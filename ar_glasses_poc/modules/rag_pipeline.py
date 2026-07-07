"""RAG pipeline: memory retrieval + LLM memory-prompt generation.

Loads per-person memory files from `data/memories/{name}.json`, builds a
context string (the "retrieval" step — deliberately simple for Phase 1),
and asks an LLM to phrase a warm 1-2 sentence memory prompt.

LLM backends, tried in order:
1. OpenAI API (if OPENAI_API_KEY is set in the environment / .env)
2. Local ollama server (if reachable)
3. Offline template fallback — so the demo always produces a prompt

Results are cached per person for the cooldown duration so the API is
never called on every detection.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from typing import Any, Dict, Optional

import config

logger = logging.getLogger(__name__)


def load_memories(name: str, memories_dir: str = config.MEMORIES_DIR) -> Optional[Dict[str, Any]]:
    """Load `data/memories/{name}.json`; None if missing or malformed."""
    path = os.path.join(memories_dir, f"{name}.json")
    if not os.path.isfile(path):
        logger.warning("No memory file for '%s' at %s", name, path)
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.exception("Failed to read memory file %s", path)
        return None


def build_context(memories: Dict[str, Any]) -> str:
    """Retrieval step: flatten all memory fields into one context string."""
    parts = [
        f"Name: {memories.get('name', 'Unknown')}",
        f"Relationship: {memories.get('relationship', 'unknown')}",
        f"Last interaction: {memories.get('last_interaction', 'unknown')}",
    ]
    facts = memories.get("facts") or []
    if facts:
        parts.append("Facts: " + "; ".join(facts))
    conversations = memories.get("recent_conversations") or []
    if conversations:
        parts.append("Recent conversations: " + "; ".join(conversations))
    return "\n".join(parts)


def _generate_via_openai(context: str) -> Optional[str]:
    """Call the OpenAI API; returns None if unavailable or on error."""
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": config.LLM_SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            timeout=config.LLM_TIMEOUT_SECONDS,
        )
        return (response.choices[0].message.content or "").strip() or None
    except Exception:
        logger.exception("OpenAI call failed")
        return None


def _generate_via_ollama(context: str) -> Optional[str]:
    """Call a local ollama server; returns None if unreachable or on error."""
    try:
        import urllib.request

        payload = json.dumps(
            {
                "model": config.OLLAMA_MODEL,
                "system": config.LLM_SYSTEM_PROMPT,
                "prompt": context,
                "stream": False,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            config.OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=config.LLM_TIMEOUT_SECONDS) as response:
            body = json.loads(response.read().decode("utf-8"))
        return (body.get("response") or "").strip() or None
    except Exception:
        logger.debug("Ollama not reachable or call failed", exc_info=True)
        return None


def _generate_fallback(memories: Dict[str, Any]) -> str:
    """Offline template prompt so the pipeline works without any LLM."""
    name = memories.get("name", "someone you know")
    relationship = memories.get("relationship")
    last = memories.get("last_interaction")
    prompt = f"This is {name}"
    if relationship:
        prompt += f", your {relationship}"
    prompt += "."
    if last:
        prompt += f" You last connected: {last}."
    return prompt


def generate_memory_prompt(name: str) -> str:
    """Full RAG flow for one person: load -> retrieve -> generate."""
    memories = load_memories(name)
    if memories is None:
        return f"This is {name}."
    context = build_context(memories)

    prompt = _generate_via_openai(context)
    if prompt:
        logger.info("Prompt for '%s' generated via OpenAI", name)
        return prompt

    prompt = _generate_via_ollama(context)
    if prompt:
        logger.info("Prompt for '%s' generated via ollama", name)
        return prompt

    logger.info("No LLM available; using offline template for '%s'", name)
    return _generate_fallback(memories)


class RagThread(threading.Thread):
    """Async RAG worker.

    Consumes person names from `rag_queue` (pushed by the recognition
    thread on new detections), generates a memory prompt, and pushes
    {"name": ..., "prompt": ...} to `prompt_queue` for the display.
    Prompts are cached per person for the cooldown duration.
    """

    def __init__(
        self,
        rag_queue: "queue.Queue[str]",
        prompt_queue: "queue.Queue[Dict[str, str]]",
        shutdown_event: threading.Event,
    ) -> None:
        super().__init__(name="RagThread", daemon=True)
        self._rag_queue = rag_queue
        self._prompt_queue = prompt_queue
        self._shutdown = shutdown_event
        self._cache: Dict[str, tuple[str, float]] = {}  # name -> (prompt, timestamp)

    def run(self) -> None:
        """Worker loop: serve prompt requests until shutdown."""
        while not self._shutdown.is_set():
            try:
                name = self._rag_queue.get(timeout=config.QUEUE_GET_TIMEOUT)
            except queue.Empty:
                continue

            now = time.time()
            cached = self._cache.get(name)
            if cached and now - cached[1] < config.FACE_COOLDOWN_SECONDS:
                prompt = cached[0]
                logger.debug("Using cached prompt for '%s'", name)
            else:
                prompt = generate_memory_prompt(name)
                self._cache[name] = (prompt, now)

            try:
                self._prompt_queue.put_nowait({"name": name, "prompt": prompt})
            except queue.Full:
                logger.warning("Prompt queue full; dropping prompt for '%s'", name)
        logger.info("RAG thread exited")
