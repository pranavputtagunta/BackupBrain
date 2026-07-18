"""RAG pipeline: memory retrieval + LLM memory-prompt generation.

Loads per-person memory files from `data/memories/{name}.json`, selects
the entries most relevant to the current moment, and asks an LLM to
phrase a warm 1-2 sentence memory prompt.

Retrieval (the context-selection step):
- The person's name/relationship/last interaction, plus any facts
  flagged `"important": true`, are always included.
- Everything else — remaining facts, conversation history, and
  `related_people` entries — becomes a ranked candidate. Candidates are
  embedded once (cached in a sidecar `{name}.embeddings.json`) and
  scored by cosine similarity against the live conversation transcript
  (or a generic cold-start query when nothing has been said yet), so
  only what's relevant right now reaches the LLM. Related people
  compete in the same ranked pool — they surface only when the current
  conversation actually touches them, never as a blanket list.
- Without a GEMINI_API_KEY (or if an embedding call fails), selection
  degrades to recency/importance rules with zero network calls.

LLM backends for the final phrasing, tried in order:
1. Gemini API (if GEMINI_API_KEY is set in the environment / .env)
2. Local ollama server (if reachable)
3. Offline template fallback — so the demo always produces a prompt

Results are cached per person for the cooldown duration so the API is
never called on every detection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import config
from shared_state import LatestValue

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


# --- Schema normalization ---------------------------------------------------
# Memory arrays accept both the old flat-string entries and the new object
# form, mixed freely, so hand-edited files never need migration.

def _normalize_fact(raw: Any) -> Tuple[str, bool]:
    """Return (text, important) from a fact entry; empty text = skip."""
    if isinstance(raw, str):
        return raw.strip(), False
    if isinstance(raw, dict):
        return str(raw.get("text", "")).strip(), bool(raw.get("important", False))
    return "", False


def _normalize_conversation(raw: Any) -> Tuple[str, Optional[str]]:
    """Return (text, iso_timestamp) from a conversation entry."""
    if isinstance(raw, str):
        return raw.strip(), None
    if isinstance(raw, dict):
        return str(raw.get("text", "")).strip(), raw.get("timestamp")
    return "", None


def _normalize_related(raw: Any) -> Tuple[Optional[str], Optional[str], bool]:
    """Return (lookup_key, note, primary) from a related_people entry."""
    if isinstance(raw, str):
        return raw.strip().lower() or None, None, False
    if isinstance(raw, dict):
        key = str(raw.get("name", "")).strip().lower() or None
        return key, raw.get("note"), bool(raw.get("primary", False))
    return None, None, False


@dataclass
class Candidate:
    """One ranked-retrieval candidate (fact, conversation, or related person)."""

    text: str
    kind: str  # "fact" | "conversation" | "related_person"
    important: bool = False  # related_person: offline-fallback preference only
    timestamp: Optional[str] = None
    source_name: Optional[str] = None
    display: Optional[str] = None


# --- Embedding cache --------------------------------------------------------
# Sidecar file per person: {"model": ..., "dimensionality": ..., "vectors":
# {sha256(text): [floats]}}. Machine-managed derived data (gitignored).

def _hash_text(text: str) -> str:
    """Stable cache key for one candidate text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embedding_cache_path(name: str, memories_dir: str = config.MEMORIES_DIR) -> str:
    """Path of the per-person embedding sidecar file."""
    return os.path.join(memories_dir, f"{name}{config.EMBEDDING_CACHE_SUFFIX}")


def _empty_cache() -> Dict[str, Any]:
    """Fresh cache structure matching current config."""
    return {
        "model": config.EMBEDDING_MODEL,
        "dimensionality": config.EMBEDDING_DIMENSIONALITY,
        "vectors": {},
    }


def _load_embedding_cache(name: str, memories_dir: str = config.MEMORIES_DIR) -> Dict[str, Any]:
    """Load the sidecar cache; empty cache on missing/malformed/mismatched.

    A stored model or dimensionality that differs from the current config
    invalidates the whole cache — mixing vectors of different models or
    dimensions would silently corrupt cosine similarity.
    """
    path = _embedding_cache_path(name, memories_dir)
    if not os.path.isfile(path):
        return _empty_cache()
    try:
        with open(path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warning("Malformed embedding cache %s; rebuilding", path)
        return _empty_cache()
    if (
        cache.get("model") != config.EMBEDDING_MODEL
        or cache.get("dimensionality") != config.EMBEDDING_DIMENSIONALITY
        or not isinstance(cache.get("vectors"), dict)
    ):
        logger.info("Embedding cache %s is stale (model/dims changed); rebuilding", path)
        return _empty_cache()
    return cache


def _save_embedding_cache(
    name: str, cache: Dict[str, Any], memories_dir: str = config.MEMORIES_DIR
) -> None:
    """Persist the sidecar cache; failures are logged, never fatal."""
    path = _embedding_cache_path(name, memories_dir)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except OSError:
        logger.exception("Failed to write embedding cache %s", path)


def _embed_texts(texts: List[str], task_type: str) -> Optional[List[np.ndarray]]:
    """Embed texts in one batched Gemini call; None if unavailable/failed.

    Returned vectors are L2-normalized: gemini-embedding-001 does not
    pre-normalize outputs truncated below 3072 dims, and normalizing an
    already-unit vector is a no-op, so we always normalize.
    """
    if not os.environ.get("GEMINI_API_KEY"):
        return None
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(
            http_options=types.HttpOptions(timeout=config.EMBEDDING_TIMEOUT_SECONDS * 1000)
        )
        response = client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=config.EMBEDDING_DIMENSIONALITY,
            ),
        )
        vectors: List[np.ndarray] = []
        for embedding in response.embeddings:
            vector = np.asarray(embedding.values, dtype=np.float64)
            norm = np.linalg.norm(vector)
            vectors.append(vector / norm if norm > 0 else vector)
        return vectors
    except Exception:
        logger.exception("Embedding call failed")
        return None


def _get_cached_or_embed(
    name: str, texts: List[str], memories_dir: str = config.MEMORIES_DIR
) -> Optional[Dict[str, np.ndarray]]:
    """Map each text to its embedding, hitting the sidecar cache first.

    Only cache misses trigger an API call (one batched call for all of
    them). Returns None wholesale if that call fails — the caller then
    falls back to offline selection rather than mixing partial state.
    """
    cache = _load_embedding_cache(name, memories_dir)
    vectors: Dict[str, np.ndarray] = {}
    misses: List[str] = []
    for text in texts:
        cached = cache["vectors"].get(_hash_text(text))
        if cached is not None:
            vectors[text] = np.asarray(cached, dtype=np.float64)
        elif text not in misses:
            misses.append(text)

    if misses:
        embedded = _embed_texts(misses, config.EMBEDDING_TASK_TYPE_DOCUMENT)
        if embedded is None:
            return None
        for text, vector in zip(misses, embedded):
            vectors[text] = vector
            cache["vectors"][_hash_text(text)] = vector.tolist()
        _save_embedding_cache(name, cache, memories_dir)
    return vectors


# --- Candidate building and selection ---------------------------------------

def _summarize_person(memories: Dict[str, Any]) -> str:
    """One-hop auto-derived summary of a related person from their own file."""
    display = memories.get("name", "")
    relationship = memories.get("relationship", "someone")
    summary = f"{display} is the patient's {relationship}" if display else f"The patient's {relationship}"
    chosen_fact = ""
    for raw in memories.get("facts") or []:
        text, important = _normalize_fact(raw)
        if text and (important or not chosen_fact):
            chosen_fact = text
            if important:
                break
    if chosen_fact:
        summary += f"; {chosen_fact}"
    return summary


def _build_candidates(
    name: str, memories: Dict[str, Any], memories_dir: str = config.MEMORIES_DIR
) -> Tuple[List[str], List[Candidate]]:
    """Split a memory file into always-include lines and ranked candidates.

    Facts, conversations, and related people all become plain candidates
    in one pool — relevance ranking treats them uniformly, which is what
    keeps unrelated people out of the prompt.
    """
    always: List[str] = [
        f"Name: {memories.get('name', 'Unknown')}",
        f"Relationship: {memories.get('relationship', 'unknown')}",
        f"Last interaction: {memories.get('last_interaction', 'unknown')}",
    ]
    candidates: List[Candidate] = []

    for raw in memories.get("facts") or []:
        text, important = _normalize_fact(raw)
        if not text:
            continue
        if important:
            always.append(f"Important fact: {text}")
        else:
            candidates.append(Candidate(text=text, kind="fact"))

    for raw in memories.get("recent_conversations") or []:
        text, timestamp = _normalize_conversation(raw)
        if text:
            candidates.append(Candidate(text=text, kind="conversation", timestamp=timestamp))

    for raw in memories.get("related_people") or []:
        key, note, primary = _normalize_related(raw)
        if key is None:
            continue
        display = key.capitalize()
        if note:
            text = str(note).strip()
        else:
            # Exactly one shallow hop into the referenced person's own
            # file — never recursive, so cross-reference cycles are
            # structurally impossible.
            related = load_memories(key, memories_dir)
            if related is None:
                logger.warning("related_people entry '%s' (in '%s') has no memory file; skipping", key, name)
                continue
            display = related.get("name", display)
            text = _summarize_person(related)
        if text:
            candidates.append(
                Candidate(
                    text=text,
                    kind="related_person",
                    important=primary,
                    source_name=key,
                    display=display,
                )
            )
    return always, candidates


def _select_offline(candidates: List[Candidate], recent_n: int) -> List[Candidate]:
    """No-network selection: recent-N conversations + at most one related person.

    Used both when GEMINI_API_KEY is absent and when an embedding call
    fails mid-flight. Non-important facts are dropped entirely here.
    A related person flagged `"primary": true` wins the single slot;
    the flag never bypasses ranking in the online path.
    """
    conversations = sorted(
        (c for c in candidates if c.kind == "conversation"),
        key=lambda c: c.timestamp or "",
        reverse=True,
    )[:recent_n]
    related = [c for c in candidates if c.kind == "related_person"]
    primary = [c for c in related if c.important]
    chosen_related = primary[:1] or related[:1]
    return conversations + chosen_related


def _format_context(always_lines: List[str], selected: List[Candidate]) -> str:
    """Assemble the final flat context string fed to the LLM."""
    lines = list(always_lines)
    facts = [c.text for c in selected if c.kind == "fact"]
    if facts:
        lines.append("Relevant facts: " + "; ".join(facts))
    conversations = [c.text for c in selected if c.kind == "conversation"]
    if conversations:
        lines.append("Relevant recent conversations: " + "; ".join(conversations))
    for candidate in selected:
        if candidate.kind == "related_person":
            label = candidate.display or candidate.source_name or "A related person"
            lines.append(f"Also relevant: {label} — {candidate.text}")
    return "\n".join(lines)


def _select_context(
    name: str,
    memories: Dict[str, Any],
    query_text: str,
    memories_dir: str = config.MEMORIES_DIR,
) -> str:
    """Retrieval-aware replacement for the old flatten-everything step."""
    always_lines, candidates = _build_candidates(name, memories, memories_dir)
    if not candidates:
        return "\n".join(always_lines)

    if not os.environ.get("GEMINI_API_KEY"):
        selected = _select_offline(candidates, config.RAG_OFFLINE_RECENT_CONVERSATIONS)
        return _format_context(always_lines, selected)

    query_vectors = _embed_texts([query_text], config.EMBEDDING_TASK_TYPE_QUERY)
    vectors = (
        _get_cached_or_embed(name, [c.text for c in candidates], memories_dir)
        if query_vectors is not None
        else None
    )
    if query_vectors is None or vectors is None:
        logger.info("Embedding unavailable; using offline selection for '%s'", name)
        selected = _select_offline(candidates, config.RAG_OFFLINE_RECENT_CONVERSATIONS)
        return _format_context(always_lines, selected)

    # Both sides are L2-normalized, so dot product == cosine similarity.
    matrix = np.stack([vectors[c.text] for c in candidates])
    similarities = matrix @ query_vectors[0]
    selected = [
        candidates[i]
        for i in np.argsort(-similarities)[: config.RAG_TOP_K]
        if similarities[i] >= config.RAG_MIN_SIMILARITY_THRESHOLD
    ]
    logger.debug(
        "Retrieval for '%s': kept %d/%d candidates (query: %.60r)",
        name,
        len(selected),
        len(candidates),
        query_text,
    )
    return _format_context(always_lines, selected)


# --- Prompt generation ------------------------------------------------------

def _generate_via_gemini(context: str) -> Optional[str]:
    """Call the Gemini API; returns None if unavailable or on error."""
    if not os.environ.get("GEMINI_API_KEY"):
        return None
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(http_options=types.HttpOptions(timeout=config.LLM_TIMEOUT_SECONDS * 1000))
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=context,
            config=types.GenerateContentConfig(system_instruction=config.LLM_SYSTEM_PROMPT),
        )
        return (response.text or "").strip() or None
    except Exception:
        logger.exception("Gemini call failed")
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


def generate_memory_prompt(name: str, transcript: str = "") -> str:
    """Full RAG flow for one person: load -> retrieve -> generate.

    `transcript` is the latest live conversation snippet; when present it
    drives retrieval so the prompt reflects what's being discussed right
    now. Empty transcript (cold start) uses a generic query instead.
    """
    memories = load_memories(name)
    if memories is None:
        return f"This is {name}."
    query_text = transcript.strip() or config.RAG_COLD_START_QUERY_TEMPLATE.format(
        name=memories.get("name", name)
    )
    context = _select_context(name, memories, query_text)

    prompt = _generate_via_gemini(context)
    if prompt:
        logger.info("Prompt for '%s' generated via Gemini", name)
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
    thread on new detections), generates a memory prompt — using the
    latest live transcript as the retrieval query when available — and
    pushes {"name": ..., "prompt": ...} to `prompt_queue` for the
    display. Prompts are cached per person for the cooldown duration.
    """

    def __init__(
        self,
        rag_queue: "queue.Queue[str]",
        prompt_queue: "queue.Queue[Dict[str, str]]",
        shutdown_event: threading.Event,
        latest_transcript: Optional[LatestValue] = None,
    ) -> None:
        super().__init__(name="RagThread", daemon=True)
        self._rag_queue = rag_queue
        self._prompt_queue = prompt_queue
        self._shutdown = shutdown_event
        self._latest_transcript = latest_transcript if latest_transcript is not None else LatestValue()
        self._cache: Dict[str, tuple[str, float]] = {}  # name -> (prompt, time.monotonic() reading)

    def run(self) -> None:
        """Worker loop: serve prompt requests until shutdown."""
        while not self._shutdown.is_set():
            try:
                name = self._rag_queue.get(timeout=config.QUEUE_GET_TIMEOUT)
            except queue.Empty:
                continue

            now = time.monotonic()
            cached = self._cache.get(name)
            if cached and now - cached[1] < config.FACE_COOLDOWN_SECONDS:
                prompt = cached[0]
                logger.debug("Using cached prompt for '%s'", name)
            else:
                transcript = str(self._latest_transcript.get() or "")
                prompt = generate_memory_prompt(name, transcript=transcript)
                self._cache[name] = (prompt, now)

            try:
                self._prompt_queue.put_nowait({"name": name, "prompt": prompt})
            except queue.Full:
                logger.warning("Prompt queue full; dropping prompt for '%s'", name)
        logger.info("RAG thread exited")
