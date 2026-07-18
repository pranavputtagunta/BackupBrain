"""Microphone capture + Whisper transcription module.

Captures audio in fixed-length chunks with `sounddevice`, applies a
simple RMS-energy VAD gate (Whisper hallucinates text on silence), and
transcribes voiced chunks with a local Whisper model. Transcripts are
pushed to `transcript_queue` for the display thread, and mirrored into
an optional `LatestValue` holder so the RAG pipeline can use the live
conversation as its retrieval query without consuming from the queue.

Phase 2+ will also feed these transcripts back into the per-person
memory store to improve prompt quality.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Optional

import numpy as np

import config
from shared_state import LatestValue

logger = logging.getLogger(__name__)


def _load_whisper_model() -> Optional[Any]:
    """Load the local Whisper model; None if the library is unavailable."""
    try:
        import whisper
    except ImportError:
        logger.warning(
            "openai-whisper is not installed — voice transcription disabled. "
            "Install with: pip install openai-whisper"
        )
        return None
    logger.info("Loading Whisper model '%s' (first load may download weights)...", config.WHISPER_MODEL)
    model = whisper.load_model(config.WHISPER_MODEL)
    logger.info("Whisper model loaded")
    return model


def rms_energy(audio: np.ndarray) -> float:
    """Root-mean-square energy of a float32 audio buffer."""
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


class VoiceThread(threading.Thread):
    """Continuously records chunks, gates on VAD, and transcribes."""

    def __init__(
        self,
        transcript_queue: "queue.Queue[str]",
        shutdown_event: threading.Event,
        latest_transcript: Optional[LatestValue] = None,
    ) -> None:
        super().__init__(name="VoiceThread", daemon=True)
        self._transcript_queue = transcript_queue
        self._shutdown = shutdown_event
        # Falls back to a private holder so run() can write unconditionally.
        self._latest_transcript = latest_transcript if latest_transcript is not None else LatestValue()

    def run(self) -> None:
        """Voice loop: record chunk -> VAD gate -> Whisper -> queue."""
        model = _load_whisper_model()
        if model is None:
            return

        try:
            import sounddevice as sd
        except (ImportError, OSError):
            logger.warning(
                "sounddevice/PortAudio unavailable — voice transcription disabled"
            )
            return

        chunk_samples = int(config.AUDIO_SAMPLE_RATE * config.AUDIO_CHUNK_SECONDS)
        logger.info(
            "Voice capture started (%ds chunks at %d Hz)",
            config.AUDIO_CHUNK_SECONDS,
            config.AUDIO_SAMPLE_RATE,
        )
        while not self._shutdown.is_set():
            try:
                recording = sd.rec(
                    chunk_samples,
                    samplerate=config.AUDIO_SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                )
                sd.wait()
            except Exception:
                logger.exception("Audio capture failed; stopping voice thread")
                return

            audio = recording.flatten()
            energy = rms_energy(audio)
            if energy < config.VAD_RMS_THRESHOLD:
                logger.debug("Chunk below VAD threshold (%.4f); skipping", energy)
                continue

            try:
                # fp16=False keeps this CPU-safe (no half-precision warning).
                result = model.transcribe(audio, fp16=False, language="en")
            except Exception:
                logger.exception("Whisper transcription failed")
                continue

            text = str(result.get("text", "")).strip()
            if not text:
                continue
            logger.info("Transcript: %s", text)
            self._latest_transcript.set(text)
            try:
                self._transcript_queue.put_nowait(text)
            except queue.Full:
                try:
                    self._transcript_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._transcript_queue.put_nowait(text)
                except queue.Full:
                    pass
        logger.info("Voice thread exited")
