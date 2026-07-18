"""Face detection and recognition module.

Loads a known-faces database from disk, consumes RGB frames from
`frame_queue`, runs recognition every Nth frame, and pushes annotated
results (frame + face list) to `result_queue`. Newly recognized faces
(first sighting, or reappearance after the cooldown window) are also
pushed to `rag_queue` so the RAG pipeline can generate a memory prompt.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import face_recognition
import numpy as np

import config

logger = logging.getLogger(__name__)

BoundingBox = Tuple[int, int, int, int]  # (top, right, bottom, left)


def load_known_faces(known_faces_dir: str = config.KNOWN_FACES_DIR) -> Dict[str, List[np.ndarray]]:
    """Walk `data/known_faces/<name>/*.jpg` and encode every photo.

    Returns a mapping of person name -> list of face encodings (one per
    photo that contained a detectable face).
    """
    known: Dict[str, List[np.ndarray]] = {}
    if not os.path.isdir(known_faces_dir):
        logger.warning("Known faces directory %s does not exist", known_faces_dir)
        return known

    for name in sorted(os.listdir(known_faces_dir)):
        person_dir = os.path.join(known_faces_dir, name)
        if not os.path.isdir(person_dir):
            continue
        encodings: List[np.ndarray] = []
        for filename in sorted(os.listdir(person_dir)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(person_dir, filename)
            try:
                image = face_recognition.load_image_file(path)
                found = face_recognition.face_encodings(image)
            except Exception:
                logger.exception("Failed to encode %s", path)
                continue
            if not found:
                logger.warning("No face found in %s; skipping", path)
                continue
            encodings.append(found[0])
        if encodings:
            known[name] = encodings
            logger.info("Loaded %d encoding(s) for '%s'", len(encodings), name)
        else:
            logger.warning("No usable photos for '%s'", name)

    if not known:
        logger.warning(
            "No known faces loaded — all detections will be 'Unknown'. "
            "Add people with: python setup_faces.py <name> <photo>"
        )
    return known


class FaceRecognitionThread(threading.Thread):
    """Consumes frames, recognizes faces, and emits results.

    Result queue payload (one per frame, so the display stays at capture
    rate even on skipped frames):
        {"frame": <RGB ndarray>, "faces": [
            {"name": str, "box": (top, right, bottom, left),
             "confidence": float, "new_detection": bool}, ...]}
    """

    def __init__(
        self,
        frame_queue: "queue.Queue[np.ndarray]",
        result_queue: "queue.Queue[Dict[str, Any]]",
        rag_queue: "queue.Queue[str]",
        shutdown_event: threading.Event,
        known_faces: Optional[Dict[str, List[np.ndarray]]] = None,
    ) -> None:
        super().__init__(name="FaceRecognitionThread", daemon=True)
        self._frame_queue = frame_queue
        self._result_queue = result_queue
        self._rag_queue = rag_queue
        self._shutdown = shutdown_event
        self._known = known_faces if known_faces is not None else load_known_faces()
        # Flat arrays for fast vectorized distance comparison.
        self._known_names: List[str] = []
        self._known_encodings: List[np.ndarray] = []
        for name, encodings in self._known.items():
            for encoding in encodings:
                self._known_names.append(name)
                self._known_encodings.append(encoding)
        self._last_seen: Dict[str, float] = {}  # name -> last RAG-trigger time.monotonic() reading
        self._frame_count = 0
        self._last_faces: List[Dict[str, Any]] = []

    def run(self) -> None:
        """Recognition loop: process every Nth frame, reuse results between."""
        while not self._shutdown.is_set():
            try:
                frame = self._frame_queue.get(timeout=config.QUEUE_GET_TIMEOUT)
            except queue.Empty:
                continue

            self._frame_count += 1
            if self._frame_count % config.INFERENCE_EVERY_N_FRAMES == 0:
                self._last_faces = self._recognize(frame)

            result = {"frame": frame, "faces": self._last_faces}
            try:
                self._result_queue.put_nowait(result)
            except queue.Full:
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._result_queue.put_nowait(result)
                except queue.Full:
                    pass
        logger.info("Face recognition thread exited")

    def _recognize(self, rgb_frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect and identify all faces in one RGB frame."""
        boxes = face_recognition.face_locations(rgb_frame, model=config.FACE_DETECTION_MODEL)
        if not boxes:
            return []
        encodings = face_recognition.face_encodings(rgb_frame, boxes)

        faces: List[Dict[str, Any]] = []
        now = time.monotonic()
        for box, encoding in zip(boxes, encodings):
            name, confidence = self._match(encoding)
            new_detection = False
            if name != "Unknown":
                last = self._last_seen.get(name, 0.0)
                if now - last >= config.FACE_COOLDOWN_SECONDS:
                    new_detection = True
                    self._last_seen[name] = now
                    self._trigger_rag(name)
            faces.append(
                {
                    "name": name,
                    "box": box,
                    "confidence": confidence,
                    "new_detection": new_detection,
                }
            )
        return faces

    def _match(self, encoding: np.ndarray) -> Tuple[str, float]:
        """Match one encoding against the known database.

        Returns ("Unknown", 0.0) when no known face is within tolerance;
        otherwise (name, confidence) where confidence = 1 - distance.
        """
        if not self._known_encodings:
            return "Unknown", 0.0
        distances = face_recognition.face_distance(self._known_encodings, encoding)
        best = int(np.argmin(distances))
        if distances[best] > config.FACE_MATCH_TOLERANCE:
            return "Unknown", 0.0
        return self._known_names[best], float(1.0 - distances[best])

    def _trigger_rag(self, name: str) -> None:
        """Ask the RAG pipeline (async, own thread) for a memory prompt."""
        logger.info("New detection: '%s' — triggering RAG", name)
        try:
            self._rag_queue.put_nowait(name)
        except queue.Full:
            logger.warning("RAG queue full; dropping trigger for '%s'", name)
