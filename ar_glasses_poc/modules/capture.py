"""Webcam capture module.

Defines the swappable `FrameSource` interface and the Phase 1
`WebcamSource` implementation. In Phase 3 an `ESP32Source` (OV2640 over
WebSocket) will implement the same interface, so nothing downstream of
this module ever touches `cv2.VideoCapture` directly.

Frames are converted BGR -> RGB exactly once, here, before they hit any
queue. Everything downstream assumes RGB.
"""

from __future__ import annotations

import logging
import queue
import threading
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)


class FrameSource(ABC):
    """Abstract frame source. Phase 3 swaps in an ESP32/WebSocket source."""

    @abstractmethod
    def start(self) -> None:
        """Open the underlying device/stream."""

    @abstractmethod
    def stop(self) -> None:
        """Release the underlying device/stream."""

    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Return the next RGB frame, or None if unavailable."""


class WebcamSource(FrameSource):
    """Laptop webcam frame source backed by OpenCV."""

    def __init__(
        self,
        index: int = config.WEBCAM_INDEX,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
    ) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._cap: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        """Open the webcam and configure resolution."""
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {self._index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        logger.info("Webcam %d opened at %dx%d", self._index, self._width, self._height)

    def stop(self) -> None:
        """Release the webcam."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Webcam released")

    def read(self) -> Optional[np.ndarray]:
        """Grab one frame; returns an RGB array or None on failure."""
        if self._cap is None:
            return None
        ok, bgr_frame = self._cap.read()
        if not ok or bgr_frame is None:
            return None
        if config.FLIP_HORIZONTAL:
            bgr_frame = cv2.flip(bgr_frame, 1)
        # Single BGR->RGB conversion point for the whole pipeline.
        return cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)


class CaptureThread(threading.Thread):
    """Continuously reads from a FrameSource and pushes RGB frames to a queue.

    If the queue is full the oldest frame is dropped — the pipeline always
    works on the freshest frame rather than building a backlog.
    """

    def __init__(
        self,
        source: FrameSource,
        frame_queue: "queue.Queue[np.ndarray]",
        shutdown_event: threading.Event,
    ) -> None:
        super().__init__(name="CaptureThread", daemon=True)
        self._source = source
        self._frame_queue = frame_queue
        self._shutdown = shutdown_event

    def run(self) -> None:
        """Capture loop: read frames until shutdown is signalled."""
        try:
            self._source.start()
        except RuntimeError:
            logger.exception("Failed to start frame source; requesting shutdown")
            self._shutdown.set()
            return
        try:
            while not self._shutdown.is_set():
                frame = self._source.read()
                if frame is None:
                    logger.warning("Frame source returned None; retrying")
                    continue
                try:
                    self._frame_queue.put_nowait(frame)
                except queue.Full:
                    # Drop the stale frame, then enqueue the fresh one.
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self._frame_queue.put_nowait(frame)
                    except queue.Full:
                        pass
        finally:
            self._source.stop()
            logger.info("Capture thread exited")
