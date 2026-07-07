"""Display module: main annotated feed + simulated 1.8" AR display.

Runs the two OpenCV windows:
1. Main feed — full-resolution webcam frames with bounding boxes and
   name labels.
2. AR simulation — fixed 240x320 (native resolution, never scaled),
   high-contrast white-on-black text matching the Pepper's Ghost optics:
   memory prompt (large), recognized name (medium), transcript (small).

The AR window is the ground truth for what Phase 3 streams to the
physical SPI display. Pressing `q` in either window signals shutdown.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_BOX_KNOWN = (0, 255, 0)    # BGR green
_BOX_UNKNOWN = (0, 0, 255)  # BGR red


def wrap_text(text: str, font_scale: float, max_width_px: int, thickness: int = 1) -> List[str]:
    """Greedy word-wrap for OpenCV text so lines fit within max_width_px."""
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        (width, _), _ = cv2.getTextSize(candidate, _FONT, font_scale, thickness)
        if width <= max_width_px or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def render_main_feed(rgb_frame: np.ndarray, faces: List[Dict[str, Any]]) -> np.ndarray:
    """Draw bounding boxes + name labels; returns a BGR frame for imshow."""
    bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    for face in faces:
        top, right, bottom, left = face["box"]
        color = _BOX_KNOWN if face["name"] != "Unknown" else _BOX_UNKNOWN
        cv2.rectangle(bgr, (left, top), (right, bottom), color, 2)
        label = face["name"]
        if face.get("confidence"):
            label += f" ({face['confidence']:.2f})"
        cv2.rectangle(bgr, (left, bottom), (right, bottom + 22), color, cv2.FILLED)
        cv2.putText(bgr, label, (left + 4, bottom + 16), _FONT, 0.5, _BLACK, 1)
    return bgr


def render_ar_display(prompt: str, name: str, transcript: str) -> np.ndarray:
    """Compose the fixed-resolution AR simulation frame (white on black)."""
    canvas = np.zeros((config.AR_DISPLAY_HEIGHT, config.AR_DISPLAY_WIDTH, 3), dtype=np.uint8)
    usable_width = config.AR_DISPLAY_WIDTH - 2 * config.AR_TEXT_MARGIN_PX
    y = config.AR_TEXT_MARGIN_PX

    def draw_block(text: str, scale: float, y_start: int, max_lines: int) -> int:
        """Draw one wrapped text block; returns the next y position."""
        (_, line_height), baseline = cv2.getTextSize("Ag", _FONT, scale, 1)
        y_cursor = y_start
        for line in wrap_text(text, scale, usable_width)[:max_lines]:
            y_cursor += line_height + config.AR_LINE_SPACING_PX
            if y_cursor + baseline > config.AR_DISPLAY_HEIGHT - config.AR_TEXT_MARGIN_PX:
                break
            cv2.putText(
                canvas, line, (config.AR_TEXT_MARGIN_PX, y_cursor), _FONT, scale, _WHITE, 1
            )
        return y_cursor

    # Line 1 (primary, large): memory prompt.
    if prompt:
        y = draw_block(prompt, config.AR_FONT_SCALE_PROMPT, y, max_lines=8)
        y += config.AR_LINE_SPACING_PX * 2
    # Line 2 (secondary, medium): recognized name.
    if name:
        y = draw_block(name, config.AR_FONT_SCALE_NAME, y, max_lines=1)
    # Line 3 (small, pinned to bottom): latest transcript snippet.
    if transcript:
        lines = wrap_text(transcript, config.AR_FONT_SCALE_TRANSCRIPT, usable_width)[-2:]
        (_, line_height), _ = cv2.getTextSize("Ag", _FONT, config.AR_FONT_SCALE_TRANSCRIPT, 1)
        y_cursor = config.AR_DISPLAY_HEIGHT - config.AR_TEXT_MARGIN_PX
        for line in reversed(lines):
            cv2.putText(
                canvas,
                line,
                (config.AR_TEXT_MARGIN_PX, y_cursor),
                _FONT,
                config.AR_FONT_SCALE_TRANSCRIPT,
                _WHITE,
                1,
            )
            y_cursor -= line_height + config.AR_LINE_SPACING_PX
    return canvas


class DisplayThread(threading.Thread):
    """Renders both windows, consuming recognition, prompt, and transcript queues.

    Note: OpenCV GUI calls are kept on this single thread — cv2 windows
    are not thread-safe across multiple threads.
    """

    def __init__(
        self,
        result_queue: "queue.Queue[Dict[str, Any]]",
        prompt_queue: "queue.Queue[Dict[str, str]]",
        transcript_queue: "queue.Queue[str]",
        shutdown_event: threading.Event,
    ) -> None:
        super().__init__(name="DisplayThread", daemon=True)
        self._result_queue = result_queue
        self._prompt_queue = prompt_queue
        self._transcript_queue = transcript_queue
        self._shutdown = shutdown_event
        self._current_prompt = ""
        self._current_name = ""
        self._current_transcript = ""

    def run(self) -> None:
        """Display loop: pull latest state from queues, redraw both windows."""
        cv2.namedWindow(config.MAIN_WINDOW_NAME)
        # AUTOSIZE keeps the AR window at its native pixel size — never scaled.
        cv2.namedWindow(config.AR_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        result: Optional[Dict[str, Any]] = None
        while not self._shutdown.is_set():
            try:
                result = self._result_queue.get(timeout=config.QUEUE_GET_TIMEOUT)
            except queue.Empty:
                pass  # Redraw with the previous frame/state.

            self._drain_updates()

            if result is not None:
                cv2.imshow(
                    config.MAIN_WINDOW_NAME,
                    render_main_feed(result["frame"], result["faces"]),
                )
            cv2.imshow(
                config.AR_WINDOW_NAME,
                render_ar_display(
                    self._current_prompt, self._current_name, self._current_transcript
                ),
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("'q' pressed — shutting down")
                self._shutdown.set()

        cv2.destroyAllWindows()
        logger.info("Display thread exited")

    def _drain_updates(self) -> None:
        """Absorb any pending prompt/transcript updates (keep only latest)."""
        while True:
            try:
                update = self._prompt_queue.get_nowait()
            except queue.Empty:
                break
            self._current_prompt = update.get("prompt", "")
            self._current_name = update.get("name", "")
        while True:
            try:
                self._current_transcript = self._transcript_queue.get_nowait()
            except queue.Empty:
                break
