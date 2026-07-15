"""BackupBrain glasses emulator — a hardware-free stand-in for the ESP32.

Speaks the exact same WebSocket protocol as the firmware (JSON display
payloads on port 81, `hello` on connect) and renders the payload with the
Phase 1 AR-display renderer, so the Phase 2 app + Phase 3 bridge can be
tested end-to-end before the glasses exist.

Usage (from repo root, venv active):
    python ar_glasses_firmware/emulator.py             # OpenCV window
    python ar_glasses_firmware/emulator.py --headless  # writes latest.png

Point the app at it by setting GLASSES_WS_URL in ar_glasses_app/src/config.ts
to ws://<laptop-ip>:81.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
from dataclasses import dataclass, field

# Reuse the Phase 1 renderer (the ground-truth display layout).
POC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ar_glasses_poc")
sys.path.insert(0, os.path.abspath(POC_DIR))

import config  # noqa: E402  (ar_glasses_poc/config.py)
from modules.display import render_ar_display  # noqa: E402

logger = logging.getLogger("glasses_emulator")

WEBSOCKET_PORT = 81
HELLO_MESSAGE = json.dumps({"type": "hello", "device": "backupbrain-glasses-emulator"})
WINDOW_NAME = "BackupBrain - Glasses Emulator (ESP32 stand-in)"


@dataclass
class EmulatorState:
    """Latest display payload, shared between server and render threads."""

    prompt: str = ""
    name: str = ""
    transcript: str = ""
    dirty: bool = True
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, payload: dict) -> None:
        """Apply one JSON payload from the app (same fields as firmware)."""
        with self.lock:
            self.prompt = str(payload.get("prompt", ""))
            self.name = str(payload.get("name", ""))
            self.transcript = str(payload.get("transcript", ""))
            self.dirty = True

    def render_if_dirty(self):
        """Return a fresh frame when the payload changed, else None."""
        with self.lock:
            if not self.dirty:
                return None
            self.dirty = False
            return render_ar_display(self.prompt, self.name, self.transcript)


state = EmulatorState()


def _handle_client(websocket) -> None:
    """Serve one app connection: greet, then apply every payload."""
    logger.info("App connected from %s", websocket.remote_address)
    websocket.send(HELLO_MESSAGE)
    for message in websocket:
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            logger.warning("Bad JSON from app: %.80s", message)
            continue
        state.update(payload)
        logger.info(
            "Payload: prompt=%.40r name=%r transcript=%.40r",
            payload.get("prompt", ""),
            payload.get("name", ""),
            payload.get("transcript", ""),
        )
    logger.info("App disconnected")


def serve_forever() -> None:
    """Run the WebSocket server (mirrors the firmware's server role)."""
    from websockets.sync.server import serve

    with serve(_handle_client, "0.0.0.0", WEBSOCKET_PORT) as server:
        logger.info("Glasses emulator listening on ws://0.0.0.0:%d", WEBSOCKET_PORT)
        server.serve_forever()


def main() -> None:
    """Start the server thread and the render loop."""
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    parser = argparse.ArgumentParser(description="BackupBrain glasses emulator")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="No window; write each frame to latest.png next to this script",
    )
    args = parser.parse_args()

    threading.Thread(target=serve_forever, name="WsServerThread", daemon=True).start()

    import cv2

    png_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latest.png")
    try:
        while True:
            frame = state.render_if_dirty()
            if args.headless:
                if frame is not None:
                    cv2.imwrite(png_path, frame)
                    logger.info("Wrote %s", png_path)
                threading.Event().wait(0.1)
            else:
                if frame is not None:
                    cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(50) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    logger.info("Emulator stopped")


if __name__ == "__main__":
    main()
