"""BackupBrain Phase 1 desktop PoC — entry point.

Wires the four pipeline modules together with queues and threads:

    [capture] --frame_queue--> [recognition] --result_queue--> [display]
    [voice]   --transcript_queue-------------------------------^
    [recognition] --rag_queue--> [rag] --prompt_queue--> [display]

Modules never import each other; all wiring happens here.
Run with: python main.py   (press 'q' in a window, or Ctrl+C, to quit)
"""

from __future__ import annotations

import logging
import queue
import threading

from dotenv import load_dotenv

import config
from modules.capture import CaptureThread, WebcamSource
from modules.display import DisplayThread
from modules.face_recognition_module import FaceRecognitionThread
from modules.rag_pipeline import RagThread
from modules.voice import VoiceThread

logger = logging.getLogger(__name__)


def main() -> None:
    """Start all pipeline threads and block until shutdown."""
    load_dotenv()  # OPENAI_API_KEY (optional) from .env
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
    )

    shutdown_event = threading.Event()

    frame_queue: "queue.Queue" = queue.Queue(maxsize=config.FRAME_QUEUE_SIZE)
    result_queue: "queue.Queue" = queue.Queue(maxsize=config.RESULT_QUEUE_SIZE)
    transcript_queue: "queue.Queue" = queue.Queue(maxsize=config.TRANSCRIPT_QUEUE_SIZE)
    rag_queue: "queue.Queue" = queue.Queue(maxsize=config.RAG_QUEUE_SIZE)
    prompt_queue: "queue.Queue" = queue.Queue(maxsize=config.RAG_QUEUE_SIZE)

    threads = [
        CaptureThread(WebcamSource(), frame_queue, shutdown_event),
        FaceRecognitionThread(frame_queue, result_queue, rag_queue, shutdown_event),
        RagThread(rag_queue, prompt_queue, shutdown_event),
        VoiceThread(transcript_queue, shutdown_event),
        DisplayThread(result_queue, prompt_queue, transcript_queue, shutdown_event),
    ]

    logger.info("Starting %d pipeline threads", len(threads))
    for thread in threads:
        thread.start()

    try:
        # Block until the display thread signals shutdown ('q') or Ctrl+C.
        while not shutdown_event.is_set():
            shutdown_event.wait(timeout=0.5)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — shutting down")
        shutdown_event.set()

    for thread in threads:
        thread.join(timeout=5)
        if thread.is_alive():
            logger.warning("Thread %s did not exit cleanly", thread.name)
    logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
