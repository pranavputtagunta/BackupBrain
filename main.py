"""BackupBrain Vision Worker — Phase 2 backend API.

FastAPI service that exposes the Phase 1 AI pipeline (face recognition,
RAG memory prompts, Whisper transcription) over HTTP so the Phase 2
React Native iOS app (`ar_glasses_app/`) can use the laptop as its
"API-based AI" backend per the project spec.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health                 — liveness + pipeline status
    POST /recognize              — JPEG upload -> recognized faces
    GET  /memory-prompt/{name}   — RAG memory prompt (cached per cooldown)
    POST /transcribe             — audio upload -> Whisper transcript
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import face_recognition
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Reuse the Phase 1 pipeline modules.
POC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ar_glasses_poc")
sys.path.insert(0, POC_DIR)

import config  # noqa: E402  (ar_glasses_poc/config.py)
from modules.face_recognition_module import load_known_faces  # noqa: E402
from modules.rag_pipeline import generate_memory_prompt  # noqa: E402
from modules.voice import rms_energy  # noqa: E402

KNOWN_FACES_DIR = os.path.join(POC_DIR, "data", "known_faces")
MEMORIES_DIR = os.path.join(POC_DIR, "data", "memories")

load_dotenv()
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger("vision_worker")

app = FastAPI(title="BackupBrain Vision Worker")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Response models -------------------------------------------------------

class FaceResult(BaseModel):
    """One recognized (or unknown) face in the uploaded image."""

    name: str
    box: Tuple[int, int, int, int]  # (top, right, bottom, left) in image pixels
    confidence: float


class RecognizeResponse(BaseModel):
    """All faces found in the uploaded image, plus its pixel dimensions."""

    faces: List[FaceResult]
    image_width: int
    image_height: int


class PromptResponse(BaseModel):
    """RAG-generated memory prompt for one person."""

    name: str
    prompt: str
    cached: bool


class TranscribeResponse(BaseModel):
    """Whisper transcript of an uploaded audio chunk."""

    text: str
    voiced: bool  # False when the chunk failed the VAD energy gate


# --- Shared pipeline state (loaded once at startup) ------------------------

class PipelineState:
    """Holds the known-faces DB, whisper model, and prompt cache."""

    def __init__(self) -> None:
        self.known_names: List[str] = []
        self.known_encodings: List[np.ndarray] = []
        self.whisper_model: Optional[Any] = None
        self.whisper_error: Optional[str] = None
        self.prompt_cache: Dict[str, Tuple[str, float]] = {}
        self.prompt_lock = threading.Lock()

    def load_faces(self) -> None:
        """(Re)load the known-faces database from the Phase 1 data dir."""
        known = load_known_faces(KNOWN_FACES_DIR)
        names: List[str] = []
        encodings: List[np.ndarray] = []
        for name, encs in known.items():
            for enc in encs:
                names.append(name)
                encodings.append(enc)
        self.known_names = names
        self.known_encodings = encodings
        logger.info("Vision worker loaded %d encoding(s) for %d people", len(encodings), len(known))

    def load_whisper(self) -> None:
        """Load Whisper in the background so startup isn't blocked."""
        if shutil.which("ffmpeg") is None:
            self.whisper_error = "ffmpeg not found on PATH — /transcribe disabled"
            logger.warning(self.whisper_error)
            return
        try:
            import whisper

            logger.info("Loading Whisper model '%s'...", config.WHISPER_MODEL)
            self.whisper_model = whisper.load_model(config.WHISPER_MODEL)
            logger.info("Whisper model ready")
        except Exception as exc:  # pragma: no cover - depends on host setup
            self.whisper_error = f"Whisper failed to load: {exc}"
            logger.exception("Whisper failed to load")


state = PipelineState()


@app.on_event("startup")
def startup() -> None:
    """Load known faces synchronously; load Whisper in the background."""
    state.load_faces()
    threading.Thread(target=state.load_whisper, name="WhisperLoader", daemon=True).start()


# --- Endpoints --------------------------------------------------------------

@app.get("/health")
def health() -> JSONResponse:
    """Liveness probe with pipeline readiness details."""
    return JSONResponse(
        {
            "status": "ok",
            "known_people": sorted(set(state.known_names)),
            "whisper_ready": state.whisper_model is not None,
            "whisper_error": state.whisper_error,
        }
    )


@app.post("/recognize", response_model=RecognizeResponse)
async def recognize(image: UploadFile = File(...)) -> RecognizeResponse:
    """Detect and identify faces in an uploaded JPEG/PNG frame."""
    data = await image.read()
    buffer = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model=config.FACE_DETECTION_MODEL)
    faces: List[FaceResult] = []
    if boxes:
        encodings = face_recognition.face_encodings(rgb, boxes)
        for box, encoding in zip(boxes, encodings):
            name, confidence = _match(encoding)
            faces.append(FaceResult(name=name, box=box, confidence=confidence))
    return RecognizeResponse(
        faces=faces, image_width=rgb.shape[1], image_height=rgb.shape[0]
    )


@app.get("/memory-prompt/{name}", response_model=PromptResponse)
def memory_prompt(name: str) -> PromptResponse:
    """Generate (or serve cached) RAG memory prompt for a known person."""
    name = name.strip().lower()
    now = time.time()
    with state.prompt_lock:
        cached = state.prompt_cache.get(name)
        if cached and now - cached[1] < config.FACE_COOLDOWN_SECONDS:
            return PromptResponse(name=name, prompt=cached[0], cached=True)

    prompt = generate_memory_prompt(name, memories_dir=MEMORIES_DIR)
    with state.prompt_lock:
        state.prompt_cache[name] = (prompt, now)
    return PromptResponse(name=name, prompt=prompt, cached=False)


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(audio: UploadFile = File(...)) -> TranscribeResponse:
    """Transcribe an uploaded audio chunk (m4a/wav) with local Whisper."""
    if state.whisper_model is None:
        detail = state.whisper_error or "Whisper model still loading — retry shortly"
        raise HTTPException(status_code=503, detail=detail)

    suffix = os.path.splitext(audio.filename or "chunk.m4a")[1] or ".m4a"
    data = await audio.read()
    # Whisper decodes via ffmpeg, which needs a real file path.
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(data)
        tmp.close()
        try:
            import whisper

            waveform = whisper.load_audio(tmp.name)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not decode audio: {exc}")
        if rms_energy(waveform) < config.VAD_RMS_THRESHOLD:
            return TranscribeResponse(text="", voiced=False)
        result = state.whisper_model.transcribe(waveform, fp16=False, language="en")
        return TranscribeResponse(text=str(result.get("text", "")).strip(), voiced=True)
    finally:
        os.unlink(tmp.name)


def _match(encoding: np.ndarray) -> Tuple[str, float]:
    """Match one face encoding against the known DB (same rule as Phase 1)."""
    if not state.known_encodings:
        return "Unknown", 0.0
    distances = face_recognition.face_distance(state.known_encodings, encoding)
    best = int(np.argmin(distances))
    if distances[best] > config.FACE_MATCH_TOLERANCE:
        return "Unknown", 0.0
    return state.known_names[best], float(1.0 - distances[best])
