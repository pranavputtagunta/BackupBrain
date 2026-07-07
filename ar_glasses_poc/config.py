"""Central configuration for the Phase 1 desktop PoC.

Every tuneable constant lives here. Modules must import from this file
rather than hardcoding values, so that Phase 2/3 ports only need to
touch one place.
"""

# --- Webcam capture ---
WEBCAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAPTURE_FPS = 30
FLIP_HORIZONTAL = True  # Mirror the feed to match real-world perspective

# --- Face recognition ---
INFERENCE_EVERY_N_FRAMES = 5      # Run recognition on every Nth frame
FACE_COOLDOWN_SECONDS = 10        # Re-trigger RAG only after this many seconds
FACE_MATCH_TOLERANCE = 0.6        # face_recognition distance threshold (lower = stricter)
FACE_DETECTION_MODEL = "hog"      # "hog" (CPU-friendly) or "cnn" (GPU)
KNOWN_FACES_DIR = "data/known_faces"

# --- RAG / LLM ---
MEMORIES_DIR = "data/memories"
LLM_MODEL = "gpt-4o-mini"         # OpenAI model
OLLAMA_MODEL = "llama3.2"         # Fallback local model served by ollama
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_TIMEOUT_SECONDS = 20
LLM_SYSTEM_PROMPT = (
    "You are a memory assistant for an Alzheimer's patient. Given facts about "
    "the person in front of them, generate a single warm, clear, 1-2 sentence "
    "memory prompt. Use simple language. Start with who the person is."
)

# --- Voice / Whisper ---
WHISPER_MODEL = "base"
AUDIO_SAMPLE_RATE = 16000         # Whisper expects 16 kHz mono
AUDIO_CHUNK_SECONDS = 3           # Length of each transcription chunk
VAD_RMS_THRESHOLD = 0.01          # Skip chunks quieter than this (avoid hallucination)

# --- Simulated AR display (matches the physical 1.8" SPI TFT) ---
AR_DISPLAY_WIDTH = 240
AR_DISPLAY_HEIGHT = 320
AR_FONT_SCALE_PROMPT = 0.45       # Line 1: memory prompt (large)
AR_FONT_SCALE_NAME = 0.4          # Line 2: recognized name (medium)
AR_FONT_SCALE_TRANSCRIPT = 0.35   # Line 3: transcript snippet (small)
AR_TEXT_MARGIN_PX = 6
AR_LINE_SPACING_PX = 6

# --- Window names ---
MAIN_WINDOW_NAME = "BackupBrain - Main Feed"
AR_WINDOW_NAME = "BackupBrain - AR Display (1.8\" simulation)"

# --- Queues / threading ---
FRAME_QUEUE_SIZE = 2              # Small: drop stale frames rather than backlog
RESULT_QUEUE_SIZE = 2
TRANSCRIPT_QUEUE_SIZE = 8
RAG_QUEUE_SIZE = 8
QUEUE_GET_TIMEOUT = 0.1           # Seconds; keeps threads responsive to shutdown

# --- Logging ---
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s [%(threadName)s] %(levelname)s %(name)s: %(message)s"
