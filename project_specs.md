# Phase 1 Build Prompt — Memory Assistive AR Glasses (Desktop PoC)

## Who you are

You are an expert Python engineer helping build the Phase 1 desktop proof-of-concept for
a memory-assistive AR glasses system designed for Alzheimer's patients. Your job is to
scaffold and implement the full Python pipeline described below — writing clean, modular,
well-commented code that is explicitly designed to be ported to React Native (Phase 2)
and hardware (Phase 3) later.

---

## Project background

This project is a set of AR glasses for Alzheimer's patients. The glasses use a
Pepper's Ghost optical system to overlay high-contrast text into the patient's field of
view. When the wearer looks at a person, the glasses identify that person by face and
surface a memory prompt — e.g. "This is your son, John. You last spoke on Tuesday." —
helping patients retain their independence and connection to family.

The hardware stack (already designed, not yet built) is:
- **MCU:** Seeed Studio XIAO ESP32S3 Sense (with integrated microphone)
- **Display:** 1.8-inch SPI TFT display (128×160 or 240×320px)
- **Camera:** OV2640 wide-angle, connected via 120mm FPC ribbon cable
- **Power:** Tethered to iPhone via USB-C (no onboard battery)

The full three-phase software roadmap is:
1. **Phase 1 (this task):** Python desktop PoC — validate the full AI pipeline using a
   webcam and laptop
2. **Phase 2:** Port to a React Native iOS app using the iPhone camera and on-device /
   API-based AI
3. **Phase 3:** Integrate the Phase 2 app with the ESP32 hardware via a WebSocket bridge
   over USB-C, streaming rendered text to the physical SPI display

You are building Phase 1 only. Python is used here specifically to validate AI pipeline
logic fast — this is NOT embedded code. C++ is reserved for Phase 3 ESP32 firmware.

---

## What Phase 1 must simulate

The desktop PoC must run all four of the following simultaneously:

### 1. Webcam facial detection with name overlays
- Capture frames from the laptop webcam using OpenCV
- Run face recognition on incoming frames using the `face_recognition` library (dlib-backed)
- Match detected faces against a local known-faces database (a folder of labeled photos)
- Render bounding boxes and name labels over each detected face
- Display this annotated feed in an OpenCV window

### 2. RAG / LLM memory prompt generation
- When a known face is detected, trigger a memory prompt for that person
- Use a RAG pipeline to retrieve relevant memories/facts about that person from a local
  store (e.g. a simple JSON or text file per person: name, relationship, last interaction,
  shared memories)
- Pass the retrieved context plus the person's name to an LLM (OpenAI API or a local
  model via `ollama`) to generate a natural-language memory prompt
- Example output: "This is your son, John. You last spoke on Tuesday. He lives in Austin."
- Avoid regenerating the prompt on every frame — debounce so it only triggers once per
  new face detection or after a cooldown period (e.g. 10 seconds)

### 3. Whisper-based voice AI transcription
- Capture live microphone audio using `pyaudio` or `sounddevice`
- Run OpenAI Whisper (local, `whisper` Python library) to transcribe speech in real time
  or near-real time
- Display the live transcript in the AR window and/or terminal
- Future use: transcribed speech will be fed into the RAG store to update conversation
  history and improve prompt quality

### 4. Simulated 1.8" AR display window
- Render a separate OpenCV or tkinter window that simulates exactly what the physical
  1.8" SPI display will show
- This window should be fixed at the display's native resolution (240×320 or 128×160)
  — do not scale it up
- Render high-contrast white text on black background (matching the Pepper's Ghost
  optical setup, which requires bright text for visibility)
- Show: the current memory prompt (large text, primary), the recognized person's name
  (secondary), and optionally the live transcript at the bottom
- This window is the ground truth for what will eventually stream to the hardware display

---

## Architecture requirements

### File structure

Scaffold the project with this structure before writing any module logic:

```
ar_glasses_poc/
├── main.py                  # Entry point — wires all modules together
├── config.py                # All tuneable constants (resolution, cooldowns, model names)
├── modules/
│   ├── __init__.py
│   ├── capture.py           # Webcam capture module
│   ├── face_recognition_module.py   # Face detection + recognition
│   ├── voice.py             # Microphone capture + Whisper transcription
│   ├── rag_pipeline.py      # Memory retrieval + LLM prompt generation
│   └── display.py           # Simulated AR display window
├── data/
│   ├── known_faces/         # Subdirs per person: data/known_faces/john/photo1.jpg
│   └── memories/            # Per-person memory files: data/memories/john.json
├── requirements.txt
└── README.md
```

### Threading model

The pipeline must use Python threading with queues between modules. Do NOT block the
main thread on any I/O or inference operation. The architecture is:

```
[capture thread] --frame_queue--> [recognition thread] --result_queue--> [display thread]
[voice thread]   --transcript_queue--> [display thread]
[rag thread]     (triggered by recognition results, async)
```

- `capture.py` puts raw frames into `frame_queue` at 30fps
- `face_recognition_module.py` consumes frames, runs inference every Nth frame
  (configurable, default N=5), puts results into `result_queue`
- `display.py` consumes from `result_queue` and `transcript_queue`, renders both windows
- `voice.py` runs its own thread, puts transcript strings into `transcript_queue`
- `rag_pipeline.py` is called async when a new recognized face is detected

### Swappable capture interface

The capture module must be written with a clean abstract interface so that in Phase 3,
swapping from webcam to OV2640/WebSocket source requires only changing the source class,
not the downstream pipeline. Define a base class or protocol:

```python
class FrameSource:
    def start(self): ...
    def stop(self): ...
    def read(self) -> np.ndarray | None: ...
```

Implement `WebcamSource(FrameSource)` for Phase 1. The rest of the pipeline only ever
calls `.read()` on a `FrameSource` — never directly touches `cv2.VideoCapture`.

### Color space

OpenCV reads frames as BGR. `face_recognition` requires RGB. Always convert:
```python
rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
```
Do this once in the capture module before the frame hits any queue — not scattered
throughout the codebase.

---

## Module-by-module specs

### `capture.py`
- `WebcamSource` implements `FrameSource`
- Default resolution: 640×480
- Flip frame horizontally (`cv2.flip`) so it matches real-world perspective
- Run on its own thread, pushing to `frame_queue`
- Expose `stop()` that sets a threading event to cleanly exit the loop

### `face_recognition_module.py`
- On startup, load all known faces from `data/known_faces/` — walk subdirs, encode each
  photo, store `{name: encoding}` dict
- Process every Nth frame from `frame_queue` (skip others to maintain performance)
- For each face detected: return name (or "Unknown"), bounding box coordinates, confidence
- Push results to `result_queue` as a list of dicts:
  `[{"name": "John", "box": (top, right, bottom, left), "new_detection": True/False}]`
- Track `new_detection` = True only when a face transitions from unknown→known or
  re-appears after the cooldown window (use this to trigger RAG, not on every frame)

### `rag_pipeline.py`
- Load memories from `data/memories/{name}.json`
- Memory schema:
  ```json
  {
    "name": "John",
    "relationship": "son",
    "last_interaction": "Tuesday phone call",
    "facts": ["Lives in Austin", "Has two kids", "Works in tech"],
    "recent_conversations": ["Talked about Thanksgiving plans"]
  }
  ```
- Build a simple retrieval step: concatenate all fields into a context string
- Call OpenAI API (gpt-4o-mini for cost efficiency) or ollama with a system prompt like:
  > "You are a memory assistant for an Alzheimer's patient. Given facts about the person
  > in front of them, generate a single warm, clear, 1-2 sentence memory prompt. Use
  > simple language. Start with who the person is."
- Return the generated prompt string
- Cache the result per person for the cooldown duration — do not re-call the API on
  every detection

### `voice.py`
- Use `sounddevice` to capture audio in chunks
- Pass chunks to `whisper` (use `base` or `small` model for speed)
- Transcribe and push latest transcript string to `transcript_queue`
- Run continuously on its own thread
- Include a simple VAD (voice activity detection) heuristic — only transcribe chunks
  where RMS energy is above a threshold, to avoid hallucinating on silence

### `display.py`
- Maintain two windows:
  1. **Main feed window:** full-res webcam feed with bounding boxes and name labels
     rendered via OpenCV
  2. **AR simulation window:** fixed 240×320 (or 128×160) black background, white text
     - Line 1 (large): Memory prompt text, word-wrapped to fit
     - Line 2 (medium): Recognized name
     - Line 3 (small, bottom): Latest transcript snippet
- Both windows update in the display thread's loop, consuming from queues
- Pressing `q` in either window cleanly shuts down all threads

### `config.py`
All tuneable values go here — never hardcode in modules:
```python
WEBCAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
INFERENCE_EVERY_N_FRAMES = 5
FACE_COOLDOWN_SECONDS = 10
WHISPER_MODEL = "base"
LLM_MODEL = "gpt-4o-mini"
AR_DISPLAY_WIDTH = 240
AR_DISPLAY_HEIGHT = 320
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHUNK_SECONDS = 3
VAD_RMS_THRESHOLD = 0.01
```

---

## Known faces data setup

Include a helper script `setup_faces.py` that:
- Accepts a person's name and a path to a photo as CLI args
- Encodes the face and saves it to `data/known_faces/{name}/`
- Creates a starter `data/memories/{name}.json` with placeholder fields

Also include sample/dummy data for two people (e.g. "john" and "sarah") so the pipeline
can be tested immediately without real photos.

---

## Requirements and dependencies

Generate a `requirements.txt` with pinned versions where stability matters:

```
opencv-python
face-recognition
numpy
openai
whisper
sounddevice
scipy
python-dotenv
```

OpenAI API key should be read from a `.env` file via `python-dotenv`. Never hardcode it.

---

## README

Write a `README.md` that covers:
1. Project overview (one paragraph)
2. Installation instructions (venv setup, pip install, system deps for dlib/portaudio)
3. How to add a known face (the `setup_faces.py` workflow)
4. How to run (`python main.py`)
5. What each window shows
6. The three-phase roadmap (brief, 3 bullet points)
7. Architecture notes: why the capture module is swappable, what changes in Phase 2 vs 3

---

## Code quality standards

- Type hints on all function signatures
- Docstrings on all classes and public methods
- No magic numbers — everything via `config.py`
- All threads must handle `KeyboardInterrupt` and `threading.Event` for clean shutdown
- Log to console with Python `logging` (not `print`), with timestamps
- Never import from sibling modules in a way that creates circular dependencies —
  `main.py` wires everything, modules do not import each other

---

## What to build first

Work in this order:
1. Scaffold the full file/folder structure with empty files and docstring stubs
2. `config.py` — all constants
3. `capture.py` — `FrameSource` base + `WebcamSource`
4. `face_recognition_module.py` — known face loading + recognition loop
5. `display.py` — both windows, consuming from queues
6. Wire `main.py` so steps 3–5 run together (working demo before voice/RAG)
7. `rag_pipeline.py` — memory loading + LLM call
8. `voice.py` — audio capture + Whisper
9. Integrate all four in `main.py`
10. `setup_faces.py` + sample data
11. `README.md` + `requirements.txt`

At each step, confirm the partial pipeline runs before moving to the next step.

---

## What NOT to do

- Do not write ESP32 firmware or C++ — that is Phase 3
- Do not build a React Native app — that is Phase 2
- Do not use a GPU-dependent model unless explicitly asked — the PoC must run on a
  standard laptop CPU
- Do not put display logic in the capture or recognition modules — keep concerns separate
- Do not hardcode any names, API keys, file paths, or resolution values outside of
  `config.py`