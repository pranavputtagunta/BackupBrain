# BackupBrain — Phase 1 Desktop PoC

Desktop proof-of-concept for memory-assistive AR glasses for Alzheimer's
patients. A webcam feed is scanned for known faces; when someone familiar is
recognized, a RAG pipeline retrieves stored facts about them and an LLM
generates a warm memory prompt (e.g. *"This is your son, John. You last spoke
on Tuesday."*), which is rendered on a simulated 1.8" AR display alongside a
live Whisper transcript of the conversation. This validates the full AI
pipeline on a laptop before it is ported to an iPhone app (Phase 2) and
ESP32-based glasses hardware (Phase 3).

## Installation

Requires Python 3.10+.

```bash
cd ar_glasses_poc
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### System dependencies

- **dlib** (pulled in by `face-recognition`) needs a C++ toolchain and CMake.
  - Windows: install [CMake](https://cmake.org/download/) and Visual Studio
    Build Tools, or `pip install dlib` from a prebuilt wheel.
  - macOS: `brew install cmake`
  - Linux: `sudo apt install build-essential cmake`
- **PortAudio** (needed by `sounddevice`):
  - Windows: bundled with the pip wheel, nothing to do.
  - macOS: `brew install portaudio`
  - Linux: `sudo apt install libportaudio2`
- **ffmpeg** (needed by Whisper): `winget install ffmpeg` / `brew install ffmpeg` /
  `sudo apt install ffmpeg`

### LLM backend (optional)

Create a `.env` file in `ar_glasses_poc/` with:

```
OPENAI_API_KEY=sk-...
```

If no key is set, the pipeline tries a local [ollama](https://ollama.com)
server, and finally falls back to an offline template — so the demo always
runs, just with simpler prompts.

## Adding a known face

```bash
python setup_faces.py john path/to/john.jpg
```

This validates the photo (exactly one face), copies it to
`data/known_faces/john/`, and creates a starter `data/memories/john.json`.
Edit that JSON to add real facts — sample memory files for `john` and `sarah`
are already included.

## Running

```bash
python main.py
```

Press `q` in either window (or Ctrl+C in the terminal) to shut down cleanly.

## What each window shows

1. **Main feed** — the full-resolution webcam feed with bounding boxes
   (green = known, red = unknown) and name labels with match confidence.
2. **AR display simulation** — a fixed 240×320 window (never scaled) that is
   the ground truth for the physical 1.8" SPI display: white text on black,
   showing the current memory prompt (large), the recognized name (medium),
   and the latest transcript snippet (bottom).

## Roadmap

- **Phase 1 (this repo):** Python desktop PoC — validate the AI pipeline with
  a webcam and laptop.
- **Phase 2:** React Native iOS app using the iPhone camera and on-device /
  API-based AI.
- **Phase 3:** ESP32S3 glasses hardware — the app streams rendered text over
  a USB-C WebSocket bridge to the physical SPI display.

## Architecture notes

- **Threading:** capture, recognition, RAG, voice, and display each run on
  their own thread, connected by bounded queues. Nothing blocks the main
  thread; stale frames are dropped rather than backlogged.
- **Swappable capture:** the pipeline only ever calls `.read()` on a
  `FrameSource` (see `modules/capture.py`). Phase 1 uses `WebcamSource`;
  Phase 3 swaps in an OV2640/WebSocket source without touching anything
  downstream.
- **Phase 2 vs 3:** Phase 2 replaces the capture source and UI (iPhone camera,
  React Native views) but keeps the same pipeline logic; Phase 3 keeps the
  Phase 2 app as the compute host and adds the hardware display as a render
  target — the AR simulation window here defines exactly what that display
  will show.
- **Color space:** frames are converted BGR→RGB once, inside the capture
  module, before entering any queue. All downstream code assumes RGB.
