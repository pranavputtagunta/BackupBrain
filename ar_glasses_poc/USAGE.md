# Phase 1 PoC — Usage Guide

A step-by-step walkthrough for setting up and running the desktop PoC, plus
a config reference and troubleshooting notes. For the high-level overview
and architecture notes, see [README.md](README.md).

## 1. Prerequisites

- Python 3.10+
- A webcam and microphone
- A C++ toolchain + CMake (needed to build `dlib`)
- PortAudio (needed by `sounddevice`)
- ffmpeg (needed by Whisper)

See the [README's System dependencies section](README.md#system-dependencies)
for OS-specific install commands.

## 2. Set up the environment

```bash
cd ar_glasses_poc
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

This installs OpenCV, `face-recognition` (dlib-backed), Whisper, and the
audio/LLM libraries. The `face-recognition` install is the slowest step
(compiling dlib) — expect several minutes on first install.

## 3. (Optional) Configure an LLM backend

Without any LLM configured, the RAG pipeline still works — it falls back to
a simple offline template like *"This is John, your son."* To get warmer,
fuller prompts, add a [Gemini API key](https://aistudio.google.com/apikey):

```bash
# ar_glasses_poc/.env
GEMINI_API_KEY=...
```

By default this calls `gemini-3.5-flash` (`GEMINI_MODEL` in `config.py`).
Or run a local [ollama](https://ollama.com) server with a model pulled
(`ollama pull llama3.2` — matches `OLLAMA_MODEL` in `config.py`). The
pipeline tries Gemini first, then ollama, then the offline template — no
code changes needed either way.

The key also gates **semantic retrieval**: with it set, stored memories are
embedded (`gemini-embedding-001`, cached in `data/memories/*.embeddings.json`)
and ranked against the live conversation, so only the most relevant facts,
conversations, and related people reach the LLM. Without it, retrieval falls
back to recency/importance rules — still fully offline-capable.

## 4. Enroll known faces

Two sample people are already set up with placeholder memories
(`data/memories/john.json`, `data/memories/sarah.json`), but no face photos
are included. Enroll a real photo for each name you want recognized:

```bash
python setup_faces.py john path/to/a/clear/front-facing/photo.jpg
python setup_faces.py sarah path/to/another/photo.jpg
```

Requirements for the photo:
- Exactly one face in the frame
- Reasonably well-lit and front-facing (side profiles hurt recognition)

This copies the photo into `data/known_faces/<name>/` and creates a starter
memory file if one doesn't already exist (it won't overwrite the samples).
To add facts, edit the JSON directly, e.g. `data/memories/john.json`:

```json
{
  "name": "John",
  "relationship": "son",
  "last_interaction": "Tuesday phone call",
  "facts": [
    {"text": "Lives in Austin", "important": true},
    "Has two kids"
  ],
  "recent_conversations": [
    {"text": "Talked about Thanksgiving plans", "timestamp": "2026-06-20T18:00:00"}
  ],
  "related_people": [
    {"name": "sarah", "note": "Sarah is John's sister; they talk every week"},
    "ted"
  ]
}
```

Schema notes:

- Every array accepts plain strings (the old format) or objects, mixed
  freely — existing hand-edited files keep working unchanged.
- Facts flagged `"important": true` are always included in the LLM context;
  everything else competes for inclusion by relevance to the current
  conversation.
- `related_people` entries are either a bare name (`"ted"` — a one-line
  summary is auto-derived from `ted.json`) or `{"name", "note"}` where the
  note is hand-written context. Either way they're ranked like any other
  memory: a related person only appears in the prompt when the conversation
  is actually about them. An optional `"primary": true` marks who to prefer
  in offline mode.
- Conversation `timestamp`s are ISO-8601 strings; they drive the
  most-recent-N selection when running without an API key.

You can enroll multiple photos per person (run `setup_faces.py` again with
the same name) — recognition matches against all of them and picks the
closest.

## 5. Run it

```bash
python main.py
```

Two windows open:

- **Main Feed** — your webcam, mirrored, with a bounding box around each
  detected face. Green box + name + confidence score for known people, red
  box + "Unknown" for anyone not enrolled.
- **AR Display simulation** — a small 240×320 black window with white text.
  This is what the physical glasses display will eventually show:
  - Top: the memory prompt (e.g. *"This is your son, John. You last spoke
    on Tuesday."*)
  - Middle: the recognized name
  - Bottom: the latest line of live transcript, once Whisper picks up speech

Console output is timestamped and tells you what's happening under the hood
— known faces loaded, new detections, RAG calls, transcripts.

**Controls:** press `q` in either window, or `Ctrl+C` in the terminal, to
shut down. All threads join cleanly within a few seconds.

### What to expect the first time

- The **first run** downloads Whisper's model weights (`base` by default,
  ~140 MB) — this can take a minute depending on your connection.
- A person's memory prompt is only regenerated after they've been out of
  frame for `FACE_COOLDOWN_SECONDS` (10s by default) — so stepping in and
  out of frame quickly won't re-trigger the LLM/template every time.
- Recognition doesn't run on every frame (see `INFERENCE_EVERY_N_FRAMES`
  below) — bounding boxes update a few times per second, not at full 30fps.

## 6. Config reference

Everything tuneable lives in [config.py](config.py); nothing is hardcoded
in the modules. The values most worth adjusting while testing:

| Setting | Default | Effect |
|---|---|---|
| `WEBCAM_INDEX` | `0` | Which camera OpenCV opens. Try `1`, `2`, ... if you have multiple cameras or the default fails. |
| `INFERENCE_EVERY_N_FRAMES` | `5` | Lower = more responsive recognition, higher CPU use. Raise this on a slower laptop. |
| `FACE_COOLDOWN_SECONDS` | `10` | How long before a re-appearing face re-triggers a memory prompt. |
| `FACE_MATCH_TOLERANCE` | `0.6` | Lower = stricter matching (fewer false positives, more false "Unknown"s). |
| `WHISPER_MODEL` | `"base"` | `tiny` is faster/less accurate; `small`/`medium` are slower/more accurate. |
| `VAD_RMS_THRESHOLD` | `0.01` | Raise if silence is being transcribed as hallucinated text; lower if quiet speech is being skipped. |
| `AR_DISPLAY_WIDTH` / `HEIGHT` | `240` / `320` | Must match your target physical display — don't scale this window up for readability, it needs to stay 1:1. |

## 7. Troubleshooting

**"Could not open webcam at index 0"**
The webcam is in use by another app, or OS camera permissions are blocking
desktop apps. On Windows: Settings → Privacy & security → Camera → enable
"Let desktop apps access your camera". Also try a different `WEBCAM_INDEX`.

**Everyone shows up as "Unknown"**
No known faces are loaded yet — run `setup_faces.py` (step 4), or check the
startup log for "Loaded N encoding(s) for '<name>'" to confirm enrollment
worked. Also check lighting and that the webcam photo resembles the
enrolled photo (recognition tolerance is `FACE_MATCH_TOLERANCE`).

**`face-recognition` / `dlib` fails to install**
Almost always a missing C++ toolchain or CMake. See the README's system
dependencies section — on Windows this usually means installing Visual
Studio Build Tools and CMake before retrying `pip install`.

**No transcript ever appears**
Confirm your OS microphone permission is granted, and check the console for
`sounddevice`/PortAudio errors. Very quiet input can also fail the VAD gate
— try lowering `VAD_RMS_THRESHOLD`.

**Memory prompts are always the generic offline template**
No LLM backend is reachable — check `.env` has `GEMINI_API_KEY` set and the
key is valid, or that a local ollama server is running at `OLLAMA_URL`. This
isn't a failure mode exactly (the pipeline is designed to degrade
gracefully), just a sign no LLM is configured.

**High CPU / laggy main feed**
Raise `INFERENCE_EVERY_N_FRAMES`, or drop `FRAME_WIDTH`/`FRAME_HEIGHT` in
`config.py`. `FACE_DETECTION_MODEL = "cnn"` is more accurate but much
slower on CPU-only machines — leave it on `"hog"` unless you have a GPU
build of dlib.
