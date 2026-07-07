# BackupBrain — Phase 2 iOS App (React Native / Expo)

Phase 2 ports the Phase 1 desktop PoC to an iPhone. The app captures the
iPhone camera and microphone and renders the UI; all AI (face recognition,
RAG memory prompts, Whisper transcription) runs on the **Vision Worker** — a
FastAPI server on your laptop (repo-root `main.py`) that reuses the Phase 1
Python pipeline. This is the "API-based AI" option from the project spec, and
it means the app runs in **Expo Go** with no Mac or Xcode required.

```
iPhone (Expo app)                      Laptop (Vision Worker, port 8000)
  camera snapshot  --POST /recognize-->  face_recognition (Phase 1 module)
  known face seen  --GET  /memory-prompt/{name}-->  RAG + LLM (Phase 1 module)
  5s audio chunks  --POST /transcribe-->  Whisper + VAD (Phase 1 module)
```

## 1. Start the Vision Worker (laptop)

Prereqs: the Phase 1 setup (`.venv` with `ar_glasses_poc/requirements.txt`
installed, plus `fastapi uvicorn python-multipart`) and **ffmpeg** on PATH
(`winget install Gyan.FFmpeg`) for transcription.

```powershell
cd BackupBrain
.venv\Scripts\activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` is required so the phone can reach it. Allow Python through
the Windows firewall on private networks if prompted. Check it works from a
browser: `http://<laptop-ip>:8000/health`.

Known faces and memories come from the Phase 1 data dirs
(`ar_glasses_poc/data/`) — enroll people with `setup_faces.py` as before.

## 2. Configure and run the app (phone)

1. Find your laptop's LAN IP (`ipconfig` → IPv4 Address) and set
   `WORKER_URL` in [src/config.ts](src/config.ts). Phone and laptop must be
   on the same Wi-Fi network.
2. Install the **Expo Go** app from the App Store.
3. Start the dev server and scan the QR code with the iPhone camera:

```powershell
cd ar_glasses_app
npm install
npx expo start
```

## What the screen shows

- **Top (main feed):** live camera preview with bounding boxes — green with
  name + confidence for known faces, red for unknown.
- **Bottom (AR display):** the simulated 1.8" SPI display at its native
  240×320 — white-on-black memory prompt, recognized name, and the latest
  transcript. Exactly this component's content streams to the physical
  display in Phase 3.
- **Status line:** worker connectivity, enrolled people, whisper readiness.

## Architecture notes

- `src/config.ts` mirrors `ar_glasses_poc/config.py` — same cooldowns and
  display resolution, no constants hardcoded in components.
- `useRecognition` implements the same cooldown-gated `new_detection` rule as
  the Phase 1 recognition thread; `useTranscription` mirrors the voice
  thread's chunked capture (VAD runs server-side).
- Phase 3 adds a WebSocket bridge from this app to the ESP32 over USB-C; the
  `ARDisplay` component's props (`prompt`, `name`, `transcript`) are exactly
  the payload that bridge will stream to the physical display.
- Recognition is snapshot-based (~1.5s cadence) rather than per-frame — a
  deliberate Phase 2 tradeoff to stay inside Expo Go. A dev build with
  `react-native-vision-camera` frame processors is the upgrade path if
  higher frame rates are needed before Phase 3.
