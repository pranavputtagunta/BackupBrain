/**
 * Central configuration — the Phase 2 counterpart of ar_glasses_poc/config.py.
 * Every tuneable value lives here; components never hardcode constants.
 */
export const CONFIG = {
  /**
   * Base URL of the BackupBrain Vision Worker (FastAPI on the laptop).
   * Replace with your laptop's LAN IP — find it with `ipconfig` (Windows)
   * or `ifconfig` (macOS/Linux). Phone and laptop must share a network.
   */
  WORKER_URL: "http://192.168.1.100:8000",

  /** How often to snapshot the camera and ask the worker for faces. */
  RECOGNIZE_INTERVAL_MS: 1500,

  /** JPEG quality for uploaded frames (0-1). Low keeps uploads fast. */
  PHOTO_QUALITY: 0.3,

  /** Re-trigger a memory prompt for the same person only after this long. */
  FACE_COOLDOWN_SECONDS: 10,

  /** Length of each recorded audio chunk sent for transcription. */
  AUDIO_CHUNK_SECONDS: 5,

  /** Request timeout for worker calls. */
  REQUEST_TIMEOUT_MS: 15000,

  /**
   * Phase 3 glasses bridge. The ESP32 firmware runs an access point
   * (default SSID "BackupBrain-Glasses") and a WebSocket server at
   * ws://192.168.4.1:81. Point this at the glasses — or at the Python
   * emulator (`ar_glasses_firmware/emulator.py`) on your laptop for
   * hardware-free testing, e.g. "ws://<laptop-ip>:81".
   */
  GLASSES_WS_URL: "ws://192.168.4.1:81",

  /** Master switch for the bridge (set false to run app-only Phase 2 mode). */
  GLASSES_ENABLED: true,

  /** How long to wait before retrying a dropped glasses connection. */
  GLASSES_RECONNECT_MS: 3000,

  /**
   * Simulated 1.8" SPI display — must stay at the hardware's native
   * resolution (matches AR_DISPLAY_* in ar_glasses_poc/config.py).
   * This component is the ground truth for what Phase 3 streams to the
   * physical display over the WebSocket bridge.
   */
  AR_DISPLAY_WIDTH: 240,
  AR_DISPLAY_HEIGHT: 320,
} as const;
