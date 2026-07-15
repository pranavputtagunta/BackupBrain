# BackupBrain — Phase 3 Glasses Firmware (ESP32S3)

Firmware for the Seeed Studio XIAO ESP32S3 Sense that turns the glasses into
a wireless display: it runs a WebSocket server, receives display payloads
from the Phase 2 iPhone app, and renders them as high-contrast white-on-black
text on the 1.8" SPI TFT — the source image for the Pepper's Ghost optics.
The phone provides power over USB-C; the display data travels over the
glasses' own WiFi access point.

```
iPhone (Phase 2 app) --WiFi--> ESP32S3 WebSocket server (ws://192.168.4.1:81)
                                  └-> SPI TFT (white text on black)
```

## Bridge protocol

One JSON text message per display update, pushed by the app whenever the AR
display content changes (`useDisplayBridge` in the app):

```json
{"prompt": "This is your son, John. You last spoke on Tuesday.",
 "name": "john",
 "transcript": "we were talking about thanksgiving"}
```

On connect the glasses send `{"type": "hello", "device": "backupbrain-glasses"}`.
The layout mirrors the app's `ARDisplay` component and the Phase 1 simulation
window: prompt (large, top), name (medium), transcript (small, bottom).
A payload older than 60 s blanks the display rather than showing stale
guidance.

## Testing without hardware

`emulator.py` is a laptop stand-in for the glasses — same port, same
protocol, rendered with the Phase 1 display code:

```powershell
# from repo root, venv active (needs: pip install websockets)
python ar_glasses_firmware\emulator.py            # OpenCV window, q to quit
python ar_glasses_firmware\emulator.py --headless # writes latest.png instead
```

Point the app at your laptop by setting `GLASSES_WS_URL` in
`ar_glasses_app/src/config.ts` to `ws://<laptop-ip>:81`.

## Building and flashing (real hardware)

Uses [PlatformIO](https://platformio.org) (`pip install platformio`):

```powershell
cd ar_glasses_firmware
pio run                 # build
pio run -t upload       # flash over USB-C
pio device monitor      # serial logs at 115200
```

First flash on a factory board: hold **BOOT**, tap **RESET**, release BOOT,
then upload.

## Wiring (XIAO ESP32S3 → 1.8" SPI TFT)

| TFT pin | XIAO pin | Notes                        |
| ------- | -------- | ---------------------------- |
| SCK     | D8       | Hardware SPI clock           |
| SDA/MOSI| D10      | Hardware SPI data            |
| CS      | D3       | `PIN_TFT_CS` in config.h     |
| DC/A0   | D2       | `PIN_TFT_DC`                 |
| RST     | D1       | `PIN_TFT_RST`                |
| VCC     | 3V3      |                              |
| BL/LED  | 3V3      | Backlight always on          |
| GND     | GND      |                              |

## Configuration

Everything tuneable is in [include/config.h](include/config.h):

- `DISPLAY_ST7735` — 1 for the 128×160 panel (default), 0 for 240×320 ST7789.
  Text sizes adapt automatically.
- `AP_MODE` — 1 (default): glasses broadcast the `BackupBrain-Glasses`
  network (password `backupbrain`), app connects to `ws://192.168.4.1:81`.
  0: join an existing WiFi network instead (set `STA_SSID`/`STA_PASSWORD`);
  the display shows the assigned IP on boot.
- Text sizes, margins, stale timeout.

## Why WiFi and not wired USB-C?

The USB-C tether powers the glasses from the phone. iOS does not allow apps
to open arbitrary wired data channels to non-MFi USB accessories, so the
data path is the glasses' own access point — the phone stays connected to it
while the worker traffic (Phase 2) can ride cellular. A wired CDC bridge can
replace this transparently later: only `useDisplayBridge` (app) and the
transport in `main.cpp` (firmware) would change; the payload stays the same.
