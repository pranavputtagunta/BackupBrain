// BackupBrain Phase 3 firmware configuration.
// All tuneable values live here — the .cpp files never hardcode them
// (same rule as config.py in Phase 1 and config.ts in Phase 2).

#pragma once

// --- Display -----------------------------------------------------------
// The 1.8" SPI TFT is either an ST7735 (128x160) or ST7789 (240x320).
// Pick one; the render layout adapts to the resolution automatically.
#define DISPLAY_ST7735 1  // 1 = ST7735 128x160, 0 = ST7789 240x320

#if DISPLAY_ST7735
static const int DISPLAY_WIDTH = 128;
static const int DISPLAY_HEIGHT = 160;
#else
static const int DISPLAY_WIDTH = 240;
static const int DISPLAY_HEIGHT = 320;
#endif

// SPI wiring on the XIAO ESP32S3 (hardware SPI: SCK = D8, MOSI = D10).
// Only the control pins are configurable:
static const int PIN_TFT_CS = D3;   // Chip select
static const int PIN_TFT_DC = D2;   // Data/command
static const int PIN_TFT_RST = D1;  // Reset (or wire to 3V3 and set to -1)

// Text sizes (Adafruit GFX classic font: 6x8 px per char at size 1).
static const int PROMPT_TEXT_SIZE = DISPLAY_WIDTH >= 240 ? 2 : 1;
static const int NAME_TEXT_SIZE = DISPLAY_WIDTH >= 240 ? 2 : 1;
static const int TRANSCRIPT_TEXT_SIZE = 1;
static const int TEXT_MARGIN_PX = 4;
static const int LINE_SPACING_PX = 2;
static const int BLOCK_SPACING_PX = 8;

// --- WiFi / WebSocket bridge -------------------------------------------
// The glasses run their own access point; the iPhone joins it and the
// app connects to ws://192.168.4.1:81 (AP mode keeps the pair working
// anywhere, with USB-C providing power from the phone). Set AP_MODE to 0
// to join an existing network instead (dev convenience).
#define AP_MODE 1

static const char *AP_SSID = "BackupBrain-Glasses";
static const char *AP_PASSWORD = "backupbrain";  // WPA2 needs >= 8 chars

// Used only when AP_MODE is 0:
static const char *STA_SSID = "your-wifi-ssid";
static const char *STA_PASSWORD = "your-wifi-password";

static const uint16_t WEBSOCKET_PORT = 81;

// Serial log speed.
static const unsigned long SERIAL_BAUD = 115200;

// Blank the display if no update arrives for this long (stale prompt is
// worse than no prompt for the wearer).
static const unsigned long DISPLAY_STALE_TIMEOUT_MS = 60000;
