// BackupBrain Phase 3 firmware — Seeed XIAO ESP32S3 Sense.
//
// Receives display payloads from the Phase 2 iPhone app over a WebSocket
// bridge and renders them as high-contrast white-on-black text on the
// 1.8" SPI TFT (the Pepper's Ghost source image).
//
// Protocol (one JSON text message per update, pushed by the app):
//   {"prompt": "This is your son, John...", "name": "john", "transcript": "..."}
//
// The layout mirrors the ARDisplay component in the app and the Phase 1
// simulation window: prompt (large, top), name (medium), transcript
// (small, pinned to the bottom).

#include <Arduino.h>
#include <SPI.h>
#include <WiFi.h>
#include <ArduinoJson.h>
#include <WebSocketsServer.h>
#include <Adafruit_GFX.h>

#include "config.h"

#if DISPLAY_ST7735
#include <Adafruit_ST7735.h>
static Adafruit_ST7735 tft(PIN_TFT_CS, PIN_TFT_DC, PIN_TFT_RST);
#else
#include <Adafruit_ST7789.h>
static Adafruit_ST7789 tft(PIN_TFT_CS, PIN_TFT_DC, PIN_TFT_RST);
#endif

static WebSocketsServer webSocket(WEBSOCKET_PORT);

// Latest payload from the app. Rendering happens in loop(), never inside
// the WebSocket callback, so slow SPI writes can't stall the socket.
struct DisplayState {
  String prompt;
  String name;
  String transcript;
  bool dirty = false;
  unsigned long lastUpdateMs = 0;
};
static DisplayState state;

// --- Text layout ---------------------------------------------------------

// Classic GFX font is 6x8 px per character at size 1.
static int charsPerLine(int textSize) {
  return (DISPLAY_WIDTH - 2 * TEXT_MARGIN_PX) / (6 * textSize);
}

static int lineHeight(int textSize) {
  return 8 * textSize + LINE_SPACING_PX;
}

// Greedy word-wrap (same rule as Phase 1's wrap_text). Returns the number
// of lines written into `lines`, capped at maxLines.
static int wrapText(const String &text, int textSize, String *lines, int maxLines) {
  const int maxChars = charsPerLine(textSize);
  int lineCount = 0;
  String current = "";
  int wordStart = 0;

  while (wordStart < (int)text.length() && lineCount < maxLines) {
    int wordEnd = text.indexOf(' ', wordStart);
    if (wordEnd < 0) wordEnd = text.length();
    String word = text.substring(wordStart, wordEnd);
    wordStart = wordEnd + 1;
    if (word.length() == 0) continue;

    if (current.length() == 0) {
      current = word;
    } else if ((int)(current.length() + 1 + word.length()) <= maxChars) {
      current += ' ';
      current += word;
    } else {
      lines[lineCount++] = current;
      current = word;
    }
  }
  if (current.length() > 0 && lineCount < maxLines) {
    lines[lineCount++] = current;
  }
  return lineCount;
}

// Draw one wrapped block starting at y; returns the y below the block.
static int drawBlock(const String &text, int textSize, int y, int maxLines) {
  String lines[12];
  if (maxLines > 12) maxLines = 12;
  const int count = wrapText(text, textSize, lines, maxLines);
  tft.setTextSize(textSize);
  for (int i = 0; i < count; i++) {
    if (y + lineHeight(textSize) > DISPLAY_HEIGHT - TEXT_MARGIN_PX) break;
    tft.setCursor(TEXT_MARGIN_PX, y);
    tft.print(lines[i]);
    y += lineHeight(textSize);
  }
  return y;
}

static void renderDisplay() {
  tft.fillScreen(0x0000);  // black
  tft.setTextColor(0xFFFF);  // white — Pepper's Ghost needs max brightness
  tft.setTextWrap(false);

  int y = TEXT_MARGIN_PX;
  if (state.prompt.length() > 0) {
    y = drawBlock(state.prompt, PROMPT_TEXT_SIZE, y, 8);
    y += BLOCK_SPACING_PX;
  }
  if (state.name.length() > 0) {
    drawBlock(state.name, NAME_TEXT_SIZE, y, 1);
  }
  if (state.transcript.length() > 0) {
    // Bottom-pinned: wrap first, then draw the last lines that fit.
    String lines[4];
    const int count = wrapText(state.transcript, TRANSCRIPT_TEXT_SIZE, lines, 4);
    const int shown = count > 2 ? 2 : count;
    int ty = DISPLAY_HEIGHT - TEXT_MARGIN_PX - shown * lineHeight(TRANSCRIPT_TEXT_SIZE);
    tft.setTextSize(TRANSCRIPT_TEXT_SIZE);
    for (int i = count - shown; i < count; i++) {
      tft.setCursor(TEXT_MARGIN_PX, ty);
      tft.print(lines[i]);
      ty += lineHeight(TRANSCRIPT_TEXT_SIZE);
    }
  }
}

static void showStatusScreen(const String &line1, const String &line2) {
  tft.fillScreen(0x0000);
  tft.setTextColor(0xFFFF);
  tft.setTextSize(1);
  tft.setCursor(TEXT_MARGIN_PX, TEXT_MARGIN_PX);
  tft.print(line1);
  tft.setCursor(TEXT_MARGIN_PX, TEXT_MARGIN_PX + lineHeight(1));
  tft.print(line2);
}

// --- WebSocket bridge ----------------------------------------------------

static void onWebSocketEvent(uint8_t client, WStype_t type, uint8_t *payload, size_t length) {
  switch (type) {
    case WStype_CONNECTED:
      Serial.printf("[ws] client %u connected\n", client);
      webSocket.sendTXT(client, "{\"type\":\"hello\",\"device\":\"backupbrain-glasses\"}");
      break;
    case WStype_DISCONNECTED:
      Serial.printf("[ws] client %u disconnected\n", client);
      break;
    case WStype_TEXT: {
      JsonDocument doc;
      const DeserializationError error = deserializeJson(doc, payload, length);
      if (error) {
        Serial.printf("[ws] bad JSON: %s\n", error.c_str());
        return;
      }
      state.prompt = doc["prompt"] | "";
      state.name = doc["name"] | "";
      state.transcript = doc["transcript"] | "";
      state.dirty = true;
      state.lastUpdateMs = millis();
      break;
    }
    default:
      break;
  }
}

// --- Arduino lifecycle -----------------------------------------------------

void setup() {
  Serial.begin(SERIAL_BAUD);

#if DISPLAY_ST7735
  tft.initR(INITR_BLACKTAB);
#else
  tft.init(DISPLAY_WIDTH, DISPLAY_HEIGHT);
#endif
  tft.setRotation(0);

  IPAddress ip;
#if AP_MODE
  WiFi.softAP(AP_SSID, AP_PASSWORD);
  ip = WiFi.softAPIP();  // 192.168.4.1 by default
  Serial.printf("[wifi] AP '%s' up, ip %s\n", AP_SSID, ip.toString().c_str());
#else
  WiFi.begin(STA_SSID, STA_PASSWORD);
  showStatusScreen("BackupBrain", "joining wifi...");
  while (WiFi.status() != WL_CONNECTED) {
    delay(250);
  }
  ip = WiFi.localIP();
  Serial.printf("[wifi] joined '%s', ip %s\n", STA_SSID, ip.toString().c_str());
#endif

  webSocket.begin();
  webSocket.onEvent(onWebSocketEvent);
  Serial.printf("[ws] listening on ws://%s:%u\n", ip.toString().c_str(), WEBSOCKET_PORT);

  showStatusScreen("BackupBrain ready", "ws://" + ip.toString() + ":" + String(WEBSOCKET_PORT));
}

void loop() {
  webSocket.loop();

  if (state.dirty) {
    state.dirty = false;
    renderDisplay();
  }

  // Blank a stale prompt rather than leave outdated guidance visible.
  if (state.lastUpdateMs != 0 && millis() - state.lastUpdateMs > DISPLAY_STALE_TIMEOUT_MS) {
    state = DisplayState{};
    tft.fillScreen(0x0000);
  }
}
