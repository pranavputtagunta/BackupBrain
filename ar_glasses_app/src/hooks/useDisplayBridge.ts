/**
 * Phase 3 glasses bridge — streams the AR display payload to the ESP32
 * over WebSocket.
 *
 * The `ARDisplay` component's props are the ground truth for the physical
 * display, so this hook takes exactly that payload and pushes it whenever
 * it changes. The firmware (ar_glasses_firmware/) renders it on the SPI
 * TFT; the Python emulator renders it in an OpenCV window for
 * hardware-free development. Reconnects automatically when the glasses
 * drop off.
 */
import { useEffect, useRef, useState } from "react";

import { CONFIG } from "../config";

export interface DisplayPayload {
  prompt: string;
  name: string;
  transcript: string;
}

export type BridgeStatus = "disabled" | "connecting" | "connected";

export function useDisplayBridge(payload: DisplayPayload): BridgeStatus {
  const [status, setStatus] = useState<BridgeStatus>(
    CONFIG.GLASSES_ENABLED ? "connecting" : "disabled"
  );
  const socketRef = useRef<WebSocket | null>(null);
  // Latest payload, so a freshly opened socket can send current state
  // immediately instead of waiting for the next change.
  const payloadRef = useRef<DisplayPayload>(payload);

  // Keep one auto-reconnecting socket for the app's lifetime.
  useEffect(() => {
    if (!CONFIG.GLASSES_ENABLED) {
      return;
    }
    let cancelled = false;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (cancelled) {
        return;
      }
      setStatus("connecting");
      const socket = new WebSocket(CONFIG.GLASSES_WS_URL);
      socketRef.current = socket;

      socket.onopen = () => {
        if (cancelled) {
          return;
        }
        setStatus("connected");
        socket.send(JSON.stringify(payloadRef.current));
      };
      socket.onclose = () => {
        if (cancelled) {
          return;
        }
        setStatus("connecting");
        reconnectTimer = setTimeout(connect, CONFIG.GLASSES_RECONNECT_MS);
      };
      socket.onerror = () => {
        // onclose fires after onerror; reconnection is handled there.
        socket.close();
      };
    };

    connect();
    return () => {
      cancelled = true;
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
      }
      socketRef.current?.close();
    };
  }, []);

  // Push every payload change to the glasses.
  useEffect(() => {
    payloadRef.current = payload;
    const socket = socketRef.current;
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(payload));
    }
  }, [payload.prompt, payload.name, payload.transcript]);

  return status;
}
