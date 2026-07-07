/**
 * Simulated 1.8" SPI display — the Phase 2 counterpart of the Phase 1
 * AR simulation window, and still the ground truth for what Phase 3
 * streams to the physical display over the WebSocket bridge.
 *
 * Fixed at the display's native resolution (240x320 logical pixels,
 * never scaled up), high-contrast white text on black to match the
 * Pepper's Ghost optics.
 */
import React from "react";
import { StyleSheet, Text, View } from "react-native";

import { CONFIG } from "../config";

interface Props {
  prompt: string;
  name: string;
  transcript: string;
}

export function ARDisplay({ prompt, name, transcript }: Props) {
  return (
    <View style={styles.screen}>
      <View style={styles.top}>
        {prompt ? <Text style={styles.prompt}>{prompt}</Text> : null}
        {name ? <Text style={styles.name}>{name}</Text> : null}
      </View>
      {transcript ? (
        <Text style={styles.transcript} numberOfLines={2}>
          {transcript}
        </Text>
      ) : null}
    </View>
  );
}

const styles = StyleSheet.create({
  screen: {
    width: CONFIG.AR_DISPLAY_WIDTH,
    height: CONFIG.AR_DISPLAY_HEIGHT,
    backgroundColor: "#000",
    padding: 8,
    justifyContent: "space-between",
    borderWidth: 1,
    borderColor: "#333",
  },
  top: {
    gap: 10,
  },
  prompt: {
    color: "#fff",
    fontSize: 17,
    fontWeight: "700",
    lineHeight: 23,
  },
  name: {
    color: "#fff",
    fontSize: 14,
    fontWeight: "500",
    textTransform: "capitalize",
  },
  transcript: {
    color: "#fff",
    fontSize: 11,
    lineHeight: 15,
  },
});
