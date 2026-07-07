/**
 * Bounding-box overlay for the camera preview — the Phase 2 counterpart
 * of the Phase 1 main-feed annotations (green = known, red = unknown).
 *
 * Boxes arrive in uploaded-photo pixel coordinates; the camera preview
 * uses "cover" resizing, so we scale by the larger axis ratio and center
 * the overflow before positioning each box.
 */
import React from "react";
import { StyleSheet, Text, View } from "react-native";

import type { Face, ImageSize } from "../types";

interface Props {
  faces: Face[];
  imageSize: ImageSize | null;
  viewWidth: number;
  viewHeight: number;
}

export function FaceOverlay({ faces, imageSize, viewWidth, viewHeight }: Props) {
  if (!imageSize || viewWidth === 0 || viewHeight === 0) {
    return null;
  }
  const scale = Math.max(viewWidth / imageSize.width, viewHeight / imageSize.height);
  const offsetX = (imageSize.width * scale - viewWidth) / 2;
  const offsetY = (imageSize.height * scale - viewHeight) / 2;

  return (
    <View style={StyleSheet.absoluteFill} pointerEvents="none">
      {faces.map((face, index) => {
        const [top, right, bottom, left] = face.box;
        const known = face.name !== "Unknown";
        const boxStyle = {
          left: left * scale - offsetX,
          top: top * scale - offsetY,
          width: (right - left) * scale,
          height: (bottom - top) * scale,
          borderColor: known ? "#00e676" : "#ff5252",
        };
        const label = known
          ? `${face.name} (${face.confidence.toFixed(2)})`
          : "Unknown";
        return (
          <View key={`${face.name}-${index}`} style={[styles.box, boxStyle]}>
            <Text
              style={[
                styles.label,
                { backgroundColor: known ? "#00e676" : "#ff5252" },
              ]}
            >
              {label}
            </Text>
          </View>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  box: {
    position: "absolute",
    borderWidth: 2,
    borderRadius: 4,
  },
  label: {
    position: "absolute",
    bottom: -22,
    left: -2,
    color: "#000",
    fontSize: 12,
    fontWeight: "600",
    paddingHorizontal: 4,
    paddingVertical: 2,
    borderRadius: 2,
    overflow: "hidden",
  },
});
