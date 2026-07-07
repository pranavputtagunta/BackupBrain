/**
 * BackupBrain — Phase 2 React Native app.
 *
 * Screen layout:
 *   - Top: live camera preview with face bounding boxes (main feed)
 *   - Bottom: simulated 1.8" AR display (240x320, white-on-black) showing
 *     the memory prompt, recognized name, and live transcript
 *
 * All AI runs on the laptop's Vision Worker (see repo-root main.py);
 * this app only captures camera/audio and renders results.
 */
import React, { useEffect, useRef, useState } from "react";
import { StatusBar } from "expo-status-bar";
import {
  Button,
  SafeAreaView,
  StyleSheet,
  Text,
  View,
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import { requestRecordingPermissionsAsync } from "expo-audio";

import { getHealth } from "./src/api/visionWorker";
import { ARDisplay } from "./src/components/ARDisplay";
import { FaceOverlay } from "./src/components/FaceOverlay";
import { CONFIG } from "./src/config";
import { useRecognition } from "./src/hooks/useRecognition";
import { useTranscription } from "./src/hooks/useTranscription";

export default function App() {
  const cameraRef = useRef<CameraView | null>(null);
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [micGranted, setMicGranted] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [previewSize, setPreviewSize] = useState({ width: 0, height: 0 });
  const [workerStatus, setWorkerStatus] = useState<string>("connecting…");

  const recognition = useRecognition(cameraRef, cameraReady);
  const transcript = useTranscription(micGranted);

  useEffect(() => {
    requestRecordingPermissionsAsync().then(({ granted }) => setMicGranted(granted));
  }, []);

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const health = await getHealth();
        if (!cancelled) {
          const people = health.known_people.join(", ") || "none enrolled";
          const voice = health.whisper_ready ? "voice ready" : "voice loading";
          setWorkerStatus(`worker ok — knows: ${people} — ${voice}`);
        }
      } catch {
        if (!cancelled) {
          setWorkerStatus(`worker unreachable at ${CONFIG.WORKER_URL}`);
        }
      }
    };
    check();
    const interval = setInterval(check, 10000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  if (!cameraPermission) {
    return <SafeAreaView style={styles.container} />;
  }
  if (!cameraPermission.granted) {
    return (
      <SafeAreaView style={[styles.container, styles.center]}>
        <Text style={styles.statusText}>
          BackupBrain needs the camera to recognize faces.
        </Text>
        <Button title="Grant camera access" onPress={requestCameraPermission} />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar style="light" />
      <View
        style={styles.cameraWrap}
        onLayout={(event) => setPreviewSize(event.nativeEvent.layout)}
      >
        <CameraView
          ref={cameraRef}
          style={StyleSheet.absoluteFill}
          facing="back"
          animateShutter={false}
          onCameraReady={() => setCameraReady(true)}
        />
        <FaceOverlay
          faces={recognition.faces}
          imageSize={recognition.imageSize}
          viewWidth={previewSize.width}
          viewHeight={previewSize.height}
        />
      </View>

      <View style={styles.bottom}>
        <ARDisplay
          prompt={recognition.prompt}
          name={recognition.promptName}
          transcript={transcript}
        />
        <Text style={styles.statusText} numberOfLines={2}>
          {recognition.workerError ?? workerStatus}
        </Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#111",
  },
  center: {
    justifyContent: "center",
    alignItems: "center",
    gap: 12,
  },
  cameraWrap: {
    flex: 1,
    overflow: "hidden",
  },
  bottom: {
    alignItems: "center",
    paddingVertical: 10,
    gap: 6,
  },
  statusText: {
    color: "#888",
    fontSize: 12,
    paddingHorizontal: 16,
    textAlign: "center",
  },
});
