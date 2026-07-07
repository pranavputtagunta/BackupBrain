/**
 * Recognition loop hook — the Phase 2 counterpart of the Phase 1
 * recognition + RAG threads.
 *
 * Every RECOGNIZE_INTERVAL_MS it snapshots the camera, sends the frame to
 * the vision worker, and updates the face list. When a known face is seen
 * for the first time (or reappears after the cooldown window) it fetches a
 * memory prompt — the same debounce rule as Phase 1's `new_detection`.
 */
import { useEffect, useRef, useState } from "react";
import type { CameraView } from "expo-camera";

import { getMemoryPrompt, recognize } from "../api/visionWorker";
import { CONFIG } from "../config";
import type { Face, ImageSize } from "../types";

export interface RecognitionState {
  faces: Face[];
  imageSize: ImageSize | null;
  prompt: string;
  promptName: string;
  workerError: string | null;
}

export function useRecognition(
  cameraRef: React.RefObject<CameraView | null>,
  enabled: boolean
): RecognitionState {
  const [faces, setFaces] = useState<Face[]>([]);
  const [imageSize, setImageSize] = useState<ImageSize | null>(null);
  const [prompt, setPrompt] = useState("");
  const [promptName, setPromptName] = useState("");
  const [workerError, setWorkerError] = useState<string | null>(null);
  const busyRef = useRef(false);
  const lastSeenRef = useRef<Map<string, number>>(new Map());

  useEffect(() => {
    if (!enabled) {
      return;
    }
    let cancelled = false;

    const tick = async () => {
      if (busyRef.current || cancelled || !cameraRef.current) {
        return;
      }
      busyRef.current = true;
      try {
        const photo = await cameraRef.current.takePictureAsync({
          quality: CONFIG.PHOTO_QUALITY,
          skipProcessing: true,
          shutterSound: false,
        });
        if (!photo || cancelled) {
          return;
        }
        const result = await recognize(photo.uri);
        if (cancelled) {
          return;
        }
        setFaces(result.faces);
        setImageSize({ width: result.image_width, height: result.image_height });
        setWorkerError(null);

        // Cooldown-gated prompt fetch, mirroring Phase 1's new_detection.
        const now = Date.now() / 1000;
        for (const face of result.faces) {
          if (face.name === "Unknown") {
            continue;
          }
          const last = lastSeenRef.current.get(face.name) ?? 0;
          if (now - last >= CONFIG.FACE_COOLDOWN_SECONDS) {
            lastSeenRef.current.set(face.name, now);
            const promptResult = await getMemoryPrompt(face.name);
            if (!cancelled) {
              setPrompt(promptResult.prompt);
              setPromptName(promptResult.name);
            }
          }
        }
      } catch (error) {
        if (!cancelled) {
          setWorkerError(error instanceof Error ? error.message : String(error));
        }
      } finally {
        busyRef.current = false;
      }
    };

    const interval = setInterval(tick, CONFIG.RECOGNIZE_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [cameraRef, enabled]);

  return { faces, imageSize, prompt, promptName, workerError };
}
