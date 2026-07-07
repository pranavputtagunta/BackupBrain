/**
 * Transcription loop hook — the Phase 2 counterpart of the Phase 1 voice
 * thread. Records fixed-length audio chunks and sends each to the vision
 * worker's /transcribe endpoint (Whisper + VAD run server-side).
 */
import { useEffect, useState } from "react";
import {
  RecordingPresets,
  setAudioModeAsync,
  useAudioRecorder,
} from "expo-audio";

import { transcribe } from "../api/visionWorker";
import { CONFIG } from "../config";

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

export function useTranscription(enabled: boolean): string {
  const recorder = useAudioRecorder(RecordingPresets.LOW_QUALITY);
  const [transcript, setTranscript] = useState("");

  useEffect(() => {
    if (!enabled) {
      return;
    }
    let cancelled = false;

    const loop = async () => {
      await setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });
      while (!cancelled) {
        try {
          await recorder.prepareToRecordAsync();
          recorder.record();
          await sleep(CONFIG.AUDIO_CHUNK_SECONDS * 1000);
          await recorder.stop();
          const uri = recorder.uri;
          if (cancelled || !uri) {
            continue;
          }
          const result = await transcribe(uri);
          if (!cancelled && result.voiced && result.text) {
            setTranscript(result.text);
          }
        } catch {
          // Worker offline or whisper still loading — retry after a pause
          // rather than spamming the network.
          await sleep(3000);
        }
      }
    };
    loop();

    return () => {
      cancelled = true;
    };
  }, [enabled, recorder]);

  return transcript;
}
