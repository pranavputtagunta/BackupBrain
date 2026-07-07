/**
 * HTTP client for the BackupBrain Vision Worker (FastAPI on the laptop).
 * The app never runs AI locally in Phase 2 — every inference goes through
 * these calls, which keeps the pipeline identical to Phase 1.
 */
import { CONFIG } from "../config";
import type {
  HealthResponse,
  PromptResponse,
  RecognizeResponse,
  TranscribeResponse,
} from "../types";

/** fetch with a timeout — RN fetch has no default timeout. */
async function fetchWithTimeout(url: string, init?: RequestInit): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), CONFIG.REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(`Worker returned ${response.status}: ${detail}`);
    }
    return response;
  } finally {
    clearTimeout(timer);
  }
}

/** Liveness + readiness of the worker (known people, whisper status). */
export async function getHealth(): Promise<HealthResponse> {
  const response = await fetchWithTimeout(`${CONFIG.WORKER_URL}/health`);
  return response.json();
}

/** Upload one camera snapshot; returns recognized faces + image dims. */
export async function recognize(photoUri: string): Promise<RecognizeResponse> {
  const form = new FormData();
  // React Native FormData accepts {uri, name, type} file descriptors.
  form.append("image", {
    uri: photoUri,
    name: "frame.jpg",
    type: "image/jpeg",
  } as unknown as Blob);
  const response = await fetchWithTimeout(`${CONFIG.WORKER_URL}/recognize`, {
    method: "POST",
    body: form,
  });
  return response.json();
}

/** Fetch the RAG memory prompt for a recognized person. */
export async function getMemoryPrompt(name: string): Promise<PromptResponse> {
  const response = await fetchWithTimeout(
    `${CONFIG.WORKER_URL}/memory-prompt/${encodeURIComponent(name)}`
  );
  return response.json();
}

/** Upload one recorded audio chunk (m4a) for Whisper transcription. */
export async function transcribe(audioUri: string): Promise<TranscribeResponse> {
  const form = new FormData();
  form.append("audio", {
    uri: audioUri,
    name: "chunk.m4a",
    type: "audio/m4a",
  } as unknown as Blob);
  const response = await fetchWithTimeout(`${CONFIG.WORKER_URL}/transcribe`, {
    method: "POST",
    body: form,
  });
  return response.json();
}
