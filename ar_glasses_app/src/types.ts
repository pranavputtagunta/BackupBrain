/** Shared types mirroring the Vision Worker's response models (main.py). */

/** Face bounding box in uploaded-image pixels: (top, right, bottom, left). */
export type FaceBox = [number, number, number, number];

export interface Face {
  name: string;
  box: FaceBox;
  confidence: number;
}

export interface RecognizeResponse {
  faces: Face[];
  image_width: number;
  image_height: number;
}

export interface PromptResponse {
  name: string;
  prompt: string;
  cached: boolean;
}

export interface TranscribeResponse {
  text: string;
  voiced: boolean;
}

export interface HealthResponse {
  status: string;
  known_people: string[];
  whisper_ready: boolean;
  whisper_error: string | null;
}

/** Pixel dimensions of the uploaded snapshot, for overlay scaling. */
export interface ImageSize {
  width: number;
  height: number;
}
