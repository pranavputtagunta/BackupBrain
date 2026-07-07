"""Helper script to enroll a known face.

Usage:
    python setup_faces.py <name> <path/to/photo.jpg>

Verifies the photo contains exactly one detectable face, copies it into
`data/known_faces/<name>/`, and creates a starter memory file at
`data/memories/<name>.json` (if one doesn't already exist).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys

import face_recognition

import config

logger = logging.getLogger(__name__)

STARTER_MEMORY = {
    "name": "",
    "relationship": "",
    "last_interaction": "",
    "facts": [],
    "recent_conversations": [],
}


def enroll_face(name: str, photo_path: str) -> None:
    """Validate, copy the photo, and create the starter memory file."""
    name = name.strip().lower()
    if not os.path.isfile(photo_path):
        logger.error("Photo not found: %s", photo_path)
        sys.exit(1)

    image = face_recognition.load_image_file(photo_path)
    encodings = face_recognition.face_encodings(image)
    if len(encodings) == 0:
        logger.error("No face detected in %s — use a clearer, front-facing photo", photo_path)
        sys.exit(1)
    if len(encodings) > 1:
        logger.error("Multiple faces detected in %s — use a photo with only one person", photo_path)
        sys.exit(1)

    person_dir = os.path.join(config.KNOWN_FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    existing = len(os.listdir(person_dir))
    ext = os.path.splitext(photo_path)[1].lower() or ".jpg"
    dest = os.path.join(person_dir, f"photo{existing + 1}{ext}")
    shutil.copy2(photo_path, dest)
    logger.info("Saved face photo to %s", dest)

    os.makedirs(config.MEMORIES_DIR, exist_ok=True)
    memory_path = os.path.join(config.MEMORIES_DIR, f"{name}.json")
    if os.path.isfile(memory_path):
        logger.info("Memory file already exists at %s — leaving it alone", memory_path)
    else:
        memory = dict(STARTER_MEMORY)
        memory["name"] = name.capitalize()
        with open(memory_path, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2)
        logger.info("Created starter memory file at %s — edit it to add real facts", memory_path)

    logger.info("Done. '%s' will be recognized on the next run of main.py", name)


def main() -> None:
    """Parse CLI args and enroll the face."""
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    parser = argparse.ArgumentParser(description="Enroll a known face for BackupBrain")
    parser.add_argument("name", help="Person's name (used as folder/file name)")
    parser.add_argument("photo", help="Path to a photo containing exactly one face")
    args = parser.parse_args()
    enroll_face(args.name, args.photo)


if __name__ == "__main__":
    main()
