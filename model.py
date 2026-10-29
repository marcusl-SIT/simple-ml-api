"""
Download MobileNetV2 and the ImageNet class names.

Run this before starting or deploying the API:

    python model.py
"""

import json
import urllib.request
from pathlib import Path

import tensorflow as tf


MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "mobilenet_v2.keras"
CLASS_NAMES_PATH = MODEL_DIR / "imagenet_class_index.json"

CLASS_NAMES_URL = (
    "https://storage.googleapis.com/"
    "download.tensorflow.org/data/imagenet_class_index.json"
)


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading pretrained MobileNetV2...")

    model = tf.keras.applications.MobileNetV2(
        weights="imagenet",
        include_top=True,
    )

    model.save(MODEL_PATH)

    print(f"Saved TensorFlow model to {MODEL_PATH}")

    print("Downloading ImageNet class names...")

    with urllib.request.urlopen(
        CLASS_NAMES_URL,
        timeout=60,
    ) as response:
        class_names = json.load(response)

    CLASS_NAMES_PATH.write_text(
        json.dumps(class_names, indent=2),
        encoding="utf-8",
    )

    print(f"Saved class names to {CLASS_NAMES_PATH}")


if __name__ == "__main__":
    main()