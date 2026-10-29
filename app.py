"""
FastAPI image-classification API using TensorFlow/Keras MobileNetV2.
"""

import io
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "mobilenet_v2.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "imagenet_class_index.json"

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
IMAGE_SIZE = (224, 224)
TOP_K = 5


def load_model() -> tf.keras.Model | None:
    """Load the trained TensorFlow model once at startup."""
    if not MODEL_PATH.exists():
        return None

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )


def load_class_names() -> dict[str, list[str]] | None:
    """Load ImageNet class IDs and human-readable class names."""
    if not CLASS_NAMES_PATH.exists():
        return None

    with CLASS_NAMES_PATH.open(encoding="utf-8") as file:
        return json.load(file)


# These objects are loaded once when the Python process starts.
model = load_model()
class_names = load_class_names()


app = FastAPI(
    title="TensorFlow Image Classification API",
    description=(
        "Classifies uploaded images using a pretrained "
        "TensorFlow MobileNetV2 model."
    ),
    version="1.0.0",
)


class Classification(BaseModel):
    class_id: int
    synset_id: str
    class_name: str
    probability: float


class PredictionOutput(BaseModel):
    filename: str
    model: str
    predictions: list[Classification]


@app.get("/")
def root():
    return {
        "status": "ok",
        "model": "MobileNetV2",
        "model_loaded": model is not None,
        "class_names_loaded": class_names is not None,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not found at {MODEL_PATH}",
        )

    if class_names is None:
        raise HTTPException(
            status_code=503,
            detail=f"Class names not found at {CLASS_NAMES_PATH}",
        )

    return {
        "status": "healthy",
        "model": "MobileNetV2",
    }


def prepare_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode and preprocess an uploaded image.

    MobileNetV2 expects:
    - RGB images;
    - 224 x 224 pixels;
    - a batch dimension;
    - MobileNetV2-specific pixel normalization.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        ) from error

    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    # [height, width, channels] -> [1, height, width, channels]
    image_batch = np.expand_dims(image_array, axis=0)

    return tf.keras.applications.mobilenet_v2.preprocess_input(
        image_batch
    )


@app.post("/predict", response_model=PredictionOutput)
def predict(
    image: UploadFile = File(
        ...,
        description="A JPEG, PNG, or WebP image.",
    ),
):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    if class_names is None:
        raise HTTPException(
            status_code=503,
            detail="ImageNet class names not loaded",
        )

    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported image type. "
                "Upload a JPEG, PNG, or WebP image."
            ),
        )

    image_bytes = image.file.read(MAX_FILE_SIZE + 1)

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="The uploaded image exceeds the 5 MB limit.",
        )

    image_batch = prepare_image(image_bytes)

    # Output shape: [1, 1000]
    probabilities = model.predict(
        image_batch,
        verbose=0,
    )[0]

    # Sort class IDs from highest to lowest probability.
    top_class_ids = np.argsort(probabilities)[-TOP_K:][::-1]

    predictions = []

    for class_id in top_class_ids:
        class_id = int(class_id)
        synset_id, class_name = class_names[str(class_id)]

        predictions.append(
            Classification(
                class_id=class_id,
                synset_id=synset_id,
                class_name=class_name,
                probability=round(
                    float(probabilities[class_id]),
                    6,
                ),
            )
        )

    return PredictionOutput(
        filename=image.filename or "uploaded-image",
        model="MobileNetV2",
        predictions=predictions,
    )