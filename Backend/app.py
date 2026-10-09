"""
API server for DeepSea Restore.

Endpoints
---------
GET  /api/health
GET  /api/training-history
POST /api/restore

POST /api/restore accepts:
    Multipart:
        image       (required)
        reference   (optional)

    JSON:
        {"image": "data:image/png;base64,..."}

Run locally:
    python app.py

Production:
    gunicorn app:app --workers 1 --threads 2 --timeout 300 --bind 0.0.0.0:$PORT
"""

import base64
import binascii
import io
import json
import os
import time

import cv2
import numpy as np
import torch

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

from ddpg.agent import DDPGAgent
from inference import restore_image


# =========================================================
# CONFIGURATION
# =========================================================

torch.set_num_threads(1)

CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT",
    "checkpoints/ddpg_actor_critic.pt",
)

DEVICE = os.environ.get("DEVICE", "cpu")

try:
    MAX_STEPS = int(os.environ.get("MAX_STEPS", "3"))
except ValueError:
    MAX_STEPS = 3

try:
    MAX_IMAGE_DIM = int(os.environ.get("MAX_IMAGE_DIM", "256"))
except ValueError:
    MAX_IMAGE_DIM = 256

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "*",
)

TRAINING_HISTORY_PATH = os.environ.get(
    "TRAINING_HISTORY_PATH",
    "training_history.json",
)


# =========================================================
# FLASK AND CORS
# =========================================================

app = Flask(__name__)

if ALLOWED_ORIGINS.strip() == "*":
    cors_origins = "*"
else:
    cors_origins = [
        origin.strip().rstrip("/")
        for origin in ALLOWED_ORIGINS.split(",")
        if origin.strip()
    ]

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": cors_origins,
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": [
                "Content-Type",
                "Authorization",
                "X-Requested-With",
            ],
        }
    },
)


# =========================================================
# LOAD DDPG MODEL
# =========================================================

agent = DDPGAgent(device=DEVICE)

checkpoint_loaded = False

if os.path.exists(CHECKPOINT_PATH):
    try:
        agent.load(CHECKPOINT_PATH)
        checkpoint_loaded = True

        print(
            f"Loaded trained weights: {CHECKPOINT_PATH}",
            flush=True,
        )

    except Exception as exc:
        print(
            f"WARNING: Could not load checkpoint: {exc}",
            flush=True,
        )
else:
    print(
        f"WARNING: Checkpoint not found: {CHECKPOINT_PATH}",
        flush=True,
    )


# =========================================================
# IMAGE HELPERS
# =========================================================

def decode_image(file_storage=None, data_url=None):
    """Decode an uploaded file or base64 data URL into RGB float32."""

    if file_storage is not None:
        pil_img = Image.open(
            file_storage.stream
        ).convert("RGB")

    elif data_url is not None:

        if not isinstance(data_url, str):
            raise ValueError("Image data must be a string.")

        if "," not in data_url:
            raise ValueError("Invalid image data URL.")

        header, encoded = data_url.split(",", 1)

        if "base64" not in header.lower():
            raise ValueError("Image data must use base64 encoding.")

        try:
            image_bytes = base64.b64decode(
                encoded,
                validate=True,
            )
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Invalid base64 image data.") from exc

        if not image_bytes:
            raise ValueError("Image data is empty.")

        pil_img = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    else:
        raise ValueError("No image was provided.")

    return np.asarray(
        pil_img,
        dtype=np.float32,
    ) / 255.0


def encode_image(arr):
    """Encode an RGB image as a PNG data URL."""

    arr8 = np.clip(
        arr * 255.0,
        0,
        255,
    ).astype(np.uint8)

    pil_img = Image.fromarray(arr8)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return f"data:image/png;base64,{encoded}"


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root():
    return jsonify(
        service="DeepSea Restore API",
        status="running",
        health="/api/health",
        restore="/api/restore",
        training_history="/api/training-history",
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/api/health")
def health():
    return jsonify(
        status="ok",
        checkpoint_loaded=checkpoint_loaded,
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
        max_steps=MAX_STEPS,
        max_image_dim=MAX_IMAGE_DIM,
    )


# =========================================================
# TRAINING HISTORY API
# =========================================================

@app.get("/api/training-history")
def training_history():
    """
    Return saved epoch-wise validation metrics.

    The JSON file is generated by train.py.
    It should contain epoch numbers and actual measured metrics.
    """

    if not os.path.isfile(TRAINING_HISTORY_PATH):
        return jsonify(
            error="Training history not found.",
            message=(
                "Run train.py to generate training_history.json "
                "and make the file available to the backend."
            ),
            history_path=TRAINING_HISTORY_PATH,
        ), 404

    try:
        with open(
            TRAINING_HISTORY_PATH,
            "r",
            encoding="utf-8",
        ) as file:
            history = json.load(file)

        if not isinstance(history, (dict, list)):
            return jsonify(
                error="Training history must be a JSON object or array."
            ), 500

        return jsonify(history)

    except json.JSONDecodeError:
        app.logger.exception("Invalid training history JSON")

        return jsonify(
            error="Training history contains invalid JSON."
        ), 500

    except OSError:
        app.logger.exception("Could not read training history")

        return jsonify(
            error="Could not read training history file."
        ), 500


# =========================================================
# IMAGE RESTORATION API
# =========================================================

@app.post("/api/restore")
def restore():
    """Restore an underwater image and return metrics."""

    start_time = time.time()
    reference = None

    # -----------------------------------------------------
    # Decode uploaded input
    # -----------------------------------------------------

    try:
        if "image" in request.files:

            raw = decode_image(
                file_storage=request.files["image"]
            )

            if "reference" in request.files:
                try:
                    reference = decode_image(
                        file_storage=request.files["reference"]
                    )
                except Exception as exc:
                    return jsonify(
                        error=f"Could not decode reference: {exc}"
                    ), 400

        elif request.is_json:

            data = request.get_json(silent=True)

            if not data or "image" not in data:
                return jsonify(
                    error="JSON body must contain an image field."
                ), 400

            raw = decode_image(
                data_url=data["image"]
            )

        else:
            return jsonify(
                error=(
                    "Send an image file using multipart/form-data "
                    "or a base64 image using JSON."
                )
            ), 400

    except Exception as exc:
        return jsonify(
            error=f"Could not decode image: {exc}"
        ), 400

    # -----------------------------------------------------
    # Resize images
    # -----------------------------------------------------

    try:
        height, width = raw.shape[:2]

        if MAX_IMAGE_DIM > 0 and max(height, width) > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / max(height, width)

            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))

            raw = cv2.resize(
                raw,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA,
            )

        if reference is not None:
            reference = cv2.resize(
                reference,
                (raw.shape[1], raw.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

    except Exception as exc:
        return jsonify(
            error=f"Could not process image dimensions: {exc}"
        ), 400

    print(
        f"[restore] Image: {raw.shape[1]}x{raw.shape[0]}, "
        f"reference={'yes' if reference is not None else 'no'}",
        flush=True,
    )

    # -----------------------------------------------------
    # Restore image
    # -----------------------------------------------------

    try:
        with torch.no_grad():
            restored, metrics, actions = restore_image(
                agent,
                raw,
                max_steps=MAX_STEPS,
                reference=reference,
            )

    except Exception as exc:
        app.logger.exception("Image restoration failed")

        return jsonify(
            error=f"Image restoration failed: {exc}"
        ), 500

    # -----------------------------------------------------
    # Return restored image and metrics
    # -----------------------------------------------------

    try:
        encoded_image = encode_image(restored)

        clean_metrics = {
            key: round(float(value), 4)
            for key, value in metrics.items()
            if value is not None
            and np.isfinite(float(value))
        }

    except Exception as exc:
        app.logger.exception("Could not encode restoration output")

        return jsonify(
            error=f"Could not encode restored image: {exc}"
        ), 500

    elapsed = round(time.time() - start_time, 3)

    print(
        f"[restore] Completed in {elapsed}s",
        flush=True,
    )

    return jsonify(
        image=encoded_image,
        metrics=clean_metrics,
        steps=actions,
        checkpoint_loaded=checkpoint_loaded,
        processing_time_seconds=elapsed,
    )


# =========================================================
# ERROR HANDLERS
# =========================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify(
        error="Endpoint not found."
    ), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify(
        error="HTTP method not allowed."
    ), 405


@app.errorhandler(500)
def internal_server_error(error):
    return jsonify(
        error="Internal server error."
    ), 500


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
