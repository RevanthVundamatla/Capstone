"""
API server for the DeepSea Restore frontend.

Endpoints
---------
GET  /api/health
    -> { status, checkpoint_loaded, device, max_steps }

POST /api/restore
    multipart form field 'image'
    -> restored image + metrics

POST /api/restore
    JSON:
    {
        "image": "data:image/png;base64,..."
    }
    -> restored image + metrics

Local:
    python app.py

Render / production:
    gunicorn app:app

Environment variables
---------------------
CHECKPOINT
    Path to the trained checkpoint.
    Default:
        checkpoints/ddpg_actor_critic.pt

DEVICE
    cpu / cuda
    Default:
        cpu

MAX_STEPS
    Maximum restoration steps.
    Default:
        3

ALLOWED_ORIGINS
    Comma-separated frontend origins, or "*" for development.
    Example:
        https://your-frontend.vercel.app

PORT
    Provided automatically by Render.
"""

import base64
import binascii
import io
import os

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

from ddpg.agent import DDPGAgent
from inference import restore_image


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT",
    "checkpoints/ddpg_actor_critic.pt",
)

DEVICE = os.environ.get("DEVICE", "cpu")

try:
    MAX_STEPS = int(os.environ.get("MAX_STEPS", "3"))
except ValueError:
    MAX_STEPS = 3

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")

MAX_IMAGE_DIM = 512


# ---------------------------------------------------------------------------
# Flask application & CORS Configuration
# ---------------------------------------------------------------------------

app = Flask(__name__)

if ALLOWED_ORIGINS.strip() == "*":
    cors_origins = "*"
else:
    cors_origins = [
        origin.strip()
        for origin in ALLOWED_ORIGINS.split(",")
        if origin.strip()
    ]

# Configure CORS fully to support preflight OPTIONS requests across all API routes
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": "*",
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
            "supports_credentials": True
        }
    },
)

# Guarantee header presence and dynamically allow Vercel previews on responses
@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin:
        if (
            cors_origins == "*" 
            or origin.endswith(".vercel.app") 
            or "localhost" in origin 
            or "127.0.0.1" in origin 
            or (isinstance(cors_origins, list) and origin in cors_origins)
        ):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization,X-Requested-With"
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


# ---------------------------------------------------------------------------
# Load DDPG agent
# ---------------------------------------------------------------------------

agent = DDPGAgent(device=DEVICE)

checkpoint_loaded = False

if os.path.exists(CHECKPOINT_PATH):
    try:
        agent.load(CHECKPOINT_PATH)
        checkpoint_loaded = True

        print(
            f"Loaded trained weights from: {CHECKPOINT_PATH}",
            flush=True,
        )

    except Exception as exc:
        print(
            f"WARNING: Checkpoint exists but could not be loaded: {exc}",
            flush=True,
        )

else:
    print(
        f"WARNING: No checkpoint found at {CHECKPOINT_PATH}. "
        "The API will run with randomly initialized networks. "
        "Train the model or provide a valid CHECKPOINT path.",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def decode_image(file_storage=None, data_url=None) -> np.ndarray:
    """
    Decode an uploaded image or base64 data URL into an RGB float32 NumPy array.

    Returns:
        np.ndarray:
            Shape: (height, width, 3)
            Values: 0.0 - 1.0
    """

    if file_storage is not None:
        pil_img = Image.open(file_storage.stream).convert("RGB")

    elif data_url is not None:
        if not isinstance(data_url, str):
            raise ValueError("Image data must be a string.")

        if "," not in data_url:
            raise ValueError(
                "Invalid image data URL. Expected "
                "'data:image/...;base64,...'."
            )

        header, encoded = data_url.split(",", 1)

        if "base64" not in header.lower():
            raise ValueError("Image data URL must contain base64 data.")

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

    arr = np.array(
        pil_img,
        dtype=np.float32,
    ) / 255.0

    return arr


def encode_image(arr: np.ndarray) -> str:
    """
    Encode an RGB NumPy array as a PNG data URL.
    """

    arr8 = np.clip(
        arr * 255.0,
        0,
        255,
    ).astype(np.uint8)

    pil_img = Image.fromarray(arr8)

    buf = io.BytesIO()

    pil_img.save(
        buf,
        format="PNG",
    )

    b64 = base64.b64encode(
        buf.getvalue()
    ).decode("ascii")

    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """
    Basic service information.
    """

    return jsonify(
        service="DeepSea Restore API",
        status="running",
        health="/api/health",
        restore="/api/restore",
    )


@app.get("/api/health")
def health():
    """
    Health-check endpoint.

    Useful for Render and frontend connectivity testing.
    """

    return jsonify(
        status="ok",
        checkpoint_loaded=checkpoint_loaded,
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
        max_steps=MAX_STEPS,
    )


@app.post("/api/restore")
def restore():
    """
    Restore an uploaded image.

    Supports:

    1. Multipart:
        POST /api/restore
        field name = image

    2. JSON:
        {
            "image": "data:image/png;base64,..."
        }
    """

    # ---------------------------------------------------------------
    # Decode image
    # ---------------------------------------------------------------

    try:

        if "image" in request.files:
            raw = decode_image(
                file_storage=request.files["image"]
            )

        elif request.is_json:
            data = request.get_json(silent=True)

            if not data or "image" not in data:
                return jsonify(
                    error=(
                        "JSON body must contain an "
                        "'image' field."
                    )
                ), 400

            raw = decode_image(
                data_url=data["image"]
            )

        else:
            return jsonify(
                error=(
                    "Send an 'image' file using multipart/form-data "
                    "or a base64 data URL using JSON."
                )
            ), 400

    except Exception as exc:
        return jsonify(
            error=f"Could not decode image: {exc}"
        ), 400

    # ---------------------------------------------------------------
    # Resize large images
    # ---------------------------------------------------------------

    try:

        height, width = raw.shape[:2]

        if max(height, width) > MAX_IMAGE_DIM:

            scale = MAX_IMAGE_DIM / max(
                height,
                width,
            )

            new_width = max(
                1,
                int(width * scale),
            )

            new_height = max(
                1,
                int(height * scale),
            )

            raw = cv2.resize(
                raw,
                (new_width, new_height),
                interpolation=cv2.INTER_AREA,
            )

    except Exception as exc:
        return jsonify(
            error=f"Could not process image dimensions: {exc}"
        ), 400

    # ---------------------------------------------------------------
    # Run restoration
    # ---------------------------------------------------------------

    try:

        restored, metrics, actions = restore_image(
            agent,
            raw,
            max_steps=MAX_STEPS,
        )

    except Exception as exc:

        app.logger.exception(
            "Image restoration failed"
        )

        return jsonify(
            error=f"Image restoration failed: {exc}"
        ), 500

    # ---------------------------------------------------------------
    # Encode response
    # ---------------------------------------------------------------

    try:

        encoded_image = encode_image(
            restored
        )

        clean_metrics = {
            key: round(float(value), 4)
            for key, value in metrics.items()
        }

    except Exception as exc:

        app.logger.exception(
            "Could not encode restored image"
        )

        return jsonify(
            error=f"Could not encode restored image: {exc}"
        ), 500

    return jsonify(
        image=encoded_image,
        metrics=clean_metrics,
        steps=actions,
        checkpoint_loaded=checkpoint_loaded,
    )


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Local development
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            "5000",
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
