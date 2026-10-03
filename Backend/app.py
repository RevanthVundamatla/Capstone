"""
API server for the DeepSea Restore frontend.

Endpoints
---------
GET  /api/health           -> { status, checkpoint_loaded, device }
POST /api/restore           multipart form field 'image' -> restored image + metrics
POST /api/restore (json)    { "image": "data:image/png;base64,..." } also accepted

Run:
    python app.py                       # loads checkpoints/ddpg_actor_critic.pt if present
    CHECKPOINT=path/to/model.pt python app.py

CORS is wide open for local development — lock ALLOWED_ORIGINS down before
deploying this publicly.
"""
import base64
import io
import os

import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

from ddpg.agent import DDPGAgent
from inference import restore_image

CHECKPOINT_PATH = os.environ.get("CHECKPOINT", "checkpoints/ddpg_actor_critic.pt")
DEVICE = os.environ.get("DEVICE", "cpu")
MAX_STEPS = int(os.environ.get("MAX_STEPS", 3))
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})

agent = DDPGAgent(device=DEVICE)
checkpoint_loaded = False
if os.path.exists(CHECKPOINT_PATH):
    agent.load(CHECKPOINT_PATH)
    checkpoint_loaded = True
    print(f"Loaded trained weights from {CHECKPOINT_PATH}")
else:
    print(
        f"No checkpoint at {CHECKPOINT_PATH} — serving with randomly-initialized "
        "networks. Train first with train.py for real restorations."
    )


def decode_image(file_storage=None, data_url=None) -> np.ndarray:
    if file_storage is not None:
        pil_img = Image.open(file_storage.stream).convert("RGB")
    else:
        header, encoded = data_url.split(",", 1)
        pil_img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    arr = np.array(pil_img).astype(np.float32) / 255.0
    return arr


def encode_image(arr: np.ndarray) -> str:
    arr8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(arr8)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


@app.get("/api/health")
def health():
    return jsonify(
        status="ok",
        checkpoint_loaded=checkpoint_loaded,
        checkpoint_path=CHECKPOINT_PATH,
        device=DEVICE,
        max_steps=MAX_STEPS,
    )


@app.post("/api/restore")
def restore():
    try:
        if "image" in request.files:
            raw = decode_image(file_storage=request.files["image"])
        elif request.is_json and "image" in request.json:
            raw = decode_image(data_url=request.json["image"])
        else:
            return jsonify(error="Send an 'image' file (multipart) or a base64 data URL under 'image' (JSON)."), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify(error=f"Could not decode image: {exc}"), 400

    # Cap size for responsive inference; the state encoder itself works on 64x64.
    h, w = raw.shape[:2]
    max_dim = 1024
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        raw = cv2.resize(raw, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    restored, metrics, actions = restore_image(agent, raw, max_steps=MAX_STEPS)

    return jsonify(
        image=encode_image(restored),
        metrics={k: round(float(v), 4) for k, v in metrics.items()},
        steps=actions,
        checkpoint_loaded=checkpoint_loaded,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
