"""
Continuous enhancement operations used by the DDPG actor.

The DDPG actor outputs a 7-dimensional continuous action vector:

    1. red_gain
    2. green_gain
    3. blue_gain
    4. gamma
    5. contrast
    6. saturation
    7. local_contrast   (CLAHE strength)

The actor outputs values in [-1, 1].
Those values are converted into physically meaningful
enhancement parameters using RANGES below.

All image operations use:

    H x W x 3
    RGB
    float32
    values in [0, 1]

IMPORTANT:
This module must not import enhancement.environment.
environment.py imports apply_enhancement() from this file,
so importing the environment here would create a circular import.
"""

from __future__ import annotations

import cv2
import numpy as np


# =========================================================
# ACTION RANGES
# =========================================================
# Notes on the design:
#   - saturation never goes below 1.0, so the agent cannot
#     "win" PSNR by washing the image out.
#   - red gain has a high upper bound because red light is
#     absorbed first underwater.
#   - blue gain can go well below 1.0 to remove blue casts.

RANGES = {
    "red_gain": (0.9, 1.8),
    "green_gain": (0.85, 1.2),
    "blue_gain": (0.6, 1.1),
    "gamma": (0.7, 1.4),
    "contrast": (0.9, 1.4),
    "saturation": (1.0, 1.6),
    "local_contrast": (0.0, 1.0),
}


ACTION_ORDER = [
    "red_gain",
    "green_gain",
    "blue_gain",
    "gamma",
    "contrast",
    "saturation",
    "local_contrast",
]


# =========================================================
# ACTION SCALING
# =========================================================

def scale_action(action: np.ndarray) -> dict:
    """
    Convert DDPG action values from [-1, 1] into
    physical enhancement parameters.

        -1 -> lower bound
         0 -> midpoint
        +1 -> upper bound
    """

    action = np.asarray(
        action,
        dtype=np.float32
    ).reshape(-1)

    if len(action) != len(ACTION_ORDER):
        raise ValueError(
            f"Expected {len(ACTION_ORDER)} action values, "
            f"but received {len(action)}."
        )

    action = np.clip(action, -1.0, 1.0)

    params = {}

    for name, value in zip(ACTION_ORDER, action):

        lo, hi = RANGES[name]

        params[name] = float(
            lo + (float(value) + 1.0) * 0.5 * (hi - lo)
        )

    return params


# =========================================================
# WHITE BALANCE
# =========================================================

def _white_balance(
    img: np.ndarray,
    r_gain: float,
    g_gain: float,
    b_gain: float,
) -> np.ndarray:
    """
    Apply independent RGB channel gains.
    """

    out = img.copy()

    out[..., 0] *= r_gain
    out[..., 1] *= g_gain
    out[..., 2] *= b_gain

    return np.clip(out, 0.0, 1.0)


# =========================================================
# GAMMA CORRECTION
# =========================================================

def _gamma_correct(
    img: np.ndarray,
    gamma: float
) -> np.ndarray:
    """
    output = input ^ (1 / gamma)

    gamma > 1: brighter
    gamma < 1: darker
    """

    gamma = max(float(gamma), 1e-6)

    safe_img = np.clip(img, 1e-6, 1.0)

    out = safe_img ** (1.0 / gamma)

    return np.clip(out, 0.0, 1.0)


# =========================================================
# CONTRAST
# =========================================================

def _contrast(
    img: np.ndarray,
    factor: float
) -> np.ndarray:
    """
    Adjust global image contrast around the image mean.
    """

    factor = float(factor)

    mean = float(img.mean())

    out = (img - mean) * factor + mean

    return np.clip(out, 0.0, 1.0)


# =========================================================
# SATURATION
# =========================================================

def _saturation(
    img: np.ndarray,
    factor: float
) -> np.ndarray:
    """
    Adjust HSV saturation.

    OpenCV works with uint8 HSV, so the image is
    temporarily converted to uint8.
    """

    factor = float(factor)

    img_uint8 = np.clip(
        img * 255.0,
        0,
        255
    ).astype(np.uint8)

    hsv = cv2.cvtColor(
        img_uint8,
        cv2.COLOR_RGB2HSV
    ).astype(np.float32)

    hsv[..., 1] *= factor

    hsv[..., 1] = np.clip(hsv[..., 1], 0.0, 255.0)

    out = cv2.cvtColor(
        hsv.astype(np.uint8),
        cv2.COLOR_HSV2RGB
    )

    return out.astype(np.float32) / 255.0


# =========================================================
# LOCAL CONTRAST (CLAHE)
# =========================================================

def _local_contrast(
    img: np.ndarray,
    strength: float
) -> np.ndarray:
    """
    CLAHE on the L channel, blended with the original by strength.

    strength = 0: no change
    strength = 1: full CLAHE

    This replaces the old dark-channel dehaze step, which produced
    blocky artifacts and does not suit underwater scenes.
    """

    strength = float(np.clip(strength, 0.0, 1.0))

    if strength <= 1e-3:
        return img.copy()

    rgb8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(rgb8, cv2.COLOR_RGB2LAB)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l_original = lab[..., 0].astype(np.float32)
    l_equalized = clahe.apply(lab[..., 0]).astype(np.float32)

    blended = (1.0 - strength) * l_original + strength * l_equalized

    lab[..., 0] = np.clip(blended, 0, 255).astype(np.uint8)

    out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return out.astype(np.float32) / 255.0


# =========================================================
# COMPLETE ENHANCEMENT PIPELINE
# =========================================================

def apply_enhancement(
    img: np.ndarray,
    action: np.ndarray
) -> np.ndarray:
    """
    Apply all enhancement operations selected by the DDPG actor.

    img:
        RGB image, H x W x 3, float32 in [0, 1].

    action:
        Seven-dimensional DDPG action in [-1, 1].
    """

    img = np.asarray(img, dtype=np.float32)

    if img.ndim != 3:
        raise ValueError(
            f"Expected image with 3 dimensions "
            f"(H, W, C), got shape {img.shape}"
        )

    if img.shape[2] != 3:
        raise ValueError(
            f"Expected RGB image with 3 channels, "
            f"got shape {img.shape}"
        )

    img = np.clip(img, 0.0, 1.0)

    params = scale_action(action)

    # 1. White balance
    out = _white_balance(
        img,
        params["red_gain"],
        params["green_gain"],
        params["blue_gain"],
    )

    # 2. Gamma
    out = _gamma_correct(out, params["gamma"])

    # 3. Local contrast (CLAHE)
    out = _local_contrast(out, params["local_contrast"])

    # 4. Global contrast
    out = _contrast(out, params["contrast"])

    # 5. Saturation
    out = _saturation(out, params["saturation"])

    return np.clip(out, 0.0, 1.0).astype(np.float32)
