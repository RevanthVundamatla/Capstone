"""
Continuous enhancement operations used by the DDPG actor.

The DDPG actor outputs a 7-dimensional continuous action vector:

    1. red_gain
    2. green_gain
    3. blue_gain
    4. gamma
    5. contrast
    6. saturation
    7. dehaze_strength

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

RANGES = {
    "red_gain": (0.7, 1.5),
    "green_gain": (0.7, 1.4),
    "blue_gain": (0.6, 1.2),
    "gamma": (0.6, 1.6),
    "contrast": (0.7, 1.5),
    "saturation": (0.7, 1.6),
    "dehaze_strength": (0.0, 1.0),
}


ACTION_ORDER = [
    "red_gain",
    "green_gain",
    "blue_gain",
    "gamma",
    "contrast",
    "saturation",
    "dehaze_strength",
]


# =========================================================
# ACTION SCALING
# =========================================================

def scale_action(action: np.ndarray) -> dict:
    """
    Convert DDPG action values from [-1, 1] into
    physical enhancement parameters.

    Example:

        -1 -> lower bound
         0 -> midpoint
        +1 -> upper bound

    Parameters
    ----------
    action:
        NumPy array containing 7 values in [-1, 1].

    Returns
    -------
    dict
        Named enhancement parameters.
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

    action = np.clip(
        action,
        -1.0,
        1.0
    )

    params = {}

    for name, value in zip(
        ACTION_ORDER,
        action
    ):

        lo, hi = RANGES[name]

        params[name] = float(
            lo
            + (float(value) + 1.0)
            * 0.5
            * (hi - lo)
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

    Input:
        RGB float32 image in [0, 1].

    Output:
        RGB float32 image in [0, 1].
    """

    out = img.copy()

    out[..., 0] *= r_gain
    out[..., 1] *= g_gain
    out[..., 2] *= b_gain

    return np.clip(
        out,
        0.0,
        1.0
    )


# =========================================================
# GAMMA CORRECTION
# =========================================================

def _gamma_correct(
    img: np.ndarray,
    gamma: float
) -> np.ndarray:
    """
    Apply gamma correction.

    Formula:

        output = input ^ (1 / gamma)

    gamma < 1:
        darker correction

    gamma > 1:
        brighter correction
    """

    gamma = max(
        float(gamma),
        1e-6
    )

    safe_img = np.clip(
        img,
        1e-6,
        1.0
    )

    out = safe_img ** (1.0 / gamma)

    return np.clip(
        out,
        0.0,
        1.0
    )


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

    mean = float(
        img.mean()
    )

    out = (
        (img - mean)
        * factor
        + mean
    )

    return np.clip(
        out,
        0.0,
        1.0
    )


# =========================================================
# SATURATION
# =========================================================

def _saturation(
    img: np.ndarray,
    factor: float
) -> np.ndarray:
    """
    Adjust HSV saturation.

    OpenCV internally works with uint8 HSV,
    so the image is temporarily converted to uint8.
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

    hsv[..., 1] = np.clip(
        hsv[..., 1],
        0.0,
        255.0
    )

    out = cv2.cvtColor(
        hsv.astype(np.uint8),
        cv2.COLOR_HSV2RGB
    )

    return (
        out.astype(np.float32)
        / 255.0
    )


# =========================================================
# DARK CHANNEL
# =========================================================

def _dark_channel(
    img: np.ndarray,
    patch_size: int = 15
) -> np.ndarray:
    """
    Calculate the dark channel prior.

    For every pixel:

        dark_channel =
            local minimum across RGB channels
            followed by morphological erosion.
    """

    min_channel = np.min(
        img,
        axis=2
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            patch_size,
            patch_size
        )
    )

    dark = cv2.erode(
        min_channel,
        kernel
    )

    return dark


# =========================================================
# DEHAZE
# =========================================================

def _dehaze(
    img: np.ndarray,
    strength: float,
    patch_size: int = 15,
    t_min: float = 0.1,
) -> np.ndarray:
    """
    Lightweight dark-channel-prior dehazing.

    The DDPG agent controls the strength continuously.

    strength = 0
        No dehazing.

    strength = 1
        Maximum dehazing within this implementation.
    """

    strength = float(
        np.clip(
            strength,
            0.0,
            1.0
        )
    )

    if strength <= 1e-3:
        return img.copy()

    dark = _dark_channel(
        img,
        patch_size
    )

    num_pixels = dark.size

    num_brightest = max(
        int(num_pixels * 0.001),
        1
    )

    flat_dark = dark.reshape(-1)

    flat_img = img.reshape(
        -1,
        3
    )

    # Find pixels with the brightest dark-channel values.
    idx = np.argpartition(
        flat_dark,
        -num_brightest
    )[-num_brightest:]

    atmospheric_light = (
        flat_img[idx].max(
            axis=0
        )
    )

    atmospheric_light = np.clip(
        atmospheric_light,
        0.3,
        1.0
    )

    # Normalize image by atmospheric light.
    norm_img = (
        img
        / (
            atmospheric_light
            + 1e-6
        )
    )

    dark_norm = _dark_channel(
        norm_img,
        patch_size
    )

    transmission = (
        1.0
        - strength * dark_norm
    )

    transmission = np.clip(
        transmission,
        t_min,
        1.0
    )

    transmission = transmission[
        ...,
        None
    ]

    recovered = (
        img
        - atmospheric_light
    ) / transmission + atmospheric_light

    return np.clip(
        recovered,
        0.0,
        1.0
    )


# =========================================================
# COMPLETE ENHANCEMENT PIPELINE
# =========================================================

def apply_enhancement(
    img: np.ndarray,
    action: np.ndarray
) -> np.ndarray:
    """
    Apply all enhancement operations selected by the DDPG actor.

    Parameters
    ----------
    img:
        RGB image with shape H x W x 3.
        float32 values in [0, 1].

    action:
        Seven-dimensional DDPG action in [-1, 1].

    Returns
    -------
    np.ndarray
        Enhanced RGB float32 image in [0, 1].
    """

    # -----------------------------------------------------
    # Validate input image
    # -----------------------------------------------------

    img = np.asarray(
        img,
        dtype=np.float32
    )

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

    img = np.clip(
        img,
        0.0,
        1.0
    )

    # -----------------------------------------------------
    # Convert DDPG action into parameters
    # -----------------------------------------------------

    params = scale_action(
        action
    )

    # -----------------------------------------------------
    # 1. White balance
    # -----------------------------------------------------

    out = _white_balance(
        img,
        params["red_gain"],
        params["green_gain"],
        params["blue_gain"],
    )

    # -----------------------------------------------------
    # 2. Dark-channel dehazing
    # -----------------------------------------------------

    out = _dehaze(
        out,
        params["dehaze_strength"]
    )

    # -----------------------------------------------------
    # 3. Gamma correction
    # -----------------------------------------------------

    out = _gamma_correct(
        out,
        params["gamma"]
    )

    # -----------------------------------------------------
    # 4. Contrast
    # -----------------------------------------------------

    out = _contrast(
        out,
        params["contrast"]
    )

    # -----------------------------------------------------
    # 5. Saturation
    # -----------------------------------------------------

    out = _saturation(
        out,
        params["saturation"]
    )

    # -----------------------------------------------------
    # Final safety clipping
    # -----------------------------------------------------

    return np.clip(
        out,
        0.0,
        1.0
    ).astype(np.float32)