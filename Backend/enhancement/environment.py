"""
DDPG environment for sequential underwater image restoration.

Each episode starts from a raw underwater image.
The DDPG agent applies several enhancement actions sequentially.

The reward is based on improvement in:
    - PSNR  : full-reference fidelity
    - SSIM  : full-reference structural similarity
    - UIQM  : no-reference underwater image quality
    - UCIQE : no-reference underwater image quality
    - cast  : color cast (lower is better)
"""

from __future__ import annotations

import numpy as np

from enhancement.ops import apply_enhancement
from enhancement.state import image_to_state
from metrics.metrics import psnr, ssim, uiqm, uciqe


# ---------------------------------------------------------
# Reward configuration
# ---------------------------------------------------------

REWARD_WEIGHTS = {
    "psnr": 0.30,
    "ssim": 0.25,
    "uiqm": 0.20,
    "uciqe": 0.15,
    "cast": 0.30,
}

PSNR_NORM = 30.0
UIQM_NORM = 4.0

# The uciqe() in metrics/metrics.py returns values around 3-4
# (not 0-1), so the old norm of 0.7 clipped every value to 1.0
# and the UCIQE reward was always zero.
# IMPORTANT: print uciqe() on ~20 UIEB images and set this to a
# value slightly above the largest typical result.
UCIQE_NORM = 6.0


def _clip01(value: float) -> float:
    """Keep a metric in the range [0, 1]."""
    return float(np.clip(value, 0.0, 1.0))


def _normalized_psnr(value: float) -> float:
    return _clip01(value / PSNR_NORM)


def _normalized_uiqm(value: float) -> float:
    return _clip01(value / UIQM_NORM)


def _normalized_uciqe(value: float) -> float:
    return _clip01(value / UCIQE_NORM)


def color_cast(img: np.ndarray) -> float:
    """
    Simple gray-world color-cast measure.

    Spread of the mean R, G, B values. 0 means perfectly balanced
    channels; larger values mean a stronger color cast.
    """

    channel_means = np.asarray(img, dtype=np.float32).reshape(-1, 3).mean(axis=0)

    return float(np.std(channel_means))


def compute_metrics(
    restored: np.ndarray,
    reference: np.ndarray | None = None,
) -> dict:

    info = {
        "uiqm": float(uiqm(restored)),
        "uciqe": float(uciqe(restored)),
        "cast": color_cast(restored),
    }

    if reference is not None:
        info["psnr"] = float(psnr(restored, reference))
        info["ssim"] = float(ssim(restored, reference))

    return info


def compute_reward(
    current_metrics: dict,
    previous_metrics: dict,
    reference: np.ndarray | None,
    action: np.ndarray | None = None,
) -> tuple[float, dict]:

    reward = 0.0

    # -----------------------------------------------------
    # Full-reference reward
    # -----------------------------------------------------

    if reference is not None:

        current_psnr = _normalized_psnr(current_metrics["psnr"])
        previous_psnr = _normalized_psnr(previous_metrics["psnr"])

        current_ssim = _clip01(current_metrics["ssim"])
        previous_ssim = _clip01(previous_metrics["ssim"])

        reward += REWARD_WEIGHTS["psnr"] * (current_psnr - previous_psnr)
        reward += REWARD_WEIGHTS["ssim"] * (current_ssim - previous_ssim)

    # -----------------------------------------------------
    # No-reference reward
    # -----------------------------------------------------

    current_uiqm = _normalized_uiqm(current_metrics["uiqm"])
    previous_uiqm = _normalized_uiqm(previous_metrics["uiqm"])

    current_uciqe = _normalized_uciqe(current_metrics["uciqe"])
    previous_uciqe = _normalized_uciqe(previous_metrics["uciqe"])

    reward += REWARD_WEIGHTS["uiqm"] * (current_uiqm - previous_uiqm)
    reward += REWARD_WEIGHTS["uciqe"] * (current_uciqe - previous_uciqe)

    # -----------------------------------------------------
    # Color-cast reward (positive when the cast shrinks)
    # -----------------------------------------------------

    cast_improvement = previous_metrics["cast"] - current_metrics["cast"]

    reward += REWARD_WEIGHTS["cast"] * cast_improvement

    # -----------------------------------------------------
    # Small action penalty
    # -----------------------------------------------------

    if action is not None:
        action_magnitude = float(np.mean(np.square(action)))
        reward -= 0.01 * action_magnitude

    info = {
        **current_metrics,
        "reward": float(reward),
    }

    return float(reward), info


class UnderwaterEnhanceEnv:
    """
    Sequential underwater enhancement environment.

    One episode consists of several enhancement actions applied
    to the same underwater image.
    """

    def __init__(self, max_steps: int = 3):

        self.max_steps = max_steps

        self.current = None
        self.reference = None

        self.t = 0

        self.previous_metrics = None

    def reset(
        self,
        raw_image: np.ndarray,
        reference_image: np.ndarray | None = None,
    ):

        self.current = raw_image.astype(np.float32).copy()
        self.reference = reference_image
        self.t = 0

        self.previous_metrics = compute_metrics(
            self.current,
            self.reference,
        )

        return image_to_state(self.current)

    def step(self, action: np.ndarray):

        self.current = apply_enhancement(
            self.current,
            action,
        )

        self.t += 1

        current_metrics = compute_metrics(
            self.current,
            self.reference,
        )

        reward, info = compute_reward(
            current_metrics=current_metrics,
            previous_metrics=self.previous_metrics,
            reference=self.reference,
            action=action,
        )

        self.previous_metrics = current_metrics

        done = self.t >= self.max_steps

        return (
            image_to_state(self.current),
            reward,
            done,
            info,
        )
