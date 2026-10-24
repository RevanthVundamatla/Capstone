"""
Image-quality metrics for underwater image restoration.

Metrics:
    PSNR  - Full-reference, higher is better.
    SSIM  - Full-reference, higher is better.
    UIQM  - No-reference underwater image quality.
    UCIQE - No-reference underwater image quality.

Input images:
    RGB
    float32/float64
    values in [0, 1]

PSNR and SSIM require a reference image.
UIQM and UCIQE do not require a reference image.
"""

from __future__ import annotations

import cv2
import numpy as np

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
)


# ============================================================
# COMMON HELPERS
# ============================================================

def _prepare_image(img: np.ndarray) -> np.ndarray:
    """
    Convert image to RGB float64 in [0, 1].
    """

    img = np.asarray(img)

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            "Expected an RGB image with shape HxWx3."
        )

    img = img.astype(np.float64)

    if img.max() > 1.0:
        img = img / 255.0

    return np.clip(img, 0.0, 1.0)


# ============================================================
# PSNR
# ============================================================

def psnr(
    restored: np.ndarray,
    reference: np.ndarray
) -> float:
    """
    Peak Signal-to-Noise Ratio.

    Higher is better.
    """

    restored = _prepare_image(restored)
    reference = _prepare_image(reference)

    if restored.shape != reference.shape:
        raise ValueError(
            "Restored and reference images must have "
            "the same dimensions."
        )

    return float(
        peak_signal_noise_ratio(
            reference,
            restored,
            data_range=1.0
        )
    )


# ============================================================
# SSIM
# ============================================================

def ssim(
    restored: np.ndarray,
    reference: np.ndarray
) -> float:
    """
    Structural Similarity Index.

    Higher is better.
    """

    restored = _prepare_image(restored)
    reference = _prepare_image(reference)

    if restored.shape != reference.shape:
        raise ValueError(
            "Restored and reference images must have "
            "the same dimensions."
        )

    return float(
        structural_similarity(
            reference,
            restored,
            channel_axis=2,
            data_range=1.0
        )
    )


# ============================================================
# UIQM
# ============================================================

def _uicm(img: np.ndarray) -> float:
    """
    Underwater Image Colorfulness Measure component.

    Uses RG and YB opponent-color statistics.
    """

    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]

    rg = r - g
    yb = 0.5 * (r + g) - b

    rg_mean = np.mean(rg)
    yb_mean = np.mean(yb)

    rg_std = np.std(rg)
    yb_std = np.std(yb)

    mean_term = np.sqrt(
        rg_mean ** 2 +
        yb_mean ** 2
    )

    std_term = np.sqrt(
        rg_std ** 2 +
        yb_std ** 2
    )

    return float(
        -0.0268 * mean_term +
        0.1586 * std_term
    )


def _uism(img: np.ndarray) -> float:
    """
    Underwater Image Sharpness Measure component.

    Uses Sobel edge strength.
    """

    gray = cv2.cvtColor(
        (img * 255.0).astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    ).astype(np.float64)

    sobel_x = cv2.Sobel(
        gray,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        gray,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    edge_strength = np.sqrt(
        sobel_x ** 2 +
        sobel_y ** 2
    )

    mean_edge = np.mean(
        edge_strength
    )

    return float(
        np.log(
            mean_edge + 1e-8
        )
    )


def _uiconm(
    img: np.ndarray,
    patch_size: int = 8
) -> float:
    """
    Underwater Image Contrast Measure component.
    """

    gray = cv2.cvtColor(
        (img * 255.0).astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    ).astype(np.float64)

    h, w = gray.shape

    values = []

    for y in range(
        0,
        h - patch_size + 1,
        patch_size
    ):

        for x in range(
            0,
            w - patch_size + 1,
            patch_size
        ):

            patch = gray[
                y:y + patch_size,
                x:x + patch_size
            ]

            minimum = np.min(patch)
            maximum = np.max(patch)

            denominator = (
                maximum +
                minimum +
                1e-8
            )

            if denominator > 0:

                contrast = (
                    maximum -
                    minimum
                ) / denominator

                values.append(
                    contrast
                )

    if not values:
        return 0.0

    return float(
        np.mean(values)
    )


def uiqm(
    img: np.ndarray,
    c1: float = 0.0282,
    c2: float = 0.2953,
    c3: float = 3.5753
) -> float:
    """
    Underwater Image Quality Measure.

    UIQM combines:
        UICM - colorfulness
        UISM - sharpness
        UIConM - contrast

    Higher values generally indicate better
    underwater perceptual quality.
    """

    img = _prepare_image(img)

    uicm_value = _uicm(img)
    uism_value = _uism(img)
    uiconm_value = _uiconm(img)

    value = (
        c1 * uicm_value +
        c2 * uism_value +
        c3 * uiconm_value
    )

    return float(value)


# ============================================================
# UCIQE
# ============================================================

def uciqe(
    img: np.ndarray,
    c1: float = 0.4680,
    c2: float = 0.2745,
    c3: float = 0.2576
) -> float:
    """
    Underwater Color Image Quality Evaluation.

    UCIQE is a linear combination of:
        - chroma variation
        - luminance contrast
        - mean saturation

    Chroma and luminance are normalised to [0, 1] so the score lands in
    the usual UCIQE range (roughly 0.3 - 0.8). The earlier version used
    the raw Lab chroma standard deviation, which produced values of
    about 2 - 10.

    Reference:
        Yang & Sowmya,
        "An Underwater Color Image Quality Evaluation Metric",
        IEEE Transactions on Image Processing, 2015.
    """

    img = _prepare_image(img)

    rgb8 = np.clip(
        img * 255.0,
        0,
        255
    ).astype(np.uint8)

    lab = cv2.cvtColor(
        rgb8,
        cv2.COLOR_RGB2LAB
    ).astype(np.float64)

    # OpenCV 8-bit LAB: L, a and b are all stored in [0, 255].
    L = lab[..., 0] / 255.0
    a = lab[..., 1] / 255.0
    b = lab[..., 2] / 255.0

    chroma = np.sqrt(
        a ** 2 +
        b ** 2
    )

    saturation = chroma / np.sqrt(
        chroma ** 2 +
        L ** 2 +
        1e-12
    )

    mean_chroma = float(
        np.mean(chroma)
    )

    # Spread of chroma around its mean
    chroma_spread = float(
        np.sqrt(
            np.mean(
                np.abs(
                    1.0 -
                    (mean_chroma / (chroma + 1e-12)) ** 2
                )
            )
        )
    )

    # Luminance contrast: 99th minus 1st percentile of L
    luminance_contrast = float(
        np.percentile(L, 99) -
        np.percentile(L, 1)
    )

    value = (
        c1 * chroma_spread +
        c2 * luminance_contrast +
        c3 * float(np.mean(saturation))
    )

    return float(value)


# ============================================================
# ALL METRICS
# ============================================================

def all_metrics(
    restored: np.ndarray,
    reference: np.ndarray | None = None
) -> dict:
    """
    Calculate all available metrics.

    Without reference:
        UIQM
        UCIQE

    With reference:
        PSNR
        SSIM
        UIQM
        UCIQE
    """

    restored = _prepare_image(
        restored
    )

    results = {
        "uiqm": uiqm(restored),
        "uciqe": uciqe(restored),
    }

    if reference is not None:

        reference = _prepare_image(
            reference
        )

        results["psnr"] = psnr(
            restored,
            reference
        )

        results["ssim"] = ssim(
            restored,
            reference
        )

    return results
