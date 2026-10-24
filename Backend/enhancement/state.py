"""State representation: 'compact numeric summary of the image' (workflow slide, step 3)."""
import cv2
import numpy as np

STATE_SIZE = 64  # resized square side fed into the CNN state encoder


def image_to_state(img: np.ndarray, size: int = STATE_SIZE) -> np.ndarray:
    """HxWx3 float32 [0,1] -> (3, size, size) float32 [0,1], channel-first for torch."""
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return np.transpose(resized, (2, 0, 1)).astype(np.float32)
