import colorsys
import numpy as np

HUES = [0.00, 0.08, 0.15, 0.33, 0.58, 0.78]          # one hue per cube face


def palette() -> np.ndarray:
    """colour index 0 = background, 1 + 5 f + level = face f at brightness level 0..4."""
    p = np.zeros((32, 3), dtype=np.uint8)
    p[0] = (14, 16, 24)
    for f in range(6):
        for lv in range(5):
            r, g, b = colorsys.hsv_to_rgb(HUES[f], 0.72, 0.30 + 0.17 * lv)
            p[1 + 5 * f + lv] = (int(r * 255), int(g * 255), int(b * 255))
    return p


def to_rgb(fb: np.ndarray, scale: int = 16) -> np.ndarray:
    img = palette()[fb]
    return np.kron(img, np.ones((scale, scale, 1), dtype=np.uint8))
