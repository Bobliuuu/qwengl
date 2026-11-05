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


# dungeon scene: family f -> base colour, 5 brightness levels each (index 1 + 5 f + level), 0 = void
DUNGEON = [(200, 176, 132), (92, 78, 70), (190, 78, 56), (122, 134, 160), (70, 170, 92), (246, 196, 72)]


def dungeon_palette() -> np.ndarray:
    p = np.zeros((32, 3), dtype=np.uint8)
    p[0] = (10, 10, 16)
    for f, base in enumerate(DUNGEON):
        for lv in range(5):
            k = 0.32 + 0.17 * lv
            p[1 + 5 * f + lv] = np.minimum(255, np.array(base) * k * 1.0).astype(np.uint8)
    return p


def to_rgb_pal(fb: np.ndarray, pal: np.ndarray, scale: int = 16) -> np.ndarray:
    return np.kron(pal[fb], np.ones((scale, scale, 1), dtype=np.uint8))
