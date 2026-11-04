"""Cube mesh, the quantised model-view uniform, and the integer identities the rasteriser relies on."""
from __future__ import annotations

import numpy as np

SCALE = 5            # half-size of the cube in pixels: R = round(SCALE * rotation)
S = 30               # framebuffer is S x S pixels
C0 = S // 2          # screen centre: pixel i has centred coordinate i - C0 (in -15..14, fits a 5-bit weight)
GRP = 6              # pixels per raster group (one EVAL kernel run)

# vertex v has corner signs s_j = 2*bit_j(v) - 1
SGN = np.array([[2 * ((v >> j) & 1) - 1 for j in range(3)] for v in range(8)], dtype=np.int64)


def _faces():
    """6 quads (axis k, sign) -> 4 corners in a fixed cyclic order; 12 triangles (a, b, c)."""
    faces = []
    for k in range(3):
        j, l = [x for x in range(3) if x != k]
        for sg in (-1, 1):
            corner = lambda bj, bl: (((sg + 1) // 2) << k) | (bj << j) | (bl << l)
            faces.append((k, sg, [corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1)]))
    return faces


FACES = _faces()                                                   # index f = 2*k + (sg > 0)
TRIS = [(f, (q[0], q[1], q[2])) for f, (_, _, q) in enumerate(FACES)] + \
       [(f, (q[0], q[2], q[3])) for f, (_, _, q) in enumerate(FACES)]
TRIS.sort(key=lambda t: t[0])                                      # triangles of a face stay adjacent


def rot_matrix(ax: float, ay: float, az: float) -> np.ndarray:
    cx, sx, cy, sy, cz, sz = np.cos(ax), np.sin(ax), np.cos(ay), np.sin(ay), np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def quantise(rot: np.ndarray) -> np.ndarray:
    """The model-view uniform: a 3x3 integer matrix (rows = screen x, y, depth)."""
    R = np.rint(SCALE * rot).astype(np.int64)
    assert np.abs(R).max() <= SCALE
    return R


def minors(R: np.ndarray) -> np.ndarray:
    """m = Rx x Ry (the 'normal-matrix' uniform).  Binet-Cauchy: X_a*Y_b - X_b*Y_a = m . (s_a x s_b)."""
    m = np.cross(R[0], R[1]).astype(np.int64)
    assert np.abs(m).max() <= 31, f"minor out of 5-bit activation range: {m}"
    return m


def split(v):
    """Signed ints -> (positive part, negative part) activation lanes."""
    v = np.asarray(v)
    return np.maximum(v, 0), np.maximum(-v, 0)


def uniform_lanes(R: np.ndarray):
    """The per-frame uniform as activation lanes: rows of R and the minors, each split into +/- lanes."""
    rows = [np.stack(split(R[i]), 1).reshape(-1) for i in range(3)]     # [R_i0+, R_i0-, R_i1+, ...]
    cols = [np.stack(split(R[:, k]), 1).reshape(-1) for k in range(3)]  # column k = face-normal axis k
    mm = np.stack(split(minors(R)), 1).reshape(-1)                      # [m0+, m0-, m1+, ...]
    return rows, cols, mm
