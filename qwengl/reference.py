"""Independent integer software rasteriser: same quantised geometry, written with ordinary arithmetic
(explicit cross products, comparisons, no relu/clamp tricks).  The kernel pipeline must match it bit for bit."""
from __future__ import annotations

import numpy as np

from .geometry import C0, FACES, GRP, S, SGN, TRIS
from .stages import LIGHT, THRESH


def reference(R: np.ndarray):
    X = SGN @ R[0]
    Y = SGN @ R[1]                                   # centred integer vertices
    fb = np.zeros((S, S), dtype=np.int64)
    zb = np.full((S, S), 31, dtype=np.int64)
    sat = lambda v: v + 15 <= 0 or v + 15 >= 31
    for f_, (a, b, c) in TRIS:
        if any(sat(X[v]) or sat(Y[v]) for v in (a, b, c)):
            continue
        k, sg, _ = FACES[f_]
        n = sg * R[:, k]                              # face normal, view space
        dot = int(np.dot(LIGHT, n))
        cn = 1 + 5 * f_ + sum(dot >= t for t in THRESH)
        zn = 7 + int(n[2])
        x0, x1 = max(min(X[[a, b, c]]) + 15, 0), min(max(X[[a, b, c]]) + 15, S - 1)
        y0, y1 = max(min(Y[[a, b, c]]) + 15, 0), min(max(Y[[a, b, c]]) + 15, S - 1)
        # the kernel pipeline rasterises whole 6-pixel groups inside the bbox (matters only for edge-on, zero-area faces)
        for j in range(y0, y1 + 1):
            for i in range((x0 // GRP) * GRP, (x1 // GRP) * GRP + GRP):
                qx, qy = i - C0, j - C0
                es = [(X[v] - X[u]) * (qy - Y[u]) - (Y[v] - Y[u]) * (qx - X[u]) for u, v in ((a, b), (b, c), (c, a))]
                inside = all(e >= 0 for e in es) or all(e <= 0 for e in es)
                if inside and zn < zb[j, i]:
                    zb[j, i], fb[j, i] = zn, cn
    return fb, zb
