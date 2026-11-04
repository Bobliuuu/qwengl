"""Independent integer software rasteriser: same quantised geometry, written with ordinary arithmetic
(explicit cross products, comparisons, no relu/clamp tricks).  The kernel pipeline must match it bit for bit."""
from __future__ import annotations

import numpy as np

from .geometry import GRP, S, quantise
from .mesh import Mesh, cube
from .stages import THRESH


def reference_draw(Z, C, mesh: Mesh, R, translate=(0, 0, 0), light=(-1, -1, -2)):
    tx, ty, tz = translate
    s = mesh.vertices
    X = s @ R[0] + tx + 15
    Y = s @ R[1] + ty + 15
    Lobj = R.T @ np.array(light)
    sat = lambda v: v <= 0 or v >= 31
    for t, (a, b, c) in enumerate(mesh.triangles):
        if any(sat(X[v]) or sat(Y[v]) for v in (a, b, c)):
            continue
        dot = int(mesh.normals[t] @ Lobj)
        cn = 1 + 5 * int(mesh.material[t]) + sum(dot >= th for th in THRESH)
        zn = min(max(15 + int(R[2] @ mesh.anchors[t]) + tz, 0), 31)
        x0, x1 = max(min(X[[a, b, c]]), 0), min(max(X[[a, b, c]]), S - 1)
        y0, y1 = max(min(Y[[a, b, c]]), 0), min(max(Y[[a, b, c]]), S - 1)
        for j in range(y0, y1 + 1):
            for i in range((x0 // GRP) * GRP, (x1 // GRP) * GRP + GRP):
                es = [(X[v] - X[u]) * (j - Y[u]) - (Y[v] - Y[u]) * (i - X[u]) for u, v in ((a, b), (b, c), (c, a))]
                if (all(e >= 0 for e in es) or all(e <= 0 for e in es)) and zn < Z[j, i]:
                    Z[j, i], C[j, i] = zn, cn
    return Z, C


def reference(R: np.ndarray):
    """The original cube demo."""
    Z, C = np.full((S, S), 31, dtype=np.int64), np.zeros((S, S), dtype=np.int64)
    reference_draw(Z, C, cube(), np.asarray(R))
    return C, Z
