"""Every pipeline stage as integer ReLU/clamp kernels (see kernel.py).  Nothing here is arithmetic on the host:
these tables of small integers *are* the stage, and a backend (numpy or a real Qwen3.5) runs them."""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .geometry import C0, FACES, GRP, SGN
from .kernel import CONST, Kernel, Layer

LIGHT = (-1, -1, -2)                 # light direction (toward the light), view space
THRESH = (-6, -2, 2, 6)              # 4 thresholds -> 5 brightness levels per face colour


@lru_cache(None)
def vertex_kernel(i: int) -> Kernel:
    """Vertex transform: screen coordinate i (0=x, 1=y, 2=depth) of all 8 cube corners.
    in : R[i][0..2] as (+,-) lanes (6)        out: 8 lanes, centred value + 15 (saturating = viewport clamp)."""
    L = Layer(6, 8)
    for v in range(8):
        for j in range(3):
            L.w(2 * j, v, SGN[v, j]).w(2 * j + 1, v, -SGN[v, j])
        L.b(v, 15)
    return Kernel(f"vertex{i}", [L])


@lru_cache(None)
def clip_kernel() -> Kernel:
    """Guard-band check.  in: [Xa Xb Xc Ya Yb Yc] vertex outputs.  out: 12 'saturated' flags (lo x3 x, lo y3, hi x3, hi y3)."""
    L1, L2 = Layer(6, 12), Layer(12, 12)
    for k in range(6):
        L1.w(k, k, -1).b(k, 1)             # lo = relu(1 - V)      -> 1 iff V == 0
        L1.w(k, 6 + k, 1).b(6 + k, -15)    # u  = relu(V - 15)     -> 16 iff V == 31
        L2.w(k, k, 1)                      # lo passes
        L2.w(6 + k, 6 + k, 1).b(6 + k, -15)  # hi = relu(u - 15)   -> 1 iff V == 31
    return Kernel("clip", [L1, L2])


@lru_cache(None)
def clip_any_kernel() -> Kernel:
    L = Layer(12, 1)
    for k in range(12):
        L.w(k, 0, 1)
    return Kernel("clip_any", [L])


@lru_cache(None)
def shade_kernel(f: int) -> Kernel:
    """Per-face flat shading + depth.  in: column k of R as (+,-) lanes (6).  out: [colour index 1..30, depth 2..12].
    colour = 1 + 5 f + #{t in THRESH: light . n >= t}; depth = 7 + sigma * Rz   (smaller = nearer)."""
    k, sg, _ = FACES[f]
    L1, L2 = Layer(6, 10), Layer(10, 2)
    for ti, t in enumerate(THRESH):
        for i in range(3):
            c = sg * LIGHT[i]
            L1.w(2 * i, ti, c).w(2 * i + 1, ti, -c)               # P_t = relu(dot - t + 1)
            L1.w(2 * i, 4 + ti, c).w(2 * i + 1, 4 + ti, -c)       # Q_t = relu(dot - t)
        L1.b(ti, -t + 1).b(4 + ti, -t)
        L2.w(ti, 0, 1).w(4 + ti, 0, -1)                           # level = sum (P_t - Q_t)
    L1.w(4, 8, sg).w(5, 8, -sg).b(8, 7)                           # zn = relu(7 + sigma*Rz)
    L1.b(9, CONST)                                                # constant lane
    base = 1 + 5 * f
    L2.w(9, 0, 1 if base >= CONST else 0).b(0, base - CONST if base >= CONST else base)
    L2.w(8, 1, 1)
    return Kernel(f"shade{f}", [L1, L2])


@lru_cache(None)
def eval_kernel(a: int, b: int, row: int, seg: int) -> Kernel:
    """Edge function of directed cube edge a->b at 6 pixels (row, 6*seg .. 6*seg+5), then a x15 saturating step.
        e(q) = (Ya - Yb) qx + (Xb - Xa) qy + m . (s_a x s_b)          (X, Y = vertex-kernel lanes, m = Rx x Ry)
    in : [Xa Xb Ya Yb m0+ m0- m1+ m1- m2+ m2-] (10)
    out: per pixel (v+, v-) = (min(31, 15*relu(e+1)), min(31, 15*relu(-e+1)))  ~ ([e >= 0], [e <= 0]) as 0/15/30/31."""
    c = np.cross(SGN[a], SGN[b])
    L1, L2 = Layer(10, 12), Layer(12, 12)
    qy = row - C0
    for p in range(GRP):
        qx = seg * GRP + p - C0
        for s, o in ((1, 2 * p), (-1, 2 * p + 1)):
            L1.w(2, o, s * qx).w(3, o, -s * qx)        # Ya, Yb
            L1.w(1, o, s * qy).w(0, o, -s * qy)        # Xb, Xa
            for i in range(3):
                L1.w(4 + 2 * i, o, s * c[i]).w(5 + 2 * i, o, -s * c[i])
            L1.b(o, 1)
    for o in range(12):
        L2.w(o, o, 15)
    return Kernel(f"eval{a}{b}r{row}s{seg}", [L1, L2])


@lru_cache(None)
def cover_kernel() -> Kernel:
    """Hard binarise + AND of 3 edges, two pixels per run.
    in : per pixel [v+ e0 e1 e2, v- e0 e1 e2] x 2 (12)   out: [s+ s- (pixel 0), s+ s- (pixel 1)], 31 iff all three edges agree."""
    L1, L2 = Layer(12, 13), Layer(13, 4)
    for k in range(12):
        L1.w(k, k, 15)                                   # t = 0 / 31
    L1.b(12, CONST)
    for px in range(2):
        for sgn in range(2):
            o = 2 * px + sgn
            for e in range(3):
                L2.w(6 * px + 3 * sgn + e, o, 1)
            L2.w(12, o, -4).b(o, -2)                     # - 62
    return Kernel("cover", [L1, L2])


@lru_cache(None)
def rop1_kernel() -> Kernel:
    """Raster-op, part 1 (two pixels).  in: [s+ s- Z zn C cn] x 2.  out: [s31 cov Z zn C cn] x 2 + const,
    cov = clamp(s+ + s-) ; s31 = 31 iff Z - zn >= 1 (the depth test)."""
    L1, L2 = Layer(12, 13), Layer(13, 13)
    for px in range(2):
        i, o = 6 * px, 6 * px
        L1.w(i, o, 1).w(i + 1, o, 1)                     # cov
        L1.w(i + 2, o + 1, 15).w(i + 3, o + 1, -15)      # u = relu(15 (Z - zn))
        for k in (2, 3, 4, 5):
            L1.w(i + k, o + k, 1)
        L2.w(o + 1, o, 15)                               # s31 = relu(15 u)
        L2.w(o, o + 1, 1)                                # cov
        for k in (2, 3, 4, 5):
            L2.w(o + k, o + k, 1)
    L1.b(12, CONST)
    L2.w(12, 12, 1)
    return Kernel("rop1", [L1, L2])


@lru_cache(None)
def rop2_kernel() -> Kernel:
    """Raster-op, part 2: conditional write of depth and colour.  in: rop1 output (13).  out: [Z' C'] x 2."""
    L1, L2 = Layer(13, 10), Layer(10, 4)
    for px in range(2):
        i, o = 6 * px, 5 * px
        S31, COV, Z, ZN, C, CN = (i + k for k in range(6))
        for lane, (pos, neg) in enumerate(((Z, ZN), (CN, C), (C, CN))):
            L1.w(pos, o + lane, 1).w(neg, o + lane, -1).w(S31, o + lane, 1).w(COV, o + lane, 1)
            L1.w(12, o + lane, -4).b(o + lane, -2)       # - 62
        L1.w(Z, o + 3, 1).w(C, o + 4, 1)
        L2.w(o + 3, 2 * px, 1).w(o, 2 * px, -1)          # Z' = relu(Z - g)
        L2.w(o + 4, 2 * px + 1, 1).w(o + 1, 2 * px + 1, 1).w(o + 2, 2 * px + 1, -1)   # C' = relu(C + g1 - g2)
    return Kernel("rop2", [L1, L2])
