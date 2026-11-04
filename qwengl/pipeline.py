"""The qwengl pipeline: uniform -> vertex transform -> guard-band clip -> shade/depth setup -> edge evaluation ->
coverage -> raster-op (depth test + framebuffer write).

HOST does: quantise the rotation (the model-view *uniform*) and its cross-product minors, bounding-box tile binning
from vertex-kernel outputs, lane gather/scatter between kernels, clearing the buffers.  No host arithmetic on pixel,
vertex or edge values.  All of that is done by the kernels in stages.py."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import stages as st
from .geometry import C0, FACES, GRP, S, TRIS, uniform_lanes

Z_CLEAR, C_CLEAR = 31, 0


@dataclass
class Frame:
    fb: np.ndarray                      # (S, S) colour index 0..30 (0 = background)
    zbuf: np.ndarray
    verts: np.ndarray                   # (8, 3) vertex-kernel outputs (centre + 15)
    runs: dict = field(default_factory=dict)
    dropped: list = field(default_factory=list)


def render(backend, R: np.ndarray, trace: list | None = None) -> Frame:
    rows, cols, mm = uniform_lanes(R)
    counts = {}

    def go(name, items):
        counts[name] = counts.get(name, 0) + len(items)
        return backend.run_many(items)

    # ---- vertex transform ------------------------------------------------------------------------------
    V = np.stack(go("vertex", [(st.vertex_kernel(i), rows[i]) for i in range(3)]), 1)       # (8, 3): X Y Z

    # ---- clip (guard band): triangles with a saturated vertex cannot be rasterised exactly -> dropped ---
    tri_ok, dropped = [], []
    flags = go("clip", [(st.clip_kernel(), np.array([V[a, 0], V[b, 0], V[c, 0], V[a, 1], V[b, 1], V[c, 1]]))
                        for _, (a, b, c) in TRIS])
    anyf = go("clip_any", [(st.clip_any_kernel(), f) for f in flags])
    for t, a in enumerate(anyf):
        (dropped if a[0] > 0 else tri_ok).append(t)

    # ---- per-face colour + depth (flat shading) ----------------------------------------------------------
    sh = go("shade", [(st.shade_kernel(f), cols[FACES[f][0]]) for f in range(6)])      # (cn, zn)

    # ---- edge evaluation + coverage for every kept triangle (independent of each other) ------------------
    ev_items, ev_meta = [], []
    for t in tri_ok:
        a, b, c = TRIS[t][1]
        x0, x1 = int(V[[a, b, c], 0].min()), int(V[[a, b, c], 0].max())
        y0, y1 = int(V[[a, b, c], 1].min()), int(V[[a, b, c], 1].max())
        x0, x1, y0, y1 = max(x0, 0), min(x1, S - 1), max(y0, 0), min(y1, S - 1)
        for row in range(y0, y1 + 1):
            for seg in range(x0 // GRP, x1 // GRP + 1):
                for (u, w) in ((a, b), (b, c), (c, a)):
                    ev_items.append((st.eval_kernel(u, w, row, seg),
                                     np.array([V[u, 0], V[w, 0], V[u, 1], V[w, 1], *mm])))
                ev_meta.append((t, row, seg))
    ev_out = go("eval", ev_items)
    cov_items, cov_meta = [], []
    for g, (t, row, seg) in enumerate(ev_meta):
        e = ev_out[3 * g: 3 * g + 3]                                        # three (12,) vectors
        for p in range(0, GRP, 2):
            x = []
            for q in (p, p + 1):
                x += [e[k][2 * q] for k in range(3)] + [e[k][2 * q + 1] for k in range(3)]
            cov_items.append((st.cover_kernel(), np.array(x)))
            cov_meta.append((t, row, seg, p))
    cov_out = go("cover", cov_items)
    if trace is not None:
        edge_maps = {}
        for g, (t, row, seg) in enumerate(ev_meta):
            m = edge_maps.setdefault(t, np.zeros((3, S, S), dtype=np.int64))
            for k in range(3):
                for p in range(GRP):
                    m[k, row, seg * GRP + p] = ev_out[3 * g + k][2 * p]

    # ---- raster ops: in triangle order (like a draw call sequence), depth test + framebuffer write -----
    Z = np.full((S, S), Z_CLEAR, dtype=np.int64)
    Cb = np.full((S, S), C_CLEAR, dtype=np.int64)
    by_tri = {}
    skipped = 0
    for m, o in zip(cov_meta, cov_out):
        if not o.any():                       # predication: the model reported no coverage for these 2 pixels, the
            skipped += 1                      # raster-op would be a no-op (g = g1 = g2 = 0), so it is not launched
            continue
        by_tri.setdefault(m[0], []).append((m, o))
    for t in tri_ok:
        items = by_tri.get(t, [])
        if not items:
            continue
        cn, zn = int(sh[TRIS[t][0]][0]), int(sh[TRIS[t][0]][1])
        r1 = []
        for (_, row, seg, p), s in items:
            x = np.array([s[0], s[1], Z[row, seg * GRP + p], zn, Cb[row, seg * GRP + p], cn,
                          s[2], s[3], Z[row, seg * GRP + p + 1], zn, Cb[row, seg * GRP + p + 1], cn])
            r1.append((st.rop1_kernel(), x))
        mid = go("rop1", r1)
        out = go("rop2", [(st.rop2_kernel(), y) for y in mid])
        for ((_, row, seg, p), _), y in zip(items, out):
            i = seg * GRP + p
            Z[row, i], Cb[row, i], Z[row, i + 1], Cb[row, i + 1] = y
        if trace is not None:
            cov = np.zeros((S, S), dtype=np.int64)
            for (_, row, seg, p), s2 in items:
                i = seg * GRP + p
                cov[row, i], cov[row, i + 1] = max(s2[0], s2[1]), max(s2[2], s2[3])
            trace.append(dict(tri=t, face=TRIS[t][0], edges=edge_maps.get(t), cov=cov, fb=Cb.copy(), z=Z.copy()))
    counts['rop_skipped'] = skipped
    return Frame(Cb, Z, V, counts, dropped)
