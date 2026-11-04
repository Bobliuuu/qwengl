"""A small GL-style API over the qwengl kernels.

    gl = GL(backend)                       # backend: NumpyBackend() or a QwenBackend (kernels run inside Qwen3.5)
    gl.clear()                             # glClear(COLOR | DEPTH)
    gl.light((-1, -1, -2))                 # glLight: direction toward the light, view space
    gl.draw_elements(cube(), rotation=rot_matrix(.4, .7, .1), translate=(0, 0, 0))     # glDrawElements(GL_TRIANGLES)
    gl.draw_elements(octahedron(), rotation=..., translate=(8, 0, 0))                   # depth buffer persists across draws
    img = gl.read_pixels()                 # glReadPixels: (30, 30) colour indices

Per draw call the HOST: quantises the model-view rotation (the uniform, scale chosen by the mesh), computes the cross-product
minors m = Rx x Ry and the object-space light R^T l, bins triangles by bounding box and moves lanes between kernels.
Everything per vertex, per edge, per pixel is computed by kernels.  See README for what is and is not supported.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import stages as st
from .geometry import GRP, S, minors, quantise, split
from .mesh import Mesh

Z_CLEAR, C_CLEAR = 31, 0


@dataclass
class Frame:
    fb: np.ndarray
    zbuf: np.ndarray
    verts: np.ndarray
    runs: dict = field(default_factory=dict)
    dropped: list = field(default_factory=list)


def lanes(v):
    return np.stack(split(np.asarray(v)), -1).reshape(-1)


class GL:
    def __init__(self, backend):
        self.backend = backend
        self.runs: dict = {}
        self.dropped: list = []
        self._light = (-1, -1, -2)
        self.clear()

    # ---- state ------------------------------------------------------------------------------------------
    def clear(self):
        self.Z = np.full((S, S), Z_CLEAR, dtype=np.int64)
        self.C = np.full((S, S), C_CLEAR, dtype=np.int64)

    def light(self, direction):
        self._light = tuple(int(v) for v in direction)

    def read_pixels(self):
        return self.C.copy()

    def read_depth(self):
        return self.Z.copy()

    def _go(self, name, items):
        self.runs[name] = self.runs.get(name, 0) + len(items)
        return self.backend.run_many(items)

    # ---- draw -------------------------------------------------------------------------------------------
    def draw_elements(self, mesh: Mesh, rotation: np.ndarray, translate=(0, 0, 0), trace: list | None = None, R=None, scale: int | None = None):
        R = quantise(rotation, scale or mesh.scale) if R is None else np.asarray(R)
        tx, ty, tz = (int(v) for v in translate)
        assert max(abs(tx), abs(ty)) <= 16, "screen translation must fit a 5-bit weight"
        m = minors(R)
        mm = lanes(m)
        Lobj = R.T @ np.array(self._light)                                   # light in object space (host uniform)
        assert np.abs(Lobj).max() <= 31
        verts = [tuple(int(a) for a in v) for v in mesh.vertices]
        go = self._go

        # ---- vertex transform (chunks of 16 vertices) ---------------------------------------------------
        V = np.zeros((len(verts), 3), dtype=np.int64)
        for lo in range(0, len(verts), 16):
            chunk = tuple(verts[lo: lo + 16])
            items = [(st.vertex_kernel(chunk, i), np.concatenate([lanes(R[i]), lanes((tx, ty, tz)[i])])) for i in range(3)]
            for i, y in enumerate(go("vertex", items)):
                V[lo: lo + len(chunk), i] = y

        # ---- clip (guard band): triangles with a saturated vertex cannot be rasterised exactly -> dropped ---
        tri_ok, dropped = [], []
        T = mesh.triangles
        flags = go("clip", [(st.clip_kernel(), np.array([V[a, 0], V[b, 0], V[c, 0], V[a, 1], V[b, 1], V[c, 1]])) for a, b, c in T])
        anyf = go("clip_any", [(st.clip_any_kernel(), f) for f in flags])
        for t, a in enumerate(anyf):
            (dropped if a[0] > 0 else tri_ok).append(t)
        self.dropped += dropped

        # ---- per-triangle colour + depth (flat shading) --------------------------------------------------
        sh_in = np.concatenate([lanes(Lobj), lanes(R[2]), lanes(tz)])
        sh = go("shade", [(st.shade_kernel(tuple(int(x) for x in mesh.normals[t]), tuple(int(x) for x in mesh.anchors[t]),
                                           int(mesh.material[t])), sh_in) for t in range(len(T))])

        # ---- edge evaluation + coverage for every kept triangle ------------------------------------------
        ev_items, ev_meta = [], []
        for t in tri_ok:
            a, b, c = T[t]
            x0, x1 = int(V[[a, b, c], 0].min()), int(V[[a, b, c], 0].max())
            y0, y1 = int(V[[a, b, c], 1].min()), int(V[[a, b, c], 1].max())
            x0, x1, y0, y1 = max(x0, 0), min(x1, S - 1), max(y0, 0), min(y1, S - 1)
            for row in range(y0, y1 + 1):
                for seg in range(x0 // GRP, x1 // GRP + 1):
                    for (u, w) in ((a, b), (b, c), (c, a)):
                        xa, xb, ya, yb = V[u, 0], V[w, 0], V[u, 1], V[w, 1]
                        ev_items.append((st.eval_kernel(verts[u], verts[w], row, seg, tx, ty), np.array([xa, xb, ya, yb, xa, xb, ya, yb, *mm])))
                    ev_meta.append((t, row, seg))
        ev_out = go("eval", ev_items)
        cov_items, cov_meta = [], []
        for g, (t, row, seg) in enumerate(ev_meta):
            e = ev_out[3 * g: 3 * g + 3]
            for p in range(0, GRP, 2):
                x = []
                for q in (p, p + 1):
                    x += [e[k][2 * q] for k in range(3)] + [e[k][2 * q + 1] for k in range(3)]
                cov_items.append((st.cover_kernel(), np.array(x)))
                cov_meta.append((t, row, seg, p))
        cov_out = go("cover", cov_items)
        edge_maps = {}
        if trace is not None:
            for g, (t, row, seg) in enumerate(ev_meta):
                mp = edge_maps.setdefault(t, (np.zeros((3, S, S), dtype=np.int64), np.zeros((3, S, S), dtype=np.int64)))
                for k in range(3):
                    for p in range(GRP):
                        mp[0][k, row, seg * GRP + p] = ev_out[3 * g + k][2 * p]
                        mp[1][k, row, seg * GRP + p] = ev_out[3 * g + k][2 * p + 1]

        # ---- raster ops: in triangle order (a draw-call sequence), depth test + framebuffer write ---------
        skipped = 0
        by_tri = {}
        for mt, o in zip(cov_meta, cov_out):
            if not o.any():                   # predication: the model reported no coverage for these 2 pixels
                skipped += 1
                continue
            by_tri.setdefault(mt[0], []).append((mt, o))
        for t in tri_ok:
            items = by_tri.get(t, [])
            if not items:
                continue
            cn, zn = int(sh[t][0]), int(sh[t][1])
            r1 = []
            for (_, row, seg, p), s in items:
                i = seg * GRP + p
                r1.append((st.rop1_kernel(), np.array([s[0], s[1], self.Z[row, i], zn, self.C[row, i], cn,
                                                       s[2], s[3], self.Z[row, i + 1], zn, self.C[row, i + 1], cn])))
            out = go("rop2", [(st.rop2_kernel(), y) for y in go("rop1", r1)])
            for ((_, row, seg, p), _), y in zip(items, out):
                i = seg * GRP + p
                self.Z[row, i], self.C[row, i], self.Z[row, i + 1], self.C[row, i + 1] = y
            if trace is not None:
                cov = np.zeros((S, S), dtype=np.int64)
                for (_, row, seg, p), s2 in items:
                    i = seg * GRP + p
                    cov[row, i], cov[row, i + 1] = max(s2[0], s2[1]), max(s2[2], s2[3])
                em = edge_maps.get(t)
                trace.append(dict(tri=t, face=int(mesh.material[t]), edges=em[0] if em else None, edges_neg=em[1] if em else None,
                                  cov=cov, fb=self.C.copy(), z=self.Z.copy()))
        self.runs["rop_skipped"] = self.runs.get("rop_skipped", 0) + skipped
        return V
