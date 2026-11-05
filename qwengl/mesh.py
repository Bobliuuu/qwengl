"""Static meshes with small integer vertices (what a 5-bit weight can hold), plus a few built-ins."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Mesh:
    """vertices: (n, 3) small ints.  triangles: (m, 3) vertex indices (either winding).
    material: int or (m,) colour family 0..5 (5 brightness levels each).
    normals: optional (m, 3) small-int face normals (outward).  Default: sign-quantised outward cross product.
    anchors: optional (m, 3) integer depth point per triangle (flat depth).  Default: rounded centroid."""
    vertices: np.ndarray
    triangles: np.ndarray
    material: object = 0
    normals: np.ndarray | None = None
    anchors: np.ndarray | None = None
    name: str = "mesh"
    scale: int = field(init=False)

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.int64)
        self.triangles = np.asarray(self.triangles, dtype=np.int64)
        m = len(self.triangles)
        self.material = np.broadcast_to(np.asarray(self.material, dtype=np.int64), (m,)).copy()
        assert self.vertices.ndim == 2 and self.vertices.shape[1] == 3 and np.abs(self.vertices).max() <= 15
        assert self.material.min() >= 0 and self.material.max() <= 5, "6 colour families"
        center = self.vertices.mean(0)
        if self.normals is None:
            ns = []
            for a, b, c in self.triangles:
                v = self.vertices
                n = np.cross(v[b] - v[a], v[c] - v[a]).astype(float)
                if n @ (v[[a, b, c]].mean(0) - center) < 0:
                    n = -n
                ns.append(np.rint(n / np.abs(n).max()).astype(np.int64) if np.abs(n).max() else np.zeros(3, np.int64))
            self.normals = np.array(ns)
        self.normals = np.asarray(self.normals, dtype=np.int64)
        if self.anchors is None:
            self.anchors = np.rint(self.vertices[self.triangles].mean(1)).astype(np.int64)
        self.anchors = np.asarray(self.anchors, dtype=np.int64)
        # a rotated vertex must stay inside the +-14 pixel guard band: |X| <= scale * |s|_2 (+ rounding slack)
        radius = np.sqrt((self.vertices ** 2).sum(1)).max()
        self.scale = int(max(1, min(5, np.floor(13 / radius)))) if radius else 5


def cube(size: int = 1, material=None, offset=(0, 0, 0)) -> Mesh:
    """Unit cube, vertex v has corner signs (2 bit_j - 1).  One colour family per face, face-centre depth anchors."""
    from .geometry import FACES
    v = np.array([[(2 * ((i >> j) & 1) - 1) * size for j in range(3)] for i in range(8)]) + np.asarray(offset)
    tris, mats, nrm, anc = [], [], [], []
    for f, (k, sg, q) in enumerate(FACES):
        for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3])):
            tris.append(t); mats.append(f if material is None else material)
            n = np.zeros(3, np.int64); n[k] = sg; nrm.append(n)
            a = np.zeros(3, np.int64); a[k] = sg * size; anc.append(a + np.asarray(offset))
    return Mesh(v, tris, mats, nrm, anc, name="cube")


def octahedron(size: int = 2) -> Mesh:
    v = np.array([[size, 0, 0], [-size, 0, 0], [0, size, 0], [0, -size, 0], [0, 0, size], [0, 0, -size]])
    tris = [(x, y, z) for x in (0, 1) for y in (2, 3) for z in (4, 5)]
    return Mesh(v, tris, [i % 6 for i in range(len(tris))], name="octahedron")


def tetrahedron(size: int = 2) -> Mesh:
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) * (size // 2 if size > 1 else 1)
    return Mesh(v, [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)], [0, 2, 3, 4], name="tetrahedron")


def pyramid(size: int = 2) -> Mesh:
    v = np.array([[-size, -size, -size], [size, -size, -size], [size, size, -size], [-size, size, -size], [0, 0, size]])
    tris = [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4), (0, 1, 2), (0, 2, 3)]
    return Mesh(v, tris, [0, 1, 2, 3, 5, 5], name="pyramid")
