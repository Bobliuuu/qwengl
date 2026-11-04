"""The demo scene: a cube with an octahedron and a pyramid orbiting it, three glDrawElements calls into one depth buffer."""
from __future__ import annotations

import numpy as np

from .geometry import rot_matrix
from .mesh import Mesh, cube, octahedron, pyramid


def _hue(m: Mesh, material: int) -> Mesh:
    return Mesh(m.vertices, m.triangles, material, m.normals, m.anchors, name=m.name)


CUBE, OCTA, PYR = _hue(cube(), 4), _hue(octahedron(1), 0), _hue(pyramid(1), 3)


def draw_scene(gl, f: int, total: int):
    th = 2 * np.pi * f / total
    gl.clear()
    gl.draw_elements(CUBE, rot_matrix(0.5 + 0.3 * np.sin(th), th, 0.2 * np.cos(th)), (0, 0, 0), scale=3)
    for k, (mesh, ph, sc) in enumerate(((OCTA, 0.0, 5), (PYR, np.pi, 4))):
        a = 2 * th + ph
        gl.draw_elements(mesh, rot_matrix(3 * th, 2 * th + k, 0.3), (int(round(8 * np.cos(a))), int(round(2 * np.sin(2 * a))), int(round(8 * np.sin(a)))), scale=sc)
    return gl.read_pixels()
