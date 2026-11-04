"""Backwards-compatible single-cube render (the original demo).  New code should use qwengl.gl.GL."""
from __future__ import annotations

import numpy as np

from .gl import GL, Frame  # noqa: F401
from .mesh import cube

_CUBE = cube()


def render(backend, R: np.ndarray, trace: list | None = None) -> Frame:
    gl = GL(backend)
    V = gl.draw_elements(_CUBE, None, R=R, trace=trace)
    return Frame(gl.C, gl.Z, V, dict(gl.runs), list(gl.dropped))
