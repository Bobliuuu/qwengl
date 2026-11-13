```python
from qwengl import NumpyBackend, rot_matrix
from qwengl.gl import GL
from qwengl.mesh import cube, octahedron, pyramid

gl = GL(NumpyBackend())            # or QwenBackend(...): every kernel then runs inside Qwen3.5
gl.light((-1, -1, -2))             # glLight
gl.clear()                         # glClear(COLOR | DEPTH)
gl.draw_elements(cube(),       rot_matrix(.5, .9, .2), translate=(0, 0, 0))      # glDrawElements
gl.draw_elements(octahedron(1), rot_matrix(1., .3, .8), translate=(8, 0, -4))    # same depth buffer
gl.draw_elements(pyramid(1),    rot_matrix(.2, 1., .1), translate=(-8, 2, 6))
image = gl.read_pixels()           # (30, 30) colour indices, gl.read_depth() for the depth buffer
```

Bring your own mesh: `Mesh(vertices, triangles, material=0..5)` with small-integer vertices (a 5-bit weight is the limit).

<p align="center"><img src="docs/gl-api.svg" alt="the GL-style API and what runs where" width="100%"></p>

| ✅ supported | ❌ not supported (yet) |
|---|---|
| any triangle mesh with small-integer vertices | perspective projection (orthographic only) |
| rotation, plus screen and depth translation per draw | programmable shaders (flat Lambert, 5 levels) |
| several draw calls, one persistent depth buffer | textures, blending, stencil |
| back faces drawn and resolved by the depth test | dynamic or large meshes, polygon clipping |
| built-in cube, octahedron, tetrahedron, pyramid | per-pixel depth interpolation (depth is flat per triangle) |
