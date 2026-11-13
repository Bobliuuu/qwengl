| path | what |
|---|---|
| `qwengl/kernel.py` | the kernel IR and the hardware limits it enforces (5-bit weights, 15 lanes, depth 2) |
| `qwengl/gl.py` | the GL-style API: `clear`, `light`, `draw_elements`, `read_pixels` and the per-draw kernel schedule |
| `qwengl/mesh.py` | `Mesh` and the built-in cube, octahedron, tetrahedron, pyramid |
| `qwengl/stages.py` | every pipeline stage as integer layers |
| `qwengl/scene.py` | the demo scene (3 draw calls) |
| `qwengl/reference.py` | independent integer rasteriser the pipeline must match |
| `qwengl/qwen_backend.py` | runs kernels in a real Qwen3.5 (teacher-forced or free-running) |
| `tools/`, `docs/tools/` | generators for every animation and diagram, driven by real kernel outputs |
