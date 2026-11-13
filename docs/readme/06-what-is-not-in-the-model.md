<p align="center"><img src="docs/host-vs-model.svg" alt="what runs where" width="100%"></p>

- 🧰 **Uniforms on the host.** Per draw call the rotation is quantised to a 3×3 integer matrix, and its cross product `m` and the object-space light `Rᵀl` are computed on the host, like GL's normal matrix and light uniforms.
- 🧰 **Binning and data movement on the host.** Bounding boxes from vertex outputs, lane gather and scatter between kernels, buffer clears (Z = 31, colour = 0), draw order. Raster ops whose coverage the model reported as all zero are not launched (predication).
- ✂️ **Clipping is a guard-band check only.** No polygon clipping, no new vertices: a triangle with a vertex outside the ±14 px band is dropped.
- 📏 **Depth is flat per triangle** (a rounded anchor point, the face centre for the cube), not interpolated. Exact for convex, well-separated objects, approximate when triangles of different meshes intersect.
- 🔦 **Lighting** uses a quantised face normal (components in −1..1) against a fixed Lambert staircase. Not Phong, not per pixel.
- 🔄 **Rotation is about the origin**, then a screen/depth translation. Position inside the mesh is part of the vertex data.
- 🗃️ **The framebuffer and depth buffer live in host-held tokens between kernels**, not in model state. Every value in them was produced by the model.
- 🔁 **The 36-frame run is teacher-forced.** Each program's expected continuation is fed and the model's argmax at every position is compared with it (0 mismatches). For a deterministic argmax this is what free-running generation emits, and the numbers passed to the next stage are decoded from the model's predicted tokens. To close the gap, one full frame was also run **free-running** (`--free-run`: the host does only `argmax; append`, expected tokens are never fed): 0 mismatches, framebuffer and depth identical to numpy.
- ⚡ **Only the TPU layers run.** Qwen's own muted layers are skipped, bit-identical per qwenTPU's tests.
- 🔬 5-bit colour, orthographic, flat shading. The native kernel window is 30×30 (5-bit coordinates). `GL(backend, 60, 60)` tiles the viewport: the host bins each triangle to a 30×30 window that contains it and re-centres the mesh on an integer vertex (`t' = t + R·o`, exact). Triangles wider than 12 px are dropped, so large meshes must be pre-subdivided. It is a very small GPU.
