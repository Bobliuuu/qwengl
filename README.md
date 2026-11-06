<p align="center"><img src="docs/banner.svg" alt="qwengl: a graphics pipeline computed by a language model" width="100%"></p>

<h1 align="center">qwengl 🎮🧠</h1>
<p align="center"><b>A GL-style graphics library that runs inside a language model.</b><br>
<code>glClear</code> · <code>glLight</code> · <code>glDrawElements</code> · <code>glReadPixels</code>, with vertex transform, edge functions, coverage, depth test and framebuffer computed as integer ReLU layers in a stock Qwen3.5-0.8B.<br>
No shaders. No rasteriser code. The model's own tokens are the pixels.</p>

<p align="center">
<a href="docs/qwengl_demo_video.mp4"><b>🎬 Demo video (36 s)</b></a> ·
<a href="#-the-api"><b>🧑‍💻 The API</b></a> ·
<a href="#-how-a-frame-is-drawn"><b>🖼️ How it works</b></a> ·
<a href="#-life-of-a-pixel"><b>🔬 One pixel</b></a> ·
<a href="#-measured"><b>📊 Numbers</b></a> ·
<a href="#-what-is-not-in-the-model"><b>⚠️ Honest limits</b></a> ·
<a href="#-run-it"><b>🚀 Run it</b></a>
</p>

<p align="center"><a href="docs/qwengl_demo_video.mp4"><img src="docs/qwengl_demo_video.gif" alt="36 s demo: the API, draw calls, kernel windows and the 48 frames Qwen produced" width="720"></a><br>
<sub>🎬 the 36 s demo, as a GIF so it plays right here. Click for the <a href="docs/qwengl_demo_video.mp4">full-quality mp4</a>.</sub></p>

<p align="center"><a href="docs/qwengl_dungeon.mp4"><img src="docs/qwengl_dungeon.gif" alt="a figure walking around a spinning gem in an isometric dungeon room, rendered inside Qwen3.5" width="480"></a><br>
<sub>🏰🚶💎 a walker circling a spinning gem in a torch-lit room: 48 frames, <b>60×60</b> pixels, 7 <code>glDrawElements</code> calls per frame into one depth buffer. Every kernel ran inside Qwen3.5-0.8B on an A100: 812,548 kernel runs, 4.87 M predicted tokens, 0 mismatches, framebuffer and depth identical to an integer oracle. It is an isometric view, not first person: the pipeline has no perspective divide. <a href="docs/qwengl_dungeon.mp4">mp4</a></sub></p>

<p align="center"><a href="docs/qwengl_demo.mp4"><img src="docs/qwengl_demo.gif" alt="a cube with an octahedron and a pyramid orbiting it" width="300"></a><br>
<sub>🧊🔺🔷 the earlier 30×30 scene: cube, octahedron and pyramid, 3 draws, 36 frames, also run in the model. <a href="docs/qwengl_demo.mp4">mp4</a></sub></p>

---

We took the **qwenTPU** chip ([a TPU compiled into the weights of Qwen3.5](https://github.com/Bobliuuu/qwentpu)) and wrote the layer above it: **a compiler from a GL-style API to that chip's programs**.
Each stage is a tiny integer network, `y = clip(relu(x·W + b), 0, 31)`, with 5-bit weights. The weights ride in the prompt, the model emits every output token, and the next stage reads those tokens.

| | 🏁 what you get |
|---|---|
| 🧑‍💻 | a **GL-style API**: `clear`, `light`, `draw_elements(mesh, rotation, translate)`, `read_pixels`, over any triangle mesh with small-integer vertices |
| 🔺 | a **triangle rasteriser** made only of ReLU layers: edge functions, hard coverage, both windings |
| 🧱 | a **real depth buffer shared across draw calls**: objects occlude each other, in any draw order |
| 🎨 | **flat shading**: 5 brightness levels × 6 hues from `light · normal`, computed in the model |
| ✅ | **0 mismatches** over 2.2 M predicted tokens, and one full frame generated **free-running** |

<p align="center"><img src="docs/anim/scene_depth.gif" alt="dungeon colour buffer next to its depth buffer" width="640"><br>
<sub>🏰 colour buffer and depth buffer side by side, the 48 frames Qwen produced. Bright = near.</sub></p>

## 🧑‍💻 The API

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

## 🏗️ Architecture

One **kernel** is one **prompt**. The kernel's weights are 1,024 `CELL` tokens, its input is 16 `ACT` tokens, and the model emits the rest.

<p align="center"><img src="docs/arch.svg" alt="architecture" width="100%"></p>

<p align="center"><img src="docs/pipeline.svg" alt="the pipeline" width="100%"></p>

### 🧩 Anatomy of a kernel

<p align="center"><img src="docs/kernel-anatomy.svg" alt="anatomy of a kernel" width="100%"></p>

Every stage has this shape and the hardware limits are **asserted when the kernel is built**: 5-bit weights, at most 15 lanes, depth at most 2.
A deeper stage is just two programs back to back.

### 🎟️ What the model actually sees

<p align="center"><img src="docs/token-timeline.svg" alt="token timeline" width="100%"></p>
<p align="center"><img src="docs/anim/token_trace.gif" alt="the model emitting tokens" width="640"><br>
<sub>🎞️ real decoded tokens from a real coverage run. The host fed the 1,056 prompt tokens, the model picked the other 65.</sub></p>

## 🖼️ How a frame is drawn

<p align="center"><img src="docs/anim/draw_calls.gif" alt="the dungeon frame built by 7 glDrawElements calls, colour and depth buffers" width="100%"></p>
<p align="center"><sub>🎞️ one frame, one <code>glDrawElements</code> call at a time. The yellow flash is what the call changed. The depth buffer is shared, so the walker is correctly hidden by the pedestal.</sub></p>

The kernels are 30×30, so the 60×60 viewport is tiled. Each triangle is binned to a 30×30 window that contains it and the mesh is re-centred on an integer vertex (exact), so the same 5-bit kernels draw every window:

<p align="center"><img src="docs/anim/tiles.gif" alt="the viewport tiled with 30x30 kernel windows, drawn window by window" width="100%"></p>

<p align="center"><img src="docs/kernel-zoo.svg" alt="kernel zoo" width="100%"></p>

### 📐 Edge functions without a multiplier

<p align="center"><img src="docs/edge-function.svg" alt="edge function math" width="100%"></p>

The pixel position is the **weight**, the vertex data is the **activation**. The product `Xa·Yb − Xb·Ya` that triangle setup needs is rewritten as a dot with a 3-integer uniform (Binet–Cauchy), so setup costs no multiplier.

### 🪜 Why a ReLU network can draw a hard edge

Weights are 5-bit, so one layer multiplies by at most 15. Two such layers saturate every `e ≥ 0` to exactly 31 and leave every `e < 0` at 0.

<p align="center"><img src="docs/anim/hard_edge.gif" alt="relu stack turning a ramp into a step" width="640"></p>

## 🔬 Life of a pixel

Pixel (12, 7) of the demo frame is covered by **four** triangles. The first two win the depth test, the last two are rejected.

<p align="center"><img src="docs/pixel-journey.svg" alt="life of a pixel" width="100%"></p>
<p align="center"><img src="docs/anim/life_of_a_pixel.gif" alt="depth test live" width="560"></p>

### 🧱 The depth test is four ReLU layers

<p align="center"><img src="docs/depth-test.svg" alt="depth test circuit" width="100%"></p>

No branch, no comparison instruction, no multiplier: a conditional move built from saturating adds.

## 📊 Measured

<p align="center"><img src="docs/results.svg" alt="results" width="100%"></p>

**60×60 dungeon (48 frames, 7 draws per frame, the static room is cached after frame 0):** 812,548 kernel runs (75,423 unique), 84.4 M tokens, 4,867,743 predicted positions, **0 mismatches**, 0 differing pixels in colour and depth vs numpy, 0 dropped triangles, 2,246 s on an A100 (`results/dungeon_qwen_48frames.jsonl`). Teacher-forced, like the 36-frame run below. One full dungeon frame (frame 12) was also run **free-running** (the host does only `argmax; append`): 17,217 kernel runs (7,350 unique), 8.24 M tokens, 474,262 predicted positions, **0 mismatches**, framebuffer and depth identical to numpy, 1,511 s (`results/dungeon_free_run_frame.json`).

| A100, Qwen3.5-0.8B, 30×30, 36-frame scene (3 draw calls per frame) | |
|---|---|
| kernel runs (one qwenTPU program each) | **98,432** (34,628 unique, identical runs cached) |
| tokens through the model | **38.8 M**, 2,242,564 predicted positions |
| mismatches, model argmax vs integer oracle | **0** |
| frames identical to the numpy pipeline | **36 / 36**, 0 differing pixels in colour and depth |
| triangles dropped by the guard band | 0 |
| wall time | 1,015 s (about 28 s per frame, shared GPU) |
| 🆓 one frame, true free-running generation | 2,823 runs (1,293 unique), 83,725 tokens, **0 mismatches**, 219 s |

Logs are in [`results/`](results/). `pytest` checks the pipeline bit for bit against an independent integer software rasteriser on every built-in mesh, with translation, several draw calls, occlusion and random scenes.

## ⚠️ What is not in the model

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

## 🚀 Run it

```bash
pip install -e ".[dev]" && pytest -q                     # numpy pipeline vs integer reference
python demo.py --backend numpy --frames 36 --out out/scene_numpy      # the 3-draw-call scene
python make_video.py out/scene_numpy docs/numpy_demo
```

With a GPU, `torch`, `transformers`, [qwentpu](https://github.com/Bobliuuu/qwentpu) on `PYTHONPATH` and a Qwen3.5-0.8B checkpoint:

```bash
QWEN_DIR=/path/to/Qwen3.5-0.8B python run_qwen.py --frame 5               # one scene frame, cross-checked against numpy
QWEN_DIR=/path/to/Qwen3.5-0.8B python run_qwen.py --frame 5 --free-run    # true autoregressive generation
QWEN_DIR=/path/to/Qwen3.5-0.8B python demo.py --backend qwen --frames 36 --out out/scene_qwen
```

Regenerate the diagrams and animations from real kernel data:

```bash
PYTHONPATH=.:/path/to/qwentpu python tools/collect_data.py   # docs/data.json (pixel journey, decoded tokens)
python docs/tools/gen_svgs.py                                # all SVG diagrams
PYTHONPATH=. python tools/make_demo_video.py && PYTHONPATH=. python tools/gen_anim_dungeon.py && PYTHONPATH=. python tools/gen_anim.py && PYTHONPATH=. python tools/gen_anim2.py   # the last two also write cube_* extras and token_trace / life_of_a_pixel
```

## 🧰 Layout

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

## 🙏 Credits

Built on [**qwenTPU**](https://github.com/Bobliuuu/qwentpu) and **Qwen3.5**. Design vocabulary from Qwenputer, QwenPU and TinyTPU. MIT licensed.
