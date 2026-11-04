<p align="center"><img src="docs/pipeline.svg" alt="qwengl pipeline" width="100%"></p>

<h1 align="center">qwengl</h1>
<p align="center"><b>A graphics pipeline computed by a language model.</b><br>
Vertex transform, edge functions, coverage, depth test and framebuffer write, run as integer ReLU layers inside a stock Qwen3.5-0.8B.<br>
No GPU shaders, no rasteriser code: the model's own tokens are the pixels.</p>

<p align="center"><img src="docs/qwengl_demo.gif" alt="a rotating cube rendered inside Qwen3.5" width="480"></p>
<p align="center"><sub>36 frames, one full turn, 30x30 pixels, 5-bit colour. Every kernel ran inside Qwen3.5-0.8B on an A100 and matched an integer oracle token for token. <a href="docs/qwengl_demo.mp4">mp4</a></sub></p>

---

Built on [qwenTPU](https://github.com/Bobliuuu/qwentpu), which compiles a small integer matrix unit into the weights of Qwen3.5.
qwengl is the layer above it: a compiler from graphics stages to that chip's programs.

## Measured

| A100, Qwen3.5-0.8B, 30x30, 36 frames | |
|---|---|
| kernel runs (one qwenTPU program in the prompt each) | **104,940** (27,880 unique, identical runs cached) |
| tokens through the model | **31.2 M**, 1,809,896 predicted positions |
| mismatches, model argmax vs integer oracle | **0** |
| frames identical to the numpy pipeline | **36 / 36**, 0 differing pixels |
| wall time | 812 s (about 23 s per frame, shared GPU) |

Per-frame logs are in [`results/`](results/). The numpy pipeline is itself checked bit for bit against an independent integer software rasteriser (`pytest`).

## How a frame is drawn

Every stage is a stack of `y = clip(relu(x @ W + b), 0, 31)` layers: 5-bit weights, at most 15 lanes, depth at most 2 (one qwenTPU v2 program).
The tables of small integers in [`qwengl/stages.py`](qwengl/stages.py) *are* the stages.

<p align="center"><img src="docs/anim/draw_calls.gif" alt="12 triangles drawn one by one with edge functions, coverage, depth buffer and framebuffer" width="100%"></p>
<p align="center"><sub>One frame, triangle by triangle. Left to right: the three edge-function outputs, their AND, the depth buffer, the framebuffer. All panels are real kernel outputs.</sub></p>

| stage | what the kernel computes |
|---|---|
| vertex transform | corner signs x the rotation uniform (3 runs per frame) |
| clip | saturation flags per vertex; a triangle with a saturated vertex is dropped |
| edge functions | `e = (Ya-Yb) qx + (Xb-Xa) qy + m.(s_a x s_b)` for 6 pixels per run, with `m = Rx x Ry` (Binet-Cauchy), so edge setup needs no multiplier |
| coverage | x15 saturating step twice, then the AND of three edges, both windings, so no culling |
| depth test | `Z - zn >= 1` as saturating steps, gated by coverage |
| framebuffer write | gated `Z' = Z - g`, `C' = C + g1 - g2`: a conditional write without a multiplier |
| shading | per face: four staircase thresholds on `light . normal` give 5 brightness levels x 6 hues |

### Why a ReLU network can draw a hard edge

Weights are 5-bit, so one layer can multiply by at most 15. Two such layers saturate every `e >= 0` to exactly 31 and leave every `e < 0` at 0, which makes the inside test binary.

<p align="center"><img src="docs/anim/hard_edge.gif" alt="relu stack turning a ramp into a step" width="640"></p>

### The depth buffer is doing real work

All 12 triangles are drawn with no back-face culling, in a fixed order. The back faces are covered and then overwritten by the depth test.

<p align="center"><img src="docs/anim/cube_depth.gif" alt="cube colour buffer next to its depth buffer" width="560"></p>

## What is not in the model

- **Uniforms on the host.** The rotation is quantised to a 3x3 integer matrix and its cross product `m` is computed on the host, like a GL normal matrix.
- **Binning and data movement on the host.** Bounding boxes from vertex outputs, lane gather and scatter between kernels, buffer clear values (Z = 31, colour = 0), the draw order. Raster ops whose coverage the model reported as all zero are not launched (predication).
- **Clipping is a guard-band check only.** No polygon clipping, no new vertices. The demo cube never triggers it.
- **Depth is flat per face**, not interpolated: exact for a convex cube, not for general meshes.
- **The framebuffer and depth buffer live in host-held tokens between kernels**, not in model state. Every value in them was produced by the model.
- **The 36-frame run is teacher-forced.** Each program's expected continuation is fed and the model's argmax at every position is compared with it (0 mismatches). For a deterministic argmax this is what free-running generation emits. The numbers passed to the next stage are decoded from the model's predicted tokens. A true free-running mode (host does only `argmax; append`) is implemented: `run_qwen.py --free-run`.
- **Only the TPU layers run.** Qwen's own muted layers are skipped, bit-identical per qwenTPU's tests.
- 30x30 pixels, 5-bit colour, orthographic. It is a very small GPU.

## Run it

```bash
pip install -e ".[dev]" && pytest -q                     # numpy pipeline vs integer reference
python demo.py --backend numpy --frames 36 --out out/numpy
python make_video.py out/numpy docs/numpy_demo
PYTHONPATH=. python tools/gen_anim.py                    # regenerate the animations above
```

With a GPU, `torch`, `transformers`, [qwentpu](https://github.com/Bobliuuu/qwentpu) on `PYTHONPATH` and a Qwen3.5-0.8B checkpoint:

```bash
QWEN_DIR=/path/to/Qwen3.5-0.8B python run_qwen.py --out out1               # one frame, cross-checked against numpy
QWEN_DIR=/path/to/Qwen3.5-0.8B python run_qwen.py --free-run --out out1    # true autoregressive generation
QWEN_DIR=/path/to/Qwen3.5-0.8B python demo.py --backend qwen --frames 36 --out out/qwen
```

## Layout

| path | what |
|---|---|
| `qwengl/kernel.py` | the kernel IR and the hardware limits it enforces (5-bit weights, 15 lanes, depth 2) |
| `qwengl/stages.py` | every pipeline stage as integer layers |
| `qwengl/pipeline.py` | the frame loop: which kernel runs on what, in what order |
| `qwengl/reference.py` | independent integer rasteriser the pipeline must match |
| `qwengl/qwen_backend.py` | runs kernels in a real Qwen3.5 (teacher-forced or free-running) |
| `tools/gen_anim.py` | the animations, generated from real kernel outputs |

MIT licensed. Thanks to Qwen3.5 and to Qwenputer, QwenPU and TinyTPU for the vocabulary qwenTPU builds on.
