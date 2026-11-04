# qwengl

**A graphics pipeline computed by a language model.** Vertex transform, guard-band clip check, edge evaluation, coverage,
depth test and framebuffer write run as integer ReLU/clamp layers inside a stock Qwen3.5-0.8B, using the
[qwenTPU](https://github.com/Bobliuuu/qwentpu) v2 chip. The demo is a rotating, depth-buffered, flat-shaded cube.

![demo](docs/qwengl_demo.gif)  ([mp4](docs/qwengl_demo.mp4))

## Measured (A100, Qwen3.5-0.8B, 30x30 framebuffer, 36 frames = one full turn)

| | |
|---|---|
| kernel runs (each = one qwenTPU program in the prompt) | 104,940 (27,880 unique; identical runs are cached) |
| tokens through the model | 31.2 M, 1,809,896 predicted positions |
| **mismatches between model argmax and the integer oracle** | **0** |
| frames identical to the numpy pipeline | 36 / 36 (0 differing pixels) |
| time | 812 s (about 23 s per frame, GPU shared with other jobs) |

Per-frame logs: `results/qwen_demo_frames.jsonl`. The numpy pipeline is itself checked bit for bit against an independent
integer software rasteriser (`tests/`).

## How it works

Every stage is a stack of `y = clip(relu(x @ W + b), 0, 31)` layers with 5-bit weights, at most 15 lanes wide, depth at most 2
(one qwenTPU v2 program). `qwengl/stages.py` holds the tables.

| stage | kernel |
|---|---|
| vertex transform | corner signs x rotation uniform, 3 runs per frame |
| clip | saturation flags per vertex; triangles with a saturated vertex are dropped |
| setup + edge functions | `e = (Ya-Yb) qx + (Xb-Xa) qy + m.(s_a x s_b)` with `m = Rx x Ry` (Binet-Cauchy), 6 pixels per run |
| coverage | x15 saturating step twice, then AND of 3 edges (either winding, so no culling) |
| depth test | `Z - zn >= 1` via x15 steps, gated by coverage |
| framebuffer write | gated `Z' = Z - g`, `C' = C + g1 - g2` (conditional write, no multiplier) |
| shading | per face: 4 staircase thresholds on `light . normal` give 5 brightness levels x 6 hues |

## What is NOT in the model (read this)

- **Uniforms on the host:** the rotation is quantised to a 3x3 integer matrix and its cross-product `m` is computed on the host (like a GL normal matrix).
- **Binning and data movement on the host:** bounding boxes from vertex outputs, lane gather/scatter between kernels, buffer clear values (Z=31, colour=0), and the sequential draw order.
- **Clipping is a guard-band check only.** There is no polygon clipping and no new vertices; a triangle with a vertex at the guard-band edge is dropped. The demo cube never triggers it.
- **Depth is flat per face** (face-centre depth), not interpolated. Exact for a convex cube, not for general meshes.
- **Framebuffer and depth buffer live in host-held tokens between kernels**, not in model state. Each value in them was produced by the model.
- **Verification is teacher-forced.** Each program's expected continuation is fed and the model's argmax at every position is compared with it (0 mismatches). For a deterministic argmax this is what free-running generation emits, and the values passed to the next stage are decoded from the model's predicted tokens. The demo did not run the host's free-running `argmax; append` loop. Only the TPU layers run (Qwen's own muted layers are skipped, bit-identical per qwenTPU's tests).
- 30x30 pixels, 5-bit colour, orthographic projection. It is a very small GPU.

## Run

```bash
pip install -e ".[dev]" && pytest -q                 # numpy pipeline vs integer reference
python demo.py --backend numpy --frames 36 --out out/numpy && python make_video.py out/numpy docs/numpy_demo
# real model (needs torch, transformers, qwentpu on PYTHONPATH, a Qwen3.5-0.8B checkpoint, a GPU):
QWEN_DIR=/path/to/Qwen3.5-0.8B python run_qwen.py --out out1       # one frame, cross-checked against numpy
QWEN_DIR=/path/to/Qwen3.5-0.8B python demo.py --backend qwen --frames 36 --out out/qwen
```

MIT licensed. Built on qwenTPU and Qwen3.5.
