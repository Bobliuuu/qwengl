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
