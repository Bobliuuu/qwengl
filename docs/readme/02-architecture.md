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
