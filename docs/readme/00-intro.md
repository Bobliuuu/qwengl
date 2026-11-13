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
