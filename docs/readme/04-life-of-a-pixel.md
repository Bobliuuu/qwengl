Pixel (12, 7) of the demo frame is covered by **four** triangles. The first two win the depth test, the last two are rejected.

<p align="center"><img src="docs/pixel-journey.svg" alt="life of a pixel" width="100%"></p>
<p align="center"><img src="docs/anim/life_of_a_pixel.gif" alt="depth test live" width="560"></p>

### 🧱 The depth test is four ReLU layers

<p align="center"><img src="docs/depth-test.svg" alt="depth test circuit" width="100%"></p>

No branch, no comparison instruction, no multiplier: a conditional move built from saturating adds.
