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
