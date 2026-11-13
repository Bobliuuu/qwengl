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
