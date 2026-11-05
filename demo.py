"""Render the demo scene (a cube with an octahedron and a pyramid orbiting it: 3 glDrawElements calls into one depth buffer).
    python demo.py --backend numpy --frames 36 --out out/numpy
    QWEN_DIR=... python demo.py --backend qwen --frames 24 --out out/qwen   (every kernel runs inside Qwen3.5)
Writes fb_XXX.npy per frame plus frames.jsonl (per-frame timing, kernel-run counts and, for qwen, model mismatches)."""
import argparse, json, os, time
import numpy as np
from qwengl import NumpyBackend
from qwengl.gl import GL
from qwengl import dungeon, scene

ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="orbit", choices=["orbit", "dungeon"])
ap.add_argument("--backend", default="numpy", choices=["numpy", "qwen"])
ap.add_argument("--frames", type=int, default=36)
ap.add_argument("--start", type=int, default=0, help="first frame index (for splitting a run across workers)")
ap.add_argument("--of", type=int, default=0, help="total frames in the full loop (default: --frames)")
ap.add_argument("--qwen", default=os.environ.get("QWEN_DIR"))
ap.add_argument("--device", default="cuda")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--out", default="out/demo")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
total = a.of or a.frames
if a.backend == "qwen":
    from qwengl.qwen_backend import QwenBackend
    be = QwenBackend(a.qwen, a.device, a.batch)
else:
    be = NumpyBackend()
log = open(os.path.join(a.out, "frames.jsonl"), "a")
for f in range(a.start, a.start + a.frames):
    before = dict(getattr(be, "stats", {}))
    t0 = time.time()
    if a.scene == "dungeon":
        gl = GL(be, dungeon.W, dungeon.H)                # 60x60 tiled viewport: 9 kernel windows of 30x30
        fb = dungeon.draw_scene(gl, f, total)
    else:
        gl = GL(be)                                      # a fresh GL context per frame: glClear + 3 glDrawElements
        fb = scene.draw_scene(gl, f, total)
    rec = dict(frame=f, seconds=round(time.time() - t0, 2), runs=gl.runs, dropped=len(gl.dropped))
    if a.backend == "qwen":
        for k in ("unique_runs", "tokens", "positions", "mismatches"):
            rec[k] = be.stats[k] - before.get(k, 0)
    np.save(os.path.join(a.out, f"fb_{f:03d}.npy"), fb)
    np.save(os.path.join(a.out, f"z_{f:03d}.npy"), gl.read_depth())
    log.write(json.dumps(rec) + "\n"); log.flush()
    print(json.dumps(rec), flush=True)
