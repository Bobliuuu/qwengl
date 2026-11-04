"""Render a rotating cube.  Every frame is a full pass through the qwengl pipeline.
    python demo.py --backend numpy --frames 36 --out out/numpy
    QWEN_DIR=... python demo.py --backend qwen --frames 24 --out out/qwen   (every kernel runs inside Qwen3.5)
Writes fb_XXX.npy per frame plus frames.jsonl (per-frame timing, kernel-run counts and, for qwen, model mismatches)."""
import argparse, json, os, time
import numpy as np
from qwengl import NumpyBackend, quantise, render, rot_matrix

ap = argparse.ArgumentParser()
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
    th = 2 * np.pi * f / total                          # one full turn over the loop
    R = quantise(rot_matrix(0.55 + 0.35 * np.sin(th), th, 0.25 * np.cos(th)))
    before = dict(getattr(be, "stats", {}))
    t0 = time.time()
    fr = render(be, R)
    rec = dict(frame=f, seconds=round(time.time() - t0, 2), runs=fr.runs, dropped=len(fr.dropped), R=R.tolist())
    if a.backend == "qwen":
        for k in ("unique_runs", "tokens", "positions", "mismatches"):
            rec[k] = be.stats[k] - before.get(k, 0)
    np.save(os.path.join(a.out, f"fb_{f:03d}.npy"), fr.fb)
    log.write(json.dumps(rec) + "\n"); log.flush()
    print(json.dumps({k: v for k, v in rec.items() if k != "R"}), flush=True)
