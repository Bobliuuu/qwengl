"""Render frames with every kernel executed by a real Qwen3.5, cross-checked against the numpy oracle.
    QWEN_DIR=/path/to/Qwen3.5-0.8B python run_qwen.py --frames 1 --angles 0.4,0.7,0.1"""
import argparse, json, os, time
import numpy as np
from qwengl import NumpyBackend, quantise, render, rot_matrix
from qwengl.qwen_backend import QwenBackend

ap = argparse.ArgumentParser()
ap.add_argument("--qwen", default=os.environ.get("QWEN_DIR"))
ap.add_argument("--device", default="cuda")
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--angles", default="0.4,0.7,0.1")
ap.add_argument("--out", default="out")
ap.add_argument("--graph", action="store_true")
ap.add_argument("--free-run", action="store_true", help="true autoregressive generation: host only does argmax + append")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
be = QwenBackend(a.qwen, a.device, a.batch, graph=a.graph, free_run=a.free_run)
ang = [float(v) for v in a.angles.split(",")]
R = quantise(rot_matrix(*ang))
t0 = time.time()
fr = render(be, R)
dt = time.time() - t0
ref = render(NumpyBackend(), R)
np.save(os.path.join(a.out, "fb_qwen.npy"), fr.fb)
res = dict(free_run=a.free_run, angles=ang, wall_seconds=round(dt, 1), identical_framebuffer=bool((fr.fb == ref.fb).all() and (fr.zbuf == ref.zbuf).all()),
           kernel_runs=fr.runs, **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in be.stats.items() if k != "seconds"}, model_seconds=round(be.stats["seconds"], 1))
print(json.dumps(res))
