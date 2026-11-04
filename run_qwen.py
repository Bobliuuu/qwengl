"""One scene frame (3 glDrawElements calls) with every kernel executed by a real Qwen3.5, cross-checked against numpy.
    QWEN_DIR=/path/to/Qwen3.5-0.8B python run_qwen.py --frame 5 [--free-run]"""
import argparse, json, os, time
import numpy as np
from qwengl import NumpyBackend
from qwengl.gl import GL
from qwengl.qwen_backend import QwenBackend
from qwengl.scene import draw_scene

ap = argparse.ArgumentParser()
ap.add_argument("--qwen", default=os.environ.get("QWEN_DIR"))
ap.add_argument("--device", default="cuda")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--frame", type=int, default=5)
ap.add_argument("--of", type=int, default=36)
ap.add_argument("--out", default="out")
ap.add_argument("--graph", action="store_true")
ap.add_argument("--free-run", action="store_true", help="true autoregressive generation: host only does argmax + append")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
be = QwenBackend(a.qwen, a.device, a.batch, graph=a.graph, free_run=a.free_run)
t0 = time.time()
gl = GL(be); fb = draw_scene(gl, a.frame, a.of)
dt = time.time() - t0
ref = GL(NumpyBackend()); rfb = draw_scene(ref, a.frame, a.of)
np.save(os.path.join(a.out, "fb_qwen.npy"), fb)
res = dict(free_run=a.free_run, frame=a.frame, wall_seconds=round(dt, 1),
           identical_framebuffer=bool((fb == rfb).all() and (gl.read_depth() == ref.read_depth()).all()),
           kernel_runs=gl.runs, **{k: v for k, v in be.stats.items() if k != "seconds"}, model_seconds=round(be.stats["seconds"], 1))
print(json.dumps(res))
