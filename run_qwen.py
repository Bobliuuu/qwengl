"""One scene frame (--scene orbit: 3 draws, --scene dungeon: 60x60, 7 draws) with every kernel executed by a real Qwen3.5, cross-checked against numpy.
    QWEN_DIR=/path/to/Qwen3.5-0.8B python run_qwen.py --frame 5 [--free-run]"""
import argparse, json, os, time
import numpy as np
from qwengl import NumpyBackend
from qwengl.gl import GL
from qwengl.qwen_backend import QwenBackend
from qwengl import dungeon, scene

ap = argparse.ArgumentParser()
ap.add_argument("--qwen", default=os.environ.get("QWEN_DIR"))
ap.add_argument("--device", default="cuda")
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--scene", default="orbit", choices=["orbit", "dungeon"])
ap.add_argument("--frame", type=int, default=5)
ap.add_argument("--of", type=int, default=0)
ap.add_argument("--out", default="out")
ap.add_argument("--graph", action="store_true")
ap.add_argument("--free-run", action="store_true", help="true autoregressive generation: host only does argmax + append")
a = ap.parse_args()
a.of = a.of or (36 if a.scene == 'orbit' else 48)
os.makedirs(a.out, exist_ok=True)
be = QwenBackend(a.qwen, a.device, a.batch, graph=a.graph, free_run=a.free_run)
t0 = time.time()
def make():
    return GL(be) if a.scene == "orbit" else GL(be, dungeon.W, dungeon.H)


def draw(gl):
    return (scene if a.scene == "orbit" else dungeon).draw_scene(gl, a.frame, a.of)


gl = make(); fb = draw(gl)
dt = time.time() - t0
ref = GL(NumpyBackend()) if a.scene == 'orbit' else GL(NumpyBackend(), dungeon.W, dungeon.H); rfb = draw(ref)
np.save(os.path.join(a.out, "fb_qwen.npy"), fb)
res = dict(scene=a.scene, free_run=a.free_run, frame=a.frame, wall_seconds=round(dt, 1),
           identical_framebuffer=bool((fb == rfb).all() and (gl.read_depth() == ref.read_depth()).all()),
           kernel_runs=gl.runs, **{k: v for k, v in be.stats.items() if k != "seconds"}, model_seconds=round(be.stats["seconds"], 1))
print(json.dumps(res))
