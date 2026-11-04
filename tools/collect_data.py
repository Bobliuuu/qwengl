"""Collect real numbers for the README diagrams: a pixel's journey through the kernels and a decoded token trace.
    PYTHONPATH=.:/path/to/qwentpu python tools/collect_data.py  ->  docs/data.json"""
import json
import numpy as np
from qwengl import NumpyBackend, quantise, render, rot_matrix
from qwengl import stages as st
from qwengl.geometry import S, TRIS

R = quantise(rot_matrix(0.9, 0.55, 0.2))
tr = []
fr = render(NumpyBackend(), R, trace=tr)
be = NumpyBackend()
sh = be.run_many([(st.shade_kernel(f), np.array(sum(([max(v, 0), max(-v, 0)] for v in R[:, [0, 0, 1, 1, 2, 2][f]]), []))) for f in range(6)])
# shade inputs use column k of R as (+,-) lanes; recompute properly
from qwengl.geometry import FACES, uniform_lanes
rows, cols, mm = uniform_lanes(R)
sh = be.run_many([(st.shade_kernel(f), cols[FACES[f][0]]) for f in range(6)])

# pixel covered by two different faces (front + back), so the depth test visibly decides
best = None
for j in range(S):
    for i in range(S):
        hits = [(k, t) for k, t in enumerate(tr) if t["cov"][j, i] > 0]
        faces = {t["face"] for _, t in hits}
        if len(faces) >= 2:
            best = (j, i, hits); break
    if best: break
j, i, hits = best
journey = []
for k, t in hits:
    prev_z = tr[k - 1]["z"][j, i] if k else 31
    prev_c = tr[k - 1]["fb"][j, i] if k else 0
    f = t["face"]
    journey.append(dict(tri=int(t["tri"]), face=int(f), edges_pos=[int(t["edges"][e][j, i]) for e in range(3)],
                        edges_neg=[int(t["edges_neg"][e][j, i]) for e in range(3)], cov=int(t["cov"][j, i]),
                        zn=int(sh[f][1]), cn=int(sh[f][0]), z_before=int(prev_z), z_after=int(t["z"][j, i]),
                        c_before=int(prev_c), c_after=int(t["fb"][j, i])))
# decoded token trace of the cover kernel on this pixel's real lanes
from qwentpu import circuit as C
from qwentpu.sequencer2 import ProgramMachine
seq = ProgramMachine(2, 16, 5, 5).install()
k0 = hits[0][1]
x = np.array(list(k0["edges"][:, j, i]) + list(k0["edges_neg"][:, j, i]) + [0] * 6)
ker = st.cover_kernel()
p = seq.program_prompt(ker.Ws, x); em = seq.expected(p)


def dec(t):
    if C.CELL0 <= t < C.ACT0:
        u = t - C.CELL0; cell, w = divmod(u, C.NW); return ["cell", f"W[{cell // C.NC},{cell % C.NC}] = {w - C.NW if w >= C.NW // 2 else w}"]
    if C.ACT0 <= t < C.READ0:
        u = t - C.ACT0; return ["act", f"ACT row {u // C.NOUT} = {u % C.NOUT}"]
    if C.READ0 <= t < seq.outk0:
        return ["read", f"READ col {t - C.READ0}"]
    if seq.outk0 <= t < seq.halt:
        u = t - seq.outk0; return ["out", f"OUT {u // C.NOUT} = {u % C.NOUT}"]
    return ["halt", "HALT"]


out = dict(pixel=[int(j), int(i)], journey=journey, angles=[0.9, 0.55, 0.2],
           kernel_out=[int(v) for v in ker.run_np(x[None])[0]], kernel_in=[int(v) for v in x],
           prompt_len=len(p), n_cells=sum(1 for t in p if C.CELL0 <= t < C.ACT0),
           prompt_tail=[dec(t) for t in p[-18:]], prompt_head=[dec(t) for t in p[:4]],
           emitted=[dec(t) for t in em] + [dec(seq.halt)])
json.dump(out, open("docs/data.json", "w"), indent=1)
print(out["pixel"], len(journey), out["prompt_len"], len(out["emitted"]), out["kernel_out"])
for q in journey: print(q)
