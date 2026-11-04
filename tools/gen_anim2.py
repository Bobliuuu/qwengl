"""More README animations: token trace, raster sweep, life of a pixel.   PYTHONPATH=. python tools/gen_anim2.py"""
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from qwengl import NumpyBackend, quantise, render, rot_matrix
from qwengl.geometry import S, GRP
from qwengl.palette import palette

BG, FG, DIM = (13, 17, 23), (230, 237, 243), (139, 147, 167)
TEAL, VIO, ROSE, AMB, PAN = (53, 214, 192), (157, 134, 255), (255, 107, 139), (255, 180, 84), (22, 27, 34)
F = ImageFont.load_default(size=15); FS = ImageFont.load_default(size=12); FB = ImageFont.load_default(size=20); FM = ImageFont.load_default(size=13)
D = json.load(open("docs/data.json")); pal = palette()


def save(frames, path, ms):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=False)
    print(path, len(frames))


def rr(d, box, fill, outline=None):
    d.rounded_rectangle(box, radius=7, fill=fill, outline=outline, width=2)


# ------------------------------------------------------------------ 1. token trace
em = D["emitted"]; col = {"read": VIO, "act": TEAL, "out": ROSE, "halt": AMB}
W, H = 1000, 470
frames = []
steps = list(range(0, len(em) + 1, 2)) + [len(em)] * 8
for n in steps:
    cv = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(cv)
    d.text((24, 14), "the model emits every token itself", font=FB, fill=FG)
    rr(d, (24, 54, 24 + 700, 90), (60, 40, 10), AMB); d.text((40, 63), f"PROMPT  {D['n_cells']} CELL (weights)  +  32 ACT (input)   fed by the host", font=F, fill=AMB)
    d.text((24, 104), "EMITTED  (argmax, fed back)", font=F, fill=TEAL)
    for k in range(min(n, len(em))):
        kind, txt = em[k]; r, c = divmod(k, 8); x, y = 24 + c * 120, 134 + r * 36
        cc = col[kind]
        rr(d, (x, y, x + 112, y + 28), tuple(int(v * .22) for v in cc), cc)
        d.text((x + 8, y + 7), txt.replace("ACT row ", "ACT ").replace("READ col ", "READ "), font=FM, fill=FG)
    cur = min(n, len(em))
    d.text((24, H - 52), f"{cur}/{len(em)} tokens", font=F, fill=DIM)
    d.text((24, H - 30), "READ k  ->  model emits ACT(row; y)  ->  next layer  ->  OUT j = y  ->  HALT", font=FS, fill=DIM)
    frames.append(cv)
save(frames, "docs/anim/token_trace.gif", 90)

# ------------------------------------------------------------------ 2. raster sweep over 6-pixel groups
tr = []
render(NumpyBackend(), quantise(rot_matrix(0.9, 0.55, 0.2)), trace=tr)
t = tr[5]; SC = 12
ys = np.where(t["edges"].any(axis=(0, 2)) | t["edges_neg"].any(axis=(0, 2)))[0]
groups = [(row, seg) for row in range(ys.min(), ys.max() + 1) for seg in range(S // GRP) if t["edges"][:, row, seg * GRP:(seg + 1) * GRP].any() or t["edges_neg"][:, row, seg * GRP:(seg + 1) * GRP].any()]
W, H = S * SC + 330, S * SC + 90
frames = []
for g in range(0, len(groups) + 1):
    for hold in (1,) if g < len(groups) else (1,) * 10:
        cv = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(cv)
        d.text((24, 12), "rasteriser: 6 pixels per kernel run", font=FB, fill=FG)
        img = np.zeros((S, S, 3), dtype=np.uint8) + np.array(BG, dtype=np.uint8)
        for (row, seg) in groups[:g]:
            sl = slice(seg * GRP, seg * GRP + GRP)
            img[row, sl] = np.where(t["cov"][row, sl, None] > 0, np.array(TEAL), np.array((34, 40, 52)))
        big = np.kron(img, np.ones((SC, SC, 1), dtype=np.uint8)); cv.paste(Image.fromarray(big), (24, 52))
        if g < len(groups):
            row, seg = groups[g]
            d.rectangle([24 + seg * GRP * SC, 52 + row * SC, 24 + (seg + 1) * GRP * SC, 52 + (row + 1) * SC], outline=AMB, width=3)
            d.text((S * SC + 52, 60), f"run {g + 1}/{len(groups)}  row {row}  px {seg * GRP}..{seg * GRP + 5}", font=F, fill=AMB)
            sl = slice(seg * GRP, seg * GRP + GRP)
            for e in range(3):
                d.text((S * SC + 52, 100 + e * 58), f"edge {e}:  e >= 0", font=FS, fill=DIM)
                for p in range(GRP):
                    v = int(t["edges"][e, row, sl][p]); c = (int(v / 31 * 53 + 20), int(v / 31 * 214 + 20), int(v / 31 * 192 + 20))
                    d.rectangle([S * SC + 52 + p * 38, 118 + e * 58, S * SC + 52 + p * 38 + 34, 118 + e * 58 + 26], fill=c)
                    d.text((S * SC + 58 + p * 38, 124 + e * 58), str(v), font=FS, fill=FG)
            d.text((S * SC + 52, 278), "AND of 3 edges ->", font=FS, fill=DIM)
            for p in range(GRP):
                v = int(t["cov"][row, sl][p]); d.rectangle([S * SC + 52 + p * 38, 296, S * SC + 52 + p * 38 + 34, 322], fill=TEAL if v else (34, 40, 52)); d.text((S * SC + 58 + p * 38, 302), str(v), font=FS, fill=BG if v else DIM)
        else:
            d.text((S * SC + 52, 60), f"{len(groups)} runs -> this triangle's coverage", font=F, fill=TEAL)
        d.text((24, H - 26), f"triangle 5, face {t['face']}: three edge functions per group, then a hard AND", font=FS, fill=DIM)
        frames.append(cv)
save(frames, "docs/anim/raster_sweep.gif", 110)

# ------------------------------------------------------------------ 3. life of a pixel
J = D["journey"]; W, H = 900, 400
frames = []
for n in range(len(J) + 1):
    for hold in range(5):
        cv = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(cv)
        d.text((24, 14), f"life of pixel ({D['pixel'][1]}, {D['pixel'][0]}): the depth test, live", font=FB, fill=FG)
        z = J[n - 1]["z_after"] if n else 31; c = J[n - 1]["c_after"] if n else 0
        # Z gauge
        d.text((24, 64), "Z buffer (0 = near, 31 = far)", font=FS, fill=DIM)
        d.rectangle([24, 86, 24 + 32 * 18, 112], outline=(60, 66, 80), width=2)
        d.rectangle([26, 88, 26 + max(2, z * 18), 110], fill=AMB)
        d.text((24 + 32 * 18 + 12, 90), f"Z = {z}", font=F, fill=AMB)
        d.text((24, 140), "framebuffer colour", font=FS, fill=DIM)
        d.rectangle([24, 160, 124, 210], fill=tuple(pal[c]), outline=(60, 66, 80), width=2); d.text((140, 176), f"index {c}", font=F, fill=FG)
        if n < len(J):
            q = J[n]; ok = q["z_after"] != q["z_before"]
            d.text((24, 240), f"incoming: triangle {q['tri']} (face {q['face']})  zn = {q['zn']}  colour {q['cn']}", font=F, fill=FG)
            rr(d, (24, 268, 24 + 36, 268 + 28), tuple(pal[q["cn"]]), None)
            d.text((24, 316), f"test:  Z - zn = {q['z_before'] - q['zn']}  ->  " + ("WRITE (nearer)" if ok else "REJECT (not nearer)"), font=FB, fill=TEAL if ok else ROSE)
            d.text((24, 350), "g = relu(Z - zn + s31 + cov - 62)    Z' = relu(Z - g)", font=FS, fill=DIM)
        else:
            d.text((24, 260), "final: the nearest face wins", font=FB, fill=TEAL)
        frames.append(cv)
save(frames, "docs/anim/life_of_a_pixel.gif", 150)
