"""README animations, generated from the real kernel outputs (numpy backend == Qwen backend, verified on 36 frames).
    python tools/gen_anim.py"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from qwengl import NumpyBackend, quantise, render, rot_matrix
from qwengl.geometry import S
from qwengl.palette import palette

BG, FG, DIM, ACC, ACC2 = (13, 17, 23), (230, 237, 243), (139, 147, 167), (34, 211, 238), (124, 92, 255)
try:
    F = ImageFont.load_default(size=15); FS = ImageFont.load_default(size=12); FB = ImageFont.load_default(size=19)
except TypeError:
    F = FS = FB = ImageFont.load_default()
SC = 9


def up(a, rgb):
    return Image.fromarray(np.kron(rgb[a], np.ones((SC, SC, 1), dtype=np.uint8)))


def ramp(v, hi, c0=(13, 17, 23), c1=ACC):
    t = np.clip(np.asarray(v, float) / hi, 0, 1)[..., None]
    return (np.array(c0) * (1 - t) + np.array(c1) * t).astype(np.uint8)


def panel(canvas, x, y, title, img):
    d = ImageDraw.Draw(canvas)
    d.text((x, y), title, font=FS, fill=DIM)
    canvas.paste(img, (x, y + 20))
    d.rectangle([x - 1, y + 19, x + img.width, y + 20 + img.height], outline=(48, 54, 61))


def save(frames, path, ms):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)
    print(path, len(frames), "frames")


pal = palette()

# ---------------------------------------------------------------- 1. draw calls: 12 triangles through edges -> coverage -> depth -> fb
tr = []
render(NumpyBackend(), quantise(rot_matrix(0.9, 0.55, 0.2)), trace=tr)
W, H = 4 * S * SC + 5 * 24, S * SC + 120
frames = []
for step, t in enumerate(tr):
    for hold in range(3):
        cv = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(cv)
        d.text((24, 14), f"triangle {step + 1}/{len(tr)}  |  face {t['face']}  |  every pixel below is a kernel output", font=FB, fill=FG)
        e = t["edges"]
        x0 = 24
        # edges + coverage reveal progressively across the 3 holds
        edge_img = np.zeros((S, S, 3), dtype=np.uint8)
        for k in range(3):
            edge_img[..., k] = (np.clip(e[k] / 31, 0, 1) * 255).astype(np.uint8) if hold >= 0 else 0
        panel(cv, x0, 56, "3 edge functions  (R,G,B = e0,e1,e2 >= 0)", Image.fromarray(np.kron(edge_img, np.ones((SC, SC, 1), dtype=np.uint8))))
        cov = t["cov"] if hold >= 1 else np.zeros_like(t["cov"])
        panel(cv, x0 + S * SC + 24, 56, "coverage = AND of edges", up((cov > 0).astype(int), np.array([[13, 17, 23], list(ACC)], dtype=np.uint8)))
        z = t["z"] if hold >= 2 else tr[step - 1]["z"] if step else np.full((S, S), 31)
        panel(cv, x0 + 2 * (S * SC + 24), 56, "depth buffer (bright = near)", Image.fromarray(np.kron(ramp(31 - z, 31, (13, 17, 23), (255, 220, 120)), np.ones((SC, SC, 1), dtype=np.uint8))))
        fb = t["fb"] if hold >= 2 else (tr[step - 1]["fb"] if step else np.zeros((S, S), int))
        panel(cv, x0 + 3 * (S * SC + 24), 56, "framebuffer", up(fb, pal))
        d.text((24, H - 38), "depth test: write only where covered AND Z - zn >= 1  |  gated relu, no multiplier, no branch", font=F, fill=DIM)
        frames.append(cv)
save(frames, "docs/anim/draw_calls.gif", 220)

# ---------------------------------------------------------------- 2. rotating cube: colour + depth buffer side by side
frames = []
for f in range(36):
    th = 2 * np.pi * f / 36
    fr = render(NumpyBackend(), quantise(rot_matrix(0.55 + 0.35 * np.sin(th), th, 0.25 * np.cos(th))))
    cv = Image.new("RGB", (2 * S * SC + 72, S * SC + 80), BG); d = ImageDraw.Draw(cv)
    panel(cv, 24, 10, "framebuffer", up(fr.fb, pal))
    panel(cv, 48 + S * SC, 10, "depth buffer", Image.fromarray(np.kron(ramp(31 - fr.zbuf, 31, (13, 17, 23), (255, 220, 120)), np.ones((SC, SC, 1), dtype=np.uint8))))
    d.text((24, S * SC + 40), f"frame {f + 1}/36 | same 36 frames the Qwen run produced (0 differing pixels)", font=FS, fill=DIM)
    frames.append(cv)
save(frames, "docs/anim/cube_depth.gif", 90)

# ---------------------------------------------------------------- 3. how ReLU layers make a hard edge
e = np.arange(-8, 9)
l1 = np.clip(np.maximum(e + 1, 0), 0, 31)
l2 = np.clip(15 * l1, 0, 31)
l3 = np.clip(15 * l2, 0, 31)
curves = [("edge function e", e, None), ("layer 1: relu(e + 1)", l1, 31), ("layer 2: relu(15 * x)", l2, 31), ("layer 3: relu(15 * x)", l3, 31)]
frames = []
CW, CH = 760, 360
for n in range(1, 5):
    for hold in range(5):
        cv = Image.new("RGB", (CW, CH), BG); d = ImageDraw.Draw(cv)
        d.text((24, 14), "a ReLU stack turns a smooth edge function into a hard 0/31 inside test", font=FB, fill=FG)
        ox, oy, pw, ph = 70, 60, 640, 250
        d.rectangle([ox, oy, ox + pw, oy + ph], outline=(48, 54, 61))
        d.line([ox + pw // 2, oy, ox + pw // 2, oy + ph], fill=(48, 54, 61))
        d.text((ox + pw // 2 + 6, oy + 4), "e = 0  (the edge)", font=FS, fill=DIM)
        for i in range(n):
            name, y, hi, = curves[i]
            col = [(139, 147, 167), ACC2, ACC, (255, 220, 120)][i]
            pts = []
            for xv, yv in zip(e, y):
                yy = (np.clip(yv, -8, 31) + 8) / 39 if hi is None else yv / 31
                pts.append((ox + (xv + 8) / 16 * pw, oy + ph - yy * ph * (0.9 if i else 0.9)))
            for j in range(len(pts) - 1):
                d.line([pts[j], (pts[j + 1][0], pts[j][1]), pts[j + 1]], fill=col, width=3 if i == n - 1 else 2)
            d.text((ox + 8, oy + 8 + 18 * i), name, font=FS, fill=col)
        d.text((24, CH - 34), "weights are 5-bit, so x15 is the biggest gain one layer has; two of them saturate any e >= 0 to exactly 31", font=FS, fill=DIM)
        frames.append(cv)
save(frames, "docs/anim/hard_edge.gif", 160)
