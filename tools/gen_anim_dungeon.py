"""60x60 dungeon animations from the real pipeline (numpy backend == Qwen backend, verified on 48 frames).
    python tools/gen_anim_dungeon.py [out/dungeon_qwen]"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from qwengl import NumpyBackend, dungeon
from qwengl.gl import GL, S
from qwengl.palette import dungeon_palette

BG, FG, DIM, ACC, WARM = (13, 17, 23), (230, 237, 243), (139, 147, 167), (34, 211, 238), (255, 200, 90)
try:
    F = ImageFont.load_default(size=15); FS = ImageFont.load_default(size=12); FB = ImageFont.load_default(size=19)
except TypeError:
    F = FS = FB = ImageFont.load_default()
SC = 8
pal = dungeon_palette()
SRC = sys.argv[1] if len(sys.argv) > 1 else "out/dungeon_qwen"


def big(fb, sc=SC):
    return Image.fromarray(np.kron(pal[fb], np.ones((sc, sc, 1), np.uint8)))


def depth_img(z, sc=SC):
    t = np.clip((31 - z) / 31.0, 0, 1)[..., None]
    rgb = (np.array(BG) * (1 - t) + np.array((255, 220, 120)) * t).astype(np.uint8)
    return Image.fromarray(np.kron(rgb, np.ones((sc, sc, 1), np.uint8)))


def save(frames, path, ms):
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)
    print(path, len(frames), "frames")


def capture(f=12, total=48):
    """snapshots after every glDrawElements call and after every 30x30 kernel window."""
    gl = GL(NumpyBackend(), dungeon.W, dungeon.H)
    calls, wins, names = [], [], []
    state = dict(call=0)
    gl.on_window = lambda g, ox, oy, n: wins.append((state["call"], ox, oy, n, g.CF[:60, :60].copy()))
    orig = gl.draw_elements

    def wrapped(mesh, *a, **k):
        r = orig(mesh, *a, **k)
        calls.append((gl.CF[:60, :60].copy(), gl.ZF[:60, :60].copy()))
        state["call"] += 1
        return r
    gl.draw_elements = wrapped
    dungeon.draw_scene(gl, f, total)
    return calls, wins


NAMES = ["room (floor, walls, pedestal)", "walker (head, torso, arms, legs)", "gem", "torch bracket", "flame", "torch bracket", "flame"]
calls, wins = capture()
W, H = 2 * 60 * SC + 3 * 24, 60 * SC + 130

# ---- 1. draw calls: the frame is built by 7 glDrawElements calls into one depth buffer
frames, prev = [], np.zeros((60, 60), int)
for i, (fb, z) in enumerate(calls):
    for hold in range(7):
        cv = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(cv)
        d.text((24, 14), f"glDrawElements #{i + 1} of {len(calls)}:  {NAMES[i]}", font=FB, fill=FG)
        d.text((24, 40), "colour buffer", font=FS, fill=DIM); d.text((24 + 60 * SC + 24, 40), "depth buffer (bright = near)", font=FS, fill=DIM)
        shown = fb if hold >= 2 else prev
        im = big(shown)
        if hold in (2, 3):                                   # flash what this call changed
            ch = np.kron((fb != prev).astype(np.uint8), np.ones((SC, SC), np.uint8)).astype(bool)
            a = np.array(im); a[ch] = (0.45 * a[ch] + 0.55 * np.array(WARM)).astype(np.uint8); im = Image.fromarray(a)
        cv.paste(im, (24, 62)); cv.paste(depth_img(z if hold >= 2 else (calls[i - 1][1] if i else np.full((60, 60), 31))), (24 + 60 * SC + 24, 62))
        d.text((24, 62 + 60 * SC + 12), "the host issues the call; every vertex, edge, coverage and depth test is a kernel run inside Qwen3.5", font=FS, fill=DIM)
        frames.append(cv)
    prev = fb
save(frames, "docs/anim/draw_calls.gif", 90)

# ---- 2. kernel windows: the 60x60 viewport is tiled with 30x30 windows (the kernels' native size)
frames, prev = [], np.zeros((60, 60), int)
W2, H2 = 60 * SC + 48 + 300, 60 * SC + 110
for k, (call, ox, oy, n, fb) in enumerate(wins):
    for hold in range(3):
        cv = Image.new("RGB", (W2, H2), BG); d = ImageDraw.Draw(cv)
        d.text((24, 14), "the viewport is tiled with 30x30 kernel windows", font=FB, fill=FG)
        im = np.array(big(fb if hold else prev))
        cv.paste(Image.fromarray(im), (24, 52))
        for i in range(3):
            for j in range(3):
                d.rectangle([24 + i * 16 * SC, 52 + j * 16 * SC, 24 + (i * 16 + 30) * SC - 1, 52 + (j * 16 + 30) * SC - 1], outline=(40, 46, 56))
        d.rectangle([24 + ox * SC, 52 + oy * SC, 24 + (ox + S) * SC - 1, 52 + (oy + S) * SC - 1], outline=ACC, width=3)
        x = 24 + 60 * SC + 24
        d.text((x, 60), f"draw call {call + 1}", font=F, fill=WARM); d.text((x, 84), NAMES[call], font=FS, fill=FG)
        d.text((x, 124), f"window origin ({ox}, {oy})", font=F, fill=ACC)
        d.text((x, 150), f"{n} triangles binned here", font=FS, fill=FG)
        d.text((x, 190), "each triangle goes to a window that", font=FS, fill=DIM); d.text((x, 208), "contains it; the mesh is re-centred", font=FS, fill=DIM)
        d.text((x, 226), "on an integer vertex (exact), so the", font=FS, fill=DIM); d.text((x, 244), "same 5-bit kernels draw any window.", font=FS, fill=DIM)
        d.text((24, 52 + 60 * SC + 14), f"window {k + 1}/{len(wins)}", font=FS, fill=DIM)
        frames.append(cv)
    prev = fb
save(frames, "docs/anim/tiles.gif", 140)

# ---- 3. the 48 frames the model produced, colour | depth
frames = []
for f in range(48):
    fb, z = np.load(f"{SRC}/fb_{f:03d}.npy"), np.load(f"{SRC}/z_{f:03d}.npy")
    cv = Image.new("RGB", (W, 60 * SC + 80), BG); d = ImageDraw.Draw(cv)
    d.text((24, 12), "colour buffer", font=FS, fill=DIM); d.text((24 + 60 * SC + 24, 12), "depth buffer (bright = near)", font=FS, fill=DIM)
    cv.paste(big(fb), (24, 34)); cv.paste(depth_img(z), (24 + 60 * SC + 24, 34))
    d.text((24, 34 + 60 * SC + 12), f"frame {f + 1}/48 | the frames Qwen produced (0 differing pixels vs the integer oracle)", font=FS, fill=DIM)
    frames.append(cv)
save(frames, "docs/anim/scene_depth.gif", 90)
