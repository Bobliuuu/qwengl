"""1280x720 demo video from the real run: title, API, draw calls, kernel windows, the 48 Qwen frames with live counters, outro.
    PYTHONPATH=. python tools/make_demo_video.py [out/dungeon_qwen] [docs/qwengl_demo_video.mp4]"""
import json, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from qwengl import NumpyBackend, dungeon
from qwengl.gl import GL, S
from qwengl.palette import dungeon_palette

SRC = sys.argv[1] if len(sys.argv) > 1 else "out/dungeon_qwen"
DST = sys.argv[2] if len(sys.argv) > 2 else "docs/qwengl_demo_video.mp4"
W, H, FPS = 1280, 720, 24
BG, FG, DIM, ACC, WARM, PUR = (13, 17, 23), (230, 237, 243), (139, 147, 167), (34, 211, 238), (255, 200, 90), (124, 92, 255)
pal = dungeon_palette()


def font(sz, mono=False):
    for p in (["/System/Library/Fonts/Menlo.ttc"] if mono else ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf"]):
        try:
            return ImageFont.truetype(p, sz)
        except OSError:
            pass
    return ImageFont.load_default(size=sz)


T, H1, B, SM, M = font(84), font(40), font(26), font(20), font(24, True)

ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", "-preset", "slow", "-movflags", "+faststart", DST], stdin=subprocess.PIPE)
nframes = 0


def emit(im, n=1):
    global nframes
    b = im.convert("RGB").tobytes()
    for _ in range(n):
        ff.stdin.write(b)
    nframes += n


def canvas():
    cv = Image.new("RGB", (W, H), BG)
    return cv, ImageDraw.Draw(cv)


def big(fb, sc):
    return Image.fromarray(np.kron(pal[fb], np.ones((sc, sc, 1), np.uint8)))


def depth_img(z, sc):
    t = np.clip((31 - z) / 31.0, 0, 1)[..., None]
    return Image.fromarray(np.kron((np.array(BG) * (1 - t) + np.array((255, 220, 120)) * t).astype(np.uint8), np.ones((sc, sc, 1), np.uint8)))


def center(d, y, text, f, fill):
    w = d.textlength(text, font=f)
    d.text(((W - w) / 2, y), text, font=f, fill=fill)


def fade(draw_fn, n_in=10, n_hold=0, n_out=10):
    """draw_fn() -> PIL image; emit with a fade in/out."""
    im = draw_fn()
    black = Image.new("RGB", (W, H), BG)
    for i in range(n_in):
        emit(Image.blend(black, im, (i + 1) / n_in))
    emit(im, n_hold)
    for i in range(n_out):
        emit(Image.blend(im, black, (i + 1) / n_out))


# ---------------------------------------------------------------- 1. title
def title():
    cv, d = canvas()
    center(d, 200, "qwengl", T, FG)
    center(d, 320, "a graphics library that runs inside a language model", H1, ACC)
    center(d, 410, "glClear  ·  glLight  ·  glDrawElements  ·  glReadPixels", M, DIM)
    center(d, 470, "every vertex, edge and pixel is computed by Qwen3.5-0.8B", B, DIM)
    return cv


fade(title, 12, 40, 12)

# ---------------------------------------------------------------- 2. the API, typed out
code = ['gl = GL(QwenBackend(...), 60, 60)       # every kernel runs inside Qwen3.5',
        'gl.light((1, -2, -2))                    # glLight',
        'gl.clear()                               # glClear',
        'gl.draw_elements(room,   R, t)           # glDrawElements',
        'gl.draw_elements(walker, R, t)           # ...into one shared depth buffer',
        'gl.draw_elements(gem,    R, t)',
        'img = gl.read_pixels()                   # glReadPixels']
total = sum(len(c) for c in code)
for n in list(range(0, total, 5)) + [total] * 28:
    cv, d = canvas()
    d.text((70, 60), "A GL-style API", font=H1, fill=FG)
    d.text((70, 118), "the host issues calls; the model does the arithmetic", font=SM, fill=DIM)
    left, y = n, 220
    for c in code:
        take = min(len(c), max(left, 0)); left -= len(c)
        d.text((70, y), c[:take], font=M, fill=ACC if "#" not in c[:take] else FG)
        if "#" in c[:take]:
            k = c.index("#")
            d.text((70, y), c[:k], font=M, fill=FG); d.text((70 + d.textlength(c[:k], font=M), y), c[k:take], font=M, fill=DIM)
        y += 46
    emit(cv)

# ---------------------------------------------------------------- 3. draw calls (real snapshots)
gl = GL(NumpyBackend(), dungeon.W, dungeon.H)
calls, wins, state = [], [], dict(call=0)
gl.on_window = lambda g, ox, oy, n: wins.append((state["call"], ox, oy, n, g.CF[:60, :60].copy()))
orig = gl.draw_elements


def wrapped(mesh, *a, **k):
    r = orig(mesh, *a, **k)
    calls.append((gl.CF[:60, :60].copy(), gl.ZF[:60, :60].copy())); state["call"] += 1
    return r


gl.draw_elements = wrapped
dungeon.draw_scene(gl, 12, 48)
NAMES = ["the room: floor, walls, pedestal", "the walker", "the gem", "torch bracket", "torch flame", "torch bracket", "torch flame"]
SC = 9
prev, pz = np.zeros((60, 60), int), np.full((60, 60), 31)
for i, (fb, z) in enumerate(calls):
    for hold in range(16):
        cv, d = canvas()
        d.text((60, 36), f"glDrawElements  {i + 1} / {len(calls)}", font=H1, fill=FG)
        d.text((60, 90), NAMES[i], font=B, fill=WARM)
        im = np.array(big(fb if hold >= 4 else prev, SC))
        if 4 <= hold < 9:
            ch = np.kron((fb != prev).astype(np.uint8), np.ones((SC, SC), np.uint8)).astype(bool)
            im[ch] = (0.45 * im[ch] + 0.55 * np.array(WARM)).astype(np.uint8)
        cv.paste(Image.fromarray(im), (60, 150)); cv.paste(depth_img(z if hold >= 4 else pz, SC), (60 + 60 * SC + 40, 150))
        d.text((60, 150 + 60 * SC + 10), "colour buffer", font=SM, fill=DIM); d.text((60 + 60 * SC + 40, 150 + 60 * SC + 10), "depth buffer (bright = near)", font=SM, fill=DIM)
        emit(cv)
    prev, pz = fb, z

# ---------------------------------------------------------------- 4. kernel windows
SC = 9
prev = np.zeros((60, 60), int)
for k, (call, ox, oy, n, fb) in enumerate(wins):
    for hold in range(7):
        cv, d = canvas()
        d.text((60, 36), "the kernels are 30×30, so the viewport is tiled", font=H1, fill=FG)
        cv.paste(big(fb if hold >= 2 else prev, SC), (60, 130))
        for i in range(3):
            for j in range(3):
                d.rectangle([60 + i * 16 * SC, 130 + j * 16 * SC, 60 + (i * 16 + 30) * SC - 1, 130 + (j * 16 + 30) * SC - 1], outline=(40, 46, 56))
        d.rectangle([60 + ox * SC, 130 + oy * SC, 60 + (ox + S) * SC - 1, 130 + (oy + S) * SC - 1], outline=ACC, width=4)
        x = 60 + 60 * SC + 50
        d.text((x, 150), f"draw call {call + 1}: {NAMES[call]}", font=B, fill=WARM)
        d.text((x, 200), f"window origin ({ox}, {oy})", font=B, fill=ACC)
        d.text((x, 240), f"{n} triangles binned here", font=B, fill=FG)
        for j, line in enumerate(["Each triangle goes to a 30×30 window", "that contains it. The mesh is re-centred", "on an integer vertex (exact), so the same", "5-bit kernels draw every window."]):
            d.text((x, 320 + 34 * j), line, font=SM, fill=DIM)
        emit(cv)
    prev = fb

# ---------------------------------------------------------------- 5. the 48 frames Qwen produced, live counters
recs = [json.loads(l) for l in open(f"{SRC}/frames.jsonl")]
cum_runs = np.cumsum([sum(v for k, v in r["runs"].items() if k != "rop_skipped") for r in recs])
cum_tok = np.cumsum([r["tokens"] for r in recs]); cum_pos = np.cumsum([r["positions"] for r in recs]); cum_mm = np.cumsum([r["mismatches"] for r in recs])
SC = 10
for loop in range(3):
    for f in range(48):
        fb, z = np.load(f"{SRC}/fb_{f:03d}.npy"), np.load(f"{SRC}/z_{f:03d}.npy")
        cv, d = canvas()
        d.text((60, 30), "48 frames, computed inside Qwen3.5-0.8B", font=H1, fill=FG)
        cv.paste(big(fb, SC), (60, 100)); cv.paste(depth_img(z, 4), (60 + 60 * SC + 30, 100))
        d.text((60 + 60 * SC + 30, 100 + 240 + 6), "depth", font=SM, fill=DIM)
        x, y = 60 + 60 * SC + 30, 410
        for label, val, col in (("frame", f"{f + 1} / 48", FG), ("kernel runs", f"{int(cum_runs[f]):,}", ACC), ("tokens through the model", f"{int(cum_tok[f]):,}", ACC),
                                ("predicted positions", f"{int(cum_pos[f]):,}", PUR), ("mismatches vs integer oracle", f"{int(cum_mm[f])}", (90, 220, 130))):
            d.text((x, y), label, font=SM, fill=DIM); d.text((x + 330, y - 4), val, font=B, fill=col); y += 40
        emit(cv, 2)

# ---------------------------------------------------------------- 6. outro
def outro():
    cv, d = canvas()
    center(d, 150, "qwengl", T, FG)
    center(d, 270, "github.com/Bobliuuu/qwengl", H1, ACC)
    lines = ["812,548 kernel runs · 4.87 M predicted tokens · 0 mismatches",
             "one full frame also run free-running (host does only argmax + append)",
             "isometric, flat-shaded, 5-bit colour: no perspective divide, so not first person",
             "the host still does rotation, triangle binning and draw order"]
    for i, l in enumerate(lines):
        center(d, 380 + 42 * i, l, B if i == 0 else SM, FG if i == 0 else DIM)
    return cv


fade(outro, 12, 72, 12)
ff.stdin.close(); ff.wait()
print(DST, nframes, "frames", round(nframes / FPS, 1), "s")
