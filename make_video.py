"""fb_XXX.npy frames -> GIF + MP4 (nearest-neighbour upscale).
    python make_video.py out/dungeon docs/qwengl_dungeon --palette dungeon --scale 9 --fps 12"""
import argparse, glob, os, subprocess, tempfile
import numpy as np
from PIL import Image
from qwengl.palette import dungeon_palette, palette, to_rgb_pal

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--palette", default="cube", choices=["cube", "dungeon"])
ap.add_argument("--scale", type=int, default=16)
ap.add_argument("--fps", type=int, default=12)
a = ap.parse_args()
pal = dungeon_palette() if a.palette == "dungeon" else palette()
fbs = [np.load(f) for f in sorted(glob.glob(f"{a.src}/fb_*.npy"))]
imgs = [Image.fromarray(to_rgb_pal(fb, pal, a.scale)) for fb in fbs]
imgs[0].save(a.dst + ".gif", save_all=True, append_images=imgs[1:], duration=int(1000 / a.fps), loop=0)
with tempfile.TemporaryDirectory() as tmp:
    for i, im in enumerate(imgs):
        im.save(f"{tmp}/f_{i:03d}.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", f"{tmp}/f_%03d.png", "-pix_fmt", "yuv420p", a.dst + ".mp4"], check=True)
print(len(imgs), "frames ->", a.dst + ".gif/.mp4")
