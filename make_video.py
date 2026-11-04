"""fb_XXX.npy frames -> GIF + MP4 (nearest-neighbour upscale of the 30x30 framebuffer).
    python make_video.py out/qwen docs/qwengl_demo"""
import glob, subprocess, sys
import numpy as np
from PIL import Image
from qwengl.palette import to_rgb

src, dst = sys.argv[1], sys.argv[2]
fbs = [np.load(f) for f in sorted(glob.glob(f"{src}/fb_*.npy"))]
imgs = [Image.fromarray(to_rgb(fb, 16)) for fb in fbs]
imgs[0].save(dst + ".gif", save_all=True, append_images=imgs[1:], duration=80, loop=0)
for i, im in enumerate(imgs):
    im.save(f"/tmp/_qwengl_{i:03d}.png")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "12", "-i", "/tmp/_qwengl_%03d.png", "-pix_fmt", "yuv420p", dst + ".mp4"], check=True)
print(len(imgs), "frames ->", dst + ".gif/.mp4")
