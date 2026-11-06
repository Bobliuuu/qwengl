"""docs/*.svg -> docs/png/*.png (2400 px wide, light theme) for places that do not take SVG (Twitter, slides).
The SVGs use CSS variables + a dark-mode media query, which rsvg ignores, so the light palette is inlined first.
    python docs/tools/svg_to_png.py        (needs rsvg-convert)"""
import glob, os, re, subprocess, tempfile

L = dict(bg="#fbfbfd", panel="#ffffff", soft="#eef0f9", ink="#14172b", mute="#5a6285", line="#c9cde3", teal="#0f9d8a", violet="#6246ea",
         amber="#c97a0a", rose="#d1385c", vfill="#ece8ff", tfill="#dcf5f0", afill="#fff0d6", rfill="#ffe3ea")
here = os.path.dirname(os.path.abspath(__file__))
docs = os.path.dirname(here)
os.makedirs(os.path.join(docs, "png"), exist_ok=True)
for f in sorted(glob.glob(os.path.join(docs, "*.svg"))):
    s = open(f).read()
    s = re.sub(r"var\(--(\w+)\)", lambda m: L[m.group(1)], s)
    s = re.sub(r"<style>.*?</style>", "<style>text{font-family:Helvetica,Arial,sans-serif}.b{font-weight:700}.mono{font-family:Menlo,monospace}</style>", s, flags=re.S)
    with tempfile.NamedTemporaryFile("w", suffix=".svg", delete=False) as t:
        t.write(s)
    out = os.path.join(docs, "png", os.path.basename(f)[:-4] + ".png")
    subprocess.run(["rsvg-convert", "-w", "2400", "-b", L["bg"], t.name, "-o", out], check=True)
    os.unlink(t.name)
    print(out)
