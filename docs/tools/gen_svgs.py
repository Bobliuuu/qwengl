"""Generates docs/*.svg (light/dark adaptive, CSS-animated).  Pure stdlib + numpy.  Reads docs/data.json (tools/collect_data.py)
and results/*.  Run:  python3 docs/tools/gen_svgs.py     Edit text here, not in the SVGs."""
import html, json, os
import numpy as np
DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = json.load(open(DOCS + "/data.json"))
STYLE = '''<style>
:root{--bg:#fbfbfd;--panel:#ffffff;--soft:#eef0f9;--ink:#14172b;--mute:#5a6285;--line:#c9cde3;--teal:#0f9d8a;--violet:#6246ea;--amber:#c97a0a;--rose:#d1385c;--vfill:#ece8ff;--tfill:#dcf5f0;--afill:#fff0d6;--rfill:#ffe3ea}
@media (prefers-color-scheme:dark){:root{--bg:#0b0e1a;--panel:#12172b;--soft:#171d36;--ink:#e6e9ff;--mute:#9aa4cf;--line:#2b3252;--teal:#35d6c0;--violet:#9d86ff;--amber:#ffb454;--rose:#ff6b8b;--vfill:#1d1a3d;--tfill:#10312f;--afill:#33260f;--rfill:#3a1824}}
text{font-family:Inter,"Helvetica Neue",Helvetica,Arial,sans-serif;fill:var(--ink)}
.m{fill:var(--mute)}.b{font-weight:700}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre}
.flow{stroke-dasharray:7 6;animation:dash 1.1s linear infinite}
@keyframes dash{to{stroke-dashoffset:-13}}
.pulse{animation:pl 2.4s ease-in-out infinite}@keyframes pl{50%{opacity:.45}}
.rise{animation:rs 3.2s ease-in-out infinite}@keyframes rs{50%{transform:translateY(-6px)}}
@media (prefers-reduced-motion:reduce){.flow,.pulse,.rise{animation:none}}
</style>'''


def head(w, h, defs=""):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" role="img">{STYLE}<defs>'
            f'<marker id="ar" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--mute)"/></marker>'
            f'{defs}</defs><rect width="{w}" height="{h}" rx="18" fill="var(--bg)"/>')


def t(x, y, s, size=14, cls="", anchor="start", fill=None):
    f = f' fill="{fill}"' if fill else ""
    return f'<text x="{x}" y="{y}" font-size="{size}" class="{cls}" text-anchor="{anchor}"{f}>{html.escape(str(s))}</text>'


def box(x, y, w, h, col, fill, title, lines=(), size=14, cls=""):
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="var(--{fill})" stroke="var(--{col})" stroke-width="2" class="{cls}"/>'
    s += t(x + w / 2, y + 25, title, size + 1, "b", "middle", f"var(--{col})")
    for i, l in enumerate(lines):
        s += t(x + w / 2, y + 47 + i * 17, l, size - 2, "m", "middle")
    return s


def arrow(x1, y1, x2, y2, flow=True, col="mute"):
    c = ' class="flow"' if flow else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="var(--{col})" stroke-width="2"{c} marker-end="url(#ar)"/>'


def path(d, flow=True, col="mute", w=2):
    c = ' class="flow"' if flow else ""
    return f'<path d="{d}" fill="none" stroke="var(--{col})" stroke-width="{w}"{c} marker-end="url(#ar)"/>'


def save(name, body, w, h, defs=""):
    open(f"{DOCS}/{name}.svg", "w").write(head(w, h, defs) + body + "</svg>")
    print(name)


PAL = None
def palette():
    import colorsys
    HUES = [0.00, 0.08, 0.15, 0.33, 0.58, 0.78]
    p = {0: "#0e1018"}
    for f in range(6):
        for lv in range(5):
            r, g, b = colorsys.hsv_to_rgb(HUES[f], 0.72, 0.30 + 0.17 * lv)
            p[1 + 5 * f + lv] = "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
    return p
PAL = palette()

# ============================================================ banner: the real cube, as SVG rects
fb = np.load(DOCS + "/../out/qwen/fb_006.npy")
W, H = 1200, 360
s = t(40, 62, "qwengl", 46, "b") + t(40, 96, "a graphics pipeline computed by a language model", 19, "m")
s += t(40, 132, "vertex transform · edge functions · coverage · depth test · framebuffer", 14, "m")
s += t(40, 154, "as integer ReLU layers inside a stock Qwen3.5-0.8B", 14, "m")
for i, (lab, col) in enumerate((("104,940 kernel runs", "teal"), ("31.2 M tokens", "violet"), ("0 mismatches", "amber"))):
    s += f'<rect x="{40 + i * 190}" y="190" width="176" height="46" rx="23" fill="var(--{col[0]}fill)" stroke="var(--{col})" stroke-width="2"/>'.replace("tfill", "tfill")
    s += t(40 + i * 190 + 88, 219, lab, 15, "b", "middle", f"var(--{col})")
s += t(40, 280, "no shaders · no rasteriser code · the model's own tokens are the pixels", 15, "m")
s += t(40, 306, "30×30 pixels · 5-bit colour · 36 frames · one A100", 13, "m")
px = 9
ox, oy = 760, 30
s += f'<rect x="{ox - 14}" y="{oy - 14}" width="{30 * px + 28}" height="{30 * px + 28}" rx="16" fill="#0e1018" stroke="var(--line)" stroke-width="2"/>'
for j in range(30):
    for i in range(30):
        v = int(fb[j, i])
        if v:
            s += f'<rect x="{ox + i * px}" y="{oy + j * px}" width="{px}" height="{px}" fill="{PAL[v]}"/>'
save("banner", s, W, H)

# ============================================================ arch: host <-> Qwen3.5 stack
W, H = 1200, 520
s = t(40, 40, "Architecture: one kernel = one prompt", 22, "b") + t(40, 62, "The weights of a tiny integer network travel in the prompt. The model emits every output token itself.", 13, "m")
s += box(30, 100, 210, 150, "amber", "afill", "HOST", ["quantise rotation  (3×3 int)", "bin triangles  (bbox)", "gather / scatter lanes", "clear buffers · draw order", "argmax + append"])
s += box(30, 290, 210, 120, "violet", "vfill", "kernel tables", ["stages.py", "5-bit weights · ≤15 lanes", "prompt = 1,056 tokens", "1,024 CELL + 32 ACT"])
s += path("M 135 290 L 135 254", True, "violet")
s += path("M 244 175 L 330 175", True, "amber")
s += t(287, 160, "prompt", 12, "b", "middle", "var(--amber)")
# Qwen stack
s += f'<rect x="334" y="86" width="560" height="372" rx="16" fill="var(--panel)" stroke="var(--line)" stroke-width="2"/>'
s += t(614, 112, "Qwen3.5-0.8B, with 20 chip layers appended", 15, "b", "middle")
s += box(360, 130, 508, 64, "mute", "soft", "24 original Qwen layers", ["muted on chip tokens: they are the identity, so the fast path skips them (bit-identical)"], 14).replace('stroke-width="2"', 'stroke-width="2" stroke-dasharray="6 5"')
for i in range(5):
    s += f'<rect x="{360 + i * 104}" y="222" width="92" height="46" rx="9" fill="var(--tfill)" stroke="var(--teal)" stroke-width="2"/>'
    s += t(360 + i * 104 + 46, 250, "chip layer" if i < 4 else "…", 12, "b", "middle", "var(--teal)")
s += t(614, 296, "DeltaNet state = the weight array.  Writes: CELL tokens.  Reads: READ tokens.", 12.5, "m", "middle")
s += box(360, 316, 160, 112, "teal", "tfill", "weights", ["32×32 packed", "layer 0 block", "layer 1 block"])
s += box(540, 316, 160, 112, "teal", "tfill", "multi-hot read", ["= exact sum", "of selected rows", "(adder tree)"])
s += box(720, 316, 148, 112, "teal", "tfill", "relu + clamp", ["saturating", "Horner fold", "0..31"])
s += path("M 520 372 L 536 372", False) + path("M 700 372 L 716 372", False)
s += path("M 872 372 L 926 372", True, "teal")
s += t(899, 360, "emits", 12, "b", "middle", "var(--teal)")
s += box(930, 316, 240, 112, "teal", "tfill", "READ k · ACT row;y", ["…then OUT tokens, then HALT", "65 tokens per depth-2 run", "(argmax, fed back)"], 13)
s += path("M 1050 430 L 1050 494 L 12 494 L 12 175 L 28 175", True, "teal")
s += t(600, 486, "decoded from the MODEL's tokens, gathered by the host into the next stage's input lanes", 12.5, "m", "middle")
save("arch", s, W, H)

# ============================================================ kernel anatomy
W, H = 1200, 470
s = t(40, 40, "Anatomy of a kernel (depth 2, 15 lanes, 5-bit)", 22, "b") + t(40, 62, "Every stage in the pipeline has exactly this shape. Hardware limits are enforced when the kernel is built.", 13, "m")
def lanes(x, y, n, label, col, vals=None):
    out = t(x + 36, y - 14, label, 12.5, "b", "middle", f"var(--{col})")
    for i in range(n):
        out += f'<rect x="{x}" y="{y + i * 17}" width="72" height="14" rx="4" fill="var(--{col[0]}fill)" stroke="var(--{col})" stroke-width="1.2"/>'
        if vals: out += t(x + 36, y + i * 17 + 11, vals[i], 10, "mono", "middle")
    return out
s += lanes(60, 110, 12, "input lanes", "amber", [str(v) for v in D["kernel_in"]])
s += f'<rect x="60" y="{110 + 12 * 17}" width="72" height="14" rx="4" fill="var(--soft)" stroke="var(--line)"/>' + t(96, 110 + 12 * 17 + 11, "bias = 1", 10, "mono", "middle")
s += box(210, 150, 190, 100, "violet", "vfill", "layer 1", ["x @ W₁ + b₁", "relu, clamp 0..31", "W ∈ [−16, 15]"])
s += lanes(450, 110, 13, "13 lanes", "teal")
s += box(570, 150, 190, 100, "violet", "vfill", "layer 2", ["x @ W₂ + b₂", "relu, clamp 0..31", "no shift, ever"])
s += lanes(810, 110, 4, "output", "rose", [str(v) for v in D["kernel_out"]])
s += arrow(136, 190, 206, 190) + arrow(404, 190, 446, 190) + arrow(522, 190, 566, 190) + arrow(764, 190, 806, 190)
s += t(60, 360, "Why 15 lanes?  The array is 16 wide and one lane is the model's constant 1 (the bias row). Lane 15 of every layer carries it.", 13, "m")
s += t(60, 384, "Why depth 2?  The validated 16-wide 5-bit chip is a 32×32 packed array: two 16-wide blocks. A deeper stage is just two programs back to back.", 13, "m")
s += t(60, 408, "Why 5-bit?  Bit-plane counts stay exact in bf16 only up to 5 bits; that is what makes the whole thing integer-exact on real hardware.", 13, "m")
s += t(60, 440, "values shown: the real input and output of the coverage kernel for pixel (%d, %d) of the demo frame" % (D["pixel"][1], D["pixel"][0]), 12, "m")
save("kernel-anatomy", s, W, H)

# ============================================================ token timeline (real decoded tokens)
W, H = 1200, 560
s = t(40, 40, "What the model actually sees: one coverage kernel, token by token", 22, "b") + t(40, 62, "Real tokens decoded from a real run. The host feeds the prompt. After that the model picks every token.", 13, "m")
# prompt bar
s += t(40, 100, "PROMPT  (host-fed, %d tokens)" % D["prompt_len"], 14, "b", fill="var(--amber)")
s += f'<rect x="40" y="112" width="900" height="40" rx="8" fill="var(--afill)" stroke="var(--amber)" stroke-width="2"/>'
s += t(490, 138, "%d × CELL(row, col, weight)   =   the whole 32×32 weight array" % D["n_cells"], 14, "b", "middle")
s += f'<rect x="946" y="112" width="104" height="40" rx="8" fill="var(--soft)" stroke="var(--line)" stroke-width="2"/>' + t(998, 138, "16 × ACT(0)", 12, "m", "middle")
s += f'<rect x="1056" y="112" width="104" height="40" rx="8" fill="var(--rfill)" stroke="var(--rose)" stroke-width="2"/>' + t(1108, 138, "16 × ACT(x)", 12, "b", "middle", "var(--rose)")
s += t(1108, 172, "the input", 11, "m", "middle") + t(998, 172, "program counter", 11, "m", "middle")
for k, tok in enumerate(D["prompt_head"][:3]):
    s += t(48 + k * 250, 190, tok[1], 11, "mono m")
s += t(40, 212, "…", 13, "m")
# emitted
s += t(40, 250, "EMITTED  (the model's own argmax, %d tokens)" % len(D["emitted"]), 14, "b", fill="var(--teal)")
em = D["emitted"]
colmap = {"read": "violet", "act": "teal", "out": "rose", "halt": "amber"}
x0, y0 = 40, 266
for n, (kind, txt) in enumerate(em[:44]):
    r, c = divmod(n, 11)
    x, y = x0 + c * 104, y0 + r * 40
    col = colmap[kind]
    s += f'<rect x="{x}" y="{y}" width="98" height="32" rx="7" fill="var(--{col[0]}fill)" stroke="var(--{col})" stroke-width="1.6"/>'.replace("vfill", "vfill")
    s += t(x + 49, y + 21, txt.replace("ACT row ", "ACT ").replace("READ col ", "READ "), 10.5, "mono", "middle")
s += t(40, 444, "layer 1:  READ k, then the model emits ACT(row; y): the next layer's input row, written by the model itself.", 13, "m")
s += t(40, 468, "layer 2:  READ k, then OUT j = y: the result tokens.   Last token: HALT.   Total emitted: %d." % len(em), 13, "m")
s += t(40, 504, "legend:", 12, "b") + f'<rect x="100" y="493" width="14" height="14" rx="3" fill="var(--vfill)" stroke="var(--violet)"/>' + t(120, 505, "READ (control)", 12, "m")
s += f'<rect x="240" y="493" width="14" height="14" rx="3" fill="var(--tfill)" stroke="var(--teal)"/>' + t(260, 505, "ACT (layer 1 → 2)", 12, "m")
s += f'<rect x="400" y="493" width="14" height="14" rx="3" fill="var(--rfill)" stroke="var(--rose)"/>' + t(420, 505, "OUT (result)", 12, "m")
s += f'<rect x="530" y="493" width="14" height="14" rx="3" fill="var(--afill)" stroke="var(--amber)"/>' + t(550, 505, "HALT", 12, "m")
save("token-timeline", s, W, H)

# ============================================================ pixel journey
J = D["journey"]
W, H = 1200, 120 + 120 * len(J) + 110
s = t(40, 40, "Life of a pixel: (%d, %d) through four triangles" % (D["pixel"][1], D["pixel"][0]), 22, "b")
s += t(40, 62, "Real kernel values from the demo frame. Four triangles cover this pixel. The depth test decides who wins.", 13, "m")
s += t(40, 98, "edge steps (e ≥ 0 / e ≤ 0)", 12, "b", fill="var(--mute)") + t(430, 98, "coverage", 12, "b", fill="var(--mute)") + t(560, 98, "depth test:  Z − zn ≥ 1 ?", 12, "b", fill="var(--mute)") + t(930, 98, "Z buffer / colour", 12, "b", fill="var(--mute)")
for n, q in enumerate(J):
    y = 112 + n * 120
    passed = q["z_after"] != q["z_before"]
    col = "teal" if passed else "rose"
    s += f'<rect x="30" y="{y}" width="1140" height="104" rx="14" fill="var(--panel)" stroke="var(--line)" stroke-width="1.5"/>'
    s += t(50, y + 30, f"triangle {q['tri']}", 15, "b") + t(50, y + 52, f"face {q['face']}", 12, "m") + f'<rect x="50" y="{y + 62}" width="60" height="24" rx="6" fill="{PAL[q["cn"]]}"/>' + t(120, y + 79, f"colour {q['cn']}", 12, "mono m")
    for e in range(3):
        for sgn, vals, lab in ((0, q["edges_pos"], "≥0"), (1, q["edges_neg"], "≤0")):
            v = vals[e]
            cx = 190 + e * 80
            cy = y + 34 + sgn * 36
            s += f'<rect x="{cx}" y="{cy - 14}" width="68" height="28" rx="7" fill="var(--{"tfill" if v else "soft"})" stroke="var(--{"teal" if v else "line"})" stroke-width="1.6"/>' + t(cx + 34, cy + 5, f"{lab}  {v}", 12, "mono", "middle")
    s += f'<rect x="430" y="{y + 26}" width="96" height="52" rx="10" fill="var(--tfill)" stroke="var(--teal)" stroke-width="2"/>' + t(478, y + 59, f"cov {q['cov']}", 14, "b", "middle", "var(--teal)")
    s += arrow(432 + 96, y + 52, 556, y + 52, True, col)
    s += t(568, y + 40, f"zn = {q['zn']}   Z = {q['z_before']}", 14, "mono")
    s += t(568, y + 64, ("zn < Z  →  WRITE" if passed else "zn ≥ Z  →  rejected, buffers unchanged"), 14, "b", "start", f"var(--{col})")
    s += t(568, y + 88, f"g = {q['z_before'] - q['z_after']}   (Z′ = Z − g)", 12, "mono m")
    s += t(930, y + 40, f"Z : {q['z_before']} → {q['z_after']}", 15, "mono b")
    s += f'<rect x="930" y="{y + 56}" width="40" height="26" rx="6" fill="{PAL[q["c_before"]]}"/>' + t(980, y + 74, "→", 16, "b") + f'<rect x="1000" y="{y + 56}" width="40" height="26" rx="6" fill="{PAL[q["c_after"]]}"/>'
    s += t(1050, y + 74, f"{q['c_before']} → {q['c_after']}", 12, "mono m")
yb = 112 + len(J) * 120
s += t(40, yb + 30, "The last two triangles are the two halves of the same far face. They are covered, but the near face already won, so nothing is written.", 13, "m")
s += t(40, yb + 54, "No branch, no comparison instruction: g = relu(Z − zn + s31 + cov − 62), then Z′ = relu(Z − g). It is all gated ReLU.", 13, "m")
save("pixel-journey", s, W, H)

# ============================================================ depth-test circuit
W, H = 1200, 420
s = t(40, 40, "The depth test and framebuffer write, as four ReLU layers", 22, "b") + t(40, 62, "Two programs: rop1 (layers 1–2) and rop2 (layers 3–4). Two pixels per run, 6 lanes each.", 13, "m")
cols = [("in", "amber", ["s⁺  s⁻", "Z  zn", "C  cn"]), ("L1", "violet", ["cov = relu(s⁺+s⁻)", "u = relu(15·(Z−zn))", "Z zn C cn pass"]), ("L2", "violet", ["s31 = relu(15·u)", "cov Z zn C cn pass", "const 15 lane"]),
        ("L3", "violet", ["g  = relu(Z−zn+s31+cov−62)", "g₁ = relu(cn−C+…−62)", "g₂ = relu(C−cn+…−62)"]), ("L4", "teal", ["Z′ = relu(Z − g)", "C′ = relu(C + g₁ − g₂)"])]
for i, (nm, col, ls) in enumerate(cols):
    x = 40 + i * 232
    s += box(x, 110, 212, 130, col, col[0] + "fill", nm, ls, 13)
    if i < 4: s += arrow(x + 214, 175, x + 230, 175)
s += t(40, 280, "s31 = 31 only when Z − zn ≥ 1 : two saturating ×15 steps turn a small integer into a hard 0 / 31.", 13, "m")
s += t(40, 304, "g and g₁−g₂ are gated by (s31 + cov − 62): both must be 31 for anything to pass, and 62 is built from a constant lane (15 × −4 − 2) because a bias can only reach −16.", 13, "m")
s += t(40, 328, "So the write is branch-free and multiplier-free: a conditional move made of saturating adds.", 13, "m")
s += t(40, 372, "Strict test: a fragment with equal depth does not overwrite. Z clears to 31, colour clears to 0 (background).", 12, "m")
save("depth-test", s, W, H)

# ============================================================ edge function math
W, H = 1200, 440
s = t(40, 40, "Edge functions without a multiplier: the rotation does the work", 22, "b") + t(40, 62, "Triangle setup folds into edge evaluation. Pixel position is the weight, the vertex data is the activation.", 13, "m")
s += box(40, 100, 520, 120, "violet", "vfill", "edge e(q) of directed cube edge a → b", ["e(q) = A·qx + B·qy + C", "A = Ya − Yb      B = Xb − Xa", "C = Xa·Yb − Xb·Ya"], 14)
s += box(640, 100, 520, 120, "teal", "tfill", "Binet–Cauchy turns the product into a dot", ["C = m · (s_a × s_b)", "m = Rx × Ry   (3 ints, a host uniform)", "s_a × s_b ∈ {0, ±2}  →  a 5-bit weight"], 14)
s += arrow(562, 160, 636, 160)
s += t(40, 262, "So for 6 pixels (qx, qy are the WEIGHTS), one run computes:", 14, "b")
s += t(60, 290, "e = qx·(Ya − Yb) + qy·(Xb − Xa) + 2·(±m₀ ± m₁ ± m₂)      lanes: Xa Xb Ya Yb m₀± m₁± m₂±   (10)", 13, "mono")
s += t(60, 318, "out lane 2p   = 15·relu(e + 1)   ≈  [e ≥ 0]        out lane 2p+1 = 15·relu(1 − e)   ≈  [e ≤ 0]", 13, "mono")
s += t(40, 362, "Both windings are tested (e ≥ 0 for CCW, e ≤ 0 for CW), so faces need no culling and the depth buffer has real work to do.", 13, "m")
s += t(40, 386, "Exactness: A, B come from the vertex kernel's own outputs, C from the identity, so every edge passes through its two vertices to the pixel.", 13, "m")
s += t(40, 410, "Checked against an independent integer rasteriser on 30 random rotations: 0 differing pixels.", 13, "m")
save("edge-function", s, W, H)

# ============================================================ host vs model
W, H = 1200, 470
s = t(40, 40, "What runs where (the honest split)", 22, "b") + t(40, 62, "No pixel, vertex or edge value is ever computed on the host.", 13, "m")
s += f'<rect x="30" y="90" width="560" height="340" rx="16" fill="var(--tfill)" stroke="var(--teal)" stroke-width="2"/>' + t(310, 122, "inside Qwen3.5  (integer ReLU layers)", 17, "b", "middle", "var(--teal)")
for i, l in enumerate(["vertex transform: 8 corners × (x, y, depth)", "guard-band clip flags", "edge functions, 6 pixels per run", "hard binarise + 3-edge AND, both windings", "depth test (Z − zn ≥ 1) and conditional write", "flat shading: 5 brightness levels × 6 hues", "every pixel, every depth value, every colour"]):
    s += t(60, 158 + i * 34, "✅  " + l, 14.5)
s += f'<rect x="610" y="90" width="560" height="340" rx="16" fill="var(--afill)" stroke="var(--amber)" stroke-width="2"/>' + t(890, 122, "on the host  (no pixel arithmetic)", 17, "b", "middle", "var(--amber)")
for i, l in enumerate(["quantise the rotation to a 3×3 int uniform", "its cross product m (like a GL normal matrix)", "bounding-box tile binning from vertex outputs", "gather / scatter of lanes between kernels", "clear values: Z = 31, colour = 0 · draw order", "skip raster ops the model reported as empty", "argmax + append (free-running mode)"]):
    s += t(640, 158 + i * 34, "🧰  " + l, 14.5)
s += t(40, 456, "Also not done: polygon clipping (guard-band check only) · interpolated depth (flat per face) · buffers live in host-held tokens between kernels.", 13, "m")
save("host-vs-model", s, W, H)

# ============================================================ kernel zoo
W, H = 1200, 470
kz = [("vertex", 6, 8, 1, 3), ("clip", 6, 12, 2, 12), ("clip_any", 12, 1, 1, 12), ("shade", 6, 2, 2, 6), ("eval", 10, 12, 2, 708), ("cover", 12, 4, 2, 708), ("rop1", 12, 13, 2, 278), ("rop2", 13, 4, 2, 278)]
desc = {"vertex": "corner signs × rotation row", "clip": "saturation flags", "clip_any": "any flag set?", "shade": "light · normal → colour, depth", "eval": "edge function, 6 px", "cover": "×15 ×15, AND of 3 edges", "rop1": "cov, depth step s31", "rop2": "gated Z′ and C′ write"}
s = t(40, 40, "Kernel zoo: every stage, with its real shape", 22, "b") + t(40, 62, "Runs per frame are for the demo frame at angles (0.4, 0.7, 0.1).", 13, "m")
s += t(50, 100, "kernel", 12, "b") + t(210, 100, "what it does", 12, "b") + t(560, 100, "lanes in → out", 12, "b") + t(740, 100, "depth", 12, "b") + t(830, 100, "runs / frame", 12, "b") + t(1000, 100, "share of runs", 12, "b")
tot = sum(k[4] for k in kz)
for i, (n, a, b, dpt, r) in enumerate(kz):
    y = 116 + i * 40
    s += f'<rect x="30" y="{y}" width="1140" height="34" rx="9" fill="var(--panel)" stroke="var(--line)"/>'
    s += t(50, y + 22, n, 14, "mono b") + t(210, y + 22, desc[n], 13, "m") + t(560, y + 22, f"{a} → {b}", 13, "mono") + t(750, y + 22, str(dpt), 13, "mono") + t(830, y + 22, f"{r:,}", 13, "mono")
    s += f'<rect x="1000" y="{y + 10}" width="{max(3, 150 * r / tot)}" height="14" rx="4" fill="var(--violet)"/>'
s += t(40, 458, "eval + cover + rop run once per (triangle, row, 6-pixel group). rop skips groups the model reports as uncovered: 430 of 708 on this frame.", 12.5, "m")
save("kernel-zoo", s, W, H)

# ============================================================ results cards
W, H = 1200, 300
s = t(40, 40, "Measured: A100, Qwen3.5-0.8B, 36 frames", 22, "b")
cards = [("104,940", "kernel runs", "27,880 unique"), ("31.2 M", "tokens through the model", "1.81 M predicted positions"), ("0", "mismatches", "model argmax vs integer oracle"), ("36 / 36", "frames pixel-identical", "to the numpy pipeline"), ("123 s", "one frame, free-running", "56,747 tokens, 0 mismatches")]
for i, (big, lab, sub) in enumerate(cards):
    x = 30 + i * 230
    col = ["teal", "violet", "amber", "teal", "rose"][i]
    s += box(x, 80, 214, 150, col, col[0] + "fill", big, [lab, sub], 22)
s += t(40, 268, "teacher-forced on the 36 frames, true autoregressive generation on one full frame. Raw logs are in results/.", 13, "m")
save("results", s, W, H)
