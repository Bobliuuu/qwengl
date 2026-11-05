"""Walk-through scene: an isometric dungeon room with a figure that walks a loop around a spinning gem.

Everything is ordinary glDrawElements calls (orthographic camera, exact integer matrices):
  draw 1  the static room (floor tiles, two walls, pedestal, torch brackets)  -> identical every frame, cached after frame 0
  draw 2  the walker (head, torso, arms, legs as one mesh; 4 walk poses x 4 headings)
  draw 3  the gem (octahedron) spinning and bobbing over the pedestal
  draw 4  two torch flames (flicker)
A 60x60 tiled viewport (GL(backend, 60, 60)).  The room uses coarse units (4 px), the walker and gem fine units (2 px).
There is no perspective in the pipeline, so this is a dimetric 'Diablo' view, not first person.
"""
from __future__ import annotations

import numpy as np

from .mesh import Mesh, octahedron, pyramid

# cameras (world: x, y up, z).  coarse: 1 unit = 4 px; fine: 1 unit = 2 px, i.e. half a coarse unit.  depth is -2(x+y+z) coarse.
W = H = 60
CAM = np.array([[4, 0, -4], [2, -4, 2], [-2, -2, -2]], dtype=np.int64)
CAMF = np.array([[2, 0, -2], [1, -2, 1], [-1, -1, -1]], dtype=np.int64)
CAM_C = CAM.astype(float)
SHIFT = (0, 8, 2)                                # screen / depth offset of the world origin
LIGHT_C, LIGHT_F = (1, -2, -2), (2, -3, -3)      # toward the light in view space (top = brightest, +x = mid, +z = dark)

FLOOR_A, FLOOR_B, BRICK, STONE, TUNIC, GOLD = range(6)   # colour families, see palette.dungeon_palette


def _faces(x0, x1, y0, y1, z0, z1, mats):
    """The three camera-facing faces (+x, +y, +z) of a box: (corner vertices, material, normal, depth anchor)."""
    c = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    quads = {
        "x": ([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], (1, 0, 0), (x1, c[1], c[2])),
        "y": ([(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)], (0, 1, 0), (c[0], y1, c[2])),
        "z": ([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], (0, 0, 1), (c[0], c[1], z1)),
    }
    out = []
    for k, m in zip("xyz", mats):
        if m is not None:
            q, n, a = quads[k]
            out.append((q, m, n, tuple(int(np.floor(v + 0.5)) for v in a)))
    return out


def build(boxes) -> Mesh:
    V, T, M, N, A = [], [], [], [], []
    for b in boxes:
        for q, m, n, a in _faces(*b[:6], b[6]):
            i = len(V)
            V += q
            T += [(i, i + 1, i + 2), (i, i + 2, i + 3)]
            M += [m, m]; N += [n, n]; A += [a, a]
    return Mesh(np.array(V), T, M, np.array(N), np.array(A), name="boxes")


def room() -> Mesh:
    """coarse units.  floor 6x6 checker, two 2-brick walls, a pedestal, two torch brackets."""
    boxes = []
    for x in range(-3, 3):
        for z in range(-3, 3):
            boxes.append((x, x + 1, -1, 0, z, z + 1, (None, FLOOR_A if (x + z) % 2 else FLOOR_B, None)))
    for z in range(-3, 3):
        for y in range(2):
            boxes.append((-4, -3, y, y + 1, z, z + 1, (BRICK, STONE if y == 1 else None, BRICK if z == 2 else None)))
    for x in range(-3, 3):
        for y in range(2):
            boxes.append((x, x + 1, y, y + 1, -4, -3, (BRICK if x == 2 else None, STONE if y == 1 else None, BRICK)))
    boxes.append((-4, -3, 0, 2, -4, -3, (BRICK, STONE, BRICK)))
    boxes.append((0, 1, 0, 1, 0, 1, (STONE, STONE, STONE)))      # pedestal
    return build(boxes)


def _rot(k, x, z):
    for _ in range(k % 4):
        x, z = z, -x
    return x, z


def walker(heading: int, pose: int) -> Mesh:
    """fine units, local frame: forward +x, lateral z.  pose 0/2 legs together, 1 left leg forward, 3 right leg forward."""
    sa, sb = {0: (0, 0), 1: (1, -1), 2: (0, 0), 3: (-1, 1)}[pose]
    parts = [  # (x range, y range, z range, family, x shift)
        ((0, 2), (0, 2), (0, 1), STONE, sa), ((0, 2), (0, 2), (1, 2), STONE, sb),     # legs
        ((0, 2), (2, 5), (0, 2), TUNIC, 0),                                           # torso
        ((0, 1), (2, 5), (-1, 0), TUNIC, sb), ((0, 1), (2, 5), (2, 3), TUNIC, sa),    # arms swing against the legs
        ((0, 2), (5, 7), (0, 2), FLOOR_A, 0),                                         # head
        ((2, 3), (5, 6), (0, 2), STONE, 0),                                           # visor: shows which way it faces
    ]
    boxes = []
    for (xa, xb), (ya, yb), (za, zb), mat, sw in parts:
        (x0, z0), (x1, z1) = _rot(heading, xa + sw, za), _rot(heading, xb + sw, zb)
        boxes.append((min(x0, x1), max(x0, x1), ya, yb, min(z0, z1), max(z0, z1), (mat, mat, mat)))
    return build(boxes)


PIVOT = (0.5, 0.5)                              # the walker mesh's local (x, z) centre, in coarse units


def _t(p):
    s = CAM_C @ np.asarray(p, float)
    return int(round(s[0])) + SHIFT[0], int(round(s[1])) + SHIFT[1], int(round(s[2])) + SHIFT[2]


ROOM = room()
GEM = octahedron(2)
GEM.material[:] = GOLD
WALK = {(h, p): walker(h, p) for h in range(4) for p in range(4)}
_FL = [[0, 0, 0], [1, 0, 0], [0, 0, 1]]
FLAME = [Mesh(np.array(_FL + [[0, hh, 0]]), [(0, 1, 3), (0, 2, 3), (1, 2, 3)], GOLD, name="flame") for hh in (3, 2)]
BRACKET = build([(0, 1, 0, 1, 0, 1, (STONE, STONE, STONE))])


def walker_state(f: int, total: int):
    """a square loop of side 4 coarse units around the pedestal."""
    u = (f / total) * 16.0
    side, d = int(u // 4) % 4, u % 4
    c = [(-1.5, -1.5), (2.5, -1.5), (2.5, 2.5), (-1.5, 2.5), (-1.5, -1.5)]
    (ax, az), (bx, bz) = c[side], c[side + 1]
    return (ax + (bx - ax) * d / 4, az + (bz - az) * d / 4), {0: 0, 1: 3, 2: 2, 3: 1}[side], (f // 2) % 4


def draw_scene(gl, f: int, total: int):
    gl.clear()
    gl.light(LIGHT_C)
    gl.draw_elements(ROOM, None, _t((0, 0, 0)), R=CAM)
    gl.light(LIGHT_F)
    (px, pz), h, pose = walker_state(f, total)
    ox, oz = _rot(h, *PIVOT)
    gl.draw_elements(WALK[(h, pose)], None, _t((px - ox, 0, pz - oz)), R=CAMF)
    th = 2 * np.pi * f / total
    Ry = np.array([[np.cos(4 * th), 0, np.sin(4 * th)], [0, 1, 0], [-np.sin(4 * th), 0, np.cos(4 * th)]])
    gl.draw_elements(GEM, None, _t((0.5, 2.4 + 0.3 * np.sin(2 * th), 0.5)), R=np.rint(CAMF @ Ry).astype(np.int64))
    for i, (tx, ty, tz) in enumerate(((-2.75, 1.0, -1.0), (1.0, 1.0, -2.75))):
        gl.draw_elements(BRACKET, None, _t((tx, ty - 0.5, tz)), R=CAMF)
        gl.draw_elements(FLAME[(f // 2 + i) % 2], None, _t((tx, ty, tz)), R=CAMF)
    return gl.read_pixels()
