import numpy as np
import pytest

from qwengl import NumpyBackend, rot_matrix
from qwengl.geometry import quantise
from qwengl.gl import GL
from qwengl.mesh import cube, octahedron, pyramid, tetrahedron
from qwengl.reference import reference_draw


def scene(draws, light=(-1, -1, -2)):
    gl = GL(NumpyBackend()); gl.light(light); gl.clear()
    Z, C = np.full((30, 30), 31, dtype=np.int64), np.zeros((30, 30), dtype=np.int64)
    for mesh, rot, tr in draws:
        gl.draw_elements(mesh, rot, tr)
        reference_draw(Z, C, mesh, quantise(rot, mesh.scale), tr, light)
    return gl, Z, C


@pytest.mark.parametrize("mesh", [cube(), octahedron(), tetrahedron(), pyramid()], ids=lambda m: m.name)
@pytest.mark.parametrize("ang", [(0.3, 0.8, 0.1), (1.9, 2.4, 0.7), (4.0, 0.2, 5.0)])
def test_single_mesh_matches_reference(mesh, ang):
    gl, Z, C = scene([(mesh, rot_matrix(*ang), (0, 0, 0))])
    assert not gl.dropped
    assert (gl.read_pixels() == C).all() and (gl.read_depth() == Z).all()
    assert (C > 0).sum() > 20


def test_translation_and_two_draw_calls_share_the_depth_buffer():
    draws = [(cube(), rot_matrix(0.5, 0.9, 0.2), (-5, 0, 0)), (octahedron(), rot_matrix(1.1, 0.3, 0.8), (6, 1, 0))]
    gl, Z, C = scene(draws)
    assert (gl.read_pixels() == C).all() and (gl.read_depth() == Z).all()
    assert len(set(C[C > 0] // 5)) >= 3


def test_occlusion_between_meshes_uses_depth():
    near = (cube(), rot_matrix(0.3, 0.5, 0.0), (0, 0, -6))
    far = (octahedron(), rot_matrix(0.9, 0.2, 0.1), (0, 0, 8))
    a = scene([near, far]); b = scene([far, near])
    assert (a[0].read_pixels() == b[0].read_pixels()).all() and (a[0].read_pixels() == a[2]).all()   # order independent


def test_triangle_off_the_guard_band_is_dropped_not_misdrawn():
    gl, Z, C = scene([(cube(), rot_matrix(0.4, 0.4, 0.4), (14, 0, 0))])
    assert gl.dropped and (gl.read_pixels() == C).all()


def test_random_scenes():
    rng = np.random.default_rng(3)
    meshes = [cube(), octahedron(), tetrahedron(), pyramid()]
    for _ in range(12):
        draws = [(meshes[rng.integers(4)], rot_matrix(*rng.uniform(0, 6.28, 3)), tuple(int(v) for v in rng.integers(-6, 7, 3) * [1, 1, 1])) for _ in range(2)]
        gl, Z, C = scene(draws)
        assert (gl.read_pixels() == C).all() and (gl.read_depth() == Z).all()
