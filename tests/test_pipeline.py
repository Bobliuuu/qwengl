import numpy as np
import pytest

from qwengl import NumpyBackend, quantise, render, rot_matrix
from qwengl import stages as st
from qwengl.geometry import SGN, minors
from qwengl.reference import reference


def test_binet_cauchy():
    rng = np.random.default_rng(0)
    for _ in range(50):
        R = rng.integers(-5, 6, (3, 3)); m = np.cross(R[0], R[1])
        for a in range(8):
            for b in range(8):
                assert (SGN[a] @ R[0]) * (SGN[b] @ R[1]) - (SGN[b] @ R[0]) * (SGN[a] @ R[1]) == m @ np.cross(SGN[a], SGN[b])


def test_all_kernels_fit_the_hardware():
    for f in range(6): st.shade_kernel(f)
    for i in range(3): st.vertex_kernel(i)
    for k in (st.clip_kernel(), st.clip_any_kernel(), st.cover_kernel(), st.rop1_kernel(), st.rop2_kernel()): pass
    st.eval_kernel(0, 7, 3, 2)


@pytest.mark.parametrize("ang", [(0, 0, 0), (0.4, 0.7, 0.1), (1.2, 2.5, 0.3), (3.0, 0.2, 2.2), (0.77, 5.1, 4.0)])
def test_matches_reference(ang):
    R = quantise(rot_matrix(*ang))
    fr = render(NumpyBackend(), R)
    fb, zb = reference(R)
    assert not fr.dropped
    assert (fr.fb == fb).all(), np.argwhere(fr.fb != fb)[:5]
    assert (fr.zbuf == zb).all()
    assert (fb > 0).sum() > 30


def test_random_sweep():
    rng = np.random.default_rng(1)
    bad = 0
    for _ in range(25):
        R = quantise(rot_matrix(*rng.uniform(0, 6.28, 3)))
        fr = render(NumpyBackend(), R); fb, zb = reference(R)
        bad += int((fr.fb != fb).sum() + (fr.zbuf != zb).sum())
    assert bad == 0
