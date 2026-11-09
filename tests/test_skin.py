"""Property tests for the skin metrics on synthetic skin (no MediaPipe needed).

Each test changes one thing about a synthetic skin patch and checks that only
the metric that should respond does. The key fairness property: a uniformly
warmer or darker skin tone must not register as "redness".
"""

import numpy as np
import pytest

from app import skin

SIZE = 128


def _patch(L=60.0, a=12.0, b=15.0, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    lab = np.zeros((SIZE, SIZE, 3), np.float32)
    lab[..., 0] = L + rng.normal(0, noise, (SIZE, SIZE))
    lab[..., 1] = a + rng.normal(0, noise / 3, (SIZE, SIZE))
    lab[..., 2] = b
    return lab


def _mask():
    yy, xx = np.mgrid[:SIZE, :SIZE]
    return (((yy - SIZE / 2) ** 2 + (xx - SIZE / 2) ** 2) < (0.45 * SIZE) ** 2).astype(np.uint8)


def _spots(lab, channel, delta, n=12, r=2, seed=1):
    rng = np.random.default_rng(seed)
    out = lab.copy()
    yy, xx = np.mgrid[:SIZE, :SIZE]
    for cy, cx in rng.integers(SIZE // 4, 3 * SIZE // 4, size=(n, 2)):
        out[..., channel][(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] += delta
    return out


BASE = skin.measure(_patch(), _mask())


def test_rough_texture_lowers_smoothness():
    rough = skin.measure(_patch(noise=3.0), _mask())
    assert rough["smoothness"] > BASE["smoothness"] * 3


def test_lighting_gradient_is_ignored():
    lab = _patch()
    lab[..., 0] += np.linspace(-15, 15, SIZE)[None, :]  # strong side lighting
    lit = skin.measure(lab, _mask())
    assert lit["smoothness"] == pytest.approx(BASE["smoothness"], abs=0.3)
    assert lit["evenness"] == pytest.approx(BASE["evenness"], abs=0.3)


def test_exposure_change_is_ignored():
    lab = _patch(noise=2.0)
    dark = lab.copy()
    dark[..., 0] *= 0.7
    a, b = skin.measure(lab, _mask()), skin.measure(dark, _mask())
    assert b["smoothness"] == pytest.approx(a["smoothness"], rel=0.05)


def test_dark_spots_lower_clarity():
    spotted = skin.measure(_spots(_patch(), 0, -12.0), _mask())
    assert spotted["clarity"] > BASE["clarity"] + 0.5


def test_red_patches_raise_redness():
    red = skin.measure(_spots(_patch(), 1, 10.0, n=20, r=4), _mask())
    assert red["redness"] > BASE["redness"] + 0.3


@pytest.mark.parametrize("L,a,b", [(35.0, 18.0, 25.0), (75.0, 6.0, 12.0)])
def test_skin_tone_does_not_register_as_redness(L, a, b):
    # Darker/warmer and lighter/cooler tones: uniform shifts, no patches.
    other = skin.measure(_patch(L=L, a=a, b=b), _mask())
    assert other["redness"] == pytest.approx(BASE["redness"], abs=0.1)


def test_score_maps_anchors_and_clamps():
    good = {k: g for k, (g, _) in skin.ANCHORS.items()}
    bad = {k: b for k, (_, b) in skin.ANCHORS.items()}
    worse = {k: b * 2 for k, (_, b) in skin.ANCHORS.items()}
    assert all(v == 100.0 for v in skin.score(good).values())
    assert all(v == 0.0 for v in skin.score(bad).values())
    assert all(v == 0.0 for v in skin.score(worse).values())
