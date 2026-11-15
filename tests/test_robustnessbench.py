"""Tests for RobustnessBench: compression error under noisy inputs."""

import numpy as np
import pytest

from robustnessbench import add_noise, robustness_table


def _matrix():
    t = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    return np.outer(np.sin(t), np.cos(t))


def test_add_noise_zero_is_identity():
    W = _matrix()
    assert np.array_equal(add_noise(W, 0.0, seed=1), W)


def test_add_noise_scales_with_std_and_validates():
    W = _matrix()
    noisy = add_noise(W, 0.1, seed=1)
    assert noisy.shape == W.shape
    assert not np.array_equal(noisy, W)
    with pytest.raises(ValueError):
        add_noise(W, -0.1)
    with pytest.raises(ValueError):
        add_noise(np.zeros(4), 0.1)


def test_robustness_table_covers_methods_and_levels():
    rows = robustness_table(_matrix(), noise_levels=(0.0, 0.1), seed=1)
    methods = {r["method"] for r in rows}
    noises = {r["noise"] for r in rows}
    assert methods == {"svd_rank4", "quantize8", "fourier4", "hybrid"}
    assert noises == {0.0, 0.1}
    assert all(r["relative_error"] >= 0.0 for r in rows)


def test_error_grows_or_stays_with_noise_for_every_method():
    W = _matrix()
    rows = robustness_table(W, noise_levels=(0.0, 0.2), seed=1)
    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r["relative_error"])
    for method, errs in by_method.items():
        assert errs[1] >= errs[0] - 1e-9, method


def test_zero_noise_svd_error_is_small_for_smooth_matrix():
    W = _matrix()
    rows = robustness_table(W, noise_levels=(0.0,), seed=1)
    svd_row = next(r for r in rows if r["method"] == "svd_rank4")
    assert svd_row["relative_error"] < 1e-6


def test_validates_bad_matrix():
    with pytest.raises(ValueError):
        robustness_table(np.zeros(4))
