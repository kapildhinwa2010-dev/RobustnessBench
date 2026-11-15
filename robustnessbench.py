# RobustnessBench - Compression error under noisy inputs, per method
# Part of the Kapildhinwa2010-dev Toolkit.

"""RobustnessBench - noise robustness benchmark for compression.

Lineage: robustness-layer repos. Perturbs a clean matrix with Gaussian
noise at several levels, compresses each noisy copy with four methods
(SVD rank, quantization, 1-D Fourier fit, hybrid) and measures error
against the ORIGINAL clean matrix. This separates two error sources:
the injected noise and the compression itself.

Expected honest result: quantization error stays roughly constant while
noise passes straight through; low-rank SVD absorbs some noise but
distorts the signal. The table exists to show the trade-off.
"""

from __future__ import annotations

import numpy as np

__all__ = ["add_noise", "robustness_table"]


def _fourier_basis(n_cols: int, num_coeffs: int) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n_cols, endpoint=False)
    cols = [np.ones_like(t)]
    for k in range(1, num_coeffs + 1):
        cols.append(np.sin(k * t))
        cols.append(np.cos(k * t))
    return np.stack(cols, axis=1)


def add_noise(matrix: np.ndarray, noise: float, seed: int = 0) -> np.ndarray:
    """Add Gaussian noise scaled by the matrix standard deviation."""
    M = np.asarray(matrix, dtype=np.float64)
    if M.ndim != 2 or M.size == 0:
        raise ValueError("matrix must be non-empty and 2-D")
    if noise < 0.0:
        raise ValueError("noise must be >= 0")
    rng = np.random.default_rng(seed)
    sigma = float(M.std()) or 1.0
    return M + noise * sigma * rng.standard_normal(M.shape)


def _svd_compress(matrix: np.ndarray, rank: int = 4) -> np.ndarray:
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    rank = min(rank, s.size)
    return (u[:, :rank] * s[:rank]) @ vt[:rank, :]


def _quantize(matrix: np.ndarray, bits: int = 8) -> np.ndarray:
    lo, hi = float(matrix.min()), float(matrix.max())
    levels = 2 ** bits - 1
    scale = (hi - lo) / levels if hi > lo else 1.0
    codes = np.clip(np.round((matrix - lo) / scale), 0, levels)
    return codes * scale + lo


def _fourier_compress(matrix: np.ndarray, num_coeffs: int = 4) -> np.ndarray:
    basis = _fourier_basis(matrix.shape[1], num_coeffs)
    coeffs, _, _, _ = np.linalg.lstsq(basis, matrix.T, rcond=None)
    return coeffs.T @ basis.T


def robustness_table(matrix: np.ndarray,
                     noise_levels=(0.0, 0.02, 0.05, 0.1, 0.2),
                     seed: int = 0) -> list:
    """Compress noisy copies; report error vs the clean original per method.

    Returns one row per (method, noise level) with the relative
    Frobenius error of the reconstruction against the clean matrix.
    """
    M = np.asarray(matrix, dtype=np.float64)
    if M.ndim != 2 or M.size == 0:
        raise ValueError("matrix must be non-empty and 2-D")
    norm = float(np.linalg.norm(M))
    rows = []
    for noise in noise_levels:
        if noise < 0.0:
            raise ValueError("noise levels must be >= 0")
        noisy = add_noise(M, noise, seed=seed)
        methods = {
            "svd_rank4": _svd_compress(noisy, rank=4),
            "quantize8": _quantize(noisy, bits=8),
            "fourier4": _fourier_compress(noisy, num_coeffs=4),
            "hybrid": _quantize(_svd_compress(noisy, rank=4), bits=8),
        }
        for name, approx in methods.items():
            rel_error = float(np.linalg.norm(M - approx)) / norm if norm else 0.0
            rows.append({
                "method": name,
                "noise": float(noise),
                "relative_error": rel_error,
            })
    return rows
