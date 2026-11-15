# RobustnessBench

[![CI](https://github.com/kapildhinwa2010-dev/RobustnessBench/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/RobustnessBench/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/RobustnessBench/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/RobustnessBench/actions/workflows/codeql.yml)

Compression error under noisy inputs, per method: noise vs compression error, separated.

## What it does

Perturbs a clean matrix with Gaussian noise scaled by its standard
deviation at several levels, compresses each noisy copy with four methods
(SVD rank-4, 8-bit quantization, 1-D Fourier fit, hybrid), and measures
reconstruction error against the ORIGINAL clean matrix. One row per
(method, noise level).

    from robustnessbench import robustness_table

    for row in robustness_table(W, noise_levels=(0.0, 0.05, 0.2)):
        print(row["method"], row["noise"], row["relative_error"])

## API

- add_noise(matrix, noise, seed) - Gaussian noise scaled by matrix std
- robustness_table(matrix, noise_levels, seed) - error vs clean original

## Honest results and limitations

- Expected result: quantization error stays roughly constant as noise
  grows (noise passes through); low-rank SVD absorbs some noise but also
  distorts the signal. Neither method "wins" in general.
- Error is relative Frobenius norm against the clean matrix; it mixes
  injected noise and compression error by construction.
- Gaussian noise only; adversarial or structured noise is not covered.

## Status

Part of the kapildhinwa2010-dev toolkit experiments. CI and CodeQL run on
every push to main. License: MIT.
