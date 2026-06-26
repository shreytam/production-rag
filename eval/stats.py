"""Statistical utilities for evaluation result analysis.

All randomness is seeded inside each function so repeated calls with the same
``seed`` produce identical results regardless of external numpy RNG state.
"""

from __future__ import annotations

import numpy as np


def bootstrap_ci(
    values: list[float],
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for the mean.

    Parameters
    ----------
    values:
        Observed metric values (one per evaluation example).
    n:
        Number of bootstrap resamples.
    alpha:
        Significance level; the CI covers (1 - alpha) of the distribution.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    (mean, lo, hi)
        Empirical mean and percentile-method CI bounds.
    """
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)
    means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n)])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return float(arr.mean()), lo, hi


def paired_bootstrap(
    a: list[float],
    b: list[float],
    n: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for the difference in means (b − a).

    Both lists must have the same length.  Resampling uses paired indices so
    the correlation structure between versions is preserved.

    Returns
    -------
    (mean_diff, lo, hi)
        Observed mean difference and percentile-method CI.
    """
    if len(a) != len(b):
        raise ValueError(f"a and b must be equal length, got {len(a)} vs {len(b)}")
    rng = np.random.default_rng(seed)
    arr_a = np.array(a, dtype=float)
    arr_b = np.array(b, dtype=float)
    diffs = arr_b - arr_a
    mean_diffs = np.array(
        [rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n)]
    )
    lo = float(np.percentile(mean_diffs, 100 * alpha / 2))
    hi = float(np.percentile(mean_diffs, 100 * (1 - alpha / 2)))
    return float(diffs.mean()), lo, hi
