"""Deterministic subset selection for fast CI gate runs."""

from __future__ import annotations

import random
from typing import TypeVar

T = TypeVar("T")


def fast_subset(items: list[T], n: int = 15, seed: int = 0) -> list[T]:
    """Return a deterministic random subset of up to *n* items.

    If ``len(items) <= n`` the full list is returned unchanged.  The subset
    is sampled without replacement using the given *seed* for reproducibility.
    """
    if len(items) <= n:
        return list(items)
    rng = random.Random(seed)
    return rng.sample(items, n)
