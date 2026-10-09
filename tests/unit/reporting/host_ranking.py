"""A plain NumPy ranking into weighted groups, for tests to compare against.

The model ranks with `reporting/distribution_jax.quantile_groups`. This is
the same rule written directly on the host.
"""

from __future__ import annotations

import numpy as np


def quantile_groups(
    values: np.ndarray,
    weights: np.ndarray,
    n_groups: int,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Put each valid record in one of `n_groups` groups by cumulative weight.

    A record's percentile is the cumulative weight through and including that
    record, and a record whose percentile lands exactly on a cut belongs to
    the group below it. Group 0 is the bottom of the distribution. Records
    outside `valid` come back as -1.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mask = (
        np.ones(len(values), dtype=bool) if valid is None else np.asarray(valid, bool)
    )
    groups = np.full(len(values), -1, dtype=np.int64)
    live_values = values[mask]
    live_weights = weights[mask]
    total = live_weights.sum()
    if live_values.size == 0 or total <= 0:
        return groups
    order = np.argsort(live_values, kind="stable")
    percentiles = np.cumsum(live_weights[order]) / total
    assigned = np.clip(
        np.ceil(percentiles * n_groups).astype(np.int64) - 1, 0, n_groups - 1
    )
    live_groups = np.empty(live_values.size, dtype=np.int64)
    live_groups[order] = assigned
    groups[mask] = live_groups
    return groups
