"""Put several scenarios' or years' versions of one object side by side."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp


def stack_lanes(items: Sequence[object]) -> object:
    """Stack objects of the same shape on a new leading axis.

    The objects are pytrees (a bound law, a year's outputs, a scenario's
    gains tables) with the same leaves in the same places. Stacking them leaf
    by leaf gives one object of the same structure whose arrays carry an extra
    first axis. The run uses that axis for the scenario lanes a compiled year
    runs side by side, and for the years of a projection. Device arrays stay
    on the device.
    """
    if not items:
        raise ValueError("stacking needs at least one item")
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *items)
