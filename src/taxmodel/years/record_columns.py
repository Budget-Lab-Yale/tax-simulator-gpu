"""Reading dollar columns off a set of household records.

The corporate income adjustment and the wealth channel both read the same
record dictionary. Both need a column for the record count and shape, a
column read as zero when the frame does not carry it, the list of
pass-through business income columns, and net pass-through income.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

from taxmodel.behavior import balance_sheet as bs

Array = jnp.ndarray
Records = Mapping[str, Array]

# The pass-through business income columns. The corporate path scales these
# as one bundle and the wealth channel carries them as one block. The
# self-employment columns (part_se1, part_se2) are in the bundle but not in
# net pass-through income (`net_pass_through_income`).
PASS_THROUGH_FLOW_COLUMNS = (
    "part_active",
    "part_passive",
    "part_active_loss",
    "part_passive_loss",
    "part_179",
    "scorp_active",
    "scorp_passive",
    "scorp_active_loss",
    "scorp_passive_loss",
    "scorp_179",
    "part_se1",
    "part_se2",
    "sole_prop1",
    "sole_prop2",
    "farm1",
    "farm2",
)


def first_array(records: Records) -> Array:
    """One column, used only for its shape and its dtype."""
    if not records:
        raise ValueError("reading record columns needs at least one column")
    return jnp.asarray(next(iter(records.values())))


def sum_columns(records: Records, names: tuple[str, ...], template: Array) -> Array:
    """Add up several dollar columns, treating missing ones as zero."""
    total = jnp.zeros_like(template, dtype=jnp.float64)
    for name in names:
        total = total + bs.column(records, name, template)
    return total


def net_pass_through_income(records: Records, template: Array) -> Array:
    """Net business income from sole proprietorships, partnerships, S
    corporations and farms.

    Partnership and S corporation income is active plus passive income, less
    active and passive losses and the section 179 deduction. The
    self-employment columns are left out.
    """
    get = lambda name: bs.column(records, name, template)
    partnership = (
        get("part_active")
        + get("part_passive")
        - get("part_active_loss")
        - get("part_passive_loss")
        - get("part_179")
    )
    s_corporation = (
        get("scorp_active")
        + get("scorp_passive")
        - get("scorp_active_loss")
        - get("scorp_passive_loss")
        - get("scorp_179")
    )
    return (
        get("sole_prop1")
        + get("sole_prop2")
        + partnership
        + s_corporation
        + get("farm1")
        + get("farm2")
    )
