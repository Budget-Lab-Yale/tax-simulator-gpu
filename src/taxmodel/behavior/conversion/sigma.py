"""Removal of the year's converted dollars from the households that converted them.

The conversion amounts are computed in the behavioral solve between the two
executions, on the mechanical world's marginal rates: for each leg of income, the change in the
tax advantage of taking the money as equity, times the response parameter
sigma, times the convertible part of the leg
(``behavior/kg/conversion_inflow.py``). This module applies those amounts, so
the dollars that enter the stock of unrealized gain and the dollars that leave
the households' pay are the same dollars.

Each leg is clamped to the income the household holds in it. The wealth
haircut erodes the pass-through legs earlier in the same pass, so a household
can arrive here with less income than the solve gave it to convert. The
clamp breaks the equality between the stock's inflow and the households' pay
cut, since the stock keeps the solved figure. The share clamped away is returned as a diagnostic column.

Converted dollars are salary, so base wages come down and tips and overtime
are unchanged. Nothing is added to realized gains here; the converted dollars
are unrealized, and their tax arrives in later years through the gains model.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from taxmodel.behavior import balance_sheet as bs
from taxmodel.behavior.kg.stock_inputs import CONVERSION_LEGS

Array = jnp.ndarray
Records = Mapping[str, Array]

# The frame column each conversion leg comes out of, in CONVERSION_LEGS order.
# The two wage legs are base pay: tips and overtime are separate columns and
# do not move.
LEG_SOURCE_COLUMNS: tuple[tuple[str, ...], ...] = (
    ("wages_base1",),
    ("wages_base2",),
    ("part_active",),
    ("scorp_active",),
    ("sole_prop1", "sole_prop2"),
)

# Legs whose earner-level companion columns move with them. Partnership income
# subject to self-employment tax is reported per earner alongside the
# household's active partnership income, and a conversion that shrank one
# without the other would leave the payroll base inconsistent. The companions
# are scaled by the same factor as the parent leg.
COMPANION_COLUMNS: dict[str, tuple[str, ...]] = {
    "part_active": ("part_se1", "part_se2"),
}

CLAMPED_SHARE_COLUMN = "conversion_clamped_share"

DEFAULTS: dict[str, Any] = {
    # The conversions the behavioral solve computed for this year, shaped
    # [household, leg] over CONVERSION_LEGS. The runner slices it out of the
    # stock recurrence's result and always supplies it; a scenario that prices no
    # conversions is handed an all-zero table of the same shape. None means a
    # caller supplied nothing, and the module converts nothing.
    "conversions": None,
}

MANIFEST: dict[str, Any] = {
    "name": "conversion_sigma",
    # The function in this file that is the response.
    "entry_point": "apply_conversions",
    # With no conversion table there is nothing to convert.
    "zero_response": {"conversions": None},
    "family": "conversion",
    "reads": tuple(name for columns in LEG_SOURCE_COLUMNS for name in columns)
    + tuple(name for companions in COMPANION_COLUMNS.values() for name in companions),
    "writes": tuple(name for columns in LEG_SOURCE_COLUMNS for name in columns)
    + tuple(name for companions in COMPANION_COLUMNS.values() for name in companions)
    + (CLAMPED_SHARE_COLUMN,),
    # This module reads no marginal rates; the conversion was priced in the
    # behavioral solve.
    "prices": (),
    "discrete_prices": (),
    "draws": (),
}


def apply_conversions(
    records: Records,
    prices_reform: Mapping[str, Array],
    prices_baseline: Mapping[str, Array],
    params: Mapping[str, Any],
    draws: Mapping[str, Array] | None = None,
) -> dict[str, Array]:
    """The in-year conversion module, in the behavior module contract's signature.

    Returns the pay columns it moved and the share of the year's conversion the
    clamp took away.
    """

    del prices_reform, prices_baseline, draws

    template = jnp.asarray(records["wages_base1"], dtype=jnp.float64)
    requested = params["conversions"]
    if requested is None:
        requested = jnp.zeros((template.shape[0], len(CONVERSION_LEGS)))
    requested = jnp.nan_to_num(jnp.asarray(requested, dtype=jnp.float64))

    changed: dict[str, Array] = {}
    asked = jnp.zeros_like(template)
    given = jnp.zeros_like(template)

    for position, columns in enumerate(LEG_SOURCE_COLUMNS):
        # The leg's value on the live frame; a column the frame lacks adds
        # nothing.
        leg = bs.total(records, columns, missing_like=template)
        wanted = requested[:, position]
        # A conversion cannot exceed the leg it comes out of, in either
        # direction.
        taken = jnp.clip(wanted, -jnp.abs(leg), jnp.abs(leg))
        asked = asked + wanted
        given = given + taken

        # The leg after conversion as a share of the leg before. A leg of zero
        # keeps a factor of one.
        factor = bs.divide_where_positive(leg - taken, leg, otherwise=1.0)

        # A leg reported as one column comes down by the converted amount. A
        # leg split across earners comes down proportionally, so the split is
        # preserved. The leg's companion columns come down proportionally in
        # either case.
        companions = COMPANION_COLUMNS.get(columns[0], ())
        if len(columns) == 1:
            name = columns[0]
            if name in records:
                reported = jnp.asarray(records[name], dtype=jnp.float64)
                changed[name] = bs.keep_what_was_missing(
                    reported, jnp.nan_to_num(reported) - taken
                )
            scaled_by_factor = companions
        else:
            scaled_by_factor = (*columns, *companions)

        for name in scaled_by_factor:
            if name in records:
                reported = jnp.asarray(records[name], dtype=jnp.float64)
                changed[name] = bs.keep_what_was_missing(
                    reported, jnp.nan_to_num(reported) * factor
                )

    changed[CLAMPED_SHARE_COLUMN] = bs.divide_where_positive(
        jnp.abs(given - asked), jnp.abs(asked)
    )
    return changed
