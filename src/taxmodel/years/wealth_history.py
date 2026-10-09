"""One year of wealth bathtub quantities, recorded for the diagnostics tables.

``taxmodel.reporting.wealth_diagnostics.WealthShortfallHistory`` takes one entry
per simulated year for each cell-level and household-level quantity in the
bathtub. The annual step computes all of them while advancing the shortfall;
this module returns them under the names the reporting layer uses.

Two of the pieces are read straight off the mechanical frame, where
``wealth_state.apply_carried_stock`` wrote them: ``D_alloc`` is the
household's share of its cell's carried shortfall in dollars, and
``wealth_haircut`` is that share as a fraction of gross assets. The rest are
cell totals over the same cell assignment the year's accounting used.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

from taxmodel.years import wealth_arithmetic, wealth_state

Array = jnp.ndarray
Records = Mapping[str, Array]


def year_wealth_channel_state(
    records: Records,
    assignment: wealth_state.CellAssignment,
    cell_inflow: Array,
    profile: wealth_state.WealthProfileYear,
) -> dict[str, Array]:
    """The year's cell and household wealth-channel state, in reporting's names.

    ``records`` is the frame whose finished tax burden advanced the stock (the
    mechanical frame on the first sweep), so it carries the two haircut columns
    and this year's weights. ``cell_inflow`` is this year's addition to the
    stock, returned by `wealth_state.advance_stock` beside the new stock.
    """
    n_ages, n_bins = profile.saving_shares.shape
    weight = jnp.asarray(records["weight"])
    by_cell = lambda values: wealth_arithmetic.sum_by_cell(
        values,
        assignment.row,
        assignment.bin_index,
        assignment.has_cell,
        n_ages,
        n_bins,
    )

    capital = wealth_state.capital_income(records, profile.pass_through_weight)
    gross = wealth_state.economic_gross(records)
    signed_capital = by_cell(weight * capital)
    gross_assets = by_cell(weight * gross)
    positive_net_worth = jnp.maximum(assignment.rank_net_worth, 0.0)
    net_worth_total = by_cell(weight * positive_net_worth)

    enough_gross = gross_assets > wealth_arithmetic.WEALTH_DYN_EPS
    yield_on_assets = jnp.clip(
        jnp.where(
            enough_gross, signed_capital / jnp.where(enough_gross, gross_assets, 1.0), 0.0
        ),
        0.0,
        wealth_state.CAPITAL_YIELD_MAXIMUM,
    )

    zero_households = jnp.zeros_like(weight)
    return {
        "cell_inflow": cell_inflow,
        "growth_factor": wealth_state.cell_growth(profile),
        "capital_income_per_dollar_of_assets": yield_on_assets,
        "cell_net_worth": net_worth_total,
        "allocated_dollars": jnp.asarray(records.get("D_alloc", zero_households)),
        "haircut_fraction": jnp.asarray(records.get("wealth_haircut", zero_households)),
        "weight": weight,
    }
