"""Schedule D: how much capital gain reaches AGI, and how much of it gets
the preferred rates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp

from taxmodel.taxes.calc._corners import capped_at, floored_at

Array = jnp.ndarray


@dataclass(frozen=True)
class CapitalGainsLaw:
    """Schedule D parameters, for the capital-gain step in `taxes/statutory_order.py`."""

    kg_loss_limit: Array
    kg_excl_rate: Array
    index_kg: Array
    index_kg_min_years_held: Array
    index_kg_purchased_after: Array
    index_kg_no_loss: Array


def calc_kg(
    records: Mapping[str, Array], law: CapitalGainsLaw, year: int
) -> dict[str, Array]:
    """Schedule D: the capital gain that reaches AGI, and the preferred part.

    Net capital gain -- the amount that gets the preferred rates -- is the
    smaller of the net long-term gain and the overall net gain, floored at
    zero. What reaches AGI is the overall net, with a net loss limited to the
    statutory annual deduction ($3,000 under current law).

    Basis indexation is off under every law we run. When a reform turns it on,
    long-term basis is written up by the CPI ratio the frame computed, but only
    for assets held longer than the stated minimum and bought after the stated
    effective year.
    """
    holding_years = records["kg_lt_years_held"]
    basis = records["kg_lt_basis"]
    indexation_applies = (
        (law.index_kg == 1)
        & ~jnp.isnan(basis)
        & ~jnp.isnan(holding_years)
        & (holding_years > law.index_kg_min_years_held)
        # The purchase year is the current year less the average holding
        # period rounded to the nearest whole year, halves to even.
        & ((year - jnp.round(holding_years)) > law.index_kg_purchased_after)
    )
    inflation_adjustment = jnp.where(
        indexation_applies,
        jnp.nan_to_num(basis, nan=0.0)
        * floored_at(records["kg_lt_cpi_ratio"] - 1.0, 0.0),
        0.0,
    )
    # A write-up cannot turn a gain into a loss when the law says so.
    inflation_adjustment = jnp.where(
        (law.index_kg_no_loss == 1) & (records["kg_lt"] > 0),
        capped_at(inflation_adjustment, records["kg_lt"]),
        inflation_adjustment,
    )
    kg_lt_adjusted = records["kg_lt"] - inflation_adjustment
    net_gain = records["kg_st"] + kg_lt_adjusted
    txbl_kg = floored_at(net_gain, -law.kg_loss_limit)
    # A household with exactly zero net gain is treated as having a gain, so the
    # exclusion applies. The amount is zero either way; only the slope differs.
    txbl_kg = jnp.where(txbl_kg >= 0, txbl_kg * (1.0 - law.kg_excl_rate), txbl_kg)
    # Which of the two amounts is written as the capped one decides the slope
    # when a household has exactly zero short-term gain, which is the common
    # case. Written this way, one more dollar of short-term gain leaves the
    # preferred amount at the long-term figure, where it belongs.
    return {
        "kg_pref": floored_at(capped_at(net_gain, kg_lt_adjusted), 0.0),
        "txbl_kg": txbl_kg,
    }
