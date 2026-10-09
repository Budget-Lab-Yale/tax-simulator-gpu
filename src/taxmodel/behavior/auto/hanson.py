"""Deductible auto-loan interest spending responds to an auto-interest deduction.

When a deduction for auto-loan interest appears, borrowing to buy a car gets
cheaper after tax, and households buy more car, borrow more, or refinance less
quickly. The response is linear in the change in the marginal rate on that
spending:

    new spending = old spending x [ 1 + e x (reform rate - baseline rate) ]

There is no auto-loan elasticity in the literature. The elasticity is the high
end of Hanson's estimated response of mortgage interest to the mortgage
interest deduction, on the judgment that a car is cheaper than a house and
less of a necessity, so the response should be at least as large:

    Andrew Hanson, "Taxes and Borrower Behavior: Evidence from the Mortgage
    Interest Deductibility Limit", Journal of Urban Economics 118 (2020).
    https://www.sciencedirect.com/science/article/pii/S0094119020300279

The response applies only where the law's auto-interest deduction limit is
nonzero and the year is at or past the first year the provision is in law.
Both come from the run.

The price this module reads is `mtr_auto_int_exp`, the marginal tax rate on a
dollar of deductible auto-loan interest.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from taxmodel.behavior import loader

Array = jnp.ndarray
Records = Mapping[str, Array]
Prices = Mapping[str, Array]

DEFAULTS: dict[str, Any] = {
    # Hanson (2020), high end of the mortgage-interest response, applied to
    # auto loans by analogy.
    "elasticity": -0.3,
    # First year the auto-interest deduction is in law.
    "first_year": 2025,
}

# The reform law's cap on deductible auto loan interest, as the frame names it.
# The frame joins every compiled law parameter on under its section-prefixed
# name.
AUTO_INTEREST_DEDUCTION_LIMIT = "agi.auto_int_ded_limit"

# The current year and the reform law's auto-interest deduction limit come
# from the run, on the frame; `params` stands in for a caller testing the
# module on a hand-built frame. See `loader.run_fact`.
MANIFEST: dict[str, Any] = {
    "name": "auto_interest_hanson",
    # The function in this file that is the response.
    "entry_point": "auto_interest_hanson",
    # The change in auto-loan interest is the elasticity times the change in
    # the marginal rate, so a zero elasticity leaves interest unchanged.
    "zero_response": {"elasticity": 0.0},
    "family": "auto",
    # The deduction limit is a bound-law column the frame carries under the
    # law's own name.
    "reads": ["auto_int_exp", "year", AUTO_INTEREST_DEDUCTION_LIMIT],
    "writes": ["auto_int_exp"],
    "prices": ["mtr_auto_int_exp"],
    "discrete_prices": [],
    "draws": [],
}


def auto_interest_hanson(
    records: Records,
    prices_reform: Prices,
    prices_baseline: Prices,
    params: Mapping[str, Any],
    draws: Mapping[str, Array],
) -> dict[str, Array]:
    """New deductible auto-loan interest spending after the response.

    Returns `{"auto_int_exp": ...}`. `draws` is unused; the response is
    deterministic given the price change.

    A reform that does not move `mtr_auto_int_exp` returns spending unchanged,
    as does any year before the provision starts.
    """
    rate_change = jnp.asarray(prices_reform["mtr_auto_int_exp"]) - jnp.asarray(
        prices_baseline["mtr_auto_int_exp"]
    )

    deduction_applies = loader.provision_applies(
        records, params, AUTO_INTEREST_DEDUCTION_LIMIT, "auto-interest"
    )

    percent_change = jnp.where(
        deduction_applies, params["elasticity"] * rate_change, 0.0
    )
    return {
        "auto_int_exp": jnp.asarray(records["auto_int_exp"]) * (1.0 + percent_change)
    }
