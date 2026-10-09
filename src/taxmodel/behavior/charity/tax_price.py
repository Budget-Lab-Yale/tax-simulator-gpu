"""Cash charitable giving responds to the tax price of a dollar given.

The tax price of giving is what a dollar of cash contribution costs the donor
after tax. A dollar given to charity costs the donor a dollar and saves the
donor whatever tax the deduction removes, so the price is one plus the
marginal tax change caused by that dollar. Because giving a dollar lowers tax,
the marginal rate on charitable cash is negative: a donor in a 37 percent
bracket who itemizes has `mtr_char_cash = -0.37` and a tax price of 0.63.
Someone who takes the standard deduction gets no tax saving, so the rate is
zero and the price is one.

A reform that makes giving less tax-favored raises the price. With a negative
tax-price elasticity, giving falls. The response is

    percent change in giving = e x [ (1 + reform rate) / (1 + baseline rate) - 1 ]

capped at plus or minus 100 percent.

The elasticity is a parameter. Two values are named below, -0.5 and -1.0.

`mtr_char_cash` is the derivative of net income tax with respect to
`char_cash` (`src/taxmodel/taxes/marginal_rates.py`), so a deductible dollar
given has a negative rate and `1 + rate` is the tax price.

The price this module reads is `mtr_char_cash`, stored under both laws.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

Array = jnp.ndarray
Records = Mapping[str, Array]
Prices = Mapping[str, Array]

# The default tax-price elasticity of cash giving. Scenarios set their own
# value in their behavior files.
HALF_ELASTICITY = -0.5

DEFAULTS: dict[str, float] = {
    # Tax-price elasticity of cash giving, from Randolph (1995), "Dynamic
    # Income, Progressive Taxes, and the Timing of Charitable Contributions,"
    # Journal of Political Economy 103(4).
    "elasticity": HALF_ELASTICITY,
    # Largest percent change in giving the formula may produce, in either
    # direction. A 100 percent cap keeps a household whose baseline tax price
    # is near zero from producing an implausible response.
    "max_percent_change": 1.0,
}

MANIFEST: dict[str, Any] = {
    "name": "charity_tax_price",
    # The function in this file that is the response.
    "entry_point": "charity_tax_price",
    # Giving changes by the elasticity times the proportional change in the
    # tax price, so a zero elasticity leaves giving unchanged.
    "zero_response": {"elasticity": 0.0},
    "family": "charity",
    "reads": ["char_cash"],
    "writes": ["char_cash"],
    "prices": ["mtr_char_cash"],
    "discrete_prices": [],
    "draws": [],
}


def charity_tax_price(
    records: Records,
    prices_reform: Prices,
    prices_baseline: Prices,
    params: Mapping[str, float],
    draws: Mapping[str, Array],
) -> dict[str, Array]:
    """New cash charitable contributions after the tax-price response.

    Returns `{"char_cash": ...}`. `draws` is unused; the response is
    deterministic given the price change.

    A reform that does not move `mtr_char_cash` returns giving unchanged.
    """
    elasticity = params["elasticity"]
    cap = params["max_percent_change"]

    price_reform = 1.0 + jnp.asarray(prices_reform["mtr_char_cash"])
    price_baseline = 1.0 + jnp.asarray(prices_baseline["mtr_char_cash"])

    percent_change = elasticity * (price_reform / price_baseline - 1.0)
    percent_change = jnp.clip(percent_change, -cap, cap)

    return {"char_cash": jnp.asarray(records["char_cash"]) * (1.0 + percent_change)}
