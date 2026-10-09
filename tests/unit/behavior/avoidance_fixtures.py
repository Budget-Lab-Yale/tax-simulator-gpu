"""Synthetic households and prices for the two avoidance module tests.

Nothing here reads the real frame: a handful of made-up balance sheets and a
flat marginal rate on every asset column are enough to check the arithmetic
these modules do.
"""

from __future__ import annotations

import numpy as np

from taxmodel.behavior import balance_sheet as bs
from taxmodel.behavior.wealth.avoidance import CONCEALED_INCOME_GROUPS
from taxmodel.taxes.calc.estate import ESTATE_ASSET_COLS
from taxmodel.taxes.marginal_rates import ESTATE_BASE_RATE

# Every income column the two modules read or write, from the wealth module's
# own list of concealed income.
INCOME_COLS: tuple[str, ...] = tuple(
    dict.fromkeys(
        name
        for gate, scaled, _kind in CONCEALED_INCOME_GROUPS
        for name in (*gate, *scaled)
    )
)


def households(count: int, **columns: object) -> dict[str, np.ndarray]:
    """A frame of `count` households with zeros everywhere but what is named."""
    record: dict[str, np.ndarray] = {
        name: np.zeros(count)
        for name in (*ESTATE_ASSET_COLS, *bs.DEBT_COLS, *INCOME_COLS)
    }
    for name in bs.EVASION_FACTOR_COLS:
        record[name] = np.full(count, bs.NO_EVASION)
    for name, value in columns.items():
        record[name] = np.broadcast_to(np.asarray(value, dtype=float), (count,)).copy()
    return record


def flat_prices(tax: str, rate: object, count: int) -> dict[str, np.ndarray]:
    """One marginal rate charged on every asset column, for one tax.

    `tax` is `"wealth_tax"` or `"estate_tax"`. A flat rate on every asset is
    what a tax charged on net worth as a whole produces, and it makes the
    aggregated rate on any base equal to that rate.

    The per-asset rates are always written under the wealth tax's names,
    because those are the only per-asset marginal rates the engine computes.
    The estate case adds the one rate the estate module actually reads: the tax
    on a dollar of the taxable estate itself.
    """
    rates = np.broadcast_to(np.asarray(rate, dtype=float), (count,)).copy()
    prices = {bs.wealth_tax_rate_key(name): rates for name in ESTATE_ASSET_COLS}
    if tax == "estate_tax":
        # The estate avoidance module reads the base-dollar rate; under a flat rate on everything it equals the per-asset rate.
        prices[ESTATE_BASE_RATE] = rates
    return prices


def only(value: object) -> float:
    """The single household's value, as a plain number.

    Most of these fixtures hold one household, and reading a one-household
    array as a number keeps the expected answers in the tests readable.
    """
    flat = np.asarray(value).reshape(-1)
    if flat.size != 1:
        raise ValueError(f"expected one household's value, got {flat.size}")
    return float(flat[0])
