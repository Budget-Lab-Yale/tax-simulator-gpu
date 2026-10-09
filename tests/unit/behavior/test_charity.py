"""The charitable giving response to a change in the tax price of giving."""

from __future__ import annotations

import jax.numpy as jnp

from taxmodel.behavior.charity import (
    DEFAULTS,
    charity_tax_price,
)

# Four donors: an itemizer in the top bracket, an itemizer in a middle
# bracket, a standard-deduction filer who gets no tax saving from giving, and a
# donor who gives nothing.
RECORDS = {"char_cash": jnp.array([50_000.0, 5_000.0, 1_000.0, 0.0])}
BASELINE_RATES = {"mtr_char_cash": jnp.array([-0.37, -0.24, 0.0, -0.37])}


def test_no_price_change_means_no_response() -> None:
    """A reform that leaves the charitable rate alone changes no giving."""
    result = charity_tax_price(
        RECORDS, BASELINE_RATES, BASELINE_RATES, DEFAULTS, draws={}
    )
    assert jnp.array_equal(result["char_cash"], RECORDS["char_cash"])


def test_one_household_worked_by_hand() -> None:
    """One donor, worked through the tax-price formula by hand.

    The top-bracket itemizer's rate goes from -0.37 to -0.24, so the tax price
    of a dollar given rises from 0.63 to 0.76. With the -0.5 elasticity the
    percent change is -0.5 x (0.76/0.63 - 1) = -0.1031746..., and giving of
    $50,000 falls to $44,841.27.
    """
    reform_rates = {"mtr_char_cash": jnp.array([-0.24, -0.24, 0.0, -0.37])}
    result = charity_tax_price(
        RECORDS, reform_rates, BASELINE_RATES, DEFAULTS, draws={}
    )

    percent_change = DEFAULTS["elasticity"] * (0.76 / 0.63 - 1.0)
    expected = 50_000.0 * (1.0 + percent_change)
    # To the cent.
    assert abs(float(result["char_cash"][0]) - expected) < 0.01

    # The middle-bracket itemizer and the nongiver see no price change of their
    # own, so nothing moves for them.
    assert float(result["char_cash"][1]) == 5_000.0
    assert float(result["char_cash"][3]) == 0.0


def test_the_cap_binds_at_an_extreme_price() -> None:
    """Giving cannot move by more than 100 percent, whatever the prices say.

    A baseline rate near -1 makes the baseline tax price near zero, so the
    price ratio explodes. The percent change in giving is capped at plus or
    minus 100 percent.

    Only the downward cap can ever bind. The price ratio is positive, so with a
    negative elasticity the largest possible increase in giving is the size of
    the elasticity itself, which is at most 1 for the variants we run. The
    second donor below shows that: an enormous price collapse still raises
    giving by less than the cap.
    """
    records = {"char_cash": jnp.array([1_000.0, 1_000.0])}
    # First donor: the tax price goes from almost nothing to 0.99, so giving
    # would fall by far more than 100 percent and end up negative.
    # Second donor: the mirror image, a price collapse.
    baseline = {"mtr_char_cash": jnp.array([-0.999, -0.01])}
    reform = {"mtr_char_cash": jnp.array([-0.01, -0.999])}
    result = charity_tax_price(records, reform, baseline, DEFAULTS, draws={})

    percent_change = result["char_cash"] / records["char_cash"] - 1.0
    assert float(jnp.max(jnp.abs(percent_change))) <= DEFAULTS["max_percent_change"]
    # The cap actually binds for the first donor rather than merely leaving the
    # answer inside it: giving stops at zero instead of going negative.
    assert float(result["char_cash"][0]) == 0.0
    assert float(result["char_cash"][1]) > 1_000.0
