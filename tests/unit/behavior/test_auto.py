"""The auto-loan interest response to an auto-interest deduction."""

from __future__ import annotations

import jax.numpy as jnp

from taxmodel.behavior.auto import DEFAULTS, auto_interest_hanson
from taxmodel.behavior.auto.hanson import AUTO_INTEREST_DEDUCTION_LIMIT

# Three households with auto-loan interest, and one with none.
RECORDS = {"auto_int_exp": jnp.array([2_000.0, 800.0, 5_000.0, 0.0])}

# Baseline law has no auto-interest deduction, so a dollar of auto interest
# changes tax by nothing. The reform lets itemizers deduct it, so the rate
# becomes negative for the two itemizers and stays zero for the filer who takes
# the standard deduction.
BASELINE_RATES = {"mtr_auto_int_exp": jnp.array([0.0, 0.0, 0.0, 0.0])}
REFORM_RATES = {"mtr_auto_int_exp": jnp.array([-0.24, 0.0, -0.37, -0.24])}

ACTIVE_PARAMS = {**DEFAULTS, "year": 2026, AUTO_INTEREST_DEDUCTION_LIMIT: 10_000.0}


def test_no_price_change_means_no_response() -> None:
    """A reform that leaves the rate on auto interest alone changes no spending."""
    result = auto_interest_hanson(
        RECORDS, BASELINE_RATES, BASELINE_RATES, ACTIVE_PARAMS, draws={}
    )
    assert jnp.array_equal(result["auto_int_exp"], RECORDS["auto_int_exp"])


def test_one_household_worked_by_hand() -> None:
    """The first household, worked through the formula by hand.

    The rate goes from 0 to -0.24, a change of -0.24. With the -0.3 elasticity
    the multiplier is 1 + (-0.3)(-0.24) = 1.072, so $2,000 of auto interest
    becomes $2,144.
    """
    result = auto_interest_hanson(
        RECORDS, REFORM_RATES, BASELINE_RATES, ACTIVE_PARAMS, draws={}
    )
    # To the cent.
    assert abs(float(result["auto_int_exp"][0]) - 2_144.00) < 0.01
    # The third household, in the top bracket: 1 + (-0.3)(-0.37) = 1.111.
    assert abs(float(result["auto_int_exp"][2]) - 5_555.00) < 0.01
    # The standard-deduction filer gets no deduction and so no response.
    assert float(result["auto_int_exp"][1]) == 800.0
    # A household with no auto interest has nothing to scale.
    assert float(result["auto_int_exp"][3]) == 0.0


def test_no_response_before_the_provision_starts_or_with_a_zero_limit() -> None:
    """Before the provision starts, or where the law allows no deduction."""
    for off in (
        {**ACTIVE_PARAMS, "year": 2024},
        {**ACTIVE_PARAMS, AUTO_INTEREST_DEDUCTION_LIMIT: 0.0},
    ):
        result = auto_interest_hanson(
            RECORDS, REFORM_RATES, BASELINE_RATES, off, draws={}
        )
        assert jnp.array_equal(result["auto_int_exp"], RECORDS["auto_int_exp"])


def test_every_borrower_responds_once_anyone_has_a_deduction_limit() -> None:
    """The module is off only when no household has a nonzero deduction limit.

    Otherwise every borrower responds, including those whose own limit is
    zero.
    """
    mixed = {
        **RECORDS,
        AUTO_INTEREST_DEDUCTION_LIMIT: jnp.array([10_000.0, 10_000.0, 0.0, 10_000.0]),
    }
    params = {**DEFAULTS, "year": 2026}
    result = auto_interest_hanson(mixed, REFORM_RATES, BASELINE_RATES, params, draws={})
    assert abs(float(result["auto_int_exp"][0]) - 2_144.00) < 0.01
    # The provision is live, so the zero-limit household responds too.
    assert abs(float(result["auto_int_exp"][2]) - 5_000.0) > 0.01

    dead = {
        **RECORDS,
        AUTO_INTEREST_DEDUCTION_LIMIT: jnp.zeros(4),
    }
    result = auto_interest_hanson(dead, REFORM_RATES, BASELINE_RATES, params, draws={})
    # No record has a limit, so the module returns spending untouched.
    assert jnp.array_equal(result["auto_int_exp"], RECORDS["auto_int_exp"])
