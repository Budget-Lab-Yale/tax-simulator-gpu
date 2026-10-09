"""The overtime response to an exemption for overtime pay, and the pay identity."""

from __future__ import annotations

import jax.numpy as jnp

from taxmodel.behavior.overtime import (
    DEFAULTS,
    INTENSIVE_ELASTICITY,
    MANIFEST,
    overtime_cahuc_carcillo,
    unfunded_overtime_dollars,
)
from taxmodel.behavior.overtime.cahuc_carcillo import OVERTIME_DEDUCTION_LIMIT

# Three households. The first has one earner with overtime, the second has both
# earners with overtime, the third has no overtime at all.
RECORDS = {
    "ot1": jnp.array([5_000.0, 8_000.0, 0.0]),
    "ot2": jnp.array([0.0, 3_000.0, 0.0]),
    "wages_base1": jnp.array([60_000.0, 45_000.0, 80_000.0]),
    "wages_base2": jnp.array([0.0, 30_000.0, 40_000.0]),
}

# Baseline law taxes overtime like any other pay; the reform exempts it from
# income tax, so the marginal rate on overtime falls to zero.
BASELINE_RATES = {
    "mtr_ot1": jnp.array([0.22, 0.24, 0.22]),
    "mtr_ot2": jnp.array([0.22, 0.24, 0.22]),
}
REFORM_RATES = {
    "mtr_ot1": jnp.array([0.0, 0.0, 0.0]),
    "mtr_ot2": jnp.array([0.0, 0.0, 0.0]),
}

# The exemption is in law and the response arrives in full at once.
ACTIVE_PARAMS = {
    **DEFAULTS,
    "phase_in_years": 0,
    "year": 2027,
    OVERTIME_DEDUCTION_LIMIT: 12_500.0,
}


def test_no_price_change_means_no_response() -> None:
    """A reform that leaves the overtime rate alone moves no pay."""
    result = overtime_cahuc_carcillo(
        RECORDS, BASELINE_RATES, BASELINE_RATES, ACTIVE_PARAMS, draws={}
    )
    for column in ("ot1", "ot2", "wages_base1", "wages_base2"):
        assert jnp.array_equal(result[column], RECORDS[column])


def test_one_household_worked_by_hand() -> None:
    """The first household's earner, worked through the formula by hand.

    The rate on overtime falls from 0.22 to 0, a change of -0.22. With the
    elasticity Cahuc and Carcillo imply and no phase-in, overtime rises by
    elasticity x (-0.22), which is about 32.3 percent, from $5,000 to about
    $6,616.
    """
    result = overtime_cahuc_carcillo(
        RECORDS, REFORM_RATES, BASELINE_RATES, ACTIVE_PARAMS, draws={}
    )
    expected_overtime = 5_000.0 * (1.0 + INTENSIVE_ELASTICITY * (0.0 - 0.22))
    # To the cent.
    assert abs(float(result["ot1"][0]) - expected_overtime) < 0.01
    # Base wages give up exactly what overtime gained.
    assert (
        abs(
            float(result["wages_base1"][0]) - (60_000.0 - (expected_overtime - 5_000.0))
        )
        < 0.01
    )


def test_total_pay_is_conserved() -> None:
    """New overtime comes out of base wages, dollar for dollar."""
    result = overtime_cahuc_carcillo(
        RECORDS, REFORM_RATES, BASELINE_RATES, ACTIVE_PARAMS, draws={}
    )
    before = sum(RECORDS[c] for c in MANIFEST["writes"])
    after = sum(result[c] for c in MANIFEST["writes"])
    assert float(jnp.max(jnp.abs(after - before))) < 0.01

    unfunded = unfunded_overtime_dollars(
        RECORDS, REFORM_RATES, BASELINE_RATES, ACTIVE_PARAMS, draws={}
    )
    assert float(jnp.max(jnp.abs(unfunded))) == 0.0


def test_the_base_wage_floor_binds_and_is_reported() -> None:
    """An earner with almost no base pay cannot fund all their new overtime.

    Base wages stop at zero rather than going negative, and the dollars the
    floor could not fund show up in `unfunded_overtime_dollars` instead of
    quietly breaking the identity.
    """
    records = {
        "ot1": jnp.array([50_000.0]),
        "ot2": jnp.array([0.0]),
        "wages_base1": jnp.array([100.0]),
        "wages_base2": jnp.array([0.0]),
    }
    baseline = {"mtr_ot1": jnp.array([0.22]), "mtr_ot2": jnp.array([0.0])}
    reform = {"mtr_ot1": jnp.array([0.0]), "mtr_ot2": jnp.array([0.0])}

    result = overtime_cahuc_carcillo(records, reform, baseline, ACTIVE_PARAMS, draws={})
    assert float(result["wages_base1"][0]) == 0.0

    overtime_added = float(result["ot1"][0]) - 50_000.0
    unfunded = unfunded_overtime_dollars(
        records, reform, baseline, ACTIVE_PARAMS, draws={}
    )
    # The identity breaks by exactly the part base pay could not cover.
    assert abs(float(unfunded[0]) - (overtime_added - 100.0)) < 0.01
    before = sum(records[c] for c in MANIFEST["writes"])
    after = sum(result[c] for c in MANIFEST["writes"])
    assert abs(float(after[0] - before[0]) - float(unfunded[0])) < 0.01


def test_only_earners_who_already_have_overtime_respond() -> None:
    """This is an intensive-margin response: nobody starts getting overtime."""
    result = overtime_cahuc_carcillo(
        RECORDS, REFORM_RATES, BASELINE_RATES, ACTIVE_PARAMS, draws={}
    )
    # Household 3 has no overtime, and the earner with no overtime in
    # household 1 is the second earner.
    assert float(result["ot1"][2]) == 0.0
    assert float(result["ot2"][0]) == 0.0
    assert float(result["wages_base1"][2]) == 80_000.0
    assert float(result["wages_base2"][0]) == 0.0


def test_the_response_phases_in_over_the_stated_years() -> None:
    """A one-year phase-in gives nothing in the provision's first year and the
    full response a year later; a three-year phase-in gives a third after one
    year."""
    # The provision's first year is 2025.
    first_year = {
        **DEFAULTS,
        "phase_in_years": 1,
        "year": 2025,
        OVERTIME_DEDUCTION_LIMIT: 12_500.0,
    }
    result = overtime_cahuc_carcillo(
        RECORDS, REFORM_RATES, BASELINE_RATES, first_year, draws={}
    )
    assert jnp.array_equal(result["ot1"], RECORDS["ot1"])

    # A year later it is fully in.
    later = {**first_year, "year": 2026}
    full = overtime_cahuc_carcillo(
        RECORDS, REFORM_RATES, BASELINE_RATES, later, draws={}
    )
    immediate = overtime_cahuc_carcillo(
        RECORDS, REFORM_RATES, BASELINE_RATES, ACTIVE_PARAMS, draws={}
    )
    assert jnp.allclose(full["ot1"], immediate["ot1"])

    # The three-year phase-in is one third of the way in after one year.
    third = {**first_year, "phase_in_years": 3, "year": 2026}
    partial = overtime_cahuc_carcillo(
        RECORDS, REFORM_RATES, BASELINE_RATES, third, draws={}
    )
    added_partial = float(partial["ot1"][0]) - 5_000.0
    added_full = float(immediate["ot1"][0]) - 5_000.0
    assert abs(added_partial - added_full / 3.0) < 0.01


def test_no_response_before_the_provision_starts_or_with_a_zero_limit() -> None:
    """Before the provision starts, or with no deduction allowed, nothing moves."""
    for off in (
        {**ACTIVE_PARAMS, "year": 2024},
        {**ACTIVE_PARAMS, OVERTIME_DEDUCTION_LIMIT: 0.0},
    ):
        result = overtime_cahuc_carcillo(
            RECORDS, REFORM_RATES, BASELINE_RATES, off, draws={}
        )
        for column in ("ot1", "ot2", "wages_base1", "wages_base2"):
            assert jnp.array_equal(result[column], RECORDS[column])
