"""Relabeling compensation as capital gain, on a three-household fixture."""

from __future__ import annotations

import jax
import numpy as np

from taxmodel.behavior.kg.conversion_inflow import (
    convertible_share_by_leg,
    price_conversions_all_years,
    price_conversions_one_year,
)

from .conftest import N_AGES, build_pool

# The two calibration values, from config/calibrations/kg/conversion.yaml.
SIGMA = 0.15
PASS_THROUGH_LABOR_SHARE = 0.75


def _priced(pool):
    return price_conversions_all_years(
        pool, SIGMA, PASS_THROUGH_LABOR_SHARE, n_ages=N_AGES
    )


def _priced_one_year(pool):
    one_year = jax.tree.map(lambda x: x[0], pool)
    return price_conversions_one_year(
        one_year, SIGMA, PASS_THROUGH_LABOR_SHARE, n_ages=N_AGES
    )


def test_zero_wedge_converts_exactly_nothing():
    """No change in any rate leaves no reason to relabel a dollar of pay.

    Neither the rate on the leg nor the rate on realized gain moves, so the
    wedge is zero and the conversion is zero exactly, not to a rounding
    tolerance.
    """

    priced = _priced(build_pool(marginal_rate_change=0.0))

    assert np.array_equal(
        np.asarray(priced.by_household_leg),
        np.zeros_like(priced.by_household_leg),
    )
    assert np.array_equal(
        np.asarray(priced.inflow_by_age), np.zeros_like(priced.inflow_by_age)
    )


def test_equal_rises_in_the_ordinary_and_gains_rates_convert_nothing():
    """Five points more on wages and five points more on gains cancel out.

    The two rate changes are equal but reach the wedge through different
    arithmetic, so this one closes to a cent rather than exactly.
    """

    priced = _priced(build_pool(marginal_rate_change=0.05, gains_rate_change=0.05))

    assert np.abs(np.asarray(priced.inflow_by_age)).max() < 0.01


def test_a_higher_gains_rate_converts_equity_back_to_pay():
    """A higher rate on gains with the ordinary rate unchanged makes pay cheaper.

    The wedge is minus five points, so the eligible household moves income out
    of equity and into wages and pass-through income: the conversion is
    negative and the inflow to the stock of gain is negative.
    """

    priced = _priced_one_year(
        build_pool(marginal_rate_change=0.0, gains_rate_change=0.05)
    )
    converted = np.asarray(priced.by_household_leg)

    expected_wages = SIGMA * -0.05 * 500_000.0
    assert abs(converted[1, 0] - expected_wages) < 0.01
    assert converted[1, 2] < 0.0
    assert np.asarray(priced.inflow_by_age).sum() < 0.0


def test_only_business_owners_above_the_threshold_convert():
    """The pool and the conversion are worked out by hand for one year.

    Household 0 is an employee: no active business income, so nothing.
    Household 2 has a partnership but taxable income below the top bracket, so
    also nothing.  Household 1 converts on both its wages and the labor share
    of its partnership income, against a wedge of ten points on the leg less
    five points on realized gain, which is five points net.
    """

    priced = _priced_one_year(
        build_pool(marginal_rate_change=0.10, gains_rate_change=0.05)
    )
    converted = np.asarray(priced.by_household_leg)

    assert converted[0].sum() == 0.0
    assert converted[2].sum() == 0.0

    net_wedge = 0.10 - 0.05
    expected_wages = SIGMA * net_wedge * 500_000.0
    expected_partnership = SIGMA * net_wedge * PASS_THROUGH_LABOR_SHARE * 400_000.0
    assert abs(converted[1, 0] - expected_wages) < 0.01
    assert abs(converted[1, 2] - expected_partnership) < 0.01

    # Household 1 sits in age cell 2 and carries a weight of 20.
    inflow = np.asarray(priced.inflow_by_age)
    expected_inflow = 20.0 * (expected_wages + expected_partnership)
    assert abs(inflow[2] - expected_inflow) < 0.01
    assert inflow.sum() == inflow[2]


def test_conversion_cannot_exceed_the_leg_it_comes_out_of():
    """A rate change large enough to relabel the whole leg stops at the leg."""

    priced = _priced_one_year(build_pool(marginal_rate_change=20.0))
    converted = np.asarray(priced.by_household_leg)

    assert converted[1, 0] == 500_000.0
    assert converted[1, 2] == 400_000.0


def test_only_the_labor_content_of_pass_through_income_is_convertible():
    shares = np.asarray(convertible_share_by_leg(PASS_THROUGH_LABOR_SHARE))
    assert shares.tolist() == [1.0, 1.0, 0.75, 0.75, 0.75]
