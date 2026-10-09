"""Targeted checks on the corporate incidence paths.

One hand-computed example per channel at the default parameter values, the
present value against a plainly written summation, and the case that has to
produce nothing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from taxmodel.years.corporate import paths
from taxmodel.years.corporate.paths import (
    CorporateParameters,
    MacroAggregates,
    assemble_corporate_wedge,
    build_corporate_paths,
    check_enactment_year_is_in_window,
    check_wedge_runs_one_way,
    debt_rollover_share,
    read_debt_maturity_schedule,
)

# A flat, easy-to-check economy in billions of dollars per year. After-tax
# corporate profit grows at exactly 4 percent a year so the tail growth rate
# comes out at 4 percent with no rounding argument.
FIRST_YEAR = 2026
LAST_SIMULATED_YEAR = 2035
LAST_PROJECTED_YEAR = 2055
PROFIT_IN_FIRST_YEAR = 2_000.0
PROFIT_GROWTH = 0.04
TREASURY_PERCENT = 4.0  # so the discount rate is 0.04 + 0.05 = 0.09
INTEREST_INCOME = 1_000.0
RENTAL_INCOME = 500.0
PROPRIETORS_INCOME = 2_500.0


def make_macro(
    first_year: int = FIRST_YEAR, last_year: int = LAST_PROJECTED_YEAR
) -> MacroAggregates:
    years = np.arange(first_year, last_year + 1)
    profit = PROFIT_IN_FIRST_YEAR * (1.0 + PROFIT_GROWTH) ** (years - first_year)
    ones = np.ones_like(years, dtype=float)
    return MacroAggregates(
        years=years,
        after_tax_profit=profit,
        treasury_10_year_percent=ones * TREASURY_PERCENT,
        interest_income=ones * INTEREST_INCOME,
        rental_income=ones * RENTAL_INCOME,
        proprietors_income=ones * PROPRIETORS_INCOME,
    )


def make_paths(
    wedge_per_year: float = 100.0,
    enactment_year: int = FIRST_YEAR + 1,
    through_year: int = LAST_SIMULATED_YEAR,
    sunset_year: int | None = None,
    parameters: CorporateParameters | None = None,
):
    """Paths for a flat revenue change that starts in ``enactment_year`` and,
    if ``sunset_year`` is given, stops there, with the revenue file running
    through ``through_year``."""
    years = np.arange(FIRST_YEAR, through_year + 1)
    live = years >= enactment_year
    if sunset_year is not None:
        live &= years < sunset_year
    wedge = np.where(live, wedge_per_year, 0.0)
    return build_corporate_paths(
        years,
        wedge,
        make_macro(),
        range(FIRST_YEAR, LAST_SIMULATED_YEAR + 1),
        parameters=parameters,
    )


# ---------------------------------------------------------------------------
# The debt rollover ramp
# ---------------------------------------------------------------------------


def test_debt_rollover_ramp_matches_the_default_schedule():
    tenor, cumulative = read_debt_maturity_schedule()
    share = debt_rollover_share(np.array([0, 1, 5, 10, 30, 31, 80]))

    # Nothing has been refinanced in the enactment year itself.
    assert share[0] == 0.0
    # One year out, only the paper maturing within a year has repriced.
    assert share[1] == pytest.approx(cumulative[0])
    assert share[2] == pytest.approx(cumulative[4])
    # About 82 percent of debt has refinanced at ten years.
    assert share[3] == pytest.approx(0.8260, abs=5e-4)
    # The schedule runs out at thirty years, and past it debt counts as fully
    # refinanced, at exactly one.
    assert int(tenor[-1]) == 30
    assert share[4] == pytest.approx(1.0)
    assert share[5] == 1.0
    assert share[6] == 1.0


# ---------------------------------------------------------------------------
# Channel 1 — the cut to corporate payouts and the four income factors
# ---------------------------------------------------------------------------


def test_payout_cut_in_the_enactment_year_is_hand_computable():
    parameters = CorporateParameters()
    paths = make_paths(wedge_per_year=100.0, enactment_year=FIRST_YEAR + 1)
    i = paths.row_index_for_year(FIRST_YEAR + 1)

    # No capital has turned over in the enactment year, so the whole $100B
    # still sits on corporate flows: the rent piece by definition, and the
    # normal piece because nothing has had a chance to migrate.
    assert paths.capital_turned_over_share[i] == pytest.approx(0.0)
    assert paths.burden_staying_on_corporate_flows[i] == pytest.approx(100.0)

    # Payouts fall by the burden as a share of after-tax profit, scaled by the
    # US-taxable exposure of one.
    profit = PROFIT_IN_FIRST_YEAR * (1.0 + PROFIT_GROWTH)
    assert paths.payout_cut[i] == pytest.approx(-100.0 / profit)

    # Dividends take that cut at the C corporation share of dividends.
    assert paths.dividend_factor[i] == pytest.approx(
        1.0 + parameters.corporate_share_of_dividends * paths.payout_cut[i]
    )
    # Nothing has migrated, so the three noncorporate lines are untouched.
    assert paths.interest_factor[i] == pytest.approx(1.0)
    assert paths.rent_factor[i] == pytest.approx(1.0)
    assert paths.pass_through_factor[i] == pytest.approx(1.0)


def test_migrated_burden_lands_on_the_noncorporate_lines_at_their_weights():
    parameters = CorporateParameters()
    enactment = FIRST_YEAR + 1
    paths = make_paths(wedge_per_year=100.0, enactment_year=enactment)
    year = enactment + 5
    i = paths.row_index_for_year(year)

    turned_over = 1.0 - (1.0 - parameters.depreciation_rate) ** 5
    assert paths.capital_turned_over_share[i] == pytest.approx(turned_over)

    # Of the normal-return share of the wedge, the part on replaced capital
    # that is not corporate migrates out of the corporate sector.
    migrated = (
        (1.0 - parameters.corporate_share_of_normal_capital)
        * turned_over
        * parameters.normal_return_share
        * 100.0
    )
    assert paths.migrated_burden[i] == pytest.approx(migrated)

    # It is spread over the three national totals, pass-through income entering
    # at its capital weight.
    pass_through_base = PROPRIETORS_INCOME * parameters.pass_through_capital_weight
    total_base = INTEREST_INCOME + RENTAL_INCOME + pass_through_base
    on_interest = migrated * INTEREST_INCOME / total_base
    on_rent = migrated * RENTAL_INCOME / total_base
    on_pass_through = migrated * pass_through_base / total_base

    # Only refinanced debt can reprice; the rest is a bondholder revaluation.
    rolled = debt_rollover_share(np.array([5]))[0]
    assert paths.interest_flow_cut[i] == pytest.approx(on_interest * rolled)
    assert paths.bondholder_revaluation[i] == pytest.approx(
        on_interest * (1.0 - rolled)
    )
    assert paths.rent_flow_cut[i] == pytest.approx(on_rent)
    assert paths.pass_through_flow_cut[i] == pytest.approx(on_pass_through)

    assert paths.interest_factor[i] == pytest.approx(
        1.0 - on_interest * rolled / INTEREST_INCOME
    )
    assert paths.rent_factor[i] == pytest.approx(1.0 - on_rent / RENTAL_INCOME)
    # The capital weight applies twice on purpose: the burden is spread over
    # the capital fifth of pass-through income, and then only that fifth of a
    # household's pass-through column is cut.
    assert paths.pass_through_factor[i] == pytest.approx(
        1.0
        - parameters.pass_through_capital_weight
        * on_pass_through
        / pass_through_base
    )


# ---------------------------------------------------------------------------
# Channel 2 — the equity markdown
# ---------------------------------------------------------------------------


def flat_profit_macro() -> MacroAggregates:
    """An economy with profit flat at $2,000B, so a flat wedge is hand-checkable.

    A flat wedge against flat profit makes the markdown the plain ratio of the
    two, with no growth argument in the way.
    """
    years = np.arange(FIRST_YEAR, LAST_PROJECTED_YEAR + 1)
    ones = np.ones_like(years, dtype=float)
    return MacroAggregates(
        years=years,
        after_tax_profit=ones * PROFIT_IN_FIRST_YEAR,
        treasury_10_year_percent=ones * TREASURY_PERCENT,
        interest_income=ones * INTEREST_INCOME,
        rental_income=ones * RENTAL_INCOME,
        proprietors_income=ones * PROPRIETORS_INCOME,
    )


def test_a_permanent_rent_only_change_marks_equity_down_by_a_constant():
    # With no normal-return share there is nothing to migrate, so the priced
    # burden is the whole wedge for ever. Against flat profit that makes the
    # markdown the plain ratio of wedge to profit, the same in every year.
    years = np.arange(FIRST_YEAR, LAST_SIMULATED_YEAR + 1)
    paths = build_corporate_paths(
        years,
        np.full(len(years), 100.0),
        flat_profit_macro(),
        years,
        parameters=CorporateParameters(normal_return_share=0.0),
    )
    markdown = paths.equity_markdown[paths.years >= paths.enactment_year]
    assert markdown[0] == pytest.approx(100.0 / PROFIT_IN_FIRST_YEAR, rel=1e-9)
    assert np.allclose(markdown, markdown[0], rtol=1e-9)


def test_migration_lets_the_markdown_decay_toward_the_rent_floor():
    # With a normal-return share, part of the priced burden leaves corporate
    # equity as capital turns over, so the markdown falls from its enactment
    # value toward the rent-only floor and never below it.
    parameters = CorporateParameters()
    years = np.arange(FIRST_YEAR, LAST_SIMULATED_YEAR + 1)
    paths = build_corporate_paths(
        years,
        np.full(len(years), 100.0),
        flat_profit_macro(),
        years,
        parameters=parameters,
    )
    markdown = paths.equity_markdown[paths.years >= paths.enactment_year]

    assert np.all(np.diff(markdown) <= 0)
    assert markdown[0] > markdown[20]
    rent_floor = (
        (1.0 - parameters.normal_return_share) * 100.0 / PROFIT_IN_FIRST_YEAR
    )
    assert markdown[-1] > rent_floor
    # Once essentially all the capital has turned over, only the rent piece is
    # still priced in corporate equity.
    assert markdown[-1] == pytest.approx(rent_floor, rel=5e-3)
    assert markdown[0] < 100.0 / PROFIT_IN_FIRST_YEAR


def test_nothing_is_priced_before_the_change_is_announced():
    paths = make_paths(wedge_per_year=100.0, enactment_year=FIRST_YEAR + 3)
    before = paths.years < paths.enactment_year
    assert np.all(paths.equity_markdown[before] == 0.0)
    assert np.all(paths.payout_cut[before] == 0.0)
    for factor in (
        paths.dividend_factor,
        paths.interest_factor,
        paths.rent_factor,
        paths.pass_through_factor,
    ):
        assert np.all(factor[before] == 1.0)


def test_the_present_value_recursion_matches_a_plain_summation():
    paths = make_paths(wedge_per_year=100.0, enactment_year=FIRST_YEAR + 1)
    rate = paths.discount_rate
    n = len(paths.years)
    # Every present value is the value of everything from next year on, and the
    # last row holds only the growing perpetuity past the end of the grid.
    terminal = paths.priced_burden_present_value[-1]
    for i in (0, 5, 20, n - 40, n - 2):
        summed = terminal / (1.0 + rate) ** (n - 1 - i)
        for j in range(i + 1, n):
            summed += paths.priced_burden[j] / (1.0 + rate) ** (j - i)
        expected = 0.0 if paths.years[i] < paths.enactment_year else summed
        assert paths.priced_burden_present_value[i] == pytest.approx(
            expected, rel=1e-10, abs=1e-10
        )


def test_a_sunset_the_market_believes_makes_the_markdown_vanish():
    # The revenue change sunsets inside the file, so by the last live year
    # there is nothing left to price.
    paths = make_paths(
        wedge_per_year=100.0,
        enactment_year=FIRST_YEAR + 1,
        sunset_year=FIRST_YEAR + 5,
    )
    assert paths.equity_markdown[paths.row_index_for_year(FIRST_YEAR + 1)] > 0
    assert paths.equity_markdown[paths.row_index_for_year(FIRST_YEAR + 4)] == (
        pytest.approx(0.0)
    )
    # The cut to payouts stops with the statute too.
    assert paths.payout_cut[paths.row_index_for_year(FIRST_YEAR + 5)] == 0.0


def test_a_sunset_the_market_disbelieves_keeps_pricing_it():
    believed = make_paths(
        wedge_per_year=100.0,
        enactment_year=FIRST_YEAR + 1,
        sunset_year=FIRST_YEAR + 5,
    )
    disbelieved = make_paths(
        wedge_per_year=100.0,
        enactment_year=FIRST_YEAR + 1,
        sunset_year=FIRST_YEAR + 5,
        parameters=CorporateParameters(priced_as_permanent=True),
    )
    i = disbelieved.row_index_for_year(FIRST_YEAR + 4)
    assert disbelieved.equity_markdown[i] > 0.01
    assert believed.equity_markdown[i] == pytest.approx(0.0)
    # The cut to flows still follows the statute, not the market's belief.
    j = disbelieved.row_index_for_year(FIRST_YEAR + 5)
    assert disbelieved.payout_cut[j] == 0.0


# ---------------------------------------------------------------------------
# Channel 3 — the capital-gains quantity margin
# ---------------------------------------------------------------------------


def test_the_gains_quantity_margin_follows_the_payout_cut():
    parameters = CorporateParameters()
    paths = make_paths(wedge_per_year=100.0, enactment_year=FIRST_YEAR + 1)
    i = paths.row_index_for_year(FIRST_YEAR + 1)

    # Fewer buybacks: realized corporate gains scale with the payout cut at the
    # corporate share of realized gains.
    assert paths.gain_quantity_factor[i] == pytest.approx(
        1.0 + parameters.corporate_share_of_gains * paths.payout_cut[i]
    )
    assert paths.gain_quantity_factor[i] < 1.0


# ---------------------------------------------------------------------------
# Channel 4 — what the wealth model is handed
# ---------------------------------------------------------------------------


def test_the_four_factors_are_what_the_saving_shortfall_is_built_from():
    # The per-household change in income from outside the tax system,
    # corp_dY_exog, is these four factors less one applied to a household's
    # dividends, interest, net rent and pass-through income. The paths owe the
    # wealth model exactly these four numbers and nothing else, so the check is
    # that all four move together and in the same direction for a tax increase.
    paths = make_paths(wedge_per_year=100.0, enactment_year=FIRST_YEAR + 1)
    i = paths.row_index_for_year(FIRST_YEAR + 5)
    for factor in (
        paths.dividend_factor[i],
        paths.interest_factor[i],
        paths.rent_factor[i],
        paths.pass_through_factor[i],
    ):
        assert 0.0 < factor < 1.0


# ---------------------------------------------------------------------------
# Nothing in, nothing out
# ---------------------------------------------------------------------------


def test_a_zero_revenue_change_is_refused_rather_than_run():
    years = np.arange(FIRST_YEAR, LAST_SIMULATED_YEAR + 1)
    with pytest.raises(ValueError, match="zero in every year"):
        build_corporate_paths(
            years,
            np.zeros_like(years, dtype=float),
            make_macro(),
            years,
        )


# ---------------------------------------------------------------------------
# The guards the eligibility decision runs
# ---------------------------------------------------------------------------


def test_a_timing_provision_is_refused():
    # Bonus depreciation's shape: it costs money early and raises it later.
    years = np.arange(2026, 2031)
    wedge = np.array([-61.0, -40.0, -10.0, 20.0, 35.0])
    with pytest.raises(ValueError, match="runs both ways"):
        check_wedge_runs_one_way(wedge)
    # A rate change with a little rounding wobble is allowed through.
    check_wedge_runs_one_way(np.array([100.0, 101.0, -0.2, 99.0, 98.0]))
    assert len(years) == 5


def test_a_change_enacted_before_the_simulation_starts_is_refused():
    years = np.arange(2024, 2031)
    wedge = np.where(years >= 2025, 100.0, 0.0)
    with pytest.raises(ValueError, match="would be missed"):
        check_enactment_year_is_in_window(years, wedge, range(2026, 2036))
    check_enactment_year_is_in_window(years, wedge, range(2025, 2036))


def test_the_wedge_adds_the_on_model_rate_change_to_the_off_model_residual():
    years, wedge = assemble_corporate_wedge(
        [2026, 2027, 2028],
        [10.0, 10.0, 10.0],
        [2027, 2028, 2029],
        [50.0, 60.0, 70.0],
    )
    assert years.tolist() == [2026, 2027, 2028, 2029]
    assert wedge.tolist() == [10.0, 60.0, 70.0, 70.0]


# ---------------------------------------------------------------------------
# Inputs the builder refuses
# ---------------------------------------------------------------------------


def test_a_markdown_past_the_sanity_bound_stops_the_run():
    # A wedge of the same order as all corporate profit is a units error, not a
    # policy, and the bound is there to say so.
    with pytest.raises(ValueError, match="sanity bound"):
        make_paths(wedge_per_year=2_500.0, enactment_year=FIRST_YEAR)


def test_a_discount_rate_below_the_growth_rate_stops_the_run():
    years = np.arange(FIRST_YEAR, LAST_PROJECTED_YEAR + 1)
    ones = np.ones_like(years, dtype=float)
    # Profit growing at 12 percent a year against a 9 percent discount rate
    # makes the terminal growing perpetuity diverge.
    macro = MacroAggregates(
        years=years,
        after_tax_profit=PROFIT_IN_FIRST_YEAR * 1.12 ** (years - FIRST_YEAR),
        treasury_10_year_percent=ones * TREASURY_PERCENT,
        interest_income=ones * INTEREST_INCOME,
        rental_income=ones * RENTAL_INCOME,
        proprietors_income=ones * PROPRIETORS_INCOME,
    )
    wedge = np.where(years[:10] >= FIRST_YEAR, 100.0, 0.0)
    with pytest.raises(ValueError, match="diverges"):
        build_corporate_paths(
            years[:10],
            wedge,
            macro,
            range(FIRST_YEAR, LAST_SIMULATED_YEAR + 1),
        )


def test_too_few_years_of_profit_stops_the_run():
    years = np.arange(FIRST_YEAR, FIRST_YEAR + 4)
    ones = np.ones_like(years, dtype=float)
    macro = MacroAggregates(
        years=years,
        after_tax_profit=ones * PROFIT_IN_FIRST_YEAR,
        treasury_10_year_percent=ones * TREASURY_PERCENT,
        interest_income=ones * INTEREST_INCOME,
        rental_income=ones * RENTAL_INCOME,
        proprietors_income=ones * PROPRIETORS_INCOME,
    )
    with pytest.raises(ValueError, match="fewer than 6 years"):
        build_corporate_paths(
            years,
            np.full(len(years), 100.0),
            macro,
            years,
        )


# --------------------------------------------------------------------------
# Reading the channel's assumptions out of a run's economy leg
# --------------------------------------------------------------------------


def economy_config():
    from taxmodel.config.scenario import ScenarioConfig, load_defaults, resolve

    root = Path("config/scenarios")
    return ScenarioConfig(economy=resolve(root, load_defaults(root, "economy"), None))


def test_the_corporate_parameters_come_from_the_economy_leg():
    """Every number is the scenario tree's, with its own recorded provenance."""
    read = paths.corporate_parameters_from_economy_leg(economy_config())

    # The default layer's values, by their corp.yaml names.
    assert read.normal_return_share == 0.375  # sigma_n
    assert read.corporate_share_of_normal_capital == 0.40  # kappa
    assert read.depreciation_rate == 0.057  # delta_nipa
    assert read.corporate_share_of_dividends == 0.85  # omega_div
    assert read.corporate_share_of_gains == 0.50  # omega_kg
    assert read.priced_as_permanent is False

    # The exposures are keyed by the balance-sheet column they mark down.
    assert read.asset_exposure["value.equities"] == 1.00
    assert read.asset_exposure["value.dc"] == 0.55

    # The capital content of pass-through income is the wealth channel's
    # setting.
    assert read.pass_through_capital_weight == 0.20
