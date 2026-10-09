"""The marginal tax rates a production run stores for every household."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from taxmodel.taxes import (
    individual_income_tax,
    law_for_calculators,
    marginal_rates,
    statutory_order,
)
from taxmodel.taxes.calc import wealth

from .conftest import (
    ESTATE_MEASUREMENT,
    compiled_calculators,
    law_for,
    synthetic_records,
)


def test_the_stored_columns_say_which_law_and_which_taxes():
    columns = marginal_rates.MARGINAL_RATE_COLUMNS
    # Each priced dollar, under each price, under each of the two laws.
    assert len(columns) == len(marginal_rates.PRICED_DOLLARS) * len(marginal_rates.PRICES) * 2
    assert len(set(columns)) == len(columns)
    assert "baseline_law_mtr_kg_lt" in columns
    assert "reform_law_mtr_kg_lt" in columns
    assert "baseline_law_mtr_wages_base1_plus_employee_payroll" in columns
    assert "reform_law_mtr_char_cash_plus_employee_payroll" in columns
    # Every column names one of the two laws and one of the priced dollars.
    for column in columns:
        assert column.startswith(("baseline_law_mtr_", "reform_law_mtr_"))


def test_every_priced_dollar_moves_columns_the_frame_actually_has():
    for dollar, direction in marginal_rates.PRICED_DOLLARS.items():
        for column in direction:
            assert column in individual_income_tax.DOLLAR_TAX_INPUTS, f"{dollar}: {column}"


def test_only_the_general_partners_dollar_moves_more_than_one_column():
    """Every other priced dollar is one frame column, named for it.

    The average-marginal-rate table weights each rate by the column of the
    same name. The one exception is described next to
    `GENERAL_PARTNER_DOLLAR`.
    """
    for dollar, direction in marginal_rates.PRICED_DOLLARS.items():
        if dollar == "part_active":
            assert dict(direction) == {"part_active": 1.0, "part_se1": 1.0}
            continue
        assert dict(direction) == {dollar: 1.0}, dollar


# A joint return with a dollar in every one of the twenty-seven priced columns,
# chosen to sit away from the thresholds each of them can meet. It has enough
# income to be past the earned income tax credit and past the phase-outs of the
# tip, overtime and car-loan-interest deductions; enough medical expense to be
# clear of the floor that a share of adjusted gross income sets; and a
# state-and-local tax total under the cap. That is what makes the derivative and
# the next dollar the same number below: a household sitting exactly on one of
# those corners would legitimately give two different answers.
HOUSEHOLD_AWAY_FROM_KINKS = {
    "wages_base1": 120_000.0,
    "wages_base2": 60_000.0,
    "tips1": 5_000.0,
    "tips2": 3_000.0,
    "ot1": 4_000.0,
    "ot2": 2_000.0,
    "part_active": 30_000.0,
    "part_se1": 30_000.0,
    "part_passive": 20_000.0,
    "sole_prop1": 25_000.0,
    "sole_prop2": 15_000.0,
    "scorp_active": 40_000.0,
    "scorp_passive": 10_000.0,
    "rent": 12_000.0,
    "txbl_int": 8_000.0,
    "div_pref": 9_000.0,
    "kg_lt": 50_000.0,
    "kg_st": 7_000.0,
    "gross_ss": 20_000.0,
    "ui": 6_000.0,
    "med_exp": 60_000.0,
    "salt_inc_sales": 15_000.0,
    "salt_prop": 12_000.0,
    "salt_pers": 3_000.0,
    "first_mort_int": 25_000.0,
    "auto_int_exp": 3_000.0,
    "char_cash": 20_000.0,
    "char_noncash": 10_000.0,
}


def test_each_of_the_twenty_seven_rates_matches_adding_the_dollar(compiled_baseline):
    """The derivative and the next dollar agree, priced dollar by priced dollar.

    Every rate stored here comes out of the reverse pass. Adding a dollar to the
    column and rerunning the calculator is an independent measurement of the
    same thing, and away from a threshold the two are the same number. A rate
    that came back zero because nothing reads its column would fail here.

    A cent per dollar of tolerance, because the two measurements are not
    algebraically identical: the derivative is taken at the household's own
    income and the next dollar averages the law over a dollar's width.
    """
    law = law_for(compiled_baseline, [2])
    records = synthetic_records([2], **HOUSEHOLD_AWAY_FROM_KINKS)
    rates = marginal_rates.marginal_rates(records, law, ESTATE_MEASUREMENT)

    for price in marginal_rates.PRICES:
        ending, outcome = price.ending, price.outcome
        for dollar, direction in marginal_rates.PRICED_DOLLARS.items():
            next_dollar = individual_income_tax.dollar_increment(
                records, law, direction, outcome=outcome
            )
            np.testing.assert_allclose(
                np.asarray(rates[f"mtr_{dollar}{ending}"]),
                np.asarray(next_dollar),
                atol=0.01,
                err_msg=f"mtr_{dollar}{ending}",
            )


def test_the_new_rates_are_not_all_the_same_number(compiled_baseline):
    """Four kinds of dollar the law treats differently get different rates.

    The four are ordinary income, a
    preferentially taxed dividend, a deduction (whose price is negative), and
    Social Security benefits (only part of which is included in income, so the
    price is a fraction of the ordinary rate).
    """
    law = law_for(compiled_baseline, [2])
    records = synthetic_records([2], **HOUSEHOLD_AWAY_FROM_KINKS)
    rates = marginal_rates.marginal_rates(records, law, ESTATE_MEASUREMENT)
    ordinary = float(np.asarray(rates["mtr_txbl_int"])[0])
    dividend = float(np.asarray(rates["mtr_div_pref"])[0])
    deduction = float(np.asarray(rates["mtr_salt_prop"])[0])
    benefits = float(np.asarray(rates["mtr_gross_ss"])[0])

    assert ordinary > 0.0
    assert 0.0 < dividend < ordinary
    assert deduction < 0.0
    assert 0.0 < benefits < ordinary


def test_the_wage_rates_are_the_bracket_rate_and_the_bracket_rate_plus_payroll(
    compiled_baseline,
):
    # A single filer with forty thousand dollars of ordinary pay, no
    # dependents and nothing else. That is below the Social Security taxable
    # maximum and below the additional Medicare tax threshold, so the payroll
    # price on the next dollar is just the two flat employee rates, and the
    # income tax price is the ordinary rate the dollar meets.
    wages = 40_000.0
    law = law_for(compiled_baseline, [1])
    records = synthetic_records([1], wages_base1=wages)
    # Every law parameter is one row per household, so the household is the
    # first index and the bracket slot the second.
    assert wages < float(np.asarray(law.payroll.oasdi_ee_brackets)[0, 1])
    assert wages < float(np.asarray(law.payroll.add_med_brackets)[0, 1])

    taxable_income = float(statutory_order.run_taxes(records, law)["txbl_inc"][0])
    brackets = np.nan_to_num(np.asarray(law.tax.ord_brackets)[0], nan=np.inf)
    statutory_rate = float(
        np.asarray(law.tax.ord_rates)[0, np.flatnonzero(brackets <= taxable_income)[-1]]
    )
    employee_payroll_rate = float(
        np.asarray(law.payroll.oasdi_ee_rates)[0, 0]
    ) + float(np.asarray(law.payroll.hi_ee_rates)[0, 0])

    rates = marginal_rates.marginal_rates(records, law, ESTATE_MEASUREMENT)
    income_tax_rate = float(np.asarray(rates["mtr_wages_base1"])[0])
    assert income_tax_rate == pytest.approx(statutory_rate)
    # The worker's own payroll tax is on top of that, and nothing else moves:
    # the employer's half is not in this price.
    with_payroll = float(np.asarray(rates["mtr_wages_base1_plus_employee_payroll"])[0])
    assert with_payroll - income_tax_rate == pytest.approx(employee_payroll_rate)


def test_the_general_partner_dollar_is_priced_in_both_of_its_bases(
    compiled_baseline,
):
    # A dollar earned by a general partner who works in the partnership is
    # ordinary partnership income and self-employment earnings at once, so its
    # price is the sum of the two partial derivatives. Nothing else is.
    law = law_for(compiled_baseline, [1])
    records = synthetic_records([1], part_active=80_000.0, part_se1=80_000.0)
    rates = marginal_rates.marginal_rates(records, law, ESTATE_MEASUREMENT)
    gradient = individual_income_tax.derivative(
        records,
        law,
        outcome=individual_income_tax.net_income_tax,
        input_names=("part_active", "part_se1", "sole_prop1"),
    )
    assert float(np.asarray(rates["mtr_part_active"])[0]) == pytest.approx(
        float(np.asarray(gradient["part_active"])[0])
        + float(np.asarray(gradient["part_se1"])[0])
    )
    assert float(np.asarray(rates["mtr_sole_prop1"])[0]) == pytest.approx(
        float(np.asarray(gradient["sole_prop1"])[0])
    )


def test_a_household_is_priced_on_its_own_income_and_no_one_elses(
    compiled_baseline,
):
    # The gradient is taken over the whole population's summed liability in
    # one reverse pass. That is only legitimate because the calculator is
    # rowwise, so pricing four households together has to give what pricing
    # each of them alone gives.
    filing_status = [1, 2, 3, 4]
    together = marginal_rates.marginal_rates(
        synthetic_records(
            filing_status,
            wages_base1=jnp.array([30_000.0, 250_000.0, 55_000.0, 70_000.0]),
            kg_lt=jnp.array([0.0, 100_000.0, 5_000.0, 0.0]),
        ),
        law_for(compiled_baseline, filing_status),
        ESTATE_MEASUREMENT,
    )
    for position, status in enumerate(filing_status):
        alone = marginal_rates.marginal_rates(
            synthetic_records(
                [status],
                wages_base1=[30_000.0, 250_000.0, 55_000.0, 70_000.0][position],
                kg_lt=[0.0, 100_000.0, 5_000.0, 0.0][position],
            ),
            law_for(compiled_baseline, [status]),
            ESTATE_MEASUREMENT,
        )
        for name, value in alone.items():
            assert float(np.asarray(value)[0]) == pytest.approx(
                float(np.asarray(together[name])[position])
            ), f"{name}, filing status {status}"


def test_compiling_the_gradient_changes_no_rate(compiled_baseline):
    filing_status = [1, 2]
    law = law_for(compiled_baseline, filing_status)
    records = synthetic_records(
        filing_status,
        wages_base1=jnp.array([45_000.0, 180_000.0]),
        char_cash=jnp.array([0.0, 20_000.0]),
        rent=jnp.array([0.0, 12_000.0]),
    )
    eager = marginal_rates.marginal_rates(records, law, ESTATE_MEASUREMENT)
    _taxes, prices, wrap = compiled_calculators(law, ESTATE_MEASUREMENT)
    compiled = prices(records, wrap(law))
    for name, value in eager.items():
        np.testing.assert_allclose(
            np.asarray(compiled[name]), np.asarray(value), err_msg=name
        )


def test_the_gradient_costs_no_more_compiles_than_the_calculator_does(
    compiled_baseline,
):
    # A compiled graph is retraced when the shape of an argument changes, and
    # a bound law presents the same shapes in every year, so one trace serves a
    # whole run. What matters here is that differentiating the chain costs
    # nothing on top of that: the gradient traces exactly as often as the chain
    # does, and every year runs on a graph one of them already built.
    values, years = compiled_baseline
    run_years = (2025, 2026, 2027)
    filing_status = np.array([1, 2])
    records = synthetic_records(filing_status, wages_base1=90_000.0)
    first_law = law_for_calculators.bind_year(values, years, run_years[0]).for_records(
        filing_status
    )
    taxes, rates, wrap = compiled_calculators(first_law, ESTATE_MEASUREMENT)

    for year in run_years:
        law = law_for_calculators.bind_year(values, years, year).for_records(filing_status)
        taxes(records, wrap(law))
        # Every rate is a real number for every household in every year.
        for name, value in rates(records, wrap(law)).items():
            assert np.isfinite(np.asarray(value)).all(), f"{year} {name}"

    assert rates._cache_size() == taxes._cache_size()
    assert rates._cache_size() < len(run_years)


def test_labelling_rates_with_a_law_that_is_neither_side_is_refused():
    with pytest.raises(ValueError, match="baseline_law or reform_law"):
        marginal_rates.under_law({"mtr_kg_lt": jnp.zeros(1)}, "some_other_law")


# -----------------------------------------------------------------------------
# The wealth tax and the estate tax, priced per balance-sheet column
# -----------------------------------------------------------------------------

# The three synthetic households the balance-sheet tests below run on. Each is a
# single filer with a hundred thousand dollars of ordinary pay. Their balance
# sheets are two and a half million, seven and a third million, and fifty-one
# million dollars of net worth: publicly marketable equities, a closely held
# business, cash, and a mortgage against the first two.
BALANCE_SHEETS = {
    "value.equities": [2_000_000.0, 5_000_000.0, 40_000_000.0],
    "value.pass_throughs": [500_000.0, 3_000_000.0, 10_000_000.0],
    "value.cash": [100_000.0, 250_000.0, 1_000_000.0],
    "value.primary_mortgage": [400_000.0, 900_000.0, 0.0],
}


def three_households_with_balance_sheets():
    records = synthetic_records([1, 1, 1], wages_base1=100_000.0)
    for name, values in BALANCE_SHEETS.items():
        records[name] = jnp.asarray(values)
    records["net_worth"] = sum(
        jnp.asarray(BALANCE_SHEETS[name])
        for name in BALANCE_SHEETS
        if name != "value.primary_mortgage"
    ) - jnp.asarray(BALANCE_SHEETS["value.primary_mortgage"])
    return records


def test_the_wealth_tax_rate_matches_adding_a_dollar_to_the_asset(
    compiled_baseline,
):
    """The gradient on each balance-sheet column against the next-dollar rate.

    A wealth tax is a graduated schedule on net worth, so away from a bracket
    edge the derivative with respect to one more dollar of an asset and the
    change in tax from actually adding that dollar are the same number. This
    checks the two agree for every asset and debt column on three households,
    which is what says the derivative is reaching the balance sheet at all
    rather than coming back zero because the tax reads a column that was added
    up before the calculation started.

    Baseline law has no wealth tax, so this runs on a reform that does: a flat
    two percent above fifty million dollars, which the third household is over
    and the first two are under.
    """
    law = law_for(compiled_baseline, [1, 1, 1])
    wealth_tax_law = wealth.WealthLaw(
        # A wealth tax exempting the first fifty million and charging two
        # percent above it. Written here rather than read off a law folder
        # because this test is about the derivative, not about any reform.
        brackets=jnp.asarray([[0.0, 50_000_000.0]] * 3),
        rates=jnp.asarray([0.0, 0.02]),
    )
    law = dataclasses.replace(law, wealth=wealth_tax_law)
    records = three_households_with_balance_sheets()

    gradient = individual_income_tax.derivative(
        records,
        law,
        outcome=individual_income_tax.wealth_tax,
        input_names=individual_income_tax.BALANCE_SHEET_INPUTS,
        double_run=False,
    )
    for column in individual_income_tax.BALANCE_SHEET_INPUTS:
        next_dollar = individual_income_tax.dollar_increment(
            records,
            law,
            {column: 1.0},
            outcome=individual_income_tax.wealth_tax,
            double_run=False,
        )
        np.testing.assert_allclose(
            np.asarray(gradient[column]),
            np.asarray(next_dollar),
            atol=1e-9,
            err_msg=column,
        )

    # And the rate really is the schedule's: nothing for the two households
    # under the exemption, two cents per dollar of asset for the one above it,
    # and minus two cents per dollar of debt.
    np.testing.assert_allclose(
        np.asarray(gradient["value.equities"]), [0.0, 0.0, 0.02], atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(gradient["value.primary_mortgage"]), [0.0, 0.0, -0.02], atol=1e-12
    )


def test_pricing_the_balance_sheet_leaves_the_income_rates_alone(
    compiled_baseline,
):
    """Income rates stored beside the balance-sheet rates equal the derivative
    taken on its own.

    The households hold balance sheets, and the largest owes estate tax, so an
    estate gradient that leaked into the income-tax passes would show here.
    """
    law = law_for(compiled_baseline, [1, 1, 1])
    records = three_households_with_balance_sheets()
    records = dict(records)
    records["kg_lt"] = jnp.asarray([10_000.0, 250_000.0, 3_000_000.0])
    records["rent"] = jnp.asarray([0.0, 40_000.0, 120_000.0])
    records["char_cash"] = jnp.asarray([1_000.0, 25_000.0, 400_000.0])

    stored = marginal_rates.marginal_rates(records, law, ESTATE_MEASUREMENT)
    for price in marginal_rates.PRICES:
        ending, outcome = price.ending, price.outcome
        gradient = individual_income_tax.derivative(
            records,
            law,
            outcome=outcome,
            input_names=("wages_base1", "kg_lt", "rent", "char_cash", "sole_prop1"),
        )
        for dollar in ("wages_base1", "kg_lt", "rent", "char_cash", "sole_prop1"):
            np.testing.assert_allclose(
                np.asarray(stored[f"mtr_{dollar}{ending}"]),
                np.asarray(gradient[dollar]),
                err_msg=f"mtr_{dollar}{ending}",
            )

    # The estate rate is stored under a name of its own.
    assert float(np.asarray(stored["mtr_estate_tax_on_value.equities"])[2]) > 0.0
    assert set(marginal_rates.MARGINAL_RATE_COLUMNS).isdisjoint(
        marginal_rates.BALANCE_SHEET_RATE_COLUMNS
    )


def test_the_estate_rate_is_positive_on_an_estate_above_the_exemption(
    compiled_baseline,
):
    # A sanity floor on the estate gradient. The third household's fifty-one
    # million dollar balance sheet is well above the exemption, so it pays more
    # estate tax on one more dollar of equities and less on one more dollar of
    # mortgage debt. The first two are below the exemption and owe nothing
    # either way, so their rate is zero.
    law = law_for(compiled_baseline, [1, 1, 1])
    records = three_households_with_balance_sheets()
    rates = marginal_rates.marginal_rates(records, law, ESTATE_MEASUREMENT)

    on_equities = np.asarray(rates["mtr_estate_tax_on_value.equities"])
    on_mortgage = np.asarray(rates["mtr_estate_tax_on_value.primary_mortgage"])
    assert on_equities[2] > 0.0
    assert on_mortgage[2] < 0.0
    np.testing.assert_array_equal(on_equities[:2], [0.0, 0.0])
    np.testing.assert_array_equal(on_mortgage[:2], [0.0, 0.0])


def test_the_estate_rate_on_the_taxable_base_carries_none_of_the_bridge(
    compiled_baseline,
):
    """The rate on the taxable estate excludes the SOI bridge; the rate on an
    asset includes it.

    The rate on a dollar of the taxable estate is the statutory next-dollar
    rate: zero for a household under the exemption, and the top rate of the
    schedule for one well over it, with none of the SOI measurement bridge in
    it. The rate on a dollar of an asset is that same rate times what the bridge
    does to that asset class, so a household holding nothing but closely held
    pass-through wealth prices a dollar of it at the base rate times the
    pass-through valuation rate times what survives the deduction share.

    The bridge used here reports 90 cents of every economic dollar, values a
    closely held dollar at 60 cents of that, and deducts a tenth of the reported
    estate -- deliberately different numbers from the frozen bridge's, so that
    an accidental use of the household's own asset rate cannot pass.
    """
    bridge = dataclasses.replace(
        ESTATE_MEASUREMENT,
        reporting_rate=0.90,
        pass_through_valuation_rate=0.60,
    )
    deduction_share = float(np.asarray(bridge.deduction_shares)[0])

    law = law_for(compiled_baseline, [1, 1])
    # Two single filers holding nothing but a closely held business: one with
    # two million dollars, far under the exemption, and one with three hundred
    # million, far over the top of the rate schedule.
    records = synthetic_records([1, 1], wages_base1=0.0)
    records["value.pass_throughs"] = jnp.asarray([2_000_000.0, 300_000_000.0])
    records["net_worth"] = records["value.pass_throughs"]

    rates = marginal_rates.marginal_rates(records, law, bridge)
    on_the_base = np.asarray(rates[marginal_rates.ESTATE_BASE_RATE])
    on_the_asset = np.asarray(rates["mtr_estate_tax_on_value.pass_throughs"])

    top_statutory_rate = float(np.nanmax(np.asarray(law.estate.rates)[0]))
    assert on_the_base[0] == 0.0
    assert on_the_base[1] == pytest.approx(top_statutory_rate)

    # The per-asset rate is the base rate after the bridge has shaved the
    # dollar. An economic dollar of closely held business is reported at ninety
    # cents and valued at sixty cents of that; a tenth of the reported estate is
    # deducted; and because this bridge sets the surviving spouse's unused
    # exclusion at a quarter of the reported estate, elected two times in five,
    # a larger reported estate also brings a larger exclusion with it.
    dsue_probability = float(np.asarray(bridge.dsue_probabilities)[0])
    dsue_share = float(np.asarray(bridge.dsue_shares)[0])
    shaving = (
        bridge.reporting_rate
        * bridge.pass_through_valuation_rate
        * (1.0 - deduction_share - dsue_probability * dsue_share)
    )
    assert on_the_asset[1] == pytest.approx(top_statutory_rate * shaving, rel=1e-6)
    assert on_the_asset[1] < on_the_base[1]


def test_the_estate_base_rate_is_stored_alongside_the_per_asset_rates():
    for law_side in (marginal_rates.BASELINE_LAW, marginal_rates.REFORM_LAW):
        assert (
            f"{law_side}_{marginal_rates.ESTATE_BASE_RATE}"
            in marginal_rates.balance_sheet_rate_columns(law_side)
        )
        assert (
            f"{law_side}_{marginal_rates.ESTATE_BASE_RATE}"
            in marginal_rates.ALL_RATE_COLUMNS
        )
    # It is not one of the priced dollars: no household holds any of it, so
    # there is nothing for the average-marginal-rate table to weight it by.
    assert marginal_rates.ESTATE_BASE_RATE not in marginal_rates.MARGINAL_RATE_COLUMNS


def test_pricing_the_estate_tax_without_the_measurement_bridge_is_refused(
    compiled_baseline,
):
    law = law_for(compiled_baseline, [1])
    with pytest.raises(ValueError, match="estate measurement bridge"):
        marginal_rates.marginal_rates(synthetic_records([1]), law, None)
