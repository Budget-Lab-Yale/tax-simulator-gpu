"""The derivative and next-dollar rerun of the tax outcomes."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from taxmodel.taxes import law_for_calculators, statutory_order
from taxmodel.taxes.individual_income_tax import (
    CONTINUOUS_TAX_INPUTS,
    DOLLAR_TAX_INPUTS,
    ESTATE_BASE_DOLLAR,
    GENERAL_PARTNER_DOLLAR,
    NONDIFFERENTIABLE_TAX_INPUTS,
    derivative,
    dollar_increment,
    income_tax_plus_employee_payroll,
    net_income_tax,
)

YEAR = 2027


def annual_law(compiled_baseline, number_of_records: int = 1):
    """Baseline law bound to single filers."""
    values, years = compiled_baseline
    return law_for_calculators.bind_year(values, years, YEAR).for_records(
        np.ones(number_of_records)
    )


def records(base_wages: np.ndarray | list[float]):
    """Single filers whose only income is the primary earner's ordinary pay.

    Ordinary pay means base wages: neither tips nor overtime, both of which the
    frame stores in columns of their own.
    """
    wage_array = jnp.asarray(base_wages, dtype=jnp.float64)
    number_of_records = wage_array.size
    frame = {
        name: jnp.zeros(number_of_records, dtype=jnp.float64)
        for name in (
            *statutory_order.RECORD_COLUMNS,
            *statutory_order.ESTATE_RECORD_COLUMNS,
            ESTATE_BASE_DOLLAR,
        )
    }
    frame["filing_status"] = jnp.ones(number_of_records)
    frame["filer"] = jnp.ones(number_of_records)
    frame["wages_base1"] = wage_array
    for name in (
        "age2",
        "blind2",
        "dep_age1",
        "dep_age2",
        "dep_age3",
        "divorce_year",
    ):
        frame[name] = jnp.full(number_of_records, jnp.nan)
    for name in ("r.bus_loss", "r.cdctc_takeup", "r.new_car", "r.eitc_precert"):
        frame[name] = jnp.full(number_of_records, 0.999)
    return frame


def ordinary_rate_on_the_next_dollar(law, taxable_income: float) -> float:
    """The statutory ordinary rate the next dollar of taxable income meets."""
    # These laws have been read for each household, so a rate or bracket
    # schedule is one row of slots per household. There is one household here,
    # and it is row 0.
    brackets = np.asarray(law.tax.ord_brackets)[0]
    rates = np.asarray(law.tax.ord_rates)[0]
    starts_at_or_below = np.nan_to_num(brackets, nan=np.inf) <= taxable_income
    return float(rates[np.flatnonzero(starts_at_or_below)[-1]])


def taxable_income_of(frame, law) -> float:
    """The one household's taxable income under this law."""
    return float(statutory_order.run_taxes(frame, law, double_run=False)["txbl_inc"][0])


def test_net_income_tax_is_the_calculators_finished_net_liability(compiled_baseline):
    frame = records([100_000.0])
    law = annual_law(compiled_baseline)

    expected = statutory_order.run_taxes(frame, law)["liab_iit_net"]

    np.testing.assert_array_equal(net_income_tax(frame, law), expected)


def test_the_second_outcome_adds_the_workers_own_payroll_tax(compiled_baseline):
    frame = records([100_000.0])
    law = annual_law(compiled_baseline)
    results = statutory_order.run_taxes(frame, law)

    np.testing.assert_array_equal(
        income_tax_plus_employee_payroll(frame, law),
        results["liab_iit_net"] + results["liab_pr_ee"],
    )
    # The employer's half exists but is not in the price the worker faces.
    assert float(results["liab_pr_er"][0]) > 0.0
    assert float(results["liab_pr_ee"][0]) > 0.0


def test_a_tip_dollar_and_an_ordinary_pay_dollar_are_priced_separately(
    compiled_baseline,
):
    """Tips live in their own frame column, so they get their own price."""
    law = annual_law(compiled_baseline)
    frame = records([40_000.0])
    frame["tips1"] = jnp.array([10_000.0])

    gradients = derivative(
        frame,
        law,
        outcome=net_income_tax,
        input_names=("wages_base1", "tips1"),
        double_run=False,
    )

    np.testing.assert_allclose(
        gradients["wages_base1"],
        ordinary_rate_on_the_next_dollar(law, taxable_income_of(frame, law)),
        atol=1e-12,
    )
    # Under 2027 baseline law the tip deduction cancels the income tax on a
    # tipped dollar outright, well inside its dollar limit and its phaseout.
    np.testing.assert_allclose(gradients["tips1"], 0.0, atol=1e-12)


def test_the_tipped_occupation_share_has_a_derivative_but_not_a_dollar(
    compiled_baseline,
):
    """`tips_lh1` is the share of an earner's tips that qualify, not an amount."""
    baseline = annual_law(compiled_baseline)
    # Baseline law lets every tip qualify, which leaves the share doing
    # nothing. Turn the occupation test on, then ask what a wider list of
    # qualifying occupations would be worth to this household.
    occupation_test = replace(
        baseline, below_ded=replace(baseline.below_ded, tip_ded_lh=1.0)
    )
    tips = 10_000.0
    qualifying_share = 0.5
    frame = records([40_000.0])
    frame["tips1"] = jnp.array([tips])
    frame["tips_lh1"] = jnp.array([qualifying_share])

    gradients = derivative(
        frame,
        occupation_test,
        outcome=net_income_tax,
        input_names=("tips1", "tips_lh1"),
        double_run=False,
    )

    statutory_rate = ordinary_rate_on_the_next_dollar(
        occupation_test, taxable_income_of(frame, occupation_test)
    )
    # Raising the share by one whole share -- half to one and a half, were that
    # possible -- would deduct another $10,000 of tips, worth the statutory
    # rate on every one of them. That is a rate per share, not per dollar.
    np.testing.assert_allclose(gradients["tips_lh1"], -statutory_rate * tips, atol=1e-9)
    # Half of the next tipped dollar is deductible, so it faces half the rate.
    np.testing.assert_allclose(
        gradients["tips1"], statutory_rate * (1.0 - qualifying_share), atol=1e-12
    )

    assert "tips_lh1" in CONTINUOUS_TAX_INPUTS
    assert "tips_lh1" not in DOLLAR_TAX_INPUTS
    with pytest.raises(ValueError, match="only dollar inputs"):
        dollar_increment(
            frame,
            occupation_test,
            {"tips_lh1": 1.0},
            outcome=net_income_tax,
            double_run=False,
        )


def test_every_continuous_input_arrives_from_one_reverse_pass(compiled_baseline):
    frame = records([100_000.0])
    law = annual_law(compiled_baseline)

    for outcome in (net_income_tax, income_tax_plus_employee_payroll):
        gradients = derivative(
            frame,
            law,
            outcome=outcome,
            input_names=CONTINUOUS_TAX_INPUTS,
            double_run=False,
        )

        assert tuple(gradients) == CONTINUOUS_TAX_INPUTS
        for name, values in gradients.items():
            assert values.shape == (1,), name
            assert np.isfinite(np.asarray(values)).all(), name


def test_every_calculator_input_has_a_derivative_classification():
    continuous = set(CONTINUOUS_TAX_INPUTS)
    nondifferentiable = set(NONDIFFERENTIABLE_TAX_INPUTS)

    assert continuous.isdisjoint(nondifferentiable)
    # Every column the calculators read: the chain's own list, plus what the
    # estate tax reads on top of it -- the balance sheet and the second
    # spouse's chance of already being dead.
    # `estate_base_bump` is the one continuous input that is not a column any
    # household carries: it is always zero, and it exists only so the estate
    # tax's rate on a dollar of its own taxable base can be measured by
    # differentiating with respect to it.
    assert continuous | nondifferentiable == set(statutory_order.RECORD_COLUMNS) | set(
        statutory_order.ESTATE_RECORD_COLUMNS
    ) | {ESTATE_BASE_DOLLAR}
    # The capital-gain price-index ratio and the two tipped-occupation shares
    # are the continuous inputs that are not measured in dollars.
    assert set(DOLLAR_TAX_INPUTS) == continuous - {
        "kg_lt_cpi_ratio",
        "tips_lh1",
        "tips_lh2",
    }


def test_an_arbitrary_bound_law_changes_the_derivative(compiled_baseline):
    frame = records([100_000.0])
    baseline = annual_law(compiled_baseline)
    flat_rate = 0.10
    flat_rates = jnp.where(jnp.isnan(baseline.tax.ord_rates), jnp.nan, flat_rate)
    alternative = replace(baseline, tax=replace(baseline.tax, ord_rates=flat_rates))
    statutory_rate = ordinary_rate_on_the_next_dollar(
        baseline, taxable_income_of(frame, baseline)
    )

    baseline_result = derivative(
        frame,
        baseline,
        outcome=net_income_tax,
        input_names=("wages_base1",),
        double_run=False,
    )["wages_base1"]
    alternative_result = derivative(
        frame,
        alternative,
        outcome=net_income_tax,
        input_names=("wages_base1",),
        double_run=False,
    )["wages_base1"]

    np.testing.assert_allclose(baseline_result, statutory_rate, atol=1e-12)
    np.testing.assert_allclose(alternative_result, flat_rate, atol=1e-12)


@pytest.mark.parametrize(
    "priced_column",
    ["sole_prop1", "part_active", "scorp_active"],
)
def test_the_business_income_deduction_reaches_the_next_business_dollar(
    compiled_baseline, priced_column
):
    """A dollar of active business income earns its section 199A deduction.

    The household here has $80,000 of pay and $40,000 of S corporation profit,
    so it already claims the deduction, and it has no partnership and no sole
    proprietorship. A dollar of any of the three is qualified business income
    all the same: the deduction grows by the deduction rate times that dollar,
    and the income tax on the dollar is the ordinary rate on the four fifths of
    it that remains taxable. The pay keeps the household's taxable income well
    clear of the limitation that caps the deduction at a fifth of taxable
    income, so the deduction on the next dollar is the flat rate.

    Both prices are checked against each other because they answer the question
    two different ways -- the derivative of the calculator, and running the
    calculator again a dollar higher. Before this was fixed the two parted
    company for exactly the kinds of business income the household held none
    of: the deduction's zero floor was spelled so that a household sitting on
    it stayed there, so the derivative reported the whole ordinary rate while
    the rerun reported four fifths of it.
    """
    law = annual_law(compiled_baseline)
    frame = records([80_000.0])
    frame["scorp_active"] = jnp.array([40_000.0])
    # No wage bill in any of the businesses. The household is far below the
    # phase-out threshold, so the wage-based limitation does not apply to it
    # and the deduction is the flat rate on qualified business income.
    deduction_rate = float(np.asarray(law.qbi_ded.rate)[0])

    # A sole proprietor's dollar is also self-employment earnings, so its
    # income-tax price carries the deduction for half the self-employment tax
    # as well; the partnership dollar moves both of its columns for the same
    # reason. The S corporation dollar carries neither.
    direction = {
        "sole_prop1": {"sole_prop1": 1.0},
        "part_active": dict(GENERAL_PARTNER_DOLLAR),
        "scorp_active": {"scorp_active": 1.0},
    }[priced_column]

    gradients = derivative(
        frame,
        law,
        outcome=net_income_tax,
        input_names=tuple(direction),
        double_run=False,
    )
    gradient = sum(
        moved * gradients[column] for column, moved in direction.items()
    )
    exact = dollar_increment(
        frame, law, direction, outcome=net_income_tax, double_run=False
    )
    np.testing.assert_allclose(gradient, exact, atol=1e-9)

    # And both of them are the deduction's doing: the rate on the dollar is the
    # ordinary rate on the part of it the deduction leaves taxable, which is
    # strictly below the rate the same dollar would meet with no deduction.
    taxable_income = taxable_income_of(frame, law)
    ordinary_rate = ordinary_rate_on_the_next_dollar(law, taxable_income)
    if priced_column == "scorp_active":
        # No self-employment tax on an S corporation shareholder's profit, so
        # the whole difference from the ordinary rate is the deduction.
        np.testing.assert_allclose(
            exact, ordinary_rate * (1.0 - deduction_rate), atol=1e-9
        )
    else:
        assert float(exact[0]) < ordinary_rate * (1.0 - deduction_rate / 2.0)


@pytest.mark.parametrize("column", ["part_se1", "part_se2"])
def test_self_employment_earnings_cannot_move_on_their_own(compiled_baseline, column):
    frame = records([100_000.0])
    law = annual_law(compiled_baseline)

    with pytest.raises(ValueError, match="cannot move without part_active"):
        dollar_increment(
            frame, law, {column: 1.0}, outcome=net_income_tax, double_run=False
        )

    # Paired with active partnership income it is a general partner's dollar,
    # which is allowed.
    dollar_increment(
        frame,
        law,
        {column: 1.0, "part_active": 1.0},
        outcome=net_income_tax,
        double_run=False,
    )


@pytest.mark.parametrize("name", ["filing_status", "age1", "r.new_car", "unknown"])
def test_discrete_and_unknown_inputs_are_refused(compiled_baseline, name):
    frame = records([100_000.0])
    law = annual_law(compiled_baseline)

    with pytest.raises(ValueError, match="continuous calculator inputs"):
        derivative(
            frame, law, outcome=net_income_tax, input_names=(name,), double_run=False
        )
    with pytest.raises(ValueError, match="only dollar inputs"):
        dollar_increment(
            frame, law, {name: 1.0}, outcome=net_income_tax, double_run=False
        )


def test_a_dollar_increment_refuses_the_capital_gain_price_index_ratio(
    compiled_baseline,
):
    frame = records([100_000.0])
    law = annual_law(compiled_baseline)

    with pytest.raises(ValueError, match="only dollar inputs"):
        dollar_increment(
            frame,
            law,
            {"kg_lt_cpi_ratio": 1.0},
            outcome=net_income_tax,
            double_run=False,
        )


@pytest.mark.parametrize(
    ("priced_column", "deductible_share_of_the_dollar"),
    # A dollar of tips is deductible in full. A dollar of overtime pay is
    # deductible only in the statutory premium part of it, which under current
    # law is a third of the overtime line.
    [("tips1", 1.0), ("ot1", 1.0 / 3.0)],
)
def test_the_tip_and_overtime_deductions_reach_the_next_such_dollar(
    compiled_baseline, priced_column, deductible_share_of_the_dollar
):
    """A worker with no tips still earns the tip deduction on the next dollar.

    The household here has $60,000 of ordinary pay and no tips and no overtime,
    which is most of the workforce. Its income is well under the threshold
    where the two deductions begin to phase out, so a dollar of tips is
    deductible in full and costs no income tax at all.

    The derivative used to report the whole ordinary rate on that dollar. The
    phase-out's zero floor was spelled so that a household sitting on it stayed
    there, and a household with no tips sits on it exactly: there is nothing to
    phase out and nothing left of it. The deduction is zero either way, so this
    is a change to the slope and not to anyone's tax.
    """
    law = annual_law(compiled_baseline)
    frame = records([60_000.0])
    direction = {priced_column: 1.0}

    gradients = derivative(
        frame,
        law,
        outcome=net_income_tax,
        input_names=(priced_column,),
        double_run=False,
    )
    exact = dollar_increment(
        frame, law, direction, outcome=net_income_tax, double_run=False
    )
    np.testing.assert_allclose(gradients[priced_column], exact, atol=1e-9)

    # The deduction is what makes the rate what it is: the dollar is taxed at
    # the ordinary rate on whatever part of it the deduction does not cover.
    taxable_income = taxable_income_of(frame, law)
    ordinary_rate = ordinary_rate_on_the_next_dollar(law, taxable_income)
    np.testing.assert_allclose(
        exact,
        ordinary_rate * (1.0 - deductible_share_of_the_dollar),
        atol=1e-9,
    )


def test_the_tip_deduction_phase_out_still_ends_flat(compiled_baseline):
    """A worker whose tip deduction is entirely phased out gains nothing more.

    Once income has taken the whole deduction away, another dollar of tips
    brings back none of it, so the rate on that dollar is the plain ordinary
    rate. This is the other household sitting on the same zero floor, and it
    belongs on the flat branch.
    """
    law = annual_law(compiled_baseline)
    threshold = float(np.asarray(law.below_ded.tip_ded_po_thresh)[0])
    phase_out_rate = float(np.asarray(law.below_ded.tip_ded_po_rate)[0])
    limit = float(np.asarray(law.below_ded.tip_ded_limit)[0])
    # Enough income above the phase-out threshold to remove the whole
    # deduction, and tips at the deduction's own dollar limit.
    frame = records([threshold + limit / phase_out_rate + 100_000.0])
    frame["tips1"] = jnp.array([limit])

    gradients = derivative(
        frame,
        law,
        outcome=net_income_tax,
        input_names=("tips1",),
        double_run=False,
    )
    exact = dollar_increment(
        frame, law, {"tips1": 1.0}, outcome=net_income_tax, double_run=False
    )
    np.testing.assert_allclose(gradients["tips1"], exact, atol=1e-9)
    assert float(np.asarray(exact)[0]) > 0.3


def test_the_auto_loan_interest_deduction_reaches_the_next_such_dollar(
    compiled_baseline,
):
    """A household with no car-loan interest earns the deduction on the next
    dollar of it.

    Same zero floor as the tip and overtime deductions, reached through the
    auto-loan-interest deduction's own phase-out. The household's income is
    below that phase-out, and the deduction reduces adjusted gross income, so
    the dollar's price is the negative of the rate the deduction is worth.
    """
    law = annual_law(compiled_baseline)
    frame = records([60_000.0])
    # The new-car share is a calibrated draw; this household is a new-car
    # buyer, so the deduction is available to it.
    frame["r.new_car"] = jnp.array([0.0])

    gradients = derivative(
        frame,
        law,
        outcome=net_income_tax,
        input_names=("auto_int_exp",),
        double_run=False,
    )
    exact = dollar_increment(
        frame,
        law,
        {"auto_int_exp": 1.0},
        outcome=net_income_tax,
        double_run=False,
    )
    np.testing.assert_allclose(gradients["auto_int_exp"], exact, atol=1e-9)
    assert float(np.asarray(exact)[0]) < -0.1
