"""Income adjustments on records whose dollars can be checked by hand."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from taxmodel.taxes.calc.payroll import PayrollLaw, calc_pr
from taxmodel.years import wealth_arithmetic, wealth_state
from taxmodel.years.income_adjustments import (
    CORPORATE_ZERO_TOLERANCE,
    EMPLOYER_PAYROLL_CASH_WAGE_SHARE,
    apply_corporate_adjustment,
    apply_corporate_gain_quantity_margin,
    apply_employer_payroll_adjustment,
    record_corporate_markdown_gains,
)
from tests.tolerances import CENT


def payroll_law_with_no_employer_tax() -> PayrollLaw:
    """A made-up law with no payroll rates, for round-dollar wage tests."""
    return PayrollLaw(
        tips_exempt=0.0,
        ot_exempt=0.0,
        seca_taxable_rate=0.9235,
        se_thresh=400.0,
        oasdi_ee_rates=jnp.array([0.0]),
        oasdi_ee_brackets=jnp.array([0.0]),
        oasdi_er_rates=jnp.array([0.0]),
        oasdi_er_brackets=jnp.array([0.0]),
        hi_ee_rates=jnp.array([0.0]),
        hi_ee_brackets=jnp.array([0.0]),
        hi_er_rates=jnp.array([0.0]),
        hi_er_brackets=jnp.array([0.0]),
        add_med_rates=jnp.array([0.0]),
        add_med_brackets=jnp.array([0.0]),
    )


def payroll_records() -> dict[str, jnp.ndarray]:
    """The columns the payroll calculator reads, for two households."""
    values = {
        "wages_base1": [80.0, 0.0],
        "wages_base2": [40.0, 0.0],
        "tips1": [10.0, 0.0],
        "tips2": [0.0, 0.0],
        "ot1": [10.0, 0.0],
        "ot2": [10.0, 0.0],
        "trad_contr_er1": [0.0, 0.0],
        "trad_contr_er2": [0.0, 0.0],
        "sole_prop1": [0.0, 0.0],
        "sole_prop2": [0.0, 0.0],
        "farm1": [0.0, 0.0],
        "farm2": [0.0, 0.0],
        "part_se1": [0.0, 0.0],
        "part_se2": [0.0, 0.0],
    }
    return {name: jnp.array(value) for name, value in values.items()}


def corporate_records() -> dict[str, jnp.ndarray]:
    values = {
        "div_ord": [100.0, 0.0],
        "div_pref": [0.0, 0.0],
        "txbl_int": [50.0, 0.0],
        "exempt_int": [0.0, 0.0],
        "rent": [50.0, 0.0],
        "rent_loss": [10.0, 0.0],
        "sole_prop1": [80.0, 0.0],
        "sole_prop2": [0.0, 0.0],
        "part_active": [30.0, 0.0],
        "part_passive": [0.0, 0.0],
        "part_active_loss": [10.0, 0.0],
        "part_passive_loss": [0.0, 0.0],
        "part_179": [0.0, 0.0],
        "scorp_active": [0.0, 0.0],
        "scorp_passive": [0.0, 0.0],
        "scorp_active_loss": [0.0, 0.0],
        "scorp_passive_loss": [0.0, 0.0],
        "scorp_179": [0.0, 0.0],
        "farm1": [0.0, 0.0],
        "farm2": [0.0, 0.0],
        "part_se1": [20.0, 0.0],
        "part_se2": [0.0, 0.0],
        "value.equities": [1_000.0, 0.0],
        "value.dc": [200.0, 0.0],
        "value.db": [200.0, 0.0],
        "value.primary_mortgage": [100.0, 0.0],
        "txbl_ira_dist": [100.0, 0.0],
        "txbl_pens_dist": [100.0, 0.0],
        "gross_pens_dist": [100.0, 0.0],
        "kg_lt": [100.0, 0.0],
        "kg_lt_basis": [100.0, 0.0],
        "kg_st": [50.0, 0.0],
        "net_worth": [1_300.0, 0.0],
    }
    return {name: jnp.array(value) for name, value in values.items()}


def corporate_adjustment(records):
    return apply_corporate_adjustment(
        records,
        dividend_factor=0.9,
        interest_factor=0.8,
        rent_factor=0.5,
        pass_through_factor=0.75,
        markdown_rate=0.1,
        flow_factor=-0.2,
        asset_exposure={"value.equities": 1.0, "value.dc": 0.5},
        corporate_gain_share=0.5,
        gain_dynamics_active=False,
    )


def test_corporate_adjustment_changes_flows_stocks_and_outside_income():
    result = corporate_adjustment(corporate_records())

    # Outside income falls by $10 of dividends, $10 of interest, $20 of net
    # rent, and $25 of pass-through income.
    np.testing.assert_allclose(result["corp_dY_exog"], [-65.0, 0.0])
    np.testing.assert_allclose(result["div_ord"], [90.0, 0.0])
    np.testing.assert_allclose(result["rent_loss"], [5.0, 0.0])
    np.testing.assert_allclose(result["sole_prop1"], [60.0, 0.0])
    np.testing.assert_allclose(result["part_se1"], [15.0, 0.0])

    # Corporate equity falls by $100 and defined-contribution assets by $10.
    # Defined-benefit assets and $100 of mortgage debt do not move.
    np.testing.assert_allclose(result["value.equities"], [900.0, 0.0])
    np.testing.assert_allclose(result["value.dc"], [190.0, 0.0])
    np.testing.assert_allclose(result["net_worth"], [1_190.0, 0.0])
    np.testing.assert_allclose(result["corp_markdown"], [110.0 / 1_400.0, 0.0])
    np.testing.assert_allclose(result["corp_flow_factor"], [-0.2, -0.2])

    # IRA draws follow the defined-contribution markdown. Half of this record's
    # combined pension balance is defined contribution, so pension draws fall
    # by half as much.
    np.testing.assert_allclose(result["txbl_ira_dist"], [95.0, 0.0])
    np.testing.assert_allclose(result["txbl_pens_dist"], [97.5, 0.0])
    np.testing.assert_allclose(result["gross_pens_dist"], [97.5, 0.0])

    # Sale quantity falls 10 percent. Long-term gains also lose $10 from the
    # payout change and $10 from the lower sale price.
    np.testing.assert_allclose(result["kg_lt"], [80.0, 0.0])
    np.testing.assert_allclose(result["kg_lt_basis"], [90.0, 0.0])
    np.testing.assert_allclose(result["kg_st"], [45.0, 0.0])


def test_corporate_adjustment_records_the_retirement_and_gains_changes():
    records = corporate_records()
    result = corporate_adjustment(records)

    # The IRA draw falls $5 and the gross pension draw $2.50.
    np.testing.assert_allclose(result["corp_dY_retirement"], [-7.5, 0.0], atol=CENT)
    np.testing.assert_allclose(
        result["corp_dY_retirement"],
        (result["txbl_ira_dist"] - records["txbl_ira_dist"])
        + (result["gross_pens_dist"] - records["gross_pens_dist"]),
        atol=CENT,
    )
    # Haig-Simons income counts the defined-benefit share of the pension
    # change, read off the marked-down balances: $200 of $190 + $200.
    np.testing.assert_allclose(
        result["corp_dY_retirement_accrual"],
        [-2.5 * 200.0 / 390.0, 0.0],
        atol=CENT,
    )
    # Long-term gains fall $20 and short-term gains $5.
    np.testing.assert_allclose(result["corp_dY_gains"], [-25.0, 0.0], atol=CENT)


def test_corporate_adjustment_leaves_gains_for_the_gains_model():
    records = corporate_records()
    result = apply_corporate_adjustment(
        records,
        dividend_factor=1.0,
        interest_factor=1.0,
        rent_factor=1.0,
        pass_through_factor=1.0,
        markdown_rate=0.1,
        flow_factor=-0.2,
        asset_exposure={"value.equities": 1.0, "value.dc": 0.5},
        corporate_gain_share=0.5,
        gain_dynamics_active=True,
    )

    for name in ("kg_lt", "kg_lt_basis", "kg_st"):
        np.testing.assert_allclose(result[name], records[name])
    np.testing.assert_array_equal(result["corp_dY_gains"], [0.0, 0.0])
    np.testing.assert_allclose(result["corp_dY_retirement"], [-7.5, 0.0], atol=CENT)


def test_the_sale_volume_margin_adds_its_gains_change_to_the_record():
    records = corporate_records()
    records["corp_dY_gains"] = jnp.array([-3.0, 0.0])

    scaled = apply_corporate_gain_quantity_margin(records, 0.9)

    # Long-term gains fall $10 and short-term gains $5; the basis is not income.
    np.testing.assert_allclose(scaled["corp_dY_gains"], [-18.0, 0.0], atol=CENT)

    unchanged = apply_corporate_gain_quantity_margin(
        records, 1.0 + 0.5 * CORPORATE_ZERO_TOLERANCE
    )
    np.testing.assert_array_equal(unchanged["corp_dY_gains"], records["corp_dY_gains"])


def test_the_markdown_gains_change_is_added_to_the_record():
    records = {
        "corp_dY_gains": jnp.array([-3.0, 0.0]),
        "kg_corporate_markdown": jnp.array([-40.0, 0.0]),
    }

    recorded = record_corporate_markdown_gains(records)

    np.testing.assert_allclose(recorded["corp_dY_gains"], [-43.0, 0.0], atol=CENT)
    assert record_corporate_markdown_gains({"kg_lt": jnp.ones(2)}).keys() == {"kg_lt"}


def test_corporate_gain_quantity_margin_leaves_death_gains_and_tiny_changes_alone():
    records = corporate_records()
    records["kg_deemed_full"] = jnp.array([80.0, 0.0])

    scaled = apply_corporate_gain_quantity_margin(records, 0.9)

    np.testing.assert_allclose(scaled["kg_lt"], [90.0, 0.0])
    np.testing.assert_allclose(scaled["kg_lt_basis"], [90.0, 0.0])
    np.testing.assert_allclose(scaled["kg_st"], [45.0, 0.0])
    np.testing.assert_array_equal(scaled["kg_deemed_full"], records["kg_deemed_full"])

    unchanged = apply_corporate_gain_quantity_margin(
        records, 1.0 + 0.5 * CORPORATE_ZERO_TOLERANCE
    )
    for name in ("kg_lt", "kg_lt_basis", "kg_st"):
        np.testing.assert_array_equal(unchanged[name], records[name])


def test_corporate_adjustment_is_a_jax_calculation():
    adjustment = jax.jit(corporate_adjustment)
    result = adjustment(corporate_records())

    np.testing.assert_allclose(result["corp_dY_exog"], [-65.0, 0.0])


def test_corporate_markdown_keeps_households_in_their_pre_markdown_wealth_cells():
    """Equal pre-markdown wealth stays tied despite different equity exposure."""
    ages = wealth_arithmetic.age_grid()
    profile = wealth_state.WealthProfileYear(
        saving_shares=jnp.ones((len(ages), 4)),
        transition=jnp.eye(4),
        r_total=jnp.asarray(0.0),
        ages=jnp.asarray(ages),
        aging=wealth_arithmetic.build_aging_matrix(len(ages)),
        fmax=jnp.asarray(0.9),
        pass_through_weight=jnp.asarray(0.2),
    )
    records = {
        "weight": jnp.ones(4),
        "age1": jnp.full(4, 40.0),
        "age2": jnp.zeros(4),
        "filing_status": jnp.ones(4),
        "dep_status": jnp.zeros(4),
        "net_worth": jnp.array([100.0, 200.0, 200.0, 300.0]),
        "value.cash": jnp.array([100.0, 0.0, 200.0, 300.0]),
        "value.equities": jnp.array([0.0, 200.0, 0.0, 0.0]),
    }

    def adjust_and_rank(frame):
        adjusted = apply_corporate_adjustment(
            frame,
            dividend_factor=1.0,
            interest_factor=1.0,
            rent_factor=1.0,
            pass_through_factor=1.0,
            markdown_rate=0.5,
            flow_factor=0.0,
            asset_exposure={"value.equities": 1.0},
            corporate_gain_share=0.0,
            gain_dynamics_active=False,
        )
        assignment = wealth_state.assign_cells(adjusted, profile)
        return adjusted, assignment

    adjusted, assignment = jax.jit(adjust_and_rank)(records)

    np.testing.assert_allclose(
        adjusted["net_worth_before_corporate_markdown"], [100.0, 200.0, 200.0, 300.0]
    )
    np.testing.assert_allclose(adjusted["net_worth"], [100.0, 100.0, 200.0, 300.0])
    np.testing.assert_array_equal(assignment.bin_index[1:3], [3, 3])


def test_corporate_adjustment_refuses_a_second_application():
    once = corporate_adjustment(corporate_records())

    with pytest.raises(ValueError, match="already applied"):
        corporate_adjustment(once)


def test_an_inert_corporate_adjustment_still_writes_its_audit_column():
    result = apply_corporate_adjustment(
        corporate_records(),
        dividend_factor=1.0,
        interest_factor=1.0,
        rent_factor=1.0,
        pass_through_factor=1.0,
        markdown_rate=0.0,
        flow_factor=0.0,
        asset_exposure={"value.equities": 1.0, "value.dc": 0.5},
        corporate_gain_share=0.5,
        gain_dynamics_active=False,
    )

    np.testing.assert_allclose(result["corp_dY_exog"], [0.0, 0.0])
    for name in ("corp_dY_retirement", "corp_dY_retirement_accrual", "corp_dY_gains"):
        np.testing.assert_array_equal(result[name], [0.0, 0.0])


def test_employer_payroll_adjustment_changes_base_wages_and_records_cash():
    records = payroll_records()
    result = apply_employer_payroll_adjustment(
        records,
        payroll_law_with_no_employer_tax(),
        baseline_liab_fica_er1=jnp.array([20.0, 0.0]),
        baseline_liab_fica_er2=jnp.array([10.0, 0.0]),
        baseline_wages1=jnp.array([100.0, 0.0]),
        baseline_wages2=jnp.array([50.0, 0.0]),
    )

    # The first earner starts with $100 of cash wages. Removing $20 of employer
    # tax moves 85 percent, or $17, into cash wages. Tips and overtime remain
    # $10 each, so base wages rise from $80 to $97. The second earner likewise
    # receives $8.50 on $50 of starting wages.
    np.testing.assert_allclose(result["wages_base1"], [97.0, 0.0])
    np.testing.assert_allclose(result["wages_base2"], [48.5, 0.0])
    np.testing.assert_allclose(result["tips1"], records["tips1"])
    np.testing.assert_allclose(result["ot1"], records["ot1"])
    np.testing.assert_allclose(result["pr_dY_exog"], [25.5, 0.0])


def test_employer_payroll_adjustment_is_a_jax_calculation():
    law = payroll_law_with_no_employer_tax()

    @jax.jit
    def adjust(records, baseline1, baseline2, wages1, wages2):
        return apply_employer_payroll_adjustment(
            records, law, baseline1, baseline2, wages1, wages2
        )

    result = adjust(
        payroll_records(),
        jnp.array([20.0, 0.0]),
        jnp.array([10.0, 0.0]),
        jnp.array([100.0, 0.0]),
        jnp.array([50.0, 0.0]),
    )
    np.testing.assert_allclose(result["pr_dY_exog"], [25.5, 0.0])


def test_employer_payroll_adjustment_refuses_a_second_application():
    once = apply_employer_payroll_adjustment(
        payroll_records(),
        payroll_law_with_no_employer_tax(),
        jnp.array([20.0, 0.0]),
        jnp.array([10.0, 0.0]),
        jnp.array([100.0, 0.0]),
        jnp.array([50.0, 0.0]),
    )

    with pytest.raises(ValueError, match="already applied"):
        apply_employer_payroll_adjustment(
            once,
            payroll_law_with_no_employer_tax(),
            jnp.array([20.0, 0.0]),
            jnp.array([10.0, 0.0]),
            jnp.array([100.0, 0.0]),
            jnp.array([50.0, 0.0]),
        )


# ---------------------------------------------------------------------------
# The exact employer-payroll passthrough
# ---------------------------------------------------------------------------

# The Social Security (OASDI) taxable maximum used in these made-up laws. It is
# the 2024 figure, picked only because it is a realistic place for the kink.
TEST_OASDI_TAXABLE_MAXIMUM = 168_600.0

# The employer-side Medicare (Hospital Insurance) rate under current law, which
# applies to every dollar of wages with no cap.
TEST_HI_EMPLOYER_RATE = 0.0145


def employer_payroll_law(oasdi_employer_rate: float, capped: bool) -> PayrollLaw:
    """A made-up law carrying only the employer side of Social Security and Medicare.

    With ``capped`` false the Social Security rate applies to every dollar,
    which makes the whole employer schedule one flat rate. With it true, Social
    Security stops at the taxable maximum and only the Medicare rate applies
    above it.
    """
    if capped:
        oasdi_brackets = jnp.array([0.0, TEST_OASDI_TAXABLE_MAXIMUM])
        oasdi_rates = jnp.array([oasdi_employer_rate, 0.0])
    else:
        oasdi_brackets = jnp.array([0.0])
        oasdi_rates = jnp.array([oasdi_employer_rate])
    return PayrollLaw(
        tips_exempt=0.0,
        ot_exempt=0.0,
        seca_taxable_rate=0.9235,
        se_thresh=400.0,
        oasdi_ee_rates=jnp.array([0.0]),
        oasdi_ee_brackets=jnp.array([0.0]),
        oasdi_er_rates=oasdi_rates,
        oasdi_er_brackets=oasdi_brackets,
        hi_ee_rates=jnp.array([0.0]),
        hi_ee_brackets=jnp.array([0.0]),
        hi_er_rates=jnp.array([TEST_HI_EMPLOYER_RATE]),
        hi_er_brackets=jnp.array([0.0]),
        add_med_rates=jnp.array([0.0]),
        add_med_brackets=jnp.array([0.0]),
    )


def one_earner_records(wage) -> dict[str, jnp.ndarray]:
    """One household with a single earner paid entirely in base wages."""
    wages_base1 = jnp.atleast_1d(jnp.asarray(wage))
    zero = jnp.zeros_like(wages_base1)
    records = {
        name: zero
        for name in (
            "wages_base2",
            "tips1",
            "tips2",
            "ot1",
            "ot2",
            "trad_contr_er1",
            "trad_contr_er2",
            "sole_prop1",
            "sole_prop2",
            "farm1",
            "farm2",
            "part_se1",
            "part_se2",
        )
    }
    records["wages_base1"] = wages_base1
    return records


def employer_tax_on(wage, law: PayrollLaw):
    """Employer payroll tax owed on a single earner's wage under one law."""
    return calc_pr(one_earner_records(wage), law)["liab_fica_er1"]


def test_flat_rate_reform_matches_the_closed_form_wage():
    # Far below any cap the whole employer schedule is one flat rate — Social
    # Security plus Medicare — so the solve has a closed form: total
    # compensation w(1 + baseline rate) has to cover the new wage and the tax
    # on it, giving w(1 + baseline rate)/(1 + reform rate).
    wage = 50_000.0
    baseline_rate = 0.062
    reform_rate = 0.042
    baseline_law = employer_payroll_law(baseline_rate, capped=False)
    reform_law = employer_payroll_law(reform_rate, capped=False)
    baseline_tax = employer_tax_on(wage, baseline_law)

    result = apply_employer_payroll_adjustment(
        one_earner_records(wage),
        reform_law,
        baseline_tax,
        jnp.array([0.0]),
        jnp.array([wage]),
        jnp.array([0.0]),
        cash_wage_share=1.0,
    )

    expected = wage * (
        (1.0 + baseline_rate + TEST_HI_EMPLOYER_RATE)
        / (1.0 + reform_rate + TEST_HI_EMPLOYER_RATE)
    )
    new_wage = result["wages_base1"]
    # Five passes hold the identity to under a cent even for this
    # larger-than-life reform.
    np.testing.assert_allclose(new_wage, [expected], atol=0.01)

    # Total compensation — the new wage plus the tax owed on it — is back at
    # the baseline wage plus the baseline tax, within the same few cents.
    total_compensation = new_wage + employer_tax_on(new_wage, reform_law)
    np.testing.assert_allclose(total_compensation, wage + baseline_tax, atol=0.01)


def test_a_small_rate_change_lands_on_the_closed_form_to_the_cent():
    # Two tenths of a point off the employer rate, on a $10,000 wage. The gap
    # the passes start from is small enough that what they leave behind is
    # under a cent, so the accounting identity holds at the usual standard.
    wage = 10_000.0
    baseline_rate = 0.062
    reform_rate = 0.060
    baseline_law = employer_payroll_law(baseline_rate, capped=False)
    reform_law = employer_payroll_law(reform_rate, capped=False)
    baseline_tax = employer_tax_on(wage, baseline_law)

    result = apply_employer_payroll_adjustment(
        one_earner_records(wage),
        reform_law,
        baseline_tax,
        jnp.array([0.0]),
        jnp.array([wage]),
        jnp.array([0.0]),
        cash_wage_share=1.0,
    )

    expected = wage * (
        (1.0 + baseline_rate + TEST_HI_EMPLOYER_RATE)
        / (1.0 + reform_rate + TEST_HI_EMPLOYER_RATE)
    )
    new_wage = result["wages_base1"]
    np.testing.assert_allclose(new_wage, [expected], atol=0.01)

    total_compensation = new_wage + employer_tax_on(new_wage, reform_law)
    np.testing.assert_allclose(total_compensation, wage + baseline_tax, atol=0.01)


def test_an_earner_pushed_over_the_taxable_maximum_still_lands_on_the_anchor():
    # This earner starts just below the Social Security taxable maximum, and
    # the raise the employer tax cut pays for carries the earner across it.
    # Above the maximum only the 1.45 percent Medicare rate applies, so the tax
    # on the new wage is not the old average rate times the new wage.
    wage = 168_000.0
    baseline_law = employer_payroll_law(0.062, capped=True)
    reform_law = employer_payroll_law(0.042, capped=True)
    baseline_tax = employer_tax_on(wage, baseline_law)

    result = apply_employer_payroll_adjustment(
        one_earner_records(wage),
        reform_law,
        baseline_tax,
        jnp.array([0.0]),
        jnp.array([wage]),
        jnp.array([0.0]),
        cash_wage_share=1.0,
    )
    new_wage = result["wages_base1"]

    assert float(new_wage[0]) > TEST_OASDI_TAXABLE_MAXIMUM
    total_compensation = new_wage + employer_tax_on(new_wage, reform_law)
    np.testing.assert_allclose(total_compensation, wage + baseline_tax, atol=0.01)


def test_a_reform_that_leaves_employer_payroll_tax_alone_leaves_wages_alone():
    # The rate here is one sixteenth, which is exact in binary, so "unchanged"
    # can be checked as an exact equality rather than to a tolerance.
    wage = 100_000.0
    law = PayrollLaw(
        tips_exempt=0.0,
        ot_exempt=0.0,
        seca_taxable_rate=0.9235,
        se_thresh=400.0,
        oasdi_ee_rates=jnp.array([0.0]),
        oasdi_ee_brackets=jnp.array([0.0]),
        oasdi_er_rates=jnp.array([0.0625]),
        oasdi_er_brackets=jnp.array([0.0]),
        hi_ee_rates=jnp.array([0.0]),
        hi_ee_brackets=jnp.array([0.0]),
        hi_er_rates=jnp.array([0.0]),
        hi_er_brackets=jnp.array([0.0]),
        add_med_rates=jnp.array([0.0]),
        add_med_brackets=jnp.array([0.0]),
    )
    baseline_tax = employer_tax_on(wage, law)

    result = apply_employer_payroll_adjustment(
        one_earner_records(wage),
        law,
        baseline_tax,
        jnp.array([0.0]),
        jnp.array([wage]),
        jnp.array([0.0]),
    )

    assert float(result["wages_base1"][0]) == wage
    assert float(result["pr_dY_exog"][0]) == 0.0


def test_only_the_cash_wage_share_of_the_solved_change_reaches_wages():
    wage = 50_000.0
    baseline_rate = 0.062
    reform_rate = 0.042
    baseline_law = employer_payroll_law(baseline_rate, capped=False)
    reform_law = employer_payroll_law(reform_rate, capped=False)
    baseline_tax = employer_tax_on(wage, baseline_law)
    arguments = (
        one_earner_records(wage),
        reform_law,
        baseline_tax,
        jnp.array([0.0]),
        jnp.array([wage]),
        jnp.array([0.0]),
    )

    whole = apply_employer_payroll_adjustment(*arguments, cash_wage_share=1.0)
    shared = apply_employer_payroll_adjustment(*arguments)

    solved_change = whole["wages_base1"] - wage
    np.testing.assert_allclose(
        shared["wages_base1"] - wage,
        EMPLOYER_PAYROLL_CASH_WAGE_SHARE * solved_change,
        rtol=0,
        atol=1e-9,
    )
    np.testing.assert_allclose(shared["pr_dY_exog"], shared["wages_base1"] - wage)


def test_gradients_through_the_wage_solve_are_finite():
    reform_law = employer_payroll_law(0.042, capped=True)
    baseline_law = employer_payroll_law(0.062, capped=True)

    def solved_wage(wage):
        records = one_earner_records(wage)
        baseline_tax = calc_pr(records, baseline_law)["liab_fica_er1"]
        result = apply_employer_payroll_adjustment(
            records,
            reform_law,
            baseline_tax,
            jnp.zeros_like(baseline_tax),
            records["wages_base1"],
            jnp.zeros_like(baseline_tax),
        )
        return jnp.sum(result["wages_base1"])

    for wage in (50_000.0, TEST_OASDI_TAXABLE_MAXIMUM, 400_000.0):
        slope = jax.grad(solved_wage)(wage)
        assert np.isfinite(float(slope))
