"""Worked cases from the PSL comparison of Tax-Calculator with this model.

Each test takes a household from the comparison, or one built on the same
provision, and checks the amount the statute gives under the frozen 2027 law.
"""

from dataclasses import replace

import numpy as np
import pytest

from taxmodel.taxes import statutory_order

from .conftest import law_for, synthetic_records


def run(compiled_baseline, filing_status, **columns):
    """One household through the whole chain; results come back as scalars."""
    records = synthetic_records([filing_status], **columns)
    law = law_for(compiled_baseline, [filing_status])
    out = statutory_order.run_taxes(records, law)
    return {name: float(value[0]) for name, value in out.items()}, law


def run_under(law, filing_status, **columns):
    """One household through the whole chain under a given law."""
    records = synthetic_records([filing_status], **columns)
    out = statutory_order.run_taxes(records, law)
    return {name: float(value[0]) for name, value in out.items()}


def one_dependent(age):
    return {"n_dep": 1.0, "dep_age1": float(age), "dep_ctc1": 1.0, "dep_ssn1": 1.0}


def test_childless_eitc_counts_the_earnings_of_a_spouse_over_the_age_limit(compiled_baseline):
    """IRC 32(c)(1)(A)(ii): either spouse may meet the age test."""
    over_limit_earner, _ = run(compiled_baseline, 2, age1=66.0, age2=60.0, wages_base1=9_000.0)
    within_limit_earner, _ = run(compiled_baseline, 2, age1=60.0, age2=66.0, wages_base1=9_000.0)
    assert over_limit_earner["eitc"] > 0.0
    assert over_limit_earner["eitc"] == pytest.approx(within_limit_earner["eitc"])


def test_childless_eitc_is_denied_when_neither_spouse_meets_the_age_test(compiled_baseline):
    out, _ = run(compiled_baseline, 2, age1=66.0, age2=70.0, wages_base1=9_000.0)
    assert out["eitc"] == 0.0


def test_ctc_phase_out_reduces_the_total_credit_once(compiled_baseline):
    """IRC 24(b)(1). $33,000 of excess income costs $1,650 of the combined credit."""
    out, law = run(
        compiled_baseline,
        2,
        age1=45.0,
        age2=45.0,
        wages_base1=433_000.0,
        n_dep=2.0,
        dep_age1=10.0,
        dep_ctc1=1.0,
        dep_ssn1=1.0,
        dep_age2=20.0,
        dep_ctc2=1.0,
        dep_ssn2=1.0,
    )
    child = float(law.ctc.value_old1[0])
    other = float(law.ctc.value_other[0])
    assert out["ctc_nonref"] == pytest.approx(child + other - 1_650.0)


def test_senior_deduction_requires_a_joint_return_if_married(compiled_baseline):
    """IRC 151(d)(5)(C)."""
    separate, _ = run(compiled_baseline, 3, age1=67.0, wages_base1=27_000.0)
    single, _ = run(compiled_baseline, 1, age1=67.0, wages_base1=27_000.0)
    assert separate["senior_ded"] == 0.0
    assert single["senior_ded"] > 0.0


def test_qbi_is_net_of_the_deduction_for_half_of_seca_tax(compiled_baseline):
    """Treas. Reg. 1.199A-3(b)(1)(vi)."""
    out, _ = run(compiled_baseline, 1, age1=45.0, wages_base1=75_000.0, sole_prop1=34_000.0)
    assert out["liab_seca_er"] > 0.0
    assert out["qbi_ded"] == pytest.approx(0.2 * (34_000.0 - out["liab_seca_er"]))


def test_seca_deduction_is_split_across_an_earners_businesses(compiled_baseline):
    both, _ = run(
        compiled_baseline,
        1,
        age1=45.0,
        wages_base1=75_000.0,
        sole_prop1=20_000.0,
        farm1=14_000.0,
    )
    assert both["qbi_ded"] == pytest.approx(0.2 * (34_000.0 - both["liab_seca_er"]))


def test_earned_income_is_net_of_the_deduction_for_half_of_seca_tax(compiled_baseline):
    """IRC 32(c)(2)(A)(ii)."""
    out, _ = run(compiled_baseline, 1, age1=45.0, wages_base1=14_000.0, sole_prop1=4_000.0)
    assert out["ei"] == pytest.approx(18_000.0 - out["liab_seca_er"])


def test_three_children_may_use_payroll_taxes_less_the_eitc(compiled_baseline):
    """IRC 24(d)(1)(B)(ii). Interest above the investment income limit removes the EITC."""
    children = {
        "n_dep": 3.0,
        "n_dep_eitc": 3.0,
        "dep_age1": 8.0,
        "dep_age2": 9.0,
        "dep_age3": 10.0,
        "dep_ctc1": 1.0,
        "dep_ctc2": 1.0,
        "dep_ctc3": 1.0,
        "dep_ssn1": 1.0,
        "dep_ssn2": 1.0,
        "dep_ssn3": 1.0,
    }
    out, _ = run(
        compiled_baseline, 4, age1=40.0, wages_base1=5_000.0, txbl_int=30_000.0, **children
    )
    assert out["eitc"] == 0.0
    assert out["ctc_ref"] >= 0.0765 * 5_000.0 - 1e-6
    # With an EITC larger than the payroll tax the alternative is worth nothing,
    # and the refund is the earned income formula.
    with_eitc, law = run(compiled_baseline, 4, age1=40.0, wages_base1=5_000.0, **children)
    assert with_eitc["eitc"] > with_eitc["liab_pr_ee"]
    assert with_eitc["ctc_ref"] == pytest.approx(
        float(law.ctc.pi_rate[0]) * (5_000.0 - float(law.ctc.pi_thresh[0]))
    )


def test_two_children_do_not_get_the_payroll_tax_alternative(compiled_baseline):
    children = {
        "n_dep": 2.0,
        "n_dep_eitc": 2.0,
        "dep_age1": 8.0,
        "dep_age2": 9.0,
        "dep_ctc1": 1.0,
        "dep_ctc2": 1.0,
        "dep_ssn1": 1.0,
        "dep_ssn2": 1.0,
    }
    out, law = run(
        compiled_baseline, 4, age1=40.0, wages_base1=5_000.0, txbl_int=30_000.0, **children
    )
    assert out["ctc_ref"] == pytest.approx(
        float(law.ctc.pi_rate[0]) * (5_000.0 - float(law.ctc.pi_thresh[0]))
    )


def test_eitc_phase_out_comes_off_the_maximum_credit(compiled_baseline):
    """IRC 32(a)(2). Earnings are in the phase-in range and AGI is in the phase-out range."""
    out, law = run(
        compiled_baseline,
        2,
        age1=77.0,
        age2=72.0,
        wages_base1=12_000.0,
        txbl_pens_dist=45_000.0,
        gross_pens_dist=45_000.0,
        **one_dependent(10),
        n_dep_eitc=1.0,
    )
    maximum = float(law.eitc.pi_end_1[0]) * float(law.eitc.pi_rate_1[0])
    phase_out = (out["agi"] - float(law.eitc.po_thresh_1[0])) * float(law.eitc.po_rate_1[0])
    assert 0.0 < maximum - phase_out < 12_000.0 * float(law.eitc.pi_rate_1[0])
    assert out["eitc"] == pytest.approx(maximum - phase_out)


@pytest.mark.parametrize("wages", (1_000_000.0, 3_000_000.0))
def test_amt_income_of_a_separate_filer_includes_the_increase(compiled_baseline, wages):
    """IRC 55(d)(2): the rate times the excess, up to the exemption amount."""
    out, law = run(compiled_baseline, 3, age1=45.0, wages_base1=wages)
    exemption = float(law.amt.exempt[0])
    rate = float(law.amt.exempt_po_rate[0])
    before = out["agi"]
    gone_at = float(law.amt.exempt_po_thresh[0]) + exemption / rate
    increase = np.clip(rate * max(before - gone_at, 0.0), 0.0, exemption)
    assert increase > 0.0
    assert out["amt_gross_inc"] == pytest.approx(before + increase)


def test_amt_income_of_a_single_filer_has_no_increase(compiled_baseline):
    out, _ = run(compiled_baseline, 1, age1=45.0, wages_base1=3_000_000.0)
    assert out["amt_gross_inc"] == pytest.approx(out["agi"])


def test_tip_and_overtime_deductions_require_a_joint_return_if_married(compiled_baseline):
    """IRC 224 and 225."""
    pay = {"age1": 40.0, "wages_base1": 40_000.0, "tips1": 5_000.0, "ot1": 3_000.0}
    separate, _ = run(compiled_baseline, 3, **pay)
    single, _ = run(compiled_baseline, 1, **pay)
    assert separate["tip_ded"] == 0.0
    assert separate["ot_ded"] == 0.0
    assert single["tip_ded"] > 0.0
    assert single["ot_ded"] > 0.0


def test_qbi_is_net_of_health_insurance_and_retirement_deductions(compiled_baseline):
    """Treas. Reg. 1.199A-3(b)(1)(vi); Form 8995 instructions."""
    out, _ = run(
        compiled_baseline,
        1,
        age1=45.0,
        wages_base1=75_000.0,
        sole_prop1=34_000.0,
        se_health=4_000.0,
        keogh_contr=2_000.0,
    )
    assert out["qbi_ded"] == pytest.approx(
        0.2 * (34_000.0 - out["liab_seca_er"] - 4_000.0 - 2_000.0)
    )


def test_health_insurance_of_an_s_corporation_shareholder_reduces_its_income(
    compiled_baseline,
):
    """Treas. Reg. 1.199A-3(b)(1)(vi); IRC 162(l)(5)."""
    out, _ = run(
        compiled_baseline,
        1,
        age1=45.0,
        wages_base1=75_000.0,
        scorp_active=34_000.0,
        se_health=4_000.0,
        keogh_contr=2_000.0,
    )
    assert out["qbi_ded"] == pytest.approx(0.2 * (34_000.0 - 4_000.0))


def test_qbi_limit_is_net_of_the_non_itemizer_charitable_deduction(compiled_baseline):
    """IRC 199A(e)(1). The taxable income limit binds; a gift lowers it by 20 cents a dollar."""
    columns = {"age1": 45.0, "scorp_active": 40_000.0, "kg_lt": 0.0}
    no_gift, _ = run(compiled_baseline, 1, **columns)
    gift, _ = run(compiled_baseline, 1, char_cash=500.0, **columns)
    assert no_gift["qbi_ded"] - gift["qbi_ded"] == pytest.approx(100.0)


def test_a_reform_can_allow_separate_filers_the_senior_deduction(compiled_baseline):
    _, law = run(compiled_baseline, 3, age1=67.0, wages_base1=27_000.0)
    allowed = replace(
        law, below_ded=replace(law.below_ded, senior_ded_mfs_eligible=np.ones(1))
    )
    out = run_under(allowed, 3, age1=67.0, wages_base1=27_000.0)
    assert out["senior_ded"] > 0.0


def test_ctc_phase_outs_stay_separate_when_a_reform_sets_different_rates(compiled_baseline):
    """$1,000 of excess income at 5% and 10%: $50 off the child credit, $100 off the other."""
    columns = {
        "age1": 45.0,
        "age2": 45.0,
        "n_dep": 2.0,
        "dep_age1": 10.0,
        "dep_ctc1": 1.0,
        "dep_ssn1": 1.0,
        "dep_age2": 20.0,
        "dep_ctc2": 1.0,
        "dep_ssn2": 1.0,
    }
    _, law = run(compiled_baseline, 2, wages_base1=100_000.0, **columns)
    threshold = float(np.asarray(law.ctc.po_thresh1).reshape(-1)[0])
    distinct = replace(law, ctc=replace(law.ctc, po_rate_other=np.full(1, 0.10)))
    out = run_under(distinct, 2, wages_base1=threshold + 1_000.0, **columns)
    child = float(law.ctc.value_old1[0])
    other = float(law.ctc.value_other[0])
    assert out["ctc_nonref"] == pytest.approx(child - 50.0 + other - 100.0)
