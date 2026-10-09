"""The composed chain: does it run, and is it run in the right order."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from taxmodel.taxes import statutory_order
from taxmodel.taxes.calc.payroll import calc_pr
from taxmodel.taxes.calc.wealth import WealthLaw

from .chain_steps import run_taxes_by_step
from .conftest import ESTATE_MEASUREMENT, synthetic_records


def frame_for_1040(law, records):
    """What `run_taxes` hands the 1040: the frame plus the steps above it."""
    frame = dict(records)
    frame |= statutory_order.derive_vars(frame)
    frame |= calc_pr(frame, law.payroll)
    return frame


def one_record(law, **overrides):
    """Run the whole chain on a single household; results come back as scalars."""
    filing_status = overrides.pop("filing_status", 1)
    records = synthetic_records([filing_status], **overrides)
    return {
        name: value[0] for name, value in statutory_order.run_taxes(records, law).items()
    }


def single_filer_law(baseline_law):
    """Baseline 2027 law narrowed to the first of the four bound households."""
    return _select_record(baseline_law, 0)


def _select_record(law, index):
    """Take one record's slice out of every law parameter bound per record."""

    def narrow(value):
        array = jnp.asarray(value)
        if array.ndim == 0:
            return array
        if array.shape[0] == 4 and array.ndim <= 2:
            return array[index]
        return array

    def rebuild(group):
        if not hasattr(group, "__dataclass_fields__"):
            return group
        return replace(
            group,
            **{
                name: rebuild(getattr(group, name))
                if hasattr(getattr(group, name), "__dataclass_fields__")
                else narrow(getattr(group, name))
                for name in group.__dataclass_fields__
            },
        )

    return replace(
        law,
        **{
            name: rebuild(getattr(law, name))
            for name in law.__dataclass_fields__
            if name != "year"
        },
    )


# --------------------------------------------------------------------------
# It runs, and the arithmetic is the arithmetic of the 1040
# --------------------------------------------------------------------------


def test_a_household_with_nothing_owes_nothing(baseline_law):
    out = statutory_order.run_taxes(synthetic_records([1, 2, 3, 4]), baseline_law)
    np.testing.assert_allclose(out["agi"], 0.0)
    np.testing.assert_allclose(out["txbl_inc"], 0.0)
    np.testing.assert_allclose(out["liab_iit_net"], 0.0)
    np.testing.assert_allclose(out["liab_pr"], 0.0)


def test_wealth_and_remittance_are_recorded_after_the_1040(baseline_law):
    law = replace(
        single_filer_law(baseline_law),
        wealth=WealthLaw(
            brackets=jnp.array([0.0, 1_000.0]), rates=jnp.array([0.0, 0.10])
        ),
    )
    records = synthetic_records(
        [1, 1],
        wages_base1=jnp.array([50_000.0, 0.0]),
        net_worth=jnp.array([500.0, 2_000.0]),
    )

    result, steps = run_taxes_by_step(records, law)

    # The step-by-step run has to agree with the production chain column for
    # column, or the order the assertions below check is not the order the
    # model runs.
    for name, value in statutory_order.run_taxes(records, law).items():
        np.testing.assert_array_equal(np.asarray(result[name]), np.asarray(value))

    assert {"calc_wealth", "remit_taxes"} <= steps.keys()
    assert list(steps).index("do_1040") < list(steps).index("calc_wealth")
    assert list(steps).index("calc_wealth") < list(steps).index(
        "filer_and_expanded_income"
    )
    assert list(steps).index("filer_and_expanded_income") < list(steps).index(
        "remit_taxes"
    )
    np.testing.assert_allclose(result["liab_wealth"], [0.0, 100.0])
    for column in (
        "pmt_iit_nonwithheld",
        "pmt_iit_withheld",
        "pmt_refund_nonwithheld",
        "pmt_refund_withheld",
        "pmt_pr_nonwithheld",
        "pmt_pr_withheld",
    ):
        assert column in result


def test_a_wage_only_single_filer_pays_the_ordinary_schedule(baseline_law):
    # $80,000 of wages, no dependents, no deductions beyond the standard one.
    # 2027 standard deduction for a single filer is $16,500, so taxable income
    # is $63,500 and the tax is the ordinary schedule integrated over it.
    law = single_filer_law(baseline_law)
    out = one_record(law, wages_base1=80_000.0)
    assert float(out["agi"]) == pytest.approx(80_000.0)
    assert float(out["std_ded"]) == pytest.approx(16_500.0)
    assert float(out["txbl_inc"]) == pytest.approx(63_500.0)

    rates = np.asarray(law.tax.ord_rates)
    brackets = np.asarray(law.tax.ord_brackets)
    live = ~np.isnan(rates)
    edges = np.append(brackets[live][1:], np.inf)
    expected = float(
        np.sum(np.maximum(0.0, np.minimum(63_500.0, edges) - brackets[live]) * rates[live])
    )
    assert float(out["liab"]) == pytest.approx(expected)
    # Employee and employer payroll tax on $80,000 of wages, both halves.
    assert float(out["liab_pr"]) == pytest.approx(2 * 80_000.0 * 0.0765)


def test_the_chain_says_which_columns_a_short_frame_is_missing(baseline_law):
    records = dict(synthetic_records([1]))
    del records["wagebill_farm"]
    del records["casualty_exp"]
    with pytest.raises(KeyError) as raised:
        statutory_order.run_taxes(records, baseline_law)
    assert "wagebill_farm" in str(raised.value)
    assert "casualty_exp" in str(raised.value)


def test_an_itemizer_reports_no_standard_deduction_and_the_reverse(baseline_law):
    law = single_filer_law(baseline_law)
    itemizer = one_record(
        law, wages_base1=400_000.0, char_cash=100_000.0
    )
    assert bool(itemizer["itemizing"])
    assert float(itemizer["std_ded"]) == 0.0
    assert float(itemizer["char_item_ded"]) > 0.0

    standard = one_record(law, wages_base1=80_000.0)
    assert not bool(standard["itemizing"])
    assert float(standard["item_ded"]) == 0.0
    assert float(standard["std_ded"]) == pytest.approx(16_500.0)


def test_a_single_parent_gets_the_refundable_child_credit(baseline_law):
    # A single parent with $20,000 of wages and two young children has a small
    # income tax, which the child credit wipes out, and collects the rest of
    # the credit as a refund.
    law = single_filer_law(baseline_law)
    out = one_record(
        law,
        filing_status=4,
        wages_base1=20_000.0,
        n_dep=2.0,
        dep_age1=4.0,
        dep_age2=7.0,
        dep_ctc1=1.0,
        dep_ctc2=1.0,
        dep_ssn1=1.0,
        dep_ssn2=1.0,
    )
    assert float(out["ctc_nonref"]) == pytest.approx(float(out["liab_bc"]))
    assert float(out["ctc_ref"]) > 0.0
    assert float(out["liab_iit_net"]) < 0.0


def test_a_refundable_credit_makes_a_nonfiler_a_filer():
    # The filing flag is recomputed after the credits are known, because a
    # household with no filing requirement still files when a refundable
    # credit is worth money to it. It comes back under its own name, so
    # Tax-Data's input column is still there and still says what it said.
    columns = {
        name: jnp.zeros(3)
        for name in (
            "wages_base1", "wages_base2", "tips1", "tips2", "ot1", "ot2",
            "trad_contr_er1", "trad_contr_er2", "txbl_int", "exempt_int",
            "div_ord", "div_pref", "state_ref", "txbl_ira_dist", "gross_pens_dist",
            "kg_st", "kg_lt", "other_gains", "alimony", "sole_prop1", "sole_prop2",
            "sch_e", "farm1", "farm2", "gross_ss", "ui", "other_inc",
            "salt_workaround_part",
            "salt_workaround_scorp", "liab_pr_er", "itemizing", "tip_ded",
            "ot_ded", "auto_int_ded",
        )
    }
    columns["filer"] = jnp.array([0.0, 0.0, 1.0])
    columns["become_filer_ctc"] = jnp.array([1.0, 0.0, 0.0])
    columns["become_filer_rebate"] = jnp.array([0.0, 1.0, 0.0])
    out = statutory_order.filer_and_expanded_income(columns)
    np.testing.assert_allclose(out["filer_recomputed"], [1.0, 1.0, 1.0])
    assert "filer" not in out
    # All three have nothing but wages and do not itemize, so all three are
    # returns the IRS could pre-fill -- but only for someone who files.
    np.testing.assert_array_equal(out["simple_filer"], [1, 1, 1])


# --------------------------------------------------------------------------
# The charitable double run
# --------------------------------------------------------------------------


def both_charitable_forms(law):
    """The law, with a charitable deduction available in both places.

    Under 2027 law the above-the-line charitable deduction is capped at
    $1,000, and Schedule A has no separate dollar cap, so the double run is
    live. Setting both explicitly keeps the test independent of the year.
    """
    return replace(
        law,
        agi=replace(law.agi, char_above_limit=jnp.asarray(1_000.0)),
        item_ded=replace(law.item_ded, char_limit=jnp.asarray(jnp.inf)),
    )


def only_the_itemized_form(law):
    return replace(
        law,
        agi=replace(law.agi, char_above_limit=jnp.asarray(0.0)),
        item_ded=replace(law.item_ded, char_limit=jnp.asarray(jnp.inf)),
    )


def test_the_double_run_fires_only_when_both_forms_are_open(baseline_law):
    assert statutory_order.charity_double_run_applies(both_charitable_forms(baseline_law))
    assert not statutory_order.charity_double_run_applies(only_the_itemized_form(baseline_law))


def test_with_one_form_open_the_chain_runs_the_1040_once(baseline_law):
    # With no above-the-line deduction available, forcing the itemized form
    # changes nothing, so the single run and the itemized leg agree.
    law = only_the_itemized_form(single_filer_law(baseline_law))
    records = frame_for_1040(
        law,
        synthetic_records([1], wages_base1=300_000.0, char_cash=50_000.0),
    )
    once = statutory_order.run_1040(records, law)
    forced = statutory_order.run_1040(records, law, "item")
    np.testing.assert_allclose(once["liab_iit_net"], forced["liab_iit_net"])


def test_the_double_run_gives_each_filer_the_cheaper_form(baseline_law):
    law = both_charitable_forms(single_filer_law(baseline_law))
    # A small giver who takes the standard deduction is better off above the
    # line; a large giver who itemizes anyway is better off on Schedule A.
    records = synthetic_records(
        [1, 1],
        wages_base1=jnp.array([60_000.0, 600_000.0]),
        char_cash=jnp.array([1_000.0, 120_000.0]),
    )
    frame = frame_for_1040(law, records)
    above = statutory_order.run_1040(frame, law, "above")
    item = statutory_order.run_1040(frame, law, "item")
    chosen = statutory_order.run_taxes(records, law)

    np.testing.assert_allclose(
        chosen["liab_iit_net"],
        np.minimum(above["liab_iit_net"], item["liab_iit_net"]),
    )
    # The small giver's better answer is the above-the-line one, the large
    # giver's is Schedule A, so the two legs are genuinely different.
    assert float(above["liab_iit_net"][0]) < float(item["liab_iit_net"][0])
    assert float(item["liab_iit_net"][1]) < float(above["liab_iit_net"][1])
    # The chosen leg governs the whole return, not just its bottom line: the
    # filer who reports above the line deducts nothing on Schedule A.
    assert float(chosen["char_above_ded"][0]) > 0.0
    assert float(chosen["char_item_ded"][0]) == 0.0
    assert float(chosen["char_above_ded"][1]) == 0.0
    assert float(chosen["char_item_ded"][1]) > 0.0


# --------------------------------------------------------------------------
# Order of operations
# --------------------------------------------------------------------------


def test_credits_are_limited_after_the_amt_not_before_it(baseline_law):
    # The child credit is nonrefundable up to liability before credits, and
    # liability before credits includes the alternative minimum tax. Running
    # the credits before the AMT would cap the credit at the regular tax and
    # leave the household paying more.
    #
    # The 2027 AMT exemption is large enough that almost nobody pays AMT, so
    # this sets it to zero -- a reform could -- to put a household in the
    # position where the ordering shows.
    law = single_filer_law(baseline_law)
    law = replace(law, amt=replace(law.amt, exempt=jnp.asarray(0.0)))
    out = one_record(
        law,
        wages_base1=60_000.0,
        n_dep=3.0,
        dep_age1=3.0,
        dep_age2=6.0,
        dep_age3=9.0,
        dep_ctc1=1.0,
        dep_ctc2=1.0,
        dep_ctc3=1.0,
        dep_ssn1=1.0,
        dep_ssn2=1.0,
        dep_ssn3=1.0,
    )
    regular_tax = float(out["liab"])
    amt = float(out["liab_amt"])
    credit = float(out["ctc_nonref"])
    assert amt > 0.0
    # Three children at $2,300 each is $6,900 of credit; the household's
    # regular tax alone does not cover it but regular tax plus AMT does.
    assert credit == pytest.approx(6_900.0)
    assert regular_tax < credit <= regular_tax + amt
    assert float(out["liab_bc"]) == pytest.approx(regular_tax + amt)


def test_the_alternative_maximum_tax_cap_is_applied_before_the_amt(baseline_law):
    # The Section 1A cap is taken off the four liability components before the
    # AMT is computed, and the AMT is measured against the capped figure. If
    # the cap ran after, `liab` would still be the uncapped Schedule D tax.
    #
    # The cap is switched off under 2027 law (its qualifying multiple is
    # zero), so this puts a live one in: 5% of income above $20,000, available
    # to anyone under a hundred times that.
    law = single_filer_law(baseline_law)
    law = replace(
        law,
        alt_max=replace(
            law.alt_max,
            rate=jnp.asarray(0.05),
            exempt=jnp.asarray(20_000.0),
            qualify_mult=jnp.asarray(100.0),
        ),
    )
    out = one_record(law, wages_base1=60_000.0)
    assert bool(out["alt_max_cap_binds"])
    assert float(out["liab"]) == pytest.approx(float(out["alt_max_liab"]))
    # The saving came off ordinary income first, which is where this
    # household's whole liability sat.
    assert float(out["alt_max_ord"]) > 0.0
    assert float(out["liab_ord"]) == pytest.approx(float(out["liab"]))


def test_the_declared_calculated_columns_are_exactly_what_the_chain_returns(
    baseline_law,
):
    """`CALCULATED_COLUMNS` is what a detail request may name, so it must be true.

    A column the chain creates and this list omits could never be exported; a
    name in the list the chain does not create would be accepted and then fail
    mid-run. Both are caught here.
    """
    records = dict(synthetic_records([1, 2, 3, 4]))
    for name in statutory_order.ESTATE_RECORD_COLUMNS:
        records[name] = jnp.zeros(4)
    law = baseline_law

    with_estate = statutory_order.run_taxes(records, law, ESTATE_MEASUREMENT)
    assert set(with_estate) == set(statutory_order.CALCULATED_COLUMNS)
    assert len(statutory_order.CALCULATED_COLUMNS) == len(set(statutory_order.CALCULATED_COLUMNS))

    # Without the measurement bridge the estate tax is skipped, and its four
    # columns are the only ones missing.
    without_estate = statutory_order.run_taxes(records, law)
    assert set(statutory_order.CALCULATED_COLUMNS) - set(without_estate) == set(
        statutory_order.ESTATE_COLUMNS
    )
