"""Every floor, cap and tie in the calculators, with a household sitting on it.

At a threshold, a household sitting exactly on one is treated as having crossed it in the direction of more
income. The published marginal tax rate is a derivative of the calculators, so
that convention is carried entirely by how each corner is written, and the only
way to see whether a site got it right is to put a household exactly on the
site and compare the derivative with what the next dollar actually costs.

Each case below builds one household on one threshold out of the law's own
numbers, and compares the derivative of the tax outcome with a one-sided rerun
of the calculators a dollar higher. A case says which priced dollars it checks
and, for each, whether the two measurements have to agree. They do not agree
everywhere, and the cases that cannot say so name the statutory reason:

- A notch or a cliff. The law itself jumps. The earned income credit is denied
  outright above an investment-income limit, self-employment tax applies only
  above the reporting threshold, and the minimum section 199A deduction is
  offered only above a stated amount of business income. A jump has no slope,
  so the derivative reports nothing and the next dollar reports the whole jump.
  Both answers are right and they are different numbers.

- A corner with no more-income side, which is left on whichever branch the
  default spelling stands on. These are amounts that rise with one kind of
  dollar and fall with another: a deduction that has phased out to nothing
  falls with a dollar of other income and rises with a dollar of the thing
  being deducted, and the earned income credit's tent rises over the phase-in
  range and falls over the phase-out range.

A case that expects agreement and does not get it is a defect in the
calculator, of the kind the section 199A component floor turned out to be.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np
import pytest

from taxmodel.taxes import individual_income_tax, marginal_rates
from taxmodel.taxes.calc.deductions import OBBBA_QBI_MIN_DEDUCTION_INCOME
from taxmodel.taxes.law_for_calculators import YearLaw

from .conftest import law_for, synthetic_records

# The tax schedule is piecewise linear almost everywhere, so away from a bend
# the derivative and the dollar rerun are the same number to floating-point
# noise. A billionth of a cent on the dollar is the working definition of
# agreement.
EXACT = 1e-9

# Two measures are only said to disagree when the gap is larger than any
# curvature a dollar of width could produce. A tenth of a cent on the dollar is
# far above that and far below any real bend in the law.
DISAGREEMENT = 1e-3

# Every household starts as a single filer whose primary earner is 40: old
# enough for the childless earned income credit's age test, young enough for
# neither the senior deduction nor the extra standard deduction.
BASE: dict[str, float] = {"age1": 40.0}

Overrides = dict[str, float]

AGREE = "agree"


@dataclass(frozen=True)
class Case:
    """One household on one threshold, and what the two rate measures should say.

    `site` names the threshold the way the statute names it. `build` turns the
    bound law into the household's record overrides, so the tie is made from
    the law's own numbers rather than a copied dollar figure. `expected` maps
    each priced dollar to either agreement or the statutory reason the two
    measurements differ.

    `tolerance` is loosened only where the tax function is genuinely curved
    over the dollar, which happens where a phase-out share multiplies an amount
    that also moves with income.

    `rounded_phase_outs` switches on the statutory rounding of the child
    credit, care credit and personal exemption phase-outs, which the default
    configuration leaves off.
    """

    site: str
    build: Callable[[YearLaw], Overrides]
    expected: dict[str, str]
    filing_status: int = 1
    tolerance: float = EXACT
    rounded_phase_outs: bool = False
    keep_the_newborn_bonus_phase_out: bool = False


def value(parameter) -> float:
    """One law parameter as a plain number, for a law read for one household."""
    return float(np.asarray(parameter).reshape(-1)[0])


# The care credit worksheet rounds income above the phase-out threshold up to
# the next $2,000.
CARE_CREDIT_ROUNDING_STEP = 2_000.0


def rounded_phase_outs(law: YearLaw) -> YearLaw:
    """The law with the three stepped phase-outs rounded as on the worksheets.

    The child credit, the child-and-dependent-care credit and the personal
    exemption phase out on income rounded up to the next statutory step on the
    credit worksheets. The default configuration switches the rounding off. The
    child credit and personal exemption steps are the law's own
    `po_discrete_step` values.
    """
    return replace(
        law,
        ctc=replace(law.ctc, po_discrete=jnp.ones_like(law.ctc.po_discrete)),
        cdctc=replace(
            law.cdctc,
            discrete_step=jnp.full_like(
                law.cdctc.discrete_step, CARE_CREDIT_ROUNDING_STEP
            ),
        ),
        pe_ded=replace(law.pe_ded, po_discrete=jnp.ones_like(law.pe_ded.po_discrete)),
    )


def without_the_newborn_bonus_phase_out(law: YearLaw) -> YearLaw:
    """The law with the newborn bonus phase-out threshold moved out of reach.

    The newborn bonus is worth nothing under current law, and its phase-out
    threshold is the same income as the additional Medicare tax threshold, the
    net investment income tax threshold and the child credit phase-out
    threshold. A household on any of those sits on the bonus's threshold too,
    and the bonus has a case of its own, so it is taken out of the way here.
    """
    return replace(
        law,
        ctc=replace(
            law.ctc,
            baby_bonus_po_thresh=jnp.full_like(
                jnp.asarray(law.ctc.baby_bonus_po_thresh), jnp.inf
            ),
        ),
    )


def landing_exactly_on(
    target: float, amount: Callable[[float], float], start: float
) -> float:
    """An input whose amount comes to exactly `target`, searched around `start`.

    Several thresholds are a statutory percentage of something, and a percentage
    such as five percent is not a binary fraction, so the obvious input misses
    the threshold by the last bit or two it can hold. Stepping the input one
    bit at a time either way finds an input that lands on the threshold
    exactly, which a tie-breaking convention has to be tested at.
    """
    up = down = float(start)
    for _ in range(128):
        for candidate in (up, down):
            if amount(candidate) == target:
                return candidate
        up = float(np.nextafter(up, np.inf))
        down = float(np.nextafter(down, -np.inf))
    raise AssertionError(f"nothing near {start} makes the amount exactly {target}")


# --------------------------------------------------------------------------
# Schedule A
# --------------------------------------------------------------------------

# Every Schedule A household is given state and local taxes at the deduction
# limit so that it itemizes and the piece under test reaches taxable income.
SCHEDULE_A_AGI = 400_000.0


def medical_floor(law: YearLaw) -> Overrides:
    """Medical expenses exactly at the share of AGI the statute subtracts."""
    return {
        "wages_base1": SCHEDULE_A_AGI,
        "salt_prop": value(law.item_ded.salt_limit),
        "med_exp": SCHEDULE_A_AGI * value(law.item_ded.med_floor_agi),
    }


def charitable_floor(law: YearLaw) -> Overrides:
    """A cash gift exactly at the share of AGI the statute subtracts."""
    return {
        "wages_base1": SCHEDULE_A_AGI,
        "salt_prop": value(law.item_ded.salt_limit),
        "char_cash": SCHEDULE_A_AGI * value(law.item_ded.char_floor_agi),
    }


def charitable_cash_ceiling(law: YearLaw) -> Overrides:
    """A cash gift exactly at the share of AGI that may be deducted."""
    return {
        "wages_base1": SCHEDULE_A_AGI,
        "salt_prop": value(law.item_ded.salt_limit),
        "char_cash": SCHEDULE_A_AGI * value(law.item_ded.char_cash_limit_agi),
    }


def charitable_noncash_ceiling(law: YearLaw) -> Overrides:
    """A noncash gift exactly at the share of AGI that may be deducted."""
    return {
        "wages_base1": SCHEDULE_A_AGI,
        "salt_prop": value(law.item_ded.salt_limit),
        "char_noncash": SCHEDULE_A_AGI * value(law.item_ded.char_noncash_limit_agi),
    }


def salt_cap(law: YearLaw) -> Overrides:
    """State and local taxes paid exactly at the deduction limit."""
    return {
        "wages_base1": SCHEDULE_A_AGI,
        "salt_prop": value(law.item_ded.salt_limit),
    }


def salt_phase_down_start(law: YearLaw) -> Overrides:
    """AGI exactly where the state and local limit starts phasing down."""
    return {
        "wages_base1": value(law.item_ded.salt_floor_thresh),
        "salt_prop": value(law.item_ded.salt_limit),
    }


def mortgage_balance_at_the_limit(law: YearLaw) -> Overrides:
    """A mortgage balance exactly at the limit above which interest is prorated."""
    limits = np.asarray(law.item_ded.mort_bal_limit).reshape(-1)
    newest_vintage = float(limits[~np.isnan(limits)][-1])
    return {
        "wages_base1": SCHEDULE_A_AGI,
        "salt_prop": value(law.item_ded.salt_limit),
        "first_mort_bal": newest_vintage,
        "first_mort_int": 30_000.0,
    }


def itemized_tax_value_threshold(law: YearLaw) -> Overrides:
    """Income exactly at the threshold above which itemized deductions lose value."""
    return {
        "wages_base1": value(law.txbl_inc.item_limit_tax_value_thresh),
        "other_item_exp": 50_000.0,
    }


def itemizer_with_nothing_but_state_and_local_taxes(law: YearLaw) -> Overrides:
    """An itemizer whose whole Schedule A is state and local taxes.

    Its income is below the threshold where itemized deductions start losing
    tax value, so the amount that limitation claws back is zero, and its
    non-state-and-local deductions are zero as well, so the amount and the cap
    the limitation compares are both zero. A dollar of mortgage interest is
    deductible from the first dollar, so it prices that limitation.
    """
    return {
        "wages_base1": SCHEDULE_A_AGI,
        "salt_prop": value(law.item_ded.salt_limit),
        "first_mort_int": 0.0,
    }


def itemize_or_standard(law: YearLaw) -> Overrides:
    """Itemized deductions exactly equal to the standard deduction."""
    return {"wages_base1": SCHEDULE_A_AGI, "salt_prop": value(law.std_ded.value)}


def dependent_standard_deduction_floor(law: YearLaw) -> Overrides:
    """A dependent filer's earned income exactly at the worksheet floor.

    The interest income is there so the household has taxable income at all;
    without it the deduction the tie decides never reaches a tax.
    """
    return {
        "dep_status": 1.0,
        "wages_base1": value(law.std_ded.dep_floor)
        - value(law.std_ded.dep_earned_bonus),
        "txbl_int": 50_000.0,
    }


def dependent_standard_deduction_cap(law: YearLaw) -> Overrides:
    """A dependent filer's earned income exactly at the full standard deduction."""
    return {
        "dep_status": 1.0,
        "wages_base1": value(law.std_ded.value) - value(law.std_ded.dep_earned_bonus),
        "txbl_int": 50_000.0,
    }


# --------------------------------------------------------------------------
# Above-the-line deductions and adjusted gross income
# --------------------------------------------------------------------------

# Only a new car's loan interest qualifies, so the new-car draw is put below
# the financed-new share the calculator tests it against.
BOUGHT_A_NEW_CAR = 0.1


def auto_interest_cap(law: YearLaw) -> Overrides:
    """Car loan interest exactly at the dollar limit on the deduction."""
    return {
        "wages_base1": 50_000.0,
        "auto_int_exp": value(law.agi.auto_int_ded_limit),
        "r.new_car": BOUGHT_A_NEW_CAR,
    }


def auto_interest_phase_out_start(law: YearLaw) -> Overrides:
    """Modified AGI exactly where the car loan interest deduction starts falling."""
    return {
        "wages_base1": value(law.agi.auto_int_ded_po_thresh),
        "auto_int_exp": 5_000.0,
        "r.new_car": BOUGHT_A_NEW_CAR,
    }


def auto_interest_phase_out_end(law: YearLaw) -> Overrides:
    """Modified AGI exactly where the car loan interest deduction runs out."""
    interest = 5_000.0
    return {
        "wages_base1": value(law.agi.auto_int_ded_po_thresh)
        + interest / value(law.agi.auto_int_ded_po_rate),
        "auto_int_exp": interest,
        "r.new_car": BOUGHT_A_NEW_CAR,
    }


def social_security_tier(law: YearLaw, slot: int) -> Overrides:
    """Provisional income exactly on one Social Security inclusion threshold.

    Provisional income is other income plus the statutory share of gross
    benefits, so the wage figure is the threshold less that share.
    """
    benefits = 20_000.0
    brackets = np.asarray(law.ss.brackets).reshape(-1)
    return {
        "gross_ss": benefits,
        "wages_base1": float(brackets[slot]) - benefits * value(law.ss.magi_ss_rate),
    }


def social_security_second_tier(law: YearLaw) -> Overrides:
    """Provisional income exactly where half of benefits become taxable."""
    return social_security_tier(law, 1)


def social_security_third_tier(law: YearLaw) -> Overrides:
    """Provisional income exactly where 85 percent of benefits become taxable."""
    return social_security_tier(law, 2)


# --------------------------------------------------------------------------
# The section 199A deduction
# --------------------------------------------------------------------------


def qbi_component_floor(law: YearLaw) -> Overrides:
    """A household holding none of three of the four kinds of business income.

    Its S corporation profit already earns the deduction, and its taxable
    income is well below both the phase-out threshold and the limit at a share
    of taxable income, so the next business dollar earns the flat statutory
    rate whichever kind of business it comes from.
    """
    return {"wages_base1": 100_000.0, "scorp_active": 40_000.0}


def qbi_phase_out_start(law: YearLaw) -> Overrides:
    """Taxable income exactly at the threshold where the wage limit phases in."""
    business = 23_350.0
    return {
        "wages_base1": value(law.qbi_ded.po_thresh_non_sstb)
        + value(law.std_ded.value)
        - business,
        "scorp_active": business,
    }


def qbi_phase_out_end(law: YearLaw) -> Overrides:
    """Taxable income exactly where the wage limit applies in full."""
    business = 23_350.0
    return {
        "wages_base1": value(law.qbi_ded.po_thresh_non_sstb)
        + value(law.qbi_ded.po_range_non_sstb)
        + value(law.std_ded.value)
        - business,
        "scorp_active": business,
    }


def qbi_wage_limit(law: YearLaw) -> Overrides:
    """A component deduction exactly at the share of the business's wage bill.

    The household sits half way through the phase-out range, so the wage
    limitation is partly in force and the tie reaches its taxable income.
    """
    business = 100_000.0
    tentative = business * value(law.qbi_ded.rate)
    return {
        "wages_base1": value(law.qbi_ded.po_thresh_non_sstb)
        + value(law.qbi_ded.po_range_non_sstb) / 2.0
        + value(law.std_ded.value)
        - business,
        "scorp_active": business,
        "wagebill_scorp": tentative / value(law.qbi_ded.wage_limit),
    }


def qbi_taxable_income_limit(law: YearLaw) -> Overrides:
    """The deduction exactly at the limit set as a share of taxable income.

    The deduction and the limit are the same share of two amounts, so they are
    equal when qualified business income equals taxable income: here the
    household's pay is exactly its standard deduction.
    """
    return {"wages_base1": value(law.std_ded.value), "scorp_active": 100_000.0}


def qbi_minimum_deduction_floor(law: YearLaw) -> Overrides:
    """Business income where the deduction exactly equals the statutory minimum."""
    rate = value(law.qbi_ded.rate)
    minimum = value(law.qbi_ded.min_value)
    return {
        "wages_base1": 100_000.0,
        "scorp_active": landing_exactly_on(
            minimum, lambda income: income * rate, minimum / rate
        ),
    }


def qbi_minimum_deduction_income(law: YearLaw) -> Overrides:
    """Business income exactly at the amount the statutory minimum requires."""
    return {
        "wages_base1": 100_000.0,
        "scorp_active": OBBBA_QBI_MIN_DEDUCTION_INCOME,
    }


# --------------------------------------------------------------------------
# The tip, overtime and senior deductions
# --------------------------------------------------------------------------


def tip_deduction_cap(law: YearLaw) -> Overrides:
    """Tips exactly at the dollar limit on the tip deduction."""
    return {"wages_base1": 50_000.0, "tips1": value(law.below_ded.tip_ded_limit)}


def tip_deduction_phase_out_start(law: YearLaw) -> Overrides:
    """AGI exactly where the tip deduction starts phasing out.

    The worker is given overtime as well, because the overtime deduction
    phases out from the same income and a worker with no overtime sits on that
    deduction's own zero-amount tie.
    """
    tips = 5_000.0
    overtime = 3_000.0
    return {
        "wages_base1": value(law.below_ded.tip_ded_po_thresh) - tips - overtime,
        "tips1": tips,
        "ot1": overtime,
    }


def tip_deduction_phase_out_end(law: YearLaw) -> Overrides:
    """AGI exactly where the tip deduction runs out."""
    tips = 5_000.0
    return {
        "wages_base1": value(law.below_ded.tip_ded_po_thresh)
        + tips / value(law.below_ded.tip_ded_po_rate)
        - tips,
        "tips1": tips,
    }


def overtime_deduction_cap(law: YearLaw) -> Overrides:
    """Deductible overtime premium exactly at the dollar limit.

    Only the premium half of overtime pay is deductible when the wage measure
    holds base pay and premium together, so the pay figure is the limit times
    that divisor.
    """
    divisor = 1.0 + 2.0 * value(law.below_ded.ot_ded_half)
    return {
        "wages_base1": 50_000.0,
        "ot1": value(law.below_ded.ot_ded_limit) * divisor,
    }


def overtime_deduction_phase_out_start(law: YearLaw) -> Overrides:
    """AGI exactly where the overtime deduction starts phasing out.

    The worker is given tips as well, for the same reason the tip case is
    given overtime.
    """
    overtime = 6_000.0
    tips = 3_000.0
    return {
        "wages_base1": value(law.below_ded.ot_ded_po_thresh) - overtime - tips,
        "ot1": overtime,
        "tips1": tips,
    }


def phase_out_threshold_with_nothing_to_phase_out(law: YearLaw) -> Overrides:
    """AGI exactly at the tip and overtime phase-out threshold, with neither.

    The worker has no tips and no overtime, so both deductions are zero and
    both sit on their phase-out threshold at the same time.
    """
    return {"wages_base1": value(law.below_ded.tip_ded_po_thresh)}


def newborn_bonus_phase_out_threshold(law: YearLaw) -> Overrides:
    """AGI exactly at the newborn bonus phase-out threshold.

    The bonus is worth nothing under current law, so there is nothing for the
    next dollar of income to phase out.
    """
    return {"wages_base1": value(law.ctc.baby_bonus_po_thresh)}


def senior_deduction_phase_out_start(law: YearLaw) -> Overrides:
    """AGI exactly where the senior deduction starts phasing out."""
    return {"age1": 70.0, "wages_base1": value(law.below_ded.senior_ded_po_thresh)}


def senior_deduction_phase_out_end(law: YearLaw) -> Overrides:
    """AGI exactly where the senior deduction runs out."""
    return {
        "age1": 70.0,
        "wages_base1": value(law.below_ded.senior_ded_po_thresh)
        + value(law.below_ded.senior_ded_value)
        / value(law.below_ded.senior_ded_po_rate),
    }


# --------------------------------------------------------------------------
# Schedule D and the rate schedules
# --------------------------------------------------------------------------


def capital_loss_limit(law: YearLaw) -> Overrides:
    """A net capital loss exactly at the annual deduction limit."""
    return {"wages_base1": 100_000.0, "kg_st": -value(law.kg.kg_loss_limit)}


def zero_net_capital_gain(law: YearLaw) -> Overrides:
    """A household with exactly no net capital gain."""
    return {"wages_base1": 100_000.0, "kg_lt": 0.0, "kg_st": 0.0}


def net_capital_gain_with_no_short_term(law: YearLaw) -> Overrides:
    """A household whose whole net gain is long term, so two amounts are equal."""
    return {"wages_base1": 100_000.0, "kg_lt": 40_000.0}


def ordinary_bracket_edge(law: YearLaw) -> Overrides:
    """Taxable income exactly on an ordinary bracket edge."""
    brackets = np.asarray(law.tax.ord_brackets).reshape(-1)
    return {"wages_base1": float(brackets[2]) + value(law.std_ded.value)}


def preferred_bracket_edge(law: YearLaw) -> Overrides:
    """Taxable income exactly on a preferred rate bracket edge.

    Half the taxable income is ordinary pay, so the household owes regular tax
    and does not also sit on the alternative minimum tax floor, which has a
    case of its own.
    """
    edge = float(np.asarray(law.tax.pref_brackets).reshape(-1)[1])
    gain = edge / 2.0
    return {
        "kg_lt": gain,
        "wages_base1": edge + value(law.std_ded.value) - gain,
    }


def nonrefundable_credit_capped_at_no_liability(law: YearLaw) -> Overrides:
    """A household sitting exactly where its tax begins.

    Its whole taxable income is long-term gain sitting in the zero percent
    preferred rate bracket, so it owes nothing, and it claims no nonrefundable
    credit either. Every cap that holds a credit down to the liability left to
    it therefore has zero on both sides. Taxable income is on the preferred
    rate bracket edge, so the next dollar of gain is taxed at the preferred
    rate.
    """
    edge = float(np.asarray(law.tax.pref_brackets).reshape(-1)[1])
    return {"kg_lt": edge + value(law.std_ded.value)}


def schedule_d_cap_with_no_preferred_income(law: YearLaw) -> Overrides:
    """A household with no income at preferred rates sits on the Schedule D cap.

    With nothing taken out of taxable income for preferred treatment, the
    Schedule D computation and ordinary tax on all taxable income are the same
    number term for term.
    """
    return {"wages_base1": 100_000.0}


def special_gain_stacking_with_no_preferred_gain(law: YearLaw) -> Overrides:
    """A household with no unrecaptured section 1250 or collectibles gain.

    Both special-gain amounts are capped at net capital gain, and all three are
    zero, so the household sits on both caps.
    """
    return {"wages_base1": 100_000.0, "kg_1250": 0.0, "kg_collect": 0.0}


def amt_exemption_phase_out_start(law: YearLaw) -> Overrides:
    """Alternative minimum taxable income exactly where the exemption starts falling.

    A filer taking the standard deduction adds it back for the alternative
    minimum tax, so alternative minimum gross income is adjusted gross income
    plus the net operating loss addback. The addback is what makes the
    alternative minimum tax bind at all at this income.
    """
    addback = 300_000.0
    return {
        "wages_base1": value(law.amt.exempt_po_thresh) - addback,
        "amt_nols": addback,
    }


# --------------------------------------------------------------------------
# Payroll tax
# --------------------------------------------------------------------------


def taxable_maximum(law: YearLaw) -> float:
    """The Social Security taxable maximum for this household."""
    return float(np.asarray(law.payroll.oasdi_ee_brackets).reshape(-1)[1])


def oasdi_taxable_maximum(law: YearLaw) -> Overrides:
    """Wages exactly at the Social Security taxable maximum."""
    return {"wages_base1": taxable_maximum(law)}


def oasdi_taxable_maximum_second_earner(law: YearLaw) -> Overrides:
    """The second earner's wages exactly at the Social Security taxable maximum."""
    return {
        "age2": 40.0,
        "wages_base1": taxable_maximum(law),
        "wages_base2": taxable_maximum(law),
    }


def additional_medicare_threshold(law: YearLaw) -> Overrides:
    """Wages exactly at the threshold for the additional Medicare tax."""
    brackets = np.asarray(law.payroll.add_med_brackets).reshape(-1)
    return {"wages_base1": float(brackets[1])}


def self_employment_reporting_threshold(law: YearLaw) -> Overrides:
    """Net self-employment earnings exactly at the reporting threshold."""
    return {"wages_base1": 50_000.0, "sole_prop1": value(law.payroll.se_thresh)}


def seca_stacked_on_a_filled_wage_base(law: YearLaw) -> Overrides:
    """Wages exactly filling the Social Security base, self-employment above it."""
    return {"wages_base1": taxable_maximum(law), "sole_prop1": 20_000.0}


# --------------------------------------------------------------------------
# Credits
# --------------------------------------------------------------------------

# The care credit is claimed by nine households in ten, so the take-up draw is
# put below that share.
CLAIMS_THE_CARE_CREDIT = 0.1

# One qualifying child, young enough for every age test in the credits.
ONE_YOUNG_CHILD: dict[str, float] = {
    "n_dep": 1.0,
    "n_dep_eitc": 1.0,
    "dep_age1": 4.0,
    "dep_ssn1": 1.0,
    "dep_ctc1": 1.0,
}


def care_credit_earned_income_cap(law: YearLaw) -> Overrides:
    """A household with a young dependent, no care expenses and no earnings.

    Care expenses are limited to the earner's own earned income, and both are
    zero, so the household sits on that limit.
    """
    return {
        **ONE_YOUNG_CHILD,
        "dep_age1": 3.0,
        "wages_base1": 0.0,
        "txbl_int": 60_000.0,
        "r.cdctc_takeup": CLAIMS_THE_CARE_CREDIT,
    }


def care_credit_expense_limit(law: YearLaw) -> Overrides:
    """Care expenses exactly at the statutory limit for one qualifying dependent."""
    return {
        **ONE_YOUNG_CHILD,
        "dep_age1": 3.0,
        "wages_base1": 60_000.0,
        "care_exp": value(law.cdctc.exp_limit),
        "r.cdctc_takeup": CLAIMS_THE_CARE_CREDIT,
    }


def care_credit_rate_phase_out_end(law: YearLaw) -> Overrides:
    """AGI exactly where the lowest care credit rate layer runs out."""
    rate = value(law.cdctc.young_rate3)
    phase_out_rate = value(law.cdctc.young_po_rate3)
    excess = landing_exactly_on(
        rate, lambda income: income * phase_out_rate, rate / phase_out_rate
    )
    return {
        **ONE_YOUNG_CHILD,
        "dep_age1": 3.0,
        "wages_base1": value(law.cdctc.young_po_thresh3) + excess,
        "care_exp": 2_000.0,
        "r.cdctc_takeup": CLAIMS_THE_CARE_CREDIT,
    }


def child_credit_phase_out_start(law: YearLaw) -> Overrides:
    """AGI exactly where the child credit starts phasing out."""
    return {**ONE_YOUNG_CHILD, "wages_base1": value(law.ctc.po_thresh1)}


def child_credit_phase_out_end(law: YearLaw) -> Overrides:
    """AGI exactly where the child credit runs out."""
    credit = value(law.ctc.value_young1)
    phase_out_rate = value(law.ctc.po_rate1)
    excess = landing_exactly_on(
        credit, lambda income: income * phase_out_rate, credit / phase_out_rate
    )
    return {**ONE_YOUNG_CHILD, "wages_base1": value(law.ctc.po_thresh1) + excess}


def child_credit_refundable_phase_in(law: YearLaw) -> Overrides:
    """Earned income exactly at the threshold where the refundable credit starts."""
    return {**ONE_YOUNG_CHILD, "wages_base1": value(law.ctc.pi_thresh)}


def child_credit_refund_cap(law: YearLaw) -> Overrides:
    """Earned income exactly where the refundable child credit reaches its cap."""
    cap = value(law.ctc.max_refund_young)
    rate = value(law.ctc.pi_rate)
    excess = landing_exactly_on(cap, lambda income: income * rate, cap / rate)
    return {**ONE_YOUNG_CHILD, "wages_base1": value(law.ctc.pi_thresh) + excess}


def eitc_phase_in_end(law: YearLaw) -> Overrides:
    """Earned income exactly at the end of the childless credit's phase-in."""
    return {"wages_base1": value(law.eitc.pi_end_0)}


def eitc_phase_out_start(law: YearLaw) -> Overrides:
    """Earned income exactly at the childless credit's phase-out threshold."""
    return {"wages_base1": value(law.eitc.po_thresh_0)}


def eitc_credit_exhausted(law: YearLaw) -> Overrides:
    """Earned income exactly where the childless credit runs out.

    The phase-in rate and the phase-out rate are the same number for a
    childless worker, so the credit reaches zero one plateau's width past the
    phase-out threshold.
    """
    return {
        "wages_base1": value(law.eitc.po_thresh_0) + value(law.eitc.pi_end_0)
    }


def eitc_investment_income_limit(law: YearLaw) -> Overrides:
    """Investment income exactly at the limit above which the credit is denied.

    The household has a child, because a childless worker with this much
    investment income has already phased out of the credit and would have
    nothing to lose.
    """
    return {
        **ONE_YOUNG_CHILD,
        "wages_base1": value(law.eitc.pi_end_1),
        "txbl_int": value(law.eitc.inv_inc_limit),
    }


def niit_threshold(law: YearLaw) -> Overrides:
    """AGI exactly at the net investment income tax threshold."""
    brackets = np.asarray(law.niit.brackets).reshape(-1)
    investment = 100_000.0
    return {
        "wages_base1": float(brackets[1]) - investment,
        "txbl_int": investment,
    }


# --------------------------------------------------------------------------
# The cases
# --------------------------------------------------------------------------

# The reasons a site cannot agree, written once so that the cases sharing a
# reason share the words as well.
PHASE_OUT_EXHAUSTED = (
    "the deduction has phased out to nothing, so it falls with a dollar of "
    "other income and rises with a dollar of the thing being deducted; "
    "such a corner is left on the default spelling"
)
EXPENSE_FLOOR = (
    "the amount is an expense measured against a floor that is a share of "
    "income, so it falls with a dollar of income and rises with a dollar of "
    "the expense; such a corner is left on the default spelling"
)
EITC_TENT = (
    "the credit rises over its phase-in range and falls over its phase-out "
    "range, so where it runs out there is no more-income side; the rising "
    "reading is kept, which reports the phase-out rate to a household whose "
    "credit is already gone"
)
INVESTMENT_INCOME_CLIFF = (
    "the earned income credit is denied outright once investment income "
    "exceeds the limit, so the next dollar costs the whole credit"
)
SELF_EMPLOYMENT_CLIFF = (
    "self-employment tax applies only above the reporting threshold, so the "
    "next dollar brings the whole taxable share of self-employment earnings "
    "into the tax"
)
MINIMUM_DEDUCTION_CLIFF = (
    "the minimum section 199A deduction is offered only above a stated amount "
    "of qualified business income, so the next dollar brings the whole minimum"
)
PHANTOM_PHASE_OUT = (
    "the household holds none of the thing being phased out and sits exactly "
    "on its threshold, so the derivative reports the phase-out rate on an "
    "amount of zero"
)
EXPENSE_CEILING = (
    "the amount is an expense measured against a ceiling that is a share of "
    "income, so the ceiling rises with a dollar of income while the expense "
    "stands still, and the expense rises against the ceiling with a dollar of "
    "the expense; such a corner is left on the default spelling"
)
CREDITS_CAPPED_AT_TAX = (
    "the whole of the nonrefundable credits is capped at pre-credit tax, and "
    "both sides of that cap are tax computations that move with income, so "
    "which grows faster differs from one household to the next; the site is "
    "left on the default spelling and says so in taxes/calc/liab.py"
)
DEDUCTION_LIMITED_BY_TAXABLE_INCOME = (
    "the section 199A deduction is capped at a share of taxable income, so "
    "the cap rises with a dollar of wages while the deduction stands still, "
    "and the deduction rises against the cap with a dollar of business "
    "income; such a corner is left on the default spelling"
)
STEPPED_PHASE_OUT = (
    "the credit worksheet rounds income above the threshold up to the next "
    "statutory step, so the credit falls in jumps and has no slope between "
    "them; the default configuration switches the rounding off"
)

# The phase-out share multiplies an amount that also moves with income, so the
# tax is genuinely curved over the dollar and the one-sided rerun carries a
# second-order term the derivative does not.
CURVED = 1e-4

CASES: tuple[Case, ...] = (
    Case(
        "the medical expense floor at a share of AGI",
        medical_floor,
        {"wages_base1": AGREE, "med_exp": EXPENSE_FLOOR},
    ),
    Case(
        "the charitable contribution floor at a share of AGI",
        charitable_floor,
        {"wages_base1": EXPENSE_FLOOR, "char_cash": AGREE},
    ),
    Case(
        "the AGI ceiling on deductible cash gifts",
        charitable_cash_ceiling,
        {"wages_base1": EXPENSE_CEILING, "char_cash": AGREE},
    ),
    Case(
        "the AGI ceiling on deductible noncash gifts",
        charitable_noncash_ceiling,
        {"wages_base1": EXPENSE_CEILING, "char_noncash": AGREE},
    ),
    Case(
        "the state and local tax deduction limit",
        salt_cap,
        {
            "wages_base1": AGREE,
            "salt_prop": AGREE,
            "salt_inc_sales": AGREE,
            "salt_pers": AGREE,
        },
    ),
    Case(
        "the income where the state and local limit starts phasing down",
        salt_phase_down_start,
        {"wages_base1": AGREE, "salt_prop": AGREE},
    ),
    Case(
        "the mortgage balance limit on deductible interest",
        mortgage_balance_at_the_limit,
        {"wages_base1": AGREE, "first_mort_int": AGREE},
    ),
    Case(
        "the income above which itemized deductions lose tax value",
        itemized_tax_value_threshold,
        {"wages_base1": AGREE},
    ),
    Case(
        "the itemized tax-value limitation for an itemizer with only state taxes",
        itemizer_with_nothing_but_state_and_local_taxes,
        {"first_mort_int": AGREE, "med_exp": AGREE},
    ),
    Case(
        "the boundary between itemizing and taking the standard deduction",
        itemize_or_standard,
        {"wages_base1": AGREE, "salt_prop": AGREE, "med_exp": AGREE},
    ),
    Case(
        "the dependent standard deduction floor",
        dependent_standard_deduction_floor,
        {"wages_base1": AGREE},
    ),
    Case(
        "the dependent standard deduction at the full amount",
        dependent_standard_deduction_cap,
        {"wages_base1": AGREE},
    ),
    Case(
        "the dollar limit on the car loan interest deduction",
        auto_interest_cap,
        {"wages_base1": AGREE, "auto_int_exp": AGREE},
    ),
    Case(
        "the income where the car loan interest deduction starts phasing out",
        auto_interest_phase_out_start,
        {"wages_base1": AGREE, "auto_int_exp": AGREE},
    ),
    Case(
        "the income where the car loan interest deduction runs out",
        auto_interest_phase_out_end,
        {"wages_base1": AGREE, "auto_int_exp": PHASE_OUT_EXHAUSTED},
    ),
    Case(
        "the second Social Security inclusion threshold",
        social_security_second_tier,
        {"wages_base1": AGREE, "gross_ss": AGREE, "txbl_int": AGREE},
    ),
    Case(
        "the third Social Security inclusion threshold",
        social_security_third_tier,
        {"wages_base1": AGREE, "gross_ss": AGREE, "txbl_int": AGREE},
    ),
    Case(
        "the zero floor on each kind of business income's component deduction",
        qbi_component_floor,
        {
            "part_active": AGREE,
            "part_passive": AGREE,
            "sole_prop1": AGREE,
            "sole_prop2": AGREE,
            "scorp_active": AGREE,
            "scorp_passive": AGREE,
            "rent": AGREE,
        },
    ),
    Case(
        "the taxable income where the section 199A wage limit phases in",
        qbi_phase_out_start,
        {"scorp_active": AGREE, "wages_base1": AGREE},
        tolerance=CURVED,
    ),
    Case(
        "the taxable income where the section 199A wage limit applies in full",
        qbi_phase_out_end,
        {"scorp_active": AGREE, "wages_base1": AGREE},
        tolerance=CURVED,
    ),
    Case(
        "the section 199A wage and property limit",
        qbi_wage_limit,
        {"scorp_active": AGREE, "wages_base1": AGREE},
        tolerance=CURVED,
    ),
    Case(
        "the section 199A limit at a share of taxable income",
        qbi_taxable_income_limit,
        {
            "scorp_active": AGREE,
            "wages_base1": DEDUCTION_LIMITED_BY_TAXABLE_INCOME,
        },
    ),
    Case(
        "the section 199A minimum deduction floor",
        qbi_minimum_deduction_floor,
        {"scorp_active": AGREE},
    ),
    Case(
        "the business income the section 199A minimum deduction requires",
        qbi_minimum_deduction_income,
        {"scorp_active": MINIMUM_DEDUCTION_CLIFF},
    ),
    Case(
        "the dollar limit on the tip deduction",
        tip_deduction_cap,
        {"tips1": AGREE, "wages_base1": AGREE},
    ),
    Case(
        "the income where the tip deduction starts phasing out",
        tip_deduction_phase_out_start,
        {"tips1": AGREE, "wages_base1": AGREE},
    ),
    Case(
        "the income where the tip deduction runs out",
        tip_deduction_phase_out_end,
        {"tips1": PHASE_OUT_EXHAUSTED, "wages_base1": AGREE},
    ),
    Case(
        "the dollar limit on the overtime deduction",
        overtime_deduction_cap,
        {"ot1": AGREE, "wages_base1": AGREE},
    ),
    Case(
        "the income where the overtime deduction starts phasing out",
        overtime_deduction_phase_out_start,
        {"ot1": AGREE, "wages_base1": AGREE},
    ),
    Case(
        "the below-the-line phase-out threshold with nothing to phase out",
        phase_out_threshold_with_nothing_to_phase_out,
        # The tip and overtime deductions share this threshold, so a dollar of
        # tips runs into the overtime deduction's phantom phase-out as well as
        # its own.
        {
            "wages_base1": PHANTOM_PHASE_OUT,
            "tips1": PHANTOM_PHASE_OUT,
            "ot1": PHANTOM_PHASE_OUT,
        },
    ),
    Case(
        "the newborn bonus phase-out threshold",
        newborn_bonus_phase_out_threshold,
        {"wages_base1": PHANTOM_PHASE_OUT, "txbl_int": PHANTOM_PHASE_OUT},
        keep_the_newborn_bonus_phase_out=True,
    ),
    Case(
        "the income where the senior deduction starts phasing out",
        senior_deduction_phase_out_start,
        {"wages_base1": AGREE, "txbl_int": AGREE},
    ),
    Case(
        "the income where the senior deduction runs out",
        senior_deduction_phase_out_end,
        {"wages_base1": AGREE, "txbl_int": AGREE},
    ),
    Case(
        "the annual capital loss deduction limit",
        capital_loss_limit,
        {"kg_st": AGREE, "kg_lt": AGREE},
    ),
    Case(
        "a household with exactly no net capital gain",
        zero_net_capital_gain,
        {"kg_st": AGREE, "kg_lt": AGREE},
    ),
    Case(
        "net capital gain for a household with no short-term gain",
        net_capital_gain_with_no_short_term,
        {"kg_st": AGREE, "kg_lt": AGREE},
    ),
    Case(
        "an ordinary bracket edge",
        ordinary_bracket_edge,
        {"wages_base1": AGREE, "txbl_int": AGREE},
    ),
    Case(
        "a preferred rate bracket edge",
        preferred_bracket_edge,
        {"kg_lt": AGREE, "div_pref": AGREE},
    ),
    Case(
        "a nonrefundable credit capped at a liability of nothing",
        nonrefundable_credit_capped_at_no_liability,
        {"kg_lt": CREDITS_CAPPED_AT_TAX, "div_pref": CREDITS_CAPPED_AT_TAX},
    ),
    Case(
        "the Schedule D cap for a household with no preferred income",
        schedule_d_cap_with_no_preferred_income,
        {"kg_lt": AGREE, "div_pref": AGREE, "wages_base1": AGREE},
    ),
    Case(
        "the special gain stacking caps for a household with no preferred gain",
        special_gain_stacking_with_no_preferred_gain,
        {"kg_lt": AGREE, "div_pref": AGREE},
    ),
    Case(
        "the income where the alternative minimum tax exemption phases out",
        amt_exemption_phase_out_start,
        {"wages_base1": AGREE, "txbl_int": AGREE},
    ),
    Case(
        "the Social Security taxable maximum",
        oasdi_taxable_maximum,
        {"wages_base1": AGREE, "tips1": AGREE, "ot1": AGREE},
    ),
    Case(
        "the Social Security taxable maximum for the second earner",
        oasdi_taxable_maximum_second_earner,
        {"wages_base2": AGREE},
        filing_status=2,
    ),
    Case(
        "the additional Medicare tax threshold",
        additional_medicare_threshold,
        {"wages_base1": AGREE},
    ),
    Case(
        "the self-employment tax reporting threshold",
        self_employment_reporting_threshold,
        {"sole_prop1": SELF_EMPLOYMENT_CLIFF},
    ),
    Case(
        "self-employment earnings stacked on a filled Social Security base",
        seca_stacked_on_a_filled_wage_base,
        {"sole_prop1": AGREE, "part_active": AGREE},
    ),
    Case(
        "the care credit's earned income limit for a household with neither",
        care_credit_earned_income_cap,
        {"wages_base1": AGREE, "txbl_int": AGREE},
    ),
    Case(
        "the care credit's expense limit",
        care_credit_expense_limit,
        {"wages_base1": AGREE},
    ),
    Case(
        "the income where the lowest care credit rate layer runs out",
        care_credit_rate_phase_out_end,
        {"wages_base1": AGREE},
    ),
    Case(
        "the income where the child credit starts phasing out",
        child_credit_phase_out_start,
        {"wages_base1": AGREE, "txbl_int": AGREE},
    ),
    Case(
        "the income where the child credit runs out",
        child_credit_phase_out_end,
        {"wages_base1": AGREE, "txbl_int": AGREE},
    ),
    Case(
        "the child credit's phase-out threshold with the rounding switched on",
        child_credit_phase_out_start,
        {"wages_base1": STEPPED_PHASE_OUT, "txbl_int": STEPPED_PHASE_OUT},
        rounded_phase_outs=True,
    ),
    Case(
        "the earned income where the refundable child credit phases in",
        child_credit_refundable_phase_in,
        {"wages_base1": AGREE},
    ),
    Case(
        "the refundable child credit's per-child cap",
        child_credit_refund_cap,
        {"wages_base1": AGREE},
    ),
    Case(
        "the end of the childless earned income credit's phase-in",
        eitc_phase_in_end,
        {"wages_base1": AGREE},
    ),
    Case(
        "the childless earned income credit's phase-out threshold",
        eitc_phase_out_start,
        {"wages_base1": AGREE},
    ),
    Case(
        "the earnings where the childless earned income credit runs out",
        eitc_credit_exhausted,
        {"wages_base1": EITC_TENT},
    ),
    Case(
        "the earned income credit's investment income limit",
        eitc_investment_income_limit,
        {"txbl_int": INVESTMENT_INCOME_CLIFF, "div_pref": INVESTMENT_INCOME_CLIFF},
    ),
    Case(
        "the net investment income tax threshold",
        niit_threshold,
        {"wages_base1": AGREE, "txbl_int": AGREE},
    ),
)


def both_measures(records, law, dollar: str, outcome) -> tuple[float, float]:
    """The derivative of the tax outcome, and what the next dollar really costs."""
    direction = marginal_rates.PRICED_DOLLARS[dollar]
    gradients = individual_income_tax.derivative(
        records,
        law,
        outcome=outcome,
        input_names=tuple(direction),
        double_run=False,
    )
    slope = sum(moved * gradients[column] for column, moved in direction.items())
    rerun = individual_income_tax.dollar_increment(
        records, law, direction, outcome=outcome, double_run=False
    )
    return (
        float(np.asarray(slope).reshape(-1)[0]),
        float(np.asarray(rerun).reshape(-1)[0]),
    )


def test_every_case_prices_a_dollar_the_model_publishes():
    """A misspelled priced dollar would leave a site unchecked."""
    for case in CASES:
        unknown = set(case.expected) - set(marginal_rates.PRICED_DOLLARS)
        assert not unknown, f"{case.site}: {sorted(unknown)}"


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.site)
def test_the_derivative_matches_the_next_dollar_on_every_threshold(
    compiled_baseline, case: Case
):
    """Both measures of the rate on one threshold, under both prices."""
    law = law_for(compiled_baseline, [case.filing_status])
    if case.rounded_phase_outs:
        law = rounded_phase_outs(law)
    if not case.keep_the_newborn_bonus_phase_out:
        law = without_the_newborn_bonus_phase_out(law)
    overrides = dict(BASE)
    overrides.update(case.build(law))
    records = synthetic_records([case.filing_status], **overrides)

    for price in marginal_rates.PRICES:
        for dollar, expectation in case.expected.items():
            slope, rerun = both_measures(records, law, dollar, price.outcome)
            priced = f"{case.site}, a dollar of {dollar}"
            in_price = price.ending or " (income tax only)"
            if expectation == AGREE:
                assert abs(slope - rerun) <= case.tolerance, (
                    f"{priced}{in_price}: the derivative says {slope!r} and the "
                    f"next dollar costs {rerun!r}. The model treats a household "
                    "on the threshold as having crossed it toward more income, "
                    "so these are the same number."
                )
            else:
                assert abs(slope - rerun) > DISAGREEMENT, (
                    f"{priced}{in_price}: the two measures agree at {slope!r}, "
                    "but they cannot at this site, because "
                    f"{expectation}."
                )
