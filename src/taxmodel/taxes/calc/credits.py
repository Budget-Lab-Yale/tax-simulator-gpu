"""Individual income-tax credits in their statutory stacking order."""

from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp

from taxmodel.taxes.calc._corners import (
    capped_at,
    floored_at,
    phase_out_share,
    phased_out,
)
from taxmodel.taxes.calc._schedules import schedule_tax

Array = jnp.ndarray


@dataclass(frozen=True)
class CdctcLaw:
    exp_limit: Array
    n_dep_limit: Array
    young_age_limit: Array
    old_age_limit: Array
    young_rate1: Array
    young_rate2: Array
    young_rate3: Array
    young_po_thresh1: Array
    young_po_thresh2: Array
    young_po_thresh3: Array
    young_po_rate1: Array
    young_po_rate2: Array
    young_po_rate3: Array
    old_rate1: Array
    old_rate2: Array
    old_rate3: Array
    old_po_thresh1: Array
    old_po_thresh2: Array
    old_po_thresh3: Array
    old_po_rate1: Array
    old_po_rate2: Array
    old_po_rate3: Array
    discrete_step: Array
    refundable: Array


@dataclass(frozen=True)
class EducationCreditLaw:
    aoc_rates: Array
    aoc_brackets: Array
    aoc_po_thresh: Array
    aoc_po_range: Array
    aoc_refund_rate: Array
    llc_rates: Array
    llc_brackets: Array
    llc_po_thresh: Array
    llc_po_range: Array


@dataclass(frozen=True)
class SaversCreditLaw:
    repeal: Array
    refundable: Array


@dataclass(frozen=True)
class CaregiverCreditLaw:
    exp_floor: Array
    min_ei: Array
    max_credit: Array
    credit_rate: Array
    po_thresh: Array
    po_rate: Array
    refundable: Array


@dataclass(frozen=True)
class CtcLaw:
    young_age_limit: Array
    old_age_limit: Array
    need_ssn: Array
    young_level: Array
    value_young1: Array
    value_young2: Array
    bonus_young1: Array
    bonus_young2: Array
    value_old1: Array
    value_old2: Array
    value_other: Array
    po_thresh1: Array
    po_thresh2: Array
    po_type: Array
    po_rate1: Array
    po_rate2: Array
    po_rate_other: Array
    po_range1: Array
    po_range2: Array
    po_range_other: Array
    po_sequential: Array
    po_discrete: Array
    po_discrete_step: Array
    mfs_eligible: Array
    min_refund_level: Array
    min_refund_young: Array
    min_refund_old: Array
    min_refund_share: Array
    max_refund_young: Array
    max_refund_old: Array
    ei_prior_yr: Array
    pi_thresh: Array
    pi_type: Array
    pi_rate: Array
    pi_range: Array
    baby_bonus_in_credit: Array
    baby_bonus: Array
    baby_bonus_pi_rate: Array
    baby_bonus_po_rate: Array
    baby_bonus_po_thresh: Array


@dataclass(frozen=True)
class EitcLaw:
    pi_rate_0: Array
    pi_rate_1: Array
    pi_rate_2: Array
    pi_rate_3: Array
    po_rate_0: Array
    po_rate_1: Array
    po_rate_2: Array
    po_rate_3: Array
    pi_end_0: Array
    pi_end_1: Array
    pi_end_2: Array
    pi_end_3: Array
    po_thresh_0: Array
    po_thresh_1: Array
    po_thresh_2: Array
    po_thresh_3: Array
    inv_inc_limit: Array
    min_age: Array
    max_age: Array
    ei_prior_yr: Array
    mfs_eligible: Array
    parent_precert: Array


@dataclass(frozen=True)
class RebateLaw:
    dep_age_limit: Array
    dep_ssn_req: Array
    value: Array
    value_dep: Array
    po_thresh: Array
    po_type: Array
    po_range: Array
    po_rate: Array


@dataclass(frozen=True)
class WageSubsidyLaw:
    pi_rate: Array
    pi_end: Array
    po_rate: Array
    po_thresh: Array


def _per_dependent(records: Mapping[str, Array], field: str) -> Array:
    # The calculator observes at most three dependents. Each per-dependent fact
    # arrives as three columns named for the dependent's position, and this puts
    # them on a trailing axis so the count can be taken across dependents.
    return jnp.stack(tuple(records[f"{field}{n}"] for n in (1, 2, 3)), axis=-1)


def _count_qualifying_ages(records: Mapping[str, Array], age_limit: Array) -> Array:
    # Missing ages are excluded from the age-qualified count.
    ages = _per_dependent(records, "dep_age")
    return jnp.sum(
        (~jnp.isnan(ages)) & (ages <= jnp.asarray(age_limit)[..., None]), axis=-1
    )


def _rounded_excess(
    income: Array, threshold: Array, step: Array, discrete: Array | None = None
) -> Array:
    # A rounded phaseout treats any positive fraction of a statutory step as a
    # full step, as on the credit worksheet. Two law parameters turn the
    # rounding off: `discrete`, the switch the child credit carries, and a
    # `step` of zero, which is how the child-care credit asks for a continuous
    # phaseout. With the rounding off the derivative of the credit is the
    # statutory phaseout rate; with it on, the derivative is zero.
    excess = floored_at(income - threshold, 0.0)
    safe_step = jnp.where(step > 0, step, 1.0)
    rounded = jnp.where(step > 0, jnp.ceil(excess / safe_step) * safe_step, excess)
    if discrete is None:
        return rounded
    return jnp.where(discrete == 1, rounded, excess)


def _phased_down_rate(
    agi: Array,
    rates: tuple[Array, ...],
    thresholds: tuple[Array, ...],
    po_rates: tuple[Array, ...],
    step: Array,
) -> Array:
    # A child-care credit rate is declared as up to three layers, each with its
    # own phaseout threshold and phaseout rate. A layer that has phased out
    # entirely contributes nothing, and the surviving layers are added.
    total = 0.0
    for rate, threshold, po_rate in zip(rates, thresholds, po_rates, strict=True):
        total = total + floored_at(
            rate - _rounded_excess(agi, threshold, step) * po_rate,
            0.0,
            sinks_against_the_limit=True,
        )
    return total


def calc_cdctc(records: Mapping[str, Array], law: CdctcLaw) -> dict[str, Array]:
    """Return refundable and nonrefundable child/dependent-care credits."""
    # The dependent count is divided into young and older groups because their
    # expense limits and AGI phaseout schedules may differ under a reform.
    young_qualified = _count_qualifying_ages(records, law.young_age_limit)
    n_young = capped_at(young_qualified, law.n_dep_limit)
    # The older group is netted against the uncapped young count and only then
    # capped by the dependent limit. The two counts differ
    # only where the young age limit exceeds the older one, which no current
    # law does but a reform could.
    n_old = capped_at(
        _count_qualifying_ages(records, law.old_age_limit) - young_qualified,
        law.n_dep_limit - n_young,
    )
    # A joint return is limited by the lower earner. When the two earners have
    # exactly the same earnings there is no "more income" side: another dollar
    # to either one of them alone leaves the lower of the two where it was.
    lower_earner_income = jnp.minimum(records["ei1"], records["ei2"])  # tie: two spouses' earnings, neither is the income side
    earned_income_limit = floored_at(
        jnp.where(
            records["filing_status"] == 2,
            lower_earner_income,
            records["ei1"],
        ),
        0.0,
    )
    # A joint return is limited by the lower earner's income; the one-earner
    # rule applies to every other filing status.
    #
    # The cap moves with earnings and the expenses do not, so the expenses sink
    # against the cap and a tie takes the flat branch.
    young_expenses = capped_at(
        capped_at(records["care_exp"], n_young * law.exp_limit),
        earned_income_limit,
        sinks_against_the_limit=True,
    )
    young_rate = _phased_down_rate(
        records["agi"],
        (law.young_rate1, law.young_rate2, law.young_rate3),
        (law.young_po_thresh1, law.young_po_thresh2, law.young_po_thresh3),
        (law.young_po_rate1, law.young_po_rate2, law.young_po_rate3),
        law.discrete_step,
    )
    # Young dependents use the shared expense and earned-income caps first;
    # older dependents may use only the remainder of both caps.
    old_expenses = capped_at(
        records["care_exp"] - young_expenses, n_old * law.exp_limit
    )
    # Same shape as the young-dependent cap above: the remaining earned-income
    # room moves with income, the expenses do not.
    old_expenses = capped_at(
        old_expenses, earned_income_limit - young_expenses, sinks_against_the_limit=True
    )
    old_rate = _phased_down_rate(
        records["agi"],
        (law.old_rate1, law.old_rate2, law.old_rate3),
        (law.old_po_thresh1, law.old_po_thresh2, law.old_po_thresh3),
        (law.old_po_rate1, law.old_po_rate2, law.old_po_rate3),
        law.discrete_step,
    )
    credit = young_expenses * young_rate + old_expenses * old_rate
    # A nonrefundable CDCTC is limited by liability remaining after the foreign
    # tax credit, its position in the order nonrefundable credits are applied.
    liability = floored_at(records["liab_bc"] - records["ftc"], 0.0)
    # The 90% take-up draw is calibrated to aggregate 2019 CDCTC claims.
    takeup = records["r.cdctc_takeup"] < 0.9
    return {
        # Same cap as the other nonrefundable credits: the liability left to
        # the credit rises with income and the credit does not, so a household
        # sitting exactly on the cap takes the flat branch.
        "cdctc_nonref": jnp.where(
            law.refundable == 1,
            0.0,
            capped_at(credit, liability, sinks_against_the_limit=True),
        )
        * takeup,
        "cdctc_ref": jnp.where(law.refundable == 0, 0.0, credit) * takeup,
    }


def calc_ed_cred(
    records: Mapping[str, Array], law: EducationCreditLaw
) -> dict[str, Array]:
    """Return American Opportunity and Lifetime Learning credit allocations."""
    # The education-expense schedules calculate each credit before its own AGI
    # phaseout; only the American Opportunity amount has a refundable share.
    aoc = schedule_tax(records["aoc_exp"], law.aoc_brackets, law.aoc_rates)
    aoc *= 1.0 - phase_out_share(records["agi"], law.aoc_po_thresh, law.aoc_po_range)
    aoc_ref = aoc * law.aoc_refund_rate
    llc = schedule_tax(records["llc_exp"], law.llc_brackets, law.llc_rates)
    llc *= 1.0 - phase_out_share(records["agi"], law.llc_po_thresh, law.llc_po_range)
    liability = floored_at(
        records["liab_bc"]
        - records["ftc"]
        - records["cdctc_nonref"]
        - records["old_cred"],
        0.0,
    )
    # The nonrefundable education amount comes after the child-care and elderly
    # credits in the credit waterfall, leaving later credits only the remaining
    # liability.
    #
    # The cap is the thing that moves with income here, not the credit: a
    # household with no education expenses has no credit and it does not grow
    # when the household earns another dollar. A household with neither a
    # credit nor any liability left sits exactly on this cap, so the credit
    # sinks against it and the tie takes the flat branch.
    return {
        "ed_nonref": capped_at(
            aoc - aoc_ref + llc, liability, sinks_against_the_limit=True
        ),
        "ed_ref": aoc_ref,
    }


def calc_savers_cred(
    records: Mapping[str, Array], law: SaversCreditLaw
) -> dict[str, Array]:
    """Allocate the PUF-abstracted saver credit by refundability."""
    # The PUF supplies the saver-credit amount; the calculator only applies a
    # policy repeal and allocates it at its place in the credit waterfall.
    credit = jnp.where(law.repeal == 0, records["savers_cred"], 0.0)
    liability = floored_at(
        records["liab_bc"]
        - records["ftc"]
        - records["cdctc_nonref"]
        - records["ed_nonref"],
        0.0,
    )
    return {
        # Same cap as the education credit above, and for the same reason: the
        # credit is a fixed amount from the data while the liability left to it
        # rises with income.
        "savers_nonref": jnp.where(
            law.refundable == 0,
            capped_at(credit, liability, sinks_against_the_limit=True),
            0.0,
        ),
        "savers_ref": jnp.where(law.refundable == 1, credit, 0.0),
    }


def calc_caregiver_cred(
    records: Mapping[str, Array], law: CaregiverCreditLaw
) -> dict[str, Array]:
    """Return the caregiver credit. The tax chain books it at zero instead."""
    # Each spouse must independently clear the earned-income test, and the
    # secondary spouse's expenses count only on a joint return.
    expense1 = floored_at(records["caregiver_exp1"] - law.exp_floor, 0.0)
    expense2 = floored_at(records["caregiver_exp2"] - law.exp_floor, 0.0)
    expense1 *= records["ei1"] > law.min_ei
    expense2 = jnp.where(
        records["filing_status"] == 2,
        expense2 * (records["ei2"] > law.min_ei),
        0.0,
    )
    credit = capped_at((expense1 + expense2) * law.credit_rate, law.max_credit)
    credit = phased_out(credit, records["agi"], law.po_thresh, law.po_rate)
    liability = floored_at(
        records["liab_bc"]
        - records["ftc"]
        - records["cdctc_nonref"]
        - records["ed_nonref"]
        - records["old_cred"],
        0.0,
    )
    return {
        # Same cap as the education credit, and for the same reason: the credit
        # falls with income through its own phase-out while the liability left
        # to it rises.
        "caregiver_cred_nonref": jnp.where(
            law.refundable == 1,
            0.0,
            capped_at(credit, liability, sinks_against_the_limit=True),
        ),
        "caregiver_cred_ref": jnp.where(law.refundable == 0, 0.0, credit),
    }


def _ctc_dependent_counts(
    records: Mapping[str, Array], law: CtcLaw
) -> tuple[Array, Array, Array]:
    # CTC eligibility is recorded separately from age and SSN information so a
    # reform can relax the SSN requirement without changing the child universe.
    ages = jnp.nan_to_num(_per_dependent(records, "dep_age"), nan=jnp.inf)
    ssns = jnp.nan_to_num(_per_dependent(records, "dep_ssn"), nan=0.0).astype(bool)
    eligible = jnp.nan_to_num(_per_dependent(records, "dep_ctc"), nan=0.0).astype(bool)
    no_ssn_required = jnp.asarray(law.need_ssn != 1)[..., None]
    qualifies = eligible & (ssns | no_ssn_required)
    young = jnp.sum(
        qualifies & (ages <= jnp.asarray(law.young_age_limit)[..., None]), axis=-1
    )
    old = (
        jnp.sum(
            qualifies & (ages <= jnp.asarray(law.old_age_limit)[..., None]), axis=-1
        )
        - young
    )
    newborn = jnp.sum(qualifies & (ages == 0), axis=-1)
    return young, old, newborn


def calc_ctc(records: Mapping[str, Array], law: CtcLaw) -> dict[str, Array]:
    """Return nonrefundable CTC, ACTC, and induced-filer status."""
    # Eligible children are divided among young, older, and newborn categories;
    # other dependents receive their distinct statutory amount.
    n_young, n_old, n_newborn = _ctc_dependent_counts(records, law)
    n_other = records["n_dep"] - n_young - n_old
    young_value1 = jnp.where(
        law.young_level == 1, law.value_young1, law.value_old1 + law.bonus_young1
    )
    young_value2 = jnp.where(
        law.young_level == 1, law.value_young2, law.value_old2 + law.bonus_young2
    )
    maximum1 = young_value1 * n_young + law.value_old1 * n_old
    maximum2 = (
        young_value2 * n_young
        + law.value_old2 * n_old
        + jnp.where(law.baby_bonus_in_credit == 1, n_newborn * law.baby_bonus, 0.0)
    )
    maximum_other = law.value_other * n_other
    maximum_baby = jnp.where(
        law.baby_bonus_in_credit == 1, 0.0, n_newborn * law.baby_bonus
    )
    excess1 = _rounded_excess(
        records["agi"], law.po_thresh1, law.po_discrete_step, law.po_discrete
    )
    excess2 = _rounded_excess(
        records["agi"], law.po_thresh2, law.po_discrete_step, law.po_discrete
    )
    rate1 = jnp.where(law.po_type == 1, law.po_rate1, maximum1 / law.po_range1)
    rate2 = jnp.where(law.po_type == 1, law.po_rate2, maximum2 / law.po_range2)
    rate_other = jnp.where(
        law.po_type == 1, law.po_rate_other, maximum_other / law.po_range_other
    )
    combined_after_first = floored_at(
        maximum1 + maximum2 - excess1 * rate1, 0.0, sinks_against_the_limit=True
    )
    # Sequential phaseout uses the first reduction against the combined child
    # credit, then assigns the residual between the two child-credit layers.
    value1 = jnp.where(
        law.po_sequential == 1,
        combined_after_first,
        floored_at(maximum1 - excess1 * rate1, 0.0, sinks_against_the_limit=True),
    )
    value2 = jnp.where(
        law.po_sequential == 1,
        floored_at(value1 - excess2 * rate2, 0.0, sinks_against_the_limit=True),
        floored_at(maximum2 - excess2 * rate2, 0.0, sinks_against_the_limit=True),
    )
    value1 = jnp.where(law.po_sequential == 1, value1 - value2, value1)
    value_other = jnp.where(
        law.po_sequential == 1,
        floored_at(maximum_other - excess2 * rate_other, 0.0, sinks_against_the_limit=True),
        floored_at(maximum_other - excess1 * rate_other, 0.0, sinks_against_the_limit=True),
    )
    filing_eligible = (records["filing_status"] != 3) | (law.mfs_eligible == 1)
    value1 *= filing_eligible
    value2 *= filing_eligible
    value_other *= filing_eligible
    prior_nonrefundable = (
        records["ftc"]
        + records["cdctc_nonref"]
        + records["ed_nonref"]
        + records["savers_nonref"]
        + records["old_cred"]
        + records["caregiver_cred_nonref"]
    )
    # CTC is nonrefundable only to the liability left after the credits applied
    # before it.
    liability = floored_at(records["liab_bc"] - prior_nonrefundable, 0.0)
    # The credit falls with income through its phase-out while the liability
    # left to it rises, so the credit sinks against this cap and a household
    # sitting exactly on it takes the flat branch. A household with no children
    # and no liability left is on the cap with zero on both sides.
    ctc_nonref = capped_at(
        value1 + value2 + value_other, liability, sinks_against_the_limit=True
    )
    remaining_ctc = floored_at(
        value1 + value2 - floored_at(ctc_nonref - value_other, 0.0), 0.0
    )
    minimum_refund = jnp.where(
        law.min_refund_level == 1,
        n_young * law.min_refund_young + n_old * law.min_refund_old,
        law.min_refund_share * (maximum1 + maximum2),
    )
    maximum_refund_young = jnp.where(
        jnp.isinf(law.max_refund_young),
        jnp.inf,
        n_young * law.max_refund_young,
    )
    maximum_refund_old = jnp.where(
        jnp.isinf(law.max_refund_old), jnp.inf, n_old * law.max_refund_old
    )
    qualifying_ei = jnp.where(
        law.ei_prior_yr == 1,
        floored_at(records["ei"], records["ei_prior_yr"]),
        records["ei"],
    )
    # Refundability can use prior-year earned income, but only when the law
    # activates that election.
    phase_in_rate = jnp.where(
        law.pi_type == 1, law.pi_rate, (maximum1 + maximum2) / law.pi_range
    )
    ctc_ref = capped_at(
        minimum_refund
        + floored_at(qualifying_ei - law.pi_thresh, 0.0) * phase_in_rate,
        capped_at(remaining_ctc, maximum_refund_young + maximum_refund_old),
    )
    baby_bonus = capped_at(
        floored_at(qualifying_ei, 0.0) * law.baby_bonus_pi_rate, maximum_baby
    )
    # A separately refundable newborn bonus has its own earned-income phase-in
    # and AGI phaseout when it is not embedded in the child-credit maximum.
    baby_bonus = phased_out(
        baby_bonus, records["agi"], law.baby_bonus_po_thresh, law.baby_bonus_po_rate
    )
    ctc_ref += baby_bonus
    return {
        "ctc_nonref": ctc_nonref,
        "ctc_ref": ctc_ref,
        "become_filer_ctc": (
            (records["filer"] == 0) & (qualifying_ei == 0) & (ctc_ref > 0)
        ).astype(jnp.int64),
    }


def _eitc_parameter(child_count: Array, values: tuple[Array, ...]) -> Array:
    # EITC schedules top-code the child count at three qualifying children.
    # mode="clip" repeats that cap inside the lookup; it changes no value and
    # lets the whole chain compile under jax.jit (a traced index needs it).
    top_coded = jnp.minimum(child_count, 3)  # tie: an integer child count used as a lookup index, so no derivative flows
    return jnp.choose(top_coded.astype(jnp.int32), values, mode="clip")


def calc_eitc(records: Mapping[str, Array], law: EitcLaw) -> dict[str, Array]:
    """Return EITC after age, filing-status, investment-income, and precert rules."""
    # A worker without a qualifying child must satisfy the age test; a joint
    # return tests each spouse before adding that spouse's earnings.
    has_child = records["n_dep_eitc"] > 0
    qualifies1 = has_child | (
        (records["age1"] >= law.min_age) & (records["age1"] <= law.max_age)
    )
    qualifies2 = (~jnp.isnan(records["age2"])) & (
        has_child
        | ((records["age2"] >= law.min_age) & (records["age2"] <= law.max_age))
    )
    earned_income = records["ei1"] * qualifies1 + jnp.where(
        records["filing_status"] == 2, records["ei2"] * qualifies2, 0.0
    )
    # The dependent flag is compared with zero, so any nonzero value marks a
    # dependent.
    earned_income *= (records["dep_status"] == 0) & (
        (law.mfs_eligible == 1) | (records["filing_status"] != 3)
    )
    investment_income = (
        records["txbl_int"]
        + records["exempt_int"]
        + records["div_ord"]
        + records["div_pref"]
        + floored_at(records["txbl_kg"], 0.0)
        + floored_at(records["sch_e"] - records["part_scorp"], 0.0)
    )
    # Eligibility survives exactly at the investment-income limit and is lost
    # immediately above it.
    earned_income *= investment_income <= law.inv_inc_limit
    n_children = records["n_dep_eitc"]
    # Each child-count schedule has its own phase-in rate, plateau endpoint, and
    # phaseout threshold.
    pi_rate = _eitc_parameter(
        n_children, (law.pi_rate_0, law.pi_rate_1, law.pi_rate_2, law.pi_rate_3)
    )
    po_rate = _eitc_parameter(
        n_children, (law.po_rate_0, law.po_rate_1, law.po_rate_2, law.po_rate_3)
    )
    pi_end = _eitc_parameter(
        n_children, (law.pi_end_0, law.pi_end_1, law.pi_end_2, law.pi_end_3)
    )
    po_thresh = _eitc_parameter(
        n_children,
        (law.po_thresh_0, law.po_thresh_1, law.po_thresh_2, law.po_thresh_3),
    )

    def credit(ei: Array) -> Array:
        # The credit phases in on earned income but phases out against the larger
        # of earned income and AGI, as on the EITC worksheet.
        maximum = capped_at(ei, pi_end) * pi_rate
        phaseout_income = floored_at(records["agi"], ei)
        # This amount rises with earnings over the phase-in range and falls over
        # the phaseout range, so it hits zero twice and has no single more-income
        # side. It is left on the rising reading, which gives the phase-in rate to
        # a worker with exactly zero earnings -- a case that is everywhere in the
        # data -- at the cost of still reporting the phaseout rate to a household
        # sitting exactly where the credit runs out, where there are no records.
        return floored_at(
            maximum - floored_at(phaseout_income - po_thresh, 0.0) * po_rate,
            0.0,
        )

    eitc = credit(earned_income)
    prior_eitc = credit(records["ei_prior_yr"] * (earned_income > 0))
    eitc = jnp.where(law.ei_prior_yr == 1, floored_at(eitc, prior_eitc), eitc)
    # The 3.1% denial draw implements the parent-precertification score.
    denied_by_precert = (
        (law.parent_precert != 0)
        & (n_children > 0)
        & (records["r.eitc_precert"] < 0.031)
    )
    return {"eitc": jnp.where(denied_by_precert, 0.0, eitc)}


def calc_rebate(records: Mapping[str, Array], law: RebateLaw) -> dict[str, Array]:
    """Return the fully refundable per-person rebate and induced-filer status."""
    # Qualifying dependents must meet both the age condition and, when required,
    # the SSN condition; adults are counted from filing status.
    ages = _per_dependent(records, "dep_age")
    ssns = jnp.nan_to_num(_per_dependent(records, "dep_ssn"), nan=0.0).astype(bool)
    dependent_qualifies = (~jnp.isnan(ages)) & (
        ages <= jnp.asarray(law.dep_age_limit)[..., None]
    )
    dependent_qualifies &= ssns | (jnp.asarray(law.dep_ssn_req) == 0)[..., None]
    n_dependents = jnp.sum(dependent_qualifies, axis=-1)
    n_adults = 1 + (records["filing_status"] == 2).astype(jnp.int64)
    rebate = law.value * n_adults + law.value_dep * n_dependents
    rate_based_range = jnp.where(rebate == 0, jnp.inf, rebate / law.po_rate)
    phaseout_range = jnp.where(law.po_type == 1, rate_based_range, law.po_range)
    phaseout_share = phase_out_share(records["agi"], law.po_thresh, phaseout_range)
    # Dependent filers receive no rebate, preventing duplication with the
    # qualifying-dependent amount on another return.
    rebate = rebate * (1.0 - phaseout_share) * (records["dep_status"] == 0)
    return {
        "rebate": rebate,
        "become_filer_rebate": ((records["filer"] == 0) & (rebate > 0)).astype(
            jnp.int64
        ),
    }


def calc_wage_subsidy(
    records: Mapping[str, Array], law: WageSubsidyLaw
) -> dict[str, Array]:
    """Return the hypothetical subsidy calculated separately for each earner."""
    # The wage subsidy is an earner-level phase-in/phaseout schedule, not a
    # filing-unit credit, so spouses are calculated independently.
    # Like the EITC, this amount rises over the phase-in range and falls over
    # the phaseout range, so the floor below stays on the rising reading; see the
    # note in calc_eitc.
    return {
        "wage_subsidy1": _one_earner_wage_subsidy(records["ei1"], law),
        "wage_subsidy2": _one_earner_wage_subsidy(records["ei2"], law),
    }


def _one_earner_wage_subsidy(earnings: Array, law: WageSubsidyLaw) -> Array:
    maximum = capped_at(earnings, law.pi_end) * law.pi_rate
    return floored_at(
        maximum - floored_at(earnings - law.po_thresh, 0.0) * law.po_rate, 0.0
    )
