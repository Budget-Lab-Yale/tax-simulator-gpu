import jax.numpy as jnp
import numpy as np

from taxmodel.taxes.calc.credits import (
    CaregiverCreditLaw,
    CdctcLaw,
    CtcLaw,
    EducationCreditLaw,
    EitcLaw,
    RebateLaw,
    SaversCreditLaw,
    WageSubsidyLaw,
    calc_caregiver_cred,
    calc_cdctc,
    calc_ctc,
    calc_ed_cred,
    calc_eitc,
    calc_rebate,
    calc_savers_cred,
    calc_wage_subsidy,
)


def test_cdctc_shared_expense_cap_and_strict_takeup_boundary():
    law = CdctcLaw(
        3_000.0,
        2.0,
        5.0,
        12.0,
        0.5,
        0.0,
        0.0,
        100_000.0,
        100_000.0,
        100_000.0,
        0.0,
        0.0,
        0.0,
        0.25,
        0.0,
        0.0,
        100_000.0,
        100_000.0,
        100_000.0,
        0.0,
        0.0,
        0.0,
        1_000.0,
        0.0,
    )
    records = {
        "dep_age1": jnp.array([4.0, 4.0]),
        "dep_age2": jnp.array([10.0, 10.0]),
        "dep_age3": jnp.array([jnp.nan, jnp.nan]),
        "care_exp": jnp.array([5_000.0, 5_000.0]),
        "filing_status": jnp.array([2, 2]),
        "ei1": jnp.array([4_000.0, 4_000.0]),
        "ei2": jnp.array([4_000.0, 4_000.0]),
        "agi": jnp.zeros(2),
        "liab_bc": jnp.full(2, 2_000.0),
        "ftc": jnp.zeros(2),
        "r.cdctc_takeup": jnp.array([0.899, 0.9]),
    }
    result = calc_cdctc(records, law)

    # Young expenses use $3,000 first at 50%; the $4,000 earned-income cap
    # leaves $1,000 for the older dependent at 25%, producing $1,750.
    np.testing.assert_allclose(result["cdctc_nonref"], [1_750.0, 0.0])


def test_education_and_saver_credits_stack_against_liability():
    education_law = EducationCreditLaw(
        jnp.array([0.5]),
        jnp.array([0.0]),
        100_000.0,
        10_000.0,
        0.4,
        jnp.array([0.2]),
        jnp.array([0.0]),
        100_000.0,
        10_000.0,
    )
    records = {
        "agi": jnp.array([0.0]),
        "aoc_exp": jnp.array([2_000.0]),
        "llc_exp": jnp.array([1_000.0]),
        "liab_bc": jnp.array([700.0]),
        "ftc": jnp.zeros(1),
        "cdctc_nonref": jnp.zeros(1),
        "old_cred": jnp.zeros(1),
    }
    education = calc_ed_cred(records, education_law)

    # AOC is $1,000: $400 refundable and $600 nonrefundable. LLC is $200, but
    # the combined nonrefundable amount is capped by $700 of liability.
    np.testing.assert_allclose(education["ed_nonref"], [700.0])
    np.testing.assert_allclose(education["ed_ref"], [400.0])
    saver_records = {
        "liab_bc": jnp.array([700.0]),
        "ftc": jnp.zeros(1),
        "cdctc_nonref": jnp.zeros(1),
        "ed_nonref": education["ed_nonref"],
        "savers_cred": jnp.array([300.0]),
    }
    saver = calc_savers_cred(saver_records, SaversCreditLaw(0.0, 0.0))
    np.testing.assert_allclose(saver["savers_nonref"], [0.0])


def test_disabled_caregiver_credit_function_matches_worksheet():
    law = CaregiverCreditLaw(1_000.0, 2_000.0, 1_000.0, 0.2, 20_000.0, 0.05, 0.0)
    records = {
        "caregiver_exp1": jnp.array([6_000.0]),
        "caregiver_exp2": jnp.array([4_000.0]),
        "ei1": jnp.array([10_000.0]),
        "ei2": jnp.array([1_000.0]),
        "filing_status": jnp.array([2]),
        "agi": jnp.array([22_000.0]),
        "liab_bc": jnp.array([2_000.0]),
        "ftc": jnp.zeros(1),
        "cdctc_nonref": jnp.zeros(1),
        "ed_nonref": jnp.zeros(1),
        "old_cred": jnp.zeros(1),
    }
    result = calc_caregiver_cred(records, law)

    # Only filer 1 clears the EI test: ($6,000 - $1,000) × 20% = $1,000;
    # the $2,000 AGI excess reduces it by $100.
    np.testing.assert_allclose(result["caregiver_cred_nonref"], [900.0])


def _ctc_law():
    return CtcLaw(
        young_age_limit=5.0,
        old_age_limit=17.0,
        need_ssn=1.0,
        young_level=1.0,
        value_young1=2_000.0,
        value_young2=0.0,
        bonus_young1=0.0,
        bonus_young2=0.0,
        value_old1=2_000.0,
        value_old2=0.0,
        value_other=500.0,
        po_thresh1=100_000.0,
        po_thresh2=100_000.0,
        po_type=1.0,
        po_rate1=0.05,
        po_rate2=0.05,
        po_rate_other=0.05,
        po_range1=jnp.inf,
        po_range2=jnp.inf,
        po_range_other=jnp.inf,
        po_sequential=0.0,
        po_discrete=0.0,
        po_discrete_step=1_000.0,
        mfs_eligible=0.0,
        min_refund_level=1.0,
        min_refund_young=0.0,
        min_refund_old=0.0,
        min_refund_share=0.0,
        max_refund_young=1_600.0,
        max_refund_old=1_600.0,
        ei_prior_yr=0.0,
        pi_thresh=0.0,
        pi_type=1.0,
        pi_rate=0.2,
        pi_range=jnp.inf,
        baby_bonus_in_credit=1.0,
        baby_bonus=0.0,
        baby_bonus_pi_rate=0.0,
        baby_bonus_po_rate=0.0,
        baby_bonus_po_thresh=jnp.inf,
    )


def test_ctc_nonrefundable_then_refundable_ladder():
    records = {
        "dep_age1": jnp.array([5.0]),
        "dep_age2": jnp.array([jnp.nan]),
        "dep_age3": jnp.array([jnp.nan]),
        "dep_ssn1": jnp.array([1.0]),
        "dep_ssn2": jnp.array([jnp.nan]),
        "dep_ssn3": jnp.array([jnp.nan]),
        "dep_ctc1": jnp.array([1.0]),
        "dep_ctc2": jnp.array([jnp.nan]),
        "dep_ctc3": jnp.array([jnp.nan]),
        "n_dep": jnp.array([1.0]),
        "agi": jnp.array([10_000.0]),
        "filing_status": jnp.array([1]),
        "liab_bc": jnp.array([500.0]),
        "ftc": jnp.zeros(1),
        "cdctc_nonref": jnp.zeros(1),
        "ed_nonref": jnp.zeros(1),
        "savers_nonref": jnp.zeros(1),
        "old_cred": jnp.zeros(1),
        "caregiver_cred_nonref": jnp.zeros(1),
        "ei": jnp.array([10_000.0]),
        "ei_prior_yr": jnp.zeros(1),
        "filer": jnp.ones(1),
    }
    result = calc_ctc(records, _ctc_law())

    # $500 of the $2,000 credit offsets liability. The $1,500 remainder is below
    # both the $2,000 earned-income phase-in and the $1,600 per-child refund cap.
    np.testing.assert_allclose(result["ctc_nonref"], [500.0])
    np.testing.assert_allclose(result["ctc_ref"], [1_500.0])


def _eitc_law():
    return EitcLaw(
        0.1,
        0.4,
        0.4,
        0.4,
        0.1,
        0.2,
        0.2,
        0.2,
        5_000.0,
        10_000.0,
        10_000.0,
        10_000.0,
        10_000.0,
        20_000.0,
        20_000.0,
        20_000.0,
        1_000.0,
        25.0,
        64.0,
        0.0,
        0.0,
        0.0,
    )


def test_eitc_investment_income_cliff_and_batch_vectorization():
    records = {
        "dep_status": jnp.array([False, False]),
        "filing_status": jnp.array([1, 1]),
        "n_dep_eitc": jnp.array([1, 1]),
        "age1": jnp.array([30.0, 30.0]),
        "age2": jnp.array([jnp.nan, jnp.nan]),
        "ei1": jnp.array([10_000.0, 10_000.0]),
        "ei2": jnp.zeros(2),
        "txbl_int": jnp.array([1_000.0, 1_001.0]),
        "exempt_int": jnp.zeros(2),
        "div_ord": jnp.zeros(2),
        "div_pref": jnp.zeros(2),
        "txbl_kg": jnp.zeros(2),
        "sch_e": jnp.zeros(2),
        "part_scorp": jnp.zeros(2),
        "r.eitc_precert": jnp.ones(2),
        "agi": jnp.array([10_000.0, 10_000.0]),
        "ei_prior_yr": jnp.zeros(2),
    }
    actual = calc_eitc(records, _eitc_law())["eitc"]

    # Exactly $1,000 of investment income remains eligible for the $4,000
    # one-child maximum; one dollar more eliminates the credit.
    np.testing.assert_allclose(actual, [4_000.0, 0.0])
    expected = jnp.array(
        [
            calc_eitc({key: value[i] for key, value in records.items()}, _eitc_law())[
                "eitc"
            ]
            for i in range(2)
        ]
    )
    np.testing.assert_allclose(actual, expected)


def test_rebate_counts_adults_and_qualified_dependents_then_phases_out():
    law = RebateLaw(17.0, 1.0, 1_000.0, 500.0, 20_000.0, 1.0, jnp.inf, 0.10)
    records = {
        "dep_status": jnp.array([False]),
        "filing_status": jnp.array([2]),
        "dep_age1": jnp.array([10.0]),
        "dep_age2": jnp.array([jnp.nan]),
        "dep_age3": jnp.array([jnp.nan]),
        "dep_ssn1": jnp.array([1.0]),
        "dep_ssn2": jnp.array([jnp.nan]),
        "dep_ssn3": jnp.array([jnp.nan]),
        "agi": jnp.array([25_000.0]),
        "filer": jnp.array([0]),
    }
    result = calc_rebate(records, law)

    # Two adults and one dependent yield $2,500 before a 10% phaseout on the
    # $5,000 excess, leaving $2,000 and inducing the nonfiler to file.
    np.testing.assert_allclose(result["rebate"], [2_000.0])
    np.testing.assert_array_equal(result["become_filer_rebate"], [1])


def test_wage_subsidy_is_calculated_per_earner_and_vectorizes():
    law = WageSubsidyLaw(0.20, 10_000.0, 0.10, 15_000.0)
    records = {"ei1": jnp.array([10_000.0, 20_000.0]), "ei2": jnp.array([5_000.0, 0.0])}
    result = calc_wage_subsidy(records, law)

    # Earner 1 receives $2,000 at the plateau and $1,500 after phaseout;
    # earner 2 receives 20% of $5,000 and zero on the second record.
    np.testing.assert_allclose(result["wage_subsidy1"], [2_000.0, 1_500.0])
    np.testing.assert_allclose(result["wage_subsidy2"], [1_000.0, 0.0])


# The rebate treats a 0/1 integer dependent flag the same as a bool: a
# dependent filer gets nothing.
def test_rebate_dependent_flag_accepts_numeric_indicator():
    law = RebateLaw(
        dep_age_limit=17.0,
        dep_ssn_req=0.0,
        value=1_400.0,
        value_dep=1_400.0,
        po_thresh=75_000.0,
        po_type=1.0,
        po_range=jnp.inf,
        po_rate=0.05,
    )
    missing = jnp.array([jnp.nan, jnp.nan])
    unit = {
        "filing_status": jnp.array([1, 1]),
        "agi": jnp.array([0.0, 0.0]),
        "filer": jnp.array([0, 0]),
        "dep_age1": missing,
        "dep_age2": missing,
        "dep_age3": missing,
        "dep_ssn1": missing,
        "dep_ssn2": missing,
        "dep_ssn3": missing,
    }
    # A single nondependent adult gets the $1,400 rebate; a dependent filer
    # gets nothing. Supplying the indicator as an integer gives -1,400 / -2,800.
    np.testing.assert_allclose(
        calc_rebate({**unit, "dep_status": jnp.array([0, 1])}, law)["rebate"],
        [1_400.0, 0.0],
    )


# The count of older qualifying children is the total less the young count
# before the young count is capped at the dependent limit. The order matters
# only when the young age limit is above the old one, which a reform can set.
def test_cdctc_old_count_uses_uncapped_young_count():
    law = CdctcLaw(
        3_000.0, 2.0, 17.0, 12.0,
        0.35, 0.0, 0.0,
        jnp.inf, jnp.inf, jnp.inf,
        0.0, 0.0, 0.0,
        0.20, 0.0, 0.0,
        jnp.inf, jnp.inf, jnp.inf,
        0.0, 0.0, 0.0,
        1.0, 0.0,
    )
    unit = {
        "dep_age1": jnp.array([5.0]),
        "dep_age2": jnp.array([10.0]),
        "dep_age3": jnp.array([15.0]),
        "filing_status": jnp.array([1]),
        "ei1": jnp.array([100_000.0]),
        "ei2": jnp.array([jnp.nan]),
        "care_exp": jnp.array([10_000.0]),
        "agi": jnp.array([0.0]),
        "liab_bc": jnp.array([100_000.0]),
        "ftc": jnp.array([0.0]),
        "r.cdctc_takeup": jnp.array([0.0]),
    }
    # Three dependents all under the inverted young limit of 17, two under the
    # old limit of 12, dependent limit 2. R: n_young = min(3, 2) = 2 but
    # n_old = min(2 - 3, 2 - 2) = -1, so $6,000 of young expenses at 35% plus
    # -$3,000 of old expenses at 20% = $1,500. Python gets n_old = 0 and $2,100.
    np.testing.assert_allclose(calc_cdctc(unit, law)["cdctc_nonref"], [1_500.0])


def test_cdctc_phaseout_is_continuous_when_the_rounding_step_is_zero():
    """A rounding step of zero turns the worksheet's $2,000 steps off.

    One young dependent, $3,000 of expenses, a 35 percent rate that phases
    down by one point per $2,000 of AGI over $15,000. Rounded, a household
    $500 over the threshold loses a full step; continuous, it loses a quarter
    of one.
    """

    def law(step):
        return CdctcLaw(
            exp_limit=3_000.0,
            n_dep_limit=2.0,
            young_age_limit=5.0,
            old_age_limit=12.0,
            young_rate1=0.35,
            young_rate2=0.0,
            young_rate3=0.0,
            young_po_thresh1=15_000.0,
            young_po_thresh2=jnp.inf,
            young_po_thresh3=jnp.inf,
            young_po_rate1=0.01 / 2_000.0,
            young_po_rate2=0.0,
            young_po_rate3=0.0,
            old_rate1=0.0,
            old_rate2=0.0,
            old_rate3=0.0,
            old_po_thresh1=jnp.inf,
            old_po_thresh2=jnp.inf,
            old_po_thresh3=jnp.inf,
            old_po_rate1=0.0,
            old_po_rate2=0.0,
            old_po_rate3=0.0,
            discrete_step=step,
            refundable=0.0,
        )

    records = {
        "dep_age1": jnp.array([3.0]),
        "dep_age2": jnp.array([jnp.nan]),
        "dep_age3": jnp.array([jnp.nan]),
        "care_exp": jnp.array([3_000.0]),
        "filing_status": jnp.array([1]),
        "ei1": jnp.array([50_000.0]),
        "ei2": jnp.array([jnp.nan]),
        "agi": jnp.array([15_500.0]),
        "liab_bc": jnp.array([10_000.0]),
        "ftc": jnp.zeros(1),
        "r.cdctc_takeup": jnp.zeros(1),
    }
    rounded = calc_cdctc(records, law(2_000.0))["cdctc_nonref"]
    continuous = calc_cdctc(records, law(0.0))["cdctc_nonref"]
    np.testing.assert_allclose(rounded, [3_000.0 * 0.34])
    np.testing.assert_allclose(continuous, [3_000.0 * 0.3475])
