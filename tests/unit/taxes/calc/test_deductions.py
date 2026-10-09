import jax
import jax.numpy as jnp
import numpy as np

from taxmodel.taxes.calc.deductions import (
    BelowDeductionLaw,
    ItemizedDeductionLaw,
    PersonalExemptionLaw,
    QbiLaw,
    StandardDeductionLaw,
    calc_below_ded,
    calc_item_ded,
    calc_pe_ded,
    calc_qbi_ded,
    calc_std_ded,
)


def test_standard_deduction_and_personal_exemption_boundaries():
    out = calc_std_ded(
        {
            "dep_status": jnp.array([False, True]),
            "ei": jnp.array([0.0, 1000.0]),
            "age1": jnp.array([65.0, 20.0]),
            "age2": jnp.array([jnp.nan, jnp.nan]),
            "blind1": jnp.array([False, False]),
            "blind2": jnp.array([jnp.nan, jnp.nan]),
        },
        StandardDeductionLaw(10000.0, 1500.0, 1200.0, 400.0, 0.0),
    )
    # $10,000 + $1,500 age bonus; dependent: max($1,200, $1,000 + $400).
    np.testing.assert_allclose(out["std_ded"], [11500.0, 1400.0])
    pe = calc_pe_ded(
        {
            "filing_status": jnp.array([1.0]),
            "n_dep": jnp.array([0.0]),
            "agi": jnp.array([150.0]),
        },
        PersonalExemptionLaw(1000.0, 100.0, 100.0, 1.0, 0.25, 0.0, 1.0),
    )
    np.testing.assert_allclose(pe["pe_ded"], [500.0])


def test_standard_deduction_counts_each_age_and_blindness_bonus():
    records = {
        "dep_status": jnp.array([False]),
        "ei": jnp.array([0.0]),
        "age1": jnp.array([65.0]),
        "age2": jnp.array([70.0]),
        "blind1": jnp.array([True]),
        "blind2": jnp.array([True]),
    }
    law = StandardDeductionLaw(10_000.0, 1_500.0, 1_200.0, 400.0, 0.0)

    # Two aged spouses plus two blind spouses produce four $1,500 bonuses.
    np.testing.assert_allclose(calc_std_ded(records, law)["std_ded"], [16_000.0])


def test_qbi_limit_binds_and_is_slack():
    law = QbiLaw(0.2, 1e6, 1e6, 1e6, 1e6, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0)
    values = {
        "agi": jnp.array([10000.0, 100.0]),
        "std_ded": jnp.zeros(2),
        "item_ded": jnp.zeros(2),
        "pe_ded": jnp.zeros(2),
        "sole_prop1": jnp.array([1000.0, 1000.0]),
        "sole_prop2": jnp.zeros(2),
        "part": jnp.zeros(2),
        "scorp": jnp.zeros(2),
        "farm1": jnp.zeros(2),
        "farm2": jnp.zeros(2),
        "wagebill_sole_prop": jnp.zeros(2),
        "wagebill_part": jnp.zeros(2),
        "wagebill_scorp": jnp.zeros(2),
        "wagebill_farm": jnp.zeros(2),
        "sstb_sole_prop": jnp.zeros(2),
        "sstb_part": jnp.zeros(2),
        "sstb_scorp": jnp.zeros(2),
        "sstb_farm": jnp.zeros(2),
        "div_pref": jnp.zeros(2),
        "kg_pref": jnp.zeros(2),
    }
    # $1,000 × 20% = $200 is slack at $10,000 taxable income and capped at $100.
    np.testing.assert_allclose(calc_qbi_ded(values, law)["qbi_ded"], [200.0, 100.0])


def test_itemized_and_below_line_deductions():
    item_law = ItemizedDeductionLaw(
        0.1,
        jnp.inf,
        10000.0,
        jnp.array([jnp.inf]),
        jnp.array([750000.0]),
        jnp.inf,
        0.0,
        jnp.inf,
        0.02,
        jnp.inf,
        0.0,
        jnp.inf,
        1.0,
        1.0,
        jnp.inf,
        0.0,
        0.0,
        jnp.inf,
        0.0,
        jnp.inf,
        0.0,
    )
    item_data = {
        "agi": jnp.array([-100.0]),
        "med_exp": jnp.array([100.0]),
        "salt_inc_sales": jnp.array([200.0]),
        "salt_prop": jnp.zeros(1),
        "salt_pers": jnp.zeros(1),
        "first_mort_bal": jnp.array([100.0]),
        "second_mort_bal": jnp.zeros(1),
        "first_mort_year": jnp.array([jnp.inf]),
        "second_mort_year": jnp.array([jnp.inf]),
        "first_mort_int": jnp.array([10.0]),
        "second_mort_int": jnp.zeros(1),
        "inv_int_exp": jnp.zeros(1),
        "auto_int_exp": jnp.zeros(1),
        "char_cash": jnp.zeros(1),
        "char_noncash": jnp.zeros(1),
        "casualty_exp": jnp.zeros(1),
        "job_exp": jnp.zeros(1),
        "tax_prep_exp": jnp.zeros(1),
        "other_misc_exp": jnp.zeros(1),
        "other_item_exp": jnp.zeros(1),
    }
    # Negative AGI is floored at zero: $100 medical expense remains deductible.
    np.testing.assert_allclose(
        calc_item_ded(item_data, item_law)["med_item_ded"], [100.0]
    )

    # A zero reported balance makes all reported mortgage interest deductible.
    # Its derivative is zero until the balance reaches the statutory cap; it
    # must not be NaN merely because the worksheet once divided by zero.
    zero_balance = dict(item_data)
    zero_balance["first_mort_bal"] = jnp.zeros(1)

    def mortgage_deduction(first_balance):
        changed = dict(zero_balance)
        changed["first_mort_bal"] = first_balance
        return jnp.sum(calc_item_ded(changed, item_law)["mort_int_item_ded"])

    np.testing.assert_allclose(jax.grad(mortgage_deduction)(jnp.zeros(1)), [0.0])

    below_law = BelowDeductionLaw(
        2000.0,
        0.0,
        500.0,
        jnp.inf,
        1.0,
        0.0,
        1.0,
        2000.0,
        1.0,
        500.0,
        jnp.inf,
        1.0,
        0.0,
        1.0,
        1000.0,
        jnp.inf,
        0.0,
    )
    below_data = {
        "tips1": jnp.array([100.0]),
        "tips2": jnp.zeros(1),
        # $810 of ordinary pay plus $100 of tips and $90 of overtime is
        # $1,000 of wages for the earner, which is what the eligibility limit
        # is tested against.
        "wages_base1": jnp.array([810.0]),
        "wages_base2": jnp.zeros(1),
        "tips_lh1": jnp.ones(1),
        "tips_lh2": jnp.zeros(1),
        "agi": jnp.zeros(1),
        "ot1": jnp.array([90.0]),
        "ot2": jnp.zeros(1),
        "age1": jnp.array([65.0]),
        "age2": jnp.array([jnp.nan]),
    }
    below = calc_below_ded(below_data, below_law)
    # $90 of overtime divided by three is the FLSA half-only deduction.
    np.testing.assert_allclose(
        [below["tip_ded"][0], below["ot_ded"][0], below["senior_ded"][0]],
        [100.0, 30.0, 1000.0],
    )


def test_below_line_deduction_counts_both_seniors():
    law = BelowDeductionLaw(
        2_000.0,
        0.0,
        500.0,
        jnp.inf,
        1.0,
        0.0,
        1.0,
        2_000.0,
        1.0,
        500.0,
        jnp.inf,
        1.0,
        0.0,
        1.0,
        1_000.0,
        jnp.inf,
        0.0,
    )
    records = {
        "tips1": jnp.zeros(1),
        "tips2": jnp.zeros(1),
        "wages_base1": jnp.zeros(1),
        "wages_base2": jnp.zeros(1),
        "tips_lh1": jnp.zeros(1),
        "tips_lh2": jnp.zeros(1),
        "agi": jnp.zeros(1),
        "ot1": jnp.zeros(1),
        "ot2": jnp.zeros(1),
        "age1": jnp.array([65.0]),
        "age2": jnp.array([65.0]),
    }

    # Two qualifying spouses each receive the invented $1,000 senior deduction.
    np.testing.assert_allclose(calc_below_ded(records, law)["senior_ded"], [2_000.0])


# The dependent flag may arrive as a bool or as a 0/1 number; both give the
# dependent standard deduction.
def test_std_ded_dependent_flag_accepts_numeric_indicator():
    law = StandardDeductionLaw(
        value=14_600.0,
        bonus=1_950.0,
        bonus_other=0.0,
        dep_floor=1_300.0,
        dep_earned_bonus=450.0,
    )
    person = {
        "age1": jnp.array([70.0]),
        "age2": jnp.array([jnp.nan]),
        "blind1": jnp.array([0.0]),
        "blind2": jnp.array([jnp.nan]),
        "ei": jnp.array([500.0]),
    }
    # A 70-year-old dependent with $500 of earned income: the worksheet gives
    # max($1,300, $950) = $1,300, plus the $1,950 age bonus = $3,250.
    expected = 3_250.0
    np.testing.assert_allclose(
        calc_std_ded({**person, "dep_status": jnp.array([True])}, law)["std_ded"],
        [expected],
    )
    np.testing.assert_allclose(
        calc_std_ded({**person, "dep_status": jnp.array([1])}, law)["std_ded"],
        [expected],
    )


# The §199A phase-out thresholds and ranges vary by filing status, so they
# arrive as one value per record, while the deduction is computed for four
# business types at once.
def test_qbi_accepts_per_record_law_parameters():
    n = 2
    ones = jnp.ones(n)
    law = QbiLaw(
        rate=0.20 * ones,
        po_thresh_sstb=jnp.array([191_950.0, 383_900.0]),
        po_thresh_non_sstb=jnp.array([191_950.0, 383_900.0]),
        po_range_sstb=jnp.array([50_000.0, 100_000.0]),
        po_range_non_sstb=jnp.array([50_000.0, 100_000.0]),
        wage_exception_sstb=0.0 * ones,
        wage_exception_non_sstb=ones,
        wage_limit=0.50 * ones,
        txbl_inc_limit=ones,
        po_type=jnp.zeros(n),
        min_value=jnp.zeros(n),
    )
    zeros = jnp.zeros(n)
    records = {
        "agi": jnp.array([100_000.0, 100_000.0]),
        "std_ded": jnp.array([14_600.0, 29_200.0]),
        "item_ded": zeros,
        "pe_ded": zeros,
        "div_pref": zeros,
        "kg_pref": zeros,
        "sole_prop1": jnp.array([50_000.0, 50_000.0]),
        "sole_prop2": zeros,
        "part": zeros,
        "scorp": zeros,
        "farm1": zeros,
        "farm2": zeros,
        "wagebill_sole_prop": zeros,
        "wagebill_part": zeros,
        "wagebill_scorp": zeros,
        "wagebill_farm": zeros,
        "sstb_sole_prop": zeros,
        "sstb_part": zeros,
        "sstb_scorp": zeros,
        "sstb_farm": zeros,
    }
    # Both records are far below the phaseout threshold, so the deduction is
    # 20% of $50,000 for each.
    np.testing.assert_allclose(calc_qbi_ded(records, law)["qbi_ded"], [10_000.0, 10_000.0])


def test_mortgage_balance_limit_by_vintage():
    # Five vintages, each with its own balance limit, and one mortgage of
    # $1,000,000 of interest-free debt per tier, so the deductible share reads
    # straight off the limit.
    law = ItemizedDeductionLaw(
        med_floor_agi=0.0,
        med_limit=jnp.inf,
        salt_limit=jnp.inf,
        mort_bal_limit_years=jnp.array([1987.0, 2000.0, 2017.0, 2030.0, jnp.inf]),
        mort_bal_limit=jnp.array(
            [jnp.inf, 1_200_000.0, 1_000_000.0, 900_000.0, 750_000.0]
        ),
        mort_int_limit=jnp.inf,
        auto_int_deduction=0.0,
        casualty_limit=jnp.inf,
        misc_floor_agi=0.0,
        misc_limit=jnp.inf,
        char_floor_agi=0.0,
        char_limit=jnp.inf,
        char_cash_limit_agi=1.0,
        char_noncash_limit_agi=1.0,
        pease_thresh=jnp.inf,
        pease_rate=0.0,
        pease_max_share=0.0,
        limit=jnp.inf,
        salt_floor=0.0,
        salt_floor_thresh=jnp.inf,
        salt_po_rate=0.0,
    )
    zeros = jnp.zeros(6)
    balance = jnp.full(6, 1_000_000.0)
    records = {
        "agi": zeros,
        "med_exp": zeros,
        "salt_inc_sales": zeros,
        "salt_prop": zeros,
        "salt_pers": zeros,
        "first_mort_bal": balance,
        "second_mort_bal": zeros,
        # One loan per vintage, plus a missing year.
        "first_mort_year": jnp.array([1980.0, 1995.0, 2010.0, 2025.0, 2040.0, jnp.nan]),
        "second_mort_year": jnp.full(6, jnp.inf),
        "first_mort_int": jnp.full(6, 40_000.0),
        "second_mort_int": zeros,
        "inv_int_exp": zeros,
        "auto_int_exp": zeros,
        "char_cash": zeros,
        "char_noncash": zeros,
        "casualty_exp": zeros,
        "job_exp": zeros,
        "tax_prep_exp": zeros,
        "other_misc_exp": zeros,
        "other_item_exp": zeros,
    }
    # $40,000 of interest on $1m of debt, times the share of the balance the
    # vintage's limit covers: unlimited, 100%, 100%, 90%, 75%, and unlimited
    # again for the loan with no year recorded.
    np.testing.assert_allclose(
        calc_item_ded(records, law)["mort_int_item_ded"],
        [40_000.0, 40_000.0, 40_000.0, 36_000.0, 30_000.0, 40_000.0],
    )


def test_mortgage_balance_limit_across_two_vintages():
    # A unit holding two mortgages of different vintages:
    # the larger loan uses its own limit, but no more than the smaller loan's.
    law = ItemizedDeductionLaw(
        med_floor_agi=0.0,
        med_limit=jnp.inf,
        salt_limit=jnp.inf,
        mort_bal_limit_years=jnp.array([2017.0, jnp.inf]),
        mort_bal_limit=jnp.array([1_000_000.0, 750_000.0]),
        mort_int_limit=jnp.inf,
        auto_int_deduction=0.0,
        casualty_limit=jnp.inf,
        misc_floor_agi=0.0,
        misc_limit=jnp.inf,
        char_floor_agi=0.0,
        char_limit=jnp.inf,
        char_cash_limit_agi=1.0,
        char_noncash_limit_agi=1.0,
        pease_thresh=jnp.inf,
        pease_rate=0.0,
        pease_max_share=0.0,
        limit=jnp.inf,
        salt_floor=0.0,
        salt_floor_thresh=jnp.inf,
        salt_po_rate=0.0,
    )
    zeros = jnp.zeros(1)
    records = {
        "agi": zeros,
        "med_exp": zeros,
        "salt_inc_sales": zeros,
        "salt_prop": zeros,
        "salt_pers": zeros,
        # An $800,000 pre-2018 loan and a $200,000 post-2017 one.
        "first_mort_bal": jnp.array([800_000.0]),
        "second_mort_bal": jnp.array([200_000.0]),
        "first_mort_year": jnp.array([2010.0]),
        "second_mort_year": jnp.array([2020.0]),
        "first_mort_int": jnp.array([32_000.0]),
        "second_mort_int": jnp.array([8_000.0]),
        "inv_int_exp": zeros,
        "auto_int_exp": zeros,
        "char_cash": zeros,
        "char_noncash": zeros,
        "casualty_exp": zeros,
        "job_exp": zeros,
        "tax_prep_exp": zeros,
        "other_misc_exp": zeros,
        "other_item_exp": zeros,
    }
    # The limit is min($1,000,000, max($800,000, $750,000)) = $800,000 against
    # a $1,000,000 combined balance, so 80% of the $40,000 of interest.
    np.testing.assert_allclose(
        calc_item_ded(records, law)["mort_int_item_ded"], [32_000.0]
    )
