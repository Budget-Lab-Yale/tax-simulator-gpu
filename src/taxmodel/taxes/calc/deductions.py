"""Standard, itemized, exemption, QBI, and other below-line deductions."""

from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp

from taxmodel.taxes.calc._corners import (
    capped_at,
    floored_at,
    phase_out_share,
    phased_out,
)

Array = jnp.ndarray

# Two numbers the OBBBA §199A calculation uses that are not yet law parameters.
# They belong in config/scenarios/.../qbi.yaml with a statutory citation once
# their source is settled.

# Under the OBBBA design the deduction is cut by 75 cents for every dollar of
# taxable income above the phase-out threshold.
OBBBA_QBI_REDUCTION_PER_DOLLAR = 0.75

# The OBBBA minimum deduction is offered only to a filer with at least $1,000 of
# combined qualified business income.
OBBBA_QBI_MIN_DEDUCTION_INCOME = 1000.0


@dataclass(frozen=True)
class StandardDeductionLaw:
    value: Array
    bonus: Array
    dep_floor: Array
    dep_earned_bonus: Array
    bonus_other: Array


@dataclass(frozen=True)
class ItemizedDeductionLaw:
    med_floor_agi: Array
    med_limit: Array
    salt_limit: Array
    mort_bal_limit_years: Array
    mort_bal_limit: Array
    mort_int_limit: Array
    auto_int_deduction: Array
    casualty_limit: Array
    misc_floor_agi: Array
    misc_limit: Array
    char_floor_agi: Array
    char_limit: Array
    char_cash_limit_agi: Array
    char_noncash_limit_agi: Array
    pease_thresh: Array
    pease_rate: Array
    pease_max_share: Array
    limit: Array
    salt_floor: Array
    salt_floor_thresh: Array
    salt_po_rate: Array


@dataclass(frozen=True)
class PersonalExemptionLaw:
    value: Array
    po_thresh: Array
    po_range: Array
    po_discrete: Array
    po_discrete_step: Array
    dep_qualify: Array
    nondep_qualify: Array


@dataclass(frozen=True)
class QbiLaw:
    rate: Array
    po_thresh_sstb: Array
    po_thresh_non_sstb: Array
    po_range_sstb: Array
    po_range_non_sstb: Array
    wage_exception_sstb: Array
    wage_exception_non_sstb: Array
    wage_limit: Array
    txbl_inc_limit: Array
    po_type: Array
    min_value: Array


@dataclass(frozen=True)
class BelowDeductionLaw:
    tip_ded_wage_limit: Array
    tip_ded_lh: Array
    tip_ded_limit: Array
    tip_ded_po_thresh: Array
    tip_ded_po_type: Array
    tip_ded_po_rate: Array
    tip_ded_po_range: Array
    ot_ded_wage_limit: Array
    ot_ded_half: Array
    ot_ded_limit: Array
    ot_ded_po_thresh: Array
    ot_ded_po_type: Array
    ot_ded_po_rate: Array
    ot_ded_po_range: Array
    senior_ded_value: Array
    senior_ded_po_thresh: Array
    senior_ded_po_rate: Array


def calc_std_ded(
    records: Mapping[str, Array], law: StandardDeductionLaw
) -> dict[str, Array]:
    """Return the standard deduction, including the dependent worksheet order."""
    bonuses = (
        (records["age1"] >= 65).astype(jnp.int64)
        + ((~jnp.isnan(records["age2"])) & (records["age2"] >= 65)).astype(jnp.int64)
        + records["blind1"].astype(jnp.int64)
        + ((~jnp.isnan(records["blind2"])) & (records["blind2"] == 1)).astype(jnp.int64)
    )
    bonus_value = law.bonus * bonuses
    nondependent = law.value + bonus_value + law.bonus_other
    # The dependent worksheet caps earned income at the base deduction before
    # adding age, blindness, and other statutory bonuses.
    dependent = (
        capped_at(
            floored_at(records["ei"] + law.dep_earned_bonus, law.dep_floor), law.value
        )
        + bonus_value
        + law.bonus_other
    )
    # The dependent flag is compared with zero, so any nonzero value marks a
    # dependent.
    return {"std_ded": jnp.where(records["dep_status"] == 0, nondependent, dependent)}


def _mortgage_balance_limit(year: Array, law: ItemizedDeductionLaw) -> Array:
    """Look up a mortgage's balance limit from the year the loan was incurred.

    The vintages are given as a rising list of years with a matching list of
    limits; a loan takes the limit of the first vintage whose year it does not
    exceed. The number of vintages is whatever the law declares. A missing
    year is unlimited.
    """
    year = jnp.asarray(year)
    years = jnp.asarray(law.mort_bal_limit_years)
    limits = jnp.asarray(law.mort_bal_limit)
    # A NaN year fails every comparison, so a missing vintage bound is passed
    # over.
    fits = year[..., None] <= years
    vintage = jnp.argmax(fits, axis=-1)
    # A loan newer than every declared vintage takes the last declared one. The
    # last year bound is Inf in every law we have, so this only fires on a law
    # that stops short.
    declared = ~jnp.isnan(limits)
    last = limits.shape[-1] - 1 - jnp.argmax(declared[..., ::-1], axis=-1)
    vintage = jnp.where(jnp.any(fits, axis=-1), vintage, last)
    limit = jnp.take_along_axis(
        jnp.broadcast_to(limits, fits.shape), vintage[..., None], axis=-1
    )[..., 0]
    return jnp.where(jnp.isnan(year), jnp.inf, limit)


def calc_item_ded(
    records: Mapping[str, Array], law: ItemizedDeductionLaw
) -> dict[str, Array]:
    """Return Schedule A components and the Pease-limited total."""
    # AGI is floored at zero before applying every AGI-based Schedule A floor,
    # cap, and phaseout.
    agi = floored_at(records["agi"], 0.0)
    med = capped_at(
        floored_at(
            records["med_exp"] - agi * law.med_floor_agi, 0.0, sinks_against_the_limit=True
        ),
        law.med_limit,
        sinks_against_the_limit=True,
    )
    salt_paid = records["salt_inc_sales"] + records["salt_prop"] + records["salt_pers"]
    # SALT phases down from its stated limit but cannot fall below the reform's
    # statutory floor.
    salt_cap = floored_at(
        law.salt_limit
        - law.salt_po_rate * floored_at(agi - law.salt_floor_thresh, 0.0),
        law.salt_floor,
        sinks_against_the_limit=True,
    )
    salt = capped_at(salt_paid, salt_cap)
    # The balance limit on deductible mortgage interest has vintages: under
    # 2024 law a pre-1988 loan is unlimited, a 1988-2017 loan is limited above
    # $1m, and a post-2017 loan above $750k. Each mortgage gets the limit of
    # the year it was incurred.
    #
    # The mortgage years arrive as record columns, so the vintage lookup is
    # available to a reform that wants it. The chain passes
    # Inf for both years, so every loan takes the last, newest vintage.
    first_bal_limit = _mortgage_balance_limit(records["first_mort_year"], law)
    second_bal_limit = _mortgage_balance_limit(records["second_mort_year"], law)
    # A unit with two mortgages: the larger balance may use
    # its own limit, but only up to the smaller loan's limit. The three
    # comparisons this takes are between two mortgage balances or between two
    # balance limits, never against a household income amount, so there is no
    # "more income" side to a tie and no marginal rate runs through them.
    larger_bal = jnp.maximum(records["first_mort_bal"], records["second_mort_bal"])  # tie: two mortgage balances, not an income threshold
    first_is_smaller = records["first_mort_bal"] < records["second_mort_bal"]
    smaller_bal_limit = jnp.where(first_is_smaller, first_bal_limit, second_bal_limit)
    larger_bal_limit = jnp.where(first_is_smaller, second_bal_limit, first_bal_limit)
    usable_bal = jnp.maximum(larger_bal, smaller_bal_limit)  # tie: mortgage balance against a balance limit, not an income threshold
    bal_limit = jnp.minimum(larger_bal_limit, usable_bal)  # tie: two balance limits, not an income threshold
    total_mortgage_balance = records["first_mort_bal"] + records["second_mort_bal"]
    # A filer with no mortgage balance still divides here. The limit divided
    # by zero is infinity, so the deductible share would be one, which is
    # harmless because there is no interest to deduct. Substituting a denominator of one for those filers gives the
    # same tax answer and keeps the derivative finite: zero divided by zero
    # differentiates to NaN, and one NaN in a reverse pass spreads to every
    # input in the gradient.
    safe_mortgage_balance = jnp.where(
        total_mortgage_balance == 0, 1.0, total_mortgage_balance
    )
    deductible_share = capped_at(bal_limit / safe_mortgage_balance, 1.0)
    mort = capped_at(
        (records["first_mort_int"] + records["second_mort_int"]) * deductible_share,
        law.mort_int_limit,
    )
    inv = records["inv_int_exp"]
    interest = mort + inv + law.auto_int_deduction * records["auto_int_exp"]
    charity = capped_at(
        records["char_cash"], agi * law.char_cash_limit_agi
    ) + capped_at(records["char_noncash"], agi * law.char_noncash_limit_agi)
    charity = capped_at(
        floored_at(charity - agi * law.char_floor_agi, 0.0), law.char_limit
    )
    # Cash and noncash gifts first face their separate AGI ceilings, then the
    # combined AGI floor and dollar limit.
    casualty = capped_at(records["casualty_exp"], law.casualty_limit)
    misc = capped_at(
        floored_at(
            records["job_exp"]
            + records["tax_prep_exp"]
            + records["other_misc_exp"]
            - agi * law.misc_floor_agi,
            0.0,
            sinks_against_the_limit=True,
        ),
        law.misc_limit,
        sinks_against_the_limit=True,
    )
    other = records["other_item_exp"]
    before_limits = med + salt + interest + charity + casualty + misc + other
    # Pease excludes medical expenses, investment interest, and casualty losses
    # from its percentage cap on the overall itemized-deduction reduction.
    pease_reduction = floored_at(agi - law.pease_thresh, 0.0) * law.pease_rate
    pease_base = floored_at(before_limits - med - inv - casualty, 0.0)
    item = capped_at(
        before_limits - capped_at(pease_reduction, law.pease_max_share * pease_base),
        law.limit,
    )
    return {
        "med_item_ded": med,
        "salt_item_ded": salt,
        "mort_int_item_ded": mort,
        "inv_int_item_ded": inv,
        "int_item_ded": interest,
        "char_item_ded": charity,
        "casualty_item_ded": casualty,
        "misc_item_ded": misc,
        "other_item_ded": other,
        "item_ded_ex_limits": before_limits,
        "item_ded": item,
    }


def calc_pe_ded(
    records: Mapping[str, Array], law: PersonalExemptionLaw
) -> dict[str, Array]:
    """Return personal exemptions after the statutory AGI phaseout."""
    # The exemption count separately admits filing-unit members and dependents,
    # allowing reforms to turn either group on or off.
    n_pe = (law.nondep_qualify == 1) * (1 + (records["filing_status"] == 2)) + (
        law.dep_qualify == 1
    ) * records["n_dep"]
    po_share = phase_out_share(records["agi"], law.po_thresh, law.po_range)
    po_share = jnp.where(
        law.po_discrete == 1,
        jnp.ceil(po_share / law.po_discrete_step) * law.po_discrete_step,
        po_share,
    )
    return {"pe_ded": n_pe * law.value * (1 - po_share)}


def calc_qbi_ded(records: Mapping[str, Array], law: QbiLaw) -> dict[str, Array]:
    """The §199A deduction, limited by a provisional taxable income."""
    # The provisional taxable income subtracts the larger of the standard and
    # itemized deductions, before calc_txbl_inc chooses a deduction or applies
    # the tip, overtime, and senior deductions.
    txbl_inc = floored_at(
        records["agi"]
        - floored_at(records["item_ded"], records["std_ded"])
        - records["pe_ded"],
        0.0,
    )
    # §199A works on the household's income from each kind of business, so
    # the two earners' sole-proprietorship and farm income are added together.
    inc = jnp.stack(
        (
            records["sole_prop1"] + records["sole_prop2"],
            records["part"],
            records["scorp"],
            records["farm1"] + records["farm2"],
        ),
        axis=-1,
    )
    wages = jnp.nan_to_num(
        jnp.stack(
            (
                records["wagebill_sole_prop"],
                records["wagebill_part"],
                records["wagebill_scorp"],
                records["wagebill_farm"],
            ),
            axis=-1,
        ),
        nan=0.0,
    )
    sstb = jnp.nan_to_num(
        jnp.stack(
            (
                records["sstb_sole_prop"],
                records["sstb_part"],
                records["sstb_scorp"],
                records["sstb_farm"],
            ),
            axis=-1,
        ),
        nan=0.0,
    )
    # The four business types sit on the trailing axis while the phaseout
    # thresholds and ranges are mapped by filing status, so the law values get
    # the extra axis before they meet a per-business array.
    threshold = jnp.where(
        sstb == 1,
        jnp.asarray(law.po_thresh_sstb)[..., None],
        jnp.asarray(law.po_thresh_non_sstb)[..., None],
    )
    po_range = jnp.where(
        sstb == 1,
        jnp.asarray(law.po_range_sstb)[..., None],
        jnp.asarray(law.po_range_non_sstb)[..., None],
    )
    po_share = phase_out_share(txbl_inc[..., None], threshold, po_range)
    wage_exception = jnp.where(
        sstb == 1,
        jnp.asarray(law.wage_exception_sstb)[..., None],
        jnp.asarray(law.wage_exception_non_sstb)[..., None],
    )
    rate = jnp.asarray(law.rate)[..., None]
    wage_limit = jnp.asarray(law.wage_limit)[..., None]
    tentative = inc * rate
    # The TCJA calculation phases out the excess of a component deduction over
    # its wage-based limitation; SSTBs and other businesses have separate law.
    #
    # The zero floor takes the default spelling because a component deduction
    # rises against it: one more dollar of that kind of business income raises
    # the component by the deduction rate. A household with none of that kind
    # of business income sits exactly on the floor and is treated as having
    # crossed in the direction of more income -- so the next dollar
    # earns the deduction.
    tcja = jnp.sum(
        floored_at(
            tentative
            - po_share
            * floored_at(tentative - wages * wage_limit * wage_exception, 0.0),
            0.0,
        ),
        axis=-1,
    )
    obbb_step1 = jnp.sum(
        capped_at(inc * rate * (sstb == 0), wages * wage_limit), axis=-1
    )
    # The alternative OBBBA calculation compares the wage-limited non-SSTB
    # amount with a taxable-income-based amount after its high-income reduction.
    total_inc = jnp.sum(inc, axis=-1)
    # Same zero floor, same reason as the TCJA branch above: the amount rises
    # with the household's business income, so a household with no business
    # income at all is treated as crossing the floor when it earns a dollar.
    obbb_step2 = floored_at(
        total_inc * law.rate
        - OBBBA_QBI_REDUCTION_PER_DOLLAR * floored_at(txbl_inc - law.po_thresh_sstb, 0.0),
        0.0,
    )
    # The OBBBA deduction is the better of two whole statutory computations, both
    # of which rise with the same business income, so neither one is the "more
    # income" side of a tie between them.
    obbb = jnp.maximum(obbb_step1, obbb_step2)  # tie: two alternative statutory formulas, neither is the income side
    qbi = jnp.where(law.po_type == 0, tcja, obbb)
    qbi = floored_at(qbi, law.min_value * (total_inc > OBBBA_QBI_MIN_DEDUCTION_INCOME))
    return {
        "qbi_ded": capped_at(
            qbi,
            floored_at(txbl_inc - records["div_pref"] - records["kg_pref"], 0.0)
            * law.txbl_inc_limit,
        )
    }


def _unless_wages_exceed(amount: Array, wages: Array, limit: Array) -> Array:
    """`amount`, or nothing once the earner's own wages exceed `limit`."""
    return jnp.where(wages > limit, 0.0, amount)


def _capped_then_phased_out(
    amount: Array,
    agi: Array,
    *,
    limit: Array,
    po_type: Array,
    po_rate: Array,
    po_range: Array,
    po_thresh: Array,
) -> Array:
    """A deduction capped at `limit`, then phased out above `po_thresh`.

    The tip and overtime deductions share this schedule. The phase-out rate is
    either the one the law states (`po_type` of 1) or whatever rate takes the
    capped amount to zero over `po_range` dollars of AGI.
    """
    amount = capped_at(amount, limit)
    rate = jnp.where(po_type == 1, po_rate, amount / po_range)
    return phased_out(amount, agi, po_thresh, rate)


def calc_below_ded(
    records: Mapping[str, Array], law: BelowDeductionLaw
) -> dict[str, Array]:
    """Return the tip, overtime, and senior deductions below AGI."""
    # Tips and overtime are unavailable once an earner's own wages exceed the
    # applicable eligibility limit; a joint return evaluates each earner alone.
    # The limit is tested against the earner's whole pay, tips and overtime
    # included, which is what that earner's W-2 reports.
    wages1 = records["wages_base1"] + records["tips1"] + records["ot1"]
    wages2 = records["wages_base2"] + records["tips2"] + records["ot2"]
    tip1 = _unless_wages_exceed(records["tips1"], wages1, law.tip_ded_wage_limit)
    tip2 = _unless_wages_exceed(records["tips2"], wages2, law.tip_ded_wage_limit)
    tips_lh = tip1 * records["tips_lh1"] + tip2 * records["tips_lh2"]
    tip = _capped_then_phased_out(
        jnp.where(law.tip_ded_lh == 1, tips_lh, tip1 + tip2),
        records["agi"],
        limit=law.tip_ded_limit,
        po_type=law.tip_ded_po_type,
        po_rate=law.tip_ded_po_rate,
        po_range=law.tip_ded_po_range,
        po_thresh=law.tip_ded_po_thresh,
    )
    ot = _unless_wages_exceed(
        records["ot1"], wages1, law.ot_ded_wage_limit
    ) + _unless_wages_exceed(records["ot2"], wages2, law.ot_ded_wage_limit)
    # Only the statutory overtime premium is deductible when the wage measure
    # includes both base pay and one-half premium.
    ot = _capped_then_phased_out(
        ot / (1 + 2 * law.ot_ded_half),
        records["agi"],
        limit=law.ot_ded_limit,
        po_type=law.ot_ded_po_type,
        po_rate=law.ot_ded_po_rate,
        po_range=law.ot_ded_po_range,
        po_thresh=law.ot_ded_po_thresh,
    )
    seniors = (records["age1"] >= 65).astype(jnp.int64) + (
        (~jnp.isnan(records["age2"])) & (records["age2"] >= 65)
    ).astype(jnp.int64)
    # The senior amount stacks for each qualifying spouse and phases out per
    # qualifying person under the model's stated rate.
    senior = phased_out(
        law.senior_ded_value * seniors,
        records["agi"],
        law.senior_ded_po_thresh,
        seniors * law.senior_ded_po_rate,
    )
    return {"tip_ded": tip, "ot_ded": ot, "senior_ded": senior}
