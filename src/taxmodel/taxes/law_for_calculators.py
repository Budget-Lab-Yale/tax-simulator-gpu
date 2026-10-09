"""Turn one year of compiled law into the objects each tax calculator takes.

The law compiler writes every parameter as one float64 array shaped
[year, filing status, slot] (`law/compile.py`). The calculators take small
frozen dataclasses whose fields are named the way the statute names them:
`CtcLaw.value_young1`, `PayrollLaw.oasdi_ee_brackets`, and so on. This module
maps one naming onto the other.

The correspondence is written out as a table per calculator: dataclass field on
the left, compiled field on the right, plus which slot of it. Nothing here
infers a mapping from a name, and nothing defaults a missing parameter to zero.
A calculator field the compiled schema cannot supply stops the bind with the
whole list of gaps.

A bound law holds every parameter as its four filing-status values, and a rate
or bracket schedule as four rows of slots. `YearLaw.for_records` gives each
household the value for its own filing status. It runs inside the compiled
program as a lookup into a four-element table, so the calculators see one value
per household (`CdctcLaw.young_po_thresh1` is an array as long as the
population) without a population-length copy of the law being stored. The
frame layer gathers the year's law columns onto records the same way
(`frame/steps_jax.join_law`).

Every law therefore has the same array shapes, and the storage a law needs
does not depend on the size of the population. A batch of laws compiled
together (`law/batch.py`) shares one compiled program and fits in memory at
once.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields

import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_dataclass

from taxmodel.law.compile import FILING_STATUSES
from taxmodel.law.errors import LawError
from taxmodel.taxes.calc.agi import AgiLaw, SocialSecurityLaw
from taxmodel.taxes.calc.credits import (
    CaregiverCreditLaw,
    CdctcLaw,
    CtcLaw,
    EducationCreditLaw,
    EitcLaw,
    RebateLaw,
    SaversCreditLaw,
    WageSubsidyLaw,
)
from taxmodel.taxes.calc.deductions import (
    BelowDeductionLaw,
    ItemizedDeductionLaw,
    PersonalExemptionLaw,
    QbiLaw,
    StandardDeductionLaw,
)
from taxmodel.taxes.calc.estate import EstateLaw, EstateParameters
from taxmodel.taxes.calc.kg import CapitalGainsLaw
from taxmodel.taxes.calc.liab import LiabilityLaw
from taxmodel.taxes.calc.niit import NiitLaw
from taxmodel.taxes.calc.payroll import PayrollLaw
from taxmodel.taxes.calc.surtax import SurtaxLaw
from taxmodel.taxes.calc.tax import AlternativeMaximumTaxLaw, AmtLaw, TaxLaw
from taxmodel.taxes.calc.txbl_inc import TaxableIncomeLaw
from taxmodel.taxes.calc.wealth import WealthLaw

Array = jnp.ndarray


@dataclass(frozen=True)
class Ref:
    """Which compiled field, and which of its slots, a calculator field reads.

    `slot` is `None` for a whole rate schedule, where the calculator wants all
    MAX_BRACKETS slots, NaN padding included, because `taxes/calc/_schedules.py`
    neutralizes the empty ones itself.
    """

    name: str
    slot: int | None = 0


def whole_schedule(name: str) -> Ref:
    """All slots of a compiled field: a bracket or rate schedule."""
    return Ref(name, slot=None)


# --------------------------------------------------------------------------
# The tables. Dataclass field on the left, compiled field on the right.
#
# Where the law states a parameter once per layer or segment (the CTC's two
# credit layers, the CDCTC's three phaseout segments), the compiled law keeps
# them in consecutive slots of one field and the calculator names them with a
# trailing number. Those are the entries with an explicit slot.
# --------------------------------------------------------------------------

SOCIAL_SECURITY = {
    "magi_ss_rate": Ref("ss.magi_ss_rate"),
    "rates": whole_schedule("ss.rates"),
    "brackets": whole_schedule("ss.brackets"),
}

AGI = {
    "alimony_repeal_year": Ref("agi.alimony_repeal_year"),
    "bus_loss_limit": Ref("agi.bus_loss_limit"),
    "tuition_ded_limit": Ref("agi.tuition_ded_limit"),
    "dpad_limit": Ref("agi.dpad_limit"),
    "char_above_limit": Ref("char.above_limit"),
    "auto_int_ded_limit": Ref("agi.auto_int_ded_limit"),
    "auto_int_ded_po_thresh": Ref("agi.auto_int_ded_po_thresh"),
    "auto_int_ded_po_rate": Ref("agi.auto_int_ded_po_rate"),
    "auto_int_ded_new": Ref("agi.auto_int_ded_new"),
}

CAPITAL_GAINS = {
    "kg_loss_limit": Ref("agi.kg_loss_limit"),
    "kg_excl_rate": Ref("agi.kg_excl_rate"),
    "index_kg": Ref("pref.index_kg"),
    "index_kg_min_years_held": Ref("pref.index_kg_min_years_held"),
    "index_kg_purchased_after": Ref("pref.index_kg_purchased_after"),
    "index_kg_no_loss": Ref("pref.index_kg_no_loss"),
}

STANDARD_DEDUCTION = {
    "value": Ref("std.value"),
    "bonus": Ref("std.bonus"),
    "dep_floor": Ref("std.dep_floor"),
    "dep_earned_bonus": Ref("std.dep_earned_bonus"),
    "bonus_other": Ref("std.bonus_other"),
}

ITEMIZED_DEDUCTION = {
    "med_floor_agi": Ref("item.med_floor_agi"),
    "med_limit": Ref("item.med_limit"),
    "salt_limit": Ref("item.salt_limit"),
    "mort_bal_limit_years": whole_schedule("item.mort_bal_limit_years"),
    "mort_bal_limit": whole_schedule("item.mort_bal_limit"),
    "mort_int_limit": Ref("item.mort_int_limit"),
    "auto_int_deduction": Ref("item.auto_int_deduction"),
    "casualty_limit": Ref("item.casualty_limit"),
    "misc_floor_agi": Ref("item.misc_floor_agi"),
    "misc_limit": Ref("item.misc_limit"),
    # The charitable deduction's Schedule A parameters live under their own
    # parameter file in the law tree, not under `item`.
    "char_floor_agi": Ref("char.item_floor_agi"),
    "char_limit": Ref("char.item_limit"),
    "char_cash_limit_agi": Ref("char.item_cash_limit_agi"),
    "char_noncash_limit_agi": Ref("char.item_noncash_limit_agi"),
    "pease_thresh": Ref("item.pease_thresh"),
    "pease_rate": Ref("item.pease_rate"),
    "pease_max_share": Ref("item.pease_max_share"),
    "limit": Ref("item.limit"),
    "salt_floor": Ref("item.salt_floor"),
    "salt_floor_thresh": Ref("item.salt_floor_thresh"),
    "salt_po_rate": Ref("item.salt_po_rate"),
}

PERSONAL_EXEMPTION = {
    "value": Ref("pe.value"),
    "po_thresh": Ref("pe.po_thresh"),
    "po_range": Ref("pe.po_range"),
    "po_discrete": Ref("pe.po_discrete"),
    "po_discrete_step": Ref("pe.po_discrete_step"),
    "dep_qualify": Ref("pe.dep_qualify"),
    "nondep_qualify": Ref("pe.nondep_qualify"),
}

QBI = {
    "rate": Ref("qbi.rate"),
    "po_thresh_sstb": Ref("qbi.po_thresh_sstb"),
    "po_thresh_non_sstb": Ref("qbi.po_thresh_non_sstb"),
    "po_range_sstb": Ref("qbi.po_range_sstb"),
    "po_range_non_sstb": Ref("qbi.po_range_non_sstb"),
    "wage_exception_sstb": Ref("qbi.wage_exception_sstb"),
    "wage_exception_non_sstb": Ref("qbi.wage_exception_non_sstb"),
    "wage_limit": Ref("qbi.wage_limit"),
    "txbl_inc_limit": Ref("qbi.txbl_inc_limit"),
    "po_type": Ref("qbi.po_type"),
    "min_value": Ref("qbi.min_value"),
}

BELOW_DEDUCTION = {
    "tip_ded_wage_limit": Ref("below.tip_ded_wage_limit"),
    "tip_ded_lh": Ref("below.tip_ded_lh"),
    "tip_ded_limit": Ref("below.tip_ded_limit"),
    "tip_ded_po_thresh": Ref("below.tip_ded_po_thresh"),
    "tip_ded_po_type": Ref("below.tip_ded_po_type"),
    "tip_ded_po_rate": Ref("below.tip_ded_po_rate"),
    "tip_ded_po_range": Ref("below.tip_ded_po_range"),
    "ot_ded_wage_limit": Ref("below.ot_ded_wage_limit"),
    "ot_ded_half": Ref("below.ot_ded_half"),
    "ot_ded_limit": Ref("below.ot_ded_limit"),
    "ot_ded_po_thresh": Ref("below.ot_ded_po_thresh"),
    "ot_ded_po_type": Ref("below.ot_ded_po_type"),
    "ot_ded_po_rate": Ref("below.ot_ded_po_rate"),
    "ot_ded_po_range": Ref("below.ot_ded_po_range"),
    "senior_ded_value": Ref("below.senior_ded_value"),
    "senior_ded_po_thresh": Ref("below.senior_ded_po_thresh"),
    "senior_ded_po_rate": Ref("below.senior_ded_po_rate"),
}

TAXABLE_INCOME = {
    "item_limit_tax_value_thresh": Ref("item.limit_tax_value_thresh"),
    "item_salt_limit_tax_value_rate": Ref("item.salt_limit_tax_value_rate"),
    "item_limit_tax_value_rate": Ref("item.limit_tax_value_rate"),
}

TAX = {
    "ord_rates": whole_schedule("ord.rates"),
    "ord_brackets": whole_schedule("ord.brackets"),
    "pref_rates": whole_schedule("pref.rates"),
    "pref_brackets": whole_schedule("pref.brackets"),
    "unrecapture_rate": Ref("pref.unrecapture_rate"),
    "collectibles_rate": Ref("pref.collectibles_rate"),
    "tax_at_ord": Ref("pref.tax_at_ord"),
    "no_ord_cap": Ref("pref.no_ord_cap"),
    "kg_top_rate_addon": Ref("pref.kg_top_rate_addon"),
    "div_top_rate_addon": Ref("pref.div_top_rate_addon"),
}

ALTERNATIVE_MAXIMUM_TAX = {
    "rate": Ref("alt_max.rate"),
    "qualify_mult": Ref("alt_max.qualify_mult"),
    "exempt": Ref("alt_max.exempt"),
}

AMT = {
    "pe_pref": Ref("amt.pe_pref"),
    "exempt": Ref("amt.exempt"),
    "exempt_po_thresh": Ref("amt.exempt_po_thresh"),
    "exempt_po_rate": Ref("amt.exempt_po_rate"),
    "rates": whole_schedule("amt.rates"),
    "brackets": whole_schedule("amt.brackets"),
}

PAYROLL = {
    "tips_exempt": Ref("pr.tips_exempt"),
    "ot_exempt": Ref("pr.ot_exempt"),
    "seca_taxable_rate": Ref("pr.seca_taxable_rate"),
    "se_thresh": Ref("pr.se_thresh"),
    "oasdi_ee_rates": whole_schedule("pr.oasdi_ee_rates"),
    "oasdi_ee_brackets": whole_schedule("pr.oasdi_ee_brackets"),
    "oasdi_er_rates": whole_schedule("pr.oasdi_er_rates"),
    "oasdi_er_brackets": whole_schedule("pr.oasdi_er_brackets"),
    "hi_ee_rates": whole_schedule("pr.hi_ee_rates"),
    "hi_ee_brackets": whole_schedule("pr.hi_ee_brackets"),
    "hi_er_rates": whole_schedule("pr.hi_er_rates"),
    "hi_er_brackets": whole_schedule("pr.hi_er_brackets"),
    "add_med_rates": whole_schedule("pr.add_med_rates"),
    "add_med_brackets": whole_schedule("pr.add_med_brackets"),
}

CDCTC = {
    "exp_limit": Ref("cdctc.exp_limit"),
    "n_dep_limit": Ref("cdctc.n_dep_limit"),
    "young_age_limit": Ref("cdctc.young_age_limit"),
    "old_age_limit": Ref("cdctc.old_age_limit"),
    "young_rate1": Ref("cdctc.young_rate", 0),
    "young_rate2": Ref("cdctc.young_rate", 1),
    "young_rate3": Ref("cdctc.young_rate", 2),
    "young_po_thresh1": Ref("cdctc.young_po_thresh", 0),
    "young_po_thresh2": Ref("cdctc.young_po_thresh", 1),
    "young_po_thresh3": Ref("cdctc.young_po_thresh", 2),
    "young_po_rate1": Ref("cdctc.young_po_rate", 0),
    "young_po_rate2": Ref("cdctc.young_po_rate", 1),
    "young_po_rate3": Ref("cdctc.young_po_rate", 2),
    "old_rate1": Ref("cdctc.old_rate", 0),
    "old_rate2": Ref("cdctc.old_rate", 1),
    "old_rate3": Ref("cdctc.old_rate", 2),
    "old_po_thresh1": Ref("cdctc.old_po_thresh", 0),
    "old_po_thresh2": Ref("cdctc.old_po_thresh", 1),
    "old_po_thresh3": Ref("cdctc.old_po_thresh", 2),
    "old_po_rate1": Ref("cdctc.old_po_rate", 0),
    "old_po_rate2": Ref("cdctc.old_po_rate", 1),
    "old_po_rate3": Ref("cdctc.old_po_rate", 2),
    "discrete_step": Ref("cdctc.discrete_step"),
    "refundable": Ref("cdctc.refundable"),
}

EDUCATION_CREDIT = {
    "aoc_rates": whole_schedule("ed.aoc_rates"),
    "aoc_brackets": whole_schedule("ed.aoc_brackets"),
    "aoc_po_thresh": Ref("ed.aoc_po_thresh"),
    "aoc_po_range": Ref("ed.aoc_po_range"),
    "aoc_refund_rate": Ref("ed.aoc_refund_rate"),
    "llc_rates": whole_schedule("ed.llc_rates"),
    "llc_brackets": whole_schedule("ed.llc_brackets"),
    "llc_po_thresh": Ref("ed.llc_po_thresh"),
    "llc_po_range": Ref("ed.llc_po_range"),
}

SAVERS_CREDIT = {
    "repeal": Ref("savers.repeal"),
    "refundable": Ref("savers.refundable"),
}

CAREGIVER_CREDIT = {
    "exp_floor": Ref("caregiver.exp_floor"),
    "min_ei": Ref("caregiver.min_ei"),
    "max_credit": Ref("caregiver.max_credit"),
    "credit_rate": Ref("caregiver.credit_rate"),
    "po_thresh": Ref("caregiver.po_thresh"),
    "po_rate": Ref("caregiver.po_rate"),
    "refundable": Ref("caregiver.refundable"),
}

CTC = {
    "young_age_limit": Ref("ctc.young_age_limit"),
    "old_age_limit": Ref("ctc.old_age_limit"),
    "need_ssn": Ref("ctc.need_ssn"),
    "young_level": Ref("ctc.young_level"),
    "value_young1": Ref("ctc.value_young", 0),
    "value_young2": Ref("ctc.value_young", 1),
    "bonus_young1": Ref("ctc.bonus_young", 0),
    "bonus_young2": Ref("ctc.bonus_young", 1),
    "value_old1": Ref("ctc.value_old", 0),
    "value_old2": Ref("ctc.value_old", 1),
    "value_other": Ref("ctc.value_other"),
    "po_thresh1": Ref("ctc.po_thresh", 0),
    "po_thresh2": Ref("ctc.po_thresh", 1),
    "po_type": Ref("ctc.po_type"),
    "po_rate1": Ref("ctc.po_rate", 0),
    "po_rate2": Ref("ctc.po_rate", 1),
    "po_rate_other": Ref("ctc.po_rate_other"),
    "po_range1": Ref("ctc.po_range", 0),
    "po_range2": Ref("ctc.po_range", 1),
    "po_range_other": Ref("ctc.po_range_other"),
    "po_sequential": Ref("ctc.po_sequential"),
    "po_discrete": Ref("ctc.po_discrete"),
    "po_discrete_step": Ref("ctc.po_discrete_step"),
    "mfs_eligible": Ref("ctc.mfs_eligible"),
    "min_refund_level": Ref("ctc.min_refund_level"),
    "min_refund_young": Ref("ctc.min_refund_young"),
    "min_refund_old": Ref("ctc.min_refund_old"),
    "min_refund_share": Ref("ctc.min_refund_share"),
    "max_refund_young": Ref("ctc.max_refund_young"),
    "max_refund_old": Ref("ctc.max_refund_old"),
    "ei_prior_yr": Ref("ctc.ei_prior_yr"),
    "pi_thresh": Ref("ctc.pi_thresh"),
    "pi_type": Ref("ctc.pi_type"),
    "pi_rate": Ref("ctc.pi_rate"),
    "pi_range": Ref("ctc.pi_range"),
    "baby_bonus_in_credit": Ref("ctc.baby_bonus_in_credit"),
    "baby_bonus": Ref("ctc.baby_bonus"),
    "baby_bonus_pi_rate": Ref("ctc.baby_bonus_pi_rate"),
    "baby_bonus_po_rate": Ref("ctc.baby_bonus_po_rate"),
    "baby_bonus_po_thresh": Ref("ctc.baby_bonus_po_thresh"),
}

EITC = {
    "pi_rate_0": Ref("eitc.pi_rate_0"),
    "pi_rate_1": Ref("eitc.pi_rate_1"),
    "pi_rate_2": Ref("eitc.pi_rate_2"),
    "pi_rate_3": Ref("eitc.pi_rate_3"),
    "po_rate_0": Ref("eitc.po_rate_0"),
    "po_rate_1": Ref("eitc.po_rate_1"),
    "po_rate_2": Ref("eitc.po_rate_2"),
    "po_rate_3": Ref("eitc.po_rate_3"),
    "pi_end_0": Ref("eitc.pi_end_0"),
    "pi_end_1": Ref("eitc.pi_end_1"),
    "pi_end_2": Ref("eitc.pi_end_2"),
    "pi_end_3": Ref("eitc.pi_end_3"),
    "po_thresh_0": Ref("eitc.po_thresh_0"),
    "po_thresh_1": Ref("eitc.po_thresh_1"),
    "po_thresh_2": Ref("eitc.po_thresh_2"),
    "po_thresh_3": Ref("eitc.po_thresh_3"),
    "inv_inc_limit": Ref("eitc.inv_inc_limit"),
    "min_age": Ref("eitc.min_age"),
    "max_age": Ref("eitc.max_age"),
    "ei_prior_yr": Ref("eitc.ei_prior_yr"),
    "mfs_eligible": Ref("eitc.mfs_eligible"),
    "parent_precert": Ref("eitc.parent_precert"),
}

REBATE = {
    "dep_age_limit": Ref("rebate.dep_age_limit"),
    "dep_ssn_req": Ref("rebate.dep_ssn_req"),
    "value": Ref("rebate.value"),
    "value_dep": Ref("rebate.value_dep"),
    "po_thresh": Ref("rebate.po_thresh"),
    "po_type": Ref("rebate.po_type"),
    "po_range": Ref("rebate.po_range"),
    "po_rate": Ref("rebate.po_rate"),
}

WAGE_SUBSIDY = {
    "pi_rate": Ref("wagesub.pi_rate"),
    "pi_end": Ref("wagesub.pi_end"),
    "po_rate": Ref("wagesub.po_rate"),
    "po_thresh": Ref("wagesub.po_thresh"),
}

NIIT = {
    "include_active": Ref("niit.include_active"),
    "rates": whole_schedule("niit.rates"),
    "brackets": whole_schedule("niit.brackets"),
}

SURTAX = {
    "threshold1": Ref("surtax.threshold1"),
    "rate1": Ref("surtax.rate1"),
    "threshold2": Ref("surtax.threshold2"),
    "rate2": Ref("surtax.rate2"),
    "threshold3": Ref("surtax.threshold3"),
    "rate3": Ref("surtax.rate3"),
}

LIABILITY = {
    "repeal_ftc": Ref("credits.repeal_ftc"),
    "repeal_res_energy": Ref("credits.repeal_res_energy"),
    "repeal_old": Ref("credits.repeal_old"),
    "repeal_gbc": Ref("credits.repeal_gbc"),
    "repeal_other": Ref("credits.repeal_other"),
}

ESTATE = {
    "exemption": Ref("estate.exemption"),
    "brackets": whole_schedule("estate.brackets"),
    "rates": whole_schedule("estate.rates"),
    "portability": Ref("estate.portability"),
    "income_tax_ded": Ref("estate.income_tax_ded"),
}

WEALTH = {
    "brackets": whole_schedule("wealth.brackets"),
    "rates": whole_schedule("wealth.rates"),
}


# The whole law-to-calculator map: every calculator, the attribute it gets on
# `YearLaw`, its dataclass, and its table.
BINDINGS: tuple[tuple[str, type, Mapping[str, Ref]], ...] = (
    ("ss", SocialSecurityLaw, SOCIAL_SECURITY),
    ("agi", AgiLaw, AGI),
    ("kg", CapitalGainsLaw, CAPITAL_GAINS),
    ("std_ded", StandardDeductionLaw, STANDARD_DEDUCTION),
    ("item_ded", ItemizedDeductionLaw, ITEMIZED_DEDUCTION),
    ("pe_ded", PersonalExemptionLaw, PERSONAL_EXEMPTION),
    ("qbi_ded", QbiLaw, QBI),
    ("below_ded", BelowDeductionLaw, BELOW_DEDUCTION),
    ("txbl_inc", TaxableIncomeLaw, TAXABLE_INCOME),
    ("tax", TaxLaw, TAX),
    ("alt_max", AlternativeMaximumTaxLaw, ALTERNATIVE_MAXIMUM_TAX),
    ("amt", AmtLaw, AMT),
    ("payroll", PayrollLaw, PAYROLL),
    ("cdctc", CdctcLaw, CDCTC),
    ("ed_cred", EducationCreditLaw, EDUCATION_CREDIT),
    ("savers_cred", SaversCreditLaw, SAVERS_CREDIT),
    ("caregiver_cred", CaregiverCreditLaw, CAREGIVER_CREDIT),
    ("ctc", CtcLaw, CTC),
    ("eitc", EitcLaw, EITC),
    ("rebate", RebateLaw, REBATE),
    ("wage_subsidy", WageSubsidyLaw, WAGE_SUBSIDY),
    ("niit", NiitLaw, NIIT),
    ("surtax", SurtaxLaw, SURTAX),
    ("liab", LiabilityLaw, LIABILITY),
    ("estate", EstateLaw, ESTATE),
    ("wealth", WealthLaw, WEALTH),
)


# One calculator's law object holds another's: the AGI calculator runs the
# Social Security inclusion worksheet inside itself, so `AgiLaw.ss` is the
# `SocialSecurityLaw` bound above.
NESTED: dict[tuple[str, str], str] = {("agi", "ss"): "ss"}


def filing_status_rows(filing_status: Array) -> Array:
    """Which row of a law table each household reads.

    Tax-Data codes filing status 1 through 4 and the compiled law stores its
    four filing statuses in that order, so the row number is the code minus
    one. This runs inside the compiled program, on a traced array, which is
    why it does no checking; `check_filing_status_codes` does that on the host
    once, when the population block is read.
    """
    return jnp.asarray(filing_status, dtype=jnp.int32) - 1


def check_filing_status_codes(filing_status: np.ndarray) -> None:
    """Refuse a population whose filing status is not one of the four codes.

    A code outside 1 through 4 would silently read the wrong row of every law
    table, or read past the end of it, so it is caught here rather than
    producing a tax.
    """
    codes = np.asarray(filing_status)
    unknown = np.unique(codes[(codes < FILING_STATUSES[0]) | (codes > FILING_STATUSES[-1])])
    if unknown.size:
        raise LawError(f"filing status codes outside 1-4: {unknown.tolist()}")


@dataclass(frozen=True)
class YearLaw:
    """Every calculator's law for one year, as its four filing-status values.

    Each parameter is a four-element array — single, joint, married filing
    separately, head of household — and each rate or bracket schedule is four
    rows of slots. `for_records` turns that into the one-value-per-household
    form the calculators read.
    """

    # The calendar year, as a float array, so it is an input to the compiled
    # chain.
    year: jnp.ndarray
    ss: SocialSecurityLaw
    agi: AgiLaw
    kg: CapitalGainsLaw
    std_ded: StandardDeductionLaw
    item_ded: ItemizedDeductionLaw
    pe_ded: PersonalExemptionLaw
    qbi_ded: QbiLaw
    below_ded: BelowDeductionLaw
    txbl_inc: TaxableIncomeLaw
    tax: TaxLaw
    alt_max: AlternativeMaximumTaxLaw
    amt: AmtLaw
    payroll: PayrollLaw
    cdctc: CdctcLaw
    ed_cred: EducationCreditLaw
    savers_cred: SaversCreditLaw
    caregiver_cred: CaregiverCreditLaw
    ctc: CtcLaw
    eitc: EitcLaw
    rebate: RebateLaw
    wage_subsidy: WageSubsidyLaw
    niit: NiitLaw
    surtax: SurtaxLaw
    liab: LiabilityLaw
    estate: EstateLaw
    wealth: WealthLaw

    def for_records(self, filing_status: Array) -> YearLaw:
        """The same law with every parameter read for each household's status.

        Every calculator field comes back as one value per household (a
        schedule as one row of slots per household). Nothing is computed here,
        only looked up.

        Call this inside the compiled program. The lookup is a gather from a
        four-element table, so XLA folds it into the arithmetic that reads it
        and no population-length copy of the law is stored.

        The calendar year passes through unchanged, and one calculator's law
        nested inside another's stays the same object.
        """
        rows = filing_status_rows(filing_status)
        read = lambda table: jnp.asarray(table)[rows]
        return assemble_year_law(
            {
                group: {
                    attribute: read(getattr(getattr(self, group), attribute))
                    for attribute in table
                }
                for group, _cls, table in BINDINGS
            },
            self.year,
        )


# --------------------------------------------------------------------------
# Checking the tables against a compiled schema
# --------------------------------------------------------------------------


def declared_fields() -> dict[str, list[str]]:
    """Every compiled field the tables read, and who reads it."""
    readers: dict[str, list[str]] = {}
    for group, _cls, table in BINDINGS:
        for attribute, ref in table.items():
            readers.setdefault(ref.name, []).append(f"{group}.{attribute}")
    return readers


def missing_fields(available: Sequence[str]) -> list[str]:
    """Compiled fields the calculators need that this law does not have."""
    have = set(available)
    return sorted(name for name in declared_fields() if name not in have)



def check_dataclass_coverage() -> list[str]:
    """Dataclass fields with no row in their table.

    Guards against a calculator gaining a parameter without this module
    gaining the line that supplies it.
    """
    gaps: list[str] = []
    for group, cls, table in BINDINGS:
        for field in fields(cls):
            if field.name in table or (group, field.name) in NESTED:
                continue
            gaps.append(f"{cls.__name__}.{field.name} (bound as law.{group})")
    return sorted(gaps)


def check_bindings(available: Sequence[str]) -> None:
    """Stop unless every calculator parameter has a compiled field behind it."""
    problems: list[str] = []
    for gap in check_dataclass_coverage():
        problems.append(f"  no table entry for {gap}")
    readers = declared_fields()
    for name in missing_fields(available):
        who = ", ".join(readers[name])
        problems.append(f"  compiled law has no field '{name}', needed by {who}")
    if problems:
        raise LawError(
            "the compiled law cannot supply every calculator parameter:\n"
            + "\n".join(problems)
        )


# --------------------------------------------------------------------------
# Binding
# --------------------------------------------------------------------------


def resolve_years(
    values: Mapping[str, np.ndarray],
    law_years: Sequence[int],
    years: Sequence[int],
) -> dict[str, dict[str, np.ndarray]]:
    """Read every calculator parameter for a whole run of years at once.

    `values` maps a compiled field name to its [year, filing status, slot]
    array; `law_years` is the year axis of those arrays. What comes back is the
    same table this module's tables describe (calculator group, then the
    field's name on the calculator's dataclass), with each parameter as one
    NumPy array carrying the requested years on a leading axis: [year, filing
    status] for a scalar parameter and [year, filing status, slot] for a rate
    or bracket schedule.

    All years are read in one array operation. The compiler has already
    written each year's value, indexation included, into its own row, so this
    step is a lookup.
    """
    check_bindings(tuple(values))

    year_axis = [int(y) for y in law_years]
    wanted = [int(y) for y in years]
    for year in wanted:
        if year not in year_axis:
            raise LawError(
                f"the compiled law covers {year_axis[0]}-{year_axis[-1]}, not {year}"
            )
    rows = np.asarray([year_axis.index(year) for year in wanted])

    # A scalar parameter that is missing for a year is a hole in the law, not
    # padding: it would run through the calculator as NaN and come out the
    # other end as a NaN liability. Schedules are exempt because NaN is how the
    # compiler marks an unused bracket slot.
    #
    # A parameter can be missing in some years and stated in others, so the
    # complaints are collected per year and the earliest bad year is reported.
    empty: dict[int, list[str]] = {}
    resolved: dict[str, dict[str, np.ndarray]] = {}
    for group, _cls, table in BINDINGS:
        resolved[group] = {}
        for attribute, ref in table.items():
            block = values[ref.name][rows]
            if ref.slot is None:
                resolved[group][attribute] = block
                continue
            if ref.slot >= block.shape[-1]:
                for year in wanted:
                    empty.setdefault(year, []).append(
                        f"  {group}.{attribute} wants slot {ref.slot + 1} of "
                        f"'{ref.name}', which has only {block.shape[-1]}"
                    )
                continue
            column = block[:, :, ref.slot]
            all_nan = np.all(np.isnan(column), axis=1)
            for year, missing in zip(wanted, all_nan, strict=True):
                if missing:
                    empty.setdefault(year, []).append(
                        f"  {group}.{attribute} reads '{ref.name}' slot "
                        f"{ref.slot + 1}, which {year} law leaves empty"
                    )
            if all_nan.any():
                continue
            resolved[group][attribute] = column
    if empty:
        year = min(empty)
        raise LawError(
            f"the compiled law is missing values the calculators need in {year}:\n"
            + "\n".join(empty[year])
        )
    return resolved


def assemble_year_law(
    parameters: Mapping[str, Mapping[str, object]], year: object
) -> YearLaw:
    """Put resolved parameters into each calculator's law dataclass.

    The arrays are used as handed over. This function knows how the
    calculators' law objects nest inside one another, and it works the same
    whether the arrays hold one year or a whole projection.
    """
    bound: dict[str, object] = {}
    for group, cls, _table in BINDINGS:
        arguments = dict(parameters[group])
        for (owner, attribute), source in NESTED.items():
            if owner == group:
                arguments[attribute] = bound[source]
        bound[group] = cls(**arguments)
    return YearLaw(year=year, **bound)  # type: ignore[arg-type]


def bind_year(
    values: Mapping[str, np.ndarray],
    years: Sequence[int],
    year: int,
) -> YearLaw:
    """Build every calculator's law object for one year.

    `values` maps a compiled field name to its [year, filing status, slot]
    array; `years` is the year axis of those arrays. What comes back holds the
    year's four filing-status values for every parameter, in filing-status
    order. It knows nothing about any population; giving each household the
    value for its own filing status is `YearLaw.for_records`, which runs inside
    the compiled program.
    """
    resolved = resolve_years(values, years, [int(year)])
    return assemble_year_law(
        {
            group: {
                attribute: jnp.asarray(array[0]) for attribute, array in table.items()
            }
            for group, table in resolved.items()
        },
        jnp.asarray(year, dtype=jnp.float64),
    )


# --------------------------------------------------------------------------
# Registering the law dataclasses as containers of arrays
# --------------------------------------------------------------------------
#
# `jax.jit` and `jax.grad` take an argument apart into its numeric pieces and
# put it back together. Registering these dataclasses lets the whole chain be
# compiled, or differentiated with respect to a tax parameter, with the law
# passed in as an ordinary argument.
#
# Every field is a numeric piece, the calendar year included. The capital-gain
# step reads the year and subtracts a rounded holding period from it. Because
# the law's values are inputs, one compiled chain serves every year of a run.
#
# The law dataclasses are declared in `taxmodel.taxes.calc`, next to the
# functions that read them. They are registered here because this module holds
# the full list of them and builds them.

for _group, _cls, _table in BINDINGS:
    register_dataclass(
        _cls, data_fields=[f.name for f in fields(_cls)], meta_fields=[]
    )

register_dataclass(
    EstateParameters,
    data_fields=[f.name for f in fields(EstateParameters)],
    meta_fields=[],
)

register_dataclass(
    YearLaw, data_fields=[f.name for f in fields(YearLaw)], meta_fields=[]
)
