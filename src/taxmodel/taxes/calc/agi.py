"""Adjusted gross income and taxable Social Security benefits."""

from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp

from taxmodel.taxes.calc._corners import capped_at, floored_at, phased_out
from taxmodel.taxes.calc._schedules import last_live_rate, live_slots, upper_edges

Array = jnp.ndarray


@dataclass(frozen=True)
class SocialSecurityLaw:
    """IRC §86 inclusion schedule, ordered from the first threshold upward."""

    magi_ss_rate: Array
    rates: Array
    brackets: Array


@dataclass(frozen=True)
class AgiLaw:
    alimony_repeal_year: Array
    bus_loss_limit: Array
    tuition_ded_limit: Array
    dpad_limit: Array
    char_above_limit: Array
    auto_int_ded_limit: Array
    auto_int_ded_po_thresh: Array
    auto_int_ded_po_rate: Array
    auto_int_ded_new: Array
    ss: SocialSecurityLaw


def calc_ss(records: Mapping[str, Array], law: SocialSecurityLaw) -> dict[str, Array]:
    """Return the portion of OASI benefits included in income under IRC §86."""
    # IRC §86 provisional income is modified AGI plus the law-specified share
    # of OASI benefits; negative provisional income has no taxable benefit.
    gross_ss = records["gross_ss"]
    magi_plus_ss = floored_at(gross_ss * law.magi_ss_rate + records["magi_ss"], 0.0)
    # ss.yaml maps the thresholds by filing status, so the schedule can differ
    # from one return to the next. Slots live on the trailing axis, as they do
    # everywhere else in the package.
    brackets, rates = live_slots(law.brackets, law.rates)
    tier_income = floored_at(
        capped_at(magi_plus_ss[..., None], upper_edges(brackets)) - brackets, 0.0
    )
    tier_inclusion = capped_at(tier_income, gross_ss[..., None]) * rates
    # Only the excess over each IRC §86 threshold is included; the final cap is
    # the highest statutory inclusion percentage of gross benefits.
    #
    # Left on the default. The cap moves with benefits but not with other income,
    # so which side of the tie is the more-income side depends on which kind of
    # dollar is being priced: a dollar of other income raises the inclusion
    # against a fixed cap, while a dollar of benefits raises the cap by the
    # statutory percentage and the inclusion by less. That is the "moves either
    # way" case, which stays at the default.
    txbl_ss = capped_at(
        jnp.sum(tier_inclusion, axis=-1), gross_ss * last_live_rate(law.rates)
    )
    return {"txbl_ss": txbl_ss}


def calc_agi(records: Mapping[str, Array], law: AgiLaw) -> dict[str, Array]:
    """Adjusted gross income and the pieces reported beside it."""
    # The TCJA alimony rule applies symmetrically: qualifying pre-repeal
    # divorces include alimony received and deduct alimony paid.
    alimony_qualifies = (~jnp.isnan(records["divorce_year"])) & (
        records["divorce_year"] < law.alimony_repeal_year
    )
    # Line 1 of the 1040 is the household's whole wage line, so both earners'
    # base pay, tips and overtime are added up here. Sole-proprietorship and
    # farm income are likewise the two earners' amounts together.
    wages = (
        records["wages_base1"] + records["tips1"] + records["ot1"]
        + records["wages_base2"] + records["tips2"] + records["ot2"]
    )
    inc_ex_ss = (
        wages + records["txbl_int"] + records["div_ord"]
        + records["div_pref"] + records["state_ref"] + records["txbl_ira_dist"]
        + records["txbl_pens_dist"] + records["txbl_kg"] + records["other_gains"]
        + records["alimony"] * alimony_qualifies
        + records["sole_prop1"] + records["sole_prop2"]
        + records["sch_e"] + records["farm1"] + records["farm2"]
        + records["ui"] + records["other_inc"]
    )
    # This calibrated reporting amount is not subtracted from AGI or from any
    # deduction.
    excess_bus_loss = (records["r.bus_loss"] < 0.1) * floored_at(
        -records["pt"] - law.bus_loss_limit, 0.0, sinks_against_the_limit=True
    )
    char_above_ded = capped_at(
        records["char_cash"] + records["char_noncash"], law.char_above_limit
    )
    # These Schedule 1 deductions are assembled before taxable benefits so
    # they reduce the modified AGI used by the IRC §86 worksheet.
    above_ded_ex_sl = (
        char_above_ded + records["other_above_ded"] + records["ed_exp"]
        + records["hsa_contr"] + records["liab_seca_er"] + records["keogh_contr"]
        + records["se_health"] + records["early_penalty"]
        + records["alimony_exp"] * alimony_qualifies + records["trad_contr_ira"]
        + capped_at(records["tuition_ded"], law.tuition_ded_limit)
        + capped_at(records["dpad"], law.dpad_limit)
    )
    magi = inc_ex_ss + records["gross_ss"] - above_ded_ex_sl
    # The auto-interest deduction is separately limited and phased out before
    # it enters Social Security modified AGI.
    auto_int_ded = capped_at(records["auto_int_exp"], law.auto_int_ded_limit)
    auto_int_ded = phased_out(
        auto_int_ded, magi, law.auto_int_ded_po_thresh, law.auto_int_ded_po_rate
    )
    # The 58% new-car share is the calibrated Experian 2024 financing share.
    auto_int_ded *= (law.auto_int_ded_new == 0) | (records["r.new_car"] < 0.58)
    above_ded_ex_sl += auto_int_ded
    txbl_ss = calc_ss(
        {"gross_ss": records["gross_ss"], "magi_ss": inc_ex_ss + records["exempt_int"] - above_ded_ex_sl},
        law.ss,
    )["txbl_ss"]
    gross_inc = inc_ex_ss + txbl_ss
    above_ded = above_ded_ex_sl + records["sl_int_ded"]
    return {
        "txbl_ss": txbl_ss,
        "excess_bus_loss": excess_bus_loss,
        "char_above_ded": char_above_ded,
        "auto_int_ded": auto_int_ded,
        "gross_inc": gross_inc,
        "above_ded": above_ded,
        "agi": gross_inc - above_ded,
    }
