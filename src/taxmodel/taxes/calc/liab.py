"""Final individual income-tax liability and credit allocation."""

from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp

from taxmodel.taxes.calc._corners import capped_at

Array = jnp.ndarray


@dataclass(frozen=True)
class LiabilityLaw:
    repeal_ftc: Array
    repeal_res_energy: Array
    repeal_old: Array
    repeal_gbc: Array
    repeal_other: Array


def calc_liab(records: Mapping[str, Array], law: LiabilityLaw) -> dict[str, Array]:
    """Allocate credits against income tax, other taxes, and outlays."""
    # Repealed credits are zeroed at the point they enter the waterfall, rather
    # than changing the ordering shared by baseline law and reforms.
    ftc = jnp.where(law.repeal_ftc == 0, records["ftc"], 0.0)
    residential_energy = jnp.where(
        law.repeal_res_energy == 0, records["res_energy_cred"], 0.0
    )
    elderly = jnp.where(law.repeal_old == 0, records["old_cred"], 0.0)
    general_business = jnp.where(law.repeal_gbc == 0, records["gbc"], 0.0)
    other_nonrefundable = jnp.where(law.repeal_other == 0, records["other_nonref"], 0.0)
    # Nonrefundable credits, in the order listed here, cannot reduce
    # pre-credit income-tax liability below zero.
    nonrefundable = capped_at(
        ftc
        + records["cdctc_nonref"]
        + records["ed_nonref"]
        + records["savers_nonref"]
        + records["caregiver_cred_nonref"]
        + residential_energy
        + elderly
        + records["ctc_nonref"]
        + general_business
        + records["prior_yr_cred"]
        + other_nonrefundable,
        records["liab_bc"],
    )
    # Left on the default. Both sides here are tax computations that move with
    # income, so the tie turns on which grows faster, and there is no single
    # answer: a credit phasing in at 15 percent grows faster than tax on a
    # household in the 10 percent bracket and slower than tax on one in the 24
    # percent bracket, and a credit past its phase-out shrinks while tax keeps
    # growing. That is the "moves either way" case, which stays at the default.
    # It needs equality to the dollar to matter at all, which takes a household
    # sitting exactly where its tax begins.
    liability_after_nonrefundable = records["liab_bc"] - nonrefundable
    refundable = (
        records["ctc_ref"]
        + records["ed_ref"]
        + records["net_ptc"]
        + records["eitc"]
        + records["rebate"]
        + records["wage_subsidy1"]
        + records["wage_subsidy2"]
        + records["cdctc_ref"]
        + records["savers_ref"]
        + records["caregiver_cred_ref"]
    )
    # Refundable credits first offset the remaining individual income-tax base;
    # only their unused portion can offset the model's other taxes.
    refundable_against_iit = capped_at(refundable, liability_after_nonrefundable)
    liability_after_credits = liability_after_nonrefundable - refundable_against_iit
    liability_iit = (
        liability_after_credits + records["liab_niit"] + records["liab_surtax"]
    )
    other_taxes = (
        records["liab_seca"] + records["recapture_tax"] + records["ira_penalty"]
    )
    refundable_against_other = capped_at(
        refundable - refundable_against_iit, other_taxes
    )
    refund = refundable - refundable_against_iit - refundable_against_other

    # The count of credits claimed is a filing-burden measure, so a caregiver
    # credit belongs in it like any other.
    count_dtype = jnp.int64

    # A credit counts here if it is worth at least a cent, so that fractions of
    # a penny left over by floating-point arithmetic cannot inflate the
    # filing-burden statistic. Repayments of the premium tax credit are
    # negative, so the test is on the size of the credit either way.
    def claimed(amount: Array) -> Array:
        return (jnp.abs(amount) >= 0.01).astype(count_dtype)  # tie: counting test, cast to an integer, so no derivative flows through it

    number_of_credits = (
        jnp.zeros_like(records["liab_bc"], dtype=count_dtype)
        + claimed(ftc)
        + claimed(records["cdctc_nonref"] + records["cdctc_ref"])
        + claimed(records["ed_nonref"] + records["ed_ref"])
        + claimed(records["savers_nonref"] + records["savers_ref"])
        + claimed(records["caregiver_cred_nonref"] + records["caregiver_cred_ref"])
        + claimed(residential_energy)
        + claimed(elderly)
        + claimed(records["ctc_nonref"] + records["ctc_ref"])
        + claimed(general_business)
        + claimed(records["prior_yr_cred"])
        + claimed(other_nonrefundable)
        + claimed(records["net_ptc"])
        + claimed(records["eitc"])
        + claimed(records["rebate"])
        + claimed(records["wage_subsidy1"] + records["wage_subsidy2"])
    )
    return {
        "nonref": nonrefundable,
        "ref": refundable,
        "ref_iit": refundable_against_iit,
        "ref_other": refundable_against_other,
        "refund": refund,
        "liab_iit": liability_iit,
        "liab_iit_net": liability_iit - (refundable_against_other + refund),
        "number_of_credits": number_of_credits,
    }
