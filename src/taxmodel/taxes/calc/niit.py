"""Net investment income tax under IRC §1411."""

from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp

from taxmodel.taxes.calc._corners import floored_at
from taxmodel.taxes.calc._schedules import stacked_tax_inclusive

Array = jnp.ndarray


@dataclass(frozen=True)
class NiitLaw:
    include_active: Array
    rates: Array
    brackets: Array


def calc_niit(records: Mapping[str, Array], law: NiitLaw) -> dict[str, Array]:
    """Return NIIT after active-business and investment-interest deductions."""
    # The §1411 threshold schedule applies to nonnegative AGI even when the
    # investment-income calculation has separate loss and deduction terms.
    agi = floored_at(records["agi"], 0.0)
    net_active_business = (
        records["scorp_active"]
        - records["scorp_active_loss"]
        - records["scorp_179"]
        + records["part_active"]
        - records["part_active_loss"]
        - records["part_179"]
    )
    # When active pass-through earnings enter the NIIT base, IRC §1411(c)(6)
    # still removes partnership earnings already subject to self-employment tax.
    active_deduction = jnp.where(
        law.include_active == 1,
        records["part_se1"] + records["part_se2"],
        net_active_business,
    )
    net_investment_income = floored_at(
        records["txbl_int"]
        + records["div_ord"]
        + records["div_pref"]
        + records["txbl_kg"]
        + records["sch_e"]
        - active_deduction
        - records["inv_int_item_ded"],
        0.0,
    )
    # Assumed: 70% of the computed base, to allow for Form 8960 components the
    # PUF does not report.
    net_investment_income *= 0.7
    return {
        "liab_niit": stacked_tax_inclusive(
            net_investment_income, agi, law.brackets, law.rates
        )
    }
