"""Taxable-income calculation after deductions."""

from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp

from taxmodel.taxes.calc._corners import capped_at, floored_at

Array = jnp.ndarray


@dataclass(frozen=True)
class TaxableIncomeLaw:
    item_limit_tax_value_thresh: Array
    item_salt_limit_tax_value_rate: Array
    item_limit_tax_value_rate: Array


def calc_txbl_inc(records: Mapping[str, Array], law: TaxableIncomeLaw) -> dict[str, Array]:
    """Choose between the standard and itemized deduction and compute taxable income."""
    # The tax-value limitation separates SALT from other itemized deductions so
    # each portion receives its own clawback rate above the threshold.
    ex_item = records["agi"] - records["pe_ded"] - records["qbi_ded"]
    # Both caps have zero on either side for an itemizer below the threshold
    # whose Schedule A holds only one of the two kinds of deduction. The amount
    # is held at zero by the floor inside the cap, while the cap itself rises
    # with the next dollar of the deduction it is measured on, so the amount
    # sinks against the cap and the tie takes the sloped branch. At that tie
    # the household's next dollar of deduction is not subject to the
    # limitation.
    salt_excess = capped_at(
        floored_at(ex_item - law.item_limit_tax_value_thresh, 0.0),
        records["salt_item_ded"],
        sinks_against_the_limit=True,
    )
    other_excess = capped_at(
        floored_at(
            ex_item - records["salt_item_ded"] - law.item_limit_tax_value_thresh, 0.0
        ),
        floored_at(records["item_ded"] - records["salt_item_ded"], 0.0),
        sinks_against_the_limit=True,
    )
    item_limited = (
        records["item_ded"]
        - salt_excess * law.item_salt_limit_tax_value_rate
        - other_excess * law.item_limit_tax_value_rate
    )
    # The itemizing flag is set only when the post-clawback itemized value is
    # strictly greater. The deduction taken is the same number either way when
    # the two are equal, so the floor below is written to take the itemized
    # slope at a tie.
    itemizing = item_limited > records["std_ded"]
    deduction = floored_at(item_limited, records["std_ded"])
    # The senior deduction is reported inside `ded` for a non-itemizer and
    # subtracted separately for an itemizer. Taxable income is the same
    # either way.
    senior_for_taxable_income = jnp.where(itemizing, records["senior_ded"], 0.0)
    deduction += jnp.where(~itemizing, records["senior_ded"], 0.0)
    taxable_income = floored_at(
        records["agi"]
        - deduction
        - records["pe_ded"]
        - records["qbi_ded"]
        - records["tip_ded"]
        - records["ot_ded"]
        - senior_for_taxable_income,
        0.0,
    )
    return {"item_ded_limited": item_limited, "itemizing": itemizing, "ded": deduction, "txbl_inc": taxable_income}
