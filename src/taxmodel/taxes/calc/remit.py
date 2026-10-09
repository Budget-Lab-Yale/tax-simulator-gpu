"""Split tax liabilities into the payments that are booked to receipts."""

from collections.abc import Mapping

import jax.numpy as jnp

from taxmodel.taxes.calc._corners import capped_at, floored_at

Array = jnp.ndarray


def remit_taxes(records: Mapping[str, Array]) -> dict[str, Array]:
    """Allocate income-tax, credit, and payroll payments to remittance lines.

    Interest, dividends, and taxable capital gains are paid when the return is
    filed.  The remaining income-tax payment is withheld or paid quarterly.
    Refundable credits are recorded as filing-season payments, and the current
    model records every payroll-tax dollar as withheld or paid quarterly.

    Estate and wealth liabilities are not changed here; their receipt timing is
    applied only after the weighted totals are assembled.
    """
    agi = jnp.asarray(records["agi"])
    nonwithheld_income = (
        jnp.asarray(records["txbl_int"])
        + jnp.asarray(records["div_ord"])
        + jnp.asarray(records["div_pref"])
        + jnp.asarray(records["txbl_kg"])
    )
    nonwithheld_share = jnp.where(
        agi == 0,
        0.0,
        capped_at(floored_at(nonwithheld_income / agi, 0.0), 1.0),
    )
    income_tax_payment = jnp.asarray(records["liab_iit"]) - jnp.asarray(
        records["ref_other"]
    )
    income_tax_nonwithheld = income_tax_payment * nonwithheld_share
    return {
        "pmt_iit_nonwithheld": income_tax_nonwithheld,
        "pmt_iit_withheld": income_tax_payment - income_tax_nonwithheld,
        "pmt_refund_nonwithheld": jnp.asarray(records["refund"]),
        "pmt_refund_withheld": jnp.zeros_like(jnp.asarray(records["refund"])),
        "pmt_pr_nonwithheld": jnp.zeros_like(jnp.asarray(records["liab_pr"])),
        "pmt_pr_withheld": jnp.asarray(records["liab_pr"]),
    }
