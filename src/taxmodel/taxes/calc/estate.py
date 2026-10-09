"""Estate tax liability conditional on death."""

from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp

from taxmodel.frame.steps_jax import (
    ESTATE_ASSET_COLS,
    ESTATE_DEBT_COLS,
    sum_missing_as_zero,
)
from taxmodel.taxes.calc._corners import floored_at
from taxmodel.taxes.calc._schedules import schedule_tax

Array = jnp.ndarray



@dataclass(frozen=True)
class EstateLaw:
    exemption: Array
    brackets: Array
    rates: Array
    portability: Array
    income_tax_ded: Array


@dataclass(frozen=True)
class EstateParameters:
    """Frozen SOI measurement bridge, not reformable estate tax law."""

    reporting_rate: Array
    pass_through_valuation_rate: Array
    gift_ratio: Array
    bin_lower_bounds: Array
    deduction_shares: Array
    dsue_probabilities: Array
    dsue_shares: Array


def calc_estate(
    records: Mapping[str, Array], law: EstateLaw, params: EstateParameters
) -> dict[str, Array]:
    """Return no-DSUE/DSUE estate liabilities and distributable estate."""
    # The estate calculator first maps economic net worth into the SOI-reported
    # estate distribution before applying estate-tax law.
    economic_gross = sum_missing_as_zero(records, ESTATE_ASSET_COLS)
    estate_debts = sum_missing_as_zero(records, ESTATE_DEBT_COLS)
    pass_through_value = jnp.nan_to_num(records["value.pass_throughs"], nan=0.0)
    positive_gross = economic_gross > 0
    pass_through_share = jnp.where(
        positive_gross,
        pass_through_value / jnp.where(positive_gross, economic_gross, 1.0),
        0.0,
    )
    # Closely held pass-through wealth receives its additional minority-interest
    # and marketability discount within the bridge to the reported gross estate.
    reported_gross = (
        economic_gross
        * params.reporting_rate
        * (1.0 + (params.pass_through_valuation_rate - 1.0) * pass_through_share)
    )
    slot = jnp.searchsorted(params.bin_lower_bounds, reported_gross, side="right") - 1
    bin_index = jnp.maximum(0, slot)  # tie: an integer bin index, so no derivative flows
    # Reported gross estate selects the calibrated deduction and portability
    # shares from its SOI measurement bin.
    deduction_share = params.deduction_shares[bin_index]
    dsue_probability = params.dsue_probabilities[bin_index]
    dsue_share = params.dsue_shares[bin_index]
    estate_distributable = floored_at(
        reported_gross - estate_debts - deduction_share * reported_gross, 0.0
    )
    income_tax_deduction = records.get("estate_income_tax_ded", 0.0)
    concealed_fraction = records.get("estate_concealed_frac", 0.0)
    base_bump = records.get("estate_base_bump", 0.0)
    # Income tax at death and concealed wealth reduce only the taxable base;
    # lifetime gifts are added back after the zero floor, while the optional MTR
    # bump sits inside that floor to measure the next-dollar right derivative.
    estate_base = (
        floored_at(
            estate_distributable
            - law.income_tax_ded * income_tax_deduction
            - concealed_fraction * reported_gross
            + base_bump,
            0.0,
        )
        + params.gift_ratio * reported_gross
    )
    married = (records["filing_status"] == 2) & (
        jnp.nan_to_num(records["q_death2"], nan=0.0) > 0
    )

    # The schedule is evaluated at the estate base and at each unified-credit
    # exclusion; a married decedent receives the joint exclusion in either case.
    tentative_base = schedule_tax(estate_base, law.brackets, law.rates)
    tentative_joint_exclusion = schedule_tax(
        2.0 * law.exemption, law.brackets, law.rates
    )
    tentative_no_dsue_exclusion = schedule_tax(law.exemption, law.brackets, law.rates)
    tentative_dsue_exclusion = schedule_tax(
        law.exemption + dsue_share * reported_gross, law.brackets, law.rates
    )
    liability_joint = floored_at(tentative_base - tentative_joint_exclusion, 0.0)
    liability_no_dsue = floored_at(tentative_base - tentative_no_dsue_exclusion, 0.0)
    liability_dsue = floored_at(tentative_base - tentative_dsue_exclusion, 0.0)
    return {
        "liab_estate_nodsue": jnp.where(married, liability_joint, liability_no_dsue),
        "liab_estate_dsue": jnp.where(married, liability_joint, liability_dsue),
        "estate_p_dsue": jnp.where(
            married, 0.0, dsue_probability * (law.portability == 1)
        ),
        "estate_distributable": estate_distributable,
    }
