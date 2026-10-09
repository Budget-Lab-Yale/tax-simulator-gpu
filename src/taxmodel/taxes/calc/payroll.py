"""FICA and SECA employment-tax calculations."""

from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp

from taxmodel.taxes.calc._corners import floored_at
from taxmodel.taxes.calc._schedules import schedule_tax, stacked_tax_exclusive

Array = jnp.ndarray

# The two payroll taxes, the two sides that pay each, and the two earners on a
# return. The law names its schedules `{tax}_{side}_brackets` and
# `{tax}_{side}_rates`.
TAXES = ("oasdi", "hi")
SIDES = ("ee", "er")
EARNERS = (1, 2)


@dataclass(frozen=True)
class PayrollLaw:
    """Employment-tax exclusions, taxable share, and marginal schedules."""

    tips_exempt: Array
    ot_exempt: Array
    seca_taxable_rate: Array
    se_thresh: Array
    oasdi_ee_rates: Array
    oasdi_ee_brackets: Array
    oasdi_er_rates: Array
    oasdi_er_brackets: Array
    hi_ee_rates: Array
    hi_ee_brackets: Array
    hi_er_rates: Array
    hi_er_brackets: Array
    add_med_rates: Array
    add_med_brackets: Array


def _added_up(
    programs: tuple[dict[tuple[int, str, str], Array], ...],
    *,
    sides: tuple[str, ...] = SIDES,
    taxes: tuple[str, ...] = TAXES,
    earners: tuple[int, ...] = EARNERS,
) -> Array:
    """The amounts for these sides, taxes and earners in each program, added up.

    A program is FICA or SECA, each a table keyed by (earner, tax, side). The
    order of addition is fixed because a floating-point sum depends on its
    order: employee side before employer, FICA before SECA, OASDI before HI,
    first earner before second.
    """
    total = 0.0
    for side in sides:
        for program in programs:
            for tax in taxes:
                for earner in earners:
                    total = total + program[earner, tax, side]
    return total


def calc_pr(records: Mapping[str, Array], law: PayrollLaw) -> dict[str, Array]:
    """Return FICA and SECA liabilities, split by tax, side, and earner."""
    # Each earner's W-2 wages are that earner's base pay plus tips plus
    # overtime. Payroll-tax exemptions then remove the qualifying tips and
    # FLSA overtime again, while employer retirement contributions remain in
    # the wage base.
    wages1 = records["wages_base1"] + records["tips1"] + records["ot1"]
    wages2 = records["wages_base2"] + records["tips2"] + records["ot2"]
    gross_wages1 = (
        wages1
        + records["trad_contr_er1"]
        - records["tips1"] * law.tips_exempt
        - records["ot1"] * law.ot_exempt
    )
    gross_wages2 = (
        wages2
        + records["trad_contr_er2"]
        - records["tips2"] * law.tips_exempt
        - records["ot2"] * law.ot_exempt
    )
    gross_wages = gross_wages1 + gross_wages2

    se1 = records["sole_prop1"] + records["farm1"] + records["part_se1"]
    se2 = records["sole_prop2"] + records["farm2"] + records["part_se2"]
    se = se1 + se2
    reporting = se > law.se_thresh
    # SECA applies the statutory taxable share (92.35% under baseline law) only
    # when unit-level net self-employment income exceeds the reporting threshold.
    txbl_se1 = floored_at(se1, 0.0) * law.seca_taxable_rate * reporting
    txbl_se2 = floored_at(se2, 0.0) * law.seca_taxable_rate * reporting

    # Each earner owes each of the two taxes on each of the two sides: FICA on
    # wages, and SECA on self-employment earnings stacked above that earner's
    # wages so the OASDI taxable maximum is shared between the two.
    gross_wages_of = {1: gross_wages1, 2: gross_wages2}
    txbl_se_of = {1: txbl_se1, 2: txbl_se2}
    fica: dict[tuple[int, str, str], Array] = {}
    seca: dict[tuple[int, str, str], Array] = {}
    for earner in EARNERS:
        for tax in TAXES:
            for side in SIDES:
                brackets = getattr(law, f"{tax}_{side}_brackets")
                rates = getattr(law, f"{tax}_{side}_rates")
                fica[earner, tax, side] = schedule_tax(
                    gross_wages_of[earner], brackets, rates
                )
                seca[earner, tax, side] = stacked_tax_exclusive(
                    txbl_se_of[earner], gross_wages_of[earner], brackets, rates
                )

    # Additional Medicare tax uses combined wages and taxable self-employment
    # earnings for the unit and has no employer-side match.
    liab_add_med = schedule_tax(
        gross_wages + txbl_se1 + txbl_se2,
        law.add_med_brackets,
        law.add_med_rates,
    )

    liab_fica_er1 = _added_up((fica,), sides=("er",), earners=(1,))
    liab_fica_er2 = _added_up((fica,), sides=("er",), earners=(2,))
    liab_fica = _added_up((fica,))
    liab_seca_er = _added_up((seca,), sides=("er",))
    liab_seca = _added_up((seca,), sides=("ee",)) + liab_seca_er
    liab_oasdi = _added_up((fica, seca), taxes=("oasdi",))
    liab_hi = _added_up((fica, seca), taxes=("hi",)) + liab_add_med
    liab_pr_ee = _added_up((fica, seca), sides=("ee",)) + liab_add_med
    liab_pr_er = liab_fica_er1 + liab_fica_er2 + liab_seca_er

    # Earned income retains gross Schedule C, farm, and partnership income;
    # the SECA taxable-share adjustment is limited to payroll-tax liability.
    return {
        "gross_wages1": gross_wages1,
        "gross_wages2": gross_wages2,
        "gross_wages": gross_wages,
        "se1": se1,
        "se2": se2,
        "se": se,
        "ei1": wages1 + se1,
        "ei2": wages2 + se2,
        "ei": wages1 + se1 + wages2 + se2,
        "liab_fica_er1": liab_fica_er1,
        "liab_fica_er2": liab_fica_er2,
        "liab_fica": liab_fica,
        "liab_seca": liab_seca,
        "liab_seca_er": liab_seca_er,
        "liab_oasdi": liab_oasdi,
        "liab_hi": liab_hi,
        "liab_add_med": liab_add_med,
        "liab_pr_ee": liab_pr_ee,
        "liab_pr_er": liab_pr_er,
        "liab_pr": liab_oasdi + liab_hi,
    }
