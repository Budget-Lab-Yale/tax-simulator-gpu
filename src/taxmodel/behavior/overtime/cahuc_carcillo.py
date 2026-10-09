"""Workers who already get overtime report more of it when overtime is untaxed.

The evidence is the French experiment of 2007-2012, which exempted overtime
hours from income tax and social contributions. Cahuc and Carcillo find that
reported overtime hours rose while total hours worked did not, which they read
as reporting and relabeling rather than more work:

    Pierre Cahuc and Stephane Carcillo, "The Detaxation of Overtime Hours:
    Lessons from the French Experiment", Journal of Labor Economics 32(2),
    April 2014, pages 361-400.

The response is a semi-elasticity: a percentage-point fall in the marginal tax
rate on overtime raises reported overtime by a fixed percent. Per earner,

    percent change in overtime =
        elasticity x (reform rate - baseline rate) x phase-in factor

applied only to earners who already have overtime. An exemption lowers the
rate, so the rate change is negative and the negative elasticity gives more
overtime.

The phase-in length is a parameter; three values are named below.

Base wages and overtime
-----------------------
`wages_base1` is the first earner's pay that is neither tips nor overtime, and
that earner's W-2 wages are `wages_base1 + tips1 + ot1`
(`src/taxmodel/taxes/statutory_order.py`). The French finding is relabeling,
not extra work, so every dollar of new overtime comes out of that earner's
base wages and total pay is unchanged.

An earner whose new overtime exceeds all of their base pay has base wages
floored at zero, and total pay rises by the part the floor could not fund.
`unfunded_overtime_dollars` computes that shortfall per record.

The prices this module reads are `mtr_ot1` and `mtr_ot2`, the marginal tax
rate on a dollar of each earner's overtime pay.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from taxmodel.behavior import loader

Array = jnp.ndarray
Records = Mapping[str, Array]
Prices = Mapping[str, Array]

# The elasticity implied by Cahuc and Carcillo's numbers, computed from three
# published inputs.
#
# Average overtime hours per week in the French data, Cahuc and Carcillo
# Table D4.
FRENCH_MEAN_OVERTIME_HOURS = 0.46
# Net rise in reported overtime hours per week after the exemption, Table 2
# column 6 row 3.
FRENCH_OVERTIME_HOURS_CHANGE = 0.2
# The French marginal tax rate the exemption removed: the average of the
# single and married rates at average wages, OECD Taxing Wages 2007-2008,
# Table I7.
FRENCH_MARGINAL_RATE = (0.317 + 0.275) / 2
# Percent change in overtime per one-unit (100 percentage point) fall in the
# marginal rate on overtime. Negative because a lower rate means more overtime.
INTENSIVE_ELASTICITY = (
    FRENCH_OVERTIME_HOURS_CHANGE / FRENCH_MEAN_OVERTIME_HOURS
) / -FRENCH_MARGINAL_RATE

# The default phase-in length: the number of years the response takes to
# arrive in full. Zero would mean the full response arrives in the first year
# the exemption is in law.
THREE_YEAR_PHASE_IN = 3

DEFAULTS: dict[str, Any] = {
    "elasticity_intensive": INTENSIVE_ELASTICITY,
    "phase_in_years": THREE_YEAR_PHASE_IN,
    # First year the overtime exemption is in law.
    "first_year": 2025,
}

# The reform law's cap on the overtime deduction, as the frame names it. The
# frame joins every compiled law parameter on under its section-prefixed name.
OVERTIME_DEDUCTION_LIMIT = "below.ot_ded_limit"

# The current year and the reform law's overtime deduction limit come from the
# run, on the frame; `params` stands in for a caller testing the module on a
# hand-built frame. See `loader.run_fact`.
MANIFEST: dict[str, Any] = {
    "name": "overtime_cahuc_carcillo",
    # The function in this file that is the response.
    "entry_point": "overtime_cahuc_carcillo",
    # Overtime changes by the intensive elasticity times the change in the
    # marginal rate, so a zero elasticity relabels no pay.
    "zero_response": {"elasticity_intensive": 0.0},
    "family": "overtime",
    # The deduction limit is a bound-law column the frame carries under the
    # law's own name.
    "reads": [
        "ot1",
        "ot2",
        "wages_base1",
        "wages_base2",
        "year",
        OVERTIME_DEDUCTION_LIMIT,
    ],
    "writes": ["ot1", "ot2", "wages_base1", "wages_base2"],
    "prices": ["mtr_ot1", "mtr_ot2"],
    "discrete_prices": [],
    "draws": [],
}


def _phase_in_factor(params: Mapping[str, Any], year: Array) -> Array:
    """How much of the full response has arrived by this year, from 0 to 1.

    A phase-in of zero years means the whole response arrives in the first year
    the exemption is in law, so the factor is 1 from that year on. The
    phase-in length is traced data, so the zero case is handled with a
    `where` and not a Python branch.
    """
    first_year = jnp.asarray(params["first_year"], dtype=jnp.float64)
    phase_in_years = jnp.asarray(params["phase_in_years"], dtype=jnp.float64)
    years_since_reform = jnp.maximum(
        0.0, jnp.asarray(year, dtype=jnp.float64) - first_year
    )
    phases_in = phase_in_years > 0
    # The divisor is held at 1 where there is no phase-in, so the division
    # does not produce an infinity.
    arrived = years_since_reform / jnp.where(phases_in, phase_in_years, 1.0)
    return jnp.where(phases_in, jnp.minimum(1.0, arrived), 1.0)


def _earner_changes(
    records: Records,
    prices_reform: Prices,
    prices_baseline: Prices,
    params: Mapping[str, Any],
    earner: int,
) -> tuple[Array, Array, Array]:
    """One earner's new overtime, new base wages, and unfunded dollars.

    Unfunded dollars are the new overtime that the base-wage floor could not
    cover; this earner's total pay rises by that amount.
    """
    overtime = jnp.asarray(records[f"ot{earner}"])
    base_wages = jnp.asarray(records[f"wages_base{earner}"])

    rate_change = jnp.asarray(prices_reform[f"mtr_ot{earner}"]) - jnp.asarray(
        prices_baseline[f"mtr_ot{earner}"]
    )

    year = loader.run_fact(records, params, "year", "overtime")
    exemption_applies = loader.provision_applies(
        records, params, OVERTIME_DEDUCTION_LIMIT, "overtime"
    )
    # Only earners who already have overtime respond; nobody starts getting
    # overtime because of the tax change.
    responds = exemption_applies & (overtime > 0)

    percent_change = jnp.where(
        responds,
        params["elasticity_intensive"] * rate_change * _phase_in_factor(params, year),
        0.0,
    )
    new_overtime = overtime * (1.0 + percent_change)
    overtime_dollars_added = new_overtime - overtime

    # The new overtime is taken out of base wages, so total pay is conserved.
    # Only the positive part of base wages can fund overtime: a few records
    # whose Tax-Data tips and overtime exceed their wages carry negative base
    # wages, and a minimum against a negative
    # number would move pay when the response is zero. A blank base wage funds
    # nothing.
    fundable = jnp.maximum(jnp.nan_to_num(base_wages, nan=0.0), 0.0)
    wages_given_up = jnp.minimum(overtime_dollars_added, fundable)
    new_base_wages = base_wages - wages_given_up
    unfunded = overtime_dollars_added - wages_given_up

    return new_overtime, new_base_wages, unfunded


def overtime_cahuc_carcillo(
    records: Records,
    prices_reform: Prices,
    prices_baseline: Prices,
    params: Mapping[str, Any],
    draws: Mapping[str, Array],
) -> dict[str, Array]:
    """New overtime and base wages after the overtime-exemption response.

    Returns `{"ot1", "ot2", "wages_base1", "wages_base2"}`. `draws` is unused;
    the response is deterministic given the price change.

    A reform that does not move `mtr_ot1` or `mtr_ot2` returns overtime and
    base wages unchanged.
    """
    changed: dict[str, Array] = {}
    for earner in (1, 2):
        new_overtime, new_base_wages, _unfunded = _earner_changes(
            records, prices_reform, prices_baseline, params, earner
        )
        changed[f"ot{earner}"] = new_overtime
        changed[f"wages_base{earner}"] = new_base_wages
    return changed


def unfunded_overtime_dollars(
    records: Records,
    prices_reform: Prices,
    prices_baseline: Prices,
    params: Mapping[str, Any],
    draws: Mapping[str, Array],
) -> Array:
    """Dollars per household by which this module fails to conserve total pay.

    Zero for every household unless some earner's new overtime is larger than
    all of their base pay, in which case base pay stops at zero and the rest of
    the new overtime is an increase in total pay.
    """
    total = None
    for earner in (1, 2):
        _new_overtime, _new_base_wages, unfunded = _earner_changes(
            records, prices_reform, prices_baseline, params, earner
        )
        total = unfunded if total is None else total + unfunded
    return total
