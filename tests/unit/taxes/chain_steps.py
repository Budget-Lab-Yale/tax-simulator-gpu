"""Run the chain one step at a time, keeping what each step produced.

`statutory_order.run_taxes` runs the calculators in statutory order and hands
back the finished columns. It does not say which calculator produced which
column, and several columns are written more than once on the way down the
form, so the finished frame cannot be read backwards to find out.

`run_taxes_by_step` below calls the model's step functions itself, one at a
time, in the order `run_taxes` calls them, and keeps each one's output under
the step's name.

This is a second statement of that order, which is only worth having because
`test_statutory_order.py` checks it against the first: running the steps this way has to
produce exactly the columns `run_taxes` produces, value for value. If someone
reorders the chain and does not reorder this file, that test fails.

Nothing in the model imports this. It lives under `tests/` because the model
has one way of running the chain.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

from taxmodel.taxes import statutory_order
from taxmodel.taxes.calc.estate import EstateParameters, calc_estate
from taxmodel.taxes.calc.payroll import calc_pr
from taxmodel.taxes.calc.remit import remit_taxes
from taxmodel.taxes.calc.wealth import calc_wealth

Array = jnp.ndarray


def run_taxes_by_step(
    records: Mapping[str, Array],
    law,
    estate_params: EstateParameters | None = None,
    double_run: bool | None = None,
) -> tuple[dict[str, Array], dict[str, dict[str, Array]]]:
    """Every tax for one record-year, plus what each step produced.

    The first return value is what `run_taxes` returns. The second is one
    dictionary per step, keyed by the step's name, in the order the steps ran.
    The whole 1040 is one step here, `do_1040`, because that is the
    granularity the outer chain works at; the charitable double run's two legs
    are kept separately beside it.
    """
    absent = statutory_order.missing_record_columns(
        records, with_estate=estate_params is not None
    )
    if absent:
        raise KeyError(f"the frame is missing: {', '.join(absent)}")

    out: dict[str, Array] = {}
    frame = dict(records)
    steps: dict[str, dict[str, Array]] = {}

    def step(name: str, produced: dict[str, Array]) -> None:
        steps[name] = produced
        out.update(produced)
        frame.update(produced)

    step("derive_vars", statutory_order.derive_vars(frame))
    step("calc_pr", calc_pr(frame, law.payroll))

    if double_run is None:
        double_run = statutory_order.charity_double_run_applies(law)
    if double_run:
        above = statutory_order.run_1040(frame, law, "above")
        item = statutory_order.run_1040(frame, law, "item")
        _, income_tax = statutory_order._choose_cheaper_leg(above, item)
        steps["do_1040_leg_above"] = dict(above)
        steps["do_1040_leg_item"] = dict(item)
    else:
        income_tax = statutory_order.run_1040(frame, law, None)
    step("do_1040", income_tax)

    if estate_params is not None:
        step("calc_estate", calc_estate(frame, law.estate, estate_params))

    step("calc_wealth", calc_wealth(frame["net_worth"], law.wealth))
    step("filer_and_expanded_income", statutory_order.filer_and_expanded_income(frame))
    step("remit_taxes", remit_taxes(frame))
    return out, steps
