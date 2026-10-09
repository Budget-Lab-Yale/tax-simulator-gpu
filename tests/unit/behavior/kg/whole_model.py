"""One scenario's gains model, end to end, for the tests that want one.

A run solves the selling choice for a whole group of scenarios at once, in one
compiled program (``behavior/kg/many_scenarios.py``).  Several tests want a
single scenario's cell tables from its households and law: as an end-to-end
check that the population cells, the schedule solver and the stock recurrence
fit together on real household data, and as the answer a batched group of
several scenarios is graded against.

So this runs the batched program on a group of one.  It is test scaffolding,
not a second implementation: every line of arithmetic lives in the model, and
what this adds is the made-up law and prices in the shapes the program reads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import jax
import jax.numpy as jnp

from taxmodel.behavior.kg import (
    many_scenarios,
    prepass,
    regime,
    schedule,
    stepper,
)
from taxmodel.behavior.kg.inputs import Calibration
from taxmodel.law.compile import FILING_STATUSES
from taxmodel.years.stacking import stack_lanes

Array = jnp.ndarray

# The three cell rate matrices the schedule solver reads, in the names
# `years/gains_solve.py` gives them.
PRICE_NAMES = ("baseline", "reform", "law_only")


def scenario_law(
    n_years: int,
    *,
    regime_codes: Sequence[int],
    section_121: float = 0.0,
    death_gain_exclusion: float = 0.0,
    joint_death_gain_exclusion: float | None = None,
    corporate_exposure: Mapping[str, float] | None = None,
    equity_markdown: float = 0.0,
) -> many_scenarios.ScenarioLaw:
    """A made-up law on gains at death, the same in every year.

    The compiled law states the two exclusions by filing status; here one
    amount serves every status, and the joint filing status's death-gain
    exclusion defaults to the same amount.
    """
    n_statuses = len(FILING_STATUSES)
    joint = (
        death_gain_exclusion
        if joint_death_gain_exclusion is None
        else joint_death_gain_exclusion
    )
    return many_scenarios.ScenarioLaw(
        section_121_by_status=jnp.full((n_years, n_statuses), section_121),
        death_gain_by_status=jnp.full((n_years, n_statuses), death_gain_exclusion),
        joint_death_gain=jnp.full(n_years, joint),
        regime_code_by_class=jnp.tile(jnp.asarray(regime_codes), (n_years, 1)),
        becomes_by_class=jnp.tile(
            jnp.asarray(regime.what_becomes_of_a_dollar(regime_codes)),
            (n_years, 1, 1),
        ),
        corporate_exposure_by_class=jnp.asarray(
            [
                (corporate_exposure or {}).get(f"value.{name}", 0.0)
                for name in prepass.DEEMED_ASSET_CLASSES
            ]
        ),
        equity_markdown=jnp.full(n_years, equity_markdown),
    )


def cell_prices(
    n_cells: int,
    n_years: int,
    *,
    baseline_rate_on_gains: float,
    reform_rate_on_gains: float,
) -> dict[str, Array]:
    """Cell prices that are the same in every cell and every year, [cell, year]."""

    def everywhere(value: float) -> Array:
        return jnp.full((n_cells, n_years), value)

    return {
        "baseline": everywhere(baseline_rate_on_gains),
        "reform": everywhere(reform_rate_on_gains),
        "law_only": everywhere(reform_rate_on_gains),
    }


def scenario_prices(
    prices: Mapping[str, Array], calibration: Calibration
) -> many_scenarios.ScenarioPrices:
    """One scenario's prices and calibrated numbers in the program's form."""
    return many_scenarios.ScenarioPrices(
        marginal_rate_baseline=prices["baseline"],
        marginal_rate_reform=prices["reform"],
        marginal_rate_reform_law_only=prices["law_only"],
        elasticity=jnp.asarray(calibration.elasticity, dtype=jnp.float64),
        timeable_share=jnp.asarray(calibration.timeable_share, dtype=jnp.float64),
        timing_reference_wedge=jnp.asarray(
            calibration.timing_reference_wedge, dtype=jnp.float64
        ),
    )


def run_one_scenario(
    records_by_year: Sequence[prepass.Records],
    years: Sequence[int],
    *,
    macro_root: str | Path,
    charity: prepass.CharitableBequestSettings,
    calibration: Calibration,
    law: many_scenarios.ScenarioLaw,
    prices: Mapping[str, Array] | None = None,
    frozen: bool = False,
) -> stepper.StockRecurrenceResult:
    """Households and law in, one scenario's cell tables out.

    The batched program on a group of one.  ``frozen`` runs the frozen pass,
    which pins turnover to the baseline pace and reads no price, so ``prices``
    is ignored and zeros go in, as the model does for a run's frozen pass.
    Nothing here converts compensation into capital gain: the conversion
    channel works household by household and is priced by its own module, and
    the tests that care about it drive that module directly.
    """
    n_years = len(years)
    # Sixty-three ages, 18 to 80, with 80 a pooled top-coded cell.
    n_cells = prepass.KG_N_AGES
    shared = many_scenarios.shared_solve_inputs(
        records_by_year,
        years,
        macro_root=macro_root,
        charity=charity,
        n_ages=n_cells,
    )
    laws = stack_lanes([law])
    cells = many_scenarios.scenario_cells(shared, laws, n_ages=n_cells)
    if frozen or prices is None:
        prices = dict.fromkeys(PRICE_NAMES, jnp.zeros((n_cells, n_years)))
    stacked_prices = stack_lanes([scenario_prices(prices, calibration)])
    solved = many_scenarios.solve_schedules_for_scenarios(
        shared,
        laws,
        cells,
        stacked_prices,
        frozen=frozen,
        shaping=many_scenarios.ProgramShapingSettings.from_calibration(calibration),
    )
    if not frozen:
        # The solver checks run on the host, on this scenario's slice of the
        # bundle, because a compiled program cannot raise.
        bundle = solved.bundle._replace(
            **{
                name: getattr(solved.bundle, name)[0]
                for name in solved.bundle._fields
                if name != "frozen" and getattr(solved.bundle, name) is not None
            }
        )
        inputs = jax.tree.map(lambda leaf: leaf[0], solved.inputs)
        schedule.check_schedule(bundle, inputs)

    nothing = jnp.zeros((1, n_years, n_cells))
    # The corporate channel's markdown is a stock: the level of the price
    # decline that year, against the corporate equity standing behind the
    # cell's gains. A frozen pass marks nothing down.
    corporate_gain_debit = (
        nothing
        if frozen
        else laws.equity_markdown[:, :, None] * cells.corporate_exposed_value
    )
    stepped = many_scenarios.step_stock_for_scenarios(
        shared, solved.bundle, nothing, corporate_gain_debit, frozen=frozen
    )
    return stepper.StockRecurrenceResult(
        cell_tables=stepped.cell_tables[0],
        stock_change_end_of_year=stepped.stock_change_end_of_year[0],
        conversions_by_household_leg=stepped.conversions_by_household_leg,
    )
