"""The calibrate command's pricing pass under current law.

The capital-gains calibration needs the marginal tax rate each age cell faces
on a realized long-term gain, under current law, for every year of the window.
This file runs one current-law scenario through the same preparation and
frozen gains solve a run uses (`run/scenarios.py`) and returns the rate sums
it formed. Nothing here writes a run directory.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from taxmodel.behavior import loader as behavior_loader
from taxmodel.behavior.gains import settings
from taxmodel.config import DEFAULT_LAYER, leg_or_none
from taxmodel.config.errors import ConfigError
from taxmodel.config.runscript import (
    ScenarioRow,
)
from taxmodel.frame.population import prepare_population
from taxmodel.run.grouping import (
    compile_partition_laws,
    runs_the_gains_model,
    timed,
)
from taxmodel.run.plan import ScenarioSpec, population_root_for
from taxmodel.run.scenarios import (
    batch_scenario,
    gains_state_for_scenario,
    heir_structure,
    shared_gains_pieces,
)
from taxmodel.scenario import prepare
from taxmodel.scenario import provenance as scenario_provenance
from taxmodel.years import gains_solve, loop

# Scenario id of the pricing pass. It appears only in the log lines the pass
# prints.
CALIBRATION_SCENARIO_ID = "calibration_current_law"

# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PricedGainsCells:
    """The population priced under current law, as the gains model reads it.

    ``rate_exports`` holds the marginal tax rate each age cell faces on a
    realized long-term gain, for every year of the window, as the weighted
    sums the compiled program formed. `years/gains_solve.current_law_cell_prices`
    divides them out.

    ``solve_group`` is the gains solve group the pass was built on, a group of
    one scenario: the population's age cells, the current-law gains law, and
    the placeholder calibration. The calibrate command lays its solver inputs
    out from it. ``vintages`` are the upstream data vintages, which the
    derivation record reports.
    """

    years: tuple[int, ...]
    rate_exports: dict[str, Any]
    solve_group: gains_solve.GainsSolveGroup
    vintages: dict[str, str]
    population_root: Path


def price_gains_cells_under_current_law(
    years: Sequence[int],
    *,
    config_root: str | Path = scenario_provenance.DEFAULT_CONFIG_ROOT,
    economy: str | None = None,
    behavior: str | None = None,
    population_root: str | Path | None = None,
    provenance_root: str | Path = scenario_provenance.DEFAULT_PROVENANCE_ROOT,
) -> PricedGainsCells:
    """Run the population through the model once under current law.

    Only the calibrate command calls this. It reads the population, compiles
    current law, solves the frozen pass as a group of one scenario, and takes
    the whole window through the year loop once, the same as the first
    execution of a gains scenario in a run. It returns the age cells' marginal
    rates and writes no run directory.

    One execution is enough because the calibration applies its two shocks to
    the rates current law produced instead of pricing the population again
    under a shocked statute (`calibration/kg_moments.py` explains why that is
    second-order).

    The behavior leg must bind the capital-gains machinery. Its calibrated
    numbers are not read: the frozen pass is handed placeholders, and the
    selling choice is not solved here.
    """
    years = tuple(int(year) for year in years)
    config_root = Path(config_root)

    # The calibrate command produces the fitted numbers, so it does not
    # refuse to run when they are stale or missing.
    provenance = scenario_provenance.resolve_scenario_provenance(
        config_root,
        economy=economy,
        behavior=behavior,
        provenance_root=provenance_root,
        law_layers=(),
        accept_stale_calibration=True,
    )
    if not runs_the_gains_model(provenance):
        raise ConfigError(
            f"the behavior leg '{behavior or DEFAULT_LAYER}' binds no capital-gains "
            "machinery, so there is nothing for the calibrate command to fit. "
            "Name a behavior alternative whose kg_dynamics section binds the "
            "realization model, such as kg_dynamics_only."
        )

    root = (
        Path(population_root)
        if population_root is not None
        else population_root_for(config_root, economy)
    )
    with timed("reading the population block"):
        population = prepare_population(root, years)

    spec = ScenarioSpec(
        id=CALIBRATION_SCENARIO_ID,
        row=ScenarioRow(
            id=CALIBRATION_SCENARIO_ID,
            tax_law=DEFAULT_LAYER,
            economy=economy or "",
            behavior=behavior or "",
            years=list(years),
            detail="",
        ),
        law_layers=(),
    )
    economy_leg = economy or DEFAULT_LAYER

    with timed("compiling current law"):
        compiled = compile_partition_laws([spec], years, economy_leg, config_root)

    heirs = heir_structure(years, population, economy_leg, config_root)

    with timed("preparing the scenario and solving the frozen pass"):
        ready = prepare.prepare_scenario(
            population,
            years,
            config_root=config_root,
            alternative=None,
            economy=leg_or_none(economy_leg),
            behavior=leg_or_none(behavior or ""),
            name=spec.id,
            compiled_law=compiled.laws[0],
            compiled_baseline_law=compiled.compiled_baseline,
            provenance=provenance,
            shared_baseline=compiled.shared_baseline,
        )
        pieces = shared_gains_pieces(population, years, economy_leg, config_root)
        # The frozen pass, solved as a group of one scenario. The calibration's
        # fitted numbers do not exist yet, so the group is handed a placeholder
        # calibration. The frozen pass does not read them. Nothing is converted
        # and no selling choice is solved.
        solve_group = gains_solve.prepare_gains_solve_group(
            pieces.host_records,
            years,
            scenario_ids=[spec.id],
            compiled_laws=[ready.compiled_law],
            behavior_legs=[provenance.behavior_leg],
            macro_root=pieces.macro_root,
            corporate_asset_exposures=[ready.corporate_asset_exposure],
            equity_markdowns=[ready.equity_markdown_by_year],
            mechanical_equity_markdowns=[ready.mechanical_equity_markdown_by_year],
            calibrations=[settings.placeholder_calibration()],
        )
        (frozen,) = gains_solve.solve_frozen_tables_for_group(solve_group)
        gains_state = gains_state_for_scenario(
            pieces,
            ready,
            law=solve_group.laws_by_scenario[0],
            frozen_tables=frozen.first_order,
            mechanical_frozen_tables=frozen.mechanical,
            converts=False,
        )

    with timed("pricing the population under current law, compile included"):
        executed = loop.run_scenarios(
            population,
            [
                batch_scenario(
                    spec,
                    ready,
                    gains_state,
                    behavior_leg=provenance.behavior_leg,
                    plain_behavior_builder=behavior_loader.plain_behavior_builder(),
                )
            ],
            years=years,
            heir_structure=heirs,
            gains_cell_weights=pieces.cell_weights,
            program_cache={},
            # One scenario, so the compiled year carries one lane.
            scenarios_per_program=1,
        )

    return PricedGainsCells(
        years=years,
        rate_exports=dict(executed.runs[spec.id].gains_rate_exports),
        solve_group=solve_group,
        vintages=dict(provenance.vintages),
        population_root=root,
    )
