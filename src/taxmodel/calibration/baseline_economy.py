"""The baseline economy the two realization moments are measured against.

This module runs the model once under current law, with the capital-gains
pre-pass, over a window of thirty-one years. The long-run moment is read ten
years in, leaving the solver's stationary terminal year well past it.

The baseline economy is thirty-one years of the age cells the gains model
runs on. For each cell it holds the stock of unrealized gain, the
realizations under current law, mortality, the chance a decedent's gain goes
to charity, and the marginal tax rate on the cell's realized dollars. Reform
law equals current law throughout; :mod:`taxmodel.calibration.kg_moments`
applies the shocks on top of this economy by raising the rate the cells face.
The population is not repriced under a shocked statute; the approximation is
described at ``kg_moments.raise_the_rate_on_gains``.

The two fitted numbers are not read here. The frozen pass is handed
placeholders and neither calibration file is opened. The baseline run
binds the gains machinery and no other behavior, so the calibration is
conditioned on nothing outside it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from taxmodel.behavior.gains import settings
from taxmodel.years import gains_solve

from . import kg_moments
from . import targets as target_definitions

# The year the projection window opens. The window runs thirty years past
# it, so the long-run moment, read ten years in, is read in 2035.
DEFAULT_FIRST_YEAR = 2025

# The behavior leg a calibration is derived under when the command line does
# not name one: the capital-gains model and no other margin.
DEFAULT_BEHAVIOR_LEG = "kg_dynamics_only"


@dataclass(frozen=True)
class BaselineEconomy:
    """The moment scenarios, and what they were built from.

    The measurement takes ``scenarios``. The remaining fields go into the
    derivation record: which years were run, which upstream data vintages were
    in force, and where the population came from.
    """

    scenarios: kg_moments.MomentScenarios
    years: tuple[int, ...]
    vintages: dict[str, str]
    population_root: Path
    behavior: str
    economy: str


def window_years(first_year: int = DEFAULT_FIRST_YEAR) -> tuple[int, ...]:
    """The calibration window: the opening year plus the thirty after it."""
    n_years = target_definitions.CALIBRATION_WINDOW_YEARS + 1
    return tuple(range(int(first_year), int(first_year) + n_years))


def build_moment_scenarios(
    *,
    config_root: str | Path = "config/scenarios",
    economy: str = "",
    behavior: str = DEFAULT_BEHAVIOR_LEG,
    first_year: int = DEFAULT_FIRST_YEAR,
    population_root: str | Path | None = None,
) -> BaselineEconomy:
    """Price the population under current law and lay out the two shocks.

    The model runs once under current law and keeps the age cells' marginal
    rates (`run/calibrate.price_gains_cells_under_current_law`). Those rates
    and the solve group become the gains model's solver inputs, with the
    baseline leg's rates also used for the reform leg. The result goes to
    ``kg_moments.moment_scenarios_from_baseline``, which applies the two
    shocks.
    """
    # Imported here because the pricing pass pulls in the whole simulation,
    # which the calibrate command's dry run does not need.
    from taxmodel.run import calibrate

    judgments = settings.read_response_settings()

    years = window_years(first_year)
    priced = calibrate.price_gains_cells_under_current_law(
        years,
        config_root=config_root,
        economy=economy or None,
        behavior=behavior or None,
        population_root=population_root,
    )

    group = priced.solve_group
    # One set of cell prices serves both legs of the selling choice, so the
    # schedule solver's reform rate matrix equals its baseline one. The
    # moment builder checks for that equality.
    prices = gains_solve.current_law_cell_prices(priced.rate_exports)
    baseline_inputs = gains_solve.schedule_inputs_for_scenario(group, 0, prices)
    baseline_inputs.validate()

    scenarios = kg_moments.moment_scenarios_from_baseline(
        baseline_inputs,
        gains_solve.population_cells(group),
        appreciation=judgments.appreciation,
        timing_reference_wedge=judgments.timing_reference_wedge,
        timing_window=judgments.timing_window,
    )
    return BaselineEconomy(
        scenarios=scenarios,
        years=years,
        vintages=priced.vintages,
        population_root=priced.population_root,
        behavior=behavior or DEFAULT_BEHAVIOR_LEG,
        economy=economy or "default",
    )


def describe(economy: BaselineEconomy) -> list[str]:
    """Lines describing what the command priced, for the console."""
    years: Sequence[int] = economy.years
    return [
        (
            f"Priced the population under current law for {years[0]} to "
            f"{years[-1]}, on the {economy.behavior} behavior leg and the "
            f"{economy.economy} economy leg."
        ),
        f"Population read from {economy.population_root}.",
        (
            "The long-run moment is read in "
            f"{years[target_definitions.PERMANENT_HORIZON_YEARS]}; the "
            "announcement moment in "
            f"{years[kg_moments.ANNOUNCEMENT_YEAR_OFFSET]}, the year before "
            f"the rise takes effect in "
            f"{years[kg_moments.EFFECTIVE_YEAR_OFFSET]}."
        ),
    ]
