"""End to end on the real population: baseline law scored against itself.

Three years, the full sample, baseline law used as both the baseline and the
reform. Nothing has changed, so every channel of the reform's revenue effect
must be zero and the wealth bathtub must stay empty. That is the
cheapest true statement that the whole projection is wired together: the
population block, the frame steps, the two laws, the calculator chain, the
wealth channel, the compiled projection year, and the reporting layer all ran,
and the one answer we know in advance came out right.

Zero to the last bit the arithmetic can promise, which is not quite exactly
zero. The baseline sweep and the four reform worlds are computed in one
compiled program, and a compiler is free to add the same numbers in a
different order in one world than in another, so a population total of $1.7
trillion can land a fraction of a cent apart. The tolerance below is one part
in a trillion of the baseline total -- about a thousand times the rounding
this can produce, and about a trillion times smaller than any tax change
anyone would report.

    python tests/null_reform_smoke.py <tax-data root> <first year> <last year>
"""

from __future__ import annotations

import sys

import numpy as np

from taxmodel.behavior import loader as behavior_loader
from taxmodel.frame.population import prepare_population
from taxmodel.reporting import assemble
from taxmodel.scenario import prepare
from taxmodel.years import loop, world_names

# This script prints revenue in billions to keep the columns readable. The
# files a run writes are in dollars.
DOLLARS_PER_BILLION = 1e9

DEFAULT_TAX_DATA = (
    "/nfs/roberts/project/pi_nrs36/shared/model_data/"
    "Tax-Data-Bridge/v1/2026070814/baseline"
)
DEFAULT_FIRST_YEAR = 2026
DEFAULT_LAST_YEAR = 2028

# One part in a trillion of the baseline receipts the difference is measured
# against. See the module docstring for why this is not zero.
RELATIVE_TOLERANCE = 1e-12


def _scale(projection: assemble.ProjectionRun) -> dict[str, float]:
    """The largest baseline total of each column, which sets what "zero" means."""
    baseline = projection.totals[
        (world_names.Frame.RAW, world_names.Law.BASELINE)
    ]
    return {
        column: float(np.abs(np.asarray(values)).max())
        for column, values in baseline.items()
    }


def failures(projection: assemble.ProjectionRun) -> list[str]:
    """Every way this run failed to be a null reform. Empty means it passed."""
    problems: list[str] = []
    scale = _scale(projection)

    def too_big(column: str, values: np.ndarray) -> np.ndarray:
        return np.abs(np.asarray(values)) > RELATIVE_TOLERANCE * scale.get(column, 0.0)

    for difference, totals in projection.channel_totals.items():
        for column, values in totals.items():
            moved = too_big(column, values)
            if moved.any():
                years = [
                    year for year, did in zip(projection.years, moved, strict=True) if did
                ]
                problems.append(
                    f"{difference.description}: {column} moved in {years} "
                    f"(largest {np.abs(np.asarray(values)).max():.6g}, against a "
                    f"baseline of {scale.get(column, 0.0):.6g})"
                )

    for column, values in projection.total_effect.items():
        if too_big(column, values).any():
            problems.append(f"the whole reform effect moved {column}")

    # The tub is filled out of the tax change, so it inherits the same
    # tolerance: measured against total receipts, not against zero dollars.
    receipts = sum(scale.values())
    shortfall = np.asarray(projection.wealth_history.cell_shortfall)
    if np.abs(shortfall).max() > RELATIVE_TOLERANCE * receipts:
        problems.append(
            f"the wealth shortfall is not zero: largest cell holds "
            f"{np.abs(shortfall).max():.6g}"
        )
    inflow = np.asarray(projection.wealth_history.cell_inflow)
    if np.abs(inflow).max() > RELATIVE_TOLERANCE * receipts:
        problems.append(
            f"the wealth channel took in {np.abs(inflow).max():.6g} in a cell"
        )
    return problems


def run_the_null_reform(tax_data: str, years: range) -> assemble.ProjectionRun:
    """Prepare baseline law as its own reform, run it, and arrange the tables.

    The same three steps a production run takes, with one scenario instead of
    many: read the population, prepare the scenario on the host
    (`prepare.prepare_scenario`), and run it through the batch runner
    (`years/loop.run_scenarios`), which is the model's one compute path. One
    scenario means one lane, so the compiled projection year is traced at a
    width of one.
    """
    requested = tuple(int(year) for year in years)
    population = prepare_population(tax_data, requested)
    prepared = prepare.prepare_scenario(population, requested)
    scenario = loop.BatchScenario(
        name="baseline law as its own reform",
        program=loop.ScenarioProgram(
            switch_fingerprint=prepared.switch_fingerprint,
            frame_builder=prepared.frame_builder,
            tax_calculator=prepared.tax_calculator,
            price_calculator=prepared.price_calculator,
            # Baseline law against itself binds no behavioral response.
            behavior_builder=behavior_loader.plain_behavior_builder(),
            taxes_and_prices=prepared.taxes_and_prices,
        ),
        reform_laws=prepared.reform_laws,
        baseline_laws=prepared.baseline_laws,
        wealth_profiles=prepared.wealth_profiles,
    )
    batched = loop.run_scenarios(
        population, [scenario], scenarios_per_program=1, years=requested
    )
    return assemble.projection_from_simulation(prepared, batched.runs[scenario.name])


def main(argv: list[str]) -> int:
    tax_data = argv[1] if len(argv) > 1 else DEFAULT_TAX_DATA
    first = int(argv[2]) if len(argv) > 2 else DEFAULT_FIRST_YEAR
    last = int(argv[3]) if len(argv) > 3 else DEFAULT_LAST_YEAR
    years = range(first, last + 1)

    print(f"population: {tax_data}")
    print(f"years: {first}-{last}, baseline law as its own reform")
    projection = run_the_null_reform(tax_data, years)

    print("\nrevenue totals under baseline law, in billions of dollars")
    for column, values in projection.totals[
        (world_names.Frame.RAW, world_names.Law.BASELINE)
    ].items():
        billions = np.asarray(values) / DOLLARS_PER_BILLION
        print(f"  {column}: {np.round(billions, 1)}")

    problems = failures(projection)
    if problems:
        print("\nTHE NULL REFORM DID NOT COME OUT NULL")
        for line in problems:
            print(f"  {line}")
        return 1
    print(
        "\nevery channel is zero to within one part in a trillion of receipts, "
        "and the wealth stock stayed empty"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
