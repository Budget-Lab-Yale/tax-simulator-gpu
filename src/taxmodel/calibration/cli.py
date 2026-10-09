"""The `calibrate` command.

The command reads the declared targets and solves for the two numbers inside
the gains model that reproduce them (the realization elasticity and the
timeable share). It writes the two numbers, with the data vintages they were
fitted under, to the calibration file a run reads, and writes the record of the
solve to the derived-internals file. A model run never calibrates on its own.

Run it from the repository root as::

    uv run taxmodel-calibrate --config-root config/scenarios --economy ...

It solves the full model several times, so on real data it runs in a Slurm
job.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from taxmodel.behavior.gains.settings import read_fitted_calibration
from taxmodel.config import scenario as scenario_config

from . import baseline_economy, derived, kg_moments, newton, paths, realization_file
from . import targets as target_definitions

# The economically allowed range of each parameter. The elasticity is a
# positive response of realizations to the net-of-tax price; the timeable
# share is a share of a year's realizations, between zero and one.
SENSIBLE_RANGES = ((0.0, 20.0), (0.0, 1.0))


def build_parser() -> argparse.ArgumentParser:
    """The command-line arguments."""

    parser = argparse.ArgumentParser(
        prog="taxmodel-calibrate",
        description=(
            "Solve for the two numbers inside the capital-gains model that "
            "reproduce the declared realization targets. The numbers go to "
            "the calibration file a run reads; the record of the solve goes "
            "to the derived-internals file."
        ),
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=paths.DEFAULT_TARGETS_FILE,
        help="the declared targets file to fit to",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=paths.DEFAULT_DERIVED_INTERNALS_FILE,
        help="the derived-internals file, which records the solve",
    )
    parser.add_argument(
        "--realization-file",
        type=Path,
        default=paths.DEFAULT_REALIZATION_FILE,
        help=(
            "the realization file a run reads, which holds the fitted "
            "realization elasticity and timeable share"
        ),
    )
    parser.add_argument(
        "--config-root",
        type=Path,
        default=Path("config/scenarios"),
        help="the scenario configuration tree",
    )
    parser.add_argument(
        "--economy",
        default="",
        help=(
            "the economy-leg alternative whose data vintages the calibration "
            "is derived under; the default layer when not given"
        ),
    )
    parser.add_argument(
        "--behavior",
        default=baseline_economy.DEFAULT_BEHAVIOR_LEG,
        help=(
            "the behavior-leg alternative the baseline economy is run on. It "
            "must bind the capital-gains model and no other response, so that "
            "the calibration does not depend on other margins"
        ),
    )
    parser.add_argument(
        "--first-year",
        type=int,
        default=baseline_economy.DEFAULT_FIRST_YEAR,
        help=(
            "the year the calibration window opens. The window runs thirty "
            "years past this; the long-run moment is read ten years in"
        ),
    )
    parser.add_argument(
        "--population-root",
        type=Path,
        default=None,
        help=(
            "the bridge population folder to price. Left out, the economy "
            "leg's own Tax-Data pin says which one, as in an ordinary run"
        ),
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=newton.NewtonSettings.tolerance,
        help="how close a moment must come to its target",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=newton.NewtonSettings.max_iterations,
        help="how many Newton steps to take before giving up; each runs the model five times",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="read and report the targets, then stop without running the model",
    )
    return parser


def _data_vintages(config_root: Path, economy: str) -> dict[str, str]:
    """The upstream vintages this calibration is derived under."""

    layer = scenario_config.load_defaults(config_root, "economy")
    resolved = scenario_config.resolve(config_root, layer, economy or None)
    return scenario_config.interface_vintages(resolved)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    declared = target_definitions.read_targets(args.targets)
    print(f"Declared targets, read from {args.targets}:")
    for name, entry in (
        (paths.PERMANENT_TARGET, declared.permanent_realization_elasticity),
        (paths.TRANSITORY_TARGET, declared.transitory_realization_elasticity),
    ):
        print(f"  {name.replace('_', ' ')}: {entry.value:g} ({entry.source})")
    print(
        "The permanent target is measured "
        f"{target_definitions.PERMANENT_HORIZON_YEARS} years out under a "
        f"{target_definitions.PERMANENT_SHOCK_SIZE:.2%} rise in the rate on "
        "gains; the transitory target is measured in the year a rise is "
        "announced. Both are converted through the "
        f"{target_definitions.TOP_RATE_DIVISOR:g} top-rate divisor."
    )

    # The solve starts from the values a run reads today, which is where the
    # last fit landed.
    starting_values = read_fitted_calibration()
    print(
        f"Starting the solve at the values in {paths.DEFAULT_REALIZATION_FILE}: "
        f"realization elasticity {starting_values.elasticity:.6g} and timeable share "
        f"{starting_values.timeable_share:.6g}."
    )

    vintages = _data_vintages(args.config_root, args.economy)
    print("Data vintages in force: " + (", ".join(
        f"{name} {value}" for name, value in sorted(vintages.items())
    ) or "none recorded"))

    if args.dry_run:
        print("Dry run: stopping before the model runs.")
        return 0

    economy = baseline_economy.build_moment_scenarios(
        config_root=args.config_root,
        economy=args.economy,
        behavior=args.behavior,
        first_year=args.first_year,
        population_root=args.population_root,
    )
    for line in baseline_economy.describe(economy):
        print(line)
    scenarios = economy.scenarios

    settings = newton.NewtonSettings(
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
    )
    try:
        result = newton.solve(
            kg_moments.moment_evaluation(scenarios),
            targets=declared.as_pair(),
            start=(starting_values.elasticity, starting_values.timeable_share),
            settings=settings,
            parameter_names=target_definitions.PARAMETER_NAMES,
            moment_names=target_definitions.MOMENT_NAMES,
            sensible_ranges=SENSIBLE_RANGES,
            report=print,
        )
    except newton.NewtonFailure as failure:
        print(str(failure), file=sys.stderr)
        print("Nothing was written; both calibration files are unchanged.", file=sys.stderr)
        return 1

    from taxmodel.config import manifest  # imported here; no other path needs it

    elasticity, timeable_share = result.parameters
    git_commit = manifest.git_state(Path.cwd()).get("commit")
    date = derived.today()

    # Each of the two numbers carries the Tax-Data and Macro-Projections
    # vintages the priced run resolved, so that a later run on other data
    # stops on the staleness check.
    realization_file.write_realization_file(
        args.realization_file,
        elasticity=elasticity,
        timeable_share=timeable_share,
        data_vintages=economy.vintages,
        targets=declared.as_record(),
        git_commit=git_commit,
        date=date,
    )
    print(
        f"Wrote {args.realization_file}, the file runs read: realization elasticity "
        f"{elasticity:.6g} (a change of "
        f"{elasticity - starting_values.elasticity:+.6g}) and timeable share "
        f"{timeable_share:.6g} (a change of "
        f"{timeable_share - starting_values.timeable_share:+.6g})."
    )

    internals = derived.DerivedInternals(
        derived_from=str(args.targets),
        achieved_moments=dict(
            zip(target_definitions.MOMENT_NAMES, result.moments, strict=True)
        ),
        derivation=derived.DerivationRecord(
            targets=declared.as_record(),
            git_commit=git_commit,
            date=date,
            notes=(
                f"Newton's method converged in {result.iterations} "
                + ("step" if result.iterations == 1 else "steps")
                + "; the furthest a moment finished from its target was "
                + f"{result.worst_residual:.3g}."
            ),
        ),
    )
    derived.write_derived_internals(args.output, internals)
    print(f"Wrote {args.output}, the record of the solve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
