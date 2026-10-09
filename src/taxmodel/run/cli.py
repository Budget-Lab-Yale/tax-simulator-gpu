"""The terminal command that runs one runscript and writes one run directory.

    taxmodel-run config/runscripts/clausing_v2.csv --dry-run
    taxmodel-run config/runscripts/clausing_v2.csv --slurm

The flags are operational only: where the output goes, whether the work runs
here or as a Slurm job, and whether to run at all. Everything that affects a
reported number, such as the years, the reform, the detail columns, the
stacking mode, and the population vintage, is read from versioned files that
the run copies into its own directory, so a published estimate can be traced
to a committed file. `tests/unit/run/test_cli.py`
asserts the flag set stays that way.

A dry run parses the runscript, resolves every folder and file it names,
expands any stacks, prints the run directory it would write and everything that
would land in it, and runs nothing.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from taxmodel.run import scenarios
from taxmodel.run.plan import (
    describe_run_plan,
    plan_run,
)

# The Slurm job script the `--slurm` flag submits. It requests the resources a
# whole-population run needs and calls this same command inside the job.
SLURM_JOB_SCRIPT = Path("jobs/runscript_run.sbatch")


def build_parser() -> argparse.ArgumentParser:
    """The command line: a runscript and four operational flags."""
    parser = argparse.ArgumentParser(
        prog="taxmodel-run",
        description="Run every scenario a runscript names and write a run directory.",
    )
    parser.add_argument(
        "runscript",
        type=Path,
        help="the runscript to run: which scenarios this run produces",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="where run directories go, instead of the scratch default in "
        "config/output_roots.yaml",
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help=f"submit {SLURM_JOB_SCRIPT} instead of running here, and print the "
        "job ID. A whole-population run does not fit on a login node",
    )
    parser.add_argument(
        "--accept-stale-calibration",
        action="store_true",
        help="run on capital-gains numbers calibrated to different targets or "
        "different data vintages than this run declares. The usual fix is to "
        "re-run taxmodel-calibrate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="read and check everything, print the run directory and the files "
        "it would hold, and run nothing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    plan = plan_run(args.runscript, output_root=args.output_root)
    if args.dry_run:
        for line in describe_run_plan(plan):
            print(line)
        print("dry run: nothing was computed and nothing was written")
        return 0

    if args.slurm:
        job_id = submit_to_slurm(args.runscript, args.output_root)
        print(f"submitted Slurm job {job_id} for {args.runscript}")
        print(f"it will write {plan.run_directory}")
        return 0

    run_directory = scenarios.execute_run(
        plan,
        execution=where_this_is_running(),
        accept_stale_calibration=args.accept_stale_calibration,
    )
    print(f"wrote {run_directory}")
    return 0


def where_this_is_running() -> dict[str, object]:
    """Local or inside a Slurm job, for the run manifest's execution section."""
    job_id = os.environ.get("SLURM_JOB_ID")
    return {
        "where": "slurm" if job_id else "local",
        "slurm_job_id": job_id,
    }


def submit_to_slurm(runscript: Path, output_root: Path | None) -> str:
    """Hand the run to Slurm and return the job ID `sbatch` printed."""
    if not SLURM_JOB_SCRIPT.is_file():
        raise FileNotFoundError(
            f"the Slurm job script {SLURM_JOB_SCRIPT} is not there; run from the "
            "repository root"
        )
    # The job script is passed the checkout that holds this job script, so the
    # job runs the same code as the command that submitted it.
    checkout = SLURM_JOB_SCRIPT.resolve().parent.parent
    command = [
        "sbatch",
        "--parsable",
        str(SLURM_JOB_SCRIPT),
        str(checkout),
        str(runscript),
    ]
    if output_root is not None:
        command.append(str(output_root))
    submitted = subprocess.run(command, capture_output=True, text=True, check=True)
    return submitted.stdout.strip()


if __name__ == "__main__":
    sys.exit(main())
