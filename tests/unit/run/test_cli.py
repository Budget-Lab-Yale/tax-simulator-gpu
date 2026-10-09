"""The command line: operational flags only, and a dry run that writes nothing."""

from __future__ import annotations

import datetime as dt

import pytest

from taxmodel.run import cli, scenarios
from taxmodel.run.plan import plan_run
from tests.unit.run.conftest import RUNSCRIPT_HEADER

FIXED_CODE_VERSION = {"commit": "abcdef1234567890", "branch": "test", "dirty": False}

# What the command line is allowed to decide: where the output goes, whether the
# work runs here or as a Slurm job, and whether to run at all. Nothing else may
# join this list. Anything that changes a reported number, or what an archived
# run contains, belongs in a versioned file — the scenario folders or the
# runscript — so that a published estimate traces back to something somebody
# wrote down and committed. --accept-stale-calibration decides whether the run
# happens at all, so it is a command-line flag, not a runscript setting.
OPERATIONAL_FLAGS = {
    "-h",
    "--help",
    "--output-root",
    "--slurm",
    "--dry-run",
    "--accept-stale-calibration",
}

# The things a user might reach for on the command line that must stay in files.
# Each of these would change a reported number or what the run directory holds.
FLAGS_THAT_WOULD_BLEED = (
    "--years",
    "--alternative",
    "--reform",
    "--tax-law",
    "--economy",
    "--behavior",
    "--detail",
    "--variables",
    "--worlds",
    "--stack",
    "--mode",
    "--tax-data",
    "--vintage",
    "--no-marginal-rates",
    "--scenario",
)


@pytest.fixture
def runscript(tmp_path):
    path = tmp_path / "smoke.csv"
    path.write_text(RUNSCRIPT_HEADER + "baseline,default,default,default,2026:2027,\n")
    return path


def option_strings() -> set[str]:
    return {
        option
        for action in cli.build_parser()._actions
        for option in action.option_strings
    }


def test_the_command_line_holds_only_operational_flags():
    assert option_strings() == OPERATIONAL_FLAGS


def test_nothing_that_changes_a_reported_number_can_be_said_on_the_command_line():
    assert option_strings().isdisjoint(FLAGS_THAT_WOULD_BLEED)


def test_the_runscript_is_the_one_positional_argument():
    positionals = [
        action.dest for action in cli.build_parser()._actions if not action.option_strings
    ]
    assert positionals == ["runscript"]


def test_a_dry_run_describes_the_run_and_writes_nothing(runscript, tmp_path, capsys):
    output_root = tmp_path / "runs"
    assert cli.main([str(runscript), "--output-root", str(output_root), "--dry-run"]) == 0
    printed = capsys.readouterr().out
    assert "scenarios: 1" in printed
    assert "scenarios/baseline/totals.csv" in printed
    assert "nothing was computed and nothing was written" in printed
    assert not output_root.exists()


def test_the_slurm_submission_names_the_job_script_the_runscript_and_the_output_root(
    monkeypatch, runscript, tmp_path
):
    """Heavy compute never runs on a login node, so --slurm hands the run to sbatch."""
    submitted: dict[str, list[str]] = {}

    class Submission:
        stdout = "654321\n"

    def fake_run(command, **kwargs):
        submitted["command"] = command
        return Submission()

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert cli.SLURM_JOB_SCRIPT.is_file(), "the job script ships with the repository"
    assert cli.submit_to_slurm(runscript, tmp_path / "runs") == "654321"
    assert submitted["command"] == [
        "sbatch",
        "--parsable",
        str(cli.SLURM_JOB_SCRIPT),
        # The checkout to run is named outright, so the directory sbatch was
        # called from cannot decide which code the job executes.
        str(cli.SLURM_JOB_SCRIPT.resolve().parent.parent),
        str(runscript),
        str(tmp_path / "runs"),
    ]


def test_asking_for_slurm_submits_instead_of_running_here(monkeypatch, runscript, tmp_path, capsys):
    monkeypatch.setattr(cli, "submit_to_slurm", lambda *args: "654321")
    monkeypatch.setattr(
        scenarios,
        "execute_run",
        lambda *args, **kwargs: pytest.fail("a submitted run must not run here"),
    )
    assert (
        cli.main([str(runscript), "--slurm", "--output-root", str(tmp_path / "runs")])
        == 0
    )
    printed = capsys.readouterr().out
    assert "submitted Slurm job 654321" in printed


def test_a_run_inside_a_slurm_job_records_the_job_id(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    assert cli.where_this_is_running() == {"where": "slurm", "slurm_job_id": "123456"}


def test_a_run_on_this_machine_records_that_instead(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    assert cli.where_this_is_running() == {"where": "local", "slurm_job_id": None}


def test_the_cli_writes_where_the_driver_planned(monkeypatch, runscript, tmp_path, capsys):
    """The one thing the command line does to the plan: choose its output root."""
    planned = plan_run(
        runscript,
        output_root=tmp_path / "runs",
        code_version=FIXED_CODE_VERSION,
        started_at=dt.datetime(2026, 8, 3, 14, 30, 5, tzinfo=dt.UTC),
    )
    assert str(tmp_path / "runs") in str(planned.run_directory)

    def refuse_to_run(*args, **kwargs):
        raise AssertionError("a dry run must not execute anything")

    monkeypatch.setattr(scenarios, "execute_run", refuse_to_run)
    assert cli.main([str(runscript), "--output-root", str(tmp_path / "runs"), "--dry-run"]) == 0
    capsys.readouterr()
