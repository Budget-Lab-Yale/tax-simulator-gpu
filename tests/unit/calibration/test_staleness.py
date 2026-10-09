"""The staleness check for the capital-gains model's calibration."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from taxmodel.calibration import derived, paths, realization_file, staleness
from taxmodel.config.errors import StaleCalibrationError

REPO_ROOT = Path(__file__).resolve().parents[3]

# Two real commits in this repository's history. The gains model's source
# folder changed between them: 057b598 merged the bathtub stepper, and
# 0b00c9c reworked the schedule bundle both halves read.
BEFORE_THE_GAINS_MODEL_CHANGED = "057b598"

VINTAGES = {"tax_data": "2026070814", "macro_projections": "2026022522"}

# The fitted numbers written into the test's calibration file.
ELASTICITY = 1.6342
TIMEABLE_SHARE = 0.2452


def current_commit() -> str:
    finished = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return finished.stdout.strip()


def internals(commit: str | None, targets_fit_to=None):
    return derived.DerivedInternals(
        derivation=derived.DerivationRecord(
            targets=targets_fit_to
            or {paths.PERMANENT_TARGET: -0.6, paths.TRANSITORY_TARGET: 1.2},
            git_commit=commit,
            date="2026-08-04",
        ),
    )


def behavior_leg(
    tmp_path,
    record,
    permanent: float = -0.6,
    transitory: float = 1.2,
    realization_vintages=None,
):
    """A resolved behavior leg pointing at the three files the check reads,
    all written into the test's own directory.

    This is the shape `check_for_run` reads: a `kg_dynamics` section naming
    the derived-internals file the last fit was recorded in, the calibration
    file a run reads its two numbers from, and the targets file saying what
    the run believes about how realizations respond.
    """

    internals_path = tmp_path / "derived_internals.yaml"
    derived.write_derived_internals(internals_path, record)
    targets_path = tmp_path / "targets.yaml"
    targets_path.write_text(
        f"{paths.PERMANENT_TARGET}:\n"
        f"  value: {permanent}\n"
        "  source: a judgment, for the test\n"
        f"{paths.TRANSITORY_TARGET}:\n"
        f"  value: {transitory}\n"
        "  source: a judgment, for the test\n"
    )
    realization_path = tmp_path / "realization.yaml"
    realization_file.write_realization_file(
        realization_path,
        elasticity=ELASTICITY,
        timeable_share=TIMEABLE_SHARE,
        data_vintages=dict(VINTAGES if realization_vintages is None else realization_vintages),
        targets={
            paths.PERMANENT_TARGET: permanent,
            paths.TRANSITORY_TARGET: transitory,
        },
        git_commit=record.derivation.git_commit,
        date=record.derivation.date,
    )
    return {
        paths.KG_DYNAMICS_SECTION: {
            paths.TARGETS_KEY: str(targets_path),
            paths.DERIVED_INTERNALS_KEY: str(internals_path),
            paths.REALIZATION_KEY: str(realization_path),
        }
    }


def test_the_fitted_numbers_carry_the_vintages_they_were_fitted_under(tmp_path):
    behavior = behavior_leg(tmp_path, internals(current_commit()))

    entries = realization_file.read_realization_entries(
        behavior[paths.KG_DYNAMICS_SECTION][paths.REALIZATION_KEY]
    )

    assert [entry.name for entry in entries] == ["eta", "timeable_share"]
    for entry in entries:
        assert entry.derived_under == VINTAGES


def test_a_run_on_the_data_the_fitted_numbers_were_fitted_under_passes(tmp_path):
    behavior = behavior_leg(tmp_path, internals(current_commit()))

    found = staleness.check_for_run(behavior, VINTAGES, repo_root=REPO_ROOT)

    assert found.stops == []


def test_a_fitted_number_fitted_under_other_data_stops_the_run(tmp_path):
    doctored = dict(VINTAGES, tax_data="2031010100")
    behavior = behavior_leg(
        tmp_path, internals(current_commit()), realization_vintages=doctored
    )

    found = staleness.check_for_run(behavior, VINTAGES, repo_root=REPO_ROOT)

    assert len(found.stops) == 2
    assert all("was fitted against vintage 2031010100" in line for line in found.stops)
    with pytest.raises(StaleCalibrationError):
        found.raise_if_stopping()


def test_nothing_has_moved_and_the_check_says_nothing(tmp_path):
    behavior = behavior_leg(tmp_path, internals(current_commit()))

    found = staleness.check_for_run(behavior, VINTAGES, repo_root=REPO_ROOT)

    assert found.stops == []
    assert found.warnings == []


def test_a_changed_target_gets_one_sentence_naming_it(tmp_path):
    behavior = behavior_leg(tmp_path, internals(current_commit()), permanent=-0.8)

    found = staleness.check_for_run(behavior, VINTAGES, repo_root=REPO_ROOT)

    assert len(found.stops) == 1
    assert "permanent realization elasticity" in found.stops[0]
    assert "-0.8" in found.stops[0] and "-0.6" in found.stops[0]


def test_a_changed_data_vintage_gets_one_sentence_for_each_number(tmp_path):
    behavior = behavior_leg(tmp_path, internals(current_commit()))

    found = staleness.check_for_run(
        behavior, {**VINTAGES, "tax_data": "2026080114"}, repo_root=REPO_ROOT
    )

    assert len(found.stops) == 2
    for line in found.stops:
        assert "tax data" in line
        assert "2026080114" in line and "2026070814" in line
    assert any("eta in realization.yaml" in line for line in found.stops)
    assert any("timeable share in realization.yaml" in line for line in found.stops)


def test_a_new_vintage_of_an_input_the_fit_does_not_use_does_not_stop_the_run(
    tmp_path,
):
    """Only the Tax-Data and Macro-Projections vintages enter the fit."""
    behavior = behavior_leg(tmp_path, internals(current_commit()))
    run_vintages = {
        **VINTAGES,
        "off_model_estimates": "20270101",
        "cost_recovery_simulator": "202701011200",
    }

    found = staleness.check_for_run(behavior, run_vintages, repo_root=REPO_ROOT)

    assert found.stops == []


def test_changed_gains_model_code_gets_one_sentence_naming_the_files(tmp_path):
    """Exercised against this repository's real history."""

    behavior = behavior_leg(tmp_path, internals(BEFORE_THE_GAINS_MODEL_CHANGED))

    found = staleness.check_for_run(behavior, VINTAGES, repo_root=REPO_ROOT)

    assert found.stops == []
    assert len(found.warnings) == 1
    assert "code has changed" in found.warnings[0]
    assert "src/taxmodel/behavior/kg/" in found.warnings[0]
    assert BEFORE_THE_GAINS_MODEL_CHANGED in found.warnings[0]


def test_a_missing_commit_is_said_out_loud_rather_than_crashing(tmp_path):
    behavior = behavior_leg(tmp_path, internals(None))

    found = staleness.check_for_run(behavior, VINTAGES, repo_root=REPO_ROOT)

    assert found.stops == []
    assert len(found.warnings) == 1
    assert "does not say which commit" in found.warnings[0]


def test_a_commit_this_repository_does_not_have_is_said_out_loud(tmp_path):
    behavior = behavior_leg(
        tmp_path, internals("0123456789abcdef0123456789abcdef01234567")
    )

    found = staleness.check_for_run(behavior, VINTAGES, repo_root=REPO_ROOT)

    assert found.stops == []
    assert len(found.warnings) == 1
    assert "could not be compared" in found.warnings[0]


def test_all_three_kinds_of_drift_are_reported_together(tmp_path):
    behavior = behavior_leg(
        tmp_path,
        internals(BEFORE_THE_GAINS_MODEL_CHANGED),
        permanent=-0.8,
        transitory=1.5,
    )

    found = staleness.check_for_run(
        behavior, {"tax_data": "2026080114"}, repo_root=REPO_ROOT
    )

    # Two targets moved. For each of the two fitted numbers, the Tax-Data
    # vintage moved and the Macro-Projections vintage is missing, which is
    # four more findings. The code moved too, which only warns.
    assert len(found.stops) == 6
    assert len(found.warnings) == 1


def test_a_run_with_the_gains_machinery_off_is_never_checked():
    """A run that does not use the gains model is not checked against its calibration."""

    assert staleness.check_for_run({"kg_dynamics": "none"}, VINTAGES).clean
    assert staleness.check_for_run({"modules": []}, VINTAGES).clean


def test_a_run_that_cannot_find_its_calibrated_numbers_stops(
    tmp_path,
):
    behavior = {
        "kg_dynamics": {
            "targets": str(paths.DEFAULT_TARGETS_FILE),
            "derived_internals": str(tmp_path / "not_there.yaml"),
        }
    }

    said = staleness.check_for_run(behavior, VINTAGES, repo_root=REPO_ROOT)

    # Unreadable calibrated numbers stop the run: whether they match what
    # this run believes is not unknown-and-probably-fine, it is unknown.
    assert len(said.stops) == 1
    assert said.warnings == []
    assert "could not be read" in said.stops[0]
    assert "calibrate command" in said.stops[0]


def test_a_run_reads_the_targets_and_internals_its_behavior_leg_names(tmp_path):
    internals_path = tmp_path / "derived_internals.yaml"
    derived.write_derived_internals(internals_path, internals(current_commit()))
    behavior = {
        "kg_dynamics": {
            "targets": str(paths.DEFAULT_TARGETS_FILE),
            "derived_internals": str(internals_path),
        }
    }

    # The default targets file declares the -0.6 and +1.2 these internals
    # were fit to, and the commit is the one we are on, so nothing has moved.
    assert staleness.check_for_run(behavior, VINTAGES, repo_root=REPO_ROOT).clean


# --------------------------------------------------------------------------
# Which findings stop the run, and which only warn
# --------------------------------------------------------------------------


def test_a_changed_target_stops_the_run_and_names_the_fix(tmp_path):
    """The model will not reproduce what this run says it believes."""
    internals_path = tmp_path / "derived_internals.yaml"
    was = internals(current_commit())
    derived.write_derived_internals(internals_path, was)
    targets_path = tmp_path / "targets.yaml"
    targets_path.write_text(
        "permanent_realization_elasticity: -0.4\n"
        "transitory_realization_elasticity: 1.2\n"
    )
    behavior = {
        "kg_dynamics": {
            "targets": str(targets_path),
            "derived_internals": str(internals_path),
        }
    }

    found = staleness.check_for_run(behavior, VINTAGES, repo_root=REPO_ROOT)

    assert found.stops, "a target this run does not believe in has to stop it"
    with pytest.raises(StaleCalibrationError, match="accept-stale-calibration"):
        found.raise_if_stopping()

    # The flag typed at run time is the only way past.
    found.raise_if_stopping(accept_stale=True)


def test_a_changed_data_vintage_stops_the_run(tmp_path):
    internals_path = tmp_path / "derived_internals.yaml"
    derived.write_derived_internals(internals_path, internals(current_commit()))
    behavior = {
        "kg_dynamics": {
            "targets": str(paths.DEFAULT_TARGETS_FILE),
            "derived_internals": str(internals_path),
        }
    }
    moved = {name: "9999999999" for name in VINTAGES}

    found = staleness.check_for_run(behavior, moved, repo_root=REPO_ROOT)

    assert found.stops
    with pytest.raises(StaleCalibrationError):
        found.raise_if_stopping()


def test_changed_gains_model_code_only_warns(tmp_path):
    """A git diff cannot tell a comment edit from a Bellman rewrite."""
    internals_path = tmp_path / "derived_internals.yaml"
    # An old commit, so the code check finds the gains model has moved since.
    old = internals("HEAD~1")
    derived.write_derived_internals(internals_path, old)
    behavior = {
        "kg_dynamics": {
            "targets": str(paths.DEFAULT_TARGETS_FILE),
            "derived_internals": str(internals_path),
        }
    }

    found = staleness.check_for_run(behavior, VINTAGES, repo_root=REPO_ROOT)

    assert found.stops == []
    found.raise_if_stopping()
