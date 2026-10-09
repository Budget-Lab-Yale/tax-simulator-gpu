"""The staleness check for the gains model's calibrated numbers.

Three things can have changed since the calibrate command last ran: the
declared targets, the upstream data vintages, or the gains model's source
code. The check reports each in one sentence.

The targets and the commit are read from the record of the last fit in the
derived-internals file. The data vintages are read from the calibration file a
run reads, where each of the two numbers records the Tax-Data and
Macro-Projections vintages it was fitted under. Other upstream interfaces do
not enter the fit and are not checked.

A changed target or data vintage stops the run, because the model would not
reproduce the targets the run declares. The fix is to re-run
`taxmodel-calibrate`; the `--accept-stale-calibration` flag lets the run go
ahead anyway. Changed code only warns, since a git diff cannot tell a comment
edit from a change to the selling choice.

The code check asks git what changed under the gains model's source folder
between the commit recorded at derivation and HEAD. Uncommitted edits are
outside this check; the run manifest records a dirty tree.
"""

from __future__ import annotations

import math
import subprocess
from collections.abc import Mapping
from pathlib import Path

from ..config.staleness import StalenessReport
from . import paths
from . import targets as target_definitions
from .derived import DerivedInternals, DerivedInternalsError, read_derived_internals
from .realization_file import RealizationFileError, read_realization_entries
from .targets import KgDynamicsTargets, TargetsError

# Two declared numbers count as equal when they agree this closely, so that
# a value written to yaml and read back does not look like a changed target.
SAME_NUMBER = 1e-12


def _readable(name: str) -> str:
    return name.replace("_", " ")


def _target_sentences(
    internals: DerivedInternals,
    declared: KgDynamicsTargets,
) -> list[str]:
    """Sentences for declared targets that differ from the fitted ones."""

    was = internals.derivation.targets
    now = declared.as_record()
    sentences = []

    for name, value_now in now.items():
        if name not in was:
            sentences.append(
                f"The gains model's calibrated numbers were derived without a "
                f"{_readable(name)} target, but this run declares one, of "
                f"{value_now:g}. Re-run the calibrate command to fit it."
            )
            continue
        value_then = was[name]
        if not math.isclose(value_then, value_now, rel_tol=SAME_NUMBER, abs_tol=SAME_NUMBER):
            sentences.append(
                f"This run declares a {_readable(name)} of {value_now:g}, but "
                f"the gains model's calibrated numbers were fit to "
                f"{value_then:g}. Until the calibrate command is re-run the "
                "model will not reproduce the target this run declares."
            )

    for name in was:
        if name not in now:
            sentences.append(
                f"The gains model's calibrated numbers were fit to a "
                f"{_readable(name)} target of {was[name]:g}, which this run "
                "no longer declares."
            )

    return sentences


def _realization_file_sentences(
    realization_path: Path,
    data_vintages: Mapping[str, str],
) -> list[str]:
    """Sentences for a fitted number in the realization file that was fitted
    under other data vintages.

    These are the two numbers a run reads. Each carries the vintages its fit
    ran under; a run on other vintages is reporting a number that was not
    fitted to its data.
    """

    try:
        entries = read_realization_entries(realization_path)
    except RealizationFileError as problem:
        return [
            (
                "This run reads the capital-gains model's two calibrated "
                f"numbers, but their record could not be read: {problem}"
            )
        ]

    sentences = []
    for entry in entries:
        for name, vintage_then in entry.derived_under.items():
            vintage_now = data_vintages.get(name)
            if vintage_now is None:
                sentences.append(
                    f"The {_readable(entry.name)} in {realization_path.name} was fitted against a "
                    f"{_readable(name)} vintage ({vintage_then}) that this run "
                    "does not use."
                )
            elif str(vintage_now) != str(vintage_then):
                sentences.append(
                    f"This run uses {_readable(name)} vintage {vintage_now}, "
                    f"but the {_readable(entry.name)} in {realization_path.name} "
                    f"({entry.value:g}) was fitted against vintage {vintage_then}."
                )
    return sentences


def _code_sentences(
    internals: DerivedInternals,
    repo_root: Path,
    source_folder: str,
) -> list[str]:
    """A sentence if the gains model's source has changed since the fit."""

    commit = internals.derivation.git_commit
    if not commit or not str(commit).strip():
        no_commit_recorded = (
            "The gains model's calibration record does not say which commit "
            "it was derived at, so whether the gains-model code has changed "
            "since cannot be checked."
        )
        return [no_commit_recorded]

    commit = str(commit).strip()
    try:
        finished = subprocess.run(
            [
                "git",
                "diff",
                "--name-only",
                f"{commit}..HEAD",
                "--",
                source_folder,
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as problem:
        git_would_not_run = (
            "The gains model's code could not be compared against the commit "
            f"it was calibrated at ({commit[:12]}): git would not run "
            f"({problem})."
        )
        return [git_would_not_run]

    if finished.returncode != 0:
        detail = finished.stderr.strip().splitlines()
        reason = detail[0] if detail else "git gave no reason"
        git_refused = (
            "The gains model's code could not be compared against the commit "
            f"it was calibrated at ({commit[:12]}): {reason}. That commit may "
            "no longer be in this repository."
        )
        return [git_refused]

    changed = [line.strip() for line in finished.stdout.splitlines() if line.strip()]
    if not changed:
        return []

    listed = ", ".join(changed[:5])
    if len(changed) > 5:
        listed += f", and {len(changed) - 5} more"
    return [
        f"The gains model's code has changed in {len(changed)} file"
        + ("" if len(changed) == 1 else "s")
        + f" since it was calibrated at commit {commit[:12]}: {listed}. The "
        "calibrated numbers were fit to the model as it was then."
    ]


def uses_the_gains_machinery(behavior: Mapping[str, object]) -> bool:
    """Whether this run reads the gains model's calibrated numbers.

    A behavior leg may say `kg_dynamics: none`, in which case the run skips
    the gains model and its calibration is not checked.
    """

    section = behavior.get(paths.KG_DYNAMICS_SECTION)
    if section is None:
        return False
    if isinstance(section, str):
        return section.strip().lower() not in {"none", ""}
    return bool(section)


# Printed above the findings when a run stops on a stale calibration. The
# report's `stops` hold changed targets or data vintages; its `warnings`
# hold changed gains-model code.
GAINS_STALENESS_EXPLANATION = (
    "STALE CAPITAL-GAINS CALIBRATION\n\n"
    "This run declares targets or reads data vintages that the gains\n"
    "model's calibrated numbers were not derived under, so the model\n"
    "will not reproduce what this run says it believes.\n\n"
    "The fix is to re-run the calibrate command, which takes minutes:\n"
    "    taxmodel-calibrate\n\n"
    "To run on the stale numbers anyway, knowing why, add\n"
    "    --accept-stale-calibration"
)


def check_for_run(
    behavior: Mapping[str, object],
    data_vintages: Mapping[str, str],
    *,
    repo_root: Path | str | None = None,
) -> StalenessReport:
    """The staleness check for one run, from its resolved configuration.

    Run planning calls this as it resolves a scenario's behavior leg. The
    check only reports; the caller decides whether to stop by calling
    ``raise_if_stopping``, so a multi-scenario runscript can collect every
    finding first.

    A run whose behavior leg does not bind the gains machinery is not checked.
    """

    found = StalenessReport(explanation=GAINS_STALENESS_EXPLANATION)
    if not uses_the_gains_machinery(behavior):
        return found

    internals_path = paths.file_named_by_behavior(
        behavior, paths.DERIVED_INTERNALS_KEY, paths.DEFAULT_DERIVED_INTERNALS_FILE
    )
    realization_path = paths.file_named_by_behavior(
        behavior, paths.REALIZATION_KEY, paths.DEFAULT_REALIZATION_FILE
    )
    targets_path = target_definitions.targets_path_from_behavior(behavior)

    # A calibration record that cannot be read stops the run, because whether
    # the numbers match this run's targets is then unknown.
    try:
        internals = read_derived_internals(internals_path)
    except DerivedInternalsError as problem:
        found.stops.append(
            "This run uses the capital-gains machinery, but the record of "
            f"its calibration could not be read: {problem}"
        )
        return found

    try:
        declared = target_definitions.read_targets(targets_path)
    except TargetsError as problem:
        found.stops.append(
            "This run uses the capital-gains machinery, but the targets it "
            "declares could not be read, so whether the calibrated numbers "
            f"still match them is unknown: {problem}"
        )
        return found

    root = Path(repo_root) if repo_root is not None else Path.cwd()
    found.stops.extend(_target_sentences(internals, declared))
    found.stops.extend(_realization_file_sentences(realization_path, data_vintages))
    found.warnings.extend(
        _code_sentences(internals, root, paths.GAINS_MODEL_SOURCE_FOLDER)
    )
    return found
