"""Reading and writing the derived-internals file.

The file is the calibrate command's record of its last solve: the targets
it fit to, the moments it achieved, the git commit, and the date. The two
fitted numbers and the data vintages they were fitted under are in the
calibration file a run reads (see :mod:`.realization_file`). The staleness check reads
the targets and the commit from this record.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import paths

HUMAN_HEADER = """\
# Derived internals for the capital-gains model. WRITTEN BY THE MACHINE.
#
# No human edits this file. The `calibrate` command rewrites the whole file
# each time it runs; anything typed in here by hand is lost.
#
# This is the record of the last solve. The two fitted numbers, and the data
# vintages they were fitted under, are in realization.yaml.
#
# What you edit instead is the targets file, named under derived_from below.
# Change a target, then run calibrate.
"""


class DerivedInternalsError(Exception):
    """The derived-internals file is missing, or is not what it claims."""


@dataclass(frozen=True)
class DerivationRecord:
    """What the last solve fit to, and when."""

    # The declared target values that were fit to, by target name.
    targets: dict[str, float]
    # The commit of the gains-model source when the solve ran. None when the
    # solve ran outside a git repository; the staleness check reports that
    # case as unknown.
    git_commit: str | None
    # The date the solve ran, ISO format.
    date: str
    # Notes for a reader: how many Newton steps it took, how close it
    # landed. Nothing reads this field.
    notes: str = ""


@dataclass(frozen=True)
class DerivedInternals:
    """The record of the gains model's last calibration solve."""

    derivation: DerivationRecord
    # Where the targets were read from when the solve ran.
    derived_from: str = str(paths.DEFAULT_TARGETS_FILE)
    # The moments as the solve last measured them.
    achieved_moments: dict[str, float] = field(default_factory=dict)


def write_derived_internals(path: Path | str, internals: DerivedInternals) -> None:
    """Write the file, header first, replacing whatever was there."""

    path = Path(path)
    body: dict[str, Any] = {
        "derived_from": internals.derived_from,
        "achieved_moments": {
            name: float(value) for name, value in internals.achieved_moments.items()
        },
        "derivation": {
            "targets": {
                name: float(value)
                for name, value in internals.derivation.targets.items()
            },
            "git_commit": internals.derivation.git_commit,
            "date": internals.derivation.date,
            "notes": internals.derivation.notes,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HUMAN_HEADER + "\n" + yaml.safe_dump(body, sort_keys=False))


def read_derived_internals(path: Path | str) -> DerivedInternals:
    """Read the derived-internals file. Keys the record does not carry are
    ignored."""

    path = Path(path)
    if not path.exists():
        raise DerivedInternalsError(
            f"there is no calibration record at {path}. Run the calibrate "
            "command to produce it."
        )
    body = yaml.safe_load(path.read_text()) or {}
    if not isinstance(body, dict):
        raise DerivedInternalsError(f"{path} is not a mapping of names to values")

    record = body.get("derivation")
    if not isinstance(record, dict):
        raise DerivedInternalsError(f"{path} has no derivation record")

    return DerivedInternals(
        derived_from=str(body.get("derived_from", paths.DEFAULT_TARGETS_FILE)),
        achieved_moments={
            str(name): float(value)
            for name, value in (body.get("achieved_moments") or {}).items()
        },
        derivation=DerivationRecord(
            targets={
                str(name): float(value)
                for name, value in (record.get("targets") or {}).items()
            },
            git_commit=(
                str(record["git_commit"])
                if record.get("git_commit") is not None
                else None
            ),
            date=str(record.get("date", "")),
            notes=str(record.get("notes", "")),
        ),
    )


def today() -> str:
    """The date to stamp a derivation with."""

    return dt.datetime.now(dt.UTC).date().isoformat()
