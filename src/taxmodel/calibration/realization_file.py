"""The realization file: the calibration file a run reads.

A run reads its two capital-gains numbers from the file its behavior leg
binds, by default ``config/calibrations/kg/realization.yaml``. This module
writes that file for the calibrate command and reads back the record each
entry carries, so that the numbers a run uses and the conditions they were
fitted under cannot drift apart.

Each entry records the data vintages the fit ran under, in the ``derived_under``
block the staleness check compares against the vintages a later run reads, and
alongside it the declared targets and the git commit of the model the fit ran
against.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import paths

# The upstream interfaces the capital-gains calibration is conditioned on: the
# microdata the population is priced from and the projections the discount
# rates and appreciation come from. The other interfaces a run reads do not
# enter the fit, so a change in them does not make these numbers stale.
CALIBRATION_INTERFACES = ("tax_data", "macro_projections")

# The two entry names in the file, and the command that rewrites it.
ELASTICITY_ENTRY = "eta"
TIMEABLE_SHARE_ENTRY = "timeable_share"
REDERIVE_COMMAND = "taxmodel-calibrate"

HEADER = """\
# Generated file -- written by the calibrate command, not by hand.
#
# The two calibrated response parameters of the capital gains model, and the
# record of the fit that produced them. `taxmodel-calibrate` rewrites the whole
# file; anything typed in here by hand is lost the next time it runs.
#
# Read by the behavior leg. A scenario reaches these values by binding the
# `realization` piece of kg_dynamics to this file; a sweep over either number binds
# a different file of the same shape, so the grid point is recorded in the
# vintage.
#
# Each entry's `derived_under` block holds the data vintages the fit ran under.
# A run on other vintages stops on the staleness check; the ways past are
# re-running the calibrate command, overriding the value in an alternative, or
# `--accept-stale-calibration`. `fitted_at` holds the declared targets and the
# commit of the model the fit ran against; the achieved moments and the other
# notes on the solve are in derived_internals.yaml.
"""

ELASTICITY_TARGET = """\
Curvature of the nontax benefit of selling in the age and gain-share selling
choice. Fitted by the calibrate command so that a permanent {shock:.0%} rise in
every gains rate lowers realizations {horizon} years on by the declared
permanent realization elasticity of {permanent:g}, converted through the
{divisor:g} top-rate divisor, with the timeable share fitted at the same time
to the transitory target. Future values are discounted at the nominal ten-year
Treasury rate.
"""

TIMEABLE_SHARE_TARGET = """\
The share of a year's realizations that can be moved between years. Fitted by
the calibrate command, together with the elasticity, to a transitory
realization elasticity of {transitory:g}: a rate increase people see coming
pulls realizations forward, and the response is read in the year the increase
is announced.
"""


class RealizationFileError(Exception):
    """The realization file is missing something it must carry."""


@dataclass(frozen=True)
class RealizationEntry:
    """One fitted number from the realization file, with the record of the fit
    that produced it."""

    name: str
    value: float
    # The data vintages the fit ran under, by interface name.
    derived_under: dict[str, str]
    # The date the fit ran, ISO format.
    set: str


def write_realization_file(
    path: Path | str,
    *,
    elasticity: float,
    timeable_share: float,
    data_vintages: Mapping[str, str],
    targets: Mapping[str, float],
    git_commit: str | None,
    date: str,
) -> None:
    """Rewrite the file a run reads, replacing whatever was there."""

    from . import targets as target_definitions

    vintages = {
        name: str(data_vintages[name])
        for name in CALIBRATION_INTERFACES
        if name in data_vintages
    }
    missing = [name for name in CALIBRATION_INTERFACES if name not in vintages]
    if missing:
        raise RealizationFileError(
            "the calibration was run without a vintage for "
            + ", ".join(missing)
            + ", which the realization file has to record"
        )

    record = {
        "git_commit": git_commit,
        **{str(name): float(value) for name, value in targets.items()},
    }

    def entry(value: float, target_text: str) -> dict[str, object]:
        return {
            "value": float(value),
            "kind": "calibrated",
            "set": date,
            "target": " ".join(target_text.split()),
            "derived_under": dict(vintages),
            "invalidated_by": [],
            "rederive": REDERIVE_COMMAND,
            "fitted_at": dict(record),
        }

    body = {
        "_channel": {"role": "state"},
        ELASTICITY_ENTRY: entry(
            elasticity,
            ELASTICITY_TARGET.format(
                shock=target_definitions.PERMANENT_SHOCK_SIZE,
                horizon=target_definitions.PERMANENT_HORIZON_YEARS,
                permanent=targets.get(paths.PERMANENT_TARGET, float("nan")),
                divisor=target_definitions.TOP_RATE_DIVISOR,
            ),
        ),
        TIMEABLE_SHARE_ENTRY: entry(
            timeable_share,
            TIMEABLE_SHARE_TARGET.format(
                transitory=targets.get(paths.TRANSITORY_TARGET, float("nan"))
            ),
        ),
    }
    written = yaml.safe_dump(body, sort_keys=False, width=76)
    # A blank line before each entry, so the file reads as three blocks.
    for name in (ELASTICITY_ENTRY, TIMEABLE_SHARE_ENTRY):
        written = written.replace(f"\n{name}:\n", f"\n\n{name}:\n")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(HEADER + "\n" + written)


def read_realization_entries(path: Path | str) -> list[RealizationEntry]:
    """The two entries in the realization file and the vintages each was
    fitted under."""

    path = Path(path)
    if not path.exists():
        raise RealizationFileError(f"there is no calibration file at {path}")
    body = yaml.safe_load(path.read_text()) or {}
    if not isinstance(body, dict):
        raise RealizationFileError(f"{path} is not a mapping of names to entries")

    entries = []
    for name in (ELASTICITY_ENTRY, TIMEABLE_SHARE_ENTRY):
        block = body.get(name)
        if not isinstance(block, dict):
            raise RealizationFileError(f"{path} has no entry for {name}")
        vintages = block.get("derived_under")
        if not isinstance(vintages, dict) or not vintages:
            raise RealizationFileError(
                f"{path}'s {name} entry records no derived_under vintages, so "
                "whether the fitted number matches this run's data cannot be "
                f"checked. Re-run {REDERIVE_COMMAND}."
            )
        entries.append(
            RealizationEntry(
                name=name,
                value=float(block["value"]),
                derived_under={
                    str(key): str(value) for key, value in vintages.items()
                },
                set=str(block.get("set", "")),
            )
        )
    return entries
