"""The two declared targets for the capital-gains model, and the moment
definitions that turn them into quantities the model measures.

A person supplies the targets in economic units: how much realizations
respond to the tax rate on gains, in the long run and in the year a change is
announced. The calibrate command solves for the internal parameters that
reproduce them and writes those to the file :mod:`taxmodel.calibration.derived`
manages.

A targets file holds two entries, each a number and one line naming its
source. The moment definitions are constants at the top of this module, not
configuration.

The permanent target is the long-run realization elasticity. The model
measures it ten years out, under a permanent five percentage point increase
in the tax rate on gains: the log ratio of realizations under the shock to
realizations without it, divided by the average change in the marginal rate
the realized dollars faced, multiplied by the top statutory rate on long-term
gains (20 percent plus the 3.8 percent net investment income tax). The last
step converts the measured semi-elasticity to an elasticity.

The transitory target is the response to an announced rate increase, which
pulls realizations forward. The model measures the response in the
announcement year, under a rate increase announced one year before it takes
effect. Law is unchanged in the announcement year, so every dollar that moves
that year is retiming. The divisor is the average rate change in the year the
rate rises, since in the announcement year that change is zero.

The target values themselves are in `config/calibrations/kg/targets.yaml`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import paths

# The top statutory rate on long-term capital gains: 20 percent plus the
# 3.8 percent net investment income tax. Converts between a declared
# elasticity and the semi-elasticity the simulation measures.
TOP_RATE_DIVISOR = 0.238

# The permanent moment is a five percentage point increase in the gains
# rate, held forever, read ten years after the window opens.
PERMANENT_SHOCK_SIZE = 0.05
PERMANENT_HORIZON_YEARS = 10

# The priced calibration window is thirty-one years long, so that the
# solver's stationary terminal year is well past the year the target is read.
CALIBRATION_WINDOW_YEARS = 30

# The announcement moment is a two point increase announced one year early.
# The moment divides by the average rate change the realized dollars saw, so
# the two shock sizes remain comparable. The year of the announcement and
# the year the rise takes effect are ANNOUNCEMENT_YEAR_OFFSET and
# EFFECTIVE_YEAR_OFFSET in kg_moments.
ANNOUNCEMENT_SHOCK_SIZE = 0.02

# The order the two moments and the two parameters appear in throughout
# this package.
MOMENT_NAMES = (paths.PERMANENT_TARGET, paths.TRANSITORY_TARGET)
PARAMETER_NAMES = (paths.REALIZATION_ELASTICITY, paths.TIMEABLE_SHARE)


class TargetsError(Exception):
    """A targets file is missing something, or says something impossible."""


@dataclass(frozen=True)
class DeclaredTarget:
    """One declared target: a number, and whose judgment it is."""

    value: float
    source: str


@dataclass(frozen=True)
class KgDynamicsTargets:
    """The pair of targets the calibrate command fits the gains model to."""

    permanent_realization_elasticity: DeclaredTarget
    transitory_realization_elasticity: DeclaredTarget

    def as_pair(self) -> tuple[float, float]:
        """The two target values, in the order the moments are measured."""
        return (
            self.permanent_realization_elasticity.value,
            self.transitory_realization_elasticity.value,
        )

    def as_record(self) -> dict[str, float]:
        """The two target values, for the derivation record.

        The source lines are left out, so that editing a citation does not make
        the fitted numbers stale.
        """
        return {
            paths.PERMANENT_TARGET: float(
                self.permanent_realization_elasticity.value
            ),
            paths.TRANSITORY_TARGET: float(
                self.transitory_realization_elasticity.value
            ),
        }


def targets_from_mapping(body: Any, *, where: str = "the targets block") -> KgDynamicsTargets:
    """Read the two targets out of an already-parsed mapping."""

    if not isinstance(body, dict):
        raise TargetsError(f"{where} should be a mapping of target names to entries")

    entries = {}
    for name in (paths.PERMANENT_TARGET, paths.TRANSITORY_TARGET):
        entry = body.get(name)
        if entry is None:
            raise TargetsError(f"{where} does not declare {name}")
        if not isinstance(entry, dict):
            raise TargetsError(
                f"{name} in {where} should be a mapping with a value and a source"
            )
        if "value" not in entry:
            raise TargetsError(f"{name} in {where} has no value")
        source = entry.get("source")
        if not source or not str(source).strip():
            raise TargetsError(
                f"{name} in {where} has no source line. Every declared target "
                "needs a `source` line saying whose judgment the number is."
            )
        try:
            value = float(entry["value"])
        except (TypeError, ValueError) as problem:
            raise TargetsError(
                f"{name} in {where} has a value that is not a number: "
                f"{entry['value']!r}"
            ) from problem
        entries[name] = DeclaredTarget(value=value, source=str(source).strip())

    unknown = sorted(set(body) - {paths.PERMANENT_TARGET, paths.TRANSITORY_TARGET})
    if unknown:
        raise TargetsError(
            f"{where} declares targets the gains model does not have: "
            + ", ".join(unknown)
        )

    return KgDynamicsTargets(
        permanent_realization_elasticity=entries[paths.PERMANENT_TARGET],
        transitory_realization_elasticity=entries[paths.TRANSITORY_TARGET],
    )


def read_targets(path: Path | str) -> KgDynamicsTargets:
    """Read a targets file."""

    path = Path(path)
    if not path.exists():
        raise TargetsError(f"there is no targets file at {path}")
    body = yaml.safe_load(path.read_text()) or {}
    return targets_from_mapping(body, where=str(path))


def targets_path_from_behavior(behavior: dict[str, Any]) -> Path:
    """The targets file a resolved behavior leg points at.

    A behavior leg either switches the gains machinery off
    (``kg_dynamics: none``) or names the files it binds. When the leg names no
    targets file, the shared default applies.
    """

    return paths.file_named_by_behavior(
        behavior, paths.TARGETS_KEY, paths.DEFAULT_TARGETS_FILE
    )

