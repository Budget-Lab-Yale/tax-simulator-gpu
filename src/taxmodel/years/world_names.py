"""The world names, and tax results keyed by the household frame and law that made them."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

import jax.numpy as jnp

Array = jnp.ndarray
YearResults = Mapping[str, Array]


class Frame(str, Enum):
    """The household information used for one tax calculation."""

    RAW = "raw"
    FLOWS = "flows"
    MECHANICAL = "mechanical"
    BEHAVIOR = "behavior"


class Law(str, Enum):
    """The tax law applied to a household frame."""

    BASELINE = "baseline"
    REFORM = "reform"


# The one-word name of each evaluated cell, used in output files and column
# headers.
#
# * `baseline` -- current law on the household as Tax-Data delivers it.
# * `first_order` -- the reform's law on that same household, before any
#   income flow, wealth, or behavioral change.
# * `mechanical_ex_saving` -- the reform's law after the corporate-tax and
#   employer-payroll income adjustments.
# * `mechanical` -- the above plus the prior year's wealth shortfall marked
#   against the household's balance sheet.
# * `conventional` -- the above plus household behavior: the scenario's
#   behavior modules, with the capital-gains applier at the front where the
#   scenario runs the gains model. For a scenario whose behavioral parameters
#   are all at neutral values, this world equals `mechanical`.
WORLD_CELLS: dict[str, tuple[Frame, Law]] = {
    "baseline": (Frame.RAW, Law.BASELINE),
    "first_order": (Frame.RAW, Law.REFORM),
    "mechanical_ex_saving": (Frame.FLOWS, Law.REFORM),
    "mechanical": (Frame.MECHANICAL, Law.REFORM),
    "conventional": (Frame.BEHAVIOR, Law.REFORM),
}

WORLD_NAMES: tuple[str, ...] = tuple(WORLD_CELLS)

# The world running baseline law on the untouched household. A reform's
# revenue change is measured against it.
BASELINE_WORLD: str = WORLD_NAMES[0]

_CELL_WORLDS = {cell: name for name, cell in WORLD_CELLS.items()}


def world_cell(world: str) -> tuple[Frame, Law]:
    """The (frame, law) cell a world name refers to, refusing an unknown name."""
    try:
        return WORLD_CELLS[world]
    except KeyError:
        raise KeyError(
            f"{world!r} is not one of the five worlds: {', '.join(WORLD_NAMES)}"
        ) from None


def world_name(frame: Frame, law: Law) -> str:
    """The one-word name for an evaluated cell, refusing a cell without one."""
    try:
        return _CELL_WORLDS[(frame, law)]
    except KeyError:
        raise KeyError(
            f"({frame.value}, {law.value}) is not one of the five named worlds"
        ) from None


def describe_cell(frame: Frame, law: Law) -> str:
    """A cell's name for error messages; never raises.

    A named world is called by its name. A cell with no name (baseline law on
    the behavioral frame, say) is described by its frame and law.
    """
    named = _CELL_WORLDS.get((frame, law))
    if named is not None:
        return f"the {named} world"
    return f"the {frame.value} frame under {law.value} law"


class ResultNotEvaluatedError(KeyError):
    """The requested frame, law, and year were not evaluated."""


@dataclass
class ResultGrid:
    """Per-year tax results for the frame and law cells that were evaluated."""

    _cells: dict[tuple[Frame, Law], dict[int, YearResults]] = field(
        default_factory=dict
    )

    def put(self, frame: Frame, law: Law, year: int, results: YearResults) -> None:
        """Store one evaluated year, refusing an accidental overwrite."""
        key = (frame, law)
        years = self._cells.setdefault(key, {})
        if year in years:
            raise ValueError(
                f"results for {describe_cell(frame, law)} in {year} already exist"
            )
        years[year] = results

    def get(self, frame: Frame, law: Law, year: int) -> YearResults:
        """Return one evaluated year or state exactly which cell is absent."""
        try:
            return self._cells[(frame, law)][year]
        except KeyError as error:
            raise ResultNotEvaluatedError(
                f"results for {describe_cell(frame, law)} in {year} "
                "were not evaluated"
            ) from error
