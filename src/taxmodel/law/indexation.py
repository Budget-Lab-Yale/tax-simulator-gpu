"""Indexation of stated dollar amounts.

A dollar amount in the statute is grown by an index (CPI-U before the TCJA,
chained CPI-U after it, the average wage index for the Social Security taxable
maximum) and then rounded to the increment the statute names: down to the next
$25 for the ordinary brackets, up to the next $300 for the taxable maximum, to
the nearest $50 for others.

The index ratio is lagged by one year: the amount in force in year y uses
growth measured through y-1, because the IRS sets next year's tables from this
year's inflation.

Rounding to the nearest increment breaks a tie to the even multiple: $12.50 at
a $1 increment rounds to $12, $13.50 to $14. Ties occur by construction,
because the statute states round base values and the increments are $50 and
$100. The fixtures in tests/golden/fixtures/law check this.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from .errors import LawError

# The three rounding rules `i_direction` selects.
ROUND_DOWN = -1
ROUND_NEAREST = 0
ROUND_UP = 1


@dataclass(frozen=True)
class IndexTable:
    """Year-over-year growth of each index series the law can name.

    The series are CPI-U and chained CPI-U as the IRS measures them, the
    average wage index, and pre-1970 CPI-U spliced on for the capital gains
    basis chain.
    """

    growth: Mapping[str, Mapping[int, float]]

    @classmethod
    def from_rows(cls, rows: Iterable[tuple[str, int, float]]) -> IndexTable:
        growth: dict[str, dict[int, float]] = {}
        for series, year, rate in rows:
            growth.setdefault(str(series), {})[int(year)] = float(rate)
        return cls(growth)

    @property
    def series_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.growth))


def index_levels(
    measure: Mapping[int, str | None],
    table: IndexTable,
    simulated_years: Iterable[int],
    label: str,
) -> dict[int, float]:
    """Accumulate the price (or wage) level along a subparameter's chain.

    The level in year y is the compounded growth from the start of the chain,
    so the ratio of two levels is the growth between those two years. A year
    whose measure is missing is a freeze, and every later level is missing
    too (the TCJA froze the lifetime learning credit phaseout in 2020 by
    writing the measure out from that year). Indexation cannot resume after a
    freeze; lint.py refuses a config that writes one.
    """
    simulated = {int(y) for y in simulated_years}
    levels: dict[int, float] = {}
    level = 1.0
    broken_at: int | None = None
    frozen = False
    dead_years: list[int] = []

    for year in sorted(measure):
        series = measure[year]
        if series is None:
            # A missing measure is a freeze. Nothing later is indexed, and
            # nothing from here on is reported as missing data.
            frozen = True
            level = np.nan
        else:
            rate = table.growth.get(series, {}).get(year)
            if rate is None:
                # A live measure with no growth rate: the window runs past the
                # end of the series, the series has a gap, or the measure names
                # a series the index data does not carry. Left alone this
                # reverts the parameter to its nominal base-year value.
                if broken_at is None and not frozen:
                    broken_at = year
                level = np.nan
            else:
                level = level * (1.0 + rate)
        levels[year] = level
        if broken_at is not None and year in simulated and np.isnan(level):
            dead_years.append(year)

    if dead_years:
        named = sorted({s for s in measure.values() if s is not None})
        raise LawError(
            f'indexation series for {label} (measure "{"/".join(named)}") has no '
            f"growth rate in simulated year(s) "
            f"{', '.join(str(y) for y in dead_years)}. Either the simulation "
            "window runs past the end of the index series, the series has a "
            "gap, or the measure names a series absent from the index data. An "
            "unindexed year silently reverts the parameter to its nominal "
            "base-year value."
        )

    return levels


def index_ratios(
    measure: Mapping[int, str | None],
    i_base_year: np.ndarray,
    table: IndexTable,
    years: tuple[int, ...],
    label: str,
) -> np.ndarray:
    """The factor each slot's base value is multiplied by, `[year, slot]`.

    Growth from the statute's base year through the year before the one in
    force.
    """
    levels = index_levels(measure, table, years, label)

    ratios = np.full(i_base_year.shape, np.nan, dtype=np.float64)
    for row, year in enumerate(years):
        lagged = levels.get(year - 1, np.nan)
        for slot in range(i_base_year.shape[1]):
            base_year = i_base_year[row, slot]
            if np.isnan(base_year):
                # This slot is not indexed, as in the donut-hole design where
                # the bracket above the taxable maximum stays nominal.
                continue
            ratios[row, slot] = lagged / levels.get(int(base_year), np.nan)
    return ratios


def apply_indexation(
    base_value: np.ndarray,
    ratio: np.ndarray | None,
    direction: np.ndarray | None,
    increment: np.ndarray | None,
    label: str = "this parameter",
) -> np.ndarray:
    """The amount in force: the base value grown by the index, then rounded.

    A slot with no rounding rule or no index ratio keeps its base value. An
    infinite base value stays infinite under every rule, so `Inf` means "this
    limit does not bind". A rounding direction other than -1 (down), 0
    (nearest) or 1 (up) is refused; `label` names the parameter in that error.
    """
    if ratio is None or direction is None or increment is None:
        return base_value.astype(np.float64, copy=True)

    unknown = ~np.isin(direction, (ROUND_DOWN, ROUND_NEAREST, ROUND_UP)) & ~np.isnan(
        direction
    )
    if np.any(unknown):
        bad = sorted({float(d) for d in np.asarray(direction)[unknown]})
        raise LawError(
            f"{label} has i_direction {bad}; the rounding direction must be "
            "-1 (down), 0 (to the nearest increment) or 1 (up)"
        )

    with np.errstate(invalid="ignore", divide="ignore"):
        quotient = base_value * ratio / increment
        rounded = np.where(
            direction == ROUND_UP,
            np.ceil(quotient),
            np.where(
                direction == ROUND_NEAREST,
                # np.round breaks ties to even.
                np.round(quotient),
                np.floor(quotient),
            ),
        )
        indexed = rounded * increment

    keeps_base = np.isnan(direction) | np.isnan(ratio)
    return np.where(keeps_base, base_value, indexed)
