"""The stacked table: each provision's own contribution to a package.

A stacked reform is scored as a ladder of runs. The first run puts provision
1 over baseline law, the second puts provisions 1 and 2 over baseline law, and
so on, so each run reports a whole-package-so-far effect against the same
baseline. A provision's own effect is the step from one rung to the next.

Because each rung is a complete run against the same baseline, the steps add
up to the last rung's effect in every year and in every channel. The test
suite checks this to the cent (`tests/tolerances.py`), since floating point
does not make the cancellation exact on numbers in the trillions.

One table holds a whole stack, wide in years: a leading `channel` column, one
block per channel, one row per provision inside that block, and a Total row
closing it. The combined-total block comes first, before the four channels it
decomposes into.

The table reports nominal dollars only. Baseline dollars and share of GDP
would need the deflator and the GDP series, which reporting does not read off
disk (`revenue.py`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from taxmodel.config.law_stacks import ProvisionIncrement
from taxmodel.reporting.revenue import (
    CHANNEL_DIFFERENCES,
    TOTAL_DIFFERENCE,
    CellDifference,
    channel_name,
)

Array = np.ndarray

# One run's revenue channels: for each comparison, one total per result column
# per year (`scenario/prepare.py` calls this `channel_totals`).
ChannelTotals = Mapping[CellDifference, Mapping[str, Array]]

# The row label of the block that adds the provisions up. For an independent
# package that sum is not a package estimate, because it counts interactions
# more than once.
TOTAL_ROW = "Total"

# The channels reported for every provision, in order: the combined effect of
# the whole reform first, then the four channels that add up to it.
REPORTED_CHANNELS: tuple[CellDifference, ...] = (TOTAL_DIFFERENCE, *CHANNEL_DIFFERENCES)


@dataclass(frozen=True)
class StackedTable:
    """One channel block per channel, one row per provision, columns by year.

    `header` is `channel`, `provision`, then one column per year. Each
    row's values are that provision's own contribution to that channel, in the
    units the result column carries (dollars, for a liability column).
    """

    header: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]
    years: tuple[int, ...]
    column: str

    def write_csv(self, path: str | Path) -> None:
        with open(path, "w") as out:
            out.write(",".join(str(name) for name in self.header) + "\n")
            out.writelines(
                ",".join(_csv_cell(value) for value in row) + "\n"
                for row in self.rows
            )


def _csv_cell(value: object) -> str:
    # A numpy scalar's repr says its type ("np.float64(1.5)"); going through
    # the plain Python float first keeps the cell a bare number at full
    # precision.
    if isinstance(value, float):
        return repr(float(value))
    return str(value)


def provision_increments(
    channel_totals_by_scenario: Mapping[str, ChannelTotals],
    increments: Sequence[ProvisionIncrement],
    column: str,
    years: Sequence[int],
) -> dict[str, dict[CellDifference, Array]]:
    """Each provision's own annual contribution to each channel.

    `channel_totals_by_scenario` holds one run's channel totals per scenario
    ID, as `scenario/prepare.py`'s `ProjectionRun.channel_totals` gives them.
    `increments` says which scenario each provision is measured against, which
    is what `config/law_stacks.expand_stack` worked out from the stack file: the
    rung below it in a stacked package, or the run's own baseline for the first
    provision and for every provision of an independent package.

    Where a provision is measured against the run's own baseline there is
    nothing to subtract: each run already reports its comparison against
    baseline law, so that channel is the provision's whole contribution.
    """
    n_years = len(years)
    result: dict[str, dict[CellDifference, Array]] = {}

    for increment in increments:
        minuend = _channels_for(
            channel_totals_by_scenario, increment.minuend_id, column, n_years
        )
        if increment.subtrahend_id is None:
            result[increment.provision] = minuend
            continue
        subtrahend = _channels_for(
            channel_totals_by_scenario, increment.subtrahend_id, column, n_years
        )
        result[increment.provision] = {
            channel: minuend[channel] - subtrahend[channel]
            for channel in REPORTED_CHANNELS
        }

    return result


def _channels_for(
    channel_totals_by_scenario: Mapping[str, ChannelTotals],
    scenario_id: str,
    column: str,
    n_years: int,
) -> dict[CellDifference, Array]:
    """One scenario's reported channels for one result column, as year series."""
    try:
        totals = channel_totals_by_scenario[scenario_id]
    except KeyError:
        raise KeyError(
            f"no run was supplied for scenario '{scenario_id}'; the stack's "
            f"scenarios are {', '.join(sorted(channel_totals_by_scenario))}"
        ) from None

    series: dict[CellDifference, Array] = {}
    for channel in REPORTED_CHANNELS:
        if channel not in totals:
            raise KeyError(
                f"scenario '{scenario_id}' did not report the "
                f"{channel_name(channel)} channel"
            )
        if column not in totals[channel]:
            raise KeyError(
                f"scenario '{scenario_id}' has no result column '{column}' in "
                f"the {channel_name(channel)} channel"
            )
        values = np.asarray(totals[channel][column], dtype=np.float64)
        if values.shape != (n_years,):
            raise ValueError(
                f"scenario '{scenario_id}' reports {values.shape[0]} years for "
                f"the {channel_name(channel)} channel, but the table covers "
                f"{n_years}"
            )
        series[channel] = values
    return series


def stacked_table(
    channel_totals_by_scenario: Mapping[str, ChannelTotals],
    increments: Sequence[ProvisionIncrement],
    years: Sequence[int],
    column: str,
) -> StackedTable:
    """Build the table: one block per channel, one row per provision, by year.

    `column` is the result column being reported, a total liability column for
    a revenue table.

    The Total row closing each channel's block adds that channel's provisions
    up, and equals the last rung's own effect in that channel.
    """
    years = tuple(int(year) for year in years)
    per_provision = provision_increments(
        channel_totals_by_scenario, increments, column, years
    )

    rows: list[tuple[object, ...]] = []
    for channel in REPORTED_CHANNELS:
        label = channel_name(channel)
        total = np.zeros(len(years), dtype=np.float64)
        for increment in increments:
            values = per_provision[increment.provision][channel]
            rows.append((label, increment.provision, *values))
            total = total + values
        rows.append((label, TOTAL_ROW, *total))

    return StackedTable(
        header=("channel", "provision", *(str(year) for year in years)),
        rows=tuple(rows),
        years=years,
        column=column,
    )
