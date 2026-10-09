"""The capital-gains model: how a change in the tax on realized gains changes
when people sell, how much unrealized gain they still hold at death, and what
becomes of that gain.

The model works on age cells. It adds households up into cells, solves the
selling choice for every cell and year at once, steps the stock of unrealized
gain year by year, and spreads each year's cell table back over the
households.

- ``prepass``            adds households up within their age cell: the stock
                         of gain, realizations, mortality, the charitable
                         bequest share and the cell-level marginal tax rates
- ``regime``             averages the treatment of gains at death (step-up,
                         carryover, deemed realization) over each cell
- ``life_table``         Social Security mortality for ages above the top cell
- ``inputs``             what the schedule solver is given, and the two age
                         grids it works on
- ``bellman_age``        the selling choice by age and gain share
- ``timing``             the short-run retiming overlay
- ``schedule``           the solver's driver, the schedule bundle it returns,
                         and the checks a caller runs afterwards
- ``stock_inputs``       the baseline age cells, the conversion pool, and the
                         aging and heir-routing operators
- ``conversion_inflow``  compensation relabeled as capital gain, priced for
                         every year at once
- ``stepper``            the scan over years that carries the stock, and the
                         cell table it emits for each year
- ``assemble``           lays the cell aggregates out for the solver and runs
                         the solve for one scenario
- ``many_scenarios``     the same solve for a group of scenarios in one
                         compiled program, as a run does it
- ``applier``            spreads a year's cell table over the households in
                         those cells, and prices the tax on gains deemed
                         realized at death

``ScheduleBundle``, defined in ``schedule``, is what the solver returns and
the stock recurrence reads.
"""

from .inputs import Calibration, ScheduleInputs
from .schedule import (
    BASELINE_TURNOVER_FIT_TOLERANCE,
    ScheduleCheckFailure,
    check_schedule,
    solve_frozen_schedule,
    solve_schedule,
)
from .stepper import cell_table_column, run_stock_recurrence
from .stock_inputs import BaselineCells

# Only the names callers import from the package are listed here; everything
# else is reached through its own module (``kg.prepass``, ``kg.applier``).
__all__ = [
    "BASELINE_TURNOVER_FIT_TOLERANCE",
    "BaselineCells",
    "Calibration",
    "ScheduleCheckFailure",
    "ScheduleInputs",
    "cell_table_column",
    "check_schedule",
    "run_stock_recurrence",
    "solve_frozen_schedule",
    "solve_schedule",
]
