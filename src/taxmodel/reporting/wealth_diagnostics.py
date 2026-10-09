"""Diagnostic tables for the wealth shortfall.

The wealth channel assumes a household finances part of any extra tax it pays
out of its wealth rather than out of consumption. That drawn-down wealth is
the *shortfall*. It is accounted for by cohort cell (one age paired with one
net-worth percentile within that age) and each year the cell's accumulated
shortfall moves across percentiles, ages a year, grows, and takes in that
year's fresh saving out of the tax change. A separate step takes each
household's share of its cell's shortfall out of its own balance sheet, as a
fraction of gross assets held within a cap.

The arithmetic is in ``taxmodel.years.wealth_arithmetic``. This module reads
the state that arithmetic produced and arranges it so a reviewer can see how
much shortfall there is, where in the age grid and the wealth distribution it
sits, how fast it is growing, and how much of it reached a household's balance
sheet.

Nothing here re-runs the shortfall recurrence or writes a file.

Inputs from the runner
----------------------
``WealthShortfallHistory`` below is the contract. Every field is a plain
NumPy or JAX array; the runner stacks one entry per simulated year, in year
order, and passes them in. The shapes are:

    n_years  the simulated years, in order
    n_ages   rows of the cohort age grid (18 through 80, so 63 rows)
    n_bins   net-worth percentile bins within an age
    n_slots  household slots, the same stable slots the population uses

Per-cell state, each [n_years, n_ages, n_bins], all recorded *after* the
year's recurrence step has run:

    cell_shortfall     the accumulated shortfall standing in the cell
    cell_inflow        that year's fresh saving out of the tax change, added
                       at face value on top of the carried shortfall
    growth_factor      G, the factor the carried shortfall grew by
    capital_income_per_dollar_of_assets    the cell's capital income as a
                       fraction of its gross assets
    cell_net_worth     the cell's total weighted positive net worth, which is
                       what the shortfall is spread across

Per-household state, each [n_years, n_slots]:

    allocated_dollars   the household's share of its cell's shortfall, in
                        dollars, before weighting
    haircut_fraction    that allocation as a fraction of gross assets, held
                        within plus or minus the cap
    weight              the household's population weight that year, zero in a
                        year the slot holds no household

And two scalars: ``years`` (the year numbers, [n_years]) and ``haircut_cap``
(the largest fraction of gross assets one year's haircut may move).

Averages over cells are weighted by the absolute size of each cell's
shortfall.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class WealthShortfallHistory:
    """The wealth shortfall's carried state for every simulated year.

    See the module docstring for what each field holds and what shape it has.
    ``check_shapes`` states the contract in code.
    """

    years: Array
    ages: Array
    cell_shortfall: Array
    cell_inflow: Array
    growth_factor: Array
    capital_income_per_dollar_of_assets: Array
    cell_net_worth: Array
    allocated_dollars: Array
    haircut_fraction: Array
    weight: Array
    haircut_cap: float

    def check_shapes(self) -> None:
        """Refuse a history whose pieces do not describe the same run."""
        years = np.asarray(self.years)
        if years.ndim != 1:
            raise ValueError("years must be a one-dimensional list of year numbers")
        n_years = len(years)
        ages = np.asarray(self.ages)
        if ages.ndim != 1:
            raise ValueError("ages must be the one-dimensional cohort age grid")

        cell_shape = np.asarray(self.cell_shortfall).shape
        if (
            len(cell_shape) != 3
            or cell_shape[0] != n_years
            or cell_shape[1] != len(ages)
        ):
            raise ValueError(
                "cell_shortfall must be [year, age, percentile bin]; expected "
                f"{n_years} years and {len(ages)} ages, got {cell_shape}"
            )
        for name in (
            "cell_inflow",
            "growth_factor",
            "capital_income_per_dollar_of_assets",
            "cell_net_worth",
        ):
            shape = np.asarray(getattr(self, name)).shape
            if shape != cell_shape:
                raise ValueError(
                    f"{name} is {shape} but the cell arrays are {cell_shape}"
                )

        household_shape = np.asarray(self.allocated_dollars).shape
        if len(household_shape) != 2 or household_shape[0] != n_years:
            raise ValueError(
                "allocated_dollars must be [year, household slot]; expected "
                f"{n_years} years, got {household_shape}"
            )
        for name in ("haircut_fraction", "weight"):
            shape = np.asarray(getattr(self, name)).shape
            if shape != household_shape:
                raise ValueError(
                    f"{name} is {shape} but the household arrays are {household_shape}"
                )


def _as_float(values: Array) -> Array:
    return np.asarray(values, dtype=np.float64)


def _ratio_or_nan(numerator: Array, denominator: Array, live: Array) -> Array:
    """The ratio where ``live`` says there is something to divide by, else NaN.

    The inner substitution of one for the denominator keeps NumPy from
    warning about a division by zero in the entries that are thrown away.
    """
    return np.where(live, numerator / np.where(live, denominator, 1.0), np.nan)


def _smallest_and_largest(
    values: Array, counted: Array, axis: tuple[int, ...] | int
) -> tuple[Array, Array]:
    """The range of ``values`` over the entries ``counted`` marks, per year.

    Entries not counted are pushed to plus or minus infinity so they lose the
    minimum and the maximum. A year with nothing counted reports not-a-number,
    because there is no range to report.
    """
    any_counted = counted.any(axis=axis)
    smallest = np.where(
        any_counted, np.min(np.where(counted, values, np.inf), axis=axis), np.nan
    )
    largest = np.where(
        any_counted, np.max(np.where(counted, values, -np.inf), axis=axis), np.nan
    )
    return smallest, largest


def _weighted_average_over_cells(values: Array, cell_weights: Array) -> Array:
    """Average one per-cell quantity across cells, one answer per year.

    The weight is the absolute size of the cell's shortfall, so that a cell
    holding a negative shortfall (the household saved a tax cut into its
    wealth) counts the same as a positive one. A year in which no cell holds
    anything reports not-a-number.
    """
    values = _as_float(values)
    cell_weights = np.abs(_as_float(cell_weights))
    total = cell_weights.sum(axis=(1, 2))
    weighted = (values * cell_weights).sum(axis=(1, 2))
    return _ratio_or_nan(weighted, total, total > 0.0)


def shortfall_totals_by_year(history: WealthShortfallHistory) -> dict[str, Array]:
    """The total shortfall each year, and how much of it reached households.

    Every entry is an array with one number per simulated year, in year order.

        total_shortfall_dollars
            the accumulated shortfall standing in every cell at the end of
            the year
        inflow_dollars
            that year's fresh saving out of the tax change
        carried_in_dollars
            the rest of the year's shortfall: what last year's shortfall
            became after it moved, aged and grew
        implied_growth_of_the_carried_shortfall
            the carried part divided by last year's total. Not-a-number in
            the first year, which has nothing to carry in. It should sit near
            one plus the assumed return, a little below it wherever tax on the
            forgone capital income damps the growth.
        cell_net_worth_dollars
            the weighted positive net worth the shortfall is spread across
        shortfall_as_share_of_net_worth
            the shortfall relative to the wealth it is drawn from
        allocated_to_households_dollars
            the same shortfall after it is split across the households in
            each cell, weighted up to the population
        share_of_shortfall_reaching_households
            the two previous lines as a ratio. Below one means some cell held
            a shortfall but no household with positive net worth to take it
            out of, so those dollars went nowhere.
    """
    history.check_shapes()
    shortfall = _as_float(history.cell_shortfall).sum(axis=(1, 2))
    inflow = _as_float(history.cell_inflow).sum(axis=(1, 2))
    net_worth = _as_float(history.cell_net_worth).sum(axis=(1, 2))

    carried_in = shortfall - inflow
    previous_total = np.concatenate(([np.nan], shortfall[:-1]))
    implied_growth = _ratio_or_nan(
        carried_in, previous_total, np.abs(previous_total) > 0.0
    )

    allocated = (
        _as_float(history.allocated_dollars) * _as_float(history.weight)
    ).sum(axis=1)

    return {
        "year": np.asarray(history.years),
        "total_shortfall_dollars": shortfall,
        "inflow_dollars": inflow,
        "carried_in_dollars": carried_in,
        "implied_growth_of_the_carried_shortfall": implied_growth,
        "cell_net_worth_dollars": net_worth,
        "shortfall_as_share_of_net_worth": _ratio_or_nan(
            shortfall, net_worth, net_worth > 0.0
        ),
        "allocated_to_households_dollars": allocated,
        "share_of_shortfall_reaching_households": _ratio_or_nan(
            allocated, shortfall, np.abs(shortfall) > 0.0
        ),
    }


def cell_rates_by_year(history: WealthShortfallHistory) -> dict[str, Array]:
    """The shortfall's growth factor and the cells' capital income, by year.

    Averages are taken over cells, weighted by how much shortfall the cell
    holds.

        average_growth_factor
            G. One plus the assumed nominal return.
        average_capital_income_per_dollar_of_assets
            the cell's capital income as a fraction of its gross assets.

    The smallest and largest growth factor over the cells that hold any
    shortfall are reported alongside, to show a cell pinned at the growth
    floor.
    """
    history.check_shapes()
    shortfall = _as_float(history.cell_shortfall)
    growth = _as_float(history.growth_factor)
    holds_shortfall = np.abs(shortfall) > 0.0
    smallest, largest = _smallest_and_largest(growth, holds_shortfall, (1, 2))

    return {
        "year": np.asarray(history.years),
        "average_growth_factor": _weighted_average_over_cells(growth, shortfall),
        "smallest_growth_factor_in_a_filled_cell": smallest,
        "largest_growth_factor_in_a_filled_cell": largest,
        "average_capital_income_per_dollar_of_assets": _weighted_average_over_cells(
            history.capital_income_per_dollar_of_assets, shortfall
        ),
        "filled_cells": holds_shortfall.sum(axis=(1, 2)).astype(np.float64),
    }


def shortfall_by_age_by_year(history: WealthShortfallHistory) -> dict[str, Array]:
    """The shortfall by age, one row per year.

    ``ages`` names the columns of ``shortfall_dollars``, which is
    [year, age]: the cell shortfall added across the net-worth percentile
    bins.
    """
    history.check_shapes()
    return {
        "year": np.asarray(history.years),
        "ages": np.asarray(history.ages),
        "shortfall_dollars": _as_float(history.cell_shortfall).sum(axis=2),
    }


def shortfall_by_percentile_bin_by_year(
    history: WealthShortfallHistory,
) -> dict[str, Array]:
    """The shortfall by net-worth percentile bin, one row per year.

    ``shortfall_dollars`` is [year, percentile bin]: the cell shortfall added
    across ages. Bin zero is the bottom of the within-age net-worth
    distribution. A wealth tax should put nearly all of the shortfall in the
    top bins.
    """
    history.check_shapes()
    return {
        "year": np.asarray(history.years),
        "shortfall_dollars": _as_float(history.cell_shortfall).sum(axis=1),
    }


def haircut_summary_by_year(history: WealthShortfallHistory) -> dict[str, Array]:
    """The haircut applied to households, one row per year.

    The haircut is the fraction of a household's gross assets the shortfall
    takes out. It is held within plus or minus a cap, and a household at the
    cap is one whose balance sheet cannot absorb its full share.

        share_of_households_at_the_cap
            weighted share whose haircut is at the cap in absolute value
        average_haircut_fraction
            weighted mean across every household with a nonzero haircut
        largest_haircut_fraction, smallest_haircut_fraction
            the extremes across households with any weight
    """
    history.check_shapes()
    fraction = _as_float(history.haircut_fraction)
    weight = _as_float(history.weight)
    total_weight = weight.sum(axis=1)

    at_the_cap = np.abs(fraction) >= history.haircut_cap
    moved = fraction != 0.0
    moved_weight = (weight * moved).sum(axis=1)

    in_the_sample = weight > 0.0
    smallest, largest = _smallest_and_largest(fraction, in_the_sample, 1)

    return {
        "year": np.asarray(history.years),
        "share_of_households_at_the_cap": _ratio_or_nan(
            (weight * at_the_cap).sum(axis=1), total_weight, total_weight > 0.0
        ),
        "share_of_households_with_any_haircut": _ratio_or_nan(
            moved_weight, total_weight, total_weight > 0.0
        ),
        "average_haircut_fraction": _ratio_or_nan(
            (weight * fraction).sum(axis=1), moved_weight, moved_weight > 0.0
        ),
        "largest_haircut_fraction": largest,
        "smallest_haircut_fraction": smallest,
    }


def wealth_diagnostics(history: WealthShortfallHistory) -> dict[str, dict[str, Array]]:
    """Every wealth-channel diagnostic table, keyed by what it shows."""
    return {
        "shortfall totals": shortfall_totals_by_year(history),
        "cell rates": cell_rates_by_year(history),
        "shortfall by age": shortfall_by_age_by_year(history),
        "shortfall by percentile bin": shortfall_by_percentile_bin_by_year(history),
        "haircut": haircut_summary_by_year(history),
    }
