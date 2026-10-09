"""Inputs and constants for the capital-gains schedule solver.

The schedule solver runs before any year of the stock of unrealized gains is
stepped forward.  Its inputs are age-cell aggregates and matrices of rates by
age and year; its output is the schedule bundle in ``schedule.py``.

Two age grids
-------------
The stock of unrealized gains is tracked over ages 18 to 80, with 80 a pooled
top-coded cell.  The selling choice is solved over a longer grid, 18 to
119, because the Social Security life tables run out at 119 and the choice
needs a last age to work backward from; the model treats 119 as the end of
life (``life_table.py``).  Past age 80 the tax rates,
baseline turnover, the charity probability and the share of gains taxed at
death all repeat their age-80 values, and mortality comes from the life
table.

So the arrays below come in two shapes.  Anything the selling choice reads
is on the long grid, ``[n_extended_ages, n_years]``.  Anything about the stock
of gains itself, and everything the solver hands downstream, is on the short
grid, ``[n_stock_ages, n_years]``.  The long grid's first rows are the short
grid's ages, in the same order, so the solver takes the short grid's rows
positionally.

The caller assembles these arrays from Tax-Data cell aggregates, the compiled
law, and the marginal rates the pricing pass measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .life_table import OLDEST_AGE
from .stock_inputs import KG_AGE_MIN, KG_N_AGES

Array = jnp.ndarray


# -----------------------------------------------------------------------------
# The two age grids
# -----------------------------------------------------------------------------

# The stock's own age grid, 18 to 80, is declared in stock_inputs.py.

# The selling choice is solved out to the last age the SSA life tables carry,
# which life_table.py declares.
KG_AGE_MAX_CHOICE_GRID = OLDEST_AGE

N_STOCK_AGES = KG_N_AGES                            # 63
N_CHOICE_GRID_AGES = KG_AGE_MAX_CHOICE_GRID - KG_AGE_MIN + 1  # 102

# Points on the gain-share grid the selling choice is solved on, evenly spaced
# from a portfolio that is all basis to one that is all gain.  The solved
# policy is handed to the stock ledger on this grid, which reads it at the gain
# share its cells actually carry.
GAIN_SHARE_GRID_POINTS = 101

# Two realizations in the same window are treated as equally good when their
# rate changes are within this of each other.
TIMING_TIE_TOLERANCE = 1e-12

# A floor under denominators that can legitimately come out at zero: a cell
# that holds no assets, a cell that sells nothing.  Dividing by this instead of
# by zero keeps an infinity out of the solve, and it is small enough never to
# bind on a denominator that is genuinely positive.
SMALLEST_SAFE_DENOMINATOR = 1e-12

# How far the three shares of a decedent's gain may add up from one before the
# run stops.
DEATH_SHARE_SUM_TOLERANCE = 1e-6

# The largest change the last sweep of the terminal year's stationary solve may
# still be making.  Past this the terminal value has not settled and the run
# stops (``schedule.check_schedule``).
TERMINAL_RESIDUAL_TOLERANCE = 1e-7


# -----------------------------------------------------------------------------
# The inputs
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ScheduleInputs:
    """Everything the schedule solver reads, as arrays of age by year.

    Short-grid fields are shaped ``[n_stock_ages, n_years]``; long-grid
    fields are shaped ``[n_extended_ages, n_years]``.  Which is which is said
    for each field below.

    Baseline cell aggregates (short grid), built from Tax-Data:

    baseline_gain_stock
        The stock of unrealized gains held by the cell, in dollars.
    baseline_realizations
        Long-term capital gains the cell realizes in the year under baseline
        law, in dollars.
    baseline_turnover
        Realizations over the stock of gains, the quantity the data measures.
        Selling is pro-rata, so it is also the share of the cell's asset value
        sold in a year under baseline law.
    charity_probability
        Share of the decedent's gain that goes to charity, where it is never
        taxed under any treatment of gains at death.
    baseline_asset_value
        Market value of the assets standing behind the cell's stock of
        unrealized gains, summed over the five tracked classes.  The stock of
        gains over this value is the cell's gain share.  The selling choice
        prices the tax cost of raising a dollar of cash at that share.

    The treatment of gains at death (short grid).  These come from averaging
    each asset class's regime — step-up, carryover, or deemed realization —
    over the cell's gains by class.  The first three are shares of the
    decedent's gain and add up to one; the charity share above is a share of
    that same gain.

    death_share_forgiven
        Share that vanishes because the heir takes a stepped-up basis.
    death_share_routed_to_heirs
        Share that keeps its basis and moves into the heirs' cells, to be
        realized there in later years.
    death_share_taxed_at_death
        Share deemed realized and taxed in the year of death.

    Rates and mortality on the long grid:

    gain_weighted_mortality
        Chance that the holder of a dollar of unrealized gain dies during the
        year.  Past age 80 this is the life table rate.
    marginal_rate_baseline
        Marginal tax rate on realized long-term gains under baseline law.
        Read by the baseline pass of the selling choice and by the retiming
        comparison.
    marginal_rate_reform
        Marginal rate under reform law, measured on the mechanical world's
        frame after the frozen state was injected.  Read by the reform pass of
        the selling choice.
    marginal_rate_reform_law_only
        Marginal rate under reform law measured on the frame *before* the
        frozen state was injected, so it reflects the change in law alone.
        Only the retiming overlay reads it.

    Operators and the discount series:

    aging_operator
        Square, ``[n_stock_ages, n_stock_ages]``.  Moves each age's stock
        to the next age, with the top-coded age pointing at itself.  Used as
        ``A' x`` — the column for an age holds where that age's dollars go.
    heir_operator
        Square, ``[n_stock_ages, n_stock_ages]``.  Row is the decedent's
        age, column the heir's; every row is the same observed distribution of
        heir ages weighted by dollars inherited.
    discount_factor_by_year
        Nominal discount factor from each year to the next, one over one plus
        the ten-year Treasury rate, length ``n_years``.
    """

    baseline_gain_stock: Array
    baseline_realizations: Array
    baseline_turnover: Array
    charity_probability: Array
    baseline_asset_value: Array

    death_share_forgiven: Array
    death_share_routed_to_heirs: Array
    death_share_taxed_at_death: Array

    gain_weighted_mortality: Array
    marginal_rate_baseline: Array
    marginal_rate_reform: Array
    marginal_rate_reform_law_only: Array

    aging_operator: Array
    heir_operator: Array
    discount_factor_by_year: Array

    @property
    def n_stock_ages(self) -> int:
        """Rows on the short grid, one per age."""
        return int(self.baseline_gain_stock.shape[0])

    @property
    def n_extended_ages(self) -> int:
        """Rows on the long grid, one per age."""
        return int(self.gain_weighted_mortality.shape[0])

    @property
    def n_years(self) -> int:
        return int(self.baseline_gain_stock.shape[1])

    def validate(self) -> None:
        """Check the shapes agree, and the death inputs are shares.

        The shape checks are plain Python, so they run at trace time.  The
        checks on the death shares read numbers, which a JAX tracer does not
        carry, so they are skipped when the arrays are being traced.  The
        economics checks, which need the solved answer, are in
        ``schedule.check_schedule()``.
        """
        short = (self.n_stock_ages, self.n_years)
        long = (self.n_extended_ages, self.n_years)
        square = (self.n_stock_ages, self.n_stock_ages)

        for name in (
            "baseline_gain_stock",
            "baseline_realizations",
            "baseline_turnover",
            "charity_probability",
            "baseline_asset_value",
            "death_share_forgiven",
            "death_share_routed_to_heirs",
            "death_share_taxed_at_death",
        ):
            got = getattr(self, name).shape
            if got != short:
                raise ValueError(
                    f"{name} is shaped {got}; the stock of gains is tracked over "
                    f"{short[0]} ages and {short[1]} years, so it must be {short}."
                )

        for name in (
            "gain_weighted_mortality",
            "marginal_rate_baseline",
            "marginal_rate_reform",
            "marginal_rate_reform_law_only",
        ):
            got = getattr(self, name).shape
            if got != long:
                raise ValueError(
                    f"{name} is shaped {got}; the selling choice is solved over "
                    f"{long[0]} ages and {long[1]} years, so it must be {long}."
                )

        if self.n_extended_ages < self.n_stock_ages:
            raise ValueError(
                "The selling choice's age grid is shorter than the stock's "
                f"({self.n_extended_ages} against {self.n_stock_ages} ages)."
            )
        for name in ("aging_operator", "heir_operator"):
            got = getattr(self, name).shape
            if got != square:
                raise ValueError(f"{name} is shaped {got}; it must be {square}.")
        if self.discount_factor_by_year.shape != (self.n_years,):
            raise ValueError(
                "discount_factor_by_year is shaped "
                f"{self.discount_factor_by_year.shape}; it must be ({self.n_years},)."
            )

        self._check_death_shares()

    def _check_death_shares(self) -> None:
        """The charity share and the three death shares are shares.

        Each is a share of the decedent's gain, so each has to be a number
        between zero and one, and the three treatments of a dollar of gain at
        death — stepped up, carried to the heirs, taxed at death — have to
        account for the whole dollar wherever the cell holds gain. Both
        checks allow the rounding a share picks up when it is built from
        sums of household weights.
        """
        death_shares = (
            "death_share_forgiven",
            "death_share_routed_to_heirs",
            "death_share_taxed_at_death",
        )
        names = ("charity_probability", *death_shares)
        values = [jnp.asarray(getattr(self, name)) for name in names]
        if any(isinstance(value, jax.core.Tracer) for value in values):
            return

        for name, value in zip(names, values):
            if not bool(jnp.all(jnp.isfinite(value))):
                raise ValueError(f"{name} has non-finite values.")
            smallest = float(jnp.min(value))
            largest = float(jnp.max(value))
            if smallest < -DEATH_SHARE_SUM_TOLERANCE or largest > 1.0 + DEATH_SHARE_SUM_TOLERANCE:
                raise ValueError(
                    f"{name} runs from {smallest} to {largest}; it is a share "
                    "of the decedent's gain, so every value must be between "
                    "zero and one."
                )

        # A cell holding no gain has nothing to classify and carries zero
        # shares; everywhere else the three shares account for the whole
        # dollar.
        holds_gain = jnp.asarray(self.baseline_gain_stock) > 0.0
        total = sum(values[1:])
        worst = float(jnp.max(jnp.where(holds_gain, jnp.abs(total - 1.0), jnp.abs(total))))
        if worst > DEATH_SHARE_SUM_TOLERANCE:
            raise ValueError(
                "death_share_forgiven, death_share_routed_to_heirs and "
                f"death_share_taxed_at_death miss adding up to one by {worst} "
                "in some cell and year that holds gain. Every dollar of a "
                "decedent's gain is either stepped up, carried to the heirs, "
                "or taxed at death, so the three shares must add up to one."
            )


@dataclass(frozen=True)
class Calibration:
    """The calibrated numbers and settings the schedule solver reads.

    The three numeric fields are ordinary JAX values, so the whole solve can be
    differentiated with respect to them, as fitting them requires.

    elasticity
        The curvature of the nontax benefit of selling in the household's
        problem: with a larger value the household moves further from its
        baseline pace for the same change in the tax cost of selling.
    timeable_share
        Share of a cell-year's baseline realizations that can move a year
        either way when a rate change is announced.
    appreciation
        The yearly rate at which assets appreciate, nominal and the same for
        every age and asset class.  It is a structural setting, read from
        ``config/calibrations/kg/settings.yaml``.
    timing_reference_wedge
        The rate change at which all of the timeable realizations move.  At the
        default 0.05, five points of rate saving moves all of them and one
        point moves a fifth.
    timing_window
        How many years either side of a realization the retiming overlay
        considers.  The default configuration uses one.
    """

    elasticity: Array
    timeable_share: Array
    appreciation: float
    timing_reference_wedge: Array = 0.05
    timing_window: int = 1

    def validate(self) -> None:
        """Check the retiming window.

        The numeric fields may be tracers under ``jax.grad``, so only the
        static retiming window is checked here; the numbers are checked after
        the solve, in ``schedule.check_schedule()``.
        """
        if not isinstance(self.timing_window, int) or self.timing_window < 0:
            raise ValueError(
                "The retiming window must be a whole number of years, zero or "
                f"more; got {self.timing_window!r}."
            )


jax.tree_util.register_dataclass(
    ScheduleInputs,
    data_fields=[f.name for f in ScheduleInputs.__dataclass_fields__.values()],
    meta_fields=[],
)

jax.tree_util.register_dataclass(
    Calibration,
    data_fields=["elasticity", "timeable_share", "timing_reference_wedge"],
    meta_fields=["appreciation", "timing_window"],
)


# -----------------------------------------------------------------------------
# Building the long age grid from the short one
# -----------------------------------------------------------------------------


def extend_to_choice_grid(
    stock_values: Array,
    tail_values: Array | None = None,
    *,
    n_choice_ages: int = N_CHOICE_GRID_AGES,
) -> Array:
    """Stretch an age-by-year matrix from the stock's grid to the choice's grid.

    Every row past the top-coded age repeats the top-coded age's value, unless
    the caller supplies its own rows for them.  Mortality is the one input that
    supplies its own: past age 80 it comes from the life table rather than from
    a pooled cell.

    Turnover is held flat past the top code: every age above 80 sells at the
    age-80 cell's observed pace.

    Parameters
    ----------
    stock_values
        ``[n_stock_ages, n_years]``.
    tail_values
        ``[n_tail_ages, n_years]``, or None to repeat the top-coded row.
    n_choice_ages
        Rows on the choice's grid when the tail is repeated.  The default grid
        runs from 18 to 119; the tests' small grids pass their own length.
    """
    stock_values = jnp.asarray(stock_values)
    if tail_values is not None:
        return jnp.concatenate([stock_values, jnp.asarray(tail_values)], axis=0)
    n_tail_ages = n_choice_ages - stock_values.shape[0]
    return jnp.concatenate(
        [stock_values, jnp.tile(stock_values[-1:, :], (n_tail_ages, 1))], axis=0
    )
