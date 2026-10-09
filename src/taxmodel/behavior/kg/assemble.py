"""Inputs to the capital-gains solve, and their layout.

The gains model runs on age cells, not households: the pre-pass adds the
households up into cells (``prepass``), the mix of death regimes averages the
year's treatment of gains at death over each cell's asset classes (``regime``),
the schedule solve works out how much of its gain each cell realizes and what a
deferred dollar is worth (``schedule``), and the stock recurrence steps the
stock of unrealized gain year by year (``stepper``).  A run solves every
scenario of a group through those steps in one compiled program
(``many_scenarios.py``).  This module supplies the pieces that program needs
from outside those steps:

- the two numbers read from the Macro-Projections publication, the nominal
  discount factor for each year and the price deflator to the charitable
  logit's base year;
- the two operators that carry the stock of gain from one year to the next,
  built on the host;
- the pre-pass's answers laid out the way the schedule solver reads them, age
  by year on the longer age grid the selling choice is solved on;
- the choice between the frozen solve and the behavioral one.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from taxmodel.law.indexes import read_macro_spliced

from . import prepass, regime, stepper
from .inputs import Calibration, ScheduleInputs, extend_to_choice_grid
from .schedule import ScheduleBundle, solve_frozen_schedule, solve_schedule
from .stock_inputs import aging_operator, heir_operator

Array = jnp.ndarray

# The Macro-Projections columns the gains model reads: the price level, which
# turns a future estate into today's dollars for the charitable bequest logit,
# and the ten-year Treasury rate, which is the interest rate a taxpayer
# discounts a future tax payment at.
PRICE_LEVEL_COLUMN = "cpiu"
TREASURY_TEN_YEAR_COLUMN = "tsy_10y"

# The published Treasury rate is in percentage points; the arithmetic wants a
# plain rate.
PERCENTAGE_POINTS_PER_UNIT = 100.0


def nominal_discount_factor_by_year(
    macro_root: str | Path, years: Sequence[int]
) -> np.ndarray:
    """The value this year of a dollar next year, for each year of the run.

    The rate is the nominal ten-year Treasury rate, because the quantities it
    discounts are nominal: assets appreciate at a nominal rate and a dollar of
    basis is a dollar of historical cost, not an indexed one.

        discount factor = 1 / (1 + Treasury rate)
    """

    macro = read_macro_spliced(macro_root).sort_values("year")
    if TREASURY_TEN_YEAR_COLUMN not in macro.columns:
        raise ValueError(
            f"{macro_root} does not publish {TREASURY_TEN_YEAR_COLUMN}, which "
            "the capital-gains model needs to discount a tax payment a "
            "taxpayer puts off"
        )

    treasury = macro[TREASURY_TEN_YEAR_COLUMN] / PERCENTAGE_POINTS_PER_UNIT
    factor = (1.0 / (1.0 + treasury)).to_numpy()

    by_year = dict(zip(macro["year"].to_numpy(), factor, strict=True))
    missing = [
        year for year in years if year not in by_year or not np.isfinite(by_year[year])
    ]
    if missing:
        raise ValueError(
            f"{macro_root} gives no ten-year Treasury rate for "
            + ", ".join(str(year) for year in missing)
        )
    return np.asarray([by_year[year] for year in years], dtype=np.float64)


def price_deflator_to_base_year(
    macro_root: str | Path, years: Sequence[int], base_year: int
) -> np.ndarray:
    """The base year's price level over each run year's.

    The charitable bequest logit was fit on estates measured in the dollars of
    one particular year, so an estate in any other year has to be brought back
    to it before the coefficients mean anything.
    """

    macro = read_macro_spliced(macro_root)
    level = dict(
        zip(macro["year"].to_numpy(), macro[PRICE_LEVEL_COLUMN].to_numpy(), strict=True)
    )
    wanted = [*years, base_year]
    missing = [year for year in wanted if year not in level]
    if missing:
        raise ValueError(
            f"{macro_root} gives no price level for "
            + ", ".join(str(year) for year in missing)
        )
    return np.asarray([level[base_year] / level[year] for year in years], dtype=np.float64)


def mortality_above_the_top_code_for_tax_units(
    probability_of_death: Array, joint_share_of_gain: Array
) -> Array:
    """The chance a dollar of gain loses its holder, for ages above the cells.

    Below age 81 the selling choice reads the cell's gain-weighted mortality,
    which is a rate per tax unit: on a joint return the gain passes on when
    both spouses have died, so the household's chance is the product of the two
    spouses' chances, and on any other return it is the filer's own.  Above age
    80 there is no cell, and the life table gives one person's chance of dying.
    Putting the same convention on the life table:

        (1 - w) * q + w * q * q

    where ``q`` is the life table's probability of death and ``w`` is the share
    of the top cell's dollars of gain that sit on joint returns
    (:func:`~.prepass.joint_share_of_top_cell_gain`).  Reading the raw life
    table instead takes the series from about 1.7 percent at age 80 to about 5
    percent at age 81, most of which is the filing convention rather than
    ageing.

    ``probability_of_death`` is [age 81 through 119, year], and
    ``joint_share_of_gain`` is one number per year.
    """

    q = jnp.asarray(probability_of_death, dtype=jnp.float64)
    w = jnp.asarray(joint_share_of_gain, dtype=jnp.float64)[None, :]
    return (1.0 - w) * q + w * q * q


def stock_operators(heir_age_shares: Array, n_cells: int) -> tuple[Array, Array]:
    """The two operators that carry the stock of gain from one year to the next.

    The first moves each cell's stock to the next age, and the second routes a
    decedent's gains to the ages of their heirs.

    The batched solve builds them here on the host, once per run: the heir
    operator checks that its distribution sums to something positive, which
    needs a real number and so cannot happen inside a compiled program.
    """
    return (aging_operator(n_cells), heir_operator(heir_age_shares))


def schedule_inputs_for_run(
    cells: prepass.PrepassResult,
    death_shares: regime.DeathShares,
    *,
    discount_factor_by_year: Sequence[float],
    mortality_above_top_code: Array,
    operators: tuple[Array, Array],
) -> ScheduleInputs:
    """Lay the pre-pass's answers out the way the schedule solver reads them.

    The pre-pass works year by year and hands back arrays with the year first,
    because that is the order a run's frames arrive in.  The solver works age
    by age within a year and takes the age first.  This turns the one into the
    other, stretches everything the selling choice reads onto the longer age
    grid, and builds the two operators (one moving each age's stock to the next
    age, the other routing a decedent's gains to the ages of their heirs).

    Everything past the top-coded age of 80 repeats age 80's value, except
    mortality, which comes from the Social Security life tables because the
    model has no cell above 80.  Turnover is held flat there (see
    ``inputs.extend_to_choice_grid``).

    ``mortality_above_top_code`` is shaped [age 81 through 119, year], already
    on the tax-unit convention the cells use
    (:func:`mortality_above_the_top_code_for_tax_units`).

    ``operators`` is the aging operator and the heir operator
    (:func:`stock_operators`), which the batched solve builds once per run.
    """

    baseline = cells.cells
    # The pre-pass's cells are laid out year by age and everything below is age
    # by year, so each array is transposed here.
    stock = baseline.stock.T
    realizations = baseline.realizations.T
    turnover = baseline.turnover.T
    charity = jnp.clip(baseline.charitable_bequest_share.T, 0.0, 1.0)

    # The chance that the holder of a dollar of unrealized gain dies during the
    # year.  It differs from the chance that a taxpayer in the cell dies,
    # because mortality and holdings vary together within an age cell.
    mortality = stepper.effective_mortality(baseline).T
    aging, heirs = operators

    return ScheduleInputs(
        baseline_gain_stock=stock,
        baseline_realizations=realizations,
        baseline_turnover=turnover,
        charity_probability=charity,
        baseline_asset_value=cells.tracked_asset_value.T,
        death_share_forgiven=death_shares.forgiven,
        death_share_routed_to_heirs=death_shares.routed_to_heirs,
        death_share_taxed_at_death=death_shares.taxed_at_death,
        gain_weighted_mortality=extend_to_choice_grid(
            mortality, jnp.asarray(mortality_above_top_code, dtype=jnp.float64)
        ),
        marginal_rate_baseline=extend_to_choice_grid(cells.marginal_rate_baseline),
        marginal_rate_reform=extend_to_choice_grid(cells.marginal_rate_reform),
        marginal_rate_reform_law_only=extend_to_choice_grid(
            cells.marginal_rate_reform_law_only
        ),
        aging_operator=aging,
        heir_operator=heirs,
        discount_factor_by_year=jnp.asarray(discount_factor_by_year, dtype=jnp.float64),
    )


def solve_schedules(
    inputs: ScheduleInputs, calibration: Calibration, frozen: bool
) -> ScheduleBundle:
    """Solve the realization schedules, or pin them to baseline law.

    The frozen solve runs before the static world.  It puts the pure law
    arithmetic of changing how gains at death are treated (carryover routing
    gains to heirs, realized at baseline rates) into the static world, so that
    the behavioral estimate is the conventional estimate less the static one.
    It is the same recurrence, with turnover pinned to the baseline pace.
    """

    return solve_frozen_schedule(inputs) if frozen else solve_schedule(inputs, calibration)
