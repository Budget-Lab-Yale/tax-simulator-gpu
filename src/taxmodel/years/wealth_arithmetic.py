"""The wealth bathtub arithmetic: the reduction in wealth from extra tax paid
during life.

The channel assumes households finance a share ``s`` of any increase in tax
they pay during life out of their wealth rather than out of consumption.  That
shortfall compounds year after year and leaves a smaller estate at death and
less capital income along the way.

Accounting is done by cohort.  A cell is one age paired with one net-worth
percentile within that age.  Each year the cell's accumulated shortfall moves
across percentiles, ages a year, grows, and takes in this year's saving out of
the tax change.  A separate step then takes each record's share of its cell's
shortfall out of its balance sheet.

This module holds the arithmetic.  Reading detail files, forming the tax
change, and carrying state by year belong to the annual simulation step that
calls these functions.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

Array = jnp.ndarray


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# Cohort age grid, top-coded at 80 to match the capital gains bathtub and the
# Tax-Data age1 top code.
WEALTH_DYN_AGE_MIN = 18
WEALTH_DYN_AGE_MAX = 80

# Floor for the growth factor G, which cannot go negative.
WEALTH_DYN_G_FLOOR = 1e-6

# Numerical floor for denominators: a cell's total net worth, and a record's
# gross assets.
WEALTH_DYN_EPS = 1e-8

# Raking the transition matrix: tolerance on the row and column sums, and the
# maximum number of passes.
SINKHORN_TOL = 1e-10
SINKHORN_MAX_ITER = 1000

# A row or column that is entirely zero cannot be scaled to sum to one, so this
# is added to every cell of the matrix first.
SINKHORN_EMPTY_LINE_SEED = 1e-12


def age_grid(
    age_min: int = WEALTH_DYN_AGE_MIN, age_max: int = WEALTH_DYN_AGE_MAX
) -> np.ndarray:
    """Return the contiguous cohort age grid, youngest first."""
    return np.arange(age_min, age_max + 1, dtype=np.int64)


# -----------------------------------------------------------------------------
# Percentile cut points within each age
# -----------------------------------------------------------------------------


def eligible_for_cutoffs(
    value: Array,
    weight: Array,
    age_cohort: Array,
    ages: np.ndarray,
) -> Array:
    """Return which records help set their age's percentile cut points.

    A record counts if its age is on the grid, its weight is positive, and its
    ranking value is a finite number.  A record at or below zero net worth is
    dropped, since it has nothing to draw down.
    """
    on_grid = (age_cohort >= int(ages[0])) & (age_cohort <= int(ages[-1]))
    keep = on_grid & (weight > 0) & jnp.isfinite(value) & (value > 0)
    return keep


def within_age_cutoffs(
    value: Array,
    weight: Array,
    age_cohort: Array,
    ages: np.ndarray,
    n_bins: int,
) -> Array:
    """Return the boundaries that split each age into equal-headcount percentiles.

    The cut point for probability p is the ranking value at which the age's
    cumulative share of weight first reaches p.  A record sitting exactly on a
    boundary falls in the lower percentile.  An age with no eligible records
    gets cut points of positive infinity, which puts every one of its records in
    the bottom percentile.

    Boundaries are returned so that the step that accumulates the shortfall
    and the step that drains it assign the same percentiles.

    Arguments:
        value: the ranking value per record, normally raw net worth.
        weight: record weights.
        age_cohort: each record's cohort age.
        ages: the contiguous age grid.
        n_bins: number of percentiles, a configuration value (``n_pctiles``).

    Returns an array with one row per age and ``n_bins - 1`` interior cut points
    per row.
    """
    n_ages = len(ages)
    age_min = int(ages[0])
    probs = jnp.arange(1, n_bins, dtype=jnp.float64) / n_bins

    keep = eligible_for_cutoffs(value, weight, age_cohort, ages)

    # Every ineligible record is parked in a spare slot one past the last age,
    # so it never lands inside an age's stretch of the sorted order.
    slot = jnp.where(keep, age_cohort - age_min, n_ages)
    # A negative weight would break the running total below, and an ineligible
    # record has to contribute nothing either way.
    weight_used = jnp.where(keep, weight, 0.0)
    # Sorting ineligible records to the very end also keeps their values from
    # ever being read out as a cut point.
    value_used = jnp.where(keep, value, jnp.inf)

    # One sort, by age and then by value.  Records of the same age are then
    # contiguous, so an age's cumulative weight is the running total minus the
    # running total where that age starts.  (jnp.lexsort takes the primary key
    # last.)
    order = jnp.lexsort((value_used, slot))
    slot_sorted = slot[order]
    value_sorted = value_used[order]
    # A leading zero makes "the running total just before position j" a plain
    # lookup at j.
    running_weight = jnp.concatenate(
        [jnp.zeros(1, dtype=jnp.float64), jnp.cumsum(weight_used[order])]
    )

    age_slots = jnp.arange(n_ages)
    start = jnp.searchsorted(slot_sorted, age_slots, side="left")
    end = jnp.searchsorted(slot_sorted, age_slots, side="right")
    base = running_weight[start][:, None]
    total = (running_weight[end] - running_weight[start])[:, None]

    # The first record whose cumulative weight within its age reaches p of the
    # age's total weight.  The comparison is multiplied through by the age
    # total so that one sorted array can be searched.
    target = base + probs[None, :] * total
    found = jnp.searchsorted(running_weight[1:], target.ravel(), side="left")
    found = found.reshape(n_ages, n_bins - 1)
    found = jnp.clip(found, start[:, None], jnp.maximum(end - 1, 0)[:, None])

    cuts = value_sorted[found]
    return jnp.where(total > 0, cuts, jnp.inf)


def assign_within_age_bin(
    value: Array,
    age_cohort: Array,
    cutoffs: Array,
    ages: np.ndarray,
    n_bins: int,
) -> tuple[Array, Array]:
    """Assign each record to its percentile within its age.

    A record's percentile is the number of its age's cut points at or below its
    ranking value, so a record sitting exactly on a boundary falls in the lower
    percentile, matching the way the boundaries were found.

    Records at or below zero, records whose
    value is not a finite number, and records whose age is off the grid get no
    cell at all.  Unlike the cut points, this does not look at the weight: a
    zero-weight record still gets a percentile, and contributes no people and no
    dollars to it.

    Returns the zero-based percentile index, which is -1 where the record has no
    cell, together with the mask saying which records do have one.
    """
    n_ages = len(ages)
    age_min = int(ages[0])

    on_grid = (age_cohort >= age_min) & (age_cohort <= int(ages[-1]))
    has_cell = on_grid & jnp.isfinite(value) & (value > 0)

    row = jnp.clip(age_cohort - age_min, 0, n_ages - 1)
    bin_index = _count_cuts_at_or_below(cutoffs, row, value, n_bins)
    bin_index = jnp.minimum(bin_index, n_bins - 1)
    return jnp.where(has_cell, bin_index, -1), has_cell


def _count_cuts_at_or_below(
    cutoffs: Array, row: Array, value: Array, n_bins: int
) -> Array:
    """Count how many of a record's age's cut points sit at or below its value.

    Reading out the whole row of cut points for every record would mean an array
    of records by percentiles, which is hundreds of gigabytes at the model's own
    size.  Walking the row by halving instead reads one cut point per record per
    pass, and seven passes cover a hundred percentiles.
    """
    n_cuts = n_bins - 1
    low = jnp.zeros(jnp.shape(value), dtype=jnp.int64)
    high = jnp.full(jnp.shape(value), n_cuts, dtype=jnp.int64)
    for _ in range(max(1, math.ceil(math.log2(n_cuts + 1)))):
        middle = (low + high) // 2
        cut = cutoffs[row, jnp.clip(middle, 0, n_cuts - 1)]
        at_or_below = (middle < n_cuts) & (cut <= value)
        low = jnp.where(at_or_below, middle + 1, low)
        high = jnp.where(at_or_below, high, middle)
    return low


# -----------------------------------------------------------------------------
# Raking the percentile transition matrix
# -----------------------------------------------------------------------------


def _seed_empty_lines(matrix: Array) -> Array:
    """Add a tiny amount everywhere if any row or column is entirely zero."""
    empty = jnp.any(matrix.sum(axis=1) == 0) | jnp.any(matrix.sum(axis=0) == 0)
    return jnp.where(empty, matrix + SINKHORN_EMPTY_LINE_SEED, matrix)


def _rake_pass(matrix: Array) -> Array:
    """One pass: make every row sum to one, then every column."""
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    return matrix / matrix.sum(axis=0, keepdims=True)


def _rake_converged(matrix: Array) -> Array:
    """True once both the row sums and the column sums are within tolerance."""
    row_gap = jnp.max(jnp.abs(matrix.sum(axis=1) - 1.0))
    column_gap = jnp.max(jnp.abs(matrix.sum(axis=0) - 1.0))
    return (row_gap < SINKHORN_TOL) & (column_gap < SINKHORN_TOL)


def rake_to_doubly_stochastic(
    matrix: Array, max_iter: int = SINKHORN_MAX_ITER
) -> Array:
    """Scale a nonnegative square matrix until its rows and columns both sum to one.

    The recurrence works in percentile space, where every percentile holds the
    same weighted headcount, so the transition has to preserve that.  Rows
    summing to one conserves dollars; columns summing to one keeps the equal
    headcount.  A sensibly built transition matrix nearly satisfies both
    already, and this makes it exact.  It leaves the identity and a uniform
    matrix alone.

    The last step is a row normalization, so the rows sum to one exactly and
    the columns only to within tolerance.  Rows summing to one conserves
    dollars.
    """
    working = _seed_empty_lines(jnp.asarray(matrix, dtype=jnp.float64))

    def keep_going(state):
        raked, passes = state
        return (passes < max_iter) & jnp.logical_not(_rake_converged(raked))

    def step(state):
        raked, passes = state
        return _rake_pass(raked), passes + 1

    # Convergence is tested after a pass, so the first pass always runs.
    raked, _ = jax.lax.while_loop(keep_going, step, (_rake_pass(working), 1))
    return raked / raked.sum(axis=1, keepdims=True)


# -----------------------------------------------------------------------------
# Stepping the cells forward one year
# -----------------------------------------------------------------------------


def build_aging_matrix(n_ages: int) -> Array:
    """Build the operator that moves each age to the next.

    The top-coded age points at itself, so that pool accumulates rather than
    falling off the end of the grid.  It is applied as ``A.T @ X`` in
    ``cohort_recurrence_step``, so each age gathers from the age one year
    younger, and the top age also gathers itself.
    """
    matrix = np.zeros((n_ages, n_ages))
    for i in range(n_ages - 1):
        matrix[i, i + 1] = 1.0
    matrix[n_ages - 1, n_ages - 1] = 1.0
    return jnp.asarray(matrix)


def apply_percentile_transition(cells: Array, transition: Array) -> Array:
    """Move each age's cells across percentiles.

    ``cells`` has ages on rows and percentiles on columns, and the transition is
    right-multiplied, so entry (i, j) of the transition is the share of
    percentile i's shortfall that ends up in percentile j.  The identity means
    everyone stays where they are.  This is also the re-binning onto the new
    year's percentile grid.

    One matrix applies to every age.
    """
    return cells @ transition


def cohort_recurrence_step(
    cells_last_year: Array,
    growth: Array,
    inflow: Array,
    aging: Array,
    transition: Array,
) -> Array:
    """Step the cell shortfall forward one year.

    The order is: last year's shortfall moves across percentiles, then ages a
    year, then grows; and this year's inflow is added at face value on top.
    This year's saving out of this year's tax change does not grow in its first
    year, because it is taken to happen at year end.  There is no mortality term
    here: deaths enter when the shortfall is aggregated, through each record's
    own mortality rate.
    """
    moved = apply_percentile_transition(cells_last_year, transition)
    # Each age's cells move to the next age, with the top age absorbing.
    carried = aging.T @ moved
    return growth * carried + inflow


def growth_factor(
    r_total: float | Array,
    saving_share: Array,
) -> Array:
    """Return the growth factor G for each cell: the shortfall grows at 1 + r.

    The tax on the income the missing wealth would have earned is already
    reflected, because the carried shortfall is removed from assets before the
    calculators run (``wealth_state.apply_carried_stock``, called from
    ``worlds.annual_step``).
    """
    grown = jnp.broadcast_to(
        1.0 + jnp.asarray(r_total, dtype=jnp.float64), jnp.shape(saving_share)
    )
    # The growth factor cannot go negative.
    return jnp.maximum(grown, WEALTH_DYN_G_FLOOR)


def sum_by_cell(
    values: Array,
    row: Array,
    bin_index: Array,
    has_cell: Array,
    n_ages: int,
    n_bins: int,
) -> Array:
    """Total a per-record quantity into the age-by-percentile grid.

    Records with no cell are dropped.  This forms the yearly inflow from the
    change in tax, and the cell net worth totals the haircut divides by.
    """
    spare = n_ages * n_bins
    flat = jnp.where(has_cell, row * n_bins + bin_index, spare)
    totals = jnp.zeros(spare + 1, dtype=jnp.float64)
    totals = totals.at[flat].add(jnp.where(has_cell, values, 0.0))
    return totals[:spare].reshape(n_ages, n_bins)


def cell_rows(age_cohort: Array, ages: np.ndarray) -> Array:
    """Return each record's row in the age grid, clipped so it is safe to index with."""
    return jnp.clip(age_cohort - int(ages[0]), 0, len(ages) - 1)


# -----------------------------------------------------------------------------
# Draining the shortfall out of records
# -----------------------------------------------------------------------------


def haircut_fractions(
    cell_shortfall: Array,
    row: Array,
    bin_index: Array,
    has_cell: Array,
    net_worth: Array,
    weight: Array,
    gross_assets: Array,
    fmax: float,
) -> tuple[Array, Array]:
    """Split each cell's shortfall across its records and turn it into a fraction.

    A record takes the share of its cell's accumulated shortfall that matches its
    share of the cell's total weighted positive net worth:

        allocated = shortfall * net worth / cell total weighted positive net worth
        f         = allocated / gross assets, held within plus or minus fmax

    The clamp is symmetric because the shortfall can be negative: a tax cut is
    saved into wealth, and scales the balance sheet up.

    A record with no cell, and a record whose cell holds no positive net worth,
    is allocated nothing.  A record with no assets gets a fraction of zero.

    Returns the dollars allocated and the fraction f.
    """
    n_ages, n_bins = cell_shortfall.shape
    positive_net_worth = jnp.maximum(net_worth, 0.0)

    cell_total = sum_by_cell(
        weight * positive_net_worth, row, bin_index, has_cell, n_ages, n_bins
    )
    safe_bin = jnp.where(has_cell, bin_index, 0)
    shortfall_here = jnp.where(has_cell, cell_shortfall[row, safe_bin], 0.0)
    denominator = jnp.where(has_cell, cell_total[row, safe_bin], 0.0)

    enough_net_worth = has_cell & (denominator > WEALTH_DYN_EPS)
    allocated = jnp.where(
        enough_net_worth,
        shortfall_here
        * positive_net_worth
        / jnp.where(enough_net_worth, denominator, 1.0),
        0.0,
    )
    has_assets = gross_assets > WEALTH_DYN_EPS
    fraction = jnp.where(
        has_assets, allocated / jnp.where(has_assets, gross_assets, 1.0), 0.0
    )
    return allocated, jnp.clip(fraction, -fmax, fmax)


def apply_haircut(
    assets: Array,
    debts: Array,
    pure_capital_flows: Array,
    pass_through_flows: Array,
    retirement_distributions: Array,
    fraction: Array,
    pass_through_weight: float,
) -> dict[str, Array]:
    """Scale a record's balance sheet and capital income flows by the haircut.

    Every asset column is scaled by the same factor, so the whole balance sheet
    shrinks together and the pass-through share of it is left where it was.
    Pure capital income flows and long-term gain basis scale by the same factor,
    because those dollars are earned on the assets that shrank.  Pass-through
    flows scale by only the capital share of business income, the weight from
    the economy configuration, since the rest of that income is a return to
    labor.  Retirement distributions scale with the balances they are drawn
    from, or a faster drawdown would undo the erosion.

    Debts are left alone: the household still owes what it owed.  Net worth is
    recomputed as the eroded assets less those unchanged debts.

    Each argument holding columns is an array of records by columns; missing
    values are read as zero.

    Returns the scaled arrays and the recomputed net worth.
    """
    asset_scale = (1.0 - fraction)[:, None]
    pass_through_scale = (1.0 - pass_through_weight * fraction)[:, None]

    scaled_assets = assets * asset_scale
    debt_total = jnp.nan_to_num(debts).sum(axis=1)
    return {
        "assets": scaled_assets,
        "pure_capital_flows": pure_capital_flows * asset_scale,
        "pass_through_flows": pass_through_flows * pass_through_scale,
        "retirement_distributions": retirement_distributions * asset_scale,
        "net_worth": jnp.nan_to_num(scaled_assets).sum(axis=1) - debt_total,
    }


# -----------------------------------------------------------------------------
# Financing profiles, read on the host
# -----------------------------------------------------------------------------

# The saving share and the transition matrix come from a profile folder under
# config/calibrations/wealth_profiles/.  The economy value
# wealth.financing_profile names it and takes one of three forms:
#
#   'none' or 'off'  the channel is off, s = 0 everywhere
#   'flat:<s>'       a constant saving share at every cell, identity transition
#   <folder name>    that profile folder

WEALTH_DEFAULT_PROFILE = "default"


@dataclass(frozen=True)
class ProfileSpec:
    """The profile source a scenario names, before anything is read."""

    kind: str  # 'off', 'folder' or 'scalar'
    value: str | float | None


@dataclass(frozen=True)
class FinancingProfile:
    """A resolved profile: the saving share by cell and the percentile transition."""

    source: str
    saving_shares: np.ndarray  # ages by percentiles
    transition: np.ndarray  # percentiles by percentiles


def parse_profile_spec(financing_profile: str | None) -> ProfileSpec:
    """Read the economy value naming the profile, without building any matrices."""
    text = "" if financing_profile is None else str(financing_profile).strip()
    if not text:
        return ProfileSpec("folder", WEALTH_DEFAULT_PROFILE)
    if text.lower() in ("none", "off"):
        return ProfileSpec("off", None)
    if text.startswith("flat:"):
        rest = re.sub(r"^flat:", "", text)
        try:
            share = float(rest)
        except ValueError as error:
            raise ValueError(
                f"economy.wealth.financing_profile = '{text}' -- the flat "
                "shorthand is flat:<number>, e.g. flat:0.5"
            ) from error
        return ProfileSpec("scalar", share)
    return ProfileSpec("folder", text)


def load_saving_shares(path: str | Path, ages: np.ndarray, n_bins: int) -> np.ndarray:
    """Read a profile's s.csv into a saving share matrix of age by percentile.

    Every cell of the grid has to appear exactly once, with a share between zero
    and one.  An incomplete grid is an error.
    """
    path = Path(path)
    table = np.atleast_1d(
        np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    )
    names = table.dtype.names or ()
    missing = [column for column in ("age", "nw_pctile", "s") if column not in names]
    if missing:
        raise ValueError(
            f"wealth bathtub profile s.csv ({path}) is missing column(s): "
            f"{', '.join(missing)}. Required: age, nw_pctile, s."
        )

    age = table["age"].astype(np.int64)
    pctile = table["nw_pctile"].astype(np.int64)
    share = table["s"].astype(np.float64)

    if np.isnan(share).any():
        raise ValueError(
            f"wealth bathtub profile s.csv ({path}) has non-numeric entries."
        )
    offending = np.flatnonzero((share < 0) | (share > 1))
    if offending.size:
        first = offending[0]
        raise ValueError(
            f"wealth bathtub profile s.csv ({path}): s must be in [0, 1] "
            f"(s = 1 - MPC). First offending cell: age {age[first]}, "
            f"pctile {pctile[first]}, s = {share[first]}."
        )
    if (age < ages[0]).any() or (age > ages[-1]).any():
        raise ValueError(
            f"wealth bathtub profile s.csv ({path}): age outside "
            f"[{ages[0]}, {ages[-1]}]."
        )
    if (pctile < 1).any() or (pctile > n_bins).any():
        raise ValueError(
            f"wealth bathtub profile s.csv ({path}): nw_pctile outside [1, {n_bins}]."
        )
    cells = np.stack([age, pctile], axis=1)
    if np.unique(cells, axis=0).shape[0] != cells.shape[0]:
        raise ValueError(
            f"wealth bathtub profile s.csv ({path}) has duplicate "
            "(age, nw_pctile) rows."
        )
    if len(age) != len(ages) * n_bins:
        raise ValueError(
            f"wealth bathtub profile s.csv ({path}) must cover all "
            f"{len(ages) * n_bins} (age x pctile) cells exactly once; "
            f"found {len(age)} rows."
        )

    shares = np.full((len(ages), n_bins), np.nan)
    shares[age - ages[0], pctile - 1] = share
    if np.isnan(shares).any():
        raise ValueError(
            f"wealth bathtub profile s.csv ({path}) does not cover every "
            "(age, pctile) cell."
        )
    return shares


def load_transition_matrix(folder: str | Path, n_bins: int) -> np.ndarray:
    """Read a profile's transition matrix M, raked to doubly stochastic.

    M.csv is a headerless numeric grid of percentiles by percentiles.  No M file
    means the identity: everyone stays in the percentile they were in.

    Only a single M.csv is read.
    """
    csv = Path(folder) / "M.csv"
    if not csv.exists():
        return np.eye(n_bins)

    matrix = np.atleast_2d(np.genfromtxt(csv, delimiter=",", dtype=np.float64))
    if matrix.shape != (n_bins, n_bins):
        raise ValueError(
            f"wealth bathtub profile M.csv ({csv}) must be {n_bins} x {n_bins}; "
            f"got {matrix.shape[0]} x {matrix.shape[1]}."
        )
    if np.isnan(matrix).any() or (matrix < 0).any():
        raise ValueError(
            f"wealth bathtub profile M.csv ({csv}) has negative or missing entries."
        )
    return np.asarray(rake_to_doubly_stochastic(jnp.asarray(matrix)))


def resolve_profile(
    financing_profile: str | None,
    profiles_root: str | Path,
    ages: np.ndarray,
    n_bins: int,
) -> FinancingProfile:
    """Turn the economy's financing_profile value into a saving surface and a transition."""
    spec = parse_profile_spec(financing_profile)
    identity = np.eye(n_bins)

    if spec.kind == "off":
        return FinancingProfile("off", np.zeros((len(ages), n_bins)), identity)
    if spec.kind == "scalar":
        share = float(spec.value)  # type: ignore[arg-type]
        if not 0.0 <= share <= 1.0:
            raise ValueError(
                f"economy.wealth.financing_profile = 'flat:{share}' -- the saving "
                "share s must be in [0, 1], since s = 1 - MPC."
            )
        return FinancingProfile(
            f"flat:{share}", np.full((len(ages), n_bins), share), identity
        )

    folder = Path(profiles_root) / str(spec.value)
    if not folder.is_dir():
        raise ValueError(
            f"wealth bathtub: financing profile '{spec.value}' not found at "
            f"{folder}. Point economy.wealth.financing_profile at a folder under "
            f"{profiles_root}, at 'none', or at 'flat:<number>'."
        )
    return FinancingProfile(
        str(spec.value),
        load_saving_shares(folder / "s.csv", ages, n_bins),
        load_transition_matrix(folder, n_bins),
    )
