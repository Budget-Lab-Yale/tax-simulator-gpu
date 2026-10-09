"""The wealth state carried from one year to the next, and the steps that advance it.

The functions in :mod:`taxmodel.years.wealth_arithmetic` do one piece of the
accounting at a time.  This module supplies the order used by the annual
simulation: form this year's cells from the frame before the wealth haircut,
apply the shortfall carried into the year, then use this year's finished tax
burden to form the shortfall carried into next year.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields

import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_dataclass

from taxmodel.behavior import balance_sheet as bs
from taxmodel.frame.steps_jax import ESTATE_ASSET_COLS, ESTATE_DEBT_COLS, FILING_JOINT
from taxmodel.years import wealth_arithmetic
from taxmodel.years.record_columns import (
    PASS_THROUGH_FLOW_COLUMNS,
    first_array,
    net_pass_through_income,
)

Array = jnp.ndarray
Records = Mapping[str, Array]

# The capital-income columns that are not pass-through business income; the
# pass-through columns are ``record_columns.PASS_THROUGH_FLOW_COLUMNS``. Loss
# columns are scaled with their income columns so net income changes in the
# same proportion.
PURE_CAPITAL_FLOW_COLUMNS = (
    "txbl_int",
    "exempt_int",
    "div_ord",
    "div_pref",
    "kg_st",
    "kg_lt",
    "kg_1250",
    "kg_collect",
    "rent",
    "rent_loss",
    "estate",
    "estate_loss",
    "kg_lt_basis",
)
RETIREMENT_DISTRIBUTION_COLUMNS = (
    "txbl_ira_dist",
    "txbl_pens_dist",
    "gross_pens_dist",
)

# A cell's capital income is capped at one dollar per dollar of gross assets.
CAPITAL_YIELD_MAXIMUM = 1.0


@dataclass(frozen=True)
class WealthProfileYear:
    """The wealth-channel inputs that can affect one year's calculation.

    ``saving_shares`` has age on rows and within-age wealth percentile on
    columns, and ``transition`` moves last year's percentiles into this year's.
    ``r_total`` is this year's total nominal return, and ``ages`` and ``aging``
    give the cohort grid. ``fmax`` is the largest share of gross assets one
    year's haircut may take, and ``pass_through_weight`` is the capital share of
    pass-through business income; both come from the economy configuration.
    """

    saving_shares: Array
    transition: Array
    r_total: Array
    ages: Array
    aging: Array
    fmax: Array
    pass_through_weight: Array


register_dataclass(
    WealthProfileYear,
    data_fields=[field.name for field in fields(WealthProfileYear)],
    meta_fields=[],
)


@dataclass(frozen=True)
class CellAssignment:
    """This year's record-to-cell assignment and its cut points."""

    cutoffs: Array
    row: Array
    bin_index: Array
    has_cell: Array
    rank_net_worth: Array
    age_cohort: Array


register_dataclass(
    CellAssignment,
    data_fields=[field.name for field in fields(CellAssignment)],
    meta_fields=[],
)


def _columns(records: Records, names: tuple[str, ...], template: Array) -> Array:
    present = tuple(name for name in names if name in records)
    if not present:
        return jnp.zeros((template.shape[0], 1), dtype=jnp.float64)
    return jnp.stack(
        tuple(bs.column(records, name, template) for name in present), axis=1
    )


def age_cohort(records: Records, ages: Array) -> Array:
    """Assign joint filers the older spouse's age and top-code to the grid."""
    age1 = jnp.asarray(records["age1"])
    age2 = jnp.nan_to_num(jnp.asarray(records.get("age2", age1)), nan=-jnp.inf)
    filing_status = jnp.asarray(records["filing_status"])
    chosen = jnp.where(filing_status == FILING_JOINT, jnp.maximum(age1, age2), age1)
    return jnp.clip(chosen, ages[0], ages[-1]).astype(jnp.int64)


def assign_cells(records: Records, profile: WealthProfileYear) -> CellAssignment:
    """Rank records on this year's net worth before the wealth haircut.

    Cutoffs are recomputed each year from that year's population. Dependent
    returns receive no cell.

    The ranking is on net worth before any corporate-equity markdown. A frame
    that carries a markdown also carries the pre-markdown figure, which is
    ranked on.
    """
    rank_net_worth = jnp.asarray(
        records.get("net_worth_before_corporate_markdown", records["net_worth"])
    )
    weight = jnp.asarray(records["weight"])
    ages = jnp.asarray(profile.ages)
    cohort = age_cohort(records, ages)
    n_bins = profile.saving_shares.shape[1]
    # The cutoff functions take the age grid as host constants; the grid is
    # fixed for a run.
    ages_host = np.arange(
        wealth_arithmetic.WEALTH_DYN_AGE_MIN,
        wealth_arithmetic.WEALTH_DYN_AGE_MIN + profile.saving_shares.shape[0],
        dtype=np.int64,
    )
    cutoffs = wealth_arithmetic.within_age_cutoffs(
        rank_net_worth, weight, cohort, ages_host, n_bins
    )
    bin_index, has_cell = wealth_arithmetic.assign_within_age_bin(
        rank_net_worth, cohort, cutoffs, ages_host, n_bins
    )
    if "dep_status" in records:
        has_cell = has_cell & (jnp.asarray(records["dep_status"]) == 0)
        bin_index = jnp.where(has_cell, bin_index, -1)
    return CellAssignment(
        cutoffs=cutoffs,
        row=wealth_arithmetic.cell_rows(cohort, ages_host),
        bin_index=bin_index,
        has_cell=has_cell,
        rank_net_worth=rank_net_worth,
        age_cohort=cohort,
    )


def capital_income(records: Records, pass_through_weight: Array) -> Array:
    """Return the capital-income bundle used by the wealth channel."""
    template = first_array(records)
    get = lambda name: bs.column(records, name, template)
    pure = (
        get("txbl_int")
        + get("exempt_int")
        + get("div_ord")
        + get("div_pref")
        + get("kg_st")
        + get("kg_lt")
        + get("kg_1250")
        + get("kg_collect")
        + get("rent")
        - get("rent_loss")
        + get("estate")
        - get("estate_loss")
    )
    pass_through = net_pass_through_income(records, template)
    return pure + jnp.asarray(pass_through_weight) * pass_through


def economic_gross(records: Records) -> Array:
    """Sum the balance-sheet assets used by the estate and wealth channels."""
    template = first_array(records)
    return jnp.sum(_columns(records, ESTATE_ASSET_COLS, template), axis=1)


def apply_carried_stock(
    records: Records,
    stock: Array,
    assignment: CellAssignment,
    profile: WealthProfileYear,
) -> dict[str, Array]:
    """Apply the shortfall carried into this year to its household records."""
    template = first_array(records)
    assets = _columns(records, ESTATE_ASSET_COLS, template)
    allocated, fraction = wealth_arithmetic.haircut_fractions(
        cell_shortfall=stock,
        row=assignment.row,
        bin_index=assignment.bin_index,
        has_cell=assignment.has_cell,
        net_worth=jnp.asarray(records["net_worth"]),
        weight=jnp.asarray(records["weight"]),
        gross_assets=jnp.sum(jnp.nan_to_num(assets, nan=0.0), axis=1),
        fmax=profile.fmax,
    )
    out = scale_by_haircut(records, fraction, profile.pass_through_weight)
    out["nw_pctile"] = assignment.bin_index
    out["D_alloc"] = allocated
    return out


def scale_by_haircut(
    records: Records, fraction: Array, pass_through_weight: Array
) -> dict[str, Array]:
    """Scale each household's balance sheet and capital income by its haircut.

    `fraction` is the share of the household's assets removed, and
    `pass_through_weight` the capital share of pass-through income
    (`wealth_arithmetic.apply_haircut`). The fraction is written on the record
    as `wealth_haircut`.
    """
    template = first_array(records)
    scaled = wealth_arithmetic.apply_haircut(
        assets=_columns(records, ESTATE_ASSET_COLS, template),
        debts=_columns(records, ESTATE_DEBT_COLS, template),
        pure_capital_flows=_columns(records, PURE_CAPITAL_FLOW_COLUMNS, template),
        pass_through_flows=_columns(records, PASS_THROUGH_FLOW_COLUMNS, template),
        retirement_distributions=_columns(
            records, RETIREMENT_DISTRIBUTION_COLUMNS, template
        ),
        fraction=fraction,
        pass_through_weight=pass_through_weight,
    )

    out = dict(records)
    for index, name in enumerate(name for name in ESTATE_ASSET_COLS if name in records):
        out[name] = scaled["assets"][:, index]
    for index, name in enumerate(
        name for name in PURE_CAPITAL_FLOW_COLUMNS if name in records
    ):
        out[name] = scaled["pure_capital_flows"][:, index]
    for index, name in enumerate(
        name for name in PASS_THROUGH_FLOW_COLUMNS if name in records
    ):
        out[name] = scaled["pass_through_flows"][:, index]
    for index, name in enumerate(
        name for name in RETIREMENT_DISTRIBUTION_COLUMNS if name in records
    ):
        out[name] = scaled["retirement_distributions"][:, index]
    out["net_worth"] = scaled["net_worth"]
    out["wealth_haircut"] = fraction
    return out


def cell_growth(profile: WealthProfileYear) -> Array:
    """The factor each cell's carried shortfall grows by this year.

    The shortfall grows at the nominal return. The tax the household does not
    pay on the wealth it does not have is accounted for in the tax
    calculators (see ``wealth_arithmetic.growth_factor``).

    The factor is the same in every cell.
    """
    return wealth_arithmetic.growth_factor(profile.r_total, profile.saving_shares)


def advance_stock(
    stock: Array,
    weighted_burden: Array,
    assignment: CellAssignment,
    profile: WealthProfileYear,
) -> tuple[Array, Array]:
    """Advance the cell shortfall after this year's taxes have been computed.

    Returns next year's stock and the fresh inflow that entered it. The
    diagnostics tables report the inflow.
    """
    n_ages, n_bins = profile.saving_shares.shape
    burden_cells = wealth_arithmetic.sum_by_cell(
        weighted_burden,
        assignment.row,
        assignment.bin_index,
        assignment.has_cell,
        n_ages,
        n_bins,
    )
    growth = cell_growth(profile)
    # The fresh inflow is aged one step before it enters the stock. The stock
    # this function returns is applied to next year's records, when the
    # households whose tax change created the inflow are a year older, so
    # aging the inflow keeps the burden attached to the households who paid
    # it.
    fresh_inflow = profile.saving_shares * burden_cells
    aged_inflow = profile.aging.T @ fresh_inflow
    next_stock = wealth_arithmetic.cohort_recurrence_step(
        cells_last_year=stock,
        growth=growth,
        inflow=aged_inflow,
        aging=profile.aging,
        transition=profile.transition,
    )
    return next_stock, aged_inflow
