"""Reading household columns, and the asset classes the avoidance modules use.

The column helpers here are used across the model. Both avoidance modules split the household balance sheet the same way and read
the same columns off the frame, so those pieces are defined here.

Marketable wealth is publicly traded and liquid; closely held wealth is
private business and other nonfinancial wealth, whose value is a matter of
appraisal. The two lists together cover all fourteen asset columns.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp

from taxmodel.frame.steps_jax import ESTATE_ASSET_COLS, ESTATE_DEBT_COLS

Array = jnp.ndarray
Records = Mapping[str, Array]

# Publicly traded and liquid wealth. These assets have an exchange price, so
# the only avoidance response is to not report them.
MARKETABLE_ASSET_COLS: tuple[str, ...] = (
    "value.cash",
    "value.equities",
    "value.bonds",
    "value.dc",
    "value.db",
    "value.life_ins",
    "value.annuities",
    "value.trusts",
    "value.other_fin",
)

# Closely held wealth: private business, real estate, and other nonfinancial
# assets. Valuation is a matter of appraisal, so part of the response to a
# wealth tax is a lower reported value.
CLOSELY_HELD_ASSET_COLS: tuple[str, ...] = (
    "value.pass_throughs",
    "value.primary_home",
    "value.other_home",
    "value.re_fund",
    "value.other_nonfin",
)

# Debts are not part of the avoidance response: understating a debt would
# raise the net-worth base.
DEBT_COLS: tuple[str, ...] = ESTATE_DEBT_COLS

# The two lists together must be the fourteen asset columns the estate and
# wealth calculators tax. An asset added to the frame but not classified is
# caught here at import time.
if set(MARKETABLE_ASSET_COLS) | set(CLOSELY_HELD_ASSET_COLS) != set(
    ESTATE_ASSET_COLS
) or len(MARKETABLE_ASSET_COLS) + len(CLOSELY_HELD_ASSET_COLS) != len(
    ESTATE_ASSET_COLS
):
    raise ImportError(
        "the marketable and closely held asset lists must partition the "
        "balance sheet's asset columns exactly once each"
    )

# The three factors the income-evasion module writes on the record: the share
# of each kind of closely held income still reported after evasion. A factor
# of one means nothing was evaded, and is what the avoidance modules assume
# when the evasion module did not run; 0.9 means a tenth of that kind of
# income went unreported. The evasion module writes the factors for every
# household, including households with no income of that kind.
EVASION_FACTOR_COLS: tuple[str, ...] = (
    "evasion_g_schc",  # sole proprietorship and farm income
    "evasion_g_pt",  # partnership and S corporation income
    "evasion_g_rent",  # rental income
)
NO_EVASION = 1.0


def column(
    records: Records, name: str, missing_like: Array | None = None
) -> Array:
    """One frame column as an array, with a missing entry read as zero.

    Balance-sheet and income columns arrive from the estate microdata with
    gaps in them, and every base in both models treats a gap as zero.

    A column the run did not supply is an error unless ``missing_like`` is
    given, in which case it reads as zeros of that array's shape. Works for
    price columns too.
    """
    if missing_like is not None and name not in records:
        return jnp.zeros_like(missing_like)
    return jnp.nan_to_num(jnp.asarray(records[name], dtype=float), nan=0.0)


def as_given(
    records: Records, name: str, missing_like: Array | None = None
) -> Array:
    """One frame column as the frame holds it, missing entries included.

    ``missing_like`` means the same as in ``column``.
    """
    if missing_like is not None and name not in records:
        return jnp.zeros_like(missing_like)
    return jnp.asarray(records[name], dtype=float)


def keep_what_was_missing(given: Array, changed: Array) -> Array:
    """The module's value where the frame had a value, and the frame's gap where it did not.

    A missing entry goes back out missing. The response is computed on a
    reading of the column that treats a gap as zero, but that reading does not
    reach the frame.
    """
    return jnp.where(jnp.isfinite(jnp.asarray(given, dtype=float)), changed, given)


def fill_when_written_into(given: Array, changed: Array, moved: Array) -> Array:
    """Like `keep_what_was_missing`, except a column that receives dollars is filled.

    When a module moves money into or out of a column (entity shifting taking
    its gains offset out of long-term capital gains), a missing entry is
    treated as zero and written back, so the moved dollars are not lost. A
    record the module moved nothing for keeps its gap.
    """
    given = jnp.asarray(given, dtype=float)
    return jnp.where(jnp.isfinite(given) | (moved != 0), changed, given)


def total(
    records: Records, names: tuple[str, ...], missing_like: Array | None = None
) -> Array:
    """Several columns added up per household.

    ``missing_like`` means the same as in ``column``: a column the run did not
    supply adds nothing.
    """
    return sum(
        (column(records, name, missing_like) for name in names), jnp.asarray(0.0)
    )


def divide_where_positive(
    numerator: Array, denominator: Array, otherwise: Array | float = 0.0
) -> Array:
    """Numerator over denominator where the denominator is positive, and
    ``otherwise`` (default zero) where it is not.

    The denominator is replaced before dividing so that no NaN is formed. A
    NaN denominator reads as not positive. Elementwise, so a scalar
    denominator works as well as a column.
    """
    positive = denominator > 0.0
    return jnp.where(
        positive, numerator / jnp.where(positive, denominator, 1.0), otherwise
    )


def wealth_tax_rate_key(asset_column: str) -> str:
    """The price column holding the wealth tax charged on a dollar of an asset."""
    return f"mtr_wealth_tax_on_{asset_column}"


# The tax on one more dollar of the taxable estate itself, without the SOI
# measurement bridge's adjustment. The estate reporting response reads this
# rate. The name matches `taxes/marginal_rates.ESTATE_BASE_RATE`.
ESTATE_TAXABLE_BASE_RATE_KEY = "mtr_estate_tax_on_taxable_base"


def rate_on_base(
    records: Records,
    prices: Mapping[str, Array],
    columns: tuple[str, ...],
) -> Array:
    """The marginal tax rate on a base made of several asset columns.

    The pricing pass gives one rate per asset column: the tax on one more
    dollar of cash, or of closely held business. A response to the rate on a
    whole class of wealth needs those collapsed into a single rate for its
    base:

        rate on the base = sum over its columns of (rate on the column times
        the dollars in that column), divided by the dollars in the base

    This is the tax cost of scaling the whole base up proportionally, per
    dollar of base. A household with nothing in the base gets a rate of zero.

    The prices were measured on the mechanical frame before any behavioral
    module ran, and `stop_gradient` holds them fixed: differentiating a
    module's output with respect to an asset column gives the response at the
    priced rate and does not re-price the law.
    """
    missing = [c for c in columns if wealth_tax_rate_key(c) not in prices]
    if missing:
        raise KeyError(
            "the rate on a base needs the marginal tax rate on every asset "
            "column in it; the prices handed in have no "
            + ", ".join(wealth_tax_rate_key(c) for c in missing)
        )
    amounts = {c: column(records, c) for c in columns}
    base = sum(amounts.values(), jnp.asarray(0.0))
    tax_on_scaling_the_base_up = sum(
        (
            jnp.asarray(prices[wealth_tax_rate_key(c)], dtype=float) * amounts[c]
            for c in columns
        ),
        jnp.asarray(0.0),
    )
    rate = divide_where_positive(tax_on_scaling_the_base_up, base)
    return jax.lax.stop_gradient(rate)


def evaded_closely_held_income_share(records: Records) -> Array:
    """The share of the household's closely held income the evasion module hid.

    The evasion module leaves three retained-reporting factors on the record,
    one per kind of closely held income. The hidden share of each kind is
    weighted by that kind's positive income. Only positive income counts; a
    loss is not hidden.

    Both avoidance modules read this. Income that is not reported implies
    assets that are not reported either, so the assets behind evaded income
    drop out of reported wealth and the reported estate.
    """
    factors = {
        name: jnp.asarray(records.get(name, NO_EVASION), dtype=float)
        for name in EVASION_FACTOR_COLS
    }
    # The frame carries the two earners' sole proprietorship income
    # separately; the household total is their sum.
    sole_prop = jnp.maximum(
        column(records, "sole_prop1") + column(records, "sole_prop2"), 0.0
    )
    pass_through = (
        jnp.maximum(column(records, "part_active"), 0.0)
        + jnp.maximum(column(records, "part_passive"), 0.0)
        + jnp.maximum(column(records, "scorp_active"), 0.0)
        + jnp.maximum(column(records, "scorp_passive"), 0.0)
    )
    rental = jnp.maximum(column(records, "rent"), 0.0)
    all_closely_held_income = sole_prop + pass_through + rental
    hidden_dollars = (
        sole_prop * (1.0 - factors["evasion_g_schc"])
        + pass_through * (1.0 - factors["evasion_g_pt"])
        + rental * (1.0 - factors["evasion_g_rent"])
    )
    share = divide_where_positive(hidden_dollars, all_closely_held_income)
    return jnp.clip(share, 0.0, 1.0)
