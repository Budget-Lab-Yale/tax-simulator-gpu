"""Average marginal tax rates, weighted by the dollars they are charged on.

For each projection year, each income group, and each of the twenty-seven
dollars the model prices, this table reports the marginal tax rate on the
average dollar of that thing. The wage row for the middle quintile in 2031 is

    sum over households of (weight * wage dollars * wage marginal rate)
    -------------------------------------------------------------------
    sum over households of (weight * wage dollars)

This is a dollar-weighted average, not a population-weighted one.

Each row carries a baseline rate, a reform rate, and the change. The model
prices the mechanical frame under both laws with the household held fixed, so
the difference is the price change the reform causes.

The weighting base for each rate
--------------------------------
The priced dollars are listed in ``PRICED_DOLLARS`` in
``taxmodel.taxes.marginal_rates``, which records the frame column each priced
dollar moves. There is one row per priced dollar.

Each priced dollar is named for the frame column it moves, and that column is
its weighting base. The exception is the general partner's dollar, which the
law puts in two bases at once: ordinary partnership income on Schedule E and
net earnings from self-employment subject to SECA. ``PRICED_DOLLARS`` moves
both ``part_active`` and ``part_se1``, but the weighting base is
``part_active`` alone, since ``part_se1`` is the same partnership income
counted a second time.

Several of the twenty-seven price a deduction rather than income: medical
expenses, the three state-and-local tax columns, first-mortgage interest,
car-loan interest, and the two charitable columns. The weighting base is the
deduction column itself, so the reported number is the marginal rate on the
average deducted dollar. It is normally negative.

Each base is floored at zero before weighting, so a household with a loss in
that column carries no weight. Partnership income, sole-proprietor income,
S-corporation income, long-term capital gain and rent can all be negative. A
household with a rental loss earned no rental dollar. Negative weights would
net losses against other households' income, which can drive a group's
denominator to nearly zero and put the reported average outside the range of
every rate any household in the group faces.

The reported rate is a ratio of two weighted sums, taken group by group. Rates
are never averaged across households directly and never clipped. A group with
nothing in the denominator reports not-a-number, as
``reporting/distribution_jax.py`` does for an average over no households.

Data flow across the compilation boundary
-----------------------------------------
``AnnualResults`` carries ``mechanical_prices_reform`` and
``mechanical_prices_baseline`` as one value per household, and per-household
results do not leave the compiled year. The reduction therefore happens inside
the compiled loop, and two accumulators per (group, rate) cross the boundary:
the weighted sum of rate times dollars, and the weighted sum of dollars. The
division happens on the host, once, after the run.

Two accumulators can be combined across years or groups after the fact: adding
them and dividing once gives the same answer as a single pass over the combined
population. A finished per-year rate cannot be re-averaged without the dollars
it was weighted by.

Arguments from the orchestrator
-------------------------------
`mtr_table_sums` is called once per year from
`reporting/device_sums.mtr_table_sums_for_year`. `rates` must hold every name
in `marginal_rates.MARGINAL_RATE_COLUMNS`. The price dictionaries also carry
the wealth tax's and the estate tax's rate on each balance-sheet column; this
table leaves those out because each would need a weighting base of its own.
`records` holds the base column of each priced dollar. `group_labels` is one
group per slot, or -1 for none. `n_groups` is a plain integer, because it sets
array shapes. The year loop stacks the two accumulators on a leading year
axis, and `mtr_table` reads that stacked form.

Records with no group are accumulated in one extra column past the last
group, so that the group rows plus the no-group row add up to the whole
population. The Overall row is that sum.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from taxmodel.taxes import marginal_rates

Array = jnp.ndarray
Records = Mapping[str, Array]

# The two accumulators that cross the compiled loop's boundary.
RATE_TIMES_DOLLARS = "weighted_rate_times_dollars"
DOLLARS = "weighted_dollars"

# The label a group row gets when the caller supplies no names of its own.
GROUP_LABEL_TEMPLATE = "Group {number}"
NO_GROUP_LABEL = "No group"
OVERALL_LABEL = "Overall"

# How each price is described in the printed table. The keys are the endings
# `marginal_rates.PRICES` puts on a rate's column name.
TAXES_INCLUDED_LABELS: dict[str, str] = {
    "": "individual income tax after refundable credits",
    "_plus_employee_payroll": (
        "individual income tax after refundable credits plus employee payroll tax"
    ),
}


# Every rate this table reports: the stored column's name and the dollar it
# prices, under each law in turn. The names are `MARGINAL_RATE_COLUMNS` in its
# own order; the dollar comes from the same entry the name was built from.
RATE_LAYOUT: tuple[tuple[str, str], ...] = tuple(
    (column, rate.dollar)
    for law_side in (marginal_rates.BASELINE_LAW, marginal_rates.REFORM_LAW)
    for column, rate in zip(
        marginal_rates.rate_columns(law_side),
        marginal_rates.PRICED_DOLLAR_RATES,
        strict=True,
    )
)

# The order of the rate axis of both accumulators.
RATE_NAMES = marginal_rates.MARGINAL_RATE_COLUMNS

# Each rate is weighted by the frame column the priced dollar is named for, so
# that column has to exist.
for _priced_dollar, _moves in marginal_rates.PRICED_DOLLARS.items():
    if _priced_dollar not in _moves:
        raise ImportError(
            f"the priced dollar {_priced_dollar!r} does not move a frame column of "
            "the same name, so there is no column holding the dollars it prices"
        )

# The (priced dollar, taxes included) pairs a row is reported for, in the order
# the rows come out. Each pair has a baseline rate and a reform rate.
PRICED_ROWS = tuple(
    (priced_dollar, price.ending)
    for price in marginal_rates.PRICES
    for priced_dollar in marginal_rates.PRICED_DOLLARS
)


def mtr_table_sums(
    rates: Mapping[str, Array],
    records: Records,
    group_labels: Array,
    weight: Array,
    n_groups: int,
) -> dict[str, Array]:
    """Add up one year's rates and dollar bases within each income group.

    This runs inside the compiled year loop. It returns two accumulators: for
    every (group, rate) cell, the weighted sum of the rate times the dollars it
    is charged on, and the weighted sum of those dollars. Both come back shaped
    ``[n_groups + 1, number of rates]``, with the rate axis in the order of
    ``RATE_NAMES`` and the last group column holding the records
    ``group_labels`` gave no group to.

    ``rates`` must hold every name in
    ``marginal_rates.MARGINAL_RATE_COLUMNS``; anything else in it is ignored.
    ``records`` is the frame those rates were priced on. The module docstring
    states the whole contract, including which dollars weight which rate.
    """
    if n_groups < 1:
        raise ValueError(f"a table needs at least one group, got {n_groups}")
    weight = jnp.asarray(weight)
    group_labels = jnp.asarray(group_labels)

    # A record with no group goes in one column past the last real group, so
    # that adding every column back up returns the whole population.
    no_group_column = n_groups
    column_of_record = jnp.where(group_labels >= 0, group_labels, no_group_column)

    rate_times_dollars = []
    dollars = []
    for name, priced_dollar in RATE_LAYOUT:
        # A household with a loss in this column carries no weight.
        base = jnp.maximum(jnp.asarray(records[priced_dollar]), 0.0)
        weighted_base = weight * base
        dollars.append(weighted_base)
        rate_times_dollars.append(weighted_base * jnp.asarray(rates[name]))

    def by_group(per_record: list[Array]) -> Array:
        # Each record's group becomes a one-hot row and one matrix product
        # adds every record into its group. On the GPU this is faster than a
        # float64 segment_sum.
        stacked = jnp.stack(per_record, axis=1)
        in_group = (
            column_of_record[:, None] == jnp.arange(n_groups + 1)[None, :]
        ).astype(stacked.dtype)
        return jnp.einsum("ng,nr->gr", in_group, stacked)

    return {
        RATE_TIMES_DOLLARS: by_group(rate_times_dollars),
        DOLLARS: by_group(dollars),
    }


def _group_row_labels(n_groups: int, group_names: Sequence[str] | None) -> list[str]:
    """The name of every row of the group axis, ending with the no-group row."""
    if group_names is None:
        named = [
            GROUP_LABEL_TEMPLATE.format(number=index + 1) for index in range(n_groups)
        ]
    else:
        if len(group_names) != n_groups:
            raise ValueError(
                f"there are {n_groups} groups and {len(group_names)} names for them"
            )
        named = list(group_names)
    return named + [NO_GROUP_LABEL]


def mtr_table(
    sums: Mapping[str, Array],
    *,
    years: Sequence[int] | None = None,
    group_names: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    """Divide the accumulated sums and lay the table out one row at a time.

    ``sums`` is what ``mtr_table_sums`` returned for every year, stacked on a
    leading year axis: two arrays shaped ``[years, groups + 1, rates]``. To
    report several years as one figure, add the two arrays across the year
    axis first (keeping the axis) and pass the result here; adding the
    accumulators and dividing once gives the same answer as one pass over the
    combined population.

    Each row is one year, one group, one priced dollar and one definition of
    which taxes are in the price. It carries the dollars being
    priced, the average marginal rate under baseline law, the same under the
    reform's law, and the change. A group with no dollars of the thing being
    priced reports not-a-number for all three.

    Rows come out for each group in order, then the records with no group, then
    an Overall row that is every one of those columns added together.
    """
    numerator = np.asarray(sums[RATE_TIMES_DOLLARS], dtype=np.float64)
    denominator = np.asarray(sums[DOLLARS], dtype=np.float64)
    if numerator.shape != denominator.shape:
        raise ValueError(
            f"the two accumulators are shaped {numerator.shape} and "
            f"{denominator.shape}; they must match"
        )
    if numerator.ndim != 3:
        raise ValueError(
            "the accumulators must be shaped [years, groups + 1, rates]; got "
            f"{numerator.shape}"
        )
    n_years, n_columns, n_rates = numerator.shape
    if n_rates != len(RATE_NAMES):
        raise ValueError(
            f"the accumulators carry {n_rates} rates; this table reports "
            f"{len(RATE_NAMES)}"
        )
    n_groups = n_columns - 1
    if n_groups < 1:
        raise ValueError(
            "the accumulators need one column per group plus one for the records "
            f"with no group; {n_columns} columns leaves no groups"
        )

    if years is None:
        calendar_years: list[int] = list(range(n_years))
    else:
        if len(years) != n_years:
            raise ValueError(
                f"the accumulators cover {n_years} years and {len(years)} years "
                "were named"
            )
        calendar_years = [int(year) for year in years]

    # The Overall row adds every group and the no-group records together, so it
    # covers the whole population whether or not every record was grouped.
    numerator = np.concatenate(
        [numerator, numerator.sum(axis=1, keepdims=True)], axis=1
    )
    denominator = np.concatenate(
        [denominator, denominator.sum(axis=1, keepdims=True)], axis=1
    )
    row_labels = _group_row_labels(n_groups, group_names) + [OVERALL_LABEL]

    position_of_rate = {name: index for index, name in enumerate(RATE_NAMES)}

    rows: list[dict[str, object]] = []
    for year_index, year in enumerate(calendar_years):
        for column, label in enumerate(row_labels):
            for priced_dollar, ending in PRICED_ROWS:
                baseline = position_of_rate[
                    f"{marginal_rates.BASELINE_LAW}_mtr_{priced_dollar}{ending}"
                ]
                reform = position_of_rate[
                    f"{marginal_rates.REFORM_LAW}_mtr_{priced_dollar}{ending}"
                ]
                base_dollars = float(denominator[year_index, column, reform])
                if base_dollars == 0.0:
                    baseline_rate = float("nan")
                    reform_rate = float("nan")
                else:
                    baseline_rate = (
                        float(numerator[year_index, column, baseline]) / base_dollars
                    )
                    reform_rate = (
                        float(numerator[year_index, column, reform]) / base_dollars
                    )
                rows.append(
                    {
                        "year": year,
                        "group": label,
                        "priced_dollar": priced_dollar,
                        "taxes_included": TAXES_INCLUDED_LABELS[ending],
                        "base_dollars": base_dollars,
                        "average_marginal_rate_baseline_law": baseline_rate,
                        "average_marginal_rate_reform_law": reform_rate,
                        "change_in_average_marginal_rate": reform_rate - baseline_rate,
                    }
                )
    return rows


# The fields every row carries, in the order they are reported.
TABLE_COLUMNS: tuple[str, ...] = (
    "year",
    "group",
    "priced_dollar",
    "taxes_included",
    "base_dollars",
    "average_marginal_rate_baseline_law",
    "average_marginal_rate_reform_law",
    "change_in_average_marginal_rate",
)
