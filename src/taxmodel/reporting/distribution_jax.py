"""The distribution table for one year, computed on the device.

Every intermediate value is a JAX array so the table can be computed inside
the compiled projection. ``reporting/distribution.py`` states the table's
layout: its cuts, group labels, row order, columns and rounding steps. The
result has one row for every group in that layout, including an empty row
when a small sample has no household in a group, because a compiled projection
needs the same shape every year.

The table
---------
`distribution_table` forms the table under one way of spreading the
corporate income tax; `reporting/device_sums.distribution_table_for_year`
calls it once for each spread in `distribution.CORPORATE_SPREADS`. It reports
three worlds: the baseline, the static world
(first_order plus each household's top-down share of the reform's static
corporate change) and the conventional world. For each world a household has
six taxes: income tax, payroll tax (employer share included), estate tax and
tax on gains deemed realized at death (both on the heir copy), the wealth
tax, and its share of the corporate income tax. Its pretax income in each
world is that world's own income measure plus the bequest (heir copy) plus
its share of current-law corporate tax; the conventional world's also adds
back what the record-level corporate and employer-payroll steps took out of
income. Expanded income is the cash measure and Haig-Simons income the
accrual measure.

The table has one block of rows for all six taxes together and one for each
tax alone. Within a block, the change columns compare the static world with
the baseline for that block's tax, on baseline pretax cash income, and the
level columns report each world's tax, each world's pretax cash and accrual
income, and the two effective rates (group tax over group income, with no
rate where group income is not positive). Group membership comes from
baseline amounts: the Income cut ranks on baseline pretax cash income, the
Accrual income cut on baseline pretax accrual income, and the AGI and Net
worth cuts keep their own rankings. The per-tax blocks add up to the total
block in tax dollars, net change, every effective rate, the percent change in
after-tax income and, within the five-dollar rounding, the average tax change.
The counts, cutoffs, incomes and baseline after-tax income are the same on
every block of a group.

Ranking and records with no group
---------------------------------
Each cut says what to do with records that have no group. A quintile cut labels
them: negative income has no percentile, so those records are reported as their
own group. A top-percent cut drops them from the printed table after the
metrics are computed, so they still count in the denominator when a top group's
share of the total change is worked out. The top one percent's share of a
reform's cost is therefore a share of the whole population's cost.

A household is ranked if its value is zero or more, so a household with exactly
zero income sits at the bottom of the income distribution. The wealth bathtub
excludes zeros, because it ranks a different population.

Averages are reported to the nearest five dollars, and the net change in a
group adds up each household's tax change rounded to the nearest dollar first.

Heirs
-----
`heir_and_non_heir_copies` doubles the household axis. Which households
inherit in a given year is unknown, only how likely each one is to, so the
reporting population splits each household into an heir copy weighted by that
chance and a non-heir copy weighted by the rest. The heir copy carries the
bequest and the taxes owed at death; the non-heir copy carries neither. The
doubling happens only in this reporting reduction; the tax calculation and the
wealth channel run on one copy of each household.

Tax owed at death reaches this table through
``reporting/estate_allocator_jax.py``, which moves estate tax and tax on gains
deemed realized at death off the decedent and onto heirs.
"""

from __future__ import annotations

import jax.numpy as jnp

from taxmodel.reporting import distribution

Array = jnp.ndarray

def quantile_groups(
    values: Array,
    weights: Array,
    n_groups: int,
    valid: Array | None = None,
) -> Array:
    """Rank records into `n_groups` equal-weight groups by cumulative weight.

    A record's rank uses cumulative weight through that record. A record whose
    cumulative share lands exactly on a boundary stays in the lower group.
    Valid records with zero weight still receive a group, while records outside
    ``valid`` receive -1 and do not affect anyone's rank.
    """
    values = jnp.asarray(values, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    keep = jnp.ones(values.shape, dtype=bool) if valid is None else jnp.asarray(valid)

    # Invalid records sort after the ranked population and carry no weight.
    # Stable sorting preserves source order among tied values.
    order = jnp.argsort(jnp.where(keep, values, jnp.inf), stable=True)
    sorted_weight = jnp.where(keep[order], weights[order], 0.0)
    total_weight = jnp.sum(sorted_weight)
    cumulative_weight = jnp.cumsum(sorted_weight)
    safe_total = jnp.where(total_weight > 0.0, total_weight, 1.0)
    assigned_in_order = jnp.clip(
        jnp.ceil(cumulative_weight / safe_total * n_groups).astype(jnp.int64) - 1,
        0,
        n_groups - 1,
    )
    assigned = jnp.empty(values.shape, dtype=jnp.int64).at[order].set(
        assigned_in_order
    )
    return jnp.where(keep & (total_weight > 0.0), assigned, -1)


def heir_and_non_heir_copies(
    households: dict[str, Array],
    probability_of_inheriting: Array,
) -> dict[str, Array]:
    """Split every household into an heir copy and a non-heir copy.

    `households` holds the baseline amounts the distribution table ranks on,
    one value per household. What comes back holds the same names, each array
    twice as long: the heir copies first, in household order, then the
    non-heir copies. The heir copy's weight is the household's weight times
    its chance of inheriting, the non-heir copy's weight is the remainder, and
    the copies are otherwise identical.

    A copy carrying no weight is marked invalid, since a compiled array has a
    fixed length. That covers a household certain not to inherit and, because
    the published chance of inheriting is above one for a handful of records, a
    household whose non-heir copy would have a negative weight.
    """
    chance = jnp.asarray(probability_of_inheriting, dtype=jnp.float64)
    weight = jnp.asarray(households["weight"], dtype=jnp.float64)
    heir_weight = weight * chance
    non_heir_weight = weight * (1.0 - chance)

    keep = jnp.asarray(households.get("valid", jnp.ones(weight.shape, dtype=bool)))
    dependent = households.get("is_dependent_return")
    if dependent is not None:
        keep = keep & ~jnp.asarray(dependent, dtype=bool)

    copies: dict[str, Array] = {
        name: jnp.concatenate([value, value])
        for name, value in households.items()
        if name not in ("weight", "valid", "is_dependent_return")
    }
    copies["weight"] = jnp.concatenate([heir_weight, non_heir_weight])
    copies["valid"] = jnp.concatenate(
        [keep & (heir_weight > 0.0), keep & (non_heir_weight > 0.0)]
    )
    return copies


def baseline_rank_groups(values: Array, weight: Array, keep: Array) -> Array:
    """Return the baseline rank label for each nonnegative kept record."""
    ranked = keep & (values >= 0.0)
    return quantile_groups(values, weight, distribution.RANK_GROUPS, ranked)


def _cut_masks(ranked: Array, negative: Array, groups: Array) -> list[Array]:
    """The eleven rows one ranked cut reports.

    Five quintiles, then the records whose amount is negative, then the five
    top cuts. `ranked` says who the percentiles were computed over, `negative`
    who is below zero and so was left out of them, and `groups` is each
    record's position among the ten thousand equal-weight groups, or -1 for a
    record that was not ranked.
    """
    quintile = jnp.clip(
        groups // (distribution.RANK_GROUPS // len(distribution.QUINTILE_LABELS)),
        0,
        len(distribution.QUINTILE_LABELS) - 1,
    )
    masks = [
        ranked & (quintile == index)
        for index in range(len(distribution.QUINTILE_LABELS))
    ]
    masks.append(negative)
    masks += [groups >= lowest_group for _label, lowest_group in distribution.TOP_CUTS]
    return masks


def _ranked_group_masks(values: Array, keep: Array, groups: Array) -> list[Array]:
    """The quintile, negative-value, and top-percent rows for one amount."""
    return _cut_masks(keep & (values >= 0.0), keep & (values < 0.0), groups)


def _group_masks(
    *,
    keep: Array,
    age: Array,
    has_child_under_18: Array,
    cash_income: Array,
    cash_rank_groups: Array,
    accrual_income: Array,
    accrual_rank_groups: Array,
    agi: Array,
    agi_rank_groups: Array,
    net_worth: Array,
    net_worth_rank_groups: Array,
) -> Array:
    """Which copies belong to each group of the table, shaped [group, copy].

    One row per entry of ``distribution.GROUPS``, in that order. Every amount
    is a baseline amount, so a copy sits in the same group in every world.
    """
    age_group = jnp.searchsorted(
        jnp.asarray(distribution.AGE_GROUP_UPPER_EDGES), age, side="right"
    )
    masks: list[Array] = [keep]
    masks += [
        keep & (age_group == index)
        for index in range(len(distribution.AGE_GROUP_LABELS))
    ]
    masks += _ranked_group_masks(cash_income, keep, cash_rank_groups)
    masks += _ranked_group_masks(accrual_income, keep, accrual_rank_groups)
    masks += [keep & ~has_child_under_18, keep & has_child_under_18]
    masks += _ranked_group_masks(agi, keep, agi_rank_groups)
    masks += _ranked_group_masks(net_worth, keep, net_worth_rank_groups)
    masks.append(keep & (net_worth >= distribution.BILLIONAIRE_NET_WORTH_DOLLARS))
    return jnp.stack(masks)


def distribution_table(
    *,
    weight: Array,
    valid: Array,
    age: Array,
    has_child_under_18: Array,
    agi: Array,
    net_worth: Array,
    agi_rank_groups: Array,
    net_worth_rank_groups: Array,
    pretax_cash_income: Array,
    pretax_accrual_income: Array,
    tax: Array,
) -> Array:
    """The distribution table for one year.

    Every per-household input is laid out over the reporting population: the
    heir copies of the households, then the non-heir copies, as
    :func:`heir_and_non_heir_copies` builds them, or the households once when
    the run has no heir structure. `valid` marks the copies the table counts.

    `agi` and `net_worth` are baseline amounts, and `agi_rank_groups` and
    `net_worth_rank_groups` their baseline rank labels (:func:`baseline_rank_groups`).

    `pretax_cash_income` and `pretax_accrual_income` are shaped
    [world, copy], the worlds in `distribution.LEVEL_WORLDS` order. Each is
    that world's own income measure (expanded income, Haig-Simons income)
    plus the bequest on the heir copy plus the household's share of
    current-law corporate tax; the conventional world's also adds back what
    the record-level corporate and employer-payroll steps took out of income.
    Row 0 is the baseline, and the Income and Accrual income cuts rank on it.

    `tax` is shaped [world, tax, copy], the taxes in `distribution.TAXES`
    order, in dollars. The corporate entry is the household's top-down share
    of corporate tax in that world; the estate and deemed-at-death entries are
    on the heir copy only.

    Returns [tax row, group, column]: tax rows in `distribution.TAX_ROWS`
    order (the total first), groups in `distribution.GROUPS` order, columns in
    `distribution.value_column_names()` order. The change columns compare the
    static world (row 1) with the baseline (row 0) on cash income.

    A household's change in total tax is the sum of its changes in the six
    taxes, and its change rounded to the dollar, which the net change adds up,
    is the sum of the six changes each rounded to the dollar. The per-tax rows
    of the net change then add up to the total row exactly.
    """
    weight = jnp.asarray(weight, dtype=jnp.float64)
    keep = jnp.asarray(valid, dtype=bool)
    agi = jnp.asarray(agi, dtype=jnp.float64)
    net_worth = jnp.asarray(net_worth, dtype=jnp.float64)
    cash = jnp.asarray(pretax_cash_income, dtype=jnp.float64)
    accrual = jnp.asarray(pretax_accrual_income, dtype=jnp.float64)
    tax = jnp.asarray(tax, dtype=jnp.float64)
    baseline_world = distribution.LEVEL_WORLDS.index("baseline")
    static_world = distribution.LEVEL_WORLDS.index("static")
    baseline_cash = cash[baseline_world]
    baseline_accrual = accrual[baseline_world]

    masks = _group_masks(
        keep=keep,
        age=jnp.asarray(age, dtype=jnp.float64),
        has_child_under_18=jnp.asarray(has_child_under_18, dtype=bool),
        cash_income=baseline_cash,
        cash_rank_groups=baseline_rank_groups(baseline_cash, weight, keep),
        accrual_income=baseline_accrual,
        # The Accrual income cut ranks on baseline pretax Haig-Simons income,
        # counting the current-law corporate share and the bequest, as the
        # Income cut does on cash income.
        accrual_rank_groups=baseline_rank_groups(baseline_accrual, weight, keep),
        agi=agi,
        agi_rank_groups=jnp.asarray(agi_rank_groups),
        net_worth=net_worth,
        net_worth_rank_groups=jnp.asarray(net_worth_rank_groups),
    )

    # Each world's tax for every tax row, [tax row, world, copy]: the total
    # first, then each tax alone.
    tax_by_row = jnp.concatenate([jnp.sum(tax, axis=1)[None], tax.transpose(1, 0, 2)])
    change = tax_by_row[:, static_world] - tax_by_row[:, baseline_world]
    # The total row's net change adds each household's per-tax changes
    # rounded to the dollar, so that the per-tax net changes add up to it
    # exactly.
    rounded_change_by_tax = jnp.round(change[1:])
    rounded_change = jnp.concatenate(
        [jnp.sum(rounded_change_by_tax, axis=0)[None], rounded_change_by_tax]
    )
    # Every tax row reports the group's whole baseline after-tax income, and
    # its percent change in after-tax income is minus that tax's change over
    # it.
    baseline_after_tax = baseline_cash - tax_by_row[0, baseline_world]

    # Every amount the table adds up within groups, one column per feature,
    # so that the group totals are one weighted matrix product. The first
    # seven features are the same for every tax row; the rest repeat once per
    # tax row.
    smallest_change = float(distribution.TAX_CHANGE_THRESHOLDS_DOLLARS[0])
    receives_cut = change <= -smallest_change
    receives_increase = change >= smallest_change
    shared_features = [baseline_after_tax, *cash, *accrual]
    per_row_features = [
        change,
        jnp.where(receives_cut, change, 0.0),
        receives_cut.astype(jnp.float64),
        jnp.where(receives_increase, change, 0.0),
        receives_increase.astype(jnp.float64),
        *(
            (change <= -threshold).astype(jnp.float64)
            for threshold in distribution.TAX_CHANGE_THRESHOLDS_DOLLARS
        ),
        *(
            (change >= threshold).astype(jnp.float64)
            for threshold in distribution.TAX_CHANGE_THRESHOLDS_DOLLARS
        ),
        rounded_change,
        *(tax_by_row[:, world] for world in range(len(distribution.LEVEL_WORLDS))),
    ]
    n_rows = len(distribution.TAX_ROWS)
    features = jnp.concatenate(
        [
            jnp.stack(shared_features, axis=-1),
            jnp.stack(per_row_features, axis=-1)
            .transpose(1, 0, 2)
            .reshape(weight.shape[0], -1),
        ],
        axis=-1,
    )
    mask_weight = masks.astype(jnp.float64) * weight[None, :]
    totals = mask_weight @ features
    group_weight = jnp.sum(mask_weight, axis=1)
    n_shared = len(shared_features)
    shared = totals[:, :n_shared]
    # [tax row, group, feature]
    by_row = totals[:, n_shared:].reshape(
        group_weight.shape[0], n_rows, len(per_row_features)
    ).transpose(1, 0, 2)

    count = jnp.broadcast_to(group_weight, (n_rows, group_weight.shape[0]))
    after_tax = jnp.broadcast_to(shared[:, 0], count.shape)
    has_record = jnp.any(masks, axis=1)

    def cutoff(values: Array, step: float) -> Array:
        minimum = jnp.min(jnp.where(masks, values[None, :], jnp.inf), axis=1)
        rounded = jnp.round(minimum / step) * step
        return jnp.broadcast_to(jnp.where(has_record, rounded, jnp.nan), count.shape)

    def rounded_mean(total: Array, denominator: Array) -> Array:
        step = distribution.AVERAGE_ROUNDING_DOLLARS
        mean = jnp.where(denominator != 0.0, total / denominator, jnp.nan)
        return jnp.round(mean / step) * step

    def share_of_group(total: Array) -> Array:
        return jnp.where(count != 0.0, total / count, jnp.nan)

    def ratio(numerator: Array, denominator: Array) -> Array:
        return jnp.where(denominator > 0.0, numerator / denominator, jnp.nan)

    # The group totals of the per-row features, in the order they are listed
    # above, each shaped [tax row, group].
    by_feature = list(jnp.moveaxis(by_row, -1, 0))
    n_thresholds = len(distribution.TAX_CHANGE_THRESHOLDS_DOLLARS)
    change_total, cut_total, cut_weight, increase_total, increase_weight = (
        by_feature[:5]
    )
    cut_counts = by_feature[5 : 5 + n_thresholds]
    increase_counts = by_feature[5 + n_thresholds : 5 + 2 * n_thresholds]
    net_change = by_feature[5 + 2 * n_thresholds]
    tax_dollars = by_feature[6 + 2 * n_thresholds :]
    # A tax row's share of net change is over the whole population's net
    # change in that tax.
    overall_net_change = net_change[:, :1]

    columns = [
        cutoff(baseline_cash, distribution.AVERAGE_ROUNDING_DOLLARS),
        cutoff(baseline_accrual, distribution.AVERAGE_ROUNDING_DOLLARS),
        cutoff(agi, distribution.AVERAGE_ROUNDING_DOLLARS),
        cutoff(net_worth, 1.0),
        count,
        after_tax,
        rounded_mean(change_total, count),
        rounded_mean(cut_total, cut_weight),
        rounded_mean(increase_total, increase_weight),
        # After-tax income in the static world less that in the baseline is
        # minus the tax change, since both use baseline pretax income.
        jnp.where(after_tax != 0.0, -change_total / after_tax, jnp.nan),
        *(share_of_group(total) for total in cut_counts),
        *(share_of_group(total) for total in increase_counts),
        net_change,
        jnp.where(overall_net_change != 0.0, net_change / overall_net_change, jnp.nan),
    ]
    n_worlds = len(distribution.LEVEL_WORLDS)
    for world in range(n_worlds):
        cash_dollars = jnp.broadcast_to(shared[:, 1 + world], count.shape)
        accrual_dollars = jnp.broadcast_to(shared[:, 1 + n_worlds + world], count.shape)
        columns += [
            tax_dollars[world],
            cash_dollars,
            accrual_dollars,
            ratio(tax_dollars[world], cash_dollars),
            ratio(tax_dollars[world], accrual_dollars),
        ]
    return jnp.stack(columns, axis=-1)
