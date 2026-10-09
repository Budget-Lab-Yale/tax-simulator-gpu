"""The compiled distribution calculation against the host specification."""

from __future__ import annotations

import jax
import numpy as np
import pytest

from taxmodel.reporting import distribution, distribution_jax
from tests.unit.reporting.host_ranking import quantile_groups as host_quantile_groups


@pytest.mark.parametrize(
    ("values", "weights", "n_groups"),
    [
        # The second and fourth records land exactly on a half-population cut.
        ([0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0], 2),
        # Ties, a zero-weight record, and unequal weights all retain a rank.
        ([0.0, 10.0, 10.0, 20.0, 30.0], [0.0, 2.0, 1.0, 3.0, 4.0], 5),
        # Negative values are outside the reporting rank but stay in the table.
        ([-20.0, -1.0, 0.0, 5.0, 10.0], [2.0, 0.0, 1.0, 1.0, 2.0], 3),
    ],
)
def test_whole_population_ranking_matches_the_host(values, weights, n_groups):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    ranked = values >= 0.0
    expected = host_quantile_groups(values, weights, n_groups, valid=ranked)

    actual = jax.jit(distribution_jax.quantile_groups, static_argnums=2)(
        values, weights, n_groups, ranked
    )

    np.testing.assert_array_equal(actual, expected)


def test_ranking_returns_no_groups_when_the_population_has_no_weight():
    values = np.array([0.0, 1.0, 2.0])
    weights = np.zeros(3)
    expected = host_quantile_groups(values, weights, 5, valid=values >= 0.0)

    actual = distribution_jax.quantile_groups(values, weights, 5, values >= 0.0)

    np.testing.assert_array_equal(actual, expected)


def test_negative_values_have_their_own_reported_group_and_zero_is_ranked():
    weight = np.array([2.0, 1.0, 4.0])
    amount = np.array([-1.0, 0.0, 10.0])
    rank_labels = distribution_jax.baseline_rank_groups(
        amount, weight, np.ones(3, dtype=bool)
    )
    n_worlds = len(distribution.LEVEL_WORLDS)
    income_by_world = np.broadcast_to(amount, (n_worlds, 3))
    table = distribution_jax.distribution_table(
        weight=weight,
        valid=np.ones(3, dtype=bool),
        age=np.array([40.0, 40.0, 40.0]),
        has_child_under_18=np.zeros(3, dtype=bool),
        agi=amount,
        net_worth=amount,
        agi_rank_groups=rank_labels,
        net_worth_rank_groups=rank_labels,
        pretax_cash_income=income_by_world,
        pretax_accrual_income=income_by_world,
        tax=np.zeros((n_worlds, len(distribution.TAXES), 3)),
    )[distribution.TAX_ROWS.index(distribution.TAX_TOTAL)]
    count_column = distribution.value_column_names().index("n_tax_units")
    income_negative_row = distribution.GROUPS.index(
        (distribution.DIMENSION_INCOME, distribution.LABEL_NEGATIVE_INCOME)
    )
    income_first_quintile_row = distribution.GROUPS.index(
        (distribution.DIMENSION_INCOME, distribution.QUINTILE_LABELS[0])
    )

    assert table[income_negative_row, count_column] == 2.0
    assert table[income_first_quintile_row, count_column] == 1.0
