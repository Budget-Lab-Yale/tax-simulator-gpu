"""Unit tests for the pure pieces of the above-threshold income table."""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "projects" / "top_tax"))
import above_threshold_income as table


def test_bisection_brings_each_record_to_the_threshold():
    # Each record's income is its factor times its full income plus a fixed
    # part that does not scale, standing in for a loss.
    full = jnp.array([200.0, 150.0, 1000.0, 50.0])
    fixed = jnp.array([0.0, -20.0, 10.0, 0.0])
    threshold = 100.0

    def income_at(factor):
        return full * factor + fixed

    active = jnp.array([True, True, True, False])
    unreachable = jnp.array([False, False, False, False])
    factor, _ = table.bisect_scale_factors(income_at, active, unreachable, threshold)
    expected = (threshold - fixed[:3]) / full[:3]
    np.testing.assert_allclose(factor[:3], expected, rtol=1e-6)
    # A record below the threshold keeps all its income.
    assert float(factor[3]) == 1.0


def test_unreachable_records_get_a_factor_of_zero():
    active = jnp.array([True, False])
    unreachable = jnp.array([False, True])
    factor, _ = table.bisect_scale_factors(
        lambda f: 300.0 * f + jnp.array([0.0, 500.0]), active, unreachable, 100.0
    )
    assert float(factor[1]) == 0.0
    assert float(factor[0]) == pytest.approx(1.0 / 3.0, rel=1e-6)


def test_threshold_is_the_last_unit_inside_the_top_share():
    income = np.array([10.0, 50.0, 40.0, 30.0, 20.0, 1000.0])
    weight = np.ones(6)
    # The unit with income 1000 is not eligible (a dependent).
    eligible = np.array([True, True, True, True, True, False])
    # Top 60 percent of five units: the cumulative weights 1 and 2 are below
    # 3.0, so the second-highest eligible income is the threshold.
    assert table.threshold_for_share(income, weight, eligible, 0.6) == 40.0
    # A share smaller than one unit falls back to the highest income.
    assert table.threshold_for_share(income, weight, eligible, 0.01) == 50.0


def test_accrual_scaling_adds_the_accrued_gains():
    assert set(table.scaled_columns("accrual")) - set(table.scaled_columns("agi")) == set(
        table.ACCRUAL_COLUMNS
    )


def test_a_missing_column_is_refused():
    with pytest.raises(KeyError, match="wages_base1"):
        table.check_frame_columns(["weight"])
