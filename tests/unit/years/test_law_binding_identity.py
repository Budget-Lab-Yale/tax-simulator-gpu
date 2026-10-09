"""Binding a whole projection at once gives the arrays binding year by year gave.

The law preparation a run does before anything reaches the card used to bind
each year on its own and stack the thirty results. It now reads each parameter
across the whole year axis in one array operation. The two have to produce the
same arrays down to the dtype, because those arrays are arguments to the
compiled program and a different dtype or shape is a different program.

The reference path below is the old code, moved here whole: ``bind_annual_law``
binds each year on its own and ``bind_year_by_year`` stacks the years with
``stacking.stack_lanes``. It lives in this test file rather than in the model so
that the model has one way of doing this and the test still has something to
compare against. Two other tests borrow the one-year functions from here for
the same reason.

This runs on the repository's own configuration tree — baseline law and the
`tests/top_rate_396` reform, over the full 2026-2055 window a production run
uses — because the point is the real law, not a fixture. It is host-side NumPy
work and takes a few tens of seconds.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from taxmodel.config import law_layers
from taxmodel.frame import steps_jax
from taxmodel.law.batch import compile_batch
from taxmodel.law.indexes import index_table_from_frame
from taxmodel.scenario import prepare, provenance
from taxmodel.scenario.law_by_year import AnnualLaw
from taxmodel.taxes import law_for_calculators
from taxmodel.years.stacking import stack_lanes
from tests.conftest import CONFIG_ROOT, INDEX_TABLE

# The window a production run projects over, from the runscript
# config/runscripts/tests/perf_reporting_30yr.csv.
YEARS = tuple(range(2026, 2056))
REFORM = "tests/top_rate_396"


@pytest.fixture(scope="module")
def scenarios():
    """The reform and baseline scenarios, compiled together as a run compiles them."""
    resolved = [
        law_layers.resolve_law_stack(CONFIG_ROOT, (REFORM,))[-1],
        law_layers.resolve_law_stack(CONFIG_ROOT, ())[0],
    ]
    compiled = compile_batch(
        resolved, YEARS, index_table_from_frame(pd.read_csv(INDEX_TABLE))
    ).laws
    reform = provenance.load_scenario(
        YEARS, CONFIG_ROOT, INDEX_TABLE, REFORM, compiled_law=compiled[0]
    )
    baseline = provenance.load_scenario(
        YEARS, CONFIG_ROOT, INDEX_TABLE, None, compiled_law=compiled[1]
    )
    return reform, baseline


def annual_frame_law_columns(law_table: pd.DataFrame, year: int) -> dict[str, jnp.ndarray]:
    """One year of the wide law table as the array join wants it.

    The wide table has one row per year and filing status and one column per
    law parameter, which is the shape the R model joins onto records. Every
    column comes over as its four rows in filing-status order, which is what
    ``steps_jax.join_law`` gathers with. A column whose four values are the
    same is four copies of one number, and every law presents the same shapes
    for that reason.
    """
    rows = law_table[law_table["year"] == int(year)].sort_values("filing_status")
    statuses = rows["filing_status"].tolist()
    expected = list(range(1, steps_jax.N_FILING_STATUSES + 1))
    if statuses != expected:
        raise ValueError(
            f"the {year} law table covers filing statuses {statuses}, expected "
            f"{expected}"
        )
    return {
        name: jnp.asarray(rows[name].to_numpy(dtype=np.float64))
        for name in rows.columns
        if name not in ("year", "filing_status")
    }


def bind_annual_law(
    values: Mapping[str, np.ndarray],
    law_years: Sequence[int],
    law_table: pd.DataFrame,
    baseline_values: Mapping[str, np.ndarray],
    baseline_law_years: Sequence[int],
    year: int,
) -> AnnualLaw:
    """Bind one year of one law, with baseline payroll law alongside it.

    Nothing here depends on the population: every law parameter comes over as
    its four filing-status values, and the lookup that gives each household its
    own status's value happens inside the compiled program.
    """
    calculator = law_for_calculators.bind_year(values, law_years, year)
    if values is baseline_values:
        baseline_payroll = calculator.payroll
    else:
        baseline_payroll = law_for_calculators.bind_year(
            baseline_values, baseline_law_years, year
        ).payroll
    return AnnualLaw(
        calculator=calculator,
        frame_columns=annual_frame_law_columns(law_table, year),
        baseline_payroll=baseline_payroll,
    )


def bind_year_by_year(scenario, baseline_scenario):
    """The old path: bind each year on its own, then stack the years."""
    reform_years = []
    baseline_years = []
    for year in YEARS:
        reform_years.append(
            bind_annual_law(
                scenario.values,
                scenario.years,
                scenario.law_table,
                baseline_scenario.values,
                baseline_scenario.years,
                int(year),
            )
        )
        baseline_years.append(
            bind_annual_law(
                baseline_scenario.values,
                baseline_scenario.years,
                baseline_scenario.law_table,
                baseline_scenario.values,
                baseline_scenario.years,
                int(year),
            )
        )
    return (
        stack_lanes(reform_years),
        stack_lanes(baseline_years),
    )


def assert_same_arrays(old, new, what: str) -> None:
    """Every leaf equal value for value, with the same dtype and shape."""
    old_leaves, old_shape = jax.tree.flatten(old)
    new_leaves, new_shape = jax.tree.flatten(new)
    assert old_shape == new_shape, f"{what}: the law objects are shaped differently"
    assert len(old_leaves) == len(new_leaves)
    for index, (was, now) in enumerate(zip(old_leaves, new_leaves, strict=True)):
        was = np.asarray(was)
        now = np.asarray(now)
        assert was.dtype == now.dtype, f"{what}: leaf {index} changed dtype"
        assert was.shape == now.shape, f"{what}: leaf {index} changed shape"
        # NaN marks an unused bracket slot and has to stay exactly where it was,
        # so the comparison counts two NaNs in the same place as equal.
        assert np.array_equal(was, now, equal_nan=True), (
            f"{what}: leaf {index} changed value"
        )


def test_the_whole_projection_binds_to_the_same_arrays(scenarios):
    """Reform and baseline law both come out of the new path unchanged."""
    reform_scenario, baseline_scenario = scenarios
    was_reform, was_baseline = bind_year_by_year(reform_scenario, baseline_scenario)
    now_reform, now_baseline = prepare.stacked_annual_laws(
        reform_scenario, baseline_scenario, YEARS
    )
    assert_same_arrays(was_reform, now_reform.stacked, "the reform's stacked law")
    assert_same_arrays(was_baseline, now_baseline.stacked, "baseline stacked law")


def test_each_year_on_its_own_holds_that_year_of_the_stack(scenarios):
    """The per-year laws the host reads agree with the stack it scans."""
    reform_scenario, baseline_scenario = scenarios
    reform_series, _baseline = prepare.stacked_annual_laws(
        reform_scenario, baseline_scenario, YEARS
    )
    stacked, by_year = reform_series.stacked, reform_series.by_year
    assert len(by_year) == len(YEARS)
    stacked_leaves, _ = jax.tree.flatten(stacked)
    for index in (0, len(YEARS) // 2, len(YEARS) - 1):
        leaves, _ = jax.tree.flatten(by_year[index])
        for whole, one in zip(stacked_leaves, leaves, strict=True):
            assert np.array_equal(
                np.asarray(whole)[index], np.asarray(one), equal_nan=True
            )
