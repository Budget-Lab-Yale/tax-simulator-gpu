"""The production path and the compiled chain.

The path from raw Tax-Data to results cannot be tested end to end off the
cluster -- Tax-Data is not in the repo and never will be. What is tested here
is everything either side of the file read: that the run-level inputs load and
have the shapes the frame steps and the estate calculator want, that the frame
steps and the chain agree on the columns between them, and that compiling the
chain changes no number.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from taxmodel.frame import steps_jax
from taxmodel.scenario import law_by_year, provenance
from taxmodel.taxes import law_for_calculators, statutory_order
from tests.conftest import INDEX_TABLE

from .conftest import YEAR, compiled_calculators, synthetic_records

YEARS = range(2025, 2037)


@pytest.fixture(scope="module")
def scenario():
    return provenance.load_scenario(YEARS, index_table=INDEX_TABLE)


def test_the_estate_bridge_gives_the_calculator_and_the_frame_what_each_needs():
    params, cluster_cap = provenance.read_estate_bridge()
    # The frame's mortality step caps the expected deaths in one cluster of
    # replicated donor records.
    assert cluster_cap > 0
    assert 0 < float(params.reporting_rate) <= 1
    assert 0 < float(params.pass_through_valuation_rate) <= 1
    # Four SOI size bins, in rising order of reported gross estate.
    bounds = np.asarray(params.bin_lower_bounds)
    assert bounds.shape == (4,)
    assert (np.diff(bounds) > 0).all()


def test_the_wide_law_table_is_what_the_frame_join_expects(scenario):
    law = scenario.law_table
    # One row per year and filing status, keyed the way the frame joins.
    assert len(law) == len(list(YEARS)) * 4
    assert set(law["filing_status"]) == {1, 2, 3, 4}
    # Schedule columns carry a trailing slot number.
    assert "ord.rates1" in law.columns
    assert "item.mort_bal_limit_years3" in law.columns
    # A single-valued parameter carries no number.
    assert "std.value" in law.columns


def test_the_scenario_carries_the_law_in_both_shapes_for_the_same_years(scenario):
    assert scenario.years == tuple(YEARS)
    # The 2027 single-filer standard deduction, read off the compiled arrays
    # and off the wide table, is the same number.
    from_arrays = scenario.values["std.value"][scenario.years.index(YEAR), 0, 0]
    row = scenario.law_table.query("year == @YEAR and filing_status == 1")
    assert float(row["std.value"].iloc[0]) == pytest.approx(float(from_arrays))


def test_the_index_table_covers_the_run_years(scenario):
    # The frame's capital-gain basis ratio needs a CPI level for every run
    # year, spliced from CPI-U and chained CPI-U.
    years, _levels = law_by_year.cpi_level_series(scenario.indexes)
    for year in YEARS:
        assert year in years


def test_the_frame_steps_and_the_chain_agree_on_the_columns_between_them(scenario):
    # The chain reads three columns no Tax-Data file has; the frame steps
    # make them.
    made_by_the_steps = {
        "kg_lt_cpi_ratio",
        "salt_workaround_part",
        "salt_workaround_scorp",
    }
    assert made_by_the_steps <= set(statutory_order.RECORD_COLUMNS)
    assert hasattr(steps_jax, "build_world_frame")


def test_the_law_objects_are_jax_containers(baseline_law):
    # Registered as pytrees, so the law can be an argument to a compiled or
    # differentiated function instead of a constant baked into the graph.
    leaves = jax.tree_util.tree_leaves(baseline_law)
    assert len(leaves) > 200
    # The year is metadata, not a leaf: the capital-gain step compares a
    # purchase year against it while the graph is being built.
    assert all(not isinstance(leaf, int) for leaf in leaves)
    rebuilt = jax.tree_util.tree_unflatten(
        jax.tree_util.tree_structure(baseline_law), leaves
    )
    assert rebuilt.year == baseline_law.year
    np.testing.assert_array_equal(
        np.asarray(rebuilt.tax.ord_rates), np.asarray(baseline_law.tax.ord_rates)
    )


def test_a_law_can_be_differentiated_through(baseline_law):
    # The reason for registering the law: a derivative of tax with respect to
    # a tax parameter, which the R model can only get by re-running.
    records = synthetic_records([1], wages_base1=90_000.0)
    single = jax.tree_util.tree_map(
        lambda x: x[0] if getattr(x, "ndim", 0) and x.shape[0] == 4 else x,
        baseline_law,
    )

    def tax_of_standard_deduction(value):
        law = replace(single, std_ded=replace(single.std_ded, value=value))
        return jnp.sum(statutory_order.run_1040(_ready(records, law), law)["liab"])

    slope = jax.grad(tax_of_standard_deduction)(jnp.asarray(16_500.0))
    # A dollar more standard deduction is a dollar less taxable income, taxed
    # at this household's marginal rate, so the derivative is minus that rate.
    assert -0.5 < float(slope) < 0.0


def _ready(records, law):
    """The frame as the 1040 receives it: derived columns and payroll first."""
    from taxmodel.taxes.calc.payroll import calc_pr

    frame = dict(records)
    frame |= statutory_order.derive_vars(frame)
    frame |= calc_pr(frame, law.payroll)
    return frame


def test_the_chain_compiles_as_it_stands(compiled_baseline):
    values, years = compiled_baseline
    filing_status = np.array([1, 2])
    law = law_for_calculators.bind_year(values, years, YEAR).for_records(filing_status)
    records = synthetic_records(filing_status, wages_base1=50_000.0)
    taxes, _prices, wrap = compiled_calculators(law, None)
    taxes(records, wrap(law))


def test_compiling_the_chain_changes_no_number(compiled_baseline):
    values, years = compiled_baseline
    filing_status = np.array([1, 2, 3, 4])
    law = law_for_calculators.bind_year(values, years, YEAR).for_records(filing_status)
    records = synthetic_records(
        filing_status,
        wages_base1=jnp.array([40_000.0, 200_000.0, 12_000.0, 85_000.0]),
        wages_base2=jnp.array([0.0, 50_000.0, 0.0, 0.0]),
        n_dep_eitc=jnp.array([0.0, 0.0, 1.0, 2.0]),
        char_cash=jnp.array([0.0, 40_000.0, 0.0, 0.0]),
    )
    eager = statutory_order.run_taxes(records, law)
    taxes, _prices, wrap = compiled_calculators(law, None)
    compiled = taxes(records, wrap(law))
    for name, value in eager.items():
        np.testing.assert_allclose(
            np.asarray(compiled[name]), np.asarray(value), err_msg=name
        )


def test_the_compiled_chain_is_reused_across_years(compiled_baseline):
    # One compiled graph serves every year, because the law goes in as an
    # argument. If it did not, each year would pay the compile again.
    values, years = compiled_baseline
    filing_status = np.array([1, 2])
    records = synthetic_records(filing_status, wages_base1=80_000.0)
    taxes, _prices, wrap = compiled_calculators(
        law_for_calculators.bind_year(values, years, 2025).for_records(filing_status), None
    )
    liabilities = []
    for year in (2025, 2026, 2027):
        law = law_for_calculators.bind_year(values, years, year).for_records(filing_status)
        liabilities.append(float(taxes(records, wrap(law))["liab"][0]))
    # Inflation-indexed brackets mean the same wage owes a little less each
    # year, which is also proof the law argument is being read and not frozen
    # into the graph.
    assert liabilities[0] > liabilities[1] > liabilities[2]


# ---------------------------------------------------------------------------
# Marginal tax rates under the two laws a run compares
# ---------------------------------------------------------------------------

# A reform that replaces the whole ordinary rate schedule from 2026 on, so
# every household's price on a dollar of wages moves and the comparison has
# something in it.
REFORM = "tests/clinton_rates"


@pytest.fixture(scope="module")
def reform_scenario():
    return provenance.load_scenario(YEARS, index_table=INDEX_TABLE, alternative=REFORM)


def test_a_baseline_run_names_no_alternative(scenario):
    assert scenario.alternative is None


def test_a_reform_run_differs_from_baseline_only_where_the_reform_says(
    reform_scenario, scenario
):
    assert reform_scenario.alternative == REFORM
    assert reform_scenario.years == scenario.years
    # The reform changes the ordinary rate schedule and nothing else.
    differing = [
        name
        for name in scenario.values
        if not np.array_equal(
            reform_scenario.values[name], scenario.values[name], equal_nan=True
        )
    ]
    assert differing == ["ord.brackets", "ord.rates"]
