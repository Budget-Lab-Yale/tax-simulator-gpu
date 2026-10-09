"""A law compiled in a batch gives the same answers as the law compiled alone.

The test runs at unit-test scale: three hand-written households over two years, with the laws
compiled from the repository's own configuration tree. For each law it runs the
frame steps and the whole calculator chain twice — once with the law compiled on
its own, once with the same law compiled in a batch that forces it into a wider
layout — and requires the two to agree exactly, column by column.

The three laws are picked because they disagree about shapes. Baseline law has
no wealth tax and states the twelve-bracket estate schedule;
`public/estate_2009` restores the fourteen-bracket 2009 schedule; and
`public/wealth_tax_warren` adds a three-bracket wealth tax stated per filing
status. Compiled together they widen each other in both of the ways a batch can:
a parameter written out per filing status, and a schedule given more columns.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from taxmodel.config.law_layers import resolve_law
from taxmodel.law.batch import compile_batch
from taxmodel.law.compile import compile_resolved
from taxmodel.law.indexes import index_table_from_frame
from taxmodel.scenario import provenance
from taxmodel.taxes import statutory_order
from tests.conftest import CONFIG_ROOT, INDEX_TABLE

# The cache-clearing fixture is autouse in whichever module imports it, which
# is why it is imported by name here rather than written out again.
from .test_integration import (
    YEARS,
    release_compiled_programs_after_each_test,  # noqa: F401
    synthetic_population,
)

LAWS = (None, "public/estate_2009", "public/wealth_tax_warren")


@pytest.fixture(scope="module")
def run_inputs():
    """Everything a year of the model needs besides the law."""
    from taxmodel.frame import build_frame
    from taxmodel.years import one_year

    growth, _source = provenance.index_series(index_table=INDEX_TABLE)
    estate_params, cluster_cap = provenance.read_estate_bridge()
    population = synthetic_population()
    builder = build_frame.make_frame_builder(growth, cluster_cap)
    by_year = one_year.population_arrays_for_years(population, YEARS)
    return {
        "index_table": index_table_from_frame(pd.read_csv(INDEX_TABLE)),
        "estate_params": estate_params,
        "builder": builder,
        "years_arrays": [
            {name: value[i] for name, value in by_year.items()}
            for i in range(len(YEARS))
        ],
    }


def one_year(law, run_inputs, year_index: int):
    """The frame and the tax results for one year under one compiled law."""
    from tests.unit.years.test_law_binding_identity import bind_annual_law

    year = YEARS[year_index]
    arrays = run_inputs["years_arrays"][year_index]
    annual = bind_annual_law(
        law.values,
        law.years,
        provenance.wide_law_table(law),
        law.values,
        law.years,
        year,
    )
    read = annual.for_records(arrays["filing_status"])
    frame = run_inputs["builder"](
        arrays, read, jnp.asarray(float(year)), apply_adjusters=False
    )
    taxes = statutory_order.run_taxes(frame, read.calculator, run_inputs["estate_params"])
    return annual, frame, taxes


@pytest.fixture(scope="module")
def compiled(run_inputs):
    """Each law compiled on its own, and the three compiled as one batch."""
    index_table = run_inputs["index_table"]
    alone = [
        compile_resolved(resolve_law(CONFIG_ROOT, name), YEARS, index_table)
        for name in LAWS
    ]
    batch = compile_batch(
        [resolve_law(CONFIG_ROOT, name) for name in LAWS], YEARS, index_table
    )
    return alone, batch


def test_the_batch_really_does_widen_these_laws(compiled):
    """Otherwise the test below would pass by proving nothing."""
    alone, batch = compiled
    baseline_alone, baseline_batched = alone[0], batch.laws[0]

    # The estate schedule gets the fourteen columns the 2009 law fills, where
    # baseline law on its own fills twelve.
    assert baseline_alone.columns_in_table("estate.brackets") == 12
    assert baseline_batched.columns_in_table("estate.brackets") == 14


def test_every_law_in_the_batch_presents_the_same_shapes(compiled, run_inputs):
    """One program per batch means one set of argument shapes for the batch."""
    _alone, batch = compiled
    signatures = []
    for law in batch.laws:
        for year_index in range(len(YEARS)):
            annual, _frame, _taxes = one_year(law, run_inputs, year_index)
            leaves, structure = jax.tree.flatten(annual)
            signatures.append((structure, tuple(leaf.shape for leaf in leaves)))
    assert len(set(signatures)) == 1


@pytest.mark.parametrize("law_index", range(len(LAWS)))
def test_a_law_gives_the_same_numbers_alone_and_in_a_batch(
    compiled, run_inputs, law_index
):
    alone, batch = compiled
    for year_index in range(len(YEARS)):
        _annual, frame_alone, taxes_alone = one_year(
            alone[law_index], run_inputs, year_index
        )
        _annual, frame_batched, taxes_batched = one_year(
            batch.laws[law_index], run_inputs, year_index
        )
        label = f"{LAWS[law_index] or 'baseline'} in {YEARS[year_index]}"

        assert taxes_alone.keys() == taxes_batched.keys()
        for name, value in taxes_alone.items():
            np.testing.assert_array_equal(
                np.asarray(taxes_batched[name]),
                np.asarray(value),
                err_msg=f"{label}: {name}",
            )

        # The frame carries the joined law columns, so the batch changes the
        # frame's columns as well as its shapes. A schedule the batch widens
        # gains columns; a single-valued parameter the batch widens has its
        # column numbered, which is a rename. Both are followed here, and every
        # column that ends up in both frames has to hold the same numbers.
        renamed = law_column_renames(alone[law_index], batch.laws[law_index])
        for name, value in frame_alone.items():
            in_the_batch = renamed.get(name, name)
            np.testing.assert_array_equal(
                np.asarray(frame_batched[in_the_batch]),
                np.asarray(value),
                err_msg=f"{label}: {name}",
            )
        added = set(frame_batched) - {renamed.get(n, n) for n in frame_alone}
        for name in added:
            assert np.all(np.isnan(np.asarray(frame_batched[name]))), (
                f"{label}: the batch added law column {name}, which is not a "
                "bracket this law leaves empty"
            )


def law_column_renames(alone, batched) -> dict[str, str]:
    """What each of a law's own table columns is called in the batch's table.

    A field filling one slot has a column with no number on it and a field
    filling more has numbered columns, so a batch that widens a single-valued
    field renames its column rather than adding one. Nothing in the frame layer
    reads a renamed column — `law/batch.py` refuses the batch outright for the
    five law columns the frame steps do read by name — but a test comparing the
    two frames has to know which column is which.
    """
    renames: dict[str, str] = {}
    for field in alone.schema:
        before = _table_columns(field, alone.columns_in_table(field))
        after = _table_columns(field, batched.columns_in_table(field))
        for old, new in zip(before, after, strict=False):
            if old != new:
                renames[old] = new
    return renames


def _table_columns(field: str, slots: int) -> list[str]:
    if slots <= 1:
        return [field]
    return [f"{field}{i + 1}" for i in range(slots)]
