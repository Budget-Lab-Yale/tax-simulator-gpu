"""Dependent filers are left out of the capital-gains model's age cells.

A dependent filer has no balance sheet on the file, so it holds no unrealized
gain, but it can report realized gains. Counted in its cell, those realizations
would raise the cell's observed turnover with nothing behind them in the stock.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from taxmodel.behavior.kg import prepass
from taxmodel.behavior.kg.applier import LOCKIN_COLUMN, apply_cell_table

from .test_applier import cell_table, households  # noqa: F401  (fixtures)
from .test_prepass import CHARITY
from .test_prepass import households as prepass_households


def test_a_dependent_filers_realized_gains_stay_out_of_its_cell():
    """An 18-year-old holding $100 of gain and realizing $5, and a dependent
    filer aged 16 (also in the youngest cell) realizing $500 with no balance
    sheet. The cell holds $100 and realizes $5, a turnover of 0.05; counting
    the dependent would make it 5.05."""

    frame = prepass_households(
        age1=[18.0, 16.0],
        weight=[1.0, 1.0],
        kg_lt=[5.0, 500.0],
        dep_status=[0.0, 1.0],
        **{"value.equities": [100.0, 0.0]},
    )
    cells = prepass.population_age_cells(
        frame, deflator_to_base_year=1.0, charity=CHARITY, n_ages=63
    )

    assert float(cells.stock[0]) == pytest.approx(100.0)
    assert float(cells.realizations[0]) == pytest.approx(5.0)
    assert float(cells.turnover[0]) == pytest.approx(0.05)

    weights = prepass.cell_weights_from_records(frame, n_ages=63)
    assert float(weights.weighted_realizations[1]) == 0.0


def test_a_frame_without_the_dependent_column_counts_everyone():
    frame = prepass_households(age1=[18.0], kg_lt=[5.0], **{"value.equities": [100.0]})
    cells = prepass.population_age_cells(
        frame, deflator_to_base_year=1.0, charity=CHARITY, n_ages=63
    )
    assert float(cells.realizations[0]) == pytest.approx(5.0)


def test_the_cells_answers_are_not_spread_to_a_dependent_filer(cell_table, households):  # noqa: F811
    """The first household of the applier's fixture, made a dependent filer,
    keeps its $60,000 of realized gains (its cell's rate factor is 1.2) and
    takes none of the cell's $24,000 of extra realizations."""

    households = {**households, "dep_status": jnp.asarray([1, 0, 0, 0])}
    changed = apply_cell_table(households, {}, {}, {"cell_table": cell_table})

    assert float(changed["kg_lt"][0]) == pytest.approx(60_000.0)
    assert float(changed[LOCKIN_COLUMN][0]) == 0.0
    # The other household in the cell is spread to as before.
    assert float(changed[LOCKIN_COLUMN][1]) == pytest.approx(12_000.0)
