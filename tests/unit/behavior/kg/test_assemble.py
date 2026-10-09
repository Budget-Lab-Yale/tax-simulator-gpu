"""The whole capital-gains model on a small made-up population.

Everything from households to the cell table the applier spends runs here, with
nothing standing in for anything: the pre-pass adds the households up, the mix
of death regimes reads the year's law, the schedules are solved on the real
63-cell age grid with the real Social Security life tables, and the stock
recurrence steps five years of it. The population is forty households.

A run solves the choice for a whole group of scenarios at once, so one
scenario is run here as a group of one (``whole_model.run_one_scenario``),
for the frozen pass and the behavioral pass alike.

These read the pinned Macro-Projections publication for the price level and the
Treasury rate, so they skip where it is not mounted.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from taxmodel.behavior.kg import assemble, cell_table_column, prepass
from taxmodel.behavior.kg.inputs import Calibration

from . import whole_model

YEARS = tuple(range(2026, 2031))
N_HOUSEHOLDS = 40

# The five tracked asset classes' death treatments, in order.
STEP_UP_EVERYTHING = (0, 0, 0, 0, 0)
DEEMED_EVERYTHING = (2, 2, 2, 2, 2)
CARRYOVER_EVERYTHING = (1, 1, 1, 1, 1)

# Fixed response parameters, so that a refit does not change what these tests
# assert. The appreciation rate is the settings file's (config/calibrations/kg/settings.yaml).
CALIBRATION = Calibration(
    elasticity=jnp.asarray(0.72),
    timeable_share=jnp.asarray(0.2),
    appreciation=0.0677,
)


def households() -> dict[str, jnp.ndarray]:
    """Forty single filers spread over the age grid, each holding equities.

    Ages run from 25 to 79 so that every household sits in its own cell with
    room above it, and the oldest are the wealthiest, which is the shape the
    real data has.
    """

    age = np.linspace(25, 79, N_HOUSEHOLDS)
    value = np.linspace(50_000.0, 5_000_000.0, N_HOUSEHOLDS)
    frame = {
        "weight": jnp.full(N_HOUSEHOLDS, 1_000.0),
        "age1": jnp.asarray(age),
        "age2": jnp.full(N_HOUSEHOLDS, jnp.nan),
        "filing_status": jnp.ones(N_HOUSEHOLDS),
        # A one-in-fifty chance of dying, rising with age.
        "m_household": jnp.asarray(0.001 + 0.0004 * (age - 25.0)),
        "p_widow": jnp.zeros(N_HOUSEHOLDS),
        "estate_economic_gross": jnp.asarray(2.0 * value),
        # Each household sells a fortieth of what it holds.
        "kg_lt": jnp.asarray(0.025 * value),
    }
    for name in prepass.DEEMED_ASSET_CLASSES:
        frame[f"value.{name}"] = jnp.zeros(N_HOUSEHOLDS)
        frame[f"basis.{name}"] = jnp.zeros(N_HOUSEHOLDS)
    frame["value.equities"] = jnp.asarray(value)
    # Half the market value is basis, so half is unrealized gain.
    frame["basis.equities"] = jnp.asarray(0.5 * value)
    return frame


def run(
    macro_root: Path,
    *,
    reform_rate_on_gains: float,
    regime_codes=STEP_UP_EVERYTHING,
):
    """The gains model for one scenario over the fixture's five years."""

    # Twenty-three cents on a dollar of gain is roughly today's top rate with
    # the investment income tax; the reform's rate is the argument.
    baseline_rate_on_gains = 0.238
    return whole_model.run_one_scenario(
        [households() for _ in YEARS],
        YEARS,
        macro_root=macro_root,
        charity=prepass.read_charitable_bequest_settings(),
        calibration=CALIBRATION,
        law=whole_model.scenario_law(
            len(YEARS), regime_codes=regime_codes
        ),
        prices=whole_model.cell_prices(
            prepass.KG_N_AGES,
            len(YEARS),
            baseline_rate_on_gains=baseline_rate_on_gains,
            reform_rate_on_gains=reform_rate_on_gains,
        ),
    )


def run_frozen(macro_root: Path, *, regime_codes):
    """The frozen pass over the same five years, straight from the model."""

    return whole_model.run_one_scenario(
        [households() for _ in YEARS],
        YEARS,
        macro_root=macro_root,
        charity=prepass.read_charitable_bequest_settings(),
        calibration=CALIBRATION,
        law=whole_model.scenario_law(
            len(YEARS), regime_codes=regime_codes
        ),
        frozen=True,
    )


def test_the_model_runs_from_households_to_a_cell_table_for_every_year(macro_root):
    """Nothing missing in the chain, and no missing numbers coming out."""

    result = run(macro_root, reform_rate_on_gains=0.238)

    assert result.cell_tables.shape[0] == len(YEARS)
    # The model's own grid: 63 cells, ages 18 through 80.
    assert result.cell_tables.shape[1] == prepass.KG_N_AGES
    assert bool(jnp.all(jnp.isfinite(result.cell_tables)))


def test_a_reform_that_changes_nothing_leaves_realizations_alone(macro_root):
    """The reform law is the baseline law, so every cell realizes as before.

    This is the check that the whole chain is consistent with itself: the same
    rate on both legs has to come back as a rate factor of one and no extra
    realizations, whatever the intermediate arrays hold.
    """

    result = run(macro_root, reform_rate_on_gains=0.238)

    rate_factor = cell_table_column(result.cell_tables, "rate_factor")
    extra = cell_table_column(result.cell_tables, "extra_realizations")
    stock = cell_table_column(result.cell_tables, "baseline_stock")

    holds_gain = stock > 0
    assert float(jnp.max(jnp.abs(jnp.where(holds_gain, rate_factor - 1.0, 0.0)))) < 1e-9
    assert float(jnp.max(jnp.abs(extra))) < 1e-6


def test_a_higher_rate_on_gains_makes_cells_sell_less(macro_root):
    """Lock-in: taxing a realization more heavily delays it.

    Ten points on the rate has to reduce the share of its gain a cell turns
    over.
    """

    unchanged = run(macro_root, reform_rate_on_gains=0.238)
    higher = run(macro_root, reform_rate_on_gains=0.338)

    def turnover(result):
        stock = cell_table_column(result.cell_tables, "baseline_stock")
        rate = cell_table_column(result.cell_tables, "scenario_realization_rate")
        return jnp.sum(rate * stock) / jnp.sum(stock)

    assert float(turnover(higher)) < float(turnover(unchanged))


def test_taxing_gains_at_death_makes_cells_sell_more(macro_root):
    """Deemed realization at death takes the reward for waiting away.

    Under a stepped-up basis, holding until death is how a gain escapes tax
    entirely; once death is a taxable event, that reason to wait is gone and
    realizations during life rise.
    """

    stepped_up = run(macro_root, reform_rate_on_gains=0.238)
    deemed = run(
        macro_root, reform_rate_on_gains=0.238, regime_codes=DEEMED_EVERYTHING
    )

    def realizations(result):
        stock = cell_table_column(result.cell_tables, "baseline_stock")
        rate = cell_table_column(result.cell_tables, "scenario_realization_rate")
        return float(jnp.sum(rate * stock))

    assert realizations(deemed) > realizations(stepped_up)


def test_carryover_basis_moves_a_decedents_gain_into_the_heirs_cells(macro_root):
    """The stock leaves the old and arrives at the young.

    Under a stepped-up basis a decedent's gain vanishes, so no cell gains
    anything from a death. Under carryover it is inherited, and the cells the
    heirs are in end the year holding more than the baseline says.
    """

    stepped_up = run(macro_root, reform_rate_on_gains=0.238)
    carryover = run(
        macro_root, reform_rate_on_gains=0.238, regime_codes=CARRYOVER_EVERYTHING
    )

    arriving = cell_table_column(
        carryover.cell_tables, "stock_change_from_inheritance"
    )
    none_arriving = cell_table_column(
        stepped_up.cell_tables, "stock_change_from_inheritance"
    )

    assert float(jnp.sum(arriving)) > 0.0
    assert float(jnp.sum(jnp.abs(none_arriving))) == 0.0


def test_the_frozen_pass_leaves_turnover_at_baseline(macro_root):
    """It is the pure law arithmetic of gains at death, and nothing else.

    The static and mechanical worlds read this pass: the routing of gains to
    heirs stays live, but no behavioral choice is solved, so the rate factor
    the applier spends is exactly one.
    """

    frozen = run_frozen(macro_root, regime_codes=CARRYOVER_EVERYTHING)

    rate_factor = cell_table_column(frozen.cell_tables, "rate_factor")
    stock = cell_table_column(frozen.cell_tables, "baseline_stock")
    holds_gain = stock > 0

    assert float(jnp.max(jnp.abs(jnp.where(holds_gain, rate_factor - 1.0, 0.0)))) < 1e-12
    # Carryover still moves gains to heirs.
    assert (
        float(
            jnp.sum(cell_table_column(frozen.cell_tables, "stock_change_from_inheritance"))
        )
        > 0.0
    )


def test_the_discount_factor_is_the_nominal_ten_year_treasury_rate(macro_root):
    """A taxpayer putting off a tax payment discounts it at a nominal rate.

    The appreciation rate the gain grows at and the basis the gain is measured
    against are both nominal, so the rate that discounts the payment is too.
    """

    factor = assemble.nominal_discount_factor_by_year(macro_root, YEARS)

    assert factor.shape == (len(YEARS),)
    # A nominal ten-year Treasury rate between zero and ten per cent.
    assert bool(np.all((factor > 1.0 / 1.10) & (factor <= 1.0)))


def test_the_mortality_tail_blends_the_two_conventions_by_the_joint_share():
    """Three quarters joint at a 20 percent rate: 0.25(0.2) + 0.75(0.04)."""

    tail = assemble.mortality_above_the_top_code_for_tax_units([[0.2]], [0.75])

    assert float(tail[0, 0]) == pytest.approx(0.25 * 0.2 + 0.75 * 0.04)


def test_the_mortality_tail_is_a_probability_and_keeps_certain_death_certain():
    """The rate stays inside [0, 1], and a rate of one comes through as one.

    The oldest age the life table reaches has to stay certain death whatever
    the joint share, because the selling choice works backward from it.
    """

    q = np.array([[0.0, 0.02], [0.5, 0.9], [1.0, 1.0]])
    for share in (0.0, 0.3, 0.77, 1.0):
        tail = np.asarray(
            assemble.mortality_above_the_top_code_for_tax_units(q, [share, share])
        )
        assert tail.min() >= 0.0
        assert tail.max() <= 1.0
        assert np.array_equal(tail[2], np.ones(2))
