"""The annual simulation's timing and independent wealth accounts."""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np

from taxmodel.scenario import law_by_year
from taxmodel.years import wealth_arithmetic, wealth_state, worlds
from tests.unit.years.one_scenario import (
    passthrough_frame_builder,
    taxes_then_prices,
    unchanged_frame,
)

N_AGES = len(wealth_arithmetic.age_grid())
N_BINS = 2
REFORM_LAW = jnp.asarray(1.0)
BASELINE_LAW = jnp.asarray(0.0)


@dataclass(frozen=True)
class GainsLaw:
    """The rate and corporate factor the gains-margin test needs."""

    calculator: jnp.ndarray
    corporate: law_by_year.CorporateYearFactors


def records(*, age: float = 40.0, assets: float = 100.0):
    return {
        "weight": jnp.array([1.0]),
        "age1": jnp.array([age]),
        "age2": jnp.array([0.0]),
        "filing_status": jnp.array([1.0]),
        "dep_status": jnp.array([0.0]),
        "income": jnp.array([0.0]),
        "net_worth": jnp.array([assets]),
        "value.cash": jnp.array([assets]),
    }


def taxes(frame, law):
    base = jnp.asarray(frame["income"]) + jnp.asarray(frame["value.cash"])
    liability = jnp.asarray(law) * base
    zero = jnp.zeros_like(liability)
    return {
        "liab_iit_net": liability,
        "liab_pr": zero,
        "liab_wealth": zero,
    }


def prices(frame, law):
    """A stand-in price calculator: one named rate, so the wiring is visible."""
    del law
    return {"mtr_wages_base1": jnp.zeros_like(frame["weight"])}


def profile(saving_share: float = 0.5):
    return wealth_state.WealthProfileYear(
        saving_shares=jnp.full((N_AGES, N_BINS), saving_share),
        transition=jnp.eye(N_BINS),
        r_total=jnp.asarray(0.0),
        ages=jnp.asarray(wealth_arithmetic.age_grid()),
        aging=wealth_arithmetic.build_aging_matrix(N_AGES),
        fmax=jnp.asarray(0.9),
        pass_through_weight=jnp.asarray(0.2),
    )


def zero_state():
    zero = jnp.zeros((N_AGES, N_BINS))
    return worlds.SimulationState(zero, zero)


def run_one(raw, state):
    return worlds.annual_step(
        raw,
        REFORM_LAW,
        BASELINE_LAW,
        taxes(raw, BASELINE_LAW),
        state,
        profile(),
        jnp.asarray(2026.0),
        passthrough_frame_builder,
        taxes,
        prices,
        unchanged_frame,
        taxes_then_prices(taxes, prices),
    )


def test_conventional_sweeps_state_is_independent():
    raw = records()
    row = 40 - wealth_arithmetic.WEALTH_DYN_AGE_MIN
    occupied_bin = N_BINS - 1
    changed = jnp.zeros((N_AGES, N_BINS)).at[row, occupied_bin].set(20.0)
    zero = jnp.zeros_like(changed)

    _, all_zero = run_one(raw, worlds.SimulationState(zero, zero))
    _, mechanical_changed = run_one(raw, worlds.SimulationState(changed, zero))
    _, conventional_changed = run_one(raw, worlds.SimulationState(zero, changed))

    np.testing.assert_array_equal(
        mechanical_changed.conventional["liab_iit_net"],
        all_zero.conventional["liab_iit_net"],
    )
    np.testing.assert_array_equal(
        conventional_changed.mechanical["liab_iit_net"],
        all_zero.mechanical["liab_iit_net"],
    )
    assert not np.array_equal(
        mechanical_changed.mechanical["liab_iit_net"],
        all_zero.mechanical["liab_iit_net"],
    )
    assert not np.array_equal(
        conventional_changed.conventional["liab_iit_net"],
        all_zero.conventional["liab_iit_net"],
    )


def test_state_handoff_makes_year_t_burden_change_only_year_t_plus_one_taxes():
    raw = records(assets=100.0)
    state_after_2026, result_2026 = run_one(raw, zero_state())

    # Baseline tax is 0 and reform tax is 100.  With s = 0.5, the 2026 burden
    # therefore puts 0.5 * (100 - 0) = 50 into the cell at year end.  It cannot
    # change 2026 tax.  The inflow is recorded one age up, because the
    # household is a year older when the shortfall lands.  So the household enters 2027 aged 41,
    # meets its own $50 shortfall there, cash falls from 100 to 50, and reform
    # tax falls from 100 to 50.
    raw_2027 = dict(raw)
    raw_2027["age1"] = raw["age1"] + 1.0
    _, result_2027 = worlds.annual_step(
        raw_2027,
        REFORM_LAW,
        BASELINE_LAW,
        taxes(raw_2027, BASELINE_LAW),
        state_after_2026,
        profile(),
        jnp.asarray(2027.0),
        passthrough_frame_builder,
        taxes,
        prices,
        unchanged_frame,
        taxes_then_prices(taxes, prices),
    )

    np.testing.assert_allclose(result_2026.mechanical["liab_iit_net"], [100.0])
    np.testing.assert_allclose(result_2027.mechanical["liab_iit_net"], [50.0])


def test_corporate_sale_volume_scales_gains_only_after_the_gains_applier():
    raw = {
        **records(),
        "kg_lt": jnp.array([100.0]),
        "kg_st": jnp.array([20.0]),
        "kg_lt_basis": jnp.array([40.0]),
    }
    corporate = replace(
        law_by_year.CorporateYearFactors.identity(),
        gain_quantity_factor=jnp.asarray(0.9),
    )
    reform_law = GainsLaw(calculator=jnp.asarray(1.0), corporate=corporate)
    baseline_law = GainsLaw(calculator=jnp.asarray(0.0), corporate=corporate)

    def taxes_with_gains(frame, law):
        base = frame["value.cash"] + frame["kg_lt"] + frame["kg_st"]
        zero = jnp.zeros_like(base)
        return {
            "liab_iit_net": law.calculator * base,
            "liab_pr": zero,
            "liab_wealth": zero,
        }

    def gains_applier(frame, *_unused, **_also_unused):
        written = dict(frame)
        written["kg_lt"] = written["kg_lt"] + 20.0
        written["kg_st"] = written["kg_st"] + 10.0
        written["kg_lt_basis"] = written["kg_lt_basis"] + 30.0
        # This is the separately priced gain from death, not a sale.
        written["kg_deemed_full"] = jnp.array([80.0])
        return written

    _, result = worlds.annual_step(
        raw,
        reform_law,
        baseline_law,
        taxes_with_gains(raw, baseline_law),
        zero_state(),
        profile(),
        jnp.asarray(2026.0),
        passthrough_frame_builder,
        taxes_with_gains,
        prices,
        behavior_builder=gains_applier,
        taxes_and_prices=taxes_then_prices(taxes_with_gains, prices),
        frozen_gains_builder=gains_applier,
        runs_the_gains_model=True,
    )

    # The first-order world receives the gains applier's law arithmetic, but
    # not the corporate transmission that changes the number of shares sold.
    np.testing.assert_allclose(result.raw_frame["kg_lt"], [120.0])
    np.testing.assert_allclose(result.raw_frame["kg_st"], [30.0])
    np.testing.assert_allclose(result.raw_frame["kg_lt_basis"], [70.0])

    for frame in (result.mechanical_frame, result.behavior_frame):
        np.testing.assert_allclose(frame["kg_lt"], [108.0])
        np.testing.assert_allclose(frame["kg_st"], [27.0])
        np.testing.assert_allclose(frame["kg_lt_basis"], [63.0])
        np.testing.assert_allclose(frame["kg_deemed_full"], [80.0])


@dataclass(frozen=True)
class TwoCorporatePathsLaw:
    """A law carrying both corporate incidence paths, as a scenario's does."""

    calculator: jnp.ndarray
    corporate: law_by_year.CorporateYearFactors
    corporate_mechanical: law_by_year.CorporateYearFactors


def test_each_world_reads_its_own_corporate_incidence_path():
    """The mechanical worlds take the fixed-base markdown, the conventional
    world the eroded-base one, and the first-order world neither."""
    raw = records(assets=100.0)

    def marked_down(markdown):
        return replace(
            law_by_year.CorporateYearFactors.identity(),
            markdown_rate=jnp.asarray(markdown),
        )

    reform_law = TwoCorporatePathsLaw(
        calculator=jnp.asarray(1.0),
        corporate=marked_down(0.1),
        corporate_mechanical=marked_down(0.2),
    )
    baseline_law = TwoCorporatePathsLaw(
        calculator=jnp.asarray(0.0),
        corporate=marked_down(0.0),
        corporate_mechanical=marked_down(0.0),
    )

    def frame_builder(raw_arrays, law, year, *, apply_adjusters):
        del year
        frame = dict(raw_arrays)
        if apply_adjusters:
            frame["value.cash"] = frame["value.cash"] * (
                1.0 - law.corporate.markdown_rate
            )
        return frame

    def taxes_on_cash(frame, law):
        liability = law.calculator * frame["value.cash"]
        zero = jnp.zeros_like(liability)
        return {"liab_iit_net": liability, "liab_pr": zero, "liab_wealth": zero}

    _, result = worlds.annual_step(
        raw,
        reform_law,
        baseline_law,
        taxes_on_cash(raw, baseline_law),
        zero_state(),
        profile(),
        jnp.asarray(2026.0),
        frame_builder,
        taxes_on_cash,
        prices,
        unchanged_frame,
        taxes_then_prices(taxes_on_cash, prices),
    )

    np.testing.assert_allclose(result.first_order["liab_iit_net"], [100.0])
    np.testing.assert_allclose(result.mechanical_ex_saving["liab_iit_net"], [80.0])
    np.testing.assert_allclose(result.mechanical["liab_iit_net"], [80.0])
    np.testing.assert_allclose(result.conventional["liab_iit_net"], [90.0])


def test_the_mechanical_world_measures_gains_at_death_after_the_drawdown():
    """The mechanical world's frozen gains pass sees the drawn-down balance
    sheet; the mechanical-ex-saving world's sees the balance sheet before it."""
    raw = records(assets=100.0)
    row = 40 - wealth_arithmetic.WEALTH_DYN_AGE_MIN
    shortfall = jnp.zeros((N_AGES, N_BINS)).at[row, N_BINS - 1].set(20.0)

    def frozen_gains(frame, *_unused, **_also_unused):
        # A stand-in for the frozen gains pass: it records the asset value the
        # gain at death would be measured on.
        return {**frame, "value_at_death": frame["value.cash"]}

    _, result = worlds.annual_step(
        raw,
        REFORM_LAW,
        BASELINE_LAW,
        taxes(raw, BASELINE_LAW),
        worlds.SimulationState(shortfall, jnp.zeros_like(shortfall)),
        profile(),
        jnp.asarray(2026.0),
        passthrough_frame_builder,
        taxes,
        prices,
        unchanged_frame,
        taxes_then_prices(taxes, prices),
        frozen_gains_builder=frozen_gains,
    )

    np.testing.assert_allclose(result.flows_frame["value_at_death"], [100.0])
    np.testing.assert_allclose(
        result.mechanical_frame["value_at_death"], result.mechanical_frame["value.cash"]
    )
    assert float(result.mechanical_frame["value_at_death"][0]) < 100.0
