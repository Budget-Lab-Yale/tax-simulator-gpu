"""Where the capital-gains model plugs into a run, and where it stays out of it.

The thing worth testing hardest is the negative: a scenario whose behavior leg
binds no gains machinery runs once, with no host solve and no runtime tables,
and with every behavioral margin at the values that leave the columns alone.
For a gains scenario, what matters is the shape of the program: the applier
composed as shared code, the scenario's own tables and its own behavioral
numbers riding along as runtime data, and the deemed-death fold present exactly
when some year's law deems.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from taxmodel.behavior import loader as behavior_loader
from taxmodel.behavior import parameters as behavior_parameters
from taxmodel.behavior.config import BehaviorLeg
from taxmodel.behavior.gains import builders
from taxmodel.reporting import run_tables
from taxmodel.run import output_directory, scenarios
from taxmodel.run.plan import plan_run
from taxmodel.years import gains_solve
from tests.unit.run.conftest import RUNSCRIPT_HEADER

YEARS = (2026, 2027)

FIXED_CODE_VERSION = {"commit": "abcdef1234567890", "branch": "test", "dirty": False}
FIXED_START = dt.datetime(2026, 8, 3, 14, 30, 5, tzinfo=dt.UTC)


def _spec(scenario_id="baseline", behavior="default"):
    return SimpleNamespace(id=scenario_id, behavior=behavior)


def _prepared():
    """Only the fields the batch scenario is built out of."""
    return SimpleNamespace(
        switch_fingerprint="charity_double_run=no",
        frame_builder=object(),
        tax_calculator=object(),
        price_calculator=object(),
        taxes_and_prices=None,
        detail_request=None,
        reform_laws=object(),
        baseline_laws=object(),
        wealth_profiles=object(),
        compiled_law=SimpleNamespace(values={}),
        compiled_baseline_law=SimpleNamespace(values={}),
        charity_double_run=False,
        corporate_asset_exposure=None,
        equity_markdown_by_year=None,
        # No national corporate figures: the distribution table reads zeros.
        corporate_dollars=None,
    )


def _provenance(leg):
    return SimpleNamespace(behavior_leg=leg)


NO_GAINS_LEG = BehaviorLeg(name="default")
GAINS_LEG = BehaviorLeg(
    name="kg_dynamics_only",
    kg_dynamics={"realization": "config/calibrations/kg/realization.yaml"},
)
# The same gains machinery, with the one margin this file uses to tell two
# scenarios' numbers apart: giving responds to its tax price with an elasticity
# of -0.5, and may change by at most 100 percent either way.
GIVING_RESPONDS_LEG = BehaviorLeg(
    name="kg_dynamics_and_giving",
    kg_dynamics=GAINS_LEG.kg_dynamics,
    margins={"charity": {"elasticity": -0.5, "max_percent_change": 1.0}},
)


def test_a_scenario_with_no_gains_leg_runs_one_execution_at_neutral():
    """No host solve, no gains applier, no runtime tables.

    The behavioral margins still all run, so this scenario and a responding
    one share one compiled program, but every one of them is at the parameter
    values that return the columns it was given, and the gains applier's slot
    is filled by the placeholder that does the same.
    """
    states, solve_group = scenarios._gains_states_for_group(
        [_spec()],
        [_prepared()],
        {"baseline": _provenance(NO_GAINS_LEG)},
        # A family without the gains model never builds the shared gains
        # pieces, so the group sees none — and needs none.
        pieces=None,
    )
    assert states == {}
    # And nothing was prepared for a batched solve there was no scenario for.
    assert solve_group is None

    plain = behavior_loader.plain_behavior_builder()
    scenario = scenarios.batch_scenario(
        _spec(), _prepared(), None, plain_behavior_builder=plain
    )
    assert scenario.program.behavior_builder is plain
    # Nothing fills the frozen slot, which belongs to the gains machinery alone.
    assert scenario.program.frozen_gains_builder is None
    assert scenario.program.runs_the_gains_model is False
    assert scenario.program.gains_rate_pricers is None
    assert scenario.gains_tables is None
    assert scenario.behavior_parameters.keys() == (
        behavior_parameters.all_neutral().keys()
    )
    assert float(scenario.behavior_parameters["charity"]["elasticity"]) == 0.0


def _law_stub(regime_by_year):
    """A compiled-law stub whose equities regime is stated year by year."""
    from taxmodel.law.compile import FILING_STATUSES

    stated = np.repeat(
        np.asarray(regime_by_year, dtype=np.float64).reshape(len(YEARS), 1, 1),
        len(FILING_STATUSES),
        axis=1,
    )
    return SimpleNamespace(values={"pref.kg_death_regime_equities": stated})


def _shared_pieces():
    """What every gains scenario of a partition shares, with the reads faked."""
    chain_spec = builders.read_gains_chain_spec()
    return scenarios.SharedGainsPieces(
        years=YEARS,
        # Two households, one row per run year. Nothing here is solved; the
        # driver reads the count to size the conversion table, which the
        # conversion module reads in every program.
        host_records=[{"weight": np.ones(2)} for _ in YEARS],
        frozen_builder=builders.frozen_gains_builder_for(chain_spec),
        behavioral_builder=builders.behavioral_gains_builder_for(chain_spec),
        pricers_by_switch={},
        macro_root=Path("."),
        charity_deflator_by_year=None,
        cell_weights=None,
    )


def _gains_state(regime_by_year=(1, 1), shared=None):
    """A gains scenario's host-side state, with a solved frozen stack faked."""
    from taxmodel.behavior.kg import stepper

    law = gains_solve.read_gains_law(_law_stub(regime_by_year), len(YEARS))
    return scenarios.GainsScenarioState(
        law=law,
        frozen_tables=np.zeros((len(YEARS), 63, len(stepper.CELL_TABLE_FIELDS))),
        pricers=SimpleNamespace(),
        shared=_shared_pieces() if shared is None else shared,
    )


def test_a_gains_scenario_composes_the_machinery_and_carries_its_tables_as_data():
    state = _gains_state(regime_by_year=(1, 1))
    scenario = scenarios.batch_scenario(
        _spec("reform"),
        _prepared(),
        state,
        plain_behavior_builder=behavior_loader.plain_behavior_builder(),
    )

    assert scenario.program.runs_the_gains_model is True
    assert scenario.program.behavior_builder is state.shared.behavioral_builder
    assert scenario.program.frozen_gains_builder is state.shared.frozen_builder
    assert scenario.program.frozen_gains_builder is not (
        scenario.program.behavior_builder
    )
    # The tables ride along as per-scenario data, with the behavioral stack at
    # the do-nothing values before the host solve has run.
    tables = scenario.gains_tables
    assert tables.frozen_cell_table.shape[0] == len(YEARS)
    rate_factor = gains_solve.stepper.cell_table_column(
        tables.behavioral_cell_table, "rate_factor"
    )
    np.testing.assert_allclose(np.asarray(rate_factor), 1.0)


def test_two_gains_scenarios_share_a_program_even_with_different_responses():
    """Different laws and different elasticities, same shape.

    The program is the grouping key in `years/loop.py`, so two scenarios whose
    programs compare equal run in one compiled program, each on its own tables
    and its own behavioral numbers. A scenario that lets giving respond and one
    that does not differ in a number, not in code.
    """
    carryover = _gains_state(regime_by_year=(1, 1))
    also_carryover = scenarios.GainsScenarioState(
        law=gains_solve.read_gains_law(_law_stub((0, 1)), len(YEARS)),
        frozen_tables=carryover.frozen_tables,
        pricers=carryover.pricers,
        shared=carryover.shared,
    )
    # The driver gives every member of a family the same calculator objects
    # (`scenarios.SHARED_CALCULATOR_FIELDS`); the test does the same by hand.
    prepared = _prepared()
    plain = behavior_loader.plain_behavior_builder()
    first = scenarios.batch_scenario(
        _spec("k1"),
        prepared,
        carryover,
        behavior_leg=GAINS_LEG,
        plain_behavior_builder=plain,
    )
    second = scenarios.batch_scenario(
        _spec("k2"),
        prepared,
        also_carryover,
        behavior_leg=GIVING_RESPONDS_LEG,
        plain_behavior_builder=plain,
    )
    assert first.program == second.program
    # Same program, different numbers: only the second scenario lets giving
    # respond to its tax price.
    assert float(first.behavior_parameters["charity"]["elasticity"]) == 0.0
    assert float(second.behavior_parameters["charity"]["elasticity"]) == -0.5

    # A deeming law is a different program shape and must not share.
    deeming = _gains_state(regime_by_year=(0, 2), shared=carryover.shared)
    deeming = scenarios.GainsScenarioState(
        law=deeming.law,
        frozen_tables=deeming.frozen_tables,
        pricers=carryover.pricers,
        shared=carryover.shared,
    )
    third = scenarios.batch_scenario(
        _spec("k3"), prepared, deeming, plain_behavior_builder=plain
    )
    # Since the deemed gain became a number rather than a shape, a law that
    # deems realization at death shares the program with one that does not.
    assert third.program == first.program


def test_only_a_gains_scenario_is_planned_to_write_the_gains_cells_file(tmp_path):
    path = tmp_path / "smoke.csv"
    path.write_text(
        RUNSCRIPT_HEADER
        + "baseline,default,default,default,2026:2027,\n"
        + "gains,default,default,kg_dynamics_only,2026:2027,\n"
    )
    plan = plan_run(
        path,
        output_root=tmp_path / "runs",
        code_version=FIXED_CODE_VERSION,
        started_at=FIXED_START,
    )
    names = output_directory.planned_output_names(plan)

    assert f"scenarios/gains/{output_directory.GAINS_CELLS_FILE}" in names
    assert f"scenarios/baseline/{output_directory.GAINS_CELLS_FILE}" not in names


def test_the_gains_cells_table_is_one_row_per_year_per_age_cell():
    from taxmodel.behavior.kg import stepper

    n_ages = 63
    tables = np.zeros((len(YEARS), n_ages, len(stepper.CELL_TABLE_FIELDS)))
    tables[1, 5, stepper.CELL_TABLE_FIELDS.index("rate_factor")] = 1.25
    table = run_tables.gains_cells_table(
        SimpleNamespace(years=YEARS, gains_cell_tables=tables)
    )

    assert len(table) == len(YEARS) * n_ages
    assert list(table.columns[:2]) == ["year", "age"]
    # Age 23 is the sixth cell, because the grid starts at 18.
    row = table[(table["year"] == 2027) & (table["age"] == 23)]
    assert float(row["rate_factor"].iloc[0]) == 1.25
