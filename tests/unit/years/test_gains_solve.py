"""The plumbing that runs the capital-gains model inside a scenario.

The structure under test is one compiled program executed twice around one
host solve. What is checked here: the scenario's law is read off the compiled
law with its filing-status axis intact; the runtime tables silence the applier
when they are the do-nothing tables and hand each year its own cell table when
they are not; the frozen builder composes the applier and nothing else; the
rate exports come back one row per year through the runner and the batch
runner; the host solves produce cell tables from records built from the data;
and the tax on gains deemed realized at death is actually charged.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from taxmodel.behavior import loader
from taxmodel.behavior.config import BehaviorLeg
from taxmodel.behavior.gains import builders, settings
from taxmodel.behavior.kg import applier, prepass, stepper
from taxmodel.behavior.kg.stock_inputs import KG_N_AGES
from taxmodel.law.compile import FILING_STATUSES
from taxmodel.reporting import estate_allocator_jax
from taxmodel.years import gains_solve, world_names, worlds
from tests.unit.years.one_scenario import (
    run_one_scenario,
    taxes_then_prices,
    unchanged_frame,
)

from . import test_one_year

YEARS = test_one_year.YEARS

# A full pass compiles five programs: the projection year, the population
# totals, and the three report programs (the tax at death, the distribution
# table and the marginal-rate sums).
PROGRAMS_IN_A_FULL_PASS = 5


# ---------------------------------------------------------------------------
# The scenario's law
# ---------------------------------------------------------------------------


class _CompiledLawStub:
    """A compiled law is a dictionary of [year, filing status, slot] arrays."""

    def __init__(self, values):
        self.values = values


def _by_filing_status(rows):
    """One parameter stated by filing status, for two years."""
    return np.asarray(rows, dtype=np.float64).reshape(2, len(FILING_STATUSES), 1)


def _for_everyone(by_year):
    """One parameter stated once for everyone, for two years."""
    return np.repeat(
        np.asarray(by_year, dtype=np.float64).reshape(2, 1, 1),
        len(FILING_STATUSES),
        axis=1,
    )


def _law():
    return _CompiledLawStub(
        {
            "pref.kg_death_regime_equities": _for_everyone([0, 2]),
            "pref.kg_death_regime_pass_throughs": _for_everyone([0, 1]),
            "pref.kg_sec121_excl": _by_filing_status(
                [
                    [250_000, 500_000, 250_000, 250_000],
                    [250_000, 500_000, 250_000, 250_000],
                ]
            ),
            "pref.kg_death_gain_excl": _by_filing_status(
                [[0, 0, 0, 0], [1_000_000, 2_000_000, 1_000_000, 1_000_000]]
            ),
        }
    )


def _stepup_law():
    """A law that gives every asset class a stepped-up basis at death."""
    return _CompiledLawStub(
        {
            "pref.kg_death_regime_equities": _for_everyone([0, 0]),
            "pref.kg_death_regime_pass_throughs": _for_everyone([0, 0]),
        }
    )


def _carryover_only_law():
    """A law that moves equities to carryover basis and deems nothing."""
    return _CompiledLawStub(
        {
            "pref.kg_death_regime_equities": _for_everyone([1, 1]),
        }
    )


def test_the_law_keeps_its_filing_status_axis():
    law = settings.read_gains_law(_law(), len(YEARS))

    assert law.section_121_exclusion.shape == (2, len(FILING_STATUSES))
    # A married couple's §121 exclusion is twice a single filer's, and reading
    # the parameter must not lose that.
    assert law.section_121_exclusion[0].tolist() == [
        250_000,
        500_000,
        250_000,
        250_000,
    ]
    # The three asset classes the stub law does not mention are under a
    # stepped-up basis, which is what a law saying nothing means.
    assert law.death_regime_by_class[1].tolist() == [2, 1, 0, 0, 0]


# ---------------------------------------------------------------------------
# The runtime tables and the two builders
# ---------------------------------------------------------------------------

N_CELLS = 63


def _cell_tables(rate_factor_by_year):
    """A cell-table stack whose only live column is the rate factor."""
    stack = np.zeros(
        (len(rate_factor_by_year), N_CELLS, len(stepper.CELL_TABLE_FIELDS))
    )
    rate_factor = stepper.CELL_TABLE_FIELDS.index("rate_factor")
    for year, factor in enumerate(rate_factor_by_year):
        stack[year, :, rate_factor] = factor
    return jnp.asarray(stack)


def _households():
    return {
        "kg_lt": jnp.asarray([100.0, 200.0]),
        "age1": jnp.asarray([20.0, 20.0]),
        "age2": jnp.asarray([0.0, 0.0]),
        "filing_status": jnp.asarray([1.0, 2.0]),
        "m_household": jnp.asarray([0.0, 0.0]),
        # The decedent's gross estate, which the applier reads to work out the
        # chance of a charitable bequest.
        applier.ESTATE_GROSS_COLUMN: jnp.asarray([0.0, 0.0]),
        applier.UNREALIZED_GAIN_STOCK_COLUMN: jnp.asarray([0.0, 0.0]),
        **{f"value.{name}": jnp.zeros(2) for name in applier.DEEMED_ASSET_CLASSES},
        **{f"basis.{name}": jnp.zeros(2) for name in applier.DEEMED_ASSET_CLASSES},
    }


def _chain_spec():
    """The gains applier alone, with none of the behavioral margins behind it.

    What these tests are about is the applier and the runtime tables it spends.
    The made-up households below carry the handful of columns the applier reads
    and no prices at all, which is not enough for the margins that read a real
    frame; the chain those margins make with the applier is checked in
    tests/unit/behavior/test_neutral_identity.py, on a frame that has them.
    """
    return replace(builders.read_gains_chain_spec(), module_entries=())


def test_the_chain_spec_carries_every_behavioral_margin_behind_the_applier():
    """A gains scenario runs the whole pinned chain, applier first."""
    spec = builders.read_gains_chain_spec()
    families = [entry["manifest"]["family"] for entry in spec.module_entries]
    assert families == list(loader.FAMILY_ORDER[1:])


def _tables(law, behavioral=None):
    frozen = _cell_tables([1.0, 1.0])
    tables = builders.runtime_tables(
        settings.read_gains_law(law, len(YEARS)),
        frozen,
        behavioral,
        # One price level per year, all at the year the charitable bequest
        # logits were fit in, so the deflator is 1 and prices nothing away.
        charity_deflator=jnp.ones(len(YEARS)),
    )
    return tables


def _year_of(tables, year):
    return jax.tree.map(lambda value: value[year], tables)


def test_the_do_nothing_tables_leave_every_household_alone():
    """The first execution's behavioral tables silence the applier."""
    build = builders.behavioral_gains_builder_for(_chain_spec())
    tables = _tables(_law())  # behavioral left out: the do-nothing tables

    for year in range(len(YEARS)):
        out = build(_households(), _year_of(tables, year), {}, {})
        np.testing.assert_allclose(out["kg_lt"], [100.0, 200.0])
        np.testing.assert_allclose(out[applier.LOCKIN_COLUMN], [0.0, 0.0])


def test_each_year_reads_its_own_row_of_the_behavioral_stack():
    """The year loop hands each year its own slice of the runtime tables."""
    build = builders.behavioral_gains_builder_for(_chain_spec())
    tables = _tables(_law(), behavioral=_cell_tables([1.0, 2.0]))

    first = build(_households(), _year_of(tables, 0), {}, {})
    second = build(_households(), _year_of(tables, 1), {}, {})

    np.testing.assert_allclose(first["kg_lt"], [100.0, 200.0])
    np.testing.assert_allclose(second["kg_lt"], [200.0, 400.0])


def test_the_frozen_builder_reads_the_frozen_table_and_not_the_behavioral_one():
    """The frozen slot is the applier alone, spending the frozen tables."""
    build = builders.frozen_gains_builder_for(_chain_spec())
    # The behavioral stack doubles gains; the frozen stack leaves them alone.
    tables = _tables(_law(), behavioral=_cell_tables([2.0, 2.0]))

    for with_corporate_markdown in (False, True):
        out = build(
            _households(),
            _year_of(tables, 0),
            {},
            {},
            with_corporate_markdown=with_corporate_markdown,
        )
        np.testing.assert_allclose(out["kg_lt"], [100.0, 200.0])


def test_the_applier_gets_the_law_by_filing_status_out_of_the_runtime_tables():
    """Each household reads the §121 exclusion for its own filing status."""
    seen = {}
    original = applier.apply_cell_table

    def spy(records, prices_reform, prices_baseline, params, draws=None):
        seen["cell_table"] = params["cell_table"]
        seen["section_121_exclusion"] = params["section_121_exclusion"]
        return original(records, prices_reform, prices_baseline, params, draws)

    applier.apply_cell_table = spy
    try:
        build = builders.behavioral_gains_builder_for(_chain_spec())
        tables = _tables(_law(), behavioral=_cell_tables([1.0, 2.0]))
        build(_households(), _year_of(tables, 1), {}, {})
    finally:
        applier.apply_cell_table = original

    assert seen["cell_table"].shape == (N_CELLS, len(stepper.CELL_TABLE_FIELDS))
    np.testing.assert_array_equal(
        seen["section_121_exclusion"], [250_000.0, 500_000.0]
    )


# ---------------------------------------------------------------------------
# The rate exports: through the runner, one row per year
# ---------------------------------------------------------------------------


class _FakePricers:
    """Duck-typed rate pricers, so the export plumbing is tested on its own.

    The exports a year makes mark the frame they were measured on by reading a
    number off it. They also carry the cell weights the year was handed, which
    is how the test sees that the year's weights reached the year.
    """

    def year_exports(
        self, *, cell_weights, mechanical_frame, pre_injection_frame, **_ignored
    ):
        return {
            "reform_on_mechanical_mtr_kg_lt": jnp.asarray(mechanical_frame["income"]),
            "reform_law_only_mtr_kg_lt": jnp.asarray(pre_injection_frame["income"]),
            "weighted_stock_the_year_was_handed": cell_weights.weighted_stock,
        }


# Two households in two cells, each holding and selling a stated number of
# dollars. The plumbing test only needs the weights to be recognisable.
_FAKE_CELL_WEIGHTS = prepass.CellWeights(
    cell_row=jnp.zeros((len(YEARS), 2), dtype=jnp.int32),
    weighted_realizations=jnp.full((len(YEARS), 2), 3.0),
    weighted_stock=jnp.full((len(YEARS), 2), 7.0),
)


def _run(with_pricers: bool):

    return run_one_scenario(
        test_one_year.population(),
        jnp.asarray([1.1, 1.1]),
        jnp.ones(len(YEARS)),
        test_one_year.profiles(0.5),
        years=YEARS,
        frame_builder=test_one_year.FRAME_BUILDER,
        tax_calculator=test_one_year.taxes,
        price_calculator=test_one_year.prices,
        gains_rate_pricers=_FakePricers() if with_pricers else None,
        gains_cell_weights=_FAKE_CELL_WEIGHTS if with_pricers else None,
    )


def test_asking_for_the_rate_exports_changes_no_reported_number():
    plain = _run(with_pricers=False)
    with_rates = _run(with_pricers=True)

    assert not plain.gains_rate_exports
    for year in YEARS:
        for frame in (world_names.Frame.MECHANICAL, world_names.Frame.BEHAVIOR):
            np.testing.assert_array_equal(
                with_rates.grid.get(frame, world_names.Law.REFORM, year)[
                    "liab_iit_net"
                ],
                plain.grid.get(frame, world_names.Law.REFORM, year)["liab_iit_net"],
            )


def test_the_rate_exports_come_back_one_row_per_year():
    run = _run(with_pricers=True)

    for name in ("reform_on_mechanical_mtr_kg_lt", "reform_law_only_mtr_kg_lt"):
        assert run.gains_rate_exports[name].shape == (len(YEARS), 2)
    # The year's cell weights reached the year loop, which is where the rates
    # are averaged into cells.
    np.testing.assert_allclose(
        run.gains_rate_exports["weighted_stock_the_year_was_handed"], 7.0
    )
    # The test's frame builder adds (law - 1) * 10 to income on the reform
    # side, so the mechanical frame reads 1.0 under the 1.1 reform law.
    np.testing.assert_allclose(
        run.gains_rate_exports["reform_on_mechanical_mtr_kg_lt"],
        np.full((2, 2), 1.0),
    )


def _a_gains_program():
    """One gains scenario's program, built once so that calls share a cache entry."""
    from taxmodel.years import loop

    return loop.ScenarioProgram(
        switch_fingerprint="charitable double run: no",
        frame_builder=test_one_year.FRAME_BUILDER,
        tax_calculator=test_one_year.taxes,
        price_calculator=test_one_year.prices,
        behavior_builder=unchanged_frame,
        taxes_and_prices=taxes_then_prices(test_one_year.taxes, test_one_year.prices),
        runs_the_gains_model=True,
        gains_rate_pricers=_FakePricers(),
    )


def _run_the_batch_runner(program, rates_only: bool, cache: dict | None = None):
    """One gains scenario through the batch runner, both ways."""
    from taxmodel.years import loop

    return loop.run_scenarios(
        test_one_year.population(),
        [
            loop.BatchScenario(
                name="scenario",
                program=program,
                reform_laws=jnp.asarray([1.1] * len(YEARS)),
                baseline_laws=jnp.ones(len(YEARS)),
                wealth_profiles=test_one_year.profiles(0.5),
            )
        ],
        years=YEARS,
        scenarios_per_program=1,
        gains_cell_weights=_FAKE_CELL_WEIGHTS,
        program_cache=cache,
        rates_only=rates_only,
    )


def test_the_batch_runner_brings_the_rate_exports_back_too():
    batched = _run_the_batch_runner(_a_gains_program(), rates_only=False)

    exports = batched.runs["scenario"].gains_rate_exports
    assert "reform_on_mechanical_mtr_kg_lt" in exports
    assert exports["reform_on_mechanical_mtr_kg_lt"].shape == (len(YEARS), 2)


def test_the_rates_only_pass_exports_the_rates_the_full_pass_does():
    program = _a_gains_program()
    full = _run_the_batch_runner(program, rates_only=False)
    rates = _run_the_batch_runner(program, rates_only=True)

    full_exports = full.runs["scenario"].gains_rate_exports
    rates_exports = rates.runs["scenario"].gains_rate_exports
    assert set(rates_exports) == set(full_exports)
    for name, exported in full_exports.items():
        np.testing.assert_array_equal(rates_exports[name], exported)

    # The pass forms no reported number, and only the year program is
    # compiled.
    only_rates = rates.runs["scenario"]
    assert only_rates.distribution_table is None
    assert only_rates.mtr_table_sums is None
    assert only_rates.detail is None
    assert rates.counters.programs_compiled == 1
    assert full.counters.programs_compiled == PROGRAMS_IN_A_FULL_PASS


def test_the_second_execution_reuses_the_year_the_rates_only_pass_compiled():
    program = _a_gains_program()
    cache: dict = {}
    rates = _run_the_batch_runner(program, rates_only=True, cache=cache)
    full = _run_the_batch_runner(program, rates_only=False, cache=cache)
    again = _run_the_batch_runner(program, rates_only=False, cache=cache)

    # The rates-only pass compiles the year alone and leaves it in the cache.
    # The execution after it compiles the totals and the three report
    # programs, and a third call finds all five.
    assert rates.counters.programs_compiled == 1
    assert full.counters.programs_compiled == PROGRAMS_IN_A_FULL_PASS - 1
    assert again.counters.programs_compiled == 0
    for year in YEARS:
        np.testing.assert_array_equal(
            again.runs["scenario"]
            .grid.get(world_names.Frame.MECHANICAL, world_names.Law.REFORM, year)[
                "liab_iit_net"
            ],
            full.runs["scenario"]
            .grid.get(world_names.Frame.MECHANICAL, world_names.Law.REFORM, year)[
                "liab_iit_net"
            ],
        )


def _population_with_gains_columns():
    """The two-household test population, with the columns the applier reads."""
    from taxmodel.frame.population import Population, stored_values

    base = test_one_year.population()
    shape = (len(YEARS), 2)
    values = dict(base.values)
    values["kg_lt"] = np.broadcast_to(np.array([50.0, 80.0]), shape).copy()
    values["m_household"] = np.zeros(shape)
    # The decedent's gross estate, which the applier reads for the chance of a
    # charitable bequest.
    values[applier.ESTATE_GROSS_COLUMN] = np.zeros(shape)
    values[applier.UNREALIZED_GAIN_STOCK_COLUMN] = np.zeros(shape)
    for name in applier.DEEMED_ASSET_CLASSES:
        values[f"value.{name}"] = np.zeros(shape)
        values[f"basis.{name}"] = np.zeros(shape)
    return Population(
        universe=base.universe,
        field_names=("id", "year", *values),
        values=stored_values(values),
        draws={},
        dropped_text_columns=(),
    )


def _taxes_reading_gains(frame, law):
    """The toy calculator, with realized gains in the base so a moved kg_lt shows."""
    base = (
        jnp.asarray(frame["income"])
        + jnp.asarray(frame["value.cash"])
        + jnp.asarray(frame["kg_lt"])
    )
    liability = jnp.asarray(law) * base
    zero = jnp.zeros_like(liability)
    return {
        "liab_iit_net": liability,
        "liab_pr": zero,
        "liab_wealth": zero,
        "expanded_inc": base,
        "agi": jnp.asarray(frame["income"]),
        "filer_recomputed": jnp.ones_like(liability),
        "simple_filer": zero,
    }


def test_the_do_nothing_tables_reproduce_a_no_gains_run_exactly():
    """A gains-shaped program on identity tables is the plain run, bitwise.

    This is the identity that lets one program shape serve the first
    execution: composing the applier changes nothing until the tables say
    something.
    """

    population = _population_with_gains_columns()
    spec = _chain_spec()
    # Regime codes all zero would be the cleanest identity; carryover regime
    # codes still change no household because the frozen stack is the
    # do-nothing stack here too.
    tables = builders.GainsTables(
        behavioral_cell_table=builders.identity_cell_tables(len(YEARS)),
        frozen_cell_table=builders.identity_cell_tables(len(YEARS)),
        mechanical_frozen_cell_table=builders.identity_cell_tables(len(YEARS)),
        death_regime=jnp.zeros((len(YEARS), 5), dtype=jnp.int32),
        realized_at_death_share=jnp.zeros((len(YEARS), 5)),
        section_121_exclusion=jnp.zeros((len(YEARS), 4)),
        death_gain_exclusion=jnp.zeros((len(YEARS), 4)),
        conversions=jnp.zeros((len(YEARS), 0, 5)),
        charity_deflator=jnp.ones(len(YEARS)),
    )

    def run(**gains_kwargs):
        return run_one_scenario(
            population,
            jnp.asarray([1.1, 1.1]),
            jnp.ones(len(YEARS)),
            test_one_year.profiles(0.5),
            years=YEARS,
            frame_builder=test_one_year.FRAME_BUILDER,
            tax_calculator=_taxes_reading_gains,
            price_calculator=test_one_year.prices,
            **gains_kwargs,
        )

    plain = run()
    gains_shaped = run(
        gains_tables=tables,
        runs_the_gains_model=True,
        behavior_builder=builders.behavioral_gains_builder_for(spec),
        frozen_gains_builder=builders.frozen_gains_builder_for(spec),
    )

    for year in YEARS:
        for frame in (
            world_names.Frame.RAW,
            world_names.Frame.FLOWS,
            world_names.Frame.MECHANICAL,
            world_names.Frame.BEHAVIOR,
        ):
            np.testing.assert_array_equal(
                gains_shaped.grid.get(frame, world_names.Law.REFORM, year)[
                    "liab_iit_net"
                ],
                plain.grid.get(frame, world_names.Law.REFORM, year)["liab_iit_net"],
            )


# ---------------------------------------------------------------------------
# The host solves, on made-up households
# ---------------------------------------------------------------------------


def _host_records(n_households=8):
    age = np.linspace(30.0, 75.0, n_households)
    value = np.linspace(100_000.0, 4_000_000.0, n_households)
    per_year = {
        "weight": np.full(n_households, 1_000.0),
        "age1": age,
        "age2": np.zeros(n_households),
        "filing_status": np.ones(n_households),
        "m_household": 0.001 + 0.0004 * (age - 30.0),
        "p_widow": np.zeros(n_households),
        "unrealized_gain_stock": 0.5 * value,
        "estate_economic_gross": 2.0 * value,
        "kg_lt": 0.025 * value,
        "value.equities": value,
        "basis.equities": 0.5 * value,
    }
    for name in applier.DEEMED_ASSET_CLASSES:
        per_year.setdefault(f"value.{name}", np.zeros(n_households))
        per_year.setdefault(f"basis.{name}", np.zeros(n_households))
    return [dict(per_year) for _year in YEARS], n_households


def _averaged_into_cells(records, household_rates):
    """The cell sums the compiled program would export from these rates.

    A test says what rate each household faces, which is the natural way to
    state what a scenario prices. The run itself averages those rates into age
    cells before they leave the device, so the same averaging is done here and
    the solve is handed what it would really be handed.
    """
    n_years = len(records)
    rates = dict(household_rates)
    by_year = []
    for year in range(n_years):
        weights = prepass.cell_weights_from_records(records[year])
        one_year = {name: value[year] for name, value in rates.items()}
        by_year.append(builders.average_rates_into_cells(weights, one_year))
    return {
        name: jnp.stack([year[name] for year in by_year]) for name in by_year[0]
    }


DEFAULT_CONVERSION_FILE = "config/calibrations/kg/conversion.yaml"


def _leg(conversion_file=None):
    """A behavior leg binding the default realization file, and a conversion
    file when one is given."""
    kg_dynamics = {"realization": "config/calibrations/kg/realization.yaml"}
    if conversion_file is not None:
        kg_dynamics["conversion"] = str(conversion_file)
    return BehaviorLeg(name="test", kg_dynamics=kg_dynamics)


def _solve_behavioral(
    records,
    exports,
    *,
    reform_law,
    macro_root,
    conversion_file=None,
    corporate_asset_exposure=None,
    equity_markdown_by_year=None,
):
    """One scenario through the batched solve, which is the only solve there is.

    A group of one is still a group: the run always goes through
    `prepare_gains_solve_group` and `solve_behavioral_tables_for_group`, so a
    test of the behavioral solve goes the same way.
    """
    group = gains_solve.prepare_gains_solve_group(
        records,
        YEARS,
        scenario_ids=["test"],
        compiled_laws=[reform_law],
        behavior_legs=[_leg(conversion_file)],
        macro_root=macro_root,
        corporate_asset_exposures=[corporate_asset_exposure],
        equity_markdowns=[equity_markdown_by_year],
    )
    return gains_solve.solve_behavioral_tables_for_group(group, [exports])[0]


def test_the_frozen_solve_runs_from_the_data_alone(macro_root):
    """No rates anywhere: the frozen tables exist before any execution."""
    records, _n = _host_records()

    group = gains_solve.prepare_gains_solve_group(
        records,
        YEARS,
        scenario_ids=["test"],
        compiled_laws=[_law()],
        behavior_legs=[_leg()],
        macro_root=macro_root,
        corporate_asset_exposures=[None],
        equity_markdowns=[None],
    )
    frozen = gains_solve.solve_frozen_tables_for_group(group)[0].first_order

    # One cell per age from 18 to 80.
    assert frozen.shape == (
        len(YEARS),
        KG_N_AGES,
        len(stepper.CELL_TABLE_FIELDS),
    )
    # Frozen means selling behaviour is held at baseline: the rate factor is
    # one everywhere, whatever the reform does to gains at death.
    rate_factor = np.asarray(stepper.cell_table_column(frozen[0], "rate_factor"))
    np.testing.assert_allclose(rate_factor, 1.0)


def test_the_corporate_markdown_enters_the_mechanical_frozen_table_only(macro_root):
    """A markdown of share prices lowers the gain the mechanical worlds' frozen
    pass sells, and leaves the first-order world's frozen table as it was."""
    records, _n = _host_records()
    markdown = 0.1

    def frozen_tables(exposure, markdown_by_year):
        group = gains_solve.prepare_gains_solve_group(
            records,
            YEARS,
            scenario_ids=["test"],
            compiled_laws=[_carryover_only_law()],
            behavior_legs=[_leg()],
            macro_root=macro_root,
            corporate_asset_exposures=[exposure],
            equity_markdowns=[markdown_by_year],
        )
        return gains_solve.solve_frozen_tables_for_group(group)[0]

    marked_down = frozen_tables({"value.equities": 1.0}, [markdown] * len(YEARS))
    no_markdown = frozen_tables(None, None)

    # The first-order table is the same with or without the markdown, and
    # without a markdown the two tables are the same.
    np.testing.assert_array_equal(marked_down.first_order, no_markdown.first_order)
    np.testing.assert_array_equal(no_markdown.mechanical, no_markdown.first_order)

    def column(table, field):
        return np.asarray(stepper.cell_table_column(table, field))

    # The mechanical table's sales fall by baseline turnover times the debit,
    # the markdown times the equity value behind each cell.
    debit = column(marked_down.mechanical, "corporate_gain_debit")
    weight = np.asarray(records[0]["weight"])
    value = np.asarray(records[0]["value.equities"])
    np.testing.assert_allclose(debit[0].sum(), markdown * (weight * value).sum())
    turnover = column(marked_down.mechanical, "scenario_realization_rate")
    lower_sales = column(marked_down.first_order, "extra_realizations") - column(
        marked_down.mechanical, "extra_realizations"
    )
    np.testing.assert_allclose(lower_sales, turnover * debit, atol=1e-6)
    assert lower_sales.sum() > 0.0


def test_the_behavioral_solve_reads_the_exported_rates(macro_root):
    """Rates in, cell tables out: a higher rate on gains means less selling."""
    records, n_households = _host_records()

    def stacked(value):
        return jnp.asarray(np.full((len(YEARS), n_households), value))

    exports = {
        # Today's top rate on a long-term gain with the investment income
        # tax, and a reform five points above it.
        "reform_on_mechanical_mtr_kg_lt": stacked(0.288),
        "baseline_on_mechanical_mtr_kg_lt": stacked(0.238),
        "reform_law_only_mtr_kg_lt": stacked(0.288),
    }

    solved = _solve_behavioral(
        records,
        _averaged_into_cells(records, exports),
        reform_law=_law(),
        macro_root=macro_root,
    )

    cell_tables = jnp.asarray(solved.cell_tables)
    assert cell_tables.shape == (
        len(YEARS),
        KG_N_AGES,
        len(stepper.CELL_TABLE_FIELDS),
    )
    rate_factor = np.asarray(
        stepper.cell_table_column(cell_tables[0], "rate_factor")
    )
    # The eight households have eight different ages, so they fill eight
    # cells. An empty cell stays at one.
    assert np.all(rate_factor <= 1.0)
    assert (rate_factor < 1.0).sum() == n_households


def test_equal_rates_on_the_mechanical_frame_mean_no_response(macro_root):
    """The baseline leg is baseline law on the mechanical frame.

    When the reform leaves the rate on the mechanical frame unchanged, the
    selling choice has nothing to respond to.
    """
    records, n_households = _host_records()

    def stacked(value):
        return jnp.asarray(np.full((len(YEARS), n_households), value))

    exports = {
        "reform_on_mechanical_mtr_kg_lt": stacked(0.288),
        "baseline_on_mechanical_mtr_kg_lt": stacked(0.288),
        "reform_law_only_mtr_kg_lt": stacked(0.288),
    }
    # A stepped-up basis at death, the treatment the baseline leg assumes, so
    # only the two legs' rates could differ.
    solved = _solve_behavioral(
        records,
        _averaged_into_cells(records, exports),
        reform_law=_stepup_law(),
        macro_root=macro_root,
    )
    rate_factor = np.asarray(
        stepper.cell_table_column(jnp.asarray(solved.cell_tables)[0], "rate_factor")
    )
    np.testing.assert_allclose(rate_factor, 1.0)


# ---------------------------------------------------------------------------
# The two channels: conversion and the corporate markdown
# ---------------------------------------------------------------------------


def _channel_exports(
    records,
    n_households,
    reform_rate=0.288,
    baseline_rate=0.238,
    ordinary_rate_change=0.10,
):
    """Rate exports for the made-up households.

    ``reform_rate`` and ``baseline_rate`` are the marginal rates on realized
    long-term gain; every ordinary-income leg's rate rises from 37 percent by
    ``ordinary_rate_change``.
    """

    def stacked(value):
        return jnp.asarray(np.full((len(YEARS), n_households), value))

    rates = {
        "reform_on_mechanical_mtr_kg_lt": stacked(reform_rate),
        "baseline_on_mechanical_mtr_kg_lt": stacked(baseline_rate),
        "reform_law_only_mtr_kg_lt": stacked(reform_rate),
    }
    exports = {
        **_averaged_into_cells(records, rates),
        # The conversion pool's quantities, which stay per household: every
        # household has active sole-proprietorship income and taxable income
        # above the top-bracket threshold, so the whole pool converts.
        "mechanical_txbl_inc": stacked(1_000_000.0),
        "mechanical_filing_status": stacked(1.0),
        "mechanical_wages_base1": stacked(200_000.0),
        "mechanical_wages_base2": stacked(0.0),
        "mechanical_part_active": stacked(0.0),
        "mechanical_scorp_active": stacked(0.0),
        "mechanical_sole_prop1": stacked(50_000.0),
        "mechanical_sole_prop2": stacked(0.0),
        # The rate on realized long-term gain the conversion wedge reads,
        # per household.
        "reform_on_mechanical_mtr_kg_lt": stacked(reform_rate),
        "baseline_on_mechanical_mtr_kg_lt": stacked(baseline_rate),
    }
    for name in settings.CONVERSION_LEG_RATE_NAMES:
        exports[f"reform_on_mechanical_mtr_{name}"] = stacked(
            0.37 + ordinary_rate_change
        )
        exports[f"baseline_on_mechanical_mtr_{name}"] = stacked(0.37)
    return exports


def _deemed_law():
    """A law that taxes equities' unrealized gain as if sold at death."""
    return _CompiledLawStub(
        {
            "pref.kg_death_regime_equities": _for_everyone([2, 2]),
        }
    )


def _law_with_brackets(law=None):
    """A law (carryover by default) plus an ordinary schedule whose top
    bracket starts low."""
    values = dict((law or _carryover_only_law()).values)
    brackets = np.full((len(YEARS), len(FILING_STATUSES), 7), np.nan)
    brackets[:, :, 0] = 0.0
    brackets[:, :, 1] = 100_000.0
    values["ord.brackets"] = brackets
    return _CompiledLawStub(values)


def test_the_conversion_channel_conserves_dollars_between_pay_and_stock(macro_root):
    """What leaves the households' pay is what enters the stock of gain."""
    records, n_households = _host_records()

    solved = _solve_behavioral(
        records,
        _channel_exports(records, n_households),
        reform_law=_law_with_brackets(),
        macro_root=macro_root,
        conversion_file=DEFAULT_CONVERSION_FILE,
    )

    conversions = np.asarray(solved.conversions_by_household_leg)
    assert conversions.shape == (len(YEARS), n_households, 5)
    # The ordinary rate rose five points more than the gains rate, so pay is
    # relabeled as equity: conversions are positive somewhere.
    assert conversions.sum() > 0.0
    # Conservation: the weighted household total equals the cell inflow, year
    # by year, to the cent standard.
    weight = np.asarray(records[0]["weight"])
    inflow = np.asarray(
        stepper.cell_table_column(
            jnp.asarray(solved.cell_tables), "conversion_inflow"
        )
    )
    for year in range(len(YEARS)):
        np.testing.assert_allclose(
            inflow[year].sum(),
            (weight * conversions[year].sum(axis=-1)).sum(),
            atol=0.01,
        )


def test_deemed_realization_with_no_statutory_rate_change_converts_nothing(
    macro_root,
):
    """Deeming gains at death raises the tax on a deferred dollar but leaves
    the rate on a realized dollar and the ordinary rates where they were, so
    no pay is relabeled."""
    records, n_households = _host_records()

    solved = _solve_behavioral(
        records,
        _channel_exports(
            records,
            n_households,
            reform_rate=0.238,
            baseline_rate=0.238,
            ordinary_rate_change=0.0,
        ),
        reform_law=_law_with_brackets(_deemed_law()),
        macro_root=macro_root,
        conversion_file=DEFAULT_CONVERSION_FILE,
    )

    np.testing.assert_array_equal(np.asarray(solved.conversions_by_household_leg), 0.0)
    inflow = stepper.cell_table_column(
        jnp.asarray(solved.cell_tables), "conversion_inflow"
    )
    np.testing.assert_array_equal(np.asarray(inflow), 0.0)


def test_the_solve_reads_the_conversion_file_the_behavior_leg_names(
    macro_root, tmp_path
):
    """A leg that binds a conversion file with twice the default sigma
    converts twice as much."""
    records, n_households = _host_records()
    default_sigma, labor_share = settings.read_conversion_settings(
        DEFAULT_CONVERSION_FILE
    )
    doubled = tmp_path / "conversion.yaml"
    doubled.write_text(
        f"conv:\n  value: {2 * default_sigma}\n"
        f"pt_labor_share:\n  value: {labor_share}\n"
    )

    def converted(conversion_file):
        solved = _solve_behavioral(
            records,
            _channel_exports(records, n_households),
            reform_law=_law_with_brackets(),
            macro_root=macro_root,
            conversion_file=conversion_file,
        )
        return np.asarray(solved.conversions_by_household_leg)

    at_default = converted(DEFAULT_CONVERSION_FILE)
    at_double = converted(doubled)

    assert at_default.sum() > 0.0
    np.testing.assert_allclose(at_double, 2.0 * at_default, rtol=1e-12)


def test_the_corporate_markdown_debits_the_stock_of_gain(macro_root):
    """The debit is the markdown times the corporate equity behind the cells."""
    records, n_households = _host_records()
    markdown = 0.1
    exposure = {"value.equities": 1.0}

    solved = _solve_behavioral(
        records,
        _channel_exports(records, n_households),
        reform_law=_carryover_only_law(),
        macro_root=macro_root,
        corporate_asset_exposure=exposure,
        equity_markdown_by_year=[markdown] * len(YEARS),
    )
    without = _solve_behavioral(
        records,
        _channel_exports(records, n_households),
        reform_law=_carryover_only_law(),
        macro_root=macro_root,
    )

    debit = np.asarray(
        stepper.cell_table_column(
            jnp.asarray(solved.cell_tables), "corporate_gain_debit"
        )
    )
    # Every made-up household holds only equities, fully exposed, so the
    # year's debit is the markdown times the cell's whole equity value.
    weight = np.asarray(records[0]["weight"])
    value = np.asarray(records[0]["value.equities"])
    np.testing.assert_allclose(debit[0].sum(), markdown * (weight * value).sum())
    # And without a corporate reform the debit is zero everywhere.
    debit_without = np.asarray(
        stepper.cell_table_column(
            jnp.asarray(without.cell_tables), "corporate_gain_debit"
        )
    )
    np.testing.assert_array_equal(debit_without, 0.0)


def test_no_corporate_rate_change_means_no_corporate_paths():
    """The channel is off for every scenario that leaves the corporate rate alone."""
    from taxmodel.years.corporate import channel as corporate_channel

    same = corporate_channel.corporate_rate_by_year(
        _CompiledLawStub(
            {corporate_channel.CORPORATE_RATE_COLUMN: _for_everyone([0.21, 0.21])}
        ),
        len(YEARS),
    )
    assert (
        corporate_channel.corporate_paths_for_scenario(
            same, same, YEARS, macro_root="unused", config=None, base_erodes=True
        )
        is None
    )


# ---------------------------------------------------------------------------
# The tax on gains deemed realized at death
# ---------------------------------------------------------------------------


def _flat_rate_calculator(rate: float):
    """A toy income tax: a flat rate on realized long-term gains."""

    def taxes(frame, law):
        del law
        liability = rate * jnp.asarray(frame["kg_lt"])
        return {
            "liab_iit": liability,
            "liab_iit_net": liability,
            "pmt_iit_nonwithheld": jnp.zeros_like(liability),
        }

    return taxes


def _priced_with_extra_calls(frame, calculator):
    """What one world does: run the calculator once per evaluation point."""
    results = dict(calculator(frame, None))
    added = worlds.deemed_death_gain_by_evaluation_point(frame)
    if not added:
        return results, None
    realized_gain = jnp.asarray(frame["kg_lt"], dtype=jnp.float64)
    with_the_deemed_gain = [
        calculator({**frame, "kg_lt": realized_gain + gain}, None)[
            worlds.DEEMED_DEATH_LIABILITY_FIELD
        ]
        for gain in added
    ]
    expected_tax, estate_deduction = worlds.the_deemed_death_tax(
        results, with_the_deemed_gain, frame
    )
    folded = worlds.fold_in_the_deemed_death_tax(
        results,
        frame,
        None,
        calculator,
        expected_tax,
        estate_deduction,
    )
    return folded, estate_deduction


def test_a_frame_with_no_deemed_gain_gets_no_extra_calculator_call():
    """Every scenario that runs no gains model, which is nearly all of them.

    Nothing is deemed, so there is no extra evaluation point and the
    calculator runs once, on the households themselves. The baseline sweep
    still gets the column, as zeros, because the five worlds' results are
    differenced against each other and two worlds holding different columns
    cannot be differenced at all.
    """
    frame = {"kg_lt": jnp.asarray([100.0, 200.0])}
    calculator = _flat_rate_calculator(0.2)

    assert worlds.deemed_death_gain_by_evaluation_point(frame) == ()
    folded, _ = _priced_with_extra_calls(frame, calculator)
    np.testing.assert_array_equal(folded["liab_iit_net"], [20.0, 40.0])
    assert estate_allocator_jax.DEEMED_GAINS_COLUMN not in folded

    everywhere = worlds.with_a_zero_deemed_death_tax(calculator(frame, None))
    np.testing.assert_array_equal(
        everywhere[estate_allocator_jax.DEEMED_GAINS_COLUMN], [0.0, 0.0]
    )


def test_a_zero_fill_never_erases_a_deemed_tax_already_priced():
    """A world that priced its own deemed tax keeps it through the zero-fill."""
    frame = {"kg_lt": jnp.asarray([100.0, 200.0])}
    calculator = _flat_rate_calculator(0.2)
    already_priced = dict(calculator(frame, None))
    already_priced[estate_allocator_jax.DEEMED_GAINS_COLUMN] = jnp.asarray([7.0, 9.0])

    refolded = worlds.with_a_zero_deemed_death_tax(already_priced)
    np.testing.assert_array_equal(
        refolded[estate_allocator_jax.DEEMED_GAINS_COLUMN], [7.0, 9.0]
    )


def test_the_deemed_death_tax_is_charged_and_lands_in_the_three_columns():
    """Expected tax is the death-gain tax times the chance of dying this year."""
    rate = 0.2
    frame = {
        "kg_lt": jnp.asarray([100.0, 200.0]),
        applier.DEEMED_FULL_COLUMN: jnp.asarray([1_000.0, 0.0]),
        applier.MORTALITY_COLUMN: jnp.asarray([0.01, 0.5]),
    }
    calculator = _flat_rate_calculator(rate)

    folded, estate_deduction = _priced_with_extra_calls(frame, calculator)

    # The first household's death gain of $1,000 taxed at 20 cents is $200 if
    # it dies, and it dies with probability one in a hundred. The second holds
    # no death gain, so its expected deemed tax is zero however likely it is
    # to die.
    expected = np.array([0.01 * rate * 1_000.0, 0.0])
    np.testing.assert_allclose(folded[estate_allocator_jax.DEEMED_GAINS_COLUMN], expected)
    for name in worlds.DEEMED_DEATH_TAX_LANDS_IN:
        np.testing.assert_allclose(
            folded[name], np.asarray(calculator(frame, None)[name]) + expected
        )
    # What §2053 lets the estate deduct is the tax if death happens, not the
    # expected amount.
    np.testing.assert_allclose(estate_deduction, [rate * 1_000.0, 0.0])


def test_a_law_that_deems_nothing_leaves_the_household_exactly_where_it_was():
    """The identity that lets one program serve every treatment of death gains.

    Under a stepped-up basis the applier writes a deemed gain of zero, so the
    extra evaluation is the household's own return and every difference is
    exactly zero. Deeming is a number rather than a shape.
    """
    frame = {
        "kg_lt": jnp.asarray([100.0, 200.0]),
        applier.DEEMED_FULL_COLUMN: jnp.zeros(2),
        applier.MORTALITY_COLUMN: jnp.asarray([0.01, 0.5]),
    }
    calculator = _flat_rate_calculator(0.2)

    folded, estate_deduction = _priced_with_extra_calls(frame, calculator)
    plain = calculator(frame, None)
    np.testing.assert_array_equal(
        folded[estate_allocator_jax.DEEMED_GAINS_COLUMN], np.zeros(2)
    )
    np.testing.assert_array_equal(estate_deduction, np.zeros(2))
    for name in worlds.DEEMED_DEATH_TAX_LANDS_IN:
        np.testing.assert_array_equal(folded[name], np.asarray(plain[name]))


def test_the_two_exclusion_branches_are_blended_by_the_chance_of_widowhood():
    """An unmarried filer's death-gain exclusion has two branches.

    The one it has, and the married amount it would have if it were widowed at
    death. Each is priced by its own calculator call, and the two taxes are
    blended by the chance of widowhood.
    """
    rate = 0.2
    frame = {
        "kg_lt": jnp.asarray([100.0]),
        applier.DEEMED_FULL_COLUMN: jnp.asarray([600.0]),
        applier.DEEMED_FULL_OWN_EXCLUSION_COLUMN: jnp.asarray([1_000.0]),
        applier.DEEMED_FULL_MARRIED_EXCLUSION_COLUMN: jnp.asarray([600.0]),
        applier.WIDOWHOOD_COLUMN: jnp.asarray([0.25]),
        applier.MORTALITY_COLUMN: jnp.asarray([1.0]),
    }
    calculator = _flat_rate_calculator(rate)

    added = worlds.deemed_death_gain_by_evaluation_point(frame)
    assert len(added) == 2
    folded, _ = _priced_with_extra_calls(frame, calculator)

    blended_gain = 0.25 * 600.0 + 0.75 * 1_000.0
    np.testing.assert_allclose(
        folded[estate_allocator_jax.DEEMED_GAINS_COLUMN], [rate * blended_gain]
    )
