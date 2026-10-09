"""The run directory: its layout, its manifest, and the partial-name rule."""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import types

import pandas as pd
import pytest

from taxmodel.config import law_stacks
from taxmodel.config.runscript import read_runscript
from taxmodel.reporting import distribution, revenue, run_tables
from taxmodel.run import output_directory
from taxmodel.run.plan import (
    RunPlan,
    ScenarioSpec,
    StackMembership,
    StackPlan,
    SubparameterConflict,
    plan_run,
)
from taxmodel.run.scenarios import ScenarioResult, execute_run
from taxmodel.scenario.provenance import DEFAULT_CONFIG_ROOT
from taxmodel.years import world_names
from tests.unit.run.conftest import (
    RUNSCRIPT_HEADER,
    UNMEASURED_COUNTERS,
    assert_marginal_rate_columns_are_the_reporting_layers,
    read_table,
    stand_in_projection,
)

TEST_REFORM = "tests/sd_bump_10k"
FIXED_CODE_VERSION = {"commit": "abcdef1234567890", "branch": "test", "dirty": False}
FIXED_START = dt.datetime(2026, 8, 3, 14, 30, 5, tzinfo=dt.UTC)
YEARS = "2026:2027"


@pytest.fixture
def two_scenario_run(tmp_path):
    """A planned two-scenario run: baseline, and one reform that wants rates."""
    runscript = tmp_path / "smoke.csv"
    runscript.write_text(
        RUNSCRIPT_HEADER
        + f"baseline,default,default,default,{YEARS},\n"
        + f"sd_bump,{TEST_REFORM},default,default,{YEARS},\n"
    )
    return plan_run(
        runscript,
        output_root=tmp_path / "runs",
        code_version=FIXED_CODE_VERSION,
        started_at=FIXED_START,
    )


def test_a_finished_run_holds_exactly_the_frozen_layout(two_scenario_run, stand_in_runner, tmp_path):
    written = execute_run(
        two_scenario_run,
        population_root=tmp_path / "population",
        group_runner=stand_in_runner,
    )
    found = sorted(str(path.relative_to(written)) for path in written.rglob("*"))
    assert found == sorted(
        [
            "manifest.json",
            "runscript.csv",
            "scenarios",
            "scenarios/baseline",
            "scenarios/baseline/manifest.json",
            "scenarios/baseline/totals.csv",
            "scenarios/baseline/distribution.parquet",
            "scenarios/baseline/wealth_diagnostics.csv",
            "scenarios/baseline/tax_law.csv",
            "scenarios/baseline/mtr_table.parquet",
            "scenarios/baseline/form1040.csv",
            "scenarios/baseline/payroll.csv",
            "scenarios/baseline/estate.csv",
            "scenarios/baseline/wealth.csv",
            "scenarios/baseline/receipts.csv",
            "scenarios/baseline/revenue_estimates.csv",
            "scenarios/sd_bump",
            "scenarios/sd_bump/manifest.json",
            "scenarios/sd_bump/totals.csv",
            "scenarios/sd_bump/distribution.parquet",
            "scenarios/sd_bump/wealth_diagnostics.csv",
            "scenarios/sd_bump/tax_law.csv",
            "scenarios/sd_bump/mtr_table.parquet",
            "scenarios/sd_bump/form1040.csv",
            "scenarios/sd_bump/payroll.csv",
            "scenarios/sd_bump/estate.csv",
            "scenarios/sd_bump/wealth.csv",
            "scenarios/sd_bump/receipts.csv",
            "scenarios/sd_bump/revenue_estimates.csv",
        ]
    )


def test_the_runscript_is_copied_in_verbatim(two_scenario_run, stand_in_runner, tmp_path):
    written = execute_run(
        two_scenario_run,
        population_root=tmp_path / "population",
        group_runner=stand_in_runner,
    )
    assert (written / "runscript.csv").read_text() == (
        two_scenario_run.runscript_path.read_text()
    )


def test_a_finished_run_has_no_partial_directory_left(two_scenario_run, stand_in_runner, tmp_path):
    written = execute_run(
        two_scenario_run,
        population_root=tmp_path / "population",
        group_runner=stand_in_runner,
    )
    assert written.is_dir()
    assert not written.with_name(written.name + output_directory.PARTIAL_SUFFIX).exists()


def test_an_interrupted_run_leaves_only_a_partial_directory(
    two_scenario_run, tmp_path, monkeypatch
):
    """A crashed job must not leave something that looks like a complete run.

    Scenarios are written group by group now, so a run cut down mid-way — here
    after its first one-scenario group — leaves a `.partial` directory that
    may hold finished scenario folders. What it never holds is a run manifest:
    that is written last, by `finish_run`, and the missing manifest plus the
    `.partial` suffix are what say the run is unfinished.
    """
    # This test looks for the partial directory at the run's permanent home,
    # so it pins the direct-write path: inside a Slurm job the run would stage
    # the partial directory on the node's local drive instead.
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    def runner_that_stops(scenarios, provenance, *, deliver, population, config_root):
        # The first scenario finishes and is written; the walltime hits
        # before the second group starts.
        deliver(
            (
                ScenarioResult(
                    spec=scenarios[0],
                    projection=stand_in_projection(tuple(scenarios[0].row.years)),
                    provenance=provenance[scenarios[0].id],
                ),
            )
        )
        raise KeyboardInterrupt("the walltime ran out")

    with pytest.raises(KeyboardInterrupt):
        execute_run(
            two_scenario_run,
            population_root=tmp_path / "population",
            group_runner=runner_that_stops,
        )
    final = two_scenario_run.run_directory
    partial = final.with_name(final.name + output_directory.PARTIAL_SUFFIX)
    assert partial.is_dir()
    assert not final.exists()
    # The group that finished before the interruption is on disk, tables and
    # all; the run manifest is not, so nothing claims the run is complete.
    assert (partial / "scenarios" / "baseline" / "totals.csv").is_file()
    assert not (partial / "manifest.json").exists()


def test_a_rerun_in_the_same_second_clears_the_leftover_partial_directory(
    two_scenario_run, stand_in_runner, tmp_path
):
    def runner_that_stops(scenarios, provenance, *, deliver, population, config_root):
        raise RuntimeError("stopped")

    with pytest.raises(RuntimeError):
        execute_run(
            two_scenario_run,
            population_root=tmp_path / "population",
            group_runner=runner_that_stops,
        )
    written = execute_run(
        two_scenario_run,
        population_root=tmp_path / "population",
        group_runner=stand_in_runner,
    )
    assert written.is_dir()
    assert not written.with_name(written.name + output_directory.PARTIAL_SUFFIX).exists()


# --------------------------------------------------------------------------
# The run-level manifest
# --------------------------------------------------------------------------


@pytest.fixture
def finished_run(two_scenario_run, stand_in_runner, tmp_path):
    return execute_run(
        two_scenario_run,
        population_root=tmp_path / "population",
        group_runner=stand_in_runner,
    )


def read_manifest(run_directory):
    return json.loads((run_directory / "manifest.json").read_text())


def test_the_manifest_has_every_section_the_freeze_document_names(finished_run):
    manifest = read_manifest(finished_run)
    assert set(manifest) >= {
        "runscript",
        "scenarios",
        "stacks",
        "detail",
        "outputs",
        "execution",
        "code_version",
        "runtime",
    }


def test_the_manifest_names_every_file_written_with_its_row_count(finished_run):
    manifest = read_manifest(finished_run)
    written = [entry for entry in manifest["outputs"] if entry["written"]]
    for entry in written:
        path = finished_run / entry["path"]
        assert path.is_file(), entry["path"]
        # A manifest entry either claims a row count, in which case the file
        # has to have that many rows, or it is one of the verbatim copies,
        # which have none.
        if entry["rows"] is not None:
            assert entry["rows"] == len(read_table(path)), entry["path"]
    # Every file on disk except the run manifest itself, which is written last
    # and cannot name itself.
    on_disk = {
        str(path.relative_to(finished_run))
        for path in finished_run.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    named = {entry["path"] for entry in written}
    assert on_disk - named == set()


def test_a_run_without_a_stack_leaves_no_file_unwritten(finished_run):
    """Every file this run's layout reserves is on disk."""
    manifest = read_manifest(finished_run)
    absent = [entry for entry in manifest["outputs"] if not entry["written"]]
    assert absent == []


def test_the_manifest_records_the_scenarios_in_run_order_with_their_legs(finished_run):
    manifest = read_manifest(finished_run)
    assert [entry["id"] for entry in manifest["scenarios"]] == ["baseline", "sd_bump"]
    reform = manifest["scenarios"][1]
    assert reform["law_layers"] == [TEST_REFORM]
    assert reform["economy"] == "default"
    assert reform["years"] == [2026, 2027]
    assert reform["from_stack"] is None


def test_the_manifest_carries_the_group_runners_counters(finished_run):
    """A stand-in run says honestly that no compiled program was measured.

    The counters section always has its seats — the compile count, the
    per-family entries, the device high-water mark — and a run whose results
    came from a test stand-in rather than from the batch runner says so
    instead of leaving the seats out.
    """
    counters = read_manifest(finished_run)["execution"]["counters"]
    assert counters["measured"] is False
    assert "no compiled program ran" in counters["why_not"]
    assert set(counters) >= {
        "compiled_programs",
        "families",
        "peak_device_bytes_in_use",
    }


def test_the_manifest_says_where_the_run_executed_and_on_what(finished_run):
    execution = read_manifest(finished_run)["execution"]
    assert execution["where"] == "local"
    assert execution["slurm_job_id"] is None
    # The card the numbers were computed on, read off the runtime stack when
    # the caller does not name one. On CPU this is the host device.
    assert execution["device"]


def test_the_manifest_records_which_tax_data_vintage_the_population_came_from(
    two_scenario_run, stand_in_runner, tmp_path
):
    """A run whose population is converted Tax-Data records the vintage.

    Without this a reader of an archived run could see only the folder the
    population was read from, and had to guess from its name which published
    Tax-Data the numbers rest on.
    """
    population = tmp_path / "population"
    population.mkdir()
    (population / "manifest.yaml").write_text(
        "source_vintage: /model_data/tax-data/2026-07-01\n"
        "converted_at: '2026-07-02T09:00:00+00:00'\n"
        "random_draw_seed: 5150\n"
    )
    written = execute_run(
        two_scenario_run, population_root=population, group_runner=stand_in_runner
    )
    provenance = read_manifest(written)["execution"]["population_provenance"]
    assert provenance["population_form"] == "bridge"
    assert provenance["source_vintage"] == "/model_data/tax-data/2026-07-01"
    assert provenance["bridge_converted_at"] == "2026-07-02T09:00:00+00:00"
    assert provenance["random_draw_seed"] == 5150


def test_a_population_folder_that_cannot_be_read_still_leaves_a_manifest(finished_run):
    """The rest of the manifest is written even when the population folder is gone."""
    provenance = read_manifest(finished_run)["execution"]["population_provenance"]
    assert provenance == {"population_form": "not readable at the path recorded here"}


def test_the_manifest_round_trips_through_json(finished_run):
    manifest = read_manifest(finished_run)
    assert json.loads(json.dumps(manifest)) == manifest


def test_each_scenario_keeps_its_own_manifest_naming_its_law(finished_run):
    per_scenario = json.loads(
        (finished_run / "scenarios" / "sd_bump" / "manifest.json").read_text()
    )
    assert per_scenario["scenario_id"] == "sd_bump"
    assert per_scenario["legs"]["tax_law"] == TEST_REFORM
    assert per_scenario["years"] == [2026, 2027]
    assert per_scenario["scenario"]["law_layers"] == [TEST_REFORM]


# --------------------------------------------------------------------------
# What is inside the tables
# --------------------------------------------------------------------------


def test_the_totals_table_carries_the_world_levels_and_the_channel_effects(finished_run):
    totals = pd.read_csv(finished_run / "scenarios" / "sd_bump" / "totals.csv")
    assert set(totals.columns) == {
        "year",
        "kind",
        "line",
        "description",
        "revenue_column",
        "dollars",
    }
    assert set(totals["kind"]) == {"level", "difference"}
    assert set(totals["year"]) == {2026, 2027}
    # Every level is one of the five worlds, called by its name, and every
    # difference names the two worlds it subtracts.
    levels = totals[totals["kind"] == "level"]
    assert set(levels["line"]) == set(world_names.WORLD_NAMES)
    differences = totals[totals["kind"] == "difference"]
    assert set(differences["line"]) == {
        revenue.channel_name(channel)
        for channel in (*revenue.CHANNEL_DIFFERENCES, revenue.TOTAL_DIFFERENCE)
    }


def test_the_marginal_rate_table_carries_the_reporting_layers_own_columns(finished_run):
    assert_marginal_rate_columns_are_the_reporting_layers()
    table = pd.read_parquet(
        finished_run / "scenarios" / "sd_bump" / "mtr_table.parquet"
    )
    assert list(table["year"]) == [2026, 2027]


def test_a_three_year_run_writes_one_distribution_row_per_group_tax_spread_and_year(
    tmp_path, stand_in_runner
):
    runscript = tmp_path / "three_years.csv"
    runscript.write_text(
        RUNSCRIPT_HEADER + f"sd_bump,{TEST_REFORM},default,default,2026:2028,\n"
    )
    plan = plan_run(
        runscript,
        output_root=tmp_path / "runs",
        code_version=FIXED_CODE_VERSION,
        started_at=FIXED_START,
    )
    written = execute_run(
        plan, population_root=tmp_path / "population", group_runner=stand_in_runner
    )
    folder = written / "scenarios" / "sd_bump"
    table = pd.read_parquet(folder / output_directory.DISTRIBUTION_FILE)

    assert list(table.columns) == list(distribution.table_columns())
    assert len(table) == (
        3
        * len(distribution.CORPORATE_SPREADS)
        * len(distribution.TAX_ROWS)
        * len(distribution.GROUPS)
    )
    keys = list(distribution.KEY_COLUMNS)
    assert not table.duplicated(keys).any()
    assert set(table["year"]) == {2026, 2027, 2028}
    assert set(table["tax"]) == set(distribution.TAX_ROWS)
    assert set(table["corporate_spread"]) == set(distribution.CORPORATE_SPREADS)


def test_a_scenario_without_a_distribution_table_says_why_in_the_manifest(
    two_scenario_run, tmp_path
):
    def runner_without_the_table(scenarios, provenance, *, deliver, population, config_root):
        deliver(
            tuple(
                ScenarioResult(
                    spec=spec,
                    projection=dataclasses.replace(
                        stand_in_projection(tuple(spec.row.years)),
                        distribution_table=None,
                    ),
                    provenance=provenance[spec.id],
                )
                for spec in scenarios
            )
        )
        return dict(UNMEASURED_COUNTERS)

    written = execute_run(
        two_scenario_run,
        population_root=tmp_path / "population",
        group_runner=runner_without_the_table,
    )
    assert not (written / "scenarios" / "sd_bump" / "distribution.parquet").exists()
    absent = [
        entry for entry in read_manifest(written)["outputs"] if not entry["written"]
    ]
    assert {entry["path"] for entry in absent} == {
        "scenarios/baseline/distribution.parquet",
        "scenarios/sd_bump/distribution.parquet",
    }
    assert {entry["not_written_because"] for entry in absent} == {
        output_directory.DISTRIBUTION_NOT_BUILT
    }


# --------------------------------------------------------------------------
# What a stack's run will look like, before the stack fold exists
# --------------------------------------------------------------------------


@pytest.fixture
def run_of_a_hand_built_stack(tmp_path, stand_in_runner):
    """A two-provision stack, assembled by hand where WP3's fold will assemble it.

    The stack expansion is another work package's, and its placeholder refuses a
    stack row rather than guessing. That would leave the manifest's stack
    section, the verbatim stack-file copy and the same-subparameter conflict
    records untested until it lands, so this builds the expansion's output
    directly — exactly the shape the seam promises — and runs the writer on it.
    """
    runscript = tmp_path / "ctc.csv"
    runscript.write_text(RUNSCRIPT_HEADER + f"ctc,decomposition.yaml,default,default,{YEARS},\n")
    stack_file = tmp_path / "decomposition.yaml"
    stack_file.write_text("mode: stacked\nprovisions: [max_value, refundability]\n")

    row = read_runscript(runscript)[0]
    provisions = ("max_value", "refundability")
    scenarios = tuple(
        ScenarioSpec(
            id=f"ctc.{provision}",
            row=row,
            law_layers=provisions[: position],
            stack=StackMembership(
                row_id="ctc",
                stack_file=stack_file,
                provision=provision,
                position=position,
            ),
        )
        for position, provision in enumerate(provisions, start=1)
    )
    plan = RunPlan(
        runscript_path=runscript,
        scenarios=scenarios,
        stacks=(
            StackPlan(
                row_id="ctc",
                stack_file=stack_file,
                provisions=provisions,
                scenario_ids=tuple(spec.id for spec in scenarios),
                increments=(
                    law_stacks.ProvisionIncrement("max_value", "ctc.max_value", None),
                    law_stacks.ProvisionIncrement(
                        "refundability", "ctc.refundability", "ctc.max_value"
                    ),
                ),
                conflicts=(
                    SubparameterConflict(
                        subparameter="ctc.max_value",
                        winning_provision="refundability",
                        displaced_provisions=("max_value",),
                    ),
                ),
            ),
        ),
        config_root=DEFAULT_CONFIG_ROOT,
        run_directory=tmp_path / "runs" / "ctc" / "2026-08-03T143005_abcdef12",
        code_version=FIXED_CODE_VERSION,
    )
    # The two cumulative laws are not real reform folders here, so provenance
    # resolution would have nothing to resolve; the run is written from the
    # stand-in results.
    provenance = {spec.id: _provenance_of_nothing() for spec in scenarios}
    # The stand-in group runner hands finished groups to `deliver`; the run is
    # then written the way the driver writes it — begin, one call per
    # scenario, finish.
    delivered: dict[str, ScenarioResult] = {}

    def keep(results):
        delivered.update({result.spec.id: result for result in results})

    counters = stand_in_runner(
        scenarios,
        provenance,
        deliver=keep,
        population=tmp_path,
        config_root=plan.config_root,
    )
    state = output_directory.begin_run(plan)
    aggregates = {
        spec.id: output_directory.write_scenario_results(
            state, delivered[spec.id], tmp_path, plan.config_root
        )
        for spec in scenarios
    }
    return output_directory.finish_run(
        state, plan, aggregates, dict(counters), tmp_path, {}
    )


def _provenance_of_nothing():
    """Resolved legs for a scenario whose law folders are made up for the test."""
    from taxmodel.scenario.provenance import Provenance

    return Provenance(
        law=None, economy=None, behavior=None, vintages={}, staleness=None, config=None
    )


def test_a_stacks_scenarios_say_which_stack_and_position_they_came_from(
    run_of_a_hand_built_stack,
):
    manifest = read_manifest(run_of_a_hand_built_stack)
    assert [entry["id"] for entry in manifest["scenarios"]] == [
        "ctc.max_value",
        "ctc.refundability",
    ]
    second = manifest["scenarios"][1]["from_stack"]
    assert second["provision"] == "refundability"
    assert second["position"] == 2


def test_the_stack_file_is_copied_in_verbatim(run_of_a_hand_built_stack):
    copied = run_of_a_hand_built_stack / "stacks" / "decomposition.yaml"
    assert "mode: stacked" in copied.read_text()


def test_the_manifest_records_every_same_subparameter_conflict(run_of_a_hand_built_stack):
    """Two provisions replacing one subparameter is routine; both replacements are recorded."""
    (stack,) = read_manifest(run_of_a_hand_built_stack)["stacks"]
    assert stack["provisions_in_order"] == ["max_value", "refundability"]
    assert stack["adjacent_differences"][0]["subtrahend"] == "baseline"
    (conflict,) = stack["same_subparameter_conflicts"]
    assert conflict["winning_provision"] == "refundability"
    assert conflict["displaced_provisions"] == ["max_value"]


def test_the_stacked_table_lands_at_the_run_root_with_a_row_per_provision(
    run_of_a_hand_built_stack,
):
    """A stacked estimate belongs to the run, not to any one rung of the ladder."""
    table = pd.read_csv(run_of_a_hand_built_stack / "stacked" / "ctc.csv")
    assert list(table.columns) == ["channel", "provision", "2026", "2027"]
    assert set(table["provision"]) == {"max_value", "refundability", "Total"}

    manifest = read_manifest(run_of_a_hand_built_stack)
    (entry,) = [row for row in manifest["outputs"] if row["path"] == "stacked/ctc.csv"]
    assert entry["written"] is True
    assert entry["rows"] == len(table)

    # No cell carries a numpy type's repr; every number reads as a number.
    for column in ("2026", "2027"):
        assert table[column].dtype.kind == "f"


def test_the_stacked_revenue_estimates_attribute_the_whole_budget(
    run_of_a_hand_built_stack,
):
    """Each provision's row is the step between adjacent rungs' estimates.

    The stand-in projections carry zeros, so every step is zero; what the test
    pins is the file's existence, its shape — the revenue-estimates columns
    behind a leading provision column — and that both provisions of the ladder
    appear, which is the attribution structure R's stacked workbook carries.
    """
    table = pd.read_csv(
        run_of_a_hand_built_stack / "stacked" / "ctc_revenue_estimates.csv"
    )
    assert list(table.columns) == [
        "provision",
        "year",
        "world",
        "line",
        "dollars",
        "share_of_gdp",
    ]
    assert set(table["provision"]) == {"max_value", "refundability"}
    assert (table["dollars"] == 0.0).all()

    manifest = read_manifest(run_of_a_hand_built_stack)
    (entry,) = [
        row
        for row in manifest["outputs"]
        if row["path"] == "stacked/ctc_revenue_estimates.csv"
    ]
    assert entry["written"] is True
    assert entry["rows"] == len(table)


def test_revenue_estimates_are_differenced_by_year_world_and_line():
    minuend = pd.DataFrame(
        [
            (2030, "baseline", "total_budget_effect", 10.0, 0.010),
            (2030, "conventional", "total_budget_effect", 120.0, 0.120),
            (2031, "baseline", "total_budget_effect", 20.0, 0.020),
            (2031, "conventional", "total_budget_effect", 140.0, 0.140),
        ],
        columns=["year", "world", "line", "dollars", "share_of_gdp"],
    )
    subtrahend = pd.DataFrame(
        [
            (2031, "conventional", "total_budget_effect", 40.0, 0.040),
            (2030, "baseline", "total_budget_effect", 1.0, 0.001),
            (2031, "baseline", "total_budget_effect", 4.0, 0.004),
            (2030, "conventional", "total_budget_effect", 20.0, 0.020),
        ],
        columns=["year", "world", "line", "dollars", "share_of_gdp"],
    )

    difference = run_tables._difference_revenue_estimates(minuend, subtrahend)
    indexed = difference.set_index(["year", "world", "line"])

    assert indexed.loc[(2030, "baseline", "total_budget_effect"), "dollars"] == 9.0
    assert indexed.loc[(2030, "conventional", "total_budget_effect"), "dollars"] == 100.0
    assert indexed.loc[(2031, "baseline", "total_budget_effect"), "dollars"] == 16.0
    assert indexed.loc[(2031, "conventional", "total_budget_effect"), "dollars"] == 100.0
    assert indexed.loc[(2031, "conventional", "total_budget_effect"), "share_of_gdp"] == 0.1


# --------------------------------------------------------------------------
# The split writers: begin, write each scenario, finish
# --------------------------------------------------------------------------


def run_in_three_calls(plan, stand_in_runner, population_root):
    """Drive the incremental path by hand: begin, write each scenario, finish.

    This is what the rewritten driver does for each group of scenarios; here
    the stand-in results are all in hand up front, so each scenario is written
    in its own call and only its aggregates are kept, exactly as the driver
    will keep them.
    """
    provenance = {spec.id: _provenance_of_nothing() for spec in plan.scenarios}
    results: dict[str, ScenarioResult] = {}

    def keep(group):
        results.update({result.spec.id: result for result in group})

    stand_in_runner(
        plan.scenarios,
        provenance,
        deliver=keep,
        population=population_root,
        config_root=plan.config_root,
    )
    state = output_directory.begin_run(plan)
    aggregates = {
        spec.id: output_directory.write_scenario_results(
            state, results[spec.id], population_root, plan.config_root
        )
        for spec in plan.scenarios
    }
    final = output_directory.finish_run(
        state,
        plan,
        aggregates,
        dict(UNMEASURED_COUNTERS),
        population_root,
        {},
    )
    return state, aggregates, final


def test_the_split_writers_finish_the_run_at_its_permanent_home(
    two_scenario_run, stand_in_runner, tmp_path, monkeypatch
):
    # Outside Slurm the run is written directly at its permanent home.
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    state, _aggregates, final = run_in_three_calls(
        two_scenario_run, stand_in_runner, tmp_path / "population"
    )
    assert state.staging_root is None
    assert final == two_scenario_run.run_directory
    assert final.is_dir()
    assert not final.with_name(final.name + output_directory.PARTIAL_SUFFIX).exists()
    # Spot-check the layout: the run's inputs, one scenario folder per
    # scenario, and the run manifest written last.
    assert (final / "runscript.csv").is_file()
    assert (final / "scenarios" / "baseline" / "totals.csv").is_file()
    assert (final / "scenarios" / "sd_bump" / "revenue_estimates.csv").is_file()
    assert (final / "manifest.json").is_file()


def test_finish_run_puts_the_handed_in_counters_in_the_manifest(
    two_scenario_run, stand_in_runner, tmp_path, monkeypatch
):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    _state, _aggregates, final = run_in_three_calls(
        two_scenario_run, stand_in_runner, tmp_path / "population"
    )
    counters = read_manifest(final)["execution"]["counters"]
    assert counters == dict(UNMEASURED_COUNTERS)


def test_write_scenario_results_retains_what_the_end_of_the_run_needs(
    two_scenario_run, stand_in_runner, tmp_path, monkeypatch
):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    _state, aggregates, final = run_in_three_calls(
        two_scenario_run, stand_in_runner, tmp_path / "population"
    )
    retained = aggregates["sd_bump"]
    assert retained.spec.id == "sd_bump"
    assert retained.years == (2026, 2027)
    # The channel effects are host numpy arrays, never device arrays.
    for totals in retained.channel_totals.values():
        for values in totals.values():
            assert type(values).__module__ == "numpy"
    for values in retained.total_effect.values():
        assert type(values).__module__ == "numpy"
    # The retained revenue-estimates frame is the one on disk: same rows,
    # same columns, built once and used for both.
    on_disk = pd.read_csv(final / "scenarios" / "sd_bump" / "revenue_estimates.csv")
    assert list(retained.revenue_estimates.columns) == list(on_disk.columns)
    assert len(retained.revenue_estimates) == len(on_disk)


# --------------------------------------------------------------------------
# Staging on the Slurm node's local drive
# --------------------------------------------------------------------------


def ample_free_space(path):
    """A stand-in for shutil.disk_usage reporting more than staging requires."""
    return types.SimpleNamespace(free=output_directory.STAGING_MINIMUM_FREE_BYTES + 1)


def test_no_slurm_job_means_no_staging(tmp_path, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    assert output_directory.staging_root_or_none(tmp_path / "runs" / "smoke" / "x") is None


def test_a_slurm_job_without_a_usable_tmpdir_means_no_staging(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    monkeypatch.delenv("TMPDIR", raising=False)
    assert output_directory.staging_root_or_none(tmp_path / "runs" / "smoke" / "x") is None
    monkeypatch.setenv("TMPDIR", str(tmp_path / "does-not-exist"))
    assert output_directory.staging_root_or_none(tmp_path / "runs" / "smoke" / "x") is None


def test_a_full_node_drive_means_no_staging(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    monkeypatch.setenv("TMPDIR", str(tmp_path))

    def nearly_full(path):
        return types.SimpleNamespace(free=output_directory.STAGING_MINIMUM_FREE_BYTES - 1)

    monkeypatch.setattr(output_directory.shutil, "disk_usage", nearly_full)
    assert output_directory.staging_root_or_none(tmp_path / "runs" / "smoke" / "x") is None


def test_a_slurm_job_with_room_stages_under_tmpdir(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setattr(output_directory.shutil, "disk_usage", ample_free_space)
    staging = output_directory.staging_root_or_none(tmp_path / "runs" / "smoke" / "x")
    assert staging is not None
    assert staging.is_relative_to(tmp_path)
    # The runscript-stem folder's name keeps two different runscripts staged
    # in the same job apart, and the job id keeps two jobs sharing a node apart.
    assert staging.name == "smoke"
    assert staging.parent.name == "123456"


def test_a_staged_run_ends_up_at_its_permanent_home_with_no_staged_copy_left(
    two_scenario_run, stand_in_runner, tmp_path, monkeypatch
):
    node_drive = tmp_path / "node_local_drive"
    node_drive.mkdir()
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    monkeypatch.setenv("TMPDIR", str(node_drive))
    monkeypatch.setattr(output_directory.shutil, "disk_usage", ample_free_space)

    state, _aggregates, final = run_in_three_calls(
        two_scenario_run, stand_in_runner, tmp_path / "population"
    )

    # The run was built on the node drive, not at the permanent home.
    assert state.staging_root is not None
    assert state.partial_path.is_relative_to(node_drive)
    # The finished tree is at the permanent home, complete and unsuffixed.
    assert final == two_scenario_run.run_directory
    assert final.is_dir()
    assert (final / "manifest.json").is_file()
    assert (final / "scenarios" / "sd_bump" / "totals.csv").is_file()
    assert not final.with_name(final.name + output_directory.PARTIAL_SUFFIX).exists()
    # The staged copy is gone; nothing on the node drive still holds the run.
    assert not state.partial_path.exists()
    assert list(node_drive.rglob(f"*{output_directory.PARTIAL_SUFFIX}")) == []


def test_the_wealth_diagnostics_file_holds_all_five_diagnostic_tables(finished_run):
    table = pd.read_csv(
        finished_run / "scenarios" / "baseline" / "wealth_diagnostics.csv"
    )
    assert set(table["table"]) == {
        "shortfall totals",
        "cell rates",
        "shortfall by age",
        "shortfall by percentile bin",
        "haircut",
    }
