"""One real scenario, start to finish: runscript in, run directory out.

Everything else in this folder substitutes a stand-in for the projection so the
tests stay quick. This one runs the model: the real frame steps, the real law
compiled from the repository's configuration tree, the real calculator chain and
wealth channel, the real reporting tables, and the real run directory. The
population is the three hand-written households the simulation's own integration
test uses, over two years, on CPU — the smallest thing that can still produce
every table the run directory holds.

It is the test that says the driver's pieces fit together with the model rather
than with a fixture.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from taxmodel.config import ConfigError
from taxmodel.reporting import revenue
from taxmodel.run import scenarios
from taxmodel.run.plan import plan_run

# The three-household population and its two years come from the simulation's
# integration test, which is where they are written out and explained.
from tests.unit.run.conftest import RUNSCRIPT_HEADER, read_table
from tests.unit.years.test_integration import YEARS, synthetic_population

# A $10,000 bump to the standard deduction: a reform folder the repository's
# configuration tree holds, whose effect these three households will show.
TEST_REFORM = "tests/sd_bump_10k"
FIXED_CODE_VERSION = {"commit": "0123456789ab", "branch": "test", "dirty": False}
FIXED_START = dt.datetime(2026, 8, 3, 15, 0, 0, tzinfo=dt.UTC)


@pytest.fixture(scope="module")
def finished_run(tmp_path_factory):
    """Plan and run one reform scenario on the three-household population."""
    tmp_path = tmp_path_factory.mktemp("real_run")
    runscript = tmp_path / "sd_bump.csv"
    years = f"{YEARS[0]}:{YEARS[-1]}"
    runscript.write_text(
        RUNSCRIPT_HEADER + f"sd_bump,{TEST_REFORM},default,default,{years},\n"
    )
    plan = plan_run(
        runscript,
        output_root=tmp_path / "runs",
        code_version=FIXED_CODE_VERSION,
        started_at=FIXED_START,
    )
    return scenarios.execute_run(
        plan,
        population_root=tmp_path / "three-households",
        population=synthetic_population(),
    )


def test_the_run_directory_holds_the_frozen_layout(finished_run):
    found = sorted(
        str(path.relative_to(finished_run)) for path in finished_run.rglob("*")
    )
    assert found == sorted(
        [
            "manifest.json",
            "runscript.csv",
            "scenarios",
            "scenarios/sd_bump",
            "scenarios/sd_bump/manifest.json",
            "scenarios/sd_bump/totals.csv",
            "scenarios/sd_bump/distribution.parquet",
            "scenarios/sd_bump/wealth_diagnostics.csv",
            "scenarios/sd_bump/mtr_table.parquet",
            "scenarios/sd_bump/form1040.csv",
            "scenarios/sd_bump/payroll.csv",
            "scenarios/sd_bump/estate.csv",
            "scenarios/sd_bump/wealth.csv",
            "scenarios/sd_bump/tax_law.csv",
            "scenarios/sd_bump/receipts.csv",
            "scenarios/sd_bump/revenue_estimates.csv",
        ]
    )


def test_the_manifest_row_counts_match_the_files_on_disk(finished_run):
    manifest = json.loads((finished_run / "manifest.json").read_text())
    tables = [
        entry
        for entry in manifest["outputs"]
        if entry["written"] and entry["rows"] is not None
    ]
    assert tables
    for entry in tables:
        assert entry["rows"] == len(read_table(finished_run / entry["path"]))


def test_the_standard_deduction_bump_cut_somebody_a_tax_cut(finished_run):
    """The run's own numbers, not a fixture's: a bigger deduction lowers receipts."""
    totals = pd.read_csv(finished_run / "scenarios" / "sd_bump" / "totals.csv")
    income_tax = totals[
        (totals["kind"] == "difference") & (totals["revenue_column"] == "liab_iit_net")
    ]
    assert (income_tax["dollars"] < 0).any()


def test_the_scenario_manifest_names_the_reform_that_ran(finished_run):
    per_scenario = json.loads(
        (finished_run / "scenarios" / "sd_bump" / "manifest.json").read_text()
    )
    assert per_scenario["legs"]["tax_law"] == TEST_REFORM
    assert per_scenario["law_replacements"], "a reform replaced at least one subparameter"


# --------------------------------------------------------------------------
# A stacked reform through the same driver path
# --------------------------------------------------------------------------

# A three-rung stack: ordinary rates up two points, then the standard deduction
# cut, then a rung that changes no tax law and turns the saving-financing
# channel off instead. The file explains itself.
TEST_STACK = "tests_two_provisions_then_economy.yaml"
STACK_ROW_ID = "provisions"
# The scenario IDs the stack expands into: the row's ID and the provision's own
# name.
STACK_SCENARIOS = (
    f"{STACK_ROW_ID}.1_ord_rates",
    f"{STACK_ROW_ID}.2_standard_deduction",
    f"{STACK_ROW_ID}.3_no_saving_channel",
)
# The economy leg the third rung asks for, which is not the one the runscript
# row names.
RUNG_ECONOMY = "no_saving_channel"
ROW_ECONOMY = "saving_50"

# A stack whose second rung reads other microdata than the rest of the run, for
# the test that the driver refuses that rather than running it on the wrong
# population.
STACK_PINNING_OTHER_MICRODATA = "tests_rung_pins_other_microdata.yaml"


@pytest.fixture(scope="module")
def finished_stack_run(tmp_path_factory):
    """Plan and run a whole stacked reform on the three-household population."""
    tmp_path = tmp_path_factory.mktemp("real_stack_run")
    runscript = tmp_path / "stack.csv"
    years = f"{YEARS[0]}:{YEARS[-1]}"
    runscript.write_text(
        RUNSCRIPT_HEADER + f"{STACK_ROW_ID},{TEST_STACK},{ROW_ECONOMY},default,{years},\n"
    )
    plan = plan_run(
        runscript,
        output_root=tmp_path / "runs",
        code_version=FIXED_CODE_VERSION,
        started_at=FIXED_START,
    )
    return scenarios.execute_run(
        plan,
        population_root=tmp_path / "three-households",
        population=synthetic_population(),
    )


def test_a_stacked_reform_runs_every_rung_and_writes_each_ones_folder(
    finished_stack_run,
):
    """The defect this covers: the driver refused to run any stack at all.

    Provenance was resolved through a property that raised as soon as a
    scenario carried more than one reform folder, which every rung of a stack
    does from the second one on, so a stacked runscript stopped before any
    arithmetic happened.
    """
    folders = sorted(
        path.name for path in (finished_stack_run / "scenarios").iterdir()
    )
    assert folders == sorted(STACK_SCENARIOS)
    for scenario_id in STACK_SCENARIOS:
        totals = pd.read_csv(
            finished_stack_run / "scenarios" / scenario_id / "totals.csv"
        )
        assert len(totals) > 0


def test_each_rung_ran_the_law_of_every_provision_below_it(finished_stack_run):
    """Rung two's law is both provisions folded, not the second folder alone."""
    manifests = {
        scenario_id: json.loads(
            (
                finished_stack_run / "scenarios" / scenario_id / "manifest.json"
            ).read_text()
        )
        for scenario_id in STACK_SCENARIOS
    }
    assert manifests[STACK_SCENARIOS[0]]["law_layers"] == [
        "default",
        "tests/stack_provisions/1_ord_rates",
    ]
    both_provisions = [
        "default",
        "tests/stack_provisions/1_ord_rates",
        "tests/stack_provisions/2_standard_deduction",
    ]
    assert manifests[STACK_SCENARIOS[1]]["law_layers"] == both_provisions
    # The third rung changes no tax law, so it runs the second rung's law.
    assert manifests[STACK_SCENARIOS[2]]["law_layers"] == both_provisions


def test_raising_rates_and_then_cutting_the_deduction_raises_more_than_either(
    finished_stack_run,
):
    """The run's own numbers: each rung's law really was the fold, not one folder.

    Both provisions raise the income tax of a household with taxable income, so
    the second rung — both of them together — has to collect more than the
    first.
    """

    def whole_effect_on_income_tax(scenario_id: str) -> float:
        totals = pd.read_csv(
            finished_stack_run / "scenarios" / scenario_id / "totals.csv"
        )
        wanted = totals[
            (totals["kind"] == "difference")
            & (totals["line"] == revenue.channel_name(revenue.TOTAL_DIFFERENCE))
            & (totals["revenue_column"] == "liab_iit_net")
        ]
        return float(wanted["dollars"].sum())

    rates_only = whole_effect_on_income_tax(STACK_SCENARIOS[0])
    both = whole_effect_on_income_tax(STACK_SCENARIOS[1])
    assert rates_only > 0
    assert both > rates_only


def test_the_rung_that_changed_the_economy_leg_is_recorded_as_running_it(
    finished_stack_run,
):
    """The manifest says what the rung ran, and separately what the row asked for.

    This is the defect that mattered most: the run manifest read the runscript
    row's economy cell, so the rung that changes the saving assumption would
    have been described as running the row's leg. The numbers would have been
    the rung's and the provenance beside them somebody else's.
    """
    manifest = json.loads((finished_stack_run / "manifest.json").read_text())
    entries = {entry["id"]: entry for entry in manifest["scenarios"]}
    assert entries[STACK_SCENARIOS[1]]["economy"] == ROW_ECONOMY
    changed = entries[STACK_SCENARIOS[2]]
    assert changed["economy"] == RUNG_ECONOMY
    # What the user wrote is still recorded, as what the user wrote.
    assert changed["runscript_row"]["economy"] == ROW_ECONOMY
    assert changed["runscript_row"]["tax_law"] == TEST_STACK

    # And the per-scenario manifest, which is built from the provenance the run
    # resolved rather than from the plan, agrees.
    per_scenario = json.loads(
        (
            finished_stack_run / "scenarios" / STACK_SCENARIOS[2] / "manifest.json"
        ).read_text()
    )
    assert per_scenario["legs"]["economy"] == RUNG_ECONOMY


def test_a_rung_pinning_other_microdata_stops_the_run_before_it_starts(tmp_path):
    """A run whose scenarios do not all read the same population is refused.

    Which Tax-Data vintage a scenario reads comes from its economy leg, and a
    stack rung may change that leg. The driver reads one population block and
    shares it, so the scenarios have to agree; before this was fixed the driver
    asked the runscript row instead of the rung, and a rung pinning other
    microdata would have run on the row's population without saying so.
    """
    runscript = tmp_path / "other_pin.csv"
    years = f"{YEARS[0]}:{YEARS[-1]}"
    runscript.write_text(
        RUNSCRIPT_HEADER + f"pinned,{STACK_PINNING_OTHER_MICRODATA},default,default,{years},\n"
    )
    plan = plan_run(
        runscript,
        output_root=tmp_path / "runs",
        code_version=FIXED_CODE_VERSION,
        started_at=FIXED_START,
    )
    with pytest.raises(ConfigError, match="pin different Tax-Data populations"):
        scenarios.execute_run(plan, population=synthetic_population())


# --------------------------------------------------------------------------
# Per-household detail through the same driver path
# --------------------------------------------------------------------------

# The repository's own detail spec: the income tax and the child credit's two
# halves, taken from the household frame under baseline law and under the
# reform's law.
DETAIL_SPEC = "ctc_audit.yaml"
DETAIL_WORLDS = ("baseline", "first_order")
DETAIL_COLUMNS = ("ctc_nonref", "ctc_ref", "liab_iit")
# The columns every detail file carries whether or not they were asked for.
DETAIL_ALWAYS = ("household_id", "year", "weight", "valid")


@pytest.fixture(scope="module")
def finished_detail_run(tmp_path_factory):
    """Run two scenarios, one of which asked for per-household detail."""
    tmp_path = tmp_path_factory.mktemp("real_detail_run")
    runscript = tmp_path / "detail.csv"
    years = f"{YEARS[0]}:{YEARS[-1]}"
    runscript.write_text(
        RUNSCRIPT_HEADER
        + f"plain,{TEST_REFORM},default,default,{years},\n"
        + f"audited,{TEST_REFORM},default,default,{years},{DETAIL_SPEC}\n"
    )
    plan = plan_run(
        runscript,
        output_root=tmp_path / "runs",
        code_version=FIXED_CODE_VERSION,
        started_at=FIXED_START,
    )
    return scenarios.execute_run(
        plan,
        population_root=tmp_path / "three-households",
        population=synthetic_population(),
    )


def test_the_run_writes_the_detail_files_the_runscript_asked_for(
    finished_detail_run,
):
    """The defect this covers: the batch runner dropped the request in silence.

    The request was read, checked and attached to the scenario, and then the
    batch path ran the scenario without it — no Parquet file and no complaint.
    A scenario that did not ask for detail still gets none, which is the other
    half of the same claim.
    """
    written = sorted(
        path.name
        for path in (finished_detail_run / "scenarios" / "audited" / "detail").iterdir()
    )
    assert written == [f"{world}.parquet" for world in DETAIL_WORLDS]
    assert not (finished_detail_run / "scenarios" / "plain" / "detail").exists()


def test_each_detail_file_holds_the_columns_and_years_asked_for(finished_detail_run):
    for world in DETAIL_WORLDS:
        table = pd.read_parquet(
            finished_detail_run / "scenarios" / "audited" / "detail" / f"{world}.parquet"
        )
        assert set(table.columns) == set(DETAIL_COLUMNS) | set(DETAIL_ALWAYS)
        assert sorted(table["year"].unique()) == list(YEARS)
        # Every household slot the run simulated gets a row in every year.
        assert len(table) % len(YEARS) == 0
        assert table["valid"].any(), "the run's households reached the detail file"


def test_the_detail_shows_the_reform_the_scenario_ran(finished_detail_run):
    """The two worlds differ by the tax change alone, and the change is visible.

    A $10,000 larger standard deduction lowers somebody's income tax, so the
    same household's `liab_iit` under the reform's law is below its baseline
    figure. That is what says these columns came from the run rather than from
    an empty array.
    """
    frames = {
        world: pd.read_parquet(
            finished_detail_run / "scenarios" / "audited" / "detail" / f"{world}.parquet"
        )
        for world in DETAIL_WORLDS
    }
    baseline_law = frames["baseline"]
    reform_law = frames["first_order"]
    assert list(baseline_law["household_id"]) == list(reform_law["household_id"])
    assert (reform_law["liab_iit"] < baseline_law["liab_iit"]).any()


def test_each_detail_file_appears_in_the_manifest_exactly_once(finished_detail_run):
    """The second defect, which the first one was hiding.

    The manifest lists the files a run wrote and then, separately, the files
    the layout reserves that it did not write. The detail files were on both
    lists — planned and absent, and also written — so as soon as detail
    started flowing the manifest contradicted itself about every one of them.
    """
    manifest = json.loads((finished_detail_run / "manifest.json").read_text())
    for world in DETAIL_WORLDS:
        path = f"scenarios/audited/detail/{world}.parquet"
        entries = [entry for entry in manifest["outputs"] if entry["path"] == path]
        assert len(entries) == 1, entries
        (entry,) = entries
        assert entry["written"] is True
        assert entry["rows"] == len(pd.read_parquet(finished_detail_run / path))
    # And the run manifest's own account of what was asked for still names the
    # spec file, the columns and the years.
    (asked,) = manifest["detail"]
    assert asked["scenario_id"] == "audited"
    assert asked["spec_file"].endswith(DETAIL_SPEC)
    assert asked["years"] == list(YEARS)
