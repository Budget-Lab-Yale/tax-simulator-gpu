"""Planning a run: what the driver settles before anything is computed."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from taxmodel.config import ConfigError
from taxmodel.run import grouping, scenarios
from taxmodel.run import plan as run_plan
from tests.unit.run.conftest import RUNSCRIPT_HEADER

# The reform folder the tests point at, which the repository's own
# configuration tree holds: a $10,000 bump to the standard deduction.
TEST_REFORM = "tests/sd_bump_10k"

# A fixed commit and start time, so a planned run directory's name is
# predictable in a test rather than being whatever the clock said.
FIXED_CODE_VERSION = {"commit": "abcdef1234567890", "branch": "test", "dirty": False}
FIXED_START = dt.datetime(2026, 8, 3, 14, 30, 5, tzinfo=dt.UTC)


def write_runscript(tmp_path, rows, name="smoke.csv"):
    path = tmp_path / name
    path.write_text(RUNSCRIPT_HEADER + "".join(rows))
    return path


def plan(path, tmp_path, **kwargs):
    return run_plan.plan_run(
        path,
        output_root=tmp_path / "runs",
        code_version=FIXED_CODE_VERSION,
        started_at=FIXED_START,
        **kwargs,
    )


def test_a_plain_runscript_becomes_one_scenario_per_row(tmp_path):
    path = write_runscript(
        tmp_path,
        [
            "baseline,default,default,default,2026:2027,\n",
            f"sd_bump,{TEST_REFORM},no_saving_channel,default,2026:2027,\n",
        ],
    )
    planned = plan(path, tmp_path)
    assert [spec.id for spec in planned.scenarios] == ["baseline", "sd_bump"]
    assert planned.scenarios[0].law_layers == ()
    assert planned.scenarios[1].law_layers == (TEST_REFORM,)


def test_the_run_directory_carries_the_runscript_stem_the_time_and_the_commit(tmp_path):
    path = write_runscript(tmp_path, ["baseline,default,default,default,2026,\n"])
    planned = plan(path, tmp_path)
    assert planned.run_directory == (
        tmp_path / "runs" / "smoke" / "2026-08-03T143005_abcdef12"
    )


def test_a_run_directory_without_a_commit_says_so_in_its_name(tmp_path):
    """A source tree that is not a repository still gets a run folder, honestly named."""
    path = write_runscript(tmp_path, ["baseline,default,default,default,2026,\n"])
    planned = run_plan.plan_run(
        path,
        output_root=tmp_path / "runs",
        code_version={"commit": None},
        started_at=FIXED_START,
    )
    assert planned.run_directory.name.endswith("_nocommit")


def test_two_rows_with_one_scenario_id_are_refused(tmp_path):
    """Scenario IDs name folders in the run directory, so they have to be unique."""
    path = write_runscript(
        tmp_path,
        [
            "baseline,default,default,default,2026,\n",
            "baseline,default,default,default,2027,\n",
        ],
    )
    with pytest.raises(ConfigError, match="one scenario ID is used twice"):
        plan(path, tmp_path)


def test_a_row_with_no_scenario_id_is_refused(tmp_path):
    path = write_runscript(tmp_path, [",default,default,default,2026,\n"])
    with pytest.raises(ConfigError, match="a row has an empty ID cell"):
        plan(path, tmp_path)


def test_a_stack_row_expands_into_the_package_s_provisions(tmp_path):
    """One row naming a stack file becomes one scenario per provision, in order.

    The Clausing package is the repository's own stack file: seven tax-law
    provisions and an eighth that changes the corporate saving assumption
    instead. The eighth carries the seventh's law, which is why its
    `law_layers` list is the same seven folders.
    """
    path = write_runscript(
        tmp_path, ["clausing,clausing_v2.yaml,saving_50,kg_entity_shifting,2030,\n"]
    )
    planned = plan(path, tmp_path)

    assert [spec.id for spec in planned.scenarios] == [
        "clausing.01_clinton_rates",
        "clausing.02_restore_bottom_rates",
        "clausing.03_199a",
        "clausing.04_carryover_basis",
        "clausing.05_pref_rates",
        "clausing.06_niit_reform",
        "clausing.07_estate",
        "clausing.08_corporate",
    ]
    # Each provision carries the ones below it, so the eighth rung's law is the
    # seventh's and the economy leg is what moved.
    seventh, eighth = planned.scenarios[-2], planned.scenarios[-1]
    assert eighth.law_layers == seventh.law_layers
    assert eighth.economy == "clausing_corporate_saving_50"
    assert seventh.economy == "saving_50"
    # And the provisions the row did not speak about keep the row's behavior.
    assert all(spec.behavior == "kg_entity_shifting" for spec in planned.scenarios)

    (stack,) = planned.stacks
    assert stack.increments[0].subtrahend_id is None
    assert stack.increments[1].subtrahend_id == "clausing.01_clinton_rates"


def test_a_stack_file_that_does_not_read_is_refused_before_anything_runs(tmp_path):
    alternatives = tmp_path / "scenarios" / "tax_law" / "alternatives"
    alternatives.mkdir(parents=True)
    (alternatives / "ctc_decomposition.yaml").write_text("provisions: []\n")
    path = write_runscript(
        tmp_path, ["ctc,ctc_decomposition.yaml,default,default,2026,\n"]
    )
    with pytest.raises(ConfigError, match="missing required key `version`"):
        plan(path, tmp_path, config_root=tmp_path / "scenarios")


def test_a_detail_spec_is_read_and_attached_to_its_scenario(tmp_path):
    detail_root = tmp_path / "detail"
    detail_root.mkdir()
    (detail_root / "ctc_audit.yaml").write_text(
        "variables: [liab_iit, ctc_nonref]\nworlds: [baseline, first_order]\nyears: 2026\n"
    )
    path = write_runscript(
        tmp_path,
        ["baseline,default,default,default,2026:2027,ctc_audit.yaml\n"],
    )
    planned = plan(path, tmp_path, detail_root=detail_root)
    spec = planned.scenarios[0].detail
    assert spec is not None
    assert spec.variables == ("liab_iit", "ctc_nonref")
    assert spec.kept_years(planned.scenarios[0].row.years) == (2026,)


def test_a_broken_detail_spec_stops_the_plan_by_name(tmp_path):
    detail_root = tmp_path / "detail"
    detail_root.mkdir()
    (detail_root / "bad.yaml").write_text("variables: [liab_iit]\nworlds: [static]\n")
    path = write_runscript(
        tmp_path, ["baseline,default,default,default,2026,bad.yaml\n"]
    )
    with pytest.raises(ConfigError, match="worlds that do not exist: static"):
        plan(path, tmp_path, detail_root=detail_root)


def test_the_dry_run_description_names_the_scenarios_and_the_files(tmp_path):
    path = write_runscript(
        tmp_path,
        [
            "baseline,default,default,default,2026:2027,\n",
            f"sd_bump,{TEST_REFORM},default,default,2026:2027,\n",
        ],
    )
    lines = run_plan.describe_run_plan(plan(path, tmp_path))
    text = "\n".join(lines)
    assert "scenarios: 2" in text
    assert f"sd_bump: {TEST_REFORM}, 2026-2027" in text
    # The marginal rate table is written for every scenario.
    assert "scenarios/sd_bump/mtr_table.parquet" in text
    assert "scenarios/baseline/mtr_table.parquet" in text
    assert "scenarios/baseline/totals.csv" in text


def test_every_active_runscript_either_plans_or_says_which_folder_is_missing():
    """A dry run of everything in config/runscripts/ leaves nothing unexplained.

    Every active runscript is expected to plan, except test runscripts that
    name a missing reform folder on purpose. Those must be refused by name,
    not run with the scenario silently skipped.
    """
    active = [
        path
        for path in sorted(Path("config/runscripts").rglob("*.csv"))
        if "archive" not in path.parts
    ]
    # A guard against an empty glob, not a count to maintain.
    assert len(active) > 20
    refused: dict[Path, str] = {}
    for path in active:
        try:
            planned = run_plan.plan_run(
                path, output_root="/tmp/planned", code_version=FIXED_CODE_VERSION
            )
        except ConfigError as error:
            refused[path] = str(error)
            continue
        assert planned.scenarios, f"{path} planned no scenarios"
    for path, message in refused.items():
        assert "names neither a reform folder nor a stack file" in message, path
        assert path.parent == Path("config/runscripts/tests"), path


def test_economy_legs_that_resolve_alike_share_a_compiled_program_group(tmp_path):
    """Grouping matches what an economy leg resolves to, not the leg's name.

    `no_saving_channel` is the default economy with the financing profile set
    to none — same growth series, same heir and macro publications — so its
    scenarios must share a group with default-economy scenarios. The two
    reform rows here differ only in their economy cell.
    """
    gains_reform = "top_tax/dials/s_cg_r50"
    gains_leg = "kg_dynamics_only"
    path = write_runscript(
        tmp_path,
        [
            "baseline,default,no_saving_channel,,2025:2027,\n",
            f"rate_off_saving,{gains_reform},no_saving_channel,{gains_leg},2025:2027,\n",
            f"rate_default,{gains_reform},default,{gains_leg},2025:2027,\n",
        ],
    )
    planned = plan(path, tmp_path)
    groups = grouping.groups_that_can_share_a_program(planned.scenarios)
    assert len(groups) == 1
    (members,) = groups.values()
    assert {spec.id for spec in members} == {spec.id for spec in planned.scenarios}


def test_families_split_by_detail_and_the_gains_model(tmp_path):
    """Within one partition, three things separate the compiled programs.

    The law's structural-switch fingerprint, the per-household detail request,
    and whether the capital-gains model runs. Here all four scenarios share
    one partition (same window, same economy) and one law fingerprint, so the
    split falls on the other two: the scenario with a detail request and the
    one whose behavior leg binds the gains machinery each get a family of
    their own, and everyone else shares one.
    """
    from types import SimpleNamespace

    detail_root = tmp_path / "detail"
    detail_root.mkdir()
    (detail_root / "audit.yaml").write_text(
        "variables: [liab_iit]\nworlds: [baseline]\nyears: 2026\n"
    )
    path = write_runscript(
        tmp_path,
        [
            "baseline,default,default,default,2026:2027,\n",
            f"bump,{TEST_REFORM},default,default,2026:2027,\n",
            f"audited,{TEST_REFORM},default,default,2026:2027,audit.yaml\n",
            "gains,default,default,kg_dynamics_only,2026:2027,\n",
        ],
    )
    planned = plan(path, tmp_path, detail_root=detail_root)
    (((years, _economy), partition),) = grouping.groups_that_can_share_a_program(
        planned.scenarios
    ).items()
    # Stand-in resolved legs: only the last scenario binds the gains model.
    provenance = {
        spec.id: SimpleNamespace(
            behavior_leg=(
                SimpleNamespace(runs_the_gains_model=True)
                if spec.id == "gains"
                else None
            )
        )
        for spec in partition
    }
    # One fingerprint for all four: the repository's test laws flip no
    # structural switch, and this test is about the other two key parts.
    families = grouping.split_into_families(
        partition, ["plain"] * len(partition), provenance, years
    )

    members = [
        tuple(partition[position].id for position in family.positions)
        for family in families
    ]
    assert members == [("baseline", "bump"), ("audited",), ("gains",)]
    assert families[0].detail_request is None
    assert families[1].detail_request is not None
    assert [family.runs_the_gains_model for family in families] == [
        False,
        False,
        True,
    ]


def test_entity_shifting_without_the_gains_model_is_refused(tmp_path):
    """Entity shifting needs the gains model.

    The retained-earnings part of the shareholder price is the tax a deferred
    dollar bears, which only the gains model computes. The pre-run check stops
    a leg that turns the margin on without binding kg_dynamics before anything
    compiles.
    """
    from taxmodel.behavior import loader as behavior_loader

    leg = "tests/entity_shifting_without_gains"
    path = write_runscript(
        tmp_path,
        [f"es_alone,{TEST_REFORM},default,{leg},2026:2027,\n"],
    )
    planned = plan(path, tmp_path)
    with pytest.raises(behavior_loader.ManifestError, match="turns on entity shifting without"):
        run_plan.check_behavior_legs(planned)


def test_a_population_read_that_fails_raises_where_the_run_collects_it(tmp_path):
    """The background read is not a way for a bad folder to pass unnoticed."""
    pending = scenarios.PendingPopulation(tmp_path / "no such bridge output", (2027,))
    with pytest.raises(FileNotFoundError, match="not a population the model can read"):
        pending.block_until_read()


def test_an_abandoned_population_read_does_not_hold_the_run_up(tmp_path):
    """A run that fails before it collects the read stops waiting for it.

    `abandon` is what the failing run calls on its way out. It must not raise
    on its own account, and calling it twice -- once from the collection point,
    once from the cleanup -- must be harmless.
    """
    pending = scenarios.PendingPopulation(tmp_path / "no such bridge output", (2027,))
    pending.abandon()
    pending.abandon()
