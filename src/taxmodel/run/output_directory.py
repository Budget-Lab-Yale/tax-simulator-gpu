"""The run directory and its manifest.

The layout:

    {output root}/{runscript stem}/{date}T{time}_{short commit}/
      manifest.json                    the run-level manifest
      runscript.csv                    verbatim copy of the runscript as run
      stacks/{name}.yaml               verbatim copy of each stack file named
      scenarios/{ID}/
        manifest.json                  the per-scenario manifest
        totals.csv                     revenue totals by year and channel
        form1040.csv                   individual income tax, filers only
        payroll.csv                    payroll tax, every record
        estate.csv                     estate tax, every record
        wealth.csv                     wealth tax, every record
        distribution.parquet           the distribution table: taxes by
                                       year, corporate spread, group and
                                       tax (see below)
        wealth_diagnostics.csv         the wealth channel's diagnostic tables
        mtr_table.parquet              the average marginal rate table
        tax_law.csv                    the resolved law the scenario ran under
        receipts.csv                   federal receipts by fiscal year and world
        revenue_estimates.csv          those receipts less the baseline world's
        gains_cells.csv                the gains model's stock of unrealized
                                       gain by year and age cell, only when
                                       the scenario runs the gains model
        detail/{world}.parquet         only when the row carried a detail spec
      stacked/{stack row ID}.csv       only when a runscript row was a stack
      stacked/{stack row ID}_revenue_estimates.csv
                                       the same, for revenue estimates

The directory name carries the timestamp and the commit, so a rerun lands
beside its predecessor. The directory is built under a `{name}.partial` suffix
and renamed once everything is written, so a directory without the suffix is a
complete run.

The distribution table has one row per year, corporate spread, group and
tax. The `tax` column is `total` or one of income_tax, payroll, estate,
deemed_at_death, wealth and corporate, and every number on a row counts that
tax alone. Averages, net changes, the percent change in after-tax income and
effective rates add up across the six taxes to the `total` row; the shares
of households with a tax cut or increase do not. The employer's share of
payroll tax is counted in payroll, the estate tax and the tax on gains
deemed realized at death are charged to the heir, and the corporate income
tax is spread over households top-down, over the baseline world's bases in
every world.

The `corporate_spread` column says how the corporate tax is spread, and the
whole table is given once for each. Both split the tax between the
above-normal return, the normal return to capital and labor, and spread the
labor part over labor income. `income` spreads both capital parts over
capital income. `assets` spreads the above-normal part over corporate equity
(equities, defined-contribution balances, trusts and real estate funds, each
times the share of it that is corporate equity) and the normal-return part
over net capital (assets less debts, with owner-occupied homes counted at the
structure share of net home equity, floored at zero). The two versions are
different models: each ranks its Income and Accrual income cuts on its own
pretax income, which includes its own share of current-law corporate tax, so
the groups of those two cuts differ between versions.

The change columns compare the static world (the first_order world plus
each household's share of the reform's static change in corporate tax) with
the baseline, on cash income. The level columns give each group's tax,
pretax cash and accrual income, and effective tax rates in the baseline,
static and conventional worlds. `reporting/distribution.py` lists the
groups and columns.

A known limitation: the conventional world's individual taxes and dissaving
were computed on incomes the model's own corporate step had changed, while
the corporate tax is counted top-down. In a scenario raising the corporate
rate from 21 to 28 percent, this moved no group's conventional effective rate
by more than about a quarter of a percentage point.

A scenario whose run formed no distribution table writes no file, and the
manifest says why.

No `detail/` folder means no per-household columns were asked for. The
manifest's `outputs` section lists the files written, with row counts.

Most published tables are CSV. The per-household detail files, the
distribution table and the marginal rate table are Parquet, since they are
large and read by programs.

The files are written by a helper thread (`run/file_writer.py`) while the run
moves on to the next scenario. Everything is on disk before the run manifest
is written.

Every table names the five worlds baseline, first_order,
mechanical_ex_saving, mechanical, and conventional. `years/world_names.py`
maps each world to the (household frame, tax law) pair it stands for.
"""

from __future__ import annotations

import datetime as dt
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from taxmodel.behavior.config import BehaviorConfigError, read_behavior_leg
from taxmodel.config import leg_or_none
from taxmodel.config import manifest as manifest_module
from taxmodel.config.runscript import BASELINE_ID
from taxmodel.frame import prepare
from taxmodel.reporting import detail as detail_output
from taxmodel.reporting import run_tables, tax_totals
from taxmodel.reporting.run_tables import ScenarioAggregates
from taxmodel.run import file_writer

if TYPE_CHECKING:  # type names only; run/scenarios.py imports this module
    from taxmodel.run.plan import (
        RunPlan,
        ScenarioSpec,
        StackPlan,
    )
    from taxmodel.run.scenarios import ScenarioResult

# The date and time the run started, in local wall time, for the run folder's
# name.
TIMESTAMP_FORMAT = "%Y-%m-%dT%H%M%S"
# The length of the short commit hash in the folder name.
SHORT_COMMIT_CHARACTERS = 8
# Stands in for the commit when the source tree is not a git repository.
NO_COMMIT_LABEL = "nocommit"
# The suffix a run directory carries while it is being written.
PARTIAL_SUFFIX = ".partial"

# The free space the Slurm node's local drive must have before a run is built
# there instead of on the shared filesystem. A run directory with per-household
# detail files can reach tens of gigabytes, and filling the drive part way
# through would kill the run.
STAGING_MINIMUM_FREE_BYTES = 50 * 1024**3

MANIFEST_NAME = "manifest.json"
RUNSCRIPT_COPY_NAME = "runscript.csv"
STACKS_FOLDER = "stacks"
SCENARIOS_FOLDER = "scenarios"
STACKED_FOLDER = "stacked"
DETAIL_FOLDER = "detail"

TOTALS_FILE = "totals.csv"
# The two Parquet tables. Both are wide and read by programs.
DISTRIBUTION_FILE = "distribution.parquet"
MTR_TABLE_FILE = "mtr_table.parquet"
WEALTH_DIAGNOSTICS_FILE = "wealth_diagnostics.csv"
RECEIPTS_FILE = "receipts.csv"
REVENUE_ESTIMATES_FILE = "revenue_estimates.csv"

# The capital-gains model's account of the stock of unrealized gain, one row
# per year per age cell. Only a scenario that runs the gains model writes it.
GAINS_CELLS_FILE = "gains_cells.csv"

# The four tax-specific totals files, one per tax, written as CSV with a
# `world` column. Their contents are `reporting/tax_totals.py`'s.
TAX_TOTALS_FILES = ("form1040.csv", "payroll.csv", "estate.csv", "wealth.csv")

# The resolved year-by-parameter law the scenario ran under.
TAX_LAW_FILE = "tax_law.csv"

# The CSV marker for a missing number, such as a bracket slot the tax law does
# not use or an average over a group that holds nobody.
CSV_MISSING = "NA"

# Manifest text for a requested per-household detail file that was not written.
DETAIL_ASKED_FOR_BUT_ABSENT = (
    "the runscript row asked for this detail file and the run did not write "
    "it; the run's numbers stand but this world's households were not exported"
)
RECEIPTS_NOT_BUILT = (
    "the projection did not total the calendar-year payment columns receipts "
    "turns into fiscal-year collections, so no receipts file could be built; "
    "reporting/receipts.py's MODEL_COLUMNS names what receipts needs and "
    "reporting/assemble.py's REVENUE_COLUMNS names what the projection totals today"
)
DISTRIBUTION_NOT_BUILT = (
    "the run formed no distribution table for this scenario, so there was "
    "nothing to write; the table is built on the device by "
    "reporting/device_sums.py's distribution_table_for_year"
)


# --------------------------------------------------------------------------
# Naming the directory and building it under a partial name
# --------------------------------------------------------------------------


def run_directory_path(
    output_root: str | Path,
    runscript_stem: str,
    commit: str | None,
    started_at: dt.datetime | None = None,
) -> Path:
    """The run directory path: output root, runscript stem, stamp."""
    when = started_at or dt.datetime.now().astimezone()
    short_commit = (
        commit[:SHORT_COMMIT_CHARACTERS] if commit else NO_COMMIT_LABEL
    )
    return (
        Path(output_root)
        / runscript_stem
        / f"{when.strftime(TIMESTAMP_FORMAT)}_{short_commit}"
    )


@dataclass
class Building:
    """A run directory under construction and its final location."""

    partial_path: Path
    final_path: Path
    # The directory under $TMPDIR holding the partial directory when the run
    # is built on the Slurm node's local drive. `finish_run` copies the
    # finished tree to the permanent root and removes the staged copy. None
    # means the run is written directly at its permanent home.
    staging_root: Path | None = None
    # Every file written so far, relative to the run directory, with its row
    # count. The row count is None for the manifests and the verbatim copies of
    # the runscript and stack files.
    written: list[dict[str, Any]] = field(default_factory=list)
    # The helper thread that writes the files. A file is recorded as written
    # when it is handed over. `finish_run` waits for the thread before writing
    # the run manifest.
    writer: file_writer.BackgroundWriter = field(
        default_factory=file_writer.BackgroundWriter
    )
    # The commit the run was planned under, repeated in every manifest this run
    # writes.
    code_version: dict[str, Any] = field(default_factory=dict)

    def write_csv(self, path: Path, table: pd.DataFrame) -> None:
        """Hand one CSV table to the writer and record it in the manifest."""
        self.writer.write_csv(path, table, missing=CSV_MISSING)
        self.record(path, rows=len(table))

    def write_parquet(self, path: Path, table: pd.DataFrame) -> None:
        """Hand one Parquet table to the writer and record it in the manifest."""
        self.writer.write_parquet(path, table)
        self.record(path, rows=len(table))

    def record(self, path: Path, rows: int | None) -> None:
        self.written.append(
            {
                "path": str(path.relative_to(self.partial_path)),
                "rows": None if rows is None else int(rows),
                "written": True,
            }
        )


def staging_root_or_none(final_path: str | Path) -> Path | None:
    """The staging directory on the Slurm node's local drive, or None.

    Writing many small files over NFS is slow. Inside a Slurm job, when
    `$TMPDIR` names an existing directory with at least
    `STAGING_MINIMUM_FREE_BYTES` free, this returns a directory under $TMPDIR
    to build the partial run directory in, and `finish_run` copies the
    finished tree home in one pass. Otherwise it returns None and the run is
    written at its permanent home.

    The returned directory carries the Slurm job id, which separates two jobs
    sharing a node, and the runscript-stem folder's name, which separates two
    runscripts staged in the same job.
    """
    final = Path(final_path)
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    tmpdir = os.environ.get("TMPDIR")
    if not tmpdir or not Path(tmpdir).is_dir():
        return None
    if shutil.disk_usage(tmpdir).free < STAGING_MINIMUM_FREE_BYTES:
        return None
    return Path(tmpdir) / "run-staging" / job_id / final.parent.name


def begin_run(plan: RunPlan) -> Building:
    """Create the partial run directory and copy the run's inputs into it.

    The directory is created at its permanent home, or on the Slurm node's
    local drive when `staging_root_or_none` returns one. A leftover partial
    directory from a failed attempt is removed first. The runscript and the
    stack files are copied in before any scenario runs, so an interrupted run's
    partial directory still says what was asked for.
    """
    final = Path(plan.run_directory)
    staging = staging_root_or_none(final)
    if staging is None:
        partial = final.with_name(final.name + PARTIAL_SUFFIX)
    else:
        partial = staging / (final.name + PARTIAL_SUFFIX)
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    state = Building(
        partial_path=partial,
        final_path=final,
        staging_root=staging,
        code_version=plan.code_version,
    )
    _copy_run_inputs(state, plan)
    return state


def _copy_run_inputs(state: Building, plan: RunPlan) -> None:
    """Copy the runscript and the stack files into the run directory verbatim."""
    shutil.copyfile(plan.runscript_path, state.partial_path / RUNSCRIPT_COPY_NAME)
    state.record(state.partial_path / RUNSCRIPT_COPY_NAME, rows=None)
    for stack in plan.stacks:
        folder = state.partial_path / STACKS_FOLDER
        folder.mkdir(exist_ok=True)
        copy = folder / Path(stack.stack_file).name
        shutil.copyfile(stack.stack_file, copy)
        state.record(copy, rows=None)


# --------------------------------------------------------------------------
# The planned output files
# --------------------------------------------------------------------------


def planned_output_names(plan: RunPlan) -> list[str]:
    """Every path a run of this plan will write, relative to the run directory.

    A dry run prints this list. A scenario whose runscript row named a detail
    spec gets one line per world it asked for.
    """
    names = [MANIFEST_NAME, RUNSCRIPT_COPY_NAME]
    for stack in plan.stacks:
        names.append(f"{STACKS_FOLDER}/{Path(stack.stack_file).name}")
    for spec in plan.scenarios:
        folder = f"{SCENARIOS_FOLDER}/{spec.id}"
        names.append(f"{folder}/{MANIFEST_NAME}")
        names.append(f"{folder}/{TOTALS_FILE}")
        names.append(f"{folder}/{DISTRIBUTION_FILE}")
        names.append(f"{folder}/{WEALTH_DIAGNOSTICS_FILE}")
        names.extend(f"{folder}/{name}" for name in TAX_TOTALS_FILES)
        names.append(f"{folder}/{TAX_LAW_FILE}")
        names.append(f"{folder}/{RECEIPTS_FILE}")
        names.append(f"{folder}/{REVENUE_ESTIMATES_FILE}")
        names.append(f"{folder}/{MTR_TABLE_FILE}")
        if _planned_leg_binds_the_gains_model(plan.config_root, spec):
            names.append(f"{folder}/{GAINS_CELLS_FILE}")
        names.extend(detail_file_names(spec))
    for stack in plan.stacks:
        names.append(f"{STACKED_FOLDER}/{stack.row_id}.csv")
        names.append(f"{STACKED_FOLDER}/{stack.row_id}_revenue_estimates.csv")
    return names


def _planned_leg_binds_the_gains_model(config_root: str | Path, spec: ScenarioSpec) -> bool:
    """Whether this scenario's behavior leg, read from the configuration at
    planning time, binds the capital-gains model.

    A leg that cannot be read returns False. Whether a behavior leg is usable
    is checked when the scenario's provenance is resolved.
    """
    try:
        return read_behavior_leg(config_root, spec.behavior).runs_the_gains_model
    except BehaviorConfigError:
        return False


def detail_file_names(spec: ScenarioSpec) -> list[str]:
    """The per-household detail files one scenario writes, relative to the run.

    One file per world the detail spec asked for, under that scenario's
    folder. Empty when the runscript row named no detail spec.
    """
    if spec.detail is None:
        return []
    folder = f"{SCENARIOS_FOLDER}/{spec.id}/{DETAIL_FOLDER}"
    return [f"{folder}/{world}.parquet" for world in spec.detail.worlds]


# --------------------------------------------------------------------------
# Writing the run
# --------------------------------------------------------------------------


def write_scenario_results(
    state: Building,
    result: ScenarioResult,
    population_root: str | Path,
    config_root: str | Path,
) -> ScenarioAggregates:
    """Write one scenario's folder and return the aggregates the end of the run needs.

    The stacked tables and the run manifest read the returned aggregates in
    `finish_run`. Everything else about the scenario, including the projection
    and its device arrays, can be freed once this returns. The
    revenue-estimates frame is built once, written to the scenario's own file,
    and kept in the aggregates.
    """
    revenue_estimates = _write_scenario(state, result, population_root, config_root)
    projection = result.projection
    return ScenarioAggregates(
        spec=result.spec,
        years=tuple(int(year) for year in result.spec.row.years),
        # np.asarray brings any device array to host; a plain numpy array
        # passes through unchanged.
        channel_totals={
            difference: {
                name: np.asarray(values) for name, values in totals.items()
            }
            for difference, totals in projection.channel_totals.items()
        },
        total_effect={
            name: np.asarray(values)
            for name, values in projection.total_effect.items()
        },
        revenue_estimates=revenue_estimates,
    )


def finish_run(
    state: Building,
    plan: RunPlan,
    aggregates: dict[str, ScenarioAggregates],
    counters: dict[str, Any],
    population_root: str | Path,
    execution: dict[str, Any],
) -> Path:
    """Write the run-level outputs and move the directory to its permanent home.

    This writes the stacked tables, the manifest entries for files the run
    declared but did not write, and the run-level manifest, whose counters
    section is the `counters` dict handed in. The helper thread is waited on
    before the manifest is written, so the manifest lists only files that are
    on disk, and a write that failed anywhere in the run raises here. A run
    staged on the Slurm node's local drive is then copied to the permanent
    root. The directory loses its `.partial` suffix in a single rename.
    Returns the finished run directory.
    """
    # The stacked tables sit at the run root, since a stacked estimate spans
    # several scenarios. They are small and are written here on the main
    # thread, not by the helper thread.
    for row_id, table in run_tables.stacked_tables(plan, aggregates).items():
        path = state.partial_path / STACKED_FOLDER / f"{row_id}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        table.write_csv(path)
        state.record(path, rows=len(table.rows))

    for row_id, frame in run_tables.stacked_revenue_estimates(
        plan, aggregates
    ).items():
        path = state.partial_path / STACKED_FOLDER / f"{row_id}_revenue_estimates.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, na_rep=CSV_MISSING)
        state.record(path, rows=len(frame))

    started_waiting = time.perf_counter()
    writing_seconds = state.writer.finish()
    waited = time.perf_counter() - started_waiting
    print(
        f"host stopwatch: the file writer spent {writing_seconds:.1f} s putting "
        f"files on disk, and the run waited {waited:.1f} s for it at the end"
    )

    not_written = _declared_but_not_written(
        plan, {entry["path"] for entry in state.written}
    )
    manifest = build_run_manifest(
        plan, state, not_written, population_root, execution, counters
    )
    manifest_module.write_manifest(state.partial_path / MANIFEST_NAME, manifest)

    if state.staging_root is not None:
        # Copy the staged tree home under its `.partial` name, then rename.
        # The staged copy is removed last, since until the rename lands it is
        # the only complete copy of the run.
        home_partial = state.final_path.with_name(
            state.final_path.name + PARTIAL_SUFFIX
        )
        if home_partial.exists():
            shutil.rmtree(home_partial)
        home_partial.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(state.partial_path, home_partial)
        home_partial.rename(state.final_path)
        shutil.rmtree(state.partial_path)
    else:
        state.partial_path.rename(state.final_path)
    return state.final_path


def _write_scenario(
    state: Building,
    result: ScenarioResult,
    population_root: str | Path,
    config_root: str | Path,
) -> pd.DataFrame | None:
    """Write one scenario's folder: its tables, then its own manifest.

    Returns the revenue-estimates frame it wrote, for the stacked revenue
    estimates, or None when receipts could not be built for this scenario.
    """
    folder = state.partial_path / SCENARIOS_FOLDER / result.spec.id
    folder.mkdir(parents=True)
    projection = result.projection

    for name, table in (
        (TOTALS_FILE, run_tables.totals_table(projection)),
        (WEALTH_DIAGNOSTICS_FILE, run_tables.wealth_diagnostics_table(projection)),
    ):
        state.write_csv(folder / name, table)

    # The distribution table is the widest table a scenario writes.
    distribution_table = run_tables.distribution_table(projection)
    if distribution_table is not None:
        state.write_parquet(folder / DISTRIBUTION_FILE, distribution_table)

    # The tax-specific totals: one CSV file per tax, one row per world per
    # year.
    for name, table in tax_totals.files_and_tables(projection):
        state.write_csv(folder / name, table)

    state.write_csv(folder / TAX_LAW_FILE, run_tables.law_table(projection))

    written_receipts = run_tables.receipts_tables(
        projection,
        [int(year) for year in result.spec.row.years],
        leg_or_none(result.spec.economy),
        config_root,
    )
    if written_receipts is not None:
        receipts_table, revenue_estimates_table = written_receipts
        state.write_csv(folder / RECEIPTS_FILE, receipts_table)
        state.write_csv(folder / REVENUE_ESTIMATES_FILE, revenue_estimates_table)

    # The marginal rate table is wide and read by programs.
    state.write_parquet(folder / MTR_TABLE_FILE, run_tables.marginal_rate_table(projection))

    # Only a scenario that runs the gains model writes the gains cells file.
    if projection.gains_cell_tables is not None:
        state.write_csv(folder / GAINS_CELLS_FILE, run_tables.gains_cells_table(projection))

    # One Parquet file per requested world, under this scenario's folder. The
    # tables come from `reporting/detail.py`. Each world's table is built only
    # after the previous one is on disk: a table holds a row per household per
    # year, about a gigabyte over a thirty-year run.
    if projection.detail is not None:
        for described, table in detail_output.tables_to_write(
            projection.detail, folder / DETAIL_FOLDER
        ):
            state.write_parquet(described.path, table)
            del table
            state.writer.wait_until_written()

    manifest = manifest_module.build_manifest(
        scenario_id=result.spec.id,
        years=[int(year) for year in result.spec.row.years],
        economy=result.provenance.economy,
        behavior=result.provenance.behavior,
        law=result.provenance.law,
        vintages=result.provenance.vintages,
        staleness=result.provenance.staleness,
        code_version=state.code_version or None,
    )
    manifest["scenario"] = {
        "law_layers": list(result.spec.law_layers),
        "population_root": str(population_root),
        "detail_spec": (
            None if result.spec.detail is None else str(result.spec.detail.path)
        ),
        "run_directory": str(state.final_path),
    }
    # The helper thread writes in order, so the scenario's manifest lands
    # after the tables it describes.
    manifest_path = folder / MANIFEST_NAME
    state.writer.write_with(
        manifest_path,
        lambda: manifest_module.write_manifest(manifest_path, manifest),
    )
    state.record(manifest_path, rows=None)
    return None if written_receipts is None else written_receipts[1]


def _declared_but_not_written(
    plan: RunPlan, already_written: set[str]
) -> list[dict[str, Any]]:
    """The layout's files that this run did not produce, each with the reason.

    `already_written` is every path the run did write, so that a file appears
    in the manifest once.
    """
    entries: list[dict[str, Any]] = []
    for spec in plan.scenarios:
        folder = f"{SCENARIOS_FOLDER}/{spec.id}"
        # A requested detail file that is not on disk is recorded as absent.
        entries.extend(
            {
                "path": name,
                "rows": None,
                "written": False,
                "not_written_because": DETAIL_ASKED_FOR_BUT_ABSENT,
            }
            for name in detail_file_names(spec)
            if name not in already_written
        )
        # When the projection did not total the columns receipts needs, both
        # receipts files are absent, and the manifest names the columns wanted.
        entries.extend(
            {
                "path": f"{folder}/{name}",
                "rows": None,
                "written": False,
                "not_written_because": RECEIPTS_NOT_BUILT,
            }
            for name in (RECEIPTS_FILE, REVENUE_ESTIMATES_FILE)
            if f"{folder}/{name}" not in already_written
        )
        if f"{folder}/{DISTRIBUTION_FILE}" not in already_written:
            entries.append(
                {
                    "path": f"{folder}/{DISTRIBUTION_FILE}",
                    "rows": None,
                    "written": False,
                    "not_written_because": DISTRIBUTION_NOT_BUILT,
                }
            )
    return entries


# --------------------------------------------------------------------------
# The run-level manifest
# --------------------------------------------------------------------------


def build_run_manifest(
    plan: RunPlan,
    state: Building,
    not_written: list[dict[str, Any]],
    population_root: str | Path,
    execution: dict[str, Any],
    counters: dict[str, Any],
) -> dict[str, Any]:
    """The run-level manifest.

    The per-scenario manifests say what law each scenario ran and what
    configuration was in force. This one says which runscript produced them,
    which scenarios came out of which stack and in what order, what
    per-household detail was asked for, every file the run wrote with its row
    count, and where and how the computation executed. The git state and
    runtime stack are repeated here.

    The population is recorded by folder and by provenance: which form of
    Tax-Data the folder holds and, for converted bridge output, which published
    Tax-Data vintage it was converted from, when, and under which random-draw
    seed.
    """
    runtime = manifest_module.runtime_stack()
    return {
        "written_at": dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds"),
        "runscript": {
            "path": str(plan.runscript_path),
            "stem": plan.stem,
            "verbatim_copy": RUNSCRIPT_COPY_NAME,
        },
        "scenarios": [_scenario_entry(spec) for spec in plan.scenarios],
        "stacks": [_stack_entry(stack) for stack in plan.stacks],
        "detail": [
            {
                "scenario_id": spec.id,
                "spec_file": str(spec.detail.path),
                "variables": list(spec.detail.variables),
                "worlds": list(spec.detail.worlds),
                "years": list(spec.detail.kept_years(spec.row.years)),
            }
            for spec in plan.scenarios
            if spec.detail is not None
        ],
        "outputs": [*state.written, *not_written],
        "execution": {
            "population_root": str(population_root),
            "population_provenance": prepare.population_provenance(population_root),
            **_where_it_ran(execution, runtime),
            "counters": dict(counters),
        },
        "code_version": plan.code_version or manifest_module.git_state(),
        "runtime": runtime,
    }


def _where_it_ran(execution: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    """Local or Slurm, and which card the numbers were computed on."""
    devices = runtime.get("jax_devices") or []
    return {
        "where": execution.get("where", "local"),
        "slurm_job_id": execution.get("slurm_job_id"),
        "device": devices[0] if devices else None,
    }


def _scenario_entry(spec: ScenarioSpec) -> dict[str, Any]:
    """One scenario in the manifest: its legs, its years, and its stack, if any.

    `economy` and `behavior` are the legs the scenario ran, which for a rung
    of a stacked reform may differ from the legs its runscript row named. The
    row's own cells are kept under `runscript_row`.
    """
    entry: dict[str, Any] = {
        "id": spec.id,
        "law_layers": list(spec.law_layers),
        "economy": spec.economy,
        "behavior": spec.behavior,
        "years": [int(year) for year in spec.row.years],
        "runscript_row": {
            "tax_law": spec.row.tax_law,
            "economy": spec.row.economy,
            "behavior": spec.row.behavior,
        },
        "from_stack": None,
    }
    if spec.stack is not None:
        entry["from_stack"] = {
            "row_id": spec.stack.row_id,
            "stack_file": str(spec.stack.stack_file),
            "provision": spec.stack.provision,
            "position": spec.stack.position,
        }
    return entry


def _stack_entry(stack: StackPlan) -> dict[str, Any]:
    """One stack in the manifest: its order and its conflicts.

    Each subparameter two provisions both replaced is recorded with the
    provision whose replacement won and the ones it displaced.
    """
    return {
        "row_id": stack.row_id,
        "stack_file": str(stack.stack_file),
        "provisions_in_order": list(stack.provisions),
        "scenario_ids_in_order": list(stack.scenario_ids),
        "adjacent_differences": [
            {
                "provision": increment.provision,
                "minuend": increment.minuend_id,
                "subtrahend": (
                    BASELINE_ID
                    if increment.subtrahend_id is None
                    else increment.subtrahend_id
                ),
            }
            for increment in stack.increments
        ],
        "same_subparameter_conflicts": [
            {
                "subparameter": conflict.subparameter,
                "winning_provision": conflict.winning_provision,
                "displaced_provisions": list(conflict.displaced_provisions),
            }
            for conflict in stack.conflicts
        ],
    }
