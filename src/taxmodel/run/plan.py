"""Reading a runscript into a run plan.

This is the first step of a run, and all a dry run does. It reads the
runscript, resolves each row's tax_law cell into baseline law, a reform
folder, or a stack file, and its detail cell into a detail spec; expands every
stack row into the ordinary scenarios it stands for; checks the behavior legs;
and returns a `RunPlan`. `describe_run_plan` prints the plan, and
`population_for` names the bridge population the plan's economy legs pin.

Each expanded scenario carries two things. `ScenarioSpec.row` is the runscript
row as written. The folded law in `law_layers` and the effective legs in
`economy` and `behavior` are what the scenario runs, since a stack's rungs may
change any of the three legs. Execution, provenance, and the manifest read
the effective values. The row appears in the manifest only under the heading
that says it is what the runscript said.

Running the plan is `run/scenarios.py`.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from taxmodel.behavior import loader as behavior_loader
from taxmodel.behavior.config import BehaviorConfigError, read_behavior_leg
from taxmodel.config import law_stacks, leg_or_none
from taxmodel.config import manifest as manifest_module
from taxmodel.config.detail import DEFAULT_DETAIL_ROOT, DetailSpec, read_detail_spec
from taxmodel.config.errors import ConfigError
from taxmodel.config.interfaces import (
    DEFAULT_OUTPUT_ROOTS,
    interface_pin,
)
from taxmodel.config.runscript import (
    ScenarioRow,
    TaxLawTarget,
    classify_tax_law_cells,
    read_runscript,
)
from taxmodel.run import output_directory
from taxmodel.scenario import provenance as scenario_provenance

# Runs land under model_data, then the model's name, then the interface
# version: model_data/tax-simulator-gpu/v1/<runscript stem>/<stamp>.
RUNS_FOLDER_NAME = Path("model_data") / "tax-simulator-gpu" / "v1"

# The population every scenario reads: the bridge output for the Tax-Data
# vintage the economy leg pins. The bridge tree is shared lab data outside the
# repository.
DEFAULT_POPULATION_ROOT = Path(
    "/nfs/roberts/project/pi_nrs36/shared/model_data/Tax-Data-Bridge/v1"
)

# --------------------------------------------------------------------------
# The parts of a planned run
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StackMembership:
    """The stack a scenario came from.

    A stack's scenarios are the cumulative laws: provision 1, then provisions
    1 and 2, and so on. `position` is 1 for the first provision.
    """

    row_id: str
    stack_file: Path
    provision: str
    position: int


@dataclass(frozen=True)
class ScenarioSpec:
    """One ordinary scenario the run produces.

    After stack expansion every scenario is one of these, whether it came from
    a plain runscript row or from one provision of a stack. `law_layers` is the
    list of reform folders to lay over baseline law in order: empty for a
    baseline scenario, one folder for a plain reform, and the cumulative list
    for a stack member.

    `row` is the runscript row this scenario came from, kept as source
    metadata. For a stack member the row's tax_law cell names the stack file,
    and its economy and behavior cells are the starting point a rung may
    change. The scenario runs `law_layers`, `economy` and `behavior`.
    """

    id: str
    row: ScenarioRow
    law_layers: tuple[str, ...]
    detail: DetailSpec | None = None
    stack: StackMembership | None = None
    # A stack provision may change the economy or behavior leg instead of the
    # tax law. None means the runscript row's own cell stands.
    economy_override: str | None = None
    behavior_override: str | None = None

    @property
    def economy(self) -> str:
        """The economy alternative this scenario runs: the provision's, or the row's."""
        return self.economy_override or self.row.economy

    @property
    def behavior(self) -> str:
        """The behavior alternative this scenario runs: the provision's, or the row's."""
        return self.behavior_override or self.row.behavior

    @property
    def alternative_or_stack_name(self) -> str | None:
        """The name of this scenario's law: the reform folder, or the last provision.

        A plain reform row is its folder. A stack member is the folder of the
        provision it adds, which names its row in the stacked table. The full
        cumulative list is in `law_layers`.
        """
        if not self.law_layers:
            return None
        return self.law_layers[-1]


@dataclass(frozen=True)
class SubparameterConflict:
    """Two provisions of one stack replacing the same subparameter.

    The later provision wins and both replacements are recorded in the
    manifest. Splitting one credit change into several provisions
    of the same parameter file is ordinary.
    """

    subparameter: str
    winning_provision: str
    displaced_provisions: tuple[str, ...]


@dataclass(frozen=True)
class StackPlan:
    """The expansion of one stack row, for the manifest and the stacked table."""

    row_id: str
    stack_file: Path
    provisions: tuple[str, ...]
    scenario_ids: tuple[str, ...]
    # The scenario each provision's own effect is measured against
    # (`config/law_stacks.ProvisionIncrement`). The stacked table differences
    # the finished runs along these.
    increments: tuple[law_stacks.ProvisionIncrement, ...]
    conflicts: tuple[SubparameterConflict, ...]


@dataclass(frozen=True)
class StackExpansion:
    """The scenarios one stack row becomes, and how to difference them."""

    scenarios: tuple[ScenarioSpec, ...]
    plan: StackPlan


@dataclass(frozen=True)
class RunPlan:
    """The plan for a run, settled before anything runs.

    A dry run prints this and stops. A real run hands it to `execute_run`,
    which writes to the directory named in `run_directory`.
    """

    runscript_path: Path
    scenarios: tuple[ScenarioSpec, ...]
    stacks: tuple[StackPlan, ...]
    config_root: Path
    run_directory: Path
    code_version: dict[str, Any] = field(default_factory=dict)

    @property
    def stem(self) -> str:
        """The runscript's file name without its extension. It names the run folder."""
        return self.runscript_path.stem


# --------------------------------------------------------------------------
# Stack expansion
# --------------------------------------------------------------------------


def expand_stack_file(
    target: TaxLawTarget,
    row: ScenarioRow,
    config_root: Path,
    detail: DetailSpec | None = None,
) -> StackExpansion:
    """Expand one runscript row naming a stack file into the stack's scenarios.

    A stack file lists a reform package's provisions in scoring order. Each
    provision becomes an ordinary scenario running the cumulative law through
    that provision. The expansion also records the adjacent differences the
    reporting layer takes and which provision won each subparameter two of
    them both wrote. The fold, the file's schema and the conflict records are
    in `config/law_stacks.py`.

    A provision may change the economy or behavior leg instead of the tax law.
    It then runs the law of the provision below it with its own leg change in
    force. A leg no provision names stays as the runscript row set it.
    """
    assert target.path is not None
    expanded = law_stacks.expand_stack(target.path, config_root, row_id=row.id)

    scenarios = tuple(
        ScenarioSpec(
            id=scenario.id,
            row=row,
            law_layers=scenario.law_layers,
            detail=detail,
            stack=StackMembership(
                row_id=row.id,
                stack_file=target.path,
                provision=scenario.provision,
                position=position,
            ),
            economy_override=scenario.economy,
            behavior_override=scenario.behavior,
        )
        for position, scenario in enumerate(expanded.scenarios, start=1)
    )

    return StackExpansion(
        scenarios=scenarios,
        plan=StackPlan(
            row_id=row.id,
            stack_file=target.path,
            provisions=expanded.stack.provisions,
            scenario_ids=tuple(scenario.id for scenario in scenarios),
            increments=expanded.increments,
            conflicts=_subparameter_conflicts(expanded),
        ),
    )


def _subparameter_conflicts(
    expanded: law_stacks.ExpandedStack,
) -> tuple[SubparameterConflict, ...]:
    """Every subparameter two provisions of this stack both replaced.

    The last provision to write a subparameter is the law in force. Both
    replacement records are kept so the manifest can say what displaced what.
    """
    if not expanded.scenarios:
        return ()
    law_in_force = expanded.scenarios[-1].law
    return tuple(
        SubparameterConflict(
            subparameter=subparameter,
            winning_provision=writers[-1],
            displaced_provisions=tuple(writers[:-1]),
        )
        for subparameter, writers in sorted(law_in_force.conflicts().items())
    )

# --------------------------------------------------------------------------
# Planning a run
# --------------------------------------------------------------------------


def default_output_root(output_roots: str | Path = DEFAULT_OUTPUT_ROOTS) -> Path:
    """The default run location: the scratch model_data tree."""
    roots = yaml.safe_load(Path(output_roots).read_text())
    return Path(roots["local"]) / RUNS_FOLDER_NAME


def plan_run(
    runscript_path: str | Path,
    output_root: str | Path | None = None,
    config_root: str | Path = scenario_provenance.DEFAULT_CONFIG_ROOT,
    detail_root: str | Path = DEFAULT_DETAIL_ROOT,
    started_at: dt.datetime | None = None,
    code_version: dict[str, Any] | None = None,
) -> RunPlan:
    """Read a runscript and settle everything about the run except its numbers.

    A retired or unknown column, a tax_law cell that resolves to nothing, or a
    detail spec asking for a world nothing can write is refused by name before
    any compute happens.

    `code_version` is the commit the run directory's name carries, read from
    git when it is not supplied. A caller planning many runscripts at once can
    supply it and make the git calls once.
    """
    runscript_path = Path(runscript_path)
    config_root = Path(config_root)
    rows = read_runscript(runscript_path)
    _check_scenario_ids(rows, runscript_path)

    targets = classify_tax_law_cells(rows, config_root)
    details = _read_detail_specs(rows, detail_root)

    scenarios: list[ScenarioSpec] = []
    stacks: list[StackPlan] = []
    for row in rows:
        target = targets[row.id]
        if target.is_stack:
            expansion = expand_stack_file(
                target, row, config_root, details.get(row.id)
            )
            scenarios.extend(expansion.scenarios)
            stacks.append(expansion.plan)
            continue
        scenarios.append(
            ScenarioSpec(
                id=row.id,
                row=row,
                law_layers=() if target.path is None else (target.cell,),
                detail=details.get(row.id),
            )
        )

    if code_version is None:
        code_version = manifest_module.git_state()
    root = Path(output_root) if output_root is not None else default_output_root()
    return RunPlan(
        runscript_path=runscript_path,
        scenarios=tuple(scenarios),
        stacks=tuple(stacks),
        config_root=config_root,
        run_directory=output_directory.run_directory_path(
            root, runscript_path.stem, code_version.get("commit"), started_at
        ),
        code_version=code_version,
    )


def _check_scenario_ids(rows: Sequence[ScenarioRow], runscript_path: Path) -> None:
    """Check that every row has a distinct ID. The ID names a folder in the run directory."""
    problems: list[str] = []
    if not rows:
        problems.append("it has no scenario rows")
    if any(not row.id for row in rows):
        problems.append("a row has an empty ID cell")
    counts = [row.id for row in rows]
    duplicates = sorted({row.id for row in rows if counts.count(row.id) > 1})
    if duplicates:
        problems.append(f"one scenario ID is used twice: {', '.join(duplicates)}")
    if problems:
        raise ConfigError(
            f"the runscript {runscript_path} cannot be run:\n  - "
            + "\n  - ".join(problems)
            + "\nEach scenario ID names that scenario's folder in the run "
            "directory, so every row needs one and no two rows may share it."
        )


def _read_detail_specs(
    rows: Sequence[ScenarioRow], detail_root: str | Path
) -> dict[str, DetailSpec]:
    """Read every detail spec the runscript names, reporting all problems at once.

    Each spec is also turned into the request the run will make, against that
    row's years, and discarded. A spec naming a column the model does not
    calculate, or asking a world for marginal tax rates it cannot supply, is
    refused at planning time.
    """
    specs: dict[str, DetailSpec] = {}
    problems: list[str] = []
    for row in rows:
        if not row.detail:
            continue
        try:
            spec = read_detail_spec(Path(detail_root) / row.detail)
            spec.request(row.years)
        except (ConfigError, KeyError, ValueError) as error:
            problems.append(f"row '{row.id}': {error}")
            continue
        specs[row.id] = spec
    if problems:
        raise ConfigError("unusable detail specs:\n  - " + "\n  - ".join(problems))
    return specs


def check_behavior_legs(plan: RunPlan) -> None:
    """Check the behavior chain and every leg this run reads, before it runs.

    There are two checks.

    First the chain. Every scenario runs the same behavioral margins, so the
    modules are checked once: that each can be imported, that every marginal
    rate it declares is a rate this model computes, and that it names a
    family, which the loader needs to order the responses. The frame's columns
    are not checked here, since the frame does not exist until the run builds
    it; a module reading a column nobody carries is caught where the chain is
    composed.

    Then each behavior leg the runscript names. Reading the leg refuses an
    unknown margin, an unknown parameter name, or a margin turned on without
    one of its parameters. Every leg's complaint is collected into one message
    before anything compiles.
    """
    from taxmodel.taxes import marginal_rates

    computed_rates = set(marginal_rates.priced_rate_names())
    problems: list[str] = []

    for entry in behavior_loader.chain_entries():
        manifest = entry["manifest"]
        name = manifest.get("name", "<unnamed module>")
        # No frame exists yet, so only the prices and the family are checked.
        for problem in behavior_loader.manifest_problems(
            manifest, None, computed_rates
        ):
            problems.append(f'the behavior module "{name}" {problem}')

    for behavior in sorted({spec.behavior for spec in plan.scenarios}):
        try:
            leg = read_behavior_leg(plan.config_root, behavior)
        except BehaviorConfigError as error:
            problems.append(f"behavior leg '{behavior}': {error}")
            continue
        # Entity shifting needs the gains model. Without it retained earnings
        # are priced at zero, which understates the incentive to stay in
        # pass-through form.
        if "entity_shifting" in leg.margins and not leg.runs_the_gains_model:
            problems.append(
                f"behavior leg '{behavior}' turns on entity shifting without "
                "the gains model. Entity shifting prices the income it moves "
                "at the tax a deferred dollar bears, which only the gains "
                "machinery computes; without it retained earnings would be "
                "priced at zero. Bind kg_dynamics in this leg or drop the "
                "entity_shifting margin."
            )

    if problems:
        raise behavior_loader.ManifestError(
            "this runscript cannot run its behavioral responses:\n  - "
            + "\n  - ".join(problems)
        )


def describe_run_plan(plan: RunPlan) -> list[str]:
    """The plan as the lines a dry run prints.

    One line per scenario with its law, years, and output, then the files the
    run directory will hold.
    """
    check_behavior_legs(plan)
    lines = [
        f"runscript: {plan.runscript_path}",
        f"run directory: {plan.run_directory}",
        f"scenarios: {len(plan.scenarios)}",
    ]
    for spec in plan.scenarios:
        law = "baseline law" if not spec.law_layers else " + ".join(spec.law_layers)
        years = spec.row.years
        lines.append(
            f"  {spec.id}: {law}, {years[0]}-{years[-1]} "
            f"({len(years)} years), economy {spec.economy}, "
            f"behavior {spec.behavior}"
        )
        if spec.detail is not None:
            lines.append(
                f"    detail from {spec.detail.path}: "
                f"{len(spec.detail.variables)} variables in worlds "
                f"{', '.join(spec.detail.worlds)}"
            )
    for stack in plan.stacks:
        lines.append(
            f"  stack {stack.row_id}: "
            f"{len(stack.provisions)} provisions from {stack.stack_file}"
        )
    lines.append("files this run will write, relative to the run directory:")
    lines.extend(f"  {name}" for name in output_directory.planned_output_names(plan))
    return lines

def population_root_for(
    config_root: str | Path,
    economy: str | None,
    population_root: str | Path = DEFAULT_POPULATION_ROOT,
) -> Path:
    """The bridge population folder one economy leg's Tax-Data pin names."""
    vintage, scenario_id = interface_pin("Tax-Data", economy, Path(config_root))
    root = Path(population_root) / vintage / scenario_id
    if not root.exists():
        raise ConfigError(
            f"the economy leg pins Tax-Data vintage {vintage} ({scenario_id}), "
            f"and there is no bridge output for it at {root}"
        )
    return root


def population_for(
    plan: RunPlan, population_root: str | Path = DEFAULT_POPULATION_ROOT
) -> Path:
    """The bridge population folder this run reads.

    The economy leg's `interfaces.yaml` pins the Tax-Data vintage and scenario
    ID, and the run reads the bridge output for that pin. Every scenario in a
    run must pin the same population, since the scenarios share one population
    block and are differenced against each other.

    The leg checked is the one each scenario runs, since a stack rung that
    changes the economy leg can change the Tax-Data pin with it.
    """
    pins = {
        spec.id: interface_pin(
            "Tax-Data", leg_or_none(spec.economy), plan.config_root
        )
        for spec in plan.scenarios
    }
    distinct = sorted(set(pins.values()))
    if len(distinct) > 1:
        described = "; ".join(f"{name}: {vintage}/{scenario_id}"
                              for name, (vintage, scenario_id) in pins.items())
        raise ConfigError(
            "the scenarios in this runscript pin different Tax-Data "
            f"populations, so they cannot be compared: {described}"
        )
    # Every scenario pins the same population, so any leg's pin names it.
    return population_root_for(
        plan.config_root, leg_or_none(plan.scenarios[0].economy), population_root
    )
