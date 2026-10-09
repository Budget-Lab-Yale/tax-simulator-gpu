"""A stacked reform: a list of provisions, in scoring order.

A stack file names the provisions of one reform package and the order they
are scored in. Each provision folder holds only its own change, and the fold
in `law_layers.py` builds the cumulative laws.

The file sits beside the reform folders, at
`config/scenarios/tax_law/alternatives/<name>.yaml`. A runscript `tax_law`
cell naming a directory is a single reform, and one naming a `.yaml` file is
a stack.

A stack file looks like this:

    ---
    #-------------------------------------------------------------
    # Clausing-Sarin package, eight provisions in scoring order
    #-------------------------------------------------------------

    version: 1

    provisions:
      - clausing_provisions/01_clinton_rates
      - clausing_provisions/02_restore_bottom_rates
      - clausing_provisions/03_199a
      - name: 08_corporate
        economy: clausing_corporate_saving_50

A provision is a whole scenario pointer. A rung written as a bare string is
a tax-law folder. A rung written as a mapping may name any of the three legs
(`tax_law`, `economy`, `behavior`) and must name at least one. For example,
the Clausing package's eighth provision keeps the seventh provision's tax law
and changes the corporate saving assumption.

A rung that names no tax-law folder runs the law of the rung below it, and
its reported increment is the effect of its economy or behavior change. A leg
no rung names is supplied by the runscript row that asked for the stack, as
for any ordinary scenario.

Every rung has a name, which becomes the scenario's name in the run
directory and its row in the stacked table. A rung takes its `name` key if
it has one, and otherwise the first leg it names, in the order tax law,
economy, behavior. No two rungs of one stack may have the same name.

Each provision is scored on top of the ones before it. The scenario for
provision 3 runs baseline law plus provisions 1, 2 and 3, and provision 3's
reported effect is that scenario minus provision 2's. The increments add up
to the whole package's effect.

`expand_stack` returns an ordered list of ordinary scenarios, each a
resolved law with an ID, plus the list of differences the reporting layer
takes between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError
from .law_layers import ResolvedLaw, resolve_law_stack
from .scenario import DEFAULT_LAYER

# The schema version this reader understands. A file stating a different
# version is refused.
STACK_FILE_VERSION = 1

# The keys a stack file may hold. `note` is free text and nothing reads it.
REQUIRED_KEYS = ("version", "provisions")
OPTIONAL_KEYS = ("note",)

# The three legs a provision may change. `config/scenario.py` calls the last
# two LEGS; a stack rung can point at all three.
RUNG_LEGS = ("tax_law", "economy", "behavior")

# The keys a rung written as a mapping may hold.
RUNG_KEYS = RUNG_LEGS + ("name",)


@dataclass(frozen=True)
class StackRung:
    """One provision of a package: the legs it changes and its name.

    A leg the rung does not change is None. The leg keeps whatever the rung
    below it set, or what the runscript row supplied if no rung set it.

    `provision` is the rung's name: the folder path for a bare-string rung,
    otherwise the explicit `name` key.
    """

    provision: str
    tax_law: str | None = None
    economy: str | None = None
    behavior: str | None = None


@dataclass(frozen=True)
class LawStack:
    """One stacked reform as its file states it."""

    name: str
    rungs: tuple[StackRung, ...]
    path: Path

    @property
    def provisions(self) -> tuple[str, ...]:
        """Every provision's name, in scoring order."""
        return tuple(rung.provision for rung in self.rungs)

    @property
    def tax_law_folders(self) -> tuple[str, ...]:
        """The tax-law provision folders, in the order they are folded on.

        Shorter than the rung list whenever a provision works through the
        economy or behavior leg instead of the law.
        """
        return tuple(rung.tax_law for rung in self.rungs if rung.tax_law)


@dataclass(frozen=True)
class StackScenario:
    """One ordinary scenario a stack expands into.

    `id` is `{row ID}.{provision folder name}` when a runscript row asked for
    the stack, and the
    provision folder path when the stack is expanded without a row, as in a
    test.

    `provision` is the name of the rung this scenario adds to the one before
    it.

    `law_layers` is the tax-law provision folders folded into `law`. `economy`
    and `behavior` are the alternative layers the provisions so far have put
    those legs on, and are None when no provision named the leg, in which
    case the runscript row's setting stands.
    """

    id: str
    law: ResolvedLaw
    provision: str
    law_layers: tuple[str, ...] = ()
    economy: str | None = None
    behavior: str | None = None


@dataclass(frozen=True)
class ProvisionIncrement:
    """One provision's own effect, as a difference between two scenarios.

    `subtrahend_id` is None when the comparison is against the run's own
    baseline, which is the case for the first provision of a stack.
    """

    provision: str
    minuend_id: str
    subtrahend_id: str | None


@dataclass(frozen=True)
class ExpandedStack:
    """The scenarios the run executes and the differences the reporting layer takes."""

    stack: LawStack
    scenarios: tuple[StackScenario, ...]
    increments: tuple[ProvisionIncrement, ...]


def read_law_stack(path: str | Path) -> LawStack:
    """Read and check one stack file.

    Every problem is reported at once. An unrecognised key is an error, so a
    misspelled key cannot pass as a setting that took effect.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"no stack file at {path}")

    with open(path) as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path} is not a mapping of stack settings")

    problems: list[str] = []
    for key in loaded:
        if key not in REQUIRED_KEYS + OPTIONAL_KEYS:
            problems.append(
                f"unknown key `{key}` (a stack file holds exactly "
                f"{', '.join(REQUIRED_KEYS)}, and optionally "
                f"{', '.join(OPTIONAL_KEYS)})"
            )
    for key in REQUIRED_KEYS:
        if key not in loaded:
            problems.append(f"missing required key `{key}`")

    version = loaded.get("version")
    if version is not None and version != STACK_FILE_VERSION:
        problems.append(
            f"version {version!r}, but this model reads stack file version "
            f"{STACK_FILE_VERSION}"
        )

    rungs: tuple[StackRung, ...] = ()
    provisions = loaded.get("provisions")
    if provisions is not None:
        rungs, rung_problems = _read_rungs(provisions)
        problems.extend(rung_problems)

    if problems:
        raise ConfigError(
            f"invalid stack file '{path}':\n  - " + "\n  - ".join(problems)
        )

    return LawStack(name=path.stem, rungs=rungs, path=path)


def _read_rungs(provisions: Any) -> tuple[tuple[StackRung, ...], list[str]]:
    """Read the provision list, and say everything that is wrong with it."""
    if not isinstance(provisions, list) or not provisions:
        return (), [
            (
                "`provisions` must be a non-empty list of law folder names, or "
                "of mappings naming the legs each provision changes "
                f"({', '.join(RUNG_LEGS)})"
            )
        ]

    rungs: list[StackRung] = []
    problems: list[str] = []
    for entry in provisions:
        rung, entry_problems = _read_one_rung(entry)
        problems.extend(entry_problems)
        if rung is not None:
            rungs.append(rung)

    problems.extend(_name_clashes([rung.provision for rung in rungs]))
    return tuple(rungs), problems


def _read_one_rung(entry: Any) -> tuple[StackRung | None, list[str]]:
    """One entry of the provision list, written as a folder name or as legs."""
    if isinstance(entry, str):
        folder = entry.strip()
        if not folder:
            return None, [f"provision {entry!r} is not a folder name"]
        if folder == DEFAULT_LAYER:
            return None, [
                (
                    f"provision `{DEFAULT_LAYER}` is the reserved name for "
                    "baseline law, which every provision is already laid over"
                )
            ]
        return StackRung(provision=folder, tax_law=folder), []

    if not isinstance(entry, dict):
        return None, [
            (
                f"provision {entry!r} is neither a tax-law folder name nor a "
                f"mapping naming the legs it changes ({', '.join(RUNG_LEGS)})"
            )
        ]

    problems: list[str] = []
    for key in entry:
        if key not in RUNG_KEYS:
            problems.append(
                f"provision {entry!r} has an unknown key `{key}` (a provision "
                f"names any of {', '.join(RUNG_LEGS)}, and optionally `name`)"
            )

    legs: dict[str, str | None] = {}
    for leg in RUNG_LEGS:
        value = entry.get(leg)
        if value is None:
            legs[leg] = None
        elif not isinstance(value, str) or not value.strip():
            problems.append(
                f"provision {entry!r} gives `{leg}` as {value!r}, not a name"
            )
            legs[leg] = None
        else:
            legs[leg] = value.strip()

    # Baseline law is already under every provision, so naming it as a
    # provision's tax law is an error. The economy and behavior default layers
    # may be named, since a later provision may put a leg back to its default.
    if legs["tax_law"] == DEFAULT_LAYER:
        problems.append(
            f"provision `{DEFAULT_LAYER}` is the reserved name for baseline "
            "law, which every provision is already laid over"
        )
        legs["tax_law"] = None

    if not any(legs.values()):
        problems.append(
            f"provision {entry!r} changes none of {', '.join(RUNG_LEGS)}, so it "
            "would score a scenario identical to the one below it"
        )
        return None, problems

    name = entry.get("name")
    if name is not None and (not isinstance(name, str) or not name.strip()):
        problems.append(f"provision {entry!r} gives `name` as {name!r}, not a name")
        name = None

    if problems:
        return None, problems

    if name is None:
        # Named after whichever leg it changes, tax law first, so an ordinary
        # tax-law provision written as a mapping is called what it would be
        # called written as a bare string.
        name = next(legs[leg] for leg in RUNG_LEGS if legs[leg])

    return (
        StackRung(
            provision=str(name).strip(),
            tax_law=legs["tax_law"],
            economy=legs["economy"],
            behavior=legs["behavior"],
        ),
        [],
    )


def _name_clashes(names: list[str]) -> list[str]:
    """Two provisions that would expand into two scenarios with one name."""
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        return [
            (
                "these provisions appear twice, so two scenarios would carry "
                f"the same name: {', '.join(repeated)}. A provision that "
                "changes the same law twice is two folders with two names."
            )
        ]

    # A scenario is named after the provision's last path component, so two
    # provisions with the same folder name under different parents would give
    # two scenarios one name, and the second would overwrite the first's output.
    leaves = [Path(n).name for n in names]
    shared = sorted({leaf for leaf in leaves if leaves.count(leaf) > 1})
    if shared:
        return [
            (
                "these provisions have different paths but the same folder "
                "name, and a scenario is named after its folder: "
                f"{', '.join(shared)}. Rename one of the folders."
            )
        ]
    return []


def scenario_id(provision: str, row_id: str | None) -> str:
    """The name of one expanded scenario.

    `{row ID}.{provision folder name}` when a runscript row asked for the
    stack, and the provision folder path as the stack file writes it when no
    row did.
    """
    if row_id is None:
        return provision
    return f"{row_id}.{Path(provision).name}"


def expand_stack(
    stack_path: str | Path,
    config_root: str | Path,
    row_id: str | None = None,
) -> ExpandedStack:
    """Turn a stack file into ordinary scenarios and the differences to report.

    `config_root` is the scenario tree holding the provision folders and
    baseline law, `config/scenarios` in a real run.

    `row_id` is the ID of the runscript row whose `tax_law` cell named this
    stack file. It prefixes the expanded scenarios' names.

    The scenarios are the cumulative laws. Each provision's increment is its
    scenario minus the scenario before it.
    """
    stack = read_law_stack(stack_path)
    scenarios = _stacked_scenarios(stack, Path(config_root), row_id)

    # The first provision is measured against the run's own baseline and
    # every later one against the provision below it.
    increments = tuple(
        ProvisionIncrement(
            provision=scenario.provision,
            minuend_id=scenario.id,
            subtrahend_id=scenarios[position - 1].id if position > 0 else None,
        )
        for position, scenario in enumerate(scenarios)
    )
    return ExpandedStack(stack=stack, scenarios=scenarios, increments=increments)


def _stacked_scenarios(
    stack: LawStack, config_root: Path, row_id: str | None
) -> tuple[StackScenario, ...]:
    """Each provision scored on top of the ones below it.

    The tax-law fold runs once over the law folders the stack names, in order,
    and returns baseline law followed by the law after each folder. A provision
    that names no law folder takes the law standing when it is reached. The
    economy and behavior legs carry forward the same way: whatever the last
    provision to name one set, until another names it again.
    """
    law_folders = stack.tax_law_folders
    cumulative = resolve_law_stack(config_root, law_folders)

    scenarios: list[StackScenario] = []
    folders_folded = 0
    economy: str | None = None
    behavior: str | None = None
    for position, rung in enumerate(stack.rungs):
        if rung.tax_law:
            folders_folded += 1
        if rung.economy:
            economy = rung.economy
        if rung.behavior:
            behavior = rung.behavior
        scenarios.append(
            StackScenario(
                id=scenario_id(rung.provision, row_id),
                law=cumulative[folders_folded],
                provision=rung.provision,
                law_layers=law_folders[:folders_folded],
                economy=economy,
                behavior=behavior,
            )
        )
    return tuple(scenarios)

