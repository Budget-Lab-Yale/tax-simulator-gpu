"""Reading a scenario's behavior leg out of its behavior.yaml.

A scenario's behavior cell names a folder under ``config/scenarios/behavior/``:
the reserved word ``default``, the reserved word ``none``, or a path under
``alternatives/``. The folder holds one file, ``behavior.yaml``:

    kg_dynamics:
      realization: config/calibrations/kg/realization.yaml
      conversion: config/calibrations/kg/conversion.yaml

    margins:
      charity:
        elasticity: -0.5
        max_percent_change: 1.0

Every behavioral margin runs in every compiled program; the file supplies
parameter values.

The reader enforces five rules.

1. A margin absent from ``margins`` runs at the neutral parameter values in
   its module's manifest, at which the module returns its columns unchanged.

2. A margin present must state every parameter it takes. An unknown margin
   name, an unknown parameter name, or a missing parameter stops the run
   before anything compiles, with a message listing the margins this model
   has.

3. An alternative replaces a section whole. A section the alternative leaves
   out is inherited from the default layer.

4. The capital-gains machinery is bound in the ``kg_dynamics`` section (the
   word ``none`` turns it off). The in-year conversion of salary into equity
   is part of that machinery, with its response parameter sigma in the
   calibration file the section binds, so ``conversion`` is not a margin.

5. Any other top-level key stops the run, with a message naming it.

The behavior leg does not go through ``config/scenario.py``, which reads
parameter entries with provenance and staleness fields.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import loader

# The reserved word for a scenario with no behavioral response: the gains
# model does not run and no calibrated behavior entry is read.
# Every margin still runs, at neutral.
NO_BEHAVIOR = "none"

# The reserved word for the default layer.
DEFAULT_LAYER = "default"

# The two sections a behavior.yaml carries. An alternative replaces either one
# whole, or leaves it out and inherits the default layer's.
KG_DYNAMICS_SECTION = "kg_dynamics"
MARGINS_SECTION = "margins"
SECTIONS = (KG_DYNAMICS_SECTION, MARGINS_SECTION)

# Parameters a margin takes that the run or the compiled law supplies, so a
# scenario neither states them nor is asked for them. The corporate rates come
# from the law being scored; the conversion table comes from the capital-gains
# behavioral solve.
RUN_SUPPLIED_PARAMS: dict[str, tuple[str, ...]] = {
    "entity_shifting": ("corp.rate", "corp.rate_baseline"),
    "conversion": ("conversions",),
}


class BehaviorConfigError(ValueError):
    """A scenario's behavior leg cannot be read, or names something that is not
    a margin this model has."""


@dataclass(frozen=True)
class BehaviorLeg:
    """One scenario's resolved behavior leg.

    ``kg_dynamics`` maps each piece of the gains machinery this scenario uses
    to its calibration file. It is empty when the scenario runs no gains
    model.

    ``margins`` maps each behavioral margin the scenario turns on to the
    parameter values it states. A margin absent from here runs at its neutral
    values.
    """

    name: str
    kg_dynamics: Mapping[str, str] = field(default_factory=dict)
    margins: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    @property
    def runs_the_gains_model(self) -> bool:
        """Whether this scenario runs the capital-gains model."""
        return bool(self.kg_dynamics)

    @property
    def converts(self) -> bool:
        """Whether the gains machinery prices conversions of salary into equity."""
        return "conversion" in self.kg_dynamics


def leg_folder(config_root: str | Path, name: str | None) -> Path:
    """The folder holding a behavior cell's behavior.yaml.

    An empty cell and the word `default` mean the same folder.
    """
    root = Path(config_root) / "behavior"
    if not name or name.strip().lower() == DEFAULT_LAYER:
        return root / DEFAULT_LAYER
    return root / "alternatives" / name.strip()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise BehaviorConfigError(
            f"the behavior leg at {path.parent} has no behavior.yaml in it"
        )
    body = yaml.safe_load(path.read_text()) or {}
    if not isinstance(body, dict):
        raise BehaviorConfigError(
            f"{path} should hold a kg_dynamics section and a margins block, "
            f"but it holds a {type(body).__name__}"
        )
    unknown = sorted(str(key) for key in body if key not in SECTIONS)
    if unknown:
        raise BehaviorConfigError(
            f"{path}: `{unknown[0]}` is not a section of a behavior leg. The "
            f"file holds only {KG_DYNAMICS_SECTION} and {MARGINS_SECTION}."
        )
    return body


def _read_kg_dynamics(body: Mapping[str, Any], where: Path) -> dict[str, str]:
    """The gains machinery this leg binds, piece by piece to its config file."""

    section = body.get(KG_DYNAMICS_SECTION)
    if section is None:
        return {}
    if isinstance(section, str):
        if section.strip().lower() in {NO_BEHAVIOR, ""}:
            return {}
        raise BehaviorConfigError(
            f"{where}: kg_dynamics is either the word `none` or a block "
            f"binding each piece of the gains machinery to its calibration "
            f'file, not "{section}"'
        )
    if not isinstance(section, dict):
        raise BehaviorConfigError(
            f"{where}: kg_dynamics should be the word `none` or a block of "
            f"piece-to-file bindings, not a {type(section).__name__}"
        )
    return {str(piece): str(path) for piece, path in section.items()}


def _configurable_parameters(family: str) -> tuple[str, ...]:
    """The parameter names a scenario must state to turn one margin on.

    Every parameter the module ships a default for, except the ones the run or
    the law supplies.
    """

    _function, _manifest, defaults = loader.import_family(family)
    supplied = set(RUN_SUPPLIED_PARAMS.get(family, ()))
    return tuple(name for name in defaults if name not in supplied)


def _read_margins(
    body: Mapping[str, Any], where: Path
) -> dict[str, dict[str, float]]:
    """The margins this leg turns on, each with the numbers it states."""

    listed = body.get(MARGINS_SECTION)
    if listed is None:
        return {}
    if not isinstance(listed, dict):
        raise BehaviorConfigError(
            f"{where}: margins should be a block naming each behavioral margin "
            f"this scenario turns on, with its parameter values under it, not "
            f"a {type(listed).__name__}"
        )

    margins: dict[str, dict[str, float]] = {}
    for named, stated in listed.items():
        family = str(named).strip()
        if family not in configurable_families():
            raise BehaviorConfigError(
                f"{where}: `{family}` is not a behavioral margin. The margins "
                "a scenario can turn on are: "
                + ", ".join(configurable_families())
                + ". The capital-gains machinery, the conversion of salary "
                "into equity included, is bound in the kg_dynamics section "
                "instead."
            )
        if not isinstance(stated, dict):
            raise BehaviorConfigError(
                f"{where}: the `{family}` margin should carry its parameter "
                f"values as named numbers, not a {type(stated).__name__}. A "
                "margin that is on states every parameter it takes; there is "
                "no on-at-defaults shorthand."
            )
        wanted = _configurable_parameters(family)
        given = {str(name): value for name, value in stated.items()}
        unknown = sorted(set(given) - set(wanted))
        if unknown:
            raise BehaviorConfigError(
                f"{where}: the `{family}` margin has no parameter called "
                + ", ".join(unknown)
                + ". Its parameters are: "
                + ", ".join(wanted)
                + "."
            )
        missing = [name for name in wanted if name not in given]
        if missing:
            raise BehaviorConfigError(
                f"{where}: the `{family}` margin is on but does not state "
                + ", ".join(missing)
                + ". A margin that is on states every parameter it takes; a "
                "margin left out of the file altogether runs at its neutral "
                "values."
            )
        margins[family] = {name: float(given[name]) for name in wanted}
    return margins


def configurable_families() -> tuple[str, ...]:
    """The margins a behavior.yaml can name, in the order they run."""

    return tuple(
        family for family in loader.RESPONSE_FAMILIES if family != "conversion"
    )


def read_behavior_leg(config_root: str | Path, name: str | None) -> BehaviorLeg:
    """Resolve one scenario's behavior cell into the leg it runs.

    The reserved word `none` means no gains model and no margin on, and reads
    no file. Anything else reads the default layer and then lays the named
    alternative over it, one whole section at a time.
    """

    if name and name.strip().lower() == NO_BEHAVIOR:
        return BehaviorLeg(name=NO_BEHAVIOR)

    default_path = leg_folder(config_root, None) / "behavior.yaml"
    names_an_alternative = bool(name) and name.strip().lower() != DEFAULT_LAYER

    # A tree with no behavior layer declares no behavioral response, the same
    # as the reserved word `none`. A named alternative with no default layer
    # under it is an error, because an alternative inherits the sections it
    # leaves out.
    if not default_path.is_file() and not names_an_alternative:
        return BehaviorLeg(name=NO_BEHAVIOR)

    default_body = _read_yaml(default_path)
    kg = _read_kg_dynamics(default_body, default_path)
    margins = _read_margins(default_body, default_path)
    label = DEFAULT_LAYER

    if names_an_alternative:
        label = name.strip()
        alternative_path = leg_folder(config_root, label) / "behavior.yaml"
        alternative_body = _read_yaml(alternative_path)
        # An alternative replaces a section whole; a section it leaves out is
        # inherited from the default layer.
        if KG_DYNAMICS_SECTION in alternative_body:
            kg = _read_kg_dynamics(alternative_body, alternative_path)
        if MARGINS_SECTION in alternative_body:
            margins = _read_margins(alternative_body, alternative_path)

    return BehaviorLeg(name=label, kg_dynamics=kg, margins=margins)
