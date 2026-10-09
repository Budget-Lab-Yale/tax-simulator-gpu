"""The two layers of the tax law leg and the replacement rule for reforms.

A reform replaces whole subparameters. Its `ctc.po_thresh` block replaces
the baseline's block, value and indexation fields together, so a reform that
gives a new phaseout threshold and omits `i_measure` un-indexes that
threshold from then on.

This module does the layering and records what was replaced. The law
compiler turns the merged parameter tree into year-by-filing-status arrays.

A reform may name a subparameter the baseline does not have. The layering
records the case, and the compiler's lint refuses it unless the block says
`new_subparameter: true`.

A stacked reform lays several provision folders over baseline law in a
stated order, and `resolve_law_stack` returns the cumulative law after each
one: baseline plus provision 1, then plus provisions 1 and 2, and so on.
Three rules govern the fold:

  Merging is at the subparameter level. One provision can raise the child
  credit's maximum value while a later one changes its refundability, each
  folder holding only its own block.

  When two provisions replace the same subparameter, the later one wins and
  both replacements are recorded.

  Whether a subparameter is new is judged against baseline law, not against
  the partly-folded tree. A provision naming a subparameter baseline law
  lacks writes `new_subparameter: true` even when an earlier provision in
  the stack already created it.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError
from .scenario import DEFAULT_LAYER, TAX_LAW_LEG, leg_path

# The four fields that say how a subparameter grows. A reform that restates a
# subparameter without them has un-indexed it. Every parameter file lists
# them in this order, and the parser checks the order.
INDEXATION_FIELDS = ("i_measure", "i_base_year", "i_direction", "i_increment")


@dataclass(frozen=True)
class Replacement:
    """One subparameter a reform replaced, and what the replacement changed.

    `written_by` is the law layer whose file holds the block: a reform folder
    for a single reform, or the individual provision folder inside a stack.
    """

    parameter: str
    subparameter: str
    baseline: Any
    reform: Any
    new_subparameter: bool
    dropped_indexation: bool
    written_by: str

    @property
    def label(self) -> str:
        return f"{self.parameter}.{self.subparameter}"


@dataclass
class ResolvedLaw:
    """The merged parameter tree, and the reform's replacements for the manifest.

    `alternative` names this law: a reform folder, `default` for baseline law,
    or, for one rung of a stack, the last provision folder folded in, which is
    the name the runscript row and the reported table carry.

    `layers` is every law layer that went into the tree, in the order they
    were laid down, baseline first. For a single reform that is baseline and
    the reform folder; for a stack it is baseline and every provision folded
    in so far. The lint's scope rule reads it.

    `baseline_subparameters` keeps the baseline's own names per parameter
    file, which the merged tree no longer distinguishes. The lint reads it to
    tell a new lever from a misspelled one and to name the nearest baseline
    subparameter in the refusal.
    """

    alternative: str
    parameters: dict[str, dict[str, Any]]
    replacements: list[Replacement]
    baseline_subparameters: dict[str, tuple[str, ...]] = field(default_factory=dict)
    layers: tuple[str, ...] = (DEFAULT_LAYER,)

    def written_by(self, parameter: str, subparameter: str) -> str:
        """Which layer's file the merged tree took this block from.

        The last provision to write a subparameter is the one whose block is
        in force, so the search runs from the end of the list backwards.
        Nothing having written it means the block is baseline law's own.
        """
        for replacement in reversed(self.replacements):
            if (
                replacement.parameter == parameter
                and replacement.subparameter == subparameter
            ):
                return replacement.written_by
        return DEFAULT_LAYER

    def conflicts(self) -> dict[str, tuple[str, ...]]:
        """Subparameters more than one provision replaced, and which folders did.

        Empty for a single reform, since a folder holds each subparameter
        once. In a stack it is the audit trail: the later folder's block is
        the law in force, and this says which earlier folders it overrode.
        """
        writers: dict[str, list[str]] = {}
        for replacement in self.replacements:
            writers.setdefault(replacement.label, []).append(replacement.written_by)
        return {
            label: tuple(folders)
            for label, folders in writers.items()
            if len(folders) > 1
        }


def load_layer(root: Path, name: str) -> dict[str, dict[str, Any]]:
    """Read one law layer: one YAML file per tax parameter, its stem the parameter name."""
    folder = leg_path(root, TAX_LAW_LEG, name)
    if not folder.is_dir():
        raise ConfigError(f"unknown tax law layer '{name}' -- no folder {folder}")

    parameters: dict[str, dict[str, Any]] = {}
    for path in sorted(folder.glob("*.yaml")):
        with open(path) as f:
            loaded = yaml.safe_load(f)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{path} is not a mapping of subparameters")
        parameters[path.stem] = loaded
    return parameters


def resolve_law(root: Path, alternative: str | None = None) -> ResolvedLaw:
    """Lay one reform over baseline law, one whole subparameter at a time.

    A single reform is a stack of one, so the fold below does the work.
    Baseline law on its own is `alternative=None` or the reserved word
    `default`.
    """
    root = Path(root)
    if not alternative or str(alternative) == DEFAULT_LAYER:
        return resolve_law_stack(root, [])[-1]
    return resolve_law_stack(root, [str(alternative)])[-1]


def resolve_law_stack(root: Path, provisions: Sequence[str]) -> list[ResolvedLaw]:
    """Fold provision folders onto baseline law, keeping every cumulative law.

    `provisions` is the stack in scoring order. What comes back is baseline
    law plus each successive prefix of the list: entry 0 is baseline law
    alone, entry 1 is baseline plus the first provision, entry 2 is baseline
    plus the first two, and so on. So an N-provision stack returns N+1 laws,
    and an empty list returns baseline law by itself.
    """
    root = Path(root)

    shadow = root / "tax_law" / "alternatives" / DEFAULT_LAYER
    if shadow.is_dir():
        raise ConfigError(
            f"`{DEFAULT_LAYER}` is the reserved runscript cell naming baseline "
            f"law, so {shadow} can never be selected. Rename it."
        )

    baseline = load_layer(root, DEFAULT_LAYER)
    baseline_names = {p: tuple(subs) for p, subs in baseline.items()}

    merged = copy.deepcopy(baseline)
    replacements: list[Replacement] = []
    layers = [DEFAULT_LAYER]

    cumulative = [
        ResolvedLaw(
            alternative=DEFAULT_LAYER,
            parameters=copy.deepcopy(merged),
            replacements=list(replacements),
            baseline_subparameters=baseline_names,
            layers=tuple(layers),
        )
    ]

    for provision in provisions:
        name = str(provision)
        layers.append(name)
        for parameter, subparams in load_layer(root, name).items():
            # Whether a subparameter is new is judged against baseline law.
            baseline_param = baseline.get(parameter, {})
            target = merged.setdefault(parameter, {})
            for subparameter, body in subparams.items():
                old = baseline_param.get(subparameter)
                replacements.append(
                    Replacement(
                        parameter=parameter,
                        subparameter=subparameter,
                        baseline=old,
                        reform=body,
                        new_subparameter=subparameter not in baseline_param,
                        dropped_indexation=_dropped_indexation(old, body),
                        written_by=name,
                    )
                )
                # The whole block is replaced, value and indexation together.
                target[subparameter] = copy.deepcopy(body)

        cumulative.append(
            ResolvedLaw(
                alternative=name,
                parameters=copy.deepcopy(merged),
                replacements=list(replacements),
                baseline_subparameters=baseline_names,
                layers=tuple(layers),
            )
        )

    return cumulative


def _dropped_indexation(baseline: Any, reform: Any) -> bool:
    """Whether restating this subparameter turned its indexation off.

    The law compiler's lint requires a reform that does this to say
    `remove_indexation: true`.
    """
    if not isinstance(baseline, dict) or not isinstance(reform, dict):
        return False
    had = any(f in baseline for f in INDEXATION_FIELDS)
    has = any(f in reform for f in INDEXATION_FIELDS)
    return had and not has
