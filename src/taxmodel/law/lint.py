"""Lint checks on a law tree that parses but says something its author did
not write.

These are errors, not warnings (ARCHITECTURE.md D4). Each one describes a
reform whose compiled output differs from the reform its author described.

  Dropped indexation. A reform replaces a whole subparameter, value and
  indexation fields together, so a reform that gives the child credit
  phaseout a new threshold and does not repeat `i_measure` has frozen that
  threshold in nominal terms for the rest of the budget window. An author who
  means it writes `remove_indexation: true` in the block.

  Schedule overflow. Compiled schedules are a fixed number of slots wide. A
  reform with more brackets than that would be truncated.

  Indexation resumed after a freeze. A config freezes indexation by writing the
  measure out as of a year, as the TCJA froze the lifetime learning credit
  phaseout in 2020. Naming a live measure again in a later year does not
  restart indexation, because the chain it multiplies into is already missing.
  A config that tries compiles as a permanent freeze.

  A schedule whose rates and thresholds have different lengths. The compiler
  pads every subparameter in a parameter file out to the widest one with NaN,
  and the calculator counts a bracket slot as part of the schedule only where
  both the threshold and the rate are present (`taxes/calc/_schedules.py`). So
  a law stating six ordinary-rate thresholds and five rates has five brackets,
  and the income above the fifth threshold is taxed at the fifth rate instead
  of the sixth. A stack that folds a provision restating the thresholds
  together with a later provision restating the rates can end up with such a
  mismatch. This check runs on the compiled arrays, after indexation and the
  filing-status mappers, because those arrays are what the calculator reads.

  The same check catches a reform that states the ordinary rates for 2026
  onward and says nothing about earlier years. Every earlier year of the run
  then has thresholds and no rates, an empty schedule collecting no tax. Each
  reform folder must restate a schedule's whole history.

  A lever the baseline does not have. Without this check `ctc.po_thresh_` (a
  typo for `ctc.po_thresh`) would become a new name nothing consumes, the
  phaseout threshold would keep its baseline value, and the reform would look
  like it ran. A genuinely new lever is written `new_subparameter: true` in
  the block, and the manifest records it either way.

The schedule-length check has no layer scoping, because it is a statement
about the arrays this compile produced. A mismatch stops the compile
(`check_schedules`, called from `compile_resolved`).

The other four checks are hard errors for the law layers a runscript names.
Findings in the rest of the tree do not block, since many reform folders drop
a baseline's indexation without saying so, and a run that never
compiles them has no reason to stop. Every finding carries the layer that
wrote the offending block, and `check_law` raises only for the layers in
scope.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config.law_layers import Replacement, ResolvedLaw
from ..config.scenario import DEFAULT_LAYER
from .errors import LawLintError
from .parse import (
    RESERVED_KEYS,
    carry_forward,
    indexation_fields_written,
    measure_by_year,
    parse_inf,
    parse_years,
    resolve_indexation_defaults,
)

DROPPED_INDEXATION = "dropped_indexation"
SCHEDULE_OVERFLOW = "schedule_overflow"
FREEZE_RESUME = "freeze_resume"
UNKNOWN_SUBPARAMETER = "unknown_subparameter"
SCHEDULE_LENGTH_MISMATCH = "schedule_length_mismatch"

# The name suffixes of the two halves of a rate schedule. `ord.rates` goes
# with `ord.brackets`, `pr.oasdi_ee_rates` with `pr.oasdi_ee_brackets`. A
# vector with neither suffix, such as the itemized deduction's mortgage-balance
# limit by loan vintage, is not a rate schedule.
_RATES_SUFFIX = "rates"
_THRESHOLDS_SUFFIX = "brackets"

# The minimum difflib similarity ratio at which a baseline name is offered as
# the intended one. 0.7 catches a dropped or doubled character in a name of
# ordinary length (`po_thresh_` for `po_thresh`) without proposing an
# unrelated lever.
_SUGGESTION_CUTOFF = 0.7


@dataclass(frozen=True)
class Finding:
    """One lint finding.

    `layer` is the law layer whose file holds the offending block (a reform
    folder, or `default` for baseline law). Scoping in `check_law` reads it.
    """

    rule: str
    field: str
    detail: str
    layer: str = DEFAULT_LAYER

    def __str__(self) -> str:
        return f"[{self.rule}] {self.field}: {self.detail}"


def lint_law(
    resolved: ResolvedLaw, years: Sequence[int], max_brackets: int
) -> list[Finding]:
    """Every violation in a layered law tree, in a stable order."""
    findings: list[Finding] = []
    findings.extend(_dropped_indexation(resolved))
    findings.extend(_unknown_subparameter(resolved))
    findings.extend(_schedule_overflow(resolved, years, max_brackets))
    findings.extend(_freeze_resume(resolved, years))
    return sorted(findings, key=lambda f: (f.rule, f.field))


def check_law(
    resolved: ResolvedLaw,
    years: Sequence[int],
    max_brackets: int,
    scope: Iterable[str] | None = None,
) -> list[Finding]:
    """Raise on a compile of a law tree that says something unintended.

    `scope` is the set of law layers whose findings raise: the cells of the
    runscript being compiled, plus `default`, which every run reads. Findings
    from any other layer are returned. The default scope is the
    layers that went into `resolved`.
    """
    # The default scope is baseline plus every reform folder stacked over it.
    in_scope = set(scope) if scope is not None else {DEFAULT_LAYER, *resolved.layers}
    in_scope.add(DEFAULT_LAYER)

    findings = lint_law(resolved, years, max_brackets)
    blocking = [f for f in findings if f.layer in in_scope]
    if not blocking:
        return findings

    plural = "problem" if len(blocking) == 1 else "problems"
    detail = "\n  ".join(str(f) for f in blocking)
    raise LawLintError(
        f"tax law layer '{resolved.alternative}' has {len(blocking)} "
        f"unresolved {plural}:\n  {detail}"
    )


def _effective_replacements(resolved: ResolvedLaw) -> list[Replacement]:
    """The replacements whose blocks survive in the merged tree.

    In a stack a later provision can overwrite an earlier one's block, and
    only the surviving block is checked: an earlier provision that dropped a
    threshold's indexation is not a problem if a later provision restated the
    threshold with its indexation intact. Every replacement stays in the
    record for the manifest.
    """
    in_force: dict[tuple[str, str], Replacement] = {}
    for replacement in resolved.replacements:
        in_force[(replacement.parameter, replacement.subparameter)] = replacement
    return list(in_force.values())


def _dropped_indexation(resolved: ResolvedLaw):
    for replacement in _effective_replacements(resolved):
        if not replacement.dropped_indexation:
            continue
        block = replacement.reform
        if isinstance(block, Mapping) and block.get("remove_indexation") is True:
            continue
        yield Finding(
            DROPPED_INDEXATION,
            replacement.label,
            "the baseline indexes this subparameter and the reform restates it "
            "without indexation fields, which freezes it in nominal terms. Say "
            "`remove_indexation: true` in the block if that is the reform.",
            replacement.written_by,
        )


def _unknown_subparameter(resolved: ResolvedLaw):
    """A reform naming a lever the baseline does not have, undeclared.

    A subparameter is new when baseline law lacks it, whatever the stack
    order, so a provision naming a lever an earlier provision created still
    has to declare it.
    """
    for replacement in _effective_replacements(resolved):
        if not replacement.new_subparameter:
            continue
        if replacement.subparameter in RESERVED_KEYS:
            continue
        block = replacement.reform
        if isinstance(block, Mapping) and block.get("new_subparameter") is True:
            continue
        known = resolved.baseline_subparameters
        what = "subparameter" if replacement.parameter in known else "parameter file"
        yield Finding(
            UNKNOWN_SUBPARAMETER,
            replacement.label,
            f"baseline law has no such {what}"
            f"{_did_you_mean(known, replacement.parameter, replacement.subparameter)}"
            ". A misspelled name silently becomes a lever nothing reads, "
            "leaving the parameter it was meant to change at its baseline "
            "value. Say `new_subparameter: true` in the block if the lever is "
            "genuinely new.",
            replacement.written_by,
        )


def _did_you_mean(
    known: Mapping[str, Sequence[str]], parameter: str, subparameter: str
) -> str:
    """The nearest baseline name, if one is close enough to be the intended one."""
    if parameter in known:
        wanted, candidates = subparameter, [
            n for n in known[parameter] if n not in RESERVED_KEYS
        ]
        prefix = f"{parameter}."
    else:
        # The whole file is unrecognised, so the misspelling is in the
        # parameter name and the subparameters under it say nothing.
        wanted, candidates, prefix = parameter, list(known), ""
    close = difflib.get_close_matches(
        wanted, candidates, n=1, cutoff=_SUGGESTION_CUTOFF
    )
    return f" -- did you mean `{prefix}{close[0]}`?" if close else ""


def _schedule_overflow(
    resolved: ResolvedLaw, years: Sequence[int], max_brackets: int
):
    window = parse_years(max(int(y) for y in years))
    for parameter, subparams in resolved.parameters.items():
        for name, block in subparams.items():
            if name in RESERVED_KEYS or not isinstance(block, Mapping):
                continue
            if "value" not in block:
                continue
            width = _widest(block["value"], window)
            if width > max_brackets:
                yield Finding(
                    SCHEDULE_OVERFLOW,
                    f"{parameter}.{name}",
                    f"{width} slots, past the compiled maximum of {max_brackets}",
                    resolved.written_by(parameter, name),
                )


def _widest(value: Any, window: tuple[int, ...]) -> int:
    widths = [
        len(v) if isinstance(v, list) else 1
        for v in carry_forward(parse_inf(value), window)
        if v is not None
    ]
    return max(widths) if widths else 1


def _freeze_resume(resolved: ResolvedLaw, years: Sequence[int]):
    window = parse_years(max(int(y) for y in years))
    for parameter, subparams in resolved.parameters.items():
        defaults = subparams.get("indexation_defaults")
        for name, block in subparams.items():
            if name in RESERVED_KEYS or not isinstance(block, Mapping):
                continue
            if block.get("i_measure") is None:
                continue
            label = f"{parameter}.{name}"
            fields = resolve_indexation_defaults(
                indexation_fields_written(block),
                defaults,
                label,
            )
            measure = measure_by_year(fields["i_measure"], window, label)

            frozen_at: int | None = None
            for year in sorted(measure):
                if measure[year] is None:
                    if frozen_at is None:
                        frozen_at = year
                elif frozen_at is not None:
                    yield Finding(
                        FREEZE_RESUME,
                        label,
                        f"indexation stops in {frozen_at} and the config names "
                        f'"{measure[year]}" again in {year}. The index chain '
                        "cannot restart, so this compiles as a permanent "
                        "freeze. Restate the base value for the resumption "
                        "year and index from there.",
                        resolved.written_by(parameter, name),
                    )
                    break


# ---------------------------------------------------------------------------
# The schedule-length check, on the compiled arrays
# ---------------------------------------------------------------------------


def schedule_pairs(field_names: Iterable[str]) -> list[tuple[str, str]]:
    """The rate-and-threshold pairs among a compiled law's fields.

    A pair is a field whose name ends in `rates` together with the field of the
    same name ending in `brackets`: `ord.rates` with `ord.brackets`,
    `pr.oasdi_ee_rates` with `pr.oasdi_ee_brackets`. Both halves must exist,
    so a lone vector such as the itemized deduction's mortgage balance limit
    by loan vintage is not a pair.
    """
    names = set(field_names)
    pairs = []
    for name in sorted(names):
        if not name.endswith(_RATES_SUFFIX):
            continue
        thresholds = name[: -len(_RATES_SUFFIX)] + _THRESHOLDS_SUFFIX
        if thresholds in names:
            pairs.append((thresholds, name))
    return pairs


def _folders_behind(resolved: ResolvedLaw, field_name: str) -> tuple[str, ...]:
    """The law layers that wrote the blocks a compiled field was built from.

    A compiled field is a parameter file's subparameter after the
    filing-status mappers have run, so `ord.brackets` was built from
    `brackets_single`, `brackets_married` and `brackets_head`. A mapper's
    inputs are the field's own name plus a suffix. Baseline law is named when
    no reform folder wrote any of them.
    """
    parameter, _, subparameter = field_name.partition(".")
    folders: list[str] = []
    for replacement in resolved.replacements:
        if replacement.parameter != parameter:
            continue
        name = replacement.subparameter
        if name != subparameter and not name.startswith(f"{subparameter}_"):
            continue
        if replacement.written_by not in folders:
            folders.append(replacement.written_by)
    return tuple(folders) if folders else (DEFAULT_LAYER,)


def schedule_length_findings(
    active: Mapping[str, Any],
    years: Sequence[int],
    resolved: ResolvedLaw,
    filing_status_labels: Mapping[int, str],
) -> list[Finding]:
    """Every schedule whose live rate and threshold slots disagree.

    `active` is the compiled law's per-field mask, shaped
    [year, filing status, slot] and true where the law states a number. This
    is the test `taxes/calc/_schedules.py` applies when it decides whether a
    bracket slot is part of the schedule. `filing_status_labels` maps the
    filing status codes, in the order of the arrays' second axis, to their
    names.

    One finding per schedule per year per filing status, so the message can say
    which year and which filers the missing bracket belongs to.
    """
    statuses = list(filing_status_labels)
    findings: list[Finding] = []

    for thresholds_name, rates_name in schedule_pairs(active):
        thresholds = np.asarray(active[thresholds_name])
        rates = np.asarray(active[rates_name])
        if thresholds.shape != rates.shape:
            raise LawLintError(
                f"the compiled schedule {thresholds_name} is shaped "
                f"{thresholds.shape} and {rates_name} is shaped {rates.shape}; "
                "a compiled law's fields all share one shape"
            )

        wrote_thresholds = _folders_behind(resolved, thresholds_name)
        wrote_rates = _folders_behind(resolved, rates_name)
        blame = _last_folder_in(resolved, (*wrote_thresholds, *wrote_rates))

        for year_index, year in enumerate(years):
            for status_index, status in enumerate(statuses):
                live_thresholds = thresholds[year_index, status_index]
                live_rates = rates[year_index, status_index]
                if np.array_equal(live_thresholds, live_rates):
                    continue
                findings.append(
                    Finding(
                        SCHEDULE_LENGTH_MISMATCH,
                        f"{thresholds_name} and {rates_name}",
                        _schedule_length_detail(
                            year,
                            status,
                            filing_status_labels[status],
                            live_thresholds,
                            live_rates,
                            thresholds_name,
                            wrote_thresholds,
                            rates_name,
                            wrote_rates,
                        ),
                        blame,
                    )
                )

    return findings


def _last_folder_in(resolved: ResolvedLaw, folders: Iterable[str]) -> str:
    """The last of these folders in stack order.

    The latest folder to touch either half of the schedule is the one whose
    edit made the schedule inconsistent.
    """
    named = set(folders)
    for layer in reversed(resolved.layers):
        if layer in named:
            return layer
    return DEFAULT_LAYER


def _schedule_length_detail(
    year: int,
    status: int,
    status_label: str,
    live_thresholds: Any,
    live_rates: Any,
    thresholds_name: str,
    wrote_thresholds: Sequence[str],
    rates_name: str,
    wrote_rates: Sequence[str],
) -> str:
    """The error message, naming the year, the filers, and the folders."""
    threshold_slots = [int(i) + 1 for i in np.nonzero(live_thresholds)[0]]
    rate_slots = [int(i) + 1 for i in np.nonzero(live_rates)[0]]
    orphans = sorted(set(threshold_slots) ^ set(rate_slots))
    return (
        f"in {year}, filing status {status} ({status_label}), the law states "
        f"{len(threshold_slots)} bracket threshold(s) and {len(rate_slots)} "
        f"rate(s). A bracket counts only where both are present, so "
        f"bracket(s) {', '.join(str(s) for s in orphans)} disappear and the "
        "income in them is taxed at the rate of the bracket below. "
        f"{thresholds_name} comes from {_name_folders(wrote_thresholds)}; "
        f"{rates_name} comes from {_name_folders(wrote_rates)}. Restate both "
        "halves at the same length in whichever folder is meant to set the "
        "schedule."
    )


def _name_folders(folders: Sequence[str]) -> str:
    if list(folders) == [DEFAULT_LAYER]:
        return "baseline law"
    return ", ".join(f"`{folder}`" for folder in folders)


def check_schedules(
    active: Mapping[str, Any],
    years: Sequence[int],
    resolved: ResolvedLaw,
    filing_status_labels: Mapping[int, str],
) -> list[Finding]:
    """Stop a compile whose schedules have mismatched rates and thresholds.

    This check has no layer scoping, because it is a statement about the
    arrays this compile just produced.
    """
    findings = schedule_length_findings(active, years, resolved, filing_status_labels)
    if not findings:
        return findings

    plural = "schedule" if len(findings) == 1 else "schedules"
    detail = "\n  ".join(str(f) for f in findings)
    raise LawLintError(
        f"tax law layer '{resolved.alternative}' has {len(findings)} mismatched "
        f"{plural}:\n  {detail}"
    )
