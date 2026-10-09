"""The law compiler: a folder of YAML files becomes the arrays a calculator reads.

The output is one array per compiled field, shaped `[year, filing_status,
slot]`, plus a mask saying which slots are part of the schedule that year. A
field is a parameter file's subparameter after the filing-status mappers have
run: `ord.brackets` exists in the compiled law, `ord.brackets_single` does not,
because the mapper consumed it.

Three shape conventions:

  Filing status is always four wide, in the model's order (1 single, 2 joint,
  3 married filing separately, 4 head of household), even for a parameter that
  does not vary by it. A rate schedule that is the same for everyone is stored
  four times, so the calculator has a single indexing convention.

  Slots are always MAX_BRACKETS wide, so the two laws of a repricing have the
  same shape.

  Unused slots hold NaN. A bracket that does not exist is not a bracket at
  zero dollars, and a calculator that forgets the mask produces a visibly
  wrong answer.

`to_frame()` and `write_csv()` produce the human-readable table: rows long in
year and filing status, one column per field, vector fields spread across
numbered columns.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..config.law_layers import Replacement, ResolvedLaw, resolve_law
from . import mappers
from .errors import LawError
from .indexation import IndexTable, apply_indexation, index_ratios
from .lint import check_law, check_schedules
from .parse import (
    ParsedSubparameter,
    parse_parameter,
    parse_years,
)

# Filing status codes, as the model numbers them throughout.
FILING_STATUSES = (1, 2, 3, 4)
FILING_STATUS_LABELS = {
    1: "single",
    2: "married filing jointly",
    3: "married filing separately",
    4: "head of household",
}

# The most slots a schedule can hold. The widest schedule in the law tree is
# the fourteen-bracket estate schedule under alternatives/public/estate_2009;
# the baseline estate schedule has twelve and the ordinary rate schedule seven.
# A law that needs more than sixteen is a lint error (ARCHITECTURE.md D4).
MAX_BRACKETS = 16

# Some subparameter names end in a number: the EITC parameters per child count
# (eitc.po_rate_0 through eitc.po_rate_3), the surtax rate and threshold per
# bracket (surtax.rate1 through rate3). The number is removed before the unit
# rules are matched.
_ENUMERATED_SUFFIX = re.compile(r"_?\d+$")

# The units of each compiled field, matched in order against the subparameter
# name; first match wins. Units are assigned by name, so a field of all zeros
# is still dollars if the statute writes it in dollars.
UNIT_RULES: tuple[tuple[str, str], ...] = (
    # The medical and miscellaneous itemized floors and the charitable
    # ceilings are written in the Code as percentages of AGI.
    (r"_agi$", "share of AGI"),
    # Ages, counts of dependents, and holding periods in whole years.
    (
        (
            r"_age$|_age_limit$|^max_age$|^min_age$|_years_held$|_limit_years$"
            r"|^n_dep_limit$"
        ),
        "count",
    ),
    # A year written into the statute: when a provision starts or stops.
    (r"_year$|_purchased_after$", "calendar year"),
    # One of a few named regimes for gains held at death (step-up, carryover,
    # deemed realization). An integer code, not a quantity.
    (r"_regime_", "regime code"),
    # On or off. Repeals, eligibility, phaseout-form switches, and the flags
    # that pick between two ways of stating the same parameter.
    (
        (
            r"^repeal$|^repeal_|_repeal$|_eligible$|_exempt$|_type$|_level$"
            r"|_qualify$|^refundable$|_precert$|^portability$"
            r"|_discrete$|_sequential$|_prior_yr$|_in_credit$|_ssn$|_ssn_req$"
            r"|_new$|_half$|_lh$|^include_active$|^tax_at_ord$|^no_ord_cap$"
            r"|^income_tax_ded$|^index_kg$|_no_loss$"
        ),
        "indicator",
    ),
    # A fraction: of the base a schedule applies to, of a credit's value, of
    # income in a phaseout range, or of the taxpayers who can use a provision.
    # The pass-through SALT workaround is written as the share of the tax
    # allowed through it, which is a share and not a switch.
    (r"(^|_)rates?$|_rate_addon$|_share$|_allowed_", "rate"),
    # A multiple of another parameter. The alternative maximum tax's
    # qualifying threshold is stated as a multiple of the standard deduction.
    (r"_mult$", "ratio"),
)

# The unit assigned when nothing above matches. Almost everything in the tree
# that is not a rate or a switch is a dollar amount.
DEFAULT_UNIT = "dollars"

# Units whose values are not continuous quantities, so a derivative with
# respect to them means nothing (ARCHITECTURE.md D4).
NON_DIFFERENTIABLE_UNITS = frozenset({"indicator", "count", "calendar year", "regime code"})


@dataclass(frozen=True)
class FieldSpec:
    """Description of one compiled field."""

    name: str
    parameter: str
    subparameter: str
    dtype: str
    units: str
    slots_used: int
    varies_by_filing_status: bool
    indexed: bool
    from_mapper: bool
    differentiable: bool
    padding: str


@dataclass
class CompiledLaw:
    """One scenario's law, compiled.

    `values[name]` and `active[name]` are `[year, filing_status, slot]`, aligned
    to `years` and to FILING_STATUSES. `active` is false where the law has
    nothing: a slot beyond the schedule's length, or a year before the first the
    author stated a value for.

    `table_slots` is empty for a law compiled on its own and filled in when
    several laws are compiled together and share one layout (`law/batch.py`).
    It affects only the layout, never the values.
    """

    alternative: str
    years: tuple[int, ...]
    max_brackets: int
    values: dict[str, np.ndarray]
    active: dict[str, np.ndarray]
    schema: dict[str, FieldSpec]
    replacements: list[Replacement]
    # The number of numbered columns a field takes in the wide law table, where
    # the batch gives it more than the slots this law fills. Empty means every
    # field gets exactly the slots it fills.
    table_slots: dict[str, int] = field(default_factory=dict)

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(self.schema)

    def columns_in_table(self, name: str) -> int:
        """The number of columns of the wide law table this field takes."""
        return max(self.table_slots.get(name, 0), self.schema[name].slots_used)

    def differentiable_mask(self, name: str) -> np.ndarray:
        """The entries of a field that belong in the differentiable parameter
        vector.

        Padding is excluded, and so are infinite entries, since a derivative
        through an unbounded limit is `0 * Inf`. Switches and counts are
        excluded wholesale.
        """
        spec = self.schema[name]
        if not spec.differentiable:
            return np.zeros_like(self.active[name])
        return self.active[name] & np.isfinite(self.values[name])

    # -- the human-readable table -------------------------------------------

    def column_names(self) -> list[str]:
        """Column names of the wide law table, in order."""
        columns: list[str] = []
        for name in self.schema:
            slots = self.columns_in_table(name)
            if slots <= 1:
                columns.append(name)
            else:
                columns.extend(f"{name}{i + 1}" for i in range(slots))
        return columns

    def to_frame(self) -> tuple[list[str], list[list[Any]]]:
        """Rows long in year and filing status and wide in field, header first."""
        header = ["year", "filing_status", *self.column_names()]
        rows: list[list[Any]] = []
        for y, year in enumerate(self.years):
            for f, status in enumerate(FILING_STATUSES):
                row: list[Any] = [year, status]
                for name in self.schema:
                    slots = max(self.columns_in_table(name), 1)
                    row.extend(self.values[name][y, f, :slots].tolist())
                rows.append(row)
        return header, rows

    def write_csv(self, path: str | Path) -> None:
        header, rows = self.to_frame()
        with open(path, "w") as out:
            out.write(",".join(header) + "\n")
            out.writelines(",".join(_csv_cell(v) for v in row) + "\n" for row in rows)

    # -- the frozen artefacts -----------------------------------------------

    def save_npz(self, path: str | Path) -> None:
        payload: dict[str, np.ndarray] = {
            "years": np.asarray(self.years, dtype=np.int32),
            "filing_statuses": np.asarray(FILING_STATUSES, dtype=np.int32),
        }
        for name in self.schema:
            payload[f"value/{name}"] = self.values[name]
            payload[f"active/{name}"] = self.active[name]
        np.savez_compressed(path, **payload)

    def schema_dict(self) -> dict[str, Any]:
        return {
            "alternative": self.alternative,
            "years": list(self.years),
            "filing_statuses": list(FILING_STATUSES),
            "max_brackets": self.max_brackets,
            "array_shape": [len(self.years), len(FILING_STATUSES), self.max_brackets],
            "fields": [asdict(spec) for spec in self.schema.values()],
        }

    def write_schema_json(self, path: str | Path) -> None:
        with open(path, "w") as out:
            json.dump(self.schema_dict(), out, indent=2, sort_keys=False)
            out.write("\n")


def _csv_cell(value: Any) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return "NA"
        if np.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return repr(value)
    return str(value)


# --------------------------------------------------------------------------
# Compiling
# --------------------------------------------------------------------------


def compile_law(
    config_root: str | Path,
    years: Sequence[int],
    index_table: IndexTable,
    alternative: str | None = None,
    max_brackets: int = MAX_BRACKETS,
    lint: bool = True,
    lint_scope: Iterable[str] | None = None,
) -> CompiledLaw:
    """Compile one law layer laid over baseline, for a run's years."""
    resolved = resolve_law(Path(config_root), alternative)
    return compile_resolved(
        resolved, years, index_table, max_brackets, lint, lint_scope
    )


def compile_resolved(
    resolved: ResolvedLaw,
    years: Sequence[int],
    index_table: IndexTable,
    max_brackets: int = MAX_BRACKETS,
    lint: bool = True,
    lint_scope: Iterable[str] | None = None,
) -> CompiledLaw:
    """Compile an already-layered parameter tree (see config/law_layers.py).

    `lint_scope` is the set of law layers whose lint findings stop the
    compile, normally the runscript's `tax_law` cells. Left out, it is the
    layers stacked into `resolved`. Findings from any other folder of the tree
    do not stop the compile.
    """
    run_years = tuple(int(y) for y in years)
    if not run_years:
        raise LawError("a compiled law needs at least one year")

    # The lint runs before any arithmetic.
    if lint:
        check_law(resolved, run_years, max_brackets, lint_scope)

    # The parse window starts at 2014 whatever the run's window, because
    # indexation and carry-forward both read years the run does not simulate.
    window = parse_years(max(run_years))
    keep = [window.index(y) for y in run_years]

    values: dict[str, np.ndarray] = {}
    active: dict[str, np.ndarray] = {}
    schema: dict[str, FieldSpec] = {}

    for parameter, raw_file in resolved.parameters.items():
        for name, per_status, spec_bits in _compile_parameter(
            parameter, raw_file, window, index_table, max_brackets
        ):
            block = np.full(
                (len(window), len(FILING_STATUSES), max_brackets), np.nan, np.float64
            )
            for f, status in enumerate(FILING_STATUSES):
                arr = per_status[status]
                block[:, f, : arr.shape[1]] = arr

            block = block[keep]
            live = ~np.isnan(block)
            slots_used = _slots_used(live)

            values[name] = block
            active[name] = live
            schema[name] = _field_spec(name, slots_used, **spec_bits)

    # This lint needs the finished arrays: a schedule whose live rate slots
    # and live threshold slots disagree loses a bracket silently, and whether
    # they agree depends on the padding and the filing-status mappers.
    if lint:
        check_schedules(active, run_years, resolved, FILING_STATUS_LABELS)

    return CompiledLaw(
        alternative=resolved.alternative,
        years=run_years,
        max_brackets=max_brackets,
        values=values,
        active=active,
        schema=schema,
        replacements=list(resolved.replacements),
    )


def _slots_used(live: np.ndarray) -> int:
    """The highest slot the law ever fills, in any year or filing status.

    A subparameter that never fills more than one slot is scalar and is named
    in the wide table without a slot number.
    """
    occupied = np.nonzero(live.any(axis=(0, 1)))[0]
    return int(occupied[-1]) + 1 if occupied.size else 1


def _compile_parameter(
    parameter: str,
    raw_file: Mapping[str, Any],
    window: tuple[int, ...],
    index_table: IndexTable,
    max_brackets: int,
):
    """Yield (field name, {filing status: [year, slot]}, spec bits) for one file."""
    parsed = parse_parameter(dict(raw_file), parameter, window)
    mapper = raw_file.get("filing_status_mapper") or {}

    # Every subparameter, indexed and rounded, on a common slot width so the
    # mapper arithmetic lines up slot by slot.
    resolved_values: dict[str, np.ndarray] = {}
    for name, sub in parsed.items():
        resolved_values[name] = _resolve_subparameter(sub, index_table)

    width = max((v.shape[1] for v in resolved_values.values()), default=1)
    if width > max_brackets:
        raise LawError(
            f"{parameter} has a schedule {width} slots long, past the compiled "
            f"maximum of {max_brackets}. Widen MAX_BRACKETS in "
            "src/taxmodel/law/compile.py and recompile every frozen fixture."
        )
    operands = {name: _pad(arr, width) for name, arr in resolved_values.items()}

    # A subparameter named in a mapper expression is an input to the mapper and
    # is not emitted as a field of its own.
    consumed = set().union(
        *(
            mappers.identifiers(expression)
            for by_status in mapper.values()
            for expression in by_status.values()
        )
    )

    # Fields are ordered by subparameter name within the file.
    emitted: list[tuple[str, dict[int, np.ndarray], dict[str, Any]]] = []

    for new_name, by_status in mapper.items():
        per_status = {}
        for status in FILING_STATUSES:
            expression = by_status.get(str(status), by_status.get(status))
            if expression is None:
                raise LawError(
                    f"{parameter}.{new_name} has no mapper expression for filing "
                    f"status {status} ({FILING_STATUS_LABELS[status]})"
                )
            result = mappers.evaluate(expression, operands)
            per_status[status] = np.broadcast_to(
                np.asarray(result, dtype=np.float64), (len(window), width)
            )
        inputs = set().union(
            *(mappers.identifiers(e) for e in by_status.values())
        )
        emitted.append(
            (
                f"{parameter}.{new_name}",
                per_status,
                {
                    "parameter": parameter,
                    "subparameter": new_name,
                    "varies_by_filing_status": len(set(by_status.values())) > 1,
                    "indexed": any(
                        parsed[n].indexed for n in inputs if n in parsed
                    ),
                    "from_mapper": True,
                },
            )
        )

    for name, sub in parsed.items():
        if name in consumed:
            continue
        emitted.append(
            (
                f"{parameter}.{name}",
                dict.fromkeys(FILING_STATUSES, operands[name]),
                {
                    "parameter": parameter,
                    "subparameter": name,
                    "varies_by_filing_status": False,
                    "indexed": sub.indexed,
                    "from_mapper": False,
                },
            )
        )

    yield from sorted(emitted, key=lambda item: item[2]["subparameter"])


def _resolve_subparameter(
    sub: ParsedSubparameter, index_table: IndexTable
) -> np.ndarray:
    """Base values grown by the index and rounded, `[year, slot]`."""
    if not sub.indexed:
        return sub.base_value.astype(np.float64, copy=True)

    assert sub.i_base_year is not None
    assert sub.i_direction is not None
    assert sub.i_increment is not None
    assert sub.measure is not None

    width = max(
        sub.base_value.shape[1],
        sub.i_base_year.shape[1],
        sub.i_direction.shape[1],
        sub.i_increment.shape[1],
    )
    ratio = index_ratios(
        sub.measure, _pad(sub.i_base_year, width), index_table, sub.years, sub.label
    )
    return apply_indexation(
        _pad(sub.base_value, width),
        ratio,
        _pad(sub.i_direction, width),
        _pad(sub.i_increment, width),
        sub.label,
    )


def _pad(arr: np.ndarray, width: int) -> np.ndarray:
    if arr.shape[1] == width:
        return arr
    out = np.full((arr.shape[0], width), np.nan, dtype=np.float64)
    out[:, : arr.shape[1]] = arr
    return out


def units_of(subparameter: str) -> str:
    """The units of a subparameter, per UNIT_RULES."""
    stem = _ENUMERATED_SUFFIX.sub("", subparameter)
    for pattern, unit in UNIT_RULES:
        if re.search(pattern, stem):
            return unit
    return DEFAULT_UNIT


def _field_spec(
    name: str,
    slots_used: int,
    parameter: str,
    subparameter: str,
    varies_by_filing_status: bool,
    indexed: bool,
    from_mapper: bool,
) -> FieldSpec:
    units = units_of(subparameter)
    return FieldSpec(
        name=name,
        parameter=parameter,
        subparameter=subparameter,
        dtype="float64",
        units=units,
        slots_used=slots_used,
        varies_by_filing_status=varies_by_filing_status,
        indexed=indexed,
        from_mapper=from_mapper,
        differentiable=units not in NON_DIFFERENTIABLE_UNITS,
        padding="nan",
    )


__all__ = [
    "FILING_STATUSES",
    "FILING_STATUS_LABELS",
    "MAX_BRACKETS",
    "CompiledLaw",
    "FieldSpec",
    "compile_law",
    "compile_resolved",
    "units_of",
]
