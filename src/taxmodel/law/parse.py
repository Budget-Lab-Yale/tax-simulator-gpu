"""Tax law YAML parsing.

A parameter file (`ctc.yaml`) holds subparameters (`po_thresh_married`). Each
subparameter has a `value` (a scalar, a vector across schedule slots, or a
year-keyed set of either) and optionally four fields saying how the value
grows with prices or wages: `i_measure`, `i_base_year`, `i_direction`,
`i_increment`. Two reserved top-level keys are not subparameters:
`indexation_defaults`, the file's house indexation rule, and
`filing_status_mapper`, which builds the by-filing-status subparameters the
calculator reads.

Three conventions of the law tree:

  Years are stated only when they change. A value given for 2018 holds until
  the next stated year: last observation carried forward, never interpolated,
  and never carried backward. Years before the first stated one are NaN with
  the active mask off.

  `Inf` means no limit. A phaseout threshold of `Inf` is a phaseout that never
  starts; a refundability cap of `Inf` is a fully refundable credit. It is a
  real value in the schedule, not a missing one.

  `NA` inside a numeric indexation field means that slot alone is not indexed,
  as in the donut-hole payroll design, where the taxable maximum is indexed and
  the second bracket above it is frozen in nominal terms.

Everything here is per-subparameter, over the parse window. By-filing-status
arrays are built in `compile.py`; the indexation arithmetic is in
`indexation.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config.law_layers import INDEXATION_FIELDS
from .errors import LawError

# Every scenario is parsed from 2014 forward regardless of the years it
# simulates. An indexation chain reads the prior year's index, so the parse
# window has to start before the first simulated year.
FIRST_PARSE_YEAR = 2014

RESERVED_KEYS = ("indexation_defaults", "filing_status_mapper")

# Declaration keys a reform author writes. `remove_indexation: true` declares
# that the author meant to drop the indexation; without it, lint.py refuses a
# reform that drops a baseline's indexation (ARCHITECTURE.md D4).
# `new_subparameter: true` declares that this lever does not exist in baseline
# law; without it, lint.py reads an unrecognised name as a typo. Neither key
# carries a value into the compiled law.
DECLARATION_KEYS = ("remove_indexation", "new_subparameter")

# The value an author writes to mean "use this file's house rule".
DEFAULT_SENTINEL = "default"


@dataclass(frozen=True)
class ParsedSubparameter:
    """One subparameter's base values and indexation rule, year by slot.

    Every matrix is `[year, slot]` over `years`, NaN where the year precedes
    the first year the author stated or the slot does not exist that year (a
    schedule that gains a bracket in 2026 has nothing in slot 3 before then).

    `measure` is the index series named for each year of the chain, over its own
    year range: the chain starts at the first year the author named a measure
    for, which is usually well before the parse window, because the index has to
    accumulate from the base year of the statute.
    """

    parameter: str
    subparameter: str
    years: tuple[int, ...]
    base_value: np.ndarray
    indexed: bool
    measure: dict[int, str | None] | None
    i_base_year: np.ndarray | None
    i_direction: np.ndarray | None
    i_increment: np.ndarray | None

    @property
    def label(self) -> str:
        return f"{self.parameter}.{self.subparameter}"


def parse_years(last_year: int) -> tuple[int, ...]:
    """The years the parser generates values for, given the last simulated year."""
    if last_year < FIRST_PARSE_YEAR:
        raise LawError(
            f"last simulated year {last_year} precedes {FIRST_PARSE_YEAR}, the "
            "first year the law tree states values for"
        )
    return tuple(range(FIRST_PARSE_YEAR, last_year + 1))


# --------------------------------------------------------------------------
# 'Inf' and 'NA'
# --------------------------------------------------------------------------


def _walk_atomic(value: Any, fn):
    """Apply `fn` to a year-keyed block's values, or to the value itself."""
    if isinstance(value, dict):
        return {k: fn(v) for k, v in value.items()}
    return fn(value)


def parse_inf(value: Any) -> Any:
    """Turn the string `Inf` in a base value into an unbounded number.

    A vector with `Inf` in any slot is converted to numbers as a whole, so
    `[200000, Inf]` is a schedule: phase out the child credit above $200k, and
    never phase out the second component.
    """

    def convert(x):
        items = x if isinstance(x, list) else [x]
        if any(isinstance(i, str) and i == "Inf" for i in items):
            converted = [_as_float(i) for i in items]
            return converted if isinstance(x, list) else converted[0]
        return x

    return _walk_atomic(value, convert)


def parse_na(value: Any) -> Any:
    """Coerce an indexation field to numbers, `NA` becoming a missing slot.

    A missing slot means that slot is not indexed: its value stays where the
    author put it while its neighbours grow.
    """
    return _walk_atomic(value, lambda x: [_as_float(i) for i in x]
                        if isinstance(x, list) else _as_float(x))


def _as_float(x: Any) -> float:
    if x is None:
        return math.nan
    if isinstance(x, bool):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    text = str(x).strip()
    if text in ("NA", "", "NaN", "null", "None"):
        return math.nan
    try:
        return float(text)
    except ValueError as exc:
        raise LawError(
            f"tax law value {x!r} is neither a number, 'Inf', nor 'NA'"
        ) from exc


# --------------------------------------------------------------------------
# Years stated only when they change
# --------------------------------------------------------------------------


def carry_forward(value: Any, years: tuple[int, ...]) -> list[Any]:
    """Expand a minimally stated value into one entry per year.

    A scalar or a vector with no year keys holds for every year, and a
    year-keyed block holds each stated value until the next stated year. Years
    before the first stated one get `None`; values are never carried backward.
    """
    if not isinstance(value, dict):
        return [value] * len(years)

    stated = {}
    for key, item in value.items():
        try:
            stated[int(key)] = item
        except (TypeError, ValueError) as exc:
            raise LawError(f"year key {key!r} is not a year") from exc

    out: list[Any] = []
    current: Any = None
    for year in years:
        if year in stated:
            current = stated[year]
        out.append(current)
    return out


def to_matrix(per_year: list[Any], years: tuple[int, ...]) -> np.ndarray:
    """Stack per-year values into `[year, slot]`, NaN where a year has no slot.

    Slot count can differ by year. The child credit states a single `NA` base
    year through 2024 and a two-slot base year from 2025, so slot 2 has no
    indexation rule until the year the second component appears.
    """
    widths = [len(v) if isinstance(v, list) else 1 for v in per_year if v is not None]
    n_slots = max(widths) if widths else 1

    out = np.full((len(years), n_slots), np.nan, dtype=np.float64)
    for row, value in enumerate(per_year):
        if value is None:
            continue
        items = value if isinstance(value, list) else [value]
        for slot, item in enumerate(items):
            out[row, slot] = _as_float(item)
    return out


# --------------------------------------------------------------------------
# One subparameter
# --------------------------------------------------------------------------


def parse_subparameter(
    raw: Any,
    parameter: str,
    subparameter: str,
    indexation_defaults: dict[str, Any] | None,
    years: tuple[int, ...],
) -> ParsedSubparameter:
    """Read one subparameter block into base values and an indexation rule."""

    if not isinstance(raw, dict):
        raise LawError(
            f"{parameter}.{subparameter} is not a block of value and indexation "
            "fields"
        )
    if "value" not in raw:
        raise LawError(f"{parameter}.{subparameter} states no value")

    base_value = to_matrix(carry_forward(parse_inf(raw["value"]), years), years)

    # No measure means no indexation, whatever else the block says. A reform
    # that restates a threshold with a base year and an increment but drops the
    # measure has un-indexed it; lint.py reports that as an error.
    if raw.get("i_measure") is None:
        return ParsedSubparameter(
            parameter=parameter,
            subparameter=subparameter,
            years=years,
            base_value=base_value,
            indexed=False,
            measure=None,
            i_base_year=None,
            i_direction=None,
            i_increment=None,
        )

    fields = resolve_indexation_defaults(
        indexation_fields_written(raw),
        indexation_defaults,
        f"{parameter}.{subparameter}",
    )

    measure = measure_by_year(fields["i_measure"], years, f"{parameter}.{subparameter}")

    return ParsedSubparameter(
        parameter=parameter,
        subparameter=subparameter,
        years=years,
        base_value=base_value,
        indexed=True,
        measure=measure,
        i_base_year=to_matrix(
            carry_forward(parse_na(fields["i_base_year"]), years), years
        ),
        i_direction=to_matrix(
            carry_forward(parse_na(fields["i_direction"]), years), years
        ),
        i_increment=to_matrix(
            carry_forward(parse_na(fields["i_increment"]), years), years
        ),
    )


def indexation_fields_written(block: dict[str, Any]) -> dict[str, Any]:
    """The indexation fields a block states, without the value or the
    declaration keys."""
    return {
        k: v for k, v in block.items() if k != "value" and k not in DECLARATION_KEYS
    }


def resolve_indexation_defaults(
    fields: dict[str, Any],
    indexation_defaults: dict[str, Any] | None,
    label: str,
) -> dict[str, Any]:
    """Replace each `default` with the file's house rule for that field."""
    if indexation_defaults is None:
        missing = [f for f in INDEXATION_FIELDS if f not in fields]
        if missing:
            raise LawError(
                f"{label} is indexed but its file states no indexation_defaults "
                f"and the block omits {', '.join(missing)}"
            )
        return dict(fields)

    if tuple(fields.keys()) != INDEXATION_FIELDS:
        raise LawError(
            f"{label} lists indexation fields as {tuple(fields.keys())}; they "
            f"are paired with the file's defaults by position, so they must "
            f"read {INDEXATION_FIELDS}"
        )
    if tuple(indexation_defaults.keys()) != INDEXATION_FIELDS:
        raise LawError(
            f"the indexation_defaults for {label.split('.')[0]} read "
            f"{tuple(indexation_defaults.keys())}; they must read "
            f"{INDEXATION_FIELDS}"
        )

    return {
        name: (
            indexation_defaults[name]
            if isinstance(fields[name], str) and fields[name] == DEFAULT_SENTINEL
            else fields[name]
        )
        for name in INDEXATION_FIELDS
    }


def measure_by_year(
    raw_measure: Any, years: tuple[int, ...], label: str
) -> dict[int, str | None]:
    """The index series in force each year, from the year the chain starts.

    The chain starts at the first year the author named a measure for: a
    bracket with a 1992 base year needs price growth accumulated from 1992
    forward. Ordinary brackets switch from CPI-U to chained CPI-U in 2017 under
    the TCJA, written as a measure of `cpi` from 1986 and `chained_cpi` from
    2017.
    """
    if not isinstance(raw_measure, dict):
        raise LawError(
            f"{label} names index measure {raw_measure!r} with no start year. "
            "The measure must be a year-keyed block, because the index chain "
            "starts where the measure starts."
        )

    stated = {int(k): (None if v is None else str(v)) for k, v in raw_measure.items()}
    first = min(stated)
    chain_years = range(first, max(years) + 1)

    out: dict[int, str | None] = {}
    current: str | None = None
    for year in chain_years:
        if year in stated:
            current = stated[year]
        # 'NA' as a measure is a freeze: indexing stops as of that year, as
        # the TCJA froze the lifetime learning credit phaseout in 2020.
        out[year] = None if current in (None, "NA") else current
    return out


def parse_parameter(
    raw_file: dict[str, Any], parameter: str, years: tuple[int, ...]
) -> dict[str, ParsedSubparameter]:
    """Read one parameter file's subparameters, defaults already folded in."""
    defaults = raw_file.get("indexation_defaults")
    return {
        name: parse_subparameter(block, parameter, name, defaults, years)
        for name, block in raw_file.items()
        if name not in RESERVED_KEYS
    }
