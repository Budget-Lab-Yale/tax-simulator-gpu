"""Compiling several laws to one shared layout.

Scenarios that run in one compiled program must pass their laws as arguments
of identical shape. `compile_batch` compiles a list of laws together and
gives them one layout.

Filing status is already uniform: every compiled law holds every parameter as
its four filing-status values, whether or not the four differ
(`taxes/law_for_calculators.py`).

Two aspects of a compiled law's layout depend on the law:

  The set of levers. A reform can create a lever baseline law does not have.
  A law without the lever carries it as NaN.

  The number of brackets a schedule fills. Every schedule is compiled
  MAX_BRACKETS slots wide, so this changes no array the calculators read. It
  does change the wide law table the frame layer joins onto records, whose
  columns are numbered one per bracket the law fills. In a batch, every law's
  table gets as many columns as the widest law fills.

Widening a law's layout changes nobody's tax under that law. Each place a
shape can widen carries a comment saying why, and
`tests/unit/law/test_batch.py` has one test per reason.

A batch must also agree on the decisions taken in Python before the
calculator chain is traced, because those change which functions run. There
is one such decision: whether a filer may choose between the above-the-line
and the itemized charitable deduction, which makes the model run the whole
1040 twice. `charity_double_run` answers it per law, and `program_fingerprint`
is the label under which scenarios that run the same code share a compiled
program.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from ..config.law_layers import ResolvedLaw
from .compile import FILING_STATUSES, MAX_BRACKETS, CompiledLaw, compile_resolved
from .errors import LawError
from .indexation import IndexTable

# The law columns the frame layer reads by their unnumbered names in
# `frame/steps_jax.py`. A batch that would give one of these a numbered column
# is refused.
FRAME_LAW_COLUMNS_READ_BY_NAME: tuple[str, ...] = (
    "amt.exempt",
    "filing.repeal_hoh",
    "item.salt_limit",
    "item.salt_workaround_allowed_non_sstb",
    "item.salt_workaround_allowed_part",
    "item.salt_workaround_allowed_sstb",
)

# The two compiled fields that decide whether a filer may choose where to
# report a charitable gift. `charity_double_run_applies` in
# `taxes/statutory_order.py` asks the same question of a bound law.
CHARITY_ABOVE_THE_LINE_LIMIT = "char.above_limit"
CHARITY_ITEMIZED_LIMIT = "char.item_limit"


@dataclass(frozen=True)
class CompiledBatch:
    """Laws compiled to one layout.

    `laws` are the compiled laws in the order they were asked for, each
    carrying the shared layout. `charity_double_run` says, law by law, whether
    the law makes the model compute the 1040 twice for the charitable choice.
    Laws that agree can share a compiled program.
    """

    laws: tuple[CompiledLaw, ...]
    charity_double_run: tuple[bool, ...]


def compile_batch(
    resolved_laws: Sequence[ResolvedLaw],
    years: Sequence[int],
    index_table: IndexTable,
    max_brackets: int = MAX_BRACKETS,
    lint: bool = True,
    lint_scope: Iterable[str] | None = None,
) -> CompiledBatch:
    """Compile a list of resolved laws to one layout they all share.

    The arguments through `lint_scope` are passed to `compile_resolved`, which
    compiles each law. This function then gives the laws a shared layout and
    answers the charitable double-run question for each.
    """
    if not resolved_laws:
        raise LawError("a batch needs at least one law")

    laws = [
        compile_resolved(resolved, years, index_table, max_brackets, lint, lint_scope)
        for resolved in resolved_laws
    ]

    laws = _give_every_law_the_same_fields(laws)
    table_slots = _shared_table_slots(laws)
    _refuse_to_renumber_columns_the_frame_reads(laws, table_slots)

    shared = tuple(replace(law, table_slots=dict(table_slots)) for law in laws)
    double_run = tuple(charity_double_run(law) for law in shared)
    return CompiledBatch(laws=shared, charity_double_run=double_run)


# --------------------------------------------------------------------------
# The two layout decisions
# --------------------------------------------------------------------------


def _shared_table_slots(laws: Sequence[CompiledLaw]) -> dict[str, int]:
    """The number of columns each field takes in every law's wide table.

    Each field gets the widest slot count any law in the batch fills, so the
    whole batch has one column set.

    The widening changes no tax. A schedule is compiled MAX_BRACKETS slots
    wide however many brackets the law creates, and the slots the law does not
    create hold NaN. The schedule readers in taxes/calc/_schedules.py count a
    slot as part of the schedule only where both its threshold and its rate
    are present. In the wide law table, a numbered column beyond the brackets
    a law creates is NaN, and the frame layer reads only the columns it names.
    """
    slots: dict[str, int] = {}
    for law in laws:
        for name, spec in law.schema.items():
            slots[name] = max(slots.get(name, 0), spec.slots_used)
    return slots


def _give_every_law_the_same_fields(
    laws: Sequence[CompiledLaw],
) -> list[CompiledLaw]:
    """Give every law the whole batch's field list, NaN where it has no lever.

    A reform may create a lever baseline law does not have (`new_subparameter`
    in `config/law_layers.py`), and restating a parameter file's filing-status
    mapper can leave a law without a field the other laws have. The law that
    lacks the lever carries it as NaN.

    A missing field that some calculator reads is refused, since a calculator
    handed NaN produces a NaN tax. Padding is applied only to levers no
    calculator reads.

    Every law comes back with its fields in one order: the first law's, with
    levers only later laws have appended.
    """
    from ..taxes import (
        law_for_calculators,  # here, so the law layer still imports on its own
    )

    field_order: dict[str, CompiledLaw] = {}
    for law in laws:
        for name in law.schema:
            field_order.setdefault(name, law)

    read_by_calculators = set(law_for_calculators.declared_fields())
    shared: list[CompiledLaw] = []
    for law in laws:
        missing = [name for name in field_order if name not in law.schema]
        unsupplied = sorted(set(missing) & read_by_calculators)
        if unsupplied:
            raise LawError(
                f"law '{law.alternative}' has no value for "
                f"{', '.join(unsupplied)}, which the calculators read, so it "
                "cannot share a compiled program with laws that do. Padding the "
                "field with NaN would give this law a NaN tax rather than the "
                "tax it states."
            )
        empty = np.full(
            (len(law.years), len(FILING_STATUSES), law.max_brackets),
            np.nan,
            np.float64,
        )
        values = {}
        active = {}
        schema = {}
        for name in field_order:
            if name in law.schema:
                values[name] = law.values[name]
                active[name] = law.active[name]
                schema[name] = law.schema[name]
                continue
            values[name] = empty.copy()
            active[name] = np.zeros_like(empty, dtype=bool)
            # The description of the field comes from the law that has the
            # lever, while the count of slots filled is this law's own, and it
            # fills none.
            schema[name] = replace(field_order[name].schema[name], slots_used=1)
        shared.append(replace(law, values=values, active=active, schema=schema))
    return shared


def _refuse_to_renumber_columns_the_frame_reads(
    laws: Sequence[CompiledLaw], table_slots: dict[str, int]
) -> None:
    """Stop if the batch would put a number on a law column a frame step names.

    A parameter that fills one slot is a column with no number on it, and one
    that fills two is columns 1 and 2. Widening a single-valued parameter
    renames its column, and the frame layer reads the law columns in
    `FRAME_LAW_COLUMNS_READ_BY_NAME` by their unnumbered names.
    """
    renumbered = [
        name
        for name in FRAME_LAW_COLUMNS_READ_BY_NAME
        if table_slots.get(name, 1) > 1
        and any(law.schema[name].slots_used == 1 for law in laws if name in law.schema)
    ]
    if renumbered:
        raise LawError(
            "these laws cannot share one wide law table: "
            + ", ".join(renumbered)
            + " fills one slot in some laws and more in others, so the batch "
            "would rename the column (from `x` to `x1`), and the frame steps "
            "read it under the unnumbered name. Run these laws in separate "
            "batches."
        )


# --------------------------------------------------------------------------
# The structural switches
# --------------------------------------------------------------------------


def charity_double_run_offered(above_the_line_limit, itemized_limit) -> bool:
    """Whether a filer can choose where to report a charitable gift.

    When the law offers both an above-the-line charitable deduction and a
    Schedule A one, the two are mutually exclusive on the form, so the filer
    computes the return both ways and reports whichever costs less. That means
    running the whole 1040 twice. The choice is offered when both limits are
    positive for some filer.
    """
    above = np.asarray(above_the_line_limit)
    item = np.asarray(itemized_limit)
    return bool(np.any((above > 0) & (item > 0)))


def program_fingerprint(charity_double_run: bool, gains_model: bool) -> str:
    """The label under which scenarios share one compiled program.

    Two things decide which functions a scenario's program runs: whether the
    1040 is computed twice for the charitable choice, and whether the
    capital-gains model is composed into the year. Scenarios with the same
    label run the same code.
    """
    return (
        f"charity_double_run={'yes' if charity_double_run else 'no'};"
        f"gains_model={'yes' if gains_model else 'no'}"
    )


def charity_double_run(law: CompiledLaw) -> bool:
    """Whether this law makes the model compute the 1040 twice.

    The law offers the choice when a filer may deduct a charitable gift either
    above the line or on Schedule A. Laws that answer differently run
    different programs and cannot share a compiled batch. The question is
    asked of all four filing statuses, as `taxes/statutory_order.py` asks it.

    The answer must be the same in every year of the run. A law offering the
    choice in some years and not others is refused.
    """
    for name in (CHARITY_ABOVE_THE_LINE_LIMIT, CHARITY_ITEMIZED_LIMIT):
        if name not in law.values:
            raise LawError(
                f"law '{law.alternative}' has no field '{name}', so whether a "
                "filer may choose where to report a charitable gift cannot be "
                "answered"
            )
    # Both limits are single-valued parameters, so slot 0 holds the value
    # `taxes/law_for_calculators.py` gives the calculators. The middle axis
    # is the four filing statuses.
    above = law.values[CHARITY_ABOVE_THE_LINE_LIMIT][:, :, 0]
    itemized = law.values[CHARITY_ITEMIZED_LIMIT][:, :, 0]
    by_year = [
        charity_double_run_offered(above[y], itemized[y]) for y in range(len(law.years))
    ]

    if len(set(by_year)) > 1:
        offered_in = [
            year for year, offered in zip(law.years, by_year, strict=True) if offered
        ]
        raise LawError(
            f"law '{law.alternative}' offers both the above-the-line and the "
            f"itemized charitable deduction in {offered_in} but not in every "
            "year of the run. That changes which functions run from year to "
            "year, and the engine settles the question once per run, so this "
            "law needs the run split into windows that answer it the same way."
        )
    return by_year[0]


__all__ = [
    "FRAME_LAW_COLUMNS_READ_BY_NAME",
    "CompiledBatch",
    "charity_double_run",
    "charity_double_run_offered",
    "compile_batch",
    "program_fingerprint",
]
