"""Compiling several laws to one layout, and the reasons that is safe.

Every widening a batch performs has a stated reason, written as a comment at
the place in `law/batch.py` that widens — a claim about what the law says, not
about arrays — and there is a test here for each of the four. The claims all
reduce to the same thing: a law presented in a wider layout is the same law, so
every household owes the same tax.

Three laws from the repository's own tree carry the real cases:

  `public/wealth_tax_warren` states the wealth-tax brackets per filing status
  and fills three of them, where baseline law fills one and states it for
  everybody — so it varies both the filing-status decision and the column count.

  `public/estate_2009` restores the 2009 estate schedule, which has fourteen
  brackets against baseline's twelve — the plain column-widening case.

  `tests/char_above_smoke` turns on the above-the-line charitable deduction
  alongside the Schedule A one, which is the one structural switch the model
  has: the whole 1040 runs twice under that law.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from taxmodel.config.law_layers import resolve_law
from taxmodel.frame import steps_jax
from taxmodel.law.batch import charity_double_run, compile_batch
from taxmodel.law.compile import compile_resolved
from taxmodel.law.errors import LawError
from taxmodel.law.indexes import index_table_from_frame
from tests.conftest import INDEX_TABLE
from tests.unit.years.test_law_binding_identity import annual_frame_law_columns

CONFIG_ROOT = Path("config/scenarios")
# Two years is enough for every shape question and keeps the compile quick.
YEARS = (2026, 2027)

WEALTH_TAX = "public/wealth_tax_warren"
ESTATE_2009 = "public/estate_2009"
CHARITY_BOTH_WAYS = "tests/char_above_smoke"
FRAME_JOIN_TEST_YEAR = 2026
FRAME_JOIN_TEST_COLUMN = "test.value"
# The invented law gives every filing status the same $12,000 value.
FRAME_JOIN_TEST_LAW_VALUE = 12_000.0


@pytest.fixture(scope="module")
def index_table():
    """The golden index table the calculator tests use."""
    return index_table_from_frame(pd.read_csv(INDEX_TABLE))


def compile_alone(alternative, index_table):
    return compile_resolved(resolve_law(CONFIG_ROOT, alternative), YEARS, index_table)


def compile_together(alternatives, index_table):
    return compile_batch(
        [resolve_law(CONFIG_ROOT, name) for name in alternatives], YEARS, index_table
    )


def write_the_charitable_limits(law_tree) -> None:
    """Give an invented tree the two limits the charitable question reads.

    Every law in the real tree states both, and a batch asks each law whether a
    filer may choose where to report a gift, so an invented tree needs them too.
    These are baseline law's answers: no above-the-line deduction, and no dollar
    cap on the Schedule A one.
    """
    law_tree.write(
        "default",
        "char",
        {"above_limit": {"value": 0}, "item_limit": {"value": "Inf"}},
    )


# ---------------------------------------------------------------------------
# The batch changes shapes and nothing else
# ---------------------------------------------------------------------------


def test_a_batch_of_one_law_is_that_law_compiled_alone(index_table):
    alone = compile_alone(None, index_table)
    batched = compile_together([None], index_table).laws[0]

    assert batched.years == alone.years
    assert batched.column_names() == alone.column_names()
    for name, block in alone.values.items():
        np.testing.assert_array_equal(batched.values[name], block, err_msg=name)


def test_a_batch_moves_no_law_value(index_table):
    """Rules 1 to 4, in one statement: only the layout changes."""
    names = [None, ESTATE_2009, WEALTH_TAX]
    batch = compile_together(names, index_table)
    for name, batched in zip(names, batch.laws, strict=True):
        alone = compile_alone(name, index_table)
        for field, block in alone.values.items():
            np.testing.assert_array_equal(
                batched.values[field], block, err_msg=f"{name or 'baseline'}.{field}"
            )


def test_every_law_in_a_batch_presents_the_same_layout(index_table):
    batch = compile_together([None, ESTATE_2009, WEALTH_TAX], index_table)
    first = batch.laws[0]
    for law in batch.laws[1:]:
        assert law.field_names == first.field_names
        assert law.column_names() == first.column_names()


# ---------------------------------------------------------------------------
# Filing status is settled by construction, not by the batch
# ---------------------------------------------------------------------------


def test_two_laws_bind_to_the_same_shapes_whatever_they_say_by_status(index_table):
    """Warren's wealth tax varies by filing status where baseline law does not.

    Warren gives married couples their own wealth-tax bracket thresholds;
    baseline law has no wealth tax at all. Both bind to four filing statuses by
    bracket slot, so the two present identical shapes and can share one
    compiled program without anything being padded out.
    """
    from taxmodel.taxes import law_for_calculators

    baseline, wealth_tax = compile_together([None, WEALTH_TAX], index_table).laws
    bound_baseline = law_for_calculators.bind_year(baseline.values, baseline.years, YEARS[0])
    bound_wealth_tax = law_for_calculators.bind_year(wealth_tax.values, wealth_tax.years, YEARS[0])

    baseline_shapes = [np.shape(leaf) for leaf in jax.tree.leaves(bound_baseline)]
    wealth_tax_shapes = [np.shape(leaf) for leaf in jax.tree.leaves(bound_wealth_tax)]
    assert baseline_shapes == wealth_tax_shapes
    # And the shapes really are the size of the law rather than of a
    # population: four filing statuses, sixteen bracket slots.
    assert np.shape(bound_wealth_tax.wealth.brackets) == (4, 16)


def test_a_law_column_gives_each_filer_the_value_for_its_own_status():
    """Four identical law rows state the same tax law as one value would.

    The frame join gathers the row for each record's filing status. When every
    row says $12,000, every filer is given $12,000, which is what a law column
    holding a single number gave them before every column carried four rows.
    """
    filing_status = np.arange(1, steps_jax.N_FILING_STATUSES + 1)
    law_table = pd.DataFrame(
        {
            "year": FRAME_JOIN_TEST_YEAR,
            "filing_status": filing_status,
            FRAME_JOIN_TEST_COLUMN: np.full(
                filing_status.size, FRAME_JOIN_TEST_LAW_VALUE
            ),
        }
    )
    columns = annual_frame_law_columns(law_table, FRAME_JOIN_TEST_YEAR)
    frame = {"filing_status": jnp.asarray(filing_status)}
    joined = steps_jax.join_law(frame, columns)

    assert columns[FRAME_JOIN_TEST_COLUMN].shape == (steps_jax.N_FILING_STATUSES,)
    np.testing.assert_array_equal(
        np.asarray(joined[FRAME_JOIN_TEST_COLUMN]),
        np.full(filing_status.size, FRAME_JOIN_TEST_LAW_VALUE),
    )


# ---------------------------------------------------------------------------
# Rules 1 and 3: bracket slots and the wide table's columns
# ---------------------------------------------------------------------------


def test_the_column_set_follows_the_widest_schedule_in_the_batch(index_table):
    # The 2009 estate schedule has fourteen brackets; baseline law's has twelve.
    alone = compile_alone(None, index_table)
    assert "estate.brackets13" not in alone.column_names()

    batch = compile_together([None, ESTATE_2009], index_table)
    for law in batch.laws:
        assert law.columns_in_table("estate.brackets") == 14
        assert "estate.brackets14" in law.column_names()


def test_the_columns_a_law_does_not_fill_say_there_is_no_such_bracket(index_table):
    """Rule 3: the extra columns are NaN, the compiler's way of saying nothing."""
    batch = compile_together([None, ESTATE_2009], index_table)
    header, rows = batch.laws[0].to_frame()
    thirteenth = header.index("estate.brackets13")
    assert all(np.isnan(row[thirteenth]) for row in rows)
    # And the law that does fill them still reports its own brackets.
    header, rows = batch.laws[1].to_frame()
    thirteenth = header.index("estate.brackets13")
    assert all(np.isfinite(row[thirteenth]) for row in rows)


def test_an_unused_bracket_slot_taxes_nobody(index_table):
    """Rule 1: NaN slots are not brackets at zero dollars.

    Compiling the ordinary rate schedule sixteen slots wide is the layout every
    law already shares, and this is why it is safe: the schedule reader stops at
    the last slot the law fills.
    """
    import jax.numpy as jnp

    from taxmodel.taxes.calc._schedules import schedule_tax

    law = compile_alone(None, index_table)
    brackets = law.values["ord.brackets"][0, 0]
    rates = law.values["ord.rates"][0, 0]
    filled = int(np.sum(~np.isnan(rates)))
    income = jnp.asarray([50_000.0, 400_000.0, 5_000_000.0])

    padded = schedule_tax(income, jnp.asarray(brackets), jnp.asarray(rates))
    trimmed = schedule_tax(
        income, jnp.asarray(brackets[:filled]), jnp.asarray(rates[:filled])
    )
    np.testing.assert_array_equal(np.asarray(padded), np.asarray(trimmed))


def test_renumbering_a_law_column_the_frame_steps_read_is_refused(
    law_tree, flat_index
):
    """Rule 4: widening a single-valued parameter renames its column.

    The SALT cap is one number, so its column is `item.salt_limit`, and the
    frame's state-tax workaround step reads it under that name. A batch with a
    law that made the cap a two-bracket schedule would rename the column to
    `item.salt_limit1`, so it is refused rather than run.
    """
    write_the_charitable_limits(law_tree)
    law_tree.write("default", "item", {"salt_limit": {"value": 10_000}})
    law_tree.write(
        "salt_two_tier", "item", {"salt_limit": {"value": [10_000, 20_000]}}
    )
    laws = [
        resolve_law(law_tree.root, None),
        resolve_law(law_tree.root, "salt_two_tier"),
    ]
    with pytest.raises(LawError, match="item.salt_limit"):
        compile_batch(laws, YEARS, flat_index, lint=False)


# ---------------------------------------------------------------------------
# Rule 6: a lever only some laws have
# ---------------------------------------------------------------------------


def test_a_lever_only_one_law_creates_is_nan_for_the_others(law_tree, flat_index):
    """A gains-at-death exclusion is a lever no ported calculator reads yet."""
    write_the_charitable_limits(law_tree)
    law_tree.write("default", "pref", {"unrecapture_rate": {"value": 0.25}})
    law_tree.write(
        "deemed_realization",
        "pref",
        {
            "unrecapture_rate": {"value": 0.25},
            "kg_death_gain_excl": {"value": 5_000_000, "new_subparameter": True},
        },
    )
    batch = compile_batch(
        [
            resolve_law(law_tree.root, None),
            resolve_law(law_tree.root, "deemed_realization"),
        ],
        YEARS,
        flat_index,
        lint=False,
    )
    baseline, reform = batch.laws
    assert "pref.kg_death_gain_excl" in baseline.field_names
    assert np.all(np.isnan(baseline.values["pref.kg_death_gain_excl"]))
    # Slot 0 is the lever; the rest of the sixteen slots are the padding every
    # field carries.
    assert np.all(reform.values["pref.kg_death_gain_excl"][:, :, 0] == 5_000_000)
    # Both laws' wide tables therefore have the column, which is what lets one
    # program serve both.
    assert baseline.column_names() == reform.column_names()


def test_a_law_missing_a_parameter_the_calculators_read_is_refused(
    law_tree, flat_index
):
    """The other half of rule 6: NaN is not a neutral value to a calculator.

    A law without the child credit's value would hand the credit NaN and get a
    NaN tax back, so it cannot be padded into a batch with laws that have it.
    Restating a parameter file's filing-status mapper is how a law in the real
    tree ends up without a field the others have, which is why this is checked
    rather than assumed away.
    """
    write_the_charitable_limits(law_tree)
    law_tree.write("default", "ctc", {"value_young": {"value": 2_000}})
    law_tree.write("bigger_ctc", "ctc", {"value_young": {"value": 3_000}})

    baseline = resolve_law(law_tree.root, None)
    stripped = resolve_law(law_tree.root, "bigger_ctc")
    del stripped.parameters["ctc"]["value_young"]

    with pytest.raises(LawError, match="ctc.value_young"):
        compile_batch([baseline, stripped], YEARS, flat_index, lint=False)


# ---------------------------------------------------------------------------
# The structural switch
# ---------------------------------------------------------------------------


def test_the_batch_answers_the_charitable_question_the_engine_asks(index_table):
    """The batch's answer has to agree with what the engine decides before tracing."""
    import jax.numpy as jnp  # noqa: F401 - imported for the engine's sake

    from taxmodel.taxes import law_for_calculators, statutory_order

    for alternative, expected in ((None, False), (CHARITY_BOTH_WAYS, True)):
        law = compile_alone(alternative, index_table)
        bound = law_for_calculators.bind_year(law.values, law.years, YEARS[0])
        assert statutory_order.charity_double_run_applies(bound) is expected
        assert charity_double_run(law) is expected


def test_laws_that_take_different_code_paths_are_told_apart(index_table):
    batch = compile_together([None, ESTATE_2009, CHARITY_BOTH_WAYS], index_table)
    assert batch.charity_double_run == (False, False, True)


def test_a_law_that_offers_both_charitable_deductions_in_some_years_only_is_refused(
    law_tree, flat_index
):
    """The engine settles this question once per run, so a law that changes its
    answer mid-run has to be split into windows rather than batched."""
    law_tree.write(
        "default",
        "char",
        {
            # The above-the-line deduction is capped at $300 in 2027 only, and
            # the Schedule A deduction is always available, so a filer has the
            # choice in 2027 and not in 2026.
            "above_limit": {"value": {2026: 0, 2027: 300}},
            "item_limit": {"value": 1e99},
        },
    )
    law = compile_resolved(
        resolve_law(law_tree.root, None), YEARS, flat_index, lint=False
    )
    with pytest.raises(LawError, match="2027"):
        charity_double_run(law)
