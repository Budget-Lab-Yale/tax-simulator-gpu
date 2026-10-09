"""The negative battery: laws that parse and mean something nobody wrote."""

from __future__ import annotations

import pytest

from taxmodel.config.law_layers import resolve_law, resolve_law_stack
from taxmodel.law import MAX_BRACKETS, compile_law
from taxmodel.law.compile import FILING_STATUS_LABELS, compile_resolved
from taxmodel.law.errors import LawLintError
from taxmodel.law.lint import (
    DROPPED_INDEXATION,
    FREEZE_RESUME,
    SCHEDULE_LENGTH_MISMATCH,
    SCHEDULE_OVERFLOW,
    UNKNOWN_SUBPARAMETER,
    check_law,
    lint_law,
    schedule_length_findings,
    schedule_pairs,
)
from tests.conftest import CONFIG_ROOT

from .conftest import LAW_TREE_YEARS

YEARS = tuple(range(2025, 2031))

INDEXED_THRESHOLD = {
    "value": {"2018": 200000},
    "i_measure": {"1987": "cpi", "2017": "chained_cpi"},
    "i_base_year": 2017,
    "i_direction": -1,
    "i_increment": 1000,
}


def rules(law_tree, alternative=None):
    resolved = resolve_law(law_tree.root, alternative)
    return [f.rule for f in lint_law(resolved, YEARS, MAX_BRACKETS)]


@pytest.fixture
def baseline(law_tree):
    law_tree.write("default", "ctc", {"po_thresh": dict(INDEXED_THRESHOLD)})
    return law_tree


# -- a reform that quietly un-indexes ---------------------------------------


def test_a_reform_that_drops_indexation_without_saying_so_is_refused(baseline):
    """Restating the threshold without i_measure freezes it in nominal terms."""
    baseline.write("bigger", "ctc", {"po_thresh": {"value": {"2026": 250000}}})
    assert rules(baseline, "bigger") == [DROPPED_INDEXATION]


def test_declaring_the_un_indexing_makes_it_legal(baseline):
    baseline.write(
        "bigger",
        "ctc",
        {"po_thresh": {"value": {"2026": 250000}, "remove_indexation": True}},
    )
    assert rules(baseline, "bigger") == []


def test_a_reform_that_keeps_indexation_is_fine(baseline):
    reform = dict(INDEXED_THRESHOLD)
    reform["value"] = {"2026": 250000}
    baseline.write("bigger", "ctc", {"po_thresh": reform})
    assert rules(baseline, "bigger") == []


def test_a_brand_new_subparameter_is_not_a_dropped_indexation(baseline):
    baseline.write(
        "bigger",
        "ctc",
        {"baby_bonus": {"value": {"2026": 2400}, "new_subparameter": True}},
    )
    assert rules(baseline, "bigger") == []


# -- a lever the baseline does not have ----------------------------------------


def findings(law_tree, alternative=None):
    return lint_law(resolve_law(law_tree.root, alternative), YEARS, MAX_BRACKETS)


def test_a_misspelled_subparameter_is_refused_and_names_the_intended_one(baseline):
    """`po_thresh_` compiles as a lever nothing reads, so the threshold never moves."""
    baseline.write("typo", "ctc", {"po_thresh_": {"value": {"2026": 250000}}})
    (finding,) = findings(baseline, "typo")
    assert finding.rule == UNKNOWN_SUBPARAMETER
    assert finding.field == "ctc.po_thresh_"
    assert "did you mean `ctc.po_thresh`?" in finding.detail


def test_declaring_the_new_lever_makes_it_legal(baseline):
    baseline.write(
        "bigger",
        "ctc",
        {"baby_bonus": {"value": {"2026": 2400}, "new_subparameter": True}},
    )
    assert rules(baseline, "bigger") == []


def test_a_declared_new_lever_still_reaches_the_manifest(baseline):
    """Declaring it says the name is deliberate, not that it stops being new."""
    baseline.write(
        "bigger",
        "ctc",
        {"baby_bonus": {"value": {"2026": 2400}, "new_subparameter": True}},
    )
    (replacement,) = resolve_law(baseline.root, "bigger").replacements
    assert replacement.label == "ctc.baby_bonus"
    assert replacement.new_subparameter


def test_a_declared_new_lever_compiles_to_a_field(baseline, flat_index):
    baseline.write(
        "bigger",
        "ctc",
        {"baby_bonus": {"value": {"2026": 2400}, "new_subparameter": True}},
    )
    law = compile_law(baseline.root, YEARS, flat_index, alternative="bigger")
    assert law.values["ctc.baby_bonus"][YEARS.index(2026), 0, 0] == 2400


def test_a_misspelled_parameter_file_names_the_nearest_file(baseline):
    baseline.write("typo", "ctcc", {"po_thresh": {"value": {"2026": 250000}}})
    (finding,) = findings(baseline, "typo")
    assert finding.rule == UNKNOWN_SUBPARAMETER
    assert "did you mean `ctc`?" in finding.detail


def test_a_new_name_nothing_resembles_is_still_refused_without_a_suggestion(baseline):
    baseline.write("typo", "ctc", {"qqqq": {"value": {"2026": 1}}})
    (finding,) = findings(baseline, "typo")
    assert finding.rule == UNKNOWN_SUBPARAMETER
    assert "did you mean" not in finding.detail


def test_the_typo_guard_stops_the_compile(baseline, flat_index):
    baseline.write("typo", "ctc", {"po_thresh_": {"value": {"2026": 250000}}})
    with pytest.raises(LawLintError, match="new_subparameter"):
        compile_law(baseline.root, YEARS, flat_index, alternative="typo")


# -- which layers the errors bind ----------------------------------------------


def test_a_finding_carries_the_layer_whose_file_holds_it(baseline):
    baseline.write("bigger", "ctc", {"po_thresh": {"value": {"2026": 250000}}})
    (finding,) = findings(baseline, "bigger")
    assert finding.layer == "bigger"


def test_a_finding_in_a_layer_the_run_does_not_name_does_not_block(baseline):
    """A finding in a reform folder the run does not name is reported only.

    Same tree, same finding: it stops the compile when the runscript names the
    folder, and is reported without stopping anything when it does not.
    """
    baseline.write("bigger", "ctc", {"po_thresh": {"value": {"2026": 250000}}})
    resolved = resolve_law(baseline.root, "bigger")

    with pytest.raises(LawLintError):
        check_law(resolved, YEARS, MAX_BRACKETS, scope=["bigger"])

    reported = check_law(resolved, YEARS, MAX_BRACKETS, scope=["some/other/folder"])
    assert [f.rule for f in reported] == [DROPPED_INDEXATION]


def test_baseline_law_is_always_in_scope(baseline):
    """Every runscript reads the default layer, whatever else its cells name."""
    baseline.write("default", "estate", {"rates": {"value": [0.1] * (MAX_BRACKETS + 2)}})
    resolved = resolve_law(baseline.root, None)
    with pytest.raises(LawLintError, match=SCHEDULE_OVERFLOW):
        check_law(resolved, YEARS, MAX_BRACKETS, scope=["some/other/folder"])


def test_the_lint_stops_the_compile(baseline, flat_index):
    baseline.write("bigger", "ctc", {"po_thresh": {"value": {"2026": 250000}}})
    with pytest.raises(LawLintError, match="remove_indexation"):
        compile_law(baseline.root, YEARS, flat_index, alternative="bigger")


# -- a schedule wider than the compiled arrays ------------------------------


def test_a_schedule_past_max_brackets_is_refused(law_tree):
    law_tree.write(
        "default", "estate", {"rates": {"value": [0.1] * (MAX_BRACKETS + 2)}}
    )
    assert rules(law_tree) == [SCHEDULE_OVERFLOW]


def test_a_schedule_that_widens_in_a_later_year_is_measured_at_its_widest(law_tree):
    law_tree.write(
        "default",
        "estate",
        {"rates": {"value": {"2025": [0.1], "2027": [0.1] * (MAX_BRACKETS + 1)}}},
    )
    assert rules(law_tree) == [SCHEDULE_OVERFLOW]


def test_a_fourteen_bracket_schedule_fits(law_tree):
    """Fourteen estate brackets, the widest schedule in the law tree."""
    law_tree.write("default", "estate", {"rates": {"value": [0.1] * 14}})
    assert rules(law_tree) == []


# -- indexation that resumes after a freeze ---------------------------------


def test_resuming_indexation_after_a_freeze_is_refused(law_tree):
    """The index chain cannot restart; this compiles as a permanent freeze."""
    law_tree.write(
        "default",
        "ctc",
        {
            "max_refund": {
                "value": 1400,
                "i_measure": {"2017": "chained_cpi", "2023": "NA", "2026": "chained_cpi"},
                "i_base_year": 2017,
                "i_direction": -1,
                "i_increment": 100,
            }
        },
    )
    assert rules(law_tree) == [FREEZE_RESUME]


def test_a_freeze_that_stays_frozen_is_fine(law_tree):
    law_tree.write(
        "default",
        "ctc",
        {
            "max_refund": {
                "value": 1400,
                "i_measure": {"2017": "chained_cpi", "2023": "NA"},
                "i_base_year": 2017,
                "i_direction": -1,
                "i_increment": 100,
            }
        },
    )
    assert rules(law_tree) == []


def test_a_freeze_stated_in_the_files_defaults_is_caught_too(law_tree):
    """The Wyden-Smith CTC config writes it this way, in indexation_defaults."""
    law_tree.write(
        "default",
        "ctc",
        {
            "indexation_defaults": {
                "i_measure": {"2017": "chained_cpi", "2023": "NA", "2026": "chained_cpi"},
                "i_base_year": {"2014": 2017, "2023": "NA", "2026": 2017},
                "i_direction": {"2014": -1, "2023": "NA", "2026": -1},
                "i_increment": {"2014": 100, "2023": "NA", "2026": 100},
            },
            "max_refund": {
                "value": 1400,
                "i_measure": "default",
                "i_base_year": "default",
                "i_direction": "default",
                "i_increment": "default",
            },
        },
    )
    assert rules(law_tree) == [FREEZE_RESUME]


# -- a schedule whose rates and thresholds are different lengths ------------
#
# The deliberate mutation these tests use: a first provision restates the
# ordinary-rate thresholds at four brackets and a second provision restates the
# rates at three. Both folders are internally sensible; folded together they
# describe a schedule whose fourth bracket does not exist, and the income above
# the fourth threshold is taxed at the third rate.


ONE_SCHEDULE = {
    "rates": {"value": {"2025": [0.1, 0.2, 0.3]}},
    "brackets": {"value": {"2025": [0, 40_000, 90_000]}},
}


@pytest.fixture
def with_a_schedule(law_tree):
    """A baseline holding one three-bracket ordinary rate schedule."""
    law_tree.write("default", "ord", ONE_SCHEDULE)
    return law_tree


def test_a_consistent_schedule_passes(with_a_schedule, flat_index):
    compile_law(with_a_schedule.root, YEARS, flat_index)


def test_a_fourth_threshold_with_no_fourth_rate_stops_the_compile(
    with_a_schedule, flat_index
):
    with_a_schedule.write(
        "wider_brackets",
        "ord",
        {"brackets": {"value": {"2025": [0, 40_000, 90_000, 200_000]}}},
    )
    with pytest.raises(LawLintError, match=SCHEDULE_LENGTH_MISMATCH) as caught:
        compile_law(with_a_schedule.root, YEARS, flat_index, alternative="wider_brackets")

    message = str(caught.value)
    assert "ord.brackets and ord.rates" in message
    assert "4 bracket threshold(s) and 3 rate(s)" in message
    assert "bracket(s) 4 disappear" in message


def test_a_fourth_rate_with_no_fourth_threshold_stops_the_compile(
    with_a_schedule, flat_index
):
    with_a_schedule.write(
        "more_rates", "ord", {"rates": {"value": {"2025": [0.1, 0.2, 0.3, 0.4]}}}
    )
    with pytest.raises(LawLintError, match=SCHEDULE_LENGTH_MISMATCH):
        compile_law(with_a_schedule.root, YEARS, flat_index, alternative="more_rates")


def test_the_mismatch_names_the_year_and_the_filing_status(with_a_schedule, flat_index):
    with_a_schedule.write(
        "wider_brackets",
        "ord",
        {
            "brackets": {
                "value": {"2025": [0, 40_000, 90_000], "2027": [0, 40_000, 90_000, 200_000]}
            }
        },
    )
    resolved = resolve_law(with_a_schedule.root, "wider_brackets")
    law = compile_resolved(resolved, YEARS, flat_index, lint=False)
    reported = schedule_length_findings(
        law.active, law.years, resolved, FILING_STATUS_LABELS
    )

    # The wider thresholds start in 2027 and carry forward, so 2025 and 2026
    # are clean and the four remaining years are not -- all four filing
    # statuses each, since nothing here varies by filing status.
    assert len(reported) == len([y for y in YEARS if y >= 2027]) * 4
    assert "in 2027, filing status 1 (single)" in reported[0].detail
    assert "in 2027, filing status 2 (married filing jointly)" in reported[1].detail


def test_restating_half_a_schedule_only_from_a_later_year_empties_the_earlier_ones(
    with_a_schedule, flat_index
):
    """A real case the lint catches, and the reason reform folders restate history.

    A reform replaces a whole subparameter, so a reform that states the rates
    for 2026 onward and nothing earlier has left the run's earlier years with
    thresholds and no rates at all — an empty schedule collecting no tax. The
    Clausing folders avoid this by restating each schedule's whole history
    (their value blocks carry a 2014 key as well as a 2030 one).
    """
    with_a_schedule.write(
        "rates_from_2026", "ord", {"rates": {"value": {"2026": [0.08, 0.18, 0.28]}}}
    )
    with pytest.raises(LawLintError) as caught:
        compile_law(with_a_schedule.root, YEARS, flat_index, alternative="rates_from_2026")
    assert "in 2025, filing status 1 (single)" in str(caught.value)
    assert "3 bracket threshold(s) and 0 rate(s)" in str(caught.value)


def test_a_stack_that_restates_only_one_half_is_caught(with_a_schedule, flat_index):
    """The case that makes the lint mandatory before composition is safe.

    Neither folder is wrong on its own. The first widens the schedule to four
    brackets, stating both halves. The second lowers the rates and states three
    of them, because when it was written the schedule had three. Folded, the
    fourth bracket vanishes.
    """
    with_a_schedule.write(
        "provisions/1_widen",
        "ord",
        {
            "rates": {"value": {"2025": [0.1, 0.2, 0.3, 0.4]}},
            "brackets": {"value": {"2025": [0, 40_000, 90_000, 200_000]}},
        },
    )
    with_a_schedule.write(
        "provisions/2_lower", "ord", {"rates": {"value": {"2025": [0.08, 0.18, 0.28]}}}
    )

    # Each provision on its own compiles.
    compile_law(with_a_schedule.root, YEARS, flat_index, alternative="provisions/1_widen")
    compile_law(with_a_schedule.root, YEARS, flat_index, alternative="provisions/2_lower")

    stacked = resolve_law_stack(
        with_a_schedule.root, ["provisions/1_widen", "provisions/2_lower"]
    )[-1]
    with pytest.raises(LawLintError) as caught:
        compile_resolved(stacked, YEARS, flat_index)

    # The message names both provision folders, so a reader knows which one to
    # edit rather than being told "the stack" is wrong.
    message = str(caught.value)
    assert "`provisions/1_widen`" in message
    assert "`provisions/2_lower`" in message


def test_the_mismatch_names_baseline_law_when_baseline_wrote_the_half(
    with_a_schedule, flat_index
):
    with_a_schedule.write(
        "wider_brackets", "ord", {"brackets": {"value": {"2026": [0, 40_000, 90_000, 200_000]}}}
    )
    with pytest.raises(LawLintError) as caught:
        compile_law(with_a_schedule.root, YEARS, flat_index, alternative="wider_brackets")
    # The reform wrote the thresholds; the rates are still baseline law's.
    assert "ord.rates comes from baseline law" in str(caught.value)


def test_the_filing_status_mapper_is_taken_into_account(law_tree, flat_index):
    """The check reads the compiled arrays, so the mapper has already run.

    Joint filers get twice the single filers' thresholds and the same rates, so
    a mapper that states three thresholds against three rates is consistent for
    every filing status even though the numbers differ.
    """
    law_tree.write(
        "default",
        "ord",
        {
            "filing_status_mapper": {
                "brackets": {
                    "1": "brackets_single",
                    "2": "brackets_single * 2",
                    "3": "brackets_single",
                    "4": "brackets_single",
                }
            },
            "rates": {"value": {"2025": [0.1, 0.2, 0.3]}},
            "brackets_single": {"value": {"2025": [0, 40_000, 90_000]}},
        },
    )
    compile_law(law_tree.root, YEARS, flat_index)


def test_a_lone_vector_is_not_mistaken_for_a_schedule(law_tree, flat_index):
    """The mortgage-balance limit by loan vintage has no rate schedule beside it."""
    law_tree.write(
        "default",
        "item",
        {
            "mort_bal_limit": {"value": {"2025": [1_000_000, 750_000]}},
            "mort_bal_limit_years": {"value": {"2025": [2017, 2026]}},
        },
    )
    assert schedule_pairs(["item.mort_bal_limit", "item.mort_bal_limit_years"]) == []
    compile_law(law_tree.root, YEARS, flat_index)


def test_the_pairs_the_check_finds_are_the_ones_the_calculators_integrate():
    """The naming rule has to pick out exactly the schedules `bind.py` declares.

    `bind.py` says which compiled fields the calculators read whole, and the
    rate-and-threshold pairs among them are the schedules that must stay
    consistent. If a new schedule is added under a name the rule misses, this
    test fails rather than the lint quietly skipping it.
    """
    from taxmodel.taxes import law_for_calculators

    declared: set[str] = set()
    for table in vars(law_for_calculators).values():
        if not isinstance(table, dict):
            continue
        for ref in table.values():
            if isinstance(ref, law_for_calculators.Ref) and ref.slot is None:
                declared.add(ref.name)

    found = set(schedule_pairs(declared))
    expected = {
        (name[: -len("rates")] + "brackets", name)
        for name in declared
        if name.endswith("rates") and name[: -len("rates")] + "brackets" in declared
    }
    assert found == expected
    # The fourteen schedules the calculators integrate: ordinary rates,
    # preferred rates, the alternative minimum tax, the estate tax, the wealth
    # tax, the net investment income tax, Social Security benefit inclusion,
    # five payroll schedules (employee and employer old-age insurance, employee
    # and employer hospital insurance, and the additional Medicare tax), and the
    # two education credits.
    assert len(found) == 14


# -- this repository's law tree ---------------------------------------------


def test_the_baseline_law_is_clean():
    resolved = resolve_law(CONFIG_ROOT)
    assert lint_law(resolved, LAW_TREE_YEARS, MAX_BRACKETS) == []


def test_the_wyden_smith_ctc_config_resumes_indexation_after_a_freeze():
    """A finding in a real reform folder, not a synthetic one.

    public/ctc/wyden_smith turns indexation off for 2023-2025 and names chained
    CPI-U again from 2026. The compiler reads that as a permanent freeze, and
    the lint reports it.
    """
    resolved = resolve_law(CONFIG_ROOT, "public/ctc/wyden_smith/indexation")
    findings = lint_law(resolved, LAW_TREE_YEARS, MAX_BRACKETS)
    assert {f.rule for f in findings} == {FREEZE_RESUME}
    assert {f.field for f in findings} == {"ctc.max_refund_old", "ctc.max_refund_young"}


def test_booker_kypa_drops_the_refundability_cap_indexation():
    """Also real: the reform restates max_refund without repeating i_measure."""
    resolved = resolve_law(CONFIG_ROOT, "public/booker-kypa")
    findings = lint_law(resolved, LAW_TREE_YEARS, MAX_BRACKETS)
    assert {f.rule for f in findings} == {DROPPED_INDEXATION}
    assert {f.field for f in findings} == {"ctc.max_refund_old", "ctc.max_refund_young"}
