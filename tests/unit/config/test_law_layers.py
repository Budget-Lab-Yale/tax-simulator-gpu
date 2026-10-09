"""Tax law layering: a reform replaces whole subparameters."""

from __future__ import annotations

import pytest

from taxmodel.config import ConfigError, resolve_law, resolve_law_stack


@pytest.fixture
def law(tree):
    """A two-parameter baseline: an indexed CTC threshold and an ordinary schedule."""
    tree.write(
        "config/scenarios/tax_law/default/ctc.yaml",
        {
            "po_thresh": {
                "value": {"2018": 200000},
                "i_measure": "chained_cpi",
                "i_base_year": 2018,
                "i_direction": "floor",
                "i_increment": 1000,
            },
            "value": {"value": {"2018": 2000}},
        },
    )
    tree.write(
        "config/scenarios/tax_law/default/ord.yaml",
        {"rates": {"value": {"2018": [0.1, 0.12, 0.22]}}},
    )
    return tree


def test_baseline_law_is_the_default_layer(law):
    resolved = resolve_law(law.root)
    assert resolved.parameters["ctc"]["value"]["value"] == {"2018": 2000}
    assert not resolved.replacements


def test_a_reform_replaces_the_subparameter_entire(law):
    """Value and indexation together: nothing merges inside a subparameter."""
    law.write(
        "config/scenarios/tax_law/alternatives/bigger_ctc/ctc.yaml",
        {"value": {"value": {"2026": 3000}}},
    )
    resolved = resolve_law(law.root, "bigger_ctc")
    assert resolved.parameters["ctc"]["value"]["value"] == {"2026": 3000}
    # Untouched subparameters and untouched parameters keep baseline law.
    assert resolved.parameters["ctc"]["po_thresh"]["i_measure"] == "chained_cpi"
    assert resolved.parameters["ord"]["rates"]["value"] == {"2018": [0.1, 0.12, 0.22]}


def test_a_reform_omitting_the_indexation_fields_un_indexes_the_subparameter(law):
    """The R model's behavior, and the reason the compiler's lint exists.

    A reform that restates the CTC phaseout threshold without saying how it
    grows has frozen it in nominal terms from that year on.
    """
    law.write(
        "config/scenarios/tax_law/alternatives/frozen/ctc.yaml",
        {"po_thresh": {"value": {"2026": 250000}}},
    )
    resolved = resolve_law(law.root, "frozen")
    subparam = resolved.parameters["ctc"]["po_thresh"]
    assert "i_measure" not in subparam
    assert "i_increment" not in subparam

    (replacement,) = resolved.replacements
    assert replacement.label == "ctc.po_thresh"
    assert replacement.dropped_indexation


def test_a_reform_restating_the_indexation_keeps_it(law):
    law.write(
        "config/scenarios/tax_law/alternatives/indexed/ctc.yaml",
        {
            "po_thresh": {
                "value": {"2026": 250000},
                "i_measure": "cpi",
                "i_base_year": 2026,
                "i_direction": "floor",
                "i_increment": 1000,
            }
        },
    )
    resolved = resolve_law(law.root, "indexed")
    assert resolved.parameters["ctc"]["po_thresh"]["i_measure"] == "cpi"
    assert not resolved.replacements[0].dropped_indexation


def test_the_baseline_layer_is_never_mutated_by_a_reform(law):
    law.write(
        "config/scenarios/tax_law/alternatives/bigger_ctc/ctc.yaml",
        {"value": {"value": {"2026": 3000}}},
    )
    resolve_law(law.root, "bigger_ctc")
    assert resolve_law(law.root).parameters["ctc"]["value"]["value"] == {"2018": 2000}


def test_a_subparameter_the_baseline_lacks_is_recorded_as_new(law):
    """Recorded here, refused by the compiler's lint unless declared.

    The layering stays permissive so the whole R tree still resolves; whether
    the new name is a lever or a typo is decided against the layers a runscript
    names, which is the lint's scope, not this module's.
    """
    law.write(
        "config/scenarios/tax_law/alternatives/new_lever/ctc.yaml",
        {"baby_bonus": {"value": {"2026": 1000}}},
    )
    resolved = resolve_law(law.root, "new_lever")
    assert resolved.replacements[0].new_subparameter


def test_the_baseline_names_are_kept_so_the_lint_can_suggest_one(law):
    resolved = resolve_law(law.root)
    assert resolved.baseline_subparameters["ctc"] == ("po_thresh", "value")


def test_an_unknown_reform_folder_stops(law):
    with pytest.raises(ConfigError, match="unknown tax law layer 'nope'"):
        resolve_law(law.root, "nope")


def test_default_under_alternatives_is_unreachable_and_refused(law):
    law.write(
        "config/scenarios/tax_law/alternatives/default/ctc.yaml",
        {"value": {"value": {"2026": 3000}}},
    )
    with pytest.raises(ConfigError, match="reserved runscript cell"):
        resolve_law(law.root, "whatever")


# ---------------------------------------------------------------------------
# The fold: several provision folders, in order
# ---------------------------------------------------------------------------


@pytest.fixture
def three_provisions(law):
    """One child credit change decomposed into three parts.

    Each folder holds one knob of the same parameter file — the credit's
    maximum value, then its phaseout threshold, then a genuinely new
    refundability lever. Nothing copies an earlier folder's block forward.
    """
    law.write(
        "config/scenarios/tax_law/alternatives/ctc/1_bigger/ctc.yaml",
        {"value": {"value": {"2026": 3000}}},
    )
    law.write(
        "config/scenarios/tax_law/alternatives/ctc/2_phaseout/ctc.yaml",
        {
            "po_thresh": {
                "value": {"2026": 150000},
                "i_measure": "chained_cpi",
                "i_base_year": 2026,
                "i_direction": "floor",
                "i_increment": 1000,
            }
        },
    )
    law.write(
        "config/scenarios/tax_law/alternatives/ctc/3_refundable/ctc.yaml",
        {"max_refund": {"value": {"2026": 2000}, "new_subparameter": True}},
    )
    return law


PROVISIONS = ("ctc/1_bigger", "ctc/2_phaseout", "ctc/3_refundable")


def test_the_fold_returns_baseline_and_every_cumulative_law(three_provisions):
    laws = resolve_law_stack(three_provisions.root, PROVISIONS)
    assert [law.alternative for law in laws] == ["default", *PROVISIONS]
    assert [law.layers for law in laws] == [
        ("default",),
        ("default", "ctc/1_bigger"),
        ("default", "ctc/1_bigger", "ctc/2_phaseout"),
        ("default", "ctc/1_bigger", "ctc/2_phaseout", "ctc/3_refundable"),
    ]


def test_an_empty_stack_is_baseline_law(three_provisions):
    (only,) = resolve_law_stack(three_provisions.root, [])
    assert only.alternative == "default"
    assert not only.replacements
    assert only.parameters["ctc"]["value"]["value"] == {"2018": 2000}


def test_each_rung_carries_the_provisions_below_it_and_no_more(three_provisions):
    _baseline, first, second, third = resolve_law_stack(
        three_provisions.root, PROVISIONS
    )
    assert first.parameters["ctc"]["value"]["value"] == {"2026": 3000}
    assert first.parameters["ctc"]["po_thresh"]["value"] == {"2018": 200000}

    # The second provision's phaseout threshold arrives without disturbing the
    # first provision's credit value: the merge is per subparameter, not per
    # file.
    assert second.parameters["ctc"]["po_thresh"]["value"] == {"2026": 150000}
    assert second.parameters["ctc"]["value"]["value"] == {"2026": 3000}

    assert "max_refund" not in second.parameters["ctc"]
    assert third.parameters["ctc"]["max_refund"]["value"] == {"2026": 2000}
    assert third.parameters["ctc"]["value"]["value"] == {"2026": 3000}


def test_every_replacement_names_the_provision_folder_that_wrote_it(three_provisions):
    third = resolve_law_stack(three_provisions.root, PROVISIONS)[-1]
    assert [(r.label, r.written_by) for r in third.replacements] == [
        ("ctc.value", "ctc/1_bigger"),
        ("ctc.po_thresh", "ctc/2_phaseout"),
        ("ctc.max_refund", "ctc/3_refundable"),
    ]
    assert third.written_by("ctc", "po_thresh") == "ctc/2_phaseout"
    # Nothing wrote the ordinary rate schedule, so it is baseline law's own.
    assert third.written_by("ord", "rates") == "default"


def test_the_baseline_layer_is_never_mutated_by_a_fold(three_provisions):
    resolve_law_stack(three_provisions.root, PROVISIONS)
    assert resolve_law(three_provisions.root).parameters["ctc"]["value"]["value"] == {
        "2018": 2000
    }


def test_a_later_provision_wins_and_both_replacements_are_recorded(law):
    """Two provisions may replace the same subparameter; the run goes ahead."""
    law.write(
        "config/scenarios/tax_law/alternatives/first/ctc.yaml",
        {"value": {"value": {"2026": 3000}}},
    )
    law.write(
        "config/scenarios/tax_law/alternatives/second/ctc.yaml",
        {"value": {"value": {"2026": 4000}}},
    )
    resolved = resolve_law_stack(law.root, ["first", "second"])[-1]

    assert resolved.parameters["ctc"]["value"]["value"] == {"2026": 4000}
    assert [(r.reform["value"], r.written_by) for r in resolved.replacements] == [
        ({"2026": 3000}, "first"),
        ({"2026": 4000}, "second"),
    ]
    assert resolved.conflicts() == {"ctc.value": ("first", "second")}
    assert resolved.written_by("ctc", "value") == "second"


def test_a_single_reform_has_no_conflicts(law):
    law.write(
        "config/scenarios/tax_law/alternatives/bigger_ctc/ctc.yaml",
        {"value": {"value": {"2026": 3000}}},
    )
    assert resolve_law(law.root, "bigger_ctc").conflicts() == {}


def test_a_new_lever_stays_new_however_late_in_the_stack_it_is_named(law):
    """The typo protection never depends on stack order.

    The first provision creates a refundability lever baseline law does not
    have. The second provision names the same lever, and it is still marked
    new — the comparison is always against baseline law — so a misspelling in
    the second folder cannot hide behind the first folder's declaration.
    """
    law.write(
        "config/scenarios/tax_law/alternatives/creates/ctc.yaml",
        {"max_refund": {"value": {"2026": 1600}, "new_subparameter": True}},
    )
    law.write(
        "config/scenarios/tax_law/alternatives/raises/ctc.yaml",
        {"max_refund": {"value": {"2026": 2000}}},
    )
    resolved = resolve_law_stack(law.root, ["creates", "raises"])[-1]

    creates, raises = resolved.replacements
    assert creates.new_subparameter and creates.written_by == "creates"
    assert raises.new_subparameter and raises.written_by == "raises"
    assert raises.baseline is None


def test_the_fold_refuses_an_unknown_provision_folder(law):
    with pytest.raises(ConfigError, match="unknown tax law layer 'nope'"):
        resolve_law_stack(law.root, ["nope"])
