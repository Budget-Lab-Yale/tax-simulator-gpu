"""The stack file, and the ordinary scenarios it expands into.

The Clausing-Sarin package is the real example. These tests check that its
eight provisions expand into eight scenarios in order, and that the audit
trail names the folder every subparameter in force came from.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from taxmodel.config import ConfigError
from taxmodel.config.law_stacks import expand_stack, read_law_stack
from tests.unit.config.law_stack_helpers import combined_law, stack_file_path

# The repository's own configuration tree, which holds the provision-only
# folders the Clausing stack file names.
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_ROOT = REPO_ROOT / "config" / "scenarios"

# The seven tax-law provisions of the Clausing-Sarin package, in scoring order.
# The eighth provision changes the corporate saving assumption instead and is
# not a law folder, so it is not in this list.
CLAUSING_LAW_PROVISIONS = (
    "clausing_provisions/01_clinton_rates",
    "clausing_provisions/02_restore_bottom_rates",
    "clausing_provisions/03_199a",
    "clausing_provisions/04_carryover_basis",
    "clausing_provisions/05_pref_rates",
    "clausing_provisions/06_niit_reform",
    "clausing_provisions/07_estate",
)


# ---------------------------------------------------------------------------
# Reading the file
# ---------------------------------------------------------------------------


def write_stack(tmp_path: Path, name: str, body: dict | str) -> Path:
    """Write a stack file where a scenario tree would keep it."""
    path = stack_file_path(tmp_path / "config" / "scenarios", name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(body, str):
        path.write_text(body)
    else:
        path.write_text(yaml.safe_dump(body, sort_keys=False))
    return path


def test_a_stack_file_states_its_provisions_in_order(tmp_path):
    path = write_stack(
        tmp_path,
        "package",
        {"version": 1, "provisions": ["one", "two", "three"]},
    )
    stack = read_law_stack(path)
    assert stack.name == "package"
    assert stack.provisions == ("one", "two", "three")


def test_the_repository_clausing_stack_file_reads(tmp_path):
    stack = read_law_stack(stack_file_path(CONFIG_ROOT, "clausing_v2"))
    # Seven tax-law provisions, then the corporate saving provision, which
    # changes no law at all.
    assert stack.provisions == CLAUSING_LAW_PROVISIONS + ("08_corporate",)
    assert stack.tax_law_folders == CLAUSING_LAW_PROVISIONS
    assert stack.rungs[-1].tax_law is None
    assert stack.rungs[-1].economy == "clausing_corporate_saving_50"


def test_a_missing_stack_file_says_where_it_looked(tmp_path):
    with pytest.raises(ConfigError, match="no stack file at"):
        read_law_stack(tmp_path / "nowhere.yaml")


def test_a_stack_file_from_a_later_schema_is_refused(tmp_path):
    path = write_stack(
        tmp_path, "package", {"version": 2, "provisions": ["one"]}
    )
    with pytest.raises(ConfigError, match="this model reads stack file version 1"):
        read_law_stack(path)


def test_an_unknown_key_is_refused_by_name(tmp_path):
    path = write_stack(
        tmp_path,
        "package",
        {"version": 1, "provisions": ["one"], "modes": "sideways"},
    )
    with pytest.raises(ConfigError, match="unknown key `modes`"):
        read_law_stack(path)


def test_every_problem_in_the_file_is_reported_at_once(tmp_path):
    path = write_stack(tmp_path, "package", {"note": "nothing else"})
    with pytest.raises(ConfigError) as caught:
        read_law_stack(path)
    message = str(caught.value)
    assert "missing required key `version`" in message
    assert "missing required key `provisions`" in message


def test_an_empty_provision_list_is_refused(tmp_path):
    path = write_stack(tmp_path, "package", {"version": 1, "provisions": []})
    with pytest.raises(ConfigError, match="non-empty list of law folder names"):
        read_law_stack(path)


def test_a_repeated_provision_is_refused_because_two_scenarios_would_share_a_name(
    tmp_path,
):
    path = write_stack(
        tmp_path,
        "package",
        {"version": 1, "provisions": ["one", "two", "one"]},
    )
    with pytest.raises(ConfigError, match="appear twice"):
        read_law_stack(path)


def test_naming_baseline_as_a_provision_is_refused(tmp_path):
    path = write_stack(
        tmp_path, "package", {"version": 1, "provisions": ["default"]}
    )
    with pytest.raises(ConfigError, match="reserved name for baseline"):
        read_law_stack(path)


# ---------------------------------------------------------------------------
# Expanding into ordinary scenarios
# ---------------------------------------------------------------------------


@pytest.fixture
def small_tree(tmp_path):
    """A two-parameter baseline and three one-knob provision folders."""
    root = tmp_path / "config" / "scenarios"

    def write(layer: str, parameter: str, body: dict) -> None:
        folder = (
            root / "tax_law" / "default"
            if layer == "default"
            else root / "tax_law" / "alternatives" / layer
        )
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{parameter}.yaml").write_text(yaml.safe_dump(body, sort_keys=False))

    write("default", "ctc", {"value": {"value": {"2018": 2000}}})
    write("default", "ord", {"rates": {"value": {"2018": [0.1, 0.12, 0.22]}}})
    write("p1", "ctc", {"value": {"value": {"2026": 3000}}})
    write("p2", "ord", {"rates": {"value": {"2026": [0.1, 0.12, 0.25]}}})
    write("p3", "ctc", {"max_refund": {"value": {"2026": 2000}, "new_subparameter": True}})
    return root


def test_a_stacked_package_expands_into_the_cumulative_laws(small_tree, tmp_path):
    path = write_stack(
        tmp_path, "package", {"version": 1, "provisions": ["p1", "p2", "p3"]}
    )
    expanded = expand_stack(path, small_tree)

    assert [s.id for s in expanded.scenarios] == ["p1", "p2", "p3"]
    assert [s.provision for s in expanded.scenarios] == ["p1", "p2", "p3"]
    # The third rung carries all three provisions' blocks at once.
    third = expanded.scenarios[-1].law
    assert third.parameters["ctc"]["value"]["value"] == {"2026": 3000}
    assert third.parameters["ord"]["rates"]["value"] == {"2026": [0.1, 0.12, 0.25]}
    assert third.parameters["ctc"]["max_refund"]["value"] == {"2026": 2000}


def test_a_stacked_package_reports_each_provision_against_the_rung_below_it(
    small_tree, tmp_path
):
    path = write_stack(
        tmp_path, "package", {"version": 1, "provisions": ["p1", "p2", "p3"]}
    )
    expanded = expand_stack(path, small_tree)
    assert [
        (i.provision, i.minuend_id, i.subtrahend_id) for i in expanded.increments
    ] == [
        # The first provision is measured against the run's own baseline.
        ("p1", "p1", None),
        ("p2", "p2", "p1"),
        ("p3", "p3", "p2"),
    ]


def test_the_combined_law_carries_every_provision(small_tree, tmp_path):
    path = write_stack(
        tmp_path,
        "package",
        {"version": 1, "provisions": ["p1", "p2", "p3"]},
    )
    combined = combined_law(path, small_tree)
    assert combined.parameters["ctc"]["value"]["value"] == {"2026": 3000}
    assert combined.parameters["ord"]["rates"]["value"] == {"2026": [0.1, 0.12, 0.25]}
    assert combined.parameters["ctc"]["max_refund"]["value"] == {"2026": 2000}


def test_a_runscript_row_names_its_scenarios_row_id_dot_provision_folder(
    small_tree, tmp_path
):
    """The frozen naming scheme, so the run directory needs no invented labels."""
    write_stack(
        tmp_path,
        "package",
        {
            "version": 1,
            "provisions": ["ctc/1_bigger", "ctc/2_phaseout"],
        },
    )
    for layer in ("ctc/1_bigger", "ctc/2_phaseout"):
        folder = small_tree / "tax_law" / "alternatives" / layer
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "ctc.yaml").write_text(
            yaml.safe_dump({"value": {"value": {"2026": 3000}}}, sort_keys=False)
        )

    expanded = expand_stack(
        stack_file_path(tmp_path / "config" / "scenarios", "package"),
        small_tree,
        row_id="ctc_package",
    )
    assert [s.id for s in expanded.scenarios] == [
        "ctc_package.1_bigger",
        "ctc_package.2_phaseout",
    ]
    # The provision keeps its full path, so a refusal can point at the folder.
    assert [s.provision for s in expanded.scenarios] == [
        "ctc/1_bigger",
        "ctc/2_phaseout",
    ]
    assert expanded.increments[1].subtrahend_id == "ctc_package.1_bigger"


def test_two_provisions_whose_folders_share_a_name_are_refused(tmp_path):
    """They would expand into two scenarios with one name, and one output."""
    path = write_stack(
        tmp_path,
        "package",
        {
            "version": 1,
            "provisions": ["ctc/bigger", "eitc/bigger"],
        },
    )
    with pytest.raises(ConfigError, match="same folder name"):
        read_law_stack(path)


# ---------------------------------------------------------------------------
# Provisions that change the economy or behavior leg
# ---------------------------------------------------------------------------


def test_a_provision_may_change_the_economy_leg_and_keep_the_law_below_it(
    small_tree, tmp_path
):
    """The Clausing package's eighth provision, in miniature.

    It changes no tax law: the provision below it stands, and the corporate
    saving assumption moves instead. The law it runs has to be the same law,
    block for block, or the increment would pick up a law change nobody wrote.
    """
    path = write_stack(
        tmp_path,
        "package",
        {
            "version": 1,
            "provisions": [
                "p1",
                "p2",
                {"name": "more_corporate_saving", "economy": "saving_75"},
            ],
        },
    )
    expanded = expand_stack(path, small_tree)

    assert [s.provision for s in expanded.scenarios] == [
        "p1",
        "p2",
        "more_corporate_saving",
    ]
    law_change, saving_change = expanded.scenarios[1], expanded.scenarios[2]
    assert saving_change.law.parameters == law_change.law.parameters
    assert saving_change.law_layers == ("p1", "p2")
    assert saving_change.economy == "saving_75"
    # The two law provisions said nothing about the economy leg, so the
    # runscript row's own cell stands for them.
    assert law_change.economy is None
    assert law_change.behavior is None
    # And the increment is still the step up from the provision below it.
    assert expanded.increments[2].subtrahend_id == "p2"


def test_an_economy_change_carries_forward_to_the_provisions_above_it(
    small_tree, tmp_path
):
    """A leg one provision moves stays moved until another provision moves it."""
    path = write_stack(
        tmp_path,
        "package",
        {
            "version": 1,
            "provisions": [
                {"tax_law": "p1", "economy": "saving_25"},
                "p2",
                {"tax_law": "p3", "behavior": "employment"},
            ],
        },
    )
    expanded = expand_stack(path, small_tree)

    assert [s.economy for s in expanded.scenarios] == [
        "saving_25",
        "saving_25",
        "saving_25",
    ]
    assert [s.behavior for s in expanded.scenarios] == [None, None, "employment"]
    # A provision written as a mapping is named after its law folder, the same
    # as one written as a bare string.
    assert [s.provision for s in expanded.scenarios] == ["p1", "p2", "p3"]


def test_a_provision_naming_no_leg_at_all_is_refused(tmp_path):
    path = write_stack(
        tmp_path,
        "package",
        {"version": 1, "provisions": ["p1", {"name": "nothing"}]},
    )
    with pytest.raises(ConfigError, match="changes none of"):
        read_law_stack(path)


def test_an_unknown_key_inside_a_provision_is_refused_by_name(tmp_path):
    path = write_stack(
        tmp_path,
        "package",
        {
            "version": 1,
            "provisions": [{"tax_law": "p1", "econonmy": "saving_25"}],
        },
    )
    with pytest.raises(ConfigError, match="unknown key `econonmy`"):
        read_law_stack(path)


def test_two_provisions_that_would_share_a_name_are_refused_however_written(tmp_path):
    path = write_stack(
        tmp_path,
        "package",
        {
            "version": 1,
            "provisions": ["p1", {"name": "p1", "economy": "saving_25"}],
        },
    )
    with pytest.raises(ConfigError, match="appear twice"):
        read_law_stack(path)


# ---------------------------------------------------------------------------
# The Clausing package, rebuilt by folding
# ---------------------------------------------------------------------------


def test_the_clausing_package_expands_into_its_eight_provisions_in_order():
    """The published table's eight rows, built by folding rather than by hand."""
    expanded = expand_stack(stack_file_path(CONFIG_ROOT, "clausing_v2"), CONFIG_ROOT)
    assert [s.provision for s in expanded.scenarios] == [
        *CLAUSING_LAW_PROVISIONS,
        "08_corporate",
    ]
    # The eighth provision changes the corporate saving assumption, not the
    # law, so it runs the seventh provision's law unchanged.
    seventh, eighth = expanded.scenarios[-2], expanded.scenarios[-1]
    assert eighth.law.parameters == seventh.law.parameters
    assert eighth.economy == "clausing_corporate_saving_50"


def test_the_clausing_fold_blames_the_right_provision_for_every_block():
    """The manifest's audit trail: which folder wrote each subparameter in force."""
    expanded = expand_stack(stack_file_path(CONFIG_ROOT, "clausing_v2"), CONFIG_ROOT)
    last = expanded.scenarios[-1].law

    # The ordinary rate schedule was written twice: the first provision put the
    # 1997 schedule in and the second spliced the two bottom brackets back out,
    # so the second one's block is the law in force and both are recorded.
    assert last.written_by("ord", "rates") == "clausing_provisions/02_restore_bottom_rates"
    assert last.conflicts()["ord.rates"] == (
        "clausing_provisions/01_clinton_rates",
        "clausing_provisions/02_restore_bottom_rates",
    )

    # Each later provision owns its own parameter file outright.
    assert last.written_by("qbi", "rate") == "clausing_provisions/03_199a"
    assert last.written_by("pref", "rates") == "clausing_provisions/05_pref_rates"
    assert (
        last.written_by("pref", "kg_death_regime_equities")
        == "clausing_provisions/04_carryover_basis"
    )
    assert last.written_by("estate", "rates") == "clausing_provisions/07_estate"
