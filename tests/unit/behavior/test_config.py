"""Reading a scenario's behavior leg, and the numbers a margin it omits runs at."""

from pathlib import Path

import pytest

from taxmodel.behavior import config as behavior_config
from taxmodel.behavior import loader, parameters

CONFIG_ROOT = "config/scenarios"


def write_leg(tmp_path, name, text):
    """Put one behavior.yaml in a scenario tree and return the tree's root."""
    root = tmp_path / "scenarios"
    (root / "behavior" / "default").mkdir(parents=True, exist_ok=True)
    (root / "behavior" / "default" / "behavior.yaml").write_text(
        "kg_dynamics: none\nmargins: {}\n"
    )
    if name != "default":
        folder = root / "behavior" / "alternatives" / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "behavior.yaml").write_text(text)
    return root


def test_the_reserved_word_none_reads_no_file_and_turns_no_margin_on():
    leg = behavior_config.read_behavior_leg(CONFIG_ROOT, "none")
    assert not leg.runs_the_gains_model
    assert leg.margins == {}


def test_a_margin_is_named_once_with_its_numbers_under_it(tmp_path):
    root = write_leg(
        tmp_path,
        "generous",
        """
kg_dynamics: none

margins:
  charity:
    elasticity: -1.0
    max_percent_change: 1.0
""",
    )
    leg = behavior_config.read_behavior_leg(root, "generous")
    assert set(leg.margins) == {"charity"}
    assert leg.margins["charity"] == {"elasticity": -1.0, "max_percent_change": 1.0}


def test_a_margin_that_is_on_states_every_parameter_it_takes(tmp_path):
    """A margin cannot be turned on at its module's defaults by naming it."""
    root = write_leg(
        tmp_path,
        "half_stated",
        "kg_dynamics: none\n\nmargins:\n  charity:\n    elasticity: -1.0\n",
    )
    with pytest.raises(
        behavior_config.BehaviorConfigError, match="max_percent_change"
    ):
        behavior_config.read_behavior_leg(root, "half_stated")


def test_a_parameter_the_margin_does_not_have_is_refused(tmp_path):
    root = write_leg(
        tmp_path,
        "misspelled",
        """
kg_dynamics: none

margins:
  charity:
    elasticity: -1.0
    max_percent_change: 1.0
    elastic1ty: -1.0
""",
    )
    with pytest.raises(behavior_config.BehaviorConfigError, match="elastic1ty"):
        behavior_config.read_behavior_leg(root, "misspelled")


def test_an_alternative_replaces_a_section_and_inherits_the_one_it_omits():
    """Sections are replaced whole, never merged."""
    leg = behavior_config.read_behavior_leg(CONFIG_ROOT, "kg_dynamics_only")
    # The kg_dynamics section was replaced.
    assert leg.runs_the_gains_model
    assert "realization" in leg.kg_dynamics
    # This alternative turns no margin on, so nothing responds but the gains
    # realization model itself.
    assert leg.margins == {}


def test_the_capital_gains_machinery_cannot_be_named_as_a_margin(tmp_path):
    """The conversion of salary into equity is bound under kg_dynamics.

    Its response parameter sigma lives in the calibration file that section
    binds, so naming it in the margins block is a mistake and stops the run.
    """
    root = write_leg(
        tmp_path,
        "hand_listed",
        "kg_dynamics: none\n\nmargins:\n  conversion:\n    sigma: 0.5\n",
    )
    with pytest.raises(behavior_config.BehaviorConfigError, match="kg_dynamics"):
        behavior_config.read_behavior_leg(root, "hand_listed")


def test_a_margin_this_model_does_not_have_is_refused_with_the_list_that_it_does(
    tmp_path,
):
    """The model has no employment response, so a leg that names one stops."""
    root = write_leg(
        tmp_path,
        "old",
        "kg_dynamics: none\n\nmargins:\n  employment: {}\n",
    )
    with pytest.raises(behavior_config.BehaviorConfigError, match="not a behavioral margin"):
        behavior_config.read_behavior_leg(root, "old")


def test_a_top_level_key_other_than_the_two_sections_is_refused(tmp_path):
    """A leg written with a key the reader does not know stops the run."""
    root = write_leg(
        tmp_path,
        "old_format",
        "modules:\n  tips: {}\n",
    )
    with pytest.raises(behavior_config.BehaviorConfigError, match="`modules`"):
        behavior_config.read_behavior_leg(root, "old_format")


def test_every_behavior_leg_in_the_config_tree_reads():
    """Every behavior alternative in the tree resolves."""

    alternatives = Path(CONFIG_ROOT, "behavior", "alternatives")
    legs = sorted(alternatives.rglob("behavior.yaml"))
    assert legs
    for leg in legs:
        name = leg.parent.relative_to(alternatives).as_posix()
        behavior_config.read_behavior_leg(CONFIG_ROOT, name)


def test_a_leg_that_omits_a_margin_runs_it_at_the_values_that_silence_it():
    """A scenario that says nothing about giving gets a zero elasticity."""
    asked_for_nothing = behavior_config.read_behavior_leg(CONFIG_ROOT, "none")
    values = parameters.values_for_leg(asked_for_nothing)

    _function, _manifest, defaults = loader.import_family("charity")
    assert values["charity"]["elasticity"] == 0.0
    # Everything the module ships that is not part of the silencing is untouched.
    assert values["charity"]["max_percent_change"] == defaults["max_percent_change"]


def test_a_leg_that_turns_a_margin_on_keeps_its_own_numbers():
    obbba = behavior_config.read_behavior_leg(CONFIG_ROOT, "obbba_stack")
    values = parameters.values_for_leg(obbba)
    assert values["overtime"]["phase_in_years"] == 3
    assert values["overtime"]["elasticity_intensive"] < 0.0


def test_every_scenario_carries_a_number_for_every_margin():
    """Every margin runs in every scenario, so every one of them has a value.

    Three legs that turn on very different responses still present the same
    parameter block, margin for margin, so they share one compiled program.
    """
    blocks = [
        parameters.values_for_leg(
            behavior_config.read_behavior_leg(CONFIG_ROOT, name)
        )
        for name in ("obbba_stack", "ot_france_1yr", "none")
    ]
    for block in blocks:
        assert set(block) == set(loader.RESPONSE_FAMILIES)
    for family in loader.RESPONSE_FAMILIES:
        parameter_names = {tuple(sorted(block[family])) for block in blocks}
        assert len(parameter_names) == 1, family
