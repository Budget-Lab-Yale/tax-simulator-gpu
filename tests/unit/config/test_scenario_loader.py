"""The two-layer scenario loader: what an alternative may and may not do."""

from __future__ import annotations

import pytest

from taxmodel.config import ConfigError, load_defaults, resolve
from taxmodel.config.scenario import interface_vintages


def economy(tree, alternative=None):
    layer = load_defaults(tree.root, "economy", base_dir=tree.base_dir)
    return resolve(tree.root, layer, alternative)


def test_default_layer_resolves_every_channel(tree):
    resolved = economy(tree)
    assert resolved.alternative == "default"
    assert resolved.values["corp"]["sigma_n"] == 0.375
    assert resolved.values["interfaces"]["tax_data_vintage"] == "2030010100"
    # The channel's role reaches every entry in it.
    assert resolved.roles["corp"]["sigma_n"] == "transmission"
    assert resolved.roles["estate"]["widowhood_share"] == "state"
    assert not resolved.overrides


def test_alternative_changes_only_what_it_names(tree):
    tree.write(
        "config/scenarios/economy/alternatives/high_sigma/corp.yaml",
        {"sigma_n": {"value": 0.5}},
    )
    resolved = economy(tree, "high_sigma")

    assert resolved.values["corp"]["sigma_n"] == 0.5
    assert resolved.values["corp"]["rate_eti"] == 0.2
    assert resolved.values["gains"]["timeable_share"] == 0.25

    (override,) = resolved.overrides
    assert (override.channel, override.name) == ("corp", "sigma_n")
    assert override.default == 0.375
    assert override.source == "alternative:high_sigma"


def test_restating_an_entry_replaces_its_justification_entire(tree):
    """An alternative giving its own kind owes that kind's provenance in full."""
    tree.write(
        "config/scenarios/economy/alternatives/judged/corp.yaml",
        {
            "sigma_n": {
                "value": 0.5,
                "kind": "judgment",
                "note": "House convention for the upper corner.",
            }
        },
    )
    resolved = economy(tree, "judged")
    assert resolved.entries["corp"]["sigma_n"].kind == "judgment"
    assert "citation" not in resolved.entries["corp"]["sigma_n"].raw


def test_restated_entry_missing_its_provenance_is_refused(tree):
    tree.write(
        "config/scenarios/economy/alternatives/unjustified/corp.yaml",
        {"sigma_n": {"value": 0.5, "kind": "sourced"}},
    )
    with pytest.raises(ConfigError, match="requires citation"):
        economy(tree, "unjustified")


def test_alternative_naming_an_unknown_entry_stops(tree):
    """A misspelled entry would otherwise change nothing and be reported as if it had."""
    tree.write(
        "config/scenarios/economy/alternatives/typo/corp.yaml",
        {"sigma_m": {"value": 0.5}},
    )
    with pytest.raises(ConfigError, match="unknown economy entry: corp.sigma_m"):
        economy(tree, "typo")


def test_alternative_naming_an_unknown_channel_stops(tree):
    tree.write(
        "config/scenarios/economy/alternatives/typo/corpse.yaml",
        {"sigma_n": {"value": 0.5}},
    )
    with pytest.raises(ConfigError, match="unknown economy channel: corpse"):
        economy(tree, "typo")


def test_locked_entry_refuses_override(tree):
    """The estate measurement bridge measures the data; a reform cannot restate it."""
    tree.write(
        "config/scenarios/economy/alternatives/meddling/estate.yaml",
        {"widowhood_share": {"value": "calibrations/timeable.yaml"}},
    )
    with pytest.raises(ConfigError, match="estate.widowhood_share is locked"):
        economy(tree, "meddling")


def test_override_without_a_value_stops(tree):
    tree.write(
        "config/scenarios/economy/alternatives/empty/corp.yaml",
        {"sigma_n": {"note": "meant to change this"}},
    )
    with pytest.raises(ConfigError, match="must supply a `value`"):
        economy(tree, "empty")


def test_unknown_alternative_stops(tree):
    with pytest.raises(ConfigError, match="unknown economy alternative 'nope'"):
        economy(tree, "nope")


def test_default_under_alternatives_is_unreachable_and_refused(tree):
    tree.write(
        "config/scenarios/economy/alternatives/default/corp.yaml",
        {"sigma_n": {"value": 0.5}},
    )
    with pytest.raises(ConfigError, match="reserved runscript cell"):
        load_defaults(tree.root, "economy", base_dir=tree.base_dir)


def test_economy_channel_must_declare_a_role(tree):
    tree.write(
        "config/scenarios/economy/default/labor.yaml",
        {"eti": {"value": 0.25, "kind": "judgment", "note": "synthetic"}},
    )
    with pytest.raises(ConfigError, match="must declare _channel"):
        load_defaults(tree.root, "economy", base_dir=tree.base_dir)


def test_every_entry_declares_a_kind_and_owes_its_provenance(tree):
    tree.write(
        "config/scenarios/economy/default/labor.yaml",
        {
            "_channel": {"role": "state"},
            "eti": {"value": 0.25},
            "frisch": {"value": 0.3, "kind": "sourced"},
            "share": 0.5,
        },
    )
    with pytest.raises(ConfigError) as caught:
        load_defaults(tree.root, "economy", base_dir=tree.base_dir)
    message = str(caught.value)
    assert "labor.eti: kind must be one of" in message
    assert "labor.frisch: kind `sourced` requires citation" in message


def test_calibrated_entry_pointing_at_a_missing_file_stops_at_load(tree):
    """A dependency that does not exist could never trip the staleness check."""
    entries = tree.read("config/scenarios/economy/default/gains.yaml")
    entries["timeable_share"]["invalidated_by"] = ["calibrations/gone.yaml"]
    tree.write("config/scenarios/economy/default/gains.yaml", entries)
    with pytest.raises(ConfigError, match="does not exist"):
        load_defaults(tree.root, "economy", base_dir=tree.base_dir)


def test_behavior_leg_carries_no_channel_values(tree):
    layer = load_defaults(tree.root, "behavior", base_dir=tree.base_dir)
    assert layer.entries == {}


def test_behavior_channel_may_not_declare_a_role(tree):
    tree.write(
        "config/scenarios/behavior/default/charity.yaml",
        {"_channel": {"role": "state"}},
    )
    with pytest.raises(ConfigError, match="roles apply to the economy leg only"):
        load_defaults(tree.root, "behavior", base_dir=tree.base_dir)


def test_interface_vintages_come_from_the_interfaces_channel(tree):
    assert interface_vintages(economy(tree)) == {"tax_data": "2030010100"}
