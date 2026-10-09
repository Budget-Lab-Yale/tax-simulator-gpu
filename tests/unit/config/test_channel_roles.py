"""Reading economy values off a resolved scenario, and the roles they carry."""

from __future__ import annotations

import pytest

from taxmodel.config import ConfigError, ScenarioConfig, load_defaults, resolve


@pytest.fixture
def config(tree) -> ScenarioConfig:
    layer = load_defaults(tree.root, "economy", base_dir=tree.base_dir)
    return ScenarioConfig(economy=resolve(tree.root, layer))


def test_economy_values_are_read_by_channel_and_name(config):
    assert config.economy_param("interfaces", "tax_data_vintage") == "2030010100"
    assert config.economy_param("gains", "timeable_share") == 0.25
    assert config.economy_param("corp", "sigma_n") == 0.375


def test_an_entry_may_carry_a_role_its_channel_does_not(tree):
    """One state value can sit in a transmission file, and the manifest says so."""
    entries = tree.read("config/scenarios/economy/default/corp.yaml")
    entries["rate_eti"]["role"] = "state"
    tree.write("config/scenarios/economy/default/corp.yaml", entries)

    layer = load_defaults(tree.root, "economy", base_dir=tree.base_dir)
    config = ScenarioConfig(economy=resolve(tree.root, layer))
    assert config.economy.roles["corp"]["rate_eti"] == "state"
    assert config.economy.roles["corp"]["sigma_n"] == "transmission"


def test_reading_with_no_economy_leg_stops(config):
    empty = ScenarioConfig()
    with pytest.raises(ConfigError, match="no economy leg has been resolved"):
        empty.economy_param("corp", "sigma_n")


def test_unknown_entry_stops(config):
    with pytest.raises(ConfigError, match="unknown economy entry: corp.sigma_q"):
        config.economy_param("corp", "sigma_q")
