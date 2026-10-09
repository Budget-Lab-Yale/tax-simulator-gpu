"""Resolving the folder that holds another model's published data."""

from __future__ import annotations

from pathlib import Path

import pytest

from taxmodel.config.errors import ConfigError
from taxmodel.config.interfaces import interface_key, interface_root


def test_the_interface_name_becomes_the_configuration_key():
    assert interface_key("Macro-Projections") == "macro_projections"
    assert interface_key("Off-Model-Estimates") == "off_model_estimates"


def test_the_path_is_root_type_name_version_vintage_scenario():
    root = interface_root("Macro-Projections")
    assert root == Path(
        "/nfs/roberts/project/pi_nrs36/shared/model_data"
        "/Macro-Projections/v3/2026022522/baseline"
    )


def test_an_economy_alternative_can_pin_a_different_vintage():
    """`multi_module_smoke` holds Tax-Data at an older publication than the
    default layer does, and says nothing about the other interfaces."""
    pinned = interface_root("Tax-Data", economy="multi_module_smoke")
    assert pinned.parent.name == "2026050315"
    assert interface_root("Tax-Data").parent.name == "2026070814"
    assert interface_root(
        "Macro-Projections", economy="multi_module_smoke"
    ) == interface_root("Macro-Projections")


def test_an_unpinned_interface_is_refused():
    with pytest.raises(ConfigError, match="not pinned"):
        interface_root("Consumption-Model")


def test_an_unknown_economy_alternative_is_refused():
    with pytest.raises(ConfigError, match="unknown economy alternative"):
        interface_root("Macro-Projections", economy="no_such_folder")
