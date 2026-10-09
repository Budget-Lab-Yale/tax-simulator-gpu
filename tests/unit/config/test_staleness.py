"""The two staleness checks, data vintages and file contents, each tripped in turn.

Each test moves exactly one input a calibrated value was derived under and
confirms the run stops rather than reporting a number whose provenance no
longer describes it.
"""

from __future__ import annotations

import pytest

from taxmodel.config import (
    StaleCalibrationError,
    check_leg,
    interface_vintages,
    load_defaults,
    resolve,
)


def resolved_economy(tree, alternative=None):
    layer = load_defaults(tree.root, "economy", base_dir=tree.base_dir)
    return resolve(tree.root, layer, alternative)


def check(tree, alternative=None, enforce=True):
    economy = resolved_economy(tree, alternative)
    report = check_leg(economy, interface_vintages(economy))
    if enforce:
        report.raise_if_stopping()
    return report


def test_a_clean_configuration_passes(tree):
    report = check(tree)
    assert report.clean


def test_a_different_data_vintage_stops(tree):
    """A share measured off one microdata vintage is not the share in another."""
    entries = tree.read("config/scenarios/economy/default/interfaces.yaml")
    entries["tax_data_vintage"]["value"] = "2031010100"
    tree.write("config/scenarios/economy/default/interfaces.yaml", entries)

    with pytest.raises(StaleCalibrationError) as caught:
        check(tree)
    message = str(caught.value)
    assert "gains.timeable_share: pinned against tax_data vintage 2030010100" in message
    assert "other/measure_timeable.py" in message


def test_a_changed_dependency_file_stops(tree):
    """The file the value was read out of has been edited since it was pinned."""
    tree.write("calibrations/timeable.yaml", {"timeable_share": 0.31})

    with pytest.raises(StaleCalibrationError, match="has changed since this value"):
        check(tree)


def test_overriding_the_value_is_the_acknowledgment(tree):
    """Naming a new value in an alternative retires the old value's provenance."""
    tree.write("calibrations/timeable.yaml", {"timeable_share": 0.31})
    tree.write(
        "config/scenarios/economy/alternatives/own_number/gains.yaml",
        {"timeable_share": {"value": 0.31}},
    )
    report = check(tree, "own_number")
    assert report.clean


def test_warn_level_entries_do_not_stop_the_run(tree):
    """The estate bridge is locked and run against older vintages on purpose.

    With no override available to it, a hard stop would make the historical
    comparison runs impossible rather than caveated.
    """
    tree.write("calibrations/widowhood.yaml", {"share_65_74": 0.44})
    report = check(tree)
    assert not report.stops
    assert any("estate.widowhood_share" in w for w in report.warnings)


def test_a_calibration_pinned_against_nothing_is_a_finding(tree):
    """A value with dependencies but no recorded hashes was never really pinned."""
    entries = tree.read("config/scenarios/economy/default/gains.yaml")
    del entries["timeable_share"]["invalidated_by_hashes"]
    tree.write("config/scenarios/economy/default/gains.yaml", entries)

    with pytest.raises(StaleCalibrationError, match="no invalidated_by_hashes"):
        check(tree)


def test_derived_under_naming_an_unused_interface_is_a_finding(tree):
    entries = tree.read("config/scenarios/economy/default/gains.yaml")
    entries["timeable_share"]["derived_under"] = {"scf": "2022"}
    tree.write("config/scenarios/economy/default/gains.yaml", entries)

    with pytest.raises(StaleCalibrationError, match="an interface this run does not use"):
        check(tree)


def test_enforce_false_reports_without_stopping(tree):
    """The re-pinning tools need to see every finding at once, not the first."""
    tree.write("calibrations/timeable.yaml", {"timeable_share": 0.31})
    report = check(tree, enforce=False)
    assert report.stops
