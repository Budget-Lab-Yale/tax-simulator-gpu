"""The targets file a person writes, and the record of the solve the calibrate
command writes."""

from __future__ import annotations

import pytest
import yaml

from taxmodel.calibration import derived, paths, targets


def test_the_targets_file_holds_the_declared_values_and_their_sources():
    declared = targets.read_targets(paths.DEFAULT_TARGETS_FILE)

    assert declared.permanent_realization_elasticity.value == pytest.approx(-0.6)
    assert declared.transitory_realization_elasticity.value == pytest.approx(1.2)
    # Each target carries one line saying whose judgment it is.
    assert declared.permanent_realization_elasticity.source.strip()
    assert declared.transitory_realization_elasticity.source.strip()


def test_the_declared_targets_convert_to_semi_elasticities():
    """As semi-elasticities the two targets are -2.5210 and +5.04."""

    # A declared elasticity divided by the top rate is the semi-elasticity
    # the simulation measures.
    assert -0.6 / targets.TOP_RATE_DIVISOR == pytest.approx(-2.5210, abs=1e-4)
    assert 1.2 / targets.TOP_RATE_DIVISOR == pytest.approx(5.042, abs=1e-3)


def test_a_target_without_a_source_is_refused():
    with pytest.raises(targets.TargetsError, match="no source line"):
        targets.targets_from_mapping(
            {
                paths.PERMANENT_TARGET: {"value": -0.6},
                paths.TRANSITORY_TARGET: {"value": 1.2, "source": "someone, today"},
            }
        )


def test_a_target_the_gains_model_does_not_have_is_refused():
    with pytest.raises(targets.TargetsError, match="does not have"):
        targets.targets_from_mapping(
            {
                paths.PERMANENT_TARGET: {"value": -0.6, "source": "a judgment"},
                paths.TRANSITORY_TARGET: {"value": 1.2, "source": "a judgment"},
                "medium_run_realization_elasticity": {
                    "value": 0.1,
                    "source": "a judgment",
                },
            }
        )


def test_a_behavior_leg_can_point_at_its_own_beliefs():
    """A scenario may declare different targets and calibrate to them."""

    own = targets.targets_path_from_behavior(
        {paths.KG_DYNAMICS_SECTION: {paths.TARGETS_KEY: "config/elsewhere.yaml"}}
    )
    assert str(own) == "config/elsewhere.yaml"

    # A leg that switches the gains machinery off, or says nothing about
    # targets, falls back to the one shared belief.
    assert targets.targets_path_from_behavior(
        {paths.KG_DYNAMICS_SECTION: "none"}
    ) == paths.DEFAULT_TARGETS_FILE


def a_record(tmp_path):
    return derived.DerivedInternals(
        derived_from=str(paths.DEFAULT_TARGETS_FILE),
        achieved_moments={
            paths.PERMANENT_TARGET: -0.5999,
            paths.TRANSITORY_TARGET: 1.2001,
        },
        derivation=derived.DerivationRecord(
            targets={paths.PERMANENT_TARGET: -0.6, paths.TRANSITORY_TARGET: 1.2},
            git_commit="0b00c9c0000000000000000000000000000000aa",
            date="2026-08-04",
            notes="Newton's method converged in 3 steps.",
        ),
    )


def test_the_derived_internals_survive_a_round_trip(tmp_path):
    path = tmp_path / "derived_internals.yaml"
    written = a_record(tmp_path)
    derived.write_derived_internals(path, written)
    read_back = derived.read_derived_internals(path)

    assert read_back == written


def test_the_file_tells_a_human_not_to_edit_it(tmp_path):
    path = tmp_path / "derived_internals.yaml"
    derived.write_derived_internals(path, a_record(tmp_path))
    text = path.read_text()

    assert "WRITTEN BY THE MACHINE" in text
    assert "No human edits this file" in text
    # And it is still ordinary yaml underneath the header.
    body = yaml.safe_load(text)
    assert body["derivation"]["targets"][paths.PERMANENT_TARGET] == pytest.approx(-0.6)


def test_the_record_does_not_hold_the_fitted_numbers(tmp_path):
    """The fitted numbers and their data vintages are in the calibration file
    a run reads, not here."""
    path = tmp_path / "derived_internals.yaml"
    derived.write_derived_internals(path, a_record(tmp_path))
    body = yaml.safe_load(path.read_text())

    assert paths.REALIZATION_ELASTICITY not in body
    assert paths.TIMEABLE_SHARE not in body
    assert "data_vintages" not in body["derivation"]


def test_a_record_that_still_holds_the_fitted_numbers_is_read(tmp_path):
    path = tmp_path / "derived_internals.yaml"
    derived.write_derived_internals(path, a_record(tmp_path))
    body = yaml.safe_load(path.read_text())
    body[paths.REALIZATION_ELASTICITY] = 5.49
    body[paths.TIMEABLE_SHARE] = 0.26
    body["derivation"]["data_vintages"] = {"value_added_tax_model": "2024050121"}
    path.write_text(yaml.safe_dump(body))

    assert derived.read_derived_internals(path) == a_record(tmp_path)


def test_internals_without_a_derivation_record_are_refused(tmp_path):
    path = tmp_path / "derived_internals.yaml"
    path.write_text(
        yaml.safe_dump({"derived_from": str(paths.DEFAULT_TARGETS_FILE)})
    )

    with pytest.raises(derived.DerivedInternalsError, match="no derivation record"):
        derived.read_derived_internals(path)
