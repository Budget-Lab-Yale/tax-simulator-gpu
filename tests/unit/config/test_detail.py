"""The detail spec: three keys, five world names, and refusals by name."""

from __future__ import annotations

import pytest

from taxmodel.config import ConfigError, read_detail_spec
from taxmodel.years.world_names import WORLD_NAMES as WORLDS


def write_spec(tmp_path, text, name="ctc_audit.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_a_well_formed_spec_reads(tmp_path):
    path = write_spec(
        tmp_path,
        "variables: [liab_iit, credit_ctc, mtr_wages]\n"
        "worlds: [baseline, first_order]\n"
        "years: 2030:2032\n",
    )
    spec = read_detail_spec(path)
    assert spec.variables == ("liab_iit", "credit_ctc", "mtr_wages")
    assert spec.worlds == ("baseline", "first_order")
    assert spec.years == (2030, 2031, 2032)


def test_the_word_all_means_every_year_the_scenario_runs(tmp_path):
    path = write_spec(tmp_path, "variables: [liab_iit]\nworlds: [baseline]\nyears: all\n")
    spec = read_detail_spec(path)
    assert spec.years is None
    assert spec.kept_years([2026, 2027]) == (2026, 2027)


def test_omitting_the_years_key_also_means_every_year(tmp_path):
    path = write_spec(tmp_path, "variables: [liab_iit]\nworlds: [baseline]\n")
    assert read_detail_spec(path).years is None


def test_years_outside_the_scenarios_own_years_are_dropped(tmp_path):
    """The years filter is applied on the way out and cannot add a year to the run."""
    path = write_spec(
        tmp_path, "variables: [liab_iit]\nworlds: [baseline]\nyears: 2026:2030\n"
    )
    assert read_detail_spec(path).kept_years([2026, 2027]) == (2026, 2027)


def test_the_five_world_names_are_the_ones_the_run_evaluates():
    assert WORLDS == (
        "baseline",
        "first_order",
        "mechanical_ex_saving",
        "mechanical",
        "conventional",
    )


def test_a_world_that_does_not_exist_is_refused_by_name(tmp_path):
    path = write_spec(tmp_path, "variables: [liab_iit]\nworlds: [static]\n")
    with pytest.raises(ConfigError, match="worlds that do not exist: static"):
        read_detail_spec(path)


def test_a_key_the_spec_does_not_have_is_refused(tmp_path):
    path = write_spec(
        tmp_path, "variables: [liab_iit]\nworlds: [baseline]\nhouseholds: top 100\n"
    )
    with pytest.raises(ConfigError, match="keys a detail spec does not have"):
        read_detail_spec(path)


def test_every_problem_in_a_spec_is_reported_at_once(tmp_path):
    path = write_spec(tmp_path, "worlds: [static, conventional]\nyears: 20x6\n")
    with pytest.raises(ConfigError) as caught:
        read_detail_spec(path)
    message = str(caught.value)
    assert "'variables' is missing or empty" in message
    assert "worlds that do not exist: static" in message
    assert "'years' does not parse" in message


def test_a_spec_that_is_not_there_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="there is no detail spec at"):
        read_detail_spec(tmp_path / "absent.yaml")
