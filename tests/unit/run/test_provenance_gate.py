"""A production run cannot get past a stale calibration.

The pieces of the provenance system have their own tests under
tests/unit/config. What is tested here is that the production entry point
uses them: that a run started from a runscript resolves every scenario's
three legs and stops on a stale one before the model runs at all, that the
toolchain guard is live on that path, and that what the checks found reaches
the per-scenario manifest.

The model itself is not run. `execute_run` takes the runner it calls as an
argument, so these tests hand it the stand-in from this folder's conftest —
which is also what proves the run stops early, because a run that stopped on
provenance never calls the runner at all.

Everything the tests read is written into a temporary folder: invented
channels, an invented elasticity, an invented Tax-Data vintage string. None
of it is derived from the microdata, so this runs anywhere.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from taxmodel.config import ConfigError, StaleCalibrationError, ToolchainError
from taxmodel.run import scenarios
from taxmodel.run.plan import plan_run
from taxmodel.scenario import provenance
from tests.unit.run.conftest import RUNSCRIPT_HEADER

# The one scenario every test here runs: baseline law, the default economy and
# behavior legs, one year.
RUN_YEARS = "2026:2026"

# The vintage of the (invented) microdata the calibrated share below was
# measured on. The run reads the same one until a test moves it.
PINNED_VINTAGE = "2030010100"
# The publications the run reads besides Tax-Data, at the vintages the real
# default economy leg pins.
UPSTREAM_VINTAGES = {
    "macro_projections": "2026022522",
    "off_model_estimates": "20260722",
    "cost_recovery_simulator": "202509251222",
    "estate_tax_distribution": "2025092512",
}
EVERY_VINTAGE = {"tax_data": PINNED_VINTAGE, **UPSTREAM_VINTAGES}


def write(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(body, sort_keys=False))


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A three-leg scenario tree with nothing stale in it.

    Returns the folder the provenance paths are relative to; the scenario
    tree itself is `config/scenarios` inside it, which is where the model
    looks.
    """
    # The file the calibrated share was read out of.
    dependency = tmp_path / "calibrations" / "timeable.yaml"
    write(dependency, {"timeable_share": 0.25})
    root = tmp_path / "config" / "scenarios"

    write(
        root / "economy" / "default" / "interfaces.yaml",
        {
            "_channel": {"role": "state"},
            "tax_data_vintage": {
                "value": PINNED_VINTAGE,
                "kind": "structural",
                "note": "Pins the synthetic microdata vintage this fixture reads.",
            },
            # The other publications the run reads, at the vintages the real
            # default leg pins: every interface's vintage and scenario id come
            # from the economy leg.
            **{
                f"{key}_vintage": {
                    "value": vintage,
                    "kind": "structural",
                    "note": f"Pins the {key} vintage the run reads.",
                }
                for key, vintage in UPSTREAM_VINTAGES.items()
            },
            **{
                f"{key}_id": {
                    "value": "baseline",
                    "kind": "structural",
                    "note": f"Pins the {key} scenario.",
                }
                for key in UPSTREAM_VINTAGES
            },
        },
    )
    write(
        root / "economy" / "default" / "gains.yaml",
        {
            "_channel": {"role": "state"},
            "timeable_share": {
                "value": 0.25,
                "kind": "calibrated",
                "set": "2030-05-01",
                "target": "Share of realized gains whose timing is discretionary.",
                "derived_under": {"tax_data": PINNED_VINTAGE},
                "invalidated_by": ["calibrations/timeable.yaml"],
                "invalidated_by_hashes": {
                    "calibrations/timeable.yaml": hashlib.md5(
                        dependency.read_bytes()
                    ).hexdigest()
                },
                "rederive": "other/measure_timeable.py",
            },
        },
    )
    # No behavior module runs on the static pass, but the leg is resolved and
    # recorded, so its default layer has to be there.
    (root / "behavior" / "default").mkdir(parents=True)
    # One tax law parameter is enough: the law leg is resolved for the
    # manifest here, and compiled separately by the law compiler.
    write(
        root / "tax_law" / "default" / "ctc.yaml",
        {"value": {"2026": 2000}},
    )
    return tmp_path


def plan(tree: Path, economy: str = "default"):
    """A one-scenario runscript against this tree, planned but not run."""
    runscript = tree / "smoke.csv"
    runscript.write_text(
        RUNSCRIPT_HEADER + f"baseline,default,{economy},default,{RUN_YEARS},\n"
    )
    return plan_run(
        runscript,
        output_root=tree / "runs",
        config_root=tree / "config" / "scenarios",
    )


def stale_run(tree: Path, economy: str = "default"):
    """A production run against this tree, with no Tax-Data anywhere near it.

    The provenance checks run before the population is read, so a run that
    stops on one never reaches the microdata, and these tests need no
    Tax-Data.
    """

    def runner_that_must_not_be_reached(*args, **kwargs):
        raise AssertionError(
            "the run reached the model, which means it did not stop on provenance"
        )

    return scenarios.execute_run(
        plan(tree, economy),
        population_root=tree / "no-tax-data-here",
        provenance_root=tree,
        group_runner=runner_that_must_not_be_reached,
    )


def test_a_production_run_stops_on_a_stale_calibration(tree):
    """The run points at a different microdata vintage than the share was measured on."""
    entries = yaml.safe_load(
        (tree / "config/scenarios/economy/default/interfaces.yaml").read_text()
    )
    entries["tax_data_vintage"]["value"] = "2031010100"
    write(tree / "config/scenarios/economy/default/interfaces.yaml", entries)

    with pytest.raises(StaleCalibrationError) as caught:
        stale_run(tree)
    message = str(caught.value)
    assert "gains.timeable_share: pinned against tax_data vintage 2030010100" in message
    assert "other/measure_timeable.py" in message


def test_a_production_run_stops_when_a_dependency_file_has_changed(tree):
    """The file the share was read out of has been edited since it was pinned."""
    write(tree / "calibrations" / "timeable.yaml", {"timeable_share": 0.31})

    with pytest.raises(StaleCalibrationError, match="has changed since this value"):
        stale_run(tree)


def test_a_production_run_stops_on_an_unverified_toolchain(tree, monkeypatch):
    """Results computed on a jax the model has not verified are not trusted."""

    def refuse() -> None:
        raise ToolchainError("jax is 0.6.2, verified stack pins 0.7.0")

    monkeypatch.setattr(provenance, "check_toolchain", refuse)
    with pytest.raises(ToolchainError):
        stale_run(tree)


def test_a_production_run_stops_when_the_scenario_names_a_folder_that_is_not_there(tree):
    with pytest.raises(ConfigError, match="unknown economy alternative"):
        stale_run(tree, economy="no_such_folder")


def test_a_clean_scenario_resolves(tree):
    provenance = scenarios.resolve_scenario_provenance(plan(tree), provenance_root=tree)[
        "baseline"
    ]
    assert provenance.staleness.clean
    assert provenance.vintages == EVERY_VINTAGE
    assert provenance.economy.alternative == "default"
    assert provenance.behavior.alternative == "default"
    assert provenance.law.alternative == "default"


def test_the_manifest_carries_the_resolved_configuration_and_the_findings(
    tree, stand_in_runner
):
    """What the checks found is written down where a reader will find it.

    The run is written for real — the stand-in supplies the numbers — so this
    is the manifest a reader of a finished scenario folder actually opens.
    """
    written = scenarios.execute_run(
        plan(tree),
        population_root=tree / "no-tax-data-here",
        provenance_root=tree,
        group_runner=stand_in_runner,
    )
    manifest = json.loads(
        (written / "scenarios" / "baseline" / "manifest.json").read_text()
    )
    rows = {(r["channel"], r["name"]): r for r in manifest["config"]}
    assert rows[("gains", "timeable_share")]["kind"] == "calibrated"
    assert rows[("gains", "timeable_share")]["overridden"] is False
    assert manifest["interface_vintages"] == EVERY_VINTAGE
    assert manifest["legs"] == {
        "tax_law": "default",
        "economy": "default",
        "behavior": "default",
    }
    assert manifest["staleness"] == {"stops": [], "warnings": []}
