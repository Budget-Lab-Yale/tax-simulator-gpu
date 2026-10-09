"""A synthetic three-leg config tree the enforcement tests trip.

Everything here is invented: made-up channels, made-up elasticities, a
made-up Tax-Data vintage string. Nothing is PUF-derived, so these
fixtures can run in public CI.

The tree has the shape a real one has — a complete `default/` layer with
one file per channel, sparse `alternatives/` folders — and one entry of
each kind, including a calibrated entry that records both the data vintages
and the file hashes it was derived under, so each test can move exactly one
thing and see the run stop.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml


@dataclass
class Tree:
    """Where the synthetic scenario tree lives."""

    base_dir: Path       # what provenance paths are relative to (a repo root)
    root: Path           # config/scenarios

    def write(self, relative: str, body: dict | str) -> Path:
        """Write a YAML file (or raw text) under the base directory."""
        path = self.base_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(body, str):
            path.write_text(body)
        else:
            path.write_text(yaml.safe_dump(body, sort_keys=False))
        return path

    def read(self, relative: str) -> dict:
        return yaml.safe_load((self.base_dir / relative).read_text())

    def md5(self, relative: str) -> str:
        return hashlib.md5((self.base_dir / relative).read_bytes()).hexdigest()


@pytest.fixture
def tree(tmp_path: Path) -> Tree:
    """The baseline synthetic configuration: everything in date, nothing stale."""
    t = Tree(base_dir=tmp_path, root=tmp_path / "config" / "scenarios")

    # The two files a calibration is pinned against.
    t.write("calibrations/widowhood.yaml", {"share_65_74": 0.41})
    t.write("calibrations/timeable.yaml", {"timeable_share": 0.25})

    # Which upstream vintages the run reads. A state channel: it describes
    # the data, not how a tax change propagates.
    t.write(
        "config/scenarios/economy/default/interfaces.yaml",
        {
            "_channel": {"role": "state"},
            "tax_data_vintage": {
                "value": "2030010100",
                "kind": "structural",
                "note": "Pins the synthetic microdata vintage this fixture reads.",
            },
            "tax_data_id": {
                "value": "baseline",
                "kind": "structural",
                "note": "Pins the synthetic microdata scenario ID.",
            },
        },
    )

    # How a corporate tax change reaches households: a transmission channel,
    # so the static pass may not read it.
    t.write(
        "config/scenarios/economy/default/corp.yaml",
        {
            "_channel": {"role": "transmission"},
            "sigma_n": {
                "value": 0.375,
                "kind": "sourced",
                "citation": "Invented for tests; no relation to the model's value.",
            },
            "rate_eti": {
                "value": 0.2,
                "kind": "judgment",
                "note": "Invented for tests.",
            },
        },
    )

    # Two calibrated values, both measured off the data rather than chosen.
    t.write(
        "config/scenarios/economy/default/estate.yaml",
        {
            "_channel": {"role": "state"},
            # Locked and warn-level, like the real valuation bridge: it
            # measures the data, so no reform may override it, and it is run
            # against older vintages on purpose.
            "widowhood_share": {
                "value": "calibrations/widowhood.yaml",
                "kind": "calibrated",
                "locked": True,
                "enforcement": "warn",
                "set": "2030-01-02",
                "target": "Share of unmarried decedents who are widowed (synthetic).",
                "derived_under": {"tax_data": "2030010100"},
                "invalidated_by": ["calibrations/widowhood.yaml"],
                "invalidated_by_hashes": {
                    "calibrations/widowhood.yaml": t.md5("calibrations/widowhood.yaml")
                },
                "rederive": "other/write_widowhood.py",
            },
        },
    )
    t.write(
        "config/scenarios/economy/default/gains.yaml",
        {
            "_channel": {"role": "state"},
            "timeable_share": {
                "value": 0.25,
                "kind": "calibrated",
                "set": "2030-01-02",
                "target": "Share of realizations that can be timed (synthetic).",
                "derived_under": {"tax_data": "2030010100"},
                "invalidated_by": ["calibrations/timeable.yaml"],
                "invalidated_by_hashes": {
                    "calibrations/timeable.yaml": t.md5("calibrations/timeable.yaml")
                },
                "rederive": "other/measure_timeable.py",
            },
        },
    )

    # A behavior leg with the gains model off and every margin at neutral.
    t.write(
        "config/scenarios/behavior/default/behavior.yaml",
        {"kg_dynamics": "none", "margins": {}},
    )

    return t
