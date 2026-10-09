"""The schema document is generated, so it cannot be edited into disagreement."""

from __future__ import annotations

import json
from pathlib import Path

from taxmodel.law.schema_doc import render
from tests.conftest import GOLDEN_LAW_FIXTURES

DOC = Path(__file__).resolve().parents[3] / "docs" / "compiled_law_schema.md"
FROZEN = GOLDEN_LAW_FIXTURES / "compiled_baseline_schema.json"


def test_the_checked_in_document_is_what_the_schema_renders():
    """Regenerate with: uv run --no-sync python -m taxmodel.law.schema_doc"""
    schema = json.loads(FROZEN.read_text())
    assert DOC.read_text() == render(schema)


def test_the_frozen_schema_describes_every_field_the_arrays_carry():
    import numpy as np

    schema = json.loads(FROZEN.read_text())
    arrays = np.load(GOLDEN_LAW_FIXTURES / "compiled_baseline.npz")
    names = [field["name"] for field in schema["fields"]]
    assert sorted(names) == sorted(
        key[len("value/") :] for key in arrays.files if key.startswith("value/")
    )
    for field in schema["fields"]:
        assert arrays["value/" + field["name"]].shape == tuple(schema["array_shape"])
        assert arrays["active/" + field["name"]].dtype == bool


def test_the_frozen_fixture_covers_2025_to_2036():
    schema = json.loads(FROZEN.read_text())
    assert schema["years"][0] == 2025
    assert schema["years"][-1] == 2036
    assert schema["max_brackets"] == 16
    assert schema["filing_statuses"] == [1, 2, 3, 4]
