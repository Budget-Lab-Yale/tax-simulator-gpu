"""Rebuild the frozen baseline law that the calculator tests read.

Compiles this repository's default tax law (`config/scenarios/tax_law/default`)
for 2025 through 2036, with the price and wage index series in
`tests/golden/fixtures/law/indexes.csv`, and writes the compiled arrays and
their field schema to `tests/golden/fixtures/law/`. It then regenerates
`docs/compiled_law_schema.md` from the new schema.

Run from the repository root:

    PYTHONPATH=src python tests/golden/freeze_compiled_law.py

Rebuilding the fixture changes the law the calculator tests run against, so
any test that pins a tax amount computed under it may need its pinned value
updated.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from taxmodel.law import compile_law
from taxmodel.law.indexes import index_table_from_frame
from taxmodel.law.schema_doc import main as write_doc

CONFIG_ROOT = Path("config/scenarios")
FIXTURES = Path("tests/golden/fixtures/law")

# The years the fixture covers. The calculator tests pin values in this window.
YEARS = range(2025, 2037)


def main() -> int:
    index_table = index_table_from_frame(pd.read_csv(FIXTURES / "indexes.csv"))
    law = compile_law(CONFIG_ROOT, YEARS, index_table)
    law.save_npz(FIXTURES / "compiled_baseline.npz")
    law.write_schema_json(FIXTURES / "compiled_baseline_schema.json")
    print(
        f"froze {len(law.schema)} fields over {len(law.years)} years "
        f"x {law.max_brackets} slots"
    )
    return write_doc([str(FIXTURES / "compiled_baseline_schema.json")])


if __name__ == "__main__":
    raise SystemExit(main())
