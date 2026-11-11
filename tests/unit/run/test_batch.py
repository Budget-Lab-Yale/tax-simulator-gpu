"""The batch command: a parquet file of records in, each record's taxes out."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from taxmodel.frame import build_frame, prepare
from taxmodel.run import batch
from taxmodel.taxes import statutory_order

YEAR = 2019
WAGES = 50_000.0
# Dollars added to the primary earner's wages for the marginal rates on wages.
WAGE_INCREMENT = 100.0


def records() -> pd.DataFrame:
    """One single filer and one married couple, each earning $50,000 in wages,
    with every other column the command reads at zero or missing."""
    columns = (set(statutory_order.RECORD_COLUMNS)
               | set(statutory_order.ESTATE_RECORD_COLUMNS)
               | set(build_frame.FRAME_STEP_RECORD_COLUMNS)
               | set(prepare.OVERLAPPING_COLUMNS))
    columns -= batch.MADE_BY_THE_MODEL | set(prepare.DRAW_COLUMNS)
    frame = pd.DataFrame({name: np.zeros(2) for name in sorted(columns)})
    frame["id"] = np.array([11, 22], dtype=np.int64)
    frame["filing_status"] = [1.0, 2.0]
    frame["weight"] = 1.0
    frame["filer"] = 1.0
    frame["age1"] = 45.0
    frame["age2"] = [np.nan, 45.0]
    for name in ("blind2", "dep_age1", "dep_age2", "dep_age3", "divorce_year"):
        if name in frame:
            frame[name] = np.nan
    frame["wages"] = WAGES
    frame["wages1"] = WAGES
    return frame


@pytest.mark.usefixtures("macro_root")
def test_batch_writes_one_row_per_record(tmp_path):
    source = tmp_path / "records.parquet"
    out = tmp_path / "taxes.parquet"
    records().to_parquet(source, index=False)

    batch.main(["--records", str(source), "--year", str(YEAR), "--out", str(out),
                "--commit", "test", "--wage-increment", str(WAGE_INCREMENT)])

    taxes = pd.read_parquet(out)
    assert list(taxes["id"]) == [11, 22]
    assert not taxes.isna().any().any()
    # A single filer with $50,000 of wages owes income tax, and both earners
    # owe payroll tax.
    assert taxes.loc[0, "liab_iit_net"] > 0
    assert (taxes["liab_pr"] > 0).all()
    # Marginal rates on wages, income and payroll tax together, lie between
    # zero and one.
    total = taxes["mtr_wages_iit"] + taxes["mtr_wages_payroll"]
    assert ((total > 0) & (total < 1)).all()

    manifest = json.loads((tmp_path / "taxes.parquet.manifest.json").read_text())
    assert manifest["law_year"] == YEAR
    assert manifest["records"] == 2
    assert manifest["missing_values_in_outputs"] == {}


def test_batch_refuses_records_missing_a_column(tmp_path):
    source = tmp_path / "records.parquet"
    records().drop(columns="filing_status").to_parquet(source, index=False)
    with pytest.raises(ValueError, match="lacks columns"):
        batch.main(["--records", str(source), "--year", str(YEAR),
                    "--out", str(tmp_path / "taxes.parquet")])


@pytest.mark.usefixtures("macro_root")
def test_batch_reads_the_student_columns_when_the_records_carry_them(tmp_path):
    """`student1` and `student2` are optional: records without them run (the
    first test), and records with them do not have them reported as unread."""
    source = tmp_path / "records.parquet"
    out = tmp_path / "taxes.parquet"
    records().assign(student1=0.0, student2=0.0).to_parquet(source, index=False)

    batch.main(["--records", str(source), "--year", str(YEAR), "--out", str(out)])

    manifest = json.loads((tmp_path / "taxes.parquet.manifest.json").read_text())
    assert not {"student1", "student2"} & set(manifest["columns_supplied_not_read"])
