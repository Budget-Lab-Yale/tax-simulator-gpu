"""Convert one Tax-Data vintage, once, into the form the model reads.

Tax-Data publishes about 200 MB of comma-separated text per year, and parsing
that text is the slow part of starting a run. Tax-Data's wage columns also
overlap (an earner's reported wages already contain that earner's tips and
overtime, and the household columns repeat the two earners' amounts), so they
are split into mutually exclusive pieces before any calculator adds them up.

This module does that work once per vintage and writes one Parquet file per
year holding the columns the model reads, plus a manifest saying what was
converted, by what code, and what was wrong with it. Everything it does is
independent of the scenario:

- parsing the text,
- the wage split into mutually exclusive per-earner pieces,
- the random draws, which are a fixed function of the run seed, the column's
  name and the record's id.

Net worth is computed by the frame steps every year
(`frame/steps_jax.add_net_worth`), since a marginal rate run differentiates
with respect to it (`taxes/individual_income_tax.py`) and the wealth tax's
avoidance module replaces it with the avoided base.

The bridge is temporary: when the Tax-Data build itself emits mutually
exclusive components, this module is deleted and the model's input path
points straight at Tax-Data.

Run it as a job, since it reads every year of the vintage:

    sbatch jobs/bridge_convert.sbatch
"""

from __future__ import annotations

import argparse
import datetime as dt
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from taxmodel.config.manifest import git_state, runtime_stack
from taxmodel.frame import prepare


@dataclass(frozen=True)
class YearReport:
    """The conversion results for one year."""

    year: int
    records: int
    # Records whose reported tips and overtime exceed that earner's wages, so
    # the earner's base pay comes out negative, a Tax-Data defect. The records
    # are carried through unchanged and counted here.
    negative_base_wage_ids: dict[int, list[int]] = field(default_factory=dict)

    @property
    def records_with_negative_base_wages(self) -> int:
        """Households where either earner's base pay is negative."""
        both = set()
        for ids in self.negative_base_wage_ids.values():
            both.update(ids)
        return len(both)

    def as_manifest_entry(self) -> dict:
        counts = {
            f"negative_base_wages_earner_{earner}": len(ids)
            for earner, ids in sorted(self.negative_base_wage_ids.items())
        }
        return {
            "records": self.records,
            **counts,
            "records_with_negative_base_wages": self.records_with_negative_base_wages,
            # The ids are listed so that a reader can look into them.
            "negative_base_wage_ids": {
                f"earner_{earner}": [int(i) for i in ids]
                for earner, ids in sorted(self.negative_base_wage_ids.items())
                if ids
            },
        }


def available_years(source_root: str | Path) -> tuple[int, ...]:
    """Every year this Tax-Data vintage publishes, in order."""
    years = sorted(
        int(path.stem.removeprefix("tax_units_"))
        for path in Path(source_root).glob("tax_units_*.csv")
    )
    if not years:
        raise FileNotFoundError(f"{source_root} holds no tax_units_<year>.csv files")
    return tuple(years)


def convert_year(
    source_root: str | Path, year: int, seed: int = prepare.DEFAULT_SEED
) -> tuple[pd.DataFrame, YearReport]:
    """One year of Tax-Data text in the form the model reads.

    The columns are in Tax-Data's order with the overlapping columns and the
    columns nothing in the model reads removed, then the two per-earner base
    wage columns, then the year and the decedent flag, then the random draws.
    """
    year = int(year)
    # The wage split warns about records with negative base pay. The same
    # records are counted into the manifest below, so the warning is silenced.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frame = prepare.read_tax_data_year_frame(source_root, year)
    # The columns no part of the model reads (prepare.NEVER_READ_COLUMNS) are
    # not written.
    frame = frame.drop(
        columns=[c for c in prepare.NEVER_READ_COLUMNS if c in frame.columns]
    )
    draws = prepare.keyed_draws(frame["id"].to_numpy(), seed=seed)
    frame = frame.merge(draws, on="id", how="left", validate="one_to_one")

    negative = {
        earner: frame.loc[frame[f"wages_base{earner}"] < 0, "id"].tolist()
        for earner in (1, 2)
    }
    return frame, YearReport(year=year, records=len(frame), negative_base_wage_ids=negative)


def convert_vintage(
    source_root: str | Path,
    out_root: str | Path,
    years: Sequence[int] | None = None,
    seed: int = prepare.DEFAULT_SEED,
) -> dict:
    """Convert every year of one vintage and write the manifest beside them.

    Years are converted one at a time and released before the next is read,
    since a single year is about 400 MB in memory.
    """
    source_root = Path(source_root)
    out_root = Path(out_root)
    years = tuple(available_years(source_root)) if years is None else tuple(
        int(y) for y in years
    )
    out_root.mkdir(parents=True, exist_ok=True)

    reports: list[YearReport] = []
    for year in years:
        frame, report = convert_year(source_root, year, seed=seed)
        frame.to_parquet(prepare.bridge_year_path(out_root, year), index=False)
        reports.append(report)
        print(
            f"{year}: {report.records:,} records, "
            f"{report.records_with_negative_base_wages} with negative base wages",
            flush=True,
        )
        del frame

    manifest = build_bridge_manifest(source_root, out_root, seed, reports)
    (out_root / prepare.BRIDGE_MANIFEST_NAME).write_text(
        yaml.safe_dump(manifest, sort_keys=False)
    )
    return manifest


def build_bridge_manifest(
    source_root: Path, out_root: Path, seed: int, reports: Sequence[YearReport]
) -> dict:
    """The manifest: what was converted, from where, by what code, and what
    was wrong with it."""
    return {
        "source_vintage": str(source_root),
        "output": str(out_root),
        "converted_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "bridge_code": {
            "module": "taxmodel.frame.bridge",
            "git": git_state(Path(__file__).resolve().parents[3]),
            "runtime": runtime_stack(),
        },
        "random_draw_seed": int(seed),
        "random_draw_columns": list(prepare.DRAW_COLUMNS),
        "dropped_overlapping_columns": list(prepare.OVERLAPPING_COLUMNS),
        # Columns Tax-Data publishes that no part of the model reads, left out
        # of the conversion. The reasons are on `prepare.NEVER_READ_COLUMNS`.
        "dropped_never_read_columns": list(prepare.NEVER_READ_COLUMNS),
        "wage_split": (
            "wages_base<earner> = wages<earner> - tips<earner> - ot<earner>; "
            "the household totals wages, sole_prop, farm, tips and ot are "
            "dropped because they repeat the per-earner amounts"
        ),
        "negative_base_wages_note": (
            "Records whose reported tips and overtime exceed that earner's "
            "wages come through with negative base pay, unchanged."
        ),
        "years": {report.year: report.as_manifest_entry() for report in reports},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="a Tax-Data vintage folder holding the CSVs")
    parser.add_argument("out", help="where the converted years and manifest go")
    parser.add_argument(
        "--years",
        default="",
        help="first:last, if only part of the vintage is wanted; "
        "the whole vintage by default",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=prepare.DEFAULT_SEED,
        help="the run seed the random draws are keyed on. Changing it means "
        "different draws, so a different conversion",
    )
    args = parser.parse_args(argv)

    years = None
    if args.years:
        first, last = (int(part) for part in args.years.split(":"))
        years = range(first, last + 1)

    manifest = convert_vintage(args.source, args.out, years=years, seed=args.seed)
    total = sum(entry["records"] for entry in manifest["years"].values())
    negative = sum(
        entry["records_with_negative_base_wages"] for entry in manifest["years"].values()
    )
    print(
        f"converted {len(manifest['years'])} years, {total:,} record-years, "
        f"{negative} of them with negative base wages"
    )
    print(f"manifest written to {Path(args.out) / prepare.BRIDGE_MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
