"""Compare every published number of two runs, file by file.

Two runs of the same scenarios on the same card do not agree to the last bit:
the order the card adds numbers up in varies between executions, so two runs of
identical code differ in the last digit or two. That is the yardstick this
prints — run it once on two control runs to see what pure run-to-run wobble
looks like, and once on control against a changed model to see whether the
change stayed inside it.

    python jobs/compare_run_outputs.py <run directory> <run directory>

A run directory is the folder holding `scenarios/`. For every published table
under it — CSV or Parquet, and a table that changed container between the two
runs is still compared against its counterpart — the script reports the
largest difference across all numeric cells, measured
against the size of the column it sits in: the gap between the two runs divided
by the largest absolute value that column takes. Scaling by the column and not
by the cell is what keeps the answer meaningful — a distributional table has
cells that are a rounding error away from zero, where a difference of one part
in ten thousand million of a dollar reads as a hundred percent of the cell and
means nothing.

A file that one run wrote and the other did not is listed on its own, not
counted as a problem. Comparing a run against one from before a table was
added or taken away is a normal thing to want — the point of the comparison is
whether the numbers both runs report agree.

The per-household detail files are not compared. They are one row per
household per year, gigabytes apiece, and differencing two of them would take
more memory than producing them did; what this script is for is the published
summary tables.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def largest_relative_difference(left: pd.DataFrame, right: pd.DataFrame):
    """The biggest relative gap between two tables, and where it is."""
    if list(left.columns) != list(right.columns):
        return None, "different columns"
    if len(left) != len(right):
        return None, f"different row counts: {len(left)} and {len(right)}"
    worst = 0.0
    worst_where = ""
    for column in left.columns:
        one = left[column]
        other = right[column]
        if not (
            pd.api.types.is_numeric_dtype(one) and pd.api.types.is_numeric_dtype(other)
        ):
            if not one.equals(other):
                return None, f"text column '{column}' differs"
            continue
        one = np.nan_to_num(one.to_numpy(dtype=float), nan=0.0)
        other = np.nan_to_num(other.to_numpy(dtype=float), nan=0.0)
        gap = np.abs(one - other)
        scale = max(np.abs(one).max(initial=0.0), np.abs(other).max(initial=0.0))
        if scale == 0.0:
            continue
        relative = gap / scale
        if relative.size and relative.max() > worst:
            worst = float(relative.max())
            worst_where = f"{column} row {int(relative.argmax())}"
    return worst, worst_where


def read_table(path: Path) -> pd.DataFrame:
    """Read one published table, whichever container it is in."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


# The per-household files a scenario writes when the runscript asked for
# detail. One holds a row per household per year and can run to gigabytes, and
# reading two of them into memory to difference them would need more memory
# than the run that produced them. What this script is for is the published
# summary tables, so the detail folder is left alone.
DETAIL_FOLDER = "detail"


def published_tables(run: Path) -> list[Path]:
    """Every summary table a run wrote, in either container, run-relative."""
    return sorted(
        path.relative_to(run)
        for pattern in ("*.csv", "*.parquet")
        for path in run.rglob(pattern)
        if DETAIL_FOLDER not in path.relative_to(run).parts
    )


def counterpart(name: Path, other_run: Path) -> Path | None:
    """The same table in the other run, allowing for a change of container.

    A table that used to be a CSV and is now a Parquet file holds the same
    columns, the same rows and the same numbers, so comparing an old run
    against a new one has to look past the suffix. Same folder, same stem,
    either container.
    """
    for suffix in (name.suffix, ".csv", ".parquet"):
        candidate = other_run / name.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def main(first: Path, second: Path) -> None:
    files = published_tables(first)
    second_files = published_tables(second)
    only_one_side = [
        f"{name}: only the first run wrote it"
        for name in files
        if counterpart(name, second) is None
    ] + [
        f"{name}: only the second run wrote it"
        for name in second_files
        if counterpart(name, first) is None
    ]
    overall = 0.0
    overall_where = ""
    trouble = []
    for name in files:
        other = counterpart(name, second)
        if other is None:
            continue
        left = read_table(first / name)
        right = read_table(other)
        worst, where = largest_relative_difference(left, right)
        if worst is None:
            trouble.append(f"{name}: {where}")
            continue
        if worst > overall:
            overall = worst
            overall_where = f"{name} ({where})"
        print(f"  {worst:.3e}  {name}")
    print(f"\nlargest relative difference anywhere: {overall:.3e}")
    if overall_where:
        print(f"  worst cell: {overall_where}")
    for line in only_one_side:
        print(f"  only in one run: {line}")
    for line in trouble:
        print(f"  PROBLEM {line}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
