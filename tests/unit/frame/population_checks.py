"""Reading a population back as pandas frames and comparing two of them, for tests.

The model reads a population through `Population.year` and the bridge's
Parquet files through `prepare.read_bridge_year_arrays`. These helpers lay
the same data out as pandas frames and list every stored difference, so the
unit tests and the bridge check in `tests/bridge_byte_identity.py` can
compare two ways of building the same households.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from taxmodel.frame import prepare
from taxmodel.frame.population import Population


def year_frame(population: Population, year: int) -> pd.DataFrame:
    """One year in the population's field order, with `valid` appended last.

    There are three kinds of source column. `year` is not stored because
    every slot has the same calendar year. A random draw is stored once per
    slot and is zeroed for a slot that holds no record this year. Every other
    field comes from that year's row in `values`, with whole-number columns
    widened to float64 and their missing values as NaN. The record id is the
    stable slot identity.
    """
    year_row = population.universe.year_index(year)
    valid = population.universe.valid[year_row]
    columns: dict[str, np.ndarray] = {}
    for name in population.field_names:
        if name == "id":
            columns[name] = population.universe.ids
        elif name == "year":
            columns[name] = np.full(
                population.universe.n_slots, int(year), dtype=np.int64
            )
        elif name in population.draws:
            columns[name] = np.where(valid, population.draws[name], 0.0)
        else:
            stored = population.values[name][year_row]
            if prepare.is_narrow_whole_number_column(name):
                stored = prepare.widen_whole_numbers(stored)
            columns[name] = stored
    columns["valid"] = valid
    return pd.DataFrame(columns)


def draws_frame(population: Population) -> pd.DataFrame:
    """The run-level record ids and their random draws in slot order."""
    return pd.DataFrame({"id": population.universe.ids, **population.draws})


def read_bridge_year_frame(
    bridge_root: str | Path, year: int, ids: np.ndarray | None = None
) -> pd.DataFrame:
    """One converted year as a pandas frame, optionally limited to named records."""
    frame = pd.read_parquet(prepare.bridge_year_path(bridge_root, year))
    frame["id"] = frame["id"].astype(np.int64)
    if ids is not None:
        frame = frame[frame["id"].isin(np.asarray(ids, dtype=np.int64))]
    return frame.reset_index(drop=True)


def population_differences(left: Population, right: Population) -> list[str]:
    """Every stored difference between two populations; empty means identical."""
    differences: list[str] = []
    if left.universe.years != right.universe.years:
        differences.append("the two populations do not cover the same years")
    if left.universe.ids.tobytes() != right.universe.ids.tobytes():
        differences.append("the two populations do not hold the same record ids")
    if left.universe.valid.tobytes() != right.universe.valid.tobytes():
        differences.append("the two populations disagree about record validity")
    if left.field_names != right.field_names:
        differences.append("the two populations do not hold fields in the same order")
    if left.dropped_text_columns != right.dropped_text_columns:
        differences.append("the two populations did not drop the same text columns")
    for name in left.values.keys() | right.values.keys():
        if name not in left.values or name not in right.values:
            differences.append(f"{name}: stored by only one population")
        elif left.values[name].tobytes() != right.values[name].tobytes():
            differences.append(f"{name}: the stored numbers are not the same bytes")
    for name in left.draws.keys() | right.draws.keys():
        if name not in left.draws or name not in right.draws:
            differences.append(f"{name}: stored by only one population")
        elif left.draws[name].tobytes() != right.draws[name].tobytes():
            differences.append(f"{name}: the stored draws are not the same bytes")
    return differences


def frame_differences(parsed: pd.DataFrame, converted: pd.DataFrame) -> list[str]:
    """Every way two frames differ, in plain sentences; empty means identical.

    Numeric columns are compared as stored bytes, so a negative zero differs
    from a zero and two missing values are equal. Text and true/false columns
    are compared as missing in the same places plus equal everywhere present.
    """
    if list(parsed.columns) != list(converted.columns):
        only_parsed = [c for c in parsed.columns if c not in set(converted.columns)]
        only_converted = [c for c in converted.columns if c not in set(parsed.columns)]
        if only_parsed or only_converted:
            return [
                (
                    "the two frames do not hold the same columns: "
                    f"only in the parsed frame {only_parsed}, "
                    f"only in the converted frame {only_converted}"
                )
            ]
        return ["the two frames hold the same columns in a different order"]
    if len(parsed) != len(converted):
        return [f"{len(parsed)} rows parsed against {len(converted)} rows converted"]

    differences: list[str] = []
    for name in parsed.columns:
        left, right = parsed[name], converted[name]
        if left.dtype != right.dtype:
            differences.append(
                f"{name}: stored as {left.dtype} parsed and {right.dtype} converted"
            )
            continue
        if pd.api.types.is_numeric_dtype(left) or pd.api.types.is_bool_dtype(left):
            if left.to_numpy().tobytes() != right.to_numpy().tobytes():
                differences.append(f"{name}: the stored numbers are not the same bytes")
            continue
        left_missing, right_missing = left.isna().to_numpy(), right.isna().to_numpy()
        if not (left_missing == right_missing).all():
            differences.append(f"{name}: missing in different rows")
            continue
        present = ~left_missing
        if not (left[present].to_numpy() == right[present].to_numpy()).all():
            differences.append(f"{name}: the values present differ")
    return differences
