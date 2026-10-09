"""Prove the bridge changed the container and not the numbers.

For every year of a converted vintage, build the model's prepared population
twice — once by parsing the Tax-Data text and once by reading the bridge
output — and compare every column of every year. The bridge's whole claim is
that these two are the same data, so the tolerance is zero: a difference of
any size in any column fails the run.

Two checks, because a run does two different things with a year. First, each
year on its own. Second, a whole run window built in one go, which is the
case that pads: a record that exists in some years and not others gets an
empty row in the years it is missing, and padding is what turns a column of
whole numbers into a column with missing values in it.

This reads two 200 MB files per year, so it belongs in a job:

    sbatch jobs/bridge_byte_identity.sbatch <tax-data root> <bridge root>
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings

from taxmodel.frame import population, prepare
from tests.unit.frame import population_checks


def compare_year(
    tax_data_root: str, bridge_root: str, year: int
) -> tuple[list[str], float, float, float]:
    """Every difference between the two ways of building one year's slice.

    Also returns how long each way took to read and how long the comparison
    itself took. The reading times are the point of the exercise — the two
    answers are the same and one of them is much faster to get — and the
    comparison time is reported so a slow run is not mistaken for slow reading.
    """
    # The text path warns about the records with negative base pay every time
    # it reads a year. The manifest already records them; here the warning
    # would just bury the result.
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from_text = prepare.read_tax_data_year_frame(tax_data_root, year)
        # The bridge leaves out the columns nothing in the model reads, so the
        # text side must leave them out too before the two are compared.
        from_text = from_text.drop(
            columns=[c for c in prepare.NEVER_READ_COLUMNS if c in from_text.columns]
        )
        from_text = from_text.merge(
            prepare.keyed_draws(from_text["id"].to_numpy()),
            on="id",
            how="left",
            validate="one_to_one",
        )
    text_seconds = time.perf_counter() - started

    started = time.perf_counter()
    from_bridge = population_checks.read_bridge_year_frame(bridge_root, year)
    # A bridge converted before a column joined the never-read list still
    # holds it; the model skips it on reading, and so does this comparison.
    from_bridge = from_bridge.drop(
        columns=[c for c in prepare.NEVER_READ_COLUMNS if c in from_bridge.columns]
    )
    bridge_seconds = time.perf_counter() - started

    started = time.perf_counter()
    differences = population_checks.frame_differences(from_text, from_bridge)
    compare_seconds = time.perf_counter() - started
    return differences, text_seconds, bridge_seconds, compare_seconds


def compare_window(
    tax_data_root: str, bridge_root: str, years: list[int]
) -> list[str]:
    """Every difference between the two ways of building a whole run window.

    A run spans several years at once, and every record keeps one slot across
    all of them. A record missing from a year gets an empty row there, which
    is the one thing a single-year comparison never exercises.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from_text = population.population_from_tax_data(tax_data_root, years)
    from_bridge = population.population_from_bridge(bridge_root, years)

    return population_checks.population_differences(from_text, from_bridge)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tax_data_root", help="the Tax-Data vintage of CSVs")
    parser.add_argument("bridge_root", help="the bridge output converted from it")
    parser.add_argument(
        "--years",
        default="",
        help="first:last, to check part of the vintage; the whole of it by default",
    )
    parser.add_argument(
        "--window",
        default="2025:2036",
        # The years a baseline run actually spans, which is where padding
        # happens. Set it to an empty string to skip the second check.
        help="first:last, a whole run window to build in one go and compare "
        "slot for slot; the 2025-2036 budget window by default",
    )
    args = parser.parse_args(argv)

    tax_data_root, bridge_root = args.tax_data_root, args.bridge_root
    manifest = prepare.read_bridge_manifest(bridge_root)
    years = sorted(int(y) for y in manifest["years"])
    if args.years:
        first, last = (int(part) for part in args.years.split(":"))
        years = [y for y in years if first <= y <= last]

    print(f"Tax-Data text : {tax_data_root}")
    print(f"bridge output : {bridge_root}")
    print(f"years         : {years[0]}-{years[-1]} ({len(years)} of them)")
    print()

    failed: dict[int, list[str]] = {}
    text_total = bridge_total = 0.0
    for year in years:
        differences, text_seconds, bridge_seconds, compare_seconds = compare_year(
            tax_data_root, bridge_root, year
        )
        text_total += text_seconds
        bridge_total += bridge_seconds
        timing = (
            f"read: text {text_seconds:5.1f}s, bridge {bridge_seconds:5.1f}s; "
            f"compared in {compare_seconds:5.1f}s"
        )
        if differences:
            failed[year] = differences
            print(f"{year}: {len(differences)} DIFFERENCES  ({timing})", flush=True)
            for line in differences:
                print(f"    {line}")
        else:
            print(f"{year}: identical  ({timing})", flush=True)

    print()
    print(
        f"reading the population: {text_total:.0f} s from the Tax-Data text, "
        f"{bridge_total:.0f} s from the bridge, over {len(years)} years "
        f"({text_total / bridge_total:.0f} times faster)"
    )
    window_differences: list[str] = []
    if args.window:
        first, last = (int(part) for part in args.window.split(":"))
        window = [y for y in years if first <= y <= last]
        print()
        print(f"the {window[0]}-{window[-1]} run window, built in one go:", flush=True)
        window_differences = compare_window(tax_data_root, bridge_root, window)
        if window_differences:
            for line in window_differences:
                print(f"    {line}")
        else:
            print(
                f"    identical on all {len(window)} years, "
                "padded slots and all",
                flush=True,
            )

    print()
    if failed or window_differences:
        if failed:
            print(f"FAIL: {len(failed)} of {len(years)} years differ: {sorted(failed)}")
        if window_differences:
            print(f"FAIL: {len(window_differences)} differences over the run window")
        return 1
    print(f"PASS: all {len(years)} years are identical, column for column, byte for byte")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
