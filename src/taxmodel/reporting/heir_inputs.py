"""The heir structure read from the upstream publication.

The estate tax is paid by an estate, but the people made worse off are the
heirs. To say which households those are, the reporting layer needs two
numbers per household per year, neither of which this model computes:

    probability_of_inheriting   the chance the household receives an
                                inheritance during the year
    inheritance                 how much it would receive, measured before any
                                estate tax is taken out

Both come from the Estate-Tax-Distribution model, which publishes one file per
year, `estate_tax_detail_{year}.csv`, under the same
`<root>/model_data/<Interface>/v<version>/<vintage>/<scenario id>/` convention
every other upstream publication this model reads uses. The vintage and
scenario id are pinned in the economy leg's `interfaces.yaml`, so the run
manifest already records which publication was read. This module only opens it.

The file also carries the upstream model's own `estate_tax_liability`, which
is not read, because this model computes the estate tax on its own microdata
and `reporting/estate_allocator_jax.py` spreads that figure over these heirs.
The heir structure does not vary by scenario, because what a decedent leaves
is measured before tax.

*Records the file does not mention are treated as non-heirs.* An id can be new
to a later Tax-Data vintage than the one the heir structure was built on.
Those records get zeros, and the count is reported.

*The published `p_inheritance` is not always a probability.* A few records
carry a value above one, which reads as an expected number of inheritances
rather than a chance of receiving one. The weight is multiplied by it
regardless, so the household's non-heir copy carries a negative weight and is
dropped, and the household appears in the distribution table only as an heir,
at more than its own weight. This module passes the published number through
unchanged; `reporting/distribution_jax.py` makes the copies.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from taxmodel.config.interfaces import DEFAULT_CONFIG_ROOT, interface_root

Array = np.ndarray

INTERFACE = "Estate-Tax-Distribution"

# The columns read out of each published file, under the file's own names.
SOURCE_COLUMNS = ("p_inheritance", "inheritance")


class HeirInputError(Exception):
    """The published heir structure cannot serve the years this run reports."""


@dataclass(frozen=True)
class HeirStructure:
    """The heir structure for one run's years, laid onto the run's record slots.

    `probability_of_inheriting` and `inheritance` are both shaped
    [year, slot], in the order `years` gives the years and in the population
    block's own slot order. A slot the published file does not mention holds
    zero in both.

    `files_read` names every file that was opened, and `unmatched_by_year` says
    how many of the run's records each year's file did not mention.

    `zero_filled_years` names the years before the publication starts, which
    are filled with zeros: nobody inherits anything in them.
    `carried_forward_years` maps a year past the publication's end to the year
    whose file was used for it. See `read_heir_structure` for both.
    """

    years: tuple[int, ...]
    probability_of_inheriting: Array
    inheritance: Array
    files_read: tuple[str, ...]
    unmatched_by_year: dict[int, int]
    zero_filled_years: tuple[int, ...] = ()
    carried_forward_years: dict[int, int] = field(default_factory=dict)


def year_file(root: str | Path, year: int) -> Path:
    """The published file holding one year's heir structure."""
    return Path(root) / f"estate_tax_detail_{int(year)}.csv"


def read_heir_structure(
    root: str | Path, years: Sequence[int], ids: Array
) -> HeirStructure:
    """Read the published heir structure for the run's years and record ids.

    `ids` is the population block's record identifier for every slot, in slot
    order. What comes back is one row per run year and one column per slot.

    A run can ask for years the publication does not cover, and which side of
    the publication the missing year falls on decides what happens.

    *Before it starts*, the year is filled with zeros: nobody inherits anything
    that year. This understates the estate-linked gains at the start of the
    window. `zero_filled_years` on the
    result says which years were filled.

    *After it ends*, the last published year's file is used again, and the run
    warns. The carry-forward holds the inheritance flow flat in nominal
    dollars, so a carried year understates inheritances by roughly a year of
    growth, and it holds the same households as heirs. The fix is the upstream
    publication covering the whole window.
    `carried_forward_years` says which years were copied and from where.
    """
    years = tuple(int(year) for year in years)
    ids = np.asarray(ids, dtype=np.int64)
    slot_of_id = pd.Index(ids)
    if not slot_of_id.is_unique:
        raise HeirInputError(
            "the population block holds the same record id in two slots, so "
            "the heir structure cannot be joined to it"
        )

    last_published = last_published_year(root)
    missing = tuple(year for year in years if not year_file(root, year).is_file())
    # A missing year past the publication's end reuses the last year it has;
    # a missing year before it starts is filled with zeros.
    carried_forward = {
        year: last_published
        for year in missing
        if last_published is not None and year > last_published
    }
    zero_filled = tuple(year for year in missing if year not in carried_forward)
    if carried_forward:
        copied = ", ".join(str(year) for year in sorted(carried_forward))
        warnings.warn(
            f"the {INTERFACE} publication ends in {last_published}, so who "
            f"inherits in {copied} is a copy of {last_published}: the same "
            "households inherit the same nominal dollars. That understates "
            "inheritances by about a year of growth per year copied, and the "
            "tax at death is spread over the wrong ladder by that much. The "
            "fix is upstream",
            stacklevel=2,
        )

    probability = np.zeros((len(years), len(ids)), dtype=np.float64)
    inheritance = np.zeros((len(years), len(ids)), dtype=np.float64)
    files_read: list[str] = []
    unmatched: dict[int, int] = {}

    for row, year in enumerate(years):
        if year in zero_filled:
            # Nobody inherits this year. The two arrays keep the zeros they
            # were created with.
            unmatched[year] = len(ids)
            continue
        path = year_file(root, carried_forward.get(year, year))
        files_read.append(str(path))
        published = pd.read_csv(path, usecols=["id", *SOURCE_COLUMNS])
        if published["id"].duplicated().any():
            raise HeirInputError(f"{path} names the same record id twice")
        slots = slot_of_id.get_indexer(published["id"].to_numpy(dtype=np.int64))
        mentioned = slots >= 0
        probability[row, slots[mentioned]] = published["p_inheritance"].to_numpy(
            dtype=np.float64
        )[mentioned]
        inheritance[row, slots[mentioned]] = published["inheritance"].to_numpy(
            dtype=np.float64
        )[mentioned]
        # Every slot the file did not mention keeps its zero.
        unmatched[year] = int(len(ids) - int(np.sum(mentioned)))

    return HeirStructure(
        years=years,
        probability_of_inheriting=probability,
        inheritance=inheritance,
        files_read=tuple(files_read),
        unmatched_by_year=unmatched,
        zero_filled_years=zero_filled,
        carried_forward_years=carried_forward,
    )


def last_published_year(root: str | Path) -> int | None:
    """The most recent year the publication has a file for, or None if it has none."""
    published = []
    for path in Path(root).glob("estate_tax_detail_*.csv"):
        stem = path.stem.rsplit("_", 1)[-1]
        if stem.isdigit():
            published.append(int(stem))
    return max(published) if published else None


def load_heir_structure(
    years: Sequence[int],
    ids: Array,
    economy: str | None = None,
    config_root: str | Path = DEFAULT_CONFIG_ROOT,
) -> HeirStructure | None:
    """Open the publication this scenario pins, or say there is none.

    `None` comes back when the pinned publication folder does not exist at all.
    That is the case a test population or a machine without the shared data
    tree is in, and the distribution table then reports only its
    death-exclusive tier.
    """
    root = interface_root(INTERFACE, economy=economy, config_root=config_root)
    if not Path(root).is_dir():
        return None
    return read_heir_structure(root, years, ids)


def describe(structure: HeirStructure | None) -> str:
    """One line a run can print saying what heir structure it is using."""
    if structure is None:
        return (
            "no Estate-Tax-Distribution data was found, so the distribution "
            "table reports its death-exclusive tier only"
        )
    unmatched = max(structure.unmatched_by_year.values(), default=0)
    if not structure.files_read:
        return (
            f"the {INTERFACE} publication covers none of the run's years, so "
            "nobody inherits anything in any of them"
        )
    line = (
        f"heir structure read from {len(structure.files_read)} "
        f"{INTERFACE} files, {structure.files_read[0]} first; at most "
        f"{unmatched:,} of the run's records were not mentioned in a year and "
        "are treated as non-heirs"
    )
    if structure.zero_filled_years:
        # Reported every run, because a year in which nobody inherits is a
        # hole in the numbers.
        years = ", ".join(str(year) for year in structure.zero_filled_years)
        line += (
            f". The publication has no file for {years}, so nobody inherits "
            "anything in those years"
        )
    if structure.carried_forward_years:
        source = max(structure.carried_forward_years.values())
        years = ", ".join(str(year) for year in sorted(structure.carried_forward_years))
        line += (
            f". The publication ends in {source}, so {years} reuse that year's "
            "file unchanged"
        )
    return line
