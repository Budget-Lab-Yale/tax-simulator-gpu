"""The capital-gains model's household inputs, read from the population data.

The age cells the gains model runs on (the stock of unrealized gain, the gains
realized under baseline law, mortality, and the chance an estate goes to
charity) describe the population as Tax-Data reports it, before any year is
simulated.

This module builds one frame of households per run year from the population
block, carrying the columns the cell aggregation and the rate averaging read.
The per-record arithmetic is the same as
``frame/steps_jax.add_capital_gains_record_attributes`` inside the compiled
program:

- **the stock of unrealized gain** is market value less cost basis over the
  five tracked asset classes, each class floored at zero, a missing pair
  counting as nothing;
- **the chance the household dies this year** is the chance both spouses die
  on a joint return, and the filer's own chance otherwise, scaled down in
  clusters of replicated donor records by the estate tax's cap on their
  expected deaths (``frame/steps_jax.calc_estate_mortality``);
- **the chance of being widowed at death** comes from the age-and-sex table
  and is zero for anyone married, divorced, or filing other than singly;
- **the gross estate** the charitable-bequest logit reads is the sum of the
  balance-sheet asset columns.

Nothing here depends on a simulated frame, so the gains recurrence can run
before any year is simulated.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np

from taxmodel.frame import widowhood
from taxmodel.frame.build_frame import DONOR_CLUSTER_GROSS_ESTATE_FLOOR
from taxmodel.frame.population import PopulationSource
from taxmodel.frame.steps_jax import (
    CAPITAL_GAINS_BASIS_COLS,
    CAPITAL_GAINS_VALUE_COLS,
    ESTATE_ASSET_COLS,
    FILING_JOINT,
    calc_estate_mortality,
)

# The Tax-Data columns this module reads off the population block. A missing
# column is an error.
POPULATION_COLUMNS: tuple[str, ...] = (
    "weight",
    "kg_lt",
    "age1",
    "age2",
    "filing_status",
    "male1",
    "divorce_year",
    "q_death1",
    "q_death2",
    *CAPITAL_GAINS_VALUE_COLS,
    *CAPITAL_GAINS_BASIS_COLS,
    *(name for name in ESTATE_ASSET_COLS if name not in CAPITAL_GAINS_VALUE_COLS),
)


def host_records_by_year(
    population: PopulationSource,
    years: Sequence[int],
    cluster_death_weight_cap: float,
    shares: widowhood.WidowhoodShares | None = None,
) -> list[dict[str, np.ndarray]]:
    """One frame of households per run year, from the population block alone.

    ``cluster_death_weight_cap`` is the estate bridge's cap on the expected
    deaths in a cluster of replicated donor records.

    Each frame carries the columns the gains model's cell aggregation and rate
    averaging read (``behavior/kg/prepass.py``): the identifying columns, the
    three derived facts described in the module header, and the value and
    basis of the five tracked asset classes. Households the year's validity
    mask excludes carry a weight of zero, so they add nothing to any cell.
    """
    missing = sorted(set(POPULATION_COLUMNS) - set(population.field_names))
    if missing:
        raise ValueError(
            "the population block is missing columns the capital-gains model "
            "reads: " + ", ".join(missing)
        )
    if shares is None:
        shares = widowhood.read_widowhood_shares()

    frames: list[dict[str, np.ndarray]] = []
    for year in years:
        # One year at a time, and only the columns below, so that a run whose
        # population is read from disk does not hold every year's households.
        one_year = population.year(int(year), columns=POPULATION_COLUMNS)
        valid = np.asarray(one_year.valid, dtype=bool)

        def column(name: str, one_year=one_year) -> np.ndarray:
            return one_year.values_as_float(name)

        values = np.stack([column(name) for name in CAPITAL_GAINS_VALUE_COLS], axis=1)
        bases = np.stack([column(name) for name in CAPITAL_GAINS_BASIS_COLS], axis=1)
        gain_by_class = np.nan_to_num(values - bases, nan=0.0)
        gain_stock = np.maximum(gain_by_class, 0.0).sum(axis=1)

        filing_status = column("filing_status")
        own_death = column("q_death1")
        spouse_death = column("q_death2")
        both_die = (filing_status == FILING_JOINT) & ~np.isnan(spouse_death)
        mortality = np.nan_to_num(
            np.where(both_die, own_death * spouse_death, own_death), nan=0.0
        )
        estate_columns = {
            name: jnp.asarray(column(name))
            for name in ("weight", "filing_status", "q_death1", "q_death2")
            + ESTATE_ASSET_COLS
        }
        donor_cluster_scale = np.asarray(
            calc_estate_mortality(
                estate_columns,
                jnp.asarray(float(cluster_death_weight_cap)),
                jnp.asarray(DONOR_CLUSTER_GROSS_ESTATE_FLOOR),
                jnp.asarray(valid),
            )["donor_cluster_scale"]
        )
        mortality = np.where(valid, mortality * donor_cluster_scale, 0.0)

        age = column("age1")
        age = np.where(np.isnan(age), shares.band_lower_bounds[0], age)
        is_male = np.nan_to_num(column("male1"), nan=0.0) == 1
        unmarried = np.isin(filing_status, widowhood.UNMARRIED_FILING_STATUSES)
        never_divorced = np.isnan(column("divorce_year"))
        p_widow = np.where(
            unmarried & never_divorced, shares.for_age_and_sex(age, is_male), 0.0
        )

        estate_gross = np.nansum(
            np.stack([column(name) for name in ESTATE_ASSET_COLS], axis=1), axis=1
        )

        frame: dict[str, np.ndarray] = {
            "weight": np.where(valid, column("weight"), 0.0),
            "kg_lt": np.nan_to_num(column("kg_lt"), nan=0.0),
            "age1": age,
            "age2": np.nan_to_num(column("age2"), nan=0.0),
            "filing_status": filing_status,
            "m_household": mortality,
            "p_widow": p_widow,
            "unrealized_gain_stock": gain_stock,
            "estate_economic_gross": estate_gross,
        }
        for name in (*CAPITAL_GAINS_VALUE_COLS, *CAPITAL_GAINS_BASIS_COLS):
            frame[name] = column(name)
        frames.append(frame)
    return frames


__all__ = ["POPULATION_COLUMNS", "host_records_by_year"]
