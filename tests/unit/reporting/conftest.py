"""A ten-household population small enough to check every reported number by hand.

The same ten households are used by the revenue tests and the distribution
tests, so that a number can be traced from one table to the other. Everything
about them is written out in full below; nothing is generated.

The households, in slot order:

    slot  weight  age  child<18  expanded income  AGI      net worth
    0        10    25    no             -5,000    -5,000     -20,000
    1        10    28    no                  0         0           0
    2        10    35    yes            20,000    18,000      10,000
    3        10    42    yes            40,000    36,000      75,000
    4        10    47    no             60,000    55,000     150,000
    5        10    55    yes            90,000    85,000     300,000
    6        10    60    no            130,000   120,000     600,000
    7        10    64    no            200,000   185,000   1,200,000
    8        10    70    no            400,000   370,000   3,000,000
    9        10    80    no          2,000,000 1,900,000  40,000,000

Every household carries the same weight of ten, which makes the weighted
quantiles easy to follow: nine households have income of zero or more, so each
of them is one ninth of the ranked population.
"""

from __future__ import annotations

import numpy as np
import pytest

from taxmodel.reporting import estate_allocator_jax

# Ten households, each standing for ten in the population.
WEIGHT = np.full(10, 10.0)

AGE = np.array([25, 28, 35, 42, 47, 55, 60, 64, 70, 80], dtype=float)

HAS_CHILD_UNDER_18 = np.array(
    [False, False, True, True, False, True, False, False, False, False]
)

EXPANDED_INCOME = np.array(
    [-5_000, 0, 20_000, 40_000, 60_000, 90_000, 130_000, 200_000, 400_000, 2_000_000],
    dtype=float,
)

AGI = np.array(
    [-5_000, 0, 18_000, 36_000, 55_000, 85_000, 120_000, 185_000, 370_000, 1_900_000],
    dtype=float,
)

NET_WORTH = np.array(
    [
        -20_000,
        0,
        10_000,
        75_000,
        150_000,
        300_000,
        600_000,
        1_200_000,
        3_000_000,
        40_000_000,
    ],
    dtype=float,
)

VALID = np.ones(10, dtype=bool)


@pytest.fixture
def households() -> dict[str, np.ndarray]:
    """The ten households, as the reporting layer wants them."""
    return {
        "weight": WEIGHT,
        "valid": VALID,
        "age": AGE,
        "has_child_under_18": HAS_CHILD_UNDER_18,
        "income": EXPANDED_INCOME,
        "agi": AGI,
        "net_worth": NET_WORTH,
    }


def named(diagnostics) -> dict[str, float]:
    """The diagnostics array read back under the names it reports."""
    return dict(
        zip(
            estate_allocator_jax.DIAGNOSTIC_NAMES,
            np.asarray(diagnostics, dtype=np.float64),
            strict=True,
        )
    )
