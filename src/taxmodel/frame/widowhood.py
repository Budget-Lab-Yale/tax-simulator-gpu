"""The probability that an unmarried filer is a widow or widower.

When gains are taxable at death, the law gives a married couple twice the
exclusion a single person gets, and a surviving spouse can generally claim
the larger amount through a survivor election. Tax-Data does not record
widowhood, so the model uses the share of unmarried people at each age who
are widowed rather than never married, separated or divorced, and blends the
single and the married exclusion by that share. It is an upper bound on
eligibility, not a measurement of who elects.

The shares come from the 2022 Survey of Consumer Finances and are stored in
`config/calibrations/estate/widowhood.yaml`. They describe the population,
not the tax law, so a reform does not change them.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import yaml

# The calibration file, read from a fixed path.
DEFAULT_WIDOWHOOD_SHARES = Path("config/calibrations/estate/widowhood.yaml")

# Only a single filer or a head of household can be a surviving spouse in this
# sense. A joint return has a living spouse by definition, and so does a
# married filer filing separately (status 3).
UNMARRIED_FILING_STATUSES = (1, 4)


@dataclass(frozen=True)
class WidowhoodShares:
    """The share widowed, by age band and sex.

    `band_lower_bounds` is ascending, so the band a person falls in is the
    last bound at or below their age. Anyone below the first bound takes the
    first band's share.
    """

    band_lower_bounds: np.ndarray
    share_widowed_male: np.ndarray
    share_widowed_female: np.ndarray

    def for_age_and_sex(self, age: np.ndarray, is_male: np.ndarray) -> np.ndarray:
        """The share widowed for each person's age band and sex."""
        band = np.searchsorted(self.band_lower_bounds, age, side="right")
        band = np.clip(band - 1, 0, len(self.band_lower_bounds) - 1)
        return np.where(
            is_male, self.share_widowed_male[band], self.share_widowed_female[band]
        )


@cache
def read_widowhood_shares(
    path: str | Path = DEFAULT_WIDOWHOOD_SHARES,
) -> WidowhoodShares:
    """Read the calibrated shares, checking that the bands make sense."""
    bands = yaml.safe_load(Path(path).read_text())["bands"]
    lower = np.asarray(bands["lo"], dtype=np.float64)
    male = np.asarray(bands["p_widow_male"], dtype=np.float64)
    female = np.asarray(bands["p_widow_female"], dtype=np.float64)

    if not (len(lower) == len(male) == len(female)) or len(lower) == 0:
        raise ValueError(
            f"{path}: the age bands and the two share lists differ in length"
        )
    if np.any(np.diff(lower) <= 0):
        raise ValueError(
            f"{path}: the age band lower bounds are not in ascending order"
        )
    shares = np.concatenate([male, female])
    if np.any(~np.isfinite(shares)) or np.any(shares < 0) or np.any(shares > 1):
        raise ValueError(f"{path}: a widowhood share is missing or outside 0 to 1")

    return WidowhoodShares(
        band_lower_bounds=lower,
        share_widowed_male=male,
        share_widowed_female=female,
    )
