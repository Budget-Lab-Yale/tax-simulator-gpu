"""The index table derived from Macro-Projections matches a stored copy.

`tests/golden/fixtures/law/indexes.csv` holds the growth rates for the
Macro-Projections publication this run is pinned to. A change to the
derivation, or a silent change in the publication, shows up as a mismatch.

These tests read the real Macro-Projections publication under the shared
project root, so they skip where it is not mounted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from taxmodel.law.indexes import derive_index_table, index_table_from_frame
from tests.conftest import INDEX_TABLE


def test_the_pinned_macro_publication_is_the_one_the_golden_came_from(macro_root):
    """The interface pin still points at the publication named in the pin file."""
    assert macro_root.parts[-3:] == ("v3", "2026022522", "baseline")


def test_derived_growth_rates_equal_the_stored_golden(macro_root):
    derived = derive_index_table(macro_root)
    golden = pd.read_csv(INDEX_TABLE)

    paired = derived.merge(
        golden, on=["series", "year"], how="outer", suffixes=("_derived", "_golden"),
        indicator=True,
    )
    assert set(paired["_merge"]) == {"both"}, "series/year rows differ from the golden"

    # Both sides are the same arithmetic on the same published levels, so the
    # only difference allowed is the last bit or two lost writing the golden out
    # as text and reading it back.
    np.testing.assert_allclose(
        paired["growth_derived"].to_numpy(dtype=float),
        paired["growth_golden"].to_numpy(dtype=float),
        rtol=1e-12,
        atol=0.0,
        equal_nan=True,
    )


def test_the_compiler_sees_the_same_series_as_the_golden(macro_root):
    """What reaches the law compiler: three series, no year with a missing rate."""
    table = index_table_from_frame(derive_index_table(macro_root))
    assert set(table.series_names) == {"cpi", "chained_cpi", "awi"}
    # The capital gains basis chain runs on CPI back to 1914, the first year the
    # pre-1970 historical file can give a growth rate for. The average wage index
    # starts with the macro publication's own first year, 1970, so its first
    # growth rate is 1971. Chained CPI-U is published only from 2012 on, which is
    # early enough: the TCJA made it the indexing measure for 2018 forward.
    assert min(table.growth["cpi"]) == 1914
    assert min(table.growth["awi"]) == 1971
    assert min(table.growth["chained_cpi"]) == 2013
