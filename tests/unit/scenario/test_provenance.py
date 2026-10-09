"""The imperative shell: reading law and reading the estate bridge."""

import numpy as np
import pytest
import yaml

from taxmodel.scenario import provenance
from tests.conftest import COMPILED_BASELINE


def test_reading_the_frozen_law_gives_the_same_arrays_the_compiler_wrote():
    values, years = provenance.read_compiled_npz(COMPILED_BASELINE)
    assert years[0] == 2025 and years[-1] == 2036
    # [year, filing status, slot], and sixteen slots is the compiler's cap.
    assert values["ord.rates"].shape == (12, 4, 16)


def test_the_estate_measurement_bins_are_read_in_size_order(tmp_path):
    # The bridge file may list its size bins in any order; the estate calculator reads the four arrays positionally, so
    # they have to come back sorted by the bin's lower bound.
    bridge = tmp_path / "bridge.yaml"
    bridge.write_text(
        yaml.safe_dump(
            {
                "r": 0.951,
                "rho_pt": 0.612,
                "gamma": 0.09,
                "cluster_death_weight_cap": 300.0,
                "bins": [
                    {"lo": 1e7, "f_ded": 0.05, "p_dsue": 0.11, "f_dsue": 0.16},
                    {"lo": 0.0, "f_ded": 0.06, "p_dsue": 0.0, "f_dsue": 0.0},
                    {"lo": 5e7, "f_ded": 0.35, "p_dsue": 0.0, "f_dsue": 0.0},
                    {"lo": 2e7, "f_ded": 0.13, "p_dsue": 0.17, "f_dsue": 0.16},
                ],
            }
        )
    )

    params, cluster_cap = provenance.read_estate_bridge(bridge)

    np.testing.assert_allclose(
        np.asarray(params.bin_lower_bounds), [0.0, 1e7, 2e7, 5e7]
    )
    np.testing.assert_allclose(
        np.asarray(params.deduction_shares), [0.06, 0.05, 0.13, 0.35]
    )
    assert float(params.reporting_rate) == pytest.approx(0.951)
    assert cluster_cap == 300.0
