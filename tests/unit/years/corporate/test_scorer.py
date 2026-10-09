"""What a corporate statutory rate change scores, checked against hand values."""

import numpy as np
import pytest

from taxmodel.years.corporate.scorer import (
    corporate_rate_revenue_change,
    corporate_rate_score_by_world,
    rate_held_flat_past_the_window,
)
from tests.tolerances import CENT

# The current top corporate statutory rate, from
# config/scenarios/tax_law/default/corp.yaml (21 percent since 2018).
BASELINE_CORPORATE_RATE = 0.21
# A 28 percent rate, the reform in config/scenarios/tax_law/alternatives/tests/
# corp_rate_28.
REFORM_CORPORATE_RATE = 0.28
# Elasticity of the corporate tax base to the net-of-tax statutory rate, from
# config/scenarios/economy/default/corp.yaml (Coles, Patel, Seegert and Smith
# 2022, the avoidance margin re-based onto the statutory rate).
BASE_ELASTICITY = 0.367
# CBO's projected corporate receipts for fiscal 2026, $404.0 billion, the
# figure named in reporting/receipts.py. Money here is in dollars.
CBO_CORPORATE_RECEIPTS = 404.0e9


def test_mechanical_score_is_the_rate_change_times_the_baseline_base():
    """Raising 21 percent to 28 with the base held fixed raises receipts by a third."""
    scored = corporate_rate_revenue_change(
        np.array([BASELINE_CORPORATE_RATE]),
        np.array([REFORM_CORPORATE_RATE]),
        np.array([CBO_CORPORATE_RECEIPTS]),
        base_erodes=False,
        base_elasticity=BASE_ELASTICITY,
    )
    # 404.0e9 * (0.28 / 0.21 - 1).
    assert scored[0] == pytest.approx(134_666_666_666.667, abs=CENT)


def test_base_erosion_shaves_the_score():
    """Letting profit shift abroad cuts the same rate rise by about $18 billion."""
    scored = corporate_rate_revenue_change(
        np.array([BASELINE_CORPORATE_RATE]),
        np.array([REFORM_CORPORATE_RATE]),
        np.array([CBO_CORPORATE_RECEIPTS]),
        base_erodes=True,
        base_elasticity=BASE_ELASTICITY,
    )
    # 404.0e9 * ((0.28 / 0.21) * (0.72 / 0.79) ** 0.367 - 1).
    assert scored[0] == pytest.approx(116_633_351_576.773, abs=CENT)


def test_a_rate_that_does_not_move_scores_nothing():
    """Same rate in both worlds, eroding base or not: no revenue change."""
    rate = np.full(3, BASELINE_CORPORATE_RATE)
    receipts = np.full(3, CBO_CORPORATE_RECEIPTS)
    for base_erodes in (False, True):
        scored = corporate_rate_revenue_change(
            rate,
            rate,
            receipts,
            base_erodes=base_erodes,
            base_elasticity=BASE_ELASTICITY,
        )
        assert np.all(scored == 0.0)


def test_a_year_with_no_rate_scores_nothing():
    """Years the rate series does not cover are left alone, not scored as a cut."""
    scored = corporate_rate_revenue_change(
        np.array([BASELINE_CORPORATE_RATE, np.nan]),
        np.array([REFORM_CORPORATE_RATE, REFORM_CORPORATE_RATE]),
        np.full(2, CBO_CORPORATE_RECEIPTS),
        base_erodes=False,
        base_elasticity=BASE_ELASTICITY,
    )
    assert scored[1] == 0.0
    assert scored[0] > 0.0


def test_only_the_conventional_world_erodes_the_base():
    """The baseline world scores zero; the mechanical worlds hold the base fixed."""
    worlds = (
        "baseline",
        "first_order",
        "mechanical_ex_saving",
        "mechanical",
        "conventional",
    )
    scores = corporate_rate_score_by_world(
        worlds,
        np.array([BASELINE_CORPORATE_RATE]),
        np.array([REFORM_CORPORATE_RATE]),
        np.array([CBO_CORPORATE_RECEIPTS]),
        base_elasticity=BASE_ELASTICITY,
    )
    assert scores["baseline"][0] == 0.0
    for world in ("first_order", "mechanical_ex_saving", "mechanical"):
        assert scores[world][0] == pytest.approx(134_666_666_666.667, abs=CENT)
    assert scores["conventional"][0] == pytest.approx(116_633_351_576.773, abs=CENT)


def test_the_rate_is_held_flat_past_the_last_simulated_year():
    """Before the rate series there is no rate; after it, the last one carries on."""
    held = rate_held_flat_past_the_window(
        years=np.array([2024, 2026, 2027, 2035]),
        rate_years=np.array([2026, 2027]),
        rate=np.array([BASELINE_CORPORATE_RATE, REFORM_CORPORATE_RATE]),
    )
    assert np.isnan(held[0])
    assert held[1] == BASELINE_CORPORATE_RATE
    assert held[2] == REFORM_CORPORATE_RATE
    assert held[3] == REFORM_CORPORATE_RATE
