"""Scoring a corporate statutory rate change on the way into a run's tables.

`tests/unit/years/corporate/test_scorer.py` checks the base-elasticity
arithmetic on its own. This file checks the wiring the arithmetic sits in: that
run preparation reads the two laws' corporate rates, calls the scorer, and hands
the answer to the place that reports it, the published corporate revenue line.
It also checks the national corporate figures run preparation builds for the
distribution table to spread over households.

Every dollar figure here is worked out from the formula in the comment above the
assertion rather than by calling the scorer again.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from taxmodel.reporting import receipts, receipts_inputs
from taxmodel.scenario import prepare
from taxmodel.years import world_names
from taxmodel.years.corporate import channel as corporate_channel
from taxmodel.years.world_names import Frame, Law
from tests.tolerances import assert_agrees_to_the_cent

YEARS = (2026, 2027)

# The scenario in most of these tests: the top corporate statutory rate goes
# from 21 percent, which is current law, to 28 percent.
BASELINE_RATE = 0.21
REFORM_RATE = 0.28

# CBO's projection of corporate receipts in the two years, in dollars.
CORPORATE_RECEIPTS = np.array([500e9, 600e9])

# The elasticity of the corporate tax base to the net-of-tax statutory rate.
# The production value is `corp.rate_eti` in the economy leg; a round number is
# used here so the hand arithmetic below is readable.
BASE_ELASTICITY = 0.4

# The corporate incidence assumptions the distribution table's spread runs
# on: `corp.sigma_n`, `distribution.corp_foreign_share`, the four
# `corp.asset_exposure_*` entries and `distribution.housing_structure_share`.
NORMAL_RETURN_SHARE = 0.375
FOREIGN_SHARE = 0.40
ASSET_EXPOSURE = {
    "asset_exposure_equities": 1.0,
    "asset_exposure_dc": 0.55,
    "asset_exposure_trusts": 0.5,
    "asset_exposure_re_fund": 0.3,
}
HOUSING_STRUCTURE_SHARE = 0.7

# Off-model corporate scores and the cost-recovery change, in dollars, for the
# distribution table's figures. The mechanical score (`corporate_static`) and
# the score with the corporate behavioral response (`corporate`) differ so a
# test can tell which world read which.
OFF_MODEL_MECHANICAL = np.array([10e9, 12e9])
OFF_MODEL_WITH_RESPONSE = np.array([7e9, 8e9])
COST_RECOVERY_CHANGE = np.array([-3e9, -4e9])

# The five evaluated worlds, as (household frame, tax law) cells.
CELLS = {
    "baseline": (Frame.RAW, Law.BASELINE),
    "first_order": (Frame.RAW, Law.REFORM),
    "mechanical_ex_saving": (Frame.FLOWS, Law.REFORM),
    "mechanical": (Frame.MECHANICAL, Law.REFORM),
    "conventional": (Frame.BEHAVIOR, Law.REFORM),
}


def mechanical_change() -> np.ndarray:
    """The rate change times the baseline base: no profit shifting, by hand."""
    return CORPORATE_RECEIPTS * (REFORM_RATE / BASELINE_RATE - 1.0)


def eroded_change() -> np.ndarray:
    """The same change with the base shrinking as the rate rises, by hand."""
    net_of_tax_ratio = (1.0 - REFORM_RATE) / (1.0 - BASELINE_RATE)
    return CORPORATE_RECEIPTS * (
        (REFORM_RATE / BASELINE_RATE) * net_of_tax_ratio**BASE_ELASTICITY - 1.0
    )


# --------------------------------------------------------------------------
# Stand-ins for the two things run preparation reads
# --------------------------------------------------------------------------


def compiled_law_with_rate(rate: float | None) -> SimpleNamespace:
    """A compiled law stating one corporate rate in both years, or none at all.

    A compiled law holds each field as [year, filing status, slot] with a flag
    saying whether the law states anything there. The corporate rate does not
    vary by filing status, so all four entries carry the same number.
    """
    if rate is None:
        return SimpleNamespace(values={}, active={})
    stated = np.full((len(YEARS), 4, 1), rate, dtype=np.float64)
    return SimpleNamespace(
        values={corporate_channel.CORPORATE_RATE_COLUMN: stated},
        active={
            corporate_channel.CORPORATE_RATE_COLUMN: np.ones_like(stated, dtype=bool)
        },
    )


class StandInConfig:
    """The economy entries the corporate channel reads, and nothing else."""

    def __init__(
        self, elasticity: float | None = BASE_ELASTICITY, states_incidence: bool = True
    ):
        self.elasticity = elasticity
        self.states_incidence = states_incidence

    def economy_param(self, channel: str, name: str) -> float:
        if (channel, name) == prepare.BASE_ELASTICITY_ENTRY:
            if self.elasticity is None:
                raise prepare.ConfigError(f"unknown economy entry: {channel}.{name}")
            return self.elasticity
        if self.states_incidence and (channel, name) == ("corp", "sigma_n"):
            return NORMAL_RETURN_SHARE
        if self.states_incidence and (channel, name) == (
            "distribution",
            "corp_foreign_share",
        ):
            return FOREIGN_SHARE
        if self.states_incidence and (channel, name) == (
            prepare.HOUSING_STRUCTURE_SHARE_ENTRY
        ):
            return HOUSING_STRUCTURE_SHARE
        if self.states_incidence and channel == "corp" and name in ASSET_EXPOSURE:
            return ASSET_EXPOSURE[name]
        raise prepare.ConfigError(f"unknown economy entry: {channel}.{name}")


def stand_in_inputs(**overrides) -> receipts_inputs.ReceiptsInputs:
    """The four upstream publications, all zero except CBO's corporate level.

    Zeroing the rest keeps every figure a test checks traceable to the rate
    score; `tests/unit/reporting/test_receipts.py` is where the other lines are
    checked.
    """
    zeros = np.zeros(len(YEARS))
    cbo = {
        name: zeros.copy() for name in receipts_inputs.MACRO_PROJECTIONS_COLUMNS
    }
    cbo["rev_corp"] = CORPORATE_RECEIPTS.copy()
    # Any positive figure will do for the share-of-GDP denominator.
    cbo["gdp_fy"] = np.full(len(YEARS), 30e12)
    return receipts_inputs.ReceiptsInputs(
        years=YEARS,
        cbo_levels=cbo | overrides.get("cbo_levels", {}),
        off_model={
            name: zeros.copy() for name in receipts_inputs.OFF_MODEL_COLUMNS
        }
        | overrides.get("off_model", {}),
        cost_recovery={"delta": zeros.copy()} | overrides.get("cost_recovery", {}),
    )


def stand_in_projection() -> SimpleNamespace:
    """A projection whose own model columns are all zero in every world.

    The model's own contribution to the corporate line — the tax that moves when
    businesses change entity form — is zero here, so whatever the corporate line
    reports beyond CBO's level is the rate score.
    """
    zeros = {name: np.zeros(len(YEARS)) for name in receipts.MODEL_COLUMNS}
    return SimpleNamespace(
        years=YEARS,
        totals={cell: dict(zeros) for cell in CELLS.values()},
    )


def channel_from(
    monkeypatch,
    baseline_rate: float | None,
    scenario_rate: float | None,
    inputs: receipts_inputs.ReceiptsInputs | Exception,
    elasticity: float | None = BASE_ELASTICITY,
) -> dict[str, np.ndarray]:
    """Each world's rate score from run preparation's corporate step."""
    return corporate_step(
        monkeypatch, baseline_rate, scenario_rate, inputs, elasticity
    ).rate_score_by_world


def corporate_step(
    monkeypatch,
    baseline_rate: float | None,
    scenario_rate: float | None,
    inputs: receipts_inputs.ReceiptsInputs | Exception,
    elasticity: float | None = BASE_ELASTICITY,
    states_incidence: bool = True,
) -> prepare.CorporateChannel:
    """Run preparation's corporate step, with the four publications stood in for."""

    def load(years, economy, config_root):
        if isinstance(inputs, Exception):
            raise inputs
        return inputs

    monkeypatch.setattr(receipts_inputs, "load_receipts_inputs", load)
    return prepare.prepare_corporate_channel(
        StandInConfig(elasticity, states_incidence),
        YEARS,
        corporate_channel.corporate_rate_by_year(
            compiled_law_with_rate(baseline_rate), len(YEARS)
        ),
        corporate_channel.corporate_rate_by_year(
            compiled_law_with_rate(scenario_rate), len(YEARS)
        ),
        None,
        "config",
    )


# --------------------------------------------------------------------------
# Reading the rate off the two laws
# --------------------------------------------------------------------------


def test_the_corporate_rate_is_read_off_the_compiled_law_year_by_year():
    rate = corporate_channel.corporate_rate_by_year(
        compiled_law_with_rate(BASELINE_RATE), len(YEARS)
    )
    assert rate.tolist() == [BASELINE_RATE, BASELINE_RATE]


def test_a_law_that_states_no_corporate_rate_reads_as_no_rate_stated():
    """Not as a rate of zero, which the scorer would refuse to divide by."""
    rate = corporate_channel.corporate_rate_by_year(compiled_law_with_rate(None), len(YEARS))
    assert np.all(np.isnan(rate))


# --------------------------------------------------------------------------
# The score itself
# --------------------------------------------------------------------------


def test_a_rate_rise_is_scored_with_base_erosion_only_in_the_conventional_world(
    monkeypatch,
):
    """21 percent to 28 percent, on CBO's receipts, at an elasticity of 0.4.

    Every world but the conventional one gets the mechanical change, which holds
    the corporate tax base where baseline law leaves it. The conventional world
    lets the base shrink as the rate rises, so it raises less.
    """
    channel = channel_from(monkeypatch, BASELINE_RATE, REFORM_RATE, stand_in_inputs())

    assert_agrees_to_the_cent(
        channel["baseline"],
        np.zeros(len(YEARS)),
        "the baseline world's corporate rate score",
    )
    for world in ("first_order", "mechanical_ex_saving", "mechanical"):
        assert_agrees_to_the_cent(
            channel[world],
            mechanical_change(),
            f"the {world} world's corporate rate score",
        )
    assert_agrees_to_the_cent(
        channel["conventional"],
        eroded_change(),
        "the conventional world's corporate rate score",
    )
    # Letting the base erode has to raise less than holding it fixed when the
    # rate goes up, or the elasticity has the wrong sign somewhere.
    assert np.all(eroded_change() < mechanical_change())


def test_a_scenario_that_leaves_the_corporate_rate_alone_scores_exactly_zero(
    monkeypatch,
):
    channel = channel_from(monkeypatch, BASELINE_RATE, BASELINE_RATE, stand_in_inputs())
    for world in world_names.WORLD_NAMES:
        assert np.array_equal(
            channel[world], np.zeros(len(YEARS))
        ), f"the {world} world's corporate rate score is not exactly zero"


def test_a_scenario_that_leaves_the_rate_alone_stops_when_receipts_cannot_be_read(
    monkeypatch,
):
    """The distribution table's corporate tax starts from CBO's receipts, so
    every scenario needs them."""
    with pytest.raises(receipts_inputs.ReceiptsInputError, match="corporate tax"):
        channel_from(
            monkeypatch,
            BASELINE_RATE,
            BASELINE_RATE,
            receipts_inputs.ReceiptsInputError("no vintage pinned"),
        )


def test_a_corporate_reform_stops_when_cbo_receipts_cannot_be_read(monkeypatch):
    with pytest.raises(receipts_inputs.ReceiptsInputError, match="corporate receipts"):
        channel_from(
            monkeypatch,
            BASELINE_RATE,
            REFORM_RATE,
            receipts_inputs.ReceiptsInputError("no vintage pinned"),
        )


def test_a_corporate_reform_stops_when_the_economy_leg_states_no_elasticity(
    monkeypatch,
):
    with pytest.raises(prepare.ConfigError, match="elasticity of the corporate tax base"):
        channel_from(
            monkeypatch,
            BASELINE_RATE,
            REFORM_RATE,
            stand_in_inputs(),
            elasticity=None,
        )


# --------------------------------------------------------------------------
# Where the score is reported
# --------------------------------------------------------------------------


def test_the_published_corporate_revenue_line_moves_by_each_world_s_score(
    monkeypatch,
):
    """What the channel is for: a corporate rate reform now moves the line.

    With every other piece of the corporate line zero, each world's line is
    CBO's level plus that world's own score.
    """
    channel = channel_from(monkeypatch, BASELINE_RATE, REFORM_RATE, stand_in_inputs())
    lines = receipts.receipts_by_world(
        stand_in_projection(), stand_in_inputs(), channel
    )

    assert_agrees_to_the_cent(
        lines["baseline"]["revenues_corp_tax"],
        CORPORATE_RECEIPTS,
        "the baseline world's corporate revenue line",
    )
    assert_agrees_to_the_cent(
        lines["mechanical"]["revenues_corp_tax"],
        CORPORATE_RECEIPTS + mechanical_change(),
        "the mechanical world's corporate revenue line",
    )
    assert_agrees_to_the_cent(
        lines["conventional"]["revenues_corp_tax"],
        CORPORATE_RECEIPTS + eroded_change(),
        "the conventional world's corporate revenue line",
    )


def test_the_corporate_line_does_not_move_when_no_score_is_handed_over():
    """The scenarios that leave the corporate rate alone report what they did."""
    lines = receipts.receipts_by_world(stand_in_projection(), stand_in_inputs())
    for world in CELLS:
        assert_agrees_to_the_cent(
            lines[world]["revenues_corp_tax"],
            CORPORATE_RECEIPTS,
            f"the {world} world's corporate revenue line with no rate change",
        )


# --------------------------------------------------------------------------
# The national figures the distribution table spreads
# --------------------------------------------------------------------------


def inputs_with_off_model_scores() -> receipts_inputs.ReceiptsInputs:
    """The stand-in publications, with off-model and cost-recovery figures."""
    return stand_in_inputs(
        off_model={
            "corporate_static": OFF_MODEL_MECHANICAL.copy(),
            "corporate": OFF_MODEL_WITH_RESPONSE.copy(),
        },
        cost_recovery={"delta": COST_RECOVERY_CHANGE.copy()},
    )


def test_the_distribution_table_s_corporate_figures_for_a_rate_rise(monkeypatch):
    """Each world's figure is CBO's receipts plus its rate and off-model scores.

    The baseline world scores nothing. The static figure takes the first_order
    world's score, which does not let the base erode, and the mechanical
    off-model score. The conventional figure takes the eroded score and the
    off-model score that includes the corporate behavioral response.
    """
    figures = corporate_step(
        monkeypatch, BASELINE_RATE, REFORM_RATE, inputs_with_off_model_scores()
    ).dollars

    assert_agrees_to_the_cent(
        figures.baseline, CORPORATE_RECEIPTS, "the baseline corporate figure"
    )
    assert_agrees_to_the_cent(
        figures.static,
        CORPORATE_RECEIPTS + mechanical_change() + OFF_MODEL_MECHANICAL,
        "the static corporate figure",
    )
    assert_agrees_to_the_cent(
        figures.conventional,
        CORPORATE_RECEIPTS + eroded_change() + OFF_MODEL_WITH_RESPONSE,
        "the conventional corporate figure",
    )
    assert_agrees_to_the_cent(
        figures.cost_recovery, COST_RECOVERY_CHANGE, "the cost-recovery change"
    )
    assert np.array_equal(figures.normal_return_share, [NORMAL_RETURN_SHARE] * 2)
    assert np.array_equal(figures.foreign_share, [FOREIGN_SHARE] * 2)
    assert np.array_equal(figures.dc_exposure, [ASSET_EXPOSURE["asset_exposure_dc"]] * 2)
    assert np.array_equal(
        figures.housing_structure_share, [HOUSING_STRUCTURE_SHARE] * 2
    )


def test_the_distribution_table_s_corporate_tax_is_zero_without_incidence(
    monkeypatch, capsys
):
    figures = corporate_step(
        monkeypatch,
        BASELINE_RATE,
        REFORM_RATE,
        inputs_with_off_model_scores(),
        states_incidence=False,
    ).dollars
    for name, series in figures._asdict().items():
        assert np.array_equal(series, np.zeros(len(YEARS))), name
    assert "corporate tax is zero" in capsys.readouterr().out
