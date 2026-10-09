"""Receipts: the fiscal-year timing, the eight lines, and the coverage refusal.

The fixtures here are hand-written files with round numbers, small enough that
every figure the tests check can be worked out on paper from the timing rules in
`reporting/receipts.py`. Nothing here reads the shared model-data tree.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from taxmodel.reporting import receipts, receipts_inputs
from taxmodel.reporting.receipts_inputs import BILLIONS, ReceiptsInputError
from taxmodel.years.world_names import Frame, Law
from tests.tolerances import assert_agrees_to_the_cent

YEARS = (2026, 2027)

# The five evaluated worlds, as (household frame, tax law) cells.
CELLS = {
    "baseline": (Frame.RAW, Law.BASELINE),
    "first_order": (Frame.RAW, Law.REFORM),
    "mechanical_ex_saving": (Frame.FLOWS, Law.REFORM),
    "mechanical": (Frame.MECHANICAL, Law.REFORM),
    "conventional": (Frame.BEHAVIOR, Law.REFORM),
}


def stand_in_projection(per_world):
    """A projection-shaped object carrying one calendar-year total per world."""
    zeros = {name: np.zeros(len(YEARS)) for name in receipts.MODEL_COLUMNS}
    return SimpleNamespace(
        years=YEARS,
        totals={
            CELLS[world]: zeros | {
                name: np.asarray(values, dtype=float)
                for name, values in columns.items()
            }
            for world, columns in per_world.items()
        },
    )


def write_inputs(tmp_path, **overrides):
    """The four upstream publications as fixture files, all figures in billions."""
    macro = tmp_path / "macro"
    macro.mkdir()
    (macro / "historical.csv").write_text(
        "year,gdp_fy,rev_corp,rev_estate,rev_excise,rev_customs,rev_misc\n"
        "2025,20,1,2,3,4,5\n"
    )
    (macro / "projections.csv").write_text(
        "year,gdp_fy,rev_corp,rev_estate,rev_excise,rev_customs,rev_misc\n"
        + overrides.get(
            "projections",
            "2026,100,10,2,3,4,5\n2027,200,20,4,6,8,10\n",
        )
    )

    off_model = tmp_path / "off_model"
    off_model.mkdir()
    (off_model / "revenues.csv").write_text(
        "year,individual,payroll,corporate,estate,vat,corporate_static\n"
        + overrides.get("off_model", "2026,1,2,7,99,0,3\n2027,1,2,7,99,0,3\n")
    )

    # The cost-recovery publication keeps its figures in a `deltas` subfolder,
    # so the folder handed to the reader is the one above it.
    cost_recovery = tmp_path / "cost_recovery"
    (cost_recovery / "deltas").mkdir(parents=True)
    (cost_recovery / "deltas" / "revenues.csv").write_text(
        "year,delta\n" + overrides.get("cost_recovery", "2026,5\n2027,5\n")
    )

    return receipts_inputs.ReceiptsInputs(
        years=YEARS,
        cbo_levels=receipts_inputs.read_macro_projections(macro, YEARS),
        off_model=receipts_inputs.read_off_model_estimates(off_model, YEARS),
        cost_recovery=receipts_inputs.read_cost_recovery(cost_recovery, YEARS),
    )


# --------------------------------------------------------------------------
# The calendar-year-to-fiscal-year timing
# --------------------------------------------------------------------------


def test_withheld_tax_is_three_quarters_this_year_and_a_quarter_last_year():
    """$100bn withheld in 2026 and $200bn in 2027, collected by fiscal year.

    Fiscal year 2026 collects only the nine months of calendar 2026 that fall
    inside it, because calendar 2025 was not simulated: 0.75 x 100 = 75. Fiscal
    2027 collects 0.75 x 200 from calendar 2027 plus 0.25 x 100 from calendar
    2026, which is 175.
    """
    collected = receipts.withheld_on_a_fiscal_year_basis(np.array([100.0, 200.0]))
    assert_agrees_to_the_cent(
        collected, [75.0, 175.0], "withheld tax spread across fiscal years"
    )


def test_the_estate_and_wealth_taxes_are_collected_the_year_after_they_are_assessed():
    collected = receipts.collected_the_following_year(np.array([40.0, 60.0]))
    assert_agrees_to_the_cent(
        collected, [0.0, 40.0], "a tax assessed one year and paid the next"
    )


def test_income_tax_settled_at_filing_is_booked_in_the_year_of_liability():
    """A known defect, checked so that a later fix has to be deliberate.

    Filing season for calendar 2026 falls inside fiscal 2027, but the model
    books it in fiscal 2026.
    """
    totals = {
        "pmt_iit_withheld": np.array([100.0, 100.0]),
        "pmt_iit_nonwithheld": np.array([8.0, 12.0]),
    }
    zeros = {name: np.zeros(2) for name in receipts.MODEL_COLUMNS}
    collected = receipts.model_receipts_by_world(zeros | totals, zeros)
    # Fiscal 2026: 0.75 x 100 withheld, plus all 8 of calendar 2026's filing-time
    # tax. Fiscal 2027: 0.75 x 100 + 0.25 x 100, plus all 12.
    assert_agrees_to_the_cent(
        collected["income_tax"],
        [83.0, 112.0],
        "income tax collections with filing-time tax booked in the year of liability",
    )


# --------------------------------------------------------------------------
# The eight lines
# --------------------------------------------------------------------------


def test_the_budget_total_is_the_eight_lines_with_the_credit_outlay_subtracted(tmp_path):
    inputs = write_inputs(tmp_path)
    projection = stand_in_projection(
        {
            world: {
                "pmt_iit_withheld": np.array([400.0, 400.0]) * BILLIONS,
                "pmt_refund_nonwithheld": np.array([30.0, 30.0]) * BILLIONS,
                "liab_pr": np.array([300.0, 300.0]) * BILLIONS,
            }
            for world in CELLS
        }
    )
    lines = receipts.receipts_by_world(projection, inputs)["conventional"]
    by_hand = sum(
        -lines[name] if name == receipts.OUTLAY_LINE else lines[name]
        for name in receipts.REVENUE_LINES
    )
    assert_agrees_to_the_cent(
        lines[receipts.TOTAL_LINE],
        by_hand,
        "the budget total against its eight lines",
    )


def test_the_corporate_line_adds_cbo_the_two_off_model_scores_and_the_shifting_change(
    tmp_path,
):
    """One year of the corporate line, worked out by hand.

    CBO projects $10bn of corporate receipts in fiscal 2026. The scenario's
    depreciation policy is scored at $5bn and its off-model corporate provisions
    at $3bn mechanically or $7bn with the corporate behavioral response. The
    model itself contributes $80bn of shifting between the individual and
    corporate sectors in calendar 2026, of which fiscal 2026 collects three
    quarters.
    """
    inputs = write_inputs(tmp_path)
    projection = stand_in_projection(
        {
            world: {"corp_tax_change": np.array([80.0, 0.0]) * BILLIONS}
            for world in CELLS
        }
    )
    lines = receipts.receipts_by_world(projection, inputs)

    assert_agrees_to_the_cent(
        lines["mechanical"]["revenues_corp_tax"][0],
        (10 + 5 + 3 + 0.75 * 80) * BILLIONS,
        "the mechanical world's corporate line",
    )
    assert_agrees_to_the_cent(
        lines["conventional"]["revenues_corp_tax"][0],
        (10 + 5 + 7 + 0.75 * 80) * BILLIONS,
        "the conventional world's corporate line, which reads the behavioral score",
    )
    # The baseline world carries no off-model score, because that is a reform's
    # effect, but it does carry the $5bn depreciation score, because that
    # restates CBO's corporate level rather than scoring a reform.
    assert_agrees_to_the_cent(
        lines["baseline"]["revenues_corp_tax"][0],
        (10 + 5 + 0.75 * 80) * BILLIONS,
        "the baseline world's corporate line, which carries no off-model score",
    )


def test_every_world_gets_the_same_depreciation_score(tmp_path):
    """The cost-recovery score is a level, not a reform effect.

    Each scenario's depreciation and expensing score is applied to its
    corporate line in every world, baseline included, so the score cancels out
    of the corporate revenue estimate when nothing else about the corporate
    line differs between the worlds.
    """
    inputs = write_inputs(tmp_path, cost_recovery="2026,-109\n2027,-115\n")
    projection = stand_in_projection({world: {} for world in CELLS})
    lines = receipts.receipts_by_world(projection, inputs)

    # CBO projects $10bn of corporate receipts in fiscal 2026 and the
    # depreciation policy is scored at -$109bn, so every world reports -$99bn
    # before any off-model score.
    for world in CELLS:
        off_model_score = 7 if world == "conventional" else 3
        expected = 10 - 109 + (0 if world == "baseline" else off_model_score)
        assert_agrees_to_the_cent(
            lines[world]["revenues_corp_tax"][0],
            expected * BILLIONS,
            f"the {world} world's corporate line",
        )

    estimates = receipts.revenue_estimates_table(
        receipts.receipts_by_world(projection, inputs), inputs
    )
    corporate = estimates[estimates["line"] == "revenues_corp_tax"]
    # Only the off-model score survives the subtraction: $3bn mechanically,
    # $7bn where the corporate behavioral response is in.
    for _, row in corporate.iterrows():
        expected = 7 if row["world"] == "conventional" else 3
        assert_agrees_to_the_cent(
            row["dollars"],
            expected * BILLIONS,
            "the corporate revenue estimate, with the depreciation score cancelled",
        )


def test_the_estate_line_is_cbos_level_plus_the_reforms_change_a_year_later(tmp_path):
    inputs = write_inputs(tmp_path)
    projection = stand_in_projection(
        {
            "baseline": {"est_tax_exp": np.array([20.0, 20.0]) * BILLIONS},
            "first_order": {"est_tax_exp": np.array([35.0, 35.0]) * BILLIONS},
            "mechanical_ex_saving": {"est_tax_exp": np.array([35.0, 35.0]) * BILLIONS},
            "mechanical": {"est_tax_exp": np.array([35.0, 35.0]) * BILLIONS},
            "conventional": {"est_tax_exp": np.array([35.0, 35.0]) * BILLIONS},
        }
    )
    lines = receipts.receipts_by_world(projection, inputs)["first_order"]
    # CBO's estate level is $2bn in fiscal 2026 and $4bn in 2027. The reform
    # raises calendar-2026 estate tax by $15bn, collected in fiscal 2027.
    assert_agrees_to_the_cent(
        lines["revenues_estate_tax"],
        np.array([2.0, 4.0 + 15.0]) * BILLIONS,
        "the estate line, whose reform change arrives a year after the deaths",
    )


def test_the_revenue_estimate_is_each_world_less_the_baseline_world(tmp_path):
    inputs = write_inputs(tmp_path)
    projection = stand_in_projection(
        {
            "baseline": {"pmt_iit_withheld": np.array([400.0, 400.0]) * BILLIONS},
            "first_order": {"pmt_iit_withheld": np.array([360.0, 360.0]) * BILLIONS},
            "mechanical_ex_saving": {
                "pmt_iit_withheld": np.array([360.0, 360.0]) * BILLIONS
            },
            "mechanical": {"pmt_iit_withheld": np.array([360.0, 360.0]) * BILLIONS},
            "conventional": {"pmt_iit_withheld": np.array([370.0, 370.0]) * BILLIONS},
        }
    )
    estimates = receipts.revenue_estimates_table(
        receipts.receipts_by_world(projection, inputs), inputs
    )

    assert receipts.BASELINE_WORLD not in set(estimates["world"])
    income = estimates[
        (estimates["world"] == "first_order")
        & (estimates["line"] == "revenues_income_tax")
    ].sort_values("year")
    # A $40bn cut in calendar-year withholding is collected as three quarters of
    # it in the first fiscal year and the whole of it thereafter. The $1bn
    # off-model individual score survives the subtraction, because the reform
    # world carries it and the baseline world does not.
    assert_agrees_to_the_cent(
        np.asarray(income["dollars"]),
        np.array([-0.75 * 40 + 1.0, -40.0 + 1.0]) * BILLIONS,
        "the income tax revenue estimate",
    )

    total = estimates[
        (estimates["world"] == "first_order") & (estimates["line"] == receipts.TOTAL_LINE)
    ].sort_values("year")
    lines_only = estimates[
        (estimates["world"] == "first_order")
        & (estimates["line"] != receipts.TOTAL_LINE)
    ]
    signed = lines_only.assign(
        signed=np.where(
            lines_only["line"] == receipts.OUTLAY_LINE,
            -lines_only["dollars"],
            lines_only["dollars"],
        )
    )
    assert_agrees_to_the_cent(
        np.asarray(total["dollars"]),
        np.asarray(signed.groupby("year")["signed"].sum().sort_index()),
        "the revenue estimate's total against the sum of its lines",
    )


def test_share_of_gdp_divides_the_estimate_by_fiscal_year_gdp(tmp_path):
    inputs = write_inputs(tmp_path)
    projection = stand_in_projection(
        {
            world: {
                "pmt_iit_withheld": np.array([0.0 if world == "baseline" else -100.0] * 2)
                * BILLIONS
            }
            for world in CELLS
        }
    )
    estimates = receipts.revenue_estimates_table(
        receipts.receipts_by_world(projection, inputs), inputs
    )
    row = estimates[
        (estimates["world"] == "conventional")
        & (estimates["line"] == "revenues_income_tax")
        & (estimates["year"] == 2026)
    ].iloc[0]
    # Fiscal-year GDP is $100bn in 2026.
    assert row["share_of_gdp"] == pytest.approx(row["dollars"] / (100 * BILLIONS))


# --------------------------------------------------------------------------
# Refuse years the inputs do not cover
# --------------------------------------------------------------------------


def test_a_blank_cost_recovery_score_stops_the_run(tmp_path):
    """A blank cost-recovery score stops the run.

    The pinned cost-recovery vintage carries rows past 2056 with the score left
    blank.
    """
    with pytest.raises(ReceiptsInputError) as raised:
        write_inputs(tmp_path, cost_recovery="2026,5\n2027,\n")
    message = str(raised.value)
    assert "delta" in message
    assert "blank" in message
    assert "2027" in message


def test_a_year_with_no_row_at_all_stops_the_run(tmp_path):
    with pytest.raises(ReceiptsInputError) as raised:
        write_inputs(tmp_path, cost_recovery="2026,5\n")
    assert "delta" in str(raised.value)
    assert "no row for 2027" in str(raised.value)


def test_an_off_model_vintage_without_the_mechanical_corporate_score_stops_the_run(
    tmp_path,
):
    folder = tmp_path / "vintage_before_version_5"
    folder.mkdir()
    (folder / "revenues.csv").write_text(
        "year,individual,payroll,corporate,estate,vat\n"
        "2026,1,2,7,99,0\n"
        "2027,1,2,7,99,0\n"
    )
    with pytest.raises(ReceiptsInputError) as raised:
        receipts_inputs.read_off_model_estimates(folder, YEARS)
    assert "corporate_static" in str(raised.value)


def test_the_upstream_files_are_read_in_billions_and_reported_in_dollars(tmp_path):
    inputs = write_inputs(tmp_path)
    assert inputs.cbo_levels["rev_corp"][0] == pytest.approx(10 * BILLIONS)
    assert inputs.cost_recovery["delta"][0] == pytest.approx(5 * BILLIONS)
