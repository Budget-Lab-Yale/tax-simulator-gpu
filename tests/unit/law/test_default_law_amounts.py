"""Statutory amounts in the default law, compiled with the golden index table.

Each test states the amount the statute or IRS guidance gives and checks that
the default law compiles to it. Filing statuses are 1 single, 2 joint, 3
married filing separately, 4 head of household.
"""

from __future__ import annotations

import pandas as pd
import pytest

from taxmodel.law import compile_law
from taxmodel.law.indexes import index_table_from_frame
from tests.conftest import CONFIG_ROOT, INDEX_TABLE

YEARS = (2024, 2025, 2026, 2027)
FILING_STATUSES = (1, 2, 3, 4)


@pytest.fixture(scope="module")
def default_law():
    index_table = index_table_from_frame(pd.read_csv(INDEX_TABLE))
    return compile_law(CONFIG_ROOT, YEARS, index_table)


def by_status(law, field: str, year: int) -> dict[int, float]:
    """One single-valued field in one year, by filing status."""
    row = law.values[field][law.years.index(year), :, 0]
    return {status: float(row[i]) for i, status in enumerate(FILING_STATUSES)}


def test_the_2025_standard_deduction_is_the_obbba_amount(default_law):
    """IRC 63(c)(7) as amended by OBBBA sec. 70102; IRS, tax year 2026
    inflation adjustments (Rev. Proc. 2025-32): $15,750 single and separate,
    $31,500 joint, $23,625 head of household."""
    assert by_status(default_law, "std.value", 2025) == {
        1: 15_750.0,
        2: 31_500.0,
        3: 15_750.0,
        4: 23_625.0,
    }


def test_the_2026_standard_deduction_is_indexed_from_2024(default_law):
    """Rev. Proc. 2025-32: $16,100 single and separate, $32,200 joint,
    $24,150 head of household, which the golden index table reproduces."""
    assert by_status(default_law, "std.value", 2026) == {
        1: 16_100.0,
        2: 32_200.0,
        3: 16_100.0,
        4: 24_150.0,
    }


@pytest.mark.parametrize(
    "year, tops",
    [
        # Rev. Proc. 2023-34, 2024-40 and 2025-32, single filers: the tops of
        # the 10, 12, 22, 24, 32 and 35 percent brackets.
        (2024, [11_600, 47_150, 100_525, 191_950, 243_725, 609_350]),
        (2025, [11_925, 48_475, 103_350, 197_300, 250_525, 626_350]),
        (2026, [12_400, 50_400, 105_700, 201_775, 256_225, 640_600]),
    ],
)
def test_single_filer_brackets_are_the_irs_amounts(default_law, year, tops):
    """The brackets are indexed from 2017 (and, for the two lowest from 2026,
    from 2016) by chained CPI-U as the IRS reads it."""
    row = default_law.values["ord.brackets"][default_law.years.index(year), 0, 1:7]
    assert [float(x) for x in row] == [float(t) for t in tops]


@pytest.mark.parametrize(
    "field, year, slot, amounts",
    [
        # Rev. Proc. 2024-40 (2025) and 2025-32 (2026). Statuses are single,
        # joint, separate, head of household; None is not checked.
        ("estate.exemption", 2025, 0, (13_990_000,) * 4),
        ("eitc.po_thresh_0", 2025, 0, (10_620, 17_730, 10_620, 10_620)),
        ("amt.exempt_po_thresh", 2025, 0, (626_350, 1_252_700, 626_350, 626_350)),
        ("amt.brackets", 2025, 1, (239_100, 239_100, 119_550, 239_100)),
        ("pref.brackets", 2025, 1, (48_350, 96_700, 48_350, 64_750)),
        ("pref.brackets", 2025, 2, (533_400, 600_050, None, 566_700)),
        ("pref.brackets", 2026, 1, (49_450, 98_900, 49_450, 66_200)),
        ("pref.brackets", 2026, 2, (545_500, 613_700, None, 579_600)),
        ("qbi.po_thresh_non_sstb", 2026, 0, (201_750, 403_500, 201_775, 201_750)),
        ("agi.bus_loss_limit", 2026, 0, (256_000, 512_000, 256_000, 256_000)),
    ],
)
def test_indexed_amounts_are_the_irs_amounts(default_law, field, year, slot, amounts):
    """Rounding rules, base years and base amounts as the statute states them:
    the estate exemption rounds to the nearest $10,000 from a 2010 base; the
    childless EITC phase-out start is indexed; the AMT 28 percent threshold
    rounds to the nearest $100 and the unmarried phase-out threshold is half
    the joint one; the capital gains thresholds are the TCJA amounts indexed
    from 2017; the separate-return QBI threshold rounds to $25; the excess
    business loss limit is OBBBA's, indexed from 2024."""
    row = default_law.values[field][default_law.years.index(year), :, slot]
    checked = [(float(x), float(a)) for x, a in zip(row, amounts) if a is not None]
    assert [x for x, _ in checked] == [a for _, a in checked]


def test_the_non_itemizer_charitable_deduction_starts_in_2026(default_law):
    """IRC 170(p) as amended by OBBBA sec. 70424: up to $1,000 of cash gifts,
    $2,000 on a joint return, for taxable years beginning after 2025."""
    assert by_status(default_law, "char.nonitem_limit", 2025) == {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    assert by_status(default_law, "char.nonitem_limit", 2026) == {
        1: 1_000.0,
        2: 2_000.0,
        3: 1_000.0,
        4: 1_000.0,
    }


@pytest.mark.parametrize(
    "alternative",
    [
        "baseline_2024",
        "baseline_2024_pre_obbba",
        "baseline_2024_tcja_ext",
        "tests/baseline_2017",
        "tests/tcja_2017",
        "tests/booker_ctc_tcja_ext",
        "public/obbba/retrospective_2026/stack_vs_2024/01-rates",
        "public/obbba/retrospective_2026/stack_vs_2024/02-amt",
        "tests/tax_expenditures/repeal_char",
        "tests/tax_expenditures/booker_repeal_char",
    ],
)
def test_pre_obbba_law_and_the_charity_repeals_have_no_non_itemizer_deduction(
    alternative,
):
    """Each of these restates the charitable law, so each states the amount."""
    # Several of these folders have lint findings that have nothing to do with
    # charity (indexed amounts restated without their indexation), so the
    # lint is not run here.
    index_table = index_table_from_frame(pd.read_csv(INDEX_TABLE))
    law = compile_law(
        CONFIG_ROOT, (2026,), index_table, alternative=alternative, lint=False
    )
    assert by_status(law, "char.nonitem_limit", 2026) == {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}


def test_the_residential_energy_credits_end_after_2025(default_law):
    """IRC 25C(i) and 25D(h) as amended by OBBBA secs. 70505 and 70506; IRS
    Fact Sheet 2025-05: neither credit is allowed for property placed in
    service, or expenditures made, after 31 December 2025."""
    assert by_status(default_law, "credits.repeal_res_energy", 2025) == {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    assert by_status(default_law, "credits.repeal_res_energy", 2026) == {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}


def test_the_savers_credit_for_retirement_contributions_ends_after_2026(default_law):
    """IRC 25B(d)(1)(B) as amended by SECURE 2.0 sec. 103 and OBBBA sec. 70116:
    contributions to retirement plans and IRAs count only in taxable years
    beginning before 1 January 2027."""
    assert by_status(default_law, "savers.repeal", 2026) == {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    assert by_status(default_law, "savers.repeal", 2027) == {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}


@pytest.mark.parametrize(
    "alternative, repealed",
    [
        ("baseline_2024", 1.0),
        ("baseline_2024_pre_obbba", 1.0),
        ("baseline_2024_tcja_ext", 1.0),
        ("tests/booker_ctc_tcja_ext", 1.0),
        ("tests/baseline_2017", 0.0),
        ("tests/tcja_2017", 0.0),
    ],
)
def test_the_savers_credit_ends_in_2024_law_and_not_in_2017_law(alternative, repealed):
    """SECURE 2.0 was enacted in December 2022, so 2024 law ends the credit
    for retirement contributions after 2026 and 2017 law does not."""
    index_table = index_table_from_frame(pd.read_csv(INDEX_TABLE))
    # Lint is off for the reason given in the charitable test above.
    law = compile_law(
        CONFIG_ROOT, (2027,), index_table, alternative=alternative, lint=False
    )
    assert by_status(law, "savers.repeal", 2027) == dict.fromkeys(FILING_STATUSES, repealed)


def test_the_additional_standard_deduction_for_a_separate_filer_is_the_married_amount(default_law):
    """IRC 63(f); Rev. Proc. 2024-40: $1,600 married, $2,000 unmarried, in 2025."""
    assert by_status(default_law, "std.bonus", 2025) == {
        1: 2_000.0,
        2: 1_600.0,
        3: 1_600.0,
        4: 2_000.0,
    }


def test_the_salt_phase_down_rate_of_a_separate_filer_is_half(default_law):
    """2025 Schedule A, line 5e worksheet: the limit is halved after the 30% reduction."""
    assert by_status(default_law, "item.salt_po_rate", 2025) == {
        1: 0.30,
        2: 0.30,
        3: 0.15,
        4: 0.30,
    }


def test_the_childless_eitc_phase_out_threshold_is_indexed(default_law):
    """Rev. Proc. 2023-34: $10,330 single and $17,250 joint in 2024. The golden index
    table carries the preliminary 2016 chained CPI, which can leave these $10 high."""
    amounts = by_status(default_law, "eitc.po_thresh_0", 2024)
    assert amounts[1] == pytest.approx(10_330.0, abs=10.0)
    assert amounts[2] == pytest.approx(17_250.0, abs=10.0)
