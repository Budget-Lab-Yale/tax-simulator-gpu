"""Binding compiled law to the calculators' law objects."""

import numpy as np
import pytest

from taxmodel.law.errors import LawError
from taxmodel.taxes import law_for_calculators

from .conftest import YEAR

FILING_STATUSES = np.array([1, 2, 3, 4])


def test_every_calculator_parameter_has_a_table_entry():
    # A calculator that gains a parameter must gain the line in bind.py that
    # supplies it, or it would be constructed with a missing argument.
    assert law_for_calculators.check_dataclass_coverage() == []


def test_frozen_baseline_supplies_every_parameter(compiled_baseline):
    values, _years = compiled_baseline
    assert law_for_calculators.missing_fields(tuple(values)) == []



def test_a_missing_compiled_field_stops_the_bind_and_names_its_reader(
    compiled_baseline,
):
    values, years = compiled_baseline
    without_ctc_value = {k: v for k, v in values.items() if k != "ctc.value_young"}
    with pytest.raises(LawError) as raised:
        law_for_calculators.bind_year(without_ctc_value, years, YEAR)
    message = str(raised.value)
    assert "ctc.value_young" in message
    assert "ctc.value_young1" in message  # the dataclass field that wanted it


def test_every_gap_is_reported_at_once(compiled_baseline):
    # A bind that stopped at the first gap would take one run per gap to
    # discover what the law is short of.
    values, years = compiled_baseline
    short = {k: v for k, v in values.items() if not k.startswith("eitc.")}
    with pytest.raises(LawError) as raised:
        law_for_calculators.bind_year(short, years, YEAR)
    assert str(raised.value).count("compiled law has no field") == 22


def test_a_slot_the_law_leaves_empty_is_a_gap_not_a_nan(compiled_baseline):
    # The CTC's second credit layer lives in slot 2 of ctc.value_young. A law
    # that declares only one layer would otherwise run the whole chain and
    # return NaN liabilities.
    values, years = compiled_baseline
    hollowed = dict(values)
    blanked = values["ctc.value_young"].copy()
    blanked[:, :, 1] = np.nan
    hollowed["ctc.value_young"] = blanked
    with pytest.raises(LawError) as raised:
        law_for_calculators.bind_year(hollowed, years, YEAR)
    assert "value_young2 reads 'ctc.value_young' slot 2" in str(raised.value)


def test_a_year_outside_the_compile_is_refused(compiled_baseline):
    values, years = compiled_baseline
    with pytest.raises(LawError, match="2050"):
        law_for_calculators.bind_year(values, years, 2050)


def test_an_unknown_filing_status_is_refused():
    with pytest.raises(LawError, match="outside 1-4"):
        law_for_calculators.check_filing_status_codes(np.array([1, 5]))


def test_a_bound_law_holds_its_four_filing_status_values(
    baseline_law_by_filing_status,
):
    # Binding reads no population: every parameter arrives as its four
    # filing-status values whether or not the four differ. The 2027 standard
    # deduction is one amount for a single filer and twice that for a joint
    # return.
    law = baseline_law_by_filing_status
    std = np.asarray(law.std_ded.value)
    assert std.shape == (4,)
    assert std[1] == pytest.approx(2 * std[0])
    # `std.bonus_other` is the hook a reform adds a flat addition to the
    # standard deduction with, and it is zero under current law for everyone.
    # It is still four numbers, which is what makes every law present the same
    # shapes.
    bonus_other = np.asarray(law.std_ded.bonus_other)
    assert bonus_other.shape == (4,)
    assert (bonus_other == 0.0).all()


def test_reading_a_law_for_records_gives_each_household_its_own_value(
    baseline_law_by_filing_status,
):
    # Four households, one per filing status, in a different order than the
    # law table stores them: each still gets its own status's amount.
    order = np.array([4, 2, 1, 3])
    read = baseline_law_by_filing_status.for_records(order)
    table = np.asarray(baseline_law_by_filing_status.std_ded.value)
    np.testing.assert_array_equal(
        np.asarray(read.std_ded.value), table[order - 1]
    )


def test_schedules_keep_their_slots(baseline_law):
    # The ordinary rate schedule is seven brackets padded to sixteen slots,
    # for each of the four households. The pad is NaN, which is what the
    # calculators' schedule helpers read as "this slot holds no bracket".
    rates = np.asarray(baseline_law.tax.ord_rates)
    assert rates.shape == (4, 16)
    assert np.count_nonzero(~np.isnan(rates[0])) == 7
    assert rates[0, 0] == pytest.approx(0.10)


def test_the_social_security_worksheet_is_nested_in_the_agi_law(baseline_law):
    # calc_agi runs the §86 inclusion worksheet inside itself, so AgiLaw.ss is
    # the same object bound for the worksheet, not a second copy.
    assert baseline_law.agi.ss is baseline_law.ss
    # Under 2027 law the second tier includes 85% of benefits above the higher
    # threshold, and the thresholds differ for a joint return.
    brackets = np.asarray(baseline_law.ss.brackets)
    assert brackets.shape == (4, 16)
    assert brackets[0, 1] == pytest.approx(25_000.0)
    assert brackets[1, 1] == pytest.approx(32_000.0)


def test_the_multi_slot_credit_parameters_land_in_the_right_order(baseline_law):
    # The CDCTC's three phaseout segments are consecutive slots of one
    # compiled field. Under 2027 law the first segment never phases out, the
    # second starts at $75,000 of AGI for a single filer and $150,000 for a
    # joint return, and the third starts at $15,000 for everyone.
    assert np.isinf(np.asarray(baseline_law.cdctc.young_po_thresh1)).all()
    second = np.asarray(baseline_law.cdctc.young_po_thresh2)
    assert second[0] == pytest.approx(75_000.0)
    assert second[1] == pytest.approx(150_000.0)
    assert np.asarray(baseline_law.cdctc.young_po_thresh3) == pytest.approx(15_000.0)


def test_every_year_of_the_frozen_law_binds(compiled_baseline):
    values, years = compiled_baseline
    for year in years:
        law_for_calculators.bind_year(values, years, year)
