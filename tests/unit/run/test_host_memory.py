"""The memory ledger reports numbers on whatever system the tests run on.

The ledger is measurement apparatus: nothing the model computes reads it. What
these tests check is that it works on this system, since the figures come from
a different place on Linux, Windows and macOS.
"""

from __future__ import annotations

import math

import pytest

from taxmodel.run import hardware, host_memory


def test_the_peak_is_a_real_number_on_this_system():
    peak = host_memory.peak_resident_gigabytes()
    assert not math.isnan(peak)
    # Python and its libraries are loaded, so the process holds something.
    assert 0.0 < peak < 1024.0


def test_resident_memory_is_a_real_number_where_the_system_offers_one():
    resident = host_memory.resident_gigabytes()
    if host_memory.LINUX or host_memory.WINDOWS:
        assert 0.0 < resident < 1024.0
    else:
        # macOS has no cheap current figure, and the ledger says so.
        assert math.isnan(resident)


def test_the_arrow_pool_reports_two_figures():
    held, peak = host_memory.arrow_pool_gigabytes()
    assert held >= 0.0
    assert peak >= held


def test_giving_back_free_memory_runs_and_returns_nothing():
    assert host_memory.give_back_free_memory() is None


def test_a_missing_figure_prints_as_not_available():
    assert host_memory._figure(float("nan")).strip() == "n/a"
    assert host_memory._figure(1.5).strip() == "1.50"


def test_the_ledger_prints_one_line_per_phase_when_it_is_asked_to(monkeypatch, capsys):
    monkeypatch.setenv(host_memory.LEDGER_VARIABLE, "on")
    host_memory.note("a phase of the run")
    printed = capsys.readouterr().out
    assert "memory ledger" in printed
    assert "a phase of the run" in printed


def test_the_ledger_is_silent_unless_it_is_asked_for(monkeypatch, capsys):
    monkeypatch.delenv(host_memory.LEDGER_VARIABLE, raising=False)
    host_memory.note("a phase of the run")
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("platforms", "expected"),
    [("cpu", False), ("cuda", True), ("cuda,cpu", True), ("CPU", False)],
)
def test_the_platform_setting_answers_the_graphics_card_question(
    monkeypatch, platforms, expected
):
    monkeypatch.setenv("JAX_PLATFORMS", platforms)
    assert hardware.graphics_card_is_present() is expected


def test_the_allocation_thresholds_are_only_set_where_there_is_no_card(monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.setattr(host_memory, "_allocation_thresholds_are_set", False)
    assert host_memory.set_the_c_library_allocation_thresholds() == ""


def test_the_allocation_thresholds_are_set_on_a_machine_with_no_card(monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setattr(host_memory, "_allocation_thresholds_are_set", False)
    first = host_memory.set_the_c_library_allocation_thresholds()
    if not host_memory.LINUX:
        assert first == ""
        return
    assert first == host_memory.ALLOCATION_THRESHOLD_NOTE
    # A second call changes nothing and reports the same thing.
    assert (
        host_memory.set_the_c_library_allocation_thresholds()
        == host_memory.ALLOCATION_THRESHOLD_NOTE
    )
