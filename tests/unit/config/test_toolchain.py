"""The startup toolchain guard (ARCHITECTURE.md D1).

Results are trusted only on the verified jax and CUDA versions. An XLA
miscompilation on one jax and CUDA pairing changed computed tax liability,
and unit tests of the tax code cannot catch that.
"""

from __future__ import annotations

import jax
import pytest

from taxmodel.config import toolchain


def test_the_verified_stack_passes():
    toolchain.check_toolchain()


def test_an_unpinned_jax_version_stops(monkeypatch):
    monkeypatch.setattr(jax, "__version__", "0.7.1")
    with pytest.raises(toolchain.ToolchainError, match="verified stack pins 0.7.0"):
        toolchain.check_toolchain()


def test_the_record_reports_the_mismatch_without_raising(monkeypatch):
    """The manifest says what stack a run used even when the guard refused it."""
    monkeypatch.setattr(jax, "__version__", "0.7.1")
    record = toolchain.toolchain_record()
    assert record["found"]["jax"] == "0.7.1"
    assert record["pinned"]["jax"] == "0.7.0"
    assert record["matches_pins"] is False
