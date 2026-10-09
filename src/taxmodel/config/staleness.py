"""Staleness checks for calibrated values whose inputs have changed.

Only calibrated values can go stale, because only they have inputs. The
check runs once, at parse time, before any of those values is used.

A value is stale in either of two ways. Under `derived_under`, the data
vintages the value was pinned against differ from the ones this run uses.
Under `invalidated_by`, a file the value depends on has different contents
than when the value was pinned. File contents are compared by md5 hash, and
only files in this repo are covered.

A stop is cleared by re-deriving and re-pinning the value, or by overriding
it in an alternative. Both are visible in the run manifest.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import StaleCalibrationError
from .scenario import ResolvedLeg

# What a stale configuration value means and what to do about it, printed
# above the findings when a run stops on them.
CONFIG_STALENESS_EXPLANATION = (
    "STALE CALIBRATION\n\n"
    "One or more calibrated values no longer match the inputs they were\n"
    "derived under. Re-derive and re-pin, or override the value in an\n"
    "alternative (which records the choice in the manifest)."
)


@dataclass
class StalenessReport:
    """The findings of a staleness check.

    `stops` holds findings that stop the run. `warnings` holds findings that
    are printed only. `explanation` is the text printed above the findings
    when the run stops. The default is the configuration tree's, and the
    capital-gains calibration check supplies its own
    (`calibration/staleness.py`).
    """

    stops: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    explanation: str = CONFIG_STALENESS_EXPLANATION

    @property
    def clean(self) -> bool:
        return not self.stops and not self.warnings

    def extend(self, other: StalenessReport) -> None:
        self.stops.extend(other.stops)
        self.warnings.extend(other.warnings)

    def raise_if_stopping(self, accept_stale: bool = False) -> None:
        """Stop on the findings in `stops`, unless the caller accepts them.

        `accept_stale` is the `--accept-stale-calibration` command-line flag.
        No file can set it.
        """
        if not self.stops or accept_stale:
            return
        raise StaleCalibrationError(
            "\n"
            + "-" * 78
            + "\n"
            + self.explanation
            + "\n\n  - "
            + "\n  - ".join(self.stops)
            + "\n"
            + "-" * 78
        )


def file_md5(path: Path) -> str:
    """Hash a dependency file's contents."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _entry_findings(entry: Any, base_dir: Path, vintages: dict[str, str]) -> list[str]:
    """Check one calibrated entry for both kinds of staleness."""
    label = entry.label
    findings: list[str] = []

    # The data vintages the value was derived under
    for dep, pinned in entry.derived_under.items():
        live = vintages.get(dep)
        if live is None:
            findings.append(
                f"{label}: derived_under names an interface this run does not "
                f"use ({dep})"
            )
            continue
        if str(live) != str(pinned):
            findings.append(
                f"{label}: pinned against {dep} vintage {pinned}, this run uses "
                f"{live} -- re-derive with {entry.rederive}"
            )

    # The files it depends on
    recorded = entry.invalidated_by_hashes
    if recorded is None:
        if entry.invalidated_by:
            findings.append(
                f"{label}: no invalidated_by_hashes recorded -- the value was "
                f"never pinned against the files it depends on"
            )
    else:
        for dep in entry.invalidated_by:
            pinned_hash = recorded.get(dep)
            if pinned_hash is None:
                findings.append(f"{label}: no hash recorded for dependency {dep}")
                continue
            if file_md5(base_dir / dep) != str(pinned_hash):
                findings.append(
                    f"{label}: {dep} has changed since this value was pinned on "
                    f"{entry.raw.get('set')} -- re-derive with {entry.rederive}, "
                    f"or override the value in an alternative"
                )

    return findings


def check_leg(resolved: ResolvedLeg, vintages: dict[str, str]) -> StalenessReport:
    """Check every calibrated entry in one resolved leg.

    An entry the scenario overrode is skipped: overriding it is the
    acknowledgment, and the override is recorded in the manifest. The caller
    decides whether to stop, with `StalenessReport.raise_if_stopping`.
    """
    report = StalenessReport()
    overridden = resolved.overridden
    base_dir = resolved.defaults.base_dir

    for entries in resolved.defaults.entries.values():
        for entry in entries.values():
            if entry.kind != "calibrated":
                continue
            if entry.label in overridden:
                continue

            findings = _entry_findings(entry, base_dir, vintages)
            if entry.enforcement == "warn":
                report.warnings.extend(findings)
            else:
                report.stops.extend(findings)

    return report
