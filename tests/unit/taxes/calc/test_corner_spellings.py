"""No averaging spellings in the calculators, unless the site was examined.

At an exact threshold the household is treated as having crossed it
in the direction of more income. Because a marginal tax rate in this model is a
gradient, that convention is enforced by how each corner is written, and there
is no way to check it after the fact -- the numbers a run produces are the same
either way. So it is checked here, in the source.

`jnp.maximum`, `jnp.minimum`, `jnp.clip`, `jnp.abs` and `jax.nn.relu` all report
a slope at a tie that the convention does not want: the first three average the
two slopes, and the other two pick a side of their own. The calculators use
`floored_at` and `capped_at` from `taxmodel.taxes.calc._corners` instead, or a plain
`jnp.where(amount >= edge, ...)`.

A handful of sites keep an averaging spelling on purpose -- an integer count,
a comparison between two law parameters, a choice between two amounts where
neither one is the household's income. Each of those carries a trailing
`# tie: <reason>` comment saying why, which is what this test looks for. The
comment means someone looked at the site and decided; it is not a way to opt
out of thinking. CONTRIBUTING.md has the rule in full.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]

# Where the calculators live. Everything under `taxes/calc/` runs inside the traced
# computation, and `taxes/statutory_order.py` is the composition layer that also does a
# few steps of arithmetic of its own.
SCANNED = (
    REPO_ROOT / "src" / "taxmodel" / "taxes" / "calc",
    REPO_ROOT / "src" / "taxmodel" / "taxes" / "statutory_order.py",
)

# `_corners.py` defines the approved replacements and names the banned spellings
# in its docstring to explain what they do wrong; it contains none of them.
NOT_SCANNED = ("_corners.py",)

BANNED = re.compile(r"\b(jnp\.(maximum|minimum|clip|abs)|jax\.nn\.relu|nn\.relu)\b")
EXAMINED = re.compile(r"#\s*tie:\s*\S")


def scanned_files() -> list[Path]:
    files: list[Path] = []
    for target in SCANNED:
        if target.is_dir():
            files.extend(sorted(target.rglob("*.py")))
        else:
            files.append(target)
    return [f for f in files if f.name not in NOT_SCANNED]


def test_the_scan_covers_the_calculators():
    """A typo in a path would make this test pass by looking at nothing."""
    names = {f.name for f in scanned_files()}
    assert {"tax.py", "credits.py", "deductions.py", "statutory_order.py"} <= names


@pytest.mark.parametrize("path", scanned_files(), ids=lambda p: p.name)
def test_every_averaging_spelling_was_examined(path: Path):
    unexplained = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if BANNED.search(line) and not EXAMINED.search(line):
            unexplained.append(f"{path.relative_to(REPO_ROOT)}:{number}: {line.strip()}")
    assert not unexplained, (
        "These lines average the two slopes at a threshold, which the model does "
        "not allow on a path a marginal tax rate runs through. Use floored_at or "
        "capped_at from taxmodel.taxes.calc._corners, or write the jnp.where out; if the "
        "site really has to keep the spelling, put a trailing '# tie: <reason>' "
        "comment on the line saying why. See CONTRIBUTING.md.\n  "
        + "\n  ".join(unexplained)
    )


def test_a_reason_has_to_say_something():
    """An empty marker is not an examined site."""
    assert not EXAMINED.search("x = jnp.maximum(a, b)  # tie:")
    assert EXAMINED.search("x = jnp.maximum(a, b)  # tie: two law rates")
