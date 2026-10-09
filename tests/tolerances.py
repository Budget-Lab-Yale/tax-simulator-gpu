"""How close two dollar figures have to be before a test calls them equal.

Dollar amounts related by an accounting identity are compared to one cent:
sums that must telescope, decompositions that must add to the whole, and
re-aggregations that must return the total they came from. Several of these
identities hold exactly in arithmetic but only nearly in binary floating
point, for example the stacked reform's increments, which subtract cumulative
runs in the trillions and add the differences back up. A cent is the smallest
unit the model publishes.

Identifiers, filing statuses, flags, counts and law sentinels are compared
exactly. So are claims that two code paths compute the same number, such as a
batch of scenarios against the same scenarios run one at a time. Rates,
shares and other computed quantities carry their own tolerances.
"""

from __future__ import annotations

import numpy as np

# A cent, in dollars. The smallest unit the model publishes.
CENT = 0.01


def assert_agrees_to_the_cent(left, right, what: str) -> None:
    """Fail with the worst offending gap named.

    `what` says which identity is being checked, in words, so the failure
    message reads as a statement about the model rather than about two arrays.
    """
    difference = np.abs(np.asarray(left) - np.asarray(right))
    worst = float(np.max(difference)) if difference.size else 0.0
    assert worst <= CENT, (
        f"{what}: the two sides differ by ${worst:,.6f} at worst, which is "
        f"more than the cent this model treats as agreement.\n"
        f"  left:  {np.asarray(left)}\n"
        f"  right: {np.asarray(right)}"
    )
