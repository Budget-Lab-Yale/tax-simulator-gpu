"""Annual wealth tax on the frame's economic net-worth measure."""

from dataclasses import dataclass

import jax.numpy as jnp

from taxmodel.taxes.calc._schedules import schedule_tax

Array = jnp.ndarray


@dataclass(frozen=True)
class WealthLaw:
    """The filing-status-resolved wealth-tax schedule for one simulation year."""

    brackets: Array
    rates: Array


def calc_wealth(net_worth: Array, law: WealthLaw) -> dict[str, Array]:
    """Return annual wealth-tax liability for each tax unit.

    The first zero-rate bracket is the wealth-tax exemption. There is no
    separate exemption or valuation discount in this calculation.
    """
    return {
        "liab_wealth": schedule_tax(jnp.asarray(net_worth), law.brackets, law.rates)
    }
