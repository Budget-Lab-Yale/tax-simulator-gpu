"""The module loader: it checks declarations, pins the order, and composes."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from taxmodel.behavior.loader import (
    ManifestError,
    compose,
    order_modules,
)

COLUMNS = ("kg_lt", "wages_base1", "part_active")
PRICES = ("mtr_kg_lt", "mtr_part_active")


def manifest(name, family, **overrides):
    declaration = {
        "name": name,
        "family": family,
        "reads": (),
        "writes": (),
        "prices": (),
        "discrete_prices": (),
        "draws": (),
        "params": (),
    }
    declaration.update(overrides)
    return declaration


def add_module(amount):
    """A toy module that adds a fixed number to realized gains."""

    def run(records, prices_reform, prices_baseline, params, draws):
        del prices_reform, prices_baseline, params, draws
        return {"kg_lt": jnp.asarray(records["kg_lt"]) + amount}

    return run


def double_module(records, prices_reform, prices_baseline, params, draws):
    """A toy module that doubles realized gains, so order shows in the answer."""

    del prices_reform, prices_baseline, params, draws
    return {"kg_lt": jnp.asarray(records["kg_lt"]) * 2.0}


def test_a_manifest_reading_a_column_the_frame_lacks_stops_the_run():
    entries = [
        {
            "function": add_module(1.0),
            "manifest": manifest("toy", "charity", reads=("no_such_column",)),
            "params": {},
        }
    ]
    with pytest.raises(ManifestError, match="no_such_column"):
        compose(entries, COLUMNS, PRICES)


def test_a_module_may_write_a_column_the_frame_does_not_carry_yet():
    """Some modules create columns: the gains applier writes the death-gain ones."""
    entries = [
        {
            "function": add_module(1.0),
            "manifest": manifest("toy", "charity", writes=("a_new_column",)),
            "params": {},
        }
    ]
    compose(entries, COLUMNS, PRICES)


def test_a_module_may_read_what_an_earlier_module_wrote():
    """Entity shifting reads the equivalent tax rates the gains applier attaches."""
    entries = [
        {
            "function": add_module(1.0),
            "manifest": manifest(
                "writer", "kg_dynamics", writes=("equivalent_tax_rate_scenario",)
            ),
            "params": {},
        },
        {
            "function": double_module,
            "manifest": manifest(
                "reader", "entity_shifting", reads=("equivalent_tax_rate_scenario",)
            ),
            "params": {},
        },
    ]
    compose(entries, COLUMNS, PRICES)

    # The other way round is a mistake, and it is caught: nothing has written
    # the column by the time the reader runs.
    backwards = [
        {
            "function": double_module,
            "manifest": manifest(
                "reader", "conversion", reads=("equivalent_tax_rate_scenario",)
            ),
            "params": {},
        },
        {
            "function": add_module(1.0),
            "manifest": manifest(
                "writer", "entity_shifting", writes=("equivalent_tax_rate_scenario",)
            ),
            "params": {},
        },
    ]
    with pytest.raises(ManifestError, match="equivalent_tax_rate_scenario"):
        compose(backwards, COLUMNS, PRICES)


def test_a_manifest_naming_a_rate_the_run_does_not_compute_stops_the_run():
    entries = [
        {
            "function": add_module(1.0),
            "manifest": manifest("toy", "charity", prices=("mtr_rent",)),
            "params": {},
        }
    ]
    with pytest.raises(ManifestError, match="mtr_rent"):
        compose(entries, COLUMNS, PRICES)


def test_families_run_in_the_pinned_order_whatever_order_they_are_listed_in():
    """Two toy modules whose composition order the answer gives away.

    Adding one and then doubling gives 4 on a starting gain of 1; doubling and
    then adding one gives 3.  Conversion runs before entity shifting, so listing
    them the other way round has to produce the first answer.
    """

    listed_backwards = [
        {
            "function": double_module,
            "manifest": manifest("doubler", "entity_shifting", writes=("kg_lt",)),
            "params": {},
        },
        {
            "function": add_module(1.0),
            "manifest": manifest("adder", "conversion", writes=("kg_lt",)),
            "params": {},
        },
    ]
    chain = compose(listed_backwards, COLUMNS, PRICES)
    frame = chain({"kg_lt": jnp.asarray([1.0])}, {}, {})
    assert frame["kg_lt"] == pytest.approx(jnp.asarray([4.0]))

    families = [
        entry["manifest"]["family"] for entry in order_modules(listed_backwards)
    ]
    assert families == ["conversion", "entity_shifting"]


def test_a_family_the_order_does_not_name_stops_the_run():
    """The pinned order is the whole list of families, so there is no room for
    one it does not name."""

    entries = [
        {
            "function": add_module(0.0),
            "manifest": manifest("newcomer", "something_new"),
            "params": {},
        },
        {
            "function": add_module(0.0),
            "manifest": manifest("charity", "charity"),
            "params": {},
        },
    ]
    with pytest.raises(ManifestError, match="something_new"):
        order_modules(entries)


def test_a_scenario_parameter_value_beats_the_module_default():
    """A module's numbers arrive when the chain is called, not when it is built.

    The entry carries the module's default; the scenario's own value
    for the same parameter is laid over it at call time.
    """

    def scale_gains(records, prices_reform, prices_baseline, params, draws):
        del prices_reform, prices_baseline, draws
        return {"kg_lt": jnp.asarray(records["kg_lt"]) * params["factor"]}

    entries = [
        {
            "function": scale_gains,
            "manifest": manifest("scaler", "charity", writes=("kg_lt",)),
            "params": {"factor": 1.0},
        }
    ]
    chain = compose(entries, COLUMNS, PRICES)
    records = {"kg_lt": jnp.asarray([10.0])}

    at_the_default = chain(records, {}, {})
    assert at_the_default["kg_lt"] == pytest.approx(jnp.asarray([10.0]))

    with_the_scenarios_number = chain(records, {}, {}, {"charity": {"factor": 3.0}})
    assert with_the_scenarios_number["kg_lt"] == pytest.approx(jnp.asarray([30.0]))
