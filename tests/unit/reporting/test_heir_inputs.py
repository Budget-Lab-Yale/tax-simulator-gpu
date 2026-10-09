"""Reading the upstream heir structure and laying it on the run's records."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from taxmodel.reporting import heir_inputs


def write_year(root, year, ids, chance, amount):
    pd.DataFrame(
        {
            "id": ids,
            "p_inheritance": chance,
            "inheritance": amount,
            # The publication also carries its own estate tax figure, which
            # this model computes for itself and so does not read.
            "estate_tax_liability": np.zeros(len(ids)),
        }
    ).to_csv(heir_inputs.year_file(root, year), index=False)


def test_the_two_columns_land_on_the_slots_their_record_ids_name(tmp_path):
    write_year(tmp_path, 2026, [30, 10, 20], [0.5, 0.1, 0.2], [900.0, 100.0, 400.0])
    write_year(tmp_path, 2027, [10, 20, 30], [0.1, 0.2, 0.5], [110.0, 440.0, 990.0])

    structure = heir_inputs.read_heir_structure(
        tmp_path, [2026, 2027], np.array([10, 20, 30])
    )

    assert structure.probability_of_inheriting.tolist() == [
        [0.1, 0.2, 0.5],
        [0.1, 0.2, 0.5],
    ]
    assert structure.inheritance.tolist() == [
        [100.0, 400.0, 900.0],
        [110.0, 440.0, 990.0],
    ]
    assert structure.unmatched_by_year == {2026: 0, 2027: 0}


def test_a_record_the_publication_never_mentions_is_a_non_heir(tmp_path):
    """Unmentioned records are non-heirs, and the description says how many."""
    write_year(tmp_path, 2026, [10, 20], [0.5, 0.2], [900.0, 400.0])

    structure = heir_inputs.read_heir_structure(
        tmp_path, [2026], np.array([10, 20, 99])
    )

    assert structure.probability_of_inheriting[0, 2] == 0.0
    assert structure.inheritance[0, 2] == 0.0
    assert structure.unmatched_by_year == {2026: 1}
    assert "1 of the run's records" in heir_inputs.describe(structure)


def test_a_year_the_publication_does_not_cover_has_nobody_inheriting(tmp_path):
    """A year before the publication starts is read as nobody inheriting.

    The pinned publication starts in 2026, so a run that starts in 2025 has
    no heir file for its first year.
    """
    write_year(tmp_path, 2026, [10], [0.5], [900.0])

    structure = heir_inputs.read_heir_structure(
        tmp_path, [2025, 2026], np.array([10])
    )

    assert structure.zero_filled_years == (2025,)
    assert structure.probability_of_inheriting[0, 0] == 0.0
    assert structure.inheritance[0, 0] == 0.0
    # The covered year is read as it always was.
    assert structure.probability_of_inheriting[1, 0] == 0.5
    assert structure.inheritance[1, 0] == 900.0
    assert "nobody inherits anything in those years" in heir_inputs.describe(structure)


def test_a_chance_of_inheriting_above_one_is_passed_through_as_published(tmp_path):
    """The pinned vintage publishes values above one (37 records).

    The weight is multiplied by whatever the file says.
    """
    write_year(tmp_path, 2026, [10], [2.03], [900.0])

    structure = heir_inputs.read_heir_structure(tmp_path, [2026], np.array([10]))

    assert structure.probability_of_inheriting[0, 0] == pytest.approx(2.03)
