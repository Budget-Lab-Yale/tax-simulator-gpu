"""The runscript: six columns, folder and file names, and no fallbacks."""

from __future__ import annotations

import pytest

from taxmodel.config import (
    ConfigError,
    TaxLawKind,
    classify_tax_law,
    classify_tax_law_cells,
    parse_year_spec,
    read_runscript,
    runscript,
)

# The runscript's header row, spelled the way the reader defines the columns.
HEADER = ",".join(runscript.COLUMNS) + "\n"


def write_runscript(tmp_path, text, name="runscript.csv"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_a_well_formed_runscript_reads(tmp_path):
    path = write_runscript(
        tmp_path,
        HEADER
        + "baseline,default,default,default,2026:2028,\n"
        + "top_rate,top_tax/five_lever,default,sigma_calib_100,2026:2035,ctc_audit.yaml\n",
    )
    rows = read_runscript(path)
    assert [r.id for r in rows] == ["baseline", "top_rate"]
    assert rows[0].years == [2026, 2027, 2028]
    assert rows[1].tax_law == "top_tax/five_lever"
    assert rows[1].detail == "ctc_audit.yaml"
    assert rows[0].detail == ""


def test_omitted_leg_columns_mean_the_default_layer(tmp_path):
    path = write_runscript(tmp_path, "ID,tax_law,years\nbaseline,default,2026\n")
    (row,) = read_runscript(path)
    assert (row.economy, row.behavior) == ("default", "default")


def test_an_unknown_column_stops_with_the_list_of_columns(tmp_path):
    path = write_runscript(tmp_path, "ID,tax_law,years,notes\nbaseline,default,2026,hi\n")
    with pytest.raises(ConfigError) as caught:
        read_runscript(path)
    message = str(caught.value)
    assert "notes : unknown column" in message
    assert ", ".join(runscript.COLUMNS) in message


def test_every_bad_column_is_reported_at_once(tmp_path):
    path = write_runscript(
        tmp_path,
        "ID,tax_law,years,s,excess_growth,first_year\nbaseline,default,2026,0.5,0.01,2026\n",
    )
    with pytest.raises(ConfigError) as caught:
        read_runscript(path)
    message = str(caught.value)
    assert "s :" in message
    assert "excess_growth :" in message
    assert "first_year :" in message


def test_a_missing_required_column_stops(tmp_path):
    path = write_runscript(tmp_path, "ID,economy,years\nbaseline,default,2026\n")
    with pytest.raises(ConfigError, match="missing required column"):
        read_runscript(path)


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("2026", [2026]),
        ("2026:2029", [2026, 2027, 2028, 2029]),
        ("2026 2030", [2026, 2030]),
        ("2030 2026:2027", [2026, 2027, 2030]),
        ("2026 2026", [2026]),
    ],
)
def test_year_specifications(spec, expected):
    assert parse_year_spec(spec) == expected


@pytest.mark.parametrize("spec", ["", "20x6", "2026:", "2026:2027:2028"])
def test_malformed_year_specifications_stop(spec):
    with pytest.raises(ConfigError):
        parse_year_spec(spec)


# --------------------------------------------------------------------------
# The tax_law cell: baseline law, one reform folder, or a stack file
# --------------------------------------------------------------------------


@pytest.fixture
def law_tree(tmp_path):
    """A scenario configuration tree holding one reform folder and one stack file."""
    alternatives = tmp_path / "tax_law" / "alternatives"
    (alternatives / "top_tax" / "five_lever").mkdir(parents=True)
    (alternatives / "ctc").mkdir()
    (alternatives / "ctc" / "decomposition.yaml").write_text("provisions: []\n")
    (alternatives / "ctc" / "notes.txt").write_text("not a stack file\n")
    return tmp_path


def test_a_directory_in_the_tax_law_cell_is_one_reform(law_tree):
    target = classify_tax_law("top_tax/five_lever", law_tree)
    assert target.kind is TaxLawKind.REFORM_FOLDER
    assert not target.is_stack
    assert target.path == law_tree / "tax_law" / "alternatives" / "top_tax" / "five_lever"


def test_a_yaml_file_in_the_tax_law_cell_is_a_stack(law_tree):
    target = classify_tax_law("ctc/decomposition.yaml", law_tree)
    assert target.kind is TaxLawKind.STACK_FILE
    assert target.is_stack


def test_the_reserved_word_default_is_baseline_law(law_tree):
    target = classify_tax_law("default", law_tree)
    assert target.kind is TaxLawKind.BASELINE
    assert target.path is None


def test_a_tax_law_cell_naming_nothing_is_refused(law_tree):
    with pytest.raises(ConfigError, match="neither a reform folder nor a stack file"):
        classify_tax_law("top_tax/five_levers", law_tree)


def test_a_file_that_is_not_a_stack_file_is_refused(law_tree):
    with pytest.raises(ConfigError, match="a file that is not a stack file"):
        classify_tax_law("ctc/notes.txt", law_tree)


def test_every_unresolvable_tax_law_cell_is_reported_at_once(tmp_path, law_tree):
    path = write_runscript(
        tmp_path,
        HEADER
        + "baseline,default,default,default,2026,\n"
        + "typo_one,top_tax/five_levers,default,default,2026,\n"
        + "typo_two,ctc/decompositon.yaml,default,default,2026,\n",
    )
    with pytest.raises(ConfigError) as caught:
        classify_tax_law_cells(read_runscript(path), law_tree)
    message = str(caught.value)
    assert "typo_one" in message
    assert "typo_two" in message
