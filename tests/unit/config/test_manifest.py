"""The run manifest: enough to read a past estimate back."""

from __future__ import annotations

import json
import subprocess

from taxmodel.config import (
    build_manifest,
    check_leg,
    interface_vintages,
    load_defaults,
    resolve,
    resolve_law,
    resolve_law_stack,
    write_manifest,
)


def resolved(tree, alternative=None):
    layer = load_defaults(tree.root, "economy", base_dir=tree.base_dir)
    return resolve(tree.root, layer, alternative)


def test_the_manifest_records_every_resolved_value_and_who_set_it(tree):
    tree.write(
        "config/scenarios/economy/alternatives/high_sigma/corp.yaml",
        {"sigma_n": {"value": 0.5}},
    )
    economy = resolved(tree, "high_sigma")
    manifest = build_manifest("reform_a", economy=economy, years=[2026, 2027])

    rows = {(r["channel"], r["name"]): r for r in manifest["config"]}
    assert rows[("corp", "sigma_n")]["value"] == 0.5
    assert rows[("corp", "sigma_n")]["overridden"] is True
    assert rows[("corp", "sigma_n")]["source"] == "alternative:high_sigma"
    assert rows[("corp", "sigma_n")]["role"] == "transmission"
    assert rows[("gains", "timeable_share")]["kind"] == "calibrated"
    assert rows[("gains", "timeable_share")]["overridden"] is False
    assert manifest["legs"]["economy"] == "high_sigma"
    assert manifest["years"] == [2026, 2027]


def law_with_two_provisions(tree):
    """A stacked law where the two provisions fight over one subparameter."""
    tree.write(
        "config/scenarios/tax_law/default/ctc.yaml",
        {"value": {"value": {"2018": 2000}}, "po_thresh": {"value": {"2018": 200000}}},
    )
    tree.write(
        "config/scenarios/tax_law/alternatives/p1/ctc.yaml",
        {"value": {"value": {"2026": 3000}}, "po_thresh": {"value": {"2026": 150000}}},
    )
    tree.write(
        "config/scenarios/tax_law/alternatives/p2/ctc.yaml",
        {"value": {"value": {"2026": 4000}}},
    )
    return resolve_law_stack(tree.root, ["p1", "p2"])[-1]


def test_the_manifest_names_the_provision_that_wrote_every_replaced_subparameter(tree):
    law = law_with_two_provisions(tree)
    manifest = build_manifest("package", law=law)

    rows = [
        (r["subparameter"], r["written_by"], r["in_force"])
        for r in manifest["law_replacements"]
    ]
    assert rows == [
        # The first provision wrote both blocks; the second overwrote the
        # credit's value, so the first provision's copy is no longer in force.
        ("value", "p1", False),
        ("po_thresh", "p1", True),
        ("value", "p2", True),
    ]
    assert manifest["law_layers"] == ["default", "p1", "p2"]


def test_the_manifest_records_the_subparameters_two_provisions_both_wrote(tree):
    """The later folder wins and both replacements are recorded."""
    manifest = build_manifest("package", law=law_with_two_provisions(tree))
    assert manifest["law_conflicts"] == {"ctc.value": ["p1", "p2"]}


def test_a_single_reform_has_no_conflicts_and_one_layer_over_baseline(tree):
    tree.write(
        "config/scenarios/tax_law/default/ctc.yaml", {"value": {"value": {"2018": 2000}}}
    )
    tree.write(
        "config/scenarios/tax_law/alternatives/bigger/ctc.yaml",
        {"value": {"value": {"2026": 3000}}},
    )
    manifest = build_manifest("reform", law=resolve_law(tree.root, "bigger"))

    assert manifest["law_conflicts"] == {}
    assert manifest["law_layers"] == ["default", "bigger"]
    (row,) = manifest["law_replacements"]
    assert (row["written_by"], row["in_force"]) == ("bigger", True)


def test_the_manifest_records_the_code_version_and_whether_the_tree_was_dirty(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "law.yaml").write_text("rate: 0.37\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "law"], cwd=repo, check=True)

    clean = build_manifest("baseline", repo_root=repo)["code_version"]
    assert len(clean["commit"]) == 40
    assert clean["dirty"] is False

    # An estimate produced from uncommitted code is not reproducible, and the
    # manifest has to say so.
    (repo / "law.yaml").write_text("rate: 0.39\n")
    dirty = build_manifest("baseline", repo_root=repo)["code_version"]
    assert dirty["dirty"] is True
    assert dirty["dirty_files"] == ["law.yaml"]


def test_the_manifest_records_the_runtime_stack_and_what_the_guard_checked(tree):
    manifest = build_manifest("baseline")
    assert manifest["runtime"]["jax"] == "0.7.0"
    assert manifest["runtime"]["python"].startswith("3.12")
    assert manifest["runtime"]["numpy"]
    assert manifest["runtime"]["platform"]
    assert manifest["toolchain"]["pinned"]["jax"] == "0.7.0"
    assert manifest["toolchain"]["matches_pins"] is True


def test_the_manifest_records_staleness_findings_including_warn_level(tree):
    tree.write("calibrations/widowhood.yaml", {"share_65_74": 0.44})
    economy = resolved(tree)
    report = check_leg(economy, interface_vintages(economy))
    manifest = build_manifest("baseline", economy=economy, staleness=report)
    assert any("estate.widowhood_share" in w for w in manifest["staleness"]["warnings"])


def test_the_manifest_is_written_as_one_readable_file(tree, tmp_path):
    economy = resolved(tree)
    path = write_manifest(
        tmp_path / "out" / "manifest.json",
        build_manifest("baseline", economy=economy, vintages=interface_vintages(economy)),
    )
    written = json.loads(path.read_text())
    assert written["interface_vintages"] == {"tax_data": "2030010100"}
