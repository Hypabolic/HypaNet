"""scripts/freeze.py: what it records, --write / --check / --revise / --dry-run, stale labels, warnings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import build_sets as bs
from scripts import freeze

SLOP = Path(__file__).resolve().parents[2]


def test_hashes_match_the_ones_their_consumers_check():
    # build_sets.py checks registry extractor.sha256 before opening a sealed repo
    assert freeze.extractor_sha256(SLOP) == bs.extractor_sha256()
    # generator/build.py checks registry generator.sha256 before building test_gen
    from generator import build as gen

    assert freeze.generator_sha256(SLOP) == gen.generator_sha256()
    assert freeze.extractor_src_sha256(SLOP) == gen.extractor_sha256()


def test_compute_records_every_component(fake_slop):
    rec = freeze.compute(fake_slop)
    for k in ("spec", "guide", "extractor", "tokeniser", "model", "runtime", "rule_layer", "generator", "harness",
              "env", "sets", "labels", "disclosed", "evaluation", "comparisons", "bootstrap"):
        assert k in rec
    assert rec["extractor"]["sha256"] == bs.extractor_sha256()
    assert rec["kit"] is None and "kit" in rec["pending"]
    assert rec["sets"]["sealed_real"] is None and "sets.sealed_real" in rec["pending"]
    assert "sets.test_gen" in rec["pending"] and "disclosed" in rec["pending"]
    assert set(rec["env"]["versions"]) == {"python", "numpy", "node"}
    assert "ui/model.py" in rec["model"]["files"]
    assert "scripts/freeze.py" in rec["harness"]["files"]


def _labels(root: Path, guide_sha: str, name="reference_real.mg.jsonl", n=3):
    d = root / "labels"
    d.mkdir(exist_ok=True)
    rows = [{"screen": f"s{i}", "loc": [1, 1, 1, 2], "class": "KEEP", "labeller": "mg", "guide_sha256": guide_sha,
             "time": "2026-11-12T10:00:00Z"} for i in range(n)]
    (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows))


def _manifest(root: Path, name: str, **extra):
    d = root / "datasets" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps({"set": name, "screens_total": 2, "screens": [], **extra}) + "\n")


def test_write_check_and_dry_run(fake_slop, capsys):
    assert freeze.main(["--root", str(fake_slop), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "extractor:" in out and "pending:" in out
    assert not (fake_slop / "registry.yaml").exists()
    rec = freeze.write(fake_slop, now="2026-11-12T00:00:00Z")
    assert (fake_slop / "registry.yaml").is_file()
    assert freeze.check(fake_slop)[0] == []
    problems, notes = freeze.check(fake_slop, strict=True)
    assert any("pending" in p for p in problems)
    with pytest.raises(freeze.FreezeError, match="exists"):
        freeze.write(fake_slop)
    # same input, same bytes
    text = (fake_slop / "registry.yaml").read_text()
    freeze.write(fake_slop, force=True, now="2026-11-12T00:00:00Z")
    assert (fake_slop / "registry.yaml").read_text() == text
    assert rec["freeze_sha256"] == freeze.load(fake_slop)["freeze_sha256"]


def test_check_names_changes_and_edits(fake_slop):
    freeze.write(fake_slop)
    (fake_slop / "ui" / "model.py").write_text((fake_slop / "ui" / "model.py").read_text() + "\n# changed\n")
    problems, _ = freeze.check(fake_slop)
    assert any(p.startswith("model: changed") and "ui/model.py" in p for p in problems)
    reg = fake_slop / "registry.yaml"
    reg.write_text(reg.read_text().replace("schema: slop.registry/1", "schema: slop.registry/1\nextra: 1"))
    assert any("freeze_sha256" in p for p in freeze.check(fake_slop)[0])


def test_stale_labels_block_write_and_check(fake_slop):
    _labels(fake_slop, "0" * 64)
    with pytest.raises(freeze.FreezeError, match="another guide"):
        freeze.write(fake_slop)
    guide = freeze.sha256_file(fake_slop / "LABELLING.md")
    _labels(fake_slop, guide)
    rec = freeze.write(fake_slop)
    assert "reference_real.mg.jsonl" in rec["labels"]
    assert freeze.check(fake_slop)[0] == []
    _labels(fake_slop, "0" * 64, name="calibration.mg.jsonl")
    problems, notes = freeze.check(fake_slop)
    assert any("calibration.mg.jsonl" in p and "guide_sha256" in p for p in problems)


def test_warnings_for_sets_built_by_another_extractor(fake_slop):
    _manifest(fake_slop, "reference_real", extractor_sha256="f" * 64)
    with pytest.raises(freeze.FreezeError, match="consistency"):
        freeze.write(fake_slop)
    rec = freeze.write(fake_slop, accept_warnings=True)
    assert "accept-warnings" in rec["note"]


def test_revise_adds_pending_and_refuses_silent_changes(fake_slop):
    freeze.write(fake_slop)
    with pytest.raises(freeze.FreezeError, match="nothing to revise"):
        freeze.revise(fake_slop, "nothing", [])
    _manifest(fake_slop, "sealed_real")
    problems, notes = freeze.check(fake_slop)
    assert problems == [] and any(n.startswith("sets.sealed_real: now present") for n in notes)
    rec, rev = freeze.revise(fake_slop, "sealed_real selected", [])
    assert rev["added"] == ["sets.sealed_real"] and rec["sets"]["sealed_real"]["screens"] == 2
    assert freeze.check(fake_slop)[0] == []
    (fake_slop / "SPEC.md").write_text((fake_slop / "SPEC.md").read_text() + "\nedit\n")
    _labels(fake_slop, freeze.sha256_file(fake_slop / "LABELLING.md"), name="sealed_real.mg.jsonl")
    with pytest.raises(freeze.FreezeError, match="only adds: changed spec"):
        freeze.revise(fake_slop, "labels", [])
    rec, rev = freeze.revise(fake_slop, "sealed labels; spec typo", ["spec"])
    assert rev["changed"] == ["spec"] and rev["added"] == ["labels.sealed_real.mg.jsonl"]
    assert len(rec["revisions"]) == 2 and rec["revisions"][1]["previous_freeze_sha256"]
    assert freeze.check(fake_slop)[0] == []


def test_deep_check_reads_the_data_behind_manifests(fake_slop):
    d = fake_slop / "datasets" / "dev"
    d.mkdir(parents=True)
    (d / "screens.jsonl").write_text("{}\n")
    (d / "manifest.json").write_text(json.dumps({"set": "dev", "screens": 1}) + "\n")
    (d / "SHA256SUMS").write_text(f"{freeze.sha256_file(d / 'screens.jsonl')}  screens.jsonl\n")
    freeze.write(fake_slop)
    assert freeze.check(fake_slop, deep=True)[0] == []
    (d / "screens.jsonl").write_text("{\"x\": 1}\n")
    assert any("differs from SHA256SUMS" in p for p in freeze.check(fake_slop, deep=True)[0])


def test_kit_and_disclosed_come_from_the_hand_over_record(fake_slop):
    (fake_slop / "kit").mkdir()
    (fake_slop / "kit" / "KIT.md").write_text("kit\n")
    (fake_slop / "kit" / "settings.json").write_text("{}\n")  # not frozen: launch.sh checks it
    rec = freeze.compute(fake_slop)
    assert rec["kit"] is None and rec["disclosed"] is None  # no hand-over yet
    hand = fake_slop / "runs" / "r1" / "HANDOVER.json"
    hand.parent.mkdir(parents=True)
    hand.write_text(json.dumps({"kind": "hypanet-slop-author", "test": False, "examples": {
        "selection": {"n": 400, "rule": "rank", "seed": 20261112, "namespace": "disclose"},
        "dev_ids_sha256": "a" * 64}}))
    rec = freeze.compute(fake_slop)
    assert rec["kit"]["handover_sha256"] == freeze.sha256_file(hand)
    assert list(rec["kit"]["files"]) == ["kit/KIT.md"]
    assert rec["disclosed"] == {"dev_ids_sha256": "a" * 64, "count": 400, "rule": "rank", "seed": 20261112,
                                "namespace": "disclose"}


def test_model_labeller_raw_files_are_hashed_per_set(fake_slop):
    raw = fake_slop / "labels" / "raw" / "reference_real" / "opus55" / "label"
    raw.mkdir(parents=True)
    (raw / "s1.json").write_text("{}\n")
    (fake_slop / "labels" / "raw" / "spend.jsonl").write_text("{}\n")
    rec = freeze.write(fake_slop)
    assert rec["labels"]["raw/reference_real"]["files"] == 1 and "raw/spend.jsonl" not in rec["labels"]
    (fake_slop / "labels" / "raw" / "spend.jsonl").write_text("{}\n{}\n")  # spend accounting may grow
    assert freeze.check(fake_slop)[0] == []
    (raw / "s1.json").write_text("{\"x\": 1}\n")
    assert any(p.startswith("labels.raw/reference_real: changed") for p in freeze.check(fake_slop)[0])
