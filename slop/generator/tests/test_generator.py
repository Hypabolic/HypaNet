"""Dev generator: determinism, the span join, label validity and the test_gen guard (SPEC.md §9, §12.4)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SLOP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SLOP))
sys.path.insert(0, str(SLOP / "labeller"))

from generator import build as gb  # noqa: E402
from generator.core import PATTERN_CLASS, PLANTABLE, PLANTABLE_K, E, Page, Rng, S  # noqa: E402
from generator.join import join_screen, run_extractor  # noqa: E402
from generator.render_tsx import TsxRenderer  # noqa: E402

N = 400


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    a = tmp_path_factory.mktemp("gen_a")
    b = tmp_path_factory.mktemp("gen_b")
    ma = gb.build("val", N, a, log=lambda *_: None)
    mb = gb.build("val", N, b, log=lambda *_: None)
    return a, b, ma, mb


def test_same_seed_same_bytes(built):
    a, b, ma, mb = built
    names = sorted(p.name for p in a.iterdir())
    assert names == sorted(p.name for p in b.iterdir())
    assert set(names) == {"screens.jsonl", "val.gen.jsonl", "sources.jsonl", "manifest.json", "SHA256SUMS"}
    for n in names:
        assert (a / n).read_bytes() == (b / n).read_bytes(), n


def test_namespaces_differ(tmp_path):
    m = gb.build("dev", 20, tmp_path, log=lambda *_: None)
    v = [json.loads(x)["id"] for x in (tmp_path / "screens.jsonl").read_text().splitlines()]
    m2 = gb.build("val", 20, tmp_path / "v", log=lambda *_: None)
    w = [json.loads(x)["id"] for x in (tmp_path / "v" / "screens.jsonl").read_text().splitlines()]
    assert not set(v) & set(w)
    assert m["files"]["screens.jsonl"] != m2["files"]["screens.jsonl"]


def test_every_node_labelled_and_checker_passes(built):
    import check as lcheck

    a = built[0]
    res = lcheck.check(a / "val.gen.jsonl", a, SLOP / "LABELLING.md")
    assert res["errors"] == []
    assert res["complete"] == N
    assert set(res["guide_hashes"]) == {gb.sha256_file(SLOP / "LABELLING.md")}


def test_every_pattern_and_k_rule_planted(built):
    counts = built[2]["counts"]
    assert counts["patterns_missing"] == []
    assert counts["k_rules_missing"] == []
    for p in PLANTABLE:
        assert counts["patterns"][f"{PATTERN_CLASS[p]} {p}"] > 0
    for k in PLANTABLE_K:
        assert counts["k_rules"][k] > 0
    assert counts["clean_screens"] > 0


def test_records_match_planted_slots(built):
    a = built[0]
    screens = {json.loads(x)["id"]: json.loads(x) for x in (a / "screens.jsonl").read_text().splitlines()}
    for line in (a / "val.gen.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "loc" not in r:
            continue
        locs = {tuple(n["loc"]) for n in screens[r["screen"]]["nodes"]}
        assert tuple(r["loc"]) in locs
        if r["target"] is not None:
            assert tuple(r["target"]) in locs and r["target"] != r["loc"]
        assert r["labeller"] == "gen" and r["time"] == gb.GEN_TIME
        if r["class"] in ("ECHO", "NARRATE", "DECOR"):
            assert PATTERN_CLASS[r["pattern"]] == r["class"]


def test_join_fails_loudly_on_a_moved_span():
    h = S("Settings", "h1")
    sub = S("Manage your settings here.", "subtitle").echo(h, "R3")
    page = Page("Settings", "app", [E("head", E("h", s=h, level=1), E("sub", s=sub))])
    src, css = TsxRenderer(Rng(1), "tailwind").render(page)
    screen = run_extractor([{"repo": "t", "path": "t/Settings.tsx", "src": src, "css": css}])[0]["screen"]
    assert join_screen(screen, page.slots())["problems"] == []
    l1, c1, l2, c2 = sub.span
    sub.span = (l1, c1 + 1, l2, c2)
    kinds = {k for k, _ in join_screen(screen, page.slots())["problems"]}
    assert "unmatched_planted" in kinds and "node_without_slot" in kinds


def test_test_gen_refuses_without_registry(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(gb, "REGISTRY", tmp_path / "registry.yaml")
    assert gb.main(["test_gen", "--n", "5", "--out", str(tmp_path / "out")]) == 2
    (tmp_path / "registry.yaml").write_text("generator:\n  sha256: " + "0" * 64 + "\n")
    assert gb.main(["test_gen", "--n", "5", "--out", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()
    assert "refusing" in capsys.readouterr().err


def test_duplicate_check():
    s1 = {"nodes": [{"text": "Paused"}, {"text": "Resume"}, {"text": "Settings"}, {"text": "Quit"}, {"text": "Audio"}]}
    s2 = {"nodes": [{"text": "PAUSED"}, {"text": "Resume"}, {"text": "Settings"}, {"text": "Quit"}, {"text": "Audio"},
                    {"text": "Controls"}]}
    s3 = {"nodes": [{"text": "Invoices"}, {"text": "New invoice"}]}
    d = gb.Dupes([s1])
    assert d.hit(s2)  # Jaccard 5/6 after normalising case
    assert not d.hit(s3)
