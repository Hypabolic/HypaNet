"""The real-set builder (SPEC.md §9): exclusion rules, hash rank, dedupe, the repo cap and the sealed gate."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import build_sets as bs


def _cand(repo, path, texts, roles=None):
    nodes = [{"i": i, "role": (roles or {}).get(i, "body"), "text": t, "flags": [], "loc": [i + 1, 1, i + 1, 2]}
             for i, t in enumerate(texts)]
    screen = {"schema": "slop.screen/1", "id": f"{repo}:{path}", "nodes": nodes}
    c = bs.Candidate(repo, path, None, "clean", Path(path), "", screen, b"")
    c.rank = bs.rank_hash("t", repo, path)
    c.texts_hash, c.texts = bs.text_hash(screen), bs.text_set(screen)
    return c


def test_rank_hash_and_norm():
    assert bs.rank_hash("reference", "R", "a.tsx") == bs.sha256_bytes(b"reference:20261116:R/a.tsx")
    assert bs.rank_hash("reference", "R", "a.tsx") != bs.rank_hash("calibration", "R", "a.tsx")
    assert bs.norm_text("  Save   CHANGES!  ") == "save changes"
    assert bs.norm_text(" user_name  Ｏff") == " user name  off"


def test_labelable_count_skips_context():
    c = _cand("R", "a", ["Save", "Save", "x"], roles={1: "aria"})
    assert bs.labelable_count(c.screen) == 2
    c.screen["nodes"][2]["flags"] = ["sr_only"]
    assert bs.labelable_count(c.screen) == 1


def test_dedupe_exact_jaccard_and_earlier_sets():
    ref = [_cand("A", "r", ["Settings", "Save", "Cancel", "Delete account", "Sign out"])]
    cands = sorted([
        _cand("B", "exact", ["settings", "SAVE", "Cancel", "Delete account", "Sign out!"]),
        _cand("B", "near", ["Settings", "Save", "Cancel", "Delete account", "Sign out", "Help"]),
        _cand("B", "new", ["Recipes", "Add recipe"]),
        _cand("B", "new-again", ["Recipes", "Add recipe"]),
    ], key=lambda c: c.rank)
    drops, per = {}, {"A": {"drops": {}}, "B": {"drops": {}}}
    kept = bs.dedupe(cands, ref, drops, per)
    assert [c.path for c in kept] in (["new"], ["new-again"])
    assert drops == {"dedupe_exact": 2, "dedupe_jaccard": 1}


def test_select_caps_and_counts():
    cands = sorted([_cand("A", f"a{i}", ["x"] * 10) for i in range(10)] + [_cand("B", f"b{i}", ["x"] * 2)
                                                                            for i in range(10)],
                   key=lambda c: c.rank)
    drops, per = {}, {"A": {"drops": {}}, "B": {"drops": {}}}
    sel = bs.select(cands, 10, {"measure": "nodes", "share": 0.5}, drops, per)
    a = sum(c.nodes for c in sel if c.repo == "A")
    assert a <= 0.5 * sum(c.nodes for c in sel)
    assert [c.rank for c in sel] == sorted(c.rank for c in sel)
    assert drops.get("repo_cap", 0) + drops.get("below_cut", 0) + len(sel) == 20
    drops, per = {}, {"A": {"drops": {}}, "B": {"drops": {}}}
    assert len(bs.select(cands, 30, {"measure": None, "share": 1.0}, drops, per)) == 20 and not drops


def test_select_dir_cap_in_rank_order():
    cands = sorted([_cand("A", f"figs/f{i}.tsx", ["x", "y"]) for i in range(10)]
                   + [_cand("A", f"pages/p{i}.tsx", ["x", "y"]) for i in range(10)], key=lambda c: c.rank)
    drops, per = {}, {"A": {"drops": {}}}
    sel = bs.select(cands, 12, {"measure": None, "share": 1.0, "dir_cap": 6}, drops, per)
    assert len(sel) == 12
    assert sum(c.directory == "figs" for c in sel) == 6 and sum(c.directory == "pages" for c in sel) == 6
    figs = [c for c in cands if c.directory == "figs"]
    assert {c.path for c in sel if c.directory == "figs"} == {c.path for c in figs[:6]}
    assert drops.get("dir_cap", 0) + drops.get("below_cut", 0) + len(sel) == 20
    assert bs.RULES.index("dir_cap") < bs.RULES.index("repo_cap")


def test_calibration_has_the_reference_node_cap():
    assert (bs.CALIBRATION_RULE["measure"], bs.CALIBRATION_RULE["share"]) == ("nodes", 0.4)
    assert bs.REFERENCE_RULE["dir_cap"] == 6 and bs.CALIBRATION_RULE["dir_cap"] == 6
    assert "of the 50 repo screens" not in bs.REFERENCE_RULE["text"]
    big = [_cand("W", f"w{i}", ["x"] * 150) for i in range(4)]
    small = [_cand(r, f"{r}{i}", ["x"] * 10) for r in "BCD" for i in range(3)]
    cands = sorted(big + small, key=lambda c: c.rank)
    drops, per = {}, {r: {"drops": {}} for r in "WBCD"}
    sel = bs.select(cands, 40, bs.CALIBRATION_RULE, drops, per)
    w = sum(c.nodes for c in sel if c.repo == "W")
    assert w <= 0.4 * sum(c.nodes for c in sel) and drops["repo_cap"] == 4 and len(sel) == 9


def _git_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "repo"
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    tracked = [r for r in files if not r.startswith("local/")]
    subprocess.run(["git", "-C", str(root), "add", "-f", *tracked], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "t"],
                   check=True)
    return root


def test_classify_rules(tmp_path):
    shared = "export const X = () => <h1>Shared</h1>;\n"
    root = _git_repo(tmp_path, {
        "src/page.tsx": "export const P = () => <h1>Page</h1>;\n",
        "src/components/ui/button.tsx": "x\n",
        "app.worktrees/x/page.tsx": "x\n",
        "DesignRefs/hero.html": "<h1>a</h1>\n",
        "design_handoff_v2/hero.html": "<h1>b</h1>\n",
        "DesignRefs/other.html": "<h1>c</h1>\n",
        "plan/prototype/page.tsx": "y\n",
        "plan/prototype/extra.tsx": "z\n",
        "src/page.test.tsx": "t\n",
        "components/blog-figures/f1-figure.tsx": "f\n",
        "AppTests/ViewTests.swift": "t\n",
        "Sources/ThingTests.swift": "t\n",
        "src/shared.tsx": shared,
        "local/scratch.tsx": "u\n",
        "notes.md": "not ui\n",
        "node_modules/m/index.js": "m\n",
        ".env.local.js": "secret\n",
        "lib/app.min.js": "m\n",
    })
    entries, commit = bs.classify(bs.Repo("R", root, "private", "", ""), {bs.sha256_bytes(shared.encode())})
    rule = {e.path: e.rule for e in entries}
    assert commit and len(commit) == 40
    assert rule == {
        "AppTests/ViewTests.swift": "test_spec_story",
        "DesignRefs/hero.html": "design_refs",
        "DesignRefs/other.html": None,
        "Sources/ThingTests.swift": "test_spec_story",
        "app.worktrees/x/page.tsx": "worktrees",
        "components/blog-figures/f1-figure.tsx": "blog_figures",
        "design_handoff_v2/hero.html": None,
        "lib/app.min.js": "not_source",
        "local/scratch.tsx": "untracked",
        "plan/prototype/extra.tsx": None,
        "plan/prototype/page.tsx": "plan_prototype",
        "src/components/ui/button.tsx": "vendored",
        "src/page.test.tsx": "test_spec_story",
        "src/page.tsx": None,
        "src/shared.tsx": "byte_identical_sealed",
    }
    (root / "src/page.tsx").write_text("changed\n")
    entries, _ = bs.classify(bs.Repo("R", root, "private", "", ""), set())
    assert {e.path: e.git for e in entries}["src/page.tsx"] == "modified"


def test_sealed_refuses_until_frozen(tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "REGISTRY", tmp_path / "registry.yaml")
    with pytest.raises(SystemExit, match="does not exist"):
        bs.frozen_extractor()
    (tmp_path / "registry.yaml").write_text("spec: {}\n")
    with pytest.raises(SystemExit, match="no extractor.sha256"):
        bs.frozen_extractor()
    (tmp_path / "registry.yaml").write_text("extractor: {sha256: '0000'}\n")
    with pytest.raises(SystemExit, match="is not the frozen one"):
        bs.frozen_extractor()
    (tmp_path / "registry.yaml").write_text(f"extractor: {{sha256: '{bs.extractor_sha256()}'}}\n")
    assert bs.frozen_extractor() == bs.extractor_sha256()


def test_sealed_command_checks_gate_first(monkeypatch):
    """`sealed` (now scripts/build_sealed.py) must refuse before it touches any sealed repo."""
    from scripts import build_sealed

    def boom(*a, **k):
        raise AssertionError("a sealed repo was opened before the gate")

    monkeypatch.setattr(build_sealed, "gate", lambda root: (_ for _ in ()).throw(SystemExit("refusing: test")))
    monkeypatch.setattr(build_sealed.bs, "gather", boom)
    monkeypatch.setattr(build_sealed.bs, "walk_ui_files", boom)
    monkeypatch.setattr(build_sealed, "sealed_repos", boom)
    with pytest.raises(SystemExit, match="refusing"):
        bs.main(["sealed"])
