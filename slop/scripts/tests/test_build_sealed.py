"""scripts/build_sealed.py: refuses before the freeze, opens nothing before its gate, and selects sealed_real by the
SPEC.md §9 rules into a private set plus a committable manifest of ids and hashes only. Every repo here is a fake one
under the test's temporary directory, with invented copy."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import build_sealed, freeze

bs = build_sealed.bs
WORDS = ["orchard", "lantern", "harbour", "meadow", "quarry", "glacier", "canyon", "thicket", "estuary", "summit",
         "pasture", "foundry", "atelier", "granary", "lagoon", "spinney"]


def _tsx(i: int, repo: str) -> str:
    a, b, c = WORDS[i % 16], WORDS[(i * 7 + 3) % 16], WORDS[(i * 5 + 1) % 16]
    return (f"export function Panel{i}() {{\n  return (\n    <section>\n      <h2>{a.title()} {b} ledger {repo}</h2>\n"
            f"      <p>Track every {c} crate shipped from the {b} depot this week.</p>\n"
            f"      <button>Add {a} crate</button>\n      <a href=\"/x\">Open the {c} archive</a>\n"
            f"    </section>\n  );\n}}\n")


def _git_repo(root: Path, files: dict[str, str], untracked: dict[str, str] | None = None) -> Path:
    root.mkdir(parents=True)
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    g = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(g[:3] + ["init", "-q"], check=True)
    subprocess.run(g + ["add", "-A"], check=True)
    subprocess.run(g + ["commit", "-q", "-m", "fake"], check=True)
    for rel, text in (untracked or {}).items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


@pytest.fixture
def sealed_world(fake_slop, tmp_path):
    """Five fake 'sealed' repos, a fake reference_real holding a copy of one of their screens, and a freeze."""
    repos = {}
    k = 0
    for r in range(5):
        name = f"FakeRepo{r}"
        files = {}
        for j in range(3):
            files[f"src/screens/Panel{k}.tsx"] = _tsx(k, name)
            k += 1
        extra = {}
        if r == 0:
            files["src/components/ui/Button.tsx"] = _tsx(90, name)  # vendored
            files["src/screens/Panel0.test.tsx"] = _tsx(91, name)  # test file
            extra["src/screens/Scratch.tsx"] = _tsx(92, name)  # untracked
        repos[name] = str(_git_repo(tmp_path / "repos" / name, files, extra))
    (tmp_path / "repos.json").write_text(json.dumps(repos))
    # reference_real: one screen with the same node texts as FakeRepo1's first screen (Panel3)
    out = tmp_path / "ref-extract"
    src = Path(repos["FakeRepo1"]) / "src/screens/Panel3.tsx"
    subprocess.run(["node", str(bs.EXTRACT_CLI), str(src), "--repo", "RefRepo", "--root", str(src.parent), "--out",
                    str(out)], check=True, capture_output=True)
    screen = json.loads((out / "Panel3.tsx.json").read_text())
    d = fake_slop / "datasets" / "reference_real"
    (d / "screens").mkdir(parents=True)
    (d / "screens" / f"{screen['id']}.json").write_text(json.dumps(screen))
    (d / "manifest.json").write_text(json.dumps({"set": "reference_real", "screens_total": 1, "screens": [
        {"id": screen["id"], "repo": "RefRepo", "path": "Panel3.tsx", "commit": None, "git": "clean",
         "source_sha256": "0" * 64, "screen_sha256": bs.sha256_file(d / "screens" / f"{screen['id']}.json")}]}))
    return fake_slop, tmp_path / "repos.json", repos


def test_refuses_without_a_registry_before_opening_anything(fake_slop, tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("a sealed repo was opened before the gate")

    monkeypatch.setattr(bs, "gather", boom)
    monkeypatch.setattr(bs, "walk_ui_files", boom)
    monkeypatch.setattr(build_sealed, "sealed_repos", boom)
    with pytest.raises(SystemExit, match="does not exist; freeze first"):
        build_sealed.build(fake_slop, 10, None, None, tmp_path / "none.json")


def test_refuses_when_the_freeze_check_fails(sealed_world, monkeypatch):
    root, repos_json, _ = sealed_world
    freeze.write(root)
    (root / "LABELLING.md").write_text((root / "LABELLING.md").read_text() + "\nrevised\n")
    monkeypatch.setattr(bs, "gather", lambda *a, **k: (_ for _ in ()).throw(AssertionError("opened")))
    with pytest.raises(SystemExit, match="freeze.py --check fails"):
        build_sealed.build(root, 10, None, None, repos_json)


def test_refuses_a_second_selection(sealed_world):
    root, repos_json, _ = sealed_world
    freeze.write(root)
    (root / "datasets" / "sealed_real").mkdir(parents=True)
    (root / "datasets" / "sealed_real" / "manifest.json").write_text("{}\n")
    with pytest.raises(SystemExit, match="selected once"):
        build_sealed.build(root, 10, None, None, repos_json)


def test_selection_on_fake_repos(sealed_world):
    root, repos_json, repos = sealed_world
    freeze.write(root)
    pub = build_sealed.build(root, 8, None, None, repos_json)
    d = root / "datasets" / "sealed_real"
    private = json.loads((d / "private_manifest.json").read_text())
    # the rules: vendored, test and untracked files dropped; the reference copy deduplicated
    assert private["drops"]["vendored"] == 1 and private["drops"]["test_spec_story"] == 1
    assert private["drops"]["untracked"] == 1
    assert private["drops"].get("dedupe_exact", 0) + private["drops"].get("dedupe_jaccard", 0) == 1
    assert "src/screens/Panel3.tsx" not in {r["path"] for r in private["screens"] if r["repo"] == "FakeRepo1"}
    # the repo cap: no repo above 25% of the nodes
    nodes = {}
    for r in pub["screens"]:
        nodes[r["repo"]] = nodes.get(r["repo"], 0) + r["nodes"]
    assert pub["screens"] and max(nodes.values()) <= 0.25 * sum(nodes.values()) + 1e-9
    # deterministic rank order, and the same ids in both manifests, order.txt and the screens dir
    assert [r["rank_sha256"] for r in pub["screens"]] == sorted(r["rank_sha256"] for r in pub["screens"])
    ids = [r["id"] for r in pub["screens"]]
    assert ids == [r["id"] for r in private["screens"]]
    assert (d / "order.txt").read_text().split() == ids
    assert sorted(p.stem for p in (d / "screens").glob("*.json")) == sorted(ids)
    # the committable manifest: ids and hashes only
    text = (d / "manifest.json").read_text()
    for r in private["screens"]:
        assert r["path"] not in text and r["path"].rsplit("/", 1)[0] not in text
        assert bs.sha256_bytes(f"{r['repo']}/{r['path']}".encode()) in text
    for repo_root in repos.values():
        assert repo_root not in text
    for w in ("ledger", "crate", "archive", "depot"):
        assert w not in text.lower()
    assert all("path" not in r for r in pub["screens"]) and pub["test"] is True
    assert "largest_dirs" not in pub
    # the freeze records it afterwards
    rec, rev = freeze.revise(root, "sealed_real selected", [])
    assert "sets.sealed_real" in rev["added"]
    with pytest.raises(SystemExit, match="already records sealed_real"):
        build_sealed.gate(root)


def test_selection_is_deterministic(sealed_world, tmp_path):
    root, repos_json, _ = sealed_world
    freeze.write(root)
    a = build_sealed.build(root, 8, None, None, repos_json)
    (root / "datasets" / "sealed_real" / "manifest.json").unlink()
    b = build_sealed.build(root, 8, None, None, repos_json)
    assert a == b
