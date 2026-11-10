"""Select sealed_real, the headline set (SPEC.md §9), once the freeze is in place. Labels nothing.

    uv run python scripts/build_sealed.py [--n 250] [--sibling SCREEN_JSON] [--work DIR]
    uv run python scripts/build_sealed.py --root TEST_ROOT --repos REPOS.json [...]   # tests only

Run from `slop/`. Refuses, before it opens any sealed repo, unless:

1. `registry.yaml` exists and `scripts/freeze.py --check` passes on it (pending components are allowed: sealed_real
   itself is pending until this script has run);
2. its `extractor.sha256` equals the extractor on disk (the hash scripts/build_sets.py computes), and it records the
   guide (LABELLING.md);
3. `datasets/sealed_real/manifest.json` does not exist and the registry does not record sealed_real already.

Selection is scripts/build_sets.py's, unchanged: the same exclusion rules 1-8 (not source, vendored primitives,
worktrees, design refs, prototype mirrors, test/spec/story files, blog figures, untracked files), extraction with the
frozen extractor, at least two labelable nodes, ranking by sha256("sealed:20261116:<repo>/<path>"), deduplication
against reference_real (exact normalised text and node-set Jaccard >= 0.8) and within the set, and the repo cap (no repo
gives more than 25% of the nodes; Atomic-Private counts as one repo). The sealed sibling fixture (--sibling, a screen
JSON) is added on top.

Writes (everything with copy or paths is private and gitignored):

- `datasets/sealed_real/screens/<id>.json`, `sources/<repo>/<path>`: screen JSON and source files (private);
- `datasets/sealed_real/private_manifest.json`: the build_sets manifest with paths (private);
- `datasets/sealed_real/manifest.json` (committed): ids and hashes only. Per screen: id, repo name, sha256 of the
  path (sha256("<repo>/<path>")), commit, screen, source and rank sha256, node counts; per repo: counts and drops;
  per rule: drop counts; no copy, no path, no directory name;
- `datasets/sealed_real/order.txt` (labelling order): ids only.

Afterwards record it: `uv run python scripts/freeze.py --revise --reason "sealed_real selected"`.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

SLOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SLOP / "scripts"))
import build_sets as bs  # noqa: E402
import freeze  # noqa: E402

NAMESPACE = "sealed"


def gate(root: Path) -> dict:
    """Refuse unless the freeze is in place (see the module docstring). Opens no sealed file."""
    reg_path = freeze.registry_path(root)
    if not reg_path.is_file():
        raise SystemExit(f"refusing: {reg_path} does not exist; freeze first "
                         "(scripts/freeze.py --write, SPEC.md §12.6)")
    problems, _ = freeze.check(root)
    if problems:
        raise SystemExit("refusing: scripts/freeze.py --check fails:\n  " + "\n  ".join(problems))
    reg = freeze.load(root)
    want = (reg.get("extractor") or {}).get("sha256")
    have = bs.extractor_sha256()
    if want != have:
        raise SystemExit(f"refusing: the extractor on disk ({have[:12]}) is not the frozen one ({str(want)[:12]})")
    if not (reg.get("guide") or {}).get("sha256"):
        raise SystemExit("refusing: registry.yaml records no guide")
    if (reg.get("sets") or {}).get("sealed_real") is not None:
        raise SystemExit("refusing: registry.yaml already records sealed_real")
    if (root / "datasets" / "sealed_real" / "manifest.json").exists():
        raise SystemExit("refusing: datasets/sealed_real/manifest.json exists (sealed_real is selected once)")
    return reg


def load_reference(root: Path) -> list[bs.Candidate]:
    d = root / "datasets" / "reference_real"
    m = d / "manifest.json"
    if not m.is_file():
        raise SystemExit(f"refusing: {m} missing (sealed_real is deduplicated against reference_real)")
    out = []
    for row in json.loads(m.read_text())["screens"]:
        p = d / "screens" / f"{row['id']}.json"
        if not p.is_file():
            raise SystemExit(f"refusing: reference_real screen {row['id']} missing ({d / 'screens'})")
        screen = json.loads(p.read_bytes())
        c = bs.Candidate(row["repo"], row["path"], row["commit"], row["git"], Path(), row["source_sha256"], screen, b"")
        c.texts_hash, c.texts = bs.text_hash(screen), bs.text_set(screen)
        out.append(c)
    return out


def sealed_repos(repos_file: Path | None) -> list[bs.Repo]:
    if repos_file is None:
        pairs = bs.SEALED_REPOS
    else:
        pairs = [(k, Path(v)) for k, v in json.loads(repos_file.read_text()).items()]
    repos = [bs.Repo(n, r, "private", "", "") for n, r in pairs]
    for r in repos:
        if r.name == "MemoryOS":
            r.licence = "Apache-2.0"
        if not r.root.is_dir():
            raise SystemExit(f"refusing: sealed repo {r.name} missing")
    return repos


def public_manifest(private: dict, test: bool) -> dict:
    """The committable manifest: ids and hashes, no copy, no path, no directory name."""
    rows = []
    for r in private["screens"]:
        rows.append({"order": r["order"], "id": r["id"], "repo": r["repo"],
                     "path_sha256": bs.sha256_bytes(f"{r['repo']}/{r['path']}".encode("utf-8")),
                     "commit": r["commit"], "git": r["git"], "screen_sha256": r["screen_sha256"],
                     "source_sha256": r["source_sha256"], "rank_sha256": r["rank_sha256"], "nodes": r["nodes"],
                     "labelable": r["labelable"]})
    keep = ("schema", "set", "seed", "namespace", "rank", "builder_sha256", "extractor_sha256", "balance_rule", "caps",
            "screens_total", "nodes_total", "labelable_total", "roles", "drops", "repos", "note", "sibling_fixture",
            "target", "shortfall")
    m = {k: private[k] for k in keep if k in private}
    m["schema"] = "slop.sealed_manifest/1"
    m["sealed_builder_sha256"] = bs.sha256_file(Path(__file__))
    m["path_hash"] = 'sha256(f"{repo}/{path}") as hex'
    m["test"] = test
    m["screens"] = rows
    return m


def build(root: Path, n: int, sibling: Path | None, work: Path | None, repos_file: Path | None) -> dict:
    root = root.resolve()
    gate(root)
    ref = load_reference(root)
    repos = sealed_repos(repos_file)
    old_datasets = bs.DATASETS
    bs.DATASETS = root / "datasets"
    work_ctx = tempfile.TemporaryDirectory(prefix="slop-sealed-") if not work else None
    wdir = Path(work or work_ctx.name).resolve()
    try:
        drops, per_repo = {}, {}
        cands = bs.gather(repos, wdir, None, drops, per_repo, NAMESPACE)
        cands = bs.dedupe(cands, ref, drops, per_repo)
        sel = bs.select(cands, n, bs.SEALED_RULE, drops, per_repo)
        extra = {"note": "deduplicated against reference_real; Atomic-Private counts as one repo", "target": n,
                 "shortfall": max(0, n - len(sel))}
        if sibling:
            b = sibling.read_bytes()
            screen = json.loads(b)
            c = bs.Candidate(screen["source"]["repo"], screen["source"]["path"], screen["source"].get("commit"),
                             "fixture", sibling, bs.sha256_file(sibling), screen, b)
            c.rank = bs.rank_hash(NAMESPACE, c.repo, c.path)
            per_repo[c.repo] = {"commit": c.commit, "licence": screen["source"].get("licence"), "files": 1,
                                "drops": {}, "eligible": 1}
            sel = sorted(sel + [c], key=lambda x: x.rank)
            extra["sibling_fixture"] = screen["id"]
        private = bs.write_set("sealed_real", NAMESPACE, sel, drops, per_repo, bs.SEALED_RULE, extra)
    finally:
        bs.DATASETS = old_datasets
        if work_ctx:
            work_ctx.cleanup()
    d = root / "datasets" / "sealed_real"
    (d / "manifest.json").rename(d / "private_manifest.json")
    pub = public_manifest(private, test=root != SLOP.resolve())
    (d / "manifest.json").write_text(json.dumps(pub, indent=1, ensure_ascii=False) + "\n")
    return pub


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--sibling", type=Path, help="screen JSON of the sealed sibling fixture (SPEC.md §9)")
    ap.add_argument("--work", type=Path, help="staging directory for the extractor's output (default: temporary)")
    ap.add_argument("--root", type=Path, default=SLOP, help="tests only: a slop-like root (registry.yaml, datasets/)")
    ap.add_argument("--repos", type=Path, help="tests only (with --root): JSON {repo name: path} replacing the sealed "
                                                 "repos")
    a = ap.parse_args(argv)
    if a.repos and a.root.resolve() == SLOP.resolve():
        ap.error("--repos is for tests and needs a --root other than slop/")
    m = build(a.root, a.n, a.sibling, a.work, a.repos)
    per = ", ".join(f"{r} {s['screens']}" for r, s in m["repos"].items() if s["screens"])
    print(f"sealed_real: {m['screens_total']} screens, {m['nodes_total']} nodes ({per}); "
          f"drops {json.dumps(m['drops'])}")
    print("record it: uv run python scripts/freeze.py --revise --reason \"sealed_real selected\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
