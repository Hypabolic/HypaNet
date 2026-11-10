"""Build the real-copy sets (SPEC.md §9): reference_real and the calibration pool now, sealed_real after the freeze.

    uv run python scripts/build_sets.py build [--work DIR] [--reference-n 50] [--calibration-n 40]
    uv run python scripts/build_sets.py sealed [--work DIR] [--n 250] [--sibling SCREEN_JSON]

Run from `slop/`. `build` writes, for each of `reference_real` and `calibration`:

- `datasets/<set>/screens/<id>.json`: the extractor's screen JSON (private copy; gitignored);
- `datasets/<set>/sources/<repo>/<path>`: the source file the screen came from, for the labeller's snippets
  (`--source-root datasets/<set>/sources`; private; gitignored);
- `datasets/<set>/manifest.json`: repo, path, commit, hashes, selection rank and drop counts per rule (no copy);
- `datasets/<set>/order.txt`: screen ids in labelling order, for the labeller's `--order` flag;

plus `datasets/kinds.json` (product kind per repo, `LABELLING.md` §2a) and the gitignored sealed hash list
`datasets/sealed_sha256.txt`.

Rules, in order; each file is counted under the first rule that drops it:

1. `not_source`: build output and dependency directories (the extractor's own skip list), minified files and
   files of 2 MB or more. Secret-looking names (`.env*`, `*secret*`, `*credential*`, `.cloudflare`) are skipped
   unread and not counted.
2. `vendored`: `components/ui/**` and `vendor/**`.
3. `worktrees`: any path through a `*.worktrees` directory.
4. `design_refs`: `DesignRefs/**` when a `design_handoff_*` directory in the repo holds a file of the same name or
   the same bytes.
5. `plan_prototype`: `plan/prototype/**` when the same relative path exists under a `src/` directory of the repo, or
   a file there has the same bytes.
6. `test_spec_story`: `*.test.*`, `*.spec.*`, `*.stories.*`, `*.story.*`, `__tests__/`, `__mocks__/`, and Swift
   tests (`*Tests.swift`, `*Test.swift`, any directory named `Tests` or ending in `Tests`).
7. `untracked`: not tracked by git at the recorded commit (generated or local files), so the manifest pins every
   file it lists. A tracked file with local changes is kept and marked `modified`, with its sha256.
8. `byte_identical_sealed`: the file's sha256 equals that of a file in a sealed repo (reference and calibration
   only). The sealed repos are read as bytes and hashed, nothing else; the hashes (no paths) go to the gitignored
   list.
9. `extract_failed`: the extractor could not parse the file.
10. `under_two_nodes`: fewer than two nodes that are not context only (`aria`, `alt`, `code`, `sr_only`;
    `LABELLING.md` §2).
11. `dedupe_exact`, `dedupe_jaccard`: in hash-rank order, a screen whose normalised node-text sequence has the same
    sha256 as, or whose node-text set has Jaccard similarity of at least 0.8 with, a screen already kept in this set
    or in an earlier set (`reference_real` before `calibration` and `sealed_real`).
12. `repo_cap`: dropped to hold the set's repo balance rule.
13. `below_cut`: ranked after the set was full.

Selection rank: screens are ordered by sha256(f"{namespace}:{seed}:{repo}/{path}") as hex, seed 20261116, namespace
`reference`, `calibration` or `sealed`. Normalised node text: NFKC, case-folded, the runs of letters and digits and
the VAR and ALT markers, joined by single spaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

SLOP = Path(__file__).resolve().parents[1]
DATASETS = SLOP / "datasets"
EXTRACT_CLI = SLOP / "extract" / "cli.js"
REGISTRY = SLOP / "registry.yaml"
SEALED_HASHES = DATASETS / "sealed_sha256.txt"
KINDS = DATASETS / "kinds.json"
FIXTURE = SLOP / "extract" / "fixtures" / "pause-menu"

SEED = 20261116
JACCARD = 0.8
HOME = Path.home()
DEV = HOME / "Development"

# The extractor's file types and its own skip list (extract/src/extract.ts, listSourceFiles).
UI_EXT = {".tsx", ".jsx", ".js", ".mjs", ".html", ".htm", ".swift"}
SKIP_DIRS = {"node_modules", ".next", "dist", "build", ".git", "out", "coverage", ".turbo", ".vercel", "Pods",
             "DerivedData", ".build", ".expo", ".svelte-kit"}
MAX_BYTES = 2_000_000
TEST_NAME = re.compile(r"\.(test|spec|stories|story)\.[a-z]+$|Tests?\.swift$")
CONTEXT_ROLES = {"aria", "alt", "code"}

RULES = ["not_source", "vendored", "worktrees", "design_refs", "plan_prototype", "test_spec_story", "untracked",
         "byte_identical_sealed", "extract_failed", "under_two_nodes", "dedupe_exact", "dedupe_jaccard", "repo_cap",
         "below_cut"]


@dataclass
class Repo:
    name: str
    root: Path
    licence: str
    kind: str
    kind_source: str
    subdir: str | None = None  # read only this directory of the repo (paths stay relative to the root)


REFERENCE_REPOS = [
    Repo("HypaChef", DEV / "hypabolic/HypaChef/HypaChef", "private",
         "recipe library and guided cooking app for iPhone and iPad", "README.md"),
    Repo("Mosaic", DEV / "hypabolic/Mosaic", "private",
         "web app for AI research over a document library: ask, deep research, reports and sharing",
         "CONTEXT-MAP.md (no README)"),
    Repo("PersonalHomepage", DEV / "Personal/PersonalHomepage", "private",
         "personal website and blog (Next.js with a Payload CMS)", "package.json dependencies (no README)"),
]
FIXTURE_REPO = Repo("slop-fixtures", FIXTURE, "MIT",
                    "first-person game (fixture note: AI-built pause menu, the motivating example)",
                    "labeller/fixtures/kinds.json")
CALIBRATION_REPOS = [
    Repo("HypaVPN", DEV / "hypabolic/HypaVPN/HypaVPN", "private",
         "Apple TV app for connecting to self-hosted WireGuard VPNs", "README.md"),
    Repo("lean-ctx", DEV / "Personal/lean-ctx", "Apache-2.0",
         "local context layer for AI coding agents (CLI and dashboard)", "README.md"),
    Repo("detectorists", DEV / "Personal/detectorists", "private",
         "mobile app for metal detectorists that maps the ground swept in a session", "README.md"),
    Repo("HyperCubeDemo", DEV / "hypabolic/HyperCubeDemo", "private",
         "4D hypercube renderer: a live web terminal demo, a .NET library and a CLI", "README.md"),
    Repo("HypaDev", DEV / "hypabolic/HypaDev", "Zed licences (LICENSE-*)",
         "multiplayer code editor (a fork of Zed)", "README.md"),
    Repo("StreetSweeperPro", DEV / "Personal/StreetSweeperPro", "private",
         "top-down pixel-art arcade driving game for the web", "README.md"),
    Repo("Hypa-Private", DEV / "hypabolic/Hypa-Private", "private",
         "local context runtime for coding agents (command-output compression); its wiki pages", "README.md",
         subdir="docs/wiki"),
]
# Sealed repos: hashed as bytes by `build`; opened only by `sealed`, after the extractor is frozen. Atomic-Private
# (three apps, one house style) is one repo for the cap.
SEALED_REPOS = [
    ("Atomic-Private", DEV / "hypabolic/Atomic-Private"), ("MemoryOS", DEV / "Personal/MemoryOS"),
    ("SandyScouts", DEV / "Personal/SandyScouts"), ("hypabolic-homepage", DEV / "hypabolic/hypabolic-homepage"),
    ("fpp-quilting-generator", DEV / "Personal/fpp-quilting-generator"), ("HypaAlarm", DEV / "hypabolic/HypaAlarm"),
    ("CoastalQuilts", DEV / "Personal/CoastalQuiltsWebsite"), ("odysseus", DEV / "Personal/odysseus"),
]

REFERENCE_RULE = {"measure": "nodes", "share": 0.4,
                  "text": "no repo gives more than 40% of the nodes of the 50 repo screens (a 25% cap, as for "
                          "sealed_real, cannot hold with three repos); the pause-menu fixture is added on top"}
CALIBRATION_RULE = {"measure": None, "share": 1.0,
                    "text": "no repo cap: up to 40 screens drawn by hash rank (SPEC.md §5)"}
SEALED_RULE = {"measure": "nodes", "share": 0.25,
               "text": "no repo gives more than 25% of the nodes (SPEC.md §9)"}


# --------------------------------------------------------------------------------------------- hashing, text


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rank_hash(namespace: str, repo: str, path: str, seed: int = SEED) -> str:
    return sha256_bytes(f"{namespace}:{seed}:{repo}/{path}".encode("utf-8"))


_NORM_WORD = re.compile(r"[^\W_]+|[]")


def norm_text(text: str) -> str:
    return " ".join(_NORM_WORD.findall(unicodedata.normalize("NFKC", text).casefold()))


def screen_texts(screen: dict) -> list[str]:
    return [norm_text(n.get("text", "")) for n in screen.get("nodes", [])]


def text_hash(screen: dict) -> str:
    return sha256_bytes("\n".join(screen_texts(screen)).encode("utf-8"))


def text_set(screen: dict) -> frozenset[str]:
    return frozenset(t for t in screen_texts(screen) if t)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def is_context(node: dict) -> bool:
    return node.get("role") in CONTEXT_ROLES or "sr_only" in (node.get("flags") or ())


def labelable_count(screen: dict) -> int:
    return sum(1 for n in screen.get("nodes", []) if not is_context(n))


def extractor_sha256() -> str:
    """sha256 over the extractor's code and pins: (relative path, sha256) of each file, sorted by path."""
    files = [SLOP / "extract" / "cli.js", SLOP / "package.json", SLOP / "package-lock.json"]
    files += sorted((SLOP / "extract" / "src").rglob("*.ts"))
    lines = sorted(f"{p.relative_to(SLOP).as_posix()} {sha256_file(p)}" for p in files)
    return sha256_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def is_secret(name: str) -> bool:
    return name.startswith(".env") or bool(re.search(r"secret|credential", name, re.I)) or name == ".cloudflare"


# --------------------------------------------------------------------------------------------- walking a repo


@dataclass
class FileEntry:
    path: str  # relative to the repo root, '/'-separated
    abspath: Path
    rule: str | None = None
    sha256: str | None = None
    git: str = "clean"  # clean | modified


def walk_ui_files(top: Path, root: Path) -> list[tuple[str, Path, int]]:
    """Every file under `top` with an extractor file type: (path relative to `root`, absolute path, size).

    Directories of the extractor's skip list and secret-looking names are pruned unread; symbolic links are not
    followed. `vendor` and test directories are kept here so that the rules can count them.
    """
    out = []
    for d, dirs, files in os.walk(top, followlinks=False):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS and not is_secret(x))
        for f in sorted(files):
            if is_secret(f) or Path(f).suffix not in UI_EXT:
                continue
            p = Path(d) / f
            try:
                st = p.lstat()
            except OSError:
                continue
            if not p.is_file() or p.is_symlink():
                continue
            out.append((p.relative_to(root).as_posix(), p, st.st_size))
    return out


def git_state(root: Path) -> tuple[str | None, set[str], set[str]]:
    """(HEAD commit, tracked paths, modified paths), paths relative to `root`."""
    def git(*a: str) -> bytes:
        return subprocess.run(["git", "-C", str(root), *a], capture_output=True, check=True).stdout

    try:
        top = Path(git("rev-parse", "--show-toplevel").decode().strip()).resolve()
        commit = git("rev-parse", "HEAD").decode().strip()
    except subprocess.CalledProcessError:
        return None, set(), set()
    prefix = root.resolve().relative_to(top).as_posix()
    prefix = "" if prefix == "." else prefix + "/"

    def rel(p: str) -> str | None:
        return p[len(prefix):] if p.startswith(prefix) else None

    tracked = {r for p in git("ls-files", "-z").decode().split("\0") if p and (r := rel(p)) is not None}
    modified = set()
    for rec in git("status", "--porcelain=v1", "-z", "--untracked-files=no").decode().split("\0"):
        if len(rec) > 3 and (r := rel(rec[3:])) is not None:
            modified.add(r)
    return commit, tracked, modified


def classify(repo: Repo, sealed_hashes: set[str] | None) -> tuple[list[FileEntry], str | None]:
    """Apply rules 1-8 to every UI file of a repo. Returns the entries (rule None = eligible) and the commit."""
    root = repo.root.resolve()
    top = root / repo.subdir if repo.subdir else root
    raw = walk_ui_files(top, root)
    commit, tracked, modified = git_state(root)
    names = [p for p, _, _ in raw]

    handoff_names, handoff_hashes = set(), set()
    src_paths, src_hashes = set(), set()
    has_handoff = any(part.startswith("design_handoff_") for p in names for part in p.split("/")[:-1])
    has_proto = any("plan/prototype/" in f"/{p}" for p in names)
    if has_handoff or has_proto:
        for p, ap, size in raw:
            parts = p.split("/")
            if has_handoff and any(x.startswith("design_handoff_") for x in parts[:-1]):
                handoff_names.add(parts[-1])
                handoff_hashes.add(sha256_file(ap))
            if has_proto and "src" in parts[:-1] and "prototype" not in parts:
                i = len(parts) - 1 - parts[::-1].index("src")
                src_paths.add("/".join(parts[i + 1:]))
                src_hashes.add(sha256_file(ap))

    out = []
    for p, ap, size in raw:
        parts = p.split("/")
        dirs, name = parts[:-1], parts[-1]
        e = FileEntry(p, ap)
        if name.endswith(".min.js") or size >= MAX_BYTES:
            e.rule = "not_source"
        elif re.search(r"(^|/)components/ui/", p) or "vendor" in dirs:
            e.rule = "vendored"
        elif any(x.endswith(".worktrees") for x in dirs):
            e.rule = "worktrees"
        elif "DesignRefs" in dirs and has_handoff and (name in handoff_names or sha256_file(ap) in handoff_hashes):
            e.rule = "design_refs"
        elif "/plan/prototype/" in f"/{p}" and (
                p.split("plan/prototype/", 1)[1] in src_paths or sha256_file(ap) in src_hashes):
            e.rule = "plan_prototype"
        elif TEST_NAME.search(name) or any(x in ("__tests__", "__mocks__") or x == "Tests" or x.endswith("Tests")
                                             for x in dirs):
            e.rule = "test_spec_story"
        elif commit is not None and p not in tracked:
            e.rule = "untracked"
        else:
            e.sha256 = sha256_file(ap)
            e.git = "modified" if p in modified else "clean"
            if sealed_hashes is not None and e.sha256 in sealed_hashes:
                e.rule = "byte_identical_sealed"
        out.append(e)
    return out, commit


def sealed_identical(entries: list[FileEntry], sealed_hashes: set[str]) -> int:
    """Files byte-identical to a sealed file whichever rule dropped them (vendored primitives included)."""
    n = 0
    for e in entries:
        if e.rule != "not_source" and (e.sha256 or sha256_file(e.abspath)) in sealed_hashes:
            n += 1
    return n


def hash_sealed_repos() -> tuple[set[str], int]:
    """sha256 of every UI file in the sealed repos, bytes only. Nothing about these files is printed or kept but
    the hashes."""
    hashes = set()
    n = 0
    for _, root in SEALED_REPOS:
        if not root.is_dir():
            raise SystemExit(f"sealed repo missing: {root.name}")
        for _, ap, size in walk_ui_files(root, root):
            if size < MAX_BYTES:
                hashes.add(sha256_file(ap))
                n += 1
    return hashes, n


# --------------------------------------------------------------------------------------------- extraction


def run_extractor(repo: Repo, commit: str | None, out: Path) -> dict:
    root = repo.root.resolve()
    top = root / repo.subdir if repo.subdir else root
    cmd = ["node", str(EXTRACT_CLI), str(top), "--repo", repo.name, "--licence", repo.licence, "--root", str(root),
           "--out", str(out / repo.name), "--keep-empty", "--summary", str(out / f"{repo.name}.summary.json")]
    if commit:
        cmd += ["--commit", commit]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads((out / f"{repo.name}.summary.json").read_text())


@dataclass
class Candidate:
    repo: str
    path: str
    commit: str | None
    git: str
    source: Path
    source_sha256: str
    screen: dict
    screen_bytes: bytes
    rank: str = ""
    texts_hash: str = ""
    texts: frozenset = field(default_factory=frozenset)

    @property
    def nodes(self) -> int:
        return len(self.screen["nodes"])


def gather(repos: list[Repo], work: Path, sealed_hashes: set[str] | None, drops: dict, per_repo: dict,
           namespace: str) -> list[Candidate]:
    """Rules 1-10 for every repo; returns the eligible screens ranked by hash."""
    cands = []
    for repo in repos:
        entries, commit = classify(repo, sealed_hashes)
        summ = run_extractor(repo, commit, work)
        failed = {f["path"] for f in summ["failures"]}
        stats = per_repo.setdefault(repo.name, {"commit": commit, "licence": repo.licence, "files": len(entries),
                                                "drops": {}, "eligible": 0})
        if sealed_hashes is not None:
            stats["byte_identical_sealed_any_rule"] = sealed_identical(entries, sealed_hashes)
        for e in entries:
            if e.rule is None and e.path in failed:
                e.rule = "extract_failed"
            screen_file = work / repo.name / f"{e.path}.json"
            if e.rule is None and not screen_file.is_file():
                raise SystemExit(f"{repo.name}: the extractor wrote no screen for an eligible file (rank "
                                 f"{rank_hash(namespace, repo.name, e.path)[:12]})")
            if e.rule is None:
                b = screen_file.read_bytes()
                screen = json.loads(b)
                if labelable_count(screen) < 2:
                    e.rule = "under_two_nodes"
                else:
                    c = Candidate(repo.name, e.path, commit, e.git, e.abspath, e.sha256 or "", screen, b)
                    c.rank = rank_hash(namespace, repo.name, e.path)
                    c.texts_hash, c.texts = text_hash(screen), text_set(screen)
                    cands.append(c)
                    stats["eligible"] += 1
            if e.rule:
                drops[e.rule] = drops.get(e.rule, 0) + 1
                stats["drops"][e.rule] = stats["drops"].get(e.rule, 0) + 1
    cands.sort(key=lambda c: c.rank)
    return cands


def fixture_candidate(work: Path, namespace: str) -> Candidate:
    """The pause-menu fixture, extracted afresh; it must equal the fixture's expected.json byte for byte."""
    src = FIXTURE / "PauseMenu.tsx"
    out = work / "slop-fixtures"
    subprocess.run(["node", str(EXTRACT_CLI), str(src), "--repo", "slop-fixtures", "--licence", "MIT", "--root",
                    str(FIXTURE), "--out", str(out)], check=True, capture_output=True, text=True)
    b = (out / "PauseMenu.tsx.json").read_bytes()
    if b != (FIXTURE / "expected.json").read_bytes():
        raise SystemExit("pause-menu fixture: the extractor's output differs from expected.json")
    screen = json.loads(b)
    c = Candidate("slop-fixtures", "PauseMenu.tsx", None, "fixture", src, sha256_file(src), screen, b)
    c.rank = rank_hash(namespace, c.repo, c.path)
    c.texts_hash, c.texts = text_hash(screen), text_set(screen)
    return c


# --------------------------------------------------------------------------------------------- selection


def dedupe(cands: list[Candidate], earlier: list[Candidate], drops: dict, per_repo: dict) -> list[Candidate]:
    """Keep the first screen by hash rank of each duplicate group, within the set and against earlier sets."""
    kept: list[Candidate] = []
    seen_hash = {c.texts_hash for c in earlier}
    pool = list(earlier)
    for c in cands:
        rule = None
        if c.texts_hash in seen_hash:
            rule = "dedupe_exact"
        elif any(jaccard(c.texts, k.texts) >= JACCARD for k in pool):
            rule = "dedupe_jaccard"
        if rule:
            drops[rule] = drops.get(rule, 0) + 1
            d = per_repo[c.repo]["drops"]
            d[rule] = d.get(rule, 0) + 1
            continue
        kept.append(c)
        pool.append(c)
        seen_hash.add(c.texts_hash)
    return kept


def over_cap(sel: list[Candidate], measure: str | None, share: float) -> str | None:
    """The repo furthest over its share, or None."""
    if not measure or not sel:
        return None
    w = {}
    for c in sel:
        w[c.repo] = w.get(c.repo, 0) + (c.nodes if measure == "nodes" else 1)
    total = sum(w.values())
    worst = max(sorted(w), key=lambda r: w[r])
    return worst if w[worst] > share * total + 1e-9 else None


def select(cands: list[Candidate], target: int, rule: dict, drops: dict, per_repo: dict) -> list[Candidate]:
    """Fill to `target` in rank order; while a repo is over its share, drop its lowest-ranked screen and refill."""
    sel: list[Candidate] = []
    i = 0
    while True:
        while len(sel) < target and i < len(cands):
            sel.append(cands[i])
            i += 1
        repo = over_cap(sel, rule["measure"], rule["share"])
        if repo is None:
            break
        victim = max((c for c in sel if c.repo == repo), key=lambda c: c.rank)
        sel.remove(victim)
        drops["repo_cap"] = drops.get("repo_cap", 0) + 1
        d = per_repo[repo]["drops"]
        d["repo_cap"] = d.get("repo_cap", 0) + 1
    for c in cands[i:]:
        drops["below_cut"] = drops.get("below_cut", 0) + 1
        d = per_repo[c.repo]["drops"]
        d["below_cut"] = d.get("below_cut", 0) + 1
    return sorted(sel, key=lambda c: c.rank)


# --------------------------------------------------------------------------------------------- output


def write_set(name: str, namespace: str, sel: list[Candidate], drops: dict, per_repo: dict, rule: dict,
              extra: dict) -> dict:
    d = DATASETS / name
    for sub in ("screens", "sources"):
        if (d / sub).exists():
            shutil.rmtree(d / sub)
        (d / sub).mkdir(parents=True)
    rows, roles = [], {}
    for n, c in enumerate(sel, 1):
        sid = c.screen["id"]
        (d / "screens" / f"{sid}.json").write_bytes(c.screen_bytes)
        dest = d / "sources" / c.repo / c.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(c.source, dest)
        if sha256_file(dest) != c.source_sha256:
            raise SystemExit(f"{c.repo}: a source file changed during the build (screen {sid})")
        for nd in c.screen["nodes"]:
            roles[nd["role"]] = roles.get(nd["role"], 0) + 1
        rows.append({"order": n, "id": sid, "repo": c.repo, "path": c.path, "commit": c.commit, "git": c.git,
                     "screen_sha256": sha256_bytes(c.screen_bytes), "source_sha256": c.source_sha256,
                     "rank_sha256": c.rank, "nodes": c.nodes, "labelable": labelable_count(c.screen)})
    (d / "order.txt").write_text("".join(f"{r['id']}\n" for r in rows))
    repos = {}
    for r, s in sorted(per_repo.items()):
        mine = [x for x in rows if x["repo"] == r]
        repos[r] = {**s, "drops": dict(sorted(s["drops"].items(), key=lambda kv: RULES.index(kv[0]))),
                    "screens": len(mine), "nodes": sum(x["nodes"] for x in mine)}
    total_nodes = sum(r["nodes"] for r in rows)
    manifest = {
        "schema": "slop.set_manifest/1", "set": name, "seed": SEED, "namespace": namespace,
        "rank": 'sha256(f"{namespace}:{seed}:{repo}/{path}") as hex, ascending',
        "builder_sha256": sha256_file(Path(__file__)), "extractor_sha256": extractor_sha256(),
        "balance_rule": rule["text"], **extra,
        "screens_total": len(rows), "nodes_total": total_nodes,
        "labelable_total": sum(r["labelable"] for r in rows),
        "roles": dict(sorted(roles.items(), key=lambda kv: (-kv[1], kv[0]))),
        "drops": {k: drops[k] for k in RULES if drops.get(k)},
        "repos": repos, "screens": rows,
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    return manifest


def write_kinds(repos: list[Repo]) -> None:
    kinds = {r.name: r.kind for r in repos}
    KINDS.parent.mkdir(parents=True, exist_ok=True)
    KINDS.write_text(json.dumps(kinds, indent=1, ensure_ascii=False) + "\n")


def kind_sources(repos: list[Repo]) -> dict:
    return {r.name: r.kind_source for r in repos}


# --------------------------------------------------------------------------------------------- commands


def cmd_build(args) -> None:
    work_ctx = tempfile.TemporaryDirectory(prefix="slop-sets-") if not args.work else None
    work = Path(args.work or work_ctx.name).resolve()
    if args.work:
        if work.exists() and any(work.iterdir()):
            raise SystemExit(f"--work {work} is not empty")
        work.mkdir(parents=True, exist_ok=True)
    try:
        sealed, n_sealed = hash_sealed_repos()
        DATASETS.mkdir(parents=True, exist_ok=True)
        SEALED_HASHES.write_text("".join(f"{h}\n" for h in sorted(sealed)))
        sealed_note = {"sealed_files_hashed": n_sealed, "sealed_hash_list_sha256": sha256_file(SEALED_HASHES)}

        drops, per_repo = {}, {}
        cands = gather(REFERENCE_REPOS, work / "reference", sealed, drops, per_repo, "reference")
        fixture = fixture_candidate(work / "reference", "reference")
        cands = dedupe(cands, [fixture], drops, per_repo)
        ref = select(cands, args.reference_n, REFERENCE_RULE, drops, per_repo)
        per_repo["slop-fixtures"] = {"commit": None, "licence": "MIT", "files": 1, "drops": {}, "eligible": 1}
        ref = sorted(ref + [fixture], key=lambda c: c.rank)
        m_ref = write_set("reference_real", "reference", ref, drops, per_repo, REFERENCE_RULE, {
            **sealed_note, "fixture": {"id": fixture.screen["id"],
                                       "origin": "slop/extract/fixtures/pause-menu/PauseMenu.tsx",
                                       "note": "the pause-menu fixture; its expected.json is the screen"},
            "kinds": kind_sources(REFERENCE_REPOS + [FIXTURE_REPO])})

        drops_c, per_repo_c = {}, {}
        cands_c = gather(CALIBRATION_REPOS, work / "calibration", sealed, drops_c, per_repo_c, "calibration")
        cands_c = dedupe(cands_c, ref, drops_c, per_repo_c)
        cal = select(cands_c, args.calibration_n, CALIBRATION_RULE, drops_c, per_repo_c)
        m_cal = write_set("calibration", "calibration", cal, drops_c, per_repo_c, CALIBRATION_RULE, {
            **sealed_note, "pool": "the calibration pool is all of reference_real plus this set (SPEC.md §5)",
            "target": args.calibration_n, "shortfall": max(0, args.calibration_n - len(cal)),
            "kinds": kind_sources(CALIBRATION_REPOS)})
        write_kinds(REFERENCE_REPOS + [FIXTURE_REPO] + CALIBRATION_REPOS)
    finally:
        if work_ctx:
            work_ctx.cleanup()

    for m in (m_ref, m_cal):
        per = ", ".join(f"{r} {s['screens']}" for r, s in m["repos"].items() if s["screens"])
        print(f"{m['set']}: {m['screens_total']} screens, {m['nodes_total']} nodes ({per})")
        print(f"  drops: {json.dumps(m['drops'])}")
    for m in (m_ref, m_cal):
        anyrule = sum(s.get("byte_identical_sealed_any_rule", 0) for s in m["repos"].values())
        print(f"{m['set']}: {m['drops'].get('byte_identical_sealed', 0)} files dropped as byte-identical to a "
              f"sealed file ({anyrule} identical under any rule); {n_sealed} sealed files hashed")


def frozen_extractor() -> str:
    """Refuse unless slop/registry.yaml records the extractor's sha256 and it matches the code on disk."""
    if not REGISTRY.is_file():
        raise SystemExit("refusing: slop/registry.yaml does not exist; freeze the extractor first (SPEC.md §12.1)")
    import yaml

    reg = yaml.safe_load(REGISTRY.read_text()) or {}
    want = ((reg.get("extractor") or {}) if isinstance(reg, dict) else {}).get("sha256")
    if not want:
        raise SystemExit("refusing: registry.yaml has no extractor.sha256; freeze the extractor first")
    have = extractor_sha256()
    if have != want:
        raise SystemExit(f"refusing: the extractor on disk ({have[:12]}) is not the frozen one ({str(want)[:12]})")
    return have


def cmd_sealed(args) -> None:
    """sealed_real (SPEC.md §9). Opens the sealed repos, so it refuses until the extractor is frozen."""
    frozen_extractor()
    ref_dir = DATASETS / "reference_real" / "screens"
    ref_manifest = json.loads((DATASETS / "reference_real" / "manifest.json").read_text())
    ref = []
    for row in ref_manifest["screens"]:
        screen = json.loads((ref_dir / f"{row['id']}.json").read_bytes())
        c = Candidate(row["repo"], row["path"], row["commit"], row["git"], Path(), row["source_sha256"], screen, b"")
        c.texts_hash, c.texts = text_hash(screen), text_set(screen)
        ref.append(c)
    repos = [Repo(n, r, "private", "", "") for n, r in SEALED_REPOS]
    for r in repos:
        if r.name == "MemoryOS":
            r.licence = "Apache-2.0"
    work_ctx = tempfile.TemporaryDirectory(prefix="slop-sealed-") if not args.work else None
    work = Path(args.work or work_ctx.name).resolve()
    try:
        drops, per_repo = {}, {}
        cands = gather(repos, work, None, drops, per_repo, "sealed")
        cands = dedupe(cands, ref, drops, per_repo)
        sel = select(cands, args.n, SEALED_RULE, drops, per_repo)
        extra = {"note": "deduplicated against reference_real; Atomic-Private counts as one repo"}
        if args.sibling:
            p = Path(args.sibling)
            b = p.read_bytes()
            screen = json.loads(b)
            c = Candidate(screen["source"]["repo"], screen["source"]["path"], screen["source"].get("commit"),
                          "fixture", p, sha256_file(p), screen, b)
            c.rank = rank_hash("sealed", c.repo, c.path)
            per_repo[c.repo] = {"commit": c.commit, "licence": screen["source"].get("licence"), "files": 1,
                                "drops": {}, "eligible": 1}
            sel = sorted(sel + [c], key=lambda x: x.rank)
            extra["sibling_fixture"] = screen["id"]
        m = write_set("sealed_real", "sealed", sel, drops, per_repo, SEALED_RULE, extra)
    finally:
        if work_ctx:
            work_ctx.cleanup()
    print(f"sealed_real: {m['screens_total']} screens, {m['nodes_total']} nodes")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="reference_real and the calibration pool")
    b.add_argument("--work", help="staging directory for the extractor's output (default: a temporary one)")
    b.add_argument("--reference-n", type=int, default=50, help="repo screens in reference_real (default 50)")
    b.add_argument("--calibration-n", type=int, default=40, help="screens in the calibration set (default 40)")
    s = sub.add_parser("sealed", help="sealed_real; refuses until the extractor hash is frozen in registry.yaml")
    s.add_argument("--work")
    s.add_argument("--n", type=int, default=250)
    s.add_argument("--sibling", help="screen JSON of the sealed sibling fixture (SPEC.md §9)")
    args = ap.parse_args(argv)
    {"build": cmd_build, "sealed": cmd_sealed}[args.cmd](args)


if __name__ == "__main__":
    main()
