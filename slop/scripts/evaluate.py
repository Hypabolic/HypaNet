"""Owner-side evaluation of a slop artefact: the unseen val set, then the owner's unseen repos.

    uv run python scripts/evaluate.py ART_DIR [--against ART2] [--run NAME] [--out DIR] [--no-val] [--no-repos]
                                              [--repo NAME=PATH ...] [--node NODE]

1. val: scores ART_DIR on the 2,000 generated val screens (datasets/val) with exactly the measures of the author's
   ./score (kit/tools/author_tool.py, imported, not copied): the screens and labels are written into a temporary
   examples root that is deleted afterwards; nothing from val is written anywhere else. The report keeps the aggregate
   lines only (the per-node "wrong nodes" list is dropped, so no val copy reaches a file). With --against, the number
   of val nodes whose call differs between the two artefacts, and which of the two the labels side with.
2. repos: compiles the artefact with lint/compile.py and runs the real extractor, rule layer and JS runtime over every
   UI file of the owner's repos (REPOS below, or --repo) in one node process (scripts/evaluate_lint.mjs). Two groups
   are reported apart: the unseen repos, none of whose files the author was shown, and the reference repos
   (REFERENCE_REPOS), the projects behind the disclosed examples/reference screens; every disclosed (repo, path) of
   datasets/reference_real/manifest.json is skipped, so only their undisclosed files are scored. Every node the
   network flags is listed with group, file:line, role, text, class, target and margin; every rule-layer finding with
   its rule, code and severity. With --against, the nodes whose class or pointer differs.

The report goes to runs/<run>/eval/ (gitignored: it quotes the owner's private copy): report.md (summary), val.txt,
flags.tsv, rules.tsv, against.tsv (with --against), errors.tsv and summary.json. <run> is the runs/<run>/ that holds
ART_DIR, or --run. Nothing here is ever handed to an author.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

SLOP = Path(__file__).resolve().parents[1]
HOME = Path.home()
VAL_DIR = SLOP / "datasets" / "val"
REF_MANIFEST = SLOP / "datasets" / "reference_real" / "manifest.json"
LINT = SLOP / "scripts" / "evaluate_lint.mjs"
# The owner's repos that are linted: the unseen ones first, then the reference ones.
REPOS = {
    "HypaVPN": HOME / "Development/hypabolic/HypaVPN/HypaVPN",
    "lean-ctx": HOME / "Development/Personal/lean-ctx",
    "detectorists": HOME / "Development/Personal/detectorists",
    "HyperCubeDemo": HOME / "Development/hypabolic/HyperCubeDemo",
    "StreetSweeperPro": HOME / "Development/Personal/StreetSweeperPro",
    "HypaChef": HOME / "Development/hypabolic/HypaChef/HypaChef",  # the root scripts/build_sets.py uses
    "Mosaic": HOME / "Development/hypabolic/Mosaic",
    "PersonalHomepage": HOME / "Development/Personal/PersonalHomepage",
}
# The projects behind the disclosed reference screens (the author's examples/README.md names them, and it saw some of
# their files): reported as their own group, never as unseen. A repo that gives a screen to
# datasets/reference_real/manifest.json counts as reference too, whatever its name.
REFERENCE_REPOS = ("HypaChef", "Mosaic", "PersonalHomepage")
GROUPS = ("unseen", "reference")
# Repos whose copy is sealed (the real test sets): evaluate.py refuses to read them.
SEALED = ("Atomic-Private", "MemoryOS", "SandyScouts", "hypabolic-homepage", "fpp-quilting-generator", "HypaAlarm",
          "CoastalQuiltsWebsite", "odysseus", "slop-private")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


at = _load("slop_author_tool", SLOP / "kit" / "tools" / "author_tool.py")
kb = _load("slop_kit_build", SLOP / "kit" / "kit_build.py")
mdl = at.mdl


class EvalError(Exception):
    pass


# -- val -----------------------------------------------------------------------------------------------------------


def load_val() -> tuple[list[dict], dict[str, list[dict]]]:
    screens = list(kb.iter_jsonl(VAL_DIR / "screens.jsonl"))
    ids = {s["id"] for s in screens}
    return screens, kb.read_labels(VAL_DIR / "val.gen.jsonl", ids)


def score_set(arts: list, screens: list[dict], labels: dict[str, list[dict]], name: str = "val") -> dict:
    """author_tool's ./score on these screens, through a temporary examples root (deleted on return). Returns the
    aggregate report lines for the first artefact and, with a second, the call differences."""
    tmp = tempfile.mkdtemp(prefix="slop-eval-")
    try:
        for s in screens:
            kb._dump(Path(tmp) / "examples" / name / f"{s['id']}.screen.json", kb.screen_json(s))
            kb._dump(Path(tmp) / "examples" / name / f"{s['id']}.labels.jsonl", kb.labels_jsonl(labels.get(s["id"], [])))
        items = at.set_items(name, at.Roots(tmp, "-"))
        preds = [at.predict(a, items) for a in arts]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    lines: list[str] = []
    at.score_report(items, preds[0], SimpleNamespace(cls=None, pattern=None, screen=None, offset=0), lines)
    # Keep the aggregates; drop the per-node list (it quotes val copy).
    cut = next((x for x, ln in enumerate(lines) if ln.startswith("wrong nodes:")), len(lines))
    out = {"lines": lines[: cut + 1] if cut < len(lines) else lines, "headline": headline(lines)}
    out["lines"] = [ln.split(";")[0] if ln.startswith("wrong nodes:") else ln for ln in out["lines"]]
    if len(arts) > 1:
        out["against"] = val_diff(items, preds[0], preds[1])
    return out


def headline(lines: list[str]) -> dict:
    for ln in lines:
        if ln.startswith("headline"):
            w = ln.split()
            get = lambda k: float(w[w.index(k) + 1])  # noqa: E731
            fa = w[w.index("alarms") + 1]
            return {"P": get("P"), "R": get("R"), "F1": get("F1"), "false_alarms": fa,
                    "TP": int(get("TP")), "FP": int(get("FP")), "FN": int(get("FN")), "TN": int(get("TN"))}
    return {}


def val_diff(items, pa, pb) -> dict:
    """Counts only: nodes whose class or pointer differs, and which artefact the label sides with."""
    c = Counter()
    for it, ra, rb in zip(items, pa, pb):
        labs = it.node_labels()
        for x, (p, q) in enumerate(zip(ra, rb)):
            if p.cls == q.cls and p.pointer == q.pointer:
                continue
            c["differ"] += 1
            rec = labs.get(x)
            if rec is None or rec["class"] not in at.CLASSES:
                c["unlabelled"] += 1
                continue
            red = rec["class"] in at.REDUNDANT
            a_ok, b_ok = (p.cls != "KEEP") == red, (q.cls != "KEEP") == red
            c["A right" if a_ok and not b_ok else "B right" if b_ok and not a_ok else "both or neither"] += 1
    return dict(c)


# -- repos ---------------------------------------------------------------------------------------------------------


def disclosed_files(manifest: Path = REF_MANIFEST) -> dict[str, list[str]]:
    """{repo: [path, ...]} of every disclosed reference screen's source file (paths relative to the repo root)."""
    try:
        m = json.loads(manifest.read_text())
    except (OSError, ValueError):
        raise EvalError(f"{manifest} missing or unreadable: the disclosed reference files cannot be skipped") from None
    out: dict[str, list[str]] = {}
    for sc in m.get("screens", []):
        if sc.get("repo") and sc.get("path"):
            out.setdefault(sc["repo"], []).append(sc["path"])
    return {k: sorted(set(v)) for k, v in out.items()}


def repo_group(name: str, disclosed: dict[str, list[str]]) -> str:
    return "reference" if name in REFERENCE_REPOS or name in disclosed else "unseen"


def compile_net(art_dir: Path, dest: Path) -> Path:
    art = mdl.load_artifact(art_dir)
    from lint import compile as net_compile  # author_tool put SLOP on sys.path
    dest.write_bytes(net_compile.compile_artifact(art, (art_dir / "vocab.txt").read_bytes()))
    return dest


def check_repo(path: Path) -> None:
    real = os.path.realpath(path)
    parts = Path(real).parts
    for s in SEALED:
        if s in parts:
            raise EvalError(f"refusing {path}: {s} is a sealed repo")


def lint_repos(repos: dict[str, Path], nets: list[Path], node: str = "node",
               skip: dict[str, list[str]] | None = None) -> list[dict]:
    """One record per repo, file and error (evaluate_lint.mjs); ``skip`` {repo: [path]} files are not linted."""
    for p in repos.values():
        check_repo(p)
    req = {"repos": [{"name": n, "root": str(p), "skip": (skip or {}).get(n, [])} for n, p in repos.items()],
           "nets": [str(n) for n in nets], "rules": True}
    r = subprocess.run([node, "--no-warnings", str(LINT)], input=json.dumps(req), capture_output=True, text=True,
                       cwd=str(SLOP))
    if r.returncode != 0:
        tail = [ln for ln in r.stderr.strip().splitlines() if ln.strip()][-3:]
        raise EvalError("evaluate_lint.mjs failed: " + " | ".join(tail))
    return [json.loads(ln) for ln in r.stdout.splitlines() if ln.strip()]


def margin(logits) -> float:
    return at.margin(logits)


def one_line(text: str) -> str:
    return text.replace("", "{…}").replace("", " | ").replace("\t", " ").replace("\n", " ").strip()


def node_ref(nodes: list[dict], x: int | None) -> str:
    if x is None:
        return "-"
    n = nodes[x]
    return f"{n['i']} {n['role']} {one_line(n['text'])[:60]}"


def summarise_repos(recs: list[dict], severities: dict[str, str], against: bool,
                    groups: dict[str, str] | None = None) -> dict:
    """Per-repo counts and the rows of flags.tsv, rules.tsv, against.tsv and errors.tsv (each row starts with the
    repo's group: unseen or reference)."""
    groups = groups or {}
    per: dict[str, Counter] = {}
    flags, rules, diffs, errors = [], [], [], []
    for rec in recs:
        if rec["kind"] == "repo":
            per[rec["name"]] = Counter(files=rec["files"], skipped=rec.get("skipped", 0))
            for path in rec.get("skip_missing", []):
                errors.append([rec["name"], path, "disclosed reference file not found in the repo (nothing to skip)"])
        elif rec["kind"] == "error":
            per[rec["repo"]]["errors"] += 1
            errors.append([rec["repo"], rec["path"], rec["error"]])
        elif rec["kind"] == "file":
            c = per[rec["repo"]]
            nodes = rec["nodes"]
            c["screens"] += 1
            c["nodes"] += len(nodes)
            pa = rec["preds"][0] if rec["preds"] else []
            c["scored"] += len(pa)
            for p in pa:
                if p["cls"] == "KEEP":
                    continue
                c["flagged"] += 1
                c[p["cls"]] += 1
                n = nodes[p["index"]]
                flags.append([groups.get(rec["repo"], "unseen"), rec["repo"], f"{rec['path']}:{n['loc'][0]}", n["role"], one_line(n["text"]), p["cls"],
                              node_ref(nodes, p["pointer"]), f"{margin(p['logits']):+.2f}"])
            for f in rec["findings"]:
                c["rule_findings"] += 1
                c[f"rule:{f['rule']}"] += 1
                for x, loc in zip(f["nodes"], f["locs"]):
                    n = next((m for m in nodes if m["i"] == x), None)
                    rules.append([groups.get(rec["repo"], "unseen"), rec["repo"], f"{rec['path']}:{loc[0] if loc else '?'}", f["rule"], f["code"],
                                  severities.get(f["rule"], "warning"), n["role"] if n else "-",
                                  one_line(n["text"]) if n else "-", f["message"]])
            if against and len(rec["preds"]) > 1:
                for p, q in zip(rec["preds"][0], rec["preds"][1]):
                    if p["cls"] != q["cls"] or p["pointer"] != q["pointer"]:
                        c["differ"] += 1
                        n = nodes[p["index"]]
                        diffs.append([groups.get(rec["repo"], "unseen"), rec["repo"], f"{rec['path']}:{n['loc'][0]}", n["role"], one_line(n["text"]),
                                      f"{p['cls']} → {node_ref(nodes, p['pointer'])} ({margin(p['logits']):+.2f})",
                                      f"{q['cls']} → {node_ref(nodes, q['pointer'])} ({margin(q['logits']):+.2f})"])
    return {"per_repo": {k: dict(v) for k, v in per.items()}, "flags": flags, "rules": rules, "diffs": diffs,
            "errors": errors, "groups": {k: groups.get(k, "unseen") for k in per}}


def rule_meta(node: str = "node") -> dict[str, str]:
    r = subprocess.run([node, "--input-type=module", "-e",
                        "import { RULES } from './rules/index.js'; process.stdout.write(JSON.stringify(RULES));"],
                       capture_output=True, text=True, cwd=str(SLOP))
    if r.returncode != 0:
        raise EvalError("could not read the rule list: " + r.stderr.strip()[-200:])
    return {x["id"]: x["severity"] for x in json.loads(r.stdout)}


# -- report --------------------------------------------------------------------------------------------------------


def tsv(path: Path, head: list[str], rows: list[list]) -> None:
    path.write_text("\t".join(head) + "\n" + "".join("\t".join(str(c) for c in r) + "\n" for r in rows), encoding="utf-8")


def run_name(art_dir: Path) -> str | None:
    """r1 for runs/r1/work (any depth under runs/<run>/), else None."""
    try:
        rel = art_dir.resolve().relative_to((SLOP / "runs").resolve())
    except ValueError:
        return None
    return rel.parts[0] if len(rel.parts) > 1 else None


def write_report(out: Path, info: dict, val: dict | None, repo: dict | None) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    md = [f"# Evaluation of {info['art']}", "", f"Artefact sha256 {info['sha'][:12]}"
          + (f", against {info['against']} ({info['sha_b'][:12]})" if info.get("against") else "")
          + f". Written {info['when']}. Local only (runs/*/eval/ is gitignored).", ""]
    summary = {"art": info["art"], "sha256": info["sha"], "against": info.get("against"), "when": info["when"]}
    if val is not None:
        (out / "val.txt").write_text("\n".join(val["lines"]) + "\n", encoding="utf-8")
        h = val["headline"]
        md += ["## Val (2,000 unseen generated screens)", "", "```", *val["lines"], "```", ""]
        summary["val"] = h
        if "against" in val:
            md += [f"Calls that differ from {info['against']} on val: {val['against']}", ""]
            summary["val_against"] = val["against"]
    if repo is not None:
        per = repo["per_repo"]
        ag = bool(info.get("against"))
        summary["repos"] = {}
        summary["repo_totals"] = {}
        all_rules = Counter()
        for group in GROUPS:
            names = [k for k in per if repo["groups"].get(k, "unseen") == group]
            if group == "unseen":
                md += ["## Unseen repos (the author was shown none of their files)", ""]
            else:
                md += ["## Reference repos (the projects behind the disclosed reference screens; those files skipped)", ""]
            if not names:
                md += ["none", ""]
                continue
            md += ["| Repo | UI files | skipped | screens | nodes | flagged | ECHO | NARRATE | DECOR | rule findings |"
                   + (" differ |" if ag else ""),
                   "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|" + ("---:|" if ag else "")]
            tot = Counter()
            for name in names:
                c = per[name]
                tot.update(c)
                md.append(f"| {name} | {c.get('files', 0)} | {c.get('skipped', 0)} | {c.get('screens', 0)} | "
                          f"{c.get('scored', 0)} | {c.get('flagged', 0)} | {c.get('ECHO', 0)} | {c.get('NARRATE', 0)} | "
                          f"{c.get('DECOR', 0)} | {c.get('rule_findings', 0)} |" + (f" {c.get('differ', 0)} |" if ag else ""))
            md.append(f"| **all** | {tot['files']} | {tot['skipped']} | {tot['screens']} | {tot['scored']} | "
                      f"{tot['flagged']} | {tot['ECHO']} | {tot['NARRATE']} | {tot['DECOR']} | {tot['rule_findings']} |"
                      + (f" {tot['differ']} |" if ag else ""))
            by_rule = Counter({k[5:]: v for k, v in tot.items() if k.startswith("rule:")})
            all_rules.update(by_rule)
            md += ["", "Rule-layer findings by rule: " + (", ".join(f"{k} {v}" for k, v in sorted(by_rule.items())) or "none"), ""]
            for name in names:
                summary["repos"][name] = {"group": group, **{kk: v for kk, v in per[name].items() if not kk.startswith("rule:")}}
            summary["repo_totals"][group] = {"files": tot["files"], "skipped": tot["skipped"], "screens": tot["screens"],
                                             "nodes": tot["scored"], "flagged": tot["flagged"],
                                             "rule_findings": tot["rule_findings"], "rules": dict(sorted(by_rule.items()))}
        md += ["Every flagged node: flags.tsv. Every rule finding: rules.tsv."
               + (" Calls that differ: against.tsv." if ag else "")
               + (f" Files the extractor could not read, and notes: errors.tsv ({len(repo['errors'])})." if repo["errors"] else ""), ""]
        tsv(out / "flags.tsv", ["group", "repo", "file:line", "role", "text", "class", "target", "margin"], repo["flags"])
        tsv(out / "rules.tsv", ["group", "repo", "file:line", "rule", "code", "severity", "role", "text", "message"], repo["rules"])
        tsv(out / "errors.tsv", ["repo", "file", "error"], repo["errors"])
        if ag:
            tsv(out / "against.tsv", ["group", "repo", "file:line", "role", "text", "A", "B"], repo["diffs"])
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return md


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="scripts/evaluate.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("art", help="artefact directory (artifact.npz, vocab.txt, meta.json), e.g. runs/r1/work")
    ap.add_argument("--against", metavar="ART2", help="a second artefact: list the nodes whose calls differ")
    ap.add_argument("--run", help="run name for runs/<run>/eval/ (default: the runs/<run>/ holding ART)")
    ap.add_argument("--out", help="report directory (default runs/<run>/eval)")
    ap.add_argument("--no-val", action="store_true", help="skip the val score")
    ap.add_argument("--no-repos", action="store_true", help="skip the repos")
    ap.add_argument("--repo", action="append", default=[], metavar="NAME=PATH", help="lint this repo instead of the "
                    "built-in list (repeatable)")
    ap.add_argument("--node", default="node")
    a = ap.parse_args(argv)
    art_dir = Path(a.art)
    run = a.run or run_name(art_dir)
    if not a.out and not run:
        ap.error("ART is not under runs/<run>/; give --run or --out")
    out = Path(a.out) if a.out else SLOP / "runs" / run / "eval"
    try:
        arts = [mdl.load_artifact(art_dir)] + ([mdl.load_artifact(Path(a.against))] if a.against else [])
    except (mdl.ArtifactError, OSError, ValueError) as e:
        print(f"evaluate: {e}", file=sys.stderr)
        return 2
    info = {"art": str(a.art), "sha": arts[0].sha256, "against": a.against, "sha_b": arts[1].sha256 if a.against else None,
            "when": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
    try:
        val = None
        if not a.no_val:
            screens, labels = load_val()
            val = score_set(arts, screens, labels, "val")
        repo = None
        if not a.no_repos:
            repos = dict(kv.split("=", 1) for kv in a.repo) if a.repo else REPOS
            repos = {k: Path(v).expanduser() for k, v in repos.items()}
            missing = [k for k, p in repos.items() if not p.is_dir()]
            repos = {k: p for k, p in repos.items() if p.is_dir()}
            disclosed = disclosed_files()
            groups = {k: repo_group(k, disclosed) for k in repos}
            with tempfile.TemporaryDirectory(prefix="slop-eval-net-") as td:
                nets = [compile_net(art_dir, Path(td) / "a.bin")]
                if a.against:
                    nets.append(compile_net(Path(a.against), Path(td) / "b.bin"))
                recs = lint_repos(repos, nets, a.node, skip=disclosed)
            repo = summarise_repos(recs, rule_meta(a.node), bool(a.against), groups)
            for k in missing:
                repo["errors"].append([k, "-", "repo not found"])
        md = write_report(out, info, val, repo)
    except EvalError as e:
        print(f"evaluate: {e}", file=sys.stderr)
        return 2
    # stdout: the summary only (the per-node lists stay in the files).
    print("\n".join(ln for ln in md if not ln.startswith("# ")))
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
