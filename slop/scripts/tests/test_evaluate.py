"""scripts/evaluate.py on tiny fixtures: the ./score path through a temporary examples root, the repo walk with the real
extractor, rule layer and runtime, the --against diff and the report files. Uses the committed r1 artefact; never
reads val."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest

from scripts import evaluate as ev

SLOP = Path(__file__).resolve().parents[2]
R1 = SLOP / "runs" / "r1" / "work"
FIX = SLOP / "rules" / "fixtures" / "screens"

pytestmark = pytest.mark.skipif(not (R1 / "artifact.npz").is_file() or shutil.which("node") is None,
                                reason="needs the r1 artefact and node")

APP = """export function App() {
  return (
    <main>
      <h1>Bake to order</h1>
      <p>Certainly! Here's a friendly message for your bakery.</p>
      <button>Start your journey</button>
    </main>
  );
}
"""


def tiny_repo(root: Path) -> Path:
    for rel, text in {
        "src/App.tsx": APP,
        "src/App.test.tsx": APP,  # test file: skipped
        "src/CardPreview.tsx": APP,  # preview file: skipped
        "node_modules/pkg/Thing.tsx": APP,  # dependency: skipped
        "blog-figures/fig.html": "<h1>Figure</h1>",  # skipped by name
        "tests/Page.tsx": APP,  # test folder: skipped
        "src/util.ts": "export const x = 1;\n",  # not a UI file
    }.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root


def labelled(name: str) -> tuple[dict, list[dict]]:
    s = json.loads((FIX / f"{name}.json").read_text(encoding="utf-8"))
    return s, [{"screen": s["id"], "loc": n["loc"], "class": "KEEP"} for n in s["nodes"]
               if n["role"] not in ("aria", "alt", "code", "comment")]


def test_score_set_keeps_aggregates_only_and_leaves_no_temp_files(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    art = ev.mdl.load_artifact(R1)
    screens, labels = [], {}
    for name in ("va_specific_neg", "clean_settings_neg"):
        s, lab = labelled(name)
        screens.append(s)
        labels[s["id"]] = lab
    res = ev.score_set([art, art], screens, labels, "val")
    assert os.listdir(tmp_path) == []  # the temporary examples root is gone
    assert res["lines"][0].startswith("2 screens, ")
    h = res["headline"]
    assert set(h) == {"P", "R", "F1", "false_alarms", "TP", "FP", "FN", "TN"}
    assert h["TP"] == 0 and h["FN"] == 0 and h["FP"] + h["TN"] == sum(len(v) for v in labels.values())
    assert res["lines"][-1].startswith("wrong nodes: ") and ";" not in res["lines"][-1]  # count only, no node list
    assert res["against"] == {}  # the same artefact twice: no call differs


def test_lint_repos_walks_ui_files_and_reports_network_and_rules(tmp_path):
    repo = tiny_repo(tmp_path / "bakery")
    net = ev.compile_net(R1, tmp_path / "a.bin")
    recs = ev.lint_repos({"bakery": repo}, [net, net])
    assert recs[0]["kind"] == "nets" and len(recs[0]["sha256"]) == 2
    head = [r for r in recs if r["kind"] == "repo"]
    assert head == [{"kind": "repo", "name": "bakery", "root": str(repo), "files": 1, "skipped": 0, "skip_missing": []}]
    files = [r for r in recs if r["kind"] == "file"]
    assert [f["path"] for f in files] == ["src/App.tsx"]
    sev = ev.rule_meta()
    assert sev["slop/chatbot-residue"] == "error"
    s = ev.summarise_repos(recs, sev, against=True)
    assert s["per_repo"]["bakery"]["files"] == 1 and s["per_repo"]["bakery"]["screens"] == 1
    rules = {(r[3], r[4], r[5]) for r in s["rules"]}
    assert ("slop/chatbot-residue", "L44", "error") in rules
    assert ("slop/stock-cta", "L26", "warning") in rules
    assert all(r[0] == "unseen" and r[2].startswith("src/App.tsx:") for r in s["rules"] + s["flags"])
    assert s["diffs"] == []  # the same net twice


def test_disclosed_reference_files_are_skipped_and_reported_apart(tmp_path):
    repo = tiny_repo(tmp_path / "HypaChef")
    (repo / "src" / "Shown.tsx").write_text(APP, encoding="utf-8")
    skip = {"HypaChef": ["src/Shown.tsx", "src/Gone.tsx"]}
    net = ev.compile_net(R1, tmp_path / "a.bin")
    recs = ev.lint_repos({"HypaChef": repo}, [net], skip=skip)
    head = next(r for r in recs if r["kind"] == "repo")
    assert head["files"] == 1 and head["skipped"] == 1 and head["skip_missing"] == ["src/Gone.tsx"]
    assert [r["path"] for r in recs if r["kind"] == "file"] == ["src/App.tsx"]
    groups = {"HypaChef": ev.repo_group("HypaChef", skip)}
    assert groups == {"HypaChef": "reference"} and ev.repo_group("bakery", skip) == "unseen"
    s = ev.summarise_repos(recs, ev.rule_meta(), against=False, groups=groups)
    assert s["groups"] == {"HypaChef": "reference"} and all(r[0] == "reference" for r in s["rules"])
    assert s["errors"] == [["HypaChef", "src/Gone.tsx", "disclosed reference file not found in the repo (nothing to skip)"]]
    # every disclosed reference screen's (repo, path) is in the skip list, and the three reference repos are named
    disclosed = ev.disclosed_files()
    man = json.loads(ev.REF_MANIFEST.read_text())
    assert sum(len(v) for v in disclosed.values()) == len({(x["repo"], x["path"]) for x in man["screens"]})
    assert set(ev.REFERENCE_REPOS) <= set(disclosed) and not set(ev.REPOS) - set(ev.REFERENCE_REPOS) & set(disclosed)


def test_main_writes_the_report_files(tmp_path, capsys):
    repo = tiny_repo(tmp_path / "bakery")
    out = tmp_path / "eval"
    rc = ev.main([str(R1), "--no-val", "--repo", f"bakery={repo}", "--repo", f"gone={tmp_path / 'nope'}", "--out", str(out),
                  "--against", str(R1)])
    assert rc == 0
    for f in ("report.md", "summary.json", "flags.tsv", "rules.tsv", "errors.tsv", "against.tsv"):
        assert (out / f).is_file(), f
    summary = json.loads((out / "summary.json").read_text())
    tot = summary["repo_totals"]["unseen"]
    assert tot["files"] == 1 and tot["rules"]["slop/chatbot-residue"] == 1 and "reference" not in summary["repo_totals"]
    assert summary["repos"]["bakery"]["group"] == "unseen"
    report = (out / "report.md").read_text()
    assert "## Unseen repos" in report and "## Reference repos" in report
    assert "val" not in summary
    assert "repo not found" in (out / "errors.tsv").read_text()
    assert (out / "rules.tsv").read_text().splitlines()[0].split("\t") == ["group", "repo", "file:line", "rule", "code",
                                                                             "severity", "role", "text", "message"]
    assert "Certainly" not in capsys.readouterr().out  # stdout carries the summary, not the copy


def test_sealed_repos_are_refused_and_run_names_come_from_the_path(tmp_path):
    for name in ev.SEALED:
        with pytest.raises(ev.EvalError):
            ev.check_repo(tmp_path / name / "app")
    ev.check_repo(tmp_path / "bakery")
    assert ev.run_name(R1) == "r1"
    assert ev.run_name(tmp_path) is None
    assert "runs/*/eval/" in (SLOP / ".gitignore").read_text().splitlines()
