"""v0.5 Phase C author kit: revision_c, tools, Phase B hand-over, disclosure screens, prepare, ./py isolation, helper,
build on top of Phase B, launcher check and refusals, audit, cost, and the prompt's leak screen.

Run ONLY with an explicit --basetemp (an author session's launcher refuses while pytest-of-$USER exists in a temp
root):

    uv run pytest tests/test_v05c_kit.py -q -p no:cacheprovider --basetemp=<scratch>/pytest_v05c

The dry prepare writes into $V05C_DRY_ROOT (default: a fresh directory under pytest's basetemp, so concurrent runs
with different basetemps do not share it; ./py denies it either way, as the author dir's siblings). It uses
the archived v0.3 Phase A artefact as a stand-in for r1's weights, a stand-in Phase B archive built from it, fake
disclosures, and never launches anything. Nothing here opens the sealed sets or the held-out sets.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from agent_loop_lab.v03.artifact import deserialize_state_dict, serialize_state_dict  # noqa: E402
from agent_loop_lab.v05 import kit_b, kit_c  # noqa: E402
from agent_loop_lab.v05.edits import READER_VERSION, zero_state  # noqa: E402
from agent_loop_lab.v05.revision import REVISION_VERSION, apply_revision, build_problems  # noqa: E402
from agent_loop_lab.v05.revision_c import REVISION_C_VERSION, apply_revision_c  # noqa: E402

KIT = REPO / "experiments" / "v0_5" / "phase_c" / "author_kit"
KIT_B = REPO / "experiments" / "v0_5" / "phase_b" / "author_kit"
STANDIN = REPO / "weight_program" / "v0_3" / "phase-a" / "artifact.dlw"
REAL_B = REPO / "weight_program" / "v0_5" / "phase-b"
REAL_R1 = REPO / "experiments" / "v0_5" / "runs" / "r1"
B_SHA = "223d074372bf726a0c89f28afe1e1fcf9b53e3bb0ff3973f0cc3acd932661001"
# hints from the post-hoc OOD diagnosis (experiments/v0_5/ood_analysis/FINDINGS.md) and the sealed evaluation that
# must never reach the author; matched case-insensitively in the Phase C prompt and the Phase C kit files handed over
FORBIDDEN_TERMS = ("progress veto", "veto", "first word", "first-word", "error line", "error-line", "false alarm",
                   "false-alarm", "sealed", "ood_analysis", "findings.md", "status.md", "artifacts/reports",
                   "in-distribution", "out-of-distribution", "out of distribution", "held out", "held-out",
                   "unseen", "generalis", "generaliz")
FORBIDDEN_WORDS = (r"\bEDR\b", r"\bOOD\b", r"\bv0\.3\b", r"\bv03\b", r"\bv0\.4\b")
B_METRICS = ("0.957", "0.959", "0.765", "31.4", "0.908", "6.8%", "3.2%", "0.321", "0.326", "0.292", "95.8")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _no_pytest_of():
    user = os.environ.get("USER", "")
    import tempfile

    for t in {os.path.realpath(tempfile.gettempdir()), "/tmp", "/private/tmp"}:
        assert not os.path.exists(os.path.join(t, f"pytest-of-{user}")), f"pytest-of-{user} appeared in {t}"


def _load_script(name: str):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), REPO / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -- revision_c: Phase B's applier with a named base -----------------------------------------------------------


def test_revision_c_is_phase_bs_applier_on_another_base():
    base = zero_state()
    base["head.bias"][:] = [1, 2, 3, 4]
    text = "head.bias[0] = 5\nhead.bias[1:3] *= 2\n"
    sd, rep = apply_revision_c(base, text)
    sd_b, rep_b = apply_revision(base, text)
    assert all(torch.equal(sd[k], sd_b[k]) for k in sd) and sd["head.bias"].tolist() == [5, 4, 6, 4]
    assert rep["revision"]["version"] == REVISION_C_VERSION and "Phase B's weights" in rep["revision"]["base"]
    assert rep_b["revision"]["version"] == REVISION_VERSION and "r1's weights" in rep_b["revision"]["base"]
    assert {k: v for k, v in rep.items() if k not in ("revision", "revision_version")} == \
        {k: v for k, v in rep_b.items() if k != "revision"}
    _sd, bad = apply_revision_c(base, "head.bias[9] = 1\n```\n")
    assert len(build_problems(bad)) == 2


def test_phase_b_harness_is_unchanged():
    files = ["src/agent_loop_lab/v05/revision.py", "src/agent_loop_lab/v05/kit_b.py", "scripts/v05b_run_build.py",
             "scripts/v05b_launch.sh", "scripts/v05b_prepare_author_dir.sh", "scripts/v05b_audit_session.py",
             "scripts/v05b_cost.py", "tests/test_v05b_kit.py"] + \
        [f"experiments/v0_5/phase_b/author_kit/{p.name}" for p in KIT_B.iterdir() if p.is_file()]
    p = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *files], cwd=REPO)
    assert p.returncode == 0, "a Phase B harness file differs from HEAD"
    rec = json.loads((REPO / "experiments/v0_5/phase_b/HANDOVER.json").read_text())
    assert rec["kit_files"] == {p.name: kit_b.sha256_file(p) for p in sorted(KIT_B.iterdir()) if p.is_file()}


# -- kit files ---------------------------------------------------------------------------------------------------


def test_vendored_files_are_phase_bs_plus_revision_c_and_the_phase_c_helper():
    files = kit_c.vendored_files()
    b = kit_b.vendored_files()
    assert set(files) == set(b) | {"lib/hn_v05/revision_c.py"}
    for k in ("lib/hn_v05/__init__.py", "lib/hn_v05/edits.py", "lib/hn_v05/revision.py", "lib/hn_v05/dlw3.py"):
        assert files[k] == b[k], k
    assert files["lib/hn_v05/revision_c.py"] == (REPO / "src/agent_loop_lab/v05/revision_c.py").read_bytes()
    assert files["lib/v05_load.py"] == (KIT / "v05_load.py").read_bytes()
    assert files["lib/v05_load.py"] != b["lib/v05_load.py"]


def test_launcher_is_phase_bs_with_the_phase_c_output_dir():
    ours = (KIT / "launch.py").read_text()
    want = (KIT_B / "launch.py").read_text().replace("v05-phase-b", "v05-phase-c").replace(
        "Phase B", "Phase C").replace("_phase_b_hook", "_phase_c_hook")
    assert ours == want
    assert 'OUTPUT = os.path.join(os.path.realpath(ROOT), "output", "v05-phase-c")' in ours


def test_continue_and_resubmit_prompts_are_phase_bs():
    for n in ("prompt_continue.md", "prompt_resubmit.md"):
        assert (KIT / n).read_bytes() == (KIT_B / n).read_bytes(), n


def test_settings_scope_reads_and_writes_to_the_author_dir():
    s = json.loads((KIT / "settings_v05_phase_c.json").read_text())
    assert s["autoMemoryEnabled"] is False and s["permissions"]["defaultMode"] == "dontAsk"
    assert s["disableBypassPermissionsMode"] == "disable"
    allow, deny = s["permissions"]["allow"], set(s["permissions"]["deny"])
    assert allow == ["Read(./**)", "Glob(./**)", "Grep(./**)", "Write(./output/v05-phase-c/**)",
                     "Edit(./output/v05-phase-c/**)", "Bash(./py:*)"]
    home = Path.home()
    roots = [REPO, REPO.parent / "HypaNet-m2-author", REPO.parent / "HypaNet-v03-author",
             REPO.parent / "HypaNet-v04-author", REPO.parent / "HypaNet-v05b-author", REPO.parent / "HypaNet-v05-*",
             Path(f"/private/tmp/claude-{os.getuid()}"), Path(f"/tmp/claude-{os.getuid()}"),
             Path("/private/var/folders"), Path("/var/folders"), home / ".claude"]
    for r in roots:
        for tool in ("Read", "Grep", "Glob", "Edit", "Write"):
            assert f"{tool}(/{r}/**)" in deny, (tool, r)
    for p in ("./packet/**", "./tools/**", "./py", "./.venv/**", "./r1/**", "./phase_b/**", "./disclosure/**"):
        assert f"Edit({p})" in deny and f"Write({p})" in deny
    assert {"WebFetch", "WebSearch"} <= deny
    # no deny rule covers the author dir itself (a deny wins over the allow list)
    assert not any("HypaNet-v05c-author" in d or d.endswith("/HypaNet*/**)") for d in deny)
    # Phase B's deny list survives whole
    b = set(json.loads((KIT_B / "settings_v05_phase_b.json").read_text())["permissions"]["deny"])
    assert {d for d in b if "phase-b" not in d} <= deny
    # every sibling HypaNet* directory that exists now is denied, by name or by the HypaNet-v05-* rule
    for d in REPO.parent.glob("HypaNet*"):
        if d.name == "HypaNet-v05c-author":
            continue
        assert f"Read(/{d}/**)" in deny or d.name.startswith("HypaNet-v05-"), d


# -- the prompt --------------------------------------------------------------------------------------------------


PLACEHOLDERS = ("{R1_SHA256}", "{B_SHA256}", "{DISCLOSURE_SHA256}", "{DISCLOSURE_W_SHA256}")


def _render_prompt(text: str) -> str:
    # the launcher's sed: every placeholder gets a 64-hex value
    for i, ph in enumerate(PLACEHOLDERS):
        text = text.replace(ph, f"{i:x}" * 64)
    return text


def _sealed_numbers() -> set[str]:
    p = REPO / "artifacts/reports/v05_sealed.json"
    if not p.is_file():
        return set()
    out: set[str] = set()

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, float) and not x.is_integer() and abs(x) < 1e6:
            out.update({f"{x:.3f}", f"{x:.4f}"})
            if abs(x) <= 1:
                out.add(f"{100 * x:.1f}%")

    walk(json.loads(p.read_text()))
    return out


def test_prompt_has_every_placeholder_and_describes_the_lineage():
    p = (KIT / "prompt_v05_phase_c.md").read_text()
    for ph in PLACEHOLDERS:
        assert ph in p, ph
    assert set(re.findall(r"\{[A-Z0-9_]+\}", p)) == set(PLACEHOLDERS)
    flat = " ".join(p.split()).lower()
    for need in ("REVISION.edits", "CIRCUITS.md", "REVISION.md", "ITERATIONS.md", "V001", "W001", "W200",
                 "./py -m v05_load check", "./py -m v05_load b", "r1/PROMPT.txt", "r1/replies/", "phase_b/artifact.dlw",
                 "phase_b/REVISION.edits", "phase_b/scratch/", "disclosure/DISCLOSURE.json",
                 "disclosure/DISCLOSURE_W.json", "output/v05-phase-c/", "applied in order on top",
                 "Phase B's weights", "a second, independently written log generator", "some of its scenario families",
                 "no** `reference_policy_class`", "one command per bash call", "US$250", "this is **phase c**",
                 "1. **r1.**", "2. **phase b.**", "phase b's network's class", "same 200 windows phase b received"):
        assert need.lower() in flat, need
    rendered = _render_prompt(p)
    assert not re.search(r"\{[A-Z0-9_]+\}", rendered)
    launch = (REPO / "scripts/v05c_launch.sh").read_text()
    for ph in PLACEHOLDERS:
        assert f's/{ph}/' in launch, ph  # the launcher fills every placeholder of the prompt
        assert f'*"{ph}"*' in launch  # and refuses a prompt with one left


def test_prompt_and_handed_over_kit_files_leak_no_diagnosis_or_evaluation():
    texts = {
        "prompt": (KIT / "prompt_v05_phase_c.md").read_text(),
        "continue": (KIT / "prompt_continue.md").read_text(),
        "resubmit": (KIT / "prompt_resubmit.md").read_text(),
        "v05_load.py": (KIT / "v05_load.py").read_text(),
        "launch.py": (KIT / "launch.py").read_text(),
        "revision_c.py": (REPO / "src/agent_loop_lab/v05/revision_c.py").read_text(),
    }
    numbers = _sealed_numbers() | set(B_METRICS)
    for name, t in texts.items():
        low = t.lower()
        for term in FORBIDDEN_TERMS:
            assert term not in low, (name, term)
        assert not re.search(r"\bEDR\b", t) and not re.search(r"\bOOD\b", t), name
        for n in numbers:
            assert n not in t, (name, n)
        if name in ("prompt", "continue", "resubmit"):  # what the author is told (as Phase B's prompt test)
            for pat in FORBIDDEN_WORDS:
                assert not re.search(pat, t), (name, pat)
            for bad in ("HypaNet", "weight_program", "experiments/", "macro-F1 0.", "family names", "how many"):
                assert bad not in t, (name, bad)
    p = texts["prompt"]
    # the prompt never says "OOD" / out-of-distribution (not even in a file name); the families are never counted or
    # named
    assert not re.search(r"(?<![a-z])ood(?![a-z])|out[- ]of[- ]distribution", _render_prompt(p), re.IGNORECASE)
    from agent_loop_lab.synth.families import FAMILY_NAMES
    from agent_loop_lab.synth_ood import FAMILIES

    for fam in tuple(FAMILY_NAMES) + tuple(FAMILIES):
        assert fam not in p and fam.replace("_", " ") not in p.lower(), fam
    assert not re.search(r"\b(?:four|six|ten|4|6|10) (?:of its )?(?:scenario )?famil", p.lower())
    assert p.count("reference_policy_class") == 1
    assert "no evaluation of your revision" in " ".join(p.split())


# -- Phase B hand-over (the real archive is read-only here; nothing sealed is opened) -----------------------------


def test_real_phase_b_archive_stages_cleanly(tmp_path):
    out = kit_c.stage_b(REAL_B, REAL_R1 / "artifact.dlw", tmp_path / "phase_b", expect_sha=B_SHA)
    assert out["artifact_sha256"] == B_SHA and out["rebuild_identical"] is True
    assert out["base_r1_sha256"] == kit_b.sha256_file(REAL_R1 / "artifact.dlw")
    files = sorted(out["files"])
    assert {"artifact.dlw", "artifact.sha256", "REVISION.edits", "CIRCUITS.md", "REVISION.md",
            "ITERATIONS.md"} <= set(files)
    assert all(f in kit_c.B_FILES + kit_c.B_DELIVERABLES or f.startswith("scratch/") for f in files), files
    for f in ("apply.json", "BUILD_RECORD.json"):
        assert not (tmp_path / "phase_b" / f).exists()
    assert not (tmp_path / "phase_b" / "attempts").exists()
    man = json.loads((tmp_path / "phase_b" / kit_c.B_MANIFEST).read_text())
    assert man["files"] == out["files"] and man["artifact_sha256"] == B_SHA
    # Phase B's own words, from before any evaluation: nothing from the sealed run or the diagnosis
    for p in (tmp_path / "phase_b").rglob("*"):
        if p.is_file() and p.suffix in (".md", ".py", ".txt", ".edits", ".json"):
            low = p.read_text(errors="replace").lower()
            for term in ("sealed", "ood_analysis", "status.md", "progress veto", "out-of-distribution"):
                assert term not in low, (p.name, term)
    with pytest.raises(kit_c.KitError, match="exists"):
        kit_c.stage_b(REAL_B, REAL_R1 / "artifact.dlw", tmp_path / "phase_b")
    with pytest.raises(kit_c.KitError, match="expected"):
        kit_c.check_b_source(REAL_B, REAL_R1 / "artifact.dlw", expect_sha="0" * 64)


def _fake_b(d: Path, r1_artifact: Path, rev: str, scratch: dict[str, str] | None = None) -> Path:
    """A stand-in Phase B archive: r1 + rev, built like scripts/v05b_run_build.py, with harness records."""
    r1 = r1_artifact.read_bytes()
    sd, _rep = apply_revision(deserialize_state_dict(r1)[0], rev)
    meta = {"experiment": "v0.5", "phase": "v05-phase-b", "builder": "revision applier",
            "reader_version": READER_VERSION, "revision_version": REVISION_VERSION,
            "base_artifact_sha256": _sha(r1), "revision_sha256": _sha(rev.encode()), "packet_sha256": "0" * 64}
    data = serialize_state_dict(sd, meta)
    out = d / "author_output"
    (out / "scratch").mkdir(parents=True)
    files = {"REVISION.edits": rev, "CIRCUITS.md": "# B circuits\n", "REVISION.md": "# B revision\n",
             "ITERATIONS.md": "# B iterations\n"}
    for k, v in files.items():
        (out / k).write_text(v)
        (d / k).write_text(v)
    for k, v in (scratch or {"look.py": "import v05_load\nprint(v05_load.load_r1())\n"}).items():
        (out / "scratch" / k).write_text(v)
    (d / "artifact.dlw").write_bytes(data)
    (d / "artifact.sha256").write_text(f"{_sha(data)}  artifact.dlw\n")
    (d / "apply.json").write_text('{"harness": "apply report: never handed over"}\n')
    (d / "attempts").mkdir()
    (d / "attempts/0001.json").write_text('{"status": "built"}\n')
    (d / "BUILD_RECORD.json").write_text(json.dumps({"artifact_sha256": _sha(data),
                                                     "output_files": kit_b.tree_hashes(out)}))
    return d


B_REV = "# stand-in Phase B revision\nhead.bias[0] = -1\nhead.bias[2] += 0.25\n"


def test_phase_b_source_checks(tmp_path):
    r1 = tmp_path / "r1.dlw"
    r1.write_bytes(STANDIN.read_bytes())
    b = _fake_b(tmp_path / "b", r1, B_REV)
    assert kit_c.check_b_source(b, r1)["rebuild_identical"] is True
    # a scratch file changed after the build
    (b / "author_output/scratch/look.py").write_text("changed\n")
    with pytest.raises(kit_c.KitError, match="output manifest"):
        kit_c.check_b_source(b, r1)
    b2 = _fake_b(tmp_path / "b2", r1, B_REV)
    (b2 / "author_output/apply.json").write_text("{}")
    with pytest.raises(kit_c.KitError, match="entries"):
        kit_c.check_b_source(b2, r1)
    b3 = _fake_b(tmp_path / "b3", r1, B_REV)
    (b3 / "author_output/REVISION.edits").write_text(B_REV + "head.bias[3] = 1\n")
    with pytest.raises(kit_c.KitError):
        kit_c.check_b_source(b3, r1)
    other = tmp_path / "other.dlw"
    other.write_bytes(serialize_state_dict(deserialize_state_dict(STANDIN.read_bytes())[0], {"x": 1}))
    with pytest.raises(kit_c.KitError, match="names base"):
        kit_c.check_b_source(_fake_b(tmp_path / "b4", r1, B_REV), other)


# -- disclosures --------------------------------------------------------------------------------------------------


def _blocks(session: str, target: str = "pkg/m", n: int = 10) -> list[str]:
    from agent_loop_lab.v03.rawop import RawOp
    from agent_loop_lab.v03.renderer import render_session

    ops = []
    for i in range(n):
        if i % 2 == 0:
            ops.append(RawOp.make(family="read_file", target=f"{target}{i}.py", args={"line_start": 1, "line_end": 9},
                                  outcome="SUCCESS", input_tokens=100 + i, output_tokens=5,
                                  timestamp=f"2026-03-14T09:05:{i:02d}.000Z").to_dict())
        else:
            ops.append(RawOp.make(family="run_tests", target="tests", args={"tests": ["t/test_a.py::test_one"]},
                                  outcome="ERROR", error_class="AssertionError", error_location="t/test_a.py",
                                  validation={"suite": "tests", "passed": 3, "total": 5}, input_tokens=50,
                                  output_tokens=9, timestamp=f"2026-03-14T09:06:{i:02d}.000Z").to_dict())
    return render_session(ops, session)


def _disclosure_doc(blocks: list[str], ids: str, artifact: Path, labels=("STALLED", "RECOVERY")) -> dict:
    from agent_loop_lab.v03 import scoring, tokenizer
    from agent_loop_lab.v03.artifact import load_artifact
    from agent_loop_lab.v03.infer import eoe_predictions

    m, _ = load_artifact(artifact)
    preds = eoe_predictions(m, tokenizer.encode_session(blocks))
    wins = []
    for n, (t, label) in enumerate(zip((7, 9), labels), start=1):
        wins.append({"id": f"{ids}{n:03d}", "prefix": scoring.render_disclosed_prefix(blocks, t), "oracle_label": label,
                     "network_class": kit_c.CLASS_NAMES[preds[t]]})
    return {"format": "v05-phase-c-disclosure/kit", "class_names": list(kit_c.CLASS_NAMES),
            "window_marker": kit_b.WINDOW_MARKER,
            "fields": {}, "windows": wins,
            "network": {"run": None, "artifact_sha256": _sha(Path(artifact).read_bytes()), "description": "kit"},
            "selection": "kit", "reference_policy": "none"}


def _write_doc(d: Path, stem: str, doc: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode()
    (d / f"{stem}.json").write_bytes(data)
    (d / f"{stem}.sha256").write_text(f"{_sha(data)}  {stem}.json\n")


def _fake_disclosures(d: Path, r1_artifact: Path, b_artifact: Path) -> tuple[Path, Path]:
    """(Phase C disclosure dir with both reports, Phase B's DISCLOSURE.json with r1's classes)."""
    v_blocks = _blocks("sess-val-1")
    _write_doc(d / "phase_c", "DISCLOSURE", _disclosure_doc(v_blocks, "V", b_artifact))
    _write_doc(d / "phase_c", "DISCLOSURE_W", _disclosure_doc(_blocks("sess-other-1", target="lib/q"), "W",
                                                              b_artifact, labels=("PRODUCTIVE", "OSCILLATING")))
    _write_doc(d / "phase_b", "DISCLOSURE", _disclosure_doc(v_blocks, "V", r1_artifact))
    return d / "phase_c", d / "phase_b" / "DISCLOSURE.json"


def test_disclosure_screens(tmp_path):
    r1 = tmp_path / "r1.dlw"
    r1.write_bytes(STANDIN.read_bytes())
    b = _fake_b(tmp_path / "b", r1, B_REV)
    dc, db = _fake_disclosures(tmp_path / "disc", r1, b / "artifact.dlw")
    v, w = dc / "DISCLOSURE.json", dc / "DISCLOSURE_W.json"
    assert kit_c.check_disclosure(v, ids="V", expect=2, artifact=b / "artifact.dlw")["windows"] == 2
    assert kit_c.check_disclosure(w, ids="W", expect=2, artifact=b / "artifact.dlw")["ids"] == "W"
    assert kit_c.check_same_windows(v, db)["windows"] == 2
    assert kit_c.check_disjoint(v, w)["shared_prefixes"] == 0
    with pytest.raises(kit_c.KitError, match="id"):
        kit_c.check_disclosure(w, ids="V", expect=2)
    with pytest.raises(kit_c.KitError, match="header names network artefact"):
        kit_c.check_disclosure(v, ids="V", expect=2, artifact=r1)  # the classes must be Phase B's, not r1's
    doc = json.loads(w.read_text())
    leaks = [("selection", "kit: families shard_migration, lockfile_churn"), ("selection", "master seed 20261109"),
             ("fields", {"note": "from abab_oscillation sessions"})]
    for key, val in leaks:
        bad = dict(doc, **{key: val})
        (tmp_path / "x.json").write_text(json.dumps(bad))
        with pytest.raises(kit_c.KitError, match="header names a"):
            kit_c.check_disclosure(tmp_path / "x.json", ids="W", expect=2)
    bad = json.loads(w.read_text())
    bad["windows"][0]["prefix"] = bad["windows"][0]["prefix"].replace("lib/q", "lib/merge_ping_pong")
    (tmp_path / "y.json").write_text(json.dumps(bad))
    with pytest.raises(kit_c.KitError, match="family name"):
        kit_c.check_disclosure(tmp_path / "y.json", ids="W", expect=2)
    bad = json.loads(w.read_text())
    bad["windows"][0]["family"] = "x"
    (tmp_path / "z.json").write_text(json.dumps(bad))
    with pytest.raises(kit_c.KitError, match="keys"):
        kit_c.check_disclosure(tmp_path / "z.json", ids="W", expect=2)
    bad = json.loads(v.read_text())
    bad["windows"][1]["oracle_label"] = "PRODUCTIVE"
    (tmp_path / "v2.json").write_text(json.dumps(bad))
    with pytest.raises(kit_c.KitError, match="differ from Phase B"):
        kit_c.check_same_windows(tmp_path / "v2.json", db)
    with pytest.raises(kit_c.KitError, match="share window ids|appear in both"):
        kit_c.check_disjoint(v, v)


# -- audit and cost ---------------------------------------------------------------------------------------------


def _ev_tool(name, inp, model="claude-fable-5-1", tid="t1"):
    return {"type": "assistant", "message": {"model": model, "content": [
        {"type": "tool_use", "id": tid, "name": name, "input": inp}]}}


def test_audit_phase_c_rules(tmp_path):
    mod = _load_script("v05c_audit_session.py")
    a = tmp_path / "author"
    (a / "output/v05-phase-c").mkdir(parents=True)
    events = [
        _ev_tool("Bash", {"command": "./py output/v05-phase-c/scratch/look.py"}, tid="a"),
        _ev_tool("Bash", {"command": "./py -m v05_load check"}, tid="b"),
        _ev_tool("Bash", {"command": "./py -m v05_load b"}, tid="b2"),
        _ev_tool("Bash", {"command": "./py -m pip install x"}, tid="c"),
        _ev_tool("Write", {"file_path": "output/v05-phase-c/REVISION.edits"}, tid="d"),
        _ev_tool("Write", {"file_path": "phase_b/REVISION.edits"}, tid="e"),
        _ev_tool("Write", {"file_path": "output/v05-phase-b/REVISION.edits"}, tid="e2"),
        _ev_tool("Read", {"file_path": str(REPO / "README.md")}, tid="f"),
        _ev_tool("Read", {"file_path": "../HypaNet-v05b-author/output/v05-phase-b/REVISION.md"}, tid="h"),
        _ev_tool("Read", {"file_path": "r1/replies/001.md"}, model="claude-opus-5-5", tid="g"),
        _ev_tool("Read", {"file_path": "phase_b/scratch/survey.py"}, tid="i"),
        {"type": "result", "total_cost_usd": 1.5, "permission_denials": [{"tool_use_id": "f"}]},
    ]
    log = tmp_path / "v05c.stream.jsonl"
    log.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    rep = mod.audit(log, a)
    v = rep["violations"]
    assert rep["phase"] == "v05-phase-c"
    assert any("./py -m pip" in x and x.startswith("[EXECUTED]") for x in v)
    assert any("read-only hand-over" in x and "phase_b/REVISION.edits" in x for x in v)
    assert any("write outside output/v05-phase-c/" in x and "v05-phase-b" in x for x in v)
    assert any("HypaNet-v05b-author" in x for x in v)
    assert any("claude-opus-5-5" in x for x in v)
    assert any(x.startswith("[refused]") and "README.md" in x for x in v)
    assert not any("-m v05_load" in x or "look.py" in x or "survey.py" in x for x in v), v
    assert not any("output/v05-phase-c/REVISION.edits" in x for x in v), v
    assert mod.main(["x", str(log), str(a)]) == 1


def test_cost_by_session_delta_and_cap(tmp_path):
    mod = _load_script("v05c_cost.py")
    assert mod.CAP_USD == 250.0 and mod.PHASE == "v05c"
    logs = tmp_path / "logs"
    logs.mkdir()
    res = lambda total, sid="S": json.dumps({"type": "result", "session_id": sid, "total_cost_usd": total})  # noqa: E731
    (logs / "v05c.stream.jsonl").write_text(res(10.0) + "\n")
    (logs / "v05c.c1.stream.jsonl").write_text(res(25.0) + "\n")
    (logs / "v05b.stream.jsonl").write_text(res(99.0, "B") + "\n")  # Phase B's spend is not Phase C's
    s = mod.summary(logs)
    assert s["phases"]["v05c"]["total_usd"] == 25.0 and mod.check(s)[0] == 0 and s["not_phase_c"] == ["v05b"]
    (logs / "v05c.r1.stream.jsonl").write_text(json.dumps({"type": "assistant", "message": {"id": "m", "model": "x"}}))
    (logs / "v05c.r1.launch.json").write_text(json.dumps({"max_budget_usd": 225.0}))
    s = mod.summary(logs)
    assert s["phases"]["v05c"]["total_usd"] == 250.0 and mod.check(s)[0] == 3


# -- the dry hand-over: prepare, ./py, helper, build, launcher check --------------------------------------------


R1_REPLIES = {"001.md": "Design.\n\n```edits\nhead.bias[0] = 0.5\n```\n", "002.md": "More.\nEND OF WEIGHTS\n"}


def _fake_run(d: Path, replies: dict[str, str], artifact: bytes) -> Path:
    (d / "replies").mkdir(parents=True)
    units = []
    for i, (name, text) in enumerate(sorted(replies.items()), start=1):
        (d / "replies" / name).write_text(text, encoding="utf-8")
        units.append({"turn": i, "file": name, "sha256": _sha(text.encode()), "cut_off": False})
    (d / "replies" / "INDEX.json").write_text(json.dumps({"units": units}))
    (d / "artifact.dlw").write_bytes(artifact)
    (d / "artifact.sha256").write_text(f"{_sha(artifact)}  artifact.dlw\n")
    return d


def _chmod_tree_writable(root: Path) -> None:
    for p in [root, *root.rglob("*")]:
        if not p.is_symlink() and ".venv" not in p.relative_to(root).parts:  # never the (kit-linked) repo venv
            p.chmod(0o755 if p.is_dir() else 0o644)


@pytest.fixture(scope="module")
def dry(tmp_path_factory):
    _no_pytest_of()
    if os.environ.get("V05C_DRY_ROOT"):
        root = Path(os.environ["V05C_DRY_ROOT"])
        if root.exists():
            _chmod_tree_writable(root)
            shutil.rmtree(root)
    else:
        root = tmp_path_factory.mktemp("v05c_dry")
    inputs = root / "inputs"
    run = _fake_run(inputs / "r1", R1_REPLIES, STANDIN.read_bytes())  # v0.3 A weights as r1 stand-in
    b = _fake_b(inputs / "phase-b", run / "artifact.dlw", B_REV)
    disc, b_disc = _fake_disclosures(inputs / "disclosures", run / "artifact.dlw", b / "artifact.dlw")
    author = root / "HypaNet-v05c-author"
    record = inputs / "HANDOVER.json"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("V05C_", "V05B_"))}
    env.update(V05C_KIT="1", V05C_R1_DIR=str(run), V05C_B_DIR=str(b), V05C_DISCLOSURE_DIR=str(disc),
               V05C_B_DISCLOSURE=str(b_disc), V05C_RECORD=str(record), V05C_NO_REBUILD_CHECK="1",
               V05C_DISCLOSURE_EXPECT="2", V05C_VENV_FROM=str(REPO / ".venv"))
    p = subprocess.run(["sh", str(REPO / "scripts/v05c_prepare_author_dir.sh"), str(author)], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=900)
    assert p.returncode == 0, p.stdout + p.stderr
    yield {"author": author, "run": run, "b": b, "disc": disc, "b_disc": b_disc, "record": record, "env": env,
           "inputs": inputs}
    _no_pytest_of()
    _chmod_tree_writable(root)  # so pytest (or the next run) can remove it


def _py(author: Path, *args: str, timeout: int = 600) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
    return subprocess.run([str(author / "py"), *args], cwd=author, env=env, capture_output=True, text=True,
                          timeout=timeout)


def _probe(a: Path, name: str, body: str) -> subprocess.CompletedProcess:
    script = a / "output/v05-phase-c/scratch" / f"{name}.py"
    script.parent.mkdir(exist_ok=True)
    script.write_text(body)
    return _py(a, f"output/v05-phase-c/scratch/{name}.py")


def test_dry_prepare_layout_and_record(dry):
    a = dry["author"]
    rec = json.loads(dry["record"].read_text())
    assert rec["kit"] is True and rec["phase"] == "v05-phase-c" and rec["fallback_model"] is None
    assert rec["model"] == "claude-fable-5-1" and rec["effort"] == "max"
    assert rec["r1_artifact_sha256"] == _sha(STANDIN.read_bytes())
    assert rec["b_artifact_sha256"] == _sha((dry["b"] / "artifact.dlw").read_bytes())
    assert rec["b_checks"]["rebuild_identical"] is True and rec["b_checks"]["base_r1_sha256"] == _sha(STANDIN.read_bytes())
    assert rec["disclosure_sha256"] == _sha((dry["disc"] / "DISCLOSURE.json").read_bytes())
    assert rec["disclosure_w_sha256"] == _sha((dry["disc"] / "DISCLOSURE_W.json").read_bytes())
    assert sorted(os.listdir(a)) == kit_c.ROOT_ENTRIES
    assert os.listdir(a / "output") == ["v05-phase-c"] and os.listdir(a / "output/v05-phase-c") == []
    assert (a / "py").read_bytes() == (REPO / "experiments/v0_3/author_kit/py").read_bytes()
    assert (a / "tools/launch.py").read_bytes() == (KIT / "launch.py").read_bytes()
    assert (a / "tools/lib/v05_load.py").read_bytes() == (KIT / "v05_load.py").read_bytes()
    assert (a / "tools/lib/hn_v05/revision_c.py").is_file()
    for k in ("packet", "tools", "r1", "phase_b", "disclosure", "py"):
        assert kit_c.writable(a / k) == [], k
    assert os.access(a / "output/v05-phase-c", os.W_OK)
    assert sorted(os.listdir(a / "disclosure")) == ["DISCLOSURE.json", "DISCLOSURE.sha256", "DISCLOSURE_W.json",
                                                    "DISCLOSURE_W.sha256"]
    assert sorted(os.listdir(a / "r1")) == ["PROMPT.txt", "R1_MANIFEST.json", "artifact.dlw", "artifact.sha256",
                                           "replies"]
    assert sorted(os.listdir(a / "phase_b")) == ["CIRCUITS.md", "ITERATIONS.md", "PHASE_B_MANIFEST.json",
                                                "REVISION.edits", "REVISION.md", "artifact.dlw", "artifact.sha256",
                                                "scratch"]
    assert (a / "phase_b/scratch/look.py").is_file()
    assert (a / "phase_b/artifact.dlw").read_bytes() == (dry["b"] / "artifact.dlw").read_bytes()
    assert (a / "r1/PROMPT.txt").read_bytes() == (REPO / "experiments/v0_5/PROMPT.txt").read_bytes()
    assert kit_c.verify(a, rec, r1_dir=dry["run"], b_dir=dry["b"], disclosure_dir=dry["disc"]) == []


def test_dry_author_dir_holds_nothing_from_the_diagnosis_reports_or_status(dry):
    a = dry["author"]
    handed = {p.relative_to(a).as_posix(): p for p in a.rglob("*")
              if p.is_file() and ".venv" not in p.relative_to(a).parts}
    names = {p.name for p in handed.values()}
    hashes = {kit_c.sha256_file(p) for p in handed.values()}
    banned = [REPO / "experiments/v0_5/STATUS.md"]
    for d in (REPO / "experiments/v0_5/ood_analysis", REPO / "artifacts/reports", REPO / "experiments/v0_5/sealed",
              REPO / "experiments/v0_5/phase_c/heldout"):
        if d.is_dir():
            banned += [p for p in d.rglob("*") if p.is_file() and p.stat().st_size < (64 << 20)]
    for p in banned:
        if p.stat().st_size:  # empty files (an __init__.py) say nothing
            assert kit_c.sha256_file(p) not in hashes, p
        if p.suffix in (".md", ".json") and p.name not in ("README.md",):
            assert p.name not in names or p.name.startswith(("DISCLOSURE", "R1_MANIFEST", "PHASE_B_MANIFEST",
                                                              "INDEX", "PACKET_MANIFEST", "state_dict")), p
    assert "DISCLOSURE_OOD_PRIVATE.json" not in names and "DISCLOSURE_PRIVATE.json" not in names
    assert not any(n in names for n in ("apply.json", "BUILD_RECORD.json", "HANDOVER.json", "FINDINGS.md"))
    # the harness that builds the author dir never names those sources outside its comments
    for script in ("scripts/v05c_prepare_author_dir.sh", "src/agent_loop_lab/v05/kit_c.py"):
        code = [ln for ln in (REPO / script).read_text().splitlines() if not ln.lstrip().startswith("#")]
        if script.endswith(".py"):
            text = (REPO / script).read_text()
            code = [ln for ln in text.split('"""', 2)[2].splitlines() if not ln.lstrip().startswith("#")]
        for bad in ("ood_analysis", "artifacts/reports", "STATUS.md", "v05_sealed", "heldout", "DISCLOSURE_OOD_PRIVATE"):
            assert not any(bad in ln for ln in code), (script, bad)


PROBE = r'''
import hashlib, json, sys
import model, v05_load
from hn_v05 import dlw3
b = v05_load.load_b()
rev, report = v05_load.apply_edits_file()
out = {"problems": v05_load.build_problems(report), "b": [], "labels": [], "ids": [], "policy": report["revision"]["base"]}
for w in v05_load.windows("all"):
    out["ids"].append(w["id"])
    out["b"].append(model.CLASS_NAMES[v05_load.classify(b, w["blocks"])])
    out["labels"].append(w["network_class"])
out["v_ids"] = [w["id"] for w in v05_load.windows()]
out["w_ids"] = [w["id"] for w in v05_load.windows_w()]
act = v05_load.activations(rev, v05_load.windows("W")[0]["blocks"], query_positions=[0])
out["act_keys"] = sorted(act)
out["rev_sha"] = hashlib.sha256(dlw3.serialize_state_dict(rev.state_dict(), {})).hexdigest()
r1 = v05_load.load_r1()
out["r1_sha"] = hashlib.sha256(dlw3.serialize_state_dict(r1.state_dict(), {})).hexdigest()
print(json.dumps(out))
'''

REVISION = "# dry-run Phase C revision\nhead.bias[0] = -3\nhead.bias[1] += 0.5\nlayers.3.mlp.fc2.bias[0:4] = [1, 2, 3, 4]\n"


def test_dry_py_helper_applies_on_top_of_phase_b_like_the_harness(dry):
    a = dry["author"]
    out = a / "output/v05-phase-c"
    (out / "scratch").mkdir(exist_ok=True)
    (out / "scratch/probe.py").write_text(PROBE)
    (out / "REVISION.edits").write_text(REVISION)
    p = _py(a, "output/v05-phase-c/scratch/probe.py")
    assert p.returncode == 0, p.stderr
    res = json.loads(p.stdout.strip().splitlines()[-1])
    assert res["problems"] == [] and "Phase B's weights" in res["policy"]
    assert res["ids"] == ["V001", "V002", "W001", "W002"] and res["v_ids"] == ["V001", "V002"]
    assert res["w_ids"] == ["W001", "W002"]
    assert res["b"] == res["labels"]  # the helper's Phase B network reproduces both disclosures' network_class
    assert {"logits", "attn", "eoe_index", "surface"} <= set(res["act_keys"])
    base, _ = deserialize_state_dict((dry["b"] / "artifact.dlw").read_bytes())
    sd, _rep = apply_revision_c(base, REVISION)
    assert res["rev_sha"] == _sha(serialize_state_dict(sd, {}))  # on top of B, exactly as the harness applies it
    r1_sd, _ = deserialize_state_dict(STANDIN.read_bytes())
    sd_r1, _ = apply_revision(r1_sd, REVISION)
    assert res["rev_sha"] != _sha(serialize_state_dict(sd_r1, {}))  # not on top of r1
    assert res["r1_sha"] == _sha(serialize_state_dict(r1_sd, {}))
    p = _py(a, "-m", "v05_load", "check")
    assert p.returncode == 0 and "would accept" in p.stdout and "phase_b/artifact.dlw" in p.stdout, p.stdout + p.stderr
    (out / "scratch/bad.edits").write_text("head.bias[4] = 1\n")
    p = _py(a, "-m", "v05_load", "check", "output/v05-phase-c/scratch/bad.edits")
    assert p.returncode == 1 and "WOULD REJECT" in p.stdout and "out of range" in p.stdout
    p = _py(a, "-m", "v05_load", "b")
    assert p.returncode == 0 and "identical weights" in p.stdout, p.stdout + p.stderr
    p = _py(a, "-m", "v05_load", "r1")
    assert p.returncode == 1 and "DIFFERENT" in p.stdout  # stand-in weights are not the fake replies' rebuild


REAL_LAYOUT_PROBE = r'''
import json, os, sys
tools, root, repo = sys.argv[1:4]
sys.path.insert(0, tools)
import launch
path = [p for p in launch._START_PATH
        if not p.startswith(os.path.dirname(tools)) and not os.path.realpath(p).startswith(repo)]
keep = launch._keep_roots(root=root, path=path, prefixes=(sys.base_prefix,))
print(json.dumps({"keep": keep, "deny": launch._deny_roots(root=root, keep=keep), "base": sys.base_prefix,
                  "output": launch.OUTPUT}))
'''


def _denied(path, deny):
    p = os.path.realpath(path)
    return any(p == d or p.startswith(d.rstrip("/") + "/") for d in deny)


def test_real_layout_denies_the_repo_phase_bs_author_dir_and_everything_else(dry):
    a = dry["author"]
    root = REPO.parent / "HypaNet-v05c-author"
    p = subprocess.run([str(a / ".venv/bin/python"), "-I", "-B", "-c", REAL_LAYOUT_PROBE, str(a / "tools"), str(root),
                        str(REPO)], capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout.strip().splitlines()[-1])
    deny, base = out["deny"], out["base"]
    must = [REPO, REPO / "datasets", REPO / "experiments/v0_5/ood_analysis", REPO / "artifacts/reports",
            REPO.parent / "HypaNet-v05b-author", REPO.parent / "HypaNet-v04-author", REPO.parent.parent / "x.md",
            Path.home() / ".claude", Path("/System/Volumes/Data") / str(REPO).lstrip("/"), Path("/Volumes/x"),
            Path("/usr/local/x"), Path("/private/tmp/x"), Path("/private/var/folders/x")]
    for m in must:
        assert _denied(m, deny), m
    for ok in (base, os.path.join(base, "lib"), "/usr/lib"):
        assert not _denied(ok, deny), ok
    assert str(root) not in deny
    text = (REPO / "scripts/v05c_prepare_author_dir.sh").read_text()
    assert 'AUTHOR_DIR="${1:-$PARENT/HypaNet-v05c-author}"' in text


def test_dry_py_confines_writes_to_the_phase_c_output(dry):
    a = dry["author"]
    stray = dry["inputs"] / "x.pt"
    cases = {
        "phase_b": (f"open({str(a / 'phase_b/REVISION.edits')!r}, 'a').write('x')\n", 3),
        "phase_b_output": (f"import os\nos.makedirs({str(a / 'output/v05-phase-b')!r})\n", 3),
        "disclosure_chmod": (f"import os\nos.chmod({str(a / 'disclosure/DISCLOSURE_W.json')!r}, 0o644)\n", 3),
        "root_file": (f"open({str(a / 'probe_root.txt')!r}, 'w').write('x')\n", 3),
        "torch_save": (f"import torch\ntorch.save({{'a': 1}}, {str(stray)!r})\n", 3),
        "sqlite": ("import sqlite3\n", 1),
        "random": ("import random\nrandom.random()\n", 3),
        "autograd": ("import torch\nx = torch.ones(2, requires_grad=True)\n(x * 2).sum().backward()\n", 3),
    }
    before = (a / "phase_b/REVISION.edits").read_bytes()
    for name, (body, code) in cases.items():
        p = _probe(a, f"w_{name}", body)
        assert p.returncode == code, (name, p.returncode, p.stderr[-400:])
    assert (a / "phase_b/REVISION.edits").read_bytes() == before
    assert not (a / "output/v05-phase-b").exists() and not stray.exists() and not (a / "probe_root.txt").exists()
    ok = a / "output/v05-phase-c/scratch/ok.pt"
    p = _probe(a, "w_ok", f"import torch\ntorch.save({{'a': 1}}, {str(ok)!r})\n")
    assert p.returncode == 0 and ok.is_file(), p.stderr[-400:]
    assert _py(a, "-c", "print(1)").returncode == 2


@pytest.mark.parametrize("target", ["repo_status", "phase_b_archive", "inputs", "claude_home"])
def test_dry_py_denies_reads_outside_the_author_dir(dry, target):
    a = dry["author"]
    path = {"repo_status": REPO / "experiments/v0_5/STATUS.md",
            "phase_b_archive": dry["b"] / "apply.json",
            "inputs": dry["disc"] / "DISCLOSURE_W.json",
            "claude_home": Path.home() / ".claude" / "settings.json"}[target]
    p = _probe(a, f"peek_{target}", f"print(len(open({str(path)!r}, 'rb').read()) > -1)\n")
    assert p.returncode == 3 and "AUTHOR GUARD: refused" in p.stderr, (p.returncode, p.stdout, p.stderr[-300:])
    p = _probe(a, f"peek_{target}", "import os\nprint(os.listdir('..'))\n")
    assert p.returncode == 3 and "AUTHOR GUARD: refused" in p.stderr


def _launch(dry, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    if env is None:
        env = dict(dry["env"], V05C_LOGDIR=str(dry["inputs"] / "logs"))
        for k in ("V05C_NO_REBUILD_CHECK", "V05C_DISCLOSURE_EXPECT", "V05C_VENV_FROM", "V05C_B_DISCLOSURE"):
            env.pop(k)
    return subprocess.run(["sh", str(REPO / "scripts/v05c_launch.sh"), *args], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=300)


def test_dry_launcher_refusals(dry):
    for args, msg in ((("phase-c", str(dry["author"])), "kit mode"), (("phase-b", str(dry["author"])), "unknown mode"),
                      (("continue", "12", str(dry["author"])), "K must be"),
                      (("resubmit", "4", "/dev/null", str(dry["author"])), "N must be")):
        p = _launch(dry, *args)
        assert p.returncode == 2 and msg in p.stderr, (args, p.stderr)
    # a directory that is not a Phase C author dir
    p = _launch(dry, "check", str(dry["inputs"]))
    assert p.returncode == 2 and "not a prepared v0.5 Phase C author dir" in p.stderr
    assert not (dry["inputs"] / "logs").exists()


def test_dry_launcher_check_mode(dry):
    p = _launch(dry, "check", str(dry["author"]))
    if p.returncode == 2 and "harness scratch" in p.stderr and "pytest-of-" in p.stderr:
        # another pytest run without --basetemp left pytest-of-$USER in a system temp root: refusing is right
        pytest.skip("pytest-of-$USER exists in a system temp root (the launcher refused, as it must)")
    assert p.returncode == 0 and "check passed" in p.stdout and "fallback=none" in p.stdout, p.stdout + p.stderr
    assert "would launch v05c " in p.stdout and "model=claude-fable-5-1" in p.stdout and "effort=max" in p.stdout
    assert "max_budget_usd=250.00" in p.stdout
    assert _sha((dry["b"] / "artifact.dlw").read_bytes()) in p.stdout
    assert _sha((dry["disc"] / "DISCLOSURE_W.json").read_bytes()) in p.stdout
    assert not (dry["inputs"] / "logs").exists()  # check writes nothing


def test_real_mode_launcher_and_prepare_refuse_without_touching_anything():
    env = {k: v for k, v in os.environ.items() if not k.startswith(("V05C_", "V05B_"))}
    real = REPO.parent / "HypaNet-v05c-author"
    if not real.exists():  # nothing prepared: check refuses before reading anything else
        p = subprocess.run(["sh", str(REPO / "scripts/v05c_launch.sh"), "check"], cwd=REPO, env=env,
                           capture_output=True, text=True, timeout=120)
        assert p.returncode == 2 and "not a prepared v0.5 Phase C author dir" in p.stderr, p.stderr
    # a kit variable without kit mode is refused before anything is created
    p = subprocess.run(["sh", str(REPO / "scripts/v05c_prepare_author_dir.sh"), "/nonexistent/HypaNet-v05c-author"],
                       cwd=REPO, env=dict(env, V05C_B_DIR="/tmp/x"), capture_output=True, text=True, timeout=120)
    assert p.returncode == 2 and "V05C_B_DIR is set but V05C_KIT is not 1" in p.stderr
    assert not Path("/nonexistent").exists()


def test_prepare_and_launcher_guards_in_text():
    prep = (REPO / "scripts/v05c_prepare_author_dir.sh").read_text()
    for need in ('[ -e "$RECORD" ] && die', 'phase_c/SESSION_ID" ] && die', "logs/v05c*.stream.jsonl",
                 'chmod -R a-w "$AUTHOR_DIR/.venv"', 'chmod 555 "$AUTHOR_DIR/output" "$AUTHOR_DIR"',
                 "HypaNet|HypaNet-m2-author|HypaNet-v03-author|HypaNet-v04-author|HypaNet-v05b-author|HypaNet-v05-*",
                 f'B_SHA_EXPECTED="{B_SHA}"', "check-same-windows", "check-disjoint", "--ids W", "--ids V",
                 '"$AUTHOR_DIR/phase_b"', 'DISCLOSURE_W.sha256 "', "stage-b"):
        assert need in prep, need
    launch = (REPO / "scripts/v05c_launch.sh").read_text()
    for need in ('MODEL="claude-fable-5-1"', 'EFFORT="max"', "export CLAUDE_CODE_NO_MODEL_FALLBACK=1",
                 "--tools Read,Glob,Grep,Write,Edit,Bash", "--disallowedTools Agent Workflow WebFetch WebSearch",
                 "--permission-mode dontAsk --permission-prompts none", "--strict-mcp-config", "--disable-slash-commands",
                 '--max-budget-usd "$REMAINING"', "scripts/v05c_cost.py", "scripts/v05c_audit_session.py",
                 "agent_loop_lab.v05.kit_c verify", 'NAME="v05c"', 'NAME="v05c.c$K"', 'NAME="v05c.r$N"',
                 '[ "$ERR_CHARS" -le 2001 ]', "HELDOUT/README.md", "git diff --quiet HEAD", B_SHA,
                 'mv "$SID_FILE" "$LOGDIR/$ATTEMPT.SESSION_ID"', "HypaNet-v05b-author"):
        assert need in launch, need
    code = "\n".join(ln for ln in launch.splitlines() if not ln.lstrip().startswith("#"))
    assert "--fallback-model" not in code and "wc -c" not in code
    b = (REPO / "scripts/v05b_launch.sh").read_text()
    claude_b = b[b.index("claude -p \"$PROMPT\""):b.index("< /dev/null")]
    claude_c = launch[launch.index("claude -p \"$PROMPT\""):launch.index("< /dev/null")]
    assert claude_b == claude_c  # the same flags as Phase B
    scrub_b = b[b.index("for v in CLAUDECODE"):b.index("cd \"$AUTHOR_DIR\"")]
    assert scrub_b in launch


# -- build on top of Phase B --------------------------------------------------------------------------------------


def _build(dry, out_dir: Path, *extra: str, b_dir: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "scripts/v05c_run_build.py"), "--author-dir", str(dry["author"]),
                           "--b-dir", str(b_dir or dry["b"]), "--out-dir", str(out_dir), *extra], cwd=REPO,
                          capture_output=True, text=True, timeout=600)


def test_dry_build_rejects_then_builds_on_top_of_phase_b(dry):
    a = dry["author"]
    out = a / "output/v05-phase-c"
    arch = dry["inputs"] / "weight_program_v0_5_phase-c"
    for f in ("CIRCUITS.md", "REVISION.md", "ITERATIONS.md"):
        if (out / f).exists():
            (out / f).unlink()
    (out / "REVISION.edits").write_text(REVISION + "head.bias[9] = 1\n")
    p = _build(dry, arch)
    assert p.returncode == 1 and "missing required file(s)" in p.stderr
    for f, t in (("CIRCUITS.md", "# circuits\n"), ("REVISION.md", "# revision\n"), ("ITERATIONS.md", "# it\n")):
        (out / f).write_text(t)
    p = _build(dry, arch)
    err = (arch / "attempts/0002.error.txt").read_text()
    assert p.returncode == 1 and "line 5: index 9 out of range" in err and len(err) <= 2001
    (out / "REVISION.edits").write_text(REVISION + "```\n")
    p = _build(dry, arch)
    assert p.returncode == 1 and "fence line" in (arch / "attempts/0003.error.txt").read_text()
    # the base must be a Phase B build (r1's run dir is not)
    p = _build(dry, arch, b_dir=dry["run"])
    assert p.returncode == 1
    (out / "REVISION.edits").write_text(REVISION)
    p = _build(dry, dry["b"])
    assert p.returncode != 0 and "Phase B's archive" in p.stderr + p.stdout
    p = _build(dry, arch)
    assert p.returncode == 0, p.stderr
    data = (arch / "artifact.dlw").read_bytes()
    assert (arch / "artifact.sha256").read_text().split()[0] == _sha(data)
    sd, meta = deserialize_state_dict(data)
    b_bytes = (dry["b"] / "artifact.dlw").read_bytes()
    want, _ = apply_revision(deserialize_state_dict(b_bytes)[0], REVISION)
    assert all(torch.equal(sd[k], want[k]) for k in sd)
    assert meta["base_artifact_sha256"] == _sha(b_bytes) and meta["phase"] == "v05-phase-c"
    assert meta["base_phase"] == "v05-phase-b" and meta["base_base_artifact_sha256"] == _sha(STANDIN.read_bytes())
    assert meta["revision_version"] == REVISION_C_VERSION
    rec = json.loads((arch / "BUILD_RECORD.json").read_text())
    assert rec["base_artifact_sha256"] == _sha(b_bytes) and rec["r1_artifact_sha256"] == _sha(STANDIN.read_bytes())
    assert rec["statements"] == 3 and rec["checks"]["author_phase_b_identical"] is True
    assert (arch / "author_output/scratch/probe.py").is_file() and (arch / "REVISION.edits").read_text() == REVISION
    apply = json.loads((arch / "apply.json").read_text())
    assert apply["artifact"]["sha256"] == _sha(data) and apply["edit_report"]["skipped"] == 0
    assert apply["revision"]["version"] == REVISION_C_VERSION
    p = _build(dry, arch)
    assert p.returncode != 0 and "first valid build" in p.stderr


def test_dry_verify_detects_tampering(dry):
    a = dry["author"]
    rec = json.loads(dry["record"].read_text())
    f = a / "phase_b/REVISION.edits"
    f.chmod(0o644)
    try:
        assert any("phase_b is writable" in x for x in kit_c.verify(a, rec))
        f.write_text(f.read_text() + "\n# tampered\n")
        assert any(x.startswith("phase_b/ differs") for x in kit_c.verify(a, rec))
    finally:
        f.write_text(f.read_text().replace("\n# tampered\n", ""))
        f.chmod(0o444)
    assert kit_c.verify(a, rec) == []
    probs = kit_c.verify(a, rec, b_dir=dry["run"])  # another base than the one handed over
    assert any("does not match its sha256 file or the handover record" in x for x in probs)
    assert any("record phase" in x for x in kit_c.verify(a, dict(rec, phase="v05-phase-b")))
    p = subprocess.run([sys.executable, "-m", "agent_loop_lab.v05.kit_c", "record", str(a), str(dry["record"]), "--kit"],
                       cwd=REPO, capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, PYTHONPATH=str(REPO / "src")))
    assert p.returncode == 2 and "never replaced silently" in p.stderr, p.stderr
