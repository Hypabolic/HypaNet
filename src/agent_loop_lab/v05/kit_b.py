"""v0.5 Phase B author kit (harness side): tools/, the r1/ hand-over, the disclosure screen and the handover record.

Used by scripts/v05b_prepare_author_dir.sh, scripts/v05b_launch.sh and scripts/v05b_run_build.py. The author
directory (default ``../HypaNet-v05b-author``, outside the repo) holds::

    packet/                 the frozen v0.3 packet (built by scripts/v03_build_packet.py, hash = registry)
    py                      experiments/v0_3/author_kit/py (byte copy): .venv Python, -I -B, tools/launch.py
    tools/launch.py         experiments/v0_5/phase_b/author_kit/launch.py: the v0.3 launcher + tools/lib on the path,
                            tightened (reads outside denied, writes only in output/v05-phase-b/)
    tools/hn_guard/         the v0.3 AUTHOR_RUN guard (agent_loop_lab.v03.kit.install_tools, unchanged)
    tools/lib/v05_load.py   experiments/v0_5/phase_b/author_kit/v05_load.py (byte copy): the author's helper
    tools/lib/hn_v05/       edits.py   = src/agent_loop_lab/v05/edits.py with its two v0.3 imports re-pointed
                            revision.py = src/agent_loop_lab/v05/revision.py (byte copy)
                            dlw3.py    = src/agent_loop_lab/v03/artifact.py with its one model import re-pointed
    r1/                     PROMPT.txt, replies/ (NNN.md, NNN.p<K>.md, INDEX.json), artifact.dlw, artifact.sha256,
                            R1_MANIFEST.json (sha256 of each of those files)
    disclosure/             DISCLOSURE.json, DISCLOSURE.sha256
    output/v05-phase-b/     the author's only writable directory (empty at hand-over)

The vendored copies differ from their sources only in the import lines named in :data:`EDITS_IMPORTS` and
:data:`ARTIFACT_IMPORTS` (each must occur exactly once); every hash goes into experiments/v0_5/phase_b/HANDOVER.json.

CLI (``uv run python -m agent_loop_lab.v05.kit_b ...``)::

    install-tools TOOLS_DIR
    stage-r1 RUN_DIR PROMPT_TXT DEST_DIR [--no-rebuild-check]
    check-disclosure DISCLOSURE.json [--expect N] [--artifact R1_ARTIFACT]
    record AUTHOR_DIR RECORD_JSON [--kit] [--extra JSON] [--force]   (refuses to replace RECORD_JSON without --force)
    verify AUTHOR_DIR RECORD_JSON [--r1-dir RUN_DIR] [--disclosure-dir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

from agent_loop_lab.v03 import kit as v03_kit

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[2]
KIT = REPO / "experiments" / "v0_5" / "phase_b" / "author_kit"
V03_KIT = REPO / "experiments" / "v0_3" / "author_kit"
PHASE = "v05-phase-b"
OUTPUT_DIR = f"output/{PHASE}"
REGISTRY_V03 = REPO / "experiments" / "registry_v0_3.yaml"

EDITS_SOURCE = _HERE / "edits.py"
REVISION_SOURCE = _HERE / "revision.py"
ARTIFACT_SOURCE = _HERE.parent / "v03" / "artifact.py"
LAUNCHER_SOURCE = KIT / "launch.py"
HELPER_SOURCE = KIT / "v05_load.py"
PY_SOURCE = V03_KIT / "py"

EDITS_IMPORTS = (
    ("from ..v03.model import PARAM_COUNT, PARAM_SHAPES",
     "from model import PARAM_COUNT, PARAM_SHAPES  # vendored (author tools/lib/hn_v05): the packet model"),
    ("    from ..v03.artifact import serialize_state_dict",
     "    from .dlw3 import serialize_state_dict  # vendored (author tools/lib/hn_v05)"),
)
ARTIFACT_IMPORTS = (
    ("from .model import PARAM_COUNT, PARAM_SHAPES, DoomLoopTransformer, zero_model",
     "from model import PARAM_COUNT, PARAM_SHAPES, DoomLoopTransformer, zero_model  # vendored: the packet model"),
)

# scripts/v05b_build_disclosure.py's header: format, class_names, window_marker, network (run, artifact sha256,
# description), selection (the rule, in words), reference_policy ("none"), fields, windows
DISCLOSURE_TOP_KEYS = {"format", "class_names", "window_marker", "network", "selection", "reference_policy", "fields",
                       "windows"}
DISCLOSURE_WINDOW_KEYS = {"id", "prefix", "oracle_label", "network_class"}
CLASS_NAMES = ("PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING")
WINDOW_MARKER = "### window start"
R1_FILES = ("PROMPT.txt", "artifact.dlw", "artifact.sha256")
R1_MANIFEST = "R1_MANIFEST.json"
NEAR_TIE = 1e-4  # logit gap below which a prefix pass may differ from a whole-session pass by rounding
FINAL_STATUSES = ("finished", "turn_cap", "context_limit", "cost_cap")  # agent_loop_lab.v05.runner.FINAL


class KitError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def read_sha_file(path: str | Path) -> str:
    p = Path(path)
    if not p.is_file() or not p.read_text().strip():
        raise KitError(f"{p} missing or empty")
    h = p.read_text().split()[0].strip()
    if not re.fullmatch(r"[0-9a-f]{64}", h):
        raise KitError(f"{p} does not hold a sha256")
    return h


def replace_once(text: str, pairs: Iterable[tuple[str, str]], what: str) -> str:
    lines = text.split("\n")
    for old, new in pairs:
        hits = [i for i, line in enumerate(lines) if line == old]
        if len(hits) != 1:
            raise KitError(f"{what}: expected exactly one line {old!r}, found {len(hits)}")
        lines[hits[0]] = new
    return "\n".join(lines)


def vendored_files() -> dict[str, bytes]:
    """``tools/``-relative path -> bytes of the Phase B additions (the v0.3 guard is installed separately)."""
    edits = replace_once(EDITS_SOURCE.read_text(encoding="utf-8"), EDITS_IMPORTS, "v05/edits.py")
    art = replace_once(ARTIFACT_SOURCE.read_text(encoding="utf-8"), ARTIFACT_IMPORTS, "v03/artifact.py")
    for name, text in (("edits.py", edits), ("dlw3.py", art)):
        if re.search(r"^\s*(from \.\.|from \.model|import agent_loop_lab|from agent_loop_lab)", text, re.MULTILINE):
            raise KitError(f"vendored {name} still imports from the harness package")
    rev = REVISION_SOURCE.read_bytes()
    if re.search(rb"^\s*from \.(?!edits\b)", rev, re.MULTILINE) or re.search(rb"^\s*from \.\.", rev, re.MULTILINE):
        raise KitError("revision.py imports something other than .edits; it cannot be vendored byte for byte")
    return {
        "lib/hn_v05/__init__.py": b"",
        "lib/hn_v05/edits.py": edits.encode("utf-8"),
        "lib/hn_v05/revision.py": rev,
        "lib/hn_v05/dlw3.py": art.encode("utf-8"),
        "lib/v05_load.py": HELPER_SOURCE.read_bytes(),
    }


def install_tools(tools_dir: str | Path) -> dict[str, Any]:
    """Write the author's tools/ (must not exist). Returns the hashes of every file written and of the sources."""
    tools = Path(tools_dir)
    if tools.exists():
        raise KitError(f"{tools} exists")
    guard = v03_kit.install_tools(tools, launcher=LAUNCHER_SOURCE)
    files = vendored_files()
    for rel, data in files.items():
        p = tools / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return {
        "files": {**guard, **{rel: sha256_bytes(d) for rel, d in files.items()}},
        "sources": {
            "launch.py": sha256_file(LAUNCHER_SOURCE),
            "v05_load.py": sha256_file(HELPER_SOURCE),
            "src/agent_loop_lab/v05/edits.py": sha256_file(EDITS_SOURCE),
            "src/agent_loop_lab/v05/revision.py": sha256_file(REVISION_SOURCE),
            "src/agent_loop_lab/v03/artifact.py": sha256_file(ARTIFACT_SOURCE),
        },
        "substitutions": {"lib/hn_v05/edits.py": [list(p) for p in EDITS_IMPORTS],
                          "lib/hn_v05/dlw3.py": [list(p) for p in ARTIFACT_IMPORTS]},
    }


# -- r1 ---------------------------------------------------------------------------------------------


def check_r1_source(run_dir: str | Path, *, rebuild: bool = True, require_final: bool = True) -> dict[str, Any]:
    """Check r1's run directory before anything is handed over; return what was checked."""
    from agent_loop_lab.v05.edits import build_run, reading_units

    run_dir = Path(run_dir)
    art = run_dir / "artifact.dlw"
    if not art.is_file():
        raise KitError(f"{art} missing (r1 has not been applied: scripts/v05_apply_reply.py)")
    want = read_sha_file(run_dir / "artifact.sha256")
    got = sha256_file(art)
    if got != want:
        raise KitError(f"{art} sha256 {got} != artifact.sha256 {want}")
    files, cuts, info = reading_units(run_dir)  # INDEX.json against the files and their sha256s
    status = None
    if require_final:
        rj = run_dir / "run.json"
        if not rj.is_file():
            raise KitError(f"{rj} missing: r1 has not finished")
        status = json.loads(rj.read_text()).get("status")
        if status not in FINAL_STATUSES:
            raise KitError(f"r1 status {status!r} is not final {FINAL_STATUSES}")
    rebuilt = None
    if rebuild:
        data, sha, _rep = build_run(run_dir)
        rebuilt = sha == got
        if not rebuilt:
            raise KitError(f"the frozen reader rebuilds r1 as {sha}, not the archived {got}")
    return {"artifact_sha256": got, "reading": info["mode"], "reading_units": [p.name for p in files],
            "status": status, "rebuild_identical": rebuilt}


def _reply_files(replies: Path) -> list[Path]:
    keep = [p for p in sorted(replies.iterdir())
            if p.is_file() and (re.fullmatch(r"\d+(\.p\d+)?\.md", p.name) or p.name == "INDEX.json")]
    stray = [p.name for p in sorted(replies.iterdir()) if p not in keep]
    if stray:
        raise KitError(f"{replies}: unexpected entries {stray} (only NNN.md, NNN.p<K>.md and INDEX.json are handed over)")
    return keep


def stage_r1(run_dir: str | Path, prompt: str | Path, dest: str | Path, *, rebuild: bool = True,
             require_final: bool = True) -> dict[str, Any]:
    """Copy r1's prompt, replies and weights into ``dest`` (must not exist) and write R1_MANIFEST.json."""
    run_dir, dest, prompt = Path(run_dir), Path(dest), Path(prompt)
    checked = check_r1_source(run_dir, rebuild=rebuild, require_final=require_final)
    if dest.exists():
        raise KitError(f"{dest} exists")
    (dest / "replies").mkdir(parents=True)
    shutil.copyfile(prompt, dest / "PROMPT.txt")
    shutil.copyfile(run_dir / "artifact.dlw", dest / "artifact.dlw")
    shutil.copyfile(run_dir / "artifact.sha256", dest / "artifact.sha256")
    for p in _reply_files(run_dir / "replies"):
        shutil.copyfile(p, dest / "replies" / p.name)
    files = {p.relative_to(dest).as_posix(): sha256_file(p) for p in sorted(dest.rglob("*")) if p.is_file()}
    manifest = {"source": "experiments/v0_5/runs/r1 (as written by the r1 harness) and experiments/v0_5/PROMPT.txt",
                "artifact_sha256": checked["artifact_sha256"], "files": files}
    (dest / R1_MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    # the copy must read exactly like the source
    from agent_loop_lab.v05.edits import reading_units

    reading_units(dest)
    return {**checked, "manifest_sha256": sha256_file(dest / R1_MANIFEST), "files": files}


# -- disclosure ---------------------------------------------------------------------------------------


def parse_prefix(text: str) -> tuple[list[str], int]:
    """A disclosed prefix -> (blocks 0..t, index of the marked block); same rule as v03.scoring.parse_disclosed_prefix."""
    blocks: list[list[str]] = []
    marked = None
    pending = False
    for line in text.split("\n"):
        if line == WINDOW_MARKER:
            if marked is not None or pending:
                raise ValueError("more than one window marker")
            pending = True
            continue
        if not line.startswith(" "):
            blocks.append([line])
            if pending:
                marked, pending = len(blocks) - 1, False
        else:
            if not blocks or pending:
                raise ValueError("body line outside a block")
            blocks[-1].append(line)
    if marked is None:
        raise ValueError("no window marker")
    return ["\n".join(b) for b in blocks], marked


def check_disclosure(path: str | Path, *, expect: int | None = 200, artifact: str | Path | None = None) -> dict[str, Any]:
    """Screen the Phase B disclosure: exactly the allowed fields (no reference_policy_class, no session id or
    statistic), ids V001.. in order, 50 per class when 200, a parsable prefix with the marker before block t-7;
    with ``artifact``, every ``network_class`` must equal that network's argmax at EOE_t (one causal pass)."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    problems: list[str] = []
    extra = set(doc) - DISCLOSURE_TOP_KEYS
    if extra:
        problems.append(f"unexpected top-level keys {sorted(extra)}")
    wins = doc.get("windows")
    if not isinstance(wins, list) or not wins:
        raise KitError(f"{path}: no windows")
    if expect is not None and len(wins) != expect:
        problems.append(f"{len(wins)} windows, expected {expect}")
    counts = {c: 0 for c in CLASS_NAMES}
    for n, w in enumerate(wins, start=1):
        keys = set(w) if isinstance(w, dict) else set()
        if keys != DISCLOSURE_WINDOW_KEYS:
            problems.append(f"window {n}: keys {sorted(keys)} != {sorted(DISCLOSURE_WINDOW_KEYS)}")
            continue
        if w["id"] != f"V{n:03d}":
            problems.append(f"window {n}: id {w['id']!r} != V{n:03d}")
        for k in ("oracle_label", "network_class"):
            if w[k] not in CLASS_NAMES:
                problems.append(f"{w['id']}: {k} {w[k]!r} is not a class name")
        if w["oracle_label"] in counts:
            counts[w["oracle_label"]] += 1
        try:
            blocks, start = parse_prefix(w["prefix"])
            if start != len(blocks) - 8:
                problems.append(f"{w['id']}: marker before block {start}, expected t-7 = {len(blocks) - 8}")
        except ValueError as exc:
            problems.append(f"{w['id']}: prefix: {exc}")
    if expect == 200 and any(v != 50 for v in counts.values()):
        problems.append(f"class counts {counts} are not 50 each")
    net = doc.get("network")
    if artifact is not None and isinstance(net, dict) and net.get("artifact_sha256") != sha256_file(artifact):
        problems.append(f"header names network artefact {net.get('artifact_sha256')}, not {sha256_file(artifact)}")
    mismatches: list[str] = []
    if artifact is not None and not problems:
        mismatches = network_class_mismatches(wins, artifact)
        if mismatches:
            problems.append(f"network_class differs from the r1 artefact's argmax for {mismatches[:10]}"
                            + (" ..." if len(mismatches) > 10 else ""))
    if problems:
        raise KitError(f"{path}: " + "; ".join(problems[:20]))
    return {"windows": len(wins), "class_counts": counts, "sha256": sha256_file(path),
            "network_class_checked_against": None if artifact is None else sha256_file(artifact)}


def network_class_mismatches(wins: list[dict], artifact: str | Path) -> list[str]:
    import torch

    from agent_loop_lab.v03 import tokenizer
    from agent_loop_lab.v03.artifact import load_artifact
    from agent_loop_lab.v03.infer import eoe_logits
    from agent_loop_lab.v03.model import predict

    m, _meta = load_artifact(artifact)
    m.eval()
    bad = []
    with torch.no_grad():
        for w in wins:
            blocks, _ = parse_prefix(w["prefix"])
            logits = eoe_logits(m, tokenizer.encode_session(blocks))[-1]
            c = int(predict(logits[None])[0])
            if CLASS_NAMES[c] != w["network_class"]:
                # the disclosure's pass runs over the whole session, this one over the prefix (the model is causal,
                # so they agree up to float32 kernel rounding): a flip within NEAR_TIE is not a mismatch
                gap = float(logits[c] - logits[CLASS_NAMES.index(w["network_class"])])
                if gap > NEAR_TIE:
                    bad.append(w["id"])
    return bad


# -- the handover record ------------------------------------------------------------------------------


def tree_hashes(root: str | Path, skip: Iterable[str] = ("__pycache__",)) -> dict[str, str]:
    root = Path(root)
    out = {}
    for p in sorted(root.rglob("*")):
        if any(part in skip for part in p.relative_to(root).parts):
            continue
        if p.is_symlink():
            raise KitError(f"{p}: symlinks are not handed over")
        if p.is_file():
            out[p.relative_to(root).as_posix()] = sha256_file(p)
    return out


def writable(root: str | Path) -> list[str]:
    root = Path(root)
    paths = [root] + (list(root.rglob("*")) if root.is_dir() else [])
    return [str(p) for p in paths if not p.is_symlink() and os.access(p, os.W_OK)]


HANDED_OVER = ("packet", "tools", "r1", "disclosure")
ROOT_ENTRIES = [".venv", "disclosure", "output", "packet", "py", "r1", "tools"]
VENV_HOOK_NAMES = ("sitecustomize.py", "usercustomize.py", "pyvenv.cfg")


def venv_state(author_dir: str | Path) -> dict[str, Any]:
    """The author's .venv: a kit symlink is recorded by its target; a private venv by an inventory digest over
    every entry (relative path, type, size, mtime_ns, symlink target) and the sha256 of every file that Python runs
    at start-up before ./py installs its guard (``*.pth``, sitecustomize.py, usercustomize.py, pyvenv.cfg)."""
    v = Path(author_dir) / ".venv"
    if v.is_symlink():
        return {"symlink": os.readlink(v)}
    h = hashlib.sha256()
    hooks: dict[str, str] = {}
    n = 0
    for p in sorted(v.rglob("*")):
        rel = p.relative_to(v).as_posix()
        st = p.lstat()
        if p.is_symlink():
            entry = f"L {rel} -> {os.readlink(p)}"
        elif p.is_dir():
            entry = f"D {rel}"
        else:
            entry = f"F {rel} {st.st_size} {st.st_mtime_ns}"
            if p.suffix == ".pth" or p.name in VENV_HOOK_NAMES:
                hooks[rel] = sha256_file(p)
        h.update(entry.encode("utf-8", "surrogateescape") + b"\n")
        n += 1
    return {"entries": n, "inventory_sha256": h.hexdigest(), "startup_files": hooks}


def author_dir_hashes(author_dir: str | Path) -> dict[str, Any]:
    d = Path(author_dir)
    return {**{k: tree_hashes(d / k) for k in HANDED_OVER}, "py": sha256_file(d / "py")}


def make_record(author_dir: str | Path, *, kit: bool, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    d = Path(author_dir).resolve()
    out_dir = d / OUTPUT_DIR
    if not out_dir.is_dir() or any(out_dir.iterdir()):
        raise KitError(f"{out_dir} missing or not empty")
    hashes = author_dir_hashes(d)
    rec = {
        "phase": PHASE,
        "kit": kit,
        "author_dir": str(d),
        "output_dir": OUTPUT_DIR,
        "packet_sha256": sha256_file(d / "packet" / "PACKET_MANIFEST.json"),
        "registry_v0_3_sha256": sha256_file(REGISTRY_V03),
        "r1_artifact_sha256": read_sha_file(d / "r1" / "artifact.sha256"),
        "r1_manifest_sha256": sha256_file(d / "r1" / R1_MANIFEST),
        "r1_prompt_sha256": sha256_file(d / "r1" / "PROMPT.txt"),
        "disclosure_sha256": read_sha_file(d / "disclosure" / "DISCLOSURE.sha256"),
        "handed_over": hashes,
        "venv_state": venv_state(d),
        "root_entries": sorted(os.listdir(d)),
        "output_entries": sorted(os.listdir(d / "output")),
        "kit_files": {p.name: sha256_file(p) for p in sorted(KIT.iterdir()) if p.is_file()},
        "model": "claude-fable-5-1",
        "fallback_model": None,
        "effort": "max",
        **(extra or {}),
    }
    return rec


def verify(author_dir: str | Path, record: dict[str, Any], *, r1_dir: str | Path | None = None,
           disclosure_dir: str | Path | None = None) -> list[str]:
    """Problems (empty: all good) with the author dir against the handover record, and, when given, the repo's
    r1 run dir and disclosure dir against what was handed over."""
    d = Path(author_dir).resolve()
    problems: list[str] = []
    if str(d) != record.get("author_dir"):
        problems.append(f"author dir {d} != recorded {record.get('author_dir')}")
    try:
        now = author_dir_hashes(d)
    except (OSError, KitError) as exc:
        return problems + [f"cannot hash the author dir: {exc}"]
    for k, v in now.items():
        if v != record["handed_over"].get(k):
            if isinstance(v, dict):
                old = record["handed_over"].get(k) or {}
                diff = sorted(set(v) ^ set(old)) + sorted(f for f in set(v) & set(old) if v[f] != old[f])
                problems.append(f"{k}/ differs from the handover record: {diff[:10]}")
            else:
                problems.append(f"{k} differs from the handover record")
    for k in HANDED_OVER + ("py",):
        w = writable(d / k)
        if w:
            problems.append(f"{k} is writable: {w[:3]}")
    if not (d / OUTPUT_DIR).is_dir():
        problems.append(f"{OUTPUT_DIR} missing")
    # what ./py could have changed without touching the handed-over trees: the author dir's own entries (it and
    # output/ are read-only), and the .venv, whose .pth / sitecustomize files run before the guard is installed
    for what, path, key in (("author dir", d, "root_entries"), ("output/", d / "output", "output_entries")):
        try:
            now_entries = sorted(os.listdir(path))
        except OSError as exc:
            problems.append(f"cannot list the {what}: {exc}")
            continue
        if now_entries != record.get(key):
            problems.append(f"{what} entries {now_entries} != recorded {record.get(key)}")
        if os.access(path, os.W_OK):
            problems.append(f"the {what} is writable")
    try:
        venv_now = venv_state(d)
    except OSError as exc:
        venv_now = {"error": str(exc)}
    if venv_now != record.get("venv_state"):
        changed = sorted(set(venv_now.get("startup_files", {}).items()) ^ set((record.get("venv_state") or {}).get(
            "startup_files", {}).items()))
        problems.append(f".venv differs from the handover record (start-up files changed: {[k for k, _ in changed][:10]})")
    if not (d / ".venv").is_symlink():
        w = writable(d / ".venv")
        if w:
            problems.append(f".venv is writable: {w[:3]}")
    kit_now = {p.name: sha256_file(p) for p in sorted(KIT.iterdir()) if p.is_file()}
    if kit_now != record.get("kit_files"):
        problems.append("experiments/v0_5/phase_b/author_kit/ differs from the handover record (prompts, settings, "
                        "launcher or helper changed after the hand-over)")
    if r1_dir is not None:
        r1 = Path(r1_dir)
        try:
            got = sha256_file(r1 / "artifact.dlw")
            if got != read_sha_file(r1 / "artifact.sha256") or got != record.get("r1_artifact_sha256"):
                problems.append(f"{r1}/artifact.dlw does not match its sha256 file or the handover record")
        except (OSError, KitError) as exc:
            problems.append(f"r1 source: {exc}")
    if disclosure_dir is not None:
        dd = Path(disclosure_dir)
        try:
            got = sha256_file(dd / "DISCLOSURE.json")
            if got != read_sha_file(dd / "DISCLOSURE.sha256") or got != record.get("disclosure_sha256"):
                problems.append(f"{dd}/DISCLOSURE.json does not match its sha256 file or the handover record")
        except (OSError, KitError) as exc:
            problems.append(f"disclosure source: {exc}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agent_loop_lab.v05.kit_b", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("install-tools")
    p.add_argument("tools_dir", type=Path)
    p = sub.add_parser("stage-r1")
    p.add_argument("run_dir", type=Path)
    p.add_argument("prompt", type=Path)
    p.add_argument("dest", type=Path)
    p.add_argument("--no-rebuild-check", action="store_true")
    p.add_argument("--no-final-check", action="store_true")
    p = sub.add_parser("check-disclosure")
    p.add_argument("path", type=Path)
    p.add_argument("--expect", type=int, default=200)
    p.add_argument("--artifact", type=Path)
    p = sub.add_parser("record")
    p.add_argument("author_dir", type=Path)
    p.add_argument("record", type=Path)
    p.add_argument("--kit", action="store_true")
    p.add_argument("--extra", type=Path, help="JSON file whose keys are added to the record")
    p.add_argument("--force", action="store_true", help="replace an existing RECORD_JSON")
    p = sub.add_parser("verify")
    p.add_argument("author_dir", type=Path)
    p.add_argument("record", type=Path)
    p.add_argument("--r1-dir", type=Path)
    p.add_argument("--disclosure-dir", type=Path)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "install-tools":
            out: Any = install_tools(a.tools_dir)
        elif a.cmd == "stage-r1":
            out = stage_r1(a.run_dir, a.prompt, a.dest, rebuild=not a.no_rebuild_check,
                           require_final=not a.no_final_check)
        elif a.cmd == "check-disclosure":
            out = check_disclosure(a.path, expect=a.expect, artifact=a.artifact)
        elif a.cmd == "record":
            if a.record.exists() and not a.force:
                raise KitError(f"{a.record} exists (the handover record is never replaced silently; --force)")
            extra = json.loads(a.extra.read_text()) if a.extra else None
            out = make_record(a.author_dir, kit=a.kit, extra=extra)
            a.record.parent.mkdir(parents=True, exist_ok=True)
            a.record.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        else:
            rec = json.loads(a.record.read_text())
            problems = verify(a.author_dir, rec, r1_dir=a.r1_dir, disclosure_dir=a.disclosure_dir)
            for line in problems:
                print("PROBLEM:", line, file=sys.stderr)
            if problems:
                return 1
            out = {"verified": str(a.author_dir), "record": str(a.record)}
    except (KitError, ValueError, OSError) as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
