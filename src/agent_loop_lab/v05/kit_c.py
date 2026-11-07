"""v0.5 Phase C author kit (harness side): tools/, the r1/ and phase_b/ hand-overs, the two disclosure screens and
the handover record.

Used by scripts/v05c_prepare_author_dir.sh, scripts/v05c_launch.sh and scripts/v05c_run_build.py. Phase C is a new
session that improves Phase B's network (weight_program/v0_5/phase-b/artifact.dlw); its REVISION.edits is applied
on top of Phase B's weights. The author directory (default ``../HypaNet-v05c-author``, outside the repo) holds::

    packet/                 the frozen v0.3 packet (built by scripts/v03_build_packet.py, hash = registry)
    py                      experiments/v0_3/author_kit/py (byte copy): .venv Python, -I -B, tools/launch.py
    tools/launch.py         experiments/v0_5/phase_c/author_kit/launch.py: Phase B's launcher with the writable
                            directory output/v05-phase-c/
    tools/hn_guard/         the v0.3 AUTHOR_RUN guard (agent_loop_lab.v03.kit.install_tools, unchanged)
    tools/lib/v05_load.py   experiments/v0_5/phase_c/author_kit/v05_load.py (byte copy): the author's helper
    tools/lib/hn_v05/       edits.py      = src/agent_loop_lab/v05/edits.py with its two v0.3 imports re-pointed
                            revision.py   = src/agent_loop_lab/v05/revision.py (byte copy)
                            revision_c.py = src/agent_loop_lab/v05/revision_c.py (byte copy)
                            dlw3.py       = src/agent_loop_lab/v03/artifact.py with its one model import re-pointed
    r1/                     as Phase B: PROMPT.txt, replies/, artifact.dlw, artifact.sha256, R1_MANIFEST.json
    phase_b/                artifact.dlw, artifact.sha256 (Phase B's network), REVISION.edits, CIRCUITS.md,
                            REVISION.md, ITERATIONS.md, scratch/ (Phase B's own output, as archived by the Phase B
                            build) and PHASE_B_MANIFEST.json; never apply.json, BUILD_RECORD.json or attempts/
    disclosure/             DISCLOSURE.json + .sha256 (Phase B's 200 validation windows V001..V200, network_class
                            = Phase B's class), DISCLOSURE_W.json + .sha256 (W001..W200 from the second generator)
    output/v05-phase-c/     the author's only writable directory (empty at hand-over)

Every hash goes into experiments/v0_5/phase_c/HANDOVER.json. Shared pieces (hashing, r1 staging, the venv
inventory, prefix parsing, network_class recomputation) are agent_loop_lab.v05.kit_b's, imported unchanged.

CLI (``uv run python -m agent_loop_lab.v05.kit_c ...``)::

    install-tools TOOLS_DIR
    stage-r1 RUN_DIR PROMPT_TXT DEST_DIR [--no-rebuild-check] [--no-final-check]
    stage-b B_DIR R1_ARTIFACT DEST_DIR [--expect-sha SHA]
    check-disclosure DISCLOSURE.json --ids V|W [--expect N] [--artifact B_ARTIFACT]
    check-same-windows PHASE_C_DISCLOSURE.json PHASE_B_DISCLOSURE.json
    check-disjoint DISCLOSURE.json DISCLOSURE_W.json
    record AUTHOR_DIR RECORD_JSON [--kit] [--extra JSON] [--force]   (refuses to replace RECORD_JSON without --force)
    verify AUTHOR_DIR RECORD_JSON [--r1-dir RUN_DIR] [--b-dir B_DIR] [--disclosure-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from agent_loop_lab.v03 import kit as v03_kit
from agent_loop_lab.v05 import kit_b
from agent_loop_lab.v05.kit_b import (
    CLASS_NAMES,
    DISCLOSURE_TOP_KEYS,
    DISCLOSURE_WINDOW_KEYS,
    VENV_HOOK_NAMES,  # noqa: F401 - re-exported for the tests
    KitError,
    network_class_mismatches,
    parse_prefix,
    read_sha_file,
    sha256_bytes,
    sha256_file,
    stage_r1,
    tree_hashes,
    venv_state,
    writable,
)

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[2]
KIT = REPO / "experiments" / "v0_5" / "phase_c" / "author_kit"
PHASE = "v05-phase-c"
OUTPUT_DIR = f"output/{PHASE}"
REGISTRY_V03 = REPO / "experiments" / "registry_v0_3.yaml"

REVISION_C_SOURCE = _HERE / "revision_c.py"
LAUNCHER_SOURCE = KIT / "launch.py"
HELPER_SOURCE = KIT / "v05_load.py"

B_DELIVERABLES = ("REVISION.edits", "CIRCUITS.md", "REVISION.md", "ITERATIONS.md")
B_SCRATCH = "scratch"
B_FILES = ("artifact.dlw", "artifact.sha256")
B_MANIFEST = "PHASE_B_MANIFEST.json"
# what Phase B's archive holds besides its author output; none of it is handed over (it records the harness's
# view: apply report, build record, rejected attempts)
B_NOT_HANDED_OVER = ("apply.json", "BUILD_RECORD.json", "attempts")

DISCLOSURES = {"V": ("DISCLOSURE.json", "DISCLOSURE.sha256"), "W": ("DISCLOSURE_W.json", "DISCLOSURE_W.sha256")}
# never in a handed-over disclosure header: the v0.5 master seeds (sealed 20261104, OOD diagnosis 20261107/08, the
# Phase C disclosure 20261109 and held-out sets 20261110/11)
SEEDS = ("20261104", "20261107", "20261108", "20261109", "20261110", "20261111")


def vendored_files() -> dict[str, bytes]:
    """``tools/``-relative path -> bytes of the Phase C additions (the v0.3 guard is installed separately): Phase
    B's vendored reader, applier and DLW3 reader, plus revision_c.py and the Phase C helper."""
    files = dict(kit_b.vendored_files())
    rc = REVISION_C_SOURCE.read_bytes()
    if re.search(rb"^\s*from \.(?!revision\b)", rc, re.MULTILINE) or re.search(rb"^\s*from \.\.", rc, re.MULTILINE) \
            or re.search(rb"^\s*(import|from) agent_loop_lab", rc, re.MULTILINE):
        raise KitError("revision_c.py imports something other than .revision; it cannot be vendored byte for byte")
    files["lib/hn_v05/revision_c.py"] = rc
    files["lib/v05_load.py"] = HELPER_SOURCE.read_bytes()
    return files


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
            "src/agent_loop_lab/v05/edits.py": sha256_file(kit_b.EDITS_SOURCE),
            "src/agent_loop_lab/v05/revision.py": sha256_file(kit_b.REVISION_SOURCE),
            "src/agent_loop_lab/v05/revision_c.py": sha256_file(REVISION_C_SOURCE),
            "src/agent_loop_lab/v03/artifact.py": sha256_file(kit_b.ARTIFACT_SOURCE),
        },
        "substitutions": {"lib/hn_v05/edits.py": [list(p) for p in kit_b.EDITS_IMPORTS],
                          "lib/hn_v05/dlw3.py": [list(p) for p in kit_b.ARTIFACT_IMPORTS]},
    }


# -- Phase B --------------------------------------------------------------------------------------------


def check_b_source(b_dir: str | Path, r1_artifact: str | Path, *, expect_sha: str | None = None) -> dict[str, Any]:
    """Check Phase B's archive before anything is handed over; return what was checked.

    ``b_dir`` (weight_program/v0_5/phase-b): artifact.dlw matches artifact.sha256 (and ``expect_sha``); its DLW3
    metadata names r1's artefact and the sha256 of author_output/REVISION.edits; r1's weights + that file, applied
    by the revision applier, give exactly Phase B's weights (and bytes); author_output/ holds only the four
    deliverables and scratch/, matching BUILD_RECORD.json's output manifest; the four top-level copies equal
    author_output's."""
    import torch

    from agent_loop_lab.v03.artifact import deserialize_state_dict, serialize_state_dict
    from agent_loop_lab.v05.revision import apply_revision, build_problems

    b_dir = Path(b_dir)
    art = b_dir / "artifact.dlw"
    if not art.is_file():
        raise KitError(f"{art} missing (Phase B has not been built: scripts/v05b_run_build.py)")
    data = art.read_bytes()
    got = sha256_bytes(data)
    if got != read_sha_file(b_dir / "artifact.sha256"):
        raise KitError(f"{art} sha256 {got} != artifact.sha256")
    if expect_sha is not None and got != expect_sha:
        raise KitError(f"{art} sha256 {got} != expected {expect_sha}")
    out = b_dir / "author_output"
    if not out.is_dir():
        raise KitError(f"{out} missing")
    entries = sorted(os.listdir(out))
    allowed = set(B_DELIVERABLES) | {B_SCRATCH}
    stray = [e for e in entries if e not in allowed]
    if stray or any(e not in entries for e in B_DELIVERABLES):
        raise KitError(f"{out}: entries {entries}; expected exactly {sorted(B_DELIVERABLES)} and optionally scratch/")
    if (out / B_SCRATCH).exists() and not (out / B_SCRATCH).is_dir():
        raise KitError(f"{out}/scratch is not a directory")
    hashes = tree_hashes(out)  # refuses symlinks
    rec_path = b_dir / "BUILD_RECORD.json"
    if not rec_path.is_file():
        raise KitError(f"{rec_path} missing")
    rec = json.loads(rec_path.read_text())
    if rec.get("artifact_sha256") != got:
        raise KitError(f"BUILD_RECORD.json names artefact {rec.get('artifact_sha256')}, not {got}")
    if rec.get("output_files") != hashes:
        raise KitError("author_output/ differs from BUILD_RECORD.json's output manifest (not what Phase B wrote)")
    for f in B_DELIVERABLES:
        if (b_dir / f).is_file() and (b_dir / f).read_bytes() != (out / f).read_bytes():
            raise KitError(f"{b_dir / f} differs from author_output/{f}")
    r1_bytes = Path(r1_artifact).read_bytes()
    r1_sha = sha256_bytes(r1_bytes)
    sd, meta = deserialize_state_dict(data)
    rev = (out / "REVISION.edits").read_bytes()
    if meta.get("base_artifact_sha256") != r1_sha:
        raise KitError(f"Phase B's artefact names base {meta.get('base_artifact_sha256')}, not r1's {r1_sha}")
    if meta.get("revision_sha256") != sha256_bytes(rev):
        raise KitError("Phase B's artefact names a different REVISION.edits")
    sd2, report = apply_revision(deserialize_state_dict(r1_bytes)[0], rev.decode("utf-8"))
    if build_problems(report):
        raise KitError("Phase B's REVISION.edits does not apply cleanly on top of r1")
    if list(sd2) != list(sd) or not all(torch.equal(sd[k], sd2[k]) for k in sd):
        raise KitError("r1 + Phase B's REVISION.edits does not give Phase B's weights")
    if serialize_state_dict(sd2, meta) != data:
        raise KitError("r1 + Phase B's REVISION.edits does not give Phase B's artefact bytes")
    return {"artifact_sha256": got, "base_r1_sha256": r1_sha, "revision_sha256": sha256_bytes(rev),
            "rebuild_identical": True, "author_output_files": len(hashes), "metadata": meta}


def stage_b(b_dir: str | Path, r1_artifact: str | Path, dest: str | Path, *,
            expect_sha: str | None = None) -> dict[str, Any]:
    """Copy Phase B's network, deliverables and scratch/ into ``dest`` (must not exist); write PHASE_B_MANIFEST.json."""
    b_dir, dest = Path(b_dir), Path(dest)
    checked = check_b_source(b_dir, r1_artifact, expect_sha=expect_sha)
    if dest.exists():
        raise KitError(f"{dest} exists")
    dest.mkdir(parents=True)
    for f in B_FILES:
        shutil.copyfile(b_dir / f, dest / f)
    out = b_dir / "author_output"
    for f in B_DELIVERABLES:
        shutil.copyfile(out / f, dest / f)
    if (out / B_SCRATCH).is_dir():
        shutil.copytree(out / B_SCRATCH, dest / B_SCRATCH, symlinks=True)
    files = tree_hashes(dest)
    for rel, h in files.items():
        src = b_dir / rel if rel in B_FILES else out / rel
        if sha256_file(src) != h:
            raise KitError(f"copied phase_b/{rel} differs from its source")
    if any(rel.split("/")[0] in B_NOT_HANDED_OVER for rel in files):
        raise KitError("a harness record would be handed over")
    manifest = {"source": "weight_program/v0_5/phase-b (artifact.dlw, artifact.sha256; author_output/: Phase B's "
                          "output directory as archived by its build)",
                "artifact_sha256": checked["artifact_sha256"], "base_r1_sha256": checked["base_r1_sha256"],
                "files": files}
    (dest / B_MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return {**{k: v for k, v in checked.items() if k != "metadata"}, "manifest_sha256": sha256_file(dest / B_MANIFEST),
            "files": files}


# -- disclosures ------------------------------------------------------------------------------------------


def _ood_family_names() -> tuple[str, ...]:
    from agent_loop_lab.synth_ood import FAMILIES

    return tuple(FAMILIES) if not isinstance(FAMILIES, dict) else tuple(FAMILIES.keys())


def _family_names() -> tuple[str, ...]:
    from agent_loop_lab.synth.families import FAMILY_NAMES

    return tuple(FAMILY_NAMES) + _ood_family_names()


def header_leaks(doc: dict[str, Any], *, family_names: tuple[str, ...] | None = None) -> list[str]:
    """What a disclosure's header (every top-level field except ``windows``) would leak: a generator family name
    or a v0.5 master seed."""
    head = json.dumps({k: v for k, v in doc.items() if k != "windows"}, sort_keys=True).lower()
    names = family_names if family_names is not None else _family_names()
    out = [f"family name {n!r}" for n in names if n.lower() in head]
    out += [f"seed {s}" for s in SEEDS if s in head]
    return out


def check_disclosure(path: str | Path, *, ids: str = "V", expect: int | None = 200,
                     artifact: str | Path | None = None,
                     family_names: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Screen a Phase C disclosure: exactly the allowed fields per window (no reference_policy_class, no session
    id or statistic), Phase B's header keys, ids ``<ids>001``.. in order, 50 per class when 200, a parsable prefix
    with the marker before block t-7, no family name or master seed in the header and no family name of the second generator in any prefix; with ``artifact`` (Phase B's), the header names it and every ``network_class`` equals its
    argmax at EOE_t (one causal pass)."""
    if ids not in DISCLOSURES:
        raise KitError(f"ids must be one of {sorted(DISCLOSURES)}")
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    problems: list[str] = []
    extra = set(doc) - DISCLOSURE_TOP_KEYS
    if extra:
        problems.append(f"unexpected top-level keys {sorted(extra)}")
    names = family_names if family_names is not None else _family_names()
    # prefixes are log text: only the second generator's family names (distinctive compound words) are looked for
    prefix_names = family_names if family_names is not None else _ood_family_names()
    problems += [f"header names a {x}" for x in header_leaks(doc, family_names=names)]
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
        if w["id"] != f"{ids}{n:03d}":
            problems.append(f"window {n}: id {w['id']!r} != {ids}{n:03d}")
        for k in ("oracle_label", "network_class"):
            if w[k] not in CLASS_NAMES:
                problems.append(f"{w['id']}: {k} {w[k]!r} is not a class name")
        if w["oracle_label"] in counts:
            counts[w["oracle_label"]] += 1
        low = str(w["prefix"]).lower()
        hit = [nm for nm in prefix_names if nm.lower() in low]
        if hit:
            problems.append(f"{w['id']}: prefix contains family name(s) {hit}")
        try:
            blocks, start = parse_prefix(w["prefix"])
            if start != len(blocks) - 8:
                problems.append(f"{w['id']}: marker before block {start}, expected t-7 = {len(blocks) - 8}")
        except ValueError as exc:
            problems.append(f"{w['id']}: prefix: {exc}")
    if expect == 200 and any(v != 50 for v in counts.values()):
        problems.append(f"class counts {counts} are not 50 each")
    net = doc.get("network")
    if artifact is not None and (not isinstance(net, dict) or net.get("artifact_sha256") != sha256_file(artifact)):
        got = net.get("artifact_sha256") if isinstance(net, dict) else None
        problems.append(f"header names network artefact {got}, not {sha256_file(artifact)}")
    if artifact is not None and not problems:
        mismatches = network_class_mismatches(wins, artifact)
        if mismatches:
            problems.append(f"network_class differs from the Phase B artefact's argmax for {mismatches[:10]}"
                            + (" ..." if len(mismatches) > 10 else ""))
    if problems:
        raise KitError(f"{path}: " + "; ".join(problems[:20]))
    return {"windows": len(wins), "ids": ids, "class_counts": counts, "sha256": sha256_file(path),
            "network_class_checked_against": None if artifact is None else sha256_file(artifact)}


def check_same_windows(path_c: str | Path, path_b: str | Path) -> dict[str, Any]:
    """The Phase C V-disclosure holds exactly Phase B's windows: same ids, prefixes and oracle labels, in order
    (only network_class may differ)."""
    c = json.loads(Path(path_c).read_text(encoding="utf-8"))["windows"]
    b = json.loads(Path(path_b).read_text(encoding="utf-8"))["windows"]
    if len(c) != len(b):
        raise KitError(f"{len(c)} windows, Phase B's disclosure has {len(b)}")
    bad = [wc.get("id") for wc, wb in zip(c, b)
           if any(wc.get(k) != wb.get(k) for k in ("id", "prefix", "oracle_label"))]
    if bad:
        raise KitError(f"windows differ from Phase B's disclosure (id, prefix or oracle_label): {bad[:10]}")
    changed = sum(1 for wc, wb in zip(c, b) if wc.get("network_class") != wb.get("network_class"))
    return {"windows": len(c), "same_as": sha256_file(path_b), "network_class_changed": changed}


def check_disjoint(path_v: str | Path, path_w: str | Path) -> dict[str, Any]:
    """No W window repeats a V window's prefix, and the two id ranges do not overlap."""
    v = json.loads(Path(path_v).read_text(encoding="utf-8"))["windows"]
    w = json.loads(Path(path_w).read_text(encoding="utf-8"))["windows"]
    if {x["id"] for x in v} & {x["id"] for x in w}:
        raise KitError("the two disclosures share window ids")
    shared = {x["prefix"] for x in v} & {x["prefix"] for x in w}
    if shared:
        raise KitError(f"{len(shared)} prefix(es) appear in both disclosures")
    return {"v": len(v), "w": len(w), "shared_prefixes": 0}


# -- the handover record ------------------------------------------------------------------------------


HANDED_OVER = ("packet", "tools", "r1", "phase_b", "disclosure")
ROOT_ENTRIES = [".venv", "disclosure", "output", "packet", "phase_b", "py", "r1", "tools"]


def author_dir_hashes(author_dir: str | Path) -> dict[str, Any]:
    d = Path(author_dir)
    return {**{k: tree_hashes(d / k) for k in HANDED_OVER}, "py": sha256_file(d / "py")}


def kit_files() -> dict[str, str]:
    return {p.name: sha256_file(p) for p in sorted(KIT.iterdir()) if p.is_file()}


def make_record(author_dir: str | Path, *, kit: bool, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    d = Path(author_dir).resolve()
    out_dir = d / OUTPUT_DIR
    if not out_dir.is_dir() or any(out_dir.iterdir()):
        raise KitError(f"{out_dir} missing or not empty")
    hashes = author_dir_hashes(d)
    return {
        "phase": PHASE,
        "kit": kit,
        "author_dir": str(d),
        "output_dir": OUTPUT_DIR,
        "packet_sha256": sha256_file(d / "packet" / "PACKET_MANIFEST.json"),
        "registry_v0_3_sha256": sha256_file(REGISTRY_V03),
        "r1_artifact_sha256": read_sha_file(d / "r1" / "artifact.sha256"),
        "r1_manifest_sha256": sha256_file(d / "r1" / kit_b.R1_MANIFEST),
        "r1_prompt_sha256": sha256_file(d / "r1" / "PROMPT.txt"),
        "b_artifact_sha256": read_sha_file(d / "phase_b" / "artifact.sha256"),
        "b_manifest_sha256": sha256_file(d / "phase_b" / B_MANIFEST),
        "disclosure_sha256": read_sha_file(d / "disclosure" / DISCLOSURES["V"][1]),
        "disclosure_w_sha256": read_sha_file(d / "disclosure" / DISCLOSURES["W"][1]),
        "handed_over": hashes,
        "venv_state": venv_state(d),
        "root_entries": sorted(os.listdir(d)),
        "output_entries": sorted(os.listdir(d / "output")),
        "kit_files": kit_files(),
        "model": "claude-fable-5-1",
        "fallback_model": None,
        "effort": "max",
        **(extra or {}),
    }


def verify(author_dir: str | Path, record: dict[str, Any], *, r1_dir: str | Path | None = None,
           b_dir: str | Path | None = None, disclosure_dir: str | Path | None = None) -> list[str]:
    """Problems (empty: all good) with the author dir against the handover record, and, when given, the repo's
    r1 run dir, Phase B archive and Phase C disclosure dir against what was handed over."""
    d = Path(author_dir).resolve()
    problems: list[str] = []
    if record.get("phase") != PHASE:
        problems.append(f"record phase {record.get('phase')!r} != {PHASE!r}")
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
    if kit_files() != record.get("kit_files"):
        problems.append("experiments/v0_5/phase_c/author_kit/ differs from the handover record (prompts, settings, "
                        "launcher or helper changed after the hand-over)")
    if r1_dir is not None:
        r1 = Path(r1_dir)
        try:
            got = sha256_file(r1 / "artifact.dlw")
            if got != read_sha_file(r1 / "artifact.sha256") or got != record.get("r1_artifact_sha256"):
                problems.append(f"{r1}/artifact.dlw does not match its sha256 file or the handover record")
        except (OSError, KitError) as exc:
            problems.append(f"r1 source: {exc}")
    if b_dir is not None:
        bd = Path(b_dir)
        try:
            got = sha256_file(bd / "artifact.dlw")
            if got != read_sha_file(bd / "artifact.sha256") or got != record.get("b_artifact_sha256"):
                problems.append(f"{bd}/artifact.dlw does not match its sha256 file or the handover record")
        except (OSError, KitError) as exc:
            problems.append(f"Phase B source: {exc}")
    if disclosure_dir is not None:
        dd = Path(disclosure_dir)
        for key, (js, sh) in (("disclosure_sha256", DISCLOSURES["V"]), ("disclosure_w_sha256", DISCLOSURES["W"])):
            try:
                got = sha256_file(dd / js)
                if got != read_sha_file(dd / sh) or got != record.get(key):
                    problems.append(f"{dd}/{js} does not match its sha256 file or the handover record")
            except (OSError, KitError) as exc:
                problems.append(f"disclosure source {js}: {exc}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m agent_loop_lab.v05.kit_c", description=__doc__,
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
    p = sub.add_parser("stage-b")
    p.add_argument("b_dir", type=Path)
    p.add_argument("r1_artifact", type=Path)
    p.add_argument("dest", type=Path)
    p.add_argument("--expect-sha")
    p = sub.add_parser("check-disclosure")
    p.add_argument("path", type=Path)
    p.add_argument("--ids", choices=sorted(DISCLOSURES), required=True)
    p.add_argument("--expect", type=int, default=200)
    p.add_argument("--artifact", type=Path)
    p = sub.add_parser("check-same-windows")
    p.add_argument("path_c", type=Path)
    p.add_argument("path_b", type=Path)
    p = sub.add_parser("check-disjoint")
    p.add_argument("path_v", type=Path)
    p.add_argument("path_w", type=Path)
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
    p.add_argument("--b-dir", type=Path)
    p.add_argument("--disclosure-dir", type=Path)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "install-tools":
            out: Any = install_tools(a.tools_dir)
        elif a.cmd == "stage-r1":
            out = stage_r1(a.run_dir, a.prompt, a.dest, rebuild=not a.no_rebuild_check,
                           require_final=not a.no_final_check)
        elif a.cmd == "stage-b":
            out = stage_b(a.b_dir, a.r1_artifact, a.dest, expect_sha=a.expect_sha)
        elif a.cmd == "check-disclosure":
            out = check_disclosure(a.path, ids=a.ids, expect=a.expect, artifact=a.artifact)
        elif a.cmd == "check-same-windows":
            out = check_same_windows(a.path_c, a.path_b)
        elif a.cmd == "check-disjoint":
            out = check_disjoint(a.path_v, a.path_w)
        elif a.cmd == "record":
            if a.record.exists() and not a.force:
                raise KitError(f"{a.record} exists (the handover record is never replaced silently; --force)")
            extra = json.loads(a.extra.read_text()) if a.extra else None
            out = make_record(a.author_dir, kit=a.kit, extra=extra)
            a.record.parent.mkdir(parents=True, exist_ok=True)
            a.record.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        else:
            rec = json.loads(a.record.read_text())
            problems = verify(a.author_dir, rec, r1_dir=a.r1_dir, b_dir=a.b_dir, disclosure_dir=a.disclosure_dir)
            for line in problems:
                print("PROBLEM:", line, file=sys.stderr)
            if problems:
                return 1
            out = {"verified": str(a.author_dir), "record": str(a.record)}
    except (KitError, ValueError, OSError) as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(out, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
