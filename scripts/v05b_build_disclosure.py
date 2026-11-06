"""Build the v0.5 Phase B disclosed report: v0.3's 200 windows with the r1 network's classes.

    uv run python scripts/v05b_build_disclosure.py RUN_DIR [--out DIR]
    uv run python scripts/v05b_build_disclosure.py --artifact PATH --out DIR      (dry run)

``RUN_DIR`` is a finished v0.5 run (``experiments/v0_5/runs/r1/``). Its ``artifact.dlw`` must match
``artifact.sha256`` and ``apply.json`` (``agent_loop_lab.v05.scoring.check_apply_report``), and its
metadata must name the run (``load_run_artifact``). Writes, under ``DIR`` (default
``experiments/v0_5/phase_b/``):

* ``disclosure/DISCLOSURE.json``: the 200 disclosed windows, ids ``V001``..``V200``. Each window has
  exactly ``prefix``, ``oracle_label`` and ``network_class`` (the run's network, argmax at EOE_t).
  v0.5 has no reference policy, so there is **no** ``reference_policy_class`` field; the header says
  so. The header also names the run, its artefact sha256 and the selection rule. There is no session
  id, family, endpoint, logit or aggregate statistic, and nothing from any v0.3 network or policy;
* ``disclosure/DISCLOSURE.sha256``: ``"<sha256>  DISCLOSURE.json"``;
* ``DISCLOSURE_PRIVATE.json`` (repo only, never handed over): each id's session id, endpoint,
  family, oracle label and key, the input hashes and the v0.3 records it was checked against.

**Same windows as v0.3 Phase B.** The selection, rendering and network path are v0.3's, by import
(``v03.scoring.disclosure_selection``, ``render_disclosed_prefix``, ``network_predictions``, and
``scripts/v03_build_disclosure.py``'s block re-rendering). The builder checks
``experiments/v0_3/disclosure/DISCLOSURE.json`` against its ``DISCLOSURE.sha256`` and against the
``disclosure_sha256`` of ``experiments/v0_3/DISCLOSURE_PRIVATE.json``, then asserts that the selected
``(session_id, endpoint, key)`` list, in id order, equals the v0.3 private record's, and that every
window's ``prefix`` and ``oracle_label`` equal v0.3's. Only those two fields are compared; v0.3's
``network_class`` and ``reference_policy_class`` are never read into the output. Any difference is
an error, and nothing is written.

The val split only (hard-coded in v0.3); no model output affects the selection. Refuses to
overwrite any output file.

``--artifact PATH --out DIR`` (dry runs and tests): use any ``.dlw`` instead of a run (it must match
the ``artifact.sha256`` beside it; there is no ``apply.json`` or run-name check). ``DIR`` is
required and may not be ``experiments/v0_5/phase_b/`` or inside it. No ``--out`` may be inside
``experiments/v0_3/`` or ``experiments/v0_5/runs/``.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:  # pragma: no cover - uv run puts the package on the path
    sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from agent_loop_lab.schema import CLASS_NAMES  # noqa: E402
from agent_loop_lab.v03 import scoring as v03_scoring  # noqa: E402
from agent_loop_lab.v03 import sealed as v03_sealed  # noqa: E402
from agent_loop_lab.v03 import tokenizer  # noqa: E402
from agent_loop_lab.v05 import scoring as v05_scoring  # noqa: E402

OUT_ROOT = REPO / "experiments" / "v0_5" / "phase_b"
PUBLIC, SHA_FILE, PRIVATE = "DISCLOSURE.json", "DISCLOSURE.sha256", "DISCLOSURE_PRIVATE.json"
FORMAT = "v05-phase-b-disclosure/1"
V03_PUBLIC = REPO / "experiments" / "v0_3" / "disclosure" / PUBLIC
V03_SHA = REPO / "experiments" / "v0_3" / "disclosure" / SHA_FILE
V03_PRIVATE = REPO / "experiments" / "v0_3" / PRIVATE
FORBIDDEN_OUT = (REPO / "experiments" / "v0_3", v05_scoring.RUNS_ROOT)
V03_COMPARED = ("prefix", "oracle_label")  # the only v0.3 public fields the builder reads

SELECTION_TEXT = (
    "validation windows that pass the runtime gate and carry a scored oracle label; for each class in the order "
    "of class_names, candidates sorted by a fixed hash key of (session, endpoint), a window taken unless its "
    "session was already used, 50 per class: 200 windows from 200 distinct sessions; ids V001..V200 in key "
    "order. The rule uses no model output and was fixed before any v0.5 network existed")
REFERENCE_POLICY_TEXT = ("none: the v0.5 runs write a network only and have no reference policy, so the windows "
                         "carry no reference_policy_class field")


class DisclosureError(ValueError):
    pass


def output_paths(root: Path) -> tuple[Path, Path, Path]:
    return root / "disclosure" / PUBLIC, root / "disclosure" / SHA_FILE, root / PRIVATE


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dump(obj: Any) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode()


@functools.lru_cache(maxsize=1)
def _v03_builder() -> Any:
    """``scripts/v03_build_disclosure.py`` as a module (frozen; imported, never edited)."""
    path = REPO / "scripts" / "v03_build_disclosure.py"
    spec = importlib.util.spec_from_file_location("_v03_build_disclosure", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _blocks_for(session_ids: set[str]) -> dict[str, list[str]]:
    """Rendered blocks of the selected val sessions (v0.3's re-rendering from the raw file)."""
    return _v03_builder()._blocks_for(session_ids)


# ---------------------------------------------------------------------------
# Inputs: the v0.3 disclosure records and the network
# ---------------------------------------------------------------------------


def load_v03_reference(public: Path = V03_PUBLIC, sha_file: Path = V03_SHA, private: Path = V03_PRIVATE
                       ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """v0.3's windows (only ``id`` + :data:`V03_COMPARED` of the public file) and its private record.

    ``public`` must match ``sha_file`` and the private record's ``disclosure_sha256``; the ids of the
    two files must be ``V001``.. in the same order."""
    raw = Path(public).read_bytes()
    actual = _sha256_bytes(raw)
    recorded = v05_scoring.read_sha256_file(Path(sha_file), PUBLIC)
    if actual != recorded:
        raise DisclosureError(f"{public}: sha256 {actual} != {sha_file} {recorded}")
    prv_raw = Path(private).read_bytes()
    prv = json.loads(prv_raw)
    if prv.get("disclosure_sha256") != actual:
        raise DisclosureError(f"{private}: disclosure_sha256 {prv.get('disclosure_sha256')} != {public} {actual}")
    pub_w = [{k: w[k] for k in ("id", *V03_COMPARED)} for w in json.loads(raw)["windows"]]
    prv_w = [{k: w[k] for k in ("id", "session_id", "endpoint", "key", "oracle_label")} for w in prv["windows"]]
    ids = [f"V{n:03d}" for n in range(1, len(pub_w) + 1)]
    if [w["id"] for w in pub_w] != ids or [w["id"] for w in prv_w] != ids:
        raise DisclosureError(f"{public} / {private}: ids are not V001..V{len(pub_w):03d} in the same order")
    hashes = {"v03_disclosure_sha256": actual, "v03_private_sha256": _sha256_bytes(prv_raw)}
    return pub_w, prv_w, hashes


def load_run(run_dir: Path) -> tuple[Any, dict[str, Any]]:
    """The run's network, checked against ``artifact.sha256``, ``apply.json`` and the run name."""
    run_dir = Path(run_dir).resolve()
    dlw, sha = run_dir / "artifact.dlw", run_dir / "artifact.sha256"
    for f in (dlw, sha, run_dir / "apply.json"):
        if not f.is_file():
            raise DisclosureError(f"{f} missing (only a finished, applied run is disclosed)")
    spec, info = v05_scoring.load_run_artifact(dlw, sha, run=run_dir.name)
    apply_sha = v05_scoring.check_apply_report(run_dir, info["artifact_sha256"])
    model = v03_sealed.load_net(spec)  # re-checks the sha256; eval mode
    return model, {"run": run_dir.name, "run_dir": str(run_dir), "artifact_sha256": info["artifact_sha256"],
                   "apply_report_artifact_sha256": apply_sha,
                   "apply_json_sha256": _sha256_bytes((run_dir / "apply.json").read_bytes()),
                   "artifact_metadata": info["artifact_metadata"], "dry_run": False}


def load_any_artifact(path: Path) -> tuple[Any, dict[str, Any]]:
    """Dry runs: any ``.dlw``, checked against the ``artifact.sha256`` beside it."""
    path = Path(path).resolve()
    sha = path.parent / "artifact.sha256"
    if not path.is_file() or not sha.is_file():
        raise DisclosureError(f"{path} or {sha} missing")
    spec, info = v05_scoring.load_run_artifact(path, sha)
    model = v03_sealed.load_net(spec)
    return model, {"run": None, "artifact_path": str(path), "artifact_sha256": info["artifact_sha256"],
                   "artifact_metadata": info["artifact_metadata"], "dry_run": True}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build(data: v03_scoring.ValData, model: Any, source: Mapping[str, Any],
          ref_public: Sequence[Mapping[str, Any]], ref_private: Sequence[Mapping[str, Any]]
          ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Public and private records; raises :class:`DisclosureError` unless v0.3's windows are reproduced."""
    chosen = v03_scoring.disclosure_selection(data.rows)
    got = [(data.rows[i].session_id, int(data.rows[i].endpoint), key) for key, i in chosen]
    want = [(w["session_id"], int(w["endpoint"]), w["key"]) for w in ref_private]
    if got != want:
        n = next((k for k, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        raise DisclosureError(f"selected windows differ from the v0.3 private record (first at V{n + 1:03d}; "
                              f"{len(got)} selected, {len(want)} recorded)")
    row_idx = [i for _, i in chosen]
    sess_idx = sorted({int(data.row_session[i]) for i in row_idx})
    if len(sess_idx) != len(chosen):
        raise DisclosureError("two disclosed windows share a session")
    net, _lat = v03_scoring.network_predictions(model, data, sessions=sess_idx)
    blocks = _blocks_for({data.session_ids[s] for s in sess_idx})
    public_windows, private_windows = [], []
    for n, ((key, i), ref_pub, ref_prv) in enumerate(zip(chosen, ref_public, ref_private), start=1):
        r = data.rows[i]
        s, pos = int(data.row_session[i]), int(data.row_pos[i])
        vid = f"V{n:03d}"
        b = blocks[r.session_id]
        if tokenizer.encode_session(b).tokens != data.sessions[s].tokens:
            raise DisclosureError(f"{vid}: re-rendered blocks differ from the scored encoding")
        if any(line == v03_scoring.WINDOW_MARKER for blk in b for line in blk.split("\n")):
            raise DisclosureError(f"{vid}: a log line equals the window marker")
        if net[i] < 0:
            raise DisclosureError(f"{vid}: missing network prediction")
        w = {
            "id": vid,
            "prefix": v03_scoring.render_disclosed_prefix(b, pos),
            "oracle_label": r.label.to_class().name,
            "network_class": CLASS_NAMES[int(net[i])],
        }
        if ref_pub["id"] != vid or ref_prv["id"] != vid:
            raise DisclosureError(f"{vid}: v0.3 records out of order")
        for f in V03_COMPARED:
            if w[f] != ref_pub[f]:
                raise DisclosureError(f"{vid}: {f} differs from the v0.3 disclosure")
        if w["oracle_label"] != ref_prv["oracle_label"]:
            raise DisclosureError(f"{vid}: oracle_label differs from the v0.3 private record")
        public_windows.append(w)
        private_windows.append({"id": vid, "session_id": r.session_id, "endpoint": r.endpoint, "block": pos,
                                "family": r.family, "oracle_label": w["oracle_label"], "key": key})
    run, sha = source.get("run"), source["artifact_sha256"]
    net_name = f"the network written in run {run}" if run else "the dry-run artefact's network"
    public = {
        "format": FORMAT,
        "class_names": list(CLASS_NAMES),
        "window_marker": v03_scoring.WINDOW_MARKER,
        "network": {"run": run, "artifact_sha256": sha,
                    "description": f"{net_name}: the network whose classes this disclosure gives"},
        "selection": SELECTION_TEXT,
        "reference_policy": REFERENCE_POLICY_TEXT,
        "fields": {
            "prefix": ("the rendered log of blocks 0..t joined with newlines; the line '### window start' (not log "
                       "text: remove it before rendering or tokenising) stands immediately before block t-7, the "
                       "window's first block; block t, the window's last block, ends the text; after removing the "
                       "marker, every line that does not start with a space begins a new block"),
            "oracle_label": "the oracle label of the window ending at block t",
            "network_class": (f"argmax at EOE_t of {net_name} (artifact sha256 {sha}), from one "
                              "causal forward pass; ties go to the first class"),
        },
        "windows": public_windows,
    }
    private = {
        "format": FORMAT + "-private",
        "note": "repo only; never handed to the weight author",
        "split": v03_scoring.SPLIT,
        "key_format": v03_scoring.DISCLOSURE_KEY_FORMAT,
        "per_class": v03_scoring.PER_CLASS,
        "same_windows_as": ("experiments/v0_3/disclosure/DISCLOSURE.json (v0.3 Phase B): selection (session_id, "
                            "endpoint, key) in id order, prefix and oracle_label checked equal"),
        "source": dict(source),
        "dataset": data.inputs.get("cache_key"),
        "windows": private_windows,
    }
    return public, private


def _check_out(root: Path, dry_run: bool) -> str | None:
    for bad in FORBIDDEN_OUT:
        b = bad.resolve()
        if root == b or b in root.parents:
            return f"--out may not be {bad} or inside it"
    official = OUT_ROOT.resolve()
    if dry_run and (root == official or official in root.parents):
        return "--out of a dry run may not be experiments/v0_5/phase_b/ or inside it"
    return None


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir", type=Path, nargs="?", help="finished v0.5 run directory (experiments/v0_5/runs/r1)")
    ap.add_argument("--artifact", type=Path, help="dry runs only: any .dlw (with artifact.sha256 beside it)")
    ap.add_argument("--out", type=Path, help="output root (disclosure/ inside it); required with --artifact")
    args = ap.parse_args(argv)
    if (args.run_dir is None) == (args.artifact is None):
        ap.error("give RUN_DIR or --artifact PATH (not both)")
    if args.artifact is not None and args.out is None:
        ap.error("--artifact needs --out DIR")
    root = (args.out if args.out is not None else OUT_ROOT).resolve()
    problem = _check_out(root, args.artifact is not None)
    if problem:
        ap.error(problem)
    if args.run_dir is not None:
        rd = args.run_dir.resolve()
        if root == rd or rd in root.parents:
            ap.error("--out may not be inside RUN_DIR")
    paths = output_paths(root)
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        print(f"refusing: {', '.join(existing)} already exist(s)", file=sys.stderr)
        return 2
    t0 = time.time()
    try:
        model, source = load_run(args.run_dir) if args.run_dir is not None else load_any_artifact(args.artifact)
        ref_public, ref_private, ref_hashes = load_v03_reference(V03_PUBLIC, V03_SHA, V03_PRIVATE)
    except (DisclosureError, v05_scoring.ScoringError, v03_sealed.CheckFailed, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    torch.set_num_threads(v03_scoring.TORCH_THREADS)
    data = v03_scoring.load_val()
    try:
        public, private = build(data, model, source, ref_public, ref_private)
    except DisclosureError as exc:
        print(f"ERROR: {exc} (nothing written)", file=sys.stderr)
        return 1
    data_bytes = _dump(public)
    digest = _sha256_bytes(data_bytes)
    private.update(ref_hashes)
    private["disclosure_sha256"] = digest
    pub, sha, prv = paths
    pub.parent.mkdir(parents=True, exist_ok=True)
    pub.write_bytes(data_bytes)
    sha.write_text(f"{digest}  {PUBLIC}\n")
    prv.write_bytes(_dump(private))
    counts = {c: sum(w["network_class"] == c for w in public["windows"]) for c in CLASS_NAMES}
    print(f"disclosure: {len(public['windows'])} windows (v0.3's, reproduced), network classes {counts}, "
          f"artifact {source['artifact_sha256']}, sha256 {digest} ({time.time() - t0:.1f}s)")
    print(f"  {pub}\n  {prv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
