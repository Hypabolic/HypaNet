"""Build the v0.3 Phase B disclosed report (spec v0.3 §9.3; experiments/v0_3/phase_b/README.md §2).

    uv run python scripts/v03_build_disclosure.py [--force]

Writes:

* ``experiments/v0_3/disclosure/DISCLOSURE.json``: the 200 disclosed windows, ids
  ``V001``..``V200`` in ascending key order. Each window has exactly the four README §2 items:
  ``prefix`` (rendered blocks 0..t, with the line ``### window start`` before block t-7),
  ``oracle_label``, ``network_class`` (archived v03-phase-a network, argmax at EOE_t) and
  ``reference_policy_class`` (archived v03-phase-a policy; null if the call failed). There is no
  session id, family, endpoint, logit or aggregate statistic;
* ``experiments/v0_3/disclosure/DISCLOSURE.sha256``: ``"<sha256>  DISCLOSURE.json"``. These two are the
  paths the frozen ``scripts/v03_launch.sh phase-b`` checks; ``scripts/v03_prepare_phase_b.sh`` copies
  them into the author directory;
* ``experiments/v0_3/DISCLOSURE_PRIVATE.json`` (repo only, never handed over): each id's
  session id, endpoint, family, oracle label and key, plus the input hashes.

The selection (``scoring.disclosure_selection``) uses the validation split only and never
depends on a model output. The script refuses to overwrite any output unless given ``--force``.

``--weight-dir DIR --out-dir DIR`` (both together, for tests and dry runs only): use the archive
in ``DIR`` instead of ``weight_program/v0_3/phase-a/`` and write all three files under
``--out-dir`` (``disclosure/`` inside it), which may not be ``experiments/v0_3/`` or inside it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "src") not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from agent_loop_lab.schema import CLASS_NAMES  # noqa: E402
from agent_loop_lab.v03 import renderer, scoring  # noqa: E402

SOURCE_PHASE = "v03-phase-a"  # README §2: predictions of the archived Phase A network and policy
OUT_ROOT = REPO / "experiments" / "v0_3"  # README §2 Records (r1): disclosure/ is the launcher's path
PUBLIC, SHA_FILE, PRIVATE = "DISCLOSURE.json", "DISCLOSURE.sha256", "DISCLOSURE_PRIVATE.json"
FORMAT = "v03-phase-b-disclosure/1"


def output_paths(root: Path) -> tuple[Path, Path, Path]:
    return root / "disclosure" / PUBLIC, root / "disclosure" / SHA_FILE, root / PRIVATE


def _blocks_for(session_ids: set[str]) -> dict[str, list[str]]:
    """Rendered blocks of the selected val sessions (re-rendered from the raw file)."""
    from agent_loop_lab.v03.build_raw import iter_raw

    out: dict[str, list[str]] = {}
    for sid, raw in iter_raw(scoring.V03_DATA, scoring.SPLIT):
        if sid in session_ids:
            out[sid] = renderer.render_session([r.to_dict() for r in raw], sid)
    if set(out) != session_ids:
        raise ValueError("selected sessions missing from the raw file")
    return out


def build(data: scoring.ValData, weight_dir: Path | None = None) -> tuple[dict, dict]:
    from agent_loop_lab.v03 import tokenizer

    chosen = scoring.disclosure_selection(data.rows)
    row_idx = [i for _, i in chosen]
    sess_idx = sorted({int(data.row_session[i]) for i in row_idx})
    if len(sess_idx) != len(chosen):
        raise ValueError("two disclosed windows share a session")
    d = scoring.phase_dir(SOURCE_PHASE, weight_dir)
    model, meta, pol, shas = scoring.load_phase(SOURCE_PHASE, d)
    net, _lat = scoring.network_predictions(model, data, sessions=sess_idx)
    ref = scoring.policy_predictions(pol, data, sessions=sess_idx, rows=row_idx, workers=1,
                                     chunk_sessions=len(sess_idx))
    blocks = _blocks_for({data.session_ids[s] for s in sess_idx})
    public_windows, private_windows = [], []
    for n, (key, i) in enumerate(chosen, start=1):
        r = data.rows[i]
        s, pos = int(data.row_session[i]), int(data.row_pos[i])
        b = blocks[r.session_id]
        if tokenizer.encode_session(b).tokens != data.sessions[s].tokens:
            raise ValueError(f"{r.session_id}: re-rendered blocks differ from the scored encoding")
        if any(line == scoring.WINDOW_MARKER for blk in b for line in blk.split("\n")):
            raise ValueError(f"{r.session_id}: a log line equals the window marker")
        vid = f"V{n:03d}"
        if net[i] < 0 or ref[i] == -1:
            raise RuntimeError(f"{vid}: missing prediction")
        public_windows.append({
            "id": vid,
            "prefix": scoring.render_disclosed_prefix(b, pos),
            "oracle_label": r.label.to_class().name,
            "network_class": CLASS_NAMES[int(net[i])],
            "reference_policy_class": None if ref[i] == -2 else CLASS_NAMES[int(ref[i])],
        })
        private_windows.append({"id": vid, "session_id": r.session_id, "endpoint": r.endpoint, "block": pos,
                                "family": r.family, "oracle_label": r.label.to_class().name, "key": key})
    public = {
        "format": FORMAT,
        "class_names": list(CLASS_NAMES),
        "window_marker": scoring.WINDOW_MARKER,
        "fields": {
            "prefix": ("the rendered log of blocks 0..t joined with newlines; the line '### window start' (not log "
                       "text: remove it before rendering or tokenising) stands immediately before block t-7, the "
                       "window's first block; block t, the window's last block, ends the text; after removing the "
                       "marker, every line that does not start with a space begins a new block"),
            "oracle_label": "the oracle label of the window ending at block t",
            "network_class": f"argmax at EOE_t of your archived Phase A network (artifact sha256 {shas['artifact_sha256']})",
            "reference_policy_class": ("class returned by your archived Phase A reference_policy on the surface tokens "
                                       "through EOE_t; null if the call failed (timeout, exception or invalid value)"),
        },
        "windows": public_windows,
    }
    private = {
        "format": FORMAT + "-private",
        "note": "repo only; never handed to the weight author (experiments/v0_3/phase_b/README.md §2)",
        "split": scoring.SPLIT,
        "key_format": scoring.DISCLOSURE_KEY_FORMAT,
        "per_class": scoring.PER_CLASS,
        "source_phase": SOURCE_PHASE,
        "archive_dir": str(d),
        **shas,
        "artifact_metadata": meta,
        "dataset": data.inputs["cache_key"],
        "windows": private_windows,
    }
    return public, private


def _dump(obj: Any) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--force", action="store_true", help="overwrite existing disclosure files")
    ap.add_argument("--weight-dir", type=Path, help="tests / dry runs only: Phase A archive directory to use")
    ap.add_argument("--out-dir", type=Path, help="with --weight-dir: where the (unofficial) files go")
    args = ap.parse_args(argv)
    if (args.weight_dir is None) != (args.out_dir is None):
        ap.error("--weight-dir and --out-dir go together")
    root = OUT_ROOT
    if args.out_dir is not None:
        root = args.out_dir.resolve()
        if root == OUT_ROOT.resolve() or OUT_ROOT.resolve() in root.parents:
            ap.error("--out-dir may not be experiments/v0_3/ or inside it (the official disclosure uses the archive)")
    paths = output_paths(root)
    if any(p.exists() for p in paths) and not args.force:
        print(f"refusing: disclosure files already exist under {root} (pass --force to overwrite)", file=sys.stderr)
        return 2
    torch.set_num_threads(scoring.TORCH_THREADS)
    scoring.phase_dir(SOURCE_PHASE, args.weight_dir)
    t0 = time.time()
    data = scoring.load_val()
    public, private = build(data, args.weight_dir)
    data_bytes = _dump(public)
    digest = hashlib.sha256(data_bytes).hexdigest()
    private["disclosure_sha256"] = digest
    pub, sha, prv = paths
    pub.parent.mkdir(parents=True, exist_ok=True)
    pub.write_bytes(data_bytes)
    sha.write_text(f"{digest}  {PUBLIC}\n")
    prv.write_bytes(_dump(private))
    counts = {c: sum(w["oracle_label"] == c for w in public["windows"]) for c in CLASS_NAMES}
    print(f"disclosure: {len(public['windows'])} windows {counts} sha256 {digest} ({time.time() - t0:.1f}s)")
    print(f"  {pub}\n  {prv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
