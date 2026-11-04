"""Build the Phase B disclosed diagnostic report (spec v0.2 §9.4; experiments/m3/README.md §2).

    uv run python scripts/m3_build_disclosure.py [--force]

Writes, under ``experiments/m3/disclosure/``:

* ``DISCLOSURE.json``: the 200 disclosed windows, ids ``D001``..``D200``. Per window exactly the
  four items of README §2: the 8 x 15 feature matrix (feature names given once at the top level),
  the oracle label, the archived A-selfcheck network's class and the archived A-selfcheck
  ``reference_policy`` class. No session id, family, endpoint, logits or aggregate statistics;
* ``DISCLOSURE.sha256``: ``"<sha256>  DISCLOSURE.json"``;
* ``DISCLOSURE_PRIVATE.json`` (repo only, never handed over): each id's session id, endpoint,
  family, oracle label and selection key, plus the inputs' hashes, so M4 can account for them.

Selection (README §2), on the **validation** split only, never depending on any model output:
population = gate-passing windows with a scored oracle label; key = ``sha256(f"20261104:phase-b-
disclosure:{session_id}:{endpoint}").hexdigest()``; per class, walk candidates by ascending key and
take a window unless the class already took one from the same session; stop at 50. The 200
windows are then sorted by key and numbered.

Refuses to overwrite existing outputs unless ``--force``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_validation as sv  # noqa: E402

from agent_loop_lab.eval import metrics  # noqa: E402
from agent_loop_lab.eval.collisions import WindowRow  # noqa: E402
from agent_loop_lab.schema import CLASS_NAMES, EVENTS_PER_WINDOW, FEATURE_NAMES, FEATURES_PER_EVENT, ClassLabel  # noqa: E402

SPLIT = "val"  # hard-coded (README §2: "200 windows from the validation split")
KEY_FORMAT = "20261104:phase-b-disclosure:{session_id}:{endpoint}"
PER_CLASS = 50
SOURCE_PHASE = "phase-a-selfcheck"  # README §2: predictions of the A-selfcheck network and reference_policy
OUT_DIR = REPO / "experiments" / "m3" / "disclosure"
PUBLIC, SHA_FILE, PRIVATE = "DISCLOSURE.json", "DISCLOSURE.sha256", "DISCLOSURE_PRIVATE.json"
FORMAT = "m3-phase-b-disclosure/1"


def window_key(session_id: str, endpoint: int) -> str:
    return hashlib.sha256(KEY_FORMAT.format(session_id=session_id, endpoint=endpoint).encode()).hexdigest()


def select(rows: Sequence[WindowRow], per_class: int = PER_CLASS) -> list[tuple[str, WindowRow]]:
    """README §2 selection: ``(key, row)`` of the chosen windows, sorted by key.

    Raises if a class cannot reach ``per_class`` windows. Uses only session ids, endpoints,
    gate results and oracle labels.
    """
    by_class: dict[ClassLabel, list[tuple[str, WindowRow]]] = {c: [] for c in ClassLabel}
    for r in rows:
        if r.split is not None:
            metrics.check_split(r.split)
        if r.gate_passed and r.label.is_scored:
            by_class[r.label.to_class()].append((window_key(r.session_id, r.endpoint), r))
    chosen: list[tuple[str, WindowRow]] = []
    for c in ClassLabel:
        cands = sorted(by_class[c], key=lambda kr: kr[0])
        if len({k for k, _ in cands}) != len(cands):
            raise ValueError(f"{c.name}: duplicate selection keys")
        taken: list[tuple[str, WindowRow]] = []
        sessions: set[str] = set()
        for k, r in cands:
            if r.session_id in sessions:
                continue
            taken.append((k, r))
            sessions.add(r.session_id)
            if len(taken) == per_class:
                break
        if len(taken) < per_class:
            raise ValueError(f"{c.name}: only {len(taken)} windows from distinct sessions, need {per_class}")
        chosen.extend(taken)
    chosen.sort(key=lambda kr: kr[0])
    if len({k for k, _ in chosen}) != len(chosen):
        raise ValueError("a window was selected under two classes")
    return chosen


def build(
    rows: Sequence[WindowRow], model: Any, policy: Any, shas: dict, inputs: dict, per_class: int = PER_CLASS
) -> tuple[dict, dict]:
    chosen = select(rows, per_class)
    sel_rows = [r for _, r in chosen]
    net, ref = sv.predictions(sel_rows, model, policy)
    public_windows, private_windows = [], []
    for n, ((key, r), pn, pr) in enumerate(zip(chosen, net.tolist(), ref.tolist()), start=1):
        did = f"D{n:03d}"
        x = np.frombuffer(r.vector, dtype=np.float32).reshape(EVENTS_PER_WINDOW, FEATURES_PER_EVENT)
        features = x.astype(np.float64).tolist()
        if not np.array_equal(np.asarray(features, dtype=np.float32), x):
            raise ValueError(f"{did}: features do not round-trip through float")
        public_windows.append(
            {
                "id": did,
                "features": features,
                "oracle_label": r.label.to_class().name,
                "network_class": CLASS_NAMES[pn],
                "reference_policy_class": CLASS_NAMES[pr],
            }
        )
        private_windows.append(
            {
                "id": did,
                "session_id": r.session_id,
                "endpoint": r.endpoint,
                "family": r.family,
                "oracle_label": r.label.to_class().name,
                "key": key,
            }
        )
    public = {
        "format": FORMAT,
        "feature_names": list(FEATURE_NAMES),
        "features_layout": "features[slot][feature]: 8 events (slot 0 oldest, slot 7 the window's last event) x 15 features in feature_names order",
        "class_names": list(CLASS_NAMES),
        "fields": {
            "oracle_label": "the oracle label of the window",
            "network_class": f"predicted class of the archived A-selfcheck network (artifact sha256 {shas['artifact_sha256']})",
            "reference_policy_class": "class returned by the archived A-selfcheck reference_policy",
        },
        "windows": public_windows,
    }
    private = {
        "format": FORMAT + "-private",
        "note": "repo only; never handed to the weight author (experiments/m3/README.md §2)",
        "split": SPLIT,
        "key_format": KEY_FORMAT,
        "per_class": per_class,
        "source_phase": SOURCE_PHASE,
        **shas,
        **inputs,
        "windows": private_windows,
    }
    return public, private


def _dump(obj: Any) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--force", action="store_true", help="overwrite existing disclosure files")
    args = ap.parse_args(argv)
    paths = [OUT_DIR / f for f in (PUBLIC, SHA_FILE, PRIVATE)]
    if any(p.exists() for p in paths) and not args.force:
        print(f"refusing: {OUT_DIR} already holds disclosure files (pass --force to overwrite)", file=sys.stderr)
        return 2
    model, _meta, policy, shas = sv.load_phase(sv.phase_dir(SOURCE_PHASE))
    rows = metrics.load_rows(sv.DATASET_DIR, SPLIT)
    public, private = build(rows, model, policy, shas, sv.dataset_inputs(sv.DATASET_DIR))
    data = _dump(public)
    digest = hashlib.sha256(data).hexdigest()
    private["disclosure_sha256"] = digest
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / PUBLIC).write_bytes(data)
    (OUT_DIR / SHA_FILE).write_text(f"{digest}  {PUBLIC}\n")
    (OUT_DIR / PRIVATE).write_bytes(_dump(private))
    counts = {c: sum(w["oracle_label"] == c for w in public["windows"]) for c in CLASS_NAMES}
    print(f"disclosure: {len(public['windows'])} windows {counts} sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
