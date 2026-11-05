"""v0.4 WSP-B disclosure (experiments/v0_4/PREREGISTRATION.md §7.1; plan v0.4 "Revisions after M0" 5). Harness only.

The same 200 windows as v0.3 Phase B: :func:`agent_loop_lab.v03.scoring.disclosure_selection` on the val rows
(50 per oracle class, at most one per session, ascending ``sha256("20261105:v03-phase-b-disclosure:{session_id}:
{endpoint}")``; the rule reads no model output). The selection must reproduce ``experiments/v0_3/
DISCLOSURE_PRIVATE.json`` window for window (id, session, endpoint, block, family, oracle label, key), and each
rendered prefix must equal the one v0.3 disclosed, byte for byte (the v0.4 export renders with the frozen v0.3
renderer). Only ``network_class`` changes: the WSP-A edit's verbaliser argmax at the window, read from the
WSP-A val column (``hypaqwen.run_readout`` with the archived edit applied, MPS, fp32). ``reference_policy_class``
is dropped. Nothing is compared with a label here: no agreement, accuracy or confusion is computed or printed.

Three files come out of :func:`build`:

* **public** (``disclosure/DISCLOSURE.json``, format :data:`FORMAT`, handed to the author by
  ``scripts/v04_prepare_phase_b.sh``): per window exactly ``id`` (``V001``..``V200`` in key order),
  ``prefix`` (blocks 0..t with the line ``### window start`` before block t-7), ``oracle_label`` and
  ``network_class``. No session id, family, endpoint, logit or aggregate;
* **windows** (``DISCLOSED_WINDOWS.json``, repo only): the model server's ``--disclosed`` file,
  ``{"windows": [{"id", "blocks", "pos"}]}`` with ``blocks`` = blocks 0..t and ``pos`` = t
  (``hypaqwen.wsp.server.read_disclosed``), so ``forward_disclosed`` runs exactly the disclosed text;
* **private** (``DISCLOSURE_PRIVATE.json``, repo only, never handed over): per id the session id, endpoint,
  block, family, oracle label and key, plus every input hash (the v0.3 disclosure, the export, the column,
  the WSP-A archive) and the sha256 of the public and windows files.

The WSP-A archive (``weight_program/v0_4/wsp-a/``, written by ``scripts/v04_run_build.py``) is checked by
:func:`check_wsp_a_archive`: ``BUILD_RECORD.json`` matches ``BUILD_RECORD.sha256``, the record is a phase
``wsp-a`` build checked against the server journal under the real guard, its edit files match the record,
and ``FREEZE.json`` names the registry it ran under. The column must carry that archive's edit-state hash in
its ``meta`` (:func:`check_column`). Every output is canonical JSON (sorted keys), so a rebuild from the same
inputs is byte-identical.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..eval.collisions import WindowRow
from ..schema import CLASS_NAMES
from ..v03 import scoring as v03_scoring

__all__ = [
    "FORMAT",
    "WINDOWS_FORMAT",
    "PUBLIC",
    "SHA_FILE",
    "PRIVATE",
    "WINDOWS",
    "DisclosureError",
    "output_paths",
    "check_wsp_a_archive",
    "check_column",
    "check_v03_disclosure",
    "read_tasks",
    "build",
    "dump",
    "write_outputs",
]

FORMAT = "v04-wsp-b-disclosure/1"
WINDOWS_FORMAT = "v04-wsp-b-disclosed-windows/1"
PUBLIC, SHA_FILE, PRIVATE, WINDOWS = "DISCLOSURE.json", "DISCLOSURE.sha256", "DISCLOSURE_PRIVATE.json", \
    "DISCLOSED_WINDOWS.json"
SOURCE_PHASE = "wsp-a"
REPO = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO / "experiments" / "v0_4"
V03_ROOT = REPO / "experiments" / "v0_3"
WSP_A_ARCHIVE = REPO / "weight_program" / "v0_4" / SOURCE_PHASE
# Files of a v0.4 build archive the disclosure depends on (hypaqwen.wsp.runner; scripts/v04_run_build.py).
RECORD_FILE, RECORD_SHA_FILE, FREEZE_FILE = "BUILD_RECORD.json", "BUILD_RECORD.sha256", "FREEZE.json"
EDIT_FILES = ("edit.safetensors", "edit_diff.safetensors")
RECORD_FORMAT_VERSION = 2  # hypaqwen.wsp.runner.RECORD_FORMAT_VERSION
# PREREGISTRATION §7.1: network_class is read out on MPS in fp32 with the frozen interface.
COLUMN_DEVICE, COLUMN_DTYPE = "mps", "fp32"
FIELDS = {
    "prefix": ("the rendered log of blocks 0..t joined with newlines; the line '### window start' (not log text: "
               "remove it before tokenising) stands immediately before block t-7, the window's first block; block "
               "t, the window's last block, ends the text; after removing the marker, every line that does not "
               "start with a space begins a new block"),
    "oracle_label": "the oracle label of the window ending at block t",
}


class DisclosureError(ValueError):
    pass


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    return p.relative_to(REPO).as_posix() if REPO in p.parents else str(p)


def dump(obj: Any) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def output_paths(root: Path) -> dict[str, Path]:
    """Where the four files go under ``root`` (``experiments/v0_4/`` for the official disclosure)."""
    return {"public": root / "disclosure" / PUBLIC, "sha": root / "disclosure" / SHA_FILE,
            "windows": root / WINDOWS, "private": root / PRIVATE}


def sha_field(path: Path) -> str:
    """The first field of a ``"<hex>  name"`` sha256 file, which must be 64 lowercase hex digits."""
    try:
        h = Path(path).read_text(encoding="utf-8").split()[0]
    except (OSError, IndexError):
        raise DisclosureError(f"{path}: missing or empty") from None
    if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
        raise DisclosureError(f"{path}: does not hold a sha256")
    return h


# -- inputs ----------------------------------------------------------------------------------------------------


def check_wsp_a_archive(archive: Path) -> dict[str, Any]:
    """The archived WSP-A build the disclosure (and WSP-B) starts from, checked; raises :class:`DisclosureError`."""
    archive = Path(archive)
    rec_path = archive / RECORD_FILE
    if not rec_path.is_file():
        raise DisclosureError(f"{rec_path} missing (WSP-A is not archived: scripts/v04_run_build.py wsp-a)")
    rec_sha = _sha256_file(rec_path)
    if sha_field(archive / RECORD_SHA_FILE) != rec_sha:
        raise DisclosureError(f"{rec_path} does not match {RECORD_SHA_FILE}")
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    checks = rec.get("checks", {})
    if rec.get("phase") != SOURCE_PHASE:
        raise DisclosureError(f"{rec_path}: phase {rec.get('phase')!r}, not {SOURCE_PHASE!r}")
    if rec.get("record_format_version") != RECORD_FORMAT_VERSION or checks.get("journal_checked") is not True \
            or checks.get("build_guard", {}).get("stub") is not False:
        raise DisclosureError(f"{rec_path}: not a build checked against the server journal under the real guard")
    for name in EDIT_FILES:
        if not (archive / name).is_file() or _sha256_file(archive / name) != rec.get("files", {}).get(name):
            raise DisclosureError(f"{archive / name} missing or different from {RECORD_FILE}")
    h = rec.get("edit_state_sha256")
    if not isinstance(h, str) or len(h) != 64:
        raise DisclosureError(f"{rec_path}: no edit_state_sha256")
    try:
        freeze = json.loads((archive / FREEZE_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise DisclosureError(f"{archive / FREEZE_FILE} missing or unreadable (PREREGISTRATION §7.3: every WSP "
                              "archive records the registry freeze it ran under)") from None
    return {"archive": _rel(archive), "build_record_sha256": rec_sha, "edit_state_sha256": h,
            "freeze_sha256": freeze.get("freeze_sha256"), "tensors_edited": sorted(rec.get("tensors", {}))}


def column_edit_hash(meta: Mapping[str, Any]) -> str | None:
    """The edit-state hash a readout column names: ``meta["edit"]["edit_state_sha256"]`` or ``meta["edit_state_sha256"]``."""
    edit = meta.get("edit")
    if isinstance(edit, Mapping) and edit.get("edit_state_sha256") is not None:
        return edit.get("edit_state_sha256")
    return meta.get("edit_state_sha256")


def check_column(meta: Mapping[str, Any], edit_state_sha256: str, *, interface_spec_sha256: str | None = None) -> None:
    """The WSP-A column is a full val readout of exactly the archived edit, on MPS in fp32 (PREREGISTRATION §7.1)."""
    problems = []
    if meta.get("split") != "val" or meta.get("limit") is not None:
        problems.append(f"not a full val readout (split {meta.get('split')!r}, limit {meta.get('limit')!r})")
    if meta.get("device") != COLUMN_DEVICE or meta.get("dtype") != COLUMN_DTYPE:
        problems.append(f"read out on {meta.get('device')!r}/{meta.get('dtype')!r}, not {COLUMN_DEVICE}/{COLUMN_DTYPE}")
    got = column_edit_hash(meta)
    if got != edit_state_sha256:
        problems.append(f"names edit state {got!r}, not the archived WSP-A edit {edit_state_sha256}")
    if interface_spec_sha256 is not None and (meta.get("interface") or {}).get("spec_sha256") != interface_spec_sha256:
        problems.append("not read out with the frozen interface")
    if problems:
        raise DisclosureError("WSP-A column: " + "; ".join(problems))


def check_v03_disclosure(public: Mapping[str, Any], private: Mapping[str, Any], public_sha256: str,
                         sha_file_value: str) -> None:
    """v0.3's disclosure files agree with each other (the reference the v0.4 selection must reproduce)."""
    if public_sha256 != sha_file_value or private.get("disclosure_sha256") != public_sha256:
        raise DisclosureError("experiments/v0_3/disclosure/DISCLOSURE.json does not match its sha256 records")
    if [w["id"] for w in public["windows"]] != [w["id"] for w in private["windows"]]:
        raise DisclosureError("v0.3 public and private disclosures list different ids")


def read_tasks(path: Path, session_ids: set[str]) -> tuple[dict[str, list[str]], dict[str, list[int]]]:
    """``(blocks, window_pos)`` of the named sessions from a v0.4 task export (no labels in that file)."""
    blocks: dict[str, list[str]] = {}
    pos: dict[str, list[int]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            if t["session_id"] in session_ids:
                blocks[t["session_id"]], pos[t["session_id"]] = list(t["blocks"]), list(t["window_pos"])
    missing = session_ids - set(blocks)
    if missing:
        raise DisclosureError(f"{len(missing)} disclosed sessions missing from {path.name}")
    return blocks, pos


# -- build -----------------------------------------------------------------------------------------------------


def build(rows: Sequence[WindowRow], row_pos: Sequence[int], pred: np.ndarray, blocks: Mapping[str, Sequence[str]],
          window_pos: Mapping[str, Sequence[int]], v03_public: Mapping[str, Any], v03_private: Mapping[str, Any],
          source: Mapping[str, Any], *, per_class: int = v03_scoring.PER_CLASS) -> tuple[dict, dict, dict]:
    """``(public, windows, private)``. ``pred`` is the WSP-A column's first-index label argmax per row (row order
    of ``rows``); ``source`` the input record (archive, column and export hashes) stored in the private file.

    Raises :class:`DisclosureError` unless the selection, the labels and every prefix equal v0.3's disclosure."""
    if not (len(rows) == len(row_pos) == len(pred)):
        raise DisclosureError("rows, row positions and predictions differ in length")
    chosen = v03_scoring.disclosure_selection(rows, per_class)
    if len({rows[i].session_id for _, i in chosen}) != len(chosen):
        raise DisclosureError("two disclosed windows share a session")
    ref_pub, ref_prv = list(v03_public["windows"]), list(v03_private["windows"])
    if len(ref_pub) != len(chosen) or len(ref_prv) != len(chosen):
        raise DisclosureError(f"{len(chosen)} windows selected, v0.3 disclosed {len(ref_prv)}")
    public_w, windows_w, private_w = [], [], []
    for n, (key, i) in enumerate(chosen, start=1):
        r, pos, vid = rows[i], int(row_pos[i]), f"V{n:03d}"
        label = r.label.to_class().name
        mine = {"id": vid, "session_id": r.session_id, "endpoint": int(r.endpoint), "block": pos, "family": r.family,
                "oracle_label": label, "key": key}
        ref = {k: ref_prv[n - 1].get(k) for k in mine}
        if mine != ref:
            raise DisclosureError(f"{vid}: the selection differs from v0.3's DISCLOSURE_PRIVATE.json: {mine} != {ref}")
        b = list(blocks[r.session_id])
        if pos not in window_pos[r.session_id]:
            raise DisclosureError(f"{vid}: block {pos} does not end a window of its session in the export")
        if any(line == v03_scoring.WINDOW_MARKER for blk in b for line in blk.split("\n")):
            raise DisclosureError(f"{vid}: a log line equals the window marker")
        prefix = v03_scoring.render_disclosed_prefix(b, pos)
        if ref_pub[n - 1].get("id") != vid or ref_pub[n - 1].get("prefix") != prefix \
                or ref_pub[n - 1].get("oracle_label") != label:
            raise DisclosureError(f"{vid}: prefix or label differs from the one v0.3 disclosed")
        k = int(pred[i])
        if not 0 <= k < len(CLASS_NAMES):
            raise DisclosureError(f"{vid}: no prediction")
        public_w.append({"id": vid, "prefix": prefix, "oracle_label": label, "network_class": CLASS_NAMES[k]})
        windows_w.append({"id": vid, "blocks": b[: pos + 1], "pos": pos})
        private_w.append({**mine, "network_class": CLASS_NAMES[k]})
    edit = source["wsp_a"]["edit_state_sha256"]
    public = {
        "format": FORMAT,
        "class_names": list(CLASS_NAMES),
        "window_marker": v03_scoring.WINDOW_MARKER,
        "fields": {**FIELDS, "network_class": (
            "the class of your archived WSP-A edit (edit hash " + edit + ") at the window: the argmax over the four "
            "label-word logits after block t, read out with the frozen interface (fp32)")},
        "windows": public_w,
    }
    windows = {"format": WINDOWS_FORMAT, "windows": windows_w}
    private = {
        "format": FORMAT + "-private",
        "note": "repo only; never handed to the weight author (PREREGISTRATION §7.1)",
        "split": v03_scoring.SPLIT,
        "key_format": v03_scoring.DISCLOSURE_KEY_FORMAT,
        "per_class": per_class,
        "source": dict(source),
        "windows": private_w,
    }
    return public, windows, private


def write_outputs(root: Path, public: Mapping[str, Any], windows: Mapping[str, Any], private: Mapping[str, Any]
                  ) -> dict[str, str]:
    """Write the four files under ``root``; returns their sha256 (the private file records the other three)."""
    paths = output_paths(root)
    pub_b, win_b = dump(public), dump(windows)
    shas = {"disclosure_sha256": _sha256_bytes(pub_b), "disclosed_windows_sha256": _sha256_bytes(win_b)}
    prv_b = dump({**private, **shas})
    paths["public"].parent.mkdir(parents=True, exist_ok=True)
    paths["public"].write_bytes(pub_b)
    paths["sha"].write_text(f"{shas['disclosure_sha256']}  {PUBLIC}\n", encoding="utf-8")
    paths["windows"].write_bytes(win_b)
    paths["private"].write_bytes(prv_b)
    return {**shas, "private_sha256": _sha256_bytes(prv_b)}
