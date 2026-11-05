"""Build the v0.4 WSP-B disclosure (experiments/v0_4/PREREGISTRATION.md §7.1; agent_loop_lab.v04.disclosure).

    uv run python scripts/v04_build_disclosure.py --column COLUMN.npz [--wsp-a-archive DIR] [--data-dir DIR]
                                                  [--out-dir DIR [--kit-unfrozen]] [--force]

Deviation D1 (PREREGISTRATION §13 log): COLUMN may instead be a readout of exactly the 200 disclosed sessions
(``--sessions ../../datasets/v0_4/DISCLOSED_SESSIONS.txt``, scripts/v04_disclosed_sessions.py); rows of other
sessions then carry no prediction, and the disclosed windows' classes are the same as from a full val readout.

Run it once, after WSP-A is archived (scripts/v04_run_build.py wsp-a) and its val column has been read out
with the archived edit applied (``uv run python -m hypaqwen.run_readout --split val --device mps --dtype fp32
--edit ../../weight_program/v0_4/wsp-a --out COLUMN.npz`` in envs/qwen; the column's meta must name the archive's
edit_state_sha256). It refuses if the column or the archive is missing or does not match. Inputs: the val
export (datasets/v0_4/val.labels.npz, checked against v0.3's val rows_sha256, and val.tasks.jsonl.gz for the
200 sessions' blocks), v0.3's disclosure (experiments/v0_3/disclosure/DISCLOSURE.json and DISCLOSURE.sha256,
experiments/v0_3/DISCLOSURE_PRIVATE.json), the WSP-A archive (default weight_program/v0_4/wsp-a/) and the
registry (experiments/registry_v0_4.yaml, required): the archive's FREEZE.json must be frozen and consistent,
with at least one launch, and its build and launch freezes must be the registry's G1 kernel freeze
(agent_loop_lab.v04.handover.check_archive_freeze). The column's meta must name the pinned model and the
archive's BUILD_RECORD.json sha256 (hypaqwen.run_readout --edit ARCHIVE). Writes:

* experiments/v0_4/disclosure/DISCLOSURE.json and DISCLOSURE.sha256 ("<sha256>  DISCLOSURE.json"): what
  scripts/v04_prepare_phase_b.sh copies into the author dir;
* experiments/v0_4/DISCLOSED_WINDOWS.json: the model server's --disclosed file (scripts/v04_launch.sh wsp-b);
* experiments/v0_4/DISCLOSURE_PRIVATE.json: repo only, never handed over.

The selection reads no model output and must reproduce v0.3's DISCLOSURE_PRIVATE.json; every prefix must equal
v0.3's byte for byte. Nothing is compared with a label: the script prints the label counts and hashes only.
Outputs are canonical JSON, so the same inputs give the same bytes. Existing outputs are kept unless --force.

``--out-dir DIR`` (tests and dry runs only): write the four files under DIR instead (same layout), which may
not be experiments/v0_4/ or inside it. ``--data-dir DIR`` reads another export directory. ``--kit-unfrozen``
(kit tests only, with --out-dir, a kit archive and no registry) accepts an archive built unfrozen. Every
refusal (archive, freeze, column, export) prints ``refusing:`` and exits 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(REPO / "src"))

from agent_loop_lab.schema import CLASS_NAMES  # noqa: E402
from agent_loop_lab.v04 import disclosure as D  # noqa: E402
from agent_loop_lab.v04 import export  # noqa: E402
from agent_loop_lab.v04 import handover as H  # noqa: E402
from agent_loop_lab.v04.scoring import ScoringError  # noqa: E402

REGISTRY = REPO / "experiments" / "registry_v0_4.yaml"


def check_archive(archive: Path, *, registry: Path = REGISTRY, kit_unfrozen: bool = False) -> tuple[dict, dict]:
    """``(wsp_a, freeze)``: the archive (``disclosure.check_wsp_a_archive``) and its freeze against the registry
    (``handover.check_archive_freeze``). Without the registry only ``--kit-unfrozen`` passes (an unfrozen kit
    archive, into ``--out-dir``)."""
    reg = registry if registry.is_file() else None
    unfrozen = kit_unfrozen and reg is None
    wsp_a = D.check_wsp_a_archive(archive, allow_unfrozen=unfrozen)
    return wsp_a, H.check_archive_freeze(wsp_a, registry=reg, allow_unfrozen=unfrozen)


def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    """Everything :func:`agent_loop_lab.v04.disclosure.build` needs, checked (module docstring)."""
    from agent_loop_lab.v04 import registry as R
    from agent_loop_lab.v04 import scoring as S

    wsp_a, freeze = check_archive(args.wsp_a_archive, kit_unfrozen=args.kit_unfrozen)
    rows, lab = S.load_rows(args.data_dir)
    if S.column_meta(args.column).get("subset") is None:
        pred, meta, _arr = S.load_column(args.column, lab)
    else:  # deviation D1: a readout of the disclosed sessions only; other rows get no prediction (-1)
        ids, info = S.read_subset(D.DISCLOSED_SESSIONS)
        if info["sha256"] != D.DISCLOSED_SESSIONS_SHA256:
            raise D.DisclosureError(f"{D.DISCLOSED_SESSIONS.name} sha256 {info['sha256']} != {D.DISCLOSED_SESSIONS_SHA256}")
        _sub_rows, sub_lab, mask = S.restrict(rows, lab, ids)
        sub_pred, meta, _arr = S.load_column(args.column, sub_lab, subset_sha256=info["sha256"])
        pred = np.full(len(rows), -1, dtype=np.int64)
        pred[mask] = sub_pred
    D.check_column(meta, wsp_a["edit_state_sha256"], interface_spec_sha256=R.INTERFACE_SPEC_SHA256,
                   model_pins=R.model_pins(REPO), build_record_sha256=wsp_a["build_record_sha256"])
    pub_path = D.V03_ROOT / "disclosure" / D.PUBLIC
    v03_public = json.loads(pub_path.read_text(encoding="utf-8"))
    v03_private = json.loads((D.V03_ROOT / "DISCLOSURE_PRIVATE.json").read_text(encoding="utf-8"))
    pub_sha = D._sha256_file(pub_path)
    D.check_v03_disclosure(v03_public, v03_private, pub_sha, D.sha_field(D.V03_ROOT / "disclosure" / D.SHA_FILE))
    tasks = export.tasks_path("val", args.data_dir)
    blocks, window_pos = D.read_tasks(tasks, {w["session_id"] for w in v03_private["windows"]})
    if meta.get("subset") is not None and set(ids) != {w["session_id"] for w in v03_private["windows"]}:
        raise D.DisclosureError(f"{D.DISCLOSED_SESSIONS.name} is not the v0.3 disclosed sessions")
    source = {
        "wsp_a": wsp_a,
        "column": {"path": D._rel(args.column), "sha256": D._sha256_file(args.column), "meta": meta},
        "export": {tasks.name: D._sha256_file(tasks),
                   export.labels_path("val", args.data_dir).name: D._sha256_file(export.labels_path("val", args.data_dir)),
                   "val_rows_sha256": S.VAL_ROWS_SHA256},
        "v03_disclosure_sha256": pub_sha,
        "v03_disclosure_private_sha256": D._sha256_file(D.V03_ROOT / "DISCLOSURE_PRIVATE.json"),
        "registry_freeze_sha256": freeze["freeze_sha256"],
    }
    return {"rows": rows, "row_pos": lab["row_pos"], "pred": pred, "blocks": blocks, "window_pos": window_pos,
            "v03_public": v03_public, "v03_private": v03_private, "source": source}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--column", type=Path, required=True, help="the WSP-A val column (run_readout .npz, edit applied)")
    ap.add_argument("--wsp-a-archive", type=Path, default=D.WSP_A_ARCHIVE)
    ap.add_argument("--data-dir", type=Path, default=export.V04_DATA)
    ap.add_argument("--out-dir", type=Path, help="tests / dry runs only: where the (unofficial) files go")
    ap.add_argument("--force", action="store_true", help="overwrite existing disclosure files")
    ap.add_argument("--kit-unfrozen", action="store_true",
                    help="kit tests only, with --out-dir and no registry: accept an unfrozen WSP-A archive")
    args = ap.parse_args(argv)
    root = D.OUT_ROOT
    if args.out_dir is not None:
        root = args.out_dir.resolve()
        if root == D.OUT_ROOT.resolve() or D.OUT_ROOT.resolve() in root.parents:
            ap.error("--out-dir may not be experiments/v0_4/ or inside it (that is the official disclosure)")
    if args.kit_unfrozen and (args.out_dir is None or REGISTRY.is_file()
                              or args.wsp_a_archive.resolve() == D.WSP_A_ARCHIVE.resolve()):
        ap.error("--kit-unfrozen needs --out-dir and a kit archive, and is refused once the registry exists")
    if not args.column.is_file():
        print(f"refusing: the WSP-A val column {args.column} does not exist (read it out first)", file=sys.stderr)
        return 2
    paths = D.output_paths(root)
    if any(p.exists() for p in paths.values()) and not args.force:
        print(f"refusing: disclosure files already exist under {root} (pass --force to overwrite)", file=sys.stderr)
        return 2
    try:
        inp = load_inputs(args)
        public, windows, private = D.build(inp["rows"], inp["row_pos"], inp["pred"], inp["blocks"], inp["window_pos"],
                                           inp["v03_public"], inp["v03_private"], inp["source"])
    except (D.DisclosureError, H.HandoverError, ScoringError) as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 2
    shas = D.write_outputs(root, public, windows, private)
    counts = {c: sum(w["oracle_label"] == c for w in public["windows"]) for c in CLASS_NAMES}
    print(f"disclosure: {len(public['windows'])} windows {counts} sha256 {shas['disclosure_sha256']}")
    print(f"disclosed windows (server --disclosed): sha256 {shas['disclosed_windows_sha256']}")
    print(f"WSP-A edit {inp['source']['wsp_a']['edit_state_sha256']} "
          f"(BUILD_RECORD.json {inp['source']['wsp_a']['build_record_sha256']})")
    for p in paths.values():
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
