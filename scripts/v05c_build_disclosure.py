"""Build the v0.5 Phase C disclosures: 200 new OOD windows and v0.3's 200 val windows, both with B's classes.

    uv run python scripts/v05c_build_disclosure.py
    uv run python scripts/v05c_build_disclosure.py --out DIR [--artifact PATH] [--sessions-per-family N]
                                                   [--part {both,ood,original}]                 (dry run)

The network whose classes both disclosures give is v0.5 Phase B's (``weight_program/v0_5/phase-b/artifact.dlw``,
sha256 ``223d0743...``, checked against ``artifact.sha256``, ``apply.json``, ``BUILD_RECORD.json`` and its metadata
phase; ``agent_loop_lab.v05.phase_c_data.load_b``). Writes, under ``DIR`` (default ``experiments/v0_5/phase_c/``):

* ``disclosure/DISCLOSURE_W.json`` (handed over): 200 windows ``W001``..``W200`` from the OOD generator
  (``synth_ood``), master seed 20261109, 200 sessions in each of the 6 seen families of the pre-registered family
  rule (``phase_c_data.family_split``), each session re-timed by its per-session clock offset
  (``phase_c_data.retime_session``; not mentioned in the handed-over file), selected by v0.3's rule
  (``v03.scoring.disclosure_selection`` by import: gate passed, scored oracle label, 50 per class, at most one
  window per session, ordered by a fixed hash key; no model output). Each window has exactly ``prefix`` (``v03.scoring.render_disclosed_prefix``, with the
  ``### window start`` marker), ``oracle_label`` and ``network_class`` (B's argmax at EOE_t). No family name,
  session id, endpoint or seed appears anywhere in the file (checked);
* ``disclosure/DISCLOSURE_W.sha256``;
* ``DISCLOSURE_OOD_PRIVATE.json`` (repo only): per id the session id, family, k, seed, endpoint, block, label and
  key (and the session's re-timing offset); the seeds, the re-timing rule, the family rule and split, the
  candidates' hashes and the source hashes;
* ``disclosure/DISCLOSURE.json`` + ``disclosure/DISCLOSURE.sha256`` (handed over): v0.3's 200 val windows
  ``V001``..``V200`` with ``network_class`` = B's class. Built by ``scripts/v05b_build_disclosure.py``'s ``build``
  (by import: the v0.3 selection is reproduced and every ``prefix`` / ``oracle_label`` is checked against v0.3's
  disclosure); then every window's ``prefix`` and ``oracle_label`` are checked equal to v0.5 Phase B's
  disclosure (``experiments/v0_5/phase_b/disclosure/DISCLOSURE.json``, checked against its ``.sha256``). Only the
  header's network description and format differ from Phase B's file, and the classes are B's, not r1's;
* ``DISCLOSURE_PRIVATE.json`` (repo only): the same windows' session ids, endpoints, families and keys.

The default run is reproducible byte for byte (no timestamps, repo-relative paths). Refuses to overwrite any
output file. Dry runs (``--out DIR``, not inside ``experiments/v0_5/phase_c/``) may use ``--artifact`` (any
``.dlw``), fewer sessions per family, and one part only.
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
from agent_loop_lab.v05 import phase_c_data as P  # noqa: E402
from agent_loop_lab.v05 import scoring as v05_scoring  # noqa: E402

OUT_ROOT = REPO / "experiments" / "v0_5" / "phase_c"
OOD_PUBLIC, OOD_SHA, OOD_PRIVATE = "DISCLOSURE_W.json", "DISCLOSURE_W.sha256", "DISCLOSURE_OOD_PRIVATE.json"
PUBLIC, SHA_FILE, PRIVATE = "DISCLOSURE.json", "DISCLOSURE.sha256", "DISCLOSURE_PRIVATE.json"
FORMAT = "v05-phase-c-disclosure/1"
PHASE_B_PUBLIC = REPO / "experiments" / "v0_5" / "phase_b" / "disclosure" / PUBLIC
PHASE_B_SHA = REPO / "experiments" / "v0_5" / "phase_b" / "disclosure" / SHA_FILE
FORBIDDEN_OUT = (REPO / "experiments" / "v0_3", REPO / "experiments" / "v0_5" / "phase_b",
                 REPO / "experiments" / "v0_5" / "runs", REPO / "weight_program", REPO / "datasets")
PARTS = ("both", "ood", "original")


class DisclosureError(ValueError):
    pass


def _dump(obj: Any) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True) + "\n").encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def output_paths(root: Path, part: str) -> list[Path]:
    ood = [root / "disclosure" / OOD_PUBLIC, root / "disclosure" / OOD_SHA, root / OOD_PRIVATE]
    orig = [root / "disclosure" / PUBLIC, root / "disclosure" / SHA_FILE, root / PRIVATE]
    return {"both": ood + orig, "ood": ood, "original": orig}[part]


@functools.lru_cache(maxsize=1)
def v05b_builder() -> Any:
    """``scripts/v05b_build_disclosure.py`` as a module (imported, never edited)."""
    path = REPO / "scripts" / "v05b_build_disclosure.py"
    spec = importlib.util.spec_from_file_location("_v05b_build_disclosure", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# The original disclosure, re-labelled with B's classes
# ---------------------------------------------------------------------------


def network_text(source: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    sha = source["artifact_sha256"]
    is_b = source.get("phase") == P.B_PHASE and sha == P.B_SHA256 and not source.get("dry_run")
    name = "the v0.5 Phase B network (the network this revision starts from)" if is_b else \
        "the dry-run artefact's network"
    return name, {"phase": P.B_PHASE if is_b else None, "artifact_sha256": sha,
                  "description": f"{name}: the network whose classes this disclosure gives"}


def load_phase_b_disclosure(public: Path = PHASE_B_PUBLIC, sha_file: Path = PHASE_B_SHA) -> list[dict[str, Any]]:
    """Phase B's handed-over windows (``id``, ``prefix``, ``oracle_label`` only), checked against its sha256 file."""
    raw = Path(public).read_bytes()
    recorded = v05_scoring.read_sha256_file(Path(sha_file), PUBLIC)
    if _sha(raw) != recorded:
        raise DisclosureError(f"{public}: sha256 {_sha(raw)} != {sha_file} {recorded}")
    return [{k: w[k] for k in ("id", "prefix", "oracle_label")} for w in json.loads(raw)["windows"]]


def relabel(public: dict[str, Any], private: dict[str, Any], source: Mapping[str, Any],
            phase_b_windows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Phase C header on v05b ``build``'s output, after checking ``prefix`` / ``oracle_label`` against Phase B's."""
    wins = public["windows"]
    if [w["id"] for w in wins] != [w["id"] for w in phase_b_windows]:
        raise DisclosureError("window ids differ from v0.5 Phase B's disclosure")
    for w, b in zip(wins, phase_b_windows):
        for f in ("prefix", "oracle_label"):
            if w[f] != b[f]:
                raise DisclosureError(f"{w['id']}: {f} differs from v0.5 Phase B's disclosure")
        if set(w) != {"id", "prefix", "oracle_label", "network_class"} or w["network_class"] not in CLASS_NAMES:
            raise DisclosureError(f"{w['id']}: unexpected fields")
    name, network = network_text(source)
    out = dict(public)
    out["format"] = FORMAT
    out["network"] = network
    out["fields"] = {**public["fields"],
                     "network_class": (f"argmax at EOE_t of {name} (artifact sha256 {source['artifact_sha256']}), "
                                       "from one causal forward pass; ties go to the first class")}
    prv = dict(private)
    prv["format"] = FORMAT + "-private"
    prv["same_windows_as"] = (private["same_windows_as"] + "; and experiments/v0_5/phase_b/disclosure/DISCLOSURE.json "
                              "(v0.5 Phase B): ids, prefix and oracle_label checked equal; network_class is B's")
    prv["source"] = dict(source)
    return out, prv


def build_original(model: Any, source: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    b = v05b_builder()
    ref_public, ref_private, ref_hashes = b.load_v03_reference(b.V03_PUBLIC, b.V03_SHA, b.V03_PRIVATE)
    data = v03_scoring.load_val()
    public, private = b.build(data, model, dict(source), ref_public, ref_private)
    public, private = relabel(public, private, source, load_phase_b_disclosure())
    private.update(ref_hashes)
    private["phase_b_disclosure_sha256"] = v05_scoring.read_sha256_file(PHASE_B_SHA, PUBLIC)
    return public, private


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _check_out(root: Path, dry_run: bool) -> str | None:
    for bad in FORBIDDEN_OUT:
        b = bad.resolve()
        if root == b or b in root.parents:
            return f"--out may not be {bad} or inside it"
    official = OUT_ROOT.resolve()
    if dry_run and (root == official or official in root.parents):
        return "--out of a dry run may not be experiments/v0_5/phase_c/ or inside it"
    return None


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, help="dry run: output root (disclosure/ inside it)")
    ap.add_argument("--artifact", type=Path, help="dry run only: any .dlw instead of v0.5 B's artefact")
    ap.add_argument("--sessions-per-family", type=int, help="dry run only: OOD candidate sessions per seen family")
    ap.add_argument("--part", choices=PARTS, default="both", help="dry run only: build one disclosure")
    args = ap.parse_args(argv)
    dry = args.out is not None
    if not dry and (args.artifact is not None or args.sessions_per_family is not None or args.part != "both"):
        ap.error("--artifact, --sessions-per-family and --part are dry-run options (they need --out)")
    if args.sessions_per_family is not None and args.sessions_per_family < 1:
        ap.error("--sessions-per-family must be >= 1")
    root = (args.out if dry else OUT_ROOT).resolve()
    problem = _check_out(root, dry)
    if problem:
        ap.error(problem)
    paths = output_paths(root, args.part)
    existing = [str(p) for p in paths if p.exists()]
    if existing:
        print(f"refusing: {', '.join(existing)} already exist(s)", file=sys.stderr)
        return 2
    per_family = args.sessions_per_family or P.DISCLOSURE_SESSIONS_PER_FAMILY
    t0 = time.time()
    try:
        model, source = P.load_b() if args.artifact is None else P.load_any(args.artifact)
    except (P.DataError, v05_scoring.ScoringError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    torch.set_num_threads(v03_scoring.TORCH_THREADS)
    outputs: list[tuple[Path, bytes]] = []
    try:
        if args.part in ("both", "ood"):
            P.checked_family_split()
            enc, meta = P.disclosure_candidates(P.DISCLOSURE_SEED, per_family)
            pub, prv = P.build_ood_disclosure(enc, meta, model, source, master_seed=P.DISCLOSURE_SEED,
                                              per_family=per_family)
            data = _dump(pub)
            prv["disclosure_sha256"] = _sha(data)
            prv["official"] = not dry
            outputs += [(root / "disclosure" / OOD_PUBLIC, data),
                        (root / "disclosure" / OOD_SHA, f"{_sha(data)}  {OOD_PUBLIC}\n".encode()),
                        (root / OOD_PRIVATE, _dump(prv))]
            counts = {c: sum(w["network_class"] == c for w in pub["windows"]) for c in CLASS_NAMES}
            print(f"OOD disclosure: {len(pub['windows'])} windows from {len(enc.data.session_ids)} candidate sessions, "
                  f"network classes {counts}, sha256 {_sha(data)} ({time.time() - t0:.0f}s)")
        if args.part in ("both", "original"):
            pub, prv = build_original(model, source)
            data = _dump(pub)
            prv["disclosure_sha256"] = _sha(data)
            outputs += [(root / "disclosure" / PUBLIC, data),
                        (root / "disclosure" / SHA_FILE, f"{_sha(data)}  {PUBLIC}\n".encode()),
                        (root / PRIVATE, _dump(prv))]
            counts = {c: sum(w["network_class"] == c for w in pub["windows"]) for c in CLASS_NAMES}
            print(f"original disclosure: {len(pub['windows'])} windows (v0.3's = Phase B's), network classes {counts}, "
                  f"sha256 {_sha(data)} ({time.time() - t0:.0f}s)")
    except (DisclosureError, P.DataError, v05b_builder().DisclosureError, ValueError) as exc:
        print(f"ERROR: {exc} (nothing written)", file=sys.stderr)
        return 1
    for p, data in outputs:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
