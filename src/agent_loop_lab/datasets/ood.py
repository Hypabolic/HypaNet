"""OOD holdout dataset (spec v0.2 §7.3), built from the independently written ``synth_ood`` generator.

Layout (data gitignored; manifest committed):
    datasets/v0_2/ood/ood.events.jsonl.gz
    datasets/v0_2/ood/ood.oracle.jsonl.gz
    datasets/v0_2/OOD_MANIFEST.json

Session seed = ``int.from_bytes(sha256(f"{master_seed}:ood:{family}:{k}").digest()[:8], "big")`` and
session id ``o-{seed:016x}``. The ``o-`` prefix and the ``:ood:`` seed namespace keep OOD sessions disjoint
from the standard set (``s-`` ids). Generation goes through ``synth_ood.FAMILIES[name](random.Random(seed), ...)``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Callable, Iterator

from ..oracle_config import OracleConfig, load_oracle_config
from ..schema import CanonicalEvent, OracleLabel, OracleRecord
from .build import BuildError, _check_events, _GzLines, _write_json, events_line, file_sha256, oracle_line

OOD_DIR = "ood"
OOD_SESSIONS_PER_FAMILY = 200
OOD_MANIFEST = "OOD_MANIFEST.json"
_SRC_ROOT = Path(__file__).resolve().parents[1]


def ood_seed(master_seed: int, family: str, k: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{master_seed}:ood:{family}:{k}".encode()).digest()[:8], "big")


def ood_session_id(seed: int) -> str:
    return f"o-{seed:016x}"


def ood_paths(out_dir: str | Path) -> tuple[Path, Path]:
    d = Path(out_dir) / OOD_DIR
    return d / "ood.events.jsonl.gz", d / "ood.oracle.jsonl.gz"


def _ood_source_sha256() -> str:
    h = hashlib.sha256()
    root = _SRC_ROOT / "synth_ood"
    for p in sorted(root.rglob("*.py"), key=lambda q: q.relative_to(root).as_posix()):
        if "__pycache__" in p.parts:
            continue
        h.update(p.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def build_ood(
    out_dir: str | Path,
    *,
    master_seed: int,
    sessions_per_family: int = OOD_SESSIONS_PER_FAMILY,
    cfg: OracleConfig | None = None,
    on_session: Callable[[list[CanonicalEvent], OracleRecord], None] | None = None,
) -> dict[str, Any]:
    """Generate the OOD set into ``out_dir`` and write ``OOD_MANIFEST.json``; return the manifest.

    ``on_session(events, record)`` (optional, observe-only) is called right
    after each session's lines are written, in file order (v0.3 raw builder).
    """
    from ..synth_ood import FAMILIES

    cfg = cfg or load_oracle_config()
    ev_path, or_path = ood_paths(out_dir)
    ev_path.parent.mkdir(parents=True, exist_ok=True)
    fe, fo = _GzLines(ev_path), _GzLines(or_path)
    label_counts: dict[str, dict[str, int]] = {}
    session_ids: list[str] = []
    try:
        for family, gen in FAMILIES.items():
            counts: dict[str, int] = {}
            for k in range(sessions_per_family):
                seed = ood_seed(master_seed, family, k)
                sid = ood_session_id(seed)
                events, record = gen(random.Random(seed), sid, cfg)
                _check_events(events, sid)
                if record.session_id != sid:
                    raise BuildError(f"OOD {family}: record session_id {record.session_id!r} != {sid!r}")
                record = dataclasses.replace(record, seed=seed)
                fe.write(events_line(sid, events))
                fo.write(oracle_line(record))
                session_ids.append(sid)
                if on_session is not None:
                    on_session(events, record)
                for lab in record.window_labels.values():
                    counts[lab.value] = counts.get(lab.value, 0) + 1
            label_counts[family] = dict(sorted(counts.items()))
    finally:
        fe.close()
        fo.close()
    if len(set(session_ids)) != len(session_ids):
        raise BuildError("duplicate OOD session ids")
    totals: dict[str, int] = {}
    for c in label_counts.values():
        for k, v in c.items():
            totals[k] = totals.get(k, 0) + v
    manifest = {
        "dataset": "v0_2-ood",
        "spec_version": "0.2",
        "generator": "agent_loop_lab.synth_ood",
        "generator_version": "ood-0.1",
        "generator_author": "Codex (isolated commission; see experiments/commissions/ood_generator/)",
        "generator_source_sha256": _ood_source_sha256(),
        "master_seed": master_seed,
        "seed_formula": 'int.from_bytes(sha256(f"{master_seed}:ood:{family}:{k}").digest()[:8], "big")',
        "session_id_format": "o-{seed:016x}",
        "families": list(FAMILIES),
        "sessions_per_family": sessions_per_family,
        "sessions": len(session_ids),
        "session_ids_sha256": hashlib.sha256("\n".join(session_ids).encode()).hexdigest(),
        "window_label_counts": dict(sorted(totals.items())),
        "window_label_counts_per_family": label_counts,
        "files": {
            f"{OOD_DIR}/{ev_path.name}": file_sha256(ev_path),
            f"{OOD_DIR}/{or_path.name}": file_sha256(or_path),
        },
    }
    _write_json(Path(out_dir) / OOD_MANIFEST, manifest)
    return manifest


def iter_ood_sessions(out_dir: str | Path) -> Iterator[tuple[list[CanonicalEvent], OracleRecord]]:
    import gzip

    ev_path, or_path = ood_paths(out_dir)
    with gzip.open(ev_path, "rt", encoding="utf-8") as fe, gzip.open(or_path, "rt", encoding="utf-8") as fo:
        for le, lo in zip(fe, fo, strict=True):
            d = json.loads(le)
            if set(d) != {"session_id", "events"}:
                raise BuildError(f"OOD events line keys {sorted(d)}")
            yield [CanonicalEvent.from_dict(e) for e in d["events"]], OracleRecord.from_dict(json.loads(lo))


def scored(label: OracleLabel) -> bool:
    return label.is_scored
