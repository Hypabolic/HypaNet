"""v0.5 Phase C data: the OOD disclosure and the held-out evaluation sets. Harness only.

Phase C gives a new author session v0.5 Phase B's network and two labelled disclosures: v0.3's 200 val
windows again (re-labelled with B's classes; no new information to the lineage) and 200 new windows from the
independently written OOD generator (``agent_loop_lab.synth_ood``), restricted to 6 of its 10 families. It is
evaluated on fresh held-out sets generated and hashed before launch and never handed over (the sealed v0.2 /
v0.3 sets are not used). The pre-registration is ``experiments/v0_5/phase_c/heldout/README.md``.

This module is never imported by an author kit (it imports the generators and the oracle).

**Seeds.** Each role has one fixed master seed (:data:`ROLE_SEEDS`): the OOD disclosure 20261109, the held-out
OOD set 20261110, the held-out in-distribution set 20261111. :func:`check_seed` refuses the reserved seeds
(:data:`FORBIDDEN_SEEDS`: 20261104, the sealed sets; 20261107 and 20261108, the post-hoc OOD diagnosis) and a
seed that belongs to another role.

**Family rule** (:func:`family_split`), fixed before any Phase C data existed: sort the OOD family names by
``sha256(f"20261109:v05c-family:{name}").hexdigest()`` and take the first 6 ("seen": the disclosure draws from
them only); the other 4 are "unseen". The result is pinned in :data:`SEEN_FAMILIES` / :data:`UNSEEN_FAMILIES`.

**Generation.** OOD sessions as ``datasets.ood.build_ood`` makes them (seed ``ood_seed(master_seed, family, k)``,
id ``o-{seed:016x}``, ``synth_ood.FAMILIES[family](random.Random(seed), sid, cfg)``, record seed replaced);
in-distribution sessions as ``datasets.build.build`` makes them (``family_counts`` over ``FAMILY_NAMES``,
``session_seed(master_seed, family, k)``, id ``s-{seed:016x}``, ``synth.families.generate_session``). Raw ops are
captured with ``v03.rawop.capture_raw`` and checked against the events (``v03.build_raw`` alignment).

**Re-timing** (OOD sessions only; decided 7 Nov 2026): ``synth_ood``'s engine starts every session's clock at
2026-03-14 00:00:00 UTC, so every rendered block header of its sessions would read ``00:0x:..`` (the renderer
prints the time of day), while the standard generator starts at a random day and second. :func:`generate_ood`
therefore shifts every timestamp of an OOD session (its canonical events' and its raw ops' alike, before
anything is rendered or written) by one per-session offset of whole seconds, :func:`retime_offset_s`:
``int.from_bytes(sha256(f"v05c-retime:{session_id}").digest()[:8], "big") % 86400``. Event order, gaps, every
other field and the oracle record are unchanged; alignment is checked before and after the shift. ``synth_ood``
itself is not modified. In-distribution sessions are not re-timed. Rendering
(``v03.renderer``), tokenising (``v03.tokenizer``), windows (``windows.window_endpoints``) and the runtime gate
(``windows.gate``) are v0.3's, step by step as ``v03.sealed._encode_split``.

**Selection** of the OOD disclosure: ``v03.scoring.disclosure_selection`` by import (v0.3's rule and key: gate
passed, scored oracle label, 50 per class in class order, ascending key, at most one window per session; it
reads only session ids, endpoints, gate results and oracle labels, never a model output).

**Held-out files** (gitignored by the ``datasets/**/*.jsonl.gz`` rule) under ``datasets/v0_5c/<set>/``:
``<set>.events.jsonl.gz``, ``<set>.oracle.jsonl.gz``, ``<set>.raw.jsonl.gz`` (v0.2 / v0.3 line formats,
deterministic gzip). The manifest (committed) records every file's sha256, the window rows' sha256
(``v03.sealed.rows_sha256``) and the token encoding's sha256 (:func:`encoding_sha256_update`).
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from ..datasets import build as dsb
from ..datasets.ood import _ood_source_sha256, ood_seed, ood_session_id
from ..eval.collisions import WindowRow
from ..oracle_config import OracleConfig, load_oracle_config
from ..schema import CLASS_NAMES, CanonicalEvent, OracleLabel, OracleRecord
from ..v03 import renderer, tokenizer
from ..v03 import scoring as v03_scoring
from ..v03 import sealed as v03_sealed
from ..v03.build_raw import RawBuildError, _check_aligned, raw_line
from ..v03.rawop import RawOp, capture_raw

__all__ = [
    "FORBIDDEN_SEEDS", "ROLE_SEEDS", "DISCLOSURE_SEED", "HELDOUT_OOD_SEED", "HELDOUT_INDIST_SEED",
    "SEEN_FAMILIES", "UNSEEN_FAMILIES", "SeedError", "DataError", "check_seed", "family_rule_key", "family_split",
    "Session", "RETIME_FORMAT", "RETIME_RULE", "retime_offset_s", "retime_session", "generate_ood", "generate_indist", "encode_sessions", "reserved_session_ids", "check_disjoint",
    "set_paths", "write_set", "iter_set", "verify_set_files", "load_set", "build_heldout", "load_b",
    "disclosure_candidates", "build_ood_disclosure", "leak_problems",
]

REPO = v03_scoring.REPO
PHASE_C_ROOT = REPO / "experiments" / "v0_5" / "phase_c"
DATA_ROOT = REPO / "datasets" / "v0_5c"
MANIFEST_PATH = PHASE_C_ROOT / "heldout" / "HELDOUT_MANIFEST.json"
DISCLOSURE_OOD_PRIVATE = PHASE_C_ROOT / "DISCLOSURE_OOD_PRIVATE.json"
B_DIR = REPO / "weight_program" / "v0_5" / "phase-b"
B_SHA256 = "223d074372bf726a0c89f28afe1e1fcf9b53e3bb0ff3973f0cc3acd932661001"
B_PHASE = "v05-phase-b"

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

FORBIDDEN_SEEDS: dict[int, str] = {
    20261104: "the master seed of the sealed v0.2/v0.3 test and OOD sets (and of dev/val)",
    20261107: "used by the post-hoc OOD diagnosis (experiments/v0_5/ood_analysis/)",
    20261108: "used by the post-hoc OOD diagnosis, in-distribution arm (experiments/v0_5/ood_analysis/)",
}
DISCLOSURE_SEED = 20261109
HELDOUT_OOD_SEED = 20261110
HELDOUT_INDIST_SEED = 20261111
ROLE_SEEDS: dict[str, int] = {"disclosure": DISCLOSURE_SEED, "heldout_ood": HELDOUT_OOD_SEED,
                              "heldout_indist": HELDOUT_INDIST_SEED}

DISCLOSURE_SESSIONS_PER_FAMILY = 200
HELDOUT_OOD_SESSIONS_PER_FAMILY = 200
HELDOUT_INDIST_SESSIONS = 4000
RESERVED_K = 2000  # indices per family checked by reserved_session_ids


class SeedError(ValueError):
    """A reserved master seed, or another role's seed."""


class DataError(ValueError):
    """Generated or stored Phase C data failed a check."""


def check_seed(seed: int, role: str) -> int:
    """``seed`` if ``role`` may use it; raises :class:`SeedError` for a reserved or another role's seed."""
    if role not in ROLE_SEEDS:
        raise ValueError(f"unknown role {role!r} (expected one of {sorted(ROLE_SEEDS)})")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise SeedError(f"master seed must be an int, got {seed!r}")
    if seed in FORBIDDEN_SEEDS:
        raise SeedError(f"refusing master seed {seed}: {FORBIDDEN_SEEDS[seed]}; it must never be used in Phase C")
    for other, s in ROLE_SEEDS.items():
        if other != role and seed == s:
            raise SeedError(f"refusing master seed {seed} for {role}: it is the {other} seed")
    return seed


# ---------------------------------------------------------------------------
# Family rule
# ---------------------------------------------------------------------------

FAMILY_RULE_FORMAT = "20261109:v05c-family:{name}"
FAMILY_RULE_TEXT = ('sort the synth_ood family names by sha256(f"20261109:v05c-family:{name}").hexdigest() and take '
                    'the first 6 (seen: the OOD disclosure draws only from them); the other 4 are unseen')
N_SEEN = 6
SEEN_FAMILIES = ("fixture_archaeology", "annotation_spiral", "merge_ping_pong", "shard_migration", "telemetry_gaps",
                 "pinned_validation")
UNSEEN_FAMILIES = ("lease_reconnect", "index_watermark", "lockfile_churn", "serializer_duel")


def family_rule_key(name: str) -> str:
    return hashlib.sha256(FAMILY_RULE_FORMAT.format(name=name).encode()).hexdigest()


def ood_row_family(name: str) -> str:
    """The ``scenario_family`` an OOD record (and so its window rows) carries for family ``name``."""
    return f"ood:{name}"


def ood_families() -> tuple[str, ...]:
    from ..synth_ood import FAMILIES

    return tuple(FAMILIES)


def family_split(families: Sequence[str] | None = None) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(seen, unseen)``: the first :data:`N_SEEN` names in :func:`family_rule_key` order, and the rest."""
    names = list(ood_families() if families is None else families)
    if len(set(names)) != len(names) or len(names) <= N_SEEN:
        raise ValueError(f"need more than {N_SEEN} distinct family names, got {names}")
    order = sorted(names, key=family_rule_key)
    return tuple(order[:N_SEEN]), tuple(order[N_SEEN:])


def checked_family_split() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The rule on the installed generator; must equal the pinned split."""
    seen, unseen = family_split()
    if (seen, unseen) != (SEEN_FAMILIES, UNSEEN_FAMILIES):
        raise DataError(f"family rule gives seen={seen} unseen={unseen}, not the pinned split")
    return seen, unseen


# ---------------------------------------------------------------------------
# Generation (with raw capture)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Session:
    session_id: str
    family: str
    k: int | None
    seed: int | None
    events: list[CanonicalEvent]
    record: OracleRecord
    raw: list[RawOp]


def _captured(sid: str, make: Callable[[], tuple[list[CanonicalEvent], OracleRecord]]
              ) -> tuple[list[CanonicalEvent], OracleRecord, list[RawOp]]:
    with capture_raw() as cap:
        events, record = make()
        raw = cap.pop(sid, events)
    dsb._check_events(events, sid)
    if record.session_id != sid:
        raise DataError(f"record session_id {record.session_id!r} != {sid!r}")
    _check_aligned(sid, events, raw)
    return events, record, raw


RETIME_FORMAT = "v05c-retime:{session_id}"
RETIME_RULE = {
    "applies_to": "every synth_ood session Phase C generates (the OOD disclosure's candidates and the held-out "
                  "OOD set); not the in-distribution set, val or the original disclosure",
    "offset_formula": 'int.from_bytes(sha256(f"v05c-retime:{session_id}").digest()[:8], "big") % 86400',
    "unit": "whole seconds, in [0, 86399]",
    "text": ("synth_ood starts every session's clock at 2026-03-14T00:00:00+00:00 (the standard generator starts at "
             "a random day and second of the day), so every timestamp of an OOD session, in its canonical events and "
             "its raw ops alike, is shifted by the session's offset before it is rendered or written; event order, "
             "gaps, every other field and the oracle record are unchanged (phase_c_data.retime_session)"),
}


def retime_offset_s(session_id: str) -> int:
    """The session's clock offset in whole seconds, ``0 <= offset < 86400`` (:data:`RETIME_RULE`)."""
    digest = hashlib.sha256(RETIME_FORMAT.format(session_id=session_id).encode()).digest()
    return int.from_bytes(digest[:8], "big") % 86400


def _shift_ts(ts: str | None, seconds: int, what: str) -> str | None:
    if ts is None:
        return None
    t = datetime.fromisoformat(ts)
    if t.isoformat() != ts:  # only the instant may change, never the string's form
        raise DataError(f"{what}: timestamp {ts!r} does not round-trip through datetime.isoformat")
    return (t + timedelta(seconds=seconds)).isoformat()


def retime_session(sid: str, events: Sequence[CanonicalEvent], raw: Sequence[RawOp]
                   ) -> tuple[list[CanonicalEvent], list[RawOp], int]:
    """``(events, raw, offset)`` with every timestamp shifted by :func:`retime_offset_s` ``(sid)`` seconds;
    nothing else changes. Checks the alignment of the result."""
    off = retime_offset_s(sid)
    ev = [dataclasses.replace(e, timestamp=_shift_ts(e.timestamp, off, f"{sid} event {n}"))
          for n, e in enumerate(events)]
    rw = [dataclasses.replace(r, timestamp=_shift_ts(r.timestamp, off, f"{sid} raw op {n}"))
          for n, r in enumerate(raw)]
    _check_aligned(sid, ev, rw)
    return ev, rw, off


def _generate_ood_unshifted(master_seed: int, families: Sequence[str], per_family: int, *, role: str,
                            cfg: OracleConfig | None = None) -> Iterator[Session]:
    """:func:`generate_ood` without the re-timing (tests compare against it; no Phase C data uses it)."""
    from ..synth_ood import FAMILIES

    check_seed(master_seed, role)
    cfg = cfg or load_oracle_config()
    unknown = [f for f in families if f not in FAMILIES]
    if unknown:
        raise ValueError(f"unknown OOD families {unknown}")
    for family in families:
        gen = FAMILIES[family]
        for k in range(per_family):
            seed = ood_seed(master_seed, family, k)
            sid = ood_session_id(seed)
            try:
                events, record, raw = _captured(sid, lambda: gen(random.Random(seed), sid, cfg))
            except (DataError, RawBuildError, dsb.BuildError):
                raise
            except Exception as exc:
                raise DataError(f"OOD {family} k={k} seed={seed}: {type(exc).__name__}: {exc}") from exc
            record = dataclasses.replace(record, seed=seed)
            if record.scenario_family not in (family, ood_row_family(family)):
                raise DataError(f"OOD {family} k={k}: record family {record.scenario_family!r}")
            yield Session(sid, record.scenario_family, k, seed, events, record, raw)


def generate_ood(master_seed: int, families: Sequence[str], per_family: int, *, role: str,
                 cfg: OracleConfig | None = None) -> Iterator[Session]:
    """OOD sessions, family by family (in the given order), ``k = 0 .. per_family - 1`` (``build_ood``'s seeding),
    re-timed (:func:`retime_session`)."""
    for s in _generate_ood_unshifted(master_seed, families, per_family, role=role, cfg=cfg):
        events, raw, _ = retime_session(s.session_id, s.events, s.raw)
        dsb._check_events(events, s.session_id)
        yield dataclasses.replace(s, events=events, raw=raw)


def generate_indist(master_seed: int, n_sessions: int, *, role: str = "heldout_indist",
                    cfg: OracleConfig | None = None) -> Iterator[Session]:
    """Standard-generator sessions (``datasets.build.build``'s seeding and family counts), family order."""
    from ..synth.families import FAMILY_NAMES, generate_session

    check_seed(master_seed, role)
    cfg = cfg or load_oracle_config()
    counts = dsb.family_counts(n_sessions, FAMILY_NAMES)
    for family in FAMILY_NAMES:
        for k in range(counts[family]):
            seed = dsb.session_seed(master_seed, family, k)
            sid = dsb.session_id_for(seed)
            try:
                events, record, raw = _captured(sid, lambda: generate_session(family, seed, sid, cfg))
            except (DataError, RawBuildError, dsb.BuildError):
                raise
            except Exception as exc:
                raise DataError(f"standard {family} k={k} seed={seed}: {type(exc).__name__}: {exc}") from exc
            if record.scenario_family != family:
                raise DataError(f"standard {family} k={k}: record family {record.scenario_family!r}")
            yield Session(sid, family, k, seed, events, record, raw)


# ---------------------------------------------------------------------------
# Encoding (v03.sealed._encode_split, step by step)
# ---------------------------------------------------------------------------

ENCODING_DEFINITION = ("sha256 over the sessions in file order of: session id, NUL, compact JSON "
                       "[tokens, event_ord, offset, eoe_index], NUL")
ROWS_DEFINITION = ("v03.sealed.rows_sha256: sha256 of compact JSON [[session_id, family, endpoint, oracle label, "
                   "gate_passed], ...] in file order")


def encoding_sha256_update(h: Any, sid: str, enc: Any) -> None:
    h.update(sid.encode() + b"\0")
    h.update(json.dumps([list(enc.tokens), list(enc.event_ord), list(enc.offset), list(enc.eoe_index)],
                        separators=(",", ":")).encode() + b"\0")


def _encode_one(sid: str, family: str, events: Sequence[CanonicalEvent], record: OracleRecord,
                raw: Sequence[RawOp]) -> tuple[list[str], Any, list[tuple[int, WindowRow]]]:
    from ..windows import gate, validate_sequence, window_endpoints

    if len(raw) != len(events):
        raise DataError(f"{sid}: {len(raw)} raw ops for {len(events)} events")
    blocks = renderer.render_session([r.to_dict() for r in raw], sid)
    enc = tokenizer.encode_session(blocks)
    if len(enc.eoe_index) != len(events):
        raise DataError(f"{sid}: EOE count differs from the event count")
    validate_sequence(events)
    rows: list[tuple[int, WindowRow]] = []
    for pos in window_endpoints(events):
        endpoint = events[pos].sequence_index
        try:
            label = OracleLabel(record.window_labels[endpoint])
        except KeyError:
            raise DataError(f"{sid}: no oracle label for window endpoint {endpoint}") from None
        rows.append((pos, WindowRow(session_id=sid, family=family, split=None, endpoint=int(endpoint), label=label,
                                    gate_passed=bool(gate(events[: pos + 1]).passed), vector=None)))
    return blocks, enc, rows


@dataclasses.dataclass
class Encoded:
    data: v03_scoring.ValData
    blocks: dict[str, list[str]]
    rows_sha256: str
    encoding_sha256: str


def encode_sessions(sessions: Iterable[Session], *, keep_blocks: bool = False) -> Encoded:
    """Rendered and tokenised sessions with their window rows (row ``split`` is None), in input order."""
    ids: list[str] = []
    fams: list[str] = []
    encs: list[v03_scoring.EncodedSession] = []
    rows: list[WindowRow] = []
    r_sess: list[int] = []
    r_pos: list[int] = []
    blocks_of: dict[str, list[str]] = {}
    seen_ids: set[str] = set()
    h = hashlib.sha256()
    for s in sessions:
        if s.session_id in seen_ids:
            raise DataError(f"duplicate session {s.session_id}")
        seen_ids.add(s.session_id)
        family = s.record.scenario_family
        if family != s.family:
            raise DataError(f"{s.session_id}: record family {family!r} != generated family {s.family!r}")
        blocks, enc, srows = _encode_one(s.session_id, family, s.events, s.record, s.raw)
        n = len(ids)
        ids.append(s.session_id)
        fams.append(family)
        encs.append(v03_scoring.EncodedSession(list(enc.tokens), list(enc.event_ord), list(enc.offset),
                                               list(enc.eoe_index), list(enc.surface)))
        encoding_sha256_update(h, s.session_id, enc)
        for pos, row in srows:
            rows.append(row)
            r_sess.append(n)
            r_pos.append(pos)
        if keep_blocks:
            blocks_of[s.session_id] = blocks
    data = v03_scoring.ValData(ids, fams, encs, rows, np.asarray(r_sess, dtype=np.int64),
                               np.asarray(r_pos, dtype=np.int64))
    return Encoded(data, blocks_of, v03_sealed.rows_sha256(rows), h.hexdigest())


# ---------------------------------------------------------------------------
# Disjointness
# ---------------------------------------------------------------------------


def reserved_session_ids(k_max: int = RESERVED_K, *, disclosure_private: Path | None = None) -> dict[str, frozenset[str]]:
    """Session ids that held-out sets must not contain, by source (computed from the seed formulas; no data file
    is opened). For each reserved master seed (:data:`FORBIDDEN_SEEDS`) and the disclosure seed: the OOD formula
    (``ood_seed``, ``o-`` ids) over the OOD and the standard family names, and the standard formula
    (``session_seed``, ``s-`` ids) over the standard family names, ``k < k_max``. With ``disclosure_private``
    (if the file exists): the session ids of the disclosed windows."""
    from ..synth.families import FAMILY_NAMES

    out: dict[str, frozenset[str]] = {}
    fams_ood = (*ood_families(), *FAMILY_NAMES)
    for seed in (*FORBIDDEN_SEEDS, DISCLOSURE_SEED):
        ids = {ood_session_id(ood_seed(seed, f, k)) for f in fams_ood for k in range(k_max)}
        ids |= {dsb.session_id_for(dsb.session_seed(seed, f, k)) for f in FAMILY_NAMES for k in range(k_max)}
        out[f"master seed {seed} (ood and standard formulas, k < {k_max})"] = frozenset(ids)
    if disclosure_private is not None and Path(disclosure_private).is_file():
        rec = json.loads(Path(disclosure_private).read_text())
        out[f"disclosed windows ({_rel(Path(disclosure_private))})"] = frozenset(w["session_id"] for w in rec["windows"])
    return out


def check_disjoint(ids: Iterable[str], what: str, reserved: Mapping[str, frozenset[str]]) -> dict[str, Any]:
    ids = frozenset(ids)
    for src, rs in reserved.items():
        both = ids & rs
        if both:
            raise DataError(f"{what}: {len(both)} session(s) also in {src} (e.g. {sorted(both)[0]})")
    return {"sessions": len(ids), "disjoint_from": sorted(reserved), "overlap": 0}


# ---------------------------------------------------------------------------
# Held-out files
# ---------------------------------------------------------------------------

SET_NAMES = ("heldout_ood", "heldout_indist")
FILE_KINDS = ("events", "oracle", "raw")


def set_paths(data_root: Path, name: str) -> dict[str, Path]:
    if name not in SET_NAMES:
        raise ValueError(f"unknown held-out set {name!r}")
    d = Path(data_root) / name
    return {kind: d / f"{name}.{kind}.jsonl.gz" for kind in FILE_KINDS}


def _rel_file(data_root: Path, p: Path) -> str:
    return p.relative_to(Path(data_root)).as_posix()


def write_set(data_root: Path, name: str, sessions: Iterable[Session]) -> dict[str, Any]:
    """Write one held-out set (refuses to overwrite) and return its manifest entry."""
    paths = set_paths(data_root, name)
    existing = [str(p) for p in paths.values() if p.exists()]
    if existing:
        raise DataError(f"refusing to overwrite {existing}")
    paths["events"].parent.mkdir(parents=True, exist_ok=True)
    writers = {k: dsb._GzLines(p) for k, p in paths.items()}
    ids: list[str] = []
    fam_counts: dict[str, int] = {}
    rows: list[WindowRow] = []
    h = hashlib.sha256()
    try:
        for s in sessions:
            writers["events"].write(dsb.events_line(s.session_id, s.events))
            writers["oracle"].write(dsb.oracle_line(s.record))
            writers["raw"].write(raw_line(s.session_id, s.raw))
            _, enc, srows = _encode_one(s.session_id, s.record.scenario_family, s.events, s.record, s.raw)
            if s.record.scenario_family != s.family:
                raise DataError(f"{s.session_id}: record family differs from the generated family")
            encoding_sha256_update(h, s.session_id, enc)
            rows.extend(r for _, r in srows)
            ids.append(s.session_id)
            fam_counts[s.family] = fam_counts.get(s.family, 0) + 1
    except BaseException:
        for w in writers.values():
            w.close()
            w.path.unlink(missing_ok=True)
        raise
    for w in writers.values():
        w.close()
    if len(set(ids)) != len(ids):
        for p in paths.values():
            p.unlink(missing_ok=True)
        raise DataError(f"{name}: duplicate session ids")
    return {
        "files": {_rel_file(data_root, w.path): {"sha256": dsb.file_sha256(w.path), "bytes": w.path.stat().st_size,
                                                 "lines": w.lines} for w in writers.values()},
        "sessions": len(ids),
        "sessions_per_family": fam_counts,
        "session_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "windows": len(rows),
        "gate_passed_windows": sum(r.gate_passed for r in rows),
        "rows_sha256": v03_sealed.rows_sha256(rows),
        "encoding_sha256": h.hexdigest(),
    }


def iter_set(data_root: Path, name: str) -> Iterator[Session]:
    """Read a held-out set back, in file order (strict keys, aligned raw ops)."""
    paths = set_paths(data_root, name)
    with gzip.open(paths["events"], "rt", encoding="utf-8") as fe, \
            gzip.open(paths["oracle"], "rt", encoding="utf-8") as fo, \
            gzip.open(paths["raw"], "rt", encoding="utf-8") as fr:
        for n, (le, lo, lr) in enumerate(zip(fe, fo, fr, strict=True)):
            d, r = json.loads(le), json.loads(lr)
            if set(d) != {"session_id", "events"} or set(r) != {"session_id", "raw"}:
                raise DataError(f"{name} line {n + 1}: unexpected keys")
            events = [CanonicalEvent.from_dict(e) for e in d["events"]]
            record = OracleRecord.from_dict(json.loads(lo))
            raw = [RawOp.from_dict(x) for x in r["raw"]]
            sid = d["session_id"]
            if record.session_id != sid or r["session_id"] != sid:
                raise DataError(f"{name} line {n + 1}: events/oracle/raw session ids differ")
            _check_aligned(sid, events, raw)
            yield Session(sid, record.scenario_family, None, record.seed, events, record, raw)


def verify_set_files(data_root: Path, name: str, entry: Mapping[str, Any]) -> dict[str, str]:
    """Hash the set's files against its manifest entry (no decoding); raises :class:`DataError`."""
    out: dict[str, str] = {}
    paths = set_paths(data_root, name)
    want = {rel: meta["sha256"] for rel, meta in entry["files"].items()}
    have = {_rel_file(data_root, p): p for p in paths.values()}
    if set(want) != set(have):
        raise DataError(f"{name}: manifest files {sorted(want)} != expected {sorted(have)}")
    for rel, p in have.items():
        if not p.is_file():
            raise DataError(f"{name}: {p} missing")
        h = dsb.file_sha256(p)
        if h != want[rel]:
            raise DataError(f"{name}: {rel} sha256 differs from the manifest")
        out[rel] = h
    return out


def check_attempt(attempt: v03_sealed.Attempt | None) -> None:
    """Held-out sets are decoded only under a written attempt record holding ``attempt.token``."""
    if attempt is None:
        raise v03_sealed.SealedAccessError("held-out sets are decoded only after the attempt record is written")
    try:
        doc = json.loads(Path(attempt.marker).read_text())
    except (OSError, ValueError) as exc:
        raise v03_sealed.SealedAccessError(f"attempt record unreadable: {exc}") from None
    if doc.get("attempt_token") != attempt.token:
        raise v03_sealed.SealedAccessError("attempt record token mismatch")


def load_set(data_root: Path, name: str, entry: Mapping[str, Any], *, attempt: v03_sealed.Attempt | None
             ) -> v03_sealed.SplitData:
    """A held-out set as ``v03.sealed.SplitData`` (needs the attempt record). Checks file hashes, then that the
    window rows and the token encoding hash to the manifest's values."""
    check_attempt(attempt)
    files = verify_set_files(data_root, name, entry)
    enc = encode_sessions(iter_set(data_root, name))
    d = enc.data
    checks = (("sessions", len(d.session_ids), entry["sessions"]), ("windows", len(d.rows), entry["windows"]),
              ("rows_sha256", enc.rows_sha256, entry["rows_sha256"]),
              ("encoding_sha256", enc.encoding_sha256, entry["encoding_sha256"]),
              ("session_ids_sha256", hashlib.sha256("\n".join(d.session_ids).encode()).hexdigest(),
               entry["session_ids_sha256"]))
    for what, got, want in checks:
        if got != want:
            raise DataError(f"{name}: {what} differs from the manifest")
    d.inputs = {"source": "v05.phase_c_data.load_set", "set": name, "files_sha256": files,
                "rows_sha256": enc.rows_sha256, "encoding_sha256": enc.encoding_sha256}
    return v03_sealed.SplitData(name, d, {}, dict(d.inputs))


# ---------------------------------------------------------------------------
# Held-out build + manifest
# ---------------------------------------------------------------------------

MANIFEST_FORMAT = "v05-phase-c-heldout-manifest/1"


def _sources_sha256() -> dict[str, str]:
    from ..v03 import rawop

    return {
        "synth (standard generator)": dsb._generator_source_sha256(),
        "synth_ood (OOD generator)": _ood_source_sha256(),
        "src/agent_loop_lab/v03/renderer.py": v03_sealed.sha256_file(Path(renderer.__file__)),
        "src/agent_loop_lab/v03/tokenizer.py": v03_sealed.sha256_file(Path(tokenizer.__file__)),
        "src/agent_loop_lab/v03/rawop.py": v03_sealed.sha256_file(Path(rawop.__file__)),
        "src/agent_loop_lab/windows.py": v03_sealed.sha256_file(REPO / "src" / "agent_loop_lab" / "windows.py"),
        "src/agent_loop_lab/v05/phase_c_data.py": v03_sealed.sha256_file(Path(__file__)),
    }


# The code that scores the held-out sets (README §5-§7). Its sha256 is pinned in HELDOUT_MANIFEST.json, which is
# committed before launch (scripts/v05c_launch.sh checks it); the official run refuses if any file differs from the
# manifest (manifest_problems(official=True)) or from HEAD (phase_c_eval.refuse_reasons).
SCORING_CODE_FILES = (
    "scripts/v05c_heldout_eval.py",
    "src/agent_loop_lab/v05/phase_c_eval.py",
    "src/agent_loop_lab/v05/phase_c_data.py",
    "src/agent_loop_lab/v05/sealed.py",
    "src/agent_loop_lab/v05/scoring.py",
    "src/agent_loop_lab/v04/scoring.py",
    "src/agent_loop_lab/v03/scoring.py",
    "src/agent_loop_lab/v03/sealed.py",
    "src/agent_loop_lab/v03/infer.py",
    "src/agent_loop_lab/v03/model.py",
    "src/agent_loop_lab/v03/renderer.py",
    "src/agent_loop_lab/v03/tokenizer.py",
    "src/agent_loop_lab/eval/metrics.py",
    "src/agent_loop_lab/eval/m4.py",
)


def scoring_code_sha256() -> dict[str, str | None]:
    """``SCORING_CODE_FILES`` -> sha256 now (``None`` for a missing file)."""
    return {f: (v03_sealed.sha256_file(REPO / f) if (REPO / f).is_file() else None) for f in SCORING_CODE_FILES}


def build_heldout(data_root: Path, *, ood_seed_: int = HELDOUT_OOD_SEED, indist_seed: int = HELDOUT_INDIST_SEED,
                  ood_per_family: int = HELDOUT_OOD_SESSIONS_PER_FAMILY, indist_sessions: int = HELDOUT_INDIST_SESSIONS,
                  official: bool = False, reserved_k: int = RESERVED_K, disclosure_private: Path | None = None,
                  cfg: OracleConfig | None = None, progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Generate both held-out sets into ``data_root`` and return the manifest (the caller writes it).

    Every generated session id is checked against :func:`reserved_session_ids` (and the OOD set against the
    in-distribution set) before the manifest is returned; a failure deletes the written files."""
    say = progress or (lambda m: None)
    check_seed(ood_seed_, "heldout_ood")
    check_seed(indist_seed, "heldout_indist")
    seen, unseen = checked_family_split()
    families = ood_families()
    cfg = cfg or load_oracle_config()
    reserved = reserved_session_ids(reserved_k, disclosure_private=disclosure_private)
    sets: dict[str, Any] = {}
    disjoint: dict[str, Any] = {}
    written: list[Path] = []
    try:
        for name, gen in (("heldout_ood", lambda: generate_ood(ood_seed_, families, ood_per_family,
                                                               role="heldout_ood", cfg=cfg)),
                          ("heldout_indist", lambda: generate_indist(indist_seed, indist_sessions, cfg=cfg))):
            written += list(set_paths(data_root, name).values())
            ids: list[str] = []

            def tap(it: Iterator[Session]) -> Iterator[Session]:
                for s in it:
                    ids.append(s.session_id)
                    yield s

            sets[name] = write_set(data_root, name, tap(gen()))
            disjoint[name] = check_disjoint(ids, name, reserved)
            reserved = {**reserved, f"held-out set {name}": frozenset(ids)}
            say(f"{name}: {sets[name]['sessions']} sessions, {sets[name]['windows']} windows")
    except BaseException:
        for p in written:
            p.unlink(missing_ok=True)
        raise
    from ..synth.families import FAMILY_NAMES

    sets["heldout_ood"] |= {
        "generator": "agent_loop_lab.synth_ood", "master_seed": ood_seed_,
        "seed_formula": 'int.from_bytes(sha256(f"{master_seed}:ood:{family}:{k}").digest()[:8], "big")',
        "session_id_format": "o-{seed:016x}", "families": list(families), "sessions_per_family_requested": ood_per_family,
        "populations": {"all": list(families), "seen": list(seen), "unseen": list(unseen)},
        "retime": dict(RETIME_RULE),
    }
    sets["heldout_indist"] |= {
        "generator": "agent_loop_lab.synth (standard)", "master_seed": indist_seed,
        "seed_formula": dsb.SEED_FORMULA, "session_id_format": "s-{seed:016x}", "families": list(FAMILY_NAMES),
        "family_count_rule": dsb.FAMILY_COUNT_RULE, "sessions_requested": indist_sessions,
    }
    return {
        "format": MANIFEST_FORMAT,
        "official": official,
        "note": ("Phase C held-out evaluation sets: generated and hashed before the Phase C author session is "
                 "launched, never handed over; data files are gitignored, this manifest is committed"),
        "pre_registration": "experiments/v0_5/phase_c/heldout/README.md",
        "data_root": _rel(Path(data_root)),
        "forbidden_seeds": {str(k): v for k, v in FORBIDDEN_SEEDS.items()},
        "role_seeds": dict(ROLE_SEEDS),
        "family_rule": {"text": FAMILY_RULE_TEXT, "format": FAMILY_RULE_FORMAT, "n_seen": N_SEEN,
                        "keys": {f: family_rule_key(f) for f in families}, "seen": list(seen), "unseen": list(unseen)},
        "oracle_config": cfg.to_dict(),
        "source_sha256": _sources_sha256(),
        "scoring_code_sha256": scoring_code_sha256(),
        "rows_definition": ROWS_DEFINITION,
        "encoding_definition": ENCODING_DEFINITION,
        "gzip": {"mtime": 0, "compresslevel": dsb.GZIP_LEVEL, "header_filename": ""},
        "disjointness": disjoint,
        "sets": sets,
    }


def manifest_problems(man: Mapping[str, Any], *, official: bool = True) -> list[str]:
    """Why a manifest cannot be used for the held-out evaluation (empty: it can). Checks the seeds, the family
    split, the generator / renderer / tokeniser source hashes and the requested sizes (no data file is read); with
    ``official``, also that the scoring code (``SCORING_CODE_FILES``) is byte for byte what the manifest pinned."""
    out: list[str] = []
    if man.get("format") != MANIFEST_FORMAT:
        out.append(f"manifest format {man.get('format')!r} != {MANIFEST_FORMAT!r}")
        return out
    if official and man.get("official") is not True:
        out.append("manifest is a dry-run manifest (official: false)")
    sets = man.get("sets", {})
    if set(sets) != set(SET_NAMES):
        out.append(f"manifest sets {sorted(sets)} != {sorted(SET_NAMES)}")
        return out
    for name, role in (("heldout_ood", "heldout_ood"), ("heldout_indist", "heldout_indist")):
        seed = sets[name].get("master_seed")
        try:
            check_seed(seed, role)
        except (SeedError, ValueError) as exc:
            out.append(f"{name}: {exc}")
        if official and seed != ROLE_SEEDS[role]:
            out.append(f"{name}: master seed {seed} != pre-registered {ROLE_SEEDS[role]}")
    if official:
        if sets["heldout_ood"].get("sessions_per_family_requested") != HELDOUT_OOD_SESSIONS_PER_FAMILY:
            out.append("heldout_ood: sessions per family differ from the pre-registration")
        if sets["heldout_indist"].get("sessions_requested") != HELDOUT_INDIST_SESSIONS:
            out.append("heldout_indist: session count differs from the pre-registration")
    fr = man.get("family_rule", {})
    if (tuple(fr.get("seen", ())), tuple(fr.get("unseen", ()))) != (SEEN_FAMILIES, UNSEEN_FAMILIES):
        out.append("family split differs from the pinned rule result")
    if sets["heldout_ood"].get("retime") != RETIME_RULE:
        out.append("heldout_ood: the re-timing rule is missing or differs from phase_c_data.RETIME_RULE")
    if "retime" in sets["heldout_indist"]:
        out.append("heldout_indist: in-distribution sessions are not re-timed")
    pops = sets["heldout_ood"].get("populations", {})
    if tuple(pops.get("seen", ())) != SEEN_FAMILIES or tuple(pops.get("unseen", ())) != UNSEEN_FAMILIES:
        out.append("heldout_ood populations differ from the pinned family split")
    now = _sources_sha256()
    for key in ("synth (standard generator)", "synth_ood (OOD generator)", "src/agent_loop_lab/v03/renderer.py",
                "src/agent_loop_lab/v03/tokenizer.py", "src/agent_loop_lab/v03/rawop.py", "src/agent_loop_lab/windows.py"):
        if man.get("source_sha256", {}).get(key) != now[key]:
            out.append(f"source sha256 of {key} differs from the manifest")
    if official:
        pinned = man.get("scoring_code_sha256")
        if not isinstance(pinned, dict) or set(pinned) != set(SCORING_CODE_FILES):
            out.append("manifest does not pin the scoring code (scoring_code_sha256 missing or a different file list)")
        else:
            code_now = scoring_code_sha256()
            changed = [f for f in SCORING_CODE_FILES if pinned[f] is None or pinned[f] != code_now[f]]
            if changed:
                out.append(f"scoring code differs from what the manifest pinned before launch: {changed}")
    return out


# ---------------------------------------------------------------------------
# The base network (v0.5 Phase B)
# ---------------------------------------------------------------------------


def load_b(weight_dir: Path = B_DIR) -> tuple[Any, dict[str, Any]]:
    """v0.5 Phase B's network, checked: ``artifact.sha256``, the pinned sha256, metadata phase, ``apply.json``
    and ``BUILD_RECORD.json``. Returns ``(model in eval mode, source record)``."""
    from . import scoring as v05_scoring

    weight_dir = Path(weight_dir)
    dlw = weight_dir / "artifact.dlw"
    spec, info = v05_scoring.load_run_artifact(dlw, weight_dir / "artifact.sha256")
    sha = info["artifact_sha256"]
    if sha != B_SHA256:
        raise DataError(f"{dlw}: sha256 {sha} is not v0.5 Phase B's {B_SHA256}")
    if info["artifact_metadata"].get("phase") != B_PHASE:
        raise DataError(f"{dlw}: artefact metadata phase is not {B_PHASE}")
    v05_scoring.check_apply_report(weight_dir, sha)
    build = json.loads((weight_dir / "BUILD_RECORD.json").read_text())
    if build.get("artifact_sha256") != sha:
        raise DataError(f"{weight_dir}/BUILD_RECORD.json artifact_sha256 differs from the artefact")
    model = v03_sealed.load_net(spec)
    return model, {"phase": B_PHASE, "artifact_path": _rel(dlw), "artifact_sha256": sha,
                   "base_artifact_sha256": build.get("base_artifact_sha256"), "dry_run": False}


def load_any(path: Path) -> tuple[Any, dict[str, Any]]:
    """Dry runs: any ``.dlw`` (checked against an ``artifact.sha256`` beside it when there is one)."""
    from . import scoring as v05_scoring

    path = Path(path)
    sha = path.parent / "artifact.sha256"
    spec, info = v05_scoring.load_run_artifact(path, sha if sha.is_file() else None)
    model = v03_sealed.load_net(spec)
    return model, {"phase": info["artifact_metadata"].get("phase"), "artifact_path": _rel(path),
                   "artifact_sha256": info["artifact_sha256"], "dry_run": True}


# ---------------------------------------------------------------------------
# OOD disclosure
# ---------------------------------------------------------------------------

# the handed-over file is DISCLOSURE_W.json; neither its name nor its format string says "OOD"
OOD_FORMAT = "v05-phase-c-disclosure-w/1"
ID_PREFIX = "W"
# the header uses exactly Phase B's top-level keys (kit_b.DISCLOSURE_TOP_KEYS, screened by kit_c.check_disclosure),
# so the description of the source is part of the selection text
OOD_SELECTION_TEXT = (
    "windows of sessions from a second, independently written log generator (not the generator of the V001..V200 "
    "windows; its sessions are rendered by the same renderer into the same log format), restricted to some of its "
    "scenario families; windows that pass the runtime gate and carry a scored oracle label; for each class in the order of "
    "class_names, candidates sorted by a fixed hash key of (session, endpoint), a window taken unless its session "
    "was already used, 50 per class: 200 windows from 200 distinct sessions; ids W001..W200 in key order. The rule "
    "uses no model output")
REFERENCE_POLICY_TEXT = ("none: the v0.5 networks have no reference policy, so the windows carry no "
                         "reference_policy_class field")
PREFIX_TEXT = ("the rendered log of blocks 0..t joined with newlines; the line '### window start' (not log text: "
               "remove it before rendering or tokenising) stands immediately before block t-7, the window's first "
               "block; block t, the window's last block, ends the text; after removing the marker, every line that "
               "does not start with a space begins a new block")


def disclosure_candidates(master_seed: int = DISCLOSURE_SEED, per_family: int = DISCLOSURE_SESSIONS_PER_FAMILY, *,
                          families: Sequence[str] = SEEN_FAMILIES, cfg: OracleConfig | None = None
                          ) -> tuple[Encoded, dict[str, dict[str, Any]]]:
    """Candidate sessions of the OOD disclosure (seen families only) and per-session ``{family, k, seed}``."""
    check_seed(master_seed, "disclosure")
    unseen = set(families) - set(SEEN_FAMILIES)
    if unseen:
        raise DataError(f"the OOD disclosure draws only from the seen families; got {sorted(unseen)}")
    meta: dict[str, dict[str, Any]] = {}

    def tap(it: Iterator[Session]) -> Iterator[Session]:
        for s in it:
            meta[s.session_id] = {"family": s.family.removeprefix("ood:"), "k": s.k, "seed": s.seed}
            yield s

    enc = encode_sessions(tap(generate_ood(master_seed, families, per_family, role="disclosure", cfg=cfg)),
                          keep_blocks=True)
    return enc, meta


def leak_problems(text: str, forbidden: Iterable[str]) -> list[str]:
    """Forbidden strings (family names, session ids, seeds) that occur in ``text``."""
    return sorted({f for f in forbidden if f and f in text})


def forbidden_strings(meta: Mapping[str, Mapping[str, Any]], master_seed: int) -> set[str]:
    from ..synth.families import FAMILY_NAMES

    out = {*ood_families(), *FAMILY_NAMES}
    if len(str(master_seed)) >= 6:  # a short test seed would match ordinary log digits
        out.add(str(master_seed))
    for sid, m in meta.items():
        out |= {sid, sid[2:], str(m["seed"])}
    return out


def build_ood_disclosure(enc: Encoded, meta: Mapping[str, Mapping[str, Any]], model: Any, source: Mapping[str, Any],
                         *, master_seed: int, per_family: int, per_class: int = v03_scoring.PER_CLASS
                         ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Public (handed over) and private (repo only) records of the OOD disclosure."""
    data = enc.data
    chosen = v03_scoring.disclosure_selection(data.rows, per_class)
    row_idx = [i for _, i in chosen]
    sess_idx = sorted({int(data.row_session[i]) for i in row_idx})
    if len(sess_idx) != len(chosen):
        raise DataError("two disclosed windows share a session")
    for i in row_idx:
        r = data.rows[i]
        if not (r.gate_passed and r.label.is_scored):
            raise DataError(f"selected window {r.session_id}:{r.endpoint} is not gate-passed and scored")
    net, _lat = v03_scoring.network_predictions(model, data, sessions=sess_idx)
    public_windows, private_windows = [], []
    width = max(3, len(str(len(chosen))))
    for n, (key, i) in enumerate(chosen, start=1):
        r = data.rows[i]
        pos = int(data.row_pos[i])
        wid = f"{ID_PREFIX}{n:0{width}d}"
        blocks = enc.blocks[r.session_id]
        if any(line == v03_scoring.WINDOW_MARKER for blk in blocks for line in blk.split("\n")):
            raise DataError(f"{wid}: a log line equals the window marker")
        if net[i] < 0:
            raise DataError(f"{wid}: missing network prediction")
        w = {"id": wid, "prefix": v03_scoring.render_disclosed_prefix(blocks, pos),
             "oracle_label": r.label.to_class().name, "network_class": CLASS_NAMES[int(net[i])]}
        public_windows.append(w)
        m = meta[r.session_id]
        private_windows.append({"id": wid, "session_id": r.session_id, "family": m["family"], "k": m["k"],
                                "seed": m["seed"], "endpoint": int(r.endpoint), "block": pos,
                                "oracle_label": w["oracle_label"], "key": key,
                                "retime_offset_s": retime_offset_s(r.session_id)})
    sha = source["artifact_sha256"]
    is_b = source.get("phase") == B_PHASE and sha == B_SHA256 and not source.get("dry_run")
    net_name = "the v0.5 Phase B network (the network this revision starts from)" if is_b else \
        "the dry-run artefact's network"
    public = {
        "format": OOD_FORMAT,
        "class_names": list(CLASS_NAMES),
        "window_marker": v03_scoring.WINDOW_MARKER,
        "network": {"phase": B_PHASE if is_b else None, "artifact_sha256": sha,
                    "description": f"{net_name}: the network whose classes this disclosure gives"},
        "selection": OOD_SELECTION_TEXT,
        "reference_policy": REFERENCE_POLICY_TEXT,
        "fields": {
            "prefix": PREFIX_TEXT,
            "oracle_label": "the oracle label of the window ending at block t",
            "network_class": (f"argmax at EOE_t of {net_name} (artifact sha256 {sha}), from one causal forward pass; "
                              "ties go to the first class"),
        },
        "windows": public_windows,
    }
    text = json.dumps(public, sort_keys=True)
    leaks = leak_problems(text, forbidden_strings(meta, master_seed))
    if leaks:
        raise DataError(f"the public OOD disclosure would contain forbidden strings: {leaks[:5]}")
    private = {
        "format": OOD_FORMAT + "-private",
        "note": "repo only; never handed to the weight author",
        "generator": "agent_loop_lab.synth_ood",
        "master_seed": master_seed,
        "seed_formula": 'int.from_bytes(sha256(f"{master_seed}:ood:{family}:{k}").digest()[:8], "big")',
        "session_id_format": "o-{seed:016x}",
        "retime": dict(RETIME_RULE),
        "family_rule": {"text": FAMILY_RULE_TEXT, "seen": list(SEEN_FAMILIES), "unseen": list(UNSEEN_FAMILIES)},
        "families_generated": sorted({m["family"] for m in meta.values()}, key=list(SEEN_FAMILIES).index),
        "sessions_per_family": per_family,
        "candidates": {"sessions": len(data.session_ids), "windows": len(data.rows),
                       "session_ids_sha256": hashlib.sha256("\n".join(data.session_ids).encode()).hexdigest(),
                       "rows_sha256": enc.rows_sha256, "encoding_sha256": enc.encoding_sha256},
        "selection": ("v03.scoring.disclosure_selection (by import): gate passed, scored oracle label, "
                      f"{per_class} per class in class order, ascending key, at most one window per session"),
        "key_format": v03_scoring.DISCLOSURE_KEY_FORMAT,
        "per_class": per_class,
        "source": dict(source),
        "source_sha256": _sources_sha256(),
        "windows": private_windows,
    }
    return public, private


def _rel(p: Path) -> str:
    p = Path(p).resolve()
    return p.relative_to(REPO).as_posix() if p.is_relative_to(REPO) else str(p)
