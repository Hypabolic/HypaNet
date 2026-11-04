"""Standard and counterfactual dataset build (M1_CONTRACTS §5; spec v0.2 §7.3).

Layout under ``out_dir`` (default ``datasets/v0_2``)::

    MANIFEST.json                              committed
    splits.json                                committed
    standard/{dev,val,test}.events.jsonl.gz    data (not committed)
    standard/{dev,val,test}.oracle.jsonl.gz
    standard/counterfactual.events.jsonl.gz
    standard/counterfactual.oracle.jsonl.gz

Seeding and splits:

* Session seed = ``int.from_bytes(sha256(f"{master_seed}:{family}:{k}").digest()[:8], "big")``
  for ``k = 0 .. n_family - 1``. ``n_family = N // 12``, and the first ``N % 12``
  families in ``FAMILY_NAMES`` order get one more.
* Session id = ``f"s-{seed:016x}"``: opaque (no family name on the inference
  object) and the seed is recoverable from it.
* Within each family the indices ``0 .. n-1`` are shuffled with
  ``random.Random(split_key)``, ``split_key`` = the first 8 bytes of
  ``sha256(f"{master_seed}:split:{family}")``. The shuffled list is cut at
  ``c1 = (6n + 5) // 10`` and ``c2 = (8n + 5) // 10`` (60/20/20, rounded half
  up): positions ``[0, c1)`` go to dev, ``[c1, c2)`` to val, the rest to test.
  Assignment depends only on the master seed, family and index, never on
  generated content or labels.
* Counterfactual pair ``k`` of kind ``kind`` uses seed
  ``sha256(f"{master_seed}:counterfactual:{kind}:{k}")[:8]`` and pair id
  ``f"cf-{seed:016x}"``. Its two sessions get opaque ids
  ``"cf-" + sha256(f"{pair_id}:{branch}")[:16]``
  (:func:`counterfactual.branch_session_id`); the branch, kind and pair id
  appear only in the oracle record's ``latent_trace["counterfactual"]``.

Files are written in family order, then index order (counterfactual: kind
order, pair order, then the pair's two sessions in the order given by
:func:`counterfactual_branch_order` of the pair seed, so neither the id nor
the line position tells which branch is which). Each line is canonical JSON (sorted keys,
no whitespace). gzip uses ``mtime=0``, an empty header filename and level
:data:`GZIP_LEVEL`, so a rebuild with the same code is byte-identical.

Generation errors propagate (with family, index and seed); nothing is skipped,
because skipping would silently change counts and seeding.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from ..oracle_config import OracleConfig, load_oracle_config
from ..schema import SPEC_VERSION, CanonicalEvent, OracleLabel, OracleRecord

__all__ = [
    "DEFAULT_MASTER_SEED",
    "STANDARD_SESSIONS",
    "PAIRS_PER_KIND",
    "GZIP_LEVEL",
    "SPLITS",
    "STANDARD_DIR",
    "BuildError",
    "BuildResult",
    "session_seed",
    "counterfactual_seed",
    "session_id_for",
    "counterfactual_branch_order",
    "pair_id_for",
    "family_counts",
    "split_cuts",
    "assign_splits",
    "events_line",
    "oracle_line",
    "data_path",
    "iter_sessions",
    "build",
    "write_reports",
    "file_sha256",
]

DEFAULT_MASTER_SEED = 20261104
STANDARD_SESSIONS = 20_000
PAIRS_PER_KIND = 300
GZIP_LEVEL = 6
SPLITS: tuple[str, ...] = ("dev", "val", "test")
COUNTERFACTUAL = "counterfactual"
STANDARD_DIR = "standard"
MANIFEST_FILE = "MANIFEST.json"
SPLITS_FILE = "splits.json"
STATS_REPORT = "m1_dataset_stats.json"
COLLISIONS_REPORT = "m1_collisions_dev.json"

SEED_FORMULA = 'int.from_bytes(sha256(f"{master_seed}:{family}:{k}").digest()[:8], "big")'
CF_SEED_FORMULA = 'int.from_bytes(sha256(f"{master_seed}:counterfactual:{kind}:{k}").digest()[:8], "big")'
CF_ORDER_RULE = (
    'per pair: lines ordered (b, a) if int.from_bytes(sha256(f"{seed}:cf-order").digest()[:8], "big") & 1 '
    'else (a, b); seed = the pair seed'
)
SPLIT_RULE = (
    'per family: indices 0..n-1 shuffled by random.Random(int.from_bytes(sha256(f"{master_seed}:split:{family}")'
    '.digest()[:8], "big")); shuffled positions [0, (6n+5)//10) -> dev, [(6n+5)//10, (8n+5)//10) -> val, rest -> test'
)
FAMILY_COUNT_RULE = "n_family = N // 12, plus 1 for the first N % 12 families in FAMILY_NAMES order"

_SRC_ROOT = Path(__file__).resolve().parents[1]  # src/agent_loop_lab


class BuildError(RuntimeError):
    """A session or pair could not be generated, or a written file failed its checks."""


@dataclass(frozen=True)
class BuildResult:
    out_dir: Path
    manifest: dict[str, Any]
    splits: dict[str, Any]
    reports: tuple[dict[str, Any], dict[str, Any]] | None


# ---------------------------------------------------------------------------
# Seeding and split assignment
# ---------------------------------------------------------------------------


def _h64(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def session_seed(master_seed: int, family: str, k: int) -> int:
    """M1_CONTRACTS §5 session seed."""
    return _h64(f"{master_seed}:{family}:{k}")


def counterfactual_seed(master_seed: int, kind: str, k: int) -> int:
    return _h64(f"{master_seed}:counterfactual:{kind}:{k}")


def counterfactual_branch_order(seed: int) -> tuple[str, str]:
    """Line order of a pair's branches: ``("b", "a")`` when the low bit of
    ``sha256(f"{seed}:cf-order")[:8]`` is set, else ``("a", "b")``."""
    return ("b", "a") if _h64(f"{seed}:cf-order") & 1 else ("a", "b")


def session_id_for(seed: int) -> str:
    return f"s-{seed:016x}"


def pair_id_for(seed: int) -> str:
    return f"cf-{seed:016x}"


def family_counts(n_sessions: int, families: Sequence[str]) -> dict[str, int]:
    """Even split across ``families``; the remainder goes to the first families in order."""
    if n_sessions < 0:
        raise ValueError("n_sessions must be >= 0")
    base, rem = divmod(n_sessions, len(families))
    return {f: base + (1 if i < rem else 0) for i, f in enumerate(families)}


def split_cuts(n: int) -> tuple[int, int]:
    """Cut points of the 60/20/20 split of ``n`` shuffled sessions (round half up)."""
    return (6 * n + 5) // 10, (8 * n + 5) // 10


def assign_splits(master_seed: int, family: str, n: int) -> list[str]:
    """``out[k]`` = split of index ``k`` of ``family``. Depends only on the arguments."""
    order = list(range(n))
    random.Random(_h64(f"{master_seed}:split:{family}")).shuffle(order)
    c1, c2 = split_cuts(n)
    out = [""] * n
    for pos, k in enumerate(order):
        out[k] = SPLITS[0] if pos < c1 else SPLITS[1] if pos < c2 else SPLITS[2]
    return out


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def _canon_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def events_line(session_id: str, events: Sequence[CanonicalEvent]) -> str:
    """One events-file line: exactly ``{"session_id", "events"}``."""
    for e in events:
        if e.session_id != session_id:
            raise BuildError(f"event {e.event_id!r} belongs to {e.session_id!r}, not {session_id!r}")
    return _canon_json({"session_id": session_id, "events": [e.to_dict() for e in events]})


def oracle_line(record: OracleRecord) -> str:
    return _canon_json(record.to_dict())


class _GzLines:
    """Deterministic gzip line writer (mtime 0, empty header filename, fixed level)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._raw = open(path, "wb")
        self._gz = gzip.GzipFile(filename="", mode="wb", fileobj=self._raw, compresslevel=GZIP_LEVEL, mtime=0)
        self.lines = 0

    def write(self, line: str) -> None:
        self._gz.write(line.encode("utf-8"))
        self._gz.write(b"\n")
        self.lines += 1

    def close(self) -> None:
        self._gz.close()
        self._raw.close()


def data_path(out_dir: str | Path, split: str, kind: str) -> Path:
    """``<out>/standard/<split>.<kind>.jsonl.gz`` with ``kind`` in {events, oracle}."""
    if kind not in ("events", "oracle"):
        raise ValueError(f"kind must be 'events' or 'oracle', got {kind!r}")
    return Path(out_dir) / STANDARD_DIR / f"{split}.{kind}.jsonl.gz"


def iter_sessions(out_dir: str | Path, split: str) -> Iterator[tuple[list[CanonicalEvent], OracleRecord]]:
    """Read one split back as ``(events, record)`` pairs, in file order.

    Each events line must hold exactly ``session_id`` and ``events``, and every
    event goes through ``CanonicalEvent.from_dict``, which rejects any field
    outside the public schema (oracle leakage).
    """
    ev_path, or_path = data_path(out_dir, split, "events"), data_path(out_dir, split, "oracle")
    with gzip.open(ev_path, "rt", encoding="utf-8") as fe, gzip.open(or_path, "rt", encoding="utf-8") as fo:
        for n, (le, lo) in enumerate(zip(fe, fo, strict=True)):
            d = json.loads(le)
            if set(d) != {"session_id", "events"}:
                raise BuildError(f"{ev_path}:{n + 1}: events line keys {sorted(d)} != ['events', 'session_id']")
            events = [CanonicalEvent.from_dict(e) for e in d["events"]]
            record = OracleRecord.from_dict(json.loads(lo))
            if record.session_id != d["session_id"]:
                raise BuildError(f"{split} line {n + 1}: oracle {record.session_id!r} != events {d['session_id']!r}")
            yield events, record


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _generator_source_sha256() -> str:
    """sha256 over the generator sources (``synth/**/*.py``, sorted relative paths and bytes)."""
    h = hashlib.sha256()
    root = _SRC_ROOT / "synth"
    for p in sorted(root.rglob("*.py"), key=lambda q: q.relative_to(root).as_posix()):
        if "__pycache__" in p.parts:
            continue
        rel = p.relative_to(root).as_posix().encode("utf-8")
        data = p.read_bytes()
        h.update(len(rel).to_bytes(8, "big") + rel + len(data).to_bytes(8, "big") + data)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


class _Counts:
    def __init__(self) -> None:
        self.sessions = 0
        self.events = 0
        self.windows: Counter = Counter()
        self.families: Counter = Counter()

    def add(self, events: Sequence[CanonicalEvent], record: OracleRecord) -> None:
        self.sessions += 1
        self.events += len(events)
        self.families[record.scenario_family] += 1
        self.windows.update(OracleLabel(v) for v in record.window_labels.values())

    def to_dict(self, families: Sequence[str]) -> dict[str, Any]:
        return {
            "sessions": self.sessions,
            "events": self.events,
            "windows": sum(self.windows.values()),
            "window_labels": {lab.value: self.windows.get(lab, 0) for lab in OracleLabel},
            "families": {f: self.families.get(f, 0) for f in families},
        }


def _check_events(events: Sequence[CanonicalEvent], session_id: str) -> None:
    from ..windows import validate_sequence

    if not events or any(e.session_id != session_id for e in events):
        raise BuildError(f"session {session_id!r}: empty or foreign events")
    validate_sequence(events)


def build(
    out_dir: str | Path,
    *,
    master_seed: int = DEFAULT_MASTER_SEED,
    sessions: int = STANDARD_SESSIONS,
    pairs_per_kind: int = PAIRS_PER_KIND,
    cfg: OracleConfig | None = None,
    reports_dir: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
    on_session: Callable[[str, list[CanonicalEvent], OracleRecord], None] | None = None,
) -> BuildResult:
    """Generate the standard splits and the counterfactual set into ``out_dir``.

    Writes the six split files, the two counterfactual files, ``MANIFEST.json``
    and ``splits.json``. When ``reports_dir`` is given, also reads the split
    files back and writes the dataset-stats and dev collision reports there.
    ``on_session(split, events, record)`` (optional, observe-only) is called
    right after each session's lines are written, in file order; the v0.3
    raw builder uses it. It does not change what is generated or written.
    """
    from ..synth import counterfactual as cf
    from ..synth.engine import GENERATOR_VERSION
    from ..synth.families import FAMILY_NAMES, generate_session

    say = progress or (lambda _msg: None)
    cfg = cfg or load_oracle_config()
    out = Path(out_dir)
    (out / STANDARD_DIR).mkdir(parents=True, exist_ok=True)
    counts = family_counts(sessions, FAMILY_NAMES)

    writers = {
        (split, kind): _GzLines(data_path(out, split, kind))
        for split in (*SPLITS, COUNTERFACTUAL)
        for kind in ("events", "oracle")
    }
    split_ids: dict[str, list[str]] = {s: [] for s in SPLITS}
    split_counts = {s: _Counts() for s in SPLITS}
    family_split: dict[str, dict[str, int]] = {}
    seen: set[str] = set()
    cf_summary: dict[str, dict[str, int]] = {}
    try:
        for family in FAMILY_NAMES:
            n = counts[family]
            assignment = assign_splits(master_seed, family, n)
            family_split[family] = {s: assignment.count(s) for s in SPLITS}
            for k in range(n):
                seed = session_seed(master_seed, family, k)
                sid = session_id_for(seed)
                if sid in seen:
                    raise BuildError(f"session id collision {sid!r} ({family} k={k})")
                seen.add(sid)
                try:
                    events, record = generate_session(family, seed, sid, cfg)
                    _check_events(events, sid)
                except Exception as exc:
                    raise BuildError(f"family {family} k={k} seed={seed}: {type(exc).__name__}: {exc}") from exc
                split = assignment[k]
                writers[(split, "events")].write(events_line(sid, events))
                writers[(split, "oracle")].write(oracle_line(record))
                split_ids[split].append(sid)
                split_counts[split].add(events, record)
                if on_session is not None:
                    on_session(split, events, record)
            say(f"{family}: {n} sessions {family_split[family]}")

        cf_counts = _Counts()
        for kind in cf.KINDS:
            with_contrast = 0
            for k in range(pairs_per_kind):
                seed = counterfactual_seed(master_seed, kind, k)
                pid = pair_id_for(seed)
                try:
                    pair = cf.generate_pair_from_seed(kind, seed, pid, cfg)
                    for events, record in pair:
                        _check_events(events, record.session_id)
                except Exception as exc:
                    raise BuildError(f"counterfactual {kind} k={k} seed={seed}: {type(exc).__name__}: {exc}") from exc
                by_branch = {rec.latent_trace["counterfactual"]["branch"]: (ev, rec) for ev, rec in pair}
                for events, record in (by_branch[b] for b in counterfactual_branch_order(seed)):
                    if record.session_id in seen:
                        raise BuildError(f"session id collision {record.session_id!r} (counterfactual {kind} k={k})")
                    seen.add(record.session_id)
                    writers[(COUNTERFACTUAL, "events")].write(events_line(record.session_id, events))
                    writers[(COUNTERFACTUAL, "oracle")].write(oracle_line(record))
                    cf_counts.add(events, record)
                    if on_session is not None:
                        on_session(COUNTERFACTUAL, events, record)
                with_contrast += bool(pair[0][1].latent_trace["counterfactual"]["contrast_endpoints"])
            cf_summary[kind] = {"pairs": pairs_per_kind, "pairs_with_contrast_endpoints": with_contrast}
            say(f"counterfactual {kind}: {pairs_per_kind} pairs, {with_contrast} with contrast")
    except BaseException:
        # A failed build must not leave plausible-looking partial data behind.
        for w in writers.values():
            w.close()
            w.path.unlink(missing_ok=True)
        for name in (MANIFEST_FILE, SPLITS_FILE):
            (out / name).unlink(missing_ok=True)
        raise
    for w in writers.values():
        w.close()

    files = {}
    for (split, kind), w in sorted(writers.items(), key=lambda kv: kv[1].path.name):
        rel = w.path.relative_to(out).as_posix()
        files[rel] = {"sha256": file_sha256(w.path), "bytes": w.path.stat().st_size, "lines": w.lines}

    manifest: dict[str, Any] = {
        "dataset": "v0_2",
        "spec_version": SPEC_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generator_source_sha256": _generator_source_sha256(),
        "oracle_config": cfg.to_dict(),
        "master_seed": master_seed,
        "seed_formula": SEED_FORMULA,
        "session_id_format": 's-{seed:016x}',
        "family_count_rule": FAMILY_COUNT_RULE,
        "split_rule": SPLIT_RULE,
        "gzip": {"mtime": 0, "compresslevel": GZIP_LEVEL, "header_filename": ""},
        "line_format": {
            "events": '{"session_id", "events": [CanonicalEvent.to_dict()...]}; canonical JSON',
            "oracle": "OracleRecord.to_dict(); canonical JSON; line i matches events line i",
        },
        "standard": {
            "sessions": sessions,
            "families": list(FAMILY_NAMES),
            "family_sessions": counts,
            "family_split_sessions": family_split,
            "splits": {s: split_counts[s].to_dict(FAMILY_NAMES) for s in SPLITS},
        },
        "counterfactual": {
            "seed_formula": CF_SEED_FORMULA,
            "pair_id_format": "cf-{seed:016x}",
            "session_id_format": 'cf-{sha256(f"{pair_id}:{branch}").hexdigest()[:16]}',
            "line_order_rule": CF_ORDER_RULE,
            "pairs_per_kind": pairs_per_kind,
            "kinds": cf_summary,
            **{k: v for k, v in cf_counts.to_dict(cf.KINDS).items() if k != "families"},
        },
        "files": files,
    }
    splits_doc = {
        "dataset": "v0_2",
        "master_seed": master_seed,
        "session_id_format": "s-{seed:016x}",
        "splits": split_ids,
    }
    _write_json(out / MANIFEST_FILE, manifest)
    _write_json(out / SPLITS_FILE, splits_doc)

    reports = None
    if reports_dir is not None:
        say("auditing (read-back)")
        reports = write_reports(out, reports_dir)
    return BuildResult(out, manifest, splits_doc, reports)


def write_reports(out_dir: str | Path, reports_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the split files back and write the two M1 audit reports.

    Dataset stats cover dev/val/test; the collision audit covers dev, plus
    identical vectors across splits.
    """
    from ..eval import audit_dataset, to_json

    stats, collisions = audit_dataset({s: iter_sessions(out_dir, s) for s in SPLITS}, audit_split="dev")
    rd = Path(reports_dir)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / STATS_REPORT).write_text(to_json(stats), encoding="utf-8")
    (rd / COLLISIONS_REPORT).write_text(to_json(collisions), encoding="utf-8")
    return stats, collisions


def _write_json(path: Path, obj: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
