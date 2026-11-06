"""Run one v0.4 val readout as several MPS worker processes and merge them into one column (deviation D2).

    uv run python scripts/v04_parallel_readout.py --out COLUMN.npz [--workers 3] [--sessions FILE] [--edit ARCHIVE]
                                                  [--seed-partial PARTIAL.npz] [--device mps] [--dtype fp32]

Matthew (2026-11-06) chose 3 parallel workers after a benchmark on this Mac (1.84x the throughput of one;
logits bit-identical to a single process). Each worker is the unchanged frozen readout
(``uv run --project envs/qwen python -m hypaqwen.run_readout --split val --sessions SHARD``, with its own
checkpoint and resume), so nothing about how a session is read out changes.

* **Sessions:** every val session in export order, or those listed in ``--sessions FILE`` (e.g.
  ``datasets/v0_4/VAL_QUICK.txt``). ``--seed-partial`` takes the rows of an interrupted single-process run
  (``OUT.partial.npz`` of ``hypaqwen.run_readout`` with the same split, device, dtype, edit and subset); its
  sessions are not read out again. It is copied into ``OUT.shards/seed.partial.npz`` on the first run, and
  that copy is what every run uses.
* **Shards:** the remaining sessions are dealt round-robin in export order to the workers. The plan
  (``OUT.shards/PLAN.json``) is fixed on the first run and must be the same on a re-run, which resumes:
  finished shards are kept (each checked against the requested device, dtype, edit and its shard list before
  the merge), and unfinished ones resume from their own checkpoints. ``OUT.shards/LOCK`` keeps a second run out.
* **Merge:** each shard's rows must be exactly its sessions' ``window_pos`` from the task export (no labels are
  read) and every shard's meta must agree on split, device, dtype, versions, model, interface and edit. The
  rows are put back in export order. The merged meta is a ``hypaqwen.run_readout`` meta: ``subset`` is the
  ``--sessions`` file's record (or null for full val) and ``parallel`` records the workers, the shards'
  sha256 and the seed. ``agent_loop_lab.v04.scoring.load_column`` and ``disclosure.check_column`` accept it
  as they accept a single-process column.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[1]
QWEN = REPO / "envs" / "qwen"
TASKS = REPO / "datasets" / "v0_4" / "val.tasks.jsonl.gz"
SPLIT = "val"
PLAN_FORMAT = "v04-parallel-readout/1"
# Meta fields every shard must share; the merged column takes them from the shards.
SHARED = ("split", "device", "dtype", "torch", "transformers", "platform", "python", "attn_implementation", "model",
          "interface_sha256", "interface", "edit")


class MergeError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_ids(path: Path) -> tuple[list[str], dict[str, Any]]:
    """As ``hypaqwen.run_readout.read_sessions``: ids and the ``meta["subset"]`` record."""
    raw = Path(path).read_bytes()
    ids = [s for s in raw.decode().split("\n") if s]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path}: duplicate session ids")
    return ids, {"file": Path(path).name, "sha256": hashlib.sha256(raw).hexdigest(), "sessions": len(ids)}


def export_windows(path: Path = TASKS) -> dict[str, list[int]]:
    """``{session_id: window_pos}`` in export order (insertion order), from the label-free task export."""
    out: dict[str, list[int]] = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            t = json.loads(line)
            out[t["session_id"]] = [int(p) for p in t["window_pos"]]
    return out


def plan(order: Sequence[str], wanted: Sequence[str] | None, seeded: Sequence[str], workers: int) -> list[list[str]]:
    """The shards: the wanted sessions (all of ``order`` when None) not in ``seeded``, round-robin in export order."""
    if workers < 1:
        raise SystemExit("--workers must be at least 1")
    want = set(order if wanted is None else wanted)
    missing = want - set(order)
    if missing:
        raise SystemExit(f"{len(missing)} listed sessions are not in {SPLIT} (e.g. {sorted(missing)[0]})")
    if set(seeded) - want:
        raise SystemExit("the seed partial holds sessions outside the requested set")
    rest = [s for s in order if s in want and s not in set(seeded)]
    return [rest[k::workers] for k in range(workers)]


def load_rows(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        return {"label_logits": z["label_logits"], "logsumexp": z["logsumexp"],
                "session_ids": [str(s) for s in z["session_ids"]], "row_pos": z["row_pos"],
                "meta": json.loads(str(z["meta"]))}


def per_session(rows: Mapping[str, Any], windows: Mapping[str, Sequence[int]], name: str) -> dict[str, tuple]:
    """``{session_id: (logits, lse, row_pos)}`` of one shard, checked against the export's ``window_pos``."""
    out, i = {}, 0
    for s in rows["session_ids"]:
        if s not in windows:
            raise MergeError(f"{name}: session {s} is not in the export")
        n = len(windows[s])
        pos = rows["row_pos"][i:i + n]
        if len(pos) != n or [int(p) for p in pos] != list(windows[s]):
            raise MergeError(f"{name}: rows of {s} are not its export window_pos")
        out[s] = (rows["label_logits"][i:i + n], rows["logsumexp"][i:i + n], pos)
        i += n
    if i != len(rows["row_pos"]) or len(rows["label_logits"]) != i or len(rows["logsumexp"]) != i:
        raise MergeError(f"{name}: {len(rows['row_pos']) - i} rows left over after its sessions")
    return out


def check_shard(name: str, meta: Mapping[str, Any], *, device: str, dtype: str, edit_sha: str | None,
                shard_sha: str) -> None:
    """A finished shard was read out with the requested arguments, over exactly its plan's session list."""
    problems = []
    if meta.get("device") != device or meta.get("dtype") != dtype:
        problems.append(f"read out on {meta.get('device')}/{meta.get('dtype')}, not {device}/{dtype}")
    got_edit = (meta.get("edit") or {}).get("build_record_sha256")
    if got_edit != edit_sha:
        problems.append(f"edit BUILD_RECORD sha256 {got_edit}, not {edit_sha}")
    if (meta.get("subset") or {}).get("sha256") != shard_sha:
        problems.append("its session list is not the plan's shard")
    if problems:
        raise MergeError(f"{name}: " + "; ".join(problems))


def merge(order: Sequence[str], windows: Mapping[str, Sequence[int]], wanted: Sequence[str],
          shards: Sequence[tuple[str, Mapping[str, Any]]], seed: tuple[str, Mapping[str, Any]] | None,
          subset: Mapping[str, Any] | None, parallel: Mapping[str, Any], seconds: float
          ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The merged column's arrays and meta (module docstring). ``shards``: ``(name, rows)`` of finished workers,
    each already checked by :func:`check_shard`; at least one is needed for the shared meta."""
    if not shards:
        raise MergeError("no worker shards (the seed alone has no full meta; read out at least one session)")
    metas = [r["meta"] for _, r in shards]
    for name, r in shards[1:]:
        diff = [k for k in SHARED if r["meta"].get(k) != metas[0].get(k)]
        if diff:
            raise MergeError(f"{name}: meta differs from {shards[0][0]} in {diff}")
    if metas[0].get("split") != SPLIT or any(m.get("limit") is not None for m in metas):
        raise MergeError("a shard is not a complete val readout")
    got: dict[str, tuple] = {}
    for name, r in list(shards) + ([seed] if seed is not None else []):
        for s, v in per_session(r, windows, name).items():
            if s in got:
                raise MergeError(f"{name}: session {s} is in two shards")
            got[s] = v
    if set(got) != set(wanted):
        raise MergeError(f"merged sessions differ from the requested ones ({len(set(wanted) - set(got))} missing, "
                         f"{len(set(got) - set(wanted))} extra)")
    ids = [s for s in order if s in got]
    arrays = {
        "label_logits": np.concatenate([got[s][0] for s in ids]).astype(np.float32),
        "logsumexp": np.concatenate([got[s][1] for s in ids]).astype(np.float32),
        "session_ids": np.array(ids),
        "row_pos": np.concatenate([got[s][2] for s in ids]).astype(np.int32),
    }
    tokens = sum(m["tokens"] for m in metas) + (seed[1]["meta"]["tokens"] if seed is not None else 0)
    meta = {k: metas[0].get(k) for k in SHARED}
    meta.update({"limit": None, "subset": None if subset is None else dict(subset),
                 "resumed_after_sessions": None if seed is None else len(seed[1]["session_ids"]),
                 "sessions": len(ids), "tokens": int(tokens), "seconds": round(seconds, 1),
                 "tokens_per_second": round(tokens / seconds, 1) if seconds > 0 else None,
                 "parallel": dict(parallel)})
    return arrays, meta


def save(path: Path, arrays: Mapping[str, np.ndarray], meta: Mapping[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays, meta=json.dumps(meta))
    os.replace(tmp, path)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--sessions", type=Path, default=None)
    ap.add_argument("--edit", type=Path, default=None)
    ap.add_argument("--seed-partial", type=Path, default=None)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dtype", default="fp32")
    a = ap.parse_args(argv)
    out = a.out.resolve()
    if out.exists():
        print(f"refusing: {out} exists", file=sys.stderr)
        return 2
    windows = export_windows()
    order = list(windows)
    wanted, subset = read_ids(a.sessions) if a.sessions is not None else (None, None)
    edit = None if a.edit is None else a.edit.resolve()
    edit_sha = None if edit is None else sha256_file(edit / "BUILD_RECORD.json")
    out_shards = out.with_name(out.name.removesuffix(".npz") + ".shards")
    out_shards.mkdir(parents=True, exist_ok=True)
    lock = open(out_shards / "LOCK", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"refusing: another run holds {out_shards}/LOCK", file=sys.stderr)
        return 2
    seed = None
    seed_copy = out_shards / "seed.partial.npz"
    if a.seed_partial is not None:
        if not seed_copy.exists():  # the source run's checkpoint can change later; the copy cannot
            shutil.copyfile(a.seed_partial, seed_copy.with_name(seed_copy.name + ".tmp"))
            os.replace(seed_copy.with_name(seed_copy.name + ".tmp"), seed_copy)
        rows = load_rows(seed_copy)
        want_args = {"split": SPLIT, "limit": None, "device": a.device, "dtype": a.dtype,
                     "edit_build_record_sha256": edit_sha, "subset_sha256": subset and subset["sha256"]}
        if rows["meta"].get("run_args") != want_args:
            print(f"refusing: {a.seed_partial} ran with {rows['meta'].get('run_args')}, not {want_args}", file=sys.stderr)
            return 2
        seed = (a.seed_partial.name, rows)
    shards = plan(order, wanted, [] if seed is None else seed[1]["session_ids"], a.workers)
    sdir = out_shards
    the_plan = {"format": PLAN_FORMAT, "workers": a.workers, "device": a.device, "dtype": a.dtype,
                "edit_build_record_sha256": edit_sha, "subset": subset,
                "tasks_sha256": sha256_file(TASKS),
                "seed_partial": None if seed is None else {"file": seed[0], "sha256": sha256_file(seed_copy),
                                                            "sessions": len(seed[1]["session_ids"])},
                "shards": [hashlib.sha256("".join(s + "\n" for s in sh).encode()).hexdigest() for sh in shards]}
    plan_file = sdir / "PLAN.json"
    if plan_file.exists() and json.loads(plan_file.read_text()) != the_plan:
        print(f"refusing: {plan_file} is another plan; move {sdir} away", file=sys.stderr)
        return 2
    plan_file.write_text(json.dumps(the_plan, indent=1, sort_keys=True) + "\n")
    t0 = time.monotonic()
    procs = []
    for k, sh in enumerate(shards):
        ids_file, col = sdir / f"shard{k}.txt", sdir / f"shard{k}.npz"
        ids_file.write_text("".join(s + "\n" for s in sh))
        if col.exists() or not sh:
            continue
        cmd = ["uv", "run", "--project", str(QWEN), "python", "-m", "hypaqwen.run_readout", "--split", SPLIT,
               "--sessions", str(ids_file), "--device", a.device, "--dtype", a.dtype, "--out", str(col)]
        if edit is not None:
            cmd += ["--edit", str(edit)]
        with open(sdir / f"shard{k}.log", "a") as log:
            procs.append((k, subprocess.Popen(cmd, cwd=QWEN, stdout=log, stderr=subprocess.STDOUT)))
        print(f"shard {k}: {len(sh)} sessions, pid {procs[-1][1].pid}", flush=True)
    failed = [k for k, p in procs if p.wait() != 0]
    if failed:
        print(f"refusing to merge: shards {failed} failed (see {sdir}/shard*.log); re-run to resume", file=sys.stderr)
        return 1
    done = [(f"shard{k}.npz", load_rows(sdir / f"shard{k}.npz")) for k, sh in enumerate(shards) if sh]
    try:
        for (name, r), sh_sha in zip(done, [h for h, sh in zip(the_plan["shards"], shards) if sh]):
            check_shard(name, r["meta"], device=a.device, dtype=a.dtype, edit_sha=edit_sha, shard_sha=sh_sha)
    except MergeError as exc:
        print(f"refusing to merge: {exc}", file=sys.stderr)
        return 2
    shard_secs = [r["meta"]["seconds"] for _, r in done]
    parallel = {"workers": a.workers, "plan_sha256": sha256_file(plan_file),
                "shards": [{"file": n, "sha256": sha256_file(sdir / n), "sessions": r["meta"]["sessions"],
                            "tokens": r["meta"]["tokens"], "seconds": r["meta"]["seconds"],
                            "resumed_after_sessions": r["meta"].get("resumed_after_sessions")} for n, r in done],
                "seed_partial": the_plan["seed_partial"],
                "seconds_note": "seconds = the slowest shard's readout seconds (plus the seed's, when seeded)"}
    seconds = max(shard_secs, default=0.0) + (seed[1]["meta"]["seconds"] if seed is not None else 0.0)
    try:
        arrays, meta = merge(order, windows, order if wanted is None else wanted, done, seed, subset, parallel, seconds)
    except MergeError as exc:
        print(f"refusing to merge: {exc}", file=sys.stderr)
        return 2
    save(out, arrays, meta)
    print(f"{out}: {meta['sessions']} sessions, {len(arrays['row_pos'])} rows, {meta['tokens']} tokens, "
          f"wall {time.monotonic() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
