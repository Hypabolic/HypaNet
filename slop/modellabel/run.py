#!/usr/bin/env python3
"""Two-model labelling harness for the slop answer key (SPEC.md §5, "Answer key").

    uv run python modellabel/run.py label --set S --labeller opus|codex|both [--screens ID ... | --first N] [--workers N]
                                          [--max-usd X] [--per-call-usd Y] [--dry-run]
    uv run python modellabel/run.py adjudicate --set S [--workers N] [--max-usd X] [--dry-run]
    uv run python modellabel/run.py report --set S
    uv run python modellabel/run.py estimate --set S|all [--json]
    uv run python modellabel/run.py build --set S          (rebuild the label files and final key from raw replies)

`label` and `adjudicate` are resumable: a screen whose raw reply is valid for the same prompt (guide, screen and
source unchanged) is skipped, and a refusal is not repeated unless --retry-refused. Screens run in the set's
order.txt order. --max-usd is a hard cap on this invocation's Opus spend (Claude Code's total_cost_usd) and
--max-total-usd one on all Opus spend in labels/raw/spend.jsonl: no call is launched once the spend plus the expected
cost of the calls in flight would pass either. One label/adjudicate run at a time (a lock in labels/raw/). --dry-run
renders the prompts (default labels/dryrun/, gitignored) and calls nothing.

Exit codes: 0 every requested screen has a valid reply (and, for build/adjudicate, the key is complete); 1 some screen
is refused, invalid or errored, the key is incomplete, or check.py found errors; 2 refused to start; 3 the spend cap
stopped launches.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modellabel import adjudicate as adj  # noqa: E402
from modellabel import agree  # noqa: E402
from modellabel import common as c  # noqa: E402
from modellabel import estimate as est  # noqa: E402
from modellabel import prompt as pr  # noqa: E402
from modellabel import runners as rn  # noqa: E402
from modellabel import to_records as tr  # noqa: E402

lk = c.lk
RAW_SCHEMA = "slop.modellabel.raw/1"
_print_lock = threading.Lock()


def say(*a):
    with _print_lock:
        print(*a, flush=True)


class Refused(SystemExit):
    pass


def open_set(name: str, datasets_dir: Path, registry_root: Path | None = None) -> c.LabelSet:
    root = c.set_root(name, datasets_dir)
    if name == "sealed_real":
        if not (root / "manifest.json").exists():
            raise Refused("refusing: sealed_real does not exist yet. It is built by scripts/build_sealed.py only after "
                          "the extractor freeze (SPEC.md §12.1, §12.6); label it then.")
        sealed_recorded(registry_root or c.SLOP_DIR)
    try:
        return c.load_set(name, datasets_dir)
    except c.SetMissing as e:
        raise Refused(f"refusing: {e}")


def sealed_recorded(root: Path):
    """sealed_real may be labelled only once the freeze record (registry.yaml, scripts/freeze.py) holds it, and only
    as recorded (its manifest unchanged since)."""
    sys.path.insert(0, str(Path(root) / "scripts"))
    try:
        import freeze  # noqa: E402
        rec = freeze.load(Path(root))
        want = (rec.get("sets") or {}).get("sealed_real")
        now = freeze.set_entry(Path(root), "sealed_real")
    except Exception as e:  # noqa: BLE001 - FreezeError, a missing yaml module, a bad record
        raise Refused(f"refusing: sealed_real is not in the freeze record ({e}); run scripts/freeze.py --revise "
                      "after build_sealed.py, then label")
    if not want:
        raise Refused("refusing: registry.yaml does not record sets.sealed_real yet (scripts/freeze.py --revise)")
    if want != now:
        raise Refused("refusing: datasets/sealed_real has changed since registry.yaml recorded it")


def pick_screens(ls: c.LabelSet, ids):
    if not ids:
        return list(ls.screens)
    want = set(ids)
    unknown = want - {s.id for s in ls.screens}
    if unknown:
        raise Refused(f"unknown screen ids for {ls.name}: {', '.join(sorted(unknown))}")
    return [s for s in ls.screens if s.id in want]


LOCK_NAME = ".modellabel.lock"


def take_lock(labels_dir: Path):
    """One label or adjudicate run at a time, for every set: two runs would pay twice for a screen, race on the raw
    files and each read the ledger total before the other's spend. Held until the process exits."""
    p = Path(labels_dir) / "raw" / LOCK_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    f = open(p, "a+")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        raise Refused(f"refusing: another modellabel label/adjudicate run holds {p}")
    return f


def release_lock(f):
    if f is not None:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def pilot_screens(ls: c.LabelSet, n: int = 3) -> list:
    """The pilot's screens, stratified by size so it calibrates thinking per node: the screens whose labelable node
    counts are nearest half the sealed mean, the sealed mean (9,474 / 422 = 22.45) and the set's largest, in set order,
    leaving out the guide's own worked example. Deterministic."""
    from modellabel.estimate import SEALED_TOTAL_NODES, SEALED_TOTAL_SCREENS
    mean = SEALED_TOTAL_NODES / SEALED_TOTAL_SCREENS
    pool = [s for s in ls.screens if s.id not in ls.guide_example_ids]
    sizes = {s.id: len(s.labelable()) for s in pool}
    targets = [mean / 2, mean, max(sizes.values(), default=0)][:n]
    picked = []
    for t in targets:
        cands = [s for s in pool if s.id not in picked]
        if not cands:
            break
        best = min(cands, key=lambda s: (abs(sizes[s.id] - t), pool.index(s)))
        picked.append(best.id)
    return [s for s in ls.screens if s.id in picked]


def sample_screens(ls: c.LabelSet, share: float, tag: str) -> list:
    """A fixed share of a set by hash rank (seed 20261116, namespace <tag>), at least one screen."""
    ranked = sorted(ls.screens, key=lambda s: hashlib.sha256(f"{tag}:20261116:{s.id}".encode()).hexdigest())
    k = max(1, round(share * len(ranked)))
    keep = {s.id for s in ranked[:k]}
    return [s for s in ls.screens if s.id in keep]


def labeller_keys(arg: str):
    return ["opus", "codex"] if arg == "both" else [arg]


def scratch_dir(arg, labels_dir) -> Path:
    """Dry-run output: --out-dir, else labels/dryrun/ (gitignored with the rest of labels/). Rendered prompts hold
    the set's source text, so they stay with the raw replies, never in a shared temporary directory."""
    return Path(arg) if arg else Path(labels_dir) / "dryrun"


def runner_for(key: str, a) -> rn.Runner:
    cwd = Path(a.cwd_root) / c.LABELLERS[key] if a.cwd_root else None
    if key == "opus":
        return rn.OpusRunner(bin_path=a.claude_bin, cwd=cwd, max_budget_usd=a.per_call_usd, timeout_s=a.timeout)
    return rn.CodexRunner(bin_path=a.codex_bin, cwd=cwd, timeout_s=a.timeout)


def save_prefix(labels_dir: Path, set_name: str, p: pr.Prompt, tag: str):
    path = Path(labels_dir) / "raw" / set_name / f"{tag}-{p.prefix_sha256[:12]}.txt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(p.prefix, encoding="utf-8")


def raw_record(phase, ls, scr, runner, guide_sha, p: pr.Prompt, out, extra=None, labeller=None) -> dict:
    return {"schema": RAW_SCHEMA, "phase": phase, "set": ls.name, "screen": scr.id,
            "labeller": labeller or runner.labeller,
            "runner": runner.describe(), "guide_sha256": guide_sha, "screen_sha256": scr.sha256,
            "source_sha256": c.sha256_text(scr.source_text) if scr.source_text is not None else None,
            "prompt_version": p.version, "prefix_sha256": p.prefix_sha256, "prompt_sha256": p.sha256,
            "prompt_body": p.body, **(extra or {}), "status": out["status"], "error": out.get("error"),
            "reply": out["reply"], "warnings": out.get("warnings") or [], "cost_usd": out["cost_usd"],
            "attempts": out["attempts"], "time": lk.now_iso()}


def resumable(raw, p: pr.Prompt, runner, retry_refused: bool) -> bool:
    if not raw or raw.get("prompt_sha256") != p.sha256 or (raw.get("runner") or {}).get("model") != runner.model:
        return False
    if (raw.get("runner") or {}).get("effort") != runner.effort:
        return False
    return raw.get("status") == "ok" or (raw.get("status") == "refused" and not retry_refused)


def run_jobs(jobs, workers: int, budget: rn.Budget):
    """jobs: list of (runner, callable). Each labeller gets its own pool of `workers` threads; jobs keep set order."""
    results = []
    by_runner = {}
    for runner, fn in jobs:
        by_runner.setdefault(runner.labeller, []).append(fn)
    pools = [ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix=lab) for lab in by_runner]
    futs = []
    for pool, fns in zip(pools, by_runner.values()):
        futs += [pool.submit(fn) for fn in fns]
    for f in futs:
        results.append(f.result())
    for p in pools:
        p.shutdown()
    return results


def tally(results):
    out = {}
    for r in results:
        out[r] = out.get(r, 0) + 1
    return out


# ---------------------------------------------------------------- label


def preflight(ls: c.LabelSet, screens: list, use_source: bool = True):
    """Refuse a set the labellers cannot judge fairly: a screen with no product kind (LABELLING.md §2a; sealed kinds
    go in datasets/sealed_real/screens/kinds.json, which is gitignored), or one whose "source" is screen JSON (the
    sealed sibling fixture, until build_sealed.py copies its real source)."""
    no_kind = [s.id for s in screens if not s.kind]
    if no_kind:
        raise Refused(f"refusing: {len(no_kind)} screens have no product kind (LABELLING.md §2a), e.g. "
                      f"{', '.join(no_kind[:3])}; record one neutral line per repo first ({ls.name}: "
                      f"{'datasets/sealed_real/screens/kinds.json, gitignored' if ls.name == 'sealed_real' else 'datasets/kinds.json'})")
    if use_source:
        bad = [s.id for s in screens if c.source_is_screen_json(s.source_text)]
        if bad:
            raise Refused(f"refusing: the source file of {', '.join(bad[:3])} is a slop.screen JSON document, not "
                          "source code; give the sealed sibling fixture its real source file first")


def pool_pins(labels_dir: Path, labeller: str) -> set:
    """(model, effort, CLI version, prompt version, prefix sha256) of every valid round-1 reply on the calibration pool
    (reference_real and calibration) from one labeller."""
    out = set()
    for name in ("reference_real", "calibration"):
        d = tr.raw_dir(labels_dir, name, labeller)
        for f in sorted(d.glob("*.json")) if d.is_dir() else []:
            raw = c.read_json(f)
            if raw and raw.get("status") == "ok":
                ru = raw.get("runner") or {}
                out.add((ru.get("model"), ru.get("effort"), ru.get("cli_version"), raw.get("prompt_version"),
                         raw.get("prefix_sha256")))
    return out


def check_sealed_pins(labels_dir: Path, runner, p: pr.Prompt):
    """sealed_real is labelled with exactly what labelled the pool: the same model, effort, CLI version, prompt
    version and prefix (guide). Anything else would make pool agreement say nothing about the sealed labels."""
    pins = pool_pins(labels_dir, runner.labeller)
    d = runner.describe()
    now = (d["model"], d["effort"], d.get("cli_version"), p.version, p.prefix_sha256)
    if not pins:
        raise Refused(f"refusing: {runner.labeller} has no pool labels (reference_real, calibration) to pin sealed_real "
                      "to; label the pool first")
    if pins != {now}:
        raise Refused(f"refusing: {runner.labeller} would label sealed_real with {now[:4]} (prefix {now[4][:12]}), but "
                      f"the pool was labelled with {sorted((x[:4] + (str(x[4])[:12],)) for x in pins)}")


def variant_screens(ls, screens, a):
    if a.variant == "retest" and not a.screens and not a.first and not a.pilot:
        return sample_screens(ls, RETEST_SHARE, "retest")
    return screens


RETEST_SHARE = 0.10


def cmd_label(a) -> int:
    labels_dir = Path(a.labels_dir)
    ls = open_set(a.set, Path(a.datasets_dir), a.registry_root)
    if a.first is not None and a.first < 1:
        raise Refused("refusing: --first must be at least 1")
    if a.pilot and (a.screens or a.first):
        raise Refused("refusing: --pilot picks its own screens; drop --screens and --first")
    if a.variant and ls.name == "sealed_real":
        raise Refused("refusing: variants (retest, nosource) run on the calibration pool only")
    screens = pilot_screens(ls) if a.pilot else pick_screens(ls, a.screens)
    if a.first is not None:
        screens = screens[:a.first]
    if a.variant:
        screens = variant_screens(ls, screens, a)
    use_source = a.variant != "nosource"
    preflight(ls, screens, use_source)
    guide = Path(a.guide)
    guide_text, guide_sha = guide.read_text(encoding="utf-8"), lk.sha256_file(guide)
    keys = labeller_keys(a.labeller)
    if a.dry_run:
        out = scratch_dir(a.out_dir, labels_dir) / ls.name
        for key in keys:
            d = out / out_id(c.LABELLERS[key], a.variant)
            d.mkdir(parents=True, exist_ok=True)
            total, p = 0, None
            for scr in screens:
                p = pr.render(scr, guide_text, source=use_source)
                (d / "prefix.txt").write_text(p.prefix, encoding="utf-8")
                (d / f"{scr.id}.prompt.txt").write_text(p.full, encoding="utf-8")
                total += len(p.full)
            say(f"dry run: {len(screens)} prompts for {d.name} in {d} ({total:,} characters"
                + (f"; prefix {len(p.prefix):,}" if p else "") + "); nothing was called")
        return 0
    if "opus" in keys and a.max_usd is None:
        raise Refused("refusing: --max-usd is required when Opus labels (it is the hard spend cap for this run)")
    lock = take_lock(labels_dir)
    try:
        return _label(a, ls, screens, use_source, guide, guide_text, guide_sha, keys, labels_dir)
    finally:
        release_lock(lock)


def out_id(labeller: str, variant) -> str:
    return agree.variant_id(labeller, variant) if variant else labeller


def make_budget(a, labels_dir: Path):
    ledger = labels_dir / "raw" / "spend.jsonl"
    prior = rn.ledger_total(ledger)
    if a.max_total_usd is not None and prior >= a.max_total_usd:
        raise Refused(f"refusing: the ledger already holds US${prior:.2f} of Opus spend, at or over --max-total-usd "
                      f"{a.max_total_usd:.2f}")
    return rn.Budget(a.max_usd, a.per_call_usd, ledger, total_cap_usd=a.max_total_usd, prior_usd=prior), ledger


def _label(a, ls, screens, use_source, guide, guide_text, guide_sha, keys, labels_dir) -> int:
    budget, ledger = make_budget(a, labels_dir)
    runners = {k: runner_for(k, a) for k in keys}
    for r in runners.values():
        rn.prepare_cwd(r.cwd)
    jobs, skipped, not_ok = [], 0, 0
    for key, runner in runners.items():
        lab_id = out_id(runner.labeller, a.variant)
        if ls.name == "sealed_real" and screens:
            check_sealed_pins(labels_dir, runner, pr.render(screens[0], guide_text))
        for scr in screens:
            p = pr.render(scr, guide_text, source=use_source)
            rp = tr.raw_path(labels_dir, ls.name, lab_id, scr.id)
            if resumable(c.read_json(rp), p, runner, a.retry_refused):
                skipped += 1
                if c.read_json(rp).get("status") != "ok":
                    not_ok += 1
                continue
            save_prefix(labels_dir, ls.name, p, "prefix")

            def job(runner=runner, scr=scr, p=p, rp=rp, lab_id=lab_id):
                if runner.costs_money and budget.stopped:
                    return "capped"
                out = rn.run_with_retry(runner, p.prefix, p.body, c.SCHEMA_PATH,
                                        lambda obj, scr=scr: tr.validate_reply(obj, scr), budget,
                                        {"set": ls.name, "labeller": lab_id, "screen": scr.id, "phase": "label"})
                if out["status"] == "capped" and not out["attempts"]:
                    return "capped"
                c.write_json(rp, raw_record("label", ls, scr, runner, guide_sha, p, out, labeller=lab_id))
                say(f"  {lab_id} {scr.id} {out['status']}"
                    + (f" US${out['cost_usd']:.3f}" if runner.costs_money else "")
                    + (f" ({out['error'][:120]})" if out.get("error") else ""))
                return out["status"]
            jobs.append((runner, job))
    say(f"label {ls.name}: {len(jobs)} calls to make, {skipped} already done; workers {a.workers} per labeller"
        + (f"; cap US${a.max_usd:.2f} this run" if "opus" in keys else "")
        + (f", US${a.max_total_usd:.2f} in all (US${budget.prior:.2f} spent before)" if a.max_total_usd is not None
           else "") + (f", US${a.per_call_usd:.2f} per call" if "opus" in keys else ""))
    results = run_jobs(jobs, a.workers, budget)
    summ = rn.ledger_summary(ledger)
    say(f"results: {json.dumps(tally(results))}; Opus spend this run US${budget.spent:.2f}"
        + (f" ({budget.estimated} calls with no reported cost charged at US${a.per_call_usd:.2f})"
           if budget.estimated else "")
        + f"; all runs US${summ['total_usd']:.2f}"
        + (f" (incl. {summ['orphan_launches']} launches with no charge recorded, at their reservation)"
           if summ["orphan_launches"] else "")
        + ("; SPEND CAP REACHED, launches stopped" if budget.stopped else ""))
    for r in runners.values():
        stray = [x.name for x in r.cwd.iterdir()] if r.cwd.exists() else []
        if stray:
            say(f"warning: {r.labeller}'s working directory {r.cwd} is no longer empty: {stray[:5]}")
    built = build(ls, labels_dir, guide, [out_id(c.LABELLERS[k], a.variant) for k in keys], final=False)
    if budget.stopped:
        return 3
    bad = not_ok + sum(1 for x in results if x != "ok")
    if bad:
        say(f"{bad} screen(s) without a valid reply this run (refused, invalid, error, cut off or capped); rerun to "
            "resume, --retry-refused to ask again after a refusal")
    return 1 if (bad or built) else 0


def build(ls, labels_dir: Path, guide: Path, labeller_ids, final=True) -> int:
    """Rebuild label files (and with final the key) from the raw replies and run check.py. Returns 1 on check errors,
    on raw replies of one labeller that mix prompts (one prefix and prompt version per set and labeller), and, with
    final, when the key is incomplete."""
    bad = 0
    for lab_id in labeller_ids:
        counts = tr.build_label_file(ls, lab_id, labels_dir, guide)
        chk = tr.check_label_file(Path(counts["file"]), ls, guide)
        incomplete = [e for e in chk["errors"] if "unlabelled" in e or "no done record" in e]
        real = [e for e in chk["errors"] if e not in incomplete]
        say(f"{counts['file']}: {counts['ok']} screens; check.py {chk['complete']} / {chk['screens']} complete, "
            f"{len(real)} errors" + (f" (first: {real[0]})" if real else ""))
        bad += len(real)
        prompts = prompt_pins(labels_dir, ls.name, lab_id)
        if len(prompts) > 1:
            say(f"MIXED PROMPTS: {lab_id} on {ls.name} has valid replies to {len(prompts)} different prompt prefixes or "
                f"versions {sorted((v, str(h)[:12]) for v, h in prompts)}; relabel the older screens (--retry-refused "
                "is not needed: a changed prompt is never resumed)")
            bad += 1
    if final:
        fc = adj.build_final(ls, labels_dir, guide)
        chk = tr.check_label_file(Path(fc["file"]), ls, guide)
        real = [e for e in chk["errors"] if "unlabelled" not in e and "no done record" not in e]
        say(f"{fc['file']}: {fc['screens']} screens ({fc['screens_skipped']} without both labels, "
            f"{fc['pending_adjudication']} awaiting a second pass, {len(fc['second_pass_refused'])} with a refused "
            f"second pass); sources {json.dumps(fc['sources'])}; check.py {len(real)} errors"
            + (f" (first: {real[0]})" if real else ""))
        if fc["screens_skipped"] or fc["pending_adjudication"]:
            miss = "; ".join(f"{sid} {json.dumps(st)}" for sid, st in list(fc["missing"].items())[:5])
            say(f"INCOMPLETE KEY: {fc['screens_skipped']} screens lack a valid label from both models"
                + (f" ({miss}{' ...' if len(fc['missing']) > 5 else ''})" if miss else "")
                + f" and {fc['pending_adjudication']} await a second pass"
                + (f" ({', '.join(fc['pending'][:5])})" if fc["pending"] else "")
                + ". The key must not be frozen or disclosed until it is complete (SPEC.md §5a).")
            bad += 1
        bad += len(real)
    return 1 if bad else 0


def prompt_pins(labels_dir: Path, set_name: str, labeller: str) -> set:
    out = set()
    d = tr.raw_dir(labels_dir, set_name, labeller)
    for f in sorted(d.glob("*.json")) if d.is_dir() else []:
        raw = c.read_json(f)
        if raw and raw.get("status") == "ok":
            out.add((raw.get("prompt_version"), raw.get("prefix_sha256")))
    return out


# ---------------------------------------------------------------- adjudicate


def cmd_adjudicate(a) -> int:
    labels_dir = Path(a.labels_dir)
    ls = open_set(a.set, Path(a.datasets_dir), a.registry_root)
    guide = Path(a.guide)
    guide_text, guide_sha = guide.read_text(encoding="utf-8"), lk.sha256_file(guide)
    r1 = adj.round1(ls, labels_dir)
    keys = labeller_keys(a.labeller)
    work = []
    for scr in ls.screens:
        if scr.id not in r1:
            continue
        disp = adj.screen_disputes(r1[scr.id])
        if disp["adjudicate"]:
            p, order = adj.adjudication_prompt(scr, guide_text, r1[scr.id], disp)
            work.append((scr, disp, p, order))
    say(f"adjudicate {ls.name}: {len(r1)} screens labelled by both, {len(work)} with disputed nodes "
        f"({sum(len(d['adjudicate']) for _, d, _, _ in work)} nodes)")
    if a.dry_run:
        out = scratch_dir(a.out_dir, labels_dir) / ls.name / "adjudicate"
        out.mkdir(parents=True, exist_ok=True)
        for scr, _, p, _ in work:
            (out / f"{scr.id}.prompt.txt").write_text(p.full, encoding="utf-8")
        say(f"dry run: {len(work)} adjudication prompts in {out}; nothing was called")
        return 0
    if work and "opus" in keys and a.max_usd is None:
        raise Refused("refusing: --max-usd is required when Opus adjudicates")
    lock = take_lock(labels_dir)
    try:
        budget, ledger = make_budget(a, labels_dir)
        runners = {k: runner_for(k, a) for k in keys} if work else {}
        for r in runners.values():
            rn.prepare_cwd(r.cwd)
        jobs, skipped, not_ok = [], 0, 0
        for key, runner in runners.items():
            for scr, disp, p, order in work:
                rp = adj.adj_raw_path(labels_dir, ls.name, runner.labeller, scr.id)
                raw = c.read_json(rp)
                if resumable(raw, p, runner, a.retry_refused):
                    skipped += 1
                    not_ok += raw.get("status") != "ok"
                    continue
                save_prefix(labels_dir, ls.name, p, "adjudicate-prefix")
                extra = {"disputed": disp["adjudicate"], "reasons": {str(k): v for k, v in disp["reasons"].items()},
                         "order": order, "round1_sha256": adj.round1_sha(r1[scr.id])}

                def job(runner=runner, scr=scr, p=p, rp=rp, disp=disp, extra=extra):
                    if runner.costs_money and budget.stopped:
                        return "capped"
                    out = rn.run_with_retry(
                        runner, p.prefix, p.body, c.ADJ_SCHEMA_PATH,
                        lambda obj, scr=scr, d=disp: adj.validate_adjudication(obj, scr, d["adjudicate"]), budget,
                        {"set": ls.name, "labeller": runner.labeller, "screen": scr.id, "phase": "adjudicate"})
                    if out["status"] == "capped" and not out["attempts"]:
                        return "capped"
                    c.write_json(rp, raw_record("adjudicate", ls, scr, runner, guide_sha, p, out, extra))
                    say(f"  adjudicate {runner.labeller} {scr.id} {out['status']}"
                        + (f" US${out['cost_usd']:.3f}" if runner.costs_money else ""))
                    return out["status"]
                jobs.append((runner, job))
        results = run_jobs(jobs, a.workers, budget)
        say(f"results: {json.dumps(tally(results))} ({skipped} already done); Opus spend this run "
            f"US${budget.spent:.2f}; all runs US${rn.ledger_total(ledger):.2f}"
            + ("; SPEND CAP REACHED, launches stopped" if budget.stopped else ""))
        built = build(ls, labels_dir, guide, [], final=True)
    finally:
        release_lock(lock)
    if budget.stopped:
        return 3
    return 1 if (built or not_ok or any(x != "ok" for x in results)) else 0


# ---------------------------------------------------------------- report, estimate, build


def cmd_report(a) -> int:
    ls = open_set(a.set, Path(a.datasets_dir), a.registry_root)
    res, md, jp, mp = agree.write_report(ls, Path(a.labels_dir))
    print(md)
    say(f"written {jp} and {mp}")
    return 0


def cmd_estimate(a) -> int:
    sets = ["all"] if a.set == "all" else [a.set]
    e = est.estimate(sets, Path(a.datasets_dir), Path(a.guide), pilot=pilot_screens)
    print(json.dumps(e, indent=1) if a.json else est.markdown(e))
    return 0


def cmd_build(a) -> int:
    ls = open_set(a.set, Path(a.datasets_dir), a.registry_root)
    labels_dir = Path(a.labels_dir)
    ids = [c.LABELLERS[k] for k in ("opus", "codex")]
    ids += [agree.variant_id(x, v) for x in list(ids) for v in agree.VARIANTS
            if tr.raw_dir(labels_dir, ls.name, agree.variant_id(x, v)).is_dir()]
    return build(ls, labels_dir, Path(a.guide), ids, final=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, calls=False):
        p.add_argument("--set", required=True)
        p.add_argument("--datasets-dir", default=str(c.DATASETS_DIR))
        p.add_argument("--labels-dir", default=str(c.LABELS_DIR))
        p.add_argument("--guide", default=str(c.GUIDE))
        p.add_argument("--registry-root", type=Path, default=c.SLOP_DIR, help=argparse.SUPPRESS)  # tests only
        if calls:
            p.add_argument("--labeller", default="both", choices=["opus", "codex", "both"])
            p.add_argument("--workers", type=int, default=3, help="parallel calls per labeller (default 3)")
            p.add_argument("--max-usd", type=float, help="hard cap on this run's Opus spend (required for Opus)")
            p.add_argument("--max-total-usd", type=float,
                           help="hard cap on all Opus spend in labels/raw/spend.jsonl (every set, phase and run)")
            p.add_argument("--per-call-usd", type=float, default=2.0,
                           help="claude --max-budget-usd per call, and the charge for a call with no reported cost")
            p.add_argument("--dry-run", action="store_true", help="render prompts to --out-dir and call nothing")
            p.add_argument("--out-dir", help="dry-run output (default: labels/dryrun/, gitignored)")
            p.add_argument("--retry-refused", action="store_true")
            p.add_argument("--claude-bin")
            p.add_argument("--codex-bin")
            p.add_argument("--cwd-root", help="parent of the labellers' empty working directories (default: "
                                              "<tmp>/slop-modellabel)")
            p.add_argument("--timeout", type=float, default=3600.0, help="seconds per call")

    pl = sub.add_parser("label", help="label a set with one or both models")
    common(pl, calls=True)
    pl.add_argument("--screens", nargs="+", help="only these screen ids")
    pl.add_argument("--first", type=int, help="only the first N screens in set order (N >= 1)")
    pl.add_argument("--pilot", action="store_true",
                    help="the 3 pilot screens: labelable node counts nearest 11, 22 (the sealed mean) and the set's "
                         "largest (pilot_screens)")
    pl.add_argument("--variant", choices=agree.VARIANTS,
                    help="a check on the key, written as <labeller>-<variant> and never keyed: retest (the same "
                         "prompt again, 10%% of the set by hash unless --screens/--first/--pilot) or nosource (the "
                         "screen without its source file)")
    pa = sub.add_parser("adjudicate", help="second pass on disputed nodes, then write <set>.final.jsonl")
    common(pa, calls=True)
    common(sub.add_parser("report", help="agreement report (markdown and JSON in labels/)"))
    common(sub.add_parser("build", help="rebuild label files and the final key from the raw replies"))
    pe = sub.add_parser("estimate", help="cost estimate from rendered prompts (calls nothing)")
    pe.add_argument("--set", required=True, help="a set name or all")
    pe.add_argument("--datasets-dir", default=str(c.DATASETS_DIR))
    pe.add_argument("--guide", default=str(c.GUIDE))
    pe.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    try:
        return {"label": cmd_label, "adjudicate": cmd_adjudicate, "report": cmd_report, "estimate": cmd_estimate,
                "build": cmd_build}[a.cmd](a)
    except Refused as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
