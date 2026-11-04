"""v0.3 rendering audits (spec v0.3 §7.3). **Dev only**; harness only (reads oracle records).

    report = run_dev_audits(v03_dir="datasets/v0_3", v02_dir="datasets/v0_2")

1. **Revisit evidence.** A REVERSION event (``OracleRecord.event_effects``) *has evidence* when one of
   its rendered ``h_new`` ids (``h0`` excluded: it only says "now empty") equals an ``h_old`` or
   ``h_new`` rendered on an earlier event of the window. Per event, the window is the 8-event window
   ending at the event (positions i-7..i-1). For each OSCILLATING-labelled window ending at t, its
   reversions (positions t-7..t) all have evidence counted within that window (t-7 <= j < i). Hunk ids
   are parsed from the rendered text, so ids hidden behind ``@ TRUNC k`` count as absent.
2. **Marker audit.** Among MUTATE events with a behaviour change (effects PROGRESS, RESOLVES_EPISODE,
   NONE, REVERSION; REDUNDANT excluded), a numpy logistic regression on rendered features (hunk lines,
   changed paths, tool one-hot, TRUNC present) predicts PROGRESS/RESOLVES vs NONE/REVERSION, 5-fold CV
   grouped by session; macro-F1 against the constant (majority) prior.
3. **Hashed-bucket collisions.** Distinct surface words that share a hashed id within one session.
4. **Length.** Tokens per session, and tokens per block by tool family.
5. **Leakage scan.** No generator marker strings in rendered text; every body line is a hunk-id line,
   an error line or the workspace line; no line of file text from ``changed`` appears in the block.

Also reported: reserved field words seen outside a field position (not followed by ``=``).
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from ..datasets.build import iter_sessions
from ..fingerprint import FAMILY_CATEGORY
from ..schema import EVENTS_PER_WINDOW, LatentEffect, OperationCategory, OracleLabel
from . import renderer, tokenizer
from .build_raw import iter_raw

__all__ = ["AUDIT_SPLIT", "LEAKAGE_MARKERS", "run_dev_audits", "logistic_cv", "macro_f1", "hunk_ids"]

AUDIT_SPLIT = "dev"
LEAKAGE_MARKERS: tuple[str, ...] = ("cleanup pass", "revised", "-ok", "enabled=", "investigation", "explanation",
                                    "+1)")
_HUNK = re.compile(r"^  @ (\S+) (h0|h[0-9a-f]{10}) > (h0|h[0-9a-f]{10})$")
_TRUNC_HUNK = re.compile(r"^  @ TRUNC [0-9]+$")
_ERROR = re.compile(r"^  ! .* at .*:.* \((pid [0-9]+|0x[0-9a-f]{8})\)$|^  ! .* TRUNC$")
_WORKSPACE = "  [ workspace untracked]"
_BEHAVIOUR = {LatentEffect.PROGRESS, LatentEffect.RESOLVES_EPISODE, LatentEffect.NONE, LatentEffect.REVERSION}
_POSITIVE = {LatentEffect.PROGRESS, LatentEffect.RESOLVES_EPISODE}
_FIELD_WORDS = ("path", "lines", "scope", "query", "limit", "bytes", "paths", "suite", "tests", "target", "body",
                "req", "resource", "backoff", "action", "in", "out")
_MUTATE_TOOLS = tuple(sorted(f for f, c in FAMILY_CATEGORY.items() if c is OperationCategory.MUTATE))
_N_FOLDS = 5


def hunk_ids(block: str) -> list[tuple[str, str, str]]:
    """``(path, h_old, h_new)`` of each rendered hunk line (TRUNC lines skipped)."""
    out = []
    for ln in block.split("\n")[1:]:
        m = _HUNK.match(ln)
        if m:
            out.append(m.groups())
    return out


def _fold(session_id: str) -> int:
    return int(hashlib.sha256(f"v03-audit-fold:{session_id}".encode()).hexdigest()[:8], 16) % _N_FOLDS


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    f1s = []
    for c in (0, 1):
        tp = int(np.sum((p == c) & (y == c)))
        fp = int(np.sum((p == c) & (y != c)))
        fn = int(np.sum((p != c) & (y == c)))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def _fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 50) -> np.ndarray:
    """Newton-Raphson logistic regression with an intercept column and a small L2 penalty."""
    xb = np.hstack([x, np.ones((len(x), 1))])
    w = np.zeros(xb.shape[1])
    reg = l2 * np.eye(xb.shape[1])
    reg[-1, -1] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(xb @ w, -30, 30)))
        g = xb.T @ (p - y) + reg @ w
        h = (xb * (p * (1 - p))[:, None]).T @ xb + reg + 1e-9 * np.eye(xb.shape[1])
        step = np.linalg.solve(h, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):  # average ranks over ties
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def logistic_cv(x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> dict[str, Any]:
    """Grouped K-fold logistic regression: out-of-fold macro-F1 (threshold 0.5 and the training-fold base
    rate), AUC, and the constant-prior macro-F1 (always the training-fold majority class)."""
    prob = np.zeros(len(y))
    pred_half = np.zeros(len(y), dtype=np.int64)
    pred_rate = np.zeros(len(y), dtype=np.int64)
    pred_prior = np.zeros(len(y), dtype=np.int64)
    weights = []
    for k in range(_N_FOLDS):
        te, tr = folds == k, folds != k
        mu, sd = x[tr].mean(0), x[tr].std(0)
        sd[sd == 0] = 1.0
        w = _fit_logistic((x[tr] - mu) / sd, y[tr])
        weights.append(w.round(4).tolist())
        z = np.hstack([(x[te] - mu) / sd, np.ones((int(te.sum()), 1))]) @ w
        prob[te] = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        rate = float(y[tr].mean())
        pred_half[te] = prob[te] >= 0.5
        pred_rate[te] = prob[te] >= rate
        pred_prior[te] = int(rate >= 0.5)
    return {
        "n": int(len(y)), "positive_rate": round(float(y.mean()), 4),
        "macro_f1_threshold_0_5": round(macro_f1(y, pred_half), 4),
        "macro_f1_threshold_base_rate": round(macro_f1(y, pred_rate), 4),
        "macro_f1_constant_prior": round(macro_f1(y, pred_prior), 4),
        "auc": round(_auc(y, prob), 4),
        "fold_weights_standardised": weights,
    }


def _dist(values) -> dict[str, float | int]:
    a = np.asarray(list(values), dtype=np.int64)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": round(float(a.mean()), 2), "p50": round(float(np.percentile(a, 50)), 2),
            "p90": round(float(np.percentile(a, 90)), 2), "p99": round(float(np.percentile(a, 99)), 2),
            "max": int(a.max())}


def _evidence(i: int, lo: int, ids: list[list[tuple[str, str, str]]]) -> bool:
    new = {h for _, _, h in ids[i] if h != "h0"}
    if not new:
        return False
    seen = {h for j in range(lo, i) for _, a, b in ids[j] for h in (a, b)}
    return bool(new & seen)


def run_dev_audits(v03_dir: str | Path, v02_dir: str | Path, *, split: str = AUDIT_SPLIT,
                   limit: int | None = None, progress=None) -> dict[str, Any]:
    if split != AUDIT_SPLIT:
        raise ValueError(f"v0.3 rendering audits run on {AUDIT_SPLIT!r} only, not {split!r}")
    say = progress or (lambda _m: None)
    W = EVENTS_PER_WINDOW

    rev_events = rev_with = 0
    rev_no_hunks = 0
    osc_windows = osc_with_rev = osc_all_evidence = osc_all_event_evidence = 0
    feats: list[list[float]] = []
    labels: list[int] = []
    folds: list[int] = []
    by_tool_hunks: dict[str, dict[str, list[int]]] = defaultdict(dict)
    marker_counts: Counter = Counter()
    marker_examples: dict[str, str] = {}
    bad_body = code_text_hits = 0
    bad_body_examples: list[str] = []
    sessions_with_collision = collision_pairs = 0
    words_in_collision = distinct_words_total = 0
    collision_examples: list[list[str]] = []
    reserved_value = Counter()
    session_tokens: list[int] = []
    family_tokens: dict[str, list[int]] = defaultdict(list)
    n_sessions = n_events = 0

    for n, ((sid, raw), (events, record)) in enumerate(zip(iter_raw(v03_dir, split), iter_sessions(v02_dir, split),
                                                          strict=True)):
        if limit is not None and n >= limit:
            break
        if sid != record.session_id or len(raw) != len(events) or len(record.event_effects) != len(events):
            raise ValueError(f"dev line {n + 1}: raw/events/oracle misaligned for {sid!r}")
        blocks = renderer.render_session([r.to_dict() for r in raw], sid)
        enc = tokenizer.encode_session(blocks)
        n_sessions += 1
        n_events += len(blocks)
        session_tokens.append(len(enc.tokens))
        prev = 0
        for r, e in zip(raw, enc.eoe_index):
            family_tokens[r.family].append(e - prev)
            prev = e
        ids = [hunk_ids(b) for b in blocks]
        effects = record.event_effects

        # (1) revisit evidence
        event_evidence: dict[int, bool] = {}
        for i, eff in enumerate(effects):
            if eff is LatentEffect.REVERSION:
                rev_events += 1
                rev_no_hunks += not ids[i]
                event_evidence[i] = _evidence(i, max(0, i - (W - 1)), ids)
                rev_with += event_evidence[i]
        pos = {e.sequence_index: k for k, e in enumerate(events)}
        for t_seq, lab in record.window_labels.items():
            if lab is not OracleLabel.OSCILLATING:
                continue
            t = pos[t_seq]
            osc_windows += 1
            lo = max(0, t - (W - 1))
            revs = [i for i in range(lo, t + 1) if effects[i] is LatentEffect.REVERSION]
            if revs:
                osc_with_rev += 1
                osc_all_evidence += all(_evidence(i, lo, ids) for i in revs)
                osc_all_event_evidence += all(event_evidence[i] for i in revs)

        # (2) marker audit features
        for i, (r, eff) in enumerate(zip(raw, effects)):
            if FAMILY_CATEGORY.get(r.family) is OperationCategory.MUTATE and eff in _BEHAVIOUR:
                body = blocks[i].split("\n")[1:]
                n_hunk_lines = sum(ln.startswith("  @ ") for ln in body)
                feats.append([float(n_hunk_lines), float(len(r.changed or [])),
                              *[float(r.family == t) for t in _MUTATE_TOOLS],
                              float(renderer.TRUNC in blocks[i])])
                labels.append(int(eff in _POSITIVE))
                cell = by_tool_hunks[r.family].setdefault(str(n_hunk_lines), [0, 0])
                cell[0] += 1
                cell[1] += int(eff in _POSITIVE)
                folds.append(_fold(sid))

        # (3) hashed-bucket collisions within the session
        by_id: dict[int, set[str]] = defaultdict(set)
        for tid, s in zip(enc.tokens, enc.surface):
            if tid >= tokenizer.HASH_BASE:
                by_id[tid].add(s)
        distinct_words_total += sum(len(v) for v in by_id.values())
        clashes = [sorted(v) for v in by_id.values() if len(v) > 1]
        if clashes:
            sessions_with_collision += 1
            collision_pairs += sum(len(v) * (len(v) - 1) // 2 for v in clashes)
            words_in_collision += sum(len(v) for v in clashes)
            if len(collision_examples) < 10:
                collision_examples.extend(clashes[: 10 - len(collision_examples)])

        # (5) leakage scan, and reserved field words outside a field position
        for r, blk in zip(raw, blocks):
            for m in LEAKAGE_MARKERS:
                if m in blk:
                    marker_counts[m] += 1
                    marker_examples.setdefault(m, blk[:240])
            for ln in blk.split("\n")[1:]:
                if not (_HUNK.match(ln) or _TRUNC_HUNK.match(ln) or _ERROR.match(ln) or ln == _WORKSPACE):
                    bad_body += 1
                    if len(bad_body_examples) < 5:
                        bad_body_examples.append(ln[:200])
            for ch in r.changed or []:
                for text in (ch["before"], ch["after"]):
                    for code_line in (text or "").split("\n"):
                        code_line = code_line.strip()
                        if len(code_line) >= 8 and code_line in blk:
                            code_text_hits += 1
            pairs = tokenizer.tokenize_block(blk)
            for k, (_tid, s) in enumerate(pairs):
                if s in _FIELD_WORDS and (k + 1 >= len(pairs) or pairs[k + 1][1] != "="):
                    reserved_value[s] += 1
        if n_sessions % 2000 == 0:
            say(f"audited {n_sessions} dev sessions")

    x = np.asarray(feats, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    marker = logistic_cv(x, y, np.asarray(folds))
    marker["features"] = ["hunk_lines", "changed_paths", *[f"tool={t}" for t in _MUTATE_TOOLS], "trunc_present"]
    marker["label"] = "1 = PROGRESS or RESOLVES_EPISODE, 0 = NONE or REVERSION (REDUNDANT excluded)"
    marker["folds"] = f"{_N_FOLDS}-fold grouped by session: sha256('v03-audit-fold:'+session_id) mod {_N_FOLDS}"
    no_tool = [0, 1, len(_MUTATE_TOOLS) + 2]  # hunk_lines, changed_paths, trunc_present
    marker["ablation_without_tool"] = {k: v for k, v in logistic_cv(x[:, no_tool], y, np.asarray(folds)).items()
                                       if k != "fold_weights_standardised"}
    marker["ablation_tool_only"] = {k: v for k, v in logistic_cv(x[:, 2:2 + len(_MUTATE_TOOLS)], y,
                                                                 np.asarray(folds)).items()
                                    if k != "fold_weights_standardised"}
    marker["counts_by_tool_and_hunk_lines"] = {
        "cell": "[events, positives]", **{t: dict(sorted(v.items(), key=lambda kv: int(kv[0])))
                                          for t, v in sorted(by_tool_hunks.items())}}

    return {
        "split": split,
        "sessions": n_sessions,
        "events": n_events,
        "renderer_source_sha256": hashlib.sha256(Path(renderer.__file__).read_bytes()).hexdigest(),
        "tokeniser_source_sha256": hashlib.sha256(Path(tokenizer.__file__).read_bytes()).hexdigest(),
        "revisit_evidence": {
            "definition": ("REVERSION event i has evidence if a rendered h_new (h0 excluded) of i equals an h_old or "
                           "h_new rendered on an earlier event j of the window. Per event: j in [i-7, i-1]. Per "
                           "OSCILLATING window ending at t, two readings: strict in-window (j in [t-7, i-1] for "
                           "every reversion i in [t-7, t]) and event-window (every reversion i in [t-7, t] has "
                           "per-event evidence)"),
            "reversion_events": rev_events,
            "reversion_events_with_evidence": rev_with,
            "fraction_reversion_events_with_evidence": round(rev_with / rev_events, 4) if rev_events else None,
            "reversion_events_without_hunk_lines": rev_no_hunks,
            "oscillating_windows": osc_windows,
            "oscillating_windows_with_reversions": osc_with_rev,
            "oscillating_windows_all_reversions_evidenced": osc_all_evidence,
            "fraction_oscillating_windows_all_evidenced": (round(osc_all_evidence / osc_with_rev, 4)
                                                           if osc_with_rev else None),
            "oscillating_windows_all_reversions_event_evidenced": osc_all_event_evidence,
            "fraction_oscillating_windows_all_event_evidenced": (round(osc_all_event_evidence / osc_with_rev, 4)
                                                                 if osc_with_rev else None),
        },
        "marker_audit": marker,
        "hashed_bucket_collisions": {
            "definition": "distinct surface words sharing a hashed id (>= 2048) within one session",
            "sessions_with_collision": sessions_with_collision,
            "fraction_sessions_with_collision": round(sessions_with_collision / n_sessions, 4) if n_sessions else None,
            "colliding_word_pairs": collision_pairs,
            "words_in_collisions": words_in_collision,
            "distinct_hashed_words_summed_over_sessions": distinct_words_total,
            "word_collision_rate": round(words_in_collision / distinct_words_total, 6) if distinct_words_total else None,
            "examples": collision_examples,
        },
        "length": {
            "tokens_per_session": _dist(session_tokens),
            "tokens_per_block_by_family": {f: _dist(v) for f, v in sorted(family_tokens.items())},
            "tokens_per_block_all": _dist(t for v in family_tokens.values() for t in v),
        },
        "leakage_scan": {
            "markers": list(LEAKAGE_MARKERS),
            "marker_counts": {m: marker_counts.get(m, 0) for m in LEAKAGE_MARKERS},
            "marker_examples": marker_examples,
            "non_grammar_body_lines": bad_body,
            "non_grammar_body_line_examples": bad_body_examples,
            "file_text_lines_found_in_blocks": code_text_hits,
            "clean": not marker_counts and bad_body == 0 and code_text_hits == 0,
        },
        "reserved_field_words_outside_field_position": dict(sorted(reserved_value.items())),
    }
