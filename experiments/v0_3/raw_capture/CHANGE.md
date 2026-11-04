# Harness change: v0.3 raw-operation capture (spec v0.3 §3, §7.1)

**Date:** 2026-11-05 · **Revision:** r1 schema (`changed`, eight-entry `dropped` vocabulary) · **Base commit:** 89801e5 (branch `m0-harness`; the engines and dataset builders are unchanged between 89801e5 and HEAD) · **Author:** harness implementer (Claude), not the weight author.

## What changed and why

Spec v0.3 feeds the network rendered raw agent logs instead of v0.2's encoded features. The renderer needs the concrete operation each event came from, which the canonical event only carries as fingerprints. Both generators therefore gain a **side channel** that records one `RawOp` (`src/agent_loop_lab/v03/rawop.py`) per emitted event. No emitted `CanonicalEvent`, `OracleRecord` or random draw changes.

This is a harness change to **both** generators, including Codex's independently written OOD generator (`synth_ood/`, commissioned under `experiments/commissions/ood_generator/`). The OOD edit is limited to recording values `perform()` already computes, plus a shallow copy of the workspace dict.

### r1 schema (spec v0.3 §3, DRAFT r1)

- `changed: [{path, before, after}]` replaces the earlier `prev_text`/`new_text` pair. It lists **every tracked file whose text differs between the start and the end of the operation, for any family**, sorted by path, with `before`/`after` null for an absent file. A no-op mutation records `[]`. So reverts and OOD package installs (an `http_call` that rewrites `vendor/cedar.artifact`) now carry their file changes, and the renderer shows hunk lines for them.
- `dropped` uses one eight-entry vocabulary for both engines: `digest_after digest_before failure input_tokens output_tokens validation_passed validation_suite validation_total` (`DROPPED_VOCAB`), mapped from each engine's own names by `STANDARD_DROP_MAP` / `OOD_DROP_MAP`.

### Standard engine (`src/agent_loop_lab/synth/engine.py`)

- New module-level helper `_changed(before, after)`: given `{path: text or None}` maps taken before and after an operation, it returns the `{path, before, after}` entries whose text differs, sorted by path. It reads only the two dicts.
- `Session._raw` (a list kept aligned with `_events`) is appended in `_commit` straight after the event is appended. It is built from `_commit`'s own arguments (`family`, `target`, `args`, `outcome`, `failure`, `validation`, `drop`) and from the emitted event (`input_tokens`, `output_tokens`, `timestamp`, so a dropped field stays null).
- `_commit` and `_finish_mutation` gain keyword-only `changed` and `extra` parameters. Subclasses (`_InstrumentedSession`, `_RampSession`) forward `**kwargs`, so they're unaffected.
- `_mutate` passes `changed=_changed({path: old_text if exists else None}, {path: new_text if new_exists else None})`, built from the `old_text`/`new_text` it already computes. `new_exists = path in w.files` replaces the inline membership test it was computed from, and that test has no side effects. The missing-file error paths go straight to `_commit` and record `changed=[]`.
- `revert` reads `w.text(p)` (or None for an absent file) for each reverted path immediately before and after the restore loop and passes `changed=_changed(pre_texts, post_texts)`. `w.text` is a pure read of the rendered file text; no rng is touched.
- `extra` is `{"failing_test": id}` on a failing `run_tests` (the id the agent observes, `OpResult.failing_test`) and `{"status": "ready"|"pending"}` on `poll_status`. Nothing latent goes in: there is no `desc`, no fault or fact ids, and none of `first[2:]`.
- `finish()` slices `raw = self._raw[m:]` exactly as it slices events, keeps it for the new `Session.raw_ops` accessor, and hands `(session_id, events, raw)` to any open capture sink (`v03.rawop.emit_finished`). With no sink open, this does nothing.
- `fork()`'s deepcopy copies `_raw` unchanged. RawOps carry no session identity, so nothing needs re-stamping (`counterfactual.py` is untouched).
- `error_class`/`error_location` are recorded exactly when `failure_fingerprint` is computed (an ERROR with a failure tuple), even if the canonical fingerprint is then dropped.

### OOD engine (`src/agent_loop_lab/synth_ood/engine.py`, Codex-authored)

- `Engine.raw` (aligned with `events`).
- At the top of `perform()`, before the operation chain, `pre_files = dict(self.repo.files)` takes a shallow snapshot of the workspace (a dict copy of immutable strings; no rng, no state change). After the existing `self.events.append(...)`, a `RawOp` is built from locals already computed there, from the appended event, and with `changed` listing every path whose text differs between `pre_files` and `self.repo.files`, sorted by path. That point is after the cost and clock rng draws, and no rng call is added or moved. Because the snapshot covers the whole repository, file changes made by package installs and uninstalls (`http_call` to the resolver) are recorded as well as `write_file`.
- `error_class`/`error_location` are recorded only on ERROR events, where `failure_fingerprint` receives them. The success paths also set `error_location`, and NEUTRAL polls set `error_class="LeaseNotReady"`, but neither reaches the fingerprint. Poll status goes in `extra`.
- `finish()` calls `emit_finished(...)` before returning the unchanged tuple. `counterfactuals.py` is untouched. Its deepcopy branches copy `raw`, and because the two branches share a session id, the sink pairs raw with events by list identity.

### Dropped-observation vocabulary (both engines)

| `DROPPED_VOCAB` entry | standard `drop=` | OOD `dropped=` |
|---|---|---|
| `input_tokens` | tokens, input_tokens | tokens (the OOD engine nulls `input_tokens` only) |
| `output_tokens` | tokens, output_tokens | — |
| `digest_before` | digests, digest_before | workspace |
| `digest_after` | digests, digest_after | workspace |
| `failure` | failure | failure (ERROR events only) |
| `validation_suite` | validation_suite | validation_suite |
| `validation_passed` | validation_counts | validation_passed |
| `validation_total` | validation_counts | validation_total |

Only deliberate, per-event withholding is recorded. The oracle's baseline-propagation marks are not. An OOD `dropped=` that withholds nothing (a `failure` drop on a passing run, a validation drop on an event without validation) records nothing.

### Dataset builders (`src/agent_loop_lab/datasets/build.py`, `ood.py`)

`build()` and `build_ood()` gain an optional, observe-only `on_session` callback. It is invoked right after each session's lines are written, and the v0.3 builder (`src/agent_loop_lab/v03/build_raw.py`) uses it to stream raw lines in v0.2 file order. With the default `None`, nothing changes.

### Alignment checks in the v0.3 builder (`src/agent_loop_lab/v03/build_raw.py`)

Every captured RawOp is checked against its canonical event (`alignment_errors`); any disagreement aborts the build and deletes the raw output. The checks: family, outcome, emitted tokens and timestamp agree; each `DROPPED_VOCAB` entry is set exactly when the field it withholds is null on the event; `validation` is present exactly when the event produced one, and every observed count equals the true value; `failure` is dropped only on ERROR events without a fingerprint; `changed` is sorted by path, unique, each entry a real change, and non-empty exactly when the event's two workspace digests (when both observed) differ.

### Side effect on v0.2 manifests

`generator_source_sha256` in v0.2's `MANIFEST.json` / `OOD_MANIFEST.json` hashes the generator sources, so a rebuild now reports a different value there. The committed v0.2 manifests are untouched. The acceptance check compares the **data file** hashes only.

## Acceptance (spec v0.3 §7.1)

`scripts/build_v03_raw.py` re-runs both v0.2 builds unchanged, from the same seeds and sizes, into a temporary directory with the capture sink open. It then compares the sha256 of every regenerated canonical file with the committed v0.2 manifests. A mismatch deletes the raw output and exits 1. The test suite (`uv run pytest -q`, including every pre-existing generator determinism and contract test) is the second check.

## Full `git diff` of both engines (against 89801e5)

```diff
diff --git a/src/agent_loop_lab/synth/engine.py b/src/agent_loop_lab/synth/engine.py
index dd9d8fc..a72551a 100644
--- a/src/agent_loop_lab/synth/engine.py
+++ b/src/agent_loop_lab/synth/engine.py
@@ -44,6 +44,7 @@ from typing import Any, Iterable, Mapping, Sequence
 
 from ..fingerprint import FAMILY_CATEGORY, failure_fingerprint, intent_fingerprint, normalise_path, workspace_digest
 from ..oracle_config import OracleConfig, load_oracle_config
+from ..v03.rawop import RawOp, emit_finished, normalise_standard_drop
 from ..schema import (
     EVENTS_PER_WINDOW,
     CanonicalEvent,
@@ -131,6 +132,12 @@ class Change:
         return self.kind
 
 
+def _changed(before: Mapping[str, str | None], after: Mapping[str, str | None]) -> list[dict]:
+    """v0.3 raw capture: files whose text differs, sorted by path (None = absent)."""
+    return [{"path": p, "before": before.get(p), "after": after.get(p)}
+            for p in sorted(set(before) | set(after)) if before.get(p) != after.get(p)]
+
+
 def _fid(fault: Fault | str) -> str:
     return fault.id if isinstance(fault, Fault) else str(fault)
 
@@ -305,6 +312,8 @@ class Session:
         self._events: list[CanonicalEvent] = []
         self._steps: list[LatentStep] = []
         self._log: list[_LogEntry] = []
+        self._raw: list[RawOp] = []  # v0.3 side channel, aligned with _events (no latent data)
+        self._finished_raw: list[RawOp] | None = None
         self._K: dict[str, int] = {}
         self._seen_content: set[tuple] = set()
         self._seen_obs: set[tuple] = set()
@@ -340,6 +349,12 @@ class Session:
     def steps(self) -> list[LatentStep]:
         return list(self._steps)
 
+    @property
+    def raw_ops(self) -> list[RawOp]:
+        """v0.3 raw operation records, aligned with :attr:`events` (after
+        :meth:`finish`, with the returned events, i.e. after any truncation)."""
+        return list(self._finished_raw if self._finished_raw is not None else self._raw)
+
     @property
     def episode(self) -> tuple[str, int] | None:
         """``(kind, attempts)`` of the open episode, else None."""
@@ -494,6 +509,7 @@ class Session:
             raise ValueError("to must be 'last' or 'initial'")
         w = self.world
         before = self._mut_snapshot()
+        pre_texts = {p: (w.text(p) if p in w.files else None) for p in paths}  # v0.3 raw capture (no rng)
         for p in paths:
             if to == "last":
                 stack = self._stacks.get(p)
@@ -505,8 +521,10 @@ class Session:
                 self._restore(p, self._initial.get(p))
         target = {p: w.text(p) for p in paths if p in w.files}
         revision = workspace_digest(target) if target else _sha("deleted:" + ",".join(paths))
+        post_texts = {p: (w.text(p) if p in w.files else None) for p in paths}
         return self._finish_mutation("revert", list(paths), {"revision": revision}, ("revert", tuple(paths), revision),
-                                     before, f"revert {','.join(paths)} to {to}", cost, drop)
+                                     before, f"revert {','.join(paths)} to {to}", cost, drop,
+                                     changed=_changed(pre_texts, post_texts))
 
     def _push(self, path: str) -> None:
         fm = self.world.files.get(path)
@@ -557,7 +575,8 @@ class Session:
                 self._push(path)
                 w.set_states(path, new_states)
             desc = change.describe()
-        new_text = w.text(path) if path in w.files else ""
+        new_exists = path in w.files
+        new_text = w.text(path) if new_exists else ""
         if family == "edit_file":
             patch = "\n".join(difflib.unified_diff(old_text.split("\n"), new_text.split("\n"), f"a/{path}", f"b/{path}",
                                                    lineterm="", n=0)) or f"--- a/{path}\n+++ b/{path}\n@@ no change @@"
@@ -567,7 +586,9 @@ class Session:
             args = {"content": new_text}
             key = (family, path, _sha(new_text))
         return self._finish_mutation(family, path, args, key, before, f"{family} {path} {desc}", cost, drop,
-                                     hint=0.6 + abs(len(new_text) - len(old_text)) / 2000.0)
+                                     hint=0.6 + abs(len(new_text) - len(old_text)) / 2000.0,
+                                     changed=_changed({path: old_text if exists else None},
+                                                      {path: new_text if new_exists else None}))
 
     def _plan_change(self, path: str, change: Change, block: str | None) -> dict:
         w = self.world
@@ -598,7 +619,8 @@ class Session:
         raise ValueError(f"unknown change kind {change.kind!r}")
 
     def _finish_mutation(self, family: str, target: Any, args: dict, key: tuple, before: tuple, desc: str,
-                         cost: float | None, drop: Sequence[str], hint: float = 1.0) -> OpResult:
+                         cost: float | None, drop: Sequence[str], hint: float = 1.0, *,
+                         changed: list[dict] | None = None) -> OpResult:
         w = self.world
         ws_b, beh_b, causes_b, _ = before
         beh_changed = w.behaviour_key() != beh_b
@@ -606,7 +628,7 @@ class Session:
         return self._commit(
             family, target, args, key=key, rel_kind="text", outcome=Outcome.SUCCESS, ws_before=ws_b,
             redundant=not beh_changed, beh_changed=beh_changed, repaired=repaired, desc=desc, cost=cost, drop=drop,
-            hint=hint,
+            hint=hint, changed=changed,
         )
 
     # -- validation -------------------------------------------------------
@@ -628,6 +650,7 @@ class Session:
             redundant=redundant, validation=(suite, passed, total), obs=("val", vkey, passed, total, first and first[0]),
             desc=f"test {suite}{'' if sel is None else f' [{len(sel)}]'} -> {passed}/{total}",
             hint=0.6 + total / 40.0 + (0.8 if first else 0.0), cost=cost, drop=drop,
+            extra={"failing_test": first[0]} if first else None,
         )
         return replace(res, failing_test=first[0] if first else None, error_class=first[1] if first else None,
                        passed=passed, total=total)
@@ -712,7 +735,7 @@ class Session:
             "poll_status", resource, args, key=("poll_status", resource), rel_kind=f"res:{resource}",
             outcome=Outcome.SUCCESS if ready else Outcome.NEUTRAL, poll=True, redundant=redundant,
             obs=("poll", resource, status), desc=f"poll {resource} -> {status}", hint=0.5, cost=cost, drop=drop,
-            seconds=1.0 + (backoff or 0),
+            seconds=1.0 + (backoff or 0), extra={"status": status},
         )
 
     # -- control ----------------------------------------------------------
@@ -777,6 +800,8 @@ class Session:
         cost: float | None = None,
         drop: Sequence[str] = (),
         seconds: float | None = None,
+        changed: list[dict] | None = None,
+        extra: Mapping[str, Any] | None = None,
     ) -> OpResult:
         if self._finished:
             raise RuntimeError("session already finished")
@@ -885,6 +910,16 @@ class Session:
         self._events.append(event)
         self._steps.append(step)
         self._log.append(_LogEntry(desc, validation[0] if validation else None, drop))
+        # v0.3 side channel: the concrete operation as executed and as emitted.
+        # Built from values computed above; no world or rng access.
+        err = failure is not None and outcome is Outcome.ERROR
+        self._raw.append(RawOp.make(
+            family=family, target=target, args=args, outcome=outcome,
+            error_class=failure[1] if err else None, error_location=failure[2] if err else None,
+            validation=validation, changed=changed, extra=extra,
+            input_tokens=event.input_tokens, output_tokens=event.output_tokens, timestamp=event.timestamp,
+            dropped=normalise_standard_drop(drop),
+        ))
         return OpResult(event, step)
 
     def _build_event(self, idx: int, family: str, target: Any, args: dict, outcome: Outcome,
@@ -933,7 +968,8 @@ class Session:
 
         The emitted prefix is re-stamped with the new ``session_id``/``event_id``;
         everything else, including the latent state and future random draws, is
-        identical until the two sessions diverge.
+        identical until the two sessions diverge. The v0.3 raw ops are copied
+        unchanged by the deepcopy: they carry no session identity.
         """
         new = copy.deepcopy(self)
         new.session_id = session_id
@@ -966,6 +1002,7 @@ class Session:
         events = self._events[m:]
         steps = self._steps[m:]
         log = self._log[m:]
+        raw = self._raw[m:]
         if m:
             events = [replace(e, provenance=replace(e.provenance, history_complete=False)) for e in events]
             p0 = self._steps[m - 1].progress_after
@@ -1018,4 +1055,6 @@ class Session:
             latent_trace=trace,
         )
         self._finished = True
+        self._finished_raw = raw
+        emit_finished(self.session_id, events, raw)
         return events, record
diff --git a/src/agent_loop_lab/synth_ood/engine.py b/src/agent_loop_lab/synth_ood/engine.py
index e76db89..a8339a9 100644
--- a/src/agent_loop_lab/synth_ood/engine.py
+++ b/src/agent_loop_lab/synth_ood/engine.py
@@ -18,6 +18,7 @@ from ..oracle_config import OracleConfig
 from ..schema import CanonicalEvent, LatentEffect as E, OracleRecord, Outcome, Provenance
 from ..synth.labeller import label_session
 from ..synth.latent import LatentStep, latent_trace
+from ..v03.rawop import RawOp, emit_finished, normalise_ood_drop
 
 
 class PackageResolver:
@@ -123,6 +124,7 @@ never supplied by scenario scripts. No encoder is imported here.
         self.service_calls: dict[str, int] = {}
         self.events: list[CanonicalEvent] = []
         self.steps: list[LatentStep] = []
+        self.raw: list[RawOp] = []  # v0.3 harness side channel, aligned with events
         self.states = [self.repo.digest()]
         self.last_failure: tuple | None = None
         self.episode_key: tuple | None = None
@@ -152,6 +154,7 @@ never supplied by scenario scripts. No encoder is imported here.
         # Relevant state for retry equality is behavioural, so cosmetic edits
         # do not make a deterministic failing validation a fresh attempt.
         key = (family, repr(target), repr(sorted(args.items())), behaviour)
+        pre_files = dict(self.repo.files)  # v0.3 raw capture: shallow snapshot, no rng
         if family == "read_file":
             text = self.repo.files[target]
             read_key = (target, text)
@@ -314,6 +317,19 @@ never supplied by scenario scripts. No encoder is imported here.
             output_tokens=cost-input_tokens,
             provenance=Provenance(source="simulator:ood-0.1"),
         ))
+        # v0.3 harness side channel (no rng, no state change): the operation as executed and emitted.
+        ev, is_error = self.events[-1], outcome is Outcome.ERROR
+        self.raw.append(RawOp.make(
+            family=family, target=target, args=args, outcome=outcome,
+            error_class=error_class if is_error else None, error_location=error_location if is_error else None,
+            validation=(suite, passed, total) if produced else None,
+            changed=[{"path": p, "before": pre_files.get(p), "after": self.repo.files.get(p)}
+                     for p in sorted(set(pre_files) | set(self.repo.files))
+                     if pre_files.get(p) != self.repo.files.get(p)],
+            extra={"status": "pending" if outcome is Outcome.NEUTRAL else "ready"} if family == "poll_status" else None,
+            input_tokens=ev.input_tokens, output_tokens=ev.output_tokens, timestamp=ev.timestamp,
+            dropped=normalise_ood_drop(dropped, is_error=is_error, produced=produced),
+        ))
         self.states.append(after)
 
     def read(self, path: str) -> None:
@@ -326,6 +342,7 @@ never supplied by scenario scripts. No encoder is imported here.
         self.perform("run_tests", self.repo.suite, {"tests": ["replica-contract"]}, dropped=dropped)
 
     def finish(self, name: str) -> tuple[list[CanonicalEvent], OracleRecord]:
+        emit_finished(self.session_id, self.events, self.raw)  # v0.3 raw capture (no-op unless a sink is open)
         return self.events, OracleRecord(
             session_id=self.session_id, generator_version="ood-0.1", seed=self.seed,
             scenario_family=f"ood:{name}", event_effects=tuple(s.effect for s in self.steps),
```

## `git diff` of the dataset builders (observe-only hook)

```diff
diff --git a/src/agent_loop_lab/datasets/build.py b/src/agent_loop_lab/datasets/build.py
index 9a15f3b..1d7a286 100644
--- a/src/agent_loop_lab/datasets/build.py
+++ b/src/agent_loop_lab/datasets/build.py
@@ -309,12 +309,16 @@ def build(
     cfg: OracleConfig | None = None,
     reports_dir: str | Path | None = None,
     progress: Callable[[str], None] | None = None,
+    on_session: Callable[[str, list[CanonicalEvent], OracleRecord], None] | None = None,
 ) -> BuildResult:
     """Generate the standard splits and the counterfactual set into ``out_dir``.
 
     Writes the six split files, the two counterfactual files, ``MANIFEST.json``
     and ``splits.json``. When ``reports_dir`` is given, also reads the split
     files back and writes the dataset-stats and dev collision reports there.
+    ``on_session(split, events, record)`` (optional, observe-only) is called
+    right after each session's lines are written, in file order; the v0.3
+    raw builder uses it. It does not change what is generated or written.
     """
     from ..synth import counterfactual as cf
     from ..synth.engine import GENERATOR_VERSION
@@ -357,6 +361,8 @@ def build(
                 writers[(split, "oracle")].write(oracle_line(record))
                 split_ids[split].append(sid)
                 split_counts[split].add(events, record)
+                if on_session is not None:
+                    on_session(split, events, record)
             say(f"{family}: {n} sessions {family_split[family]}")
 
         cf_counts = _Counts()
@@ -379,6 +385,8 @@ def build(
                     writers[(COUNTERFACTUAL, "events")].write(events_line(record.session_id, events))
                     writers[(COUNTERFACTUAL, "oracle")].write(oracle_line(record))
                     cf_counts.add(events, record)
+                    if on_session is not None:
+                        on_session(COUNTERFACTUAL, events, record)
                 with_contrast += bool(pair[0][1].latent_trace["counterfactual"]["contrast_endpoints"])
             cf_summary[kind] = {"pairs": pairs_per_kind, "pairs_with_contrast_endpoints": with_contrast}
             say(f"counterfactual {kind}: {pairs_per_kind} pairs, {with_contrast} with contrast")
diff --git a/src/agent_loop_lab/datasets/ood.py b/src/agent_loop_lab/datasets/ood.py
index 79e00b4..ba4547c 100644
--- a/src/agent_loop_lab/datasets/ood.py
+++ b/src/agent_loop_lab/datasets/ood.py
@@ -17,7 +17,7 @@ import hashlib
 import json
 import random
 from pathlib import Path
-from typing import Any, Iterator
+from typing import Any, Callable, Iterator
 
 from ..oracle_config import OracleConfig, load_oracle_config
 from ..schema import CanonicalEvent, OracleLabel, OracleRecord
@@ -61,8 +61,13 @@ def build_ood(
     master_seed: int,
     sessions_per_family: int = OOD_SESSIONS_PER_FAMILY,
     cfg: OracleConfig | None = None,
+    on_session: Callable[[list[CanonicalEvent], OracleRecord], None] | None = None,
 ) -> dict[str, Any]:
-    """Generate the OOD set into ``out_dir`` and write ``OOD_MANIFEST.json``; return the manifest."""
+    """Generate the OOD set into ``out_dir`` and write ``OOD_MANIFEST.json``; return the manifest.
+
+    ``on_session(events, record)`` (optional, observe-only) is called right
+    after each session's lines are written, in file order (v0.3 raw builder).
+    """
     from ..synth_ood import FAMILIES
 
     cfg = cfg or load_oracle_config()
@@ -85,6 +90,8 @@ def build_ood(
                 fe.write(events_line(sid, events))
                 fo.write(oracle_line(record))
                 session_ids.append(sid)
+                if on_session is not None:
+                    on_session(events, record)
                 for lab in record.window_labels.values():
                     counts[lab.value] = counts.get(lab.value, 0) + 1
             label_counts[family] = dict(sorted(counts.items()))
```

## Result (2026-11-05, r1)

`uv run python scripts/build_v03_raw.py` (r1 schema, with the r1 alignment checks of `build_raw.alignment_errors` applied to every one of the 23,800 captured sessions) was run twice: once into `datasets/v0_3/` (wall clock 4m54s for the rebuild) and once into a scratch directory (5m01s, plus the dev/val token-statistics pass). Both runs gave the same result: all 10 regenerated canonical files (`standard/{dev,val,test,counterfactual}.{events,oracle}.jsonl.gz` and `ood/ood.{events,oracle}.jsonl.gz`) are **byte-identical** to the sha256 hashes in the committed `datasets/v0_2/MANIFEST.json` and `OOD_MANIFEST.json`, and no alignment check failed. The five raw files were byte-identical between the two runs:

| file | sessions | sha256 |
|---|---|---|
| `dev.raw.jsonl.gz` | 12,000 | `d8ee7b273872c431…` |
| `val.raw.jsonl.gz` | 4,004 | `e15dcf48706f43ba…` |
| `test.raw.jsonl.gz` | 3,996 | `31594b323fc0f8e8…` |
| `counterfactual.raw.jsonl.gz` | 1,800 | `b0cf33299c052564…` |
| `ood.raw.jsonl.gz` | 2,000 | `b2785e8510262d00…` |

Full per-file hashes are in `datasets/v0_3/MANIFEST_v0_3.json` under `acceptance` and `files`. The manifest also carries the renderer and tokeniser source hashes and token statistics for dev and val only (test, counterfactual and OOD record only existence and sha256). `uv run pytest`: 2388 passed, 0 failed (including `tests/test_v03_rawop.py`, which checks every capture against its canonical event and renders engine-produced edits, reverts and OOD installs).
