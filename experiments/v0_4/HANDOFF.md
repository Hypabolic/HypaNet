# v0.4 handoff (2026-11-06, evening)

For whoever picks up v0.4 next, human or Claude. `experiments/v0_4/STATUS.md` has the full record, and `experiments/v0_4/PREREGISTRATION.md` holds the rules, with the §13 log of deviations. This file covers where things are, what is running, what to do next and the rules that are easy to break.

## The result so far

v0.4 is compared with v0.3: Fable editing Qwen's weights (v0.4) against Fable building a small network from scratch (v0.3), with the same labelled data in rows A and B (none, then the same 200 disclosed val windows); row C is trained networks, with no v0.4 counterpart. The full summary, with CIs, false alarms and sources, is at the top of `experiments/v0_4/STATUS.md` ("Summary so far").

| Val macro-F1 | v0.3 (small network) | v0.4 (edits Qwen) |
| --- | --- | --- |
| A: original, no labels | **0.750** | 0.330\* |
| B: revision, 200 labelled windows | **0.954** | 0.419\* |
| C: trained | 0.926 (C-all) | none (dropped) |
| Starting point | | base Qwen 0.212 |

\* Quick look (1,000 val sessions); WSP-A and WSP-B full val are still to run. The rest is full val `excluding_disclosed`. So far v0.3 wins clearly: on the quick look WSP-B nearly doubles base Qwen (WSP-A about 1.5×), but both reach less than half of their v0.3 counterpart. v0.3 C's val numbers are optimistic (early-stopped on part of val). Harness-only, never shown to Fable. Fable's v0.4 cost: US$62.39 (caps US$500 and US$250). These are val numbers, not sealed ones (Matthew chose "Val only, no sealed"); WSP-B's disclosure followed deviation D1.

## Qwen-0 full val: done (6 Nov, 12:55 AEDT)

The 3-worker run (deviation D2) finished cleanly: `experiments/v0_4/columns/qwen0.val.npz`, 4,004 sessions, 87,176 rows, 7,871,600 tokens. The workers held 276–281 tok/s each, and the merged file was ready 1 h 37 min after the start (1,700-session seed plus 3 × 768). Scored → `artifacts/reports/v04_val_qwen0.json` (official):

| Population | Qwen-0 macro-F1 [95% CI] | Prior (always PRODUCTIVE) | Q1: Qwen-0 − dev-majority prior |
| --- | --- | --- | --- |
| **excluding_disclosed** (3,804 sessions, 69,974 scored windows; primary) | **0.212 [0.202, 0.222]** | 0.139 | **+0.073 [+0.061, +0.085]** |
| all_val (4,004 sessions) | 0.213 [0.204, 0.223] | | +0.076 [+0.065, +0.087] |

Per-class F1 (excluding_disclosed): STALLED 0.36, RECOVERY 0.23, OSCILLATING 0.21, PRODUCTIVE 0.05. Accuracy 0.231; false-loop alarm rate 0.709. This agrees with the quick look (0.223 [0.202, 0.243]).

The driver log was in the session scratchpad; the per-worker logs and `PLAN.json` are committed in `qwen0.val.shards/`. Worker `.npz` files are gitignored and can be deleted.

## Next, in order

1. ~~Score Qwen-0 full val~~ **done** (above): `artifacts/reports/v04_val_qwen0.json`, Q1 = +0.073 [+0.061, +0.085].
2. **WSP-A and WSP-B full val**, one at a time, about 2.5–3 h each with 3 workers. Run them from the repo root in a plain terminal, so they outlive any Claude session:
   ```sh
   caffeinate -i -s uv run python scripts/v04_parallel_readout.py --out experiments/v0_4/columns/wspa.val.npz --workers 3 --edit weight_program/v0_4/wsp-a
   caffeinate -i -s uv run python scripts/v04_parallel_readout.py --out experiments/v0_4/columns/wspb.val.npz --workers 3 --edit weight_program/v0_4/wsp-b
   ```
3. **v0.3 val columns, before the final score** (Q2 = WSP-B − v0.3 B is a primary hypothesis, and the built-in v0.3 pairs are computed only with `--v03`): `uv run python scripts/v04_v03_val_columns.py --workers W` → `experiments/v0_4/columns/v03.val.npz`. It runs on CPU in the frozen v0.3 environment, so run it only when no MPS readout is running. A column that does not reproduce `v03_sealed.json` exactly is left out and cited without a paired CI; if that column is v0.3 B, Q2 is "not tested" (PREREGISTRATION §1).
4. **Score full val together**, the result of record, written once. An official report refuses to be overwritten without `--force`, so include `--v03` the first time:
   `uv run python scripts/v04_score_validation.py --arm wspb --column "Qwen-0=experiments/v0_4/columns/qwen0.val.npz" --column "WSP-A=experiments/v0_4/columns/wspa.val.npz" --column "WSP-B=experiments/v0_4/columns/wspb.val.npz" --diff "WSP-B,WSP-A" --v03 experiments/v0_4/columns/v03.val.npz`
   This writes `artifacts/reports/v04_val_wspb.json`. A WSP-A-only report (`--arm wspa`, Qwen-0 and WSP-A columns) can be written as soon as WSP-A's full val exists.
5. **WSP-F: needs Matthew's go-ahead** (optional, reported beside WSP-B with no verdict, cap US$250). Decisions first:
   - (i) whether to run it at all;
   - (ii) the `shutil` guard gap. WSP-B's `import shutil` was not refused at run time, though the comment on `V04_FORBIDDEN_MODULES` in `src/agent_loop_lab/v04/audit.py` says the guard refuses it. The run-time guard is `envs/qwen/hypaqwen/wsp/author/hn_guard/guards_v04.py`. §7.3 and §11 do not allow changing the guard after G1. Either accept the gap (the audit catches any use, as it did for WSP-B), or log a deviation that changes the guard, knowing WSP-F then no longer runs the G1 kernel by hash.

   Engineering, as new files apart from the four §7.3 kernel files:
   - the `score_dev` and `holdout_check` ops and their tests (wsp-a and wsp-b must refuse them);
   - a WSP-F launcher mode (`scripts/v04_launch.sh` has only wsp-a, wsp-b, continue and resubmit);
   - a WSP-F prompt and settings;
   - a prepare step that seeds the journal from WSP-B;
   - the reader for `wspf_holdout`.

   Then check whether `scripts/v04_run_build.py` accepts a WSP-F build. A revision that changes kernel files ends the G1 kernel chain (`handover.kernel_freezes`), and the build's freeze check may refuse it.
6. **Pending decision for Matthew: the CPU replay check** (§3.3). It is pre-registered and reported: 100 val sessions, Qwen-0 and WSP-B, on CPU, several hours per column, label-free. If MPS and CPU agree on fewer than 99.5% of windows, every MPS column carries a caveat. Matthew has not dropped it. Either write the tool (pre-registered revision 3 in §13's numbering; it would be **registry revision 4**, because registry revisions 1–3 are D1, the D1 follow-on and D2) and run it when MPS is idle, or, if he declines, log a deviation in §13 and note "CPU replay not run" beside the MPS columns. The LoRA/probe baselines were dropped by Matthew on 6 Nov.

## Rules that are easy to break

- **Sealed data:** never open or decompress `datasets/**/test.*`, `ood.*`, `counterfactual.*` or the `datasets/v0_4/fresh_test/` data. Only their committed MANIFEST/REGISTRATION JSON may be read. v0.4 scores no sealed set.
- **Fable never sees a val number.** Reports and columns are harness-only. The only thing Fable has seen is the WSP-B disclosure of 200 windows.
- **No money beyond the Fable caps, and no rented compute,** without Matthew agreeing to the cost first. He "never planned on renting a h100". Git LFS for the large binaries is also off for now (see the size notes below).
- **The registry is frozen.** Any change to a file under `envs/qwen/**`, `experiments/v0_4/author_kit/**`, `scripts/v04_*`, `src/agent_loop_lab/v04/**`, or the required files (PREREGISTRATION, plan, spec, …) needs `scripts/v04_freeze.py --revise --reason … [--allow-change PATH]`, plus a §13 log entry when it is not pre-registered. `scripts/v04_freeze.py --check` must pass, and scoring refuses otherwise. A revision that changes a kernel file ends the G1 kernel chain (`handover.KERNEL_NEUTRAL_FILES`). Do not edit the hash-pinned `fresh_test.py`, `prompt_choice.py` or `prompt_choice/README.md`, or any v0.3 or v0.2 frozen file.
- **One MPS job at a time,** apart from the parallel readout's own workers. Do not run heavy test suites during a readout; an earlier readout died under that load. 4 workers is the memory limit, and 3 is the setting in use.
- **Before any Fable launch:** remove pytest's temp dir (`rm -rf /private/var/folders/*/*/T/pytest-of-$USER`), because the launcher refuses while harness scratch is in the temp roots.
- **Git:** push `main` after each commit (Matthew authorised this on 6 Nov). Never force-push, and do not push worktree branches. End every commit with the `Co-Authored-By` and `Claude-Session` trailers used in this repo's log.
- **Shell (zsh):** a bare word starting with `=` (`echo ===`) fails with "not found" and breaks command chains. macOS has no `timeout`.

## Where things are

| What | Where |
| --- | --- |
| Registry and freeze | `experiments/registry_v0_4.yaml`. G1 `494db5e3`, current `bf81354e` (revisions 1–3) |
| Fable's author dir | `../HypaNet-v04-author` (outside the repo; `output/wsp-a/` and `disclosure/` are read-only). Session `dd5e5940-2e7c-4007-a3b8-05927e29a4c9` (`experiments/v0_4/SESSION_ID`) |
| Launch, audit and cost logs | `experiments/v0_4/logs/`: `wsp-{a,b}.stream.jsonl`, `wsp-{a,b}.stream.audit.json`, `.cost.json`, `.ledger.jsonl`, `.kernel.jsonl`, `LAUNCHES.jsonl`, `wsp-{a,b}.ADJUDICATION.md` |
| Edit archives | `weight_program/v0_4/wsp-{a,b}/`: BUILD_RECORD.json, edit_diff, build inputs, author output, MECHANISMS.json, FREEZE.json |
| WSP-B disclosure | `experiments/v0_4/disclosure/DISCLOSURE.json` (`d8c75fbc`), `experiments/v0_4/DISCLOSED_WINDOWS.json`, `experiments/v0_4/DISCLOSURE_PRIVATE.json`, `experiments/v0_4/wsp_b/HANDOVER.json` |
| Numbering | PREREGISTRATION §13's "pre-registered revisions 1–3" are not registry revision numbers. Registry revisions 1–3 are D1, the D1 follow-on and D2 |
| Readout columns | `experiments/v0_4/columns/`: `qwen0.val_quick`, `wspa.val_quick`, `wspb.val_quick`, `wspa.val_disclosed` (D1); `qwen0.val` (full val, done) and `qwen0.val.shards/` |
| Quick-look subset | `datasets/v0_4/VAL_QUICK.txt` (`b77f7415`, 1,000 sessions, no disclosed session) |
| Parallel benchmark | `experiments/v0_4/benchmarks/parallel_readout/` (README, bench.py, compare.py, results.jsonl) |

**Kept local, not in git:** each `weight_program/v0_4/*/edit.safetensors` (WSP-A 126 MB, WSP-B 84 MB) and the forward-pass journals `experiments/v0_4/logs/wsp-{a,b}/journal/` (462 MB and 630 MB). Their hashes are in `BUILD_RECORD.json`. The edits can be rebuilt from the archived `inputs/` (`build_edit.py` plus the constants), and the journals are what the build checks against. Also kept local: shard and checkpoint `.npz` files and the datasets. Matthew asked about Git LFS on 6 Nov. It handles these sizes (2–5 GB per-file cap, 1.3 GB in total), but he chose to leave it off for now.

**Recovery:** the full-val readouts of WSP-A and WSP-B need `edit.safetensors`. If it is lost, `uv run python scripts/v04_run_build.py wsp-a --archive /tmp/rebuild-wsp-a` (and the same for wsp-b) rebuilds it, provided `../HypaNet-v04-author` and `experiments/v0_4/logs/wsp-{a,b}/journal/` are in place. Check the rebuilt sha256 against `BUILD_RECORD.json` before using it. Without the author dir and the journals there is no rebuild path, so until LFS is on, keep a copy of those items (about 1.3 GB) outside this Mac, e.g. on an external disk. That needs no money.

## Known issues

- **The run-time guard did not refuse `shutil`** in WSP-B (see Next 5). The audit caught it, and the use was harmless.
- **Pre-existing test failures, not caused by v0.4:** `tests/test_m4_eval.py::test_sealed_script_refuses_second_run` and `tests/test_v03_runner.py::test_side_table_is_unreachable_at_run_time[absolute]`. `test_wsp_runner` errors under concurrent load but passes alone.
- **The launch times in the logs are UTC** (5 Nov), while the Mac's local clock reads 6 Nov.
- **Still open from STATUS "Kernel rules":** the launcher should deny Write/Edit under `tools/` and protect `./py` and `./fwd`. The WSP-A and WSP-B audits found no attempt at either.
