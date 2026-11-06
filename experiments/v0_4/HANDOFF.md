# v0.4 handoff (2026-11-06, evening)

For whoever picks up v0.4 next, human or Claude. `experiments/v0_4/STATUS.md` has the full record, and `experiments/v0_4/PREREGISTRATION.md` holds the rules, with the §13 log of deviations. This file covers where things are, what is running, what to do next and the rules that are easy to break.

## The result so far

Fable (claude-fable-5-1) edited the weights of Qwen3.5-0.8B by hand to detect agent doom loops. Scored on the val quick look (1,000 val sessions, 18,222 windows; harness-only, never shown to Fable):

| Column | Macro-F1 [95% CI] | Paired difference |
| --- | --- | --- |
| Prior: always PRODUCTIVE | 0.137 [0.131, 0.144] | |
| Qwen-0 (base) | 0.223 [0.202, 0.243] | vs prior +0.086 [+0.062, +0.108] |
| WSP-A (no labelled data) | 0.330 [0.309, 0.350] | vs Qwen-0 +0.107 [+0.083, +0.130] |
| WSP-B (plus the 200 labelled val windows) | **0.419 [0.397, 0.441]** | vs Qwen-0 **+0.196 [+0.165, +0.226]**; vs WSP-A +0.089 [+0.065, +0.113] |

The reports are `artifacts/reports/v04_val_quick_{qwen0,wspa,wspb}.json`. Fable's cost: WSP-A US$32.58 and WSP-B US$29.81, US$62.39 in total, against caps of US$500 and US$250.

**Caveats:** these are val numbers, not sealed ones (Matthew chose "Val only, no sealed"). The full-val `excluding_disclosed` numbers are the result of record and are still to come. WSP-B's disclosure followed deviation D1.

## Running right now

**Qwen-0 full val, 3 parallel MPS workers** (deviation D2), started from the main checkout:

```sh
caffeinate -i -s uv run python scripts/v04_parallel_readout.py --out experiments/v0_4/columns/qwen0.val.npz \
    --workers 3 --seed-partial experiments/v0_4/columns/qwen0.val.partial.npz
```

- The seed is 1,700 sessions from the interrupted single-process run, copied to `qwen0.val.shards/seed.partial.npz`. Each of the 3 shards holds 768 sessions. It started at 11:18 AEDT on 6 Nov. About 4.7M tokens remained at the start (7.87M minus the seed's 3.15M). At about 280–285 tok/s per worker, it should finish around 13:00 AEDT, ±30 min.
- **It runs as a background task of Claude Code session `84f94084-9d88-46ef-9b1b-8b4fda2dc583`.** The driver log (shard pids, merge result or refusal) is outside the repo: `/private/tmp/claude-501/-Users-matthew-Development-hypabolic-HypaNet/84f94084-9d88-46ef-9b1b-8b4fda2dc583/scratchpad/qwen0_parallel.log`. Closing that session may kill the workers and `caffeinate`, which lets the Mac sleep. If that happens, use "If it died" below.
- **Check progress:** `tail -1 experiments/v0_4/columns/qwen0.val.shards/shard{0,1,2}.log` and `pgrep -fl hypaqwen.run_readout`.
- **Finished** when `experiments/v0_4/columns/qwen0.val.npz` exists (4,004 sessions, 87,176 rows).
- **If it died** (session ended, reboot):
  1. First make sure nothing is still running: `pgrep -fl 'hypaqwen.run_readout|v04_parallel_readout'` must print nothing. **This run started before the lock existed, so `LOCK` does not protect it.** Never start a second copy while any of its processes is alive; 6 fp32 workers would not fit in 24 GB.
  2. Run the same command again from the repo root, in a plain terminal (not as a Claude background task), so that it outlives any session.
  3. The plan is fixed in `qwen0.val.shards/PLAN.json`, which was brought up to the reviewed format. Finished shards are kept and checked. Unfinished ones resume from `shard{k}.partial.npz` (checkpoints every 50 sessions). If all three `shard{k}.npz` exist but `qwen0.val.npz` does not, the re-run only checks and merges.
  4. From then on the reviewed script holds `LOCK`.
- **Note:** if this run completes, its merge is done by the pre-review driver code that is in memory. An independent review found that merge logic correct; the fixes concerned re-runs only. The merged file records `parallel.plan_sha256` and each shard's sha256.

## Next, in order

1. **Score Qwen-0 full val** when it finishes:
   `uv run python scripts/v04_score_validation.py --arm qwen0 --column "Qwen-0=experiments/v0_4/columns/qwen0.val.npz"`
   → `artifacts/reports/v04_val_qwen0.json`. This gives Q1 (Qwen-0 − dev-majority prior). Commit the column and report.
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
| Readout columns | `experiments/v0_4/columns/`: `qwen0.val_quick`, `wspa.val_quick`, `wspb.val_quick`, `wspa.val_disclosed` (D1); `qwen0.val.shards/` (running) |
| Quick-look subset | `datasets/v0_4/VAL_QUICK.txt` (`b77f7415`, 1,000 sessions, no disclosed session) |
| Parallel benchmark | `experiments/v0_4/benchmarks/parallel_readout/` (README, bench.py, compare.py, results.jsonl) |

**Kept local, not in git:** each `weight_program/v0_4/*/edit.safetensors` (WSP-A 126 MB, WSP-B 84 MB) and the forward-pass journals `experiments/v0_4/logs/wsp-{a,b}/journal/` (462 MB and 630 MB). Their hashes are in `BUILD_RECORD.json`. The edits can be rebuilt from the archived `inputs/` (`build_edit.py` plus the constants), and the journals are what the build checks against. Also kept local: shard and checkpoint `.npz` files and the datasets. Matthew asked about Git LFS on 6 Nov. It handles these sizes (2–5 GB per-file cap, 1.3 GB in total), but he chose to leave it off for now.

**Recovery:** the full-val readouts of WSP-A and WSP-B need `edit.safetensors`. If it is lost, `uv run python scripts/v04_run_build.py wsp-a --archive /tmp/rebuild-wsp-a` (and the same for wsp-b) rebuilds it, provided `../HypaNet-v04-author` and `experiments/v0_4/logs/wsp-{a,b}/journal/` are in place. Check the rebuilt sha256 against `BUILD_RECORD.json` before using it. Without the author dir and the journals there is no rebuild path, so until LFS is on, keep a copy of those items (about 1.3 GB) outside this Mac, e.g. on an external disk. That needs no money.

## Known issues

- **The run-time guard did not refuse `shutil`** in WSP-B (see Next 5). The audit caught it, and the use was harmless.
- **Pre-existing test failures, not caused by v0.4:** `tests/test_m4_eval.py::test_sealed_script_refuses_second_run` and `tests/test_v03_runner.py::test_side_table_is_unreachable_at_run_time[absolute]`. `test_wsp_runner` errors under concurrent load but passes alone.
- **The launch times in the logs are UTC** (5 Nov), while the Mac's local clock reads 6 Nov.
- **Still open from STATUS "Kernel rules":** the launcher should deny Write/Edit under `tools/` and protect `./py` and `./fwd`. The WSP-A and WSP-B audits found no attempt at either.
