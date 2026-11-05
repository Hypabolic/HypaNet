# v0.4 status

_Last update: 2026-11-05 21:30. Plan: `agent_doom_loop_detector_plan_v0_4.md`. Matthew accepted every recommendation in it, and the post-M0 revisions are in its "Revisions after M0" section. No Qwen output has been compared with a label._

## Done

| Step | Where | Result |
| --- | --- | --- |
| Data rebuilt on this Mac | datasets/ (gitignored) | Every canonical v0.2 file is byte-identical; every v0.3 raw file matches its manifest |
| Qwen environment | `envs/qwen` | torch 2.14.1 and transformers 5.22.1. Qwen3.5-0.8B pinned at revision `2fc06364`; every file's sha256 is checked; offline load; SDPA attention |
| Task export (dev, val) | `scripts/v04_export_tasks.py` | Rendered blocks with no labels, plus a separate labels file. val: 87,176 windows (the v0.3 rows) |
| Verbaliser readout | `hypaqwen/readout.py` | One pass per session with cache snapshot and restore. Equals a separate prompt per window on MPS and CPU. Label-row edits reach the readout |
| Token audit | `artifacts/reports/v04_token_audit.json` | val: 1,726 tokens per session with the prefix (1,559 without), 1.97× v0.3; 7.87M readout tokens |
| **Fresh sealed test set** | registration `6a778d1`, manifest `8475556` | Seed 20261106, 3,996 sessions, 86,412 windows (83,874 gate-passing). Rebuilt byte-identically under a second PYTHONHASHSEED. No label statistics recorded |
| Prompt choice, pre-registered | `experiments/v0_4/prompt_choice/README.md` | 12 candidates and a label-free rule (median label-word mass, 100 dev sessions) |
| Scoring bridge | `src/agent_loop_lab/v04/scoring.py` | Val rows checked against v0.3's `rows_sha256`. Strict column alignment. `m4.score_columns`, with H2 not applicable |
| WSP kernel contract | `docs/V04_WSP_KERNEL.md` | Two processes; forward as a separate metered tool; edit scope; guards; build runner |

## Running

- **Prompt choice** on MPS, 12 × 100 dev sessions, label-free. The report goes to `artifacts/reports/v04_prompt_choice.json`.
- **WSP kernel build**, as a workflow in three worktrees: server/client/`./fwd`, guards on torch 2.14.1, and the build runner. Each part is reviewed and fixed before it is merged.
- **v0.3 val dry run on arm64** (unofficial; report in the session scratchpad). It checks whether v0.3's exact-equality reproduction checks still hold on this machine. The first attempt stopped because the Phase C `model.pt` files were Git LFS pointers. `git lfs install --local && git lfs pull` fixed that, and the hashes match.

## Next

1. Freeze the chosen interface text in `hypaqwen/interface.py`.
2. Build the session harness: prepare script, launch with a tool allow-list and command hook, audit, cost by session delta, packet, kit tests.
3. Write the v0.4 spec with `[packet]` sections.
4. Write the probe and LoRA recipes and the cloud-GPU scripts.
5. Write the G1 pre-registration and the `registry_v0_4.yaml` freeze.

## Measurements on this Mac

| Backend | Verbaliser readout | Note |
| --- | --- | --- |
| MPS, fp32 | 445 tok/s | Compute-bound; a fixed 40–70 ms per call. One val pass takes about 5 h; 2M tokens take about 75 min |
| CPU, fp32 | under 10 tok/s | Measured with other jobs running. The 100-session CPU replay check moves to the cloud machine |

## Corrections to earlier notes

- An earlier version of this file said the bridge would call `metrics.score_report`. It calls `m4.score_columns`: `score_report` needs a reference policy and accepts val only.
- v0.3's cost was over-counted. `total_cost_usd` is cumulative across resumed launches, so v0.3's hand-written work cost US$64.62 in total, not US$122.24.

## Known pre-existing test failures (not v0.4)

`tests/test_m4_eval.py::test_sealed_script_refuses_second_run` and `tests/test_v03_runner.py::test_side_table_is_unreachable_at_run_time[absolute]` fail on this machine. pytest's temp directory is under `/private/var`, outside the repo. Every other test passes.
