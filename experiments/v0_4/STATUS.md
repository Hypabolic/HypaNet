# v0.4 status

_Last update: 2026-11-06 02:00. Plan: `agent_doom_loop_detector_plan_v0_4.md`. Matthew accepted every recommendation in it, and the post-M0 revisions are in its "Revisions after M0" section. On 6 Nov, before G1, he dropped the training baselines and all cloud compute (below; plan revision 12). No Qwen output has been compared with a label._

## Decision of 6 Nov 2026: no training baselines, everything local

Matthew decided directly: "Drop the training baselines", and "I’ve never planned on renting a h100".

- **No Qwen-probe and no Qwen-LoRA** column at any budget (32, 128, 512, 2,048, all or 200 windows). WSP-A, WSP-B and WSP-F are compared with Qwen-0 and the cited v0.3 and v0.2 columns only.
- **Q3 and Q4 are not tested.** Q1, Q2, Q2-OOD and Q5 are unchanged.
- **Nothing is rented.** The author's forward server, the harness-only val readouts and every sealed scoring run (fresh test, v0.3 test, v0.3 OOD, counterfactual and the Q5 readouts) use local MPS, fp32. The CPU replay check runs on this Mac's CPU, Qwen-0 and WSP-B against MPS (PREREGISTRATION §3.1, §3.3).
- **Time instead of money.** One Qwen column over val takes about 5 h on MPS, so the sealed attempt takes several days; an interrupted attempt resumes from per-split, per-column checkpoints and no score is computed until every readout is done (PREREGISTRATION §3.4). The CPU replay takes several hours per column; its tool is written after G1 as pre-registered revision 3 (§3.3, §13).
- **Code removed.** The probe, LoRA and cloud-runner code merged earlier (`hypaqwen/{probe, lora, cloud, backend, columns, budgets}.py`, `envs/qwen/scripts/cloud_run.sh`, and their tests) is deleted, and `peft` is out of `envs/qwen`. It stays in git history at `0540341`. `datasets/v0_4/BUDGETS.json` is kept unchanged: v0.4 uses its 200 disclosed windows and its WSP-F holdout.
- **Updated before G1:** PREREGISTRATION (§1, §3, §9 to §13; §4, §5 and §7.2 are stubs, so section numbers do not move; §8 now defines the "Small" yardstick itself), the spec's non-packet sections, the plan (revision 12) and the freeze's required components.

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
| Prompt choice, run and revised | `artifacts/reports/v04_prompt_choice*.json` | The first run chose `w3-state-full`, under which Qwen answers STALLED on all 2,069 dev windows (reported as run). Revision 1 (approved): mass × entropy of the predicted-class shares, still label-free → **`w1-state-full`** (0.742; next 0.526), the M0 draft text. Frozen as `interface.FROZEN`, `spec_sha256` `5922eb45…` |
| **WSP kernel** (§3–6) | `envs/qwen/hypaqwen/wsp/` | Model server with a kernel socket and a forward socket, author client, `./fwd`, guards on torch 2.14.1 (KERNEL_CLIENT, BUILD_EDIT), build runner. Built in three worktrees, then adversarially reviewed: 19 findings, all fixed. At merge, the server and runner were given one shared rescaling rule and edit hash (`wsp/edit_state.py`), and the runner was moved to the real guard |
| Scoring bridge | `src/agent_loop_lab/v04/scoring.py` | Val rows checked against v0.3's `rows_sha256`. Strict column alignment. `m4.score_columns`, with H2 not applicable |
| WSP kernel contract | `docs/V04_WSP_KERNEL.md` | Two processes; forward as a separate metered tool; edit scope; guards; build runner |

| **v0.4 spec and author packet** | `agent_doom_loop_detector_spec_v0_4.md`, `scripts/v04_build_packet.py` | `[packet]` sections, plus the v0.2 labelling rules exactly as v0.3 A saw them; the interface and Qwen tokenizer files; leakage checks; deterministic. Reviewed (6 findings, 5 fixed, 1 accepted at G1: the tool files name WSP-B mechanics) |
| **Session harness** | `scripts/v04_{prepare_author_dir,launch,kit_tests}.sh`, `v04_{cost,audit_session,run_build}.py`, `experiments/v0_4/author_kit/` | `./py` launcher under the KERNEL_CLIENT guard; command hook; settings; session-id cost accounting, which also charges killed launches; AST audit plus ledger cross-check; watchdog. Reviewed (7 findings, all fixed). The kit test passed end to end on a scratch dir (Haiku, server on CPU) |
| **G1 pre-registration (draft)** | `experiments/v0_4/PREREGISTRATION.md`, `scripts/v04_freeze.py` | Backends, Q5 statistic and hypotheses (the probe and LoRA recipes were dropped on 6 Nov). The freeze covers every v0.4 tree and the root-env import closure, and refuses until the WSP-B pieces exist. Reviewed (8 findings, all fixed) |

## Running (night of 5–6 Nov)

- **Qwen-0 val readout** on MPS (`experiments/v0_4/columns/qwen0.val.npz`, label-free), ETA about 04:00. Scored only after the G1 freeze.
- **Workflow: the last G1 components.** The WSP-B disclosure builder and prompt, and the Mechanisms table in EDIT.md. Each is reviewed before the freeze. The probe, LoRA and cloud-runner pieces were merged and then removed after the 6 Nov decision.
- **Then:** freeze G1 → prepare `../HypaNet-v04-author` → kit tests → score Qwen-0 on val (harness only) → launch WSP-A (Fable, cap US$500), as Matthew asked.

## Kernel rules added beyond the contract (fold into `docs/V04_WSP_KERNEL.md` at G1)

- **Rescaling refusal.** With `new = c·W + R`, an edit is refused when `|c − 1| > 1e-3` and `‖R‖ < 0.5·‖(c − 1)·W‖`. For the zero-centred `(1 + w)` norms it also applies to the gain; for the label rows, also to the row-centred block and to the changed rows.
- **Disclosed windows.** `forward` refuses any session that overlaps a WSP-B disclosed window (0.6 shingle or block share). `forward_disclosed` allows **at most 2** distinct non-empty edit states per phase. Confirm at G1.
- **Load quota.** A `./py` script may load at most half of the layer-weight bytes (distinct tensors, per phase). Every kernel op is logged to `<phase>.kernel.jsonl`.
- **Tools copied read-only.** The tools go into the author dir read-only, and `install.verify()` checks them. **Still needed in the launcher (§7):** deny Write/Edit under `tools/` and protect the `./py` and `./fwd` wrappers.
- **Guard.** stderr is kept on a private copy of fd 2, so any violation still ends the run with exit 3. Launchers must use `hn_guard.clean_run(returncode, stderr)`.

## v0.3 reproduces exactly on this Mac

The unofficial v0.3 val dry run (`scripts/v03_sealed_eval.py --dry-run-val`) ran with macOS-27.0.1-arm64, Python 3.12.16, torch 2.2.2, 2 threads and 4 workers. Its report stays in the session scratchpad. Every exact-equality check passed:

- the official v0.3 Phase A and Phase B val reports;
- all 11 Phase C early-stopping macro-F1 values;
- the scoring path, for both phases;
- the val and counterfactual row identity against v0.2;
- the disclosure rule;
- the two priors, overall and per family.

So the frozen v0.3 artefacts can be re-scored on the fresh test set here with no tolerance (plan revision 10). The first attempt stopped because the Phase C `model.pt` files were Git LFS pointers. `git lfs install --local && git lfs pull` fixed that, and the hashes match.

## Next

1. **Qwen-0 val logits** (label-free, about 5 h on MPS): `hypaqwen.run_readout --split val`. They are scored only after G1.
2. Build the session harness: prepare script, launch with a tool allow-list and command hook, audit, cost by session delta, packet, kit tests.
3. Write the v0.4 spec with `[packet]` sections.
4. ~~Write the probe and LoRA recipes and the cloud-GPU scripts.~~ Dropped on 6 Nov (no training baselines).
5. Write the G1 pre-registration and the `registry_v0_4.yaml` freeze.

## Measurements on this Mac

| Backend | Verbaliser readout | Note |
| --- | --- | --- |
| MPS, fp32 | 445 tok/s | Compute-bound; a fixed 40–70 ms per call. One val pass takes about 5 h; 2M tokens take about 75 min |
| CPU, fp32 | under 10 tok/s | Measured with other jobs running. The 100-session CPU replay check runs here, for Qwen-0 and WSP-B: several hours per column |

## Corrections to earlier notes

- An earlier version of this file said the bridge would call `metrics.score_report`. It calls `m4.score_columns`: `score_report` needs a reference policy and accepts val only.
- v0.3's cost was over-counted. `total_cost_usd` is cumulative across resumed launches, so v0.3's hand-written work cost US$64.62 in total, not US$122.24.

## Known pre-existing test failures (not v0.4)

`tests/test_m4_eval.py::test_sealed_script_refuses_second_run` and `tests/test_v03_runner.py::test_side_table_is_unreachable_at_run_time[absolute]` fail on this machine. pytest's temp directory is under `/private/var`, outside the repo. Every other test passes.
