# v0.4 status

_Last update: 2026-11-06. Plan: `agent_doom_loop_detector_plan_v0_4.md`. Matthew accepted every recommendation in it, and the post-M0 revisions are in its "Revisions after M0" section. On 6 Nov, before G1, he dropped the training baselines and all cloud compute (plan revision 12), and then took sealed scoring out of v0.4 (plan revision 13). No Qwen output has been compared with a label._

## Second decision of 6 Nov 2026: val only, no sealed scoring

Matthew decided directly: "Val only, no sealed". His context: "I need a quick answer to how the base qwen vs modified qwen performs, not 90 hours".

- **No sealed set is opened or scored in v0.4**: not the fresh test set, v0.3 test, v0.3 OOD or the counterfactual split. They stay sealed and available to a later version. There is no sealed attempt, no G2 and no sealed-eval script (PREREGISTRATION §3.4; pre-registered revision 1 removed, §13).
- **The result is Qwen-0 against the WSP edits on val**, `excluding_disclosed` primary, harness-only. Val numbers are still never shown to the weight author.
- **Quick look first** (PREREGISTRATION §3.5). Each edit (WSP-A, then WSP-B, then WSP-F) is read out first on `datasets/v0_4/VAL_QUICK.txt` (1,000 val sessions by hash key, chosen before any edit existed; sha256 `b77f7415…`) and scored paired against Qwen-0 on the same sessions → `artifacts/reports/v04_val_quick_<arm>.json`. The full val readout follows → `v04_val_<arm>.json`, the result of record.
- **Hypotheses on val** (§9): Q1 = Qwen-0 − dev-majority prior; Q2 = WSP-B − v0.3 B. Reported: WSP-A − Qwen-0, WSP-B − Qwen-0, WSP-A − v0.3 A, WSP-F beside WSP-B. Q2-OOD, Q3, Q4 and the Q5 verdict are not tested; the edit size and Mechanisms-table eligibility are still reported (§8). Val is not a sealed set, so these are weaker evidence, and the report says so.
- **v0.3 columns on val**: `scripts/v04_v03_val_columns.py` recomputes the v0.3 A, B and C networks on val in the frozen v0.3 environment and writes each column whose every block equals `v03_sealed.json` exactly; a column that differs is left out and cited without a paired CI (PREREGISTRATION §1; if that is v0.3 B, Q2 is not tested). The bridge re-checks the file against `v03_sealed.json` before using it. It has not been run yet (CPU, several workers; run it when the MPS readout is not competing, preferably before G1, so the full C blocks are known to reproduce).
- **Scoring bridge**: accepts subset columns (VAL_QUICK by sha256, exact session ids), refuses mixed coverage, adds the constant priors and the v0.3 columns, and reports paired macro-F1 differences with 95% CIs over the shared bootstrap draws (the §9 pairs built in; `--diff` adds). It reads only the fresh set's committed REGISTRATION and MANIFEST, never its data files. The scoring script refuses unless `scripts/v04_freeze.py --check` passes.

## Decision of 6 Nov 2026: no training baselines, everything local

Matthew decided directly: "Drop the training baselines", and "I’ve never planned on renting a h100".

- **No Qwen-probe and no Qwen-LoRA** column at any budget (32, 128, 512, 2,048, all or 200 windows). WSP-A, WSP-B and WSP-F are compared with Qwen-0 and the cited v0.3 and v0.2 columns only.
- **Q3 and Q4 are not tested.** Q1, Q2, Q2-OOD and Q5 were unchanged by this decision (the second decision, above, then dropped Q2-OOD and the Q5 verdict).
- **Nothing is rented.** The author's forward server and the harness-only val readouts use local MPS, fp32 (the sealed scoring runs this bullet first listed were removed by the second decision, above). The CPU replay check runs on this Mac's CPU, Qwen-0 and WSP-B against MPS (PREREGISTRATION §3.1, §3.3).
- **Time instead of money.** One Qwen column over val takes about 5 h on MPS, a quick look about 1.25 h. The CPU replay takes several hours per column; its tool is written after G1 as pre-registered revision 3 (§3.3, §13).
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

## Running (6 Nov)

- **Qwen-0 readout of the quick-look subset** on MPS (`run_readout --split val --sessions datasets/v0_4/VAL_QUICK.txt`, label-free), from the main checkout. Scored only after the G1 freeze.
- **Then:** freeze G1 → prepare `../HypaNet-v04-author` → kit tests → launch WSP-A (Fable, cap US$500), as Matthew asked → WSP-A quick look (subset readout, scored paired against Qwen-0) → full val readouts. The v0.3 val columns (`scripts/v04_v03_val_columns.py`) run on CPU when the MPS readout is not competing.

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

So the frozen v0.3 artefacts can be re-scored here with no tolerance (plan revision 10); after the second 6 Nov decision they are re-scored on val only. The first attempt stopped because the Phase C `model.pt` files were Git LFS pointers. `git lfs install --local && git lfs pull` fixed that, and the hashes match.

## Next

1. **Qwen-0 val logits** (label-free): the quick-look subset first (about 1.25 h on MPS, running), then full val (about 5 h). They are scored only after G1.
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
