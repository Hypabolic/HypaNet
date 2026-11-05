# v0.4 pre-registration (gate G1)

_Written 2026-11-05, before any Qwen output has been compared with a label and before the weight author's session exists; revised 2026-11-06, still before G1, when Matthew dropped the training baselines and all cloud compute (Authority, below). Plan: `agent_doom_loop_detector_plan_v0_4.md` (Matthew accepted every recommendation, including the "Revisions after M0"). Every frozen file, the interface hash, the model hashes and the packet hash are recorded in `experiments/registry_v0_4.yaml` by `scripts/v04_freeze.py`; `scripts/v04_freeze.py --check` must pass before every launch and every scoring run. Where the plan left a number open, this file fixes it and gives the reason in one line ("Why")._

**Authority.** Decided under the delegated authority of the plan (Matthew, 2026-11-05: "proceed on recommendations"), except one decision Matthew took directly on 2026-11-06, before G1, overriding the plan's accepted recommendation: "Drop the training baselines". v0.4 has no Qwen-probe and no Qwen-LoRA column at any budget, and no cloud GPU: nothing is rented, and every Qwen run, every sealed scoring run included, is on this Mac (§3.1). The plan's Q3 and Q4 are therefore not tested (§9). §4, §5 and §7.2 stay as stubs so that section numbers do not move. Nothing below may change after this file is committed with the registry. A change needs a dated entry in §13, an additive registry revision (`scripts/v04_freeze.py --revise`, §13) and a line in the report.

## 1. Arms and columns

Every Qwen arm starts from Qwen3.5-0.8B at revision `2fc06364` (`hypaqwen.MODEL_REVISION`, every file's sha256 in `hypaqwen.model.FILE_SHA256`) and reads out through the same verbaliser (§2).

| Column | Built by | Labelled data | Answers |
| --- | --- | --- | --- |
| **Qwen-0** | nobody: released weights, frozen interface | 0 | Q1 |
| **WSP-A** | the weight author, sealed session, no labels | 0 | Q5-A; reported beside Q2 (vs v0.3 A) |
| **WSP-B** | the same author, one revision after the 200 disclosed windows | 200 windows | Q2, Q2-OOD, Q5-B |
| **WSP-F** (separate column, never instead of WSP-B) | WSP-B plus a labelled dev budget (§7.3) | 200 windows + 50 × 128 dev sessions scored in aggregate | reported beside, no verdict |

- **No trained Qwen column** (Matthew, 2026-11-06): no Qwen-probe and no Qwen-LoRA at any budget (32, 128, 512, 2,048, all or 200w). The trained columns in the report are the cited v0.3 C and v0.2 C-all below; nothing new is trained.
- **Cited columns** (from their sealed reports, never re-run): v0.3 A, v0.3 B, v0.3 C-{32, 128, 512, 2048, all}, the rules engine and the constant priors (`artifacts/reports/v03_sealed.json`); v0.2 B and C-all (`artifacts/reports/m4_sealed_v02.json`).
- **Re-scored on the fresh test set** (plan revision 2): v0.3 A, v0.3 B, every v0.3 C seed, the rules engine and the constant priors, with their frozen artefacts in the frozen v0.3 environment on this Mac (§3.4). v0.2 B and C-all are added only if the frozen v0.2 feature path runs without touching the spent v0.2 sealed driver; otherwise they stay cited.
- For the v0.3 C budgets with 3 seeds (32, 128, 512), every seed is reported plus the mean ± sample sd and the seed-mean CI (`m4.score_columns` groups), as v0.3 did.

## 2. Interface

- **Frozen text:** `envs/qwen/hypaqwen/interface.py` `FROZEN`, candidate `w1-state-full`, `spec_sha256()` = **`5922eb4500122921ac7d07e3d35599b95cb10b462c4f50b89e5d6bb989437879`** (prefix, block template, suffix and label words). `scripts/v04_freeze.py` recomputes it from the file and refuses a freeze with any other value.
- **Label words** (class order PRODUCTIVE, RECOVERY, STALLED, OSCILLATING): ` productive`, ` recovering`, ` stalled`, ` looping`, each one Qwen token with its leading space. Answer cue `State:`, the `full` instruction style, the Qwen chat format with thinking off.
- **How it was chosen:** without labels, by the pre-registered rule of `experiments/v0_4/prompt_choice/README.md` as amended by its **Revision 1** (median label-word mass × normalised entropy of the predicted-class shares; reports `artifacts/reports/v04_prompt_choice.json` and `artifacts/reports/v04_prompt_choice_revision1.json`). The first run's choice (`w3-state-full`, a constant STALLED classifier) is reported as run.
- **Readout** (`hypaqwen/readout.py`): one pass per session, the cache branched after each block to run only the suffix; the class is the first-index argmax over the four label-word logits (the v0.3 tie rule). It equals a separate prompt per window (checked on MPS and CPU). The edited label rows are read live, so a WSP edit of the tied embedding reaches the readout.
- **Fixed for every arm.** Nobody may change the prefix, suffix, label words or tokenisation after G1. The weight author may change weights only (`docs/V04_WSP_KERNEL.md` §2).
- **Windows, scored mask, labels, populations:** v0.3's, unchanged. val 87,176 windows (rows checked against v0.3's `rows_sha256` `cdf6cce8…`), v0.3 test 87,148, v0.3 OOD 83,011, fresh test 86,412 (83,874 gate-passing).

## 3. Readout, scoring and sealed sets

### 3.1 Backends

Everything runs on this Mac. Nothing is rented.

| Run | Backend | Settings |
| --- | --- | --- |
| Harness-only val readouts of Qwen-0, WSP-A, WSP-B, WSP-F; the weight author's forward server; **every sealed scoring run** (fresh test, v0.3 test, v0.3 OOD, counterfactual, and the Q5 readouts of §8) | local MPS on this Mac (Apple M4 Pro, 24 GB) | fp32 weights and activations, SDPA attention (`hypaqwen.model.ATTN_IMPLEMENTATION`), the torch implementation of Gated DeltaNet (no `flash-linear-attention` or `causal-conv1d` installed), the `envs/qwen/uv.lock` wheels |
| CPU replay check (§3.3) | this Mac's CPU | fp32, the same wheels and implementations |

- **Why one backend for every Qwen column, the author's server included:** it removes the backend as a source of argmax flips between the val readouts, the author's own forward passes and sealed scoring.
- **Why local:** nothing is rented (Matthew, 2026-11-06), and the Mac's GPU runs the readout at about 445 tok/s (plan revision 1).
- **Determinism is not claimed on MPS.** torch's deterministic-algorithms mode does not guarantee bitwise-repeatable results on MPS, so no MPS run relies on it. The CPU replay (§3.3) is the numerical check.

### 3.2 Scoring

- **Bridge:** `src/agent_loop_lab/v04/scoring.py` in the frozen v0.3 environment. Per column it checks the rows, the session ids and row positions, refuses non-finite logits, takes the first-index argmax and scores with `eval.m4.score_columns` (as v0.3 scored Phase C) with no fidelity pairs: **H2 is not applicable** without a reference policy, and Q5 replaces it.
- **Val populations:** `excluding_disclosed` (the 200 Phase B disclosed sessions removed) is primary, `all_val` is reported alongside.
- **Metrics:** v0.2 §10.1 as v0.3: macro-F1 (headline), per-class precision and recall, false-loop alarm rate, confusion, counterfactual consistency, session-bootstrap 95% CIs (10,000 resamples, seed 20261104, `eval.metrics`), overall and per family.
- **Paired comparisons** (§9): the difference of two columns' macro-F1 over the same bootstrap draws (`score_columns` already shares the draws across columns), 95% percentile interval.
- **Val numbers are never shown to the weight author**, in any phase.

### 3.3 CPU replay check

On this Mac, the 100 val sessions with the smallest `sha256("20261105:v04-cpu-replay:{session_id}")` are read out on CPU for Qwen-0 and WSP-B, and the window-level argmax agreement CPU vs MPS is reported. No labels are read.

- **Why these two columns:** one per verbaliser path, the unedited weights (Qwen-0) and an edited model (WSP-B, which carries the WSP-A edit and its revision). CPU is slow here for the readout's many short calls (under 10 tok/s measured with another job running), so each column is several hours of CPU.
- Agreement below 99.5% of windows is reported as a numerical-backend caveat beside the affected column. It never changes a score.
- **Code:** no G1 file selects the 100 sessions or computes the agreement. The replay tool (the selection by the key above, the CPU readouts through `hypaqwen.run_readout` on its session subset, and the agreement report; no labels) is written after G1 against this section only, with a CPU-only test, and committed with an additive registry revision before the replay is run (§13, pre-registered revision 3: new files only, no `--allow-change`, every G1 hash unchanged). **Why after G1:** the replay needs the WSP-B archive, which exists only after G1, and the rule it implements is fixed here.

### 3.4 Sealed sets

- **Primary:** the fresh test set (`experiments/v0_4/fresh_test/REGISTRATION.json`, seed 20261106; `datasets/v0_4/fresh_test/MANIFEST_fresh_test.json`, 3,996 sessions). **Secondary:** v0.3 test and v0.3 OOD (no OOD-2: plan decision).
- **Once, after G2**, by the pre-registered script `scripts/v04_sealed_eval.py`. It is written after G1 against this section and §8 only, and committed with an additive registry revision (`scripts/v04_freeze.py --revise`, §13: new files only, no `--allow-change`, every G1 hash unchanged) before any sealed file is opened. It refuses a second official run (as v0.3 E7; resuming an interrupted attempt is not one, below) and refuses unless `scripts/v04_freeze.py --check` passes.
- **Order:** one attempt, one pass per split: counterfactual first (every column's counterfactual consistency, and every Q5 readout of §8: *E*, each *E∖m* and *O*), then the fresh test set, then v0.3 test and OOD. Before the attempt opens any sealed file it records the sha256 of every archived `EDIT.md` and `BUILD_RECORD.json` it reads mechanism tables and sizes from; Q5 uses only those files.
- **v0.3 artefacts on the fresh set:** re-scored here on arm64 in the v0.3 environment with no tolerance; the unofficial v0.3 val dry run matched every official v0.3 number exactly on this machine (plan revision 10). Their v0.3 test and OOD predictions are recomputed for the paired comparisons and must reproduce the cited sealed numbers exactly, or the attempt stops.
- **Backend:** the sealed attempt runs on local MPS, fp32 (§3.1). One Qwen column over val takes about 5 h there, so the attempt is several days of MPS time.
- **Interrupted attempt.** A crash, power loss or restart during the attempt does not end it.
  - **Checkpoints:** `scripts/v04_sealed_eval.py` checkpoints every readout per split and per column (the logits of every completed session, as `hypaqwen.run_readout` does), with the registry `freeze_sha256`, the readout's arguments and its session ids.
  - **Resume:** on restart it resumes the same attempt with the same frozen code, only after `scripts/v04_freeze.py --check` passes, and refuses if the checkpoint's `freeze_sha256`, arguments or session ids differ. Completed sessions are kept; the interrupted session is recomputed from its start.
  - **No score before every readout is done:** no Qwen output is compared with a label, and no score of any column is computed, until every Qwen readout of the attempt (§8's included) is complete. So a resume cannot be steered by a score. Scoring reads only the stored logits; if it is interrupted before its report is written, it is rerun on the same stored logits, with no readout recomputed.
  - **Record:** each outage is logged in the attempt's record (time, split, column, sessions kept) and reported.
  - A resume of an incomplete attempt is **not a second official run**. Starting a new attempt, or reading out again a split and column that the attempt completed, is.
  - **Why:** the attempt is several days of local MPS time, and a full val readout on this Mac has already died once without a traceback or a written output (commit `fbac9f9`, which added checkpoint and resume to `run_readout`). Without this rule, one crash would either void the single attempt or force an unregistered rerun.
- **G2** waits for the WSP-B and WSP-F archives.

## 4. Probe recipe (Qwen-probe)

Removed before G1 (Matthew, 2026-11-06): no probe column in v0.4. The heading stays so that section numbers do not move.

## 5. LoRA recipe (Qwen-LoRA)

Removed before G1 (Matthew, 2026-11-06): no LoRA column in v0.4. The heading stays so that section numbers do not move. The rank-16 LoRA parameter count that §8's "Small" yardstick uses is defined in §8.

## 6. WSP rules

The contract is **`docs/V04_WSP_KERNEL.md`, frozen at G1**, with the kernel's additional rules folded in and marked (G1): the rescaling refusal (`|c − 1| > 1e-3` and `‖R‖ < 0.5·‖(c − 1)W‖`, also on norm gains and label rows); the disclosed-overlap refusal at 0.6; at most 2 distinct non-empty edit states on the disclosed windows per phase; the weight-load quota of 0.5 of the layer bytes per phase; the read-only tools and `install.verify`; guard exit 3 and `clean_run` for launchers. The bans are v0.3's: no backward pass, autograd, optimiser, fitting to a loss, automated or numerical search over candidate weights (keep-the-best loops included) or distillation. Direct, reasoned edits only.

| | WSP-A | WSP-B | WSP-F |
| --- | --- | --- | --- |
| Server phase | `wsp-a` | `wsp-b` | `wsp-f` |
| Starts from | original weights | the archived WSP-A edit | the archived WSP-B edit |
| Session | fresh `claude -p` session (claude-fable-5-1, fallback claude-opus-5-5, effort max) | the same session, resumed (v0.3 §9.3) | the same session, resumed after the WSP-B archive |
| Forward passes | own texts | own texts + `forward_disclosed` | own texts + `score_dev` + `holdout_check` |
| Forward budget | 2,000,000 tokens, ≤ 50,000 per request | the same, new per phase | the same, new per phase (`score_dev` is counted separately) |
| Labels | none | the 200 disclosed windows | the 200 + aggregate dev scores |
| Cost cap | US$500 | US$250 | US$250 |
| Prose | EDIT.md, ITERATIONS.md | EDIT.md, REVISION.md | EDIT.md, REVISION.md |
| Result | the first valid artefact submitted as final | one submitted revision | one submitted revision, accepted or rejected (§7.3) |

- **G1 decision: the tools name later-phase mechanics, and that is accepted.** The author-readable files under `tools/` (`hn_kernel/fwd.py`, `hn_kernel/protocol.py`, `hn_kernel/client.py`, `hn_kernel/run_py.py` and the `hn_guard/` guards) name WSP-B mechanics (the `forward_disclosed` op, `disclosed_windows` requests, "WSP-B") and harness module names (`hypaqwen.wsp.*`, `agent_loop_lab` in the guards' import bans). WSP-A can read them. This is accepted, not fixed: the tools must be byte-identical across phases (the same kernel by hash, §7.3, `install.verify`), so they cannot be cut per phase, and what they reveal is that a later phase with disclosed windows may exist, which the packet already says (spec §7.8), plus module names that carry no data, labels or scores. The packet (`SPEC.md`, `AUTHOR_INSTRUCTIONS.md`) still holds none of these names (its leakage check).
- **WSP-B starts from the archived WSP-A edit** by journal: `scripts/v04_prepare_phase_b.sh` seeds the WSP-B journal with a byte-identical copy of the WSP-A journal, whose final edit hash must equal the archive's `edit_state_sha256`, and the server replays it on start (kernel contract §7). The WSP-B ledger, budget and load quota start empty.
- **Cost** is summed over every launch of a phase by session-id delta of the cumulative `total_cost_usd` (v0.3's `scripts/v03_cost.py` double-counted resumed sessions). No launch once a phase's cap is reached; `--max-budget-usd` is the remaining cap.
- **Author dir:** `../HypaNet-v04-author`, made fresh by the prepare script and never reused; layout in the kernel contract §7. Harness logs: `experiments/v0_4/logs/` (ledger `<phase>.ledger.jsonl`, kernel log `<phase>.kernel.jsonl`, journal `<phase>/journal/`, transcripts `<phase>.stream.jsonl`).
- **Packet:** `scripts/v04_build_packet.py`; packet sha256 = sha256 of its `PACKET_MANIFEST.json` bytes, recorded in the registry. It holds the `[packet]` sections of the v0.4 spec, the frozen interface, the v0.3 grammar sheet and renderer, and `packet/qwen/{config.json, tokenizer.json, tokenizer_config.json}` verbatim (sha256 as `FILE_SHA256`). No weights (the kernel serves them), no generator, oracle, dataset or eval code.

## 7. Disclosure and the labelled budgets

### 7.1 WSP-B disclosure

The same 200 windows as v0.3 Phase B (50 per oracle class, at most one per session, by ascending `sha256(f"20261105:v03-phase-b-disclosure:{session_id}:{endpoint}")`; the rule reads no model output). Ids `V001`–`V200` in key order. Per window: the rendered prefix of blocks 0..t with block t−7 marked, the oracle label, and `network_class`, the WSP-A edit's verbaliser argmax at the window (MPS, fp32). `reference_policy_class` is dropped (no reference policy). Format tag `v04-wsp-b-disclosure/1`. Nothing else is disclosed. `REVISION.md` cites the disclosed ids behind every change.

### 7.2 Probe-200w and LoRA-200w

Removed before G1 (Matthew, 2026-11-06): no probe-200w and no LoRA-200w column. The heading stays so that section numbers do not move.

### 7.3 WSP-F (rules frozen; the server ops are implemented before its launch)

**How the ops enter after G1.** One additive registry revision (§13) after the WSP-B archive and before the WSP-F launch, reason "WSP-F ops (§7.3)". It may add new files (the ops in a new module, e.g. `envs/qwen/hypaqwen/wsp/wsp_f.py`, their tests, the WSP-F prompt and settings in the author kit, and a new WSP-F launcher file if the G1 launcher does not already accept the `wsp-f` phase) and may change exactly these existing files, each named with `--allow-change`: `envs/qwen/hypaqwen/wsp/server.py` (dispatch of `score_dev` and `holdout_check` on `fwd.sock`), `envs/qwen/hypaqwen/wsp/author/protocol.py`, `envs/qwen/hypaqwen/wsp/author/client.py` and `envs/qwen/hypaqwen/wsp/author/fwd.py` (the two request types and the `./fwd` subcommands). Nothing else: not the guard, the runner, `edit_state.py`, `install.py`'s rules, the packet, the interface or any test file that existed at G1. Conditions, all checked before the WSP-F launch:

- the new ops exist only when the server's phase is `wsp-f`: in every other phase the server refuses them as unknown requests, as it does now, and a new test shows this for `wsp-a` and `wsp-b`;
- every G1 test of the kernel (`envs/qwen/tests/test_wsp_*.py`, unchanged by the revision because the registry holds their G1 hashes) passes on the revised files;
- the harness re-installs `tools/hn_kernel/` from the revised source for WSP-F (`install.verify` compares with it);
- the WSP-A and WSP-B archives record the registry `freeze_sha256` they ran under (the launcher writes it at every launch, after `--check`, into `NAME.launch.json` and `LAUNCHES.jsonl`; `scripts/v04_run_build.py` copies the registry's and every launch's into `ARCHIVE/FREEZE.json`), so what they ran is the G1 kernel by hash, whatever WSP-F later adds.

- **`score_dev(session_ids)`:** at most 128 dev sessions per call, at most **50 calls** per phase. It returns, for the current edit state and for the WSP-B edit, aggregate scores only over the sessions' gate-passing scored windows: macro-F1, per-class precision and recall, the confusion counts. No per-window label or prediction. Sessions are named by opaque ids `D00001`… of the eligible pool (dev minus the holdout, ordered by `sha256(f"20261105:v04-wsp-f-pool:{session_id}")`); the author gets no dev text. Its tokens do not draw on the 2M forward budget.
  - **Why no dev text:** the only information WSP-F adds over WSP-B is then the labelled aggregate the blog's method used.
- **Holdout:** the first 500 dev sessions by `sha256(f"20261105:v04-wsp-f-holdout:{session_id}")` that are not in the v0.3 Phase C 2,048-session budget subset (`agent_loop_lab.baselines.phase_c_v03.budget_subset`; the list is `wspf_holdout` in `datasets/v0_4/BUDGETS.json`), so disjoint from the training sessions of v0.3 C-32 to C-2048 (C-all trained on all of dev). Held only by the WSP-F server.
- **`holdout_check()`:** once, at submission. The WSP-F edit is **accepted** when its holdout macro-F1 is strictly above the WSP-B edit's; the author learns only accept or reject. A rejected WSP-F column reports the WSP-B edit with the rejection.
- **Cap:** US$250. WSP-F is reported beside WSP-B, never instead of it, and enters no hypothesis.

## 8. Q5: mechanism check and size

**Mechanism claims.** `EDIT.md`'s `## Mechanisms` table (kernel contract §6) gives per mechanism *m*: the tensors *T_m* (checkpoint names, or `label rows` for the four label rows of the embedding), a target class *c_m* (one of the four class names) and a one-sentence claim. Let *T(E)* be the set of units the edit changes (each edited tensor, with the embedding's edit as the one unit `label rows`), from `BUILD_RECORD.json`.

**Eligibility.** A row is **eligible** when *c_m* is a class name, every unit of *T_m* is in *T(E)*, *T_m* is a **strict** subset of *T(E)* (a row naming the whole edit does not isolate anything, so an edit of a single unit cannot support Q5: its mechanism check would only repeat the effect check), and *T_m* shares no unit with any other row's *T_m* (overlapping rows are all ineligible, so table order cannot pick which one is credited). Ineligible rows are reported with the reason and are not evaluated. The first 8 eligible rows in table order are evaluated; later eligible rows are reported as not evaluated. **Why 8:** each costs one counterfactual readout. **No table, an empty table, or no eligible row** is a Q5 failure for that edit, reported as such.

**Pairs.** The v0.3 counterfactual split, gate-passing contrast endpoints only (as `m4.counterfactual_consistency`). For an endpoint, the *a* side carries the benign oracle label and the *b* side the degenerate one (`synth.counterfactual.INTENDED`).

**Models.** *E* = the archived edit; *E∖m* = *E* with every unit of *T_m* reverted to the original (label rows: the four rows reverted); *O* = original weights. All on local MPS, fp32 (§3.1), inside the single sealed attempt of §3.4, before the fresh test set is opened; `scripts/v04_sealed_eval.py` runs every one of these readouts in that pass.

**Statistic (class-specific).** For target class *c*, the **units** are the sides, over all gate-passing contrast endpoints of all kinds, whose oracle label is *c* (STALLED: the *b* sides of `retry_success_vs_doomed` and `improving_vs_identical_failures`; OSCILLATING: the *b* sides of `productive_vs_abab`; PRODUCTIVE and RECOVERY: the *a* sides with that oracle label). For model *X*:

- hit rate *H_X* = the fraction of the units predicted *c*;
- false rate *F_X* = the fraction of the **other** side of the same endpoints (oracle label ≠ *c*) predicted *c*;
- score *N_X = H_X − F_X*.

So a mechanism that only makes the model say *c* everywhere gains nothing: it raises *H* and *F* together. Effect *Δ = N_E − N_O*; ablation effect *A_m = N_E − N_{E∖m}*. CIs: counterfactual pairs resampled with replacement (all of a pair's endpoints together), 10,000 resamples, seed 20261104, the same draws for *E*, every *E∖m* and *O*; 95% percentile intervals.

**A mechanism passes** when both hold:

1. the 95% CI of *Δ* lies above 0 (the edit moves class *c* the claimed way);
2. the 95% CI of *A_m − 0.5·Δ* lies above 0 (reverting *T_m* removes more than half the effect; the rule is on the interval, not the point estimate).

- **Why this statistic:** it is the `strict` counterfactual measure restricted to the claimed class, with the same-endpoint false rate subtracted, so a claim about OSCILLATING cannot pass on loops labelled STALLED, and a bias towards *c* cannot pass at all; "more than half the effect" separates a mechanism that carries the effect from one that only accompanies it.
- **Q5 is supported** for an edit when **all** of: at least one row is evaluated; every evaluated mechanism passes; and the edit is **Small** (below). Ineligible and unevaluated rows, the number passing, and every *H*, *F*, *Δ* and *A_m* with CIs are always reported.

**Size report** (from `BUILD_RECORD.json`): changed entries and their fraction of the 752,393,024 text-model parameters, per-tensor ‖ΔW‖_F/‖W‖_F, the rank of ΔW above the fp32 noise floor, the bf16-rounded changed count, and the label-row flag. **"Small"** means the edit's degrees of freedom, Σ over tensors of min(changed entries, r·(m + n − r)) (rank *r* above the noise floor, *m* × *n* with trailing dimensions flattened; 1-D tensors and the label rows count changed entries), are at most **10,223,616**. That is the trainable-parameter count of a rank-16 LoRA of Qwen3.5-0.8B (16·(m + n) per adapted *m* × *n* matrix) on these modules: `self_attn.{q_proj, k_proj, v_proj, o_proj}` of the 6 full-attention layers (3, 7, 11, 15, 19, 23); `linear_attn.{in_proj_qkv, in_proj_z, out_proj}` of the 18 Gated DeltaNet layers; and `mlp.{gate_proj, up_proj, down_proj}` of all 24 layers. It is 1.36% of the 752,393,024 text-model parameters. No LoRA is trained in v0.4: the number is only the yardstick. **Why:** a standard fine-tune of the same model is the natural yardstick, and entry counts alone would call one rank-1 MLP update (3.7M changed entries) large.

## 9. Hypotheses and the comparisons that answer them

Research targets, not gates (as v0.3). "CI > 0" below means the paired session-bootstrap 95% CI of the macro-F1 difference (§3.2) lies above 0; for a seed mean, the seeds' macro-F1 are averaged within each draw. Each hypothesis has exactly the verdicts listed; a verdict is "supported" only when its whole condition holds, and every other number is reported beside it without a verdict.

| ID | Verdict | Supported if and only if |
| --- | --- | --- |
| Q1 | Q1 | CI > 0 for Qwen-0 − constant prior (dev-majority) on the fresh test. (Reported: the same on v0.3 test and OOD; Qwen-0 − v0.3 A, the value of hand-building.) |
| Q2 | Q2 (primary) | CI > 0 for WSP-B − v0.3 B on the fresh test. |
| | Q2-OOD (secondary, separate verdict) | CI > 0 for WSP-B − v0.3 B on v0.3 OOD (v0.3 B 0.908). (Reported: the same on v0.3 test; WSP-A − v0.3 A on every set.) |
| Q3, Q4 | none: not tested | The plan's Q3 (WSP against fine-tuning the same model) and Q4 (pretraining against the trained models' OOD collapse) need the Qwen-probe and Qwen-LoRA columns, which were dropped before G1 (Matthew, 2026-11-06). v0.4 gives no Q3 or Q4 verdict. |
| Q5 | Q5-B (primary), Q5-A | §8 for the WSP-B edit and for the WSP-A edit; size report for every WSP edit. |

WSP-F enters no verdict. Every number is reported whatever its direction.

## 10. Failures

- **Build refusal:** the runner's author-facing error text only (≤ 2,000 characters), up to **3** resubmissions per phase (v0.3). A phase with no valid build after the third is reported as failed.
- **Technical continuation:** up to **9** per phase with the fixed continuation prompt (v0.3). Each counts against the phase's cost cap.
- **Guard violation, audit finding, tampering** (`KernelTampered`, a non-clean `clean_run`, a forbidden marker, an AST loop flag, more than 3 edit hashes on one text set): to the adjudicator; the run is reported with the finding, never silently dropped.
- **Cap reached:** no further launch; the last valid archive, if any, is the result.
- **Infrastructure** (server crash, machine restart): the server is restarted on the same journal and ledger, so neither the budget nor the edits reset; the outage is logged.
- **Infrastructure during the sealed attempt** (crash, power loss, restart): the attempt resumes from its checkpoints exactly as §3.4 states; each outage is logged and reported, and the resume is not a second official run.
- **A failed or poor WSP arm is a result** and is reported as such.

## 11. Not allowed after G1

- Changing the interface (prefix, suffix, label words, tokenisation, readout) or `interface.FROZEN`.
- Any tuning on val, test, OOD or counterfactual.
- Changing the WSP contract, the kernel, guards, runner, budgets, caps, disclosure or WSP-F rules, or the packet (its hash is registered). The one exception is the WSP-F ops revision, limited exactly as §7.3 states.
- Showing any val, test, OOD or counterfactual number, or any Qwen output compared with a label, to the weight author.
- Opening a sealed set before G2, or scoring one twice officially (resuming an interrupted attempt under §3.4 is not scoring it twice).
- Changing the Q5 eligibility rules, statistic or thresholds, the verdicts of §9 and their conditions, or the backends of §3.1.
- Adding or removing columns, except v0.2 B and C-all on the fresh set under the condition of §1.

## 12. Frozen files

`experiments/registry_v0_4.yaml` (`scripts/v04_freeze.py`; the lists are in `src/agent_loop_lab/v04/registry.py`):

- **Required files** (the freeze refuses without each): the v0.4 plan and spec, this file, `docs/V04_WSP_KERNEL.md`; `hypaqwen/{__init__, interface, readout, model, data, run_readout}.py` and `wsp/author/run_py.py`; `envs/qwen/{pyproject.toml, uv.lock}` and `scripts/make_author_venv.sh`; the root `pyproject.toml`, `uv.lock` and `oracle_config.v0_2.yaml`; on the root side `v04/{scoring, export}.py`, `schema.py`, `eval/{m4, metrics, collisions}.py`, `v03/{scoring, sealed, renderer}.py`, `baselines/{phase_c_v03, phase_c}.py`, `synth/counterfactual.py`; `scripts/v04_{export_tasks, score_validation, build_packet, cost}.py`, `scripts/v04_{prepare_author_dir, launch}.sh`; the author kit's WSP-A prompt, continuation and resubmission prompts, WSP-A and WSP-B settings, `py` and `fwd`; the committed manifests `datasets/v0_2/{splits, MANIFEST}.json`, `datasets/v0_3/MANIFEST_v0_3.json`, `datasets/v0_4/EXPORT_MANIFEST.json`, `datasets/v0_4/BUDGETS.json` (the 200 disclosed windows and the WSP-F holdout; its v0.3 budget lists and ES set are kept unchanged and unused), the fresh-test REGISTRATION and MANIFEST; the prompt-choice README and reports, `artifacts/reports/v03_sealed.json` and `experiments/v0_3/phase_c/README.md`.
- **Required components** (at least one file each, by pattern): the WSP-B disclosure builder, the session audit, the WSP-B prompt and the author-kit command hook.
- **Globs** (every match): all of `envs/qwen/**` except its venv and caches (any file type: code, data, tests, lock), `experiments/v0_4/author_kit/**`, `scripts/v04_*`, `src/agent_loop_lab/v04/**`. Each glob's tree is walked without following links, and any symlink in it is refused.
- **Import closure:** every `agent_loop_lab` module any frozen root-env module imports, transitively, with its package `__init__` files.
- **Values:** `interface_spec_sha256`, the model id, revision and `FILE_SHA256`, and `packet_sha256`.

`--check` also runs the v0.3 registry check, so the v0.3 artefacts re-scored here and the v0.3 renderer in the packet stay verified.

## 13. Revisions

**Procedure.** Before any launch or scoring run has used the registry, `--write --force` may replace it (a wrong G1 write). After that, the only change is an additive revision, `scripts/v04_freeze.py --revise --reason TEXT [--allow-change PATH ...]`. It refuses unless the registry is intact, refuses a removed file and any change of the interface hash, the model pins or the packet hash, and refuses a change to any existing file not named with `--allow-change` (and a named file that did not change). It appends `{date, reason, previous_freeze_sha256, globs_before, added, changed}` to `revisions`; `--check` undoes the revisions one by one and verifies each step's `previous_freeze_sha256`, back to the G1 record. Each revision also gets a dated entry below and a line in the report.

**Pre-registered revisions** (the only ones planned):

1. `scripts/v04_sealed_eval.py` (§3.4, §8): new files only, no `--allow-change`.
2. The WSP-F ops (§7.3): new files plus the four kernel files listed there.
3. The CPU replay tool (§3.3): new files only, no `--allow-change`; before the replay is run. It may be committed in the same revision as 1.

Any other revision is a deviation from this pre-registration and is reported as one.

**Log.** None yet.
