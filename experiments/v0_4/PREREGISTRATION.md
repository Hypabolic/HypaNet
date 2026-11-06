# v0.4 pre-registration (gate G1)

_Written 2026-11-05, before any Qwen output has been compared with a label and before the weight author's session exists; revised twice on 2026-11-06, still before G1: when Matthew dropped the training baselines and all cloud compute, and when he took sealed scoring out of v0.4 (Authority, below). Plan: `agent_doom_loop_detector_plan_v0_4.md` (Matthew accepted every recommendation, including the "Revisions after M0"). Every frozen file, the interface hash, the model hashes and the packet hash are recorded in `experiments/registry_v0_4.yaml` by `scripts/v04_freeze.py`; `scripts/v04_freeze.py --check` must pass before every launch and every scoring run. Where the plan left a number open, this file fixes it and gives the reason in one line ("Why")._

**Authority.** Decided under the delegated authority of the plan (Matthew, 2026-11-05: "proceed on recommendations"), except two decisions Matthew took directly on 2026-11-06, before G1, each overriding the plan's accepted recommendations:

1. "Drop the training baselines". v0.4 has no Qwen-probe and no Qwen-LoRA column at any budget, and no cloud GPU: nothing is rented, and every Qwen run is on this Mac (§3.1). The plan's Q3 and Q4 are therefore not tested (§9). §4, §5 and §7.2 stay as stubs so that section numbers do not move.
2. "Val only, no sealed" (his context: "I need a quick answer to how the base qwen vs modified qwen performs, not 90 hours"). v0.4 has **no sealed scoring at all**. The fresh test set, v0.3 test, v0.3 OOD and the counterfactual split are not opened or scored in v0.4; they stay sealed and available to a later version (§3.4). The v0.4 result is base Qwen (Qwen-0) against the WSP edits on val, harness-only, with a pre-registered quick look on a fixed 1,000-session val subset first (§3.5). Val numbers are still never shown to the weight author. Q2-OOD and the Q5 verdict are therefore not tested either (§8, §9), and the val results are weaker evidence than a sealed result would be (§9).

Nothing below may change after this file is committed with the registry. A change needs a dated entry in §13, an additive registry revision (`scripts/v04_freeze.py --revise`, §13) and a line in the report.

## 1. Arms and columns

Every Qwen arm starts from Qwen3.5-0.8B at revision `2fc06364` (`hypaqwen.MODEL_REVISION`, every file's sha256 in `hypaqwen.model.FILE_SHA256`) and reads out through the same verbaliser (§2).

| Column | Built by | Labelled data | Answers |
| --- | --- | --- | --- |
| **Qwen-0** | nobody: released weights, frozen interface | 0 | Q1; the base of every quick look (§3.5) |
| **WSP-A** | the weight author, sealed session, no labels | 0 | reported: WSP-A − Qwen-0, WSP-A − v0.3 A; size report (§8) |
| **WSP-B** | the same author, one revision after the 200 disclosed windows | 200 windows | Q2; reported: WSP-B − Qwen-0; size report (§8) |
| **WSP-F** (separate column, never instead of WSP-B) | WSP-B plus a labelled dev budget (§7.3) | 200 windows + 50 × 128 dev sessions scored in aggregate | reported beside WSP-B, no verdict |

- **No trained Qwen column** (Matthew, 2026-11-06): no Qwen-probe and no Qwen-LoRA at any budget (32, 128, 512, 2,048, all or 200w). The trained columns in the report are the v0.3 C columns below; nothing new is trained.
- **v0.3 columns, re-scored on val** in the frozen v0.3 environment on this Mac, with no tolerance (plan revision 10): **v0.3 A** and **v0.3 B** (the archived networks), **every v0.3 C seed** (C-32 × 3, C-128 × 3, C-512 × 3, C-2048, C-all) and the two **constant priors** (dev-majority, always-PRODUCTIVE).
  - **What v0.3 code can produce on val without a sealed set:** `scripts/v04_v03_val_columns.py` (`agent_loop_lab.v04.v03_val`) runs v0.3's own path, the path of the unofficial v0.3 val dry run without its counterfactual pass: `v03.sealed.build_split("val")` (val only; its rows must hash to the v0.3 val `rows_sha256`, so they are the bridge's rows in order), `load_columns()` (every archive and Phase C run sha-checked) and `network_predictions`. The file holds the classes per val row, no labels and no scores, and enters the same `score_columns` call as the Qwen columns (`--v03`), so every paired comparison shares the bootstrap draws (§3.2). It opens no sealed set and no counterfactual file.
  - **Per-column reproduction rule** (fixed here before the tool has run on the full C blocks; no tolerance). Each v0.3 network column (v0.3 A, v0.3 B, each C seed) whose every val block (overall and per family, both populations) equals `v03_sealed.json` exactly is written and enters the report with paired CIs. A column that differs anywhere is not written: the tool names it with the path of its first difference, and the report cites its val blocks from `v03_sealed.json` beside, with no paired CI (`v03_cited`). A C group (seed mean) is written only when all its seeds are written and its group blocks equal the record; otherwise it is cited the same way. Nothing is written if a prior differs (the priors are the bridge's own, so a difference means the bridge's rows or gate differ from v0.3's) or if no network reproduces. **If v0.3 B is not reproduced, Q2 has no verdict** (§9: "not tested: v0.3 B did not reproduce on val"), and WSP-B's val macro-F1 is reported beside v0.3 B's cited one with no paired CI; likewise WSP-A − v0.3 A and Qwen-0 − v0.3 A become cited comparisons without a CI if v0.3 A is not reproduced. Q1 does not depend on any v0.3 column.
  - **The bridge re-checks the file** before it enters any report (quick look included): it re-scores the file's columns and the priors on the full val rows, both populations, and refuses unless every block equals `v03_sealed.json` `splits.val.populations` exactly. It does not take the file's own reproduction record on trust.
  - **The priors** are built by the bridge for every report (quick look included), as v0.3 built them: the dev-majority class recorded in `v03_sealed.json` `dev_prior` (PRODUCTIVE), and PRODUCTIVE, on every gate-passing window. The v0.3 tool checks them against the v0.3 val blocks too.
  - **Not re-run:** the v0.3 A and B reference policies (they are not v0.4 comparison columns, and each would add hours of sandboxed policy calls); their val numbers are cited from `v03_sealed.json`.
- **Cited, beside, no paired CI:** the rules engine, v0.2 B and v0.2 C-all, from the val block of `artifacts/reports/m4_sealed_v02.json` (all 4,004 val sessions, so beside `all_val` only). The rules engine is a v0.2 column: its val predictions would need the spent v0.2 sealed driver (`scripts/m4_sealed_eval_v02.py`), which v0.4 does not run. The sealed v0.3 and v0.2 numbers (v0.3 test and OOD) are quoted as background only: no v0.4 column has a sealed number to set beside them (§3.4).
- For the v0.3 C budgets with 3 seeds (32, 128, 512), every seed is reported plus the mean ± sample sd and the seed-mean CI (`m4.score_columns` groups), as v0.3 did.

## 2. Interface

- **Frozen text:** `envs/qwen/hypaqwen/interface.py` `FROZEN`, candidate `w1-state-full`, `spec_sha256()` = **`5922eb4500122921ac7d07e3d35599b95cb10b462c4f50b89e5d6bb989437879`** (prefix, block template, suffix and label words). `scripts/v04_freeze.py` recomputes it from the file and refuses a freeze with any other value.
- **Label words** (class order PRODUCTIVE, RECOVERY, STALLED, OSCILLATING): ` productive`, ` recovering`, ` stalled`, ` looping`, each one Qwen token with its leading space. Answer cue `State:`, the `full` instruction style, the Qwen chat format with thinking off.
- **How it was chosen:** without labels, by the pre-registered rule of `experiments/v0_4/prompt_choice/README.md` as amended by its **Revision 1** (median label-word mass × normalised entropy of the predicted-class shares; reports `artifacts/reports/v04_prompt_choice.json` and `artifacts/reports/v04_prompt_choice_revision1.json`). The first run's choice (`w3-state-full`, a constant STALLED classifier) is reported as run.
- **Readout** (`hypaqwen/readout.py`): one pass per session, the cache branched after each block to run only the suffix; the class is the first-index argmax over the four label-word logits (the v0.3 tie rule). It equals a separate prompt per window (checked on MPS and CPU). The edited label rows are read live, so a WSP edit of the tied embedding reaches the readout.
- **Fixed for every arm.** Nobody may change the prefix, suffix, label words or tokenisation after G1. The weight author may change weights only (`docs/V04_WSP_KERNEL.md` §2).
- **Windows, scored mask, labels, populations:** v0.3's, unchanged. val 87,176 windows, 4,004 sessions (rows checked against v0.3's `rows_sha256` `cdf6cce8…`); the quick-look subset is 1,000 of those sessions (§3.5). No other split is scored in v0.4 (§3.4).

## 3. Readout and scoring on val; sealed sets not used

### 3.1 Backends

Everything runs on this Mac. Nothing is rented.

| Run | Backend | Settings |
| --- | --- | --- |
| Harness-only val readouts of Qwen-0, WSP-A, WSP-B, WSP-F (the quick look of §3.5 and full val); the weight author's forward server | local MPS on this Mac (Apple M4 Pro, 24 GB) | fp32 weights and activations, SDPA attention (`hypaqwen.model.ATTN_IMPLEMENTATION`), the torch implementation of Gated DeltaNet (no `flash-linear-attention` or `causal-conv1d` installed), the `envs/qwen/uv.lock` wheels |
| CPU replay check (§3.3) | this Mac's CPU | fp32, the same wheels and implementations |
| v0.3 columns on val (§1) | this Mac's CPU, frozen v0.3 environment | v0.3's settings (2 torch threads per worker), as its val dry run |

- **Why one backend for every Qwen column, the author's server included:** it removes the backend as a source of argmax flips between the val readouts and the author's own forward passes.
- **Why local:** nothing is rented (Matthew, 2026-11-06), and the Mac's GPU runs the readout at about 445 tok/s (plan revision 1): about 5 h per Qwen column over val, about 1.25 h over the quick-look subset.
- **Determinism is not claimed on MPS.** torch's deterministic-algorithms mode does not guarantee bitwise-repeatable results on MPS, so no MPS run relies on it. The CPU replay (§3.3) is the numerical check.
- **Readouts:** `hypaqwen.run_readout --split val` on MPS, `--edit ARCHIVE` for an edit, `--sessions datasets/v0_4/VAL_QUICK.txt` for the quick look. A readout interrupted by a crash or restart resumes from its checkpoint (same arguments, same session ids; completed sessions kept). The first complete readout of a column on a coverage (quick or full) is the one scored; a column is never read out again to change a score.

### 3.2 Scoring on val

- **Bridge:** `src/agent_loop_lab/v04/scoring.py` (`scripts/v04_score_validation.py`) in the frozen v0.3 environment. Per column it checks the rows, the session ids and row positions, refuses non-finite logits, takes the first-index argmax and scores with `eval.m4.score_columns` (as v0.3 scored Phase C) with no fidelity pairs: **H2 is not applicable** without a reference policy. It adds the two constant priors and, with `--v03`, the v0.3 columns of §1 (re-checked, §1). It refuses unless `experiments/v0_4/fresh_test/REGISTRATION.json` and `datasets/v0_4/fresh_test/MANIFEST_fresh_test.json` are committed unchanged and the manifest cites that registration (`scoring.fresh_test_committed`); it reads only those two committed files, never the fresh-test data files, and records the manifest's sha256 in every report.
- **Registry:** `scripts/v04_score_validation.py` runs the `--check` of the front matter itself (`registry.check_registry`) and refuses any report, official or not, unless it passes; the report records the registry's `freeze_sha256` (after revisions), its G1 `freeze_sha256` and the number of revisions (`inputs.registry`).
- **Populations:** `excluding_disclosed` (the 200 Phase B disclosed sessions removed) is **primary**: its full-val numbers are the result of record. `all_val` is reported alongside.
- **Metrics:** v0.2 §10.1 as v0.3: macro-F1 (headline), per-class precision and recall, false-loop alarm rate, confusion, session-bootstrap 95% CIs (10,000 resamples, seed 20261104, `eval.metrics`), overall and per family. Counterfactual consistency is not computed: it needs the counterfactual split, which stays sealed (§3.4).
- **Paired comparisons** (§9): the difference of two columns' macro-F1 over the same bootstrap draws, 95% percentile interval, overall. `score_columns` shares the draws across the columns of one call but reports no difference, so the bridge computes it (`scoring.paired_macro_f1`) from the same session counts, the same session order and `metrics.bootstrap_distribution` with the same seed, that is, from the very draws behind each column's CI. It is reported as `paired_macro_f1` in each population. The §9 pairs are built in (`scoring.default_pairs`), each computed whenever both its columns are in the report, under these exact column names: every Qwen column `− Qwen-0`; `Qwen-0 − prior dev-majority`; `WSP-B − v0.3 B network`; `WSP-A − v0.3 A network`; `Qwen-0 − v0.3 A network`; `WSP-F − WSP-B`. `--diff A,B` adds a pair and never removes a built-in one.
- **Coverage:** all Qwen columns of one report cover the same sessions (all full val, or all the same subset); a mix is refused. Full val is written to `artifacts/reports/v04_val_<arm>.json`, the quick look to `artifacts/reports/v04_val_quick_<arm>.json`.
- **Val numbers are never shown to the weight author**, in any phase, the quick look included.

### 3.3 CPU replay check

On this Mac, the 100 val sessions with the smallest `sha256("20261105:v04-cpu-replay:{session_id}")` are read out on CPU for Qwen-0 and WSP-B, and the window-level argmax agreement CPU vs MPS is reported. No labels are read. Being label-free, it stays meaningful with val as the only scored split: it checks the backend of every MPS column.

- **Why these two columns:** one per verbaliser path, the unedited weights (Qwen-0) and an edited model (WSP-B, which carries the WSP-A edit and its revision). CPU is slow here for the readout's many short calls (under 10 tok/s measured with another job running), so each column is several hours of CPU.
- Agreement below 99.5% of windows is reported as a numerical-backend caveat beside the affected column. It never changes a score.
- **Code:** no G1 file selects the 100 sessions or computes the agreement. The replay tool (the selection by the key above, the CPU readouts through `hypaqwen.run_readout` on its session subset, and the agreement report; no labels) is written after G1 against this section only, with a CPU-only test, and committed with an additive registry revision before the replay is run (§13, pre-registered revision 3: new files only, no `--allow-change`, every G1 hash unchanged). **Why after G1:** the replay needs the WSP-B archive, which exists only after G1, and the rule it implements is fixed here.

### 3.4 Sealed sets: not used in v0.4

Matthew, 2026-11-06: "Val only, no sealed".

- **What stays sealed:** the fresh test set (`experiments/v0_4/fresh_test/REGISTRATION.json`, seed 20261106; `datasets/v0_4/fresh_test/MANIFEST_fresh_test.json`, 3,996 sessions), v0.3 test, v0.3 OOD and the v0.3 counterfactual split. v0.4 does not open, list, decode, hash or score any of them, for any column (Qwen, WSP or v0.3). The counterfactual split, which v0.3 treated as unsealed, is sealed here too. v0.4 reads only the fresh set's committed `REGISTRATION.json` and `MANIFEST_fresh_test.json` (the bridge checks they are committed unchanged, §3.2; the freeze hashes them as required files, §12); no v0.4 step needs the data files to be present or intact.
- **Available to a later version.** No v0.4 step counts as a use of them: no Qwen output and no WSP edit meets a label on them, so a later version may score the archived v0.4 edits on them under its own pre-registration.
- **Removed:** the sealed attempt and its order, G2, the sealed-eval script (pre-registered revision 1 is gone, §13), the interrupted-attempt rule and the re-scoring of v0.3 artefacts on the fresh set.
- **Why:** a quick answer to base Qwen against modified Qwen. The sealed attempt was several days of MPS time; four full val readouts are about 20 h, and the quick look gives a first answer within hours of each archive.

### 3.5 Quick look (VAL_QUICK)

- **Subset:** `datasets/v0_4/VAL_QUICK.txt`, the 1,000 val sessions with the smallest `sha256("20261106:v04-val-quick:{session_id}")` after the 200 WSP-B disclosed sessions are removed, one id per line in key order (`scripts/v04_val_quick_subset.py`; it reads only the val task export's session ids). File sha256 **`b77f74150ecc19edf51ed57b5bc77f9e9eacd51b1992c0467ab4a4bce6cfd556`**, frozen as a required file (§12). It was fixed before any edit existed, so it cannot favour one.
- **Order, per edit:** every edit (WSP-A, then WSP-B, then WSP-F) is read out **first** on the subset (`run_readout --split val --edit ARCHIVE --sessions datasets/v0_4/VAL_QUICK.txt`) and scored **paired against Qwen-0 on the same 1,000 sessions** in one report (Qwen-0's own subset readout; the same bootstrap draws): WSP − Qwen-0 macro-F1 with its 95% CI, and every column's metrics and CIs, written to `artifacts/reports/v04_val_quick_<arm>.json` and reported as the **quick look**. The v0.3 columns and the priors join it, restricted to the subset's windows. Then the edit's full val readout follows, scored to `artifacts/reports/v04_val_<arm>.json`.
- **The bridge accepts a subset column** only when its `meta["subset"]["sha256"]` equals the sha256 of `VAL_QUICK.txt` and its session ids are exactly that file's ids in export order; the rows are restricted to those sessions, and the report records the subset (`file`, `sha256`, `sessions`).
- **Result of record:** the full-val `excluding_disclosed` numbers. The quick look is reported beside, never instead of, the full val result, and enters no verdict of §9. The subset holds no disclosed session, so on it `excluding_disclosed` and `all_val` coincide.
- **Why:** Matthew wants the base-against-modified answer within hours ("I need a quick answer to how the base qwen vs modified qwen performs, not 90 hours"). A subset readout is about a quarter of a full one, and the subset was chosen by hash before any edit existed.

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

**Not tested in v0.4.** The mechanism statistic below needs the counterfactual split, which stays sealed (§3.4), so v0.4 gives **no Q5 verdict** and runs none of its counterfactual readouts (*E*, *E∖m*, *O*). For every WSP edit v0.4 reports, without a verdict: the **size report** (below, with the "Small" definition and whether the edit is Small) and the **Mechanisms table's eligibility** (each row eligible or not, with the reason, and which rows would be evaluated). The definitions below stay as written, so that this report is defined and a later version can run the statistic on the sealed split; the parts marked "not run in v0.4" are not run.

**Mechanism claims.** `EDIT.md`'s `## Mechanisms` table (kernel contract §6) gives per mechanism *m*: the tensors *T_m* (checkpoint names, or `label rows` for the four label rows of the embedding), a target class *c_m* (one of the four class names) and a one-sentence claim. Let *T(E)* be the set of units the edit changes (each edited tensor, with the embedding's edit as the one unit `label rows`), from `BUILD_RECORD.json`.

**Eligibility.** A row is **eligible** when *c_m* is a class name, every unit of *T_m* is in *T(E)*, *T_m* is a **strict** subset of *T(E)* (a row naming the whole edit does not isolate anything, so an edit of a single unit cannot support Q5: its mechanism check would only repeat the effect check), and *T_m* shares no unit with any other row's *T_m* (overlapping rows are all ineligible, so table order cannot pick which one is credited). Ineligible rows are reported with the reason and are not evaluated. The first 8 eligible rows in table order are evaluated; later eligible rows are reported as not evaluated. **Why 8:** each costs one counterfactual readout. **No table, an empty table, or no eligible row** is a Q5 failure for that edit, reported as such (in v0.4: reported as such, with no verdict).

**Pairs** (not run in v0.4). The v0.3 counterfactual split, gate-passing contrast endpoints only (as `m4.counterfactual_consistency`). For an endpoint, the *a* side carries the benign oracle label and the *b* side the degenerate one (`synth.counterfactual.INTENDED`).

**Models** (not run in v0.4). *E* = the archived edit; *E∖m* = *E* with every unit of *T_m* reverted to the original (label rows: the four rows reverted); *O* = original weights. A later version that runs the statistic reads them out on local MPS, fp32 (§3.1), under its own sealed pre-registration.

**Statistic (class-specific; not run in v0.4).** For target class *c*, the **units** are the sides, over all gate-passing contrast endpoints of all kinds, whose oracle label is *c* (STALLED: the *b* sides of `retry_success_vs_doomed` and `improving_vs_identical_failures`; OSCILLATING: the *b* sides of `productive_vs_abab`; PRODUCTIVE and RECOVERY: the *a* sides with that oracle label). For model *X*:

- hit rate *H_X* = the fraction of the units predicted *c*;
- false rate *F_X* = the fraction of the **other** side of the same endpoints (oracle label ≠ *c*) predicted *c*;
- score *N_X = H_X − F_X*.

So a mechanism that only makes the model say *c* everywhere gains nothing: it raises *H* and *F* together. Effect *Δ = N_E − N_O*; ablation effect *A_m = N_E − N_{E∖m}*. CIs: counterfactual pairs resampled with replacement (all of a pair's endpoints together), 10,000 resamples, seed 20261104, the same draws for *E*, every *E∖m* and *O*; 95% percentile intervals.

**A mechanism passes** when both hold:

1. the 95% CI of *Δ* lies above 0 (the edit moves class *c* the claimed way);
2. the 95% CI of *A_m − 0.5·Δ* lies above 0 (reverting *T_m* removes more than half the effect; the rule is on the interval, not the point estimate).

- **Why this statistic:** it is the `strict` counterfactual measure restricted to the claimed class, with the same-endpoint false rate subtracted, so a claim about OSCILLATING cannot pass on loops labelled STALLED, and a bias towards *c* cannot pass at all; "more than half the effect" separates a mechanism that carries the effect from one that only accompanies it.
- **Q5 would be supported** for an edit when **all** of: at least one row is evaluated; every evaluated mechanism passes; and the edit is **Small** (below). Ineligible and unevaluated rows, the number passing, and every *H*, *F*, *Δ* and *A_m* with CIs would be reported. In v0.4 no row is evaluated and no *H*, *F*, *Δ* or *A_m* is computed.

**Size report** (reported in v0.4 for every WSP edit) (from `BUILD_RECORD.json`): changed entries and their fraction of the 752,393,024 text-model parameters, per-tensor ‖ΔW‖_F/‖W‖_F, the rank of ΔW above the fp32 noise floor, the bf16-rounded changed count, and the label-row flag. **"Small"** means the edit's degrees of freedom, Σ over tensors of min(changed entries, r·(m + n − r)) (rank *r* above the noise floor, *m* × *n* with trailing dimensions flattened; 1-D tensors and the label rows count changed entries), are at most **10,223,616**. That is the trainable-parameter count of a rank-16 LoRA of Qwen3.5-0.8B (16·(m + n) per adapted *m* × *n* matrix) on these modules: `self_attn.{q_proj, k_proj, v_proj, o_proj}` of the 6 full-attention layers (3, 7, 11, 15, 19, 23); `linear_attn.{in_proj_qkv, in_proj_z, out_proj}` of the 18 Gated DeltaNet layers; and `mlp.{gate_proj, up_proj, down_proj}` of all 24 layers. It is 1.36% of the 752,393,024 text-model parameters. No LoRA is trained in v0.4: the number is only the yardstick. **Why:** a standard fine-tune of the same model is the natural yardstick, and entry counts alone would call one rank-1 MLP update (3.7M changed entries) large.

## 9. Hypotheses and the comparisons that answer them

Research targets, not gates (as v0.3). Every verdict is on **full val, `excluding_disclosed`** (§3.2). "CI > 0" below means the paired session-bootstrap 95% CI of the macro-F1 difference (§3.2) lies above 0; for a seed mean, the seeds' macro-F1 are averaged within each draw. Each hypothesis has exactly the verdicts listed; a verdict is "supported" only when its whole condition holds, and every other number is reported beside it without a verdict.

| ID | Verdict | Supported if and only if |
| --- | --- | --- |
| Q1 | Q1 | CI > 0 for `Qwen-0 − prior dev-majority` on val `excluding_disclosed`. (Reported: `Qwen-0 − v0.3 A network`, the value of hand-building.) |
| Q2 | Q2 (primary) | CI > 0 for `WSP-B − v0.3 B network` (the archived v0.3 B network, re-scored on val, §1) on val `excluding_disclosed`. If v0.3 B does not reproduce on val (§1), Q2 is **not tested** and reported as such, with the two macro-F1 values side by side and no CI. |
| | reported, no verdict | `WSP-A − Qwen-0`, `WSP-B − Qwen-0`, `WSP-A − v0.3 A network`; WSP-F beside WSP-B (`WSP-F − Qwen-0`, `WSP-F − WSP-B`); every column on `all_val`; every quick look (§3.5). |
| Q2-OOD | none: not tested | It needs v0.3 OOD, which stays sealed (§3.4). |
| Q3, Q4 | none: not tested | The plan's Q3 (WSP against fine-tuning the same model) and Q4 (pretraining against the trained models' OOD collapse) need the Qwen-probe and Qwen-LoRA columns, which were dropped before G1 (Matthew, 2026-11-06). v0.4 gives no Q3 or Q4 verdict. |
| Q5 | none: not tested | The mechanism statistic needs the counterfactual split, which stays sealed (§3.4, §8). The size report and the Mechanisms table's eligibility are reported for every WSP edit. |

WSP-F enters no verdict. Every number is reported whatever its direction.

**Caveat: val is not a sealed set.** These verdicts are weaker evidence than a sealed result, and the report says so beside each one:

- the harness has read out models on val (the Qwen-0 readouts, and each edit's quick look before its full readout);
- v0.3 used val for its Phase C early stopping and for the Phase B disclosure, and the v0.3 A, B and C val numbers have been known to us who design v0.4 since v0.3 (the val dry run of plan revision 10);
- WSP-B sees 200 val windows (the same disclosure as v0.3 B); `excluding_disclosed` removes their 200 sessions, but the other val sessions come from the same generator and families as those sessions.

No weight author has seen a val number, so the WSP columns were not fitted to val, but a sealed evaluation by a later version (§3.4) is what would turn a val result into a test result.

## 10. Failures

- **Build refusal:** the runner's author-facing error text only (≤ 2,000 characters), up to **3** resubmissions per phase (v0.3). A phase with no valid build after the third is reported as failed.
- **Technical continuation:** up to **9** per phase with the fixed continuation prompt (v0.3). Each counts against the phase's cost cap.
- **Guard violation, audit finding, tampering** (`KernelTampered`, a non-clean `clean_run`, a forbidden marker, an AST loop flag, more than 3 edit hashes on one text set): to the adjudicator; the run is reported with the finding, never silently dropped.
- **Cap reached:** no further launch; the last valid archive, if any, is the result.
- **Infrastructure** (server crash, machine restart): the server is restarted on the same journal and ledger, so neither the budget nor the edits reset; the outage is logged.
- **Infrastructure during a val readout** (crash, power loss, restart): the readout resumes from its checkpoint as §3.1 states; each outage is logged and reported.
- **A failed or poor WSP arm is a result** and is reported as such.

## 11. Not allowed after G1

- Changing the interface (prefix, suffix, label words, tokenisation, readout) or `interface.FROZEN`.
- Any tuning on val, test, OOD or counterfactual.
- Changing the WSP contract, the kernel, guards, runner, budgets, caps, disclosure or WSP-F rules, or the packet (its hash is registered). The one exception is the WSP-F ops revision, limited exactly as §7.3 states.
- Showing any val, test, OOD or counterfactual number, or any Qwen output compared with a label, to the weight author.
- **Opening a sealed set in v0.4 at all** (fresh test, v0.3 test, v0.3 OOD, counterfactual): no opening, listing, decoding, readout or scoring, for any column. v0.4 reads only the fresh set's committed REGISTRATION and MANIFEST (§3.2, §3.4).
- Changing the quick-look subset or its rule (§3.5), or making a quick look the result of record.
- Changing the Q5 eligibility rules, statistic, thresholds or size definition, the verdicts of §9 and their conditions, or the backends of §3.1.
- Adding or removing columns.

## 12. Frozen files

`experiments/registry_v0_4.yaml` (`scripts/v04_freeze.py`; the lists are in `src/agent_loop_lab/v04/registry.py`):

- **Required files** (the freeze refuses without each): the v0.4 plan and spec, this file, `docs/V04_WSP_KERNEL.md`; `hypaqwen/{__init__, interface, readout, model, data, run_readout}.py` and `wsp/author/run_py.py`; `envs/qwen/{pyproject.toml, uv.lock}` and `scripts/make_author_venv.sh`; the root `pyproject.toml`, `uv.lock` and `oracle_config.v0_2.yaml`; on the root side `v04/{scoring, export}.py`, `schema.py`, `eval/{m4, metrics, collisions}.py`, `v03/{scoring, sealed, renderer}.py`, `baselines/{phase_c_v03, phase_c}.py`; `scripts/v04_{export_tasks, score_validation, build_packet, cost}.py`, `scripts/v04_{prepare_author_dir, launch}.sh`; the author kit's WSP-A prompt, continuation and resubmission prompts, WSP-A and WSP-B settings, `py` and `fwd`; the committed manifests `datasets/v0_2/{splits, MANIFEST}.json`, `datasets/v0_3/MANIFEST_v0_3.json`, `datasets/v0_4/EXPORT_MANIFEST.json`, `datasets/v0_4/BUDGETS.json` (the 200 disclosed windows and the WSP-F holdout; its v0.3 budget lists and ES set are kept unchanged and unused), **`datasets/v0_4/VAL_QUICK.txt`** (the quick-look subset, §3.5), the fresh-test REGISTRATION and MANIFEST (committed JSON only; the bridge checks they are unchanged and never reads the data files); the prompt-choice README and reports, `artifacts/reports/v03_sealed.json` and `experiments/v0_3/phase_c/README.md`.
- **Required components** (at least one file each, by pattern): the WSP-B disclosure builder, the session audit, the WSP-B prompt and the author-kit command hook.
- **Globs** (every match): all of `envs/qwen/**` except its venv and caches (any file type: code, data, tests, lock), `experiments/v0_4/author_kit/**`, `scripts/v04_*` (so `scripts/v04_val_quick_subset.py` and `scripts/v04_v03_val_columns.py`), `src/agent_loop_lab/v04/**` (so `v04/v03_val.py`). Each glob's tree is walked without following links, and any symlink in it is refused.
- **Import closure:** every `agent_loop_lab` module any frozen root-env module imports, transitively, with its package `__init__` files.
- **Values:** `interface_spec_sha256`, the model id, revision and `FILE_SHA256`, and `packet_sha256`.

`--check` also runs the v0.3 registry check, so the v0.3 artefacts re-scored here on val and the v0.3 renderer in the packet stay verified.

## 13. Revisions

**Procedure.** Before any launch or scoring run has used the registry, `--write --force` may replace it (a wrong G1 write). After that, the only change is an additive revision, `scripts/v04_freeze.py --revise --reason TEXT [--allow-change PATH ...]`. It refuses unless the registry is intact, refuses a removed file and any change of the interface hash, the model pins or the packet hash, and refuses a change to any existing file not named with `--allow-change` (and a named file that did not change). It appends `{date, reason, previous_freeze_sha256, globs_before, added, changed}` to `revisions`; `--check` undoes the revisions one by one and verifies each step's `previous_freeze_sha256`, back to the G1 record. Each revision also gets a dated entry below and a line in the report.

**Pre-registered revisions** (the only ones planned; the numbers are kept so that references do not move):

1. Removed before G1 (Matthew, 2026-11-06: "Val only, no sealed"): there is no sealed-eval script, because v0.4 scores no sealed set (§3.4).
2. The WSP-F ops (§7.3): new files plus the four kernel files listed there.
3. The CPU replay tool (§3.3): new files only, no `--allow-change`; before the replay is run.

Any other revision is a deviation from this pre-registration and is reported as one.

**Log.**

- **D1, 2026-11-06, deviation (not pre-registered).** Matthew chose to go straight to WSP-B without waiting for WSP-A's full val readout (about 7.5 h): "I'd rather go straight to WSP-B and get a number for that", then "Shortcut, report deviation". The §7.1 `network_class` now comes from a WSP-A readout of exactly the 200 disclosed sessions (`datasets/v0_4/DISCLOSED_SESSIONS.txt`, sha256 `778ed497…`, written by `scripts/v04_disclosed_sessions.py` from `BUDGETS.json` `disclosed_200w`), not from a full val readout. Each session is read out on its own, so the class at every disclosed window is the one a full val readout would give. What is disclosed, the selection, the prefixes and the format are unchanged. Changed: `src/agent_loop_lab/v04/disclosure.py` (`check_column` also accepts that subset, pinned by sha256) and `scripts/v04_build_disclosure.py` (rows outside the subset get no prediction). Added: `scripts/v04_disclosed_sessions.py`. WSP-A's full val readout, its result of record, still follows. Follow-on (revision 2, same day): the build runner refused the WSP-B archive because revision 1 changed existing files, and `handover.kernel_freezes` counted any such revision as leaving the G1 kernel (a rule written for the WSP-F ops revision). `src/agent_loop_lab/v04/handover.py` now keeps the chain through a revision that changes only `KERNEL_NEUTRAL_FILES` (this file, `disclosure.py`, `scripts/v04_build_disclosure.py`, `handover.py`). The session, model server, guard and hook do not run any of these files. Any other changed file still ends the chain.
- **D2, 2026-11-06, deviation (not pre-registered).** Readouts can run as parallel MPS processes. Matthew asked for a parallelism benchmark ("GPU only please") and then chose 3 workers. On 48 val sessions, 1/2/3/4 processes gave 457/733/841/915 tok/s; free memory was 72/54/39/14% at the lowest; every parallel run's logits were bit-identical to one process. `scripts/v04_parallel_readout.py` (new file, registry revision 3, additive) deals the sessions round-robin in export order to N unchanged `hypaqwen.run_readout --sessions SHARD` workers. It can seed from an interrupted single-process run's checkpoint, and it merges the shards back into one column in export order. Each shard's rows must equal its sessions' export `window_pos`, and the shards' metas must agree. The merged meta adds `parallel` (workers, shard sha256s, seed). Merging the benchmark's 3- and 4-process shards gave exactly the single-process column. An independent review found no merge defect. It found four gaps in re-runs, all fixed before the freeze: each kept shard is now checked against the requested device, dtype, edit and its shard list; the seed is copied into the shards dir; a lock keeps a second run out; and a run with no worker shards is refused cleanly. The readout itself, the backend (MPS, fp32) and the scoring are unchanged.
