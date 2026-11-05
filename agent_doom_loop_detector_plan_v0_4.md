# HypaNet v0.4 plan: Qwen3.5-0.8B and weight-space programming

5 Nov 2026 · Matthew Gribben. This is a draft plan, not yet a frozen spec. The live version, with comments and diagrams, is at https://claude.ai/artifact/1xPtxSME6hek7gN5o3E1t2

## Summary

v0.4 asks whether an LLM does better by **editing a pretrained model's weights** or by **building a network from zero**. Fable will program Qwen3.5-0.8B to detect agent doom loops. That edited model is then compared with:

- v0.3's fully hand-built 2.1M-parameter transformer;
- Qwen used as it comes;
- Qwen trained the usual way, with LoRA on the same label budgets as v0.3 Phase C.

Nothing about the task changes from v0.3: same rendered logs, labels, windows and metrics, so every new number lines up with the v0.3 table.

The headline questions:

1. Does a pretrained model's prior knowledge of logs and code help a hand-programmer?
2. Does weight-space programming beat ordinary fine-tuning at the same label budget?
3. Does either generalise out of distribution as well as the from-scratch v0.3 network (OOD macro-F1 0.908)?

As in v0.3, everything is pre-registered before it runs, and the weight author is a fresh, sealed Fable session.

## Where v0.3 left us

The bar for every v0.4 arm is v0.3 B: test 0.957 and OOD 0.908, from 200 labelled windows. Every model trained on the raw logs lost most of its accuracy out of distribution.

Sealed macro-F1, one attempt each (`artifacts/reports/v03_sealed.json`, `artifacts/reports/m4_sealed_v02.json`):

| Model | Labelled data | Test | OOD |
| --- | --- | --- | --- |
| **v0.3 B, hand-written** | 200 windows | **0.957** | **0.908** |
| v0.3 A, hand-written | none | 0.744 | 0.815 |
| v0.2 B, hand-written MLP | 200 windows | 0.936 | 0.870 |
| v0.2 C-all, trained MLP (engineered features) | all dev | 0.992 | 0.843 |
| v0.3 C-all, trained | 11,864 sessions | 0.924 | 0.488 |
| v0.3 C-2048, trained | 2,048 sessions | 0.755 | 0.400 |
| v0.3 C-512, trained (3 seeds) | 512 sessions | 0.707 | 0.305 |
| Rules engine | none | 0.417 | 0.415 |
| Constant prior | none | 0.136 | 0.181 |

The best in-distribution model is still the trained v0.2 MLP on engineered features (test 0.992). The v0.3 trained transformers were capped at 3–4 epochs on CPU, so LoRA-Qwen is also a test of whether a better-trained baseline closes the OOD gap.

## Questions and hypotheses

Each hypothesis is a research target, not a pass/fail gate, as in v0.3. The comparison is always at **equal labelled data**.

| ID | Question | What would support it |
| --- | --- | --- |
| Q1 | Can Qwen do the task as it comes? | Zero-shot Qwen clearly above the constant prior (test 0.136); its gap to v0.3 A (0.744) is the value of hand-building |
| Q2 | Does weight-space programming (WSP) on Qwen beat building from zero? | WSP-Qwen B above v0.3 B (test 0.957, OOD 0.908) at the same 200 labelled windows |
| Q3 | Does WSP beat ordinary fine-tuning of the same model? | WSP-Qwen above LoRA-Qwen trained on a matched budget, and on the 512-session budget |
| Q4 | Does pretraining fix the trained models' OOD collapse? | LoRA-Qwen OOD well above v0.3 C-all (0.488), with fewer false alarms than 35% |
| Q5 | Is a WSP edit explainable and small? | The edit touches a reported, small fraction of weights, and its stated mechanism checks out on counterfactual pairs |

In v0.3 the author's written policy (H1) and the network (H3) were the same function by construction. Qwen's edited weights have no such reference policy, so Q5 replaces H2 fidelity as the interpretability check.

## Experimental arms

v0.4 adds six Qwen arms (the probe and LoRA arms have one column per budget); every v0.3 and v0.2 column is cited from its sealed report. Every Qwen arm starts from the same checkpoint and uses the same readout (next section).

| Arm | Built by | Labelled data | Answers |
| --- | --- | --- | --- |
| **Qwen-0**: zero-shot | Nobody. The released weights and a fixed prompt | 0 | Q1 |
| **Qwen-probe** | A linear 4-class head on frozen Qwen states, fitted by logistic regression | 32 / 128 / 512 / 2,048 / all sessions | How much the frozen model already knows |
| **Qwen-LoRA** | LoRA fine-tuning; seeds as v0.3 E8 (3 / 3 / 3 / 1 / 1) | 32 / 128 / 512 / 2,048 / all sessions | Q3, Q4 |
| **WSP-A** | Fable edits the weights in a sealed session. Forward passes only on inputs it writes itself | 0 | Q2 (vs v0.3 A) |
| **WSP-B** | Fable makes one revision of WSP-A after the same 200 disclosed windows as v0.3 Phase B | 200 windows | Q2, Q3 (vs v0.3 B) |
| **WSP-F** (optional) | Blog-style: WSP-B plus a budget of labelled forward passes on dev, with accept or reject | 200 windows + the budget | Matches the published method; reported separately |
| Cited | v0.3 A / B / C, v0.2 columns, rules engine, constant prior | as reported | Baselines |

- **The disclosed 200 windows stay the same.** The v0.3 selection rule never looks at a model's output, so WSP-B sees exactly the windows v0.3 Phase B saw. Only the two prediction fields are refreshed, from WSP-A.
- **WSP-F is optional.** The blog's method (Exp 8) kept an edit only if it scored better on training prompts. That is data-driven selection, which v0.3 banned. Reporting it beside WSP-B, never instead of it, keeps the main comparison clean.

## Task interface

Qwen answers with one of four label words after every tool-call block. It reads the same log text as v0.3, and each decision sees only the past.

1. **Input.** The frozen v0.3 renderer's text for each session (headers, error lines, opaque diff ids), tokenised by Qwen's own tokeniser. The vision encoder is not used.
2. **Fixed prefix.** One short instruction at the start of every session, the same for all arms. It defines the four classes in plain words, taken from the spec. Thinking mode is off.
3. **Query after each block.** After block t, a fixed suffix asks for the status. The class is the argmax of the LM-head logits over four single-token label words (for example productive, recovering, stalled, oscillating). That is the "verbaliser" readout.
4. **One pass per session.** The session is processed once. At each block end the cached attention and DeltaNet state is branched to run only the suffix (about 10–15 tokens), then thrown away. This gives exactly the result of a separate prompt per window at about 1.5× the cost of one pass, and the label after block t never sees block t+1.
5. **The same readout for every arm.** Zero-shot, LoRA and WSP all use the verbaliser, so they differ only in the weights. The probe is the one exception: by design it reads the hidden state at the suffix's last token.
6. **Frozen text.** The prefix, suffix and label words are fixed in the pre-registration. Fable may change weights, not the prompt.

Windows, the scored mask, labels and populations are v0.3's, unchanged. Each Qwen column is scored on exactly the 87,176 val, 87,148 test and 83,011 OOD windows the v0.3 report used.

## Rules for Fable's weight-space programming

Fable edits Qwen's weights by reasoning, in a fresh sealed session, with the v0.3 bans unchanged. The harness is modelled on the blog's Exp 8 kernel, with two fixes for its weak points: the holdout is held by the harness, and the budget is counted in tokens.

**Packet.** The v0.4 spec and the task interface (prompt, suffix, label words). The v0.3 grammar sheet and renderer, so Fable knows the log format. The Qwen config, tokeniser and original weights, read-only. No generator, oracle or dataset code, as in v0.3.

**Kernel tools.** A persistent Python kernel on the local CPU, about 3.2 GB of fp32 weights:

- `load(name)`, `load_original(name)`, `save_edit(name, tensor)`, `drop_edit(name)`, `edited()`, `tok`, after the blog's kernel;
- `forward(texts)`: Qwen's four label-word logits after each block, plus hidden states and MLP activations at the layers Fable names. It runs on Fable's own texts, original or edited weights, measured at about 215 tokens per second.

**Banned, as in v0.3.** Backward passes, autograd and optimisers. Fitting weights to a loss. Any automated or numerical search over candidate weights, including keep-the-best loops. Distillation.

**Allowed.** Direct, reasoned edits, such as:

- rank-1 updates along a direction Fable has identified;
- projecting a direction out of a matrix;
- writing a readout direction into the label-word rows.

**Feedback by arm:**

| Arm | Forward passes on | Labels seen | Budget |
| --- | --- | --- | --- |
| WSP-A | texts Fable writes itself | none | a fixed token budget (proposed 2M tokens, about 2.6 h of CPU) |
| WSP-B | its own texts and the 200 disclosed windows | 200 windows, once | the same, plus one revision |
| WSP-F (optional) | dev windows, chosen by Fable | a fixed labelled budget | the blog's pass count; the final paired check on a harness-held holdout |

**Deliverables.**

- `build_edit.py`: rebuilds the edit deterministically from the original weights; the harness checks it bit for bit.
- `EDIT.md`: what was changed, in which tensors, and the mechanism claimed.
- `REVISION.md` for WSP-B, citing the disclosed window ids, as v0.3 Phase B did.

The harness reports how many weights changed and the relative Frobenius norm of each tensor's change.

**Caps.** Proposed as in v0.3: US$500 for WSP-A and US$250 for WSP-B, with a fresh session that has never seen generator code.

## Data, splits and sealing

Recommendation: score v0.4 on a **fresh sealed test set** as the primary result, and on v0.3's test and OOD sets as secondary results for continuity.

- **Training and disclosure data are unchanged.** Dev (12,000 sessions) trains the probe and LoRA arms. Val (4,004 sessions) is for validation, with the 200 disclosed sessions left out of the primary population.
- **The v0.3 test and OOD sets have now been scored.** No author has seen them, so they can still score new models fairly. But we, who design v0.4, now know the v0.3 results on them. Prompt wording, LoRA settings and the label words could drift toward what we know works there.
- **A fresh test set is cheap.** Same generator, a new registered seed, about 4,000 sessions, sealed before any v0.4 run. The frozen v0.3 artefacts are re-scored on it too, so the comparison stays like for like.
- **A fresh OOD set is not cheap.** It needs a newly written generator from a different model family, as v0.3's Codex-written one was. Options: reuse v0.3 OOD only, or commission an OOD-2. This is a decision for Matthew.
- **Every sealed set is still scored once,** after the WSP-B archive, by a pre-registered script, as in v0.3 (E7).

## Compute and environment

Recommendation: run Fable's kernel on the Mac's CPU, and do all Qwen evaluation and LoRA training on one rented cloud GPU. The whole cloud bill should be under about US$50.

**What runs where (tested 5 Nov 2026):**

- **The project's environment cannot run Qwen3.5.** `transformers` first ships `qwen3_5` in 5.2.0, which needs torch 2.4 or later. pip has no torch newer than 2.2.2 for Intel Macs ([PyPI](https://pypi.org/pypi/torch/json)).
- **A separate conda-forge environment works:** Python 3.12, torch 2.14.1 (cpu_mkl) and transformers 5.22.1. Gated DeltaNet runs through its pure-PyTorch fallback ([modeling_qwen3_5.py](https://github.com/huggingface/transformers/blob/v5.2.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)). The v0.3 environment stays frozen as it is.
- **Measured CPU speed:** fp32 on 8 threads, about 215 tokens/s on the forward pass, 7.4 GB peak memory, 25 s to load. llama.cpp on CPU reaches 290–350 tokens/s, but it gives only last-layer states.
- **The Mac's GPUs are not usable.** torch's Metal backend refuses the discrete W5700X ("only supported on devices with unified memory"). llama.cpp's Metal build ran about 7× slower than the CPU, or timed out.
- **Token counts.** Sessions run to 1,559 Qwen tokens on average (p95 3,311), 1.78× v0.3's. All of val is 6.24M tokens.

**Cost and time by task.** Estimates use a measured CPU rate and GPUs at 15–30% of peak. Prices are on-demand from [Modal](https://modal.com/pricing), [RunPod](https://www.runpod.io/pricing) and [Lambda](https://lambda.com/pricing).

| Task | CPU here | One H100 | Cloud cost |
| --- | --- | --- | --- |
| Verbaliser pass over all of val | about 20 h | under 1 h | under $1 |
| LoRA, 512 sessions, 3 epochs | about 15 h | minutes | under $1 |
| LoRA, all 12,000 sessions, 3 epochs | about 340 h | 0.5–1 h | about $4 |
| Sealed evaluation, about 10 Qwen columns | about 580 h | 0.3–2 h | under $10 |
| Fable's forward passes (2M-token budget) | about 2.6 h | not needed | $0 |

Batching the per-window suffixes across sessions matters more than buying a faster GPU. One window at a time, a sealed column takes 1.4–2 h on any GPU.

**Scoring backend.** v0.3 scored in fp32 on CPU. v0.4 scores in fp32 on one named GPU type with deterministic kernels, fixed in the pre-registration. The harness also replays 100 val sessions on CPU and reports how many argmaxes agree.

## Milestones

The work runs in six steps with two gates. LoRA training runs in parallel on a cloud GPU while Fable works.

1. **M0 Setup.** Conda env, Fable kernel, verbaliser readout. Token audit on val; seal a fresh test set.
2. **G1: pre-registration frozen and committed.** This freezes everything a later result could tempt us to change: prompt, label words, LoRA recipe, WSP rules, packet hash.
3. **M1 Qwen-0 and probe on val.** Harness-only scores, never shown to Fable. Checks the readout before any WSP.
4. **M2 WSP-A: Fable, no labels.** Sealed session; edit archived and scored on val. Refreshes the 200-window disclosure.
5. **M3 WSP-B: one revision.** Sees the 200 disclosed windows once. Archived before any sealed scoring.
6. **In parallel with M1–M3: training baselines on a cloud GPU.** LoRA at 32, 128, 512, 2,048 and all dev sessions (seeds as v0.3 E8); the probe fits on frozen states at the same five budgets.
7. **G2: WSP-B archived and every LoRA run done.** The sealed sets stay closed until every arm is final.
8. **M4 Sealed evaluation, run once.** The fresh test set is the primary result; v0.3 test and OOD are secondary.
9. **M5 Report.** One joint table with v0.2 and v0.3.

## Risks

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Qwen3.5 does not run on the Intel Mac (new `transformers`, DeltaNet kernels, torch pinned at 2.2.2) | Nothing runs locally | A separate environment, or a cloud GPU for every Qwen arm (see Compute) |
| GPU numerics differ from CPU | Argmax flips on borderline windows; v0.3 scored on CPU float32 | Fix the backend and dtype in the pre-registration. Report a CPU–GPU agreement check on val |
| Fable cannot hold 0.8B weights in context | WSP becomes guesswork | Give it analysis tools (tensor slices, SVD, activation probes on its own inputs). Budget a larger session than v0.3's $58 + $65 |
| WSP can unlock an ability but not add one (the blog's Gemma result) | If zero-shot Qwen is near the prior, WSP may fail | Run Qwen-0 and Qwen-probe first. A failed WSP is still a reported result |
| Prompt and label words decide zero-shot accuracy | Designer bias toward what works | Freeze the prompt before any Qwen run. Choose it on dev only, with the choice rule written down |
| Opaque diff ids tokenise badly in Qwen's vocabulary | Long sessions, higher cost, especially OOD (about 1.7× longer) | Measure tokens per session on val before freezing |
| LoRA tuned harder than WSP | An unfair comparison | One pre-registered LoRA recipe; any tuning on val only, reported |
| Fable sees generator or oracle code | Voids the zero-label result | Same sealed-packet rules as v0.3; a fresh session |

## Decisions needed from Matthew

**Decided 5 Nov 2026: Matthew accepted every recommendation below as written.** Separately, `" oscillating"` is two Qwen tokens, so it cannot be a label word. The draft interface uses `" looping"`; G1 fixes the final words (see `experiments/v0_4/STATUS.md`).

Each item has a recommendation.

- [x] **Fresh sealed test set.** Recommended: yes. It is about 4,000 new sessions from a new seed, and v0.4's primary result is scored on it.
- [x] **OOD-2.** Recommended: reuse v0.3's OOD for now. Commissioning a new independently written generator costs a separate Codex or Grok session and about a day.
- [x] **Cloud GPU.** Recommended: RunPod or Modal, H100 or A100, with a compute cap of US$100 for all Qwen evaluation and LoRA. The estimate is under US$50.
- [x] **WSP-F (blog-style labelled forward passes).** Recommended: include it as a separate, clearly labelled column after WSP-B.
- [x] **Fable caps.** Recommended: US$500 for WSP-A and US$250 for WSP-B, as in v0.3.
- [x] **LoRA budgets.** Recommended: all five (32 / 128 / 512 / 2,048 / all), seeds as v0.3 E8. On a GPU the cost difference is small.
- [x] **Label-word rows.** Qwen ties its embedding and LM head, so editing the four label words' readout rows also changes how those words are read as input. Recommended: allow it, and report it.

## Revisions after M0 (5 Nov 2026)

These follow the M0 build and an independent review of the v0.3 harness. Matthew asked for the recommendations to be taken, so each item below is decided. G1 writes them into the pre-registration.

1. **Machine.** The work runs on an Apple M4 Pro (24 GB), not the Intel Mac this plan's "Compute and environment" section measured. On this machine:
   - The Mac GPU works (MPS). Pip ships current torch, so no conda environment is needed (`envs/qwen`, a uv project).
   - The readout runs at about 445 tok/s on MPS. It is compute-bound, and each call has a fixed cost of 40–70 ms. One full val pass takes about 5 h; Fable's 2M-token budget takes about 75 min.
   - CPU is far slower for the readout's many short calls: under 10 tok/s measured while another job shared the machine.
   - Fable's forward server and the harness-only val runs use local MPS. LoRA, probe features on dev, and every sealed scoring run use one CUDA GPU (cap US$100). The CPU replay check runs on that cloud machine (CPU against CUDA), not on this Mac.
2. **Fresh sealed test set** (`experiments/v0_4/fresh_test/REGISTRATION.json`, `datasets/v0_4/fresh_test/MANIFEST_fresh_test.json`).
   - It has 3,996 sessions, 333 per family, from master seed 20261106 with the v0.2 generator unchanged: 86,412 windows, 83,874 of them gate-passing.
   - The seed was committed before generation. The manifest holds no label statistics.
   - Every v0.4 column is scored on it as the primary result, alongside v0.3 A/B/C, the rules engine and the constant prior. v0.2 B and C-all are added if the frozen v0.2 feature path can be run without touching the spent v0.2 sealed driver.
3. **Prompt and label words are chosen without labels** (`experiments/v0_4/prompt_choice/README.md`). The rule picks the highest median label-word probability mass over 12 candidates on 100 hash-picked dev sessions. That keeps Qwen-0 and WSP-A zero-label; choosing on dev labels, as first planned, would not have. ` oscillating` is two Qwen tokens, so no candidate uses it.
4. **WSP kernel design** (`docs/V04_WSP_KERNEL.md`).
   - The forward pass is a separate metered tool, one request per Bash call. It is not a function inside Fable's scripts, so a keep-the-best loop cannot be written as a script.
   - The meter, edit journal and ledger live in a harness process that never runs author code.
   - Edits are in scope only for the text-model tensors, minus the final norm. In `embed_tokens`, only the four label rows may change.
   - WSP-B's forward passes on the 200 disclosed windows return label logits only, with no activations, so they cannot be turned into a centroid probe.
5. **WSP-B disclosure.** The same 200 windows and oracle labels. `network_class` comes from WSP-A's verbaliser. `reference_policy_class` is dropped, because no reference policy exists. The format tag is `v04-wsp-b-disclosure/1`.
6. **WSP-F.**
   - **Cost cap:** US$250.
   - **Labelled budget:** 50 `score_dev` calls of at most 128 dev sessions each.
   - **Holdout:** a final paired check on 500 dev sessions picked by hash key and disjoint from every LoRA/probe budget subset. The harness holds it.
   - **Gate:** G2 waits for the WSP-F archive too.
7. **Matched budget for Q3.** Add **LoRA-200w** and **probe-200w** columns trained on exactly the 200 disclosed windows, alongside the 32 to all budgets. That gives WSP-B a like-for-like comparison.
8. **Q5 protocol.** For each mechanism EDIT.md claims, the harness checks it on the counterfactual pairs, with the edit applied and then ablated. The pass statistic is pre-registered at G1. Counterfactual is scored before the sealed attempt, as in v0.3.
9. **Scoring bridge.** It uses `eval.m4.score_columns`, as v0.3 Phase C did. H2 is not applicable without a reference policy.
10. **Re-scoring v0.3 artefacts on arm64.** v0.3's reproduction checks require exact equality, and the artefacts were scored on x86. Before G1, the unofficial v0.3 val dry run is repeated on this machine and diffed against the official reports. If it matches, local re-scoring is valid. If not, G1 pre-registers a tolerance. **Done 5 Nov: every check matched exactly on arm64, so local re-scoring is valid with no tolerance.**
11. **v0.3 cost correction.** `total_cost_usd` in Claude Code's stream is cumulative across resumed launches of one session, so `scripts/v03_cost.py` double-counted. v0.3's hand-written work cost US$64.62 in total, not US$122.24: Phase A $42.73 and Phase B $21.89. The v0.4 cost script accounts by session-id delta.

## Revision of 6 Nov 2026 (before G1)

Matthew decided this directly. It overrides everything above that assumes a Qwen-probe or Qwen-LoRA column, a cloud or CUDA GPU, or anything rented, in particular:

- the "Compute and environment" recommendation (all Qwen evaluation and LoRA training on one rented cloud GPU) and its "Scoring backend" paragraph (one named GPU type);
- milestones 3 (the probe on val), 6 (training baselines on a cloud GPU) and 7 (G2 waits for every LoRA run; it now waits for the WSP-B and WSP-F archives);
- the accepted recommendations "Cloud GPU" and "LoRA budgets";
- revision 1's last bullet (LoRA, probe features on dev and every sealed scoring run on one CUDA GPU, and the CPU replay on that cloud machine): every one of those runs is now on this Mac, as below;
- revision 6's holdout wording ("disjoint from every LoRA/probe budget subset"): the holdout is now defined as outside the v0.3 Phase C 2,048-session budget subset, which is the same set of sessions (PREREGISTRATION §7.3);
- revision 7 (LoRA-200w and probe-200w).

The sections above stay as written; the pre-registration and the spec carry the change.

12. **No training baselines and no cloud GPU.** Matthew: "Drop the training baselines", and "I’ve never planned on renting a h100".
    - There is no Qwen-probe and no Qwen-LoRA column at any budget (32, 128, 512, 2,048, all or 200 windows). WSP-A, WSP-B and WSP-F are compared with Qwen-0 and the cited v0.3 and v0.2 columns only.
    - Q3 and Q4 are not tested, because both needed those columns. Q1, Q2, Q2-OOD and Q5 are unchanged.
    - Nothing is rented. Fable's forward server, the harness-only val runs and every sealed scoring run, the Q5 readouts included, use local MPS in fp32. The CPU replay check runs on this Mac's CPU and compares Qwen-0 and WSP-B with MPS.
    - This works because the Mac's GPU works. The "Compute and environment" section assumed it would not. That held for the Intel Mac it measured, but not for the M4 Pro (revision 1).
    - The cost is time. One Qwen column over val takes about 5 h on MPS, so the sealed attempt takes several days and the CPU replay several hours per column.
    - The probe, LoRA and cloud-runner code merged before this decision has been removed. It stays in git history at commit `0540341`.

## Second revision of 6 Nov 2026 (before G1)

Matthew decided this directly too: "Val only, no sealed". His context: "I need a quick answer to how the base qwen vs modified qwen performs, not 90 hours". It overrides everything above that assumes a sealed evaluation, in particular:

- the "Data, splits and sealing" recommendation (the fresh sealed test set as the primary result, v0.3 test and OOD as secondary, every sealed set scored once after the WSP-B archive);
- milestones 7 (G2) and 8 (M4, the sealed evaluation);
- revision 2's use of the fresh test set (it was built and sealed; it is not scored), revision 8's counterfactual pass, and revision 10's re-scoring of v0.3 artefacts on the fresh set (they are re-scored on val instead);
- revision 12's sealed scoring runs and its "several days" for the sealed attempt.

13. **No sealed scoring in v0.4.**
    - The fresh test set, v0.3 test, v0.3 OOD and the counterfactual split are not opened or scored in v0.4. They stay sealed and available to a later version, which may score the archived v0.4 edits on them under its own pre-registration.
    - The v0.4 result is base Qwen (Qwen-0) against the WSP edits on val, `excluding_disclosed` primary, harness-only. Val numbers are still never shown to Fable.
    - **Quick look first.** Each edit (WSP-A, then WSP-B, then WSP-F) is read out first on `datasets/v0_4/VAL_QUICK.txt`, 1,000 val sessions chosen by hash key before any edit existed, and scored paired against Qwen-0 on the same sessions with the same bootstrap draws. Its full val readout follows, and the full-val numbers are the result of record. The quick look is reported beside, never instead of, the full result.
    - The v0.3 A, B and C networks and the constant priors are re-scored on val in the frozen v0.3 environment (`scripts/v04_v03_val_columns.py`), exactly against the v0.3 report column by column, so WSP-B − v0.3 B (Q2) has a paired CI on val. A column that does not reproduce exactly is cited from the v0.3 report without a paired CI; if that is v0.3 B, Q2 is not tested (PREREGISTRATION §1). The rules engine and the v0.2 columns are cited.
    - Q1 and Q2 are tested on val. Q2-OOD and the Q5 verdict are not, because they need v0.3 OOD and the counterfactual split. The size of every edit and the eligibility of its claimed mechanisms are still reported.
    - Val is not a sealed set: the harness reads models out on it, v0.3 used it for early stopping and the Phase B disclosure, and WSP-B sees 200 of its windows (`excluding_disclosed` removes their sessions). So a v0.4 result is weaker evidence than a sealed one, and the report says so.
    - Time: four full val readouts take about 20 h of MPS, and a quick look about 1.25 h, against several days for the sealed attempt.

## Sources

- "Sparks of Weight-Space Programming", Takehi, Teoh, Zhang and Khattab, 2 Nov 2026: https://rikiyat.github.io/blog/weight-space-programming/ and logs at https://github.com/RikiyaT/lm_understand-logs
- Qwen3.5-0.8B model card: https://huggingface.co/Qwen/Qwen3.5-0.8B
- Feasibility measurements on this Mac (environment, throughput, token counts): research run of 5 Nov 2026, scripts in the session scratchpad.
