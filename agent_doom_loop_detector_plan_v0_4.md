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

Each item has a recommendation.

- [ ] **Fresh sealed test set.** Recommended: yes. It is about 4,000 new sessions from a new seed, and v0.4's primary result is scored on it.
- [ ] **OOD-2.** Recommended: reuse v0.3's OOD for now. Commissioning a new independently written generator costs a separate Codex or Grok session and about a day.
- [ ] **Cloud GPU.** Recommended: RunPod or Modal, H100 or A100, with a compute cap of US$100 for all Qwen evaluation and LoRA. The estimate is under US$50.
- [ ] **WSP-F (blog-style labelled forward passes).** Recommended: include it as a separate, clearly labelled column after WSP-B.
- [ ] **Fable caps.** Recommended: US$500 for WSP-A and US$250 for WSP-B, as in v0.3.
- [ ] **LoRA budgets.** Recommended: all five (32 / 128 / 512 / 2,048 / all), seeds as v0.3 E8. On a GPU the cost difference is small.
- [ ] **Label-word rows.** Qwen ties its embedding and LM head, so editing the four label words' readout rows also changes how those words are read as input. Recommended: allow it, and report it.

## Sources

- "Sparks of Weight-Space Programming", Takehi, Teoh, Zhang and Khattab, 2 Nov 2026: https://rikiyat.github.io/blog/weight-space-programming/ and logs at https://github.com/RikiyaT/lm_understand-logs
- Qwen3.5-0.8B model card: https://huggingface.co/Qwen/Qwen3.5-0.8B
- Feasibility measurements on this Mac (environment, throughput, token counts): research run of 5 Nov 2026, scripts in the session scratchpad.
