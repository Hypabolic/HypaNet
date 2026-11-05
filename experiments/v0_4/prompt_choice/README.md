# v0.4 prompt choice: pre-registration

_Written 2026-11-05, before the rule has been run. No Qwen output has been compared with a label._

The interface text (instruction, answer cue and label words) is shared by every verbaliser arm. That covers Qwen-0, Qwen-LoRA and every WSP arm. Only the probe reads a hidden state instead. This file fixes how the text is chosen. The choice uses **no labels**, so Qwen-0 and WSP-A stay zero-label.

## 1. Candidates

`envs/qwen/hypaqwen/prompt_choice.py` defines the candidates as `CANDIDATES`. There are 12, the product of three label-word sets, two answer cues and two instruction styles:

| Set | PRODUCTIVE | RECOVERY | STALLED | OSCILLATING |
| --- | --- | --- | --- | --- |
| w1 | ` productive` | ` recovering` | ` stalled` | ` looping` |
| w2 | ` productive` | ` recovering` | ` stuck` | ` looping` |
| w3 | ` progress` | ` retry` | ` stuck` | ` loop` |

- **Answer cue:** `State:` or `Answer:`, the last word before the label.
- **Style:** `full` describes the log format before the definitions; `short` gives one line of context.
- **Shared by all candidates:**
  - the Qwen chat format with thinking off;
  - the four class definitions, word for word from spec v0.3 §2;
  - one block per piece (`{block}\n`);
  - the readout in `hypaqwen/readout.py`.
- **Single tokens:** every label word is one Qwen token with its leading space. ` oscillating` is two tokens, so no set uses it.

The draft interface used in M0 plumbing is candidate `w1-state-full`.

## 2. Data

- **Sessions:** the 100 dev sessions with the smallest `sha256("v04-prompt-choice:{session_id}")`.
- **Windows:** every window of those sessions (`window_pos` from the task export).
- **Inputs:** only the rendered text. The rule reads no labels, gate results or families.
- **Other splits:** val, test and OOD are not touched.

## 3. Rule

For each candidate and each window, the **label-word mass** is the probability the model puts on the four label words together:

`exp(logsumexp(label-word logits) − logsumexp(all 248,320 logits))`

The candidate with the highest **median** mass over all windows is chosen. A tie goes to the earlier candidate in `CANDIDATES` order.

The mass measures whether Qwen answers in the requested format. It does not measure whether the answer is right, and no accuracy is computed.

## 4. Run

```
cd envs/qwen && uv run python -m hypaqwen.prompt_choice --device mps
```

Backend: MPS, fp32, the pinned model files (`hypaqwen/model.py`). The report is `artifacts/reports/v04_prompt_choice.json`. It holds:

- each candidate's mass quantiles;
- each candidate's predicted-class shares (computed without labels);
- the chosen name.

The rule runs once, after the fresh test manifest is committed. G1 then writes the chosen candidate's text into `hypaqwen/interface.py` as the frozen interface and records its sha256.

## 5. Not allowed

- Running any candidate on val, or scoring any candidate against labels, before G1.
- Adding, removing or editing candidates after the rule has run. Any change needs a new pre-registration with a revision entry here, and the first run is still reported.

## Revisions

### Revision 1 (2026-11-05, after the first run; approved by Matthew)

**What the first run showed.** The rule in §3 chose `w3-state-full` (median mass 0.986). It is reported unchanged in `artifacts/reports/v04_prompt_choice.json` (commit `8dac1f2`). Under that candidate, the model gives STALLED for all 2,069 dev windows, and so does every other w3 candidate. The w2 candidates give STALLED for 89–100% of windows. The mass was high because one label word took nearly all of it, so mass alone rewarded the degenerate answer it was meant to rule out. With such an interface, Qwen-0 would be a constant classifier.

**What was seen.** Only label-free quantities: the mass quantiles and the predicted-class shares that §4 puts in the report. No labels, gate results or families were read, and no candidate was scored. Qwen-0 and WSP-A remain zero-label.

**Revised rule.** Each candidate is scored by its **median label-word mass × the normalised entropy of its predicted-class shares** (entropy ÷ log 4: 0 means one class always, 1 means uniform). The highest score wins, and a tie goes to the earlier candidate. Both factors are read from the first run's report, so nothing is run again (`uv run python -m hypaqwen.prompt_choice --revision1`, written to `artifacts/reports/v04_prompt_choice_revision1.json`).

**Result.** `w1-state-full` scores 0.742; the next best is `w1-answer-full` at 0.526, and every w3 candidate scores 0. `w1-state-full` is the M0 draft text, so every M0 measurement used this interface. It is frozen as `FROZEN` in `hypaqwen/interface.py`, and G1 records its `spec_sha256`.

**Known weakness.** The entropy factor rewards a spread of answers, not correct ones. A candidate that spread its answers at random would also score well. Here the choice is clear: the winner leads by 0.22, and among the w1 candidates it has both the highest entropy and the second-highest mass.
