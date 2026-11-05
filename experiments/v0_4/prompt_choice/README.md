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

None.
