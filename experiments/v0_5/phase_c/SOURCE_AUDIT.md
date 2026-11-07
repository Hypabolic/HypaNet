# v0.5 Phase C: source-switch and compliance audit (7 Nov 2026)

This was run **before** the one-time held-out scoring, as Matthew asked when he chose "launch as is, audit after".
The second generator can be identified from its surface (gaps under 1.7 s, `in=0 out=0`, its own vocabulary,
no `edit_file`/`search`/`list_dir`/`lint`/`revert`), so the question was whether Phase C's network detects the
source and switches rules on it. There were three audits (static weights, empirical perturbation, compliance),
each re-checked by a skeptic. The skeptics re-ran the perturbation and ablation results on new seeds
(20261201/20261202). Only fresh sessions were used: seeds 20261107 (second generator) and 20261108 (standard),
plus the new seeds. No held-out or sealed data was opened. Scratch: `~/Library/Caches/hypanet-scratch/v05c_audit/`.

## Verdict: no source switch, with three rules tuned to the second generator

**Weights.** Between B and C, the token embedding changes in exactly two entries:
- `read_file` (id 7) → d127, a new flag;
- `h0` (id 5861) → d30. `h0` is the renderer's hunk id for an empty removed side, i.e. a pure insertion.

The offset embedding is unchanged. No new unit reads timestamps, gaps, token counts, or any path, suite or
error-class word. Paths enter only as hashed codes used for equality matching. The new gather heads attend
within the current event only, so no log-level "which generator" flag can form.

**Perturbations** (second-generator logs made to look standard, and the reverse):

| Change | Windows changed by C |
|---|---|
| Clock, gaps, token counts | 0, in both directions |
| Consistent rename of paths/suites/error classes to standard vocabulary | 10 of 15,824 |
| Control: rename to other second-generator-style names | 11 of 15,824 |

The rename effect is hash-collision noise. The surface tells are not read.

**Rules that are format-specific or tuned to the second generator** (all applied identically to every log):
1. **Fresh `read_file` suppresses STALLED.** REVISION.md §2.5 calls it "a genuine difference between the
   generators" and gives up V134 and V146 for six W windows. It has the largest effect of the three. Ablating
   it on the disclosed families' fresh sessions takes C from 0.962 (FA 0.045) to 0.922 (FA 0.112), with no
   change on the other families. Standard logs show the same regularity, only weaker: 3.7% of STALLED windows
   have a fresh read, against about 0.1% in the second generator. On standard logs it trades a little
   accuracy for fewer false alarms.
2. **Append rule (H0RED).** It is triggered by `h0`, which the standard generator never emits (0 of ~10,000
   lines), so in practice its trigger is a one-way tell for the second generator. It is still a per-event content
   check (an append to an already-modified path, with no EDR), and it gates no other rule.
3. **Plan repeats keyed on `verb:argument`.** Only the second generator writes that format. On single-word plans
   the rule falls back to the verb (format-conditional, not a source switch).

Expect held-out OOD sets from the same generator to **flatter** rules 1–2. The unseen-family contrast is the
fairer test, but those 4 families are from the same generator too.

## Cost in distribution

On fresh standard sessions C is 0.007–0.017 macro-F1 below B, depending on seed (B 0.946–0.961, C 0.929–0.954).
It has fewer false alarms (0.020 against 0.030). Removing all three rules recovers only part of this. The rest is
RECOVERY→PRODUCTIVE disagreements in `polling_recovery` and `repeated_test_failure`, which probably come from the
count-keyed retry rule or the `read_file` chain breaker.

## Compliance: compliant, with rule selection disclosed

- **Generated weights.** `gen_revision.py` prints fixed constants and reproduces `REVISION.edits` byte for byte.
  The build reproduces the artefact (`a7f2890c…`).
- **What was not done.** There was no weight search, fitting, autograd, randomness or outside data.
- **Rule selection.** `rules.py` evaluated about 31 hand-written rule variants on the 400 disclosed windows,
  and the author chose by reading the results (Phase B did the same). A few choices were small numeric
  comparisons decided by the disclosed score: the retry-tally threshold 5 against 4 (W 0.955 → 0.970),
  append lookback 7 against 31, and `list_dir` excluded from the fresh rule. All of these are recorded in
  ITERATIONS.md, but REVISION.md's "checked (not tuned)" understates them. The author did not always keep the
  best-scoring variant.
- **Thinking.** Every thinking block in the stream is redacted, so the reasons behind selections can be read
  only from text, tool output and ITERATIONS.md.
- **Session audit.** One model; two read-only `ls` inside the author dir; one refused read of its own
  tool-result file (all as in Phase B).
