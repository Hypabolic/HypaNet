# REVISION.md: Phase B, the one diagnostic revision

## What was disclosed

- `disclosure/DISCLOSURE.json`, sha256 `27c8deb03432a0ad79ebcd32f867a132d6c6bca210bdd81cd011ee2987e87492`
  (verified locally; `disclosure/DISCLOSURE.sha256` carries the same digest).
- 200 validation windows `V001`..`V200`, 50 per oracle class, each with the rendered prefix (the
  `### window start` line removed before tokenising), the oracle label, the archived Phase A
  network's class and the archived Phase A `reference_policy` class. Nothing else was used.
- Base: the archived Phase A artefact (sha256 `ba6f4743e6c4c32a0a71124357511146889a94204d8263d547edaa85b251dfb6`),
  copied into this directory by `scratch/copy_phase_a.py`.

## Where Phase A stood on the disclosure

The Phase A network and policy agreed on all 200 endpoints (`scratch/peek.py`), so every error was
a policy error. Against the oracle: 141 / 200, macro-F1 0.700. STALLED recall 0.30 (34 STALLED
windows called PRODUCTIVE), RECOVERY recall 0.68 (16 called PRODUCTIVE), OSCILLATING recall 0.88,
PRODUCTIVE precision 0.46.

## Changes to the policy, each with its reason and the windows that motivated it

The analysis scripts are `scratch/analyze.py`, `scratch/eval_v3.py` (candidate evaluator) and
the candidates `scratch/policy_v2.py` .. `policy_v6.py`, `scratch/variants.py`. Every change below
was a reasoned edit; the candidates were compared on the disclosed windows by hand, not searched.

1. **Session-wide repetition.** Redundancy is judged against the whole prefix, not the last 7
   blocks: a repeated search ten blocks after its first occurrence is still redundant (V009), reads
   repeated across a long session are redundant (V125, V142, V128), a re-validation of an
   unchanged workspace is redundant however long ago the previous validation was (V044).
2. **Validation failures are episodes.** A FAILED run re-run on an unchanged workspace is a retry;
   three retries are in policy (V154, V157, V174) and the episode stays active through later reads
   and edits (V112). A later passing run of the same suite resolves the episode (V088, V160, V115),
   but only when the failure had been retried: a fix after a single failure is ordinary progress
   (V105, V052, V090, V012, V095, V058 are PRODUCTIVE).
3. **What counts as a retry.** The repeated failure must be of the same suite within the last 7
   blocks (V142: a re-run 26 blocks after the original failure is a fresh failure), with no hunk
   block since the last validation (a run after an edit is a new attempt: V021, V058, V003, V015,
   V084), and with at least as many passes as its predecessor (a narrowed re-run with a different
   count is a new attempt: V029, V046). One retry is already in policy (V177, V180, V185). Out of
   policy once the suite has failed 4 times (6 pending polls for a poll resource: V030 is in policy
   with six, V163 and V044 are out).
4. **Network episodes.** For `http_call` / `poll_status` the first successful repeat after a
   failure is a resolution, no retry needed (V036, V124, V129, V199, V200); a first failure alone
   is not an active episode (V085).
5. **Active episode.** RECOVERY needs the most recent failing-or-resolving block of the prefix to
   be an in-policy retry, or a resolution within the last four blocks; a resolution at distance
   >= 4 no longer counts (V016, V022) and an out-of-policy retry ends the evidence (V163, V125).
6. **Stall-ineffective blocks.** Added: validations that repeat their predecessor's status without
   a new high, including passing re-runs (V040) and identically failing re-runs beyond policy
   (V009, V027, V037, V060, V063); edits with hunk lines made after such a stagnant validation
   (V027, V037, V060, V062, V063, V147, V168); repeated `plan` / `ask` / `yield` actions, which
   now share the inspection class (V134). The intent of a block whose target is not a hashed word
   (`path=.`) falls back to its first non-reserved token (V128, V142). Hunk ids are read as the
   tokens around `>` so that paths starting with `.` do not shift the slots (V025).
7. **Progress.** Progress is a validation whose passed count exceeds every earlier count of the
   same suite (a new high), not "latest > earliest in the window": alternating counts no longer
   look like progress (V043, V081, V126, V133), an improvement relative to a run just before the
   window still counts (V051), a FAILED -> ok transition is a resolution rather than progress
   (V125, V093), and failed validations with dropped counts no longer fake a rise (V032).
8. **Unchanged rules.** The rule order (OSCILLATING > STALLED > RECOVERY > PRODUCTIVE), the window
   of 8, `N_stall = 5`, `N_rev = 2`, `k_rec = 4`, the within-window reversion test on hunk ids, and
   the logit encoding of the order are as in Phase A.

Result of the revised policy on the disclosure: 199 / 200, macro-F1 0.995; the one miss is V053
(see "left unchanged"). The policy scans the prefix once with running per-key statistics, so its
cost is linear in the prefix: 11.6 ms per call on the longest disclosed prefix (48 blocks) and
25.9 ms on a 139-block Phase A session (`scratch/policy_time.py`), far below the 0.5 s limit.

## Changes to the weights

The circuit was rebuilt around the new relations; `CIRCUITS.md` documents it fully. In brief:

- Layer 0 now copies tool and status, fetches the previous token's identity bits and flags (one
  exact offset lookup instead of two), detects hunk lines, and gathers the intent token into the
  EOE with a priority score (earliest hashed token at offset >= 10, else earliest other
  non-reserved token). All 13-bit identity codes now cover punctuation and digits as well.
- Layer 1 matches `h_new` against the `h_old` bits carried by `>` tokens, reads the passed count
  (digit after the status word), finds the nearest same-key block over the whole prefix (status),
  tests for a same-key block within 7, counts same-key failures with the BOS anchor, and finds the
  nearest validation-or-hunk block. Its MLP derives the status-difference, the count thresholds
  (`<= 3`, `<= 5`), reverting tokens, re-applied edits and repeated inspections.
- Layer 2 fetches the previous same-key block's count and layer-1 flags, the maximum count among
  earlier same-id blocks (hard attention on `12 * count`), and the nearest validation's flags, then
  derives progress, stagnant validations, in-policy retries, resolutions and the SI sum.
- Layer 3 averages the window (SI, reversions, progress), fetches the in-policy flag of the nearest
  failing-or-resolving block and any resolution within four blocks; the head is as in Phase A
  with `ACT` replacing the final block's own in-policy flag.
- Scales: gate `M = 1e5` (match scores reach 12,000 + 1,200), intent scale 800 so a same-key key
  at distance 63 (bias -945) still beats every other key, counts capped at 100.

Agreement of the rebuilt network with the revised policy: 4,484 / 4,484 prefixes of the disclosed
windows (`scratch/netcheck.py discall`) and 381 / 381 Phase A windows (`scratch/netcheck.py phasea`);
at the 200 endpoints it scores 199 / 200 against the oracle. Build: 0.82 s CPU, deterministic,
156,788 non-zero parameters (`scratch/build_check.py`); static audit clean (`scratch/static_audit.py`).

## Deliberately left unchanged

- **V053** (PRODUCTIVE, predicted OSCILLATING): a `.editorconfig` edit, revert and re-edit around
  passing builds, with progress that is invisible in the log (an earlier edit of another file).
  Treating a lone build success as progress would weaken the OSCILLATING and STALLED rules on many
  other windows, so this window is accepted as a loss.
- **The search intent is its scope token**, not the first query word (which the Phase A policy
  used): the four-layer, 128-dimension budget has no room for a second gathered token. On the
  disclosed windows the two definitions give the same 200 labels.
- **Reversions stay within the window** (`b-7..b-1`), as the oracle text states; no disclosed
  window suggested a longer horizon.
- **Rule order, thresholds and the logit encoding** (above).
- **Behaviour on sessions longer than 63 blocks**: the nearest-same-key heads put every distance
  >= 63 in one bias bucket, so ties there are averaged rather than resolved to the nearest block.
  The disclosed sessions have at most 48 blocks; the policy picks the nearest block.
