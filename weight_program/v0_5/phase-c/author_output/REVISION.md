# REVISION.md — v05 Phase C revision of Phase B's network

## 1. What was disclosed, and the base

- First report: `disclosure/DISCLOSURE.json`, sha256
  `fb0f9608b4703523bcc616b444e67ca29b4f9bff8d40b0b754c933ca56acf6d1` (also in `disclosure/DISCLOSURE.sha256`):
  the 200 validation windows V001–V200 Phase B saw (50 per oracle class, at most one per session), each with the
  rendered prefix (window marker at block t−7), the oracle label and **Phase B's** network's class at `EOE_t`.
- Second report: `disclosure/DISCLOSURE_W.json`, sha256
  `96cad550f7461b3c1b1f9fbe0842a0deaae7d53e842743a8fa746d8628f48d1c` (also in `disclosure/DISCLOSURE_W.sha256`):
  200 windows W001–W200 from sessions of a second, independently written log generator (50 per class, at most
  one per session), with the same three fields. Nothing about that generator was disclosed.
- No reference-policy class exists in either report (no session in this lineage wrote one).
- Base: `phase_b/artifact.dlw`, sha256 `223d074372bf726a0c89f28afe1e1fcf9b53e3bb0ff3973f0cc3acd932661001`
  (r1's weights with Phase B's `REVISION.edits`; `./py -m v05_load b` confirms the hash).
- Deliverable: `REVISION.edits` (430 statements, all applied; `./py -m v05_load check` reports that the harness
  would accept it). It is written by `scratch/gen_revision.py`, which only prints hand-chosen constants (the
  16-dimensional one-hot code blocks and the two-unit clip pairs are typed out by loops). No value in it comes
  from a search, a fit or an optimisation over the disclosed windows; every number is a threshold or a gate
  weight chosen as described below, and the rules were checked (not tuned) with `scratch/rules.py`, a plain
  Python re-statement of the observable rules, before being compiled into weights.

Self-check on the disclosed reports (not an evaluation): Phase B's network has macro-F1 0.995 on V001–V200 and
0.795 on W001–W200 (false-alarm rate 0.34: 23 PRODUCTIVE windows called STALLED, 7 RECOVERY windows called
STALLED). The revised network has macro-F1 0.985 on V (V051, V134, V146 wrong) and 0.975 on W (W002, W005, W024,
W079, W132 wrong).

## 2. Diagnosis: what the second generator does differently

Reading the 200 W windows (`scratch/survey.py`, `scratch/stats.py`) showed conventions Phase B's rules had not
met in V:

1. **The error line is constant within a session.** Every failing `run_tests` of a W session reports the same
   location (`lib/shards/contract_0.*:NNN`), whatever the passed count (W008: `FAILED 1/13`, `2/13`, `3/13`,
   `4/13`, all at `:365`). Phase B identified a failure by its line alone, so every failing run after the first
   was "a repeated failure", every write within 3 events of one was "an edit that changed nothing", and
   incremental repairs became STALLED (W008, W033, W035, W041, W049, W052, W082, W085 and 15 more). In V the line
   is the hash of the failing test and changes with the failure, which is why the line alone worked there
   (45 of 50 STALLED sessions in V have a single failing line, but so do 39 of 50 PRODUCTIVE sessions in W).
2. **Progress shows in the passed count**, and a repair's first failure after a successful edit shares its count
   with the retries that follow it (W025: `3/17` at event 7 after a write, then `3/17` ×4 at 9–12, RECOVERY at 12;
   W131, W196 likewise with `handoff` plans between). A count-based tally therefore runs one ahead of the retry
   number.
3. **Plan actions are `verb:argument`**: `record:lib/shards/contract_7.lua`, `isolate:…`, `trace:…`,
   `handoff:42:0`, `handoff:43:1`, … (329 `W:W` and 79 `W:D:D` plans in W; every V plan is a single word). Phase B
   keyed plans on the word at offset 10, so every `handoff` after the first was redundant; four consecutive
   handoffs plus one other repeat made a PRODUCTIVE window STALLED (W055, W057, W063, W077).
4. **Append-writes**: in W's stall loops the agent writes the same file again and again with an `h0 >` hunk (the
   removed side is empty: pure insertions) and the count never moves; 129 such writes sit in W's STALLED windows,
   none in any V window. When the tests are sparse (W028, W065, W099, W117: write, re-read, write, re-read, …)
   Phase B's "edit within 3 events after a repeated test result" rule missed the writes and the window fell to
   PRODUCTIVE.
5. **A first read of a file counts as progress.** W047 (`FAILED 3/12` ×10, a read of a never-read file, the same
   failure again) is PRODUCTIVE; W042, W058, W067, W081, W083 are PRODUCTIVE windows in which a fresh read is the
   only sign of progress. W's STALLED windows contain no fresh read_file at all (69 in its PRODUCTIVE windows). In
   V a fresh read is "novel but fruitless" (V134, V146 are STALLED with one), so this is a genuine difference
   between the generators; the rule costs those two V windows and gains six W windows.
6. **`http_call` events carry hunk lines** in W (28 events: `POST …/install/` changes `cedar.artifact`), and
   Phase B counted their alternating ids as reversions (W132: STALLED called OSCILLATING).
7. **A re-run right after a fresh read is a new attempt, not a retry** (W053: `FAILED 0/15`, read, `0/15`, read,
   `0/15`, read, `0/15` is PRODUCTIVE; Phase B's "same failure within 2 events" made the last run an in-policy
   retry).
8. **Reversions are judged against states held inside the window.** W080 (PRODUCTIVE), W168 (RECOVERY): the first
   event of the window reverts a state held only before the window; Phase B's 7-event lookback counted it as the
   second reversion.

## 3. The rules the revised network implements (and the windows behind them)

All thresholds are the oracle's parameters; ranges are Phase B's (7 events for reversions, 15 for lines, 31 for
targets and counts).

1. **Same failure** = same error line within 15 events AND (when the failure has a passed count) same passed
   count within 31 events. Such a failure is ineffective (W008 …: a failure with a new count is not; V006, V013,
   V062, V066: repeated line and count still are). The count tally is also what triggers "edit after a repeated
   test result" (EDR) and the `ok N/N`-after-failure resolution.
2. **In policy** = a retry (same line within 2 events, same count when there is one, no mutation and **no
   read_file** between) with fewer than 4 earlier identical lines **or** fewer than 5 earlier equal counts. The
   extra count match absorbs the repair's first failure, which shares the retries' count (W025, W131, W196 are
   RECOVERY with 3 retries after it; W116, W174 are STALLED with 8). V154, V174, V176 (3 retries, RECOVERY) and
   V037, V062, V066 (4+, STALLED) keep their classes. The read_file condition handles W053 (and V176, V157, whose
   retries do not follow a read).
3. **Plan repeats** are keyed on the word after the verb when the action is `verb:word`, on the verb when the
   action is one word, and never on `verb:digits` (W055, W057, W063, W077: handoffs are not repeats; V134: the
   repeated `checkpoint` / `review_diff` plans still are).
4. **An append** (`h0 >` hunk) to a path mutated within the last 31 events is ineffective (W028, W065, W099,
   W117). A first append to a path (W061, event 17) is not. V has no `h0 >` hunk at all.
5. **Reversions** count only for edit/write/revert events (W132, W179, W186) and only over the last 7 events of
   the window (W080, W168).
6. **A fresh read_file** (a path not read within 31 events) anywhere in the last 8 events suppresses STALLED
   (W042, W047, W058, W067, W081, W083; costs V134, V146). `list_dir` is deliberately excluded: V007, V163, V172
   are STALLED windows with a fresh `list_dir`.

## 4. Every change (statements by section of `REVISION.edits`, reason, windows)

Section letters are the `# ====` headings in `REVISION.edits`; `clip` pairs are two units with biases b, b−1.

| section | statements | change | reason | windows |
|---|---|---|---|---|
| A1 | `tok_emb.weight[5861, 30] = 1` | d30 = the hunk id `h0` | the append rule needs to see an empty removed side | W028, W065, W099, W117 |
| A2, B1 | `tok_emb.weight[7, 127] = 1`; `layers.0.attn.v.weight[40, 127] = 1`, `o.weight[127, 40] = 1` | d127 = read_file flag, broadcast over the event by head 2 | the fresh-read rule and the retry chain-breaker need read_file (not list_dir) | W047, W053; V007, V163, V172 (why list_dir is excluded) |
| C1, C2 | units 242–273 of layer-0 MLP (code A/B + hashed + offset 10 + mutation class − 3; fc2 −3 / −1 into block 3 A / B) | role-signed path keys for edit/write/revert events | "path mutated before" for the append rule, without letting reads match mutations (a read after an edit stays a non-repeat) | W028, W061 |
| C3, C4 | units 274–305 (code A/B + hashed + prev1 `:` + plan class + 24·o − o² − 143 − 4; fc2 +1) | plan key at offset 12 | `record:path`, `isolate:path` are keyed on the path; `handoff:42:0` gets no code key there | W055, W057, W063, W077 |
| C5 | unit 306: `relu(d30 + d27 + d8 − 2)` → d112 | H0 flag at the h_old position of a mutation | append detection | W028 |
| E1 | `layers.1.mlp.fc1.weight[0, 8] = 1`, `bias[0] = -4` | REVg gated on the mutation class | http_call hunks are not agent mutations | W132 (and W179, W186 unchanged) |
| E2 | `layers.1.mlp.fc1.weight[1, 18] = -10`, `[2, 18] = -10`, `[1, 8] = -10`, `[2, 8] = -10` | REPg excludes plan and mutation keys (REPg_np) | those tokens get their own flags (E5, E6) | — |
| E3 | units 13, 14 zeroed | KV (d127) dropped | d127 is re-used; the EOE-level tally is now g_kpenL / g_kpenC | — |
| E4 | units 18, 19 rewritten: `clip(8·d113 + (10/3)·ΣB2 + 10·FAILED + 10·digit − 31.5)` → d52 | FAILREP: a FAILED count seen within 31 events (was: a line seen within 15) | the EDR trigger and the `ok`-after-failure resolution must not fire on a repair's progressing failures | W008, W035, W041, W049, W052, W082, W085 |
| E5 | units 91, 92: `clip(2·d26 + 4·d9 + ⅔·ΣB3 − 19 + 10·plan − 20·o7)` → d89 | REPg_plan | the plan gather reads it at the key tokens; the −20·o7 term stops the tool token (class flag 2) from firing it | W063; the tool-token leak was seen on W061 (iteration 9) |
| E6 | units 93, 94: `clip(−2·d26 − 4·d9 − ⅔·ΣB3 − 19 + 10·mutation − 20·o7)` → d91 | REPMUT: the path was mutated within 31 events | append rule | W028, W065, W099, W117 |
| E7 | units 95, 96: `clip(8·d113 + (10/3)·ΣB2 + 10·digit − 21.5)` → d87 | CNT15g: the count was seen within 31 events | the AND of line and count (rule 1) | W008 and the other 22 PRODUCTIVE→STALLED windows |
| E8 | units 97, 98: `clip(100·d113 + (10/3)·ΣB2 + 10·digit − 101.5)` → d88 | kpenC: at least 5 earlier equal counts | rule 2 | W025, W131, W196 |
| E9 | units 99, 100 (clip partners of 22, 23; fc2 −1 into d54) | kpen becomes 0/1 | INPOL's new threshold form needs binary penalties | — |
| E10 | units 101–118 | zero d87–95 | block-1 B residue would corrupt the new flags | — |
| E11 | units 119–134 | cancel the negative block-3 B keys | keeps d48–63 clean for the flags that live there | — |
| E12 | units 135, 136 → d90 (1 at offset 10, 2 at offset 12) | plankey | lets one gather head prefer the offset-12 key | W063 |
| F1, F2 | head 0 loses the d52 and d127 values; head 1 adds d87 → d61, d88 → d62 | gathers for CNT15g and kpenC | rules 1, 2 | — |
| F3 | `layers.2.attn.v.weight[48, 127] = 1` | prevX = previous event was a mutation or a read_file | rule 2 | W053 |
| F4, F5 | head 4 keys d50 + d51 + d52 + d123, values 1 each → d70–73; heads 5–7 cleared | one "recent flag" head instead of four | frees three heads; with at most one flagged token per event the values stay ≥ 1/3 | — |
| F6, F7, F8 | heads 5 (d112 → d78), 6 (key d90, value d89 → d92), 7 (d91 → d93) | gathers for H0, PLANREP, REPMUT | rules 3, 4 | — |
| G1 | units 0, 1: drop d59, add d92 | INEFF part 1 without the line match, with plan repeats | rules 1, 3 | — |
| G2 | units 13, 14: `clip(g_SAME + g_CNT15 − g_hascount)` → d75 | SAME_lc | rule 1 | — |
| G3 | units 15, 16: `clip(g_H0 + g_REPMUT − 1 − 3·(r_FAILREP + r_CNTOK + g_REV))` → d75 | H0RED, silent when EDR already counts the event (INEFF stays ≤ 1) | rule 4 | — |
| G4 | unit 17: `relu(d127 − d67)` → d94 | FRESH | rule 6 | — |
| G4b | `fc2.weight[76, 6:12] = ±3` | RES scaled ×3 | a recent flag now reads 1/m; the resolution head and resw_b need RES ≥ 1 | iteration 8: without it every ok event with any recent flag resolved (V004, W004 …) |
| G5 | unit 12 rewritten (weights 10/10/10/−6/−4/−4/10/10/10/−10, bias −19) | INPOL with the two-tally penalty and the read_file breaker | rule 2 | W025, W053, W131, W196 |
| H1 | head 0: d66 value removed; d94 → d95 (A3) | window sum of FRESH | rule 6 | — |
| H2 | head 3: q.bias 1, key 80·d5, value d66 → d79, bias 0 at 0..6, BOS 20 | reversion sum over 7 events | rule 5 | W080, W168 |
| I1 | units 8, 9: `clip(18·A3 − 1)` → d112 | fresh_b | rule 6 | — |
| J1 | `head.weight[2, 112] = -12` | STALLED suppressed by a fresh read | rule 6 | W047 |

## 5. Deliberately left unchanged

- Phase B's whole structure (r1's code keys, the matching heads, the gathers, the window heads, the ramps and
  the logit combination), its thresholds and ranges, and its handling of http errors, polls, no-op mutations,
  repeated reads/searches/lint targets and resolutions: they are right on both reports.
- V051, W002 (PRODUCTIVE with two or three reversions and progress by the oracle's progress rule): the network
  has no numeric progress estimate, as in Phase B. W024 and W079 (PRODUCTIVE windows whose last event is a real
  edit within 3 events of a repeated failure, after a run of append-writes) are the same conflict as V's
  scenario 6 (V027, V060, V099 …), where that edit is ineffective; V wins. W005 (RECOVERY: a retry chain cut by a
  mutation between two identical failures) needs a chain-length count that attention over a fixed range cannot
  give; its count tally reaches 5 and the window is OSCILLATING by the 7-event lookback. W132 (two files
  alternating so that the joint workspace state recurs once) needs joint-state reversions; per-file reversions
  count two.
- V134 and V146 (STALLED windows containing a fresh read) are given up for the fresh-read rule, on the evidence
  above (0 of 50 W STALLED windows versus 69 fresh reads in W PRODUCTIVE windows).
- The reversion count over 7 events still over-counts an event whose match lies before the window (an exact
  window-relative count needs a per-distance lookback that one head cannot express); only the first event of
  the window is excluded.
- `.`-prefixed paths, `lines=` ranges, ports in http hosts and the 1/140 code-A collision rate keep Phase B's
  behaviour.
