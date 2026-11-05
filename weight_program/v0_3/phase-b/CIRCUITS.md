# CIRCUITS.md: the hand-programmed doom-loop detector (Phase B revision)

`build_weights.py` sets 156,788 of the 2,128,932 parameters; everything else is exactly 0.
The circuit computes `reference_policy.py` exactly (same features, same thresholds, same rule
order); on the 200 disclosed windows the network and the policy agree at every one of the 4,484
prefixes, and on the 381 Phase A windows. This file gives the policy, the residual-subspace
allocation, every head, every MLP unit, the head readout, and an account of every non-zero block.

## 1. Policy (what `reference_policy.py` computes)

Blocks are the event blocks of the prefix; the window is the last 8 blocks `t-7..t`. Relations
look at the whole prefix unless a horizon is stated.

Per-block features of block `b` (tokens `tb`, including the trailing `<EOE>`):

| feature | definition |
|---|---|
| `tool` | the token at offset 7 if it is a tool name (the timestamp fills offsets 0-6) |
| `status` | the **last** status word of the block: `ok`/`ready` = OK, `error`/`pending`/`FAILED` = FAIL |
| `pv` | the last digit group that directly follows a status word, capped at 100 (the passed count; 0 if none) |
| `has_hunk` | the block has a `@` hunk line (the workspace changed) |
| `intent` | the identity of the first hashed token at offset >= 10, else of the first other non-reserved token there (punctuation or digits; e.g. `path=.`); identities are the tokeniser id reduced to 13 bits |
| `cls` | tool class: inspect (`read_file`, `search`, `list_dir`, `plan`, `ask`, `yield`), mutate (`edit_file`, `write_file`, `revert`), validate (`run_tests`, `build`, `lint`), network (`http_call`, `poll_status`) |
| `key` | `(cls, intent)`: two blocks have the same key iff both agree |
| `h_old`, `h_new` | the hashed tokens just before / just after each `>` of the block (`h0` never counts) |

Relations of block `b`:

| relation | definition |
|---|---|
| `rev` | some `h_new` of `b` equals some `h_old` of a block in `b-7..b-1` (an undo / A-B-A return) |
| `prev` | the nearest earlier block with the same key; `exists`; `exists7` if it lies within `b-7..b-1` |
| `maxpv` | the largest `pv` among earlier same-key blocks |
| `c_fail` | the number of earlier same-key blocks with FAIL status |
| `nearest` | the nearest earlier block that is a validation (+1) or has a hunk line (-1); 0 if none |
| `same_status` | `prev` exists and has the same OK / FAIL status |
| `better` | `prev` exists and `pv > maxpv` (a new high) |

Block effect flags (approximations of the oracle's latent effects):

- `improved` = validation and `better` and (FAIL, or OK with `prev` not FAIL): measured progress; a FAILED -> OK transition is a resolution, not progress.
- `retry_base` = FAIL and `exists7` and `prev` FAIL and, for validations, `nearest == +1` (no workspace change since the last validation).
- `retry` = `retry_base` and, for validations, `pv >= prev.pv` and not `rev` (a narrowed re-run with a different count is a new attempt).
- `inpol` = `retry` and `c_fail < 4` (polls: `< 6`); `kill` = `retry` and not within those counts (`R_max = 3`, `P_max = 5`).
- `resolv` = OK and `prev` FAIL and, for validations, `prev.retry_base` (the failure had been retried; for http / polls / inspections the first successful repeat already resolves).
- `stagnant` = validation and `same_status` and not `better` and not `rev` (a re-validation that reveals nothing new).
- `futile` = mutation with a hunk line, not FAIL, not `rev`, and the nearest earlier validation (any suite) repeated its own predecessor's status.
- `noop` = mutation with no hunk line and not FAIL; `redinsp` = inspection, OK, `exists` (a repeated read / search / listing / plan).
- `SI` (stall-ineffective) = `rev` or (`revert` and not `rev`) or `noop` or `futile` or `redinsp` or (`stagnant` and not `inpol`) or (`kill` and not a validation).

Window rule at `t` (first match wins), with `V` = number of `rev` blocks, `I` = number of `SI`
blocks, `prog` = some `improved` block in the window, `active` = the most recent block of the
prefix that is FAIL or `resolv` is `inpol`, `recent` = some `resolv` block among `t-3..t`:

1. **OSCILLATING** if `V >= 2` and not `prog`.
2. **STALLED** if `I >= 5`, `SI(t)`, not `prog`, not `active`.
3. **RECOVERY** if `active` or `recent`.
4. **PRODUCTIVE** otherwise (also the default for ambiguous windows).

## 2. Conventions

- Score = `q.k/4 + B[h, c]`. Query weights carry the factor 4, so all numbers below are scores.
- `M = 1e5`. Every head uses head dim 15 as a **key gate**: `k15 = -M * (#unmet conditions) + M * n * is_bos`,
  `q15 = 4`. A key scores `0` on the gate iff all its conditions hold; BOS always scores exactly `0`
  on the gate and on every match term (its match components are nulled), so the BOS key's score
  is exactly the bias-table entry `B[h, 64]`: BOS is the **sink** (value 0) or the **anchor**
  (value 1 in an anchor dim) of every head.
- Heads not listed are all-zero (uniform attention over zero values: no effect).
- "Same block" heads: `B[0] = 0`, `B[1..63] = -M`. Window heads: `B[0..7]` as stated, `B[8..63] = -M`.
  Whole-prefix heads: `B[1..63]` as stated (bucket 63 holds every distance >= 63).
- Offset code: 7 frequencies `w_i = pi / 1.5^i`, 14 dims `(cos w_i o, sin w_i o)`. The kernel
  `K(d) = sum_i cos(w_i d)` is 7 at `d = 0` and at most 5.27 for `d != 0`, `|d| <= 191`. A shift by
  `delta` is the block-diagonal rotation `R_delta` (closed form in `W_k`).
- Identity bits: every non-reserved id `t >= 128` gets the 13 sign bits of `(t - 2048) mod 8192`:
  equal ids give dot 13, different ids at most 11.
- Intent match: 13 identity bits plus 2 class bits (inspect `(+,+)`, mutate `(+,-)`, validate
  `(-,+)`, network `(-,-)`) at scale `A_INT = 800`: same key 12,000, anything else <= 10,400. The
  nearest-same-key heads add `-15 c` per event distance (at most `-945`), so a same-key block at
  any distance still beats every other key; their sink is 10,700.
- MLP unit `clip(x) = relu(x) - relu(x - 1)` (two units). A "gated clip" adds `KGATE * (flags - n)`
  with `KGATE = 1e4`, far below any threshold term (`KSTEP = 40` times a quantity in `[-100, 100]`).

## 3. Residual-subspace allocation (d = 128; dims 123-127 unused)

| dims | name | written by | content |
|---|---|---|---|
| 0-2 | BOS, EOE, NLAT | E | 1 at `<BOS>`, `<EOE>`, the `NL@` marker |
| 3 | HASHED | E | 1 for hashed words (id >= 2048) except `h0` |
| 4 | GT | E | 1 for `>` |
| 5 | NONRES | E | 1 for every non-reserved token (id >= 128) |
| 6, 7 | DIGIT, NUM | E | 1 for digit groups; the group's value (3-digit groups: 100) |
| 8-20 | BITS | E | 13 identity bits of every non-reserved token |
| 21-23 | ISSTAT, SOK, SFAIL | E | status-word flags (`ok`/`ready`; `error`/`pending`/`FAILED`) |
| 24-29 | TINSP, TVAL, TMUT, TREVERT, THTTP, TPOLL | E | tool flags on the tool-name token |
| 30-43 | CODE | P | the 14-dim offset code |
| 44-46 | OFFLIN, OFF7, OFFGE10 | P | `offset/192`; offset 7; offset >= 10 |
| 47-52 | BTINSP..BTPOLL | H0.0 | the block's tool flags, copied to every token |
| 53, 54 | BSOK, BSFAIL | H0.1 | the block's last status word so far |
| 55-67 | PBITS | H0.2 | identity bits of the previous token |
| 68-70 | PHASHED, PGT, PISSTAT | H0.2 | the previous token is hashed / `>` / a status word |
| 71 | HASHUNK | H0.3 | an `NL@` exists before me in my block |
| 72-84 | IBITS | H0.4 | identity bits of the block's intent token (gathered into the EOE) |
| 85 | VM | MLP 0 | `clip(BTVAL + HASHUNK)`: eligible for the nearest validation/hunk lookup |
| 86 | REVMATCH | H1.0 | my id equals an `h_old` of blocks 1..7 back |
| 87 | PV | H1.1 | passed count (<= 100) |
| 88-90 | EXISTS, PSOK, PSFAIL | H1.2 | a same-key block exists; its status |
| 91 | NEAREST | H1.5 | nearest validation (+1) or hunk (-1) block |
| 92, 93 | OUTFAIL, ANCH | H1.4 | `c_fail/(1+c_fail)`, `1/(1+c_fail)` (BOS anchor) |
| 94 | EXISTS7 | H1.3 | a same-key block within 1..7 back exists |
| 95-97 | NOTIDST, CNT3, CNT5 | MLP 1 | status differs from `prev` (0..2); `c_fail <= 3`; `c_fail <= 5` |
| 98-100 | REVTOK, NOOP, REDINSP | MLP 1 | reverting `h_new` token; re-applied edit; repeated inspection |
| 101-104 | PPV, PEX7, PPSFAIL, PNEAR | H2.0 | `prev`'s count, `exists7`, `psfail`, `nearest` |
| 105 | MAXPV | H2.1 | largest count among earlier blocks with the same intent id |
| 106, 107 | NVEX, NVNOTST | H2.2 | the nearest earlier validation: exists / status differs from its own prev |
| 108 | REVBLK | H2.3 | block has a reverting hunk line |
| 109-112 | SI, INPOL, RESOLV, IMPROVED | MLP 2 | block effect flags |
| 113-116 | AVGSI, AVGREV, AVGIMP, ANCH8 | H3.0 | `I/(1+n)`, `V/(1+n)`, `#improved/(1+n)`, `1/(1+n)` over the window |
| 117 | ACT | H3.1 | INPOL of the nearest failing-or-resolving block |
| 118 | REC4 | H3.2 | a resolving block among the last four |
| 119-122 | V2, I5, PROG, RECB | MLP 3 | decision bits |

## 4. Embeddings

- `tok_emb` rows: `<BOS>`, `<EOE>`, `NL@` -> their flags; every id >= 128 -> NONRES and BITS
  (closed form from the id); ids >= 2048 -> HASHED (the row of `h0` is reset to 0); `>` -> GT;
  the 1,110 digit-group rows -> DIGIT and NUM; the 5 status words -> ISSTAT plus SOK / SFAIL;
  the tool rows -> their class flags (`revert` also TREVERT, `http_call` THTTP, `poll_status`
  TPOLL).
- `off_emb` rows: CODE (14 values), OFFLIN = `o/192`, OFF7 at row 7, OFFGE10 at rows 10..191.

## 5. Heads

Notation: `q` = query subspace, `k` = key subspace (match terms + gate conditions), `B` =
event-distance bias pattern, `v -> out` = value subspace and the residual dims written.

### Layer 0 (within-block structure; all same-block: `B[0] = 0`, `B[1..63] = -M`)

| head | q | k | B[BOS] | v -> out |
|---|---|---|---|---|
| H0.0 | gate only | gate OFF7 | -M/2 | TINSP..TPOLL -> BTINSP..BTPOLL |
| H0.1 | gate, `q14 = 4` | `k14 = 4000*OFFLIN` (latest status wins by >= 20.8); gate ISSTAT | -M/2 | SOK, SFAIL -> BSOK, BSFAIL |
| H0.2 | `4*30*CODE` | `R_+1 CODE` (BOS nulled) | 184.5 (hit 210, miss <= 158) | BITS -> PBITS; HASHED, GT, ISSTAT -> PHASHED, PGT, PISSTAT |
| H0.3 | gate only | gate NLAT | -M/2 | NLAT -> HASHUNK |
| H0.4 | `q0 = 4*6000`, `q1 = -4*3000` | `k0 = HASHED`, `k1 = OFFLIN`; gate NONRES, OFFGE10 (score `6000*hashed - 3000*offset/192`: the earliest hashed token wins, else the earliest other non-reserved token) | -M/2 | BITS -> IBITS |

MLP 0 (units 0-1): `VM = clip(BTVAL + HASHUNK)`.

### Layer 1 (token matching, same-key relations)

| head | q | k | B | v -> out |
|---|---|---|---|---|
| H1.0 | `4*20*BITS` | PBITS; gate PHASHED, GT | `B[0] = -M`, `B[1..7] = 0`, `B[8..] = -M`, BOS 240 (hit 260, miss <= 220) | GT -> REVMATCH |
| H1.1 | gate, `q14 = 4` | `k14 = 4000*OFFLIN`; gate DIGIT, PISSTAT | same-block, BOS -M/2 | NUM -> PV |
| H1.2 | intent match | intent match; gate EOE | `B[0] = -M`, `B[c] = -15c` (c = 1..63), BOS 10,700 | EOE, BSOK, BSFAIL -> EXISTS, PSOK, PSFAIL |
| H1.3 | intent match | intent match; gate EOE | `B[0] = -M`, `B[1..7] = 0`, `B[8..] = -M`, BOS 11,200 | EOE -> EXISTS7 |
| H1.4 | intent match | intent match; gate EOE | `B[0] = -M`, `B[1..63] = 0`, BOS 12,000 (anchor = exact match) | BSFAIL -> OUTFAIL; BOS -> ANCH |
| H1.5 | gate only | gate EOE, VM | `B[0] = -M`, `B[c] = -15c` (c = 1..63), BOS -960 | BTVAL - HASHUNK -> NEAREST |

Because every same-key key and the BOS anchor of H1.4 score exactly 12,000, the softmax gives
each weight `1/(1+c_fail)` so that `c_fail <= K` is the linear test `(K + 0.5) ANCH - OUTFAIL >= 0`.

MLP 1 (units 0-10): `NOTIDST = relu(BSOK - PSOK) + relu(PSOK - BSOK) + relu(BSFAIL - PSFAIL) + relu(PSFAIL - BSFAIL)`;
`CNT3 = clip(40 (3.5 ANCH - OUTFAIL))`; `CNT5 = clip(40 (5.5 ANCH - OUTFAIL))`;
`REVTOK = relu(REVMATCH + PGT + HASHED - 2)`; `NOOP = relu(BTMUT - HASHUNK - BSFAIL)`;
`REDINSP = relu(BTINSP + BSOK + EXISTS - 2)`.

### Layer 2 (second-hop relations, block flags)

| head | q | k | B | v -> out |
|---|---|---|---|---|
| H2.0 | intent match | intent match; gate EOE | as H1.2 | PV, EXISTS7, PSFAIL, NEAREST -> PPV, PEX7, PPSFAIL, PNEAR |
| H2.1 | `4*800*IBITS`, `q13 = 4*12` | IBITS, `k13 = PV`; gate EOE | `B[0] = -M`, `B[1..63] = 0`, BOS 10,200 (same id >= 10,400 + 12 pv; other <= 10,000) | PV -> MAXPV |
| H2.2 | gate only | gate EOE, BTVAL | `B[0] = -M`, `B[c] = -15c`, BOS -960 | EXISTS, NOTIDST -> NVEX, NVNOTST |
| H2.3 | gate only | gate REVTOK | same-block, BOS -M/2 | REVTOK -> REVBLK |

MLP 2 (units 0-19; the SI terms are mutually exclusive so SI stays in {0, 1}):

| units | circuit | expression -> target |
|---|---|---|
| 0-3 | measured progress | gated clip `40 (PV - MAXPV - 0.5)`, gates `BSFAIL + EXISTS = 2`, and `BSOK + EXISTS - PSFAIL = 2` -> IMPROVED |
| 4-5 | reversion | `clip(40 REVBLK - 20)` -> SI |
| 6 | revert tool | `relu(BTREVERT - REVBLK)` -> SI |
| 7, 8 | re-applied edit, repeated inspection | `relu(NOOP)`, `relu(REDINSP)` -> SI |
| 9 | futile edit | `relu(BTMUT + HASHUNK - BSFAIL + NVEX - NVNOTST - REVBLK - 2)` -> SI |
| 10-11 | stagnant validation | gated clip `40 (MAXPV - PV + 0.5)`, gate `BTVAL + EXISTS - NOTIDST - REVBLK = 2` -> SI |
| 12-13 | in-policy validation retry | gated clip `40 (PV - PPV + 0.5)`, gate `BTVAL + BSFAIL + EXISTS7 + PSFAIL + NEAREST + CNT3 - REVBLK = 6` -> INPOL (+1), SI (-1) |
| 14, 15 | http / inspection retry | `relu(BTHTTP + BTINSP + BSFAIL + EXISTS7 + PSFAIL + CNT3 - 4)` -> INPOL; the same with `- CNT3 - 3` -> SI (beyond policy) |
| 16, 17 | poll retry | `relu(BTPOLL + BSFAIL + EXISTS7 + PSFAIL + CNT5 - 4)` -> INPOL; the same with `- CNT5 - 3` -> SI |
| 18 | validation resolution | `relu(BTVAL + BSOK + PSFAIL + PEX7 + PPSFAIL + PNEAR - 5)` -> RESOLV |
| 19 | network / inspection resolution | `relu(BTHTTP + BTPOLL + BTINSP + BSOK + PSFAIL - 2)` -> RESOLV |

### Layer 3 (window aggregation, decision)

| head | q | k | B | v -> out |
|---|---|---|---|---|
| H3.0 | gate only | gate EOE | `B[0..7] = 0`, `B[8..] = -M`, BOS 0 (anchor) | SI, REVBLK, IMPROVED, BOS -> AVGSI, AVGREV, AVGIMP, ANCH8 |
| H3.1 | gate only | gate EOE and (BSFAIL or RESOLV) | `B[0] = 0`, `B[c] = -15c` (c = 1..63), BOS -960 | INPOL -> ACT |
| H3.2 | gate only | gate EOE, RESOLV | `B[0..3] = 0`, `B[4..] = -M`, BOS -M/2 | RESOLV -> REC4 |

MLP 3 (units 0-7): `V2 = clip(40 (AVGREV - 1.5 ANCH8))`; `I5 = clip(40 (AVGSI - 4.5 ANCH8))`;
`PROG = clip(40 (AVGIMP - 0.5 ANCH8))`; `RECB = clip(ACT + REC4)`.

## 6. Head readout: which policy branch each logit implements

| logit | expression | active value | inactive |
|---|---|---|---|
| OSCILLATING (3) | `60 V2 - 100 PROG - 5` | 55 | <= -5 |
| STALLED (2) | `50 I5 + 50 SI - 100 ACT - 100 PROG - 80` | 20 | <= -30 |
| RECOVERY (1) | `15 RECB - 5` | 10 | -5 |
| PRODUCTIVE (0) | 0 | | |

The argmax reproduces the first-match-wins order: OSCILLATING (55) beats STALLED (20) beats
RECOVERY (10) beats PRODUCTIVE (0); `prog` disables rules 1-2 but not 3; an active in-policy
retry disables rule 2. All bits are clipped to {0, 1}, so the logits are these integers up to
attention residue of order `e^-15`.

## 7. Account of every non-zero tensor block

| tensor | non-zero block(s) |
|---|---|
| `tok_emb.weight` (152,006) | 3 marker rows; NONRES + 13 BITS on the 10,112 rows with id >= 128; HASHED on 8,191 hashed rows; GT; DIGIT on 1,110 rows and NUM on the 1,108 of them with non-zero value; ISSTAT (5), SOK (2), SFAIL (3); 15 tool-flag entries |
| `off_emb.weight` (3,055) | all 192 rows: CODE (`sin 0 = 0` at row 0), OFFLIN (rows 1-191), OFF7, OFFGE10 (rows 10-191) |
| `layers.{l}.attn.rel_bias` | rows of the listed heads (H0.0-H0.4, H1.0-H1.5, H2.0-H2.3, H3.0-H3.2): `-M` default with the stated same-block / distance / BOS entries; unused heads stay 0 |
| `layers.{l}.attn.q.weight` | rows `16h + hd` of the match subspaces: CODE (H0.2), BITS (H1.0), IBITS + class dims (H1.2-H1.4, H2.0), IBITS (H2.1) |
| `layers.{l}.attn.q.bias` | `4` on the gate dim 15 of every listed head; `4` on dim 14 of H0.1, H1.1; `24000`, `-12000` on dims 0, 1 of H0.4; `48` on dim 13 of H2.1 |
| `layers.{l}.attn.k.weight` | per listed head: the match rows (rotated CODE with the BOS-null column, PBITS, IBITS + class dims, PV slope of H2.1, HASHED / OFFLIN of H0.4, OFFLIN of H0.1 / H1.1), and the gate row (`+M` on each condition flag, `+nM` on BOS) |
| `layers.{l}.attn.k.bias` | `-nM` on the gate dim of every listed head |
| `layers.{l}.attn.v.weight` | the value subspaces of section 5 (rows `16h + hd`) |
| `layers.{l}.attn.o.weight` | one `1.0` per value dim, routing to the "out" dims of section 5 (`-1.0` for HASHUNK in H1.5) |
| `layers.{l}.mlp.fc1.weight`, `.bias` | the units of section 5 (2, 11, 20 and 8 units per layer), weights as stated |
| `layers.{l}.mlp.fc2.weight` | `+1` / `-1` from each unit to its target dim (the `-1` entries are the second half of each clip and the SI side of the in-policy retry) |
| `head.weight`, `head.bias` | the 7 weights and 3 biases of section 6 |
| all `attn.o.bias`, `mlp.fc2.bias` | zero |
