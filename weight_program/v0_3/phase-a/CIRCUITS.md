# CIRCUITS.md: the hand-programmed doom-loop detector (Phase A)

`build_weights.py` sets 121,422 of the 2,128,932 parameters; everything else is exactly 0.
The circuit computes `reference_policy.py` exactly (same features, same thresholds, same rule
order). This file gives the policy, the residual-subspace allocation, every head, every MLP unit,
the head readout, and an account of every non-zero tensor block.

## 1. Policy (what `reference_policy.py` computes)

Blocks are the event blocks of the prefix; the window is the last 8 blocks `t-7..t`.

Per-block features of block `b` (tokens `tb`, including the trailing `<EOE>`):

| feature | definition |
|---|---|
| `tool` | the token at offset 7 if it is a tool name (the timestamp fills offsets 0-6) |
| `status` | the **last** status word of the block: `ok`/`ready` = OK, `error`/`pending` = FAIL, `FAILED` = validation failure |
| `has_hunk` | the block has a `@` hunk line (the workspace changed) |
| `has_passed`, `pv` | a `passed` word exists; `pv` = the digit group 3 tokens before the last `passed`, / 100 (`?` reads as 0) |
| `intent` | the tokeniser hash bucket of the intent-slot token: offset 13 for `http_call`, 14 for `search` (first query word), 10 otherwise; None if that token is not a hashed word |
| `cls` | tool class: inspect (`read_file`, `search`, `list_dir`), mutate (`edit_file`, `write_file`, `revert`), validate (`run_tests`, `build`, `lint`), network (`http_call`, `poll_status`), none (`plan`, `ask`, `yield`) |
| `h_old`, `h_new` | the tokens 2 and 4 after each `@` marker (`h0` is excluded as an `h_old`) |

Relations of block `b` over blocks `b-7..b-1`:

| relation | definition |
|---|---|
| `rev` | some `h_new` of `b` equals (same bucket) some `h_old` of an earlier block: an undo / A-B-A return |
| `C_all`, `C_fail` | earlier blocks with the same `intent` and the same non-none `cls`; those of them with FAIL status |
| `nearest` | the nearest earlier block that is a validation or has a hunk line: +1 validation, -1 hunk, 0 none |

Block effect flags (approximations of the oracle's latent effects):

- `redval` = validation and `nearest == +1` (re-run on an unchanged workspace)
- `u1` = FAIL and not poll and `C_fail >= 4`; `u2` = FAIL and poll and `C_fail >= 5` (retry beyond policy); `kill = u1 or u2`
- `inpol` = FAIL and not `kill`; `resolv` = OK and `C_fail >= 1` (a retry that succeeded)
- `noop` = mutate and not `has_hunk` and not FAIL (re-applied edit); `redinsp` = inspect and `C_all - C_fail >= 1`
- `SI` (stall-ineffective) = `rev` or (`revert` and not `rev`) or `noop` or `redinsp` or `redval` or `kill`

Window rule at `t` (first match wins), with `V` = number of `rev` blocks, `I` = number of `SI`
blocks, `prog` = (latest validation's `pv` > earliest validation's `pv`) or (earliest FAILED and latest OK):

1. **OSCILLATING** if `V >= 2` and not `prog`.
2. **STALLED** if `I >= 5`, `SI(t)`, not `prog`, not `inpol(t)`.
3. **RECOVERY** if, scanning `t, t-1, t-2, t-3`, the first block that is `inpol`/`resolv`/`kill` is `inpol` or `resolv`.
4. **PRODUCTIVE** otherwise (also the default for ambiguous windows).

## 2. Conventions

- Score = `q.k/4 + B[h, c]`. Query weights carry the factor 4, so all numbers below are scores.
- `M = 1e4`. Every head uses head dim 15 as a **key gate**: `k15 = -M * (#unmet conditions) + M * n * is_bos`,
  `q15 = 4`. A key scores `0` on the gate iff all its conditions hold; BOS always scores exactly `0`
  on the gate and on every match term (its match components are nulled), so the BOS key's score
  is exactly the bias-table entry `B[h, 64]`: BOS is the **sink** (value 0) or the **anchor**
  (value 1 in an anchor dim) of every head.
- Heads not listed are all-zero (uniform attention over zero values: no effect).
- "Same block" heads: `B[0] = 0`, `B[1..63] = -M`. Window heads: `B[0..7]` as stated, `B[8..63] = -M`.
- Offset code: 7 frequencies `w_i = pi / 1.5^i`, 14 dims `(cos w_i o, sin w_i o)`. The kernel
  `K(d) = sum_i cos(w_i d)` is 7 at `d = 0` and at most 5.27 for `d != 0`, `|d| <= 191`. A shift by
  `delta` is the block-diagonal rotation `R_delta` (closed form in `W_k` or `W_v`).
- Id bits: hashed word ids `2048..10239` get 13 sign bits (`+-1`) of `id - 2048`: same word gives
  dot 13, different words at most 11.
- MLP unit `clip(x) = relu(x) - relu(x - 1)` (two units). "Gated clip" adds `KGATE * (flags - n)`
  with `KGATE = 1000`, which is far below any threshold term (`KSTEP = 40` times a quantity in `[-5, 5]`).

## 3. Residual-subspace allocation (d = 128; dims not listed are unused)

| dims | name | written by | content |
|---|---|---|---|
| 0 | BOS | E | 1 at `<BOS>` |
| 1 | EOE | E | 1 at `<EOE>` |
| 2 | NLAT | E | 1 at the `NL@` marker |
| 3 | HASHED | E | 1 for hashed words (id >= 2048) |
| 4 | H0 | E | 1 for the word `h0` |
| 5 | PASSEDW | E | 1 for `passed` |
| 6, 7 | DIGIT, NUM | E | 1 for digit groups; `int(group) / 100` |
| 8-20 | BITS | E, then H1.1 at EOE | 13 sign bits of the hashed id; the EOE receives its block's intent-token bits |
| 21-24 | ISSTAT, SOK, SFAIL, SFAILED | E | status-word flags (`ok`/`ready`; `error`/`pending`; `FAILED`) |
| 25-31 | TINSP, TVAL, TMUT, TREVERT, THTTP, TPOLL, TSEARCH | E | tool flags on the tool-name token |
| 32-45 | CODE | P | the 14-dim offset code |
| 46 | OFFLIN | P | `offset / 192` |
| 47-50 | OFF7, OFF10, OFF13, OFF14 | P | one-hot flags of offsets 7, 10, 13, 14 |
| 51, 52 | HOLD, HNEW | H0.0, H0.1 | "an `NL@` sits exactly 2 / 4 tokens before me" |
| 53-59 | BTINSP..BTSEARCH | H0.2 | the block's tool flags, copied to every token |
| 60-62 | BSOK, BSFAIL, BSFAILED | H0.3 | the block's last status word so far |
| 63, 64 | HASHUNK, HASPASSED | H0.5, H0.4 | an `NL@` / a `passed` word exists before me in my block |
| 65-78 | CODEP | H0.4 | `R_-3 code(offset of the last passed)` = the code of the passed count's offset |
| 79, 80 | SLOT, VM | MLP 0 | intent-slot flag; eligible for the nearest validation/hunk lookup (`clip(BTVAL + HASHUNK)`) |
| 81 | REVMATCH | H1.0 | my id equals an `h_old` of blocks 1..7 back |
| 82, 83 | PV, NEAREST | H1.2, H1.3 | passed count / 100; nearest validation (+1) or hunk (-1) block |
| 84, 85 | REVTOK, REDVAL | MLP 1 | this token is a reverting `h_new`; redundant validation |
| 86-89 | OUTALL, OUTFAIL, ANCH2, REVBLK | H2.0, H2.1 | `C_all/(1+C_all)`, `C_fail/(1+C_all)`, `1/(1+C_all)`; block has a reverting hunk |
| 90-92, 110 | SI, INPOL, RESOLV, KILL | MLP 2 | block effect flags |
| 93-96 | AVGSI, AVGREV, ANCH8, NEAR4 | H3.0, H3.1 | `I/(1+n)`, `V/(1+n)`, `1/(1+n)` over the window (`n` = blocks present); nearest recovery/kill sign |
| 98-103 | LPV, LFAILED, LOK, FPV, FFAILED, FOK | H3.2, H3.3 | latest / earliest validation: `pv`, FAILED, OK |
| 104-109 | V2, I5, A4, PROGN, PROGF, SIT | MLP 3 | decision bits: `V >= 2`, `I >= 5`, recovery, progress (count / fixed), `clip(SI(t))` |

## 4. Embeddings

- `tok_emb` rows: `<BOS>` -> BOS; `<EOE>` -> EOE; `NL@` -> NLAT; the 5 status words -> ISSTAT plus
  SOK / SFAIL / SFAILED; `passed` -> PASSEDW; 13 tool rows -> their class flags (`revert` also
  TREVERT, `search` also TSEARCH, `http_call` THTTP, `poll_status` TPOLL; `plan`/`ask`/`yield`
  have no flags); the 1,110 digit-group rows -> DIGIT and NUM = value/100; the 8,192 hashed rows ->
  HASHED and BITS (closed form from the id); the row of `h0` -> H0 as well.
- `off_emb` rows: CODE (14 values of `cos`/`sin`), OFFLIN = `o/192`, and the four one-hot flags.

## 5. Heads

Notation: `q` = query subspace (dims read by `W_q`), `k` = key subspace, `v -> out` = value
subspace and the residual dims written through `W_o`, `B` = event-distance bias pattern.

### Layer 0 (within-block structure; all heads same-block: `B[0] = 0`, `B[1..63] = -M`)

| head | q | k (match + gate conditions) | B[BOS] | v -> out |
|---|---|---|---|---|
| H0.0 | `4*30*CODE` | `R_+2 CODE` (BOS nulled); gate NLAT | 184.5 (between hit 210 and miss <= 158) | NLAT -> HOLD |
| H0.1 | `4*30*CODE` | `R_+4 CODE`; gate NLAT | 184.5 | NLAT -> HNEW |
| H0.2 | gate only | gate OFF7 | -M/2 | TINSP..TSEARCH -> BTINSP..BTSEARCH |
| H0.3 | gate, `q14 = 4` | `k14 = 4000*OFFLIN` (latest status wins by >= 20.8); gate ISSTAT | -M/2 | SOK, SFAIL, SFAILED -> BSOK, BSFAIL, BSFAILED |
| H0.4 | gate, `q14 = 4` | `k14 = 4000*OFFLIN`; gate PASSEDW | -M/2 | `R_-3 CODE` (BOS nulled) -> CODEP; PASSEDW -> HASPASSED |
| H0.5 | gate only | gate NLAT | -M/2 | NLAT -> HASHUNK |

MLP 0 (units 0-4): `SLOT = relu(OFF10 - BTHTTP - BTSEARCH) + relu(OFF13 + BTHTTP - 1) + relu(OFF14 + BTSEARCH - 1)`;
`VM = clip(BTVAL + HASHUNK)`.

### Layer 1 (token matching, gathers into the EOE)

| head | q | k | B | v -> out |
|---|---|---|---|---|
| H1.0 | `4*20*BITS` | BITS; gate HOLD, HASHED, not H0 | `B[0] = -M`, `B[1..7] = 0`, `B[8..] = -M`, BOS 240 (hit 260, miss <= 220) | HOLD -> REVMATCH |
| H1.1 | gate only | gate SLOT | same-block, BOS -M/2 | BITS -> BITS (the EOE's own bits are 0) |
| H1.2 | `4*30*CODEP` | CODE (BOS nulled); gate DIGIT | same-block, BOS 184.5 | NUM -> PV |
| H1.3 | gate only | gate EOE, VM | `B[0] = -M`, `B[c] = -20c` (c = 1..7), `B[8..] = -M`, BOS -160 | BTVAL - HASHUNK -> NEAREST |

MLP 1 (units 0-1): `REVTOK = relu(REVMATCH + HNEW + HASHED - 2)`; `REDVAL = relu(2*BTVAL + 2*NEAREST - 3)`.

### Layer 2 (EOE-to-EOE counting, block flags)

| head | q | k | B | v -> out |
|---|---|---|---|---|
| H2.0 | `4*20*[BITS, cls1, cls2]` with `cls1 = BTINSP + BTMUT - BTVAL - BTHTTP - BTPOLL`, `cls2 = BTINSP - BTMUT + BTVAL - BTHTTP - BTPOLL` | `[BITS, cls1, cls2]`; gate EOE | `B[0] = -M`, `B[1..7] = 0`, `B[8..] = -M`, BOS 300 = exact match (13 + 2) * 20; non-match <= 260 | EOE -> OUTALL; BSFAIL -> OUTFAIL; BOS -> ANCH2 |
| H2.1 | gate only | gate REVTOK | same-block, BOS -M/2 | REVTOK -> REVBLK |

Because every matching key and the BOS anchor score exactly 300, the softmax gives each weight
`1/(1+C_all)`, so `OUTALL = C_all/(1+C_all)`, `OUTFAIL = C_fail/(1+C_all)`, `ANCH2 = 1/(1+C_all)`,
and `C_fail >= K` is the linear test `OUTFAIL - (K - 0.5) * ANCH2 >= 0` (margin `0.5/8` times `KSTEP = 40`).

MLP 2 (units 0-16, all single-stage, terms mutually exclusive so sums stay in {0, 1}):

| units | circuit | expression -> target |
|---|---|---|
| 0-1 | reversion | `clip(40*REVBLK - 20)` -> SI |
| 2 | revert tool | `relu(BTREVERT - REVBLK)` -> SI |
| 3 | re-applied edit | `relu(BTMUT - HASHUNK - BSFAIL)` -> SI |
| 4 | redundant validation | `relu(REDVAL)` -> SI |
| 5-6 | redundant inspection | gated clip `40*(OUTALL - OUTFAIL - 0.5*ANCH2) + 1000*(BTINSP - 1)` -> SI |
| 7-8 | retry beyond policy (non-poll, `C_fail >= 4`) | gated clip `40*(OUTFAIL - 3.5*ANCH2) + 1000*(BSFAIL - BTPOLL - 1)` -> SI, KILL |
| 9-10 | poll beyond policy (`C_fail >= 5`) | gated clip `40*(OUTFAIL - 4.5*ANCH2) + 1000*(BSFAIL + BTPOLL - 2)` -> SI, KILL |
| 11-12 | in-policy failure (non-poll, `C_fail <= 3`) | gated clip `40*(3.5*ANCH2 - OUTFAIL) + 1000*(BSFAIL - BTPOLL - 1)` -> INPOL |
| 13-14 | in-policy poll (`C_fail <= 4`) | gated clip `40*(4.5*ANCH2 - OUTFAIL) + 1000*(BSFAIL + BTPOLL - 2)` -> INPOL |
| 15-16 | resolution (OK and `C_fail >= 1`) | gated clip `40*(OUTFAIL - 0.5*ANCH2) + 1000*(BSOK - 1)` -> RESOLV |

### Layer 3 (window aggregation, decision)

| head | q | k | B | v -> out |
|---|---|---|---|---|
| H3.0 | gate only | gate EOE | `B[0..7] = 0`, `B[8..] = -M`, BOS 0 (anchor) | SI -> AVGSI; REVBLK -> AVGREV; BOS -> ANCH8 |
| H3.1 | gate only | gate EOE and (INPOL + RESOLV + KILL) | `B[c] = -20c` (c = 0..3), `B[4..] = -M`, BOS -80 | INPOL + RESOLV - KILL -> NEAR4 |
| H3.2 | gate only | gate EOE, HASPASSED | `B[c] = -20c` (c = 0..7), `B[8..] = -M`, BOS -160 | PV, BSFAILED, BSOK -> LPV, LFAILED, LOK (latest validation) |
| H3.3 | gate only | gate EOE, HASPASSED | `B[c] = +20(c+1)` (c = 0..7), `B[8..] = -M`, BOS 0 | PV, BSFAILED, BSOK -> FPV, FFAILED, FOK (earliest validation) |

MLP 3 (units 0-10):

| units | expression -> target | policy quantity |
|---|---|---|
| 0-1 | `clip(40*(AVGREV - 1.5*ANCH8))` -> V2 | `V >= 2` |
| 2-3 | `clip(40*(AVGSI - 4.5*ANCH8))` -> I5 | `I >= 5` |
| 4-5 | `clip(40*NEAR4 - 20)` -> A4 | RECOVERY evidence |
| 6-7 | `clip(400*(LPV - FPV) - 2)` -> PROGN | latest `pv` > earliest `pv` |
| 8 | `relu(FFAILED + LOK - 1)` -> PROGF | earliest FAILED and latest OK |
| 9-10 | `clip(SI)` -> SIT | `SI(t)` |

## 6. Head readout: which policy branch each logit implements

| logit | expression | active value | inactive |
|---|---|---|---|
| OSCILLATING (3) | `60*V2 - 100*PROGN - 100*PROGF - 5` | 55 | <= -5 |
| STALLED (2) | `50*I5 + 50*SIT - 100*INPOL - 100*PROGN - 100*PROGF - 80` | 20 | <= -30 |
| RECOVERY (1) | `15*A4 - 5` | 10 | -5 |
| PRODUCTIVE (0) | 0 | | |

The argmax reproduces the first-match-wins order: OSCILLATING (55) beats STALLED (20) beats
RECOVERY (10) beats PRODUCTIVE (0); `prog` disables rules 1-2 but not 3; an in-policy failure at
`t` disables rule 2. All bits are clipped to {0, 1}, so the logits are these integers up to
attention residue of order `e^-20`.

## 7. Account of every non-zero tensor block

| tensor | non-zero block(s) |
|---|---|
| `tok_emb.weight` (116,933) | rows and dims of section 4: 3 marker rows, 5 status rows (2 dims each), `passed`, 13 tool rows, 1,110 digit rows (DIGIT; NUM except the three zero groups), 8,192 hashed rows (HASHED + 13 BITS), `h0` |
| `off_emb.weight` (2,876) | all 192 rows: CODE (14 dims, `sin 0 = 0` at row 0), OFFLIN (rows 1-191), OFF7/10/13/14 (one entry each) |
| `layers.{l}.attn.rel_bias` | rows of the heads listed above (H0.0-H0.5, H1.0-H1.3, H2.0-H2.1, H3.0-H3.3): `-M` default, the stated same-block / distance / BOS entries; unused heads stay 0 |
| `layers.{l}.attn.q.weight` | rows `16h + hd` of the match subspaces: CODE (H0.0, H0.1, H1.2 via CODEP), BITS (H1.0, H2.0) and the class dims of H2.0 |
| `layers.{l}.attn.q.bias` | `4` on the gate dim 15 of every listed head; `4` on dim 14 of H0.3, H0.4 |
| `layers.{l}.attn.k.weight` | per listed head: the match rows (rotated CODE with the BOS-null column, BITS, class dims), the gate row (`+M` on each condition flag, `-M` on H0 for H1.0, `+nM` on BOS), `4000` on OFFLIN for H0.3, H0.4 |
| `layers.{l}.attn.k.bias` | `-nM` on the gate dim of every listed head |
| `layers.{l}.attn.v.weight` | the value subspaces of section 5 (rows `16h + hd`); H0.4 includes the rotated CODE block and its BOS-null column |
| `layers.{l}.attn.o.weight` | one `1.0` per value dim, routing to the "out" dims of section 5 |
| `layers.{l}.mlp.fc1.weight`, `.bias` | the units of sections 5 (rows 0..4, 0..1, 0..16, 0..10 per layer), weights as stated |
| `layers.{l}.mlp.fc2.weight` | `+1` / `-1` from each unit to its target dim (the `-1` entries are the second half of each clip) |
| `head.weight`, `head.bias` | the 9 weights and 3 biases of section 6 |
| all `attn.o.bias`, `mlp.fc2.bias` | zero |
