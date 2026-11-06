# CIRCUITS.md — the revised network (r1's weights + `REVISION.edits`)

This describes the network the harness builds from `r1/artifact.dlw` with `REVISION.edits` applied. Section
numbering of the statements refers to the `# ====` section letters inside `REVISION.edits` (A–J). "r1" means
the weights r1 wrote; "kept" means the block is r1's and unchanged; "changed"/"new" means this revision
wrote it. Dimension `dN` is residual dimension N. `clip(x)` means `relu(x) − relu(x − 1)` (two MLP units
with biases differing by 1 and output weights +1/−1), which maps any pre-activation to [0, 1].

The decision the network implements (the oracle's rule order, with observable proxies):

```
OSCILLATING  if  at least 2 reverting mutations among the last 8 events
else RECOVERY if  the most recent outcome event in the last 8 is an in-policy retry (act)
else STALLED  if  at least 5 of the last 8 events are ineffective AND the current event is ineffective
else RECOVERY if  a resolution happened in the last 4 events (resw)
else PRODUCTIVE
```

Logits: `PRODUCTIVE = 0.5`, `RECOVERY = 8·act + 5·resw − 2`, `STALLED = 6·sc + 6·INEFF_t − 8·act − 8`,
`OSCILLATING = 12·osc`.

## 1. Residual-subspace allocation (128 dims)

| dims | content | written by |
|---|---|---|
| d0 | constant 1 (every token) | embedding (kept) |
| d1 | hashed word (ids 2048+) and the token `.` (id 141) | embedding (kept + A7) |
| d2 | digit group | embedding (kept) |
| d3 `>`, d4 `NL@`, d5 `EOE` | token flags | embedding (kept) |
| d6 read_file/list_dir, d7 search, d8 edit/write/revert, d11 build/lint, d12 run_tests, d13 poll_status, d16 http_call, d18 plan/ask/yield | tool-class flags on the tool token; layer-0 head 2 adds the same flag to every later token of the event (value 1 there, 2 on the tool token itself) | embedding (d6–d8 kept, others A3–A6) + layer-0 attention |
| d9 `:` flag, d10 ok/ready flag, d14 error, d15 FAILED, d17 pending | status/punctuation token flags | embedding (A1, A2; d14/15/17 kept) |
| d19 offset o, d20 o², d21 o=7, d22 o=10, d23 o=14 | offset features | offset embedding (kept) |
| d24 prev1 is `:`, d25 prev1 is `>`, d26 prev2 is `>`, d27 prev2 is `NL@` | previous-token copies | layer-0 heads 0/1 (d25, d27 kept; d24, d26 new) |
| d28 FAILED, d29 ok/ready, d114 error, d115 pending | status broadcast: the event's status word, visible at every token after it | layer-0 head 4 (new) |
| d31 | hunk-line count k/(k+1) with a BOS sink | layer-0 head 3 (kept) |
| d32–47 | code A (one-hot of id mod 4, 5, 7) of hashed/digit tokens; **block 3**: targets (read/list path, search word, plan action, build/lint target) carry 2× the code after layer-0 MLP | embedding (kept) + layer-0 MLP |
| d48–63 | code B (one-hot of id mod 3, 13) in the embedding, cancelled by layer-0 MLP; **block 3 B**: code B at targets only; cancelled again by layer-1 MLP, then layer-1 flags: d48 (0), d49 NOOP, d50 HTTPERR, d51 POLLPEND, d52 SAMEFAILED, d53 outcome, d54 kpen, d55 hascount, d56 is_key, d57 POLL2g, d58 muthunk | embedding, layer-0 MLP, layer-1 MLP |
| d64–79 | **block 1 A**: role-signed keys, +code A at hunk ids (h_old/h_new), −code A at error-line digits; zeroed by layer-1 MLP; then layer-2/3 features d64 g_CNT2, d65 g_hascount, d66 g_REV, d67 g_REP, d68 g_POLL2, d69 prevmut, d70 r_HTTPERR, d71 r_POLLPEND, d72 r_SAMEF, d73 r_CNTOK, d74 kpen_ev, d75 INEFF, d76 RES, d77 INPOL, d79 A1, d80 A2, d81 act, d82 resw, d83 osc, d84 sc, d85 act_b, d86 resw_b | layer-0 MLP, layer-1 MLP, layers 2–3 |
| d80–95 | **block 1 B**: +code B at hunk ids, −code B at line digits; zeroed by layer-1 MLP (d80–86 reused as above) | layer-0 MLP |
| d96–111 | **block 2 A**: +code A at the passed-count digit, −code A at poll resources | layer-0 MLP |
| d59 g_SAME, d60 g_LINE2, d61 g_SAMEF, d62 KVev, d63 g_CNTOK | event-level gathers | layer-2 attention |
| d113 count-repeat / poll-count value, d116 REVtokA, d117 REVtokB, d118 line-count value, d119 LINE2 value, d120 CNT2/POLL2 value, d26 REP-A value, d9 REP-B value | layer-1 attention outputs (signed match values, see §4) | layer-1 attention |
| d121 REVg, d122 REPg, d123 CNTOKg, d124 SAMEg, d125 LINE2g, d126 CNT2g, d127 KV | token-level flags | layer-1 MLP |
| d30, d112, d78, d87–95 after layer 1, d96–111 after layer 1 | unused or dead | — |

Dead r1 dims: d30 (nothing writes it now), d112 (status_true; its unit is zeroed), d40 (r1's FNCev; its
head is cleared).

## 2. Embeddings

- `tok_emb` column 0: 1 for every id (kept). Column 1: ids 2048–10239 and 141 (A7). Column 2: digit groups
  224–1333 (kept). Columns 3, 4, 5: ids 147 (`>`), 4 (`NL@`), 2 (`EOE`) (kept). Columns 6, 7, 8: ids {7, 9},
  {8}, {10, 11, 12} (kept). Column 9: id 143 `:` (A1). Column 10: ids 38 `ok`, 42 `ready` (A2). Column 11:
  ids 14, 15 (A3). Column 12: id 13 (A4). Column 13: id 17; column 16: id 16 (A5). Columns 14, 15, 17: ids
  39, 40, 43 (kept). Column 18: ids 18, 19, 20 (A6).
- Columns 32–47 (code A: mod 4 → d32–35, mod 5 → d36–40, mod 7 → d41–47) for hashed ids and digit groups
  (kept) plus id 141 (d33, d37, d42; A7). Columns 48–63 (code B: mod 3 → d48–50, mod 13 → d51–63) for hashed
  ids (kept) plus id 141 (d48, d62; A7).
- `off_emb`: d19 = o, d20 = o², d21/22/23 = [o = 7/10/14] (kept).

## 3. Layer 0

### Attention (rel_bias rows 0, 1, 2, 4: 0 at distance 0, −10000 elsewhere incl. BOS; row 3: BOS +20)

| head | q (rows) | k | v → o | role |
|---|---|---|---|---|
| 0 | rows 0–2 from d0, d19, d20: `q = (−64o² + 128o − 64, 128o − 128, −64)` (r1 ×2, B1) | rows 0–2 = (d0, d19, d20) (kept) | d3 → d25 (kept), d9 → d24 (B2) | exact copy of the previous token's `>` and `:` flags: score −16(o_i − 1 − o_j)² |
| 1 | rows 16–18 (r1 ×2) | rows 16–18 (kept) | d4 → d27 (kept), d3 → d26 (B3) | copy of the token two back: `NL@` and `>` flags |
| 2 | bias q[32] = 1 (kept) | k[32] = 80·d21 (offset 7 = the tool token; B4) | d6→d6, d7→d7, d8→d8, d11→d11, d12→d12, d13→d13, d16→d16, d18→d18 (B4) | tool-class broadcast, in place |
| 3 | bias q[48] = 1 | k[48] = 80·d4, BOS bias 20 | d4 → d31 | hunk-line count k/(k+1) (kept) |
| 4 | bias q[64] = 1 | k[64] = 80·(d14 + d15 + d17 + d10) | d14→d114, d17→d115, d15→d28, d10→d29 (B5) | status broadcast within the event |
| 5–7 | zero | zero | zero | unused |

### MLP (unit groups; each unit is `relu(Σ inputs − n)`)

| units | computes | output |
|---|---|---|
| 0–15, 16–31 | hunk-id keys: `relu(codeA_d / codeB_d + hashed + prev1'>' + prev2'NL@' − 2)` = the code at h_old (prev2 `NL@`) and h_new (prev1 `>`) tokens (kept) | +1 → d64+d / d80+d |
| 32–47 | read/list path key: `relu(codeA_d + hashed + o10 + bc_rl − 3)` (bias and class dim changed, C1) | +1 → d32+d (block 3) |
| 48–63 | search word key: `relu(codeA_d + hashed + o14 + bc_search − 3)` (C1) | +1 → d32+d |
| 64–79 | passed-count key: `relu(codeA_d + digit + prev2'>' − 2)` (weights kept; d26 re-assigned) | +1 → d96+d (block 2) |
| 80, 81 | r1's status_true and FNC: zeroed (C3) | — |
| 82–97, 98–113 | error-line number key: `relu(codeA_d / codeB_d + digit + prev1':' + (bc_err + bc_pend + bc_FAILED) − 3)` (C4) | −1 → d64+d / d80+d |
| 114–129 | poll resource key: `relu(codeA_d + hashed + o10 + bc_poll − 3)` (C5) | −1 → d96+d |
| 130–145 | plan action key: `relu(codeA_d + hashed + o10 + bc_plan − 3)` (C6) | +1 → d32+d |
| 146–161 | build/lint target key: `relu(codeA_d + hashed + o10 + bc_buildlint − 3)` (C7) | +1 → d32+d |
| 162–177 | cancel raw code B: `relu(codeB_d)` (C8) | −1 → d48+d |
| 178–193, 194–209, 210–225, 226–241 | block-3 B keys: code B at read/list paths, search words, plan actions, build/lint targets (same gates as 32–47/48–63/130–145/146–161) (C9) | +1 → d48+d |

After this MLP: d32–47 = 2·codeA at targets, 1·codeA at other coded tokens; d48–63 = codeB at targets, 0
elsewhere; block 1 = ±code at hunk ids / line digits; block 2 = ±code at counts / poll resources.

## 4. Layer 1 attention: code matching (all heads: distance 0 → −10000)

Scores are q·k/4. A one-hot code with 3 ones (A) gives 32·3/4 = 24 for a full match, 16 for 2 of 3; a code
B (2 ones) gives 16 / 8. Because keys are role-signed, a query from one role scores negatively against the
other role's keys of the same block and never attends to them.

| head | q from | k from | range (bias 0) | BOS | v → o | output meaning |
|---|---|---|---|---|---|---|
| 0 | 32·d32–47 | d64–79 | 1..7 | 20 | ⅓Σd64–79 → d116 | REVtokA: a hunk id whose code A matched a hunk id of the last 7 events (kept) |
| 1 | 32·d80–95 (D1) | d80–95 | 1..7 | 12 | ½Σd80–95 → d117 | REVtokB (kept except the query source) |
| 2 | 32·d64–79 | d64–79 | 1..15 | 24 | ⅓Σd64–79 → d118 | at a line digit: −k/(k+1), k = earlier identical line numbers (D2) |
| 3 | 32·d64–79 | d64–79 | 1..2 | 20 | ⅓Σd64–79 → d119 | at a line digit: −0.98 if the same line occurred in the previous 2 events (D3) |
| 4 | 32·d96–111 | d96–111 | 1..2 | 20 | ⅓Σd96–111 → d120 | +0.98 at a count digit / −0.98 at a poll resource repeated within 2 events (D4) |
| 5 | 32·d96–111 | d96–111 | 1..31 | 24 | ⅓Σd96–111 → d113 | +k/(k+1) at a count digit (earlier equal counts), −k/(k+1) at a poll resource (earlier polls) (D5) |
| 6 | 32·d32–47 | d32–47 | 1..31 | 72 | ⅓Σd32–47 → d26 | REP-A: ≈2 at a target whose code A matched an earlier target (target–target 96 > BOS 72 > target–plain 48) (D6) |
| 7 | 32·d48–63 | d48–63 | 1..31 | 12 | ½Σd48–63 → d9 | REP-B: ≈1 at a target whose code B matched an earlier target (D7) |

The BOS column equal to the full-match score (heads 2 and 5) makes the attention split equally between
the BOS and the k matching keys, so the value is k/(k+1): 0.5, 0.667, 0.75, 0.8, 0.833, 0.857 for k = 1..6.

## 5. Layer 1 MLP: token-level flags

Notation: `ΣB1 = Σd64–79` (= +3 at a hunk id, −3 at a line digit), `ΣB2 = Σd96–111` (+3 at a count digit,
−3 at a poll resource), `ΣB3 = Σd32–47` (6 at a target, 3 at other coded tokens).

| units | flag | formula | output |
|---|---|---|---|
| 0 | REVg | `relu(d116 + d117 + hashed + prev1'>' − 3)`: an h_new that returns a file to an earlier state (kept) | d121 |
| 1, 2 | REPg | `clip(2·d26 + 4·d9 + ⅔·ΣB3 − 9)`: a target repeated (both codes) within 31 events | d122 |
| 3, 4 | CNTOKg | `clip(8·d113 + (10/3)·ΣB2 + 10·bc_ok + 10·digit − 31.5)`: an ok validation whose passed count was seen before | d123 |
| 5, 6 | SAMEg | `clip(−8·d118 − (10/3)·ΣB1 − 11.5)`: a failure whose line number was seen within 15 events | d124 |
| 7, 8 | LINE2g | `clip(−4·d119 − (10/3)·ΣB1 − 11.5)`: the same failure within the previous 2 events | d125 |
| 9, 10 | POLL2g | `clip(−4·d120 − (10/3)·ΣB2 − 11.5)`: the same poll resource within the previous 2 events | d57 |
| 11, 12 | CNT2g | `clip(4·d120 + (10/3)·ΣB2 + 10·digit − 21.5)`: the same passed count within the previous 2 events | d126 |
| 13, 14 | KV | `relu(−d118 − ΣB1/3 − 1)` + `relu(−d113 − ΣB2/3 − 1)` = k/(k+1) at a line digit / poll resource | d127 |
| 15 | NOOP | `relu(bc_mut − 2·d31)`: a mutation without a hunk line (r1's unit, moved) | d49 |
| 16, 17 | HTTPERR, POLLPEND | `relu(bc_err + bc_http + EOE − 2)`, `relu(bc_pend + bc_poll + EOE − 2)` (EOE only) | d50, d51 |
| 18, 19 | SAMEFAILED | `clip(−8·d118 − (10/3)·ΣB1 + 10·bc_FAILED + 10·bc_tests − 31.5)`: a repeated run_tests failure | d52 |
| 20, 21 | outcome | `relu(bc_err + bc_pend + bc_FAILED)` + `relu(bc_ok + bc_http + bc_poll + bc_buildlint + bc_tests − 1)` | d53 |
| 22, 23 | kpen | `relu(−50·d118 − 38.5)` (= 1.5 at k = 4, 0 at k ≤ 3) + `relu(−100·d113 − 84.5)` (= 1.2 at k = 6 polls, 0 at k ≤ 5) | d54 |
| 24 | hascount | `relu(ΣB2/3)` = 1 at the passed-count digit | d55 |
| 25, 26 | is_key | `relu(−ΣB1/3) + relu(−ΣB2/3)` = 1 at a line digit or a poll resource | d56 |
| 27, 28 | muthunk | `clip(2·bc_mut + 8·d31 − 5)`: a mutation that changed at least one file | d58 |
| 29–44 | cancel block-3 B: `relu(d48+d)` → −1 into d48+d | d48–63 |
| 45–90 | zero d64–86: `relu(d)` → −1 and `relu(−d)` → +1 for each dim | d64–86 |

## 6. Layer 2 attention: gather to the event (q.bias = 1 on all heads)

| head | key | range | values → outputs | meaning |
|---|---|---|---|---|
| 0 | 80·d56 (is_key) | same event | d124→d59 g_SAME, d125→d60 g_LINE2, d52→d61 g_SAMEF, d127→d62 KVev, d57→d68 g_POLL2, d54→d74 kpen_ev | the failure identity token's flags, at the EOE |
| 1 | 80·d55 (hascount) | same event | d123→d63 g_CNTOK, d126→d64 g_CNT2, d55→d65 g_hascount | the passed-count digit's flags |
| 2 | 80·(d121 + d122) | same event | d121→d66 g_REV, d122→d67 g_REP | reverting hunk ids / repeated target |
| 3 | 80·d5 (EOE) | distance 1 only | d8→d69 prevmut | the previous event was a mutation |
| 4, 5, 6, 7 | 80·d50 / 80·d51 / 80·d52 / 80·d123 | distances 1..3, BOS 0 | d50→d70 r_HTTPERR, d51→d71 r_POLLPEND, d52→d72 r_SAMEF, d123→d73 r_CNTOK | a flagged token exists within the previous 3 events (each output 0 or 1) |

With a flagged token present the softmax puts ≈1 on it (score 20 against 0); with none, every value read is
0, so each output is binary.

## 7. Layer 2 MLP: per-event features (valid at the EOE)

| units | feature | formula | output |
|---|---|---|---|
| 0, 1 | INEFF part 1 | `clip(g_REP + g_CNTOK + g_SAME + NOOP)` | d75 |
| 2, 3 | INEFF part 2 (EDR) | `clip(6·muthunk + 3·(r_SAMEF + r_CNTOK) + 3·g_REV − 6)`: a file change within 3 events after a repeated test result, or a reverting change | d75 |
| 4, 5 | INEFF part 3 (POLLOUT) | `clip(10·bc_pend + kpen_ev − 10)`: a pending poll beyond the sixth | d75 |
| 6–11 | RES | `clip(2·bc_ok + bc_http + r_HTTPERR − 3) + clip(2·bc_ok + bc_poll + r_POLLPEND − 3) + clip(2·bc_ok + g_hascount + r_SAMEF − 3)`: an ok/ready that resolves a recent failure of its own kind | d76 |
| 12 | INPOL | `relu(fail + g_LINE2 + g_POLL2 + g_CNT2 + 1 − g_hascount − prevmut − kpen_ev − 1)` with `fail = bc_err + bc_pend + bc_FAILED`: a retry (same failure within 2 events, same count when there is one, no mutation in between) still within policy | d77 |

INEFF ∈ {0, 1} per event (the three parts are mutually exclusive: non-mutation flags / file-changing
mutation / pending poll).

## 8. Layer 3 attention: window features at the EOE (q.bias = 1)

| head | key | bias | value → output | meaning |
|---|---|---|---|---|
| 0 | 80·d5 | 0 at distances 0..7, BOS +20 (kept) | d66→d79 A1 = Σg_REV/9, d75→d80 A2 = ΣINEFF/9 | window sums over the last 8 events |
| 1 | 240·d5 + 240·d53 − 240 (EOE AND outcome: 60; else ≤ 0) | −6·distance for 0..7, −10000 beyond | d77→d81 act | INPOL of the most recent outcome event in the window |
| 2 | 80·d5 + 80·d76 − 80 (EOE AND RES: 20) | 0 at distances 0..3 | d76→d82 resw | a resolution among the last 4 events |

## 9. Layer 3 MLP and head

| units | ramp | output |
|---|---|---|
| 0, 1 | osc = `clip(25·A1 − 3.75)` (2 reversions of 8) (kept, moved to d79→d83) | d83 |
| 2, 3 | sc = `clip(30·A2 − 14.7)` (5 ineffective of 8) (kept, moved to d80→d84) | d84 |
| 4, 5 | act_b = `clip(4·act − 1)` | d85 |
| 6, 7 | resw_b = `clip(4·resw − 1)` | d86 |

Head: `PRODUCTIVE = 0.5` (bias); `RECOVERY = 8·d85 + 5·d86 − 2`; `STALLED = 6·d84 + 6·d75 − 8·d85 − 8`;
`OSCILLATING = 12·d83`. Which circuit drives each logit:

- OSCILLATING: hunk-id keys (L0 MLP 0–31) → heads L1-0/1 → REVg (L1 MLP 0) → g_REV (L2 head 2) → A1 (L3 head 0) → osc.
- RECOVERY (active): line/count/poll keys → heads L1-2/3/4/5 → LINE2g, CNT2g, POLL2g, KV, kpen → gathers
  (L2 heads 0/1/3) → INPOL (L2 MLP 12) → act (L3 head 1) → act_b. RECOVERY (resolved): HTTPERR, POLLPEND,
  SAMEFAILED → recent heads (L2 4/5/6) → RES (L2 MLP 6–11) → resw (L3 head 2) → resw_b.
- STALLED: REPg, CNTOKg, SAMEg, NOOP, EDR, POLLOUT → INEFF (L2 MLP 0–5) → A2 (L3 head 0) → sc, plus INEFF at
  the current EOE; suppressed by act_b.
- PRODUCTIVE: the constant 0.5 wins when nothing else fires.

## 10. What was kept from r1 and what changed

Kept: the token flags d0–d8, d14, d15, d17, the code A/B embeddings, the offset features, layer-0 heads 0, 1
(doubled queries), 3, the hunk-id key units (L0 MLP 0–31), the passed-count key units (64–79), the reversion
heads (L1 0, 1) and REVg, the window-sum head (L3 head 0) and the osc/sc ramps.

Changed: the target-code units' gate bias (the defect: −2 → −3, which made every code dimension fire at
read paths and search words); the layer-0 broadcast (keyed on offset 7, eight classes, in place); the status
broadcast (moved to layer 0, keyed on the status flags); the previous-token copies (`:` and prev2 `>`); all
of layer 1 heads 2–7, the layer-1 MLP, layers 2 and 3 and the head, which replace r1's T-match / fail /
rf / sh logic with the retry, resolution and ineffectiveness circuits above. Dropped: r1's FNC and
status_true units, the has_lines head, the lines-field exception, the "fail within 5 / within 16" features.

## 11. Account of every non-zero tensor block

- `tok_emb.weight`: columns listed in §2 (columns 0–18 flags, 32–47 code A, 48–63 code B).
- `off_emb.weight`: columns 19, 20 (rows 1–191), 21 (row 7), 22 (row 10), 23 (row 14).
- `layers.0.attn.rel_bias`: rows 0, 1, 2, 4: −10000 at columns 1–64; row 3: −10000 at 1–63, +20 at 64.
- `layers.0.attn.q.weight`: rows 0–2 and 16–18 (heads 0/1 quadratic queries, ×2). `q.bias`: [32], [48], [64] = 1.
- `layers.0.attn.k.weight`: rows 0–2, 16–18 (d0, d19, d20); row 32: d21 = 80; row 48: d4 = 80; row 64: d10, d14, d15, d17 = 80.
- `layers.0.attn.v.weight`: rows 0 (d3), 2 (d9), 16 (d4), 17 (d3), 32–39 (class flags), 48 (d4), 64–67 (d14, d17, d15, d10).
- `layers.0.attn.o.weight`: rows 25←0, 24←2, 27←16, 26←17, 6/7/8/11/12/13/16/18←32–39, 31←48, 114←64, 115←65, 28←66, 29←67.
- `layers.0.mlp.fc1.weight` / `fc1.bias` / `fc2.weight`: unit groups of §3 (units 0–79, 82–241; biases −2 for 0–31 and 64–79, −3 for 32–63, 82–161, 178–241, 0 for 162–177; fc2 rows 32–47, 48–63, 64–79, 80–95, 96–111 as listed in §3).
- `layers.1.attn.rel_bias`: rows 0–7 as in §4 (row 0: 0→−10000, 8–63→−10000, 64→20; row 1: same with 64→12; row 2: 0 and 16–63→−10000, 64→24; rows 3, 4: 0 and 3–63→−10000, 64→20; row 5: 0 and 32–63→−10000, 64→24; row 6: same with 64→72; row 7: same with 64→12).
- `layers.1.attn.q.weight`: rows 0–15 and 96–111 ← d32–47 (32); rows 16–31 ← d80–95 (32); rows 32–47 and 48–63 ← d64–79 (32); rows 64–79 and 80–95 ← d96–111 (32); rows 112–127 ← d48–63 (32).
- `layers.1.attn.k.weight`: rows 0–15, 32–47, 48–63 ← d64–79 (1); rows 16–31 ← d80–95; rows 64–79, 80–95 ← d96–111; rows 96–111 ← d32–47; rows 112–127 ← d48–63.
- `layers.1.attn.v.weight`: rows 0, 32, 48 ← d64–79 (⅓); row 16 ← d80–95 (½); rows 64, 80 ← d96–111 (⅓); row 96 ← d32–47 (⅓); row 112 ← d48–63 (½).
- `layers.1.attn.o.weight`: 116←0, 117←16, 118←32, 119←48, 120←64, 113←80, 26←96, 9←112.
- `layers.1.mlp.fc1.weight` / `fc1.bias` / `fc2.weight`: units 0–90 of §5.
- `layers.2.attn.rel_bias`: rows 0–2: −10000 at columns 1–64; row 3: −10000 at 0 and 2–64; rows 4–7: −10000 at 0 and 4–63, 0 at 64. `q.bias`: [0], [16], …, [112] = 1.
- `layers.2.attn.k.weight`: row 0 ← d56; 16 ← d55; 32 ← d121, d122; 48 ← d5; 64 ← d50; 80 ← d51; 96 ← d52; 112 ← d123 (all 80).
- `layers.2.attn.v.weight` / `o.weight`: the value→output pairs of §6 (v rows 0–5, 16–18, 32–33, 48, 64, 80, 96, 112; o rows 59–74).
- `layers.2.mlp.fc1.weight` / `fc1.bias` / `fc2.weight`: units 0–12 of §7 (fc2 rows 75, 76, 77).
- `layers.3.attn.rel_bias`: row 0: −10000 at 8–63, 20 at 64 (kept); row 1: 0, −6, …, −42 at 0–7, −10000 at 8–64; row 2: 0 at 0–3, −10000 at 4–64. `q.bias`: [0], [16], [32] = 1 (kept).
- `layers.3.attn.k.weight`: row 0 ← d5 (80); row 16 ← d5, d53 (240); row 32 ← d5, d76 (80). `k.bias`: [16] = −240, [32] = −80.
- `layers.3.attn.v.weight`: 0←d66, 1←d75, 16←d77, 32←d76. `o.weight`: 79←0, 80←1, 81←16, 82←32.
- `layers.3.mlp.fc1.weight` / `fc1.bias` / `fc2.weight`: units 0–7 of §9 (fc2 rows 83–86).
- `head.weight`: row 1: d85 = 8, d86 = 5; row 2: d84 = 6, d75 = 6, d85 = −8; row 3: d83 = 12. `head.bias`: [0.5, −2, −8, 0].

Everything not listed is zero (in particular layer-0 heads 5–7, layer-1 `q.bias`, all `k.bias` except
layer 3's, all `v.bias`, `o.bias`, `fc2.bias`, and MLP units beyond those named).
