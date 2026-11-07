# CIRCUITS.md — the revised network (Phase B's weights + `REVISION.edits`)

This describes the network the harness builds from `phase_b/artifact.dlw` with `REVISION.edits` applied on
top. Section letters (A–J) refer to the `# ====` headings of `REVISION.edits`. "Phase B" means a block Phase B
wrote (on top of r1's design); "kept" means the block is Phase B's and unchanged; "changed" / "new" means this
revision wrote it. `dN` is residual dimension N. `clip(x)` means `relu(x) − relu(x − 1)` (two MLP units whose
biases differ by 1 and whose output weights are +1 / −1), which maps any pre-activation to [0, 1].

The decision the network implements (the oracle's rule order, with observable proxies):

```
OSCILLATING  if  at least 2 reverting mutations (edit/write/revert) among the last 7 events
else RECOVERY if  the most recent outcome event in the last 8 is an in-policy retry (act)
else STALLED  if  at least 5 of the last 8 events are ineffective AND the current event is ineffective
                  AND no fresh read_file among the last 8 events
else RECOVERY if  a resolution happened in the last 4 events (resw)
else PRODUCTIVE
```

Logits: `PRODUCTIVE = 0.5`, `RECOVERY = 8·act_b + 5·resw_b − 2`,
`STALLED = 6·sc + 6·INEFF_t − 8·act_b − 12·fresh_b − 8`, `OSCILLATING = 12·osc`.

What changed against Phase B, in one paragraph: a failure is now "the same failure" only when both its error
line and its passed count were seen before (Phase B used the line alone); the retry penalty needs both the
line tally (≥ 4) and the count tally (≥ 5); a retry is not one when the previous event was a read_file; repeated
plan actions are keyed on the word after the verb (`record:path`, `handoff:N:k`) when there is one; an append
(`h0 >` hunk) to a path mutated before is ineffective; reversions count only for edit/write/revert events and
only over the last 7 events of the window; and a fresh read_file within the window suppresses STALLED. Four
layer-2 "recent flag" heads were merged into one to free three heads for the new gathers.

## 1. Residual-subspace allocation (128 dims)

| dims | content | written by |
|---|---|---|
| d0 | constant 1 (every token) | embedding (kept) |
| d1 | hashed word (ids 2048+) and `.` (id 141) | embedding (kept) |
| d2 | digit group | embedding (kept) |
| d3 `>`, d4 `NL@`, d5 `EOE` | token flags | embedding (kept) |
| d6 read_file/list_dir, d7 search, d8 edit/write/revert, d11 build/lint, d12 run_tests, d13 poll_status, d16 http_call, d18 plan/ask/yield | tool-class flags on the tool token; layer-0 head 2 adds the same flag to every later token of the event (1 there, 2 on the tool token) | embedding + layer-0 attention (kept) |
| d127 | **new**: read_file flag, broadcast like the class flags (1 on the event's tokens, 2 on the tool token) | embedding A2 + layer-0 head 2 (B1) |
| d9 `:`, d10 ok/ready, d14 error, d15 FAILED, d17 pending | status / punctuation token flags | embedding (kept) |
| d19 offset o, d20 o², d21 o=7, d22 o=10, d23 o=14 | offset features | offset embedding (kept) |
| d24 prev1 is `:`, d25 prev1 is `>`, d26 prev2 is `>`, d27 prev2 is `NL@` | previous-token copies | layer-0 heads 0/1 (kept) |
| d28 FAILED, d29 ok/ready, d114 error, d115 pending | status broadcast: the event's status word, visible at every token after it | layer-0 head 4 (kept) |
| d30 | **new**: the hunk id `h0` (an empty removed-lines side) | embedding A1 |
| d31 | hunk-line count k/(k+1) with a BOS sink | layer-0 head 3 (kept) |
| d32–47 | **block 3 A**: code A (one-hot of id mod 4, 5, 7) of hashed/digit tokens; after layer-0 MLP: 2× at targets (read/list path, search word, plan action word at offset 10, build/lint target, **new** plan word at offset 12), **new** −2× at the path of an edit/write/revert (role-signed) | embedding (kept) + layer-0 MLP (kept, C1, C3) |
| d48–63 | **block 3 B**: code B (id mod 3, 13) in the embedding, cancelled by layer-0 MLP; +1 at targets (and **new** plan offset-12 words), **new** −1 at mutation paths; cancelled again by layer-1 MLP; then layer-1 flags d48 (0), d49 NOOP, d50 HTTPERR, d51 POLLPEND, d52 FAILREP (changed meaning), d53 outcome, d54 kpen (now 0/1), d55 hascount, d56 is_key, d57 POLL2g, d58 muthunk | embedding, layer-0 MLP, layer-1 MLP |
| d59–74 | event-level gathers at every token of the event (valid at the EOE): d59 g_SAME, d60 g_LINE2, **d61 g_CNT15 (new)**, **d62 g_kpenC (new)**, d63 g_CNTOK, d64 g_CNT2, d65 g_hascount, d66 g_REV, d67 g_REP, d68 g_POLL2, d69 prevX (changed: previous event was a mutation or a read_file), d70 r_HTTPERR, d71 r_POLLPEND, d72 r_FAILREP (changed), d73 r_CNTOK, d74 g_kpenL | layer-2 attention |
| d64–79, d80–86 at layers 0–1 | block 1 A (+code A at hunk ids, −code A at error-line digits) and block 1 B (+code B / −code B); zeroed by layer-1 MLP | layer-0 MLP (kept), layer-1 MLP (kept) |
| d75 INEFF, d76 RES, d77 INPOL, **d94 FRESH (new)** | per-event features | layer-2 MLP |
| **d78 g_H0 (new)**, **d92 g_PLANREP (new)**, **d93 g_REPMUT (new)** | event-level gathers | layer-2 heads 5–7 |
| d79 A1, d80 A2, **d95 A3 (new)**, d81 act, d82 resw | window features | layer-3 attention |
| d83 osc, d84 sc, d85 act_b, d86 resw_b, **d112 fresh_b (new, at the EOE)** | ramps | layer-3 MLP |
| d87–95 | block 1 B residue at layers 0–1 (code B dims 7–15); **new**: zeroed by layer-1 MLP, then token flags d87 CNT15g, d88 kpenC, d89 REPg_plan, d90 plankey (1 at offset 10, 2 at offset 12), d91 REPMUT | layer-1 MLP (E7, E8, E5, E12, E6) |
| d96–111 | **block 2**: +code A at the passed-count digit, −code A at poll resources (any status) | layer-0 MLP (kept) |
| d112 at layer 0 | **new**: H0 = an `h0` token two after `NL@` in a mutation event | layer-0 MLP C5 |
| d113 | count / poll match value ±k/(k+1) | layer-1 head 5 (kept) |
| d116 REVtokA, d117 REVtokB, d118 line count, d119 LINE2 value, d120 CNT2/POLL2 value, d26 REP-A value, d9 REP-B value | layer-1 attention outputs | layer-1 attention (kept) |
| d121 REVg, d122 REPg_np (changed), d123 CNTOKg, d124 SAMEg, d125 LINE2g, d126 CNT2g | token-level flags | layer-1 MLP |
| d40 (r1's FNCev), d127 before this revision (KV) | dead / re-used | — |

## 2. Embeddings

- `tok_emb` column 0: 1 for every id. Column 1: ids 2048–10239 and 141. Column 2: digit groups 224–1333.
  Columns 3, 4, 5: ids 147, 4, 2. Columns 6, 7, 8: ids {7, 9}, {8}, {10, 11, 12}. Column 9: id 143. Column 10:
  ids 38, 42. Column 11: ids 14, 15. Column 12: id 13. Column 13: id 17. Columns 14, 15, 17: ids 39, 40, 43.
  Column 16: id 16. Column 18: ids 18, 19, 20 (all kept).
- **New** column 30: id 5861 (`h0`, A1). **New** column 127: id 7 (`read_file`, A2).
- Columns 32–47 (code A: mod 4 → d32–35, mod 5 → d36–40, mod 7 → d41–47) for hashed ids and digit groups plus
  id 141; columns 48–63 (code B: mod 3 → d48–50, mod 13 → d51–63) for hashed ids plus id 141 (kept).
- `off_emb`: d19 = o, d20 = o², d21/22/23 = [o = 7/10/14] (kept).

## 3. Layer 0

### Attention (rel_bias rows 0, 1, 2, 4: 0 at distance 0, −10000 elsewhere incl. BOS; row 3: BOS +20; rows 5–7 zero)

| head | q (rows) | k | v → o | role |
|---|---|---|---|---|
| 0 | rows 0–2 from d0, d19, d20 (quadratic) | rows 0–2 = (d0, d19, d20) | d3 → d25, d9 → d24 | exact copy of the previous token's `>` and `:` flags (kept) |
| 1 | rows 16–18 | rows 16–18 | d4 → d27, d3 → d26 | copy of the token two back: `NL@` and `>` flags (kept) |
| 2 | bias q[32] = 1 | k[32] = 80·d21 (offset 7, the tool token) | d6→d6, d7→d7, d8→d8, d11→d11, d12→d12, d13→d13, d16→d16, d18→d18 (kept), **new** d127→d127 (value row 40, B1) | tool-class broadcast in place, now including the read_file flag |
| 3 | bias q[48] = 1 | k[48] = 80·d4, BOS +20 | d4 → d31 | hunk-line count k/(k+1) (kept) |
| 4 | bias q[64] = 1 | k[64] = 80·(d14 + d15 + d17 + d10) | d14→d114, d17→d115, d15→d28, d10→d29 | status broadcast within the event (kept) |
| 5–7 | zero | zero | zero | unused |

### MLP (each unit is `relu(Σ inputs − n)`; units 0–241 kept, 242–306 new)

| units | computes | output |
|---|---|---|
| 0–15, 16–31 | hunk-id keys: `relu(codeA_d / codeB_d + hashed + prev1'>' + prev2'NL@' − 2)` at h_old and h_new | +1 → d64+d / d80+d |
| 32–47, 48–63 | read/list path key (offset 10, class d6) and search word key (offset 14, class d7): `relu(codeA_d + hashed + offset + class − 3)` | +1 → d32+d (block 3) |
| 64–79 | passed-count key: `relu(codeA_d + digit + prev2'>' − 2)` | +1 → d96+d (block 2) |
| 80, 81 | zero (r1's status_true and FNC, dropped by Phase B) | — |
| 82–97, 98–113 | error-line number key: `relu(codeA_d / codeB_d + digit + prev1':' + (bc_err + bc_pend + bc_FAILED) − 3)` | −1 → d64+d / d80+d |
| 114–129 | poll resource key: `relu(codeA_d + hashed + o10 + bc_poll − 3)` | −1 → d96+d |
| 130–145, 146–161 | plan action word (offset 10) and build/lint target keys | +1 → d32+d |
| 162–177 | cancel raw code B: `relu(codeB_d)` | −1 → d48+d |
| 178–241 | block-3 B keys: code B at read/list paths, search words, plan words (offset 10), build/lint targets | +1 → d48+d |
| **242–257 (C1)** | mutation path key, code A: `relu(codeA_d + hashed + o10 + bc_mut − 3)` at the path of an edit/write/revert | **−3** → d32+d (net −2× code A) |
| **258–273 (C2)** | the same, code B | **−1** → d48+d (net −1× code B) |
| **274–289 (C3)** | plan word at offset 12 (the word after `verb:`), code A: `relu(codeA_d + hashed + prev1':' + bc_plan + (24·o − o² − 143) − 4)`; the offset term is 1 at o = 12 and ≤ 0 elsewhere | +1 → d32+d |
| **290–305 (C4)** | the same, code B | +1 → d48+d |
| **306 (C5)** | H0 = `relu(h0 + prev2'NL@' + bc_mut − 2)`: the h_old position holds `h0` in a mutation event (an append) | +1 → d112 |

After this MLP: d32–47 = 2·codeA at targets and plan offset-12 words, −2·codeA at mutation paths, 1·codeA at
other coded tokens; d48–63 = codeB at targets (+1) / mutation paths (−1), 0 elsewhere; block 1 = ±code at hunk
ids / line digits; block 2 = ±code at counts / poll resources.

## 4. Layer 1 attention: code matching (all heads kept; distance 0 → −10000)

Scores are q·k/4. A one-hot code A (3 ones) gives 24 for a full match at scale 32, 16 for 2 of 3; targets carry
2× code, so target–target is 96, target–plain 48, and the BOS column 72 sits between them. Mutation paths carry
−2× code A / −1× code B: a mutation query against a mutation key scores +96 (code A) / +16 (code B), against a
target or plain key it is negative; a target query against a mutation key is negative. So reads/searches/plans
match earlier targets only, and mutations match earlier mutations of the same path only.

| head | q from | k from | range (bias 0) | BOS | v → o | output meaning |
|---|---|---|---|---|---|---|
| 0 | 32·d32–47 | d64–79 | 1..7 | 20 | ⅓Σd64–79 → d116 | REVtokA: a hunk id whose code A matched a hunk id of the last 7 events |
| 1 | 32·d80–95 | d80–95 | 1..7 | 12 | ½Σd80–95 → d117 | REVtokB |
| 2 | 32·d64–79 | d64–79 | 1..15 | 24 | ⅓Σd64–79 → d118 | at a line digit: −k/(k+1), k = earlier identical line numbers |
| 3 | 32·d64–79 | d64–79 | 1..2 | 20 | ⅓Σd64–79 → d119 | at a line digit: −0.98 if the same line occurred in the previous 2 events |
| 4 | 32·d96–111 | d96–111 | 1..2 | 20 | ⅓Σd96–111 → d120 | +0.98 at a count digit / −0.98 at a poll resource repeated within 2 events |
| 5 | 32·d96–111 | d96–111 | 1..31 | 24 | ⅓Σd96–111 → d113 | +k/(k+1) at a count digit (earlier equal passed counts, ok or FAILED), −k/(k+1) at a poll resource |
| 6 | 32·d32–47 | d32–47 | 1..31 | 72 | ⅓Σd32–47 → d26 | REP-A: ≈2 at a target whose code A matched an earlier target; **≈−2 at a mutation path whose code A matched an earlier mutation path** |
| 7 | 32·d48–63 | d48–63 | 1..31 | 12 | ½Σd48–63 → d9 | REP-B: ≈1 at a matched target, **≈−1 at a matched mutation path** |

## 5. Layer 1 MLP: token-level flags

Notation: `ΣB1 = Σd64–79` (+3 at a hunk id, −3 at a line digit), `ΣB2 = Σd96–111` (+3 at a count digit, −3 at a
poll resource), `ΣB3 = Σd32–47` (6 at a target, −6 at a mutation path, 3 at other coded tokens).

| units | flag | formula | output |
|---|---|---|---|
| 0 (E1) | REVg | `relu(d116 + d117 + hashed + prev1'>' + bc_mut − 4)`: an h_new that returns a file to an earlier state, **in an edit/write/revert event** | d121 |
| 1, 2 (E2) | REPg_np | `clip(2·d26 + 4·d9 + ⅔·ΣB3 − 9 − 10·bc_plan − 10·bc_mut)`: a repeated read/list/search/build/lint target (plan and mutation tokens excluded) | d122 |
| 3, 4 | CNTOKg | `clip(8·d113 + (10/3)·ΣB2 + 10·bc_ok + 10·digit − 31.5)`: an ok validation whose passed count was seen before | d123 |
| 5, 6 | SAMEg | `clip(−8·d118 − (10/3)·ΣB1 − 11.5)`: a failure whose line number was seen within 15 events | d124 |
| 7, 8 | LINE2g | `clip(−4·d119 − (10/3)·ΣB1 − 11.5)`: the same line within the previous 2 events | d125 |
| 9, 10 | POLL2g | `clip(−4·d120 − (10/3)·ΣB2 − 11.5)`: the same poll resource within the previous 2 events | d57 |
| 11, 12 | CNT2g | `clip(4·d120 + (10/3)·ΣB2 + 10·digit − 21.5)`: the same passed count within the previous 2 events | d126 |
| 13, 14 (E3) | zero (Phase B's KV, dropped) | — | — |
| 15 | NOOP | `relu(bc_mut − 2·d31)`: a mutation without a hunk line | d49 |
| 16, 17 | HTTPERR, POLLPEND | `relu(bc_err + bc_http + EOE − 2)`, `relu(bc_pend + bc_poll + EOE − 2)` (EOE only) | d50, d51 |
| 18, 19 (E4) | FAILREP | `clip(8·d113 + (10/3)·ΣB2 + 10·bc_FAILED + 10·digit − 31.5)`: a FAILED passed-count digit whose count was seen within 31 events | d52 |
| 20, 21 | outcome | `relu(bc_err + bc_pend + bc_FAILED)` + `relu(bc_ok + bc_http + bc_poll + bc_buildlint + bc_tests − 1)` | d53 |
| 22, 99 (E9) | kpenL | `clip(−50·d118 − 38.5)`: ≥ 4 earlier identical line numbers (unit 22 kept, unit 99 is its clip partner) | d54 |
| 23, 100 (E9) | poll penalty | `clip(−100·d113 − 84.5)`: ≥ 6 earlier pending polls of the resource | d54 |
| 24 | hascount | `relu(ΣB2/3)` = 1 at the passed-count digit | d55 |
| 25, 26 | is_key | `relu(−ΣB1/3) + relu(−ΣB2/3)` = 1 at a line digit or a poll resource | d56 |
| 27, 28 | muthunk | `clip(2·bc_mut + 8·d31 − 5)`: a mutation that changed at least one file | d58 |
| 29–44 | cancel block-3 B: `relu(d48+d)` → −1 | d48–63 |
| 45–90 | zero d64–86: `relu(d)` → −1 and `relu(−d)` → +1 for each dim | d64–86 |
| **91, 92 (E5)** | REPg_plan | `clip(2·d26 + 4·d9 + ⅔·ΣB3 − 19 + 10·bc_plan − 20·o7)`: a repeated plan key (both codes); `o7` kills the tool token, where the class flag is 2 | d89 |
| **93, 94 (E6)** | REPMUT | `clip(−2·d26 − 4·d9 − ⅔·ΣB3 − 19 + 10·bc_mut − 20·o7)`: a mutation of a path mutated within the last 31 events | d91 |
| **95, 96 (E7)** | CNT15g | `clip(8·d113 + (10/3)·ΣB2 + 10·digit − 21.5)`: a passed count seen within 31 events (any status) | d87 |
| **97, 98 (E8)** | kpenC | `clip(100·d113 + (10/3)·ΣB2 + 10·digit − 101.5)`: ≥ 5 earlier equal passed counts | d88 |
| **101–118 (E10)** | zero d87–95 (block-1 B residue) | d87–95 |
| **119–134 (E11)** | cancel negative block-3 B: `relu(−d48−d)` → +1 | d48–63 |
| **135, 136 (E12)** | plankey | `relu(hashed + o10 + bc_plan − 2)` → 1; `relu(hashed + digit + prev1':' + bc_plan + (24·o − o² − 143) − 3)` → 2 (a word or digit at offset 12 after `:`) | d90 |

## 6. Layer 2 attention: gather to the event (q.bias = 1 on all heads)

| head | key | range | values → outputs | meaning |
|---|---|---|---|---|
| 0 | 80·d56 (is_key) | same event | d124→d59 g_SAME, d125→d60 g_LINE2, d57→d68 g_POLL2, d54→d74 g_kpenL (**F1**: the SAMEFAILED and KV values were dropped) | the failure identity token's flags, at the EOE |
| 1 | 80·d55 (hascount) | same event | d123→d63 g_CNTOK, d126→d64 g_CNT2, d55→d65 g_hascount, **d87→d61 g_CNT15, d88→d62 g_kpenC (F2)** | the passed-count digit's flags |
| 2 | 80·(d121 + d122) | same event | d121→d66 g_REV, d122→d67 g_REP | reverting hunk ids / repeated target (plan and mutation keys excluded from d122) |
| 3 | 80·d5 (EOE) | distance 1 only | **d8 + d127 → d69 prevX (F3)** | the previous event was a mutation or a read_file |
| **4 (F4)** | 80·(d50 + d51 + d52 + d123) | distances 1..3, BOS 0 | d50→d70 r_HTTPERR, d51→d71 r_POLLPEND, d52→d72 r_FAILREP, d123→d73 r_CNTOK | one head for all four "recent" flags: with m flagged tokens in range (at most one per event, so m ≤ 3) each present flag reads 1/m |
| **5 (F6)** | 80·d112 (H0) | same event | d112→d78 g_H0 | the event has an append hunk |
| **6 (F7)** | 80·d90 (plankey: 1 at offset 10, 2 at offset 12) | same event | d89→d92 g_PLANREP | REPg_plan of the offset-12 key when there is one (score 40 beats 20), else of the offset-10 key |
| **7 (F8)** | 80·d91 (REPMUT) | same event | d91→d93 g_REPMUT | the mutation's path was mutated before |

With a flagged token present the softmax puts ≈1 on it (score 20 against 0); with none, every value read is
0 (BOS −10000 for heads 0–2, 5–7; BOS 0 for head 4).

## 7. Layer 2 MLP: per-event features (valid at the EOE)

| units | feature | formula | output |
|---|---|---|---|
| 0, 1 (G1) | INEFF part 1 | `clip(g_REP + g_CNTOK + NOOP + g_PLANREP)` | d75 |
| 2, 3 | INEFF part 2 (EDR) | `clip(6·muthunk + 3·(r_FAILREP + r_CNTOK) + 3·g_REV − 6)`: a file change within 3 events after a repeated test result, or a reverting change | d75 |
| 4, 5 | INEFF part 3 (POLLOUT) | `clip(10·bc_pend + g_kpenL − 10)`: a pending poll beyond the sixth | d75 |
| 6–11 (G4b) | RES | `3·clip(2·bc_ok + bc_http + r_HTTPERR − 3) + 3·clip(2·bc_ok + bc_poll + r_POLLPEND − 3) + 3·clip(2·bc_ok + g_hascount + r_FAILREP − 3)`: an ok/ready that resolves a recent failure of its own kind (the ×3 restores a value ≥ 1 now that a recent flag can read 1/3) | d76 |
| 12 (G5) | INPOL | `relu(10·g_LINE2 + 10·g_POLL2 + 10·g_CNT2 − 6·g_hascount − 4·g_kpenL − 4·g_kpenC + 10·fail − 10·prevX − 19)` with `fail = bc_err + bc_pend + bc_FAILED`: a retry (same failure within 2 events, same count when there is one, no mutation and no read_file in between) still within policy: 5 for a validation retry, 1 for an http or poll retry, 0 otherwise; it is 0 when both the line tally (≥ 4) and the count tally (≥ 5) penalise it, or, for events without a count, when the line/poll tally does | d77 |
| **13, 14 (G2)** | INEFF part 4 (SAME_lc) | `clip(g_SAME + g_CNT15 − g_hascount)`: a failure whose line was seen within 15 events and, when it has a passed count, whose count was seen within 31 events | d75 |
| **15, 16 (G3)** | INEFF part 5 (H0RED) | `clip(g_H0 + g_REPMUT − 1 − 3·(r_FAILREP + r_CNTOK + g_REV))`: an append to a path mutated before, when EDR does not already count the event | d75 |
| **17 (G4)** | FRESH | `relu(bc_readfile − g_REP)`: a read_file of a path not read in the last 31 events | d94 |

INEFF ∈ {0, 1} per event: the parts are mutually exclusive (part 1: reads/plans/ok validations/no-op
mutations; parts 2 and 5: file-changing mutations, 5 only when 2 is silent; part 3: polls; part 4: failures).

## 8. Layer 3 attention: window features at the EOE (q.bias = 1)

| head | key | bias | value → output | meaning |
|---|---|---|---|---|
| 0 | 80·d5 | 0 at distances 0..7, BOS +20 | d75→d80 A2 = ΣINEFF/9, **d94→d95 A3 = ΣFRESH/9 (H1)** | window sums over the last 8 events (the reversion sum moved to head 3) |
| 1 | 240·d5 + 240·d53 − 240 (EOE AND outcome) | −6·distance for 0..7, −10000 beyond | d77→d81 act | INPOL of the most recent outcome event in the window |
| 2 | 80·d5 + 80·d76 − 80 (EOE AND RES) | 0 at distances 0..3 | d76→d82 resw | a resolution among the last 4 events |
| **3 (H2)** | 80·d5 | 0 at distances 0..6, −10000 beyond, BOS +20 | d66→d79 A1 = Σg_REV/8 | reversions among the last 7 events (the first event of a window cannot revert a state held in it) |

## 9. Layer 3 MLP and head

| units | ramp | output |
|---|---|---|
| 0, 1 | osc = `clip(25·A1 − 3.75)` (2 reversions of 7: 2/8·25 − 3.75 = 2.5; one: −0.6) | d83 |
| 2, 3 | sc = `clip(30·A2 − 14.7)` (5 ineffective of 8) | d84 |
| 4, 5 | act_b = `clip(4·act − 1)` | d85 |
| 6, 7 | resw_b = `clip(4·resw − 1)` | d86 |
| **8, 9 (I1)** | fresh_b = `clip(18·A3 − 1)` (at least one fresh read among the last 8 events) | d112 |

Head: `PRODUCTIVE = 0.5` (bias); `RECOVERY = 8·d85 + 5·d86 − 2`; `STALLED = 6·d84 + 6·d75 − 8·d85 − 12·d112 − 8`
(**J1**: the −12·d112 term is new); `OSCILLATING = 12·d83`. Which circuit drives each logit:

- OSCILLATING: hunk-id keys (L0 MLP 0–31) → heads L1-0/1 → REVg (L1 MLP 0, gated on edit/write/revert) → g_REV
  (L2 head 2) → A1 (L3 head 3, last 7 events) → osc.
- RECOVERY (active): line/count/poll keys → heads L1-2/3/4/5 → LINE2g, CNT2g, POLL2g, kpenL, kpenC → gathers (L2
  heads 0/1/3) → INPOL (L2 MLP 12) → act (L3 head 1) → act_b. RECOVERY (resolved): HTTPERR, POLLPEND, FAILREP →
  the recent head (L2 head 4) → RES (L2 MLP 6–11) → resw (L3 head 2) → resw_b.
- STALLED: REPg_np, CNTOKg, NOOP, PLANREP, SAME_lc, EDR, H0RED, POLLOUT → INEFF (L2 MLP 0–5, 13–16) → A2 (L3 head
  0) → sc, plus INEFF at the current EOE; suppressed by act_b and by fresh_b (FRESH → A3 → fresh_b).
- PRODUCTIVE: the constant 0.5 wins when nothing else fires.

## 10. What was kept from Phase B (and r1) and what changed

Kept: every embedding column of Phase B (flags, codes A/B, offsets), layer-0 heads 0–4 (one value added to head
2), layer-0 MLP units 0–241, all eight layer-1 attention heads, layer-1 MLP units 3–12, 15–17, 20–28, 29–90 (REVg,
REPg, SAMEFAILED, kpen partially rewritten; KV dropped), layer-2 heads 0–3 (values adjusted), layer-2 MLP units
2–11 (RES output ×3), layer-3 heads 0–2 (A1 moved out of head 0), layer-3 MLP units 0–7, and the head's
structure. The oracle's thresholds (2 reversions, 5 ineffective, 3 retries, 5 polls, 4-event resolution horizon,
ranges 7 / 15 / 31) are unchanged.

Changed or new (sections of `REVISION.edits`): A (two embedding flags), B1 (read_file broadcast), C1–C5 (mutation
path keys, plan offset-12 keys, H0), E1–E12 (REVg gate, REPg split, FAILREP by count, CNT15g, kpenC, clipped
kpen, zeroing d87–95, negative-key cancel, plankey), F1–F8 (gather values, merged recent head, three new gathers),
G1–G5 (INEFF parts, RES scale, INPOL), H1–H2 (A3, the 7-event reversion head), I1 (fresh_b), J1 (the STALLED
logit's fresh term).

## 11. Account of every non-zero tensor block

- `tok_emb.weight`: columns 0–18 flags, 30 (`h0`), 32–47 code A, 48–63 code B, 127 (`read_file`), as in §2.
- `off_emb.weight`: columns 19, 20 (rows 1–191), 21 (row 7), 22 (row 10), 23 (row 14).
- `layers.0.attn.rel_bias`: rows 0, 1, 2, 4: −10000 at columns 1–64; row 3: −10000 at 1–63, +20 at 64.
- `layers.0.attn.q.weight`: rows 0–2 and 16–18 (quadratic queries of heads 0/1). `q.bias`: [32], [48], [64] = 1.
- `layers.0.attn.k.weight`: rows 0–2, 16–18 (d0, d19, d20); row 32: d21 = 80; row 48: d4 = 80; row 64: d10, d14, d15, d17 = 80.
- `layers.0.attn.v.weight`: rows 0 (d3), 2 (d9), 16 (d4), 17 (d3), 32–39 (class flags), 40 (d127), 48 (d4), 64–67 (d14, d17, d15, d10).
- `layers.0.attn.o.weight`: rows 25←0, 24←2, 27←16, 26←17, 6/7/8/11/12/13/16/18←32–39, 127←40, 31←48, 114←64, 115←65, 28←66, 29←67.
- `layers.0.mlp.fc1.weight` / `fc1.bias` / `fc2.weight`: unit groups of §3 (units 0–79, 82–306; biases −2 for 0–31, 64–79, 306; −3 for 32–63, 82–161, 178–273; 0 for 162–177; −147 for 274–305; fc2 rows 32–47 ← units 32–63 (+1), 130–161 (+1), 242–257 (−3), 274–289 (+1); rows 48–63 ← 162–177 (−1), 178–241 (+1), 258–273 (−1), 290–305 (+1); rows 64–79 ← 0–15 (+1), 82–97 (−1); rows 80–95 ← 16–31 (+1), 98–113 (−1); rows 96–111 ← 64–79 (+1), 114–129 (−1); row 112 ← 306 (+1)).
- `layers.1.attn.rel_bias`: rows 0–7 as in §4 (row 0: 0 and 8–63 → −10000, 64 → 20; row 1: same with 64 → 12; row 2: 0 and 16–63 → −10000, 64 → 24; rows 3, 4: 0 and 3–63 → −10000, 64 → 20; row 5: 0 and 32–63 → −10000, 64 → 24; row 6: same with 64 → 72; row 7: same with 64 → 12).
- `layers.1.attn.q.weight`: rows 0–15 and 96–111 ← d32–47 (32); rows 16–31 ← d80–95 (32); rows 32–47 and 48–63 ← d64–79 (32); rows 64–79 and 80–95 ← d96–111 (32); rows 112–127 ← d48–63 (32).
- `layers.1.attn.k.weight`: rows 0–15, 32–47, 48–63 ← d64–79 (1); rows 16–31 ← d80–95; rows 64–79, 80–95 ← d96–111; rows 96–111 ← d32–47; rows 112–127 ← d48–63.
- `layers.1.attn.v.weight`: rows 0, 32, 48 ← d64–79 (⅓); row 16 ← d80–95 (½); rows 64, 80 ← d96–111 (⅓); row 96 ← d32–47 (⅓); row 112 ← d48–63 (½).
- `layers.1.attn.o.weight`: 116←0, 117←16, 118←32, 119←48, 120←64, 113←80, 26←96, 9←112.
- `layers.1.mlp.fc1.weight` / `fc1.bias` / `fc2.weight`: units 0–12, 15–28, 29–44, 45–90, 91–136 of §5 (units 13, 14 zero). fc2 rows: 121 ← 0; 122 ← 1, 2; 123 ← 3, 4; 124 ← 5, 6; 125 ← 7, 8; 57 ← 9, 10; 126 ← 11, 12; 49 ← 15; 50 ← 16; 51 ← 17; 52 ← 18, 19; 53 ← 20, 21; 54 ← 22, 23, 99, 100; 55 ← 24; 56 ← 25, 26; 58 ← 27, 28; 48–63 ← 29–44 (−1) and 119–134 (+1); 64–86 ← 45–90 (∓1); 87 ← 95, 96, 101, 102; 88 ← 97, 98, 103, 104; 89 ← 91, 92, 105, 106; 90 ← 107, 108, 135 (1), 136 (2); 91 ← 93, 94, 109, 110; 92–95 ← 111–118 (∓1).
- `layers.2.attn.rel_bias`: rows 0–2, 5–7: −10000 at columns 1–64; row 3: −10000 at 0 and 2–64; row 4: −10000 at 0 and 4–63, 0 at 64. `q.bias`: [0], [16], …, [112] = 1.
- `layers.2.attn.k.weight`: row 0 ← d56; 16 ← d55; 32 ← d121, d122; 48 ← d5; 64 ← d50, d51, d52, d123; 80 ← d112; 96 ← d90; 112 ← d91 (all 80).
- `layers.2.attn.v.weight` / `o.weight`: the value → output pairs of §6 (v rows 0, 1, 4, 5, 16–20, 32, 33, 48 (d8 and d127), 64–67, 80, 96, 112; o rows 59–74, 78, 92, 93).
- `layers.2.mlp.fc1.weight` / `fc1.bias` / `fc2.weight`: units 0–17 of §7 (fc2 rows 75 ← 0–5, 13–16; 76 ← 6–11 (±3); 77 ← 12; 94 ← 17).
- `layers.3.attn.rel_bias`: row 0: −10000 at 8–63, 20 at 64; row 1: 0, −6, …, −42 at 0–7, −10000 at 8–64; row 2: 0 at 0–3, −10000 at 4–64; row 3: 0 at 0–6, −10000 at 7–63, 20 at 64. `q.bias`: [0], [16], [32], [48] = 1.
- `layers.3.attn.k.weight`: rows 0 and 48 ← d5 (80); row 16 ← d5, d53 (240); row 32 ← d5, d76 (80). `k.bias`: [16] = −240, [32] = −80.
- `layers.3.attn.v.weight`: 1←d75, 2←d94, 16←d77, 32←d76, 48←d66. `o.weight`: 80←1, 95←2, 81←16, 82←32, 79←48.
- `layers.3.mlp.fc1.weight` / `fc1.bias` / `fc2.weight`: units 0–9 of §9 (fc2 rows 83–86, 112).
- `head.weight`: row 1: d85 = 8, d86 = 5; row 2: d84 = 6, d75 = 6, d85 = −8, d112 = −12; row 3: d83 = 12. `head.bias`: [0.5, −2, −8, 0].

Everything not listed is zero (in particular layer-0 heads 5–7, layer-1 `q.bias`, all `k.bias` except layer 3's,
all `v.bias`, `o.bias`, `fc2.bias`, layer-2/3 `q.weight`, layer-3 heads 4–7, and MLP units beyond those named).
