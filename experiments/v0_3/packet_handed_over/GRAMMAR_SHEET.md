# Grammar sheet (spec v0.3 §4-§5)

Each example below is one **event block** as the frozen renderer lays it out, followed by its
tokens from the frozen tokenizer (`tokenizer.tokenize_block`, then `EOE`), one per line as
`offset  id  surface`. Line-marker tokens are shown as `NL@`, `NL!` and `NL[`.

Every path, suite, test, target, host, action, error class, hunk id and number here is
**invented** for illustration. Real logs use the same grammar with other values.

A session is `BOS` followed by its blocks in order (`tokenizer.encode_session`). `event_ord` is 0
for `BOS` and k+1 for every token of block k, including its `EOE`; `offset` restarts at 0 in every
block. The network's prediction for block k is read at that block's `EOE`.

Header suffix: `-> {status}[ {passed}/{total} passed] in={input_tokens} out={output_tokens}`.
Body lines, at most 6, in this order: hunk lines (`@`, at most 5), the error line (`!`, ERROR
outcomes only) and the workspace line (`[`, when a digest was dropped). A dropped observation is
rendered as `?` in its own position.

## read_file

```text
09:14:03.250 read_file path=zork/zeta/widget.qx lines=12-40 -> ok in=1840 out=96
```

```tokens
  0    243  09
  1    143  :
  2    248  14
  3    143  :
  4    237  03
  5    141  .
  6    584  250
  7      7  read_file
  8     21  path
  9    146  =
 10   6414  zork/zeta/widget.qx
 11     22  lines
 12    146  =
 13    246  12
 14    140  -
 15    274  40
 16    140  -
 17    147  >
 18     38  ok
 19     36  in
 20    146  =
 21    518  184
 22    224  0
 23     37  out
 24    146  =
 25    330  96
 26      2  <EOE>
```

## search

```text
09:14:05.004 search scope=zork/zeta query="quux frobnicate" limit=5 -> ok in=1932 out=41
```

```tokens
  0    243  09
  1    143  :
  2    248  14
  3    143  :
  4    239  05
  5    141  .
  6    338  004
  7      8  search
  8     23  scope
  9    146  =
 10   7568  zork/zeta
 11     24  query
 12    146  =
 13    129  "
 14   8189  quux
 15   8358  frobnicate
 16    129  "
 17     25  limit
 18    146  =
 19    229  5
 20    140  -
 21    147  >
 22     38  ok
 23     36  in
 24    146  =
 25    527  193
 26    226  2
 27     37  out
 28    146  =
 29    275  41
 30      2  <EOE>
```

## list_dir

```text
09:14:06.117 list_dir path=zork/zeta -> ok in=2010 out=22
```

```tokens
  0    243  09
  1    143  :
  2    248  14
  3    143  :
  4    240  06
  5    141  .
  6    451  117
  7      9  list_dir
  8     21  path
  9    146  =
 10   7568  zork/zeta
 11    140  -
 12    147  >
 13     38  ok
 14     36  in
 15    146  =
 16    535  201
 17    224  0
 18     37  out
 19    146  =
 20    256  22
 21      2  <EOE>
```

## edit_file (one line per changed file: removed-lines id > added-lines id)

```text
09:14:09.880 edit_file path=zork/zeta/widget.qx -> ok in=2214 out=310
  @ zork/zeta/widget.qx h3f9a0c1d2e > h7b21e9c04a
```

```tokens
  0    243  09
  1    143  :
  2    248  14
  3    143  :
  4    243  09
  5    141  .
  6   1214  880
  7     10  edit_file
  8     21  path
  9    146  =
 10   6414  zork/zeta/widget.qx
 11    140  -
 12    147  >
 13     38  ok
 14     36  in
 15    146  =
 16    555  221
 17    228  4
 18     37  out
 19    146  =
 20    644  310
 21      4  NL@
 22   6414  zork/zeta/widget.qx
 23   2227  h3f9a0c1d2e
 24    147  >
 25   4182  h7b21e9c04a
 26      2  <EOE>
```

## write_file

```text
09:14:12.301 write_file path=zork/zeta/gizmo_plinth.qx -> ok in=2402 out=505
  @ zork/zeta/gizmo_plinth.qx h0 > h9e4a7710cd
```

```tokens
  0    243  09
  1    143  :
  2    248  14
  3    143  :
  4    246  12
  5    141  .
  6    635  301
  7     11  write_file
  8     21  path
  9    146  =
 10   7212  zork/zeta/gizmo_plinth.qx
 11    140  -
 12    147  >
 13     38  ok
 14     36  in
 15    146  =
 16    574  240
 17    226  2
 18     37  out
 19    146  =
 20    839  505
 21      4  NL@
 22   7212  zork/zeta/gizmo_plinth.qx
 23   5861  h0
 24    147  >
 25   7348  h9e4a7710cd
 26      2  <EOE>
```

## revert (undoes the edit_file block: the ids swap back)

```text
09:14:15.762 revert paths=zork/zeta/widget.qx -> ok in=2470 out=12
  @ zork/zeta/widget.qx h7b21e9c04a > h3f9a0c1d2e
```

```tokens
  0    243  09
  1    143  :
  2    248  14
  3    143  :
  4    249  15
  5    141  .
  6   1096  762
  7     12  revert
  8     27  paths
  9    146  =
 10   6414  zork/zeta/widget.qx
 11    140  -
 12    147  >
 13     38  ok
 14     36  in
 15    146  =
 16    581  247
 17    224  0
 18     37  out
 19    146  =
 20    246  12
 21      4  NL@
 22   6414  zork/zeta/widget.qx
 23   4182  h7b21e9c04a
 24    147  >
 25   2227  h3f9a0c1d2e
 26      2  <EOE>
```

## run_tests (failed, with validation counts and an error line)

```text
09:14:20.415 run_tests suite=alpha tests=alpha/spin_wobble -> FAILED 7/9 passed in=2611 out=188
  ! QuuxError at zork/zeta/widget.qx:417 (pid 4821)
```

```tokens
  0    243  09
  1    143  :
  2    248  14
  3    143  :
  4    254  20
  5    141  .
  6    749  415
  7     13  run_tests
  8     28  suite
  9    146  =
 10   7853  alpha
 11     29  tests
 12    146  =
 13   2519  alpha/spin_wobble
 14    140  -
 15    147  >
 16     40  FAILED
 17    231  7
 18    142  /
 19    233  9
 20     41  passed
 21     36  in
 22    146  =
 23    595  261
 24    225  1
 25     37  out
 26    146  =
 27    522  188
 28      5  NL!
 29   2810  QuuxError
 30     44  at
 31   6414  zork/zeta/widget.qx
 32    143  :
 33    751  417
 34    135  (
 35     45  pid
 36    816  482
 37    225  1
 38    136  )
 39      2  <EOE>
```

## run_tests (dropped observations and untracked workspace)

```text
09:14:22.006 run_tests suite=? -> FAILED ?/9 passed in=? out=176
  ! ? at ?:? (0x1f2e3d4c)
  [workspace untracked]
```

```tokens
  0    243  09
  1    143  :
  2    248  14
  3    143  :
  4    256  22
  5    141  .
  6    340  006
  7     13  run_tests
  8     28  suite
  9    146  =
 10    148  ?
 11    140  -
 12    147  >
 13     40  FAILED
 14    148  ?
 15    142  /
 16    233  9
 17     41  passed
 18     36  in
 19    146  =
 20    148  ?
 21     37  out
 22    146  =
 23    510  176
 24      5  NL!
 25    148  ?
 26     44  at
 27    148  ?
 28    143  :
 29    148  ?
 30    135  (
 31    224  0
 32   6279  x1f2e3d4c
 33    136  )
 34      6  NL[
 35     46  workspace
 36     47  untracked
 37    152  ]
 38      2  <EOE>
```

## build

```text
09:14:31.002 build target=zeta_bale -> ok in=2702 out=64
```

```tokens
  0    243  09
  1    143  :
  2    248  14
  3    143  :
  4    265  31
  5    141  .
  6    336  002
  7     14  build
  8     30  target
  9    146  =
 10   9591  zeta_bale
 11    140  -
 12    147  >
 13     38  ok
 14     36  in
 15    146  =
 16    604  270
 17    226  2
 18     37  out
 19    146  =
 20    298  64
 21      2  <EOE>
```

## lint (failed, hex noise)

```text
09:14:33.640 lint target=zeta_bale -> FAILED in=2750 out=71
  ! QuuxError at zork/zeta/gizmo_plinth.qx:88 (0x9c1e44b7)
```

```tokens
  0    243  09
  1    143  :
  2    248  14
  3    143  :
  4    267  33
  5    141  .
  6    974  640
  7     15  lint
  8     30  target
  9    146  =
 10   9591  zeta_bale
 11    140  -
 12    147  >
 13     40  FAILED
 14     36  in
 15    146  =
 16    609  275
 17    224  0
 18     37  out
 19    146  =
 20    305  71
 21      5  NL!
 22   2810  QuuxError
 23     44  at
 24   7212  zork/zeta/gizmo_plinth.qx
 25    143  :
 26    322  88
 27    135  (
 28    224  0
 29   9034  x9c1e44b7
 30    136  )
 31      2  <EOE>
```

## http_call

```text
09:14:40.118 http_call POST https://quuxhost.zeta/v9/gadgets/{gid} body={"flavor":"quux"} req=5d03ac91 -> error in=2810 out=33
  ! QuuxError at zeta_portal:212 (pid 30177)
```

```tokens
  0    243  09
  1    143  :
  2    248  14
  3    143  :
  4    274  40
  5    141  .
  6    452  118
  7     16  http_call
  8     51  POST
  9     49  https
 10    143  :
 11    142  /
 12    142  /
 13   2727  quuxhost.zeta/v9/gadgets/
 14    155  {
 15   2509  gid
 16    157  }
 17     31  body
 18    146  =
 19    155  {
 20    129  "
 21   7249  flavor
 22    129  "
 23    143  :
 24    129  "
 25   8189  quux
 26    129  "
 27    157  }
 28     32  req
 29    146  =
 30    229  5
 31   5868  d03ac91
 32    140  -
 33    147  >
 34     39  error
 35     36  in
 36    146  =
 37    615  281
 38    224  0
 39     37  out
 40    146  =
 41    267  33
 42      5  NL!
 43   2810  QuuxError
 44     44  at
 45   7250  zeta_portal
 46    143  :
 47    546  212
 48    135  (
 49     45  pid
 50    635  301
 51    311  77
 52    136  )
 53      2  <EOE>
```

## poll_status

```text
09:14:52.733 poll_status resource=zeta_hopper backoff=4 -> pending in=2851 out=18
```

```tokens
  0    243  09
  1    143  :
  2    248  14
  3    143  :
  4    286  52
  5    141  .
  6   1067  733
  7     17  poll_status
  8     33  resource
  9    146  =
 10   3989  zeta_hopper
 11     34  backoff
 12    146  =
 13    228  4
 14    140  -
 15    147  >
 16     43  pending
 17     36  in
 18    146  =
 19    619  285
 20    225  1
 21     37  out
 22    146  =
 23    252  18
 24      2  <EOE>
```

## plan

```text
09:14:55.090 plan action=sketch_zeta_zigzag -> ok in=2902 out=640
```

```tokens
  0    243  09
  1    143  :
  2    248  14
  3    143  :
  4    289  55
  5    141  .
  6    424  090
  7     18  plan
  8     35  action
  9    146  =
 10   6721  sketch_zeta_zigzag
 11    140  -
 12    147  >
 13     38  ok
 14     36  in
 15    146  =
 16    624  290
 17    226  2
 18     37  out
 19    146  =
 20    974  640
 21      2  <EOE>
```

## ask

```text
09:15:01.512 ask action=ponder_zeta_flavor -> ok in=3015 out=57
```

```tokens
  0    243  09
  1    143  :
  2    249  15
  3    143  :
  4    235  01
  5    141  .
  6    846  512
  7     19  ask
  8     35  action
  9    146  =
 10   5345  ponder_zeta_flavor
 11    140  -
 12    147  >
 13     38  ok
 14     36  in
 15    146  =
 16    635  301
 17    229  5
 18     37  out
 19    146  =
 20    291  57
 21      2  <EOE>
```

## yield

```text
09:15:09.377 yield action=zeta_handback -> ok in=3104 out=212
```

```tokens
  0    243  09
  1    143  :
  2    249  15
  3    143  :
  4    243  09
  5    141  .
  6    711  377
  7     20  yield
  8     35  action
  9    146  =
 10   3504  zeta_handback
 11    140  -
 12    147  >
 13     38  ok
 14     36  in
 15    146  =
 16    644  310
 17    228  4
 18     37  out
 19    146  =
 20    546  212
 21      2  <EOE>
```

## revert of 7 files (more changed files than fit: the 5th line is replaced by a count)

```text
09:15:12.020 revert paths=zork/blorb.qx,zork/frotz.qx,zork/gnusto.qx,zork/plugh.qx,zork/rezrov.qx,zork/xyzzy.qx,zork/yomin.qx -> ok in=3190 out=902
  @ zork/blorb.qx h2a6b8c4d1e > h61f0e2d9ab
  @ zork/frotz.qx h4d7c1b0e5f > h0
  @ zork/gnusto.qx h0 > h8a3e5f7c21
  @ zork/plugh.qx h3f9a0c1d2e > h5c0d81f3b6
  @ TRUNC 3
```

```tokens
  0    243  09
  1    143  :
  2    249  15
  3    143  :
  4    246  12
  5    141  .
  6    354  020
  7     12  revert
  8     27  paths
  9    146  =
 10   3432  zork/blorb.qx
 11    139  ,
 12   9004  zork/frotz.qx
 13    139  ,
 14   8119  zork/gnusto.qx
 15    139  ,
 16   2979  zork/plugh.qx
 17    139  ,
 18   8778  zork/rezrov.qx
 19    139  ,
 20   6754  zork/xyzzy.qx
 21    139  ,
 22   7221  zork/yomin.qx
 23    140  -
 24    147  >
 25     38  ok
 26     36  in
 27    146  =
 28    653  319
 29    224  0
 30     37  out
 31    146  =
 32   1236  902
 33      4  NL@
 34   3432  zork/blorb.qx
 35   2564  h2a6b8c4d1e
 36    147  >
 37   4074  h61f0e2d9ab
 38      4  NL@
 39   9004  zork/frotz.qx
 40   6665  h4d7c1b0e5f
 41    147  >
 42   5861  h0
 43      4  NL@
 44   8119  zork/gnusto.qx
 45   5861  h0
 46    147  >
 47   9006  h8a3e5f7c21
 48      4  NL@
 49   2979  zork/plugh.qx
 50   2227  h3f9a0c1d2e
 51    147  >
 52   8309  h5c0d81f3b6
 53      4  NL@
 54     48  TRUNC
 55    227  3
 56      2  <EOE>
```
