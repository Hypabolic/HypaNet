# modellabel: the two-model answer key

Nobody labels by hand (decision 2026-11-10, `SPEC.md` §5a). Two model labellers from different families label every
node of a set with `LABELLING.md`, each blind to the other, and disputed nodes get a second pass from both:

| Labeller id | Model | CLI | Effort |
|---|---|---|---|
| `opus55` | Claude Opus 5.5 (`claude-opus-5-5`) | `claude -p` (Claude Code 2.1.322) | high |
| `codex61` | GPT-6.1 Sol (`gpt-6.1-sol`) | `codex exec` (codex-cli 0.160.1), on the user's ChatGPT plan | high |

## Commands (from `slop/`)

```sh
# what it will cost (renders every prompt, calls nothing): the table, the thinking scenarios and the pilot's screens
uv run python modellabel/run.py estimate --set all

# look at the prompts first (written to labels/dryrun/, gitignored; nothing is called)
uv run python modellabel/run.py label --set reference_real --labeller both --dry-run

# pilot: 3 reference_real screens stratified by size (11, 24 and 31 labelable nodes today), both labellers
uv run python modellabel/run.py label --set reference_real --labeller both --pilot --workers 3 \
    --max-usd 5 --max-total-usd 5

# the pool, under one overall cap (resumable: rerun the same command after an interruption)
# (US$35 in all covers the "high" thinking scenario, about US$31 with the pilot; the per-run caps split it)
uv run python modellabel/run.py label --set reference_real --labeller both --workers 3 --max-usd 15 --max-total-usd 35
uv run python modellabel/run.py label --set calibration --labeller both --workers 3 --max-usd 20 --max-total-usd 35
uv run python modellabel/run.py adjudicate --set reference_real --workers 3 --max-usd 8 --max-total-usd 35
uv run python modellabel/run.py adjudicate --set calibration --workers 3 --max-usd 8 --max-total-usd 35

# agreement before and after adjudication (labels/<set>.agreement.md and .json), and the key's completeness
uv run python modellabel/run.py report --set reference_real
uv run python modellabel/run.py build --set reference_real

# checks on the key (SPEC.md §5a), never keyed: Codex without source files; both models again on 10% of the pool
uv run python modellabel/run.py label --set calibration --labeller codex --variant nosource --workers 3
uv run python modellabel/run.py label --set reference_real --labeller both --variant retest --workers 3 \
    --max-usd 3 --max-total-usd 40
uv run python modellabel/run.py label --set calibration --labeller both --variant retest --workers 3 \
    --max-usd 3 --max-total-usd 40
```

`label --set sealed_real` refuses until `datasets/sealed_real/` exists and `registry.yaml` records it unchanged
(`scripts/freeze.py --revise` after `build_sealed.py`), until every sealed screen has a product kind
(`datasets/sealed_real/screens/kinds.json`, gitignored), and unless the model, effort, CLI version, prompt version and
guide are those that labelled the pool.

Exit codes: 0 every requested screen has a valid reply (for `build` and `adjudicate`: and the key is complete); 1 a
screen is refused, invalid, errored or cut off, the key is incomplete, or `check.py` found errors; 2 refused to start
(no cap, another run holds the lock, a gate failed); 3 the spend cap stopped launches.

## Files

| File | What |
|---|---|
| `prompt.py` | Per-screen prompt: a stable prefix (preamble + the whole `LABELLING.md`) and a body (product kind, nodes, numbered source). Adjudication prompt. |
| `schema.json`, `adjudicate_schema.json` | Reply schemas (Codex strict-mode subset: every object closed, every field required). |
| `runners.py` | The two CLIs, the spend cap, one retry on an invalid reply. |
| `to_records.py` | Reply → `SPEC.md` §5 records (loc-keyed), label files rebuilt from raw replies, `check.py`. |
| `adjudicate.py` | Disputes, A/B order, second pass validation, `labels/<set>.final.jsonl`. |
| `agree.py` | Cohen's κ (binary, four-class, five-category), AC1, positive and negative agreement, pointer agreement, per role and repo, bootstrap intervals, contested rate, `role_ok` gate. |
| `estimate.py` | Cost estimate. |
| `run.py` | The CLI. |
| `tests/` | Stub `claude` and `codex` executables and the tests (invented fixtures only). |

Outputs (all under the gitignored `slop/labels/`): `labels/<set>.opus55.jsonl`, `labels/<set>.codex61.jsonl` and
their `.meta.json`; `labels/<set>.final.jsonl`; `labels/raw/<set>/<labeller>/<screen>.json` (prompt body, every
attempt's reply, errors, usage and cost); `labels/raw/<set>/adjudicate/<labeller>/<screen>.json`;
`labels/raw/<set>/prefix-<sha>.txt`; `labels/raw/spend.jsonl` (every Opus charge). Raw files hold private source
text: never commit them.

## Behaviour

- **One call per screen per labeller**, in `order.txt` order, `--workers` in parallel per labeller (`--pilot`,
  `--screens` or `--first N` for a subset). One `label` or `adjudicate` run at a time, whatever the set: a lock in
  `labels/raw/.modellabel.lock`.
- **The prompt** is the preamble plus the whole `LABELLING.md` (the cached prefix) and a body: the product kind (a
  fixture's note cut from it), the nodes as JSON lines with their window `w`, and the numbered source file in a fence
  longer than any backtick run in it. A literal ⟨, ⟩ or ‖ in copy is doubled so it never reads as an interpolation
  or branch marker. The preamble adds nothing to the guide but the window rule (a target in the same window; copy
  repeated only in another window labelled as if that window did not exist, aside `X-FILE`) and the null class of a
  node with no literal words; it carries no prior about how much copy is redundant, and uses the guide's own words
  for `UNSURE`.
- **Validation.** A reply must parse as one JSON object (a single fenced block is accepted), match the schema, label
  every node that is not context only exactly once, use real node indices with targets in the node's window, and
  pass `labelkit.validate_record` (pattern matches class, targets only on ECHO and NARRATE, `partial` only on KEEP,
  `role_fix` with `role_ok: false`). A node with no literal words (only interpolated identifiers) is set to class
  null with aside UNRESOLVED whatever the model said (`LABELLING.md` §2); null on a node with literal words is an
  error.
- **Retry.** One retry, with the validation errors appended to the same prompt; then the screen is recorded as
  `invalid` and runs again on the next invocation. A valid reply with a §5 precedence slip (an ECHO pointing at a node
  of lower rank, or an `equiv` entry not labelled ECHO) is sent back once with the slip listed; the second reply is
  accepted as it is, and if that call fails the first reply stands. A Codex reply whose event stream shows any item
  other than `agent_message`, `reasoning`, `error` or `todo_list`, started or completed, is invalid: a labeller that
  reads files could read the other's labels.
- **Refusals** (Opus `stop_reason: refusal`, `stop_details.category` or `safety_stops`; Codex a policy error, or a
  refusal sentence in a reply that holds no JSON at all) are recorded as `refused` and not retried, including on
  resume, unless `--retry-refused`. CLI errors, timeouts, cut-off replies (`stop_reason: max_tokens`: not retried in
  the run, since the same prompt would be cut off again) and per-call budget stops are recorded and run again on the
  next invocation. A crash inside one call is recorded as an error and never stops the run. Output is decoded as
  UTF-8 with replacement, and a timed-out CLI is killed with its whole process group.
- **Resume.** A screen is skipped when its raw file is `ok` (or `refused`) for the same prompt hash, which covers the
  guide, the screen and its source file, and for the same model and effort. Label files are rebuilt from the raw
  files every time; `build` fails when one labeller's valid replies in a set mix prompt prefixes or versions.
- **Spend cap.** `--max-usd` (required whenever Opus runs) caps this invocation's Opus spend and `--max-total-usd`
  everything in `labels/raw/spend.jsonl` (every set, phase and run). A call launches only if the spend so far plus
  the expected cost of every call in flight, this one included, stays within both (expected cost = mean of the calls
  that reported a cost, or `--per-call-usd` before the first). A launch that does not fit while calls are in flight
  waits for them; launches stop for good, and the run exits 3, only when nothing is in flight and the next call still
  would not fit. Codex calls never wait on the cap. A call that reports no cost (timeout, garbled output, crash) is
  charged at `--per-call-usd` and marked `estimated` in the ledger. Each launch is written to the ledger before the
  process starts and each charge after it ends, so a launch whose charge never arrived (the harness was killed
  mid-call) counts at its reservation in every later total.
- **Overshoot.** At most the in-flight calls' excess over the expected cost, so at most `--workers × --per-call-usd`,
  *provided no single call costs more than `--per-call-usd`*. `claude --max-budget-usd` is checked between turns, so
  it does not cut a single turn short: one labelling turn is bounded by `CLAUDE_CODE_MAX_OUTPUT_TOKENS` (64,000 × US$20
  per million = US$1.28 of output) plus its input (a 21,000-token body written to the cache at 2 × US$4 per million is
  US$0.17, the cached prefix about US$0.002), about US$1.50 at worst, under the US$2 default. Keep `--per-call-usd` at
  2 or more unless the output limit is lowered. If the harness itself is killed (SIGKILL) a running `claude` child is
  not killed with it: check `ps` before restarting.
- **Adjudication.** A node is disputed when the round-1 labels differ on the binary class (REDUNDANT, KEEP, UNSURE),
  on the class, or, for two ECHO or two NARRATE labels, on the target (a target in the other's `equiv` counts as the
  same). Both models see the guide, the screen, the agreed labels and both disputed labels as A and B, in an order
  fixed per node by `sha256("adjudicate:20261116:<screen>:<i>")` (the same for both adjudicators), without either
  labeller's `conf`, and answer each disputed node with a final label and a reason. Same answer from both (by the
  same tests) → `source: adjudicated`; otherwise → `source: contested`, `class: null`, `conf: "contested"`. Both
  UNSURE in round 1, or an agreed UNSURE, → contested without a second pass. Every final node record carries
  `binary`, the binary consensus of the final labels (`REDUNDANT`, `KEEP` or null), so a node contested only on its
  class or target is still scored in the binary headline. Agreed nodes merge as: `conf` judgement if either is, `ct`
  and `partial` if either is, `equiv` and `aside` united, and where the pattern codes differ one labeller's, chosen by
  hash per node. Every final record keeps both round-1 labels in `candidates`. Screen asides and missed copy enter the
  key only when both labellers raised them (missed copy: same line, one text inside the other; class UNSURE where the
  two differ). `labelkit` accepts these records only in a file whose labeller is `final`. `resolve_binary` gives the
  four fixed resolutions of the sensitivity analysis (SPEC.md §5a).
- **A complete key.** `build` (and `adjudicate`) exit 1 and print `INCOMPLETE KEY` while a screen lacks a valid
  round-1 label from both models (with each labeller's status) or a disputed screen lacks a second pass from both; a
  refused second pass leaves its nodes contested and counts as done. `scripts/freeze.py` refuses an incomplete
  `labels/<set>.final.jsonl`.
- **Report** (`run.py report`): κ, AC1 and positive and negative agreement with 95% intervals (screens resampled),
  binary with UNSURE as a third category, four- and five-category κ, pointer agreement, the inter-labeller ceiling
  (each labeller's REDUNDANT F1 and false-alarm rate against the other on headline nodes), `ct` disagreement, the
  non-control and clear/judgement subsets, per role and repo, screens missing from the key by status and repo,
  one-sided screen-level items, concession on the second pass, the κ gate (round 1) and the role gate, and the
  `retest` and `nosource` variants when present. The pause-menu fixture (its answer is in `LABELLING.md` §8) is left
  out of every statistic.
- **Variants** (`label --variant retest|nosource`) write `labels/raw/<set>/<labeller>-<variant>/` and
  `labels/<set>.<labeller>-<variant>.jsonl`; they never enter the key. `retest` takes 10% of the set by hash unless
  `--screens`, `--first` or `--pilot` is given; `nosource` sends the screen without its source file.

## Isolation (what `--help` says, and what is and is not isolated)

**Opus** (`runners.OpusRunner.argv`):

```
claude -p --model claude-opus-5-5 --effort high --output-format json
  --system-prompt <preamble + LABELLING.md> --tools "" --disallowedTools Bash Read Write Edit Glob Grep … ToolSearch
  --permission-mode dontAsk --permission-prompts none --setting-sources project
  --settings '{"autoMemoryEnabled": false}' --strict-mcp-config --disable-slash-commands --no-session-persistence
  --max-budget-usd 2.00 --safe-mode          (screen body on stdin; cwd: <tmp>/slop-modellabel/opus55, empty)
```

Environment: the Clouds launcher's scrub list is removed (`CLAUDECODE`, `CLAUDE_CODE_ENTRYPOINT`,
`CLAUDE_CODE_SESSION_ID`, `CLAUDE_CODE_EFFORT_LEVEL`, which outranks `--effort`, `ANTHROPIC_MODEL`, …), and
`CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK=1`, `CLAUDE_CODE_NO_MODEL_FALLBACK=1`, `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`,
`CLAUDE_CODE_DISABLE_CLAUDE_MDS=1`, `CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000` are set.

- Isolated: tools (`--tools ""`: "Use \"\" to disable all tools", plus the deny list); CLAUDE.md (empty working
  directory outside any repository, checked with `git rev-parse` before a run; `CLAUDE_CODE_DISABLE_CLAUDE_MDS`;
  `--safe-mode`, which disables "CLAUDE.md, skills, installed plugins, hooks, MCP servers, custom commands and
  agents, output styles, workflows"; no `~/.claude/CLAUDE.md` exists on this machine); user and local settings,
  hooks and plugins (`--setting-sources project`, and the empty directory has no project settings); auto memory
  (env and `autoMemoryEnabled: false`); MCP (`--strict-mcp-config` with no `--mcp-config`); skills and slash
  commands (`--disable-slash-commands`); transcripts (`--no-session-persistence`, so no sealed copy lands in
  `~/.claude/projects`); Claude Code's default system prompt, with its environment, git and memory sections
  (replaced by `--system-prompt`, which also makes the guide the cached prefix).
- Not isolated: admin-managed (policy) settings ("still apply" under `--safe-mode` and `--restricted`); the logged-in
  account (OAuth/keychain; `--bare` would cut keychain reads too but then needs `ANTHROPIC_API_KEY`); `~/.claude.json`
  global state; Anthropic-side prompt caching (wanted); environment variables outside the scrub list.
- Not used: `--json-schema` (structured output with `--print`): untested here, and Claude Code implements it with a
  tool; the harness validates replies itself. Worth trying in the pilot if replies often fail validation.
- `--safe-mode` was not used by the Clouds launcher; drop it (`OpusRunner(safe_mode=False)`) if the pilot shows it
  interferes with `--system-prompt`.

**Codex** (`runners.CodexRunner.argv`):

```
codex exec -m gpt-6.1-sol -c model_reasoning_effort="high" --sandbox read-only --skip-git-repo-check --ephemeral
  --ignore-user-config --ignore-rules -c web_search="disabled"
  --disable shell_tool --disable unified_exec --disable apps --disable plugins --disable multi_agent
  --disable image_generation --disable browser_use --disable computer_use --disable view_image --disable hooks
  --disable skill_search --disable tool_suggest --disable sleep_tool --disable goals --disable code_mode_host
  --disable browser_use_external --disable browser_use_full_cdp_access --disable in_app_browser
  --disable remote_plugin
  --output-schema modellabel/schema.json --output-last-message <tmp file> --json --color never
  -C <tmp>/slop-modellabel/codex61 -          (prefix + body on stdin)
```

- Isolated: `~/.codex/config.toml` (`--ignore-user-config`: its MCP server, hooks, trusted projects and profile;
  "auth still uses CODEX_HOME"); execpolicy rules (`--ignore-rules`); session files (`--ephemeral`); writes
  (`--sandbox read-only`); project instructions (empty `-C` directory, and there is no `~/.codex/AGENTS.md`); the
  tool features above (every name is listed by `codex features list` as stable and on by default); web search
  (`web_search="disabled"`: checked without a model call, `codex -c 'web_search="bogus"' features list` fails with
  "expected one of `disabled`, `cached`, `indexed`, `live` in `web_search`", so the key and value are real).
- Not isolated: Codex's own base instructions; files on disk are readable in a read-only sandbox, so tool use is
  disabled and, as a backstop, any item in the `--json` events other than `agent_message`, `reasoning`, `error` or
  `todo_list` (started, updated or completed) makes the reply invalid; skills under
  `~/.codex/skills` may still be listed to the model. If a newer Codex rejects a feature name, edit `CODEX_DISABLE`.
- Usage: `turn.completed.usage` (`input_tokens`, `cached_input_tokens`, `output_tokens`,
  `reasoning_output_tokens`) is recorded per attempt, with any model name the events carry.

Both runners record `<cli> --version` (no model call) in every raw record; `sealed_real` refuses a version, model,
effort, prompt version or prefix other than the pool's.

**Before sealed copy is sent:** confirm that training on user content is off for the Claude account and the ChatGPT
plan (SPEC.md §16).

None of the flags above has been exercised against a real model here (no paid calls were made while building this).
The pilot is the first real test of every one of them.

## Cost

`run.py estimate --set all` (2026-11-11, sets and guide as of that day, prompt version 2); assumptions are printed with the
table and listed in `estimate.py`. Claude Code marks the conversation tail for caching, so each screen body is billed
as a cache write, and the Clouds r1 log shows 1-hour cache entries (2 × input), hence the second price column.

| Item | Screens | Calls (Opus) | Opus input tok | Opus output tok (incl. thinking) | Opus US$ (5-min cache writes) | Opus US$ (1-hour writes) | Codex input tok | Codex output tok |
|---|---|---|---|---|---|---|---|---|
| pilot (3 reference_real screens of 11, 24, 31 labelable nodes, both labellers) | 3 | 3 | 0.05M | 0.02M | 0.51 | 0.60 | 0.07M | 0.02M |
| reference_real: labelling | 22 | 23 | 0.26M | 0.07M | 1.98 | 2.22 | 0.39M | 0.07M |
| reference_real: adjudication (10% of nodes disputed) | 22 | 12 | 0.15M | 0.02M | 0.80 | 0.98 | 0.21M | 0.02M |
| calibration: labelling | 25 | 26 | 0.36M | 0.12M | 3.34 | 3.79 | 0.50M | 0.12M |
| calibration: adjudication (10% of nodes disputed) | 25 | 15 | 0.23M | 0.03M | 1.28 | 1.65 | 0.31M | 0.03M |
| sealed_real: labelling | 250 | 262 | 3.99M | 1.45M | 39.11 | 44.56 | 5.44M | 1.45M |
| sealed_real: adjudication (10% of nodes disputed) | 250 | 238 | 3.62M | 0.46M | 19.30 | 24.84 | 4.93M | 0.46M |
| **Total (without pilot)** | | 577 | 8.61M | 2.17M | **65.80** | 78.04 | 11.78M | 2.17M |

Measured from rendered prompts: reference_real: 22 screens, 212 labelable nodes, prefix 7,823 tokens, body mean 3,126 / max 9,700 tokens; calibration: 25 screens, 453 labelable nodes, prefix 7,823 tokens, body mean 5,414 / max 21,078 tokens.


Scenarios (Opus US$, labelling and adjudication, 5-min / 1-hour cache writes):

| Scenario | Thinking per call | Disputed | Pilot | reference_real | calibration | sealed_real | Total |
|---|---|---|---|---|---|---|---|
| assumed | 1,500 + 120 per node | 10% | 0.51 / 0.60 | 2.77 / 3.20 | 4.62 / 5.45 | 58.41 / 69.40 | 65.80 / 78.04 |
| more thinking | 4,000 + 250 per node | 10% | 0.85 / 0.94 | 5.18 / 5.60 | 8.03 / 8.85 | 100.28 / 111.27 | 113.49 / 125.73 |
| more thinking, more disputes | 4,000 + 400 per node | 20% | 1.06 / 1.15 | 6.56 / 7.03 | 10.54 / 11.43 | 128.25 / 139.96 | 145.35 / 158.42 |
| high | 8,000 + 600 per node | 25% | 1.59 / 1.68 | 11.20 / 11.68 | 17.04 / 17.96 | 202.73 / 214.57 | 230.97 / 244.20 |

Pilot screens (run.py label --pilot): 9d693509558ed2f8, 48eff9e28be7cba1, 2194e4444913ebb4.

Thinking is the least certain figure (1,500 + 120 tokens per node is an assumption, probably low for a six-step
procedure over 22 nodes), so until the pilot has measured it the cap is a range: about US$7–30 for `reference_real`
(22 screens as rebuilt on 2026-11-11) and calibration together, labelling and adjudication, and US$58–215 for
`sealed_real`. The pilot's three screens (11, 24 and 31 labelable nodes, chosen by `--pilot`; the sealed mean is
22.45) measure thinking per node across that range; set `THINK_BASE` and `THINK_PER_NODE` in `estimate.py` from the pilot's `usage` and rerun
`estimate` before approving the sealed cap.
