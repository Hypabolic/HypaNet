# Claude Fable 5.2: facts for a headless HypaNet authoring run

Compiled 2026-11-10 for the slop-detector experiment (`slop/`). Every fact below names the page it came from. Pages were read with WebFetch on 2026-11-10; local checks were run on this machine the same day. No model was called and no session was started.

Short links used below:

- [OV] https://platform.claude.com/docs/en/models/fable-5-2/overview
- [WN] https://platform.claude.com/docs/en/models/fable-5-2/whats-new-fable-5-2
- [MG] https://platform.claude.com/docs/en/models/fable-5-2/migration-guide
- [PR] https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-2
- [EF] https://platform.claude.com/docs/en/build-with-claude/effort
- [PZ] https://platform.claude.com/docs/en/about-claude/pricing
- [RF] https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback
- [MC] https://code.claude.com/docs/en/model-config
- [CLI] https://code.claude.com/docs/en/cli-reference
- [ENV] https://code.claude.com/docs/en/env-vars
- [LOOP] https://code.claude.com/docs/en/agent-sdk/agent-loop
- [CL] https://code.claude.com/docs/en/changelog

## 1. The model

| Fact | Value | Source |
|---|---|---|
| API id | `claude-fable-5-2` | [OV] |
| Released | 9 November 2026; retirement not before 9 November 2027 | [OV] |
| Context window | 1M tokens (default and maximum) | [OV], [WN] |
| Max output | 128K tokens per synchronous request | [OV] |
| Thinking | Adaptive, always on, cannot be turned off; `thinking: disabled` and `budget_tokens` return 400 | [OV], [WN] |
| Effort levels | `low`, `medium`, `high`, `xhigh`, `max`; API default `high` (unchanged from Fable 5.1) | [WN], [MG] |
| Tokeniser | The one introduced with Claude Opus 4.7 (same as Fable 5.1) | [WN], [MG] |
| Knowledge cutoff | June 2026 | [OV] |
| Forced tool use | `tool_choice` `any` / `tool` returns 400 (unchanged from Fable 5.1) | [WN], [MG] |
| Sampling params, prefill | Omit `temperature`/`top_p`/`top_k`; prefill returns 400 (unchanged from Fable 5.1) | [MG] |
| Data retention | Requires 30-day retention; not available under ZDR unless authorised (same as Fable 5.1) | [OV], [MG] |
| Rate limit | Its own limit, separate from the one Fable 5.1 and Fable 5 share; lower token limits per tier than Opus 5.5 | [MG] |
| Priority Tier | Not supported | [MG] |

## 2. Prices, Fable 5.2 against Fable 5.1 (USD per million tokens)

| Item | Fable 5.2 | Fable 5.1 | Ratio | Source |
|---|---|---|---|---|
| Base input | 8 | 10 | 0.8 | [PZ] |
| 5-minute cache write | 10 | 12.50 | 0.8 | [PZ] |
| 1-hour cache write | 16 | 20 | 0.8 | [PZ] |
| Cache read (hit or refresh) | 0.20 | 0.25 | 0.8 | [PZ] (0.025x base input on both) |
| Output (thinking is billed as output) | 40 | 50 | 0.8 | [PZ], [MG] |
| Batch API | 4 / 20 | 5 / 25 | 0.8 | [PZ] |
| Fast mode | 16 / 80 | n/a in the tables read | | [PZ], [WN] |

- Long context: no premium above 200K tokens for Fable models. The 1M window "uses standard pricing with no premium above 200K, except Haiku 5.5" [MC]. The pricing page's long-context section names only Haiku 5.5 as having a higher tier [PZ].
- `inference_geo: "us"` multiplies every category by 1.1x; the default `global` is standard price [PZ]. The past HypaNet runs show `"inference_geo": "global"` in their result events.
- Every price category drops by exactly the same factor, 0.8. So the same token counts cost exactly 80% of what they cost on Fable 5.1.

## 3. What changes from Fable 5.1 (API level)

Four breaking changes, all from [WN] and [MG]:

1. The prefix check (editing an earlier turn, `system` or `tools` invalidates later thinking blocks, 400 error) now applies to every account. [WN] and [MG] both say: "Nothing changes for you if Claude Code ... or the Claude Agent SDK manages your conversation history."
2. Only Fable 5.2 and Mythos 5.2 can read Fable 5.2 thinking blocks. A switch to any other model (Fable 5.1, Opus 5.5, or the fallback models Opus 4.8 and Opus 5) drops that reasoning; the request still succeeds, and the first turn after the switch can cost more [WN], [MG].
3. `computer_20251124` returns 400 except on Bedrock (not used by HypaNet) [WN].
4. With the advisor tool, a Fable 5.2 executor accepts only a Fable 5.2 or Mythos 5.2 advisor (not used by HypaNet) [WN].

Unchanged from Fable 5.1: API surface, always-on adaptive thinking, forced-tool-choice restriction, refusal handling and `stop_details` categories, data retention [MG].

## 4. Behaviour changes that matter for a long unattended run

From [WN] "Behavior differences", [MG] "Behavior changes", and [PR]:

- **Thinking is harder to suppress.** Fable 5.1 usually obeys "answer directly without deliberating"; Fable 5.2 thinks anyway at `medium` and above, and "at `high` and above, the model thinks before almost every reply" [WN], [PR]. Effort is the only lever.
- **`max` is expensive by design.** "Reserve `max` for measuring the model's ceiling and for the rare task where time and tokens don't matter ... It runs slower and costs more than at `xhigh`, often for only a small gain" [EF]. At `max`, "Turns are slow and use many more tokens than at `xhigh`" [PR]. `xhigh` "raised scores slightly and used many more tokens" than `high` on Anthropic's coding evals [PR]. None of these pages gives a number for Fable 5.2 `max` against Fable 5.1 `max`.
- **Effort settings do not carry over.** "Rerun your effort sweep instead of carrying over a setting you tuned on Claude Fable 5.1" [PR]. At `low`, Fable 5.2 scored "close to Claude Fable 5.1's scores at higher effort levels" in Anthropic's testing [PR].
- **Early stops.** On long multistep tasks the model sometimes ends a turn with a list of what remains or an offer to continue, more often at `low` and `medium` [WN], [PR]. [PR] gives a "Finishing work" system-prompt block; with it, output tokens rose about 30% (Anthropic replay at `medium`).
- **Self-review subagents.** At `xhigh` and `max` in an agentic coding product, Fable 5.2 started subagents to review its own work; Fable 5.1 did not on the same tasks. "Subagents can account for a large part of what a session costs" [MG], [PR].
- **Budget text in the prompt.** "The size of that budget changes how much Claude Fable 5.2 spends. In Anthropic's testing on coding tasks at `high` and `xhigh` effort, removing a large budget lowered cost, with equivalent performance" [PR].
- **Silent tool runs.** Long runs of tool calls can pass with little user-facing text, and the first tool call can come without a preamble [WN], [PR]. Narration between tool calls arrives as progress-update `thinking` blocks whose text is empty at the default `display: "omitted"` [PR], [MG].
- **Style rules are followed closely.** A hard sentence-length rule can make prose choppy [WN], [PR].
- **Code ownership.** If the prompt says the code is the model's own, it merges repeated code and makes unrequested changes (about 1.5x output tokens in Anthropic's small-task tests) [PR].
- **Caution piles up.** Fable 5.2 follows accumulated "never" rules and notes closely and can hold back routine work such as retries; name the steps that need no approval [PR].
- **Small `max_tokens`.** Thinking counts toward `max_tokens`, so a small limit can end a turn after thinking with no text [MG]. [EF] says to set a large `max_tokens` at `high` and above.

## 5. Refusals and fallback

- Fable 5.2 runs safety classifiers. A declined request returns HTTP 200 with `stop_reason: "refusal"` and `stop_details.category` [WN], [RF].
- Categories: `cyber`, `bio`, `frontier_llm` ("could help with the development of competing AI models ... Benign machine learning work can also trigger this category"), `reasoning_extraction` (asks the model to reproduce its internal reasoning in the response text), `general_harms` [RF], [PR].
- Billing: a refusal before any output is billed when the category is `bio`, `frontier_llm` or `reasoning_extraction`; a mid-stream refusal bills input plus the output already streamed [RF].
- Permitted fallback targets for Fable 5.2: Opus 4.8 and Opus 5 [WN]. In Claude Code, Fable models re-run biology-flagged requests on Opus 5 and cybersecurity-flagged requests on Opus 4.8 [MC]. The switch is controlled by the `switchModelsOnFlag` setting or `CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK=1` [MC], [ENV]; since 2.1.312 it also covers unrecognised model names [CL].
- HypaNet history: 0 occurrences of `"refusal"` in the six Fable 5.1 transcripts checked (`experiments/v0_3/logs/*.stream.jsonl` x3, `experiments/v0_5/phase_c/logs/v05c.stream.jsonl`, `clouds/runs/r1/logs/r1*.stream.jsonl` x2; `grep -c '"refusal"'`), and the clouds r1 audit lists only `claude-fable-5-1` under `models` (`clouds/runs/r1/logs/r1.audit.json`).

## 6. Claude Code

| Fact | Source |
|---|---|
| **Fable 5.2 requires Claude Code v2.1.321 or later.** | [MC] |
| 2.1.321 changelog: "Added Claude Fable 5.2 (`claude-fable-5-2`), now the default Fable model — 1M context, $8/$40 per Mtok with $0.20/Mtok cache reads" | [CL] |
| `--model fable` resolves to Fable 5.2 from v2.1.321 (except in Claude apps gateway sessions); a full id also works | [MC] |
| Claude Code's own default effort for Fable 5.2 is `medium` (the API default is `high`) | [MC] |
| Effort resolution order: `CLAUDE_CODE_EFFORT_LEVEL` env, then an in-session pick, then `--effort`, then settings, then the model default. `maxEffortLevel` and organisation caps still apply | [MC], [ENV] |
| `max` applies to the current session only unless set through `CLAUDE_CODE_EFFORT_LEVEL`; it cannot be saved in `effortLevel` | [MC] |
| `--max-budget-usd`: print mode only; checked against Claude Code's client-side estimate; subagent spend counts; spend can pass the cap; on `--continue`/`--resume` the totals restored from earlier runs do not count toward it | [CLI] |
| The cap is checked after each model response arrives, so spend can pass it by up to one response's cost plus any running subagents; result subtype `error_max_budget_usd` | [LOOP] |
| Fable models auto-compact at about 967K tokens by default | [MC] |
| Thinking cannot be turned off on Fable models; thinking tokens are billed even when hidden; interactive sessions get redacted thinking unless `showThinkingSummaries: true` | [MC] |
| `CLAUDE_CODE_SUBAGENT_EFFORT` added in 2.1.316; from 2.1.317 subagents on a different model use that model's saved or default effort | [CL] |

Local checks (2026-11-10):

- `claude --version` prints `2.1.311 (Claude Code)`. Binary: `/opt/homebrew/Caskroom/claude-code/2.1.311/claude`.
- `claude --help` lists every flag the HypaNet launchers use: `-p`, `--model`, `--effort <level>` ("low, medium, high, xhigh, max"), `--max-budget-usd <amount>` ("only works with --print"), `--fallback-model`, `--permission-mode` (incl. `dontAsk`), `--permission-prompts host|none`, `--settings`, `--setting-sources`, `--strict-mcp-config`, `--tools`, `--disallowedTools`, `--disable-slash-commands`, `--output-format stream-json`, `--verbose`, `--session-id`, `-r/--resume`, `--autocompact`, `--system-prompt-snapshot`.
- `strings` on the 2.1.311 binary finds `claude-fable-5-1` (37 times) and no `claude-fable-5-2`. It does contain `CLAUDE_CODE_NO_MODEL_FALLBACK`, `CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK`, `CLAUDE_CODE_EFFORT_LEVEL` and `switchModelsOnFlag`.
- `brew info --cask claude-code` reports 2.1.311, so the local Homebrew metadata is not yet at 2.1.321.
- `CLAUDE_CODE_NO_MODEL_FALLBACK` (set by `clouds/scripts/launch.sh`) has no row in [ENV]; the documented switch for content-based fallback is `CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK=1` or `switchModelsOnFlag: false` [ENV], [MC].

## 7. Past HypaNet run costs, repriced at Fable 5.2 list prices

All three runs used `claude-fable-5-1` at `--effort max` through `claude -p`. Token counts come from the final `result` event's `modelUsage` (cumulative over `--resume`). Repricing applies the [PZ] rates to the same token counts.

| Run | Wall time | Output tok (of which thinking) | Cache read tok | Cache write tok | Fable 5.1 cost | Same tokens at Fable 5.2 |
|---|---|---|---|---|---|---|
| v0.3 doom-loop classifier, A + A.c1 + B (one resumed session; `experiments/v0_3/logs/v03-phase-b.stream.jsonl`) | 77.6 + 94.1 + 58.8 min | 905,992 (768,515) | 15,759,605 | 767,913 (1h) | US$64.62 (A 42.73 + B 21.89; README.md "Cost") | US$51.69 |
| v0.5 Phase C (`experiments/v0_5/phase_c/logs/v05c.stream.jsonl`) | 66.4 min | 298,698 (215,026) | 14,057,054 | 3,049,599 (5m) | US$56.58 (README.md "Cost") | US$45.26 |
| Clouds r1 + c1 (`clouds/runs/r1/logs/r1.c1.stream.jsonl`) | 98.4 + 12.9 min | 452,957 (336,402) | 47,359,501 | 1,562,613 (1h) | US$65.77 tokens + US$10.00 charged for the aborted first attempt = US$75.77 (`clouds/runs/r1/logs/PRIOR_SPEND.json`, `r1.audit.json`) | US$52.62 tokens (US$60.62 with the US$10 charge scaled) |

Note on the brief: the US$64.62 figure is the v0.3 classifier (A + B), not v0.5 C. The README's v0.5 row is r1 US$16.70 + B US$40.75 + C US$56.58.

Where the money went on Fable 5.1: v0.3 was output-heavy (US$45.30 of US$64.62 was output); v0.5 C and clouds were cache-write-heavy (US$38.12 of US$56.58 and US$31.25 of US$65.77). Clouds' last call read about 833K tokens of cached context (`iterations[0].cache_read_input_tokens` = 832,811), close to the ~967K auto-compact point [MC].

## 8. Implications for the launcher and cost cap

**Launcher**

1. **Upgrade Claude Code first.** Fable 5.2 needs v2.1.321+ [MC]; this machine has 2.1.311 and its binary does not know `claude-fable-5-2`. The launcher should refuse unless `claude --version` is at least 2.1.321. Without the upgrade, Claude Code would treat the id as unrecognised and assume a 200K window [MC].
2. **Pass the full id**, `--model claude-fable-5-2`, not the `fable` alias; the alias resolves differently in gateway sessions [MC].
3. **Pass `--effort` explicitly and scrub `CLAUDE_CODE_EFFORT_LEVEL`**, because the env var outranks the flag and Claude Code's own Fable 5.2 default is `medium` [MC]. The clouds launcher already scrubs it (`clouds/scripts/launch.sh` line 211). Record the effort actually used from the transcript.
4. **Choose between `max` and `xhigh` deliberately.** Anthropic's guidance puts `max` at "many more tokens than `xhigh`", often for a small gain [EF], [PR]. HypaNet's earlier runs used `max` on Fable 5.1, and the user's request names `--effort max`. Keeping `max` keeps the comparison clean; the cost section below assumes `max`.
5. **Turn content-based fallback off with the documented switch.** Add `CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK=1` and/or `"switchModelsOnFlag": false` in the kit settings, in addition to the undocumented `CLAUDE_CODE_NO_MODEL_FALLBACK=1` [ENV], [MC]. Keep `--fallback-model` unset. A switch to Opus 4.8 or Opus 5 would drop Fable 5.2's thinking and make the run no longer a Fable-only authoring [WN].
6. **Detect refusals.** Have the audit count `stop_reason: "refusal"` and the `stop_details.category`, and flag any model other than `claude-fable-5-2` in `modelUsage`. `frontier_llm` names benign ML work as a possible trigger and is billed before output [RF]. Hand-writing classifier weights is ML work; Fable 5.1 never refused in six HypaNet transcripts, but Fable 5.2 has not been tried.
7. **Do not ask for reasoning in the response text.** Prompts that ask the model to reproduce its thinking can trigger `reasoning_extraction` (billed) [RF], [PR]. Asking Fable to write a design note explaining its weight choices is a different request; keep the wording about files and decisions, not about "showing your thinking".
8. **Keep the Agent tool disallowed** (as `clouds/scripts/launch.sh` does). Fable 5.2 starts self-review subagents at `xhigh`/`max` [MG]; subagent spend counts toward the cap [CLI].
9. **Drop budget language from the prompt.** The clouds prompt says "The budget for this session is large" (`clouds/kit/prompt.md` line 67). [PR] reports that a large budget in the prompt raised Fable 5.2's spend. Keep the "no hurry, think it through" sense without mentioning the budget, and keep the cap in the launcher only.
10. **Review carried-over "never" rules** in the kit prompt for scope and reason, and name safe retries as needing no approval [PR]. Add the [PR] "Finishing work" block only if a smoke run shows early stops; at `max` they are less likely [PR], and the block raised output about 30% at `medium`.
11. **Prefix check needs no work** while Claude Code builds the requests [WN], [MG]. `--resume` continuations keep the recorded system prompt (`--system-prompt-snapshot` defaults to `on`, per `claude --help`). Do not hand-edit transcripts between continues.
12. **Budget flag on continues.** [CLI] now states that restored totals do not count toward `--max-budget-usd` on `--resume`. That answers the open question in `clouds/scripts/launch.sh` (lines 26–31): passing only the remaining cap on a continue is correct and does not stop early.
13. **Thinking visibility for the integrity review.** In the clouds r1.c1 transcript, 5 of 18 thinking blocks had text. Fable 5.2's narration between tool calls is also in `thinking` blocks, empty at `display: "omitted"` [PR]. If the review needs readable thinking, test `showThinkingSummaries: true` in the kit settings in a smoke run; [MC] documents it for interactive sessions only.
14. **Rate limit.** Fable 5.2 has its own, lower-tier limit [MG]. A rate-limit error does not trigger a model switch in Claude Code [MC]; the existing "continue K" path covers it.

**Cost cap**

- Same tokens, new price: repricing the three Fable 5.1 runs gives US$45.26 (v0.5 C), US$51.69 (v0.3 A + B) and US$52.62 (clouds tokens), or US$60.62 counting the clouds abort charge (section 7). This is exactly 0.8x, because every price category falls by the same factor [PZ].
- Token use is the unknown. The docs say Fable 5.2 thinks more readily and that `max` uses many more tokens than `xhigh` [WN], [PR], but they give no Fable 5.2 `max` against Fable 5.1 `max` comparison. HypaNet has no Fable 5.2 run yet. The slop task (UI copy, small model, deterministic lint rule) is new, and its size is not yet known.
- **Reasoned range for one authoring run at `--effort max`:**
  - Expected: **US$45–80.** This is the 0.8x repriced band (US$45–61), with room for up to about 1.3x the Fable 5.1 token counts.
  - Plausible high: **US$100–130.** This covers about 2x the tokens of the dearest past run (clouds, US$65.77 in tokens at Fable 5.1 prices). That could happen if Fable 5.2 at `max` thinks much more, or if the session needs a continue that re-writes the whole cache. A 1M-token 1-hour cache write costs US$16 at Fable 5.2 prices [PZ].
- **Overshoot per response:** at most about US$21. One response with 128K output costs US$5.12 [OV], [PZ]. A full re-write of a ~967K-token context at the 1-hour rate costs about US$15.47 [MC], [PZ]. A typical response costs well under US$3 (clouds averaged US$65.77 over 350 turns on Fable 5.1, about US$0.19 per turn). [LOOP] says the response that crosses the cap still completes.
- **Suggested cap:** a hard `--max-budget-usd` of about US$150–250 for the run. US$150 covers the plausible high (US$130) plus one worst-case overshoot (US$21). US$250 matches the v0.5 Phase C cap. Record the cap in the launch record before `claude` starts, as the clouds launcher does. Plan one short smoke launch first: a few turns at `max` on the real kit gives Fable 5.2's tokens per turn and refusal behaviour before the full run.
