# Prompt preview: what Fable 5.2 receives (run r2)

Rendered by `uv run python kit/kit_build.py preview --run r2` from the kit files below. This is a review copy only:
the launcher reads `kit/prompt_r2.md` and `kit/prompt_continue.md`, never this file. `preview --run r2 --check`
fails when any sha256 below no longer matches; re-render then.

| File | sha256 at render time |
|---|---|
| `kit/prompt_r2.md` | `4a8d311ce38ffa41b403833c6bb1e495a6653f3b6632e82c751cad28d667147f` |
| `kit/prompt_continue.md` | `77e2e967ce41d27fc247f68ca96551d8a2dceddc121defebc752a30eb08c6e16` |
| `kit/KIT.md` | `f5c527b3e4ebe55296d64e8966da9af48daa93fce6e0f2d3bfb22bbaed6a501a` |
| `LABELLING.md` (the frozen guide) | `3b8b7a3616c1360650b9c89cf0a4dc1b80601e40cc1ffb19f54da796a898efb6` |
| `LABELLING.md` as the author receives it (`kit_build.kit_guide()`) | `131c6c30f5be2b71f57e76b04cacb21e1f9e6b5abf93cdc62f906351627e9256` |
| `kit/r1_results.md` (`resources/r1/RESULTS.md`) | `c42f8fd67d7a0c530d4793a8c292010819815b27e0c3ad21c9e563ca811ff135` |
| `codex/CODEX.md` (`resources/codex/CODEX.md`) | `9bbd4c18aa448ef9cdb975f30afd132a13be531349fa5584fdeac16601619d4e` |

## How it is delivered

- `start` (`scripts/launch.sh --run r2 start`): the whole of `kit/prompt_r2.md` is the one user message of a
  new headless session (`claude -p "$PROMPT" --model claude-fable-5-2 --effort xhigh ...`, `SPEC.md` §13; the
  shell's `$(cat ...)` drops the final newline), run in `../HypaNet-slop-author-r2/`. No other user text is sent.
  Claude Code adds its own system prompt, which is not reproduced here; `CLAUDE.md` files and auto-memory are off.
  It is a new session: Fable does not see its r1 transcript, only what r1 left in `work/`.
- `continue K` (`scripts/launch.sh --run r2 continue K`): the whole of `kit/prompt_continue.md` as the next user
  message of the same session (`--resume`), only after a launch that ended without `MODEL COMPLETE`.
- In the author directory Fable can read `PROMPT.md` (a copy of `kit/prompt_r2.md`), `KIT.md` (a copy of
  `kit/KIT.md`), `LABELLING.md` (the hash above), `examples/` (1,200 dev screens and the reference screens),
  `resources/` (the codex: `CODEX.md`, `SOURCES.md` and `lists/`; the thesaurus: `out/*.tsv`, `README.md` and
  `ATTRIBUTION.md`; `r1/RESULTS.md`, the r1 results it may see), `HANDOVER.json` (file hashes only) and `work/`,
  which starts as its r1 `work/` without `scratch/`. `KIT.md` and `LABELLING.md` are not inlined in any message.
- The cost cap exists only as the launcher's `--max-budget-usd` flag; no message or kit file Fable receives states a
  spending budget, a cost cap or a price (the word "cost" appears only in `LABELLING.md`'s consequence test, and in the codex's and thesaurus's ordinary uses of the word).

## Message 1: `start` (verbatim, 5960 characters)

~~~~text
Your task is to **revise, by reasoning, the network you wrote in round 1**: a small transformer, every weight of it
written by hand, that finds redundant copy in user interfaces (an eyebrow that restates its heading, a subtitle that
explains the button below it, a dot pill on static content, the same fact given twice on one screen). The network
runs inside a linter, on real UI source files from projects you are never shown, so it has to be right for general
reasons, not tuned to the examples here.

Nobody trains this network. You design it: its vocabulary, the lexicon you write into the embedding table, what each
head and MLP unit computes, and every weight and bias. The question behind the experiment is how weights written by
reasoning compare with weights trained by gradient descent and with hand-written rules on the same task.

## Start here

`work/` holds your round-1 network as you left it: `build.py`, `lexicon/`, `probes/`, `MODEL.md` and the artefact.
Read `resources/r1/RESULTS.md` first: it says how round 1 did on screens you were never shown, and what kinds of
mistake it made on real copy. Then read `KIT.md` (section 14 is new; its opening and sections 1, 6–9, 11 and 13
have changed), `LABELLING.md` in full, your own `work/MODEL.md`, and `resources/codex/CODEX.md`. `examples/dev/` now
holds 1,200 screens.

## What to improve, in this order

1. **Precision on real copy.** Stop flagging sentences that add a consequence, an instruction or a reason. Every
   wrong call round 1 made on real copy was a full sentence that shares a concept with a nearby title, heading or
   control and then says something new. A shared concept is not enough: weigh what the sentence says that its anchor
   does not. The codex's decision aid (its section 7) and its consequence and instruction marker lists are a place
   to start.
2. **Synonym and paraphrase coverage.** Most of the redundancy round 1 missed on real copy was paraphrase with no
   shared stem. Use the thesaurus in `resources/thesaurus/` to build far broader synonym groups and concept codes:
   aim for thousands of stems, not hundreds, in id layout 2. Keep apart what must stay apart (antonyms, negations,
   senses that only share a spelling or a stem), because merging them erases the information that makes copy worth
   keeping.
3. **DECOR and NARRATE coverage.** Use the codex's patterns and lists (genre kickers, status words, narration verbs,
   restating frames, padding phrases that make a whole node redundant) to strengthen the decorative and narrating
   cases, within `LABELLING.md`'s definitions. Filler words inside a node that otherwise says something are the rule
   layer's (R15, `KIT.md` section 2), not the network's.
4. **The fourth layer.** Round 1 left layer 4's attention unused and most of its MLP free. Consider whether a
   mechanism you need belongs there.

False alarms on copy that should stay are scored alongside misses. A change that buys recall with false alarms on
real-looking sentences is not an improvement.

## What you deliver

In `work/`, as in round 1: `build.py`, which rewrites the artefact exactly; `vocab.txt`; your `lexicon/`;
`artifact.npz` and `meta.json` (`"ids": 2` once you use id layout 2); and `MODEL.md`, describing the network as it now
is: what every part computes and why, which resources `build.py` reads and the rule by which it selects from them,
the known failures, and a revisions section that keeps round 1's revisions and adds round 2's after them, each with
its reason. `KIT.md` section 13 gives the details.

## Rules

1. No training, fitting, optimisation or automated search of any kind: no gradient descent, regression, least squares
   or sweeps. Scoring several settings and keeping the best is a search, even by hand: change a value because you have
   a reason, then look.
2. No code that reads the examples to compute any weight, vocabulary entry or synonym group. Measure with the tools,
   think, then type the number you choose into `build.py` with a comment saying why. `build.py` and your scripts may
   read `resources/`; which thesaurus and codex entries you use, and how, is your judgement, stated in `MODEL.md`, and
   is never chosen from scores, labels, the examples or the network's outputs.
3. Every vocabulary entry and synonym group is general vocabulary, from your own knowledge or from `resources/`, not a
   key for one disclosed screen. The files in `resources/` (the thesaurus and the codex lists) are the only lexical
   material you may use besides your own knowledge.
4. numpy only.
5. Don't modify or work around the tools or the `./python3` guard, and don't read anything outside this directory,
   except the file Claude Code itself names when it saves a long tool output.
6. Keep `work/MODEL.md` current, and write each file with its own tool call as soon as it is settled, so that work is
   never lost if a response is cut short.

`KIT.md` section 11 states the rules in full, including what is fine.

## How to work

There is no hurry. Take the time the problem deserves: design before you write, score and inspect often, read the
nodes you get wrong, test one mechanism at a time with probe screens of your own, and keep revising. Quality is the
goal.

This session runs unattended: nobody will answer questions or approve steps, so make each decision yourself and carry
on. Running the tools, rebuilding the artefact and rewriting your own files in `work/` never need anyone's approval.
Keep your messages short (what you decided and which files changed); the explanations belong in `work/MODEL.md`.

## Ending

When `work/build.py` rebuilds your final artefact exactly and `work/MODEL.md` is complete, end your final message with
this line, on its own:

MODEL COMPLETE

Write that line only in your last message, one that makes no tool call, and only when you are finished. Any reply
that makes no tool call ends the session, so do not stop to report progress or to ask.
~~~~

## Message on `continue K` (verbatim, 654 characters)

~~~~text
Your previous response in this session ended before you wrote MODEL COMPLETE (an API or limit error, or a reply with
no tool call). This is a purely technical continuation: nothing has been evaluated, and there is no feedback of any
kind.

Please continue the same task under exactly the same rules (`PROMPT.md`, `KIT.md`, `LABELLING.md`), in `work/`. Check
what is already in `work/` (`build.py`, `vocab.txt`, `lexicon/`, the artefact, `MODEL.md`, your probes and scripts)
and carry on from there. Write each file with its own tool call as soon as it is settled.

When you are finished, end your final message with this line, on its own:

MODEL COMPLETE
~~~~
