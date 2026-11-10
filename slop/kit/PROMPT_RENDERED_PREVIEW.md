# Prompt preview: what Fable 5.2 receives

Rendered by `uv run python kit/kit_build.py preview` from the kit files below. This is a review copy only: the
launcher reads `kit/prompt.md` and `kit/prompt_continue.md`, never this file. `preview --check` fails when any
sha256 below no longer matches; re-render then.

| File | sha256 at render time |
|---|---|
| `kit/prompt.md` | `781c154c01cf51512c228971cab9ff6da2f2d15df97756dbb38a1ab24cbc62f3` |
| `kit/prompt_continue.md` | `77e2e967ce41d27fc247f68ca96551d8a2dceddc121defebc752a30eb08c6e16` |
| `kit/KIT.md` | `2889e0899eb49dfd63528ffd06844a8c9f883c02ec9110097cb22afb7ce16108` |
| `LABELLING.md` (the frozen guide) | `3b8b7a3616c1360650b9c89cf0a4dc1b80601e40cc1ffb19f54da796a898efb6` |
| `LABELLING.md` as the author receives it (`kit_build.kit_guide()`) | `131c6c30f5be2b71f57e76b04cacb21e1f9e6b5abf93cdc62f906351627e9256` |

## How it is delivered

- `smoke` and `start`: the whole of `kit/prompt.md` is the one user message of a new headless session
  (`claude -p "$PROMPT" --model claude-fable-5-2 --effort xhigh ...`, `SPEC.md` §13; the shell's `$(cat ...)` drops
  the final newline), run in `../HypaNet-slop-smoke/` or `../HypaNet-slop-author/`. No other user text is sent.
  Claude Code adds its own system prompt, which is not reproduced here; `CLAUDE.md` files and auto-memory are off.
- `continue K`: the whole of `kit/prompt_continue.md` as the next user message of the same session (`--resume`),
  only after a launch that ended without `MODEL COMPLETE` (an API or limit error, or a reply with no tool call).
- In the author directory Fable can read `PROMPT.md` (a copy of `kit/prompt.md`), `KIT.md` (a copy of `kit/KIT.md`),
  `LABELLING.md` (the guide with its internal preamble and its cross-references to `SPEC.md` replaced; the hash
  above), `examples/` and `HANDOVER.json` (file hashes only). `KIT.md` and `LABELLING.md` are not inlined in any
  message.
- The cost cap exists only as the launcher's `--max-budget-usd` flag; no message or kit file Fable receives states a
  spending budget, a cost cap or a price (the word "cost" appears only in `LABELLING.md`'s consequence test).

## Message 1: `start` (verbatim, 3813 characters)

~~~~text
Your task is to **write, by reasoning, every weight of a small transformer that finds redundant copy in user
interfaces**: an eyebrow that restates its heading, a subtitle that explains the button below it, a dot pill on static
content, the same fact given twice on one screen. The network will run inside a linter, on real UI source files from
projects you are never shown, so it has to be right for general reasons, not tuned to the examples here.

Nobody trains this network. You design it: its vocabulary, the lexicon you write into the embedding table, what each
head and MLP unit computes, and every weight and bias. The question behind the experiment is how weights written by
reasoning compare with weights trained by gradient descent and with hand-written rules on the same task.

## Start here

Read `KIT.md` in this directory first. It gives the input format (screens of extracted UI copy), the tokeniser, the
exact model and the artefact you produce, the examples, the tools, the rules and how the network is scored. Then read
`LABELLING.md` in full: it is the definition of what counts as redundant, and the real screens you are scored on were
labelled to it. Then look at the examples in `examples/`.

The extractor, tokeniser, model code and tools are fixed. Your whole output is the network and your account of it.

## What you deliver

In `work/`: `build.py`, which rewrites the artefact exactly; `vocab.txt`; your hand-written `lexicon/` (optional);
`artifact.npz` and `meta.json`; and `MODEL.md`, explaining what every part of the network computes and why, the
revisions you made after scoring and why, and the known failures. `KIT.md` section 13 gives the details.

## Rules

1. No training, fitting, optimisation or automated search of any kind: no gradient descent, regression, least squares
   or sweeps. Scoring several settings and keeping the best is a search, even by hand: change a value because you have
   a reason, then look.
2. No code that reads the examples to compute any weight, vocabulary entry or synonym group. Measure with the tools,
   think, then type the number you choose into `build.py` with a comment saying why.
3. Every vocabulary entry and synonym group is general UI vocabulary from your own knowledge, not a key for one
   disclosed screen. No WordNet or other lexical resource.
4. numpy only.
5. Don't modify or work around the tools or the `./python3` guard, and don't read anything outside this directory,
   except the file Claude Code itself names when it saves a long tool output.
6. Keep `work/MODEL.md` current, and write each file with its own tool call as soon as it is settled, so that work is
   never lost if a response is cut short.

`KIT.md` section 11 states the rules in full, including what is fine.

## How to work

There is no hurry. Take the time the problem deserves: design before you write, score and inspect often, read the
nodes you get wrong, test one mechanism at a time with probe screens of your own, and keep revising. Quality is the
goal, and false alarms on copy that should stay are scored alongside misses.

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
