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
