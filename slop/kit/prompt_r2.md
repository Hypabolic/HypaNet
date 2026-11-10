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
