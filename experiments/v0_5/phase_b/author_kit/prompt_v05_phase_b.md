Your task is to **improve a network that was written by hand**, using a fixed report on 200 labelled
windows, and to deliver the improvement as one revision.

## The task and the network

A small fixed transformer (`packet/model.py`, 4 layers, 8 heads, width 128) reads the rendered log of an
AI coding agent's session and, at the end of every event block, classifies the agent's recent behaviour
into one of four classes: PRODUCTIVE, RECOVERY, STALLED or OSCILLATING. `packet/SPEC.md` defines the
task, the labels, the log rendering, the tokeniser and the model. `r1/PROMPT.txt` restates all of it in
one message, together with how the network is scored and the "edits" language for writing weights.

In an earlier, separate session (called **r1** here), another session of the same model as you was given
`r1/PROMPT.txt` and wrote **every weight** of this network from zero. It had no tools: nothing was run,
it got no feedback, and it reasoned everything out in its replies. Its replies hold its design
description and its weights, written as `edits` blocks. The frozen edits reader applied every edits
block of every reply, in order, starting from all zeros. The result is r1's network,
`r1/artifact.dlw` (sha256 `{R1_SHA256}`).

This is **Phase B**. You start from r1's network, as it is. You get a fixed diagnostic report on 200
labelled windows. You may run the network and inspect it with your own scripts. You deliver **one
revision**: a set of edits applied on top of r1's weights, plus your account of the result.

## Your directory

| Path | What it is |
|---|---|
| `packet/` | The frozen task packet (read-only). `SPEC.md`, `AUTHOR_INSTRUCTIONS.md`, `model.py`, `tokenizer.py`, `renderer.py`, `GRAMMAR_SHEET.md`, `state_dict_template.json`, `PACKET_MANIFEST.json` |
| `r1/PROMPT.txt` | The full prompt r1 received |
| `r1/replies/` | r1's replies, exactly as the r1 harness wrote them. `NNN.md` is turn NNN. When Claude Code continued a reply after it was cut at the output limit, `NNN.p<K>.md` holds each part of that turn. `INDEX.json` gives the reading order and whether each part was cut off |
| `r1/artifact.dlw`, `r1/artifact.sha256` | r1's network: the weights the frozen reader built from `r1/replies/` |
| `r1/R1_MANIFEST.json` | The sha256 of every file in `r1/` |
| `disclosure/DISCLOSURE.json`, `disclosure/DISCLOSURE.sha256` | The report (below) |
| `tools/` | The `./py` launcher and its guard, and `tools/lib/`: the helper `v05_load.py` and the package `hn_v05/`. `hn_v05/` holds copies of the frozen edits reader (`edits.py`), the revision applier the harness uses to build your revision (`revision.py`), and the weight-file reader (`dlw3.py`) |
| `output/v05-phase-b/` | Your working directory. Put scratch scripts in `output/v05-phase-b/scratch/` |

Everything except `output/v05-phase-b/` is read-only.

**What still applies from the packet.** The packet was written for an earlier protocol, in which the
author delivered `build_weights.py` and `reference_policy.py`. Those two deliverables, their build rules
and interfaces, and the packet's Phase A and Phase B sections and their cost caps do **not** apply to you,
except the window-selection rule of `SPEC.md` §9.3 (see "The disclosed report"). What does apply:

- the task, the model, the log rendering, the tokeniser, windows, gate, labels and oracle rules
  (`SPEC.md` §1–§6 and the v0.2 oracle section);
- `AUTHOR_INSTRUCTIONS.md` "Running Python" (`./py`, the guard), with the module path below;
- `AUTHOR_INSTRUCTIONS.md` "Operation records" (the renderer's input);
- what `CIRCUITS.md` must cover (below).

`r1/PROMPT.txt` was addressed to r1. Its task description, its scoring section and its description of
the edits language apply to you. Its rules about having no tools and replying with edits blocks do not.

**Cost cap.** A cost cap of US$250 applies to the whole of this Phase B, including any technical
continuations and build-rejection resubmissions. When it is reached the session is stopped, and only
what is then in `output/v05-phase-b/` can be built.

## The disclosed report

- It is in `disclosure/DISCLOSURE.json` (sha256 `{DISCLOSURE_SHA256}`, also given in
  `disclosure/DISCLOSURE.sha256`).
- It holds 200 windows from the **validation** split, 50 per oracle class, at most one window per
  session. They were chosen by the rule in `packet/SPEC.md` §9.3, which uses no model output. They have
  opaque ids `V001`…`V200`.
- Each window gives exactly three things:
  - `prefix`: the rendered log of blocks 0..t. A single line `### window start` is inserted just
    before block t−7, and it is not log text, so remove that one line before you tokenise. After
    removing it, every line that does not start with a space begins a new block. `v05_load.windows()`
    does this for you.
  - `oracle_label`: the true label of the window ending at block t.
  - `network_class`: r1's network's class at `EOE_t` (the argmax of its logits at block t's EOE token).
- There is **no** `reference_policy_class`. r1 wrote no reference policy, so that field is absent.
- Nothing else is disclosed, and nothing else will be. You get no other windows, data, labels or
  metrics, and no evaluation of your revision. If the harness rejects the build, you receive only its
  error text.

## What you deliver

Put these four files in `output/v05-phase-b/`:

1. **`REVISION.edits`**: your changes to r1's weights, in the edits language of `r1/PROMPT.txt` ("HOW
   TO WRITE THE WEIGHTS"). The harness builds the revised network as **r1's weights with the statements
   of `REVISION.edits` applied in order on top**, using `tools/lib/hn_v05/revision.py` and the frozen
   reader. So:
   - the file holds statements only: one statement per line, `#` comments and blank lines. It has no
     ``` fences and no `END OF WEIGHTS` line;
   - every weight you do not touch keeps r1's value. `=` sets values, `+=` and `-=` shift r1's values,
     and `*=` scales them;
   - every statement must read and apply. The reader would skip a bad statement silently, so the
     harness **rejects** a revision that has any. Check the file with `./py -m v05_load check` before
     you finish: it applies the file exactly as the harness will and lists any problem;
   - values are plain decimal numbers, `|v| <= 1e6`, stored as float32;
   - an empty file means you leave r1's network unchanged.

   You may write `REVISION.edits` by hand, or have a script of your own write it. A script may only
   write out values you chose, or values computed from a rule you state in `REVISION.md`. Those values
   must never come from a search, fit or optimisation over the disclosed windows or anything else.
2. **`CIRCUITS.md`**: your account of the **revised** network:
   - the residual-subspace allocation (which dimensions carry what);
   - each head's role, with its query, key and value subspaces and its event-distance bias pattern;
   - the MLP units grouped by circuit;
   - which circuit drives each class logit;
   - an account of every non-zero tensor block.

   Say what you kept from r1's design and what you changed.
3. **`REVISION.md`** (requirements below).
4. **`ITERATIONS.md`**: every self-check iteration you ran. For each, record what you ran, the inputs
   (the disclosed window ids, or the logs you built), what you saw, and what you changed as a result.

You make **one** revision: a single submitted set of these files. You may iterate locally before you
finish. When you are done, the harness archives `output/v05-phase-b/` once (`scratch/` included) and
builds the revised network from `r1/artifact.dlw` and your `REVISION.edits`.

## What you may do

- **Allowed:** read r1's prompt, replies and design; load r1's network and your revised network; run
  forward passes and inspect activations (`model.activations`) on the 200 disclosed windows and on inputs
  you build yourself. Those inputs can be hand-written blocks, or `RawOp` sequences you render with
  `packet/renderer.py`. Run all of this from your own scripts with `./py`. Revise the weights by direct,
  reasoned edits.
- **Forbidden:**
  - automated or numerical weight search of any kind, including loops that try candidate weights and
    keep the best, parameter sweeps scored against an objective, and random restarts;
  - fitting of any kind (least squares, regression of weights on activations, or any closed-form fit to
    the disclosed windows);
  - gradient methods and autograd;
  - distillation;
  - any data other than the disclosed report.

The helper `tools/lib/v05_load.py` (read its docstring) does the following:

- `load_r1()` loads r1's network;
- `apply_edits_file(path)` / `apply_edits_text(text)` give a network with edits applied on top of r1's
  weights (or on top of a state dict you pass), plus a report;
- `windows()` returns the disclosed windows with their blocks recovered;
- `classify(m, blocks)` returns the class at the last block's EOE, and `eoe_logits(m, blocks)` the logits
  at every block's EOE;
- `activations(m, blocks)` returns the intermediate values.

It loads, applies what you wrote, and runs. It never chooses weights.

## Commands and files

- Run scripts with `./py <script>`, for example `./py output/v05-phase-b/scratch/look.py`. Run **one
  command per Bash call**, and only commands that start with `./py `. Compound commands are refused: no
  `;`, `&&`, `||`, pipes, redirection, backticks or `$(...)`. `./py -c` is refused. The only modules
  you may run with `-m` are `./py -m py_compile FILE` and `./py -m v05_load check [FILE]` /
  `./py -m v05_load r1`.
- The module path is `packet/` first, then `tools/lib/`, then the script's directory and its parent.
  So `import model`, `import tokenizer`, `import renderer` and `import v05_load` work in every script.
- Every script runs under the author guard, as `AUTHOR_INSTRUCTIONS.md` "Running Python" describes.
  Forward passes are allowed. Autograd, optimisers, random numbers, threads, subprocesses, network
  access, reading or listing anything outside this directory, and writing or changing any file
  (permissions included) outside `output/v05-phase-b/` are refused. So are the modules `sqlite3`,
  `dbm.ndbm`, `readline`, `tkinter` and `curses`, and `torch.save` outside `output/v05-phase-b/`.
- Read only inside this working directory. No web. Write only inside `output/v05-phase-b/`. Never write
  inside `packet/`, `tools/`, `r1/` or `disclosure/`.
- Write each deliverable with its own tool call as soon as it is settled, rather than composing
  everything in one response.

## `REVISION.md` must record

- what was disclosed to you: `disclosure/DISCLOSURE.json` and its sha256 above, and the base you revised
  (`r1/artifact.dlw`, sha256 above);
- every change you made, each with:
  - the statements in `REVISION.edits` it consists of;
  - its reason;
  - the disclosed window ids (`Vnnn`) that motivated it;
- anything you deliberately left unchanged, and why.

## When you are done

Run `./py -m v05_load check` one last time and make sure it reports that the harness would accept the
file. Re-read your four files against this message. Then state clearly that your Phase B submission is
final, and print a short summary of what changed and why, citing the disclosed window ids.
