You are the weight author for a research experiment. Your task is to write, by hand and with no
optimiser and no labelled examples, the parameters of a small fixed transformer that reads rendered
agent-log text and classifies the agent's recent behaviour into four classes. This is **Phase A**.

Everything you need is in `packet/` in this directory. Start by reading `packet/SPEC.md` (the task,
the model, the log grammar and the deliverables) and `packet/AUTHOR_INSTRUCTIONS.md` (the exact
interfaces, the rules the harness enforces and how to run things). `packet/GRAMMAR_SHEET.md` shows
the log grammar with invented examples. The packet's Python modules (`model.py`, `tokenizer.py`,
`renderer.py`) are the frozen code the harness uses; `AUTHOR_INSTRUCTIONS.md` lists the fields of the
operation records the renderer reads.

Deliverables, all in `output/v03-phase-a/`:
- `reference_policy.py`
- `build_weights.py`
- `CIRCUITS.md`
- `ITERATIONS.md`

Phase A rules (spec §9.2):
- You may run forward passes of your network and calls of your `reference_policy` on logs you
  write yourself: hand-written block text, or `RawOp` sequences you build and render with
  `packet/renderer.py`. You may compare the two and revise weights and policy by direct, reasoned
  edits.
- You may not use any other data, and you may not use automated search of any kind: no fitting, no
  gradient steps, no random restarts, no parameter sweeps scored against an objective, and no
  distilling anything into the weights, your own `reference_policy` included.
- Log every self-check iteration in `ITERATIONS.md`: a summary of the diff, the reason and the
  inputs used.

How to work here:
- Read only inside this directory. Write only inside `output/v03-phase-a/`. Never write inside
  `packet/` or `tools/`.
- Put your own scripts in `output/v03-phase-a/scratch/` and run them with `./py <file>`, for
  example `./py output/v03-phase-a/scratch/check.py`. `./py -m py_compile <file>` checks syntax.
- Run exactly one command per Bash call, and only commands that start with `./py `. Compound
  commands are refused: no `;`, `&&`, `||`, pipes, redirection, backticks, `$(...)` or `echo`.
  Write files with the file-writing tool, not with shell commands.
- Write each deliverable with its own tool call as soon as its design is settled, rather than
  composing everything in a single response.
- The harness builds and checks your files, and archives the whole `output/v03-phase-a/`
  directory, scratch scripts included. You do not build, hash or commit anything yourself.

When you are done, state clearly that your Phase A submission is final, and give a short summary
of your policy, your circuit layout and how well the network agrees with your reference policy on
your own inputs.
