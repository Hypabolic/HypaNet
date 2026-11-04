You are the **weight author** for a research experiment (spec v0.2, Phase A-strict).

Your working directory contains:
- `packet/` — the authoring packet. Read `packet/AUTHOR_INSTRUCTIONS.md` first, then
  `packet/SPEC_AUTHORING.md`, `packet/feature_index.json`, `packet/state_dict_template.json` and
  `packet/model.py`. These are your only inputs.
- `output/phase-a-strict/` — where you write your deliverables.
- `tools/shape_check.py` — runs your `build()` and checks the state_dict's shapes, dtype and
  finiteness and prints sparsity. It does no forward pass.
- `./py` — runs Python 3.12 (torch 2.2.2, numpy 1.26) with `packet/` on the module path and bytecode caching off.

Task: carry out **Phase A-strict** exactly as `packet/AUTHOR_INSTRUCTIONS.md` describes. Write
`output/phase-a-strict/reference_policy.py`, `output/phase-a-strict/build_weights.py` and
`output/phase-a-strict/CIRCUITS.md`.

Rules for this phase (they restate the packet; the packet wins if anything differs):
- Read only files inside this working directory. Do not search or open anything elsewhere on disk,
  and do not use the web.
- Zero execution: the only commands you may run are `./py -m py_compile <file>` and
  `./py tools/shape_check.py output/phase-a-strict`.
  No forward passes, no calls of `reference_policy`, no other scripts.
- Run each allowed command on its own, exactly as written: no `;`, `&&`, pipes, redirection or
  `echo` (compound commands are refused).
- Never write inside `packet/`.
- Design the policy and the circuits by reasoning about the spec. There is no data and there are no
  labelled examples; do not create any.

Take the time you need: this is the core of the experiment. Before you finish, re-read your three
files against `packet/AUTHOR_INSTRUCTIONS.md` ("What the harness rejects") so the build is accepted.
When done, print a short summary of your policy and circuit layout.
