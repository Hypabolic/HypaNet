The harness has built and archived your Phase A deliverables (artefact sha256 `{A_SHA256}`). That
archived Phase A artefact is the base for this phase. `output/v03-phase-a/` is now read-only: you
may read it but not edit it.

Your task is **Phase B**, exactly as `packet/SPEC.md` §9.3 and `packet/AUTHOR_INSTRUCTIONS.md`
("Phase B") describe it. Work in `output/v03-phase-b/`.

The disclosed report:
- It is in `disclosure/DISCLOSURE.json` (sha256 `{DISCLOSURE_SHA256}`, also given in
  `disclosure/DISCLOSURE.sha256`). Both files are read-only.
- It holds 200 windows from the **validation** split, 50 per oracle class (PRODUCTIVE, RECOVERY,
  STALLED, OSCILLATING), at most one window per session. They have opaque ids `V001`…`V200`.
- Each window gives exactly four things:
  - `prefix`: the rendered log of blocks 0..t. A single line `### window start` is inserted
    just before block t−7, and it is not log text, so remove that one line before you tokenise.
  - `oracle_label`: the true label.
  - `network_class`: your archived Phase A network's class at `EOE_t`.
  - `reference_policy_class`: your archived Phase A `reference_policy` class (`null` if the call
    failed).
- Nothing else is disclosed, and nothing else will be. You get no other windows, data, labels or
  metrics, and no evaluation of your revision. If the harness rejects the build, you receive only
  its error text.

What you may do:
- Make **one** revision: a single submitted set of files in `output/v03-phase-b/`, namely
  `build_weights.py`, `reference_policy.py`, `CIRCUITS.md` and `REVISION.md`. Start from your
  Phase A files (copy them in with a script of your own) and revise them there. You may iterate
  locally before you finish. The harness builds and archives what is in the directory once, when
  you are done.
- Run forward passes of your network and calls of `reference_policy` on the 200 disclosed windows
  and on inputs you construct yourself. Do this from your own separate scripts, never from
  `build_weights.py`. Revise weights and policy by direct, reasoned edits.
- You may not use automated or numerical weight search of any kind, including loops that try
  candidate weights and keep the best. You may not use fitting, gradient methods, distillation, or
  any data other than the disclosed report.
- Every build rule in `packet/AUTHOR_INSTRUCTIONS.md` still applies to `build_weights.py`.

`REVISION.md` must record:
- what was disclosed: the file and its sha256 above;
- every change you made (to the weights, the policy or both), each with its reason and the
  disclosed window ids (`Vnnn`) that motivated it;
- anything you deliberately left unchanged, and why.

Commands and files:
- Run scripts with `./py <script>`. One command per Bash call, and only commands that start with
  `./py `. Compound commands are refused.
- Put your analysis scripts under `output/v03-phase-b/scratch/`. The harness archives the whole
  of `output/v03-phase-b/`, scratch included.
- Read only inside this working directory. No web. Never write inside `packet/`, `tools/`,
  `disclosure/` or `output/v03-phase-a/`.
- Write each deliverable with its own tool call as soon as it is settled, rather than composing
  everything in one response.

Deliver the final `reference_policy.py`, `build_weights.py`, `CIRCUITS.md` and `REVISION.md`.
Before you finish, re-read them against `packet/AUTHOR_INSTRUCTIONS.md` so that the build is
accepted. When done, state clearly that your Phase B submission is final, and print a short
summary of what changed and why, citing the disclosed window ids.
