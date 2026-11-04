Your Phase A-selfcheck deliverables have been built and archived by the harness (archive sha256:
`{A_SELFCHECK_SHA256}`). That archived A-selfcheck artefact is the base for this phase.
`output/phase-a-selfcheck/` (and `output/phase-a-strict/`, `output/selfcheck_scripts/`) are now
read-only: you may read them but not edit them.

Task: carry out **Phase B** exactly as `packet/AUTHOR_INSTRUCTIONS.md` describes ("Phase B (§9.4): one
diagnostic revision"), in `output/phase-b-revision/`.

The disclosed report:
- `disclosure/DISCLOSURE.json` (sha256 `{DISCLOSURE_SHA256}`, also in `disclosure/DISCLOSURE.sha256`;
  both read-only). It holds 200 windows from the **validation** split, stratified 50 per oracle
  class (PRODUCTIVE, RECOVERY, STALLED, OSCILLATING), with opaque ids `D001`…`D200`.
- For each window it gives exactly four things: the 8 × 15 feature matrix (with feature names), the
  oracle label, your archived A-selfcheck network's predicted class, and your archived A-selfcheck
  `reference_policy`'s class.
- Nothing else is disclosed, and nothing else will be: no other windows, data, labels or metrics,
  and no evaluation of your revision. (If the harness rejects the build, you receive only its error
  text, as before.)

What you may do:
- Make **one** revision: one submitted set of files in `output/phase-b-revision/`:
  `build_weights.py`, `reference_policy.py`, `CIRCUITS.md` and `REVISION.md`. Start from your
  A-selfcheck files (copy them in with a script of your own) and revise them there. You may iterate
  locally before you finish; the harness builds and archives what is in the directory when you are
  done, once.
- Run forward passes of your network and calls of `reference_policy` on the 200 disclosed windows and
  on inputs you construct yourself, from your own separate scripts, never from `build_weights.py`.
  Revise weights and policy by direct, reasoned edits.
- Not allowed: automated or numerical weight search of any kind (no loops that try candidate weights
  and keep the best), fitting, gradient methods, distillation, or any data other than the disclosed
  report.
- Every build rule in `packet/AUTHOR_INSTRUCTIONS.md` ("What the harness rejects") still applies to
  `build_weights.py`: in particular, no forward computation of the network, or any part of it,
  inside `build()`.

`REVISION.md` must record:
- what was disclosed: the file and its sha256 above;
- every change you made (weights, policy or both), each with its reason and the disclosed window ids
  (`Dnnn`) that motivated it;
- anything you deliberately left unchanged, and why.

Commands and files:
- Commands: `./py <your script>` (packet on the module path) and
  `./py tools/shape_check.py output/phase-b-revision`. `./py` is the only interpreter.
- Run each command on its own: no `;`, `&&`, pipes, redirection or `echo` (compound commands
  are refused). Write scripts to files and run them with `./py <file>`.
- Put the scripts you write for analysis under `output/phase-b-revision/` (for example in a
  `scratch/` subfolder). The harness archives the whole of `output/phase-b-revision/`, scratch
  scripts included.
- Same as before: read only inside this working directory, no web, never write inside `packet/`,
  `tools/`, `disclosure/` or the archived `output/` directories above.

Deliver the final `reference_policy.py`, `build_weights.py`, `CIRCUITS.md` and `REVISION.md`. Before
you finish, re-read them against `packet/AUTHOR_INSTRUCTIONS.md` ("What the harness rejects") so the
build is accepted. When done, print a short summary of what changed and why, citing the disclosed
window ids.
