Your Phase A-strict deliverables have been built and archived by the harness (archive sha256:
`{A_STRICT_SHA256}`). You may not edit `output/phase-a-strict/` any more.

Task: carry out **Phase A-selfcheck** exactly as `packet/AUTHOR_INSTRUCTIONS.md` describes, in
`output/phase-a-selfcheck/`.

- Copy your A-strict deliverables into `output/phase-a-selfcheck/` and revise them there.
- You may now run forward passes of your network and calls of `reference_policy` **on inputs you
  construct yourself** (hand-written windows, enumerations of your own policy's branch conditions),
  from your own separate scripts — put them in `output/selfcheck_scripts/`, never in
  `build_weights.py`. Compare network and policy, and revise weights and policy by direct, reasoned
  edits.
- Not allowed: any data, labels or evaluation output (none exist in this directory), and automated
  weight search of any kind (no loops that try candidate weights and keep the best).
- Log every iteration in `output/phase-a-selfcheck/ITERATIONS.md`: diff summary, reason, inputs used.
- Commands: `./py <your script>` (packet on the module path) and
  `./py tools/shape_check.py output/phase-a-selfcheck`.
- Run each command on its own: no `;`, `&&`, pipes, redirection or `echo` (compound commands
  are refused). Write scripts to files and run them with `./py <file>`.
- Same as before: read only inside this working directory, no web, never write inside `packet/`.

Deliver the final `reference_policy.py`, `build_weights.py`, `CIRCUITS.md` and `ITERATIONS.md`.
When done, print a short summary of what changed and why, and how well the network now agrees with
your reference policy on your own inputs.
