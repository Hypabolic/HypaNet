The harness has built and archived your WSP-A submission (build record sha256 `{A_SHA256}`, edit hash
`{A_EDIT_HASH}`). That archived edit is the base for this phase: the kernel now holds exactly it, so
`edit_hash()` returns `{A_EDIT_HASH}` before you change anything. `output/wsp-a/` is now read-only: you
may read it but not edit it.

Your task is phase **WSP-B**: **one revision** of your WSP-A edit, after seeing 200 labelled windows once.
Work in `output/wsp-b/`. Everything in `packet/SPEC.md` and `packet/AUTHOR_INSTRUCTIONS.md` still applies
(kernel API, edit scope, method, `build_edit.py` rules), with `output/wsp-b/` in place of `output/wsp-a/`
and the additions below. The forward budget is new for this phase: 2,000,000 tokens, at most 50,000 per
request. A new cost cap applies to the whole of WSP-B, continuations and resubmissions included.

The disclosed windows:
- They are in `disclosure/DISCLOSURE.json` (sha256 `{DISCLOSURE_SHA256}`, also given in
  `disclosure/DISCLOSURE.sha256`). Both files are read-only.
- 200 windows from the **validation** split, 50 per oracle class (PRODUCTIVE, RECOVERY, STALLED,
  OSCILLATING), at most one window per session, with opaque ids `V001`…`V200`.
- Each window gives exactly three things:
  - `prefix`: the rendered log of blocks 0..t. A single line `### window start` stands just before block
    t−7; it is not log text, so remove that one line before you use the text. After removing it, every line
    that does not start with a space begins a new block, and block t ends the text.
  - `oracle_label`: the true label of the window ending at block t.
  - `network_class`: the class your archived WSP-A edit gives the window (the argmax of the four label-word
    logits after block t, with the frozen interface).
- Nothing else is disclosed, and nothing else will be. You get no other windows, data, labels or scores,
  and no evaluation of your revision. If the harness rejects the build, you receive only its error text.

Forward passes on the disclosed windows:
- Run them only with `./fwd` and a request file of exactly this form (ids from `DISCLOSURE.json`):
  `{"disclosed_windows": ["V001", "V002"], "model": "edited"}` (`"model"` is `"edited"`, `"original"` or
  `"both"`). The response gives, per window, the four label-word logits and their argmax **only**: no hidden
  states, MLP activations or top-k ids. Split the 200 windows over several requests to stay within 50,000
  tokens per request; every request is metered and ledgered like any other.
- At most **2 distinct edit states** can ever be run on the disclosed windows in this phase (`"original"`
  does not count): the WSP-A edit you start from, and one revision. A third edit state is refused. Every
  distinct edit state you run there counts, so run the disclosed windows only with the WSP-A edit (before
  you change anything) and with your final revision. This is what "seeing the 200 windows once" means.
- `./fwd` requests on texts you write yourself work as in WSP-A, with activations. A session whose text
  overlaps a disclosed window (the same prefix, or most of its blocks or text) is refused: the disclosed
  windows can be run only through `disclosed_windows` requests.

What you may do:
- Make **one** revision: a single submitted set of files in `output/wsp-b/`, namely `build_edit.py`,
  `edit_constants.safetensors` (optional), `EDIT.md` and `REVISION.md`. Start from your WSP-A files in
  `output/wsp-a/` (write your copies into `output/wsp-b/` with the file-writing tool or a `./py` script of
  your own) and revise them there. You may iterate before you finish; the harness builds what is in
  `output/wsp-b/` once, when you state that your submission is final.
- `build_edit.py` returns the **whole** final edit relative to the original weights, the parts of the WSP-A
  edit you keep included, and must reproduce exactly the kernel's final edit state (which starts at the
  WSP-A edit). Apply your final edit by running `build()` from a `./py` script, as in WSP-A.
- Revise by direct, reasoned edits only, as in WSP-A. Banned: fitting weights to the disclosed labels or to
  any loss, in closed form or iteratively; training a probe or classifier on the 200 windows; gradients;
  any automated or numerical search over candidate edits, including keep-the-best loops and sweeps; and
  distillation. Use the windows to see where and why your edit fails, then change it for a reason you state.

`EDIT.md` describes the final edit as a whole (the kept WSP-A parts included), as in WSP-A, and must end
with a section headed exactly `## Mechanisms`: one Markdown table with exactly the columns
`| id | tensors | target class | claim |`, in that order, and one row per mechanism you claim (format, example
and rules in `packet/AUTHOR_INSTRUCTIONS.md`, "`EDIT.md`"):
- `id`: a short unique id (`M1`, `M2`, ...);
- `tensors`: the full names of the edited tensors that implement the mechanism, comma-separated, or
  `label rows` for the four label-word rows of the embedding;
- `target class`: exactly one of `PRODUCTIVE`, `RECOVERY`, `STALLED`, `OSCILLATING`;
- `claim`: one sentence.

Each row is checked on its own, by reverting exactly its tensors to the original weights. A row is checked
only if every tensor it names is edited, it does not name the whole edit, and it shares no tensor with any
other row.

`REVISION.md` must record:
- what was disclosed: the file and its sha256 above;
- every change from the WSP-A edit (to which tensors, and how), each with its reason and the disclosed
  window ids (`Vnnn`) that motivated it;
- the ledger call ids of the forward passes each change rests on, the `disclosed_windows` requests
  included, with the edit hash each one ran under;
- anything you deliberately left unchanged, and why;
- an iteration log, kept as you go: the edit hash after each change and the call ids that tested it.

Commands and files:
- Run exactly one command per Bash call, in the foreground, and only commands that start with `./py ` or
  `./fwd `, with scripts and request files under `output/wsp-b/` (for example
  `./py output/wsp-b/scratch/inspect.py` or `./fwd output/wsp-b/scratch/disc_01.json`). Compound commands
  are refused. Read the forward results from `output/wsp-b/fwd/<call id>/`.
- Put your scripts under `output/wsp-b/scratch/` and keep every script you run, unchanged; the harness
  archives the whole of `output/wsp-b/`, scratch included.
- Read only inside this working directory. Never write inside `packet/`, `tools/`, `disclosure/`,
  `output/wsp-a/` or `output/wsp-b/fwd/`.
- Write each deliverable with its own tool call as soon as it is settled, rather than composing everything
  in one response.

Before you finish, make sure the kernel's edit state is the one `build_edit.py` builds, and re-read your
files against `packet/AUTHOR_INSTRUCTIONS.md` so that the build is accepted. When done, state clearly that
your WSP-B submission is final, and give a short summary of what changed and why, citing the disclosed
window ids.
