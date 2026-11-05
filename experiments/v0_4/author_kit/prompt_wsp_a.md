You are the weight author for a research experiment. A small pretrained language model, Qwen3.5-0.8B, is
asked to read rendered agent-log text and name the agent's recent behaviour as one of four classes, by the
logit of one label word per class. Your task is to edit the model's weights **by hand, by reasoning**, so
that it does this better: no optimiser, no gradients, no labelled examples and no automated search. This is
phase **WSP-A**.

Everything you need is in `packet/` in this directory. Start by reading `packet/SPEC.md` (the task, the
frozen interface, the model, the log format and the deliverables) and `packet/AUTHOR_INSTRUCTIONS.md` (the
kernel API, the forward tool, the rules the harness enforces and how to run things). `packet/qwen/` holds the
model's config and tokenizer; the weights are served by the kernel.

Deliverables, all in `output/wsp-a/`:
- `build_edit.py`: `def build(orig, consts)` that rebuilds your final edit deterministically from the
  original weights;
- `edit_constants.safetensors` (optional, at most 8 MB): activation-derived vectors `build_edit.py` uses;
- `EDIT.md`: what you changed, in which tensors, the mechanism you claim, and the ledger call ids behind
  every constant vector. It must end with a section headed exactly `## Mechanisms`: one Markdown table
  with exactly the columns `| id | tensors | target class | claim |`, in that order, and one row per
  mechanism you claim (the format, an example and the rules are in `packet/AUTHOR_INSTRUCTIONS.md`,
  "`EDIT.md`"):
  - `id`: a short unique id (`M1`, `M2`, ...);
  - `tensors`: the full names of the edited tensors that implement the mechanism, comma-separated, or
    `label rows` for the four label-word rows of the embedding;
  - `target class`: exactly one of `PRODUCTIVE`, `RECOVERY`, `STALLED`, `OSCILLATING`;
  - `claim`: one sentence.

  Each row is checked on its own, by reverting exactly its tensors to the original weights. A row is
  checked only if every tensor it names is edited, it does not name the whole edit, and it shares no
  tensor with any other row;
- `ITERATIONS.md`: every iteration, logged as you go.

WSP-A rules:
- You may read weights (`load`, `load_original`), save and drop edits (`save_edit`, `save_label_rows`,
  `drop_edit`) from `./py` scripts, and run forward passes of the original or edited model **only on texts
  you write yourself**, with `./fwd`. You see no labels and no data of any kind beyond the packet.
- Forward passes are metered: 2,000,000 tokens for the whole phase and at most 50,000 per request. Every
  request is ledgered with its call id, your current edit hash and a hash of its texts.
- No automated search of any kind: no fitting weights to a loss, no gradient steps, no random restarts, no
  sweeps of candidate edits scored against an objective, no keep-the-best loops, no distillation. Every
  edit must come from a reason you state.
- Run **one `./fwd` request per Bash call**. Never write a loop, in one script or across scripts, that both
  saves edits and writes or runs forward requests. Decide each edit yourself between forward calls.
- Log every iteration in `ITERATIONS.md`: what you changed and why, the edit hash after it, and the ledger
  call ids (`fwd call N` in the tool's output) of the forward passes it rests on.
- `build_edit.py` must reproduce exactly the edit state the kernel holds when you finish. The harness
  rebuilds it twice and compares it with the kernel's journal tensor for tensor.

How to work here:
- Read only inside this directory. Write only inside `output/wsp-a/`, never inside `output/wsp-a/fwd/`
  (the forward tool writes its records there), `packet/` or `tools/`.
- Put your scripts in `output/wsp-a/scratch/` and run them with `./py <file>`, for example
  `./py output/wsp-a/scratch/inspect.py`. `./py -m py_compile <file>` checks syntax. Write forward requests
  as JSON files under `output/wsp-a/` and run them with `./fwd <file>`, for example
  `./fwd output/wsp-a/scratch/req_01.json`; read the results from `output/wsp-a/fwd/<call id>/`.
- Run exactly one command per Bash call, and only commands that start with `./py ` or `./fwd `. Compound
  commands are refused: no `;`, `&&`, `||`, pipes, redirection, backticks, `$`, quotes, globs or newlines.
  Write files with the file-writing tool, not with shell commands. Run every command in the foreground
  (`run_in_background` is refused); keep each script short enough to finish within a Bash call.
- Keep every script you run, unchanged, under `output/wsp-a/` (edit a copy instead of rewriting one you
  have run); the harness audits all of them.
- Keep printed output short; write large results to files in `output/wsp-a/scratch/` and read those.
- Write each deliverable with its own tool call as soon as its design is settled, rather than composing
  everything in a single response.
- The harness builds and checks your files, and archives the whole `output/wsp-a/` directory, scratch
  included. You do not build, hash or commit anything yourself.

When you are done, make sure the kernel's edit state is the one `build_edit.py` builds, then state clearly
that your WSP-A submission is final, and give a short summary of your edit, the mechanism you claim and what
your own forward passes showed.
