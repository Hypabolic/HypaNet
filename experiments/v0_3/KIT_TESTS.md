# v0.3 author-kit tests (2026-11-05, before any weight-author launch)

These are kit tests, not Phase A attempts. They ran in `../HypaNet-v03-author` with the frozen `settings_v03_phase_a.json`. Any files they created were deleted before launch.

## 1. `./py` path

The test script did the following:
- imported `model`, `tokenizer` and `renderer` from the packet;
- rendered 40 RawOp dicts (the renderer's seeded `random.Random` is allowed under the AUTHOR_RUN guard);
- tokenised them into 1,241 tokens;
- ran a forward pass with `zero_model()` and then with non-zero weights, taking 0.10 s.

Under the same guard, reading `../HypaNet/...` and calling `random.random()` were both refused (`TrainingViolation`, rc 3).

## 2. Build budget

A fake `build_weights.py` set about 204k parameters one element at a time in Python loops, using `hashlib` sign codes for all 8,192 hashed buckets. It went through `scripts/v03_run_build.py` in 8.3 s wall time, covering both builds and the policy smoke call. The rebuild was byte-identical. That is far inside the 300 s CPU budget.

## 3. Permissions under `dontAsk`

These were throwaway `claude -p` sessions, one on `claude-haiku-5-5` and two on `claude-sonnet-5-5`.

Allowed:
- reading `packet/SPEC.md`;
- writing `output/v03-phase-a/scratch/kt.py`;
- `./py <script>`.

Refused:
- reading `../HypaNet/...`;
- reading `../HypaNet-m2-author/...`;
- editing `packet/SPEC.md`;
- `ls ..`;
- `./py x.py ; cat ../HypaNet-m2-author/...`;
- `./py x.py && head ../HypaNet/...`.

Allowed but harmless: compound commands that stay inside the directory (`./py x.py ; echo hi`, `./py x.py | cat`, `./py x.py > output/v03-phase-a/scratch/o.txt`). The audit still flags any command that does not start with `./py `.

Other observations:
- `./py -c` is not supported by the launcher; scripts must be files.
- Claude Code warns that the `Write(...)` deny rules are not matched by file permission checks. The matching `Edit(...)` deny rules cover writes, and test 3 confirmed packet edits are refused. The settings are left as frozen.

The kit sessions' transcripts are in `~/.claude/projects/-Users-matthew-Development-hypabolic-HypaNet-v03-author/`. The author cannot read them because `~/.claude/**` is denied, and auto-memory is off.
