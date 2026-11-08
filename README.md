# HypaNet

HypaNet is a set of experiments in which a frontier model writes a small neural network's weights directly, instead of training it. The weight author is Fable (`claude-fable-5-1`). It works from a written spec in an isolated session, with no optimiser, no training set and no automated search. Codex wrote two independent pieces that the author never saw: a second data generator, used for out-of-distribution (OOD) testing, and a rules-engine baseline. Each hand-written network is compared with the same architecture trained by gradient descent. One version (v0.4) instead hand-edits a pretrained language model (Qwen3.5-0.8B); it is compared with the unedited model and has no trained baseline. The benchmark task is detecting "doom loops" in coding-agent sessions. The task is a test bed; the question is how written weights compare with trained ones.

The write-up is the blog post [Building AI Models Without Gradient Descent](https://matthewgribben.com/blog/building-ai-models-without-gradient-descent). This repository holds the specs, harness, generators, hand-written networks, trained baselines, transcripts, score reports and the figure data behind that post.

## Quick start

```sh
git clone https://github.com/Hypabolic/HypaNet.git && cd HypaNet
git lfs pull        # optional: the v0.3 trained baselines' model.pt files
uv sync             # Python 3.12 environment from uv.lock
uv run python -m agent_loop_lab.v03.inspect_cli inspect model weight_program/v0_3/phase-b/artifact.dlw
```

- Hand-written networks are in [`weight_program/`](weight_program/). The v0.5 no-tools network is in [`experiments/v0_5/runs/r1/`](experiments/v0_5/runs/r1/).
- Trained baselines are in [`experiments/phase-c-trained/`](experiments/phase-c-trained/).
- A good first read is [`weight_program/v0_3/phase-b/CIRCUITS.md`](weight_program/v0_3/phase-b/CIRCUITS.md), the author's own account of what every part of its transformer does.
- Each version's [`experiments/`](experiments/) folder has a STATUS or README file with its results and history.

## The experiment

### Task

After every tool call in an agent session, the detector reads the latest 8-operation window and labels it:

| Class | Meaning (informal) |
|---|---|
| PRODUCTIVE | the agent is making progress |
| RECOVERY | a bounded retry or poll that is expected to resolve |
| STALLED | repeated work with no progress |
| OSCILLATING | undoing and redoing the same change (A/B/A/B) |

Labels come from a latent-state oracle with pre-registered numeric rules (`agent_doom_loop_detector_spec_v0_2.md` §6, [`oracle_config.v0_2.yaml`](oracle_config.v0_2.yaml)). The oracle sees the simulator's hidden state, not the log. Priority is OSCILLATING > STALLED > RECOVERY > PRODUCTIVE. Windows the oracle marks as ambiguous or under-instrumented are not scored.

**Metric.** Macro-F1 over the four classes, with a 95% session-bootstrap interval (10,000 resamples, seed 20261104). The false-alarm rate (FA) is the share of healthy (PRODUCTIVE or RECOVERY) windows called STALLED or OSCILLATING.

### Data

All data is synthetic.

- **Standard generator** (`src/agent_loop_lab/synth/`). A seeded, stateful simulator of a repository, its faults, tests and external services, with 12 scenario families. Several families are deliberate look-alikes, for example doomed retries next to retries that resolve. It produced 20,000 sessions, split 60/20/20 per family: dev 12,000 (there is no split called "train"; dev is the training pool), val 4,004 and sealed test 3,996. There is also an 1,800-session counterfactual set of paired sessions that branch on one latent variable.
- **OOD generator** (`src/agent_loop_lab/synth_ood/`). Codex wrote it in an isolated directory without seeing the standard generator's scenario code. It has 10 different families and uses the same labeller. The sealed OOD set is 2,000 sessions. "OOD" here means an independently written generator, not real transcripts.

The hashes of the sealed test and OOD sets were committed (commit `51c9fa8`) before any network existed.

### Roles and isolation

From [`experiments/roles.yaml`](experiments/roles.yaml):

| Role | Who |
|---|---|
| Weight author | Fable (`claude-fable-5-1`); `claude-opus-5-5` was the registered fallback up to v0.4 |
| Harness, standard simulator, evaluators | Claude Code (Opus 5.5) |
| OOD generator and rules baseline | Codex (codex-cli 0.166.0), each in an isolated session |
| Adjudicator | Matthew Gribben |

The weight author ran in a fresh headless Claude Code session inside a directory outside the repo. It could see only a hashed authoring packet: the spec with the sealed sections removed, the model definition, and (from v0.3) the tokeniser and renderer. It had no repo, web, subagents, generators, datasets or evaluation output. A Python guard blocked autograd, optimisers, random number generation, subprocesses, network access and reads outside its directory. Transcripts were audited after every session, and the audit findings and adjudications are committed next to the logs.

### Phases

| Phase | What the author gets |
|---|---|
| A | Zero labels. Only the packet. It may run forward passes on inputs it writes itself (v0.2 also ran an "A-strict" variant with no execution at all). |
| B | One revision after seeing 200 labelled val windows, 50 per class, chosen by a pre-registered hash rule. Each window shows the input, the oracle label and the network's own answer. In v0.2 and v0.3 it also shows the class from the author's own reference policy. Nothing else is shown (no session id, family or aggregate score). |
| C | Overloaded name. In v0.2 and v0.3, "Phase C" is the trained baselines. In v0.5, Phase C is a third Fable round with 200 extra windows from the OOD generator. |

The first valid build of each phase is the result. Each archive is committed before any scoring.

### Versions

| Version | Network | What changed and why |
|---|---|---|
| v0.2 (M0–M4) | MLP 120→24→12→4, 3,256 parameters, over 15 hand-engineered features per event | First test of the hypothesis that an LLM can construct a small network's weights from a spec. A-strict and A-selfcheck produced the same circuit. B revised it after the 200-window disclosure. Trained baseline: the same MLP with Adam, 5 data budgets × 3 seeds. |
| v0.3 | Causal transformer, 4 layers, d=128, 8 heads, no normalisation, 2,128,932 parameters, all starting at zero, reading rendered raw log text | Removes the hand-engineered features: the author must build tokenisation-level circuits itself. Phase A set 121,126 non-zero weights, Phase B 156,501. Trained baseline: 11 runs of the same transformer (budgets 32/128/512 × 3 seeds, 2048 and all × 1 seed). The largest, C-all, trained on 11,864 dev sessions = 223,930 labelled positions. |
| v0.4 | Qwen3.5-0.8B (752,393,024 text-model parameters, fp32 on Apple MPS), read through a frozen prompt and 4 label-word logits | Asks whether hand-editing a pretrained model does better than writing a small one. WSP-A (0 labels) changed 10 tensors, 10,240 entries, each a rank-1 change. WSP-B (200 windows) changed 7 tensors, 8,192 entries. By decision, no training baselines and no sealed scoring: val only, with a 1,000-session quick look. A pre-registered WSP-F arm was never run. |
| v0.5 r1 | The v0.3 transformer | Follows the premise of MIT CSAIL's "[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)" no-tools experiment: Fable writes every weight with no tools, as statements in an `edits` language, from one 73 KB prompt and three "Continue." replies. 66,014 non-zero weights. |
| v0.5 Phase B | The v0.3 transformer | A new session, with tools and the same 200 val windows, revises r1: 1,262 statements, 2,813 entries changed. |
| v0.5 Phase C | The v0.3 transformer | A new session revises B with the 200 val windows plus 200 windows from 6 of the 10 OOD families: 430 statements, 999 entries changed. Scored once on fresh held-out sets. |

## Results

All numbers are macro-F1 unless marked. Each column names its split. Intervals and per-class figures are in the reports under [`artifacts/reports/`](artifacts/reports/) and in the STATUS files.

### Written and trained networks on the sealed sets

Sealed test is the standard generator; sealed OOD is Codex's generator. Each sealed set was opened once per version (v0.2, v0.3, v0.5).

| Network | Labels used | Sealed test | Sealed OOD | FA (test / OOD) |
|---|---|---|---|---|
| **v0.2 MLP** | | | | |
| Written, Phase A | 0 | 0.667 | 0.810 | 2.5% / 1.3% |
| Written, Phase B | 200 windows | 0.936 | 0.870 | 3.1% / 8.5% |
| Trained C-32 (3 seeds) | 563 windows | 0.618 ± 0.130 | 0.517 ± 0.182 | |
| Trained C-128 (3 seeds) | 2,207 windows | 0.932 ± 0.006 | 0.644 ± 0.026 | |
| Trained C-512 (3 seeds) | 9,216 windows | 0.966 ± 0.003 | 0.727 ± 0.026 | |
| Trained C-2048 (3 seeds) | 38,120 windows | 0.985 ± 0.000 | 0.768 ± 0.034 | |
| Trained C-all (3 seeds) | 223,930 windows | 0.992 ± 0.001 | 0.843 ± 0.023 | 0.8% / 11.6% |
| **v0.3 transformer** | | | | |
| Written, Phase A | 0 | 0.744 | 0.815 | 1.9% / 4.0% |
| Written, Phase B | 200 windows | 0.957 | 0.908 | 2.5% / 6.8% |
| Trained C-32 (3 seeds) | 590 positions | 0.347 ± 0.034 | 0.241 ± 0.055 | |
| Trained C-128 (3 seeds) | 2,216 positions | 0.450 ± 0.035 | 0.288 ± 0.035 | |
| Trained C-512 (3 seeds) | 9,114 positions | 0.707 ± 0.008 | 0.305 ± 0.034 | |
| Trained C-2048 (1 seed) | 37,285 positions | 0.755 | 0.400 | 4.4% / 50.4% |
| Trained C-all (1 seed) | 223,930 positions | 0.923 | 0.488 | 6.1% / 34.5% |
| **v0.5 (v0.3 transformer)** | | | | |
| Written, r1 (no tools) | 0 | 0.326 | 0.292 | 95.7% / 81.5% |
| Written, Phase B | 200 windows | 0.959 | 0.765 | 3.2% / 31.4% |
| **Reference baselines** | | | | |
| Rules engine (Codex) | 0 | 0.417 | 0.415 | 1.1% / 2.4% |
| Constant prior (always PRODUCTIVE) | dev counts | 0.136 | 0.181 | |

Sources: [`experiments/v0_3/STATUS.md`](experiments/v0_3/STATUS.md), [`experiments/v0_5/STATUS.md`](experiments/v0_5/STATUS.md), [`artifacts/reports/m4_sealed_v02.json`](artifacts/reports/m4_sealed_v02.json), [`artifacts/reports/v03_sealed.json`](artifacts/reports/v03_sealed.json), [`artifacts/reports/v05_sealed.json`](artifacts/reports/v05_sealed.json). The v0.3 C-all sealed-test figure is 0.92350 (0.923 to three places); the 0.924 in `agent_doom_loop_detector_plan_v0_4.md` is a rounding slip.

Paired differences on the sealed sets (v0.5 STATUS): v0.5 B minus v0.3 B is +0.0015 [−0.0020, +0.0052] on sealed test and −0.143 [−0.147, −0.139] on sealed OOD. On sealed OOD, v0.5 B calls 32.6% of PRODUCTIVE windows STALLED, against 4.6% for v0.3 B (32.8% and 4.6% on a fresh OOD batch, seed 20261107). [`experiments/v0_5/ood_analysis/FINDINGS.md`](experiments/v0_5/ood_analysis/FINDINGS.md) attributes this to behavioural rules fitted to the standard generator and the lack of a progress veto. It does not test whether adding a veto would close the gap.

### Validation scores

These were scored on val excluding the 200 disclosed windows' sessions (3,804 sessions), except v0.2, which was scored on all of val.

| Network | Val macro-F1 |
|---|---|
| v0.2 written A / B | 0.668 / 0.936 |
| v0.3 written A / B | 0.750 / 0.954 |
| v0.3 trained C-all | 0.926 (optimistic: early-stopped on a 500-session val subset) |
| v0.5 r1 / B / Phase C | 0.320 / 0.957 / 0.947 |

### v0.5 Phase C on fresh held-out sets

Scored once on new sets: held-out OOD (2,000 sessions, seed 20261110) and held-out in-distribution (4,000 sessions, seed 20261111). The primary population is the 4 OOD families that Phase C never saw. Source: [`artifacts/reports/v05c_heldout.json`](artifacts/reports/v05c_heldout.json).

| Network | Held-out OOD, 4 unseen families | Held-out OOD, 6 disclosed families | Held-out OOD, all 10 | Held-out in-dist |
|---|---|---|---|---|
| v0.5 Phase C | 0.750 (FA 3.0%) | 0.957 (FA 5.0%) | 0.955 (FA 4.2%) | 0.949 (FA 2.1%) |
| v0.5 B | 0.679 (FA 19.4%) | 0.752 | 0.766 (FA 31.2%) | 0.960 (FA 3.2%) |
| v0.3 written B | 0.718 (FA 4.3%) | 0.883 | 0.906 (FA 6.9%) | 0.954 (FA 2.4%) |
| v0.3 trained C-all | 0.349 (FA 12.5%) | 0.516 | 0.494 (FA 34.7%) | 0.923 (FA 6.4%) |

Paired primaries on the unseen families: Phase C minus v0.5 B is +0.071 [+0.064, +0.079]; Phase C minus v0.3 B is +0.031 [+0.024, +0.039]. On held-out in-distribution, Phase C minus v0.5 B is −0.011 [−0.013, −0.009].

This comparison is not like for like. Only Phase C saw labels from the second generator. [`experiments/v0_5/phase_c/SOURCE_AUDIT.md`](experiments/v0_5/phase_c/SOURCE_AUDIT.md) finds no source switch (Phase C does not read the generator's surface tells), but three rules tuned to that generator and applied to every log: a fresh `read_file` blocks STALLED, an append rule triggered by head h0, and verb:argument plan keys. Both held-out sets have now been used.

### v0.4: hand edits to Qwen3.5-0.8B (val only)

| Arm | Split | Macro-F1 | FA |
|---|---|---|---|
| Qwen-0 (base model) | full val, excluding disclosed | 0.212 [0.202, 0.222] | 70.9% |
| Qwen-0 | val quick look (1,000 sessions) | 0.223 [0.202, 0.243] | 68.7% |
| WSP-A (Fable edit, 0 labels) | val quick look | 0.330 [0.309, 0.350] | 55.3% |
| WSP-B (Fable edit, 200 windows) | val quick look | 0.419 [0.397, 0.441] | 47.6% |
| Constant prior | val quick look | 0.137 | |

Paired on the quick look: WSP-A minus Qwen-0 is +0.107, WSP-B minus Qwen-0 is +0.196, and WSP-B minus WSP-A is +0.089. For scale, v0.3's written B scores 0.954 on val. WSP-A and WSP-B were never read out on full val, and no sealed set was opened for v0.4. See [`experiments/v0_4/STATUS.md`](experiments/v0_4/STATUS.md) and [`experiments/v0_4/HANDOFF.md`](experiments/v0_4/HANDOFF.md).

### Cost

Model usage at API-equivalent prices, from the session audits.

| Version | Fable cost |
|---|---|
| v0.2 | A + B US$40.05: one resumed session, whose `total_cost_usd` is cumulative ([`experiments/m3/logs/AUDIT_NOTE.md`](experiments/m3/logs/AUDIT_NOTE.md)). The per-phase figures in `experiments/v0_3/STATUS.md` add up cumulative values and are too high. |
| v0.3 | A US$42.73 + B US$21.89 = US$64.62 |
| v0.4 | WSP-A US$32.58 + WSP-B US$29.81 = US$62.39 |
| v0.5 | r1 US$16.70 + B US$40.75 + C US$56.58 = US$114.02 (summed before rounding) |

The largest trained run (v0.3 C-all) took about 3.76 hours on 2 CPU threads. The blog prices that at no more than US$26.22 at an H100 rental rate of US$6.98/h ([`blog_figures/F28.json`](blog_figures/F28.json)).

### Caveats

- All data is synthetic, and "OOD" means a second, independently written generator.
- Each phase was authored once. The confidence intervals cover session sampling only, not variation between authoring runs.
- The sealed sets were opened three times in all, once per version. Later versions were designed knowing earlier sealed results.
- The v0.4 Qwen numbers are val quick looks, except the base model's full-val score.
- The v0.2 and v0.3 author guards could be bypassed through macOS firmlink paths (`/System/Volumes/Data`). This was fixed for v0.5 B. The committed v0.2 and v0.3 transcripts contain no such path (`grep -rE '/Volumes/' experiments/m2/logs experiments/m3/logs experiments/v0_3/logs` finds nothing).
- v0.3 training and the original v0.3 sealed run used an x86_64 Python on macOS. v0.4 and v0.5 ran natively on Apple Silicon. In the v0.5 re-score of the v0.3 columns, one trained baseline (C-128 seed0, sealed OOD) predicted one window of 74,000 differently, which moved its macro-F1 by about 1e-6.

## Repository layout

```
agent_doom_loop_detector_spec_v0_{1,2,3,4}.md   specs (v0.2 is the base; later specs extend it)
agent_doom_loop_detector_plan_v0_4.md           v0.4 plan
oracle_config.v0_2.yaml                         frozen oracle parameters
pyproject.toml, uv.lock                         root environment (frozen; hashed by the registries)
src/agent_loop_lab/                             harness, generators, models, guards, scoring
tests/                                          root test suite
envs/qwen/                                      separate environment for v0.4 (Qwen, newer torch)
scripts/                                        per-version drivers (m2_, m3_, m4_, v03_, v04_, v05_, v05b_, v05c_)
docs/                                           M0/M1 contracts, engine guide, v0.4 WSP kernel contract
experiments/                                    pre-registrations, STATUS files, kits, transcripts, audits
weight_program/                                 archived hand-written networks and their notes
datasets/                                       manifests and id lists only (data files are not in git)
artifacts/reports/                              score and audit reports
blog_figures/                                   figure data for the blog post (JSON with provenance)
```

| Version | Code | Experiment records | Networks |
|---|---|---|---|
| v0.2 | `src/agent_loop_lab/` (`model.py`, `encode.py`, `windows.py`, `synth/`, `synth_ood/`, `eval/`, `baselines/phase_c.py`) | [`experiments/m2/`](experiments/m2/) (Phase A), [`experiments/m3/`](experiments/m3/) (Phase B), [`experiments/m4/`](experiments/m4/) (trained baselines, sealed run) | [`weight_program/phase-a-strict/`](weight_program/phase-a-strict/), [`phase-a-selfcheck/`](weight_program/phase-a-selfcheck/), [`phase-b-revision/`](weight_program/phase-b-revision/); trained in [`experiments/phase-c-trained/v0_2/`](experiments/phase-c-trained/v0_2/) |
| v0.3 | `src/agent_loop_lab/v03/`, `baselines/phase_c_v03.py` | [`experiments/v0_3/`](experiments/v0_3/) | [`weight_program/v0_3/phase-a/`](weight_program/v0_3/phase-a/), [`phase-b/`](weight_program/v0_3/phase-b/); trained in [`experiments/phase-c-trained/v0_3/`](experiments/phase-c-trained/v0_3/) |
| v0.4 | `src/agent_loop_lab/v04/`, [`envs/qwen/`](envs/qwen/), [`docs/V04_WSP_KERNEL.md`](docs/V04_WSP_KERNEL.md) | [`experiments/v0_4/`](experiments/v0_4/) | [`weight_program/v0_4/wsp-a/`](weight_program/v0_4/wsp-a/), [`wsp-b/`](weight_program/v0_4/wsp-b/) |
| v0.5 | `src/agent_loop_lab/v05/` | [`experiments/v0_5/`](experiments/v0_5/) | r1: [`experiments/v0_5/runs/r1/`](experiments/v0_5/runs/r1/); [`weight_program/v0_5/phase-b/`](weight_program/v0_5/phase-b/), [`phase-c/`](weight_program/v0_5/phase-c/) |
| Commissions | `src/agent_loop_lab/synth_ood/` | [`experiments/commissions/`](experiments/commissions/) (Codex OOD generator and rules baseline, with briefs and hashes) | |

Each hand-written network's archive contains `artifact.dlw` and `artifact.sha256`, a `BUILD_RECORD.json` (checks, file hashes, sparsity, packet hash), `attempts/`, and the author's own notes. For v0.2 and v0.3 these are `build_weights.py` (literal weight assignments), `reference_policy.py` (the policy the network is meant to compute) and `CIRCUITS.md` (what every unit does). For v0.5 B and C they are `REVISION.edits`, the statements applied to the previous network. `ITERATIONS.md` and `REVISION.md` record the author's working. The v0.4 archives use `inputs/build_edit.py`, `inputs/edit_constants.safetensors` and `EDIT.md`, plus `ITERATIONS.md` (WSP-A) or `REVISION.md` (WSP-B). The v0.5 r1 run keeps `apply.json`, `run.json` and the raw replies in `replies/` instead of a build record.

## Using the repository

### Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/). The root environment pins torch 2.2.2, numpy 1.26 and pyyaml.
- Git LFS, if you want the v0.3 trained `model.pt` files (`git lfs pull`).
- For v0.4 only: an Apple Silicon Mac (readouts use MPS in fp32) and the separate `envs/qwen` environment (torch 2.14.1, transformers 5.22.1). It is separate because Qwen3.5 needs a newer torch than the frozen root environment.
- For authoring runs only: the Claude Code CLI with access to `claude-fable-5-1`. This costs money (see Cost above).

### Install and test

```sh
uv sync                 # root environment from uv.lock (includes pytest)
uv run pytest           # run from the repo root
```

Known issues:

- Two tests fail when pytest's temp directory is under `/private/var` (the macOS default): `tests/test_m4_eval.py::test_sealed_script_refuses_second_run` and `tests/test_v03_runner.py::test_side_table_is_unreachable_at_run_time[absolute]`.
- `tests/test_v05b_kit.py` and `tests/test_v05c_kit.py` must be run on their own with an explicit temp directory, for example `uv run pytest tests/test_v05c_kit.py -q -p no:cacheprovider --basetemp=<scratch>/pytest_v05c`.
- Some tests skip when local data or the Qwen snapshot is missing.
- Some determinism tests (`tests/test_datasets.py`, `tests/test_v03_rawop.py`, `tests/test_v03_render.py`) regenerate a few sessions in memory from the sealed master seed 20261104, including OOD and counterfactual sessions. They print no data. To leave those sets entirely untouched, add `--ignore` for those three files.

Do not edit `pyproject.toml`, `uv.lock` or other frozen files. They are hashed into the freeze registries, and `freeze check` will fail.

### The CLI

```sh
uv run agent-loop-lab inspect window EVENTS.jsonl [--endpoint SEQ] [--artifact PATH.dlw]
uv run agent-loop-lab inspect model weight_program/phase-b-revision/artifact.dlw
uv run agent-loop-lab packet build [--spec PATH] [--model PATH] [--out DIR]
uv run agent-loop-lab packet verify DIR
uv run agent-loop-lab freeze check
```

`inspect window` shows the gate result, the 8×15 feature matrix, logits and activations. `inspect model` shows the hash, metadata, sparsity and non-zero parameters, and reads only v0.2 (DLW1) artefacts. Exit code 0 means OK, 1 means a failed verify or check, and 2 means a usage error.

### Inspect a hand-written network

Start with the author's own description, then look at the weights:

- v0.2: [`weight_program/phase-b-revision/CIRCUITS.md`](weight_program/phase-b-revision/CIRCUITS.md) and [`REVISION.md`](weight_program/phase-b-revision/REVISION.md); 295 of 3,256 weights are non-zero.
- v0.3: [`weight_program/v0_3/phase-b/CIRCUITS.md`](weight_program/v0_3/phase-b/CIRCUITS.md) and [`REVISION.md`](weight_program/v0_3/phase-b/REVISION.md).
- v0.5: [`weight_program/v0_5/phase-c/CIRCUITS.md`](weight_program/v0_5/phase-c/CIRCUITS.md) and [`REVISION.edits`](weight_program/v0_5/phase-c/REVISION.edits).
- v0.4: [`weight_program/v0_4/wsp-b/author_output/EDIT.md`](weight_program/v0_4/wsp-b/author_output/EDIT.md) and [`REVISION.md`](weight_program/v0_4/wsp-b/author_output/REVISION.md).

The v0.3 and v0.5 transformers (DLW3 format) have their own inspector:

```sh
uv run python -m agent_loop_lab.v03.inspect_cli inspect model weight_program/v0_3/phase-b/artifact.dlw
uv run python -m agent_loop_lab.v03.inspect_cli inspect log SESSION [--artifact A.dlw]
uv run python -m agent_loop_lab.v03.inspect_cli inspect heads SESSION --event T [--artifact A.dlw] [--layer L] [--top K]
```

### Check the freezes

```sh
uv run python scripts/freeze.py --check        # v0.2: experiments/registry.yaml
uv run python scripts/v03_freeze.py --check    # v0.3: experiments/registry_v0_3.yaml
uv run python scripts/v04_freeze.py --check    # v0.4 (also runs the v0.3 check)
```

### Regenerate datasets and check hashes

Data files are gitignored. Only manifests, split id lists and hashes are committed. The builds are deterministic, and gzip is written with fixed metadata, so a rebuild with the same code and seed is byte-identical.

Two warnings before you run anything:

- **The defaults recreate the sealed sets.** With the default master seed (20261104), `build_datasets.py` writes the sealed test and counterfactual files, `build_ood_dataset.py` writes the sealed OOD set (and decodes it to print label counts and write audit reports), and `build_v03_raw.py` writes raw test, counterfactual and OOD files. See [Sealed data](#sealed-data).
- **The defaults overwrite committed files.** They write to `datasets/v0_2`, `datasets/v0_3` and `artifacts/reports/`, replacing the committed manifests, `splits.json` and `m1_*` reports. Always pass `--out` (and `--reports` or `--no-reports`) pointing at a scratch directory.

For your own experiments, use a seed that this repo has not used (see the list under [Sealed data](#sealed-data)):

```sh
uv run python scripts/build_datasets.py --master-seed <new seed> --out <scratch>/v0_2 --no-reports
uv run python scripts/build_ood_dataset.py --master-seed <new seed> --out <scratch>/v0_2 --reports <scratch>/reports
```

The v0.3 raw-text build (`scripts/build_v03_raw.py --out <scratch>/v0_3 --v02 <v0.2 dir>`) re-runs the v0.2 builds with the published seeds and exits 1 unless they match the committed v0.2 manifests byte for byte. `scripts/v04_export_tasks.py` exports the dev and val tasks for `envs/qwen` from the v0.3 raw files.

To compare your local dev and val files with the committed manifests (sealed files are skipped; this hashes bytes and decodes nothing):

```sh
python3 - <<'EOF'
import hashlib, json, pathlib
for man in ["datasets/v0_2/MANIFEST.json", "datasets/v0_3/MANIFEST_v0_3.json"]:
    m = pathlib.Path(man)
    for rel, entry in json.loads(m.read_text())["files"].items():
        if not any(s in rel for s in ("dev.", "val.")):
            continue
        want = entry if isinstance(entry, str) else entry["sha256"]
        p = m.parent / rel
        got = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else "missing"
        print("OK  " if got == want else "DIFF", man, rel)
EOF
```

### Rebuild and re-check artefacts

- **v0.2, v0.3 and v0.5 networks.** The `.dlw` files are committed. Check them against `artifact.sha256` and `BUILD_RECORD.json`. Each build was run by the harness from the author's isolated directory, under a guard, and rebuilt in a fresh process to confirm byte-identical output. A rebuild needs the author directory, which is not in the repo. Archives are immutable, so build into a new directory and compare hashes, for example (v0.2):
  ```sh
  uv run python scripts/run_build.py phase-a-strict --author-dir ../HypaNet-m2-author/output/phase-a-strict --packet-dir ../HypaNet-m2-author/packet --out-dir /tmp/rebuild-phase-a-strict
  # then compare /tmp/rebuild-phase-a-strict/artifact.sha256 with weight_program/phase-a-strict/artifact.sha256
  ```
- **v0.4 Qwen edits.** `edit.safetensors` is gitignored. Committed are `inputs/build_edit.py`, `inputs/edit_constants.safetensors` and `edit_diff.safetensors` (the exact changed indices and values). The documented rebuild is `uv run python scripts/v04_run_build.py wsp-a --archive /tmp/rebuild-wsp-a`, followed by checking the sha256 against `BUILD_RECORD.json`. It needs the local author directory and the gitignored server journals, so from a fresh clone there is no scripted rebuild yet.
- **v0.5 no-tools prompt.** Check that `experiments/v0_5/PROMPT.txt` rebuilds byte for byte:
  ```sh
  uv run python scripts/v05_build_prompt.py --check
  ```
- **Trained baselines.** The trained models are committed (v0.3 `model.pt` through Git LFS), so the training scripts refuse to overwrite them, and `scripts/v03_run_phase_c_all.sh` only verifies or skips finished runs. To retrain a v0.3 run without touching the committed files: `uv run python scripts/v03_train_phase_c.py --budget B --seed K --out /tmp/phase-c-B-K` (add `--smoke` for a 2-epoch check). `scripts/m4_train_phase_c.py` (v0.2) has no output option, and `--force` overwrites the committed baselines. Both need the local dev and val data. The v0.3 C-all run took about 3.8 hours on 2 CPU threads.

### The Qwen environment (v0.4)

```sh
cd envs/qwen
uv sync
uv run pytest
```

The model is loaded offline from the local Hugging Face cache, at revision `2fc06364715b967f1860aea9cf38778875588b17`, with every file's sha256 checked. The repo does not download it; you must put it in the cache yourself. Readouts also need the val task export (`scripts/v04_export_tasks.py`, which needs the regenerated v0.3 data). `--edit` needs the archive's `edit.safetensors`, which is not in git, so from a fresh clone only the base model (Qwen-0) can be read out.

```sh
# from envs/qwen; write --out outside envs/qwen
uv run python -m hypaqwen.run_readout --split val --device mps --dtype fp32 [--edit ../../weight_program/v0_4/wsp-a] [--sessions ../../datasets/v0_4/VAL_QUICK.txt] --out /abs/path/OUT.npz
# from the repo root
uv run python scripts/v04_parallel_readout.py --out /abs/path/COLUMN.npz [--workers 3] [--sessions FILE] [--edit ARCHIVE]
uv run python scripts/v04_score_validation.py --arm qwen0 --column "Qwen-0=/abs/path/COLUMN.npz" --report-dir /tmp/v04-scores
```

Any new file under `envs/qwen` breaks `scripts/v04_freeze.py --check`, and scoring refuses until that check passes. Without `--report-dir`, scoring targets the committed official report and refuses unless given `--force`.

A full-val readout is about 7.9 million tokens. At the measured rates that is about 5 hours in one process on an M4 Pro, or about 2.5–3 hours with 3 workers. These are estimates: the one completed full-val run (Qwen-0) resumed from a 1,700-session checkpoint. Scoring runs in the root environment with the unchanged v0.3 metrics. The full-val commands for WSP-A and WSP-B, which were never run, are in [`experiments/v0_4/HANDOFF.md`](experiments/v0_4/HANDOFF.md).

### Regenerate the blog figures

```sh
uv run python scripts/blog/figures/f28.py         # one figure
uv run python scripts/blog/export_figures.py      # all of blog_figures/*.json
```

Most figures read only committed reports and can be rebuilt on their own. `export_figures.py` builds every figure before writing any, so it fails unless the inputs for F1 and F9 (local dev and val data matching the manifests) and F19b (the OOD-analysis rows) are present. Regenerate the rows with `uv run python experiments/v0_5/ood_analysis/analyse.py --per-family 100` and `uv run python experiments/v0_5/ood_analysis/analyse.py --indist --per-family 40`. F19 uses them only as a cross-check. Every JSON file records its sources, their hashes and the git commit in `_provenance`.

### How an authoring run works

This section explains the records. Do not rerun these commands in place: the prepare script deletes and rewrites `experiments/v0_3/packet_handed_over/` and `HANDOVER.json`, and the build writes into the committed archive. For a new experiment, copy the kit and scripts to a new version prefix with their own output paths. Runs need the Claude Code CLI and cost real money. The kits also hard-code sibling directories (`../HypaNet-*-author`) and absolute deny paths for the original machine. Using v0.3 as the example:

1. **Freeze.** `uv run python scripts/v03_freeze.py --check`. The registry hashes the spec, code, kit and packet.
2. **Prepare.** `scripts/v03_prepare_author_dir.sh` creates a fresh author directory outside the repo with the packet, a guarded `./py` wrapper and a private venv. It writes `experiments/v0_3/HANDOVER.json`, which is committed before launch. Kit tests with throwaway sessions check the allowed and refused tools ([`experiments/v0_3/KIT_TESTS.md`](experiments/v0_3/KIT_TESTS.md)).
3. **Launch.** `scripts/v03_launch.sh phase-a` runs `claude -p` with `--model claude-fable-5-1 --effort max`, restricted settings, no MCP and no web, agent or skill tools. Use `continue K` after API errors and `resubmit N ERROR_FILE` to pass back a build error, at most 3 times. The launcher refuses while pytest's temp directory exists, so remove it first (`rm -rf /private/var/folders/*/*/T/pytest-of-$USER` on macOS) and do not run the test suite while a session runs.
4. **Audit.** The launcher runs `scripts/v03_audit_session.py` and `scripts/v03_cost.py`. Transcripts go to `experiments/v0_3/logs/`, with an `.audit.json` next to each.
5. **Build and archive.** `uv run python scripts/v03_run_build.py v03-phase-a --author-dir ../HypaNet-v03-author/output/v03-phase-a --packet-dir ../HypaNet-v03-author/packet --out-dir weight_program/v0_3/phase-a`. Commit the archive before scoring.
6. **Score on val.** `uv run python scripts/v03_score_validation.py v03-phase-a`.
7. **Phase B.** `uv run python scripts/v03_build_disclosure.py`, then `scripts/v03_prepare_phase_b.sh`, then `scripts/v03_launch.sh phase-b`, then build and score as above.
8. **Sealed run, once.** `scripts/v03_sealed_eval.py --sealed --confirm-one-time` scored every column on the sealed sets. It is spent, and the script refuses to run again. Its `--dry-run-val` mode also scores the counterfactual set, so do not run it.

v0.2 (`m2_`, `m3_`, `m4_` scripts), v0.4 (`v04_`) and v0.5 (`v05_`, `v05b_`, `v05c_`) follow the same pattern. Their READMEs and STATUS files under `experiments/` give the exact commands. The v0.5 r1 run is different: `scripts/v05_launch.sh r1` runs a tool-free conversation and applies the `edits` blocks from the replies.

## Sealed data

These sets are spent or reserved. Do not rescore, open or inspect any of them. The sealed test and OOD sets were opened once per version (three times in all). The held-out sets were used once, and the v0.4 fresh test set was never opened.

| Set | Seed | Used by |
|---|---|---|
| Sealed test (3,996 sessions) | 20261104 | v0.2, v0.3 and v0.5 sealed runs |
| Sealed OOD (2,000 sessions) | 20261104 (OOD namespace) | v0.2, v0.3 and v0.5 sealed runs |
| Counterfactual (1,800 sessions) | 20261104 | scored with val in v0.2, v0.3 and v0.5; held back in v0.4 |
| v0.5 Phase C held-out OOD | 20261110 | one held-out scoring |
| v0.5 Phase C held-out in-dist | 20261111 | one held-out scoring |
| v0.4 fresh sealed test | 20261106 | never opened |

The one-time evaluation scripts write an attempt marker (`artifacts/reports/*.started.json`) before opening any sealed file, and refuse to run again if the marker or report exists. The data files are not in git; only their manifests and hashes are. Rebuilding with the default scripts puts them on your disk, and keeping them sealed then depends on you not opening them. Seeds 20261107 and 20261108 were used for the post hoc OOD diagnosis and the Phase C source audit, 20261109 for the Phase C disclosure, and 20261201 and 20261202 for the source audit's re-checks. For new data, use a seed not listed here.

## Citation

```bibtex
@misc{gribben2026withoutgradientdescent,
  author       = {Gribben, Matthew},
  title        = {Building AI Models Without Gradient Descent},
  year         = {2026},
  month        = nov,
  howpublished = {\url{https://matthewgribben.com/blog/building-ai-models-without-gradient-descent}},
  note         = {Code, weights and transcripts: \url{https://github.com/Hypabolic/HypaNet}}
}
```

## Licence

MIT. See [`LICENSE`](LICENSE).
