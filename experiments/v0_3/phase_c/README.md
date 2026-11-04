# v0.3 Phase C pre-registration: trained transformers (spec v0.3 §9.5, E8)

Written **2026-11-05, before any v0.3 Phase C training run** and before any v0.3 Phase C model has seen dev labels. The only training-path code run so far is a CPU timing benchmark on random token ids (no dataset file was read), used to choose the sequence handling in §4. Nothing below may change after the first Phase C training run starts. A change needs a recorded revision in §9, and the report states it.

**Authority.** This is a decision under delegated authority (spec §0.1). It **amends E8 and §9.5 for CPU feasibility**: the spec text gives the budgets, seeds, the all-dev 3-epoch cap and the optimiser, but no epoch caps for the smaller budgets, no early-stopping schedule and no gradient clipping. Measured cost on this machine (Intel Mac CPU, torch 2.2.2, 2 threads): a no-grad forward is about 110 ms per validation session, so one full-val evaluation takes about 8 minutes; a training forward plus backward is about 2 to 3 times that. Evaluating all of val after every epoch of every run is therefore not affordable. §8 lists every amendment. The adjudicator may reverse any of them; reversing one after the first training run creates a new registered Phase C configuration.

## 1. Model and initialisation

- **Architecture and tokeniser.** Exactly the v0.3 target: `agent_loop_lab.v03.model.DoomLoopTransformer` (2,128,932 parameters, §2.2) over the frozen renderer and tokeniser. float32, CPU.
- **Why an explicit init.** The frozen `model.py` builds every layer from `_ZeroLinear` / `_ZeroEmbedding`, whose `reset_parameters` zero-fill, so `DoomLoopTransformer()` draws no random numbers and starts all zero (spec §2.2, the hand-authoring start). Zero init cannot be trained (every unit and head would stay identical), so Phase C defines its "standard random initialisation" explicitly:
  1. `torch.manual_seed(seed)`;
  2. `model = DoomLoopTransformer()` (draws nothing);
  3. for each module in `model.modules()` order (registration order: `tok_emb`, `off_emb`, then per layer `attn.q`, `attn.k`, `attn.v`, `attn.o`, `mlp.fc1`, `mlp.fc2`, then `head`):
     - every `nn.Linear` (all `_ZeroLinear` instances, `head` and `attn.o` included): `nn.Linear.reset_parameters(module)`, the PyTorch default (weight `kaiming_uniform_(a=sqrt(5))`, bias `uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))`);
     - every `nn.Embedding` (`tok_emb`, `off_emb`): `nn.init.normal_(weight, mean=0.0, std=0.02)`;
  4. every `attn.rel_bias` (the event-distance bias, including the BOS column) stays **0.0**.

  No other random draw happens before training. The draw order above is fixed by the code (`agent_loop_lab.baselines.phase_c_v03.init_model`).

## 2. Data

- **Training split: dev only.** `datasets/v0_3/dev.raw.jsonl.gz` rendered with the frozen renderer and tokenised with the frozen tokeniser. Labels from `datasets/v0_2/standard/dev.oracle.jsonl.gz`; the scored mask from the v0.2 runtime gate (`windows.gate`) over the v0.2 canonical events `datasets/v0_2/standard/dev.events.jsonl.gz`. Raw/event alignment is checked per event (`build_raw.alignment_errors`) and the renderer and tokeniser source hashes are checked against `MANIFEST_v0_3.json`, as for val. The encoding is cached in `datasets/v0_3/dev.encoded.pt` (gitignored), keyed on the renderer and tokeniser source hashes and the input file hashes.
- **Training positions.** A dev window `(session_id, endpoint)` with block position `pos` is a training position when it passes the gate **and** has a scored oracle label (PRODUCTIVE, RECOVERY, STALLED, OSCILLATING). Its target is the class index (`schema.ClassLabel`, the model's logit order) and it is read at token `eoe_index[pos]` of the session's single causal forward pass. INSUFFICIENT_* windows and gate-rejected windows contribute nothing.
- **Validation split (early stopping only):** the cached `datasets/v0_3/val.encoded.pt` of `agent_loop_lab.v03.scoring.load_val`, unchanged.
- **Test, OOD and counterfactual are never read** by the Phase C trainer. The dev builder refuses any split other than `dev`.

## 3. Budgets, seeds and epoch caps

| Budget (dev sessions) | Seeds | Epoch cap | ES evaluated every | ES evaluations (max) |
|---|---|---|---|---|
| 32 | 0, 1, 2 | 60 | 6 epochs | 10 |
| 128 | 0, 1, 2 | 30 | 3 epochs | 10 |
| 512 | 0, 1, 2 | 12 | 2 epochs | 6 |
| 2,048 | 0 | 4 | 1 epoch | 4 |
| all (12,000) | 0 | 3 | 1 epoch | 3 |

- **Budget subsets.** The dev session ids (`datasets/v0_2/splits.json`, `splits.dev`, which must equal the dev data file's ids) are sorted ascending by `sha256(f"20261105:v03-phase-c-budget:{session_id}").hexdigest()`, compared as hex strings. Budget *k* is the first *k* ids. Subsets are nested and identical for every seed.
- **Training pool.** A budget session counts towards the budget even if it has no training position (for example `under_instrumented` sessions whose windows are all gate-rejected). Such sessions are dropped from the training pool, since they carry no gradient. Each run reports its budget count, pool count, training-position count and class counts.
- 11 runs in total.

## 4. Optimisation

- **Loss.** Mean cross-entropy (unweighted) over all training positions of the batch's sessions, pooled: `sum over positions / number of positions in the batch`. This is not the mean of per-session means.
- **Optimiser.** `torch.optim.Adam(model.parameters(), lr=3e-4)` with PyTorch defaults (betas (0.9, 0.999), eps 1e-8, weight decay 0, not AMSGrad). No learning-rate schedule. No weight decay. No dropout (the architecture has none).
- **Gradient clipping.** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` (global L2 norm), once per step, after all of the batch's gradients are accumulated. This is not in §9.5; it is added because the architecture has no normalisation and its residual stream is unbounded.
- **Batches.** 8 sessions from the training pool. The pool (in budget order) is shuffled every epoch with `torch.randperm(len(pool), generator=g)`, where `g = torch.Generator().manual_seed(seed)` is created once per run. The last partial batch is kept. One batch is one optimiser step.
- **Sequence handling: per-session loop with gradient accumulation.** There is no padding. For each batch: `opt.zero_grad(set_to_none=True)`; `T` = total training positions of the batch; for each session in batch order, one causal `model(tokens, event_ord, offset)` forward over the whole session, then `F.cross_entropy(logits[eoe_positions], targets, reduction="sum") / T`, then `.backward()`; after the last session, clip (above) and `opt.step()`. The accumulated gradient equals that of the pooled mean exactly up to float32 summation order. (Benchmark on random tokens: a padded `forward_batch` of 8 mixed-length sessions took 7.6 s against 3.2 s for the loop, and padding multiplies peak memory by the batch size.)
- **Runtime.** `torch.set_num_threads(2)` and `torch.use_deterministic_algorithms(True)`. A run is bit-for-bit reproducible only at the same thread count, torch build and CPU. The thread count is recorded in every training log; the registered value is 2, and any run made at another count is reported as such.

## 5. Early stopping

- **ES set: a fixed validation subset of 500 sessions.** Sort all 4,004 val session ids ascending by `sha256(f"20261105:v03-phase-c-es:{session_id}").hexdigest()` (hex string order), **remove** the 200 Phase B disclosed sessions (computed with `agent_loop_lab.v03.scoring.disclosed_sessions` on the val rows, the Phase B README §2 rule, which uses no model output), then take the **first 500** of what remains. Filtering comes before taking, so the set always has exactly 500 sessions. Its count and the sha256 of its sorted ids are written to every training log.
- **ES metric.** Macro-F1 over the ES sessions' gate-passing, scored windows, with the `eval.metrics.metrics_from_counts` definition (`agent_loop_lab.baselines.phase_c.macro_f1`). Predictions come from the reference inference path: one causal forward per session under `torch.no_grad()` (`infer.eoe_logits`), argmax at `EOE_pos`, ties to the first index.
- **Schedule.** With cap *C*, ES is evaluated after epoch *e* when `e % ceil(C / 10) == 0` or `e == C` (§3 table). So at most 10 evaluations per run.
- **Best and stop.** A strictly higher ES macro-F1 is a new best (the earliest maximum is kept). Training stops after **3 consecutive evaluations** without a new best, or at the cap. The best state is kept. With cap 3 patience cannot trigger; with cap 4 it can only coincide with the cap.

## 6. Outputs

`scripts/v03_train_phase_c.py --budget B --seed K` (trainer: `src/agent_loop_lab/baselines/phase_c_v03.py`) writes `experiments/phase-c-trained/v0_3/<B>/seed<K>/`:

- `model.pt`: `torch.save` of the best state_dict (`DoomLoopTransformer` parameter names, float32);
- `model.sha256`: `"<sha256>  model.pt"`;
- `training_log.json`: the protocol constants, budget/pool/position/class counts, budget and ES set hashes, per-epoch train loss, steps and wall time, every ES evaluation (epoch, macro-F1, wall time), best epoch and best ES macro-F1, why it stopped, total steps and wall time, code and data hashes, and the environment (torch, numpy, Python, platform, thread count, deterministic flag).

The script refuses to overwrite an existing run unless given `--force`. `scripts/v03_run_phase_c_all.sh` runs the 11 runs of §3 sequentially, in table order, and skips runs that already have a `model.pt` and `model.sha256`. The trainer is never imported by any weight-build path (runner, packet, guards, `weight_program/`), and nothing from it reaches a weight author.

A `--smoke` mode (epoch cap 2, output only outside `experiments/phase-c-trained/`) exists for tooling tests and timing. Smoke outputs are not Phase C results.

## 7. Reporting

- **Validation.** Phase C runs are scored with the same v0.3 pipeline as Phases A and B (`scoring.network_predictions`, `scoring.score`), on val **excluding the disclosed sessions** (primary), with all-val alongside. The Phase C columns are optimistic on val, because their 500 ES sessions lie inside that population; as in v0.2 M4, the sealed numbers are the comparison that counts.
- **Sealed.** Phase C enters the joint sealed evaluation (spec E7, milestone V4) with the other v0.3 and v0.2 columns. Per budget with 3 seeds: each seed and the mean ± sample sd.
- **Ablation** (no `event_ord`, §9.5 optional): not run in this configuration.

## 8. Amendments to E8 / §9.5 (CPU feasibility)

1. **Epoch caps** 60 / 30 / 12 / 4 for budgets 32 / 128 / 512 / 2,048 (E8 caps only all-dev, at 3). They keep every run within roughly one CPU-hour, except all-dev.
2. **Early stopping on a fixed 500-session val subset** that excludes the disclosed sessions, evaluated at most 10 times per run with patience 3 evaluations, instead of all of val every epoch (about 8 minutes per full-val pass).
3. **Gradient clipping** at global norm 1.0 (not in §9.5).
4. **Explicit initialisation** (§1), because the frozen model class constructs zero parameters.
5. **Per-session gradient accumulation** instead of a padded batch (same loss, §4).
6. Sessions without training positions leave the training pool (§3).

## 9. Revisions

None.

- **Note (2026-11-05, before the first registered run):** the dev encoding cache (`load_dev()`, `datasets/v0_3/dev.encoded.pt`) was built at 04:19 by a smoke run of budget 32, seed 0, 2 epochs, written to scratch and not used for any reported result. No registered Phase C model has been trained or evaluated.
- **Note (2026-11-05, interruption):** the schedule runner was killed at 09:01 by the end of its parent session, about one minute into budget 2048 seed 0 (no output directory, no model written). The runner was restarted at 09:17. It skipped the nine completed runs by sha256 and, per §3 of this README and the runner's documented behaviour, started 2048 seed 0 again from scratch. Nothing else changed.
