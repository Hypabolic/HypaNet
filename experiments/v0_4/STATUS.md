# v0.4 status

_Last update: 2026-11-05 23:30. Plan: `agent_doom_loop_detector_plan_v0_4.md`. Matthew accepted every recommendation in it, and the post-M0 revisions are in its "Revisions after M0" section. No Qwen output has been compared with a label._

## Done

| Step | Where | Result |
| --- | --- | --- |
| Data rebuilt on this Mac | datasets/ (gitignored) | Every canonical v0.2 file is byte-identical; every v0.3 raw file matches its manifest |
| Qwen environment | `envs/qwen` | torch 2.14.1 and transformers 5.22.1. Qwen3.5-0.8B pinned at revision `2fc06364`; every file's sha256 is checked; offline load; SDPA attention |
| Task export (dev, val) | `scripts/v04_export_tasks.py` | Rendered blocks with no labels, plus a separate labels file. val: 87,176 windows (the v0.3 rows) |
| Verbaliser readout | `hypaqwen/readout.py` | One pass per session with cache snapshot and restore. Equals a separate prompt per window on MPS and CPU. Label-row edits reach the readout |
| Token audit | `artifacts/reports/v04_token_audit.json` | val: 1,726 tokens per session with the prefix (1,559 without), 1.97× v0.3; 7.87M readout tokens |
| **Fresh sealed test set** | registration `6a778d1`, manifest `8475556` | Seed 20261106, 3,996 sessions, 86,412 windows (83,874 gate-passing). Rebuilt byte-identically under a second PYTHONHASHSEED. No label statistics recorded |
| Prompt choice, pre-registered | `experiments/v0_4/prompt_choice/README.md` | 12 candidates and a label-free rule (median label-word mass, 100 dev sessions) |
| Prompt choice, run and revised | `artifacts/reports/v04_prompt_choice*.json` | The first run chose `w3-state-full`, under which Qwen answers STALLED on all 2,069 dev windows (reported as run). Revision 1 (approved): mass × entropy of the predicted-class shares, still label-free → **`w1-state-full`** (0.742; next 0.526), the M0 draft text. Frozen as `interface.FROZEN`, `spec_sha256` `5922eb45…` |
| **WSP kernel** (§3–6) | `envs/qwen/hypaqwen/wsp/` | Model server with a kernel socket and a forward socket, author client, `./fwd`, guards on torch 2.14.1 (KERNEL_CLIENT, BUILD_EDIT), build runner. Built in three worktrees, then adversarially reviewed: 19 findings, all fixed. At merge, the server and runner were given one shared rescaling rule and edit hash (`wsp/edit_state.py`), and the runner was moved to the real guard |
| Scoring bridge | `src/agent_loop_lab/v04/scoring.py` | Val rows checked against v0.3's `rows_sha256`. Strict column alignment. `m4.score_columns`, with H2 not applicable |
| WSP kernel contract | `docs/V04_WSP_KERNEL.md` | Two processes; forward as a separate metered tool; edit scope; guards; build runner |

## Kernel rules added beyond the contract (fold into `docs/V04_WSP_KERNEL.md` at G1)

- **Rescaling refusal.** With `new = c·W + R`, an edit is refused when `|c − 1| > 1e-3` and `‖R‖ < 0.5·‖(c − 1)·W‖`. For the zero-centred `(1 + w)` norms it also applies to the gain; for the label rows, also to the row-centred block and to the changed rows.
- **Disclosed windows.** `forward` refuses any session that overlaps a WSP-B disclosed window (0.6 shingle or block share). `forward_disclosed` allows **at most 2** distinct non-empty edit states per phase. Confirm at G1.
- **Load quota.** A `./py` script may load at most half of the layer-weight bytes (distinct tensors, per phase). Every kernel op is logged to `<phase>.kernel.jsonl`.
- **Tools copied read-only.** The tools go into the author dir read-only, and `install.verify()` checks them. **Still needed in the launcher (§7):** deny Write/Edit under `tools/` and protect the `./py` and `./fwd` wrappers.
- **Guard.** stderr is kept on a private copy of fd 2, so any violation still ends the run with exit 3. Launchers must use `hn_guard.clean_run(returncode, stderr)`.

## v0.3 reproduces exactly on this Mac

The unofficial v0.3 val dry run (`scripts/v03_sealed_eval.py --dry-run-val`) ran with macOS-27.0.1-arm64, Python 3.12.16, torch 2.2.2, 2 threads and 4 workers. Its report stays in the session scratchpad. Every exact-equality check passed:

- the official v0.3 Phase A and Phase B val reports;
- all 11 Phase C early-stopping macro-F1 values;
- the scoring path, for both phases;
- the val and counterfactual row identity against v0.2;
- the disclosure rule;
- the two priors, overall and per family.

So the frozen v0.3 artefacts can be re-scored on the fresh test set here with no tolerance (plan revision 10). The first attempt stopped because the Phase C `model.pt` files were Git LFS pointers. `git lfs install --local && git lfs pull` fixed that, and the hashes match.

## Next

1. **Qwen-0 val logits** (label-free, about 5 h on MPS): `hypaqwen.run_readout --split val`. They are scored only after G1.
2. Build the session harness: prepare script, launch with a tool allow-list and command hook, audit, cost by session delta, packet, kit tests.
3. Write the v0.4 spec with `[packet]` sections.
4. Write the probe and LoRA recipes and the cloud-GPU scripts.
5. Write the G1 pre-registration and the `registry_v0_4.yaml` freeze.

## Measurements on this Mac

| Backend | Verbaliser readout | Note |
| --- | --- | --- |
| MPS, fp32 | 445 tok/s | Compute-bound; a fixed 40–70 ms per call. One val pass takes about 5 h; 2M tokens take about 75 min |
| CPU, fp32 | under 10 tok/s | Measured with other jobs running. The 100-session CPU replay check moves to the cloud machine |

## Corrections to earlier notes

- An earlier version of this file said the bridge would call `metrics.score_report`. It calls `m4.score_columns`: `score_report` needs a reference policy and accepts val only.
- v0.3's cost was over-counted. `total_cost_usd` is cumulative across resumed launches, so v0.3's hand-written work cost US$64.62 in total, not US$122.24.

## Known pre-existing test failures (not v0.4)

`tests/test_m4_eval.py::test_sealed_script_refuses_second_run` and `tests/test_v03_runner.py::test_side_table_is_unreachable_at_run_time[absolute]` fail on this machine. pytest's temp directory is under `/private/var`, outside the repo. Every other test passes.
