# M0 implementation contracts

This is the binding interface for M0 modules (spec: `agent_doom_loop_detector_spec_v0_2.md`, approved). Where this document and the spec disagree, the spec wins. Report the disagreement; don't resolve it silently.

**Frozen spec file [R2].** The frozen spec is the versioned file `agent_doom_loop_detector_spec_v0_2.md` at the repository root (spec §0.2). It takes the place of the `SPEC.md` named in spec §13. Nothing is renamed or copied to `SPEC.md`; every tool and hash refers to the versioned filename.

**Spec §0.2 [R3].** Spec §0.2 ("Implementation clarifications (M0)") now also records three points this contract implements: the **weight-author interfaces** (`reference_policy(x: tuple[float, ...]) -> int`; `build()` returns the populated model; the harness saves `artifact.dlw` and writes `BUILD_RECORD.json`; `build()` may not run forward passes, matrix products or linear-algebra solvers), the **harness smoke call** (one call of `reference_policy` on the all-zero input, return value not recorded, the only execution in A-strict apart from `build()`), and **failed attempts** (every rejected build is logged under the phase's `attempts/` directory; only a successful build writes the artefact). §E below is the implementation of those clauses.

### Round 3 changes

Clauses marked **[R3]** were added or changed in round 3. The `runner.py`, `packet.py` and `scripts/run_build.py` changes are implemented concurrently with this document; where the code and an **[R3]** clause disagree, report it.

| # | Clause | Change |
|---|---|---|
| R3-1 | `FROZEN_FILES` (§D) | Adds `pyproject.toml`, `uv.lock` (environment pins) and `src/agent_loop_lab/oracle_config.py` (the oracle-config loader). |
| R3-2 | `build_phase` signature (§E) | Adds one keyword, `registry_path="auto"`. The signature is otherwise unchanged. |
| R3-3 | Compute ban (§E step 3, step 4) | No fitting or forward computation while `build_weights.py` runs (spec §0.2 "Weight-author interfaces"): a runtime tripwire plus a static check. `BUILD_RECORD.json` records `"compute_ban": "enforced"`. |
| R3-4 | Attempt log (§E step 8) | Every `build_phase` call, built or rejected, writes `out_dir/attempts/NNNN.json` (spec §10.2, §0.2 "Failed attempts"). On rejection nothing **except the attempt record** is written. |
| R3-5 | Registered packet (§E step 1) | With a registry, the packet's `packet_sha256` must equal the registry's (spec §8, §15.1). `BUILD_RECORD.json` records `packet_registered`. `scripts/run_build.py` gains `--registry PATH` and `--no-registry`. |
| R3-6 | `AUTHOR_INSTRUCTIONS.md` (§D) | States the compute ban; consistent `model` import guidance; one wording for the `reference_policy` smoke call. |

**Deferred (minor).** Not part of round 3, left for later if needed: `reference_policy.py` purity checks beyond the existing static rules; phase lineage (a `base_archive` argument tying A-selfcheck and Phase B to the preceding phase's archive); recording the Python/torch/numpy environment in `BUILD_RECORD.json`; a `verify_archive` function that rebuilds an archive from its own author-file copies; new `agent-loop-lab` CLI subcommands for building or verifying.

### Round 2 changes

Clauses marked **[R2]** below were added or changed in round 2. Everything unmarked is the round-1 contract, unchanged.

| # | Clause | Change |
|---|---|---|
| R2-1 | Frozen spec filename (above) | The versioned filename is the frozen spec, not `SPEC.md` (spec §0.2). |
| R2-2 | `schema.py` timestamp (below) | `CanonicalEvent.timestamp` must be ISO-8601. It used to be any non-empty string. |
| R2-3 | `.gitattributes` (§D) | New: `* text=auto eol=lf` and `*.dlw binary`. |
| R2-4 | Packet `model.py` (§D) | Constants are inlined. It is no longer a byte copy (spec §0.2, §8). |
| R2-5 | `FROZEN_FILES` (§D) | Adds `packet.py`, `guards.py` and `runner.py`. |
| R2-6 | Registry record (§D) | Adds `packet_sha256` and `packet_source_spec_sha256`. `freeze_sha256` now covers the whole record. `check_registry` rebuilds the packet. |
| R2-7 | Build runner (§E) | New `runner.build_phase` contract. |
| R2-8 | Repeatability test (§F) | Adds a two-process artefact test (spec §11). |

Already written (do not edit; report problems instead):
- `src/agent_loop_lab/schema.py`: `CanonicalEvent`, `Provenance`, `OracleRecord`, enums, `FEATURE_NAMES`, `FEATURE_INDEX`, `FEATURE_DEFINITIONS`, `flat_index`, `EVENTS_PER_WINDOW = 8`, `FEATURES_PER_EVENT = 15`, `INPUT_DIM = 120`, `FORBIDDEN_INFERENCE_FIELDS`. Every digest field (`intent_fingerprint`, `failure_fingerprint`, `workspace_state_before`, `workspace_state_after`) must be `None` or exactly 64 lowercase hex characters (`hashlib.sha256(...).hexdigest()` form); anything else raises `SchemaError`. Digests are compared by plain string equality downstream.
  - **[R2] `timestamp`** must be a non-empty `str` (checked first) that `datetime.fromisoformat` accepts (Python 3.12 rules), with a trailing `Z` read as `+00:00`. Anything else raises `SchemaError` ("ISO-8601"). The timezone is optional and date-only strings are accepted. The value is validated, never normalised, so `to_dict`/`to_json` return it exactly as given. Lower-case `z`, surrounding whitespace and epoch numbers are rejected.
- `src/agent_loop_lab/model.py`: `DoomLoopMLP`, `zero_model`, `predict`, `predict_classes`, `PARAM_SHAPES`, `PARAM_COUNT = 3256`, deterministic `.dlw` serialisation, `sparsity`. `zero_model()` and `load_artifact()` leave the process-global torch RNG state unchanged. `deserialize_state_dict` / `load_artifact` raise `ValueError` for every malformed or truncated artefact.
- `tests/factories.py`: `make_event`, `session`, `novel_inspections`, `h`. In `make_event`, omitting `after` means "same as `before`"; an explicit `after=None` means the after-digest is unobserved. Factory timestamps are `2026-01-01T00:MM:SSZ`, so they are valid ISO-8601 for `i < 3600`.

Environment: `uv run pytest`, `uv run python`. Python 3.12, torch 2.2.2, numpy 1.26. Tests import factories as `from tests.factories import ...` (`tests/__init__.py` exists; run pytest from the repo root).

---

## A. `src/agent_loop_lab/fingerprint.py` (spec §3.2)

```python
class FingerprintError(ValueError): ...

def canonical_json(obj) -> bytes
    # json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    # tuples -> lists. Floats are rejected (TypeError/FingerprintError): fingerprints must not depend on float repr.
def sha256_hex(data: bytes | str) -> str          # str encoded UTF-8
def normalise_path(path: str) -> str
    # repo-relative POSIX: backslashes -> "/", collapse "//" and "/./", strip leading "./",
    # resolve "a/../b" lexically; reject absolute paths, drive letters, or anything escaping the root (FingerprintError).
def intent_fingerprint(family: str, target=None, args: Mapping[str, Any] | None = None) -> str
def failure_fingerprint(family: str, error_class: str, location: str | None = None) -> str
def normalise_location(location: str | None) -> str | None
def normalise_message(message: str) -> str       # diagnostics only; NOT part of any hash
def workspace_digest(files: Mapping[str, bytes | str]) -> str
FAMILY_CATEGORY: Mapping[str, OperationCategory]  # family -> category, from the §3.2 table
```

`intent_fingerprint` = `sha256_hex(canonical_json({"family": family, "target": <normalised target>, "args": <included args dict>}))`.

Per-family rules. Every key in `args` must be **included** or **excluded** for that family. Any other key raises `FingerprintError`, so definitions can't drift silently. Globally excluded keys for every family: `timestamp`, `attempt`, `attempt_number`, `retry`, `retry_count`, `timeout`, `request_id`, `reasoning`, `free_text`, `text`.

| family | target (normalised) | included args → payload key | family-specific excluded |
|---|---|---|---|
| `read_file` | path | `line_start`, `line_end` (int or None) → `line_range: [start, end]` | — |
| `search` | scope path | `query` → lower-cased, whitespace collapsed and stripped | `limit` |
| `list_dir` | path | — | — |
| `edit_file` | path | `patch` (str) → `patch_sha256` of normalised patch (CRLF/CR → LF, trailing whitespace stripped per line, trailing blank lines stripped) | — |
| `write_file` | path | `content` (str or bytes; str is UTF-8) → `content_sha256` of raw bytes | — |
| `revert` | list of paths → sorted unique normalised list | `revision` (str) → `revision` | — |
| `run_tests` | suite id (str, stripped; not a path) | `tests` (iterable of str) → `tests`, sorted and unique | `verbosity`, `parallelism` |
| `build`, `lint` | target str (stripped) | — | `flags` |
| `http_call` | mapping `{method, host, path_template}` → method upper-cased, host lower-cased, path_template as given | `body` (str, bytes or None) → `body_sha256`, **only** for non-idempotent methods (`POST`, `PATCH`). JSON-parseable bodies are hashed in canonical JSON form; anything else as raw bytes. | for idempotent methods (`GET`, `HEAD`, `OPTIONS`, `PUT`, `DELETE`) `body` is accepted and ignored |
| `poll_status` | resource id (str, stripped) | — | `backoff` |
| `plan`, `ask`, `yield` | must be None | `action_type` (str, required, lower-cased) → `action_type` | — |

An unknown family raises `FingerprintError`. A missing required target raises `FingerprintError` (only `plan`/`ask`/`yield` take no target).

`failure_fingerprint` = `sha256_hex(canonical_json({"family": family, "error_class": error_class.strip(), "location": normalise_location(location)}))`.

`normalise_location`:
- strip trailing `:<line>` and `:<line>:<col>`;
- remove hex addresses (`0x[0-9a-fA-F]+`);
- for absolute paths, keep only the part after the last `/src/`, `/tests/` or `/lib/` anchor if present, else the basename;
- keep pytest node ids (`path::test_name`) intact apart from the line stripping.
- `None` stays `None`.

`normalise_message` removes numbers, hex addresses, ISO timestamps and absolute paths. Diagnostics only.

`workspace_digest` = `sha256_hex(canonical_json(sorted([normalise_path(p), sha256_hex(content)] for p, content in files.items())))`. Duplicate normalised paths raise.

## B. `src/agent_loop_lab/encode.py` and `src/agent_loop_lab/windows.py` (spec §4, §5)

`windows.py` owns sequence validation and the gate. `encode.py` owns features. `encode.py` must not import `windows.py`; `windows.py` may import `encode.py`.

```python
# encode.py
class EncodingError(ValueError): ...
TOKEN_LOG_SCALE = math.log1p(100000)
def encode_event_matrix(prefix: Sequence[CanonicalEvent]) -> np.ndarray   # float32 (8, 15)
def encode_window(prefix: Sequence[CanonicalEvent]) -> np.ndarray         # float32 (120,) == matrix.reshape(-1)
def describe_matrix(matrix: np.ndarray) -> str                            # human-readable table with FEATURE_NAMES
```

- `prefix` is the session prefix up to and including the window endpoint. The window is the last 8 events. Earlier events are used **only** for the validation baseline (features 12–13).
- Preconditions: `len(prefix) >= 8`, and every observation required by the gate is present. Otherwise raise `EncodingError`. **Never** map an unknown value to 0. The encoder doesn't re-check sequence identity or ordering; that's the window layer's job.
- `repeated_intent`, `repeated_failure`, `state_revisited`: compare only with **earlier events inside the window**. So position 0 always has `repeated_intent = repeated_failure = 0`.
- `state_revisited`: 1 iff `workspace_changed` and `after ∈ {window[0].before} ∪ {window[j].before, window[j].after : j < i}`.
- `validation_comparable` / `validation_delta`: for a window event with `validation_produced`, search **all earlier events in the prefix** (inside and before the window) for the most recent one with `validation_produced` and the same `validation_suite_id`. If found, comparable = 1 and delta = `clip(p/t − p'/t', −1, 1)`, computed in float64 and then cast to float32. If not found, both are 0. If the most recent earlier same-suite result has unobserved passed/total, raise `EncodingError` (the gate rejects this case first).
- `relative_token_cost = min(1, log1p(in + out) / TOKEN_LOG_SCALE)`, computed in float64 and cast to float32. If both counts are 0 the result is exactly 0.0.
- Output is deterministic: the same prefix gives byte-identical arrays (`arr.tobytes()`).

```python
# windows.py
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
class SequenceError(ValueError): ...
class GateReason(str, Enum):
    TOO_SHORT, MISSING_TOKENS, MISSING_WORKSPACE_DIGEST, MISSING_FAILURE_FINGERPRINT,
    MISSING_VALIDATION_FIELDS, UNOBSERVED_VALIDATION_BASELINE
@dataclass(frozen=True) class GateDecision:
    passed: bool
    reasons: tuple[tuple[int, GateReason], ...]   # (window position 0..7, or -1 for TOO_SHORT), sorted
    @property result -> str   # "PASS" or INSUFFICIENT_EVIDENCE
@dataclass(frozen=True) class WindowResult:
    endpoint_index: int            # sequence_index of the window's last event
    gate: GateDecision
    features: np.ndarray | None    # (120,) float32 when gate passed, else None
def validate_sequence(events: Sequence[CanonicalEvent]) -> None
def gate(prefix: Sequence[CanonicalEvent]) -> GateDecision
def window_endpoints(events: Sequence[CanonicalEvent]) -> list[int]        # list positions t >= 7
def iter_windows(events: Sequence[CanonicalEvent]) -> Iterator[WindowResult]
def encode_session(events) -> tuple[list[int], np.ndarray]  # endpoints of passing windows, stacked (n, 120) float32 (shape (0, 120) if none)
```

`validate_sequence` raises `SequenceError` for:
- mixed `session_id`, `agent_id` or `task_id`;
- duplicate `event_id`s;
- `sequence_index` not strictly increasing by exactly 1;
- a first index other than 0 when `provenance.history_complete` is True.

`gate` (spec §5.2) looks at the last 8 events of `prefix` and is purely about instrumentation; it makes **no semantic judgement**. Fewer than 8 events gives `TOO_SHORT`. For each window event:
- `input_tokens` or `output_tokens` None → `MISSING_TOKENS`;
- either workspace digest None → `MISSING_WORKSPACE_DIGEST`;
- ERROR with `failure_fingerprint` None → `MISSING_FAILURE_FINGERPRINT`;
- `validation_produced` with any of suite id, passed or total None → `MISSING_VALIDATION_FIELDS`;
- `validation_produced`, and either (i) the most recent earlier same-suite result in the prefix has passed/total None, or (ii) no earlier same-suite result exists and `provenance.history_complete` is False → `UNOBSERVED_VALIDATION_BASELINE`.

A window of eight novel INSPECT/SUCCESS events with full instrumentation **passes**.

`iter_windows` calls `validate_sequence` once, then for each endpoint `t >= 7` (list position) runs `gate(events[:t+1])` and encodes only if the window passed. It never reads events after `t` (causality).

## C. `src/agent_loop_lab/guards.py` (spec §11 anti-training)

```python
class TrainingViolation(RuntimeError): ...
FORBIDDEN_MODULE_PREFIXES = ("agent_loop_lab.synth", "agent_loop_lab.synth_ood", "agent_loop_lab.datasets",
                             "agent_loop_lab.eval", "agent_loop_lab.baselines")
@contextmanager
def no_training_guard(extra_forbidden: Sequence[str] = ()) -> Iterator[None]
def run_guarded(fn: Callable[[], T]) -> T
def audit_model(model: DoomLoopMLP) -> None      # raises TrainingViolation if any param requires_grad or has .grad
def scan_source(path: str | Path) -> list[str]   # static AST audit; returns violation descriptions
```

`no_training_guard`:
1. On entry, fails if any forbidden module prefix is already in `sys.modules`.
2. Installs a `sys.meta_path` finder that raises `TrainingViolation` when a forbidden prefix is imported. Note that `torch.optim` **cannot** be import-blocked, because `import torch` loads it.
3. Patches `torch.optim.Optimizer.__init__`, `torch.Tensor.backward`, `torch.autograd.backward` and `torch.autograd.grad` to raise `TrainingViolation`.
4. Enters `torch.no_grad()` and patches `torch.set_grad_enabled` and `torch.enable_grad` to raise when they would enable grad.
5. On exit, restores every patch, including after exceptions, and asserts grad mode was never enabled.

Guards must nest safely or explicitly refuse nesting, and they must not leak patches between tests.

`scan_source` flags imports of `torch.optim`, `torch.autograd` or any forbidden prefix, and any calls to `.backward(`, `.step(`, `optim.` or `autograd.grad`. It also flags `requires_grad=True` and `requires_grad_(True)`, file reads of `datasets/` or `synth` paths, and `import random` / `np.random` used for search. Report each as a string with its line number.

## D. Oracle config, authoring packet, registry

### `oracle_config.v0_2.yaml` (repo root) and `src/agent_loop_lab/oracle_config.py`

```python
@dataclass(frozen=True) class OracleConfig: eps_p: float; f_prod: int; n_stall: int; n_grey: int; n_rev: int; r_max: int; p_max: int; k_rec: int
FROZEN_V0_2 = OracleConfig(eps_p=0.02, f_prod=2, n_stall=5, n_grey=3, n_rev=2, r_max=3, p_max=5, k_rec=4)
def load_oracle_config(path: str | Path | None = None) -> OracleConfig   # default: repo-root yaml
```

The loader validates exact key set, types and ranges: `0 <= n_grey < n_stall <= 8`, `1 <= n_rev <= 8`, `0 <= k_rec <= 8`, `eps_p > 0`. The YAML records `spec_version: "0.2"` and the approval date `2026-11-04`.

### `src/agent_loop_lab/packet.py` and `scripts/build_authoring_packet.py`

```python
EXCLUDED_SECTIONS = ("## 0.", "### 2.3", "### 7.2", "### 7.3")   # by heading prefix
def strip_spec(spec_text: str) -> str
def build_packet(spec_path, model_path, out_dir) -> dict          # returns manifest dict, also written to MANIFEST.json
def verify_packet(out_dir) -> list[str]                           # returns problems; [] = OK
```

`strip_spec` removes each excluded section from its heading up to (not including) the next heading of the same or higher level. So `## 0.` runs to `## 1.` and includes `### 0.1`; `### 2.3` runs to `## 3.`. Keep the title and status block. A `---` horizontal rule that only separated a removed section may be dropped, but nothing else.

Packet files, all deterministic:
- `SPEC_AUTHORING.md`
- **[R2]** `model.py`: the source module made standalone, **not** a byte copy (spec §0.2, §8). The harness module's single line `from .schema import CLASS_NAMES, INPUT_DIM` can't be imported from the packet directory. Importing the harness package instead would expose generator and oracle code. So the builder (`standalone_model_source`) replaces exactly that line with a marked block (`inlined_constants_block()`) of literal `INPUT_DIM` and `CLASS_NAMES` values taken from `agent_loop_lab.schema`. Nothing else changes. The builder refuses a source that repeats the import line, already contains the block, or keeps any other harness-only import (relative, or `agent_loop_lab`). The substitution is exactly invertible (`original_model_source`), and the manifest records `source_model_sha256`, so `verify_packet` can prove that the packet copy is the source module with only those constants inlined. A source without the import line is copied byte for byte.
- `feature_index.json`: positions × features → flat index, names, definitions from `schema.FEATURE_DEFINITIONS`, class order
- `state_dict_template.json`: names, shapes, `PARAM_COUNT`, "all zeros"
- `AUTHOR_INSTRUCTIONS.md`: the deliverables, A-strict/A-selfcheck/B rules and forbidden actions from spec §9, and **[R2]** the author-side `build()` / `reference_policy` interface of §E. Where the author writes is set by the adjudicator, outside the packet. The harness archives each phase under `weight_program/<phase>/` (spec §13) using `runner.build_phase`. **[R3]** It also states, in plain language, the compute ban of §E step 4 (what `build()` may and may not call); gives one consistent `model` import guidance (no example uses a `model` name the pinned import doesn't bring in, e.g. a `__main__` block calling `model.sparsity` under `from model import zero_model, DoomLoopMLP`); and words every mention of the smoke call the same way: the harness calls `reference_policy` once on the all-zero input as a type check, and the return value is not recorded.
- `MANIFEST.json`: per-file sha256, a `packet_sha256` (sha256 of canonical JSON of the sorted `[name, sha256]` list over the five content files), `source_spec_sha256`, **[R2]** `source_model_sha256`, `spec_version`, `packet_format_version` and `excluded_sections`

Sources with CR line endings are refused (`PacketError`), and every packet file is LF only.

`verify_packet` checks:
- file set and hashes;
- no file comes from `synth*/` or `datasets/`;
- `SPEC_AUTHORING.md` contains none of the excluded headings and none of the capacity-hint phrases (`per position`, `Design note on capacity`, `Generator requirements`);
- §4, §6 and §9 headings are present;
- **[R2]** `model.py` is standalone (no relative or `agent_loop_lab` imports), contains the inlined block at most once, and restores to `source_model_sha256`;
- `AUTHOR_INSTRUCTIONS.md`, `feature_index.json` and `state_dict_template.json` equal what the builder would produce now.

### `src/agent_loop_lab/registry.py` and `scripts/freeze.py`

```python
FROZEN_FILES = ("agent_doom_loop_detector_spec_v0_2.md",
                "pyproject.toml", "uv.lock",                                   # [R3] environment pins
                "oracle_config.v0_2.yaml",
                "src/agent_loop_lab/oracle_config.py",                         # [R3] oracle-config loader
                "src/agent_loop_lab/schema.py",
                "src/agent_loop_lab/fingerprint.py", "src/agent_loop_lab/encode.py", "src/agent_loop_lab/windows.py",
                "src/agent_loop_lab/model.py",
                "src/agent_loop_lab/packet.py", "src/agent_loop_lab/guards.py", "src/agent_loop_lab/runner.py")  # [R2] last three
PACKET_KEYS = ("packet_sha256", "packet_source_spec_sha256")                                              # [R2]
REGISTRY_KEYS = {"spec_version", "files", "packet_sha256", "packet_source_spec_sha256", "freeze_sha256"}  # [R2]
REGISTRY_RELPATH = "experiments/registry.yaml"
class RegistryError(RuntimeError): ...
def freeze_digest(record: Mapping) -> str        # [R2] over the full record minus freeze_sha256
def packet_hashes(root=".") -> dict[str, str]   # [R2] {packet_sha256, packet_source_spec_sha256}
def compute_freeze(root=".") -> dict            # the registry record
def write_registry(root=".", *, overwrite=False) -> Path   # experiments/registry.yaml
def check_registry(root=".") -> list[str]       # mismatches vs registry; [] = OK
```

**[R2] Registry record** (`experiments/registry.yaml`, YAML with sorted keys under a fixed comment header):

| Key | Value |
|---|---|
| `spec_version` | `"0.2"` |
| `files` | `{frozen path: sha256 of the raw file bytes}` for every entry of `FROZEN_FILES` |
| `packet_sha256` | `packet_sha256` from the manifest of the authoring packet, rebuilt into a temporary directory |
| `packet_source_spec_sha256` | `source_spec_sha256` from that manifest, i.e. the sha256 of the spec the packet was built from (equal to `files[spec]` when consistent) |
| `freeze_sha256` | sha256 of canonical JSON (`sort_keys=True`, `separators=(",", ":")`, `ensure_ascii=False`) of the record without `freeze_sha256` |

- **[R2] Packet build.** `packet_hashes(root)` calls `packet.build_packet(<root>/agent_doom_loop_detector_spec_v0_2.md, <root>/src/agent_loop_lab/model.py, <tempdir>)`, using a `tempfile.TemporaryDirectory`, and reads the two hashes from the returned manifest. Nothing is written under `root`. The builder is the *imported* `agent_loop_lab.packet`, whose source is itself frozen by hash. The build is path-free and deterministic, so identical sources give an identical `packet_sha256`. A build or verification failure raises `RegistryError("authoring packet could not be built: ...")`.
- `compute_freeze` raises `FileNotFoundError` when a frozen file is missing (including `runner.py`). It raises `RegistryError` when a frozen file contains CR, or when the packet can't be built.
- `write_registry` treats rewriting an identical registry as a no-op. Replacing a registry whose content differs requires `overwrite=True` (`--force`).
- `check_registry` never raises. It reports, as strings:
  - a missing, unreadable or non-mapping registry;
  - a wrong key set, or a wrong `spec_version`;
  - non-string packet hashes;
  - frozen files that are missing, modified, not in the registry, or not frozen;
  - CRLF line endings, named as such so a CRLF checkout isn't mistaken for a content change;
  - **[R2]** packet hashes that differ from a fresh rebuild (`"authoring packet <key> differs from the registry ..."`), or an unbuildable packet;
  - **[R2]** a `freeze_sha256` that doesn't match the record.

  The rebuild is skipped when the spec or `model.py` is missing or has CR, because those are already reported per file and the builder refuses CR sources. Since it rebuilds, `check_registry` also catches a change in what the builder produces even when no frozen file changed.
- `scripts/freeze.py [--root DIR]` prints the record as JSON. `--write [--force]` writes the registry and prints the packet and freeze sha256. `--check` exits 1 on any mismatch. `agent-loop-lab freeze write|check` behaves the same way.
- **[R2] `.gitattributes`** at the repo root pins `* text=auto eol=lf` and `*.dlw binary`, so the frozen hashes, which depend on line endings, match on every checkout, and artefacts are never re-encoded.

`tests/test_registry.py` must **skip** (not fail) when `experiments/registry.yaml` doesn't exist yet. Once it exists, the test fails on any mismatch. The registry is written at M0 sign-off (`scripts/freeze.py --write`), after `runner.py` is final. It is not written before then. Tests build their roots in `tmp_path`, using copies of the spec, model, oracle config, schema, packet builder and guard, **[R3]** `pyproject.toml` and `uv.lock`, and placeholders for the other frozen files. **[R3]** A changed torch pin in either environment file (made only in the `tmp_path` copy) is reported as `frozen file modified` and leaves the packet hashes unchanged.

## E. `src/agent_loop_lab/runner.py`: harness build runner [R2] (spec §9, §11, §15)

```python
PHASES = ("phase-a-strict", "phase-a-selfcheck", "phase-b-revision")
REQUIRED_FILES = {"phase-a-strict":    ("build_weights.py", "reference_policy.py", "CIRCUITS.md"),
                  "phase-a-selfcheck": (... , "ITERATIONS.md"),
                  "phase-b-revision":  (... , "REVISION.md")}
class BuildError(RuntimeError): ...
def build_phase(phase: str, author_dir, packet_dir, out_dir, *, rebuild_check: bool = True, force: bool = False,
                registry_path="auto") -> dict                                  # [R3] registry_path
    # returns the build record, also written to out_dir/BUILD_RECORD.json
def check_build_source(path) -> list[str]
def check_reference_policy_source(path) -> list[str]
```

Author-side interface (stated in the packet's `AUTHOR_INSTRUCTIONS.md`):
- `build_weights.py` defines a module-level `build() -> DoomLoopMLP` with no arguments. It starts from the packet's `model.zero_model()` and returns the populated model. The harness saves the artefact, so the author doesn't call `save_artifact` or `torch.save`.
- `reference_policy.py` defines `reference_policy(x: tuple[float, ...]) -> int`, taking 120 floats and returning a value in `0..3`. It is pure and deterministic, with no I/O, and imports only the standard library (no `numpy`, no `torch`).

`build_phase` steps, in order (any failure raises `BuildError` naming the file and rule, and **nothing is written** to `out_dir` **[R3] except the attempt record of step 8**):
1. **Verify the packet.** `packet.verify_packet(packet_dir)` must return `[]`. The author and output directories must be outside the packet directory.
   - **[R3] Registered packet** (spec §8 "sha256 recorded", §15.1). `registry_path="auto"` means `<repo root of the runner module>/experiments/registry.yaml` if that file exists, and no check otherwise. An explicit path must exist (`BuildError` otherwise). `None` disables the check. When checked, the packet's `MANIFEST.json` `packet_sha256` must equal the registry's `packet_sha256`, else `BuildError`.
2. **Required files.** The phase's `REQUIRED_FILES` must exist in `author_dir`.
3. **Scan.**
   - `guards.scan_source(build_weights.py)` must be clean.
   - `check_build_source` must be clean: allowed imports are the standard library minus `BANNED_MODULES`, plus `torch`, `numpy` and the packet's `model`; no I/O builtins; `build()` is defined.
   - `check_reference_policy_source` must be clean: allowed imports are the standard library minus `BANNED_MODULES`; `reference_policy` is defined.
   - **[R3]** `check_build_source` also flags the compute ban statically: the `@` operator (`ast.MatMult`) and the source substrings `.forward(`, `.dot(`, `linalg`, `lstsq`, `pinv`, `einsum` and `matmul`. Not covered (deliberately left as minor): top-level solvers outside `torch.linalg` (`torch.inverse`, `torch.svd`, …), `np.polyfit`, and forward passes written with elementwise operations. The author instructions forbid all of these in prose.
4. **Guarded build.** Everything here runs inside `guards.no_training_guard()`, with the packet directory first on `sys.path`, bytecode writing off and a runtime I/O tripwire active.
   - **[R3] Compute ban** (spec §0.2 "Weight-author interfaces": no fitting or forward computation inside `build()`). While `build_weights.py`'s module code and `build()` run, a runtime tripwire raises `BuildError` on: `torch.nn.functional.linear`; `nn.Linear.forward` and calling any `nn.Module`; `torch.matmul`, `mm`, `bmm`, `addmm`, `addmv`, `mv`, `einsum`, `tensordot`, `inner`, `outer`, `dot`; every `torch.linalg.*` function; `Tensor.__matmul__`, `__rmatmul__`, `matmul`, `mm`, `mv`, `dot`, `bmm`, `addmm`, `addmv`, `inner`, `outer` (the method forms); every `numpy.linalg.*` function; and `numpy.matmul`, `dot`, `einsum`, `tensordot`, `inner`, `outer`, `vdot`. Everything is restored afterwards, including after an exception. Elementwise and scalar tensor arithmetic and index assignment remain allowed. The fresh-process rebuild of step 5 enforces the same ban.
   - `build_weights.py` is imported **by file path** (`importlib.util.spec_from_file_location`), so its module-level code also runs under the guard.
   - `build()` is called and must return a `DoomLoopMLP`. Its `state_dict` is checked with `model.validate_state_dict` (exact keys, shapes, float32, finite), loaded into a harness `zero_model()` and checked with `guards.audit_model`.
   - `reference_policy.py` is imported the same way and called **once**, on the all-zero 120-tuple. This only checks that it returns an `int` in `0..3`. It is not a behavioural test, and its value is not recorded.
   - Afterwards `sys.path` and `sys.modules` are restored: no `model`, `build_weights` or `reference_policy` module is left behind.
5. **Rebuild check** (with `rebuild_check`, the default). The same guarded build runs again in a fresh interpreter (`python -I -B`), and its artefact bytes must be identical to the first build's (spec §15 item 2).
6. **Archive.** Only now is anything written to `out_dir` (**[R3]** apart from the attempt record):
   - `artifact.dlw` (`model.save_artifact`). Its metadata is deterministic and includes `phase`, `spec_version`, `packet_sha256` and the sha256 of `build_weights.py`, `reference_policy.py` and `CIRCUITS.md`. It has no timestamps and no paths.
   - copies of the phase's required author files;
   - `artifact.sha256` (`"<sha256>  artifact.dlw\n"`);
   - `BUILD_RECORD.json` (canonical JSON). It holds the metadata, artefact sha256 and size, per-file sha256, `sparsity`, and the result of each check. **[R3]** It also records `"compute_ban": "enforced"` and `"packet_registered": true|false` (whether the step-1 registry check ran and passed).
7. **Immutable archives.** If `out_dir/artifact.dlw` already exists, `build_phase` refuses before building and again before writing, unless `force=True`. The runner never git-commits; the adjudicator commits the archive (spec §9.2). **[R3]** The refusal also writes an attempt record (step 8).
8. **[R3] Attempt log** (spec §10.2; §0.2 "every rejected build is logged under the phase's `attempts/` directory"). Every `build_phase` call, built or rejected, writes `out_dir/attempts/NNNN.json`, where `NNNN` is the next free 4-digit index. It is canonical JSON holding: the attempt index; the phase; the outcome, `"built"` or `"rejected"`; the error message, with absolute paths replaced by `<author>`, `<packet>` and `<out>` placeholders; the sha256 of each author file present; and `packet_sha256` if known. It has no timestamps. `artifact.dlw`, `artifact.sha256`, `BUILD_RECORD.json` and the author-file copies are still written only on success.

**[R3] CLI.** `scripts/run_build.py` gains `--registry PATH` (an explicit `registry_path`) and `--no-registry` (`registry_path=None`). Without either, the default `"auto"` applies.

The runner never imports generator, dataset or evaluation code, and the build path reads no labelled data.

## F. Repeatability tests [R2] (spec §11 "Repeatability", §15)

- Encoder: identical session prefixes give byte-identical feature arrays (`test_encode.py`).
- Artefacts: identical weights and metadata give byte-identical `.dlw` bytes (`test_model.py`).
- **[R2] Two processes:** `tests/test_model.py::test_two_process_repeatability` saves a hand-written literal-weight artefact to `tmp_path`. It then loads it in two separate `sys.executable` subprocesses with different `PYTHONHASHSEED`s and runs a fixed batch (zero row, four one-hot probes, eleven dense rows). The test asserts that both processes, and the parent, produce byte-identical float32 logits and identical predictions, and that the predictions cover all four classes.
