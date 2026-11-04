"""v0.3 build runner for the weight author's deliverables (spec v0.3 §9, §11, §15).

``build_phase`` turns one phase directory of the weight author into an
immutable, hashed archive, mirroring the v0.2 runner's semantics:

1. **Packet.** ``packet.verify_packet(packet_dir)`` must report no problems;
   the packet's copies of ``model.py``, ``tokenizer.py`` and ``renderer.py``
   (``packet.PACKET_SOURCE_FILES``) must be byte-identical to the harness
   sources in ``src/agent_loop_lab/v03/`` (manifest ``sources``); and with a freeze
   registry (``registry_path="auto"``: ``experiments/registry_v0_3.yaml`` when
   it exists; an explicit path must exist; ``None`` disables the check) the
   packet hash (sha256 of ``PACKET_MANIFEST.json``) must equal the registry's.
2. **Required files.** ``build_weights.py``, ``reference_policy.py``,
   ``CIRCUITS.md`` and ``ITERATIONS.md`` (v03-phase-a) or ``REVISION.md``
   (v03-phase-b).
3. **Static rules.** ``source_rules.check_build_weights`` and the v0.2
   ``guards.scan_source`` on ``build_weights.py``;
   ``source_rules.check_reference_policy`` on ``reference_policy.py``.
4. **Build.** ``build_weights.py`` alone is copied into a fresh private
   directory, next to a copy of the packet files (nothing else of the author's
   directory is reachable: the archived files are exactly the files the build
   read). ``build()`` runs from that copy in a fresh interpreter (``python -I
   -B``, working directory the private copy) with the packet copy first on
   ``sys.path``, under the v0.3 BUILD guard (``guards.build_guard``: v0.2
   anti-training guard, RNG ban, compute ban, no I/O; the only readable files
   are Python code of the interpreter install, the packet copy and the copied
   ``build_weights.py``), one torch thread, ``RLIMIT_CPU`` set to 300 s of CPU
   beyond the interpreter's start-up, and a 600 s wall-clock timeout. The
   returned model must be the packet's ``DoomLoopTransformer``; its
   ``state_dict`` is read and validated inside the guard, loaded into a harness
   ``zero_model()`` and audited (no ``requires_grad``, ``.grad`` or
   ``grad_fn``). The child reports through a result file written by harness
   code after the guard, never through its stdout or stderr (which author code
   can print to), and ends with ``os._exit`` so no author finaliser runs
   unguarded.
5. **Rebuild.** With ``rebuild_check`` (default) the same build runs in a
   second fresh interpreter and its artefact bytes must be identical.
6. **Smoke call.** In a third fresh interpreter, a private copy of
   ``reference_policy.py`` (alone) is executed under the BUILD guard (no I/O,
   no RNG) and ``reference_policy`` is
   called **once**, on the surface tokens and event ordinals of the fixed
   grammar-sheet block :data:`SMOKE_BLOCK`, with a 0.5 s CPU limit
   (``ITIMER_PROF``, plus a ``process_time`` check after the call). It must
   return an ``int`` in ``0..3``. This is a type and time check, not a
   behavioural test; the value is not recorded.
7. **Archive.** Only then is anything written: ``artifact.dlw``
   (``artifact.serialize_state_dict`` with deterministic metadata, no
   timestamps), copies of the required files, the **whole phase directory**
   (scratch scripts, ``ITERATIONS.md`` inputs and every other file; symlinks
   are recorded, never followed; ``__pycache__`` is skipped) under
   ``author_output/``, ``artifact.sha256`` and ``BUILD_RECORD.json``
   (canonical JSON, including per-tensor sparsity: the non-zero count and the
   non-zero blocks of ``artifact.nonzero_blocks``, and ``author_output``: the
   sha256 of every archived file).

Every call, built or rejected, writes ``out_dir/attempts/NNNN.json`` (next free
index; phase, outcome, redacted error, author-file hashes, the sha256 of every
file of the phase directory, packet hash). An existing ``artifact.dlw`` is
never overwritten unless ``force=True``. The runner does not git-commit. The
parent process never executes author code.

**Error text.** A rejection's error text is what a resubmission discloses to
the author (``scripts/v03_launch.sh resubmit``), so it never carries content
that author code chose: an exception raised by author code is reported as its
type (builtin exception classes only) and the author-file line numbers of its
traceback, without its message; a child's stdout / stderr is kept for the
operator (``operator_detail`` in the attempt record) but not in the error; and
the text is capped at :data:`MAX_ERROR_CHARS`.

**Policy evaluation.** :func:`evaluate_policy` is the only supported way for
harness scoring (H1, H2) to call an author ``reference_policy``: a private copy
of the file runs in a fresh interpreter under the BUILD guard (no I/O, no RNG),
with the 0.5 s CPU limit per call; a timeout, an exception or a value that is
not an ``int`` in 0..3 gives ``None`` (a disagreement). The harness process
never imports the policy.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import artifact as harness_artifact
from . import model as harness_model
from .packet import GRAMMAR_EXAMPLES, PACKET_SOURCE_FILES, PacketError, packet_sha256, verify_packet
from .source_rules import check_build_weights, check_reference_policy

__all__ = [
    "BuildError",
    "PHASE_A",
    "PHASE_B",
    "PHASES",
    "REQUIRED_FILES",
    "SMOKE_BLOCK",
    "BUILD_CPU_BUDGET_S",
    "BUILD_TIMEOUT_S",
    "POLICY_CPU_LIMIT_S",
    "REGISTRY_RELPATH",
    "MAX_ERROR_CHARS",
    "AUTHOR_OUTPUT_DIR",
    "build_phase",
    "evaluate_policy",
]

SPEC_VERSION = "0.3"
RECORD_FORMAT_VERSION = 1

PHASE_A = "v03-phase-a"
PHASE_B = "v03-phase-b"
PHASES = (PHASE_A, PHASE_B)

BUILD_FILE = "build_weights.py"
POLICY_FILE = "reference_policy.py"
CIRCUITS_FILE = "CIRCUITS.md"
ITERATIONS_FILE = "ITERATIONS.md"
REVISION_FILE = "REVISION.md"
REQUIRED_FILES: Mapping[str, tuple[str, ...]] = {
    PHASE_A: (BUILD_FILE, POLICY_FILE, CIRCUITS_FILE, ITERATIONS_FILE),
    PHASE_B: (BUILD_FILE, POLICY_FILE, CIRCUITS_FILE, REVISION_FILE),
}
ARTIFACT_FILE = "artifact.dlw"
ARTIFACT_SHA_FILE = "artifact.sha256"
RECORD_FILE = "BUILD_RECORD.json"
ATTEMPTS_DIR = "attempts"
AUTHOR_OUTPUT_DIR = "author_output"
MAX_ERROR_CHARS = 2000
_STAGE_PREFIX = "v03_stage_"
_ATTEMPT_RE = re.compile(r"^(\d{4})\.json$")

BUILD_CPU_BUDGET_S = 300
BUILD_TIMEOUT_S = 600
POLICY_CPU_LIMIT_S = 0.5
SMOKE_TIMEOUT_S = 120

REGISTRY_RELPATH = "experiments/registry_v0_3.yaml"
REPO_ROOT = Path(__file__).resolve().parents[3]
AUTO_REGISTRY_PATH = REPO_ROOT / REGISTRY_RELPATH
HARNESS_SOURCE_DIR = Path(__file__).resolve().parent

# The read_file block of the grammar sheet (packet GRAMMAR_SHEET.md). Literal, invented words only.
SMOKE_BLOCK = GRAMMAR_EXAMPLES[0][1]
assert SMOKE_BLOCK.split()[1] == "read_file"
SMOKE_NOTE = (
    "reference_policy was called once, on the surface tokens of the fixed grammar-sheet read_file block, "
    f"under a {POLICY_CPU_LIMIT_S} s CPU limit, and returned an int in 0..3; a type and time check, not a "
    "behavioural test; the value is not recorded"
)

RESULT_FILE = "result.json"


class BuildError(RuntimeError):
    """The author's deliverables were rejected, or the archive could not be written.

    ``detail`` (operator only, never part of ``str(exc)``) holds raw child output."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.detail = detail


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


# -- static checks ---------------------------------------------------------------------


def _resolve_registry(registry_path: str | Path | None) -> Path | None:
    if registry_path is None:
        return None
    if isinstance(registry_path, str) and registry_path == "auto":
        return AUTO_REGISTRY_PATH if AUTO_REGISTRY_PATH.is_file() else None
    p = Path(registry_path)
    if not p.is_file():
        raise BuildError(f"freeze registry not found: {p}")
    return p


def _check_registered(packet_dir: Path, registry: Path) -> None:
    try:
        data = yaml.safe_load(registry.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise BuildError(f"freeze registry unreadable: {exc}") from exc
    registered = data.get("packet_sha256") if isinstance(data, dict) else None
    if not isinstance(registered, str):
        raise BuildError("freeze registry has no packet_sha256 string")
    actual = packet_sha256(packet_dir)
    if actual != registered:
        raise BuildError(
            f"authoring packet is not the registered one (packet_sha256 {actual}, registry {registered}); "
            "build the packet from the frozen sources"
        )


def _check_packet_sources(packet_dir: Path, source_dir: Path) -> None:
    for name in PACKET_SOURCE_FILES:
        pkt = packet_dir / name
        src = source_dir / name
        if not src.is_file():
            raise BuildError(f"harness source {name} missing from {source_dir}")
        if _sha256(pkt.read_bytes()) != _sha256(src.read_bytes()):
            raise BuildError(
                f"packet {name} differs from the harness source the weights are checked against; "
                "rebuild the packet with scripts/v03_build_packet.py"
            )


def _static_checks(phase: str, author_dir: Path, packet_dir: Path, registry: Path | None,
                   source_dir: Path, attempt: dict[str, Any]) -> None:
    problems = verify_packet(packet_dir)
    if problems:
        raise BuildError("authoring packet failed verification: " + "; ".join(problems))
    attempt["packet_sha256"] = packet_sha256(packet_dir)
    if registry is not None:
        _check_registered(packet_dir, registry)
    _check_packet_sources(packet_dir, source_dir)
    if not author_dir.is_dir():
        raise BuildError(f"author directory not found: {author_dir}")
    missing = [f for f in REQUIRED_FILES[phase] if not (author_dir / f).is_file()]
    if missing:
        raise BuildError(f"{phase}: required file(s) missing from the author directory: {missing}")
    found = check_build_weights(author_dir / BUILD_FILE)
    if found:
        raise BuildError(f"{BUILD_FILE}: rejected by the source rules: " + "; ".join(found))
    from ..guards import scan_source  # v0.2 static training scan (optimiser, autograd, RNG, data paths)

    scan = [s.replace(str(author_dir / BUILD_FILE), BUILD_FILE) for s in scan_source(author_dir / BUILD_FILE)]
    if scan:
        raise BuildError(f"{BUILD_FILE}: rejected by the anti-training source scan: " + "; ".join(scan))
    found = check_reference_policy(author_dir / POLICY_FILE)
    if found:
        raise BuildError(f"{POLICY_FILE}: rejected by the source rules: " + "; ".join(found))


# -- child processes ---------------------------------------------------------------------


def _metadata(phase: str, author_dir: Path, packet_dir: Path) -> dict[str, str]:
    """Artefact metadata: deterministic, no timestamps, no paths."""
    return {
        "phase": phase,
        "spec_version": SPEC_VERSION,
        "packet_sha256": packet_sha256(packet_dir),
        "build_weights_sha256": _sha256((author_dir / BUILD_FILE).read_bytes()),
        "reference_policy_sha256": _sha256((author_dir / POLICY_FILE).read_bytes()),
        "circuits_sha256": _sha256((author_dir / CIRCUITS_FILE).read_bytes()),
    }


@contextlib.contextmanager
def _staged(files: Mapping[str, bytes], packet_dir: Path | None) -> Any:
    """A fresh private directory: ``author/`` holding exactly ``files``, ``packet/`` a copy of the packet files."""
    from .packet import PACKET_FILES

    with tempfile.TemporaryDirectory(prefix=_STAGE_PREFIX) as tmp:
        root = Path(tmp).resolve()
        (root / "author").mkdir()
        for name, data in files.items():
            (root / "author" / name).write_bytes(data)
        if packet_dir is not None:
            (root / "packet").mkdir()
            for name in PACKET_FILES:
                shutil.copyfile(packet_dir / name, root / "packet" / name)
        (root / "out").mkdir()
        yield root


def _run_child(args: list[str], timeout: float, label: str, stage: Path) -> dict[str, Any]:
    """Run ``_child_main(args)`` in a fresh interpreter (cwd ``stage/author``); its verdict is ``stage/out/result.json``."""
    src_root = str(HARNESS_SOURCE_DIR.parents[1])
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from agent_loop_lab.v03.runner import _child_main; sys.exit(_child_main(sys.argv[2:]))"
    )
    result_path = stage / "out" / RESULT_FILE
    cmd = [sys.executable, "-I", "-B", "-c", code, src_root, *args, str(result_path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False, cwd=stage / "author")
    except subprocess.TimeoutExpired as exc:
        raise BuildError(f"{label}: timed out after {timeout:.0f} s (wall clock)") from exc
    detail = ((r.stderr or "") + (r.stdout or "")).strip()[-4000:] or None
    result: Any = None
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            result = None
    if isinstance(result, dict) and isinstance(result.get("error"), str):
        raise BuildError(f"{label}: {result['error']}", detail)
    if r.returncode in (-signal.SIGXCPU, -signal.SIGKILL, 128 + signal.SIGXCPU):
        raise BuildError(f"{label}: CPU budget exceeded (limit {BUILD_CPU_BUDGET_S} s)", detail)
    if r.returncode != 0 or not isinstance(result, dict) or result.get("ok") is not True:
        # The child's own output is not quoted: author code can print to it.
        raise BuildError(f"{label}: fresh interpreter failed (exit {r.returncode}) without a harness verdict", detail)
    return result


def _build_in_child(phase: str, build_source: bytes, packet_dir: Path, metadata: Mapping[str, str], label: str) -> bytes:
    with _staged({BUILD_FILE: build_source}, packet_dir) as stage:
        out = stage / "out" / ARTIFACT_FILE
        _run_child(["build", phase, str(stage / "author"), str(stage / "packet"), str(out), json.dumps(dict(metadata))],
                   BUILD_TIMEOUT_S, label, stage)
        if not out.is_file():
            raise BuildError(f"{label}: no artefact was produced")
        return out.read_bytes()


def _child_finish(result_path: Path, body: dict[str, Any]) -> int:
    """Write the verdict and leave at once: no interpreter shutdown, so no author finaliser runs unguarded."""
    try:
        result_path.write_text(json.dumps(body), encoding="utf-8")
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0 if body.get("ok") is True else 1)
    return 1  # pragma: no cover - os._exit does not return


def _exec_author_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    code = compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    sys.modules[name] = module
    exec(code, module.__dict__)  # noqa: S102 - runs under the BUILD guard
    return module


_AUTHOR_FILES: set[str] = set()  # realpaths of the executed author copies (for traceback line numbers)


def _safe_type_name(obj: Any) -> str:
    """The class name if it is a builtin class, otherwise a fixed phrase (author code can name its own classes)."""
    import builtins

    cls = obj if isinstance(obj, type) else type(obj)
    if getattr(builtins, cls.__name__, None) is cls:
        return cls.__name__
    mod = getattr(cls, "__module__", "") or ""
    if mod.split(".")[0] == "torch" and cls.__name__.isidentifier() and len(cls.__name__) <= 40:
        try:
            import torch

            if any(getattr(sub, cls.__name__, None) is cls for sub in (torch, torch.nn)):
                return f"{mod}.{cls.__name__}"
        except Exception:  # noqa: BLE001
            pass
    return "an object of a non-builtin class"


def _author_lines(exc: BaseException) -> str:
    lines = []
    for fr in traceback.extract_tb(exc.__traceback__):
        try:
            fn = os.path.realpath(fr.filename)
        except (OSError, ValueError):
            continue
        if fn in _AUTHOR_FILES and fr.lineno is not None:
            lines.append(f"{os.path.basename(fn)}:{fr.lineno}")
    if not lines:
        return "outside the author file"
    return "at " + " < ".join(reversed(lines[-5:]))


def _author_exception(exc: BaseException) -> str:
    """Type and author-file line numbers of an exception raised by author code; never its message."""
    return f"{_safe_type_name(exc)} raised {_author_lines(exc)} (the message is not disclosed)"


def _step(label: str, fn: Any) -> Any:
    from .guards import TrainingViolation

    try:
        return fn()
    except BuildError:
        raise
    except TrainingViolation as exc:
        raise BuildError(f"{label}: rejected by the BUILD guard: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - author code may raise anything
        raise BuildError(f"{label}: {_author_exception(exc)}") from exc


def _validate_returned(sd: Any) -> None:
    """``artifact.validate_state_dict`` with an error text that names only template keys."""
    try:
        keys = list(sd.keys())
    except Exception as exc:  # noqa: BLE001
        raise BuildError(f"{BUILD_FILE}: state_dict() is not a mapping ({_author_exception(exc)})") from exc
    if keys != list(harness_model.PARAM_SHAPES):
        missing = [k for k in harness_model.PARAM_SHAPES if k not in keys]
        extra = sum(1 for k in keys if k not in harness_model.PARAM_SHAPES)
        raise BuildError(
            f"{BUILD_FILE}: state_dict keys differ from the template (missing {missing}, {extra} extra key(s), "
            "or a different order)"
        )
    try:
        harness_artifact.validate_state_dict(sd)
    except ValueError as exc:  # names only template keys, shapes and dtypes
        raise BuildError(f"{BUILD_FILE}: state_dict validation: {exc}") from exc


def _child_build(phase: str, author_dir: Path, packet_dir: Path, out: Path, metadata: dict[str, str]) -> dict[str, Any]:
    import resource

    import torch

    from .guards import TrainingViolation, audit_model, build_guard

    budget = int(os.environ.get("V03_BUILD_CPU_BUDGET_S", BUILD_CPU_BUDGET_S))  # test override only
    used = resource.getrusage(resource.RUSAGE_SELF)
    soft = int(math.ceil(used.ru_utime + used.ru_stime)) + budget
    resource.setrlimit(resource.RLIMIT_CPU, (soft, soft + 5))
    torch.set_num_threads(1)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(packet_dir))
    import model as packet_model  # the packet's verbatim copy (checked against the harness source)

    if Path(packet_model.__file__).resolve() != (packet_dir / "model.py").resolve():
        return {"error": "could not import the packet's model.py"}
    author_file = (author_dir / BUILD_FILE).resolve()
    _AUTHOR_FILES.add(str(author_file))
    gc.disable()  # no author finaliser runs from a collection outside the guard
    try:
        with contextlib.redirect_stdout(sys.stderr):
            try:
                with build_guard(model_classes=(packet_model.DoomLoopTransformer, harness_model.DoomLoopTransformer),
                                 read_roots=(packet_dir, author_file)):
                    module = _step(f"{BUILD_FILE} (module-level code)", lambda: _exec_author_module("build_weights", author_file))
                    build = getattr(module, "build", None)
                    if not callable(build):
                        raise BuildError(f"{BUILD_FILE}: does not define a callable build()")
                    result = _step(f"{BUILD_FILE}: build()", build)
                    if type(result) is not packet_model.DoomLoopTransformer:
                        raise BuildError(f"{BUILD_FILE}: build() returned {_safe_type_name(result)}, not the packet's DoomLoopTransformer")
                    sd = _step(f"{BUILD_FILE}: state_dict()", result.state_dict)
                    _validate_returned(sd)
                    _step(f"{BUILD_FILE}: audit of the returned model", lambda: audit_model(result))
                    harness = harness_artifact.load_state_dict_into(harness_model.zero_model(), sd)
                    audit_model(harness)
            except TrainingViolation as exc:
                raise BuildError(f"rejected by the BUILD guard: {exc}") from exc
        data = harness_artifact.serialize_state_dict(harness.state_dict(), metadata)
    except BuildError as exc:
        return {"error": str(exc)}
    out.write_bytes(data)
    return {"ok": True, "sha256": _sha256(data)}


class _PolicyTimeout(BaseException):
    pass


def smoke_input() -> tuple[list[str], list[int]]:
    """Surface tokens and event ordinals of ``SMOKE_BLOCK`` (a one-event prefix, BOS first)."""
    from . import tokenizer

    enc = tokenizer.encode_session([SMOKE_BLOCK])
    return list(enc.surface), list(enc.event_ord)


def _child_policy(author_dir: Path, inputs_path: Path | None, strict: bool) -> dict[str, Any]:
    """Call the author's ``reference_policy`` on each input under the BUILD guard.

    ``strict`` (the smoke call): the first failure is an error. Otherwise each failure gives ``None``
    (a disagreement); a guard violation is always an error."""
    import time

    from .guards import TrainingViolation, build_guard

    if inputs_path is None:
        inputs = [smoke_input()]
    else:
        raw = json.loads(inputs_path.read_text(encoding="utf-8"))
        inputs = [(list(map(str, t)), [int(x) for x in e]) for t, e in raw]
    sys.dont_write_bytecode = True
    author_file = (author_dir / POLICY_FILE).resolve()
    _AUTHOR_FILES.add(str(author_file))

    def on_alarm(signum: int, frame: Any) -> None:
        raise _PolicyTimeout()

    values: list[int | None] = []
    gc.disable()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            try:
                with build_guard(read_roots=(author_file,)):
                    module = _step(f"{POLICY_FILE} (module-level code)", lambda: _exec_author_module("author_reference_policy", author_file))
                    policy = getattr(module, "reference_policy", None)
                    if not callable(policy):
                        raise BuildError(f"{POLICY_FILE}: does not define a callable reference_policy()")
                    signal.signal(signal.SIGPROF, on_alarm)
                    for tokens, event_ord in inputs:
                        what = "smoke input" if strict else "input"
                        value: Any = None
                        failed: str | None = None
                        t0 = time.process_time()
                        signal.setitimer(signal.ITIMER_PROF, POLICY_CPU_LIMIT_S)
                        try:
                            value = policy(list(tokens), list(event_ord))
                        except _PolicyTimeout:
                            failed = f"{POLICY_FILE}: reference_policy() exceeded {POLICY_CPU_LIMIT_S} s of CPU on the {what}"
                        except BuildError:
                            raise
                        except TrainingViolation:
                            raise
                        except Exception as exc:  # noqa: BLE001
                            failed = f"{POLICY_FILE}: reference_policy({what}): {_author_exception(exc)}"
                        finally:
                            signal.setitimer(signal.ITIMER_PROF, 0)
                        if failed is None and time.process_time() - t0 > POLICY_CPU_LIMIT_S:
                            failed = f"{POLICY_FILE}: reference_policy() exceeded {POLICY_CPU_LIMIT_S} s of CPU on the {what}"
                        if failed is None and (type(value) is not int or not 0 <= value <= 3):
                            failed = (f"{POLICY_FILE}: reference_policy() must return an int in 0..3, got "
                                      f"{_safe_type_name(value)}")
                        if failed is not None:
                            if strict:
                                raise BuildError(failed)
                            values.append(None)
                        else:
                            values.append(int(value))
            except TrainingViolation as exc:
                raise BuildError(f"{POLICY_FILE}: rejected by the BUILD guard: {exc}") from exc
    except BuildError as exc:
        return {"error": str(exc)}
    return {"ok": True, "values": None if strict else values}


def _child_main(argv: list[str]) -> int:
    mode, result_path = argv[0], Path(argv[-1])
    try:
        if mode == "build":
            phase, author_dir, packet_dir, out = argv[1], Path(argv[2]), Path(argv[3]), Path(argv[4])
            body = _child_build(phase, author_dir, packet_dir, out, json.loads(argv[5]))
        elif mode == "smoke":
            body = _child_policy(Path(argv[1]), None, strict=True)
        elif mode == "policy":
            body = _child_policy(Path(argv[1]), Path(argv[2]), strict=False)
        else:
            body = {"error": f"unknown child mode {mode!r}"}
    except BaseException:  # noqa: BLE001 - report without quoting anything author code chose
        body = {"error": "the fresh interpreter failed outside the guarded steps"}
    return _child_finish(result_path, body)


def evaluate_policy(policy_file: str | Path, inputs: list[tuple[list[str], list[int]]], *,
                    timeout: float | None = None) -> list[int | None]:
    """Call an author ``reference_policy`` on each ``(surface tokens, event_ord)`` prefix, guarded (module docstring).

    Returns one value per input: the class, or ``None`` for a timeout (0.5 s of CPU), an exception or a
    value that is not an ``int`` in 0..3. Raises ``BuildError`` if the policy cannot be loaded or breaks
    a guard rule. Only ``reference_policy.py`` itself is copied into the child's private directory."""
    data = Path(policy_file).read_bytes()
    with _staged({POLICY_FILE: data}, None) as stage:
        inputs_path = stage / "out" / "inputs.json"
        inputs_path.write_text(json.dumps([[list(t), list(e)] for t, e in inputs]), encoding="utf-8")
        wall = timeout if timeout is not None else 120 + 2.0 * len(inputs)
        res = _run_child(["policy", str(stage / "author"), str(inputs_path)], wall, f"{POLICY_FILE} evaluation", stage)
    values = res.get("values")
    if not isinstance(values, list) or len(values) != len(inputs):
        raise BuildError(f"{POLICY_FILE} evaluation: malformed result")
    return [v if (type(v) is int and 0 <= v <= 3) else None for v in values]


# -- public API ---------------------------------------------------------------------------


def _author_file_hashes(author_dir: Path) -> dict[str, str]:
    names = sorted({n for files in REQUIRED_FILES.values() for n in files})
    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = _sha256((author_dir / name).read_bytes())
        except OSError:
            continue
    return out


_STAGE_RE = re.compile(r"[^\s'\"]*" + re.escape(_STAGE_PREFIX) + r"[^/\s'\"]*/(author|packet|out)")


def _redact(message: str, places: Mapping[str, Path]) -> str:
    pairs: list[tuple[str, str]] = []
    for placeholder, path in places.items():
        for form in {str(path), str(path.resolve())}:
            if Path(form).is_absolute():
                pairs.append((form, placeholder))
    for form, placeholder in sorted(pairs, key=lambda fp: -len(fp[0])):
        message = message.replace(form, placeholder)
    return _STAGE_RE.sub(lambda m: f"<{m.group(1)}>", message)


def _cap(message: str) -> str:
    if len(message) <= MAX_ERROR_CHARS:
        return message
    return message[: MAX_ERROR_CHARS - 40] + f" ... [cut: {len(message)} characters]"


def _tree_files(root: Path, skip: Path | None = None) -> dict[str, Any]:
    """Every entry under ``root`` (relative POSIX path -> sha256, or {"symlink": target}); ``__pycache__`` and
    ``skip`` excluded; symlinks are never followed."""
    out: dict[str, Any] = {}
    if not root.is_dir():
        return out
    skip_r = skip.resolve() if skip is not None else None
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        d = Path(dirpath)
        keep = []
        for name in sorted(dirnames):
            sub = d / name
            if name == "__pycache__" or (skip_r is not None and not sub.is_symlink() and _is_within(sub.resolve(), skip_r)):
                continue
            if sub.is_symlink():
                out[sub.relative_to(root).as_posix()] = {"symlink": os.readlink(sub)}
                continue
            keep.append(name)
        dirnames[:] = keep
        for name in sorted(filenames):
            f = d / name
            rel = f.relative_to(root).as_posix()
            if f.is_symlink():
                out[rel] = {"symlink": os.readlink(f)}
            elif f.is_file():
                out[rel] = _sha256(f.read_bytes())
    return dict(sorted(out.items()))


def _archive_tree(author_dir: Path, dest: Path, skip: Path | None) -> dict[str, Any]:
    """Copy the whole phase directory into ``dest`` (regular files and symlinks as links); return its hashes."""
    if dest.exists() or dest.is_symlink():
        shutil.rmtree(dest) if dest.is_dir() and not dest.is_symlink() else dest.unlink()
    dest.mkdir(parents=True)
    listing = _tree_files(author_dir, skip)
    for rel, h in listing.items():
        src, dst = author_dir / rel, dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(h, dict):
            os.symlink(h["symlink"], dst)
        else:
            shutil.copyfile(src, dst)
            if _sha256(dst.read_bytes()) != h:
                raise BuildError(f"{AUTHOR_OUTPUT_DIR}/{rel}: changed while it was archived")
    return listing


def _write_attempt(out_dir: Path, body: dict[str, Any]) -> Path:
    d = out_dir / ATTEMPTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    taken = [int(m.group(1)) for p in d.iterdir() if (m := _ATTEMPT_RE.match(p.name))]
    n = max(taken, default=0) + 1
    while True:
        path = d / f"{n:04d}.json"
        try:
            with open(path, "xb") as f:
                f.write(_canonical_json({"attempt": n, **body}))
            return path
        except FileExistsError:
            n += 1


def _sparsity_record(model: Any) -> dict[str, Any]:
    base = harness_artifact.sparsity(model)
    sd = model.state_dict()
    out: dict[str, Any] = {}
    for k in harness_model.PARAM_SHAPES:
        blocks = harness_artifact.nonzero_blocks(sd[k])
        out[k] = {
            "nonzero": base[k]["nonzero"],
            "slots": base[k]["slots"],
            "blocks": [
                {"rows": list(b["rows"]) if b["rows"] is not None else None,
                 "cols": [list(r) for r in b["cols"]],  # type: ignore[union-attr]
                 "nonzero": b["nonzero"]}
                for b in blocks
            ],
        }
    out["overall"] = {"nonzero": base["overall"]["nonzero"], "slots": base["overall"]["slots"]}
    return out


def build_phase(
    phase: str,
    author_dir: str | Path,
    packet_dir: str | Path,
    out_dir: str | Path,
    *,
    rebuild_check: bool = True,
    force: bool = False,
    registry_path: str | Path | None = "auto",
    source_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build, check and archive one v0.3 phase. Returns the build record (also in BUILD_RECORD.json).

    ``source_dir`` (default: this package's directory) holds the harness copies of the packet
    sources (``packet.PACKET_SOURCE_FILES``) that the packet must match byte for byte.
    """
    author_dir, packet_dir, out_dir = Path(author_dir), Path(packet_dir), Path(out_dir)
    src = Path(source_dir) if source_dir is not None else HARNESS_SOURCE_DIR
    attempt: dict[str, Any] = {"packet_sha256": None}
    try:
        record = _build_phase(phase, author_dir, packet_dir, out_dir, rebuild_check, force, registry_path, src, attempt)
    except Exception as exc:
        if _is_within(out_dir.resolve(), packet_dir.resolve()):
            raise
        places = {"<author>": author_dir, "<packet>": packet_dir, "<out>": out_dir}
        error = str(exc) if isinstance(exc, BuildError) else f"{type(exc).__name__}: {exc}"
        error = _cap(_redact(error, places))
        detail = getattr(exc, "detail", None)
        body = {**attempt, "phase": phase, "outcome": "rejected", "error": error,
                "author_files": _author_file_hashes(author_dir), "author_tree": _safe_tree(author_dir, out_dir)}
        if detail:
            body["operator_detail"] = _redact(detail, places)  # raw child output: never shown to the author
        try:
            _write_attempt(out_dir, body)
        except OSError as w:
            exc.add_note(f"the attempt record could not be written: {w}")
        raise BuildError(error, detail) from exc
    _write_attempt(out_dir, {**attempt, "phase": phase, "outcome": "built", "error": None,
                             "author_files": _author_file_hashes(author_dir),
                             "author_tree": record["author_output"]})
    return record


def _safe_tree(author_dir: Path, out_dir: Path) -> dict[str, Any] | None:
    try:
        return _tree_files(author_dir, out_dir)
    except OSError:
        return None


def _build_phase(phase: str, author_dir: Path, packet_dir: Path, out_dir: Path, rebuild_check: bool,
                 force: bool, registry_path: str | Path | None, source_dir: Path,
                 attempt: dict[str, Any]) -> dict[str, Any]:
    if phase not in PHASES:
        raise BuildError(f"unknown phase {phase!r}; expected one of {PHASES}")
    pkt = packet_dir.resolve()
    if _is_within(author_dir.resolve(), pkt) or _is_within(out_dir.resolve(), pkt):
        raise BuildError("the author and output directories must be outside the authoring packet directory")
    artifact_path = out_dir / ARTIFACT_FILE
    if artifact_path.exists() and not force:
        raise BuildError(f"{artifact_path} already exists; archives are immutable (use --force to overwrite)")
    registry = _resolve_registry(registry_path)
    try:
        _static_checks(phase, author_dir, packet_dir, registry, source_dir, attempt)
    except PacketError as exc:
        raise BuildError(f"authoring packet: {exc}") from exc
    metadata = _metadata(phase, author_dir, packet_dir)
    # One snapshot of the author's files: the build, the smoke call and the archive all use these bytes.
    snap = {name: (author_dir / name).read_bytes() for name in REQUIRED_FILES[phase]}
    if (_sha256(snap[BUILD_FILE]) != metadata["build_weights_sha256"]
            or _sha256(snap[POLICY_FILE]) != metadata["reference_policy_sha256"]
            or _sha256(snap[CIRCUITS_FILE]) != metadata["circuits_sha256"]):
        raise BuildError("author files changed during the build")
    data = _build_in_child(phase, snap[BUILD_FILE], packet_dir, metadata, "build (fresh interpreter)")
    if rebuild_check:
        again = _build_in_child(phase, snap[BUILD_FILE], packet_dir, metadata, "rebuild check (second fresh interpreter)")
        if again != data:
            raise BuildError(
                f"rebuild check: the second fresh-interpreter artefact differs (sha256 {_sha256(again)} vs "
                f"{_sha256(data)}); {BUILD_FILE} is not deterministic"
            )
    with _staged({POLICY_FILE: snap[POLICY_FILE]}, None) as stage:
        _run_child(["smoke", str(stage / "author")], SMOKE_TIMEOUT_S, f"{POLICY_FILE} smoke call", stage)
    if _metadata(phase, author_dir, packet_dir) != metadata:
        raise BuildError("author files changed during the build")
    sd, meta = harness_artifact.deserialize_state_dict(data)
    if meta != metadata:
        raise BuildError("artefact metadata differs from the checked author files")
    harness = harness_artifact.load_state_dict_into(harness_model.zero_model(), sd)

    out_dir.mkdir(parents=True, exist_ok=True)
    if artifact_path.exists() and not force:
        raise BuildError(f"{artifact_path} already exists; archives are immutable (use --force to overwrite)")
    files: dict[str, str] = {}
    for name in REQUIRED_FILES[phase]:
        dst = out_dir / name
        if not (dst.exists() and (author_dir / name).resolve() == dst.resolve()):
            dst.write_bytes(snap[name])
        files[name] = _sha256(dst.read_bytes())
        if files[name] != _sha256(snap[name]):
            raise BuildError(f"{name}: changed while it was archived")
    author_output = _archive_tree(author_dir, out_dir / AUTHOR_OUTPUT_DIR, out_dir)
    for name in REQUIRED_FILES[phase]:
        if author_output.get(name) != files[name]:
            raise BuildError(f"{name}: changed while the phase directory was archived")
    artifact_path.write_bytes(data)
    artifact_sha = _sha256(artifact_path.read_bytes())
    if artifact_sha != _sha256(data):
        raise BuildError("saved artefact differs from the checked build")
    harness_artifact.load_artifact(artifact_path)  # round trip
    (out_dir / ARTIFACT_SHA_FILE).write_bytes(f"{artifact_sha}  {ARTIFACT_FILE}\n".encode("ascii"))
    record: dict[str, Any] = {
        **metadata,
        "record_format_version": RECORD_FORMAT_VERSION,
        "artifact_file": ARTIFACT_FILE,
        "artifact_format": harness_artifact.FORMAT,
        "artifact_sha256": artifact_sha,
        "artifact_size_bytes": len(data),
        "files": dict(sorted(files.items())),
        "author_output_dir": AUTHOR_OUTPUT_DIR,
        "author_output": author_output,
        "sparsity": _sparsity_record(harness),
        "checks": {
            "packet_verified": True,
            "packet_sources_match_harness": list(PACKET_SOURCE_FILES),
            "packet_registered": registry is not None,
            "required_files_present": list(REQUIRED_FILES[phase]),
            "source_rules": "passed (build_weights.py, reference_policy.py)",
            "anti_training_scan": "passed",
            "build_guard": ("BUILD mode: anti-training, RNG ban, compute ban, no I/O; built from a private copy "
                            "of build_weights.py and the packet only"),
            "cpu_budget_s": BUILD_CPU_BUDGET_S,
            "wall_timeout_s": BUILD_TIMEOUT_S,
            "audit_model": "passed",
            "state_dict": "validated: exact keys, shapes, float32, finite",
            "rebuild_check": "byte-identical in a second fresh interpreter" if rebuild_check else "not performed",
            "reference_policy_smoke_call": SMOKE_NOTE,
        },
    }
    (out_dir / RECORD_FILE).write_bytes(_canonical_json(record))
    return record
