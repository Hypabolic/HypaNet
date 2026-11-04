"""``agent-loop-lab`` command-line interface (spec v0.2 §12, M0 subset).

Subcommands::

    inspect window <events.jsonl> [--endpoint SEQ] [--artifact PATH]
    inspect model <artifact.dlw>
    packet build [--spec PATH] [--model PATH] [--out DIR]
    packet verify DIR
    freeze write [--root DIR] [--force]
    freeze check [--root DIR]

Exit codes: 0 on success (a gate rejection is a valid result, so it is 0 too);
1 when ``packet verify`` or ``freeze check`` find problems; 2 for usage or
input errors (unreadable files, invalid events or sequences, malformed
artefacts, a packet or registry that cannot be written).

Inspection only: nothing here modifies weights, runs autograd or reads oracle
data. CIRCUITS.md labels and the reference-policy class (spec §12) are Phase A
deliverables and are reported as unavailable in M0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, TextIO

import numpy as np
import torch

from ..encode import EncodingError, describe_matrix, encode_event_matrix
from ..model import (
    PARAM_SHAPES,
    SPARSITY_TOLERANCE,
    DoomLoopMLP,
    artifact_sha256,
    load_artifact,
    predict,
    sparsity,
    zero_model,
)
from ..packet import PacketError, build_packet, verify_packet
from ..registry import RegistryError, check_registry, compute_freeze, write_registry
from ..schema import (
    CLASS_NAMES,
    EVENTS_PER_WINDOW,
    FEATURE_NAMES,
    FEATURES_PER_EVENT,
    CanonicalEvent,
    SchemaError,
)
from ..windows import SequenceError, gate, validate_sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPEC = REPO_ROOT / "agent_doom_loop_detector_spec_v0_2.md"
DEFAULT_MODEL = REPO_ROOT / "src" / "agent_loop_lab" / "model.py"
DEFAULT_PACKET_DIR = REPO_ROOT / "authoring_packet"

EXIT_OK = 0
EXIT_PROBLEMS = 1
EXIT_ERROR = 2


class CliError(Exception):
    """An expected, user-facing failure (bad input file, bad argument)."""


# -- helpers ------------------------------------------------------------------


def _fmt(v: float) -> str:
    return f"{float(v):+.6f}"


def _vector(values: Sequence[float]) -> str:
    return "[" + ", ".join(_fmt(v) for v in values) + "]"


def _input_name(col: int) -> str:
    pos, feat = divmod(col, FEATURES_PER_EVENT)
    return f"pos{pos}.{FEATURE_NAMES[feat]}"


def read_events(path: str | Path) -> list[CanonicalEvent]:
    """Parse one session of CanonicalEvent JSON lines (blank lines ignored)."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise CliError(f"cannot read {p}: {exc}") from exc
    events: list[CanonicalEvent] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CliError(f"{p}:{lineno}: invalid JSON: {exc}") from exc
        if not isinstance(obj, dict):
            raise CliError(f"{p}:{lineno}: expected a JSON object, got {type(obj).__name__}")
        try:
            events.append(CanonicalEvent.from_dict(obj))
        except (SchemaError, KeyError, TypeError) as exc:
            raise CliError(f"{p}:{lineno}: invalid CanonicalEvent: {exc!r}") from exc
    if not events:
        raise CliError(f"{p}: no events")
    return events


def _load_model(path: str | Path) -> tuple[DoomLoopMLP, dict]:
    try:
        return load_artifact(path)
    except OSError as exc:
        raise CliError(f"cannot read artefact {path}: {exc}") from exc
    except (ValueError, UnicodeDecodeError, KeyError, TypeError) as exc:
        raise CliError(f"malformed artefact {path}: {exc}") from exc


def _print_activations(model: DoomLoopMLP, x: np.ndarray, out: TextIO) -> None:
    with torch.no_grad():
        acts = model.activations(torch.as_tensor(x, dtype=torch.float32).unsqueeze(0))
    acts = {k: v.squeeze(0).cpu().numpy().astype(np.float64) for k, v in acts.items()}
    logits = acts["logits"]
    pred = int(predict(torch.as_tensor(logits)).item())

    print("raw logits:", file=out)
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {i} {name:<12} {_fmt(logits[i])}", file=out)
    tie = int(np.sum(logits == logits.max())) > 1
    note = " (tie between maximal logits; argmax takes the first index)" if tie else ""
    print(f"predicted class: {CLASS_NAMES[pred]} ({pred}){note}", file=out)
    print("reference-policy class: n/a in M0 (reference_policy.py is a Phase A deliverable)", file=out)

    print("activations:", file=out)
    for layer, pre in (("h1", "h1_pre"), ("h2", "h2_pre")):
        post = acts[layer]
        print(f"  {pre} ({len(acts[pre])}): {_vector(acts[pre])}", file=out)
        print(f"  {layer} ({len(post)}): {_vector(post)}", file=out)
        active = [i for i, v in enumerate(post) if v > 0.0]
        if active:
            listing = ", ".join(f"{layer}[{i}]={_fmt(post[i])}" for i in active)
            print(f"  active {layer} units ({len(active)}): {listing}", file=out)
        else:
            print(f"  active {layer} units (0): none", file=out)
    print(f"  logits (4): {_vector(logits)}", file=out)


# -- inspect window -----------------------------------------------------------


def cmd_inspect_window(args: argparse.Namespace, out: TextIO) -> int:
    events = read_events(args.events)
    try:
        validate_sequence(events)
    except SequenceError as exc:
        raise CliError(f"invalid event sequence: {exc}") from exc

    if args.endpoint is None:
        t = len(events) - 1
    else:
        positions = {ev.sequence_index: i for i, ev in enumerate(events)}
        if args.endpoint not in positions:
            raise CliError(
                f"no event with sequence_index {args.endpoint} "
                f"(file covers {events[0].sequence_index}..{events[-1].sequence_index})"
            )
        t = positions[args.endpoint]
    # Causality: nothing after the endpoint reaches the gate or the encoder.
    prefix = events[: t + 1]
    window = prefix[-EVENTS_PER_WINDOW:]
    start = len(prefix) - len(window)

    print(f"session: {prefix[-1].session_id}  agent: {prefix[-1].agent_id}  task: {prefix[-1].task_id}", file=out)
    print(
        f"endpoint sequence_index: {prefix[-1].sequence_index}  "
        f"(prefix length {len(prefix)}, history_complete={prefix[0].provenance.history_complete})",
        file=out,
    )
    print("window events:", file=out)
    for j, ev in enumerate(window):
        pos = j if len(window) == EVENTS_PER_WINDOW else "-"
        print(
            f"  pos {pos} seq {ev.sequence_index:>4}  {ev.operation_family:<14} "
            f"{ev.operation_category.value:<8} {ev.outcome.value:<7} {ev.event_id}",
            file=out,
        )

    decision = gate(prefix)
    print(f"gate: {decision.result}", file=out)
    if not decision.passed:
        for pos, reason in decision.reasons:
            if pos < 0:
                print(f"  window: {reason.value}", file=out)
            else:
                print(f"  pos {pos} (seq {prefix[start + pos].sequence_index}): {reason.value}", file=out)
        print("no inference: the gate rejected this window (spec §5.2)", file=out)
        return EXIT_OK

    try:
        matrix = encode_event_matrix(prefix)
    except EncodingError as exc:  # gate passed, so this is an internal inconsistency
        raise CliError(f"encoder rejected a gate-passing window: {exc}") from exc
    print("feature matrix (8 x 15):", file=out)
    print(describe_matrix(matrix), file=out)

    if args.artifact is None:
        model = zero_model()
        print("model: ZERO MODEL (no --artifact given; every parameter is exactly 0.0)", file=out)
    else:
        model, _ = _load_model(args.artifact)
        print(f"model: {args.artifact}  sha256 {artifact_sha256(args.artifact)}", file=out)
    _print_activations(model, matrix.reshape(-1), out)
    return EXIT_OK


# -- inspect model ------------------------------------------------------------


def cmd_inspect_model(args: argparse.Namespace, out: TextIO) -> int:
    model, meta = _load_model(args.artifact)
    print(f"artefact: {args.artifact}", file=out)
    print(f"sha256: {artifact_sha256(args.artifact)}", file=out)
    print(f"metadata: {json.dumps(meta, sort_keys=True, ensure_ascii=False)}", file=out)
    print(f"sparsity (non-zero means |w| > {SPARSITY_TOLERANCE:g}):", file=out)
    report = sparsity(model)
    for key in (*PARAM_SHAPES, "overall"):
        r = report[key]
        print(f"  {key:<11} {r['nonzero']:>5}/{r['slots']:<5} density {r['density']:.6f}", file=out)
    print("CIRCUITS.md labels: n/a in M0 (CIRCUITS.md is a Phase A deliverable)", file=out)

    sd = model.state_dict()
    print("non-zero parameters:", file=out)
    any_nz = False
    for key in PARAM_SHAPES:
        arr = sd[key].detach().cpu().numpy().astype(np.float64)
        idx = np.argwhere(np.abs(arr) > SPARSITY_TOLERANCE)
        print(f"  {key}: {len(idx)}", file=out)
        layer = key.split(".")[0]
        for ij in idx:
            any_nz = True
            v = arr[tuple(ij)]
            if key == "fc1.weight":
                label = f"h1[{ij[0]}] <- {_input_name(int(ij[1]))}"
            elif key == "fc2.weight":
                label = f"h2[{ij[0]}] <- h1[{ij[1]}]"
            elif key == "fc3.weight":
                label = f"{CLASS_NAMES[ij[0]]} <- h2[{ij[1]}]"
            elif key == "fc3.bias":
                label = f"{CLASS_NAMES[ij[0]]} bias"
            else:
                label = f"{'h1' if layer == 'fc1' else 'h2'}[{ij[0]}] bias"
            coords = ", ".join(str(int(c)) for c in ij)
            print(f"    {key}[{coords}] = {_fmt(v)}  ({label})", file=out)
    if not any_nz:
        print("  no non-zero parameters", file=out)
    return EXIT_OK


# -- packet -------------------------------------------------------------------


def cmd_packet_build(args: argparse.Namespace, out: TextIO) -> int:
    try:
        manifest = build_packet(args.spec, args.model, args.out)
    except (PacketError, OSError, UnicodeDecodeError) as exc:
        raise CliError(f"packet build failed: {exc}") from exc
    print(f"packet written to {args.out}", file=out)
    for name, sha in manifest["files"].items():
        print(f"  {sha}  {name}", file=out)
    print(f"source spec sha256: {manifest['source_spec_sha256']}", file=out)
    print(f"packet sha256: {manifest['packet_sha256']}", file=out)
    return EXIT_OK


def cmd_packet_verify(args: argparse.Namespace, out: TextIO) -> int:
    problems = verify_packet(args.dir)
    for p in problems:
        print(f"PROBLEM: {p}", file=out)
    print("packet OK" if not problems else f"packet verification FAILED: {len(problems)} problem(s)", file=out)
    return EXIT_PROBLEMS if problems else EXIT_OK


# -- freeze -------------------------------------------------------------------


def cmd_freeze_write(args: argparse.Namespace, out: TextIO) -> int:
    try:
        path = write_registry(args.root, overwrite=args.force)
        freeze = compute_freeze(args.root)
    except (RegistryError, OSError) as exc:
        raise CliError(f"freeze write failed: {exc}") from exc
    print(f"registry written to {path}", file=out)
    for rel, sha in sorted(freeze["files"].items()):
        print(f"  {sha}  {rel}", file=out)
    print(f"packet sha256: {freeze['packet_sha256']}", file=out)
    print(f"freeze sha256: {freeze['freeze_sha256']}", file=out)
    return EXIT_OK


def cmd_freeze_check(args: argparse.Namespace, out: TextIO) -> int:
    problems = check_registry(args.root)
    for p in problems:
        print(f"MISMATCH: {p}", file=out)
    print("registry OK" if not problems else f"registry check FAILED: {len(problems)} problem(s)", file=out)
    return EXIT_PROBLEMS if problems else EXIT_OK


# -- argument parsing ---------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="agent-loop-lab",
        description="Doom-loop detector diagnostics (spec v0.2 §12, M0 subset).",
    )
    groups = ap.add_subparsers(dest="group", metavar="{inspect,packet,freeze}", required=True)

    insp = groups.add_parser("inspect", help="inspect a window or a weight artefact")
    insp_sub = insp.add_subparsers(dest="command", metavar="{window,model}", required=True)
    w = insp_sub.add_parser("window", help="gate, features and model output for one window")
    w.add_argument("events", type=Path, help="CanonicalEvent JSON lines for one session")
    w.add_argument("--endpoint", type=int, default=None, metavar="SEQ",
                   help="sequence_index of the window's last event (default: last event)")
    w.add_argument("--artifact", type=Path, default=None, metavar="PATH",
                   help=".dlw artefact (default: the zero model)")
    w.set_defaults(func=cmd_inspect_window)
    m = insp_sub.add_parser("model", help="hash, metadata, sparsity and non-zero parameters")
    m.add_argument("artifact", type=Path)
    m.set_defaults(func=cmd_inspect_model)

    pk = groups.add_parser("packet", help="build or verify the authoring packet")
    pk_sub = pk.add_subparsers(dest="command", metavar="{build,verify}", required=True)
    b = pk_sub.add_parser("build", help="build the authoring packet")
    b.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    b.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    b.add_argument("--out", type=Path, default=DEFAULT_PACKET_DIR, metavar="DIR")
    b.set_defaults(func=cmd_packet_build)
    v = pk_sub.add_parser("verify", help="verify an authoring packet")
    v.add_argument("dir", type=Path)
    v.set_defaults(func=cmd_packet_verify)

    fz = groups.add_parser("freeze", help="write or check experiments/registry.yaml")
    fz_sub = fz.add_subparsers(dest="command", metavar="{write,check}", required=True)
    fw = fz_sub.add_parser("write", help="write the freeze registry")
    fw.add_argument("--root", type=Path, default=REPO_ROOT, metavar="DIR")
    fw.add_argument("--force", action="store_true", help="replace a registry whose content differs")
    fw.set_defaults(func=cmd_freeze_write)
    fc = fz_sub.add_parser("check", help="check frozen files against the registry")
    fc.add_argument("--root", type=Path, default=REPO_ROOT, metavar="DIR")
    fc.set_defaults(func=cmd_freeze_check)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # --help or a usage error
        code = exc.code
        return code if isinstance(code, int) else (EXIT_OK if code is None else EXIT_ERROR)
    try:
        return args.func(args, sys.stdout)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
