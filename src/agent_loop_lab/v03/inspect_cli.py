"""v0.3 inspector (spec v0.3 §12). Inspection only: nothing here changes weights.

Usage::

    python -m agent_loop_lab.v03.inspect_cli inspect log   SESSION [--artifact A.dlw]
    python -m agent_loop_lab.v03.inspect_cli inspect heads SESSION --event T [--artifact A.dlw] [--layer L] [--top K]
    python -m agent_loop_lab.v03.inspect_cli inspect model [A.dlw] [--max-blocks N]

``SESSION`` is one of:

* a ``.jsonl`` file with one JSON string per line, each a rendered event block;
* a ``.json`` file holding a list of block strings, or ``{"blocks": [...]}``;
* a ``.json`` file holding an already encoded session ``{"tokens",
  "event_ord", "offset", "eoe_index"[, "surface"]}``.

Block files are tokenised with ``agent_loop_lab.v03.tokenizer``. Event ``T``
is the 0-based block index; its tokens have ``event_ord == T + 1``. Without
``--artifact`` the all-zero model is used. ``CIRCUITS.md`` labels (spec §12)
come with a Phase A artefact and are not shown here.

Exit codes: 0 success, 2 usage or input errors.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence, TextIO

import torch

from .artifact import SPARSITY_TOLERANCE, artifact_sha256, format_ranges, load_artifact, nonzero_blocks, sparsity
from .infer import encode_blocks
from .model import CLASS_NAMES, H, PARAM_SHAPES, DoomLoopTransformer, predict, zero_model

EXIT_OK = 0
EXIT_ERROR = 2


class CliError(Exception):
    """An expected, user-facing failure."""


@dataclass
class Session:
    tokens: list[int]
    event_ord: list[int]
    offset: list[int]
    eoe_index: list[int]
    surface: list[str]
    blocks: list[str] | None = field(default=None)


def _load_session(path: Path) -> Session:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CliError(f"cannot read {path}: {exc}") from exc
    try:
        if path.suffix == ".jsonl":
            doc: Any = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CliError(f"{path}: not valid JSON: {exc}") from exc
    if isinstance(doc, dict) and "tokens" in doc:
        try:
            tokens = [int(x) for x in doc["tokens"]]
            sess = Session(
                tokens=tokens,
                event_ord=[int(x) for x in doc["event_ord"]],
                offset=[int(x) for x in doc["offset"]],
                eoe_index=[int(x) for x in doc["eoe_index"]],
                surface=[str(x) for x in doc.get("surface", [""] * len(tokens))],
                blocks=doc.get("blocks"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CliError(f"{path}: malformed encoded session: {exc}") from exc
        if not (len(sess.tokens) == len(sess.event_ord) == len(sess.offset) == len(sess.surface)):
            raise CliError(f"{path}: tokens/event_ord/offset/surface lengths differ")
        return sess
    if isinstance(doc, dict) and "blocks" in doc:
        doc = doc["blocks"]
    if not isinstance(doc, list) or not all(isinstance(b, str) for b in doc):
        raise CliError(f"{path}: expected a list of block strings or an encoded session")
    try:
        enc = encode_blocks(doc)
    except RuntimeError as exc:
        raise CliError(str(exc)) from exc
    except ValueError as exc:
        raise CliError(f"{path}: tokeniser rejected the log: {exc}") from exc
    return Session(
        tokens=list(enc.tokens),
        event_ord=list(enc.event_ord),
        offset=list(enc.offset),
        eoe_index=list(enc.eoe_index),
        surface=list(enc.surface),
        blocks=list(doc),
    )


def _load_model(path: str | None) -> tuple[DoomLoopTransformer, str]:
    if path is None:
        return zero_model(), "zero model"
    try:
        model, _ = load_artifact(path)
    except OSError as exc:
        raise CliError(f"cannot read {path}: {exc}") from exc
    except ValueError as exc:
        raise CliError(f"{path}: invalid artefact: {exc}") from exc
    return model, f"{path} (sha256 {artifact_sha256(path)})"


def _tensors(s: Session) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(s.tokens, dtype=torch.long),
        torch.tensor(s.event_ord, dtype=torch.long),
        torch.tensor(s.offset, dtype=torch.long),
    )


def _forward(model: DoomLoopTransformer, s: Session) -> torch.Tensor:
    try:
        with torch.no_grad():
            return model(*_tensors(s))
    except ValueError as exc:
        raise CliError(f"model rejected the session: {exc}") from exc


def _surf(text: str) -> str:
    """Surface string with newlines and other control characters escaped."""
    return text.encode("unicode_escape").decode("ascii")


def _event_label(eo: int) -> str:
    return "BOS" if eo == 0 else f"e{eo - 1}"


def _fmt_logits(row: torch.Tensor) -> str:
    return " ".join(f"{name[:4]}={float(v):+.4f}" for name, v in zip(CLASS_NAMES, row))


# -- commands -----------------------------------------------------------------


def cmd_log(args: argparse.Namespace, out: TextIO) -> int:
    sess = _load_session(Path(args.session))
    model, desc = _load_model(args.artifact)
    logits = _forward(model, sess)
    out.write(f"model: {desc}\n")
    out.write(f"tokens: {len(sess.tokens)}  events: {len(sess.eoe_index)}\n")
    for t, eoe in enumerate(sess.eoe_index):
        idx = [i for i, eo in enumerate(sess.event_ord) if eo == t + 1]
        out.write(f"\n== event {t} (positions {idx[0] if idx else '?'}..{idx[-1] if idx else '?'})\n")
        if sess.blocks is not None and t < len(sess.blocks):
            for line in sess.blocks[t].split("\n"):
                out.write(f"  | {line}\n")
        out.write("  ids: " + " ".join(f"{sess.tokens[i]}:{_surf(sess.surface[i])}" for i in idx) + "\n")
        row = logits[eoe]
        out.write(f"  EOE@{eoe}: {_fmt_logits(row)} -> {CLASS_NAMES[int(predict(row))]}\n")
    return EXIT_OK


def cmd_heads(args: argparse.Namespace, out: TextIO) -> int:
    sess = _load_session(Path(args.session))
    model, desc = _load_model(args.artifact)
    t = args.event
    if not 0 <= t < len(sess.eoe_index):
        raise CliError(f"--event {t} outside [0, {len(sess.eoe_index)})")
    q = sess.eoe_index[t]
    try:
        with torch.no_grad():
            acts = model.activations(*_tensors(sess), query_positions=[q])
    except ValueError as exc:
        raise CliError(f"model rejected the session: {exc}") from exc
    eo = torch.tensor(sess.event_ord, dtype=torch.long)
    n_groups = int(eo.max()) + 1
    layers = range(len(acts["attn"])) if args.layer is None else [args.layer]
    out.write(f"model: {desc}\n")
    out.write(f"query: EOE of event {t} at position {q}\n")
    for li in layers:
        if not 0 <= li < len(acts["attn"]):
            raise CliError(f"--layer {li} outside [0, {len(acts['attn'])})")
        a = acts["attn"][li][:, 0, :]  # [H, n]
        out.write(f"\n-- layer {li}\n")
        for h in range(H):
            mass = torch.zeros(n_groups).index_add_(0, eo, a[h])
            top_ev = torch.argsort(mass, descending=True, stable=True)[: args.top]
            ev_txt = ", ".join(f"{_event_label(int(g))}:{float(mass[g]):.3f}" for g in top_ev if float(mass[g]) > 0)
            top_tok = torch.argsort(a[h], descending=True, stable=True)[: args.top]
            tok_txt = ", ".join(
                f"{int(j)}[{_event_label(sess.event_ord[int(j)])}] {_surf(sess.surface[int(j)]) or sess.tokens[int(j)]}:{float(a[h, j]):.3f}"
                for j in top_tok
            )
            out.write(f"  head {h}: events {ev_txt}\n")
            out.write(f"          tokens {tok_txt}\n")
    return EXIT_OK


def cmd_model(args: argparse.Namespace, out: TextIO) -> int:
    model, desc = _load_model(args.artifact)
    sd = model.state_dict()
    rep = sparsity(model)
    out.write(f"model: {desc}\n")
    out.write(f"non-zero (|w| > {SPARSITY_TOLERANCE:g}): {rep['overall']['nonzero']} / {rep['overall']['slots']}\n")
    for k in PARAM_SHAPES:
        nz = rep[k]["nonzero"]
        if nz == 0:
            continue
        out.write(f"\n{k} {tuple(sd[k].shape)}: {nz} non-zero\n")
        blocks = nonzero_blocks(sd[k])
        for blk in blocks[: args.max_blocks]:
            if blk["rows"] is None:
                out.write(f"  idx {format_ranges(blk['cols'])}\n")
            else:
                r0, r1 = blk["rows"]
                rows = f"{r0}" if r0 == r1 else f"{r0}-{r1}"
                out.write(f"  rows {rows}: cols {format_ranges(blk['cols'])} ({blk['nonzero']})\n")
        if len(blocks) > args.max_blocks:
            out.write(f"  ... {len(blocks) - args.max_blocks} more blocks\n")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m agent_loop_lab.v03.inspect_cli", description="v0.3 inspector")
    top = p.add_subparsers(dest="group", required=True)
    insp = top.add_parser("inspect")
    sub = insp.add_subparsers(dest="cmd", required=True)

    lg = sub.add_parser("log", help="rendered log with token ids and per-EOE logits")
    lg.add_argument("session")
    lg.add_argument("--artifact")
    lg.set_defaults(fn=cmd_log)

    hd = sub.add_parser("heads", help="per-head attention from EOE_t, grouped by source event")
    hd.add_argument("session")
    hd.add_argument("--event", type=int, required=True)
    hd.add_argument("--artifact")
    hd.add_argument("--layer", type=int)
    hd.add_argument("--top", type=int, default=5)
    hd.set_defaults(fn=cmd_heads)

    md = sub.add_parser("model", help="non-zero blocks per tensor")
    md.add_argument("artifact", nargs="?")
    md.add_argument("--max-blocks", type=int, default=20)
    md.set_defaults(fn=cmd_model)
    return p


def main(argv: Sequence[str] | None = None, out: TextIO | None = None, err: TextIO | None = None) -> int:
    out = out or sys.stdout
    err = err or sys.stderr
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return EXIT_OK if exc.code == 0 else EXIT_ERROR
    try:
        return args.fn(args, out)
    except CliError as exc:
        err.write(f"error: {exc}\n")
        return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
