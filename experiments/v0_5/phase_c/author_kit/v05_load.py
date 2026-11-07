"""Phase C helper (read-only): load Phase B's network (and r1's), apply an edits file on top of Phase B's weights,
run a network on the disclosed windows.

Import it from any script you run with ``./py`` (``tools/lib/`` is on the module path, after ``packet/``)::

    import v05_load
    b = v05_load.load_b()                                     # Phase B's network: your base
    rev, report = v05_load.apply_edits_file()                 # B + output/v05-phase-c/REVISION.edits
    for w in v05_load.windows("all"):                         # V001..V200, then W001..W200
        c = v05_load.classify(rev, w["blocks"])               # class index at EOE_t (the last block)
        print(w["id"], w["oracle_label"], w["network_class"], model.CLASS_NAMES[c])

or from the command line::

    ./py -m v05_load check [FILE]    # apply FILE (default output/v05-phase-c/REVISION.edits) on top of Phase B's
                                     # weights the way the harness will, and say whether the build would accept it
    ./py -m v05_load b               # check phase_b/artifact.dlw against phase_b/artifact.sha256 and against r1's
                                     # weights + phase_b/REVISION.edits applied by the revision applier
    ./py -m v05_load r1              # check r1/artifact.dlw against r1/artifact.sha256 and against the reader's
                                     # rebuild from r1/replies/

Functions:

* ``b_state()`` / ``load_b()``: Phase B's weights (``phase_b/artifact.dlw``, checked against
  ``phase_b/artifact.sha256``) as a state dict / a model. This is the base your revision is applied to;
* ``r1_state()`` / ``load_r1()``: r1's weights (``r1/artifact.dlw``, checked against ``r1/artifact.sha256``);
* ``apply_edits_text(text, base=None)`` / ``apply_edits_file(path=None, base=None)``: a model with the
  statements applied in order on top of ``base`` (a state dict; default **Phase B's weights**), and the report.
  This is ``hn_v05.revision_c.apply_revision_c`` (Phase B's ``hn_v05.revision.apply_revision``, unchanged), the
  same code the harness runs to build your revision;
* ``build_problems(report)``: why the harness would reject that file (an empty list means it would accept it);
* ``windows(which="V", path=None)``: the disclosed windows, each a dict with ``id``, ``oracle_label``,
  ``network_class``, ``prefix`` (as disclosed), ``blocks`` (the rendered blocks 0..t, window marker removed),
  ``start`` (t-7) and ``t``. ``which``: ``"V"`` (``disclosure/DISCLOSURE.json``, V001..V200), ``"W"``
  (``disclosure/DISCLOSURE_W.json``, W001..W200) or ``"all"`` (V then W). ``windows_w()`` is ``windows("W")``;
* ``encode(blocks)``, ``eoe_logits(m, blocks)`` ([number of blocks, 4], one causal pass), ``classify(m, blocks)``
  (the argmax at the last block's EOE; ties go to the first class), ``activations(m, blocks, ...)``;
* ``r1_rebuild()``: r1's artefact rebuilt by the frozen reader from ``r1/replies/``; ``b_rebuild_state()``:
  r1's weights with ``phase_b/REVISION.edits`` applied by the revision applier.

Nothing here searches, fits or changes weights by itself: it only loads, applies what you wrote, and runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import OrderedDict

import torch

import model
import tokenizer
from hn_v05 import dlw3
from hn_v05.edits import build_run
from hn_v05.revision import build_problems as _build_problems
from hn_v05.revision_c import apply_revision_c

LIB = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(os.path.dirname(LIB))
R1_DIR = os.path.join(ROOT, "r1")
R1_ARTIFACT = os.path.join(R1_DIR, "artifact.dlw")
R1_SHA_FILE = os.path.join(R1_DIR, "artifact.sha256")
B_DIR = os.path.join(ROOT, "phase_b")
B_ARTIFACT = os.path.join(B_DIR, "artifact.dlw")
B_SHA_FILE = os.path.join(B_DIR, "artifact.sha256")
B_REVISION = os.path.join(B_DIR, "REVISION.edits")
OUTPUT = os.path.join(ROOT, "output", "v05-phase-c")
DEFAULT_REVISION = os.path.join(OUTPUT, "REVISION.edits")
DISCLOSURES = {
    "V": os.path.join(ROOT, "disclosure", "DISCLOSURE.json"),
    "W": os.path.join(ROOT, "disclosure", "DISCLOSURE_W.json"),
}
WINDOW_MARKER = "### window start"
CLASS_NAMES = model.CLASS_NAMES


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_sha_file(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().split()[0].strip().lower()


def _checked_bytes(artifact: str, sha_file: str, what: str) -> bytes:
    with open(artifact, "rb") as f:
        data = f.read()
    want = _read_sha_file(sha_file)
    if _sha256(data) != want:
        raise RuntimeError(f"{what} sha256 {_sha256(data)} != {os.path.relpath(sha_file, ROOT)} {want}")
    return data


def r1_bytes() -> bytes:
    """r1/artifact.dlw, after checking it against r1/artifact.sha256."""
    return _checked_bytes(R1_ARTIFACT, R1_SHA_FILE, "r1/artifact.dlw")


def b_bytes() -> bytes:
    """phase_b/artifact.dlw, after checking it against phase_b/artifact.sha256."""
    return _checked_bytes(B_ARTIFACT, B_SHA_FILE, "phase_b/artifact.dlw")


def r1_state() -> "OrderedDict[str, torch.Tensor]":
    """r1's weights as a state dict (fresh tensors each call)."""
    sd, _meta = dlw3.deserialize_state_dict(r1_bytes())
    return sd


def b_state() -> "OrderedDict[str, torch.Tensor]":
    """Phase B's weights as a state dict (fresh tensors each call): the base of your revision."""
    sd, _meta = dlw3.deserialize_state_dict(b_bytes())
    return sd


def model_from_state(sd) -> model.DoomLoopTransformer:
    return dlw3.load_state_dict_into(model.zero_model(), sd)


def load_r1() -> model.DoomLoopTransformer:
    """r1's network."""
    return model_from_state(r1_state())


def load_b() -> model.DoomLoopTransformer:
    """Phase B's network."""
    return model_from_state(b_state())


def apply_edits_text(text: str, base=None):
    """``(model, report)``: ``text``'s statements applied in order on top of ``base`` (default Phase B's weights)."""
    sd, report = apply_revision_c(b_state() if base is None else base, text)
    return model_from_state(sd), report


def apply_edits_file(path: str | None = None, base=None):
    """``(model, report)`` for an edits file (default output/v05-phase-c/REVISION.edits), on top of ``base``
    (default Phase B's weights)."""
    with open(path or DEFAULT_REVISION, "rb") as f:
        text = f.read().decode("utf-8")
    return apply_edits_text(text, base)


def build_problems(report) -> list[str]:
    """Why the harness would reject the revision this report describes (empty: it would accept it)."""
    return _build_problems(report)


def parse_prefix(text: str) -> tuple[list[str], int]:
    """A disclosed ``prefix`` -> ``(blocks 0..t, index of the marked block t-7)``. The marker line is removed;
    after that, every line that does not start with a space starts a new block."""
    blocks: list[list[str]] = []
    marked = None
    pending = False
    for line in text.split("\n"):
        if line == WINDOW_MARKER:
            if marked is not None or pending:
                raise ValueError("more than one window marker")
            pending = True
            continue
        if not line.startswith(" "):
            blocks.append([line])
            if pending:
                marked, pending = len(blocks) - 1, False
        else:
            if not blocks or pending:
                raise ValueError("body line outside a block")
            blocks[-1].append(line)
    if marked is None:
        raise ValueError("no window marker")
    return ["\n".join(b) for b in blocks], marked


def _load_windows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    out = []
    for w in doc["windows"]:
        blocks, start = parse_prefix(w["prefix"])
        out.append({**w, "blocks": blocks, "start": start, "t": len(blocks) - 1})
    return out


def windows(which: str = "V", path: str | None = None) -> list[dict]:
    """The disclosed windows, in file order, with their blocks recovered from the prefix.

    ``which``: ``"V"`` (DISCLOSURE.json), ``"W"`` (DISCLOSURE_W.json) or ``"all"`` (V then W); ``path`` reads
    that one file instead."""
    if path is not None:
        return _load_windows(path)
    if which == "all":
        return _load_windows(DISCLOSURES["V"]) + _load_windows(DISCLOSURES["W"])
    if which not in DISCLOSURES:
        raise ValueError(f"which must be 'V', 'W' or 'all', not {which!r}")
    return _load_windows(DISCLOSURES[which])


def windows_w() -> list[dict]:
    """The W windows (disclosure/DISCLOSURE_W.json)."""
    return windows("W")


def encode(blocks: list[str]) -> tokenizer.Encoded:
    return tokenizer.encode_session(list(blocks))


def _tensors(enc):
    return (torch.tensor(enc.tokens, dtype=torch.long), torch.tensor(enc.event_ord, dtype=torch.long),
            torch.tensor(enc.offset, dtype=torch.long))


def eoe_logits(m: model.DoomLoopTransformer, blocks: list[str]) -> torch.Tensor:
    """Logits ``[len(blocks), 4]`` at each block's EOE, from one causal forward pass."""
    enc = encode(blocks)
    with torch.no_grad():
        logits = m(*_tensors(enc))
    return logits[torch.tensor(enc.eoe_index, dtype=torch.long)]


def classify(m: model.DoomLoopTransformer, blocks: list[str]) -> int:
    """The class index at the last block's EOE (argmax; ties go to the first class)."""
    return int(model.predict(eoe_logits(m, blocks)[-1:])[0])


def activations(m: model.DoomLoopTransformer, blocks: list[str], query_positions=None) -> dict:
    """``model.activations`` for these blocks; ``eoe_index`` and ``surface`` are added to the result."""
    enc = encode(blocks)
    with torch.no_grad():
        act = m.activations(*_tensors(enc), query_positions=query_positions)
    act["eoe_index"] = list(enc.eoe_index)
    act["surface"] = list(enc.surface)
    return act


def r1_rebuild() -> bytes:
    """r1's artefact rebuilt by the frozen reader (hn_v05.edits.build_run) from r1/replies/."""
    data, _sha, _report = build_run(R1_DIR)
    return data


def b_rebuild_state():
    """``(state dict, report)``: r1's weights with phase_b/REVISION.edits applied by the revision applier."""
    with open(B_REVISION, "rb") as f:
        text = f.read().decode("utf-8")
    return apply_revision_c(r1_state(), text)


def _cmd_check(path: str | None) -> int:
    p = path or DEFAULT_REVISION
    _m, report = apply_edits_file(p)
    er = report["edit_report"]
    print(f"{os.path.relpath(os.path.realpath(p), ROOT)}: sha256 {report['revision_sha256']}")
    print(f"  applied on top of phase_b/artifact.dlw ({_sha256(b_bytes())})")
    print(f"  statements {er['statements']}, applied {er['applied']}, skipped {er['skipped']}, "
          f"entries written {er['entries_written']}")
    print(f"  entries changed from the base: {report['changed_entries_total']} "
          f"(non-zero parameters {report['nonzero_before_total']} -> {report['nonzero_after_total']})")
    for k, n in report["changed_entries"].items():
        print(f"    {k}: {n}")
    problems = build_problems(report)
    if problems:
        print("  THE HARNESS WOULD REJECT THIS FILE:")
        for line in problems:
            print("    " + line)
        return 1
    print("  every statement applies: the harness would accept this file")
    return 0


def _cmd_r1() -> int:
    data = r1_bytes()
    print(f"r1/artifact.dlw matches r1/artifact.sha256 ({_sha256(data)})")
    same = r1_rebuild() == data
    print("rebuild from r1/replies/ with the frozen reader: " + ("identical" if same else "DIFFERENT"))
    return 0 if same else 1


def _cmd_b() -> int:
    data = b_bytes()
    print(f"phase_b/artifact.dlw matches phase_b/artifact.sha256 ({_sha256(data)})")
    sd, report = b_rebuild_state()
    have, _meta = dlw3.deserialize_state_dict(data)
    same = list(sd) == list(have) and all(torch.equal(sd[k], have[k]) for k in have)
    skipped = report["edit_report"]["skipped"] + len(report["structural_problems"])
    print(f"r1/artifact.dlw + phase_b/REVISION.edits ({report['edit_report']['statements']} statements, "
          f"{skipped} skipped): " + ("identical weights" if same and not skipped else "DIFFERENT"))
    return 0 if same and not skipped else 1


def main(argv: list[str]) -> int:
    if argv[:1] == ["check"] and len(argv) <= 2:
        return _cmd_check(argv[1] if len(argv) == 2 else None)
    if argv == ["r1"]:
        return _cmd_r1()
    if argv == ["b"]:
        return _cmd_b()
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
