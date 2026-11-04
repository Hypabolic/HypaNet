"""v0.3 authoring-packet builder and verifier (spec v0.3 §8, §9, §11 "Leakage").

The packet is the only input the v0.3 weight author sees:

- ``SPEC.md``: the ``[packet]``-marked sections of the v0.3 spec (marker
  stripped), then v0.2 §6.1-§6.3 (latent state, oracle parameters, labelling
  rules) and the v0.2 §7.1 family names. v0.2 §6.4 is excluded;
- ``model.py``, ``tokenizer.py``, ``renderer.py``: verbatim copies of
  ``src/agent_loop_lab/v03/`` (each is standalone). ``rawop.py`` is **not**
  shipped (spec §8: its docstring describes the generators); the renderer reads
  plain dicts and ``AUTHOR_INSTRUCTIONS.md`` lists their fields;
- ``state_dict_template.json``: tensor names, shapes and parameter count;
- ``AUTHOR_INSTRUCTIONS.md``: the operational protocol (:data:`AUTHOR_INSTRUCTIONS`);
- ``GRAMMAR_SHEET.md``: one invented example block per tool, with its tokens;
- ``PACKET_MANIFEST.json``: sha256 of every file above and of the sources.

The **packet sha256** is the sha256 of the ``PACKET_MANIFEST.json`` bytes.
Everything is deterministic (no timestamps, no absolute paths, sorted JSON,
LF line endings), so building twice gives byte-identical files.

Leakage checks (``verify_packet``): :data:`SPEC_SENTINELS` (phrases from
withheld text) must occur in no packet file, and :data:`GENERATOR_WORDS`
(``engine``, ``generator``, ``OOD``, ``oracle``, ``latent``, ``Session``,
``fault``, ...; whole words, case-sensitive) in no packet file other than
``SPEC.md``. ``SPEC.md`` carries the v0.2 oracle rules by design, so there only
the words absent from those rules are checked (:data:`SPEC_GENERATOR_WORDS`).

Section selection: a heading containing ``[packet]`` selects its whole section
(to the next heading of the same or a higher level). A paragraph that is
exactly ``**[packet]**`` inside an unmarked section (spec §6) selects the text
after it up to the next heading, which is emitted under that section's heading.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import types
from pathlib import Path
from typing import Any

__all__ = [
    "PacketError",
    "CONTENT_FILES",
    "PACKET_SOURCE_FILES",
    "MANIFEST_FILE",
    "SPEC_SENTINELS",
    "GENERATOR_WORDS",
    "SPEC_GENERATOR_WORDS",
    "AUTHOR_INSTRUCTIONS",
    "GRAMMAR_EXAMPLES",
    "extract_packet_spec",
    "grammar_sheet",
    "build_packet",
    "verify_packet",
    "packet_sha256",
]

PACKET_FORMAT_VERSION = 1
SPEC_VERSION = "0.3"

SPEC_FILE = "SPEC.md"
TEMPLATE_FILE = "state_dict_template.json"
INSTRUCTIONS_FILE = "AUTHOR_INSTRUCTIONS.md"
GRAMMAR_FILE = "GRAMMAR_SHEET.md"
MANIFEST_FILE = "PACKET_MANIFEST.json"
PACKET_SOURCE_FILES = ("model.py", "tokenizer.py", "renderer.py")
CONTENT_FILES = (SPEC_FILE, *PACKET_SOURCE_FILES, TEMPLATE_FILE, INSTRUCTIONS_FILE, GRAMMAR_FILE)
PACKET_FILES = CONTENT_FILES + (MANIFEST_FILE,)

V02_SECTIONS = ("### 6.1", "### 6.2", "### 6.3", "### 7.1")
MARKER = "[packet]"

# Phrases from withheld text (v0.3 §0, §2.1, §2.3, §3, §7, §8, §10-§15; v0.2 §6.4). None may appear in SPEC.md.
SPEC_SENTINELS = (
    "E9", "E10", "delegated authority", "cleanup pass", "telemetry_gaps", "Codex", "Bayes ceiling",
    "Feature-collision", "false-match", "BOS as a sink", "Capacity notes", "Raw operation record",
    "Rendering audits", "Research targets", "Definition of done", "Milestones", "OOD generator",
    "Phase C", "Four-lens", "four-lens", "Generator ──►", "inspect heads", "registry_v0_3",
)
# Words that name the generators and their internals (spec §8: rawop.py was withheld for its docstring).
# Whole words, case-sensitive. None may occur in a packet file other than SPEC.md.
GENERATOR_WORDS = (
    "engine", "engines", "generator", "generators", "OOD", "oracle", "latent", "Session", "fault", "faults",
    "synth", "synth_ood", "rawop", "intent_fingerprint", "failure_fingerprint", "OracleRecord",
    "dropped_observations", "fork",
)
# SPEC.md holds the v0.2 oracle rules (latent state, faults, the oracle, "the generator deliberately dropped",
# intent and failure fingerprints), so only the words that those rules do not use are checked there.
SPEC_GENERATOR_WORDS = (
    "engine", "engines", "OOD", "Session", "synth", "synth_ood", "rawop", "OracleRecord", "dropped_observations",
    "fork",
)

_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*)$")
_FENCE_RE = re.compile(r"^ {0,3}(```|~~~)")
_SEC_KEY_RE = re.compile(r"^(#{1,6})[ \t]+(\S+)")
_LEAKY_IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+(?:agent_loop_lab\b|(?:\.+)?(?:synth\w*|datasets|eval)\b)", re.MULTILINE)


class PacketError(RuntimeError):
    """The packet cannot be built or fails verification."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pretty_json(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def packet_sha256(packet_dir: str | Path) -> str:
    """sha256 of the packet's ``PACKET_MANIFEST.json`` bytes."""
    p = Path(packet_dir) / MANIFEST_FILE
    if not p.is_file():
        raise PacketError(f"{MANIFEST_FILE} missing from {packet_dir}")
    return _sha256(p.read_bytes())


# -- spec extraction -----------------------------------------------------------------


def _headings(lines: list[str]) -> dict[int, tuple[int, str]]:
    out: dict[int, tuple[int, str]] = {}
    in_fence = False
    for i, line in enumerate(lines):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if not in_fence:
            m = _HEADING_RE.match(line)
            if m:
                out[i] = (len(m.group(1)), m.group(2))
    return out


def _section_end(heads: dict[int, tuple[int, str]], i: int, n: int) -> int:
    level = heads[i][0]
    return next((j for j in sorted(heads) if j > i and heads[j][0] <= level), n)


def _strip_rules(block: list[str]) -> list[str]:
    while block and block[-1].strip() in ("", "---"):
        block = block[:-1]
    while block and block[0].strip() == "":
        block = block[1:]
    return block


def _packet_sections_v03(text: str) -> list[str]:
    lines = text.replace("\r\n", "\n").split("\n")
    heads = _headings(lines)
    out: list[str] = []
    i, n = 0, len(lines)
    order = sorted(heads)
    while i < n:
        if i in heads and MARKER in heads[i][1]:
            end = _section_end(heads, i, n)
            block = list(lines[i:end])
            block[0] = block[0].replace(" " + MARKER, "").replace(MARKER, "").rstrip()
            out.append("\n".join(_strip_rules(block)))
            i = end
            continue
        if i in heads:
            # paragraph marker inside an unmarked section: take what follows it, to the next heading
            nxt = next((j for j in order if j > i), n)
            body = lines[i + 1 : nxt]
            for k, line in enumerate(body):
                if line.strip() == f"**{MARKER}**":
                    part = _strip_rules(body[k + 1 :])
                    out.append("\n".join([lines[i], ""] + part))
                    break
        i += 1
    return out


def _v02_sections(text: str) -> list[str]:
    lines = text.replace("\r\n", "\n").split("\n")
    heads = _headings(lines)
    out = []
    for key in V02_SECTIONS:
        hit = [i for i in sorted(heads) if f"{'#' * heads[i][0]} {heads[i][1].split()[0]}" == key]
        if len(hit) != 1:
            raise PacketError(f"v0.2 spec: section {key!r} found {len(hit)} times")
        i = hit[0]
        block = list(lines[i : _section_end(heads, i, len(lines))])
        block[0] = "#" + block[0]  # one level deeper, under the v0.2 heading below
        out.append("\n".join(_strip_rules(block)))
    return out


SPEC_PREAMBLE = """\
# Agent doom-loop detector, spec v0.3: authoring specification

This file holds the sections of the v0.3 specification that are given to the weight author,
followed by the oracle rules and scenario-family names of spec v0.2, which v0.3 keeps unchanged.
`AUTHOR_INSTRUCTIONS.md` is the operational protocol; `GRAMMAR_SHEET.md` shows the log grammar
with invented examples.
"""

V02_PREAMBLE = """\
## v0.2 oracle rules and scenario families (unchanged in v0.3)

The labels are produced by the v0.2 oracle from the simulator's latent state. Its definitions,
parameters and rules (v0.2 §6.1-§6.3) and the names of the scenario families (v0.2 §7.1) are
reproduced here. "Encoded features" and "the 120-vector" refer to v0.2's input, which v0.3
replaces with the rendered log."""


def extract_packet_spec(v03_text: str, v02_text: str) -> str:
    parts = [SPEC_PREAMBLE.rstrip("\n")]
    parts += _packet_sections_v03(v03_text)
    parts.append(V02_PREAMBLE)
    parts += _v02_sections(v02_text)
    return "\n\n".join(p.rstrip("\n") for p in parts) + "\n"


# -- grammar sheet -------------------------------------------------------------------

# One invented block per tool (spec §4), plus examples of the line markers, missing observations
# and truncation. Every non-reserved word is invented; a test checks that none occurs in the
# generator vocabularies.
GRAMMAR_EXAMPLES: tuple[tuple[str, str], ...] = (
    ("read_file", "09:14:03.250 read_file path=zork/zeta/widget.qx lines=12-40 -> ok in=1840 out=96"),
    ("search", '09:14:05.004 search scope=zork/zeta query="quux frobnicate" limit=5 -> ok in=1932 out=41'),
    ("list_dir", "09:14:06.117 list_dir path=zork/zeta -> ok in=2010 out=22"),
    ("edit_file (one line per changed file: removed-lines id > added-lines id)",
     "09:14:09.880 edit_file path=zork/zeta/widget.qx -> ok in=2214 out=310\n"
     "  @ zork/zeta/widget.qx h3f9a0c1d2e > h7b21e9c04a"),
    ("write_file",
     "09:14:12.301 write_file path=zork/zeta/gizmo_plinth.qx -> ok in=2402 out=505\n"
     "  @ zork/zeta/gizmo_plinth.qx h0 > h9e4a7710cd"),
    ("revert (undoes the edit_file block: the ids swap back)",
     "09:14:15.762 revert paths=zork/zeta/widget.qx -> ok in=2470 out=12\n"
     "  @ zork/zeta/widget.qx h7b21e9c04a > h3f9a0c1d2e"),
    ("run_tests (failed, with validation counts and an error line)",
     "09:14:20.415 run_tests suite=alpha tests=alpha/spin_wobble -> FAILED 7/9 passed in=2611 out=188\n"
     "  ! QuuxError at zork/zeta/widget.qx:417 (pid 4821)"),
    ("run_tests (dropped observations and untracked workspace)",
     "09:14:22.006 run_tests suite=? -> FAILED ?/9 passed in=? out=176\n"
     "  ! ? at ?:? (0x1f2e3d4c)\n"
     "  [workspace untracked]"),
    ("build", "09:14:31.002 build target=zeta_bale -> ok in=2702 out=64"),
    ("lint (failed, hex noise)",
     "09:14:33.640 lint target=zeta_bale -> FAILED in=2750 out=71\n"
     "  ! QuuxError at zork/zeta/gizmo_plinth.qx:88 (0x9c1e44b7)"),
    ("http_call",
     '09:14:40.118 http_call POST https://quuxhost.zeta/v9/gadgets/{gid} body={"flavor":"quux"} req=5d03ac91 '
     "-> error in=2810 out=33\n"
     "  ! QuuxError at zeta_portal:212 (pid 30177)"),
    ("poll_status", "09:14:52.733 poll_status resource=zeta_hopper backoff=4 -> pending in=2851 out=18"),
    ("plan", "09:14:55.090 plan action=sketch_zeta_zigzag -> ok in=2902 out=640"),
    ("ask", "09:15:01.512 ask action=ponder_zeta_flavor -> ok in=3015 out=57"),
    ("yield", "09:15:09.377 yield action=zeta_handback -> ok in=3104 out=212"),
    ("revert of 7 files (more changed files than fit: the 5th line is replaced by a count)",
     "09:15:12.020 revert paths=zork/blorb.qx,zork/frotz.qx,zork/gnusto.qx,zork/plugh.qx,zork/rezrov.qx,zork/xyzzy.qx,zork/yomin.qx -> ok in=3190 out=902\n"
     "  @ zork/blorb.qx h2a6b8c4d1e > h61f0e2d9ab\n"
     "  @ zork/frotz.qx h4d7c1b0e5f > h0\n"
     "  @ zork/gnusto.qx h0 > h8a3e5f7c21\n"
     "  @ zork/plugh.qx h3f9a0c1d2e > h5c0d81f3b6\n"
     "  @ TRUNC 3"),
)

_GRAMMAR_INTRO = """\
# Grammar sheet (spec v0.3 §4-§5)

Each example below is one **event block** as the frozen renderer lays it out, followed by its
tokens from the frozen tokenizer (`tokenizer.tokenize_block`, then `EOE`), one per line as
`offset  id  surface`. Line-marker tokens are shown as `NL@`, `NL!` and `NL[`.

Every path, suite, test, target, host, action, error class, hunk id and number here is
**invented** for illustration. Real logs use the same grammar with other values.

A session is `BOS` followed by its blocks in order (`tokenizer.encode_session`). `event_ord` is 0
for `BOS` and k+1 for every token of block k, including its `EOE`; `offset` restarts at 0 in every
block. The network's prediction for block k is read at that block's `EOE`.

Header suffix: `-> {status}[ {passed}/{total} passed] in={input_tokens} out={output_tokens}`.
Body lines, at most 6, in this order: hunk lines (`@`, at most 5), the error line (`!`, ERROR
outcomes only) and the workspace line (`[`, when a digest was dropped). A dropped observation is
rendered as `?` in its own position.
"""


def _display(surface: str) -> str:
    if surface.startswith("\n") and len(surface) == 2:
        return "NL" + surface[1]
    if surface == "\n":
        return "NL"
    return surface


def grammar_sheet(tokenizer: Any) -> str:
    """The GRAMMAR_SHEET.md text, tokenised with ``tokenizer`` (the packet's frozen module)."""
    out = [_GRAMMAR_INTRO]
    for title, block in GRAMMAR_EXAMPLES:
        pairs = list(tokenizer.tokenize_block(block)) + [(tokenizer.EOE, "<EOE>")]
        out.append(f"## {title}\n")
        out.append("```text\n" + block + "\n```\n")
        rows = [f"{off:>3}  {tid:>5}  {_display(s)}" for off, (tid, s) in enumerate(pairs)]
        out.append("```tokens\n" + "\n".join(rows) + "\n```\n")
    return "\n".join(out).rstrip("\n") + "\n"


# -- author instructions -----------------------------------------------------------------

AUTHOR_INSTRUCTIONS = """\
# Weight-author instructions (spec v0.3, §9)

You are the **weight author**. Your inputs are the files in `packet/` and nothing else. You work in
your own directory; you never read anything outside it.

`SPEC.md` defines the task, the model, the log grammar and the deliverables. This file is the
operational protocol: the exact interfaces the harness calls, the rules it enforces and how to run
things. Where the two seem to differ on *how* something is built, run or archived, follow this file.

## Your directory

| Path | What it is |
|---|---|
| `packet/` | This packet (read-only). `PACKET_MANIFEST.json` holds the sha256 of every file |
| `packet/model.py` | The fixed transformer (`DoomLoopTransformer`, `zero_model`, `PARAM_SHAPES`, constants) |
| `packet/tokenizer.py` | The frozen tokenizer (`encode_session`, `tokenize_block`, `word_id`, vocabulary constants) |
| `packet/renderer.py` | The frozen renderer (`render_session`, `render_event`); it reads plain dicts (below) |
| `packet/state_dict_template.json` | Tensor names, shapes and parameter count |
| `packet/GRAMMAR_SHEET.md` | One invented example block per tool, with its tokens |
| `tools/` | The `./py` launcher and its guard (read-only) |
| `output/v03-phase-a/` | Phase A deliverables; put scratch scripts in `output/v03-phase-a/scratch/` |
| `output/v03-phase-b/` | Phase B deliverables (only when Phase B starts) |

Never write inside `packet/` or `tools/`. The harness archives your whole phase directory,
including `scratch/`.

## Running Python

Run every script with the wrapper, from your directory:

```
./py output/v03-phase-a/scratch/check_logs.py
./py -m py_compile output/v03-phase-a/build_weights.py
```

- `./py` runs `.venv` Python 3.12 with `torch==2.2.2` and `numpy<2` (CPU, float32). No other
  packages are installed, and nothing can be installed.
- The module path is `packet/` first, then the script's directory, then that directory's parent.
  So `import model`, `import tokenizer` and `import renderer` work everywhere, and a
  script in `output/v03-phase-a/scratch/` can `import build_weights` and `import reference_policy`.
- Every script runs under the author guard. Forward passes of the model, matrix products and
  `reference_policy` calls are allowed. Refused, and fatal to the run: autograd (grad mode,
  `.backward()`, `torch.autograd`), `torch.optim`, any random-number API (`random`,
  `numpy.random`, `torch.rand*`, `torch.normal`, `torch.Generator`, in-place `.normal_()` and the
  like, dropout), threads and subprocesses, network access, reading or listing anything outside your
  directory, and writing outside your directory. The renderer's own seeded per-event noise
  source is the one allowed use of `random.Random`. For quasi-random vectors use `hashlib`.
- **One command per Bash call.** Only commands that start with `./py ` are allowed, and compound
  commands are refused: no `;`, `&&`, `||`, pipes, redirection (`>`, `<`), backticks, `$(...)` or
  `echo`. Write a script to a file with the file-writing tool, then run it with `./py <file>`.

## Operation records (`RawOp` dicts, the renderer's input)

`renderer.render_session(ops, session_id)` returns one block string per operation; `ops` is a list
of plain dicts, one per tool call, with these keys (a missing key reads as null; `session_id` is any
string, and it seeds the hunk ids and error line numbers of SPEC.md §4.3):

| Key | Type | Meaning |
|---|---|---|
| `family` | str | The tool name: `read_file`, `search`, `list_dir`, `edit_file`, `write_file`, `revert`, `run_tests`, `build`, `lint`, `http_call`, `poll_status`, `plan`, `ask`, `yield` |
| `target` | str, list or dict | What the call acts on: the path (`read_file`, `list_dir`, `edit_file`, `write_file`), the scope (`search`), the list of paths (`revert`), the suite (`run_tests`), the target (`build`, `lint`), `{"method", "host", "path_template"}` (`http_call`), the resource (`poll_status`) |
| `args` | dict | Call arguments the header shows: `line_start` / `line_end` (`read_file`), `content` (`write_file`; not shown in the header), `query` / `limit` (`search`), `tests` (a list, `run_tests`), `body` (`http_call`), `backoff` (`poll_status`), `action_type` (`plan`, `ask`, `yield`) |
| `outcome` | str | `SUCCESS`, `ERROR` or `NEUTRAL`. The status reads `error` for `ERROR` and `ok` otherwise; `run_tests` / `build` / `lint` read `FAILED` / `ok`, and `poll_status` reads `ready` / `error` / `pending` |
| `error_class`, `error_location` | str or null | The error line of an `ERROR` outcome: `! {error_class} at {error_location}:{line} (...)` |
| `validation` | `{"suite", "passed", "total"}` or null | The validation result; shown as `{passed}/{total} passed` |
| `changed` | list of `{"path", "before", "after"}` | Every tracked file whose text differs between the start and the end of the call, sorted by path; `before` / `after` is the full text, or null when the file is absent. Rendered as hunk lines (SPEC.md §4.3) |
| `extra` | dict or null | Tool-specific detail that the header does not show |
| `input_tokens`, `output_tokens` | int or null | Shown as `in=` and `out=` |
| `timestamp` | str or null | ISO 8601 (`2026-03-01T09:14:03.250+00:00`); shown in UTC as `HH:MM:SS.mmm` |
| `dropped` | sorted list of str | Observations withheld on this call, each shown as `?` in its own position: any of `digest_after`, `digest_before`, `failure`, `input_tokens`, `output_tokens`, `validation_passed`, `validation_suite`, `validation_total` (a dropped digest adds the `[workspace untracked]` line) |

## Deliverables and pinned interfaces

Put them in `output/v03-phase-a/` (Phase A) or `output/v03-phase-b/` (Phase B).

### `reference_policy.py`

```python
def reference_policy(tokens: list[str], event_ord: list[int]) -> int:
    ...
```

- `tokens` holds the tokenizer's **surface strings** for a session prefix (`Encoded.surface`:
  `"<BOS>"` first, `"<EOE>"` at the end of every block, `"\\n@"` / `"\\n!"` / `"\\n["` / `"\\n"`
  for line markers), and `event_ord` the matching event ordinals (`Encoded.event_ord`). The prefix
  always ends with the `<EOE>` of the block being classified.
- Return an `int` (not a `bool`) in `0..3`: `PRODUCTIVE`, `RECOVERY`, `STALLED`, `OSCILLATING`. It
  states your intended policy for the **last** block of the prefix.
- Standard library only; it may not import `tokenizer`, `model`, `renderer`, `torch` or `numpy`.
  Pure and deterministic: no I/O, no clock, no randomness, no `global` or `nonlocal` statement, and
  no function may change module-level state (no item assignment, `.append()`, `.update()` and the
  like on a module-level list, dict or set, no attribute assignment on a module-level name).
  Module-level constants are fine; build per-call tables inside the function.
- Both Python deliverables: no interpreter internals. Dunder names other than `__name__`,
  `__doc__`, `__all__`, `__init__`, `__post_init__`, `__repr__` and the like are refused
  (`__file__`, `__dict__`, `__globals__`, `__builtins__`, `__class__`, ...), as are strings that
  spell them, `vars()` / `locals()` / `globals()`, and `getattr` / `setattr` / `hasattr` /
  `delattr` unless the attribute name is a literal string (or the loop variable over a literal
  tuple of strings). Special methods other than `__init__`, `__post_init__` and `__repr__` may not
  be defined.
- The harness calls `reference_policy` only in a fresh interpreter under the build guard (no I/O,
  no randomness), with the CPU limit per call.
- At most 0.5 s of CPU per call. The harness smoke-calls it once, on the tokens of the grammar
  sheet's `read_file` block, as a type and time check (the value is not recorded). A timeout in
  evaluation counts as a disagreement.

### `build_weights.py`

```python
import torch
import model
import tokenizer


def build() -> model.DoomLoopTransformer:
    m = model.zero_model()
    with torch.no_grad():
        ...  # explicit constants and closed-form rules
    return m
```

- A module-level `build()` with no arguments that starts from `model.zero_model()`, sets
  parameters by closed-form rule or explicit constants, and returns that model. Keep every tensor's
  name, shape and float32 dtype; every value must be finite.
- Imports allowed: `torch` and `torch.nn`, `model`, `tokenizer`, and the standard-library modules
  `math`, `hashlib`, `itertools`, `functools`, `collections`, `dataclasses`, `typing`, `string`,
  `re`. Nothing else (in particular not `numpy`, `renderer` or `reference_policy`).
- The tokenizer may be called only on literal word lists: arguments must be literals (`"path"`,
  `["path", "lines"]`), a module-level name assigned once to a literal list or tuple of strings, or
  the loop variable of a `for` or comprehension over one. Its constants (`V`, `EOE`,
  `RESERVED_IDS`, ...) can be read freely.
- Quasi-random vectors must come from `hashlib` (for example the bits of `sha256(name)`).
- Allowed: indexing and index assignment, slicing, `index_put_`, `fill_`, `copy_`, `clone`,
  `torch.zeros` / `ones` / `full` / `tensor` / `arange` / `eye`, elementwise and scalar
  arithmetic, reductions, `torch.outer` / `Tensor.outer` and `torch.kron`.
- Forbidden (static scan and run-time tripwires, even inside `try`): `random`, `numpy.random`,
  `torch.rand*`, `torch.Generator` and every other random-number API; all of
  `torch.nn.functional`; `softmax` / `log_softmax`; `matmul` and the `@` operator; `mm`, `bmm`,
  `addmm`, `addr`, `ger`, `baddbmm`, `addbmm`, `addmv`, `mv`, `dot`, `vdot`, `inner`, `tensordot`,
  `einsum`, `chain_matmul`, `bilinear`, `torch.linalg`; `torch.ops`, `torch._C`; calling the model
  or any `nn.Module` (`m(x)`) and any `forward` / `forward_batch` / `activations` / `_run` call;
  autograd and `torch.optim`; I/O of any kind (`open`, `print` to files, `pathlib`, `os`, `sys`,
  `io`, `subprocess`); `eval`, `exec`, `compile`, `__import__`, `importlib`, `globals`; the
  modules `base64`, `struct`, `zlib`, `pickle`, `array`, `marshal`, `json`. Names containing
  `matmul`, `linalg`, `einsum` or `softmax` are refused too, so do not use them for your own
  variables.
- Size limits: no list, tuple, set or dict literal with more than 256 elements, no string or bytes
  literal longer than 512 characters (docstrings count, so keep long prose in `CIRCUITS.md`), and
  the file is at most 200 KB.
- The build must finish within 300 s of CPU, and two builds in two fresh interpreters must give
  byte-identical weights.
- Never evaluate the network, or any part of it, inside `build_weights.py`, however it is written
  (for example a forward pass assembled from elementwise products and sums). Weights follow from
  your stated policy by assignment or closed-form rules. Run forward passes only from your own
  scratch scripts.

### `CIRCUITS.md`

- The residual-subspace allocation (which dimensions carry what).
- Each head's role: its query, key and value subspaces and its event-distance bias pattern.
- MLP units grouped by circuit.
- Which `reference_policy` branch each logit implements.
- An account of every non-zero tensor block.

### `ITERATIONS.md` (Phase A) / `REVISION.md` (Phase B)

- `ITERATIONS.md`: every self-check iteration, with a summary of the diff, the reason and the
  inputs used (each hand-written log, or the operation-record sequence you rendered).
- `REVISION.md`: what was disclosed to you, and every change with its reason and the disclosed
  windows that motivated it.

## What the harness does with your files

You do not build, archive, hash or commit anything. When you state that a submission is final,
the harness reads your phase directory and:

1. checks the packet against its registered hash;
2. requires `build_weights.py`, `reference_policy.py`, `CIRCUITS.md` and `ITERATIONS.md`
   (Phase A) or `REVISION.md` (Phase B);
3. runs the static rules above on both Python files;
4. copies `build_weights.py` **alone** (with the packet) into a fresh private directory and runs
   `build()` from there in a fresh interpreter under the build guard (the bans above, no I/O, 300 s
   CPU), validates the state dict (exact tensor names in order, shapes, float32, finite, no
   autograd state) and rebuilds in a second fresh interpreter, requiring identical bytes. So
   `build()` must not depend on any other file of yours;
5. smoke-calls `reference_policy` once (above), again from a private copy of that file alone;
6. archives the artefact, its sha256 and your whole phase directory, `scratch/` included.

If a build is rejected you receive only the harness's error text, and you may fix what it names.
For an exception raised by your code that text gives the exception type and the line numbers in
your file, not the exception message, so reproduce the failure with `./py` to see the message.

## Phase A (zero labels)

- **Allowed:** forward passes of your network and calls of your `reference_policy` on logs you
  write yourself: hand-written block text, or `RawOp` sequences you build and render with
  `packet/renderer.py`. Compare the two and revise weights and policy by direct, reasoned edits.
- **Forbidden:** any other data; and automated search of any kind: fitting, gradient steps,
  random restarts, parameter sweeps scored against an objective, or distilling anything into the
  weights (your own `reference_policy` included). A loop that tries candidate weights and keeps the
  best is search.
- Log every iteration in `ITERATIONS.md`.
- A cost cap applies to the whole of Phase A, including continuations and resubmissions. The
  result is the first valid artefact you submit as final.

## Phase B (one diagnostic revision)

Phase B starts only if you are told it has. You then receive a fixed report of 200 validation
windows (for each: the rendered prefix with the window's first block marked, the true label,
your Phase A network's class and your `reference_policy`'s class) and nothing else. You make one
submitted revision in `output/v03-phase-b/`, logging every iteration in `REVISION.md`. Phase B has
its own cost cap.
"""


# -- build / verify ------------------------------------------------------------------------


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PacketError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    import sys

    saved = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.modules[name] = mod  # dataclasses look their module up while the class is created
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = saved
        sys.modules.pop(name, None)
    return mod


def _template_document(model_mod: types.ModuleType) -> dict[str, Any]:
    import math

    shapes = model_mod.PARAM_SHAPES
    return {
        "spec_version": SPEC_VERSION,
        "tensors": [{"name": k, "shape": list(s), "numel": math.prod(s)} for k, s in shapes.items()],
        "param_count": int(model_mod.PARAM_COUNT),
        "dtype": "float32",
        "device": "cpu",
        "initial_state": "all zeros",
        "class_order": list(model_mod.CLASS_NAMES),
        "construction": (
            "start from model.zero_model() (every tensor exactly 0.0), assign entries under torch.no_grad(), "
            "and return the model from build(); the harness saves the artefact (see AUTHOR_INSTRUCTIONS.md)"
        ),
    }


def _entries(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def _is_leaky(rel: str) -> bool:
    parts = Path(rel).parts
    low = rel.lower()
    return any(p == "datasets" or p.startswith("synth") for p in parts) or any(
        w in low for w in ("oracle", "review", "labeller", "generator", "engine", "eval", "disclosure")
    )


def _word_re(words: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)) + r")(?![A-Za-z0-9_])")


_GENERATOR_RE = _word_re(GENERATOR_WORDS)
_SPEC_GENERATOR_RE = _word_re(SPEC_GENERATOR_WORDS)


def leakage_problems(texts: dict[str, str]) -> list[str]:
    """Withheld phrases (every file) and generator words (every file; a reduced list in SPEC.md)."""
    problems: list[str] = []
    for name, text in sorted(texts.items()):
        for s in SPEC_SENTINELS:
            if s in text:
                problems.append(f"{name}: contains withheld text {s!r}")
        rx = _SPEC_GENERATOR_RE if name == SPEC_FILE else _GENERATOR_RE
        hits = sorted(set(rx.findall(text)))
        if hits:
            problems.append(f"{name}: names the generators or their internals {hits}")
    return problems


def build_packet(v03_spec: str | Path, v02_spec: str | Path, source_dir: str | Path, out_dir: str | Path) -> dict[str, Any]:
    """Build the packet into ``out_dir`` (absent, empty or a previous packet) and return the manifest."""
    v03_spec, v02_spec, src, out = Path(v03_spec), Path(v02_spec), Path(source_dir), Path(out_dir)
    missing = [n for n in PACKET_SOURCE_FILES if not (src / n).is_file()]
    if missing:
        raise PacketError(
            f"packet sources missing from {src}: {missing} (the v0.3 renderer and tokenizer must exist before "
            "the packet can be built)"
        )
    if out.exists():
        if not out.is_dir():
            raise PacketError(f"{out} exists and is not a directory")
        stray = [e for e in _entries(out) if e not in PACKET_FILES]
        if stray:
            raise PacketError(f"{out} contains entries that are not packet files: {stray}")
    raw: dict[str, bytes] = {n: (src / n).read_bytes() for n in PACKET_SOURCE_FILES}
    spec3, spec2 = v03_spec.read_bytes(), v02_spec.read_bytes()
    for label, data in [("v0.3 spec", spec3), ("v0.2 spec", spec2), *raw.items()]:
        if b"\r" in data:
            raise PacketError(f"{label} has CR line endings; packet sources must be LF")
    for name, data in raw.items():
        if _LEAKY_IMPORT_RE.search(data.decode("utf-8")):
            raise PacketError(f"{name} imports from the harness, generator or dataset packages")
    tok = _load_module("_v03_packet_build_tokenizer", src / "tokenizer.py")
    mdl = _load_module("_v03_packet_build_model", src / "model.py")
    contents: dict[str, bytes] = {
        SPEC_FILE: extract_packet_spec(spec3.decode("utf-8"), spec2.decode("utf-8")).encode("utf-8"),
        **raw,
        TEMPLATE_FILE: _pretty_json(_template_document(mdl)),
        INSTRUCTIONS_FILE: AUTHOR_INSTRUCTIONS.encode("utf-8"),
        GRAMMAR_FILE: grammar_sheet(tok).encode("utf-8"),
    }
    assert tuple(contents) == CONTENT_FILES
    manifest = {
        "packet_format_version": PACKET_FORMAT_VERSION,
        "spec_version": SPEC_VERSION,
        "files": {n: _sha256(d) for n, d in sorted(contents.items())},
        "sources": {
            "spec_v0_3": _sha256(spec3),
            "spec_v0_2": _sha256(spec2),
            **{f"v03/{n}": _sha256(d) for n, d in sorted(raw.items())},
        },
        "spec_sections": "v0.3 [packet] sections; v0.2 " + ", ".join(V02_SECTIONS),
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, data in contents.items():
        (out / name).write_bytes(data)
    (out / MANIFEST_FILE).write_bytes(_pretty_json(manifest))
    problems = verify_packet(out)
    if problems:
        raise PacketError("built packet failed verification: " + "; ".join(problems))
    return {**manifest, "packet_sha256": packet_sha256(out)}


def verify_packet(packet_dir: str | Path) -> list[str]:
    """Problems with the packet in ``packet_dir``; ``[]`` means OK."""
    out = Path(packet_dir)
    if not out.is_dir():
        return [f"packet directory not found: {out}"]
    problems: list[str] = []
    entries = _entries(out)
    for e in entries:
        if e not in PACKET_FILES:
            problems.append(f"stray entry: {e}")
        if _is_leaky(e):
            problems.append(f"entry from a generator/dataset/oracle/review path: {e}")
    for name in PACKET_FILES:
        if name not in entries:
            problems.append(f"missing file: {name}")
    if problems:
        return problems
    try:
        manifest = json.loads((out / MANIFEST_FILE).read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [f"{MANIFEST_FILE}: unreadable ({exc})"]
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict) or sorted(files) != sorted(CONTENT_FILES):
        return [f"{MANIFEST_FILE}: file list differs from {sorted(CONTENT_FILES)}"]
    for name in CONTENT_FILES:
        data = (out / name).read_bytes()
        if _sha256(data) != files[name]:
            problems.append(f"hash mismatch: {name}")
        if b"\r" in data:
            problems.append(f"{name}: CR line endings")
    for name in PACKET_SOURCE_FILES:
        if manifest.get("sources", {}).get(f"v03/{name}") != files[name]:
            problems.append(f"{name}: not a verbatim copy of its recorded source")
        if _LEAKY_IMPORT_RE.search((out / name).read_text(encoding="utf-8")):
            problems.append(f"{name}: imports from the harness, generator or dataset packages")
    spec = (out / SPEC_FILE).read_text(encoding="utf-8")
    problems += leakage_problems({n: (out / n).read_text(encoding="utf-8") for n in CONTENT_FILES})
    if MARKER in spec:
        problems.append(f"{SPEC_FILE}: still contains the {MARKER} marker")
    if (out / INSTRUCTIONS_FILE).read_text(encoding="utf-8") != AUTHOR_INSTRUCTIONS:
        problems.append(f"{INSTRUCTIONS_FILE}: does not match the builder's text")
    if problems:
        return problems
    try:
        tok = _load_module("_v03_packet_verify_tokenizer", out / "tokenizer.py")
        if (out / GRAMMAR_FILE).read_text(encoding="utf-8") != grammar_sheet(tok):
            problems.append(f"{GRAMMAR_FILE}: does not match the packet tokenizer's output")
        mdl = _load_module("_v03_packet_verify_model", out / "model.py")
        if json.loads((out / TEMPLATE_FILE).read_text(encoding="utf-8")) != _template_document(mdl):
            problems.append(f"{TEMPLATE_FILE}: does not match the packet model's PARAM_SHAPES")
    except Exception as exc:  # noqa: BLE001 - a broken packet module is a packet problem
        problems.append(f"packet modules could not be loaded: {type(exc).__name__}: {exc}")
    return problems
