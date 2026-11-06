"""Build the v0.5 author prompt from the frozen v0.3 packet (experiments/v0_5/PROMPT_NOTES.md).

Usage:
    uv run python scripts/v05_build_prompt.py [--packet DIR] [--out DIR]
    uv run python scripts/v05_build_prompt.py --check [--packet DIR] [--out DIR]

v0.5 replicates the premise of "Experiment 5" of "Sparks of Weight-Space Programming": the author
model gets ONE prompt holding everything (task, oracle rules, the renderer, tokeniser and model
source, the tensor layout, how the network is scored and an "edits" language for writing weights),
has no tools, gets no feedback, and every weight starts at 0. The task information is that of the
v0.3 Phase A packet (zero labels), in the Exp 5 form.

The prompt is assembled deterministically from experiments/v0_3/packet_handed_over/ (default):
every file listed in its PACKET_MANIFEST.json is checked against its sha256, the manifest itself
against the registered v0.3 packet sha256 (experiments/registry_v0_3.yaml and PACKET_SHA256 here),
and the packet's model.py, tokenizer.py and renderer.py against the copies in src/agent_loop_lab/v03/
that the scorer runs. SPEC.md and AUTHOR_INSTRUCTIONS.md are cut at their headings (the cut is a full
partition of each file, so nothing is dropped silently), and the few textual edits are exact
replace-once edits that fail if the text is not found exactly once.

Writes OUT/PROMPT.txt (no trailing newline: the file bytes are the prompt as sent) and
OUT/PROMPT.sha256 (``<sha256>  PROMPT.txt``). --check rebuilds in memory and exits 1 if either
file differs. Prints the prompt's size in characters, words and an estimate in tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKET_DIR = REPO_ROOT / "experiments" / "v0_3" / "packet_handed_over"
OUT_DIR = REPO_ROOT / "experiments" / "v0_5"
REGISTRY_V03 = REPO_ROOT / "experiments" / "registry_v0_3.yaml"
SCORER_SOURCES = REPO_ROOT / "src" / "agent_loop_lab" / "v03"

# sha256 of the v0.3 PACKET_MANIFEST.json (experiments/registry_v0_3.yaml packet_sha256).
PACKET_SHA256 = "72d2bebf0a4c2c8607fd1ca7f451ef56ff95dea444276f8a13132921d99045f2"
MANIFEST_FILE = "PACKET_MANIFEST.json"
SOURCE_FILES = ("renderer.py", "tokenizer.py", "model.py")

PROMPT_NAME = "PROMPT.txt"
SHA_NAME = "PROMPT.sha256"
CONTINUE_MESSAGE = "Continue."
END_MARKER = "END OF WEIGHTS"
# Rough estimate only (no local Claude tokeniser): this prompt is about half source code and token
# tables, which tokenise denser than prose.
CHARS_PER_TOKEN_ESTIMATE = 3.3


class PromptBuildError(ValueError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# -- packet ----------------------------------------------------------------------------------


def load_packet(packet_dir: Path = PACKET_DIR, *, registry: Path | None = REGISTRY_V03,
                scorer_sources: Path | None = SCORER_SOURCES) -> dict[str, str]:
    """Text of every manifest-listed packet file, after every hash check (raises PromptBuildError)."""
    packet_dir = Path(packet_dir)
    man_path = packet_dir / MANIFEST_FILE
    if not man_path.is_file():
        raise PromptBuildError(f"{man_path} missing")
    man_bytes = man_path.read_bytes()
    man_sha = _sha256(man_bytes)
    if man_sha != PACKET_SHA256:
        raise PromptBuildError(f"{man_path}: sha256 {man_sha} != registered v0.3 packet {PACKET_SHA256}")
    if registry is not None:
        m = re.search(r"^packet_sha256:\s*([0-9a-f]{64})\s*$", Path(registry).read_text(), re.M)
        if m is None or m.group(1) != PACKET_SHA256:
            raise PromptBuildError(f"{registry}: packet_sha256 is not {PACKET_SHA256}")
    files = json.loads(man_bytes)["files"]
    out: dict[str, str] = {}
    for name, want in sorted(files.items()):
        p = packet_dir / name
        if not p.is_file():
            raise PromptBuildError(f"{p} missing")
        data = p.read_bytes()
        if _sha256(data) != want:
            raise PromptBuildError(f"{p}: sha256 {_sha256(data)} != PACKET_MANIFEST.json {want}")
        out[name] = data.decode("utf-8")
    for name in SOURCE_FILES:
        if name not in out:
            raise PromptBuildError(f"{name} is not listed in {MANIFEST_FILE}")
        if scorer_sources is not None:
            got = _sha256((Path(scorer_sources) / name).read_bytes())
            if got != files[name]:
                raise PromptBuildError(f"{Path(scorer_sources) / name} (run by the scorer) differs from the packet {name}")
    return out


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PromptBuildError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses look the module up while the class is created
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
    return mod


# -- text surgery ----------------------------------------------------------------------------


def split_at_headings(text: str, headings: list[str]) -> dict[str, str]:
    """Partition ``text`` at the given heading lines (each must occur exactly once, outside code
    fences, in the given order). Key ``""`` is the text before the first heading. The pieces
    concatenated in order give back ``text`` exactly."""
    lines = text.splitlines(keepends=True)
    in_fence = False
    where: dict[str, int] = {}
    for i, ln in enumerate(lines):
        if ln.startswith("```"):
            in_fence = not in_fence
            continue
        s = ln.rstrip("\n")
        if not in_fence and s in headings:
            if s in where:
                raise PromptBuildError(f"heading {s!r} occurs more than once")
            where[s] = i
    missing = [h for h in headings if h not in where]
    if missing:
        raise PromptBuildError(f"headings not found: {missing}")
    starts = [where[h] for h in headings]
    if starts != sorted(starts):
        raise PromptBuildError("headings are not in document order")
    bounds = [0, *starts, len(lines)]
    keys = ["", *headings]
    pieces = {k: "".join(lines[a:b]) for k, a, b in zip(keys, bounds, bounds[1:])}
    if "".join(pieces[k] for k in keys) != text:  # pragma: no cover - partition invariant
        raise PromptBuildError("heading split is not a partition")
    return pieces


def replace_once(text: str, old: str, new: str) -> str:
    n = text.count(old)
    if n != 1:
        raise PromptBuildError(f"expected exactly one occurrence of {old[:60]!r}, found {n}")
    return text.replace(old, new)


def drop_table(text: str, first_row: str) -> str:
    """Remove the Markdown table starting at the line ``first_row`` (every following line that starts
    with ``|``) and the blank line after it."""
    lines = text.split("\n")
    idx = [i for i, ln in enumerate(lines) if ln == first_row]
    if len(idx) != 1:
        raise PromptBuildError(f"table row {first_row!r} found {len(idx)} times")
    a = b = idx[0]
    while b < len(lines) and lines[b].startswith("|"):
        b += 1
    if b < len(lines) and lines[b] == "":
        b += 1
    return "\n".join(lines[:a] + lines[b:])


def fenced(lang: str, body: str) -> str:
    if "```" in body:
        raise PromptBuildError("source holds a code fence")
    return f"```{lang}\n{body.rstrip(chr(10))}\n```"


# -- the packet pieces -----------------------------------------------------------------------

SPEC_HEADINGS = [
    "## 1. Objective and hypotheses",
    "### 1.2 What a good result does and does not show",
    "### 2.2 Target model (immutable within a run)",
    "## 4. Log rendering (frozen)",
    "## 5. Tokeniser (frozen)",
    "## 6. Windows, gate, labels",
    "## 9. Direct-weight construction protocol",
    "## v0.2 oracle rules and scenario families (unchanged in v0.3)",
]
# Pieces of SPEC.md left out (PROMPT_NOTES.md §3): the file header (it names the packet files) and §9
# (deliverables, Phase A/B protocol, cost caps: all about the tool-using v0.3 setting).
SPEC_EXCLUDED = ("", "## 9. Direct-weight construction protocol")

AI_HEADINGS = [
    "## Your directory",
    "## Running Python",
    "## Operation records (`RawOp` dicts, the renderer's input)",
    "## Deliverables and pinned interfaces",
    "## What the harness does with your files",
    "## Phase A (zero labels)",
    "## Phase B (one diagnostic revision)",
]
AI_INCLUDED = ("## Operation records (`RawOp` dicts, the renderer's input)",)

# (where, old, new): exact replace-once edits of packet text (PROMPT_NOTES.md §4).
SPEC_EDITS = [
    ("## 1. Objective and hypotheses", "## 1. Objective and hypotheses\n", "## 1. Objective\n"),
    ("## 6. Windows, gate, labels",
     "- **Oracle rules.** The packet carries v0.2 §6.1–§6.3 (latent state definitions, parameters, rules) "
     "and the family names in v0.2 §7.1, as the v0.2 packet did.",
     "- **Oracle rules.** The labels come from the v0.2 oracle. Its §6.1–§6.3 (latent state definitions, "
     "parameters, rules) and the family names in its §7.1 follow."),
    ("## v0.2 oracle rules and scenario families (unchanged in v0.3)",
     "\n\nThese values live in `oracle_config.v0_2.yaml` and are frozen at M0.", ""),
    ("## v0.2 oracle rules and scenario families (unchanged in v0.3)",
     "#### 7.1 Scenario families (names only; this list is in the authoring packet)",
     "#### 7.1 Scenario families (names only)"),
]
H_TABLE_FIRST_ROW = "| ID | Claim | Measured as |"
AI_EDITS = [
    ("## Operation records (`RawOp` dicts, the renderer's input)",
     "error line numbers of SPEC.md §4.3)", "error line numbers of §4.3)"),
    ("## Operation records (`RawOp` dicts, the renderer's input)",
     "Rendered as hunk lines (SPEC.md §4.3)", "Rendered as hunk lines (§4.3)"),
]


def spec_pieces(spec: str) -> dict[str, str]:
    pieces = split_at_headings(spec, SPEC_HEADINGS)
    for where, old, new in SPEC_EDITS:
        pieces[where] = replace_once(pieces[where], old, new)
    key = "## 1. Objective and hypotheses"
    pieces[key] = drop_table(pieces[key], H_TABLE_FIRST_ROW)
    return {k: v.strip("\n") for k, v in pieces.items()}


def operation_records(author_instructions: str) -> str:
    pieces = split_at_headings(author_instructions, AI_HEADINGS)
    for where, old, new in AI_EDITS:
        pieces[where] = replace_once(pieces[where], old, new)
    return "\n\n".join(pieces[k].strip("\n") for k in AI_INCLUDED)


def token_id_table(tok: ModuleType) -> str:
    """The id of every fixed token, derived from the packet tokenizer's own tables."""
    if (tok.PAD, tok.BOS, tok.EOE, tok.NL) != (0, 1, 2, 3):
        raise PromptBuildError("unexpected special-token ids")
    specials = [(tok.PAD, "PAD (padding only; never produced)"), (tok.BOS, "BOS"), (tok.EOE, "EOE"), (tok.NL, "NL")]
    specials += [(i, "NL" + ch) for ch, i in sorted(tok.LINE_MARKERS.items(), key=lambda kv: kv[1])]
    reserved = sorted(tok.RESERVED_IDS.items(), key=lambda kv: kv[1])
    punct = sorted(tok.PUNCT_IDS.items(), key=lambda kv: kv[1])
    r_lo, r_hi = reserved[0][1], reserved[-1][1]
    p_lo, p_hi = punct[0][1], punct[-1][1]
    if [i for _, i in reserved] != list(range(r_lo, r_hi + 1)) or [i for _, i in punct] != list(range(p_lo, p_hi + 1)):
        raise PromptBuildError("reserved or punctuation ids are not contiguous")
    d = tok.DIGIT_IDS
    if (d["0"], d["9"], d["00"], d["99"], d["000"], d["999"]) != (224, 233, 234, 333, 334, 1333):
        raise PromptBuildError("unexpected digit-group ids")
    if (tok.HASH_BASE, tok.HASH_BUCKETS, tok.V) != (2048, 8192, 10240):
        raise PromptBuildError("unexpected hashed-word range")

    def rows(pairs: list[tuple[str, int]], per_line: int) -> list[str]:
        cells = [f"{i} {s}" for s, i in pairs]
        return ["  " + "   ".join(cells[k : k + per_line]) for k in range(0, len(cells), per_line)]

    out = ["Token ids (the tables of tokenizer.py, listed):", "Special and line-marker tokens:"]
    out += ["  " + "   ".join(f"{i} {s}" for i, s in specials)]
    out += [f"Reserved words (ids {r_lo}-{r_hi}; ids {r_hi + 1}-127 are never produced):"]
    out += rows(reserved, 7)
    out += [f"Punctuation (ids {p_lo}-{p_hi}; ids {p_hi + 1}-223 are never produced):"]
    out += rows(punct, 8)
    out += [
        "Digit groups (a number is split left to right into groups of 3 digits; the last group may be shorter):",
        '  a 1-digit group g ("0".."9") has id 224 + g (224-233); a 2-digit group ("00".."99") has id 234 + its value',
        '  (234-333); a 3-digit group ("000".."999") has id 334 + its value (334-1333). For example "09" is 243 and',
        '  "250" is 584. Ids 1334-2047 are never produced.',
        "Every other word: id 2048 + (int(sha256(word)[:8], 16) mod 8192), in 2048-10239.",
    ]
    return "\n".join(out)


LAYOUT = [
    ("tok_emb.weight", "E: row t = token id t; column j = residual dimension j"),
    ("off_emb.weight", "P: row o = offset o within the event block (BOS has offset 0); column j = residual dimension j"),
    ("attn.rel_bias", "B_l: row h = head h; column c = event-distance bucket c (0..63), column 64 = the BOS column"),
    ("attn.q.weight", "W_q, [out, in]: row i = query unit i (head h = rows 16h..16h+15); column j = residual dimension j"),
    ("attn.q.bias", "b_q: entry i = query unit i"),
    ("attn.k.weight", "W_k, [out, in]: row i = key unit i (head h = rows 16h..16h+15); column j = residual dimension j"),
    ("attn.k.bias", "b_k: entry i = key unit i"),
    ("attn.v.weight", "W_v, [out, in]: row i = value unit i (head h = rows 16h..16h+15); column j = residual dimension j"),
    ("attn.v.bias", "b_v: entry i = value unit i"),
    ("attn.o.weight", "W_o, [out, in]: row j = residual dimension j; column i = concatenated head output i (head h = columns 16h..16h+15)"),
    ("attn.o.bias", "b_o: entry j = residual dimension j"),
    ("mlp.fc1.weight", "W_1, [out, in]: row u = MLP unit u; column j = residual dimension j"),
    ("mlp.fc1.bias", "b_1: entry u = MLP unit u"),
    ("mlp.fc2.weight", "W_2, [out, in]: row j = residual dimension j; column u = MLP unit u"),
    ("mlp.fc2.bias", "b_2: entry j = residual dimension j"),
    ("head.weight", "W_out, [out, in]: row c = class c; column j = residual dimension j"),
    ("head.bias", "b_out: entry c = class c"),
]


def tensor_table(template: dict) -> str:
    tensors = template["tensors"]
    total = sum(t["numel"] for t in tensors)
    if total != template["param_count"] or total != 2_128_932:
        raise PromptBuildError(f"parameter count {total} != 2,128,932")
    kinds = {re.sub(r"^layers\.\d+\.", "", t["name"]) for t in tensors}
    if kinds != {k for k, _ in LAYOUT}:
        raise PromptBuildError(f"tensor kinds differ from the layout table: {sorted(kinds)}")
    width = max(len(t["name"]) for t in tensors)
    out = [
        "The tensors (torch state_dict names; layers.0 .. layers.3 are layers 1..4 of §2.2; every .weight of a linear",
        "map has the torch layout [out, in], so entry [i, j] is the weight from input j to output i):",
    ]
    for kind, desc in LAYOUT:
        name = kind if kind in ("tok_emb.weight", "off_emb.weight", "head.weight", "head.bias") else "layers.L." + kind
        out.append(f"  {name:<{width}}  {desc}")
    out.append(f"Every tensor, in state_dict order (name, shape; {total:,} parameters in all):")
    for t in tensors:
        shape = "[" + ", ".join(str(s) for s in t["shape"]) + "]"
        out.append(f"  {t['name']:<{width}}  {shape}")
    return "\n".join(out)


# -- the prompt ------------------------------------------------------------------------------

YOUR_TASK = f"""YOUR TASK: write, from scratch, the weights of a transformer that reads the log of an AI coding agent's session and, at every event, classifies the agent's recent behaviour into one of four classes, as well as possible. There is no training: no data, no labels, no loss, no gradients, no fine-tuning; you choose every number yourself. You have no tools: no code execution, no calculator, nothing to run, no files to read; everything you are given is in this message. You will get no feedback: nothing is run, and the network is scored only after you have finished. You may take as many replies as you need. After each reply you will receive only the message "{CONTINUE_MESSAGE}" (it carries no feedback, and nothing is run in between); after a reply that was cut off by a length limit you may instead receive a fixed notice that it was cut off. If a reply is cut off, continue in your next reply; if the cut fell inside an edits block, open a new edits block and write the cut statement again in full. Every edits block in every reply is read and applied, in the order written. First describe your design briefly (the strategy, and what the attention heads and MLP units compute), then write the edits blocks. End your final reply with a line that is exactly {END_MARKER} (outside any edits block)."""

THE_TASK = """THE TASK: an AI coding agent works through a session of tool calls (reading and editing files, searching, running tests, builds and lints, HTTP calls, polling, planning). Its session log is rendered as text, one event block per tool call, and tokenised. At every event, your network classifies the pattern of the agent's most recent eight operations into one of four classes. The sessions are synthetic: a simulator generates them, and an oracle labels each window from the simulator's hidden state. The task's specification follows; its section numbers are kept, so a reference such as §4.3 points to that section of the specification."""

THE_LOGS = """THE LOGS (the network's input): each session is rendered by renderer.py into one text block per event and tokenised by tokenizer.py (encode_session); the network reads the token ids, with each token's event ordinal and offset. Below: the specification's rendering and tokeniser sections, the id of every fixed token, the records the renderer reads, the exact source of renderer.py and tokenizer.py, and a grammar sheet with one invented example block per tool and its tokens."""

THE_NETWORK = """THE NETWORK (the same architecture as the trained network it will be compared with): a transformer with 4 layers, residual width d = 128, 8 attention heads of width 16, MLP width 512, no normalisation, an event-distance attention bias and 4 output logits; 2,128,932 parameters. Below: its specification (§2.2), the table of its tensors and the exact source of model.py, the code that runs it. EVERY WEIGHT STARTS AT 0; you set the weights you need."""

SCORING = """HOW YOUR NETWORK IS SCORED: on held-out sessions from the same simulator, which you do not see. Each session is rendered with renderer.py, tokenised with tokenizer.py and run through the network once (model.py, CPU, float32); attention is causal, so the logits at a block depend only on that block and the blocks before it. The class predicted for event block t is the argmax of the 4 logits at that block's EOE token (ties go to the first index, so all-zero logits mean PRODUCTIVE). The window ending at block t is blocks t-7 .. t, the agent's most recent eight operations, so windows ending before block 7 (counting from 0) are not scored. A window is scored when it passes the gate (§6: windows with withheld observations are not scored) and its oracle label (§6.3) is one of the four classes; windows labelled INSUFFICIENT_EVIDENCE are not scored. The score is the macro-F1 over the four classes on the scored windows (the mean of the four per-class F1 scores), with a 95% bootstrap confidence interval over sessions. The false-alarm rate is reported too: the share of scored windows labelled PRODUCTIVE or RECOVERY that the network classifies as STALLED or OSCILLATING. It is compared with a network of the same architecture trained on labelled windows."""

HOW_TO_WRITE = f"""HOW TO WRITE THE WEIGHTS (numbers only; nothing is run). Write code blocks whose opening line is three backticks followed by the word edits, one statement per line ('#' starts a comment; a long list may continue over the next lines). Tensors: the names and shapes in the table under THE NETWORK.

```edits
tok_emb.weight[13, 0] = 2.5                       # one entry: token id 13 (run_tests), residual dimension 0
off_emb.weight[0:4, 2] = [1, 0, -1, 0.5]          # explicit values along a range (as many values as entries)
tok_emb.weight[7:21, 1] = 1                       # fill a range (a:b is half-open; ':' alone = the whole axis)
layers.0.attn.q.weight[16:18, 0:3] = [[1, -1, 0.5], [0, 2, 0]]   # a 2-D list for a 2-D block (here 2 x 3)
layers.0.attn.rel_bias[1, 0:3] = [0, -1, -2]      # head 1, event distances 0, 1 and 2
layers.0.attn.rel_bias[1, 64] = -4                # head 1, the BOS column
layers.1.mlp.fc1.bias[0:512:2] = -1               # a:b:s takes every s-th index (here every 2nd MLP unit)
layers.1.mlp.fc2.weight[:, 0] *= 2                # scale (also += and -=)
head.bias[3] = 0.25                               # one entry: the OSCILLATING logit
```
Ranges: a:b is half-open, a:b:s takes every s-th index. Indices are plain non-negative integers counted from 0: no negative indices and no arithmetic (write 48:64, not 3*16:4*16); every index and every range end must lie inside its axis (a range is not clamped to the axis); a:, :b and ::s are accepted. Addressing: row t of tok_emb.weight is token id t (the ids are listed under THE LOGS) and row o of off_emb.weight is offset o; column j of either is residual dimension j. A linear map's .weight is [out, in]: entry [i, j] is the weight from input j to output i. Head h (0..7) of a layer is rows 16h to 16h+15 of its attn.q.weight, attn.k.weight and attn.v.weight (for head 3: 48:64), the same entries of attn.q.bias, attn.k.bias and attn.v.bias, columns 16h to 16h+15 of its attn.o.weight, and row h of its attn.rel_bias. Column c of attn.rel_bias: c = 0..63 is the event distance event_ord_i - event_ord_j (0 = the same event block; column 63 also covers every larger distance) and c = 64 is the BOS column. MLP unit u of a layer is row u of mlp.fc1.weight, entry u of mlp.fc1.bias and column u of mlp.fc2.weight. Row c of head.weight and entry c of head.bias give the logit of class c (0 PRODUCTIVE, 1 RECOVERY, 2 STALLED, 3 OSCILLATING).
Statements apply in order; a later statement overwrites an earlier one. Every edits block of every reply is read and applied, in the order written, so you can write the network in parts over several replies and change earlier statements later; do not write an edits block you do not want applied. A statement that cannot be read is skipped (the others still apply). If a reply is cut off inside an edits block, the statements before the cut are applied and the statement that was cut off is skipped: open a new edits block in your next reply and write that statement again in full (whatever message follows the cut). An edits block never continues into the next reply, and lines outside edits blocks are not read. Values: plain decimal numbers (|v| <= 1e6), stored as float32. When every weight you want is written, end your final reply with a line that is exactly {END_MARKER} (outside any edits block)."""


def build_prompt(packet: dict[str, str], tokenizer_module: ModuleType) -> str:
    spec = spec_pieces(packet["SPEC.md"])
    template = json.loads(packet["state_dict_template.json"])
    parts = [
        YOUR_TASK,
        THE_TASK,
        spec["## 1. Objective and hypotheses"],
        spec["### 1.2 What a good result does and does not show"],
        spec["## 6. Windows, gate, labels"],
        spec["## v0.2 oracle rules and scenario families (unchanged in v0.3)"],
        THE_LOGS,
        spec["## 4. Log rendering (frozen)"],
        spec["## 5. Tokeniser (frozen)"],
        token_id_table(tokenizer_module),
        operation_records(packet["AUTHOR_INSTRUCTIONS.md"]),
        "The renderer's exact source (renderer.py):",
        fenced("python", packet["renderer.py"]),
        "The tokeniser's exact source (tokenizer.py):",
        fenced("python", packet["tokenizer.py"]),
        packet["GRAMMAR_SHEET.md"].strip("\n"),
        THE_NETWORK,
        spec["### 2.2 Target model (immutable within a run)"],
        tensor_table(template),
        "The model's exact source (model.py):",
        fenced("python", packet["model.py"]),
        SCORING,
        HOW_TO_WRITE,
    ]
    prompt = "\n\n".join(parts)
    if "\r" in prompt or "\t" in prompt:
        raise PromptBuildError("prompt holds carriage returns or tabs")
    return prompt


def build(packet_dir: Path = PACKET_DIR, **kw) -> str:
    packet = load_packet(packet_dir, **kw)
    tok = _load_module(Path(packet_dir) / "tokenizer.py", "_v05_packet_tokenizer")
    return build_prompt(packet, tok)


def sha_line(prompt: str) -> str:
    return f"{_sha256(prompt.encode('utf-8'))}  {PROMPT_NAME}\n"


def size_report(prompt: str) -> dict[str, int]:
    return {
        "characters": len(prompt),
        "bytes": len(prompt.encode("utf-8")),
        "lines": prompt.count("\n") + 1,
        "words": len(prompt.split()),
        "tokens_estimate": round(len(prompt) / CHARS_PER_TOKEN_ESTIMATE),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packet", type=Path, default=PACKET_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--check", action="store_true", help="rebuild in memory; exit 1 if the files differ")
    args = ap.parse_args(argv)
    try:
        prompt = build(args.packet)
    except (PromptBuildError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    out_prompt, out_sha = args.out / PROMPT_NAME, args.out / SHA_NAME
    if args.check:
        ok = (out_prompt.is_file() and out_prompt.read_bytes() == prompt.encode("utf-8")
              and out_sha.is_file() and out_sha.read_text() == sha_line(prompt))
        print(("OK " if ok else "MISMATCH ") + sha_line(prompt).strip())
        return 0 if ok else 1
    args.out.mkdir(parents=True, exist_ok=True)
    out_prompt.write_bytes(prompt.encode("utf-8"))
    out_sha.write_text(sha_line(prompt))
    print(f"wrote {out_prompt} and {out_sha}")
    print(sha_line(prompt).strip())
    print(json.dumps(size_report(prompt), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
