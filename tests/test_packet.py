"""Authoring packet: spec stripping, deterministic build, verification."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from agent_loop_lab import model as model_mod
from agent_loop_lab import schema
from agent_loop_lab.packet import (
    AUTHOR_INSTRUCTIONS,
    CAPACITY_HINT_PHRASES,
    CONTENT_FILES,
    EXCLUDED_SECTIONS,
    MANIFEST_FILE,
    PACKET_FILES,
    SCHEMA_IMPORT_LINE,
    PacketError,
    build_packet,
    find_capacity_hints,
    inlined_constants_block,
    original_model_source,
    packet_digest,
    section_keys,
    standalone_model_source,
    strip_spec,
    verify_packet,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "agent_doom_loop_detector_spec_v0_2.md"
MODEL_PATH = REPO_ROOT / "src" / "agent_loop_lab" / "model.py"

EXPECTED_FILES = {
    "SPEC_AUTHORING.md",
    "model.py",
    "feature_index.json",
    "state_dict_template.json",
    "AUTHOR_INSTRUCTIONS.md",
    "MANIFEST.json",
}

_HEADING = re.compile(r"^(#{1,6}) (\S+)")
_RULE = re.compile(r"^-{3,}\s*$")


@pytest.fixture(scope="module")
def spec_text() -> str:
    return SPEC_PATH.read_text(encoding="utf-8")


@pytest.fixture
def packet(tmp_path) -> tuple[Path, dict]:
    out = tmp_path / "packet"
    manifest = build_packet(SPEC_PATH, MODEL_PATH, out)
    return out, manifest


def _files(d: Path) -> dict[str, bytes]:
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


def _rehash_manifest(out: Path) -> None:
    """Make the manifest consistent with the files on disk (simulates a careful tamperer)."""
    import hashlib

    m = json.loads((out / MANIFEST_FILE).read_text(encoding="utf-8"))
    m["files"] = {n: hashlib.sha256((out / n).read_bytes()).hexdigest() for n in CONTENT_FILES}
    m["packet_sha256"] = packet_digest(m["files"])
    (out / MANIFEST_FILE).write_text(json.dumps(m, sort_keys=True, indent=2) + "\n", encoding="utf-8")


# -- strip_spec: synthetic markdown ------------------------------------------------

SYNTHETIC = """\
# Title

**Status:** draft (see §0.1)

---

## 0. Changes
change table
### 0.1 Sign-off
signed

---

## 1. One
body one
### 1.1 Sub
```text
## 0. not a heading inside a fence
### 2.3 nor this
```
## 2. Two
### 2.2 Keep
keep 2.2
### 2.3 Drop
drop 2.3
#### 2.3.1 Nested drop
drop 2.3.1

---

## 3. Three
### 2.30 Different token, keep
keep 2.30
## 7. Seven
### 7.1 Keep
keep 7.1
### 7.2 Drop
drop 7.2
### 7.3 Drop
drop 7.3

---

## 8. Eight
## 9. Nine
end
"""

SYNTHETIC_EXPECTED = """\
# Title

**Status:** draft (see §0.1)

---

## 1. One
body one
### 1.1 Sub
```text
## 0. not a heading inside a fence
### 2.3 nor this
```
## 2. Two
### 2.2 Keep
keep 2.2
---

## 3. Three
### 2.30 Different token, keep
keep 2.30
## 7. Seven
### 7.1 Keep
keep 7.1
---

## 8. Eight
## 9. Nine
end
"""


def test_strip_spec_synthetic_exact():
    assert strip_spec(SYNTHETIC) == SYNTHETIC_EXPECTED


def test_strip_spec_synthetic_crlf_and_idempotent():
    assert strip_spec(SYNTHETIC.replace("\n", "\r\n")) == SYNTHETIC_EXPECTED
    assert strip_spec(SYNTHETIC_EXPECTED) == SYNTHETIC_EXPECTED


def test_strip_spec_section_at_end_of_document():
    text = "# T\n\n## 1. A\na\n### 7.3 Last\ngone\n"
    assert strip_spec(text) == "# T\n\n## 1. A\na\n"


def test_strip_spec_no_excluded_sections_is_identity():
    text = "# T\n\n---\n\n## 1. A\na\n\n---\n\n## 2. B\nb\n"
    assert strip_spec(text) == text


# -- strip_spec: the real spec -----------------------------------------------------


def _sections(text: str) -> list[tuple[str, list[str]]]:
    """Independent re-implementation: split into (heading key, lines) chunks."""
    chunks: list[tuple[str, list[str]]] = [("", [])]
    in_fence = False
    for line in text.split("\n"):
        if line.startswith("```"):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            chunks.append((f"{m.group(1)} {m.group(2)}", [line]))
        else:
            chunks[-1][1].append(line)
    return chunks


def _content_lines(lines) -> list[str]:
    return [l for l in lines if l.strip() and not _RULE.match(l)]


def test_strip_real_spec_removes_exactly_the_excluded_sections(spec_text):
    out = strip_spec(spec_text)
    orig = _sections(spec_text)
    # All of section 0 (every ``### 0.x`` subsection) plus the three named subsections.
    removed = {"## 0.", "### 2.3", "### 7.2", "### 7.3"} | {k for k, _ in orig if k.startswith("### 0.")}
    assert {"### 0.1", "### 0.2"} <= removed
    assert removed <= {k for k, _ in orig}, "test premise: all excluded headings exist in the spec"
    expected = [l for k, lines in orig if k not in removed for l in _content_lines(lines)]
    actual = [l for _, lines in _sections(out) for l in _content_lines(lines)]
    assert actual == expected


def test_strip_real_spec_headings(spec_text):
    keys = section_keys(strip_spec(spec_text))
    for gone in ("## 0.", "### 0.1", "### 2.3", "### 7.2", "### 7.3"):
        assert gone not in keys
    for kept in ("## 1.", "### 1.1", "### 2.1", "### 2.2", "## 2.", "## 3.", "### 3.2", "## 4.", "## 5.",
                 "### 5.2", "## 6.", "### 6.2", "### 6.3", "## 7.", "### 7.1", "## 8.", "## 9.", "### 9.1",
                 "### 9.2", "### 9.3", "## 10.", "## 13.", "## 15."):
        assert kept in keys, kept
    # Every original heading except the removed ones survives, in order.
    removed = {"## 0.", "### 2.3", "### 7.2", "### 7.3"} | {k for k in section_keys(spec_text) if k.startswith("### 0.")}
    assert not any(k.startswith("### 0.") for k in keys)
    orig = [k for k in section_keys(spec_text) if k not in removed]
    assert keys == orig


def test_strip_real_spec_keeps_title_status_block(spec_text):
    out = strip_spec(spec_text)
    head = spec_text.split("\n## 0.")[0]
    assert out.startswith(head)
    assert "**Status:** design approved 2026-11-04" in out
    assert out.split("\n")[:9][-1].startswith("## 1."), "title block, one rule, then ## 1."


def test_strip_real_spec_rules(spec_text):
    out_lines = strip_spec(spec_text).split("\n")
    nonblank = [l for l in out_lines if l.strip()]
    # No two rules adjacent (ignoring blanks), and no rule duplicated by the removal.
    for a, b in zip(nonblank, nonblank[1:]):
        assert not (_RULE.match(a) and _RULE.match(b))
    orig_rules = sum(1 for l in spec_text.split("\n") if _RULE.match(l))
    out_rules = sum(1 for l in out_lines if _RULE.match(l))
    assert out_rules == orig_rules - 1  # only the rule closing §0 goes
    # Rules before ## 3. and ## 8. are kept.
    for heading in ("## 3.", "## 8."):
        i = next(i for i, l in enumerate(nonblank) if l.startswith(heading))
        assert _RULE.match(nonblank[i - 1]), heading


def test_strip_real_spec_drops_capacity_hints(spec_text):
    assert set(find_capacity_hints(spec_text)) == set(CAPACITY_HINT_PHRASES), "premise"
    assert find_capacity_hints(strip_spec(spec_text)) == []


def test_find_capacity_hints_is_robust():
    assert find_capacity_hints("one unit **per position** here") == ["per position"]
    assert find_capacity_hints("a PER-POSITION conjunction") == ["per position"]
    assert find_capacity_hints("per\nposition") == ["per position"]
    assert find_capacity_hints("### 2.3 design note on capacity") == ["Design note on capacity"]
    assert find_capacity_hints("position, per event") == []


def test_excluded_sections_constant():
    assert EXCLUDED_SECTIONS == ("## 0.", "### 2.3", "### 7.2", "### 7.3")


# -- build ------------------------------------------------------------------------


def test_build_is_deterministic(tmp_path):
    m1 = build_packet(SPEC_PATH, MODEL_PATH, tmp_path / "a")
    m2 = build_packet(SPEC_PATH, MODEL_PATH, tmp_path / "b")
    assert m1 == m2
    assert _files(tmp_path / "a") == _files(tmp_path / "b")
    # Rebuilding in place is also byte-identical.
    before = _files(tmp_path / "a")
    m3 = build_packet(SPEC_PATH, MODEL_PATH, tmp_path / "a")
    assert m3 == m1
    assert _files(tmp_path / "a") == before


def test_packet_contains_only_contract_files(packet):
    out, manifest = packet
    assert set(_files(out)) == EXPECTED_FILES
    assert set(PACKET_FILES) == EXPECTED_FILES
    assert [p for p in out.iterdir() if p.is_dir()] == []
    assert set(manifest["files"]) == EXPECTED_FILES - {MANIFEST_FILE}
    # Nothing from src except model.py, which equals the source apart from the inlined constants.
    src_names = {p.name for p in (REPO_ROOT / "src" / "agent_loop_lab").rglob("*.py")}
    assert set(_files(out)) & src_names == {"model.py"}
    packet_model = (out / "model.py").read_text(encoding="utf-8")
    source_model = MODEL_PATH.read_text(encoding="utf-8")
    assert original_model_source(packet_model) == source_model
    # The frozen source imports from schema exactly once (the byte-copy fallback is tested separately).
    assert source_model.splitlines().count(SCHEMA_IMPORT_LINE) == 1
    assert packet_model == source_model.replace(SCHEMA_IMPORT_LINE, inlined_constants_block())
    assert packet_model.count(inlined_constants_block()) == 1


_STANDALONE_PROBE = """\
import sys
sys.path.insert(0, ".")
import model
m = model.zero_model()
assert sum(p.numel() for p in m.parameters()) == model.PARAM_COUNT == 3256
assert model.INPUT_DIM == 120
assert tuple(model.CLASS_NAMES) == ("PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING")
model.save_artifact(m, "probe.dlw")
assert model.sparsity(m)["overall"]["nonzero"] == 0
from pathlib import Path
assert Path(model.__file__).resolve().parent == Path(".").resolve(), model.__file__
leaked = sorted(k for k in sys.modules if k == "agent_loop_lab" or k.startswith("agent_loop_lab."))
assert not leaked, leaked
print("standalone ok")
"""


def test_packet_model_imports_standalone(packet, tmp_path):
    """Spec §8: the author needs nothing beyond the packet directory to run model.py."""
    out, _ = packet
    work = tmp_path / "work"
    shutil.copytree(out, work)
    r = subprocess.run([sys.executable, "-I", "-c", _STANDALONE_PROBE], cwd=work, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "standalone ok" in r.stdout
    assert (out / "model.py").read_bytes() == (work / "model.py").read_bytes()


def test_inlined_constants_match_schema():
    block = inlined_constants_block()
    ns: dict = {}
    exec(compile(block, "<block>", "exec"), ns)
    assert ns["INPUT_DIM"] == schema.INPUT_DIM
    assert ns["CLASS_NAMES"] == tuple(schema.CLASS_NAMES)


def test_build_refuses_model_with_other_harness_imports(tmp_path):
    src = tmp_path / "model.py"
    src.write_text(MODEL_PATH.read_text(encoding="utf-8") + "\nfrom .encode import encode_window\n", encoding="utf-8")
    with pytest.raises(PacketError, match="harness package"):
        build_packet(SPEC_PATH, src, tmp_path / "packet")


def test_standalone_model_source_refuses_repeated_import_line():
    """M0_CONTRACTS §D [R2]: a repeated import line is refused (the inversion must stay exact)."""
    source = MODEL_PATH.read_text(encoding="utf-8")
    with pytest.raises(PacketError, match=r"occurs 2 times"):
        standalone_model_source(source + "\n" + SCHEMA_IMPORT_LINE + "\n")


def test_standalone_model_source_refuses_preexisting_block():
    """M0_CONTRACTS §D [R2]: a source that already contains the inlined block is refused."""
    source = MODEL_PATH.read_text(encoding="utf-8")
    with pytest.raises(PacketError, match=r"already contains the inlined-constants block"):
        standalone_model_source(source + "\n" + inlined_constants_block() + "\n")
    # Also when the block replaced the import line already (a packet model.py fed back in as a source).
    with pytest.raises(PacketError, match=r"already contains the inlined-constants block"):
        standalone_model_source(standalone_model_source(source))


def test_standalone_model_source_without_import_line_is_a_byte_copy(tmp_path):
    """M0_CONTRACTS §D [R2]: a source without the import line is copied byte for byte."""
    source = MODEL_PATH.read_text(encoding="utf-8").replace(
        SCHEMA_IMPORT_LINE, f"INPUT_DIM = {schema.INPUT_DIM!r}\nCLASS_NAMES = {tuple(schema.CLASS_NAMES)!r}"
    )
    assert SCHEMA_IMPORT_LINE not in source
    assert standalone_model_source(source) == source
    src = tmp_path / "model.py"
    src.write_text(source, encoding="utf-8")
    out = tmp_path / "packet"
    manifest = build_packet(SPEC_PATH, src, out)
    assert (out / "model.py").read_bytes() == src.read_bytes()
    assert verify_packet(out) == []
    assert manifest["source_model_sha256"] == manifest["files"]["model.py"]


def test_build_refuses_crlf_sources(tmp_path):
    for name, src in (("spec.md", SPEC_PATH), ("model.py", MODEL_PATH)):
        crlf = tmp_path / name
        crlf.write_bytes(src.read_bytes().replace(b"\n", b"\r\n"))
        spec = crlf if name == "spec.md" else SPEC_PATH
        model = crlf if name == "model.py" else MODEL_PATH
        with pytest.raises(PacketError, match="CR line endings"):
            build_packet(spec, model, tmp_path / f"packet-{name}")


def test_manifest_contents(packet):
    import hashlib

    out, manifest = packet
    on_disk = json.loads((out / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert on_disk == manifest
    for name, sha in manifest["files"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == sha
    expected = hashlib.sha256(
        json.dumps(sorted([n, s] for n, s in manifest["files"].items()), separators=(",", ":")).encode()
    ).hexdigest()
    assert manifest["packet_sha256"] == expected
    assert manifest["source_spec_sha256"] == hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()
    assert manifest["spec_version"] == "0.2"
    raw = (out / MANIFEST_FILE).read_text(encoding="utf-8")
    assert str(REPO_ROOT) not in raw and str(out) not in raw, "no absolute paths in the manifest"


def test_spec_authoring_has_no_hints_and_required_sections(packet, spec_text):
    out, _ = packet
    text = (out / "SPEC_AUTHORING.md").read_text(encoding="utf-8")
    assert text == strip_spec(spec_text)
    for phrase in CAPACITY_HINT_PHRASES:
        assert phrase.lower() not in text.lower()
    assert "per-position" not in text.lower()
    keys = section_keys(text)
    for req in ("## 4.", "## 6.", "## 9."):
        assert req in keys


def test_feature_index_json(packet):
    out, _ = packet
    doc = json.loads((out / "feature_index.json").read_text(encoding="utf-8"))
    entries = doc["entries"]
    assert len(entries) == schema.INPUT_DIM == 120
    assert sorted(e["flat_index"] for e in entries) == list(range(120))
    for e in entries:
        assert e["flat_index"] == schema.flat_index(e["position"], e["feature"])
        assert e["flat_index"] == schema.flat_index(e["position"], e["feature_index"])
        assert schema.FEATURE_NAMES[e["feature_index"]] == e["feature"]
    assert [f["name"] for f in doc["features"]] == list(schema.FEATURE_NAMES)
    assert {f["name"]: f["definition"] for f in doc["features"]} == schema.FEATURE_DEFINITIONS
    assert doc["class_order"] == list(schema.CLASS_NAMES) == ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
    assert doc["events_per_window"] == 8 and doc["features_per_event"] == 15


def test_state_dict_template_json(packet):
    out, _ = packet
    doc = json.loads((out / "state_dict_template.json").read_text(encoding="utf-8"))
    assert [(t["name"], tuple(t["shape"])) for t in doc["tensors"]] == list(model_mod.PARAM_SHAPES.items())
    assert doc["param_count"] == model_mod.PARAM_COUNT == 3256
    assert sum(t["numel"] for t in doc["tensors"]) == 3256
    assert doc["initial_state"] == "all zeros"
    zero = model_mod.zero_model().state_dict()
    assert [(k, tuple(v.shape)) for k, v in zero.items()] == [(t["name"], tuple(t["shape"])) for t in doc["tensors"]]


def test_author_instructions(packet):
    out, _ = packet
    text = (out / "AUTHOR_INSTRUCTIONS.md").read_text(encoding="utf-8")
    assert text == AUTHOR_INSTRUCTIONS
    for needle in (
        # deliverables and pinned interfaces
        "reference_policy.py",
        "build_weights.py",
        "CIRCUITS.md",
        "ITERATIONS.md",
        "REVISION.md",
        "def reference_policy(x: tuple[float, ...]) -> int:",
        "exactly 120 Python floats",
        "exactly representable in\n  float32",
        "ordered as in `feature_index.json`",
        "`0..3`",
        "[PRODUCTIVE, RECOVERY, STALLED, OSCILLATING]",
        "def build() -> DoomLoopMLP:",
        "from model import zero_model, DoomLoopMLP",
        "first on `sys.path`",
        "zero_model()",
        "torch.no_grad()",
        'if __name__ == "__main__":',
        "from model import sparsity",
        "print(sparsity(build()))",
        "Do not write or read any file",
        "exactly once, on the all-zero 120-tuple",
        "compute ban",
        "byte-identical",
        # environment
        "Python 3.12",
        "torch==2.2.2",
        "numpy>=1.26,<2",
        "float32",
        "runs standalone",
        "PYTHONDONTWRITEBYTECODE=1",
        # where to write; who archives
        "<output>/phase-a-strict/",
        "<output>/phase-a-selfcheck/",
        "<output>/phase-b-revision/",
        "never inside the packet",
        "You do not archive, hash or commit anything",
        "**before A-selfcheck\nbegins**",
        # phase rules
        "No forward passes",
        "inputs you construct\n  yourself",
        "diff summary, reason, inputs used",
        "**one** revision",
        "200 windows from the validation split",
        # guard rules (each family)
        ".backward(",
        "torch.optim",
        "torch.autograd",
        "torch.func",
        "torch._C",
        "`requires_grad=`",
        "`nn.Parameter(...)`",
        "`.step(`",
        "inference_mode(False)",
        "a thread or a process is started",
        "agent_loop_lab.synth",
        "`numpy.random`",
        "`import`, `from ... import`, `importlib.import_module`",
        "`open`",
        "`datasets` path segment",
    ):
        assert needle in text, needle
    assert find_capacity_hints(text) == []
    assert "per-position" not in text.lower() and "per position" not in text.lower()
    # The author must not be told to import the harness package, nor where the harness keeps archives.
    assert "from agent_loop_lab" not in text and "import agent_loop_lab" not in text
    assert "weight_program/" not in text
    # No generator details or capacity notes leak through the instructions.
    for leak in ("Generator requirements", "counterfactual", "Lookalike", "hidden unit per", "capacity"):
        assert leak.lower() not in text.lower(), leak


def _backticked(text: str) -> set[str]:
    return set(re.findall(r"`([^`\n]+)`", text))


def _norm_name(s: str) -> str:
    s = s.strip()
    for prefix in ("torch.", "init."):
        if s.startswith(prefix):
            s = s[len(prefix) :]
    return s.lstrip(".").rstrip("(").strip()


def _build_weights_section_code_blocks() -> list[str]:
    section = AUTHOR_INSTRUCTIONS.split("### `build_weights.py`", 1)[1].split("\n### ", 1)[0]
    blocks = re.findall(r"^( *)```python\n(.*?)^\1```$", section, flags=re.MULTILINE | re.DOTALL)
    return [textwrap.dedent(body) for _indent, body in blocks]


def test_author_instructions_build_template_and_main_block_run(packet, tmp_path):
    """The pinned template plus the suggested sparsity __main__ block runs as written (no NameError)."""
    from agent_loop_lab.guards import scan_source
    from agent_loop_lab.runner import check_build_source

    out, _ = packet
    blocks = _build_weights_section_code_blocks()
    assert len(blocks) == 2, blocks
    template, main_block = blocks
    assert "from model import zero_model, DoomLoopMLP" in template
    assert main_block.startswith('if __name__ == "__main__":')
    assert "model.sparsity" not in AUTHOR_INSTRUCTIONS, "the pinned import never binds the name `model`"
    author = tmp_path / "author"
    author.mkdir()
    f = author / "build_weights.py"
    f.write_text(template + "\n\n" + main_block, encoding="utf-8")
    assert scan_source(f) == [] and check_build_source(f) == []
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env.update(PYTHONPATH=str(out), PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, str(f)], cwd=author, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "'overall'" in r.stdout and "3256" in r.stdout
    assert verify_packet(out) == [], "running the template must not write into the packet"


def test_author_instructions_list_every_guard_rule():
    """The instructions name every rule guards.py and the runner enforce, so an honest author is never surprised."""
    from agent_loop_lab import guards, runner

    text = AUTHOR_INSTRUCTIONS
    ticked = _backticked(text)
    names = {_norm_name(t) for t in ticked}
    for prefix in guards.FORBIDDEN_MODULE_PREFIXES:
        assert f"`{prefix}`" in text, prefix
    for root in guards._FORBIDDEN_IMPORT_ROOTS:
        assert f"`{root}`" in text, root
    for mod in guards._RANDOM_MODULES + ("random",):
        assert f"`{mod}`" in text, mod
    for fn in guards._RANDOM_TORCH_FUNCS:
        assert f"`torch.{fn}`" in text, fn
    for meth in guards._RNG_INPLACE_METHODS:
        assert f"`.{meth}`" in text, meth
    for leaf in guards._FORBIDDEN_LEAVES:
        assert leaf in names or f"`{leaf}`" in text or (leaf.startswith("synth") and "`synth*`" in text), leaf
    # The allowed nn.init list in the instructions is exactly the guard's deterministic set.
    m = re.search(r"`torch\.nn\.init`: only (.+?) are allowed", text, re.S)
    assert m, "allowed nn.init sentence"
    assert {_norm_name(x) for x in re.findall(r"`([^`]+)`", m.group(1))} == set(guards._DETERMINISTIC_INIT)
    # Every thread / process event the guard's audit hook blocks has a plain-language entry.
    for event in guards._AUDIT_BLOCKED:
        short = event.split(".")[0] if event.startswith(("_thread", "subprocess")) else event
        assert f"`{short}`" in text or f"`{short}*`" in text, event
    # Runner rules: banned modules, I/O builtins and I/O call names, listed exactly.
    m = re.search(r"Banned modules \(and their submodules\) for both files: (.+?)\. Safe", text, re.S)
    assert m, "banned-modules sentence"
    assert {x for x in re.findall(r"`([^`]+)`", m.group(1))} == set(runner.BANNED_MODULES)
    for b in runner.IO_BUILTINS:
        assert f"`{b}`" in text, b
    for name in runner.IO_CALL_NAMES:
        assert f"`{name}`" in text, name
    for f in runner.REQUIRED_FILES["phase-a-selfcheck"] + runner.REQUIRED_FILES["phase-b-revision"]:
        assert f"`{f}`" in text, f
    # Compute ban (spec §0.2): every static token and every name the runtime tripwire replaces.
    for tok in runner.COMPUTE_TOKENS:
        assert f"`{tok}`" in text, tok
    for n in runner.COMPUTE_BANNED_TORCH:
        assert f"`torch.{n}`" in text, n
    for n in runner.COMPUTE_BANNED_TENSOR:
        assert f"`Tensor.{n}`" in text, n
    for n in runner.COMPUTE_BANNED_NUMPY:
        assert f"`numpy.{n}`" in text, n
    for n in ("`@`", "`.forward(`", "`.dot(`", "`torch.nn.functional.linear`", "`nn.Linear.forward`", "`nn.Module`",
              "every `torch.linalg` function", "every\n`numpy.linalg` function"):
        assert n in text, n


def test_author_instructions_smoke_call_wording_is_uniform():
    """Every mention of the harness smoke call says the same thing (spec §0.2 "Harness smoke call")."""
    flat = " ".join(AUTHOR_INSTRUCTIONS.split())
    phrase = (
        "calls `reference_policy` exactly once, on the all-zero 120-tuple, as a type check "
        "(it must return an `int` in `0..3`); the return value is not recorded"
    )
    assert flat.count("all-zero 120-tuple") == flat.count(phrase) == 3
    assert "not reported" not in flat


# -- verify ---------------------------------------------------------------------


def test_verify_fresh_build_ok(packet):
    out, _ = packet
    assert verify_packet(out) == []


def test_verify_detects_tampered_file(packet):
    out, _ = packet
    p = out / "SPEC_AUTHORING.md"
    p.write_bytes(p.read_bytes() + b"\nextra line\n")
    problems = verify_packet(out)
    assert any("hash mismatch" in x and "SPEC_AUTHORING.md" in x for x in problems), problems


def test_verify_detects_tampered_model(packet):
    out, _ = packet
    p = out / "model.py"
    p.write_bytes(p.read_bytes().replace(b"HIDDEN1 = 24", b"HIDDEN1 = 25"))
    assert any("hash mismatch" in x and "model.py" in x for x in verify_packet(out))


def test_verify_detects_synth_import_in_model_after_rehash(packet):
    out, _ = packet
    p = out / "model.py"
    p.write_text(p.read_text(encoding="utf-8") + "from agent_loop_lab.synth import gen\n", encoding="utf-8")
    _rehash_manifest(out)
    problems = verify_packet(out)
    assert not any("hash mismatch" in x for x in problems)
    assert any("model.py: imports a generator/dataset module" in x for x in problems), problems


def test_verify_detects_manifest_listing_generator_path(packet):
    out, _ = packet
    m = json.loads((out / MANIFEST_FILE).read_text(encoding="utf-8"))
    m["files"]["synth/x.py"] = "0" * 64
    m["packet_sha256"] = packet_digest(m["files"])
    (out / MANIFEST_FILE).write_text(json.dumps(m), encoding="utf-8")
    problems = verify_packet(out)
    assert not any("packet_sha256" in x for x in problems), problems
    assert any("lists a generator/dataset path: synth/x.py" in x for x in problems), problems


def test_verify_detects_model_edit_after_consistent_rehash(packet):
    """The packet model must restore to source_model_sha256, even if MANIFEST files are rehashed."""
    out, _ = packet
    p = out / "model.py"
    p.write_text(p.read_text(encoding="utf-8") + "# appended\n", encoding="utf-8")
    _rehash_manifest(out)
    problems = verify_packet(out)
    assert not any("hash mismatch" in x for x in problems)
    assert any("differs from the source model" in x for x in problems), problems


def test_verify_detects_repeated_inlined_block_after_rehash(packet):
    """M0_CONTRACTS §D [R2]: verify_packet checks the inlined block occurs at most once."""
    out, _ = packet
    p = out / "model.py"
    p.write_text(p.read_text(encoding="utf-8") + "\n" + inlined_constants_block() + "\n", encoding="utf-8")
    _rehash_manifest(out)
    problems = verify_packet(out)
    assert not any("hash mismatch" in x for x in problems)
    assert "model.py: inlined-constants block occurs more than once" in problems, problems


def test_verify_detects_non_standalone_model(packet):
    out, _ = packet
    p = out / "model.py"
    p.write_text(original_model_source(p.read_text(encoding="utf-8")), encoding="utf-8")
    _rehash_manifest(out)
    problems = verify_packet(out)
    assert any("not standalone" in x for x in problems), problems


def test_verify_detects_reworded_instructions_leak(packet):
    """A capacity note that dodges the phrase list is still caught: the instructions are regenerated."""
    out, _ = packet
    p = out / "AUTHOR_INSTRUCTIONS.md"
    leak = "\n## Capacity\n\nGive each window slot its own detector unit; same-position conjunctions are cheap.\n"
    p.write_text(p.read_text(encoding="utf-8") + leak, encoding="utf-8")
    _rehash_manifest(out)
    problems = verify_packet(out)
    assert find_capacity_hints(leak) == []
    assert not any("hash mismatch" in x for x in problems)
    assert any("AUTHOR_INSTRUCTIONS.md: does not match" in x for x in problems), problems


def test_verify_reports_crlf(packet):
    out, _ = packet
    p = out / "feature_index.json"
    p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
    _rehash_manifest(out)
    problems = verify_packet(out)
    assert any("feature_index.json: contains CR line endings" in x for x in problems), problems


def test_verify_detects_stray_file(packet):
    out, _ = packet
    (out / "notes.txt").write_text("hi", encoding="utf-8")
    problems = verify_packet(out)
    assert any("stray" in x and "notes.txt" in x for x in problems), problems


def test_verify_detects_stray_generator_dir(packet):
    out, _ = packet
    (out / "synth").mkdir()
    (out / "synth" / "gen.py").write_text("x = 1\n", encoding="utf-8")
    (out / "datasets").mkdir()
    problems = verify_packet(out)
    assert any("stray" in x and "synth/gen.py" in x for x in problems), problems
    assert any("generator/dataset" in x and "synth" in x for x in problems), problems
    assert any("generator/dataset" in x and "datasets" in x for x in problems), problems


def test_verify_detects_missing_file(packet):
    out, _ = packet
    (out / "feature_index.json").unlink()
    problems = verify_packet(out)
    assert any("missing" in x and "feature_index.json" in x for x in problems), problems


def test_verify_detects_missing_manifest(packet):
    out, _ = packet
    (out / MANIFEST_FILE).unlink()
    assert any("missing" in x and MANIFEST_FILE in x for x in verify_packet(out))


def test_verify_detects_manifest_tampering(packet):
    out, _ = packet
    m = json.loads((out / MANIFEST_FILE).read_text(encoding="utf-8"))
    m["packet_sha256"] = "0" * 64
    (out / MANIFEST_FILE).write_text(json.dumps(m), encoding="utf-8")
    assert any("packet_sha256" in x for x in verify_packet(out))


def test_verify_detects_manifest_listing_extra_file(packet):
    out, _ = packet
    (out / "extra.md").write_text("x", encoding="utf-8")
    m = json.loads((out / MANIFEST_FILE).read_text(encoding="utf-8"))
    m["files"]["extra.md"] = "0" * 64
    m["packet_sha256"] = packet_digest(m["files"])
    (out / MANIFEST_FILE).write_text(json.dumps(m), encoding="utf-8")
    problems = verify_packet(out)
    assert any("file list" in x for x in problems), problems
    assert any("stray" in x and "extra.md" in x for x in problems), problems


def test_verify_content_checks_survive_consistent_rehash(packet):
    """Even with a self-consistent manifest, leaked sections and hints are caught."""
    out, _ = packet
    p = out / "SPEC_AUTHORING.md"
    p.write_text(
        p.read_text(encoding="utf-8")
        + "\n### 2.3 Design note on capacity\n\nOne unit per position.\n\n### 7.2 Generator requirements\n",
        encoding="utf-8",
    )
    _rehash_manifest(out)
    problems = verify_packet(out)
    assert not any("hash mismatch" in x for x in problems)
    for needle in ("'### 2.3'", "'### 7.2'", "'per position'", "'Design note on capacity'", "'Generator requirements'"):
        assert any(needle in x for x in problems), (needle, problems)


def test_verify_detects_missing_required_sections(packet):
    out, _ = packet
    p = out / "SPEC_AUTHORING.md"
    p.write_text(p.read_text(encoding="utf-8").replace("\n## 6. ", "\n## Six. "), encoding="utf-8")
    _rehash_manifest(out)
    assert any("'## 6.'" in x for x in verify_packet(out))


def test_verify_detects_inconsistent_feature_index(packet):
    out, _ = packet
    p = out / "feature_index.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    doc["entries"][0]["flat_index"], doc["entries"][1]["flat_index"] = 1, 0
    p.write_text(json.dumps(doc), encoding="utf-8")
    _rehash_manifest(out)
    assert any("feature_index.json" in x for x in verify_packet(out))


def test_verify_missing_dir(tmp_path):
    assert verify_packet(tmp_path / "nope") != []


def test_build_refuses_dir_with_stray_files(tmp_path):
    out = tmp_path / "packet"
    out.mkdir()
    (out / "leftover.txt").write_text("x", encoding="utf-8")
    with pytest.raises(PacketError, match="leftover.txt"):
        build_packet(SPEC_PATH, MODEL_PATH, out)
    assert sorted(p.name for p in out.iterdir()) == ["leftover.txt"]


def test_build_rejects_spec_without_required_sections(tmp_path):
    spec = tmp_path / "spec.md"
    spec.write_text("# T\n\n## 1. A\n", encoding="utf-8")
    with pytest.raises(PacketError, match="required section"):
        build_packet(spec, MODEL_PATH, tmp_path / "packet")


# -- script ---------------------------------------------------------------------


def test_build_script(tmp_path):
    script = REPO_ROOT / "scripts" / "build_authoring_packet.py"
    out = tmp_path / "pkt"
    r = subprocess.run([sys.executable, str(script), "--out", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    manifest = json.loads((out / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["packet_sha256"] in r.stdout
    r = subprocess.run([sys.executable, str(script), "--out", str(out), "--verify-only"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    (out / "stray.txt").write_text("x", encoding="utf-8")
    r = subprocess.run([sys.executable, str(script), "--out", str(out), "--verify-only"], capture_output=True, text=True)
    assert r.returncode == 1
    assert "stray" in r.stderr
    shutil.rmtree(out)
