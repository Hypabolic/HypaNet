"""v0.4 WSP authoring packet (spec v0.4 §5, §13; src/agent_loop_lab/v04/packet.py)."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from agent_loop_lab.v03 import packet as p3
from agent_loop_lab.v04 import packet as P

REPO = Path(__file__).resolve().parents[1]
V04 = REPO / "agent_doom_loop_detector_spec_v0_4.md"
V02 = REPO / "agent_doom_loop_detector_spec_v0_2.md"
SRC = REPO / "src" / "agent_loop_lab" / "v03"
HANDED_OVER_V03 = REPO / "experiments" / "v0_3" / "packet_handed_over"
WSP = REPO / "envs" / "qwen" / "hypaqwen" / "wsp"
PINS = P.qwen_pins()
SNAPSHOT = P.qwen_snapshot_dir(PINS["model_id"], PINS["revision"])
needs_snapshot = pytest.mark.skipif(not (SNAPSHOT / "tokenizer.json").is_file(),
                                    reason="the pinned Qwen snapshot is not in the local Hugging Face cache")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture(scope="module")
def packet(tmp_path_factory) -> Path:
    if not (SNAPSHOT / "tokenizer.json").is_file():
        pytest.skip("the pinned Qwen snapshot is not in the local Hugging Face cache")
    out = tmp_path_factory.mktemp("v04pkt") / "packet"
    P.build_packet(V04, V02, SRC, out)
    return out


def _retamper(copy: Path, name: str) -> None:
    """Re-hash ``name`` in the manifest (and its source entry), so only the content checks can catch it."""
    manifest = json.loads((copy / P.MANIFEST_FILE).read_text())
    manifest["files"][name] = _sha((copy / name).read_bytes())
    if f"v03/{name}" in manifest["sources"]:
        manifest["sources"][f"v03/{name}"] = manifest["files"][name]
    (copy / P.MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


# -- build -------------------------------------------------------------------------------------


def test_packet_is_deterministic(packet, tmp_path):
    again = tmp_path / "again"
    m = P.build_packet(V04, V02, SRC, again)
    assert m["packet_sha256"] == P.packet_sha256(packet)
    for name in P.PACKET_FILES:
        assert (again / name).read_bytes() == (packet / name).read_bytes(), name
    assert P.verify_packet(packet) == []
    P.build_packet(V04, V02, SRC, again)  # rebuilding over a previous packet is allowed
    assert P.packet_sha256(again) == m["packet_sha256"]


def test_packet_contents_and_manifest(packet):
    entries = sorted(p.relative_to(packet).as_posix() for p in packet.rglob("*"))
    assert entries == sorted({*P.PACKET_FILES, "qwen"})
    raw = (packet / P.MANIFEST_FILE).read_bytes()
    manifest = json.loads(raw)
    assert sorted(manifest["files"]) == sorted(P.CONTENT_FILES)
    for name, sha in manifest["files"].items():
        assert _sha((packet / name).read_bytes()) == sha, name
    assert manifest["spec_version"] == "0.4" and manifest["sources"]["spec_v0_4"] == _sha(V04.read_bytes())
    assert manifest["sources"]["spec_v0_2"] == _sha(V02.read_bytes())
    assert manifest["interface"] == {"name": "w1-state-full", "spec_sha256": P.INTERFACE_SPEC_SHA256,
                                     "label_ids": list(P.LABEL_IDS), "prefix_tokens": 167, "suffix_tokens": 11}
    assert manifest["qwen"]["revision"] == PINS["revision"]
    for n in P.PACKET_SOURCE_FILES:
        assert (packet / n).read_bytes() == (SRC / n).read_bytes()
    text = raw.decode()
    assert str(REPO) not in text and "/Users/" not in text and "utc" not in text.lower()
    assert not any("safetensors" in e or "weights" in e.lower() for e in entries)  # the kernel serves the weights


@needs_snapshot
def test_qwen_files_are_the_pinned_checkpoint_files(packet):
    pins = P.qwen_pins()
    model_src = (REPO / "envs" / "qwen" / "hypaqwen" / "model.py").read_text()
    for name in P.QWEN_FILES:
        data = (packet / "qwen" / name).read_bytes()
        assert data == (SNAPSHOT / name).read_bytes()
        assert _sha(data) == pins["file_sha256"][name]
        assert f'"{name}": "{_sha(data)}"' in model_src  # the pin as model.py spells it


@needs_snapshot
def test_qwen_sha_mismatch_is_refused(tmp_path):
    fake = tmp_path / "snap"
    fake.mkdir()
    for name in P.QWEN_FILES:
        shutil.copy(SNAPSHOT / name, fake / name)
    (fake / "config.json").write_bytes((SNAPSHOT / "config.json").read_bytes() + b"\n")
    with pytest.raises(P.PacketError, match="config.json"):
        P.build_packet(V04, V02, SRC, tmp_path / "pkt", qwen_snapshot=fake)
    (fake / "config.json").unlink()
    with pytest.raises(P.PacketError, match="missing: config.json"):
        P.build_packet(V04, V02, SRC, tmp_path / "pkt2", qwen_snapshot=fake)


def test_snapshot_dir_follows_the_hub_cache_rules():
    rev = PINS["revision"]
    tail = Path("models--Qwen--Qwen3.5-0.8B") / "snapshots" / rev
    assert P.qwen_snapshot_dir("Qwen/Qwen3.5-0.8B", rev, {"HF_HUB_CACHE": "/c"}) == Path("/c") / tail
    assert P.qwen_snapshot_dir("Qwen/Qwen3.5-0.8B", rev, {"HF_HOME": "/h"}) == Path("/h/hub") / tail
    assert P.qwen_snapshot_dir("Qwen/Qwen3.5-0.8B", rev, {"XDG_CACHE_HOME": "/x"}) == Path("/x/huggingface/hub") / tail


# -- SPEC.md -----------------------------------------------------------------------------------


def test_spec_md_keeps_packet_sections_only(packet):
    spec = (packet / "SPEC.md").read_text()
    assert "[packet]" not in spec
    for kept in ("## 1. Objective", "### 1.1 What a good result", "## 3. Target model (frozen)",
                 "## 4. Log rendering (frozen)", "### 4.3 Body lines", "### 4.7 Word-level tokens",
                 "## 5. Task interface (frozen)", "Gate rule.", "## 7. Weight-space programming protocol",
                 "### 7.6 Method: banned and allowed", "### 7.7 WSP-A (zero labels)",
                 "## 8. Deliverables and build checks", "#### 6.1 Latent state", "#### 6.3 Window labelling rules",
                 "#### 7.1 Scenario families", "`N_stall` | 5", "2,000,000", "50,000", "996,224,640"):
        assert kept in spec, kept
    for withheld in ("## 0.", "### 0.1", "## 2.", "## 9.", "## 10.", "## 11.", "## 12.", "## 13.", "## 14.",
                     "## 15.", "## 16."):
        assert not re.search(rf"^{re.escape(withheld)}", spec, re.M), withheld
    for phrase in ("Qwen-0", "LoRA", "fresh sealed", "87,176", "Revision 1", "US$100", "registry_v0_4",
                   "forward_disclosed", "score_dev", "Sections marked"):
        assert phrase not in spec, phrase
    # Section 6 contributes only what follows its paragraph marker.
    assert "Every Qwen column is scored" not in spec


def test_spec_md_v02_part_is_what_the_v03_author_read(packet):
    """WSP-A sees the same labelling rules v0.3 A saw: the v0.2 sections are byte-identical."""
    v04 = (packet / "SPEC.md").read_text()
    v03 = (HANDED_OVER_V03 / "SPEC.md").read_text()
    assert v03.count(p3.V02_PREAMBLE) == 1 and v04.count(P.V02_PREAMBLE) == 1
    assert v04.split(P.V02_PREAMBLE)[1] == v03.split(p3.V02_PREAMBLE)[1]
    assert v04.startswith(P.SPEC_PREAMBLE)


def test_spec_v04_states_the_frozen_interface():
    """The spec's §5 carries the exact prefix and suffix, label words and ids of the frozen interface."""
    iface = P.load_interface()
    text = V04.read_text()
    assert json.dumps(iface.PREFIX, ensure_ascii=False) in text
    assert json.dumps(iface.SUFFIX, ensure_ascii=False) in text
    assert iface.spec_sha256() in text and '`"{block}\\n"`' in text
    for w, i in zip(iface.LABEL_WORDS, P.LABEL_IDS, strict=True):
        assert f"`{w}` | {i} |" in text, w
    for d in iface.DEFINITIONS:  # spec v0.3 §2 / interface.DEFINITIONS, word for word
        assert d[:-1] in text


# -- INTERFACE.md ------------------------------------------------------------------------------


def test_interface_md_matches_interface_py(packet):
    iface = P.load_interface()
    text = (packet / "INTERFACE.md").read_text()
    assert iface.spec_sha256() == P.INTERFACE_SPEC_SHA256
    body = {"prefix": iface.PREFIX, "block_template": iface.BLOCK_TEMPLATE, "suffix": iface.SUFFIX,
            "label_words": list(iface.LABEL_WORDS)}
    assert _sha(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()) == P.INTERFACE_SPEC_SHA256
    assert f"`{P.INTERFACE_SPEC_SHA256}`" in text and f"`{iface.FROZEN.name}`" in text
    assert json.dumps(iface.PREFIX, ensure_ascii=False) in text
    assert json.dumps(iface.SUFFIX, ensure_ascii=False) in text
    assert json.dumps(iface.BLOCK_TEMPLATE) in text
    assert json.dumps(list(P.PREFIX_IDS)) in text and json.dumps(list(P.SUFFIX_IDS)) in text
    for k, (c, w, i) in enumerate(zip(iface.CLASS_NAMES, iface.LABEL_WORDS, P.LABEL_IDS, strict=True)):
        assert f"| {k} | {c} | `{json.dumps(w)}` | {i} |" in text
    assert text == P.interface_md(iface, (packet / "qwen" / "tokenizer.json").read_bytes())


@needs_snapshot
def test_pinned_ids_spell_the_interface():
    iface = P.load_interface()
    vocab = P._Vocab((SNAPSHOT / "tokenizer.json").read_bytes())
    assert vocab.decode(P.PREFIX_IDS) == iface.PREFIX and vocab.decode(P.SUFFIX_IDS) == iface.SUFFIX
    assert [vocab.word_id(w) for w in iface.LABEL_WORDS] == list(P.LABEL_IDS)


@needs_snapshot
def test_a_wrong_pin_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "SUFFIX_IDS", P.SUFFIX_IDS[:-1] + (26,))
    with pytest.raises(P.PacketError, match="SUFFIX_IDS"):
        P.build_packet(V04, V02, SRC, tmp_path / "pkt")
    monkeypatch.undo()
    monkeypatch.setattr(P, "LABEL_IDS", (25399, 39923, 60477, 65552))
    with pytest.raises(P.PacketError, match="label word"):
        P.build_packet(V04, V02, SRC, tmp_path / "pkt2")


def test_interface_loads_by_path_and_must_stay_stdlib(tmp_path):
    src = (REPO / "envs" / "qwen" / "hypaqwen" / "interface.py").read_text()
    bad = tmp_path / "interface.py"
    bad.write_text(src.replace("import hashlib\n", "import hashlib\nimport torch\n", 1))
    with pytest.raises(P.PacketError, match="torch"):
        P.load_interface(bad)
    changed = tmp_path / "changed.py"
    changed.write_text(src.replace('"State:"', '"Status:"'))
    with pytest.raises(P.PacketError, match="spec_sha256"):
        P.load_interface(changed)


# -- GRAMMAR_SHEET.md, AUTHOR_INSTRUCTIONS.md ---------------------------------------------------


def test_grammar_sheet_is_the_v03_authors(packet):
    assert (packet / "GRAMMAR_SHEET.md").read_bytes() == (HANDED_OVER_V03 / "GRAMMAR_SHEET.md").read_bytes()
    assert _sha((packet / "GRAMMAR_SHEET.md").read_bytes()) == P.GRAMMAR_SHEET_SHA256


def test_author_instructions_cover_the_wsp_protocol(packet):
    text = (packet / "AUTHOR_INSTRUCTIONS.md").read_text()
    assert text == P.AUTHOR_INSTRUCTIONS
    for s in ("from wsp import", "./py output/wsp-a/scratch/", "./fwd output/wsp-a/scratch/",
              "One command per Bash call", "def build(orig, consts)", "output/wsp-a/fwd/<call id>/",
              "EDIT.md", "ITERATIONS.md", "edit_constants.safetensors", "8 MiB", "200 KB", "300 s",
              "byte for byte", "2,000,000", "50,000", "996,224,640", "label_argmax_flips", "argmax_flips",
              "logsumexp", "<model>.s<session index>.<hidden|mlp_act>.L<layer>.npy", "0..24", "0..23",
              "0..20", "1e-3", "0.5", "keep-the-best", "cost cap"):
        assert s in text, s
    assert "$" not in text.split("cost cap")[1][:80]  # no number for the cap here
    # Every public function of the author client is documented.
    tree = ast.parse((WSP / "author" / "client.py").read_text())
    names = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets))
    funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")}
    for name in sorted((set(names) & funcs) - {"connect", "init_tok"}):
        assert f"`{name}(" in text, name
    for name in ("NAMES", "SHAPES", "ARCH", "LABEL_IDS", "CLASS_NAMES", "tok", "EditReceipt", "EditRefused",
                 "BudgetRefused", "RequestError", "QuotaRefused", "KernelError"):
        assert f"`{name}`" in text, name
    receipt = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EditReceipt")
    for field in (s.target.id for s in receipt.body if isinstance(s, ast.AnnAssign)):
        assert f"`{field}`" in text, field
    # The request schema keys of protocol.validate_forward_request, and the RawOp fields of v0.3.
    for key in ('"model"', '"sessions"', '"id"', '"blocks"', '"windows"', '"return"', '"hidden"', '"mlp_act"',
                '"topk"', '"edited"', '"original"', '"both"'):
        assert key in text, key
    assert P.rawop_section() in text and P.rawop_section() in p3.AUTHOR_INSTRUCTIONS
    for field in ("family", "target", "args", "outcome", "error_class", "error_location", "validation", "changed",
                  "extra", "input_tokens", "output_tokens", "timestamp", "dropped"):
        assert f"`{field}`" in text, field


def _literal(path: Path, name: str):
    return P._literal_assign(path, name)


def test_numbers_match_the_kernel_sources():
    assert _literal(WSP / "server.py", "PHASE_BUDGET") == P.PHASE_BUDGET
    assert _literal(WSP / "server.py", "REQUEST_CAP") == P.REQUEST_CAP
    assert _literal(WSP / "author" / "protocol.py", "MAX_TOPK") == 20
    assert _literal(WSP / "author" / "protocol.py", "N_LAYERS") == 24
    assert _literal(WSP / "edit_state.py", "SCALE_C_TOL") == 1e-3
    assert _literal(WSP / "edit_state.py", "SCALE_RESID_TAU") == 0.5
    assert "MAX_CONSTS_BYTES = 8 * 1024 * 1024" in (WSP / "runner.py").read_text() and P.MAX_CONSTS_BYTES == 8 << 20


@needs_snapshot
def test_load_quota_matches_the_checkpoint():
    weights = SNAPSHOT / "model.safetensors-00001-of-00001.safetensors"
    with open(weights, "rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    skip = {"model.language_model.embed_tokens.weight", "model.language_model.norm.weight"}
    layer = sum(math.prod(v["shape"]) for k, v in header.items()
                if k.startswith("model.language_model.") and k not in skip)
    frac = _literal(WSP / "server.py", "LOAD_QUOTA_FRACTION")
    assert int(frac * 4 * layer) == P.LOAD_QUOTA_BYTES


# -- leakage -----------------------------------------------------------------------------------


def test_built_packet_has_no_leaks(packet):
    texts = {n: (packet / n).read_text() for n in P.TEXT_FILES}
    texts[P.MANIFEST_FILE] = (packet / P.MANIFEST_FILE).read_text()
    assert P.leakage_problems(texts) == []
    for name in P.TEXT_FILES:
        if name.endswith(".py"):
            assert not re.search(r"^\s*(from|import)\s+(agent_loop_lab|hypaqwen|transformers)", texts[name], re.M)


@pytest.mark.parametrize("name,inject,why", [
    ("AUTHOR_INSTRUCTIONS.md", "\nThe Qwen-0 val macro-F1 was 0.41.\n", "withheld v0.4 text"),
    ("INTERFACE.md", "\nsee datasets/v0_4/val.tasks.jsonl.gz\n", "withheld v0.4 text"),
    ("GRAMMAR_SHEET.md", "\nscored on the dev split\n", "data split"),
    ("SPEC.md", "\nPhase C\n", "withheld text"),
    ("AUTHOR_INSTRUCTIONS.md", "\nthe OOD engine nulls input_tokens\n", "generators"),
    ("renderer.py", "\nimport hypaqwen.interface\n", "imports"),
    ("tokenizer.py", "\nfrom agent_loop_lab.v03 import scoring\n", "imports"),
])
def test_a_planted_leak_fails_verification(packet, tmp_path, name, inject, why):
    copy = tmp_path / "pkt"
    shutil.copytree(packet, copy)
    f = copy / name
    f.write_text(f.read_text() + inject)
    _retamper(copy, name)
    problems = P.verify_packet(copy)
    assert any(name in p and why in p for p in problems), problems


def test_a_leak_in_a_packet_section_refuses_the_build(tmp_path):
    spec = tmp_path / "spec.md"
    spec.write_text(V04.read_text().replace("### 7.8 Later phases\n", "### 7.8 Later phases\n\nLoRA is a baseline.\n"))
    with pytest.raises(P.PacketError, match="LoRA"):
        P.build_packet(spec, V02, SRC, tmp_path / "pkt")
    assert not (tmp_path / "pkt" / P.MANIFEST_FILE).exists() or P.verify_packet(tmp_path / "pkt")


def test_verify_detects_tampering_and_strays(packet, tmp_path):
    copy = tmp_path / "pkt"
    shutil.copytree(packet, copy)
    (copy / "qwen" / "config.json").write_bytes(b"{}")
    assert any("hash mismatch: qwen/config.json" in p for p in P.verify_packet(copy))
    _retamper(copy, "qwen/config.json")
    assert any("pinned checkpoint" in p for p in P.verify_packet(copy))
    shutil.rmtree(copy)
    shutil.copytree(packet, copy)
    (copy / "qwen" / "model.safetensors").write_bytes(b"x")
    assert any("stray entry: qwen/model.safetensors" in p for p in P.verify_packet(copy))
    (copy / "qwen" / "model.safetensors").unlink()
    (copy / "AUTHOR_INSTRUCTIONS.md").write_text(P.AUTHOR_INSTRUCTIONS + "\nUse ./py freely.\n")
    _retamper(copy, "AUTHOR_INSTRUCTIONS.md")
    assert any("builder's text" in p for p in P.verify_packet(copy))


def test_cli_build_and_verify_only(packet, tmp_path):
    script = REPO / "scripts" / "v04_build_packet.py"
    r = subprocess.run([sys.executable, str(script), "--out", str(packet), "--verify-only"], capture_output=True,
                       text=True)
    assert r.returncode == 0 and r.stdout.splitlines()[-1] == f"packet sha256: {P.packet_sha256(packet)}"
    src = tmp_path / "src"
    src.mkdir()
    shutil.copy(SRC / "tokenizer.py", src / "tokenizer.py")
    r = subprocess.run([sys.executable, str(script), "--out", str(tmp_path / "p2"), "--source-dir", str(src)],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "renderer.py" in r.stderr


# -- packet promises the kernel must keep (review fixes) -----------------------------------------


def test_interface_md_one_pass_is_equal_only_up_to_reassociation(packet):
    """INTERFACE.md overrides SPEC §5, so it must not claim bit-exact equality with a separate prompt."""
    text = (packet / "INTERFACE.md").read_text()
    assert "up\n  to floating-point reassociation (not bit for bit)" in text
    assert "equals the separate prompt above.\n" not in text
    assert "per window up to floating-point reassociation" in (packet / "SPEC.md").read_text()


def test_ledger_promise_matches_fwd(packet):
    """./fwd rejects an invalid request file (exit 2) before sending it, so only requests that reach the
    server are ledgered; the packet must say so (fwd.py main: validate -> return 2 before _send)."""
    src = (WSP / "author" / "fwd.py").read_text()
    main = src[src.index("def main("):]
    assert main.index("return 2") < main.index("_send(")
    for name in ("SPEC.md", "AUTHOR_INSTRUCTIONS.md"):
        flat = " ".join((packet / name).read_text().split())
        assert "every request, run or refused" not in flat.lower(), name
        assert "request that reaches the model server, run or refused" in flat, name
        assert "rejects as invalid (exit status 2) is not sent: it costs nothing, has no call id" in flat, name


def test_grammar_sheet_v03_readout_is_flagged(packet):
    """GRAMMAR_SHEET.md stays v0.3's bytes (BOS, EOE, event_ord, readout at EOE); SPEC and the
    instructions say those parts do not apply and that ./fwd blocks carry no EOE."""
    assert "read at that block's `EOE`" in (packet / "GRAMMAR_SHEET.md").read_text()
    for name in ("SPEC.md", "AUTHOR_INSTRUCTIONS.md"):
        flat = " ".join((packet / name).read_text().split())
        assert "`BOS`, `EOE`, `event_ord` and network-readout sentences describe v0.3's model and do not apply" \
            in flat, name
        assert "Block strings sent to `./fwd` contain no `EOE`" in flat, name


QWEN_PY = REPO / "envs" / "qwen" / ".venv" / "bin" / "python"
GUARD_DIR = WSP / "author" / "hn_guard"

_STUB_WSP = '''\
"""A kernel-free stand-in for tools/hn_kernel/wsp.py: the names the save recipe imports."""
import torch

EMBED = "model.language_model.embed_tokens.weight"
DOWN = "model.language_model.layers.0.mlp.down_proj.weight"
NAMES = (EMBED, DOWN, "model.language_model.norm.weight")
LABEL_IDS = (1, 3)
_ORIG = {EMBED: torch.arange(20.0).reshape(5, 4), DOWN: torch.ones(4, 6), NAMES[2]: torch.zeros(4)}
EDIT = {"stale": torch.zeros(1)}


def load_original(name):
    return _ORIG[name].clone()


def edited():
    return sorted(EDIT)


def drop_edit(name):
    return EDIT.pop(name, None) is not None


def save_edit(name, t):
    EDIT[name] = t.clone()


def save_label_rows(rows):
    EDIT[EMBED] = rows.clone()
'''

_STUB_BUILD = '''\
import torch


def build(orig, consts):
    W = orig["model.language_model.layers.0.mlp.down_proj.weight"]
    E = orig["model.language_model.embed_tokens.weight"]
    E[[1, 3]] = E[[1, 3]] + torch.outer(torch.ones(2), consts["u"])
    return {"model.language_model.layers.0.mlp.down_proj.weight": W + torch.outer(consts["u"], torch.ones(6)),
            "model.language_model.embed_tokens.weight": E}
'''

_LAUNCH = r'''
import os, runpy, sys
AD, SCRIPT = sys.argv[1], sys.argv[2]
sys.path[:0] = [os.path.join(AD, "tools", "hn_kernel"), os.path.join(AD, "packet"), os.path.dirname(SCRIPT),
                os.path.dirname(os.path.dirname(SCRIPT))]
import torch, safetensors.torch
safetensors.torch.save_file({"u": torch.full((4,), 0.5)}, os.path.join(AD, "output", "wsp-a", "edit_constants.safetensors"))
import wsp
sys.path.insert(0, os.path.join(AD, "tools"))
from hn_guard import guards_v04 as g
g.install("KERNEL_CLIENT", author_dir=AD, phase="wsp-a", kernel_endpoint=os.path.join(AD, ".kernel", "kernel.sock"))
runpy.run_path(SCRIPT, run_name="__main__")
print("EDITED " + " ".join(f"{k}:{float(v.sum())}" for k, v in sorted(wsp.EDIT.items())), flush=True)
'''


def _save_recipe(text: str) -> str:
    """The ``./py`` script AUTHOR_INSTRUCTIONS gives for applying the final edit through build()."""
    start = text.index("It must reproduce the server's edit exactly")
    block = text[text.index("  ```python\n", start) + len("  ```python\n"):]
    block = block[:block.index("  ```\n")]
    return "\n".join(line[2:] for line in block.splitlines()) + "\n"


@pytest.mark.skipif(not QWEN_PY.exists(), reason="the envs/qwen venv (torch 2.14.1) is not built")
def test_save_recipe_runs_under_the_kernel_client_guard(packet):
    """The documented final-save script must not trip the ./py guard (torch.use_deterministic_algorithms
    imports torch._inductor, which KERNEL_CLIENT refuses: exit 3)."""
    text = (packet / "AUTHOR_INSTRUCTIONS.md").read_text()
    recipe = _save_recipe(text)
    assert "torch.set_num_threads(1)" in recipe and "use_deterministic_algorithms" not in recipe
    assert "`torch.use_deterministic_algorithms`" in text.split("**Refused, and fatal to the run**")[1][:1200]
    import os
    import tempfile

    root = Path(os.path.realpath(tempfile.mkdtemp(prefix="hnr", dir="/tmp")))  # short AF_UNIX-safe path
    try:
        ad = root / "a"
        scratch = ad / "output" / "wsp-a" / "scratch"
        for d in (ad / "tools" / "hn_kernel", ad / "packet", ad / ".kernel", scratch):
            d.mkdir(parents=True)
        shutil.copytree(GUARD_DIR, ad / "tools" / "hn_guard", ignore=shutil.ignore_patterns("__pycache__"))
        (ad / "tools" / "hn_kernel" / "wsp.py").write_text(_STUB_WSP)
        (ad / "output" / "wsp-a" / "build_edit.py").write_text(_STUB_BUILD)
        (scratch / "save_final.py").write_text(recipe)
        r = subprocess.run([str(QWEN_PY), "-I", "-B", "-c", _LAUNCH, str(ad), str(scratch / "save_final.py")],
                           cwd=str(ad), capture_output=True, text=True, timeout=300)
        assert r.returncode == 0 and "HN_GUARD" not in r.stderr, r.stdout[-2000:] + r.stderr[-3000:]
        down = 6.0 * 4 + 0.5 * 6 * 4
        rows = float(sum(range(4, 8)) + sum(range(12, 16))) + 0.5 * 8
        assert r.stdout.splitlines()[-1] == (f"EDITED model.language_model.embed_tokens.weight:{rows} "
                                             f"model.language_model.layers.0.mlp.down_proj.weight:{down}")
    finally:
        shutil.rmtree(root, ignore_errors=True)
