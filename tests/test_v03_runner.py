"""v0.3 build runner (spec v0.3 §9, §11, §15): guarded fresh-interpreter builds, archive, rejections."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from agent_loop_lab.v03.artifact import deserialize_state_dict
from agent_loop_lab.v03.packet import build_packet, packet_sha256
from agent_loop_lab.v03.runner import (
    ARTIFACT_FILE,
    ARTIFACT_SHA_FILE,
    RECORD_FILE,
    BuildError,
    build_phase,
    smoke_input,
)

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "agent_loop_lab" / "v03"

GOOD_BUILD = '''\
"""Toy honest author: a few parameters set with outer products and hashlib-derived signs."""
import hashlib

import torch

import model
import tokenizer

WORDS = ("read_file", "path", "FAILED")


def _signs(name, n):
    d = hashlib.sha256(name.encode()).digest()
    return torch.tensor([1.0 if (d[i // 8] >> (i % 8)) & 1 else -1.0 for i in range(n)])


def build():
    m = model.zero_model()
    with torch.no_grad():
        u = _signs("zeta", 16)
        for i, w in enumerate(WORDS):
            m.tok_emb.weight[tokenizer.word_id(w), :16] = _signs(w, 16)
        m.layers[0].attn.q.weight[0:16, 0:16] = torch.outer(u, u)
        m.layers[0].attn.rel_bias[:, 64] = 2.0
        m.layers[2].mlp.fc1.weight[:16, :16] = torch.kron(torch.eye(4), torch.ones(4, 4))
        m.head.bias[0] = 0.5
    return m
'''
GOOD_POLICY = '''\
def reference_policy(tokens, event_ord):
    return 2 if "FAILED" in tokens else 0
'''


@pytest.fixture(scope="module")
def packet(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("v03runpkt") / "packet"
    build_packet(REPO / "agent_doom_loop_detector_spec_v0_3.md", REPO / "agent_doom_loop_detector_spec_v0_2.md", SRC, out)
    return out


def _author(tmp_path: Path, build: str = GOOD_BUILD, policy: str = GOOD_POLICY, phase_file: str = "ITERATIONS.md") -> Path:
    d = tmp_path / "author"
    d.mkdir(exist_ok=True)
    (d / "build_weights.py").write_text(build)
    (d / "reference_policy.py").write_text(policy)
    (d / "CIRCUITS.md").write_text("# circuits\n")
    (d / phase_file).write_text("# log\n")
    return d


def _attempts(out: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((out / "attempts").glob("*.json"))]


def test_valid_submission_is_archived(tmp_path, packet):
    a = _author(tmp_path)
    out = tmp_path / "out"
    rec = build_phase("v03-phase-a", a, packet, out, registry_path=None)
    data = (out / ARTIFACT_FILE).read_bytes()
    assert rec["artifact_sha256"] == hashlib.sha256(data).hexdigest()
    assert (out / ARTIFACT_SHA_FILE).read_text().split()[0] == rec["artifact_sha256"]
    assert json.loads((out / RECORD_FILE).read_text()) == rec
    assert rec["packet_sha256"] == packet_sha256(packet)
    assert rec["checks"]["rebuild_check"].startswith("byte-identical")
    sp = rec["sparsity"]
    assert sp["layers.0.attn.q.weight"]["nonzero"] == 256
    assert sp["layers.0.attn.q.weight"]["blocks"] == [{"rows": [0, 15], "cols": [[0, 15]], "nonzero": 256}]
    assert sp["layers.0.attn.rel_bias"]["nonzero"] == 8
    assert sp["layers.1.attn.q.weight"] == {"nonzero": 0, "slots": 16384, "blocks": []}
    sd, meta = deserialize_state_dict(data)
    assert meta["phase"] == "v03-phase-a" and float(sd["head.bias"][0]) == 0.5
    for f in ("build_weights.py", "reference_policy.py", "CIRCUITS.md", "ITERATIONS.md"):
        assert (out / f).read_bytes() == (a / f).read_bytes()
    assert [x["outcome"] for x in _attempts(out)] == ["built"]
    with pytest.raises(BuildError, match="immutable"):
        build_phase("v03-phase-a", a, packet, out, registry_path=None)
    assert [x["outcome"] for x in _attempts(out)] == ["built", "rejected"]


def test_phase_b_needs_revision_md(tmp_path, packet):
    a = _author(tmp_path)
    with pytest.raises(BuildError, match="REVISION.md"):
        build_phase("v03-phase-b", a, packet, tmp_path / "out", registry_path=None)


def _reject(tmp_path, packet, match, **kw):
    a = _author(tmp_path, **kw)
    out = tmp_path / "out"
    with pytest.raises(BuildError, match=match):
        build_phase("v03-phase-a", a, packet, out, registry_path=None)
    att = _attempts(out)
    assert att[-1]["outcome"] == "rejected" and str(tmp_path) not in att[-1]["error"]
    assert not (out / ARTIFACT_FILE).exists()
    return att[-1]["error"]


STATIC_REJECTS = {
    "matmul": (GOOD_BUILD.replace("    return m\n", "    _ = m.head.weight @ m.head.weight.t()\n    return m\n"), "@ operator"),
    "random": ("import random\n" + GOOD_BUILD, "random"),
    "numpy": ("import numpy\n" + GOOD_BUILD, "numpy"),
    "forward": (GOOD_BUILD.replace("    return m\n", "    m.forward(1, 2, 3)\n    return m\n"), "forward"),
    "renderer": ("import renderer\n" + GOOD_BUILD, "renderer"),
    "big literal": (GOOD_BUILD + "\nX = [" + ", ".join(["1"] * 300) + "]\n", "300 elements"),
    "tokenizer non-literal": (GOOD_BUILD.replace('tokenizer.word_id(w)', 'tokenizer.word_id(w.upper())'), "tokenizer call"),
}


@pytest.mark.parametrize("name", sorted(STATIC_REJECTS))
def test_static_rejections(tmp_path, packet, name):
    build, match = STATIC_REJECTS[name]
    _reject(tmp_path, packet, match, build=build)


def test_policy_static_rejection(tmp_path, packet):
    _reject(tmp_path, packet, "tokenizer", policy="import tokenizer\n" + GOOD_POLICY)


def test_runtime_compute_ban_catches_computed_names(tmp_path, packet, monkeypatch):
    # the static rules refuse a computed getattr name; the run-time guard must refuse it on its own too
    _no_static_scan(monkeypatch)
    evasive = GOOD_BUILD.replace(
        "    return m\n",
        "    f = getattr(torch, 'mat' + 'mul')\n    try:\n        f(m.head.weight, m.head.weight.t())\n"
        "    except Exception:\n        pass\n    return m\n",
    )
    _reject(tmp_path, packet, "BUILD guard", build=evasive)


def test_runtime_module_call_is_refused(tmp_path, packet):
    calls = GOOD_BUILD.replace("    return m\n", "    _ = m.head(torch.zeros(1, 128))\n    return m\n")
    _reject(tmp_path, packet, "calling an nn.Module", build=calls)


def test_nondeterministic_build_fails_rebuild(tmp_path, packet):
    nondet = GOOD_BUILD.replace("m.head.bias[0] = 0.5", "m.head.bias[1] = float(id(object()) % 9973)")
    _reject(tmp_path, packet, "not deterministic", build=nondet)


def test_wrong_return_type(tmp_path, packet):
    _reject(tmp_path, packet, "not the packet's DoomLoopTransformer", build="import torch\n\ndef build():\n    return torch.nn.Identity()\n")


def test_policy_must_return_int(tmp_path, packet):
    _reject(tmp_path, packet, "int in 0..3", policy="def reference_policy(tokens, event_ord):\n    return True\n")


def test_policy_time_limit(tmp_path, packet):
    slow = "def reference_policy(tokens, event_ord):\n    n = 0\n    while True:\n        n += 1\n"
    _reject(tmp_path, packet, "0.5 s of CPU", policy=slow)


def test_registry_mismatch(tmp_path, packet):
    a = _author(tmp_path)
    reg = tmp_path / "registry.yaml"
    reg.write_text("packet_sha256: '" + "0" * 64 + "'\n")
    with pytest.raises(BuildError, match="not the registered one"):
        build_phase("v03-phase-a", a, packet, tmp_path / "out", registry_path=reg)


def test_packet_source_mismatch(tmp_path, packet):
    src = tmp_path / "src"
    shutil.copytree(SRC, src, ignore=shutil.ignore_patterns("__pycache__"))
    (src / "renderer.py").write_text((src / "renderer.py").read_text() + "\n# changed\n")
    with pytest.raises(BuildError, match="renderer.py differs"):
        build_phase("v03-phase-a", _author(tmp_path), packet, tmp_path / "out", registry_path=None, source_dir=src)


def test_smoke_input_is_one_grammar_block():
    tokens, event_ord = smoke_input()
    assert tokens[0] == "<BOS>" and tokens[-1] == "<EOE>" and "read_file" in tokens
    assert event_ord[0] == 0 and set(event_ord[1:]) == {1}


def test_cpu_budget_exceeded(tmp_path, packet, monkeypatch):
    monkeypatch.setenv("V03_BUILD_CPU_BUDGET_S", "1")
    spin = "import model\n\ndef build():\n    n = 0\n    while True:\n        n += 1\n"
    _reject(tmp_path, packet, "CPU budget exceeded", build=spin)


# -- review findings: side files, exfiltration through the error text, archive of the phase directory ---------

SIDE_TABLE_BUILD = GOOD_BUILD.replace(
    "    return m\n",
    "    o = getattr(type(m).__init__, '__glo'+'bals__')['__buil'+'tins__']['op'+'en']\n"
    "    vals = [float(x) for x in o(__file__.replace('build_weights.py', 'scratch/tbl.py')).read()[2:].split()]\n"
    "    m.head.bias[1] = vals[0]\n"
    "    return m\n",
)


def test_side_table_build_is_rejected_statically(tmp_path, packet):
    err = _reject(tmp_path, packet, "interpreter internals", build=SIDE_TABLE_BUILD)
    assert "__file__" in err


def _no_static_scan(monkeypatch):
    """Skip the static rules so the run-time half (fresh interpreter, BUILD guard, private copy) is tested alone."""
    from agent_loop_lab.v03 import runner

    monkeypatch.setattr(runner, "check_build_weights", lambda p: [])
    monkeypatch.setattr(runner, "check_reference_policy", lambda p: [])
    import agent_loop_lab.guards as g02

    monkeypatch.setattr(g02, "scan_source", lambda p: [])


@pytest.mark.parametrize("how", ["sibling", "absolute", "relative"])
def test_side_table_is_unreachable_at_run_time(tmp_path, packet, monkeypatch, how):
    _no_static_scan(monkeypatch)
    a = _author(tmp_path)
    (a / "scratch").mkdir()
    (a / "scratch" / "tbl.py").write_text("# " + " ".join(["0.25"] * 1000) + "\n")
    target = {
        "sibling": "__file__.replace('build_weights.py', 'scratch/tbl.py')",
        "absolute": repr(str(a / "scratch" / "tbl.py")),
        "relative": "'scratch/tbl.py'",
    }[how]
    build = GOOD_BUILD.replace(
        "    return m\n",
        f"    vals = [float(x) for x in open({target}).read()[2:].split()]\n    m.head.bias[1] = vals[0]\n    return m\n",
    )
    err = _reject(tmp_path, packet, "BUILD guard|FileNotFoundError", build=build)
    assert "0.25" not in err
    assert "v03_stage_" not in err  # the private copy's path is redacted
    if how == "sibling":  # (the other cases quote the author's own path literal)
        assert "/var/folders" not in err and "<author>" in err


def test_reading_harness_sources_and_the_spec_is_refused(tmp_path, packet, monkeypatch):
    _no_static_scan(monkeypatch)
    secret = REPO / "src" / "agent_loop_lab" / "v03" / "audits.py"
    build = GOOD_BUILD.replace("    return m\n", f"    raise ValueError(open({str(secret)!r}).read())\n")
    err = _reject(tmp_path, packet, "BUILD guard", build=build)
    assert secret.read_text()[:60] not in err
    build = GOOD_BUILD.replace(
        "    return m\n",
        f"    raise ValueError(bytes(torch.from_file({str(REPO / 'pyproject.toml')!r}, size=60, dtype=torch.uint8).tolist()))\n",
    )
    err = _reject(tmp_path, packet, "BUILD guard", build=build)
    assert "[project]" not in err and "hatchling" not in err


def test_author_exception_messages_are_not_disclosed(tmp_path, packet):
    build = GOOD_BUILD.replace("    return m\n", "    raise ValueError('SMUGGLED-' + 'X' * 50)\n")
    err = _reject(tmp_path, packet, r"ValueError raised at build_weights.py:\d+", build=build)
    assert "SMUGGLED" not in err and "message is not disclosed" in err


def test_child_output_is_not_disclosed(tmp_path, packet):
    build = GOOD_BUILD.replace(
        "    return m\n",
        "    print('V03-CHILD-ERROR: SMUGGLED')\n    print('V03-CHILD-RESULT: SMUGGLED')\n    return m.head\n",
    )
    out = tmp_path / "out"
    with pytest.raises(BuildError) as ei:
        build_phase("v03-phase-a", _author(tmp_path, build=build), packet, out, registry_path=None)
    assert "SMUGGLED" not in str(ei.value)
    assert "SMUGGLED" not in _attempts(out)[-1]["error"]


def test_error_text_is_capped(tmp_path, packet, monkeypatch):
    from agent_loop_lab.v03 import runner

    def boom(*a, **k):
        raise BuildError("x" * 10_000)

    monkeypatch.setattr(runner, "_static_checks", boom)
    out = tmp_path / "out"
    with pytest.raises(BuildError) as ei:
        build_phase("v03-phase-a", _author(tmp_path), packet, out, registry_path=None)
    assert len(str(ei.value)) <= runner.MAX_ERROR_CHARS
    assert len(_attempts(out)[-1]["error"]) <= runner.MAX_ERROR_CHARS


def test_whole_phase_directory_is_archived(tmp_path, packet):
    import os

    a = _author(tmp_path)
    (a / "scratch").mkdir()
    (a / "scratch" / "check.py").write_text("print('self-check')\n")
    (a / "scratch" / "inputs.json").write_text("[1, 2]\n")
    (a / "scratch" / "__pycache__").mkdir()
    (a / "scratch" / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    os.symlink("/nonexistent_v03_target", a / "scratch" / "link")
    out = tmp_path / "out"
    rec = build_phase("v03-phase-a", a, packet, out, registry_path=None, rebuild_check=False)
    listing = rec["author_output"]
    assert listing["scratch/check.py"] == hashlib.sha256(b"print('self-check')\n").hexdigest()
    assert listing["scratch/inputs.json"] == hashlib.sha256(b"[1, 2]\n").hexdigest()
    assert listing["scratch/link"] == {"symlink": "/nonexistent_v03_target"}
    assert not any("__pycache__" in k for k in listing)
    for name in ("build_weights.py", "reference_policy.py", "CIRCUITS.md", "ITERATIONS.md"):
        assert listing[name] == rec["files"][name]
    arch = out / "author_output"
    assert (arch / "scratch" / "check.py").read_text() == "print('self-check')\n"
    assert (arch / "scratch" / "link").is_symlink()
    assert json.loads((out / RECORD_FILE).read_text())["author_output"] == listing
    assert _attempts(out)[-1]["author_tree"] == listing


def test_build_only_sees_build_weights(tmp_path, packet, monkeypatch):
    """The build runs from a private copy: the author's other files are not next to it."""
    _no_static_scan(monkeypatch)
    build = GOOD_BUILD.replace(
        "    return m\n",
        "    import os\n    if sorted(os.listdir(os.path.dirname(__file__))) != ['build_weights.py']:\n"
        "        raise RuntimeError('author files visible')\n    return m\n",
    )
    a = _author(tmp_path, build=build)
    (a / "extra.py").write_text("x = 1\n")
    build_phase("v03-phase-a", a, packet, tmp_path / "out", registry_path=None, rebuild_check=False)


def test_evaluate_policy_guarded(tmp_path):
    from agent_loop_lab.v03.runner import evaluate_policy

    p = tmp_path / "reference_policy.py"
    p.write_text(
        "def reference_policy(tokens, event_ord):\n"
        "    if 'SPIN' in tokens:\n        while True:\n            pass\n"
        "    if 'BAD' in tokens:\n        return 7\n"
        "    if 'BOOM' in tokens:\n        raise KeyError('x')\n"
        "    return 2 if 'FAILED' in tokens else 0\n"
    )
    tokens, eo = smoke_input()
    inputs = [(tokens, eo), (tokens + ["FAILED"], eo + [1]), (["SPIN"], [1]), (["BAD"], [1]), (["BOOM"], [1])]
    assert evaluate_policy(p, inputs) == [0, 2, None, None, None]
    p.write_text("def reference_policy(tokens, event_ord):\n    return len(open('/etc/hosts').read()) % 4\n")
    with pytest.raises(BuildError, match="BUILD guard"):
        evaluate_policy(p, inputs[:1])
