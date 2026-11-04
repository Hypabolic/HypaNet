"""Harness build runner (spec v0.2 §9, §11, §15): guarded build, archive, rejections."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest
import torch

from agent_loop_lab import runner
from agent_loop_lab.model import PARAM_SHAPES, deserialize_state_dict, load_artifact
from agent_loop_lab.packet import build_packet, verify_packet
from agent_loop_lab.runner import (
    ARTIFACT_FILE,
    ARTIFACT_SHA_FILE,
    RECORD_FILE,
    BuildError,
    build_phase,
    check_build_source,
    check_reference_policy_source,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = REPO_ROOT / "agent_doom_loop_detector_spec_v0_2.md"
MODEL_PATH = REPO_ROOT / "src" / "agent_loop_lab" / "model.py"
SCRIPT = REPO_ROOT / "scripts" / "run_build.py"

AUTHOR_MODULES = ("model", "build_weights", "reference_policy")

HONEST_BUILD = '''\
"""Toy honest author: zero_model plus a handful of literal assignments."""
import math

import torch
from model import zero_model, DoomLoopMLP

GAIN = 2.0


def build() -> DoomLoopMLP:
    m = zero_model()
    with torch.no_grad():
        m.fc1.weight[0, 0] = 1.0
        m.fc1.weight[0, 8] = -0.5
        m.fc1.bias[0] = -0.25
        m.fc2.weight[0, 0] = GAIN
        m.fc3.weight[2, 0] = 1.0
        m.fc3.bias[0] = math.ldexp(1.0, -3)
    return m


if __name__ == "__main__":
    from model import sparsity
    print(sparsity(build()))
'''

HONEST_POLICY = '''\
def reference_policy(x: tuple[float, ...]) -> int:
    return 0
'''

EXPECTED_NONZERO = {
    ("fc1.weight", (0, 0)): 1.0,
    ("fc1.weight", (0, 8)): -0.5,
    ("fc1.bias", (0,)): -0.25,
    ("fc2.weight", (0, 0)): 2.0,
    ("fc3.weight", (2, 0)): 1.0,
    ("fc3.bias", (0,)): 0.125,
}


def _build_body(body: str, header: str = "") -> str:
    """A build_weights.py whose build() runs ``body`` (indented) after zero_model()."""
    return (
        "import torch\nfrom model import zero_model, DoomLoopMLP\n"
        + textwrap.dedent(header)
        + "\n\ndef build():\n    m = zero_model()\n"
        + textwrap.indent(textwrap.dedent(body), "    ")
        + "\n    return m\n"
    )


@pytest.fixture(scope="module")
def packet_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("packet") / "authoring_packet"
    build_packet(SPEC_PATH, MODEL_PATH, out)
    return out


@pytest.fixture(autouse=True)
def _no_repo_registry(tmp_path, monkeypatch):
    """In-process tests never consult the repository's real freeze registry through registry_path="auto"."""
    monkeypatch.setattr(runner, "AUTO_REGISTRY_PATH", tmp_path / "no-registry" / "registry.yaml")


@pytest.fixture(autouse=True)
def _clean_modules_and_path(packet_dir):
    """Every test must leave sys.modules, sys.path and the packet as it found them."""
    before_path = list(sys.path)
    before_models = {n: sys.modules.get(n) for n in AUTHOR_MODULES}
    yield
    assert sys.path == before_path
    assert {n: sys.modules.get(n) for n in AUTHOR_MODULES} == before_models
    assert verify_packet(packet_dir) == []


def make_author(
    root: Path,
    build_src: str = HONEST_BUILD,
    policy_src: str = HONEST_POLICY,
    *,
    circuits: bool = True,
    extra: dict[str, str] | None = None,
) -> Path:
    d = root / "author"
    d.mkdir(parents=True, exist_ok=True)
    (d / "build_weights.py").write_text(build_src, encoding="utf-8")
    (d / "reference_policy.py").write_text(policy_src, encoding="utf-8")
    if circuits:
        (d / "CIRCUITS.md").write_text("# Circuits\n\nUnit h1[0]: toy.\n", encoding="utf-8")
    for name, text in (extra or {}).items():
        (d / name).write_text(text, encoding="utf-8")
    return d


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _assert_no_archive(out: Path) -> None:
    assert not (out / ARTIFACT_FILE).exists()
    assert not (out / RECORD_FILE).exists()


def _attempts(out: Path) -> list[dict]:
    """The attempt records under ``out/attempts/``, in index order."""
    d = out / "attempts"
    return [json.loads(p.read_bytes()) for p in sorted(d.iterdir())] if d.is_dir() else []


# -- honest author ------------------------------------------------------------------


def test_honest_build_archives_and_records_hashes(tmp_path, packet_dir):
    author = make_author(tmp_path)
    out = tmp_path / "archive" / "phase-a-strict"
    record = build_phase("phase-a-strict", author, packet_dir, out)

    assert sorted(p.name for p in out.iterdir()) == sorted(
        [ARTIFACT_FILE, ARTIFACT_SHA_FILE, RECORD_FILE, "build_weights.py", "reference_policy.py", "CIRCUITS.md",
         "attempts"]
    )
    assert [p.name for p in (out / "attempts").iterdir()] == ["0001.json"]
    for name in ("build_weights.py", "reference_policy.py", "CIRCUITS.md"):
        assert (out / name).read_bytes() == (author / name).read_bytes()

    manifest = json.loads((packet_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    metadata = {
        "phase": "phase-a-strict",
        "spec_version": "0.2",
        "packet_sha256": manifest["packet_sha256"],
        "build_weights_sha256": _sha(author / "build_weights.py"),
        "reference_policy_sha256": _sha(author / "reference_policy.py"),
        "circuits_sha256": _sha(author / "CIRCUITS.md"),
    }
    model, meta = load_artifact(out / ARTIFACT_FILE)
    assert meta == metadata, "artefact metadata: exactly these keys, no timestamps or paths"
    sd = model.state_dict()
    for name, shape in PARAM_SHAPES.items():
        expected = torch.zeros(shape)
        for (n, idx), v in EXPECTED_NONZERO.items():
            if n == name:
                expected[idx] = v
        assert torch.equal(sd[name], expected), name

    artifact_sha = _sha(out / ARTIFACT_FILE)
    assert (out / ARTIFACT_SHA_FILE).read_text(encoding="ascii") == f"{artifact_sha}  {ARTIFACT_FILE}\n"
    raw = (out / RECORD_FILE).read_bytes()
    on_disk = json.loads(raw)
    assert on_disk == record
    assert raw == json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    for k, v in metadata.items():
        assert record[k] == v
    assert record["artifact_sha256"] == artifact_sha
    assert record["files"] == {n: _sha(author / n) for n in ("CIRCUITS.md", "build_weights.py", "reference_policy.py")}
    assert record["sparsity"]["overall"]["nonzero"] == len(EXPECTED_NONZERO)
    assert record["sparsity"]["overall"]["slots"] == 3256
    checks = record["checks"]
    assert checks["scan_source_findings"] == [] and checks["anti_training_guard"] == "passed"
    assert checks["rebuild_check"] == "byte-identical in a fresh interpreter"
    assert checks["compute_ban"] == "enforced"
    assert checks["packet_registered"] is False, "no registry: auto finds none"
    assert "only call" in checks["reference_policy_smoke_call"]
    assert checks["packet_model_matches_harness"] == f"source_model_sha256 {_sha(MODEL_PATH)}"
    text = raw.decode("utf-8")
    for path in (str(tmp_path), str(packet_dir), str(REPO_ROOT)):
        assert path not in text, "no absolute paths in the record"
    assert not any(k in text.lower() for k in ("timestamp", "created", "_time", "datetime")), "no timestamps"
    # No bytecode caches were dropped next to the author's or the packet's sources.
    assert not list(author.rglob("__pycache__")) and not list(packet_dir.rglob("__pycache__"))


def test_two_independent_builds_are_byte_identical(tmp_path, packet_dir):
    a = make_author(tmp_path / "a")
    b = make_author(tmp_path / "b")  # same content, different location
    out_a, out_b = tmp_path / "out-a", tmp_path / "out-b"
    ra = build_phase("phase-a-strict", a, packet_dir, out_a, rebuild_check=False)
    rb = build_phase("phase-a-strict", b, packet_dir, out_b, rebuild_check=False)
    assert (out_a / ARTIFACT_FILE).read_bytes() == (out_b / ARTIFACT_FILE).read_bytes()
    assert (out_a / RECORD_FILE).read_bytes() == (out_b / RECORD_FILE).read_bytes()
    assert ra == rb
    # And the fresh-interpreter rebuild reproduces exactly the same bytes.
    again = runner._rebuild_in_fresh_interpreter("phase-a-strict", b, packet_dir)
    assert again == (out_a / ARTIFACT_FILE).read_bytes()
    data, meta = deserialize_state_dict(again)
    assert meta["phase"] == "phase-a-strict"


def test_selfcheck_and_revision_phases(tmp_path, packet_dir):
    author = make_author(tmp_path, extra={"ITERATIONS.md": "# Iterations\n", "REVISION.md": "# Revision\n"})
    r1 = build_phase("phase-a-selfcheck", author, packet_dir, tmp_path / "sc", rebuild_check=False)
    assert (tmp_path / "sc" / "ITERATIONS.md").read_bytes() == b"# Iterations\n"
    assert set(r1["files"]) == {"build_weights.py", "reference_policy.py", "CIRCUITS.md", "ITERATIONS.md"}
    r2 = build_phase("phase-b-revision", author, packet_dir, tmp_path / "b", rebuild_check=False)
    assert (tmp_path / "b" / "REVISION.md").exists()
    assert r1["phase"] == "phase-a-selfcheck" and r2["phase"] == "phase-b-revision"
    # The phase is part of the metadata, so the artefacts differ although the weights do not.
    assert r1["artifact_sha256"] != r2["artifact_sha256"]


def test_selfcheck_requires_iterations_log(tmp_path, packet_dir):
    author = make_author(tmp_path)
    with pytest.raises(BuildError, match=r"required file.*ITERATIONS\.md"):
        build_phase("phase-a-selfcheck", author, packet_dir, tmp_path / "out", rebuild_check=False)
    with pytest.raises(BuildError, match=r"required file.*REVISION\.md"):
        build_phase("phase-b-revision", author, packet_dir, tmp_path / "out", rebuild_check=False)
    _assert_no_archive(tmp_path / "out")


def test_unknown_phase(tmp_path, packet_dir):
    with pytest.raises(BuildError, match="unknown phase"):
        build_phase("phase-c", make_author(tmp_path), packet_dir, tmp_path / "out")


def test_preexisting_model_module_is_restored(tmp_path, packet_dir):
    sentinel = types.ModuleType("model")
    sys.modules["model"] = sentinel
    try:
        build_phase("phase-a-strict", make_author(tmp_path), packet_dir, tmp_path / "out", rebuild_check=False)
        assert sys.modules["model"] is sentinel
    finally:
        del sys.modules["model"]


# -- rejections: static rules --------------------------------------------------------

STATIC_REJECTIONS = {
    "module_level_optim": (
        "import torch\nfrom model import zero_model\n\nOPT = torch.optim.SGD([torch.zeros(1)], lr=0.1)\n\n"
        "def build():\n    return zero_model()\n",
        r"scan_source.*optim",
    ),
    "backward": (_build_body("x = torch.zeros(1)\nx.sum().backward()\n"), r"scan_source.*backward"),
    "import_random": (_build_body("m.fc1.bias[0] = random.random()\n", "import random\n"), r"scan_source.*'random'"),
    "numpy_random": (
        _build_body("m.fc1.bias[0] = float(np.random.rand())\n", "import numpy as np\n"),
        r"scan_source.*np\.random",
    ),
    "open_file": (
        _build_body("m.fc1.bias[0] = float(open('weights.txt').read())\n"),
        r"harness build rules.*builtin open\(\)",
    ),
    "numpy_load": (
        _build_body("m.fc1.weight.copy_(torch.from_numpy(np.load('w.npy')))\n", "import numpy as np\n"),
        r"harness build rules.*\.load\(\)",
    ),
    "pathlib_read": (
        _build_body("Path('w.txt').read_text()\n", "from pathlib import Path\n"),
        r"harness build rules.*banned module 'pathlib'",
    ),
    "save_artifact": (
        _build_body("save_artifact(m, 'x.dlw')\n", "from model import save_artifact\n"),
        r"harness build rules.*save_artifact\(\)",
    ),
    "thread": (
        _build_body("threading.Thread(target=lambda: None).start()\n", "import threading\n"),
        r"harness build rules.*banned module 'threading'",
    ),
    "missing_build": (
        "import torch\nfrom model import zero_model\n\ndef make():\n    return zero_model()\n",
        r"no module-level function build\(\)",
    ),
    "relative_import": (_build_body("pass\n", "from . import helpers\n"), r"relative import"),
    "scipy_import": (_build_body("pass\n", "import json\nimport scipy\n"), r"'scipy', which is not allowed"),
}


@pytest.mark.parametrize("case", sorted(STATIC_REJECTIONS))
def test_static_rejections(case, tmp_path, packet_dir):
    src, pattern = STATIC_REJECTIONS[case]
    author = make_author(tmp_path, src)
    out = tmp_path / "out"
    with pytest.raises(BuildError, match=pattern):
        build_phase("phase-a-strict", author, packet_dir, out, rebuild_check=False)
    _assert_no_archive(out)


POLICY_REJECTIONS = {
    "numpy": ("import numpy as np\n\ndef reference_policy(x):\n    return int(np.argmax(x))\n", r"'numpy', which is not allowed"),
    "torch": ("from torch import tensor\n\ndef reference_policy(x):\n    return 0\n", r"'torch', which is not allowed"),
    "random": ("import random\n\ndef reference_policy(x):\n    return random.randrange(4)\n", r"banned module 'random'"),
    "time": ("from time import time\n\ndef reference_policy(x):\n    return 0\n", r"banned module 'time'"),
    "os_path": ("import os.path\n\ndef reference_policy(x):\n    return 0\n", r"banned module 'os.path'"),
    "open": ("def reference_policy(x):\n    open('log.txt', 'a')\n    return 0\n", r"builtin open\(\)"),
    "global": ("N = 0\n\ndef reference_policy(x):\n    global N\n    N += 1\n    return 0\n", r"global statement"),
    "missing": ("def policy(x):\n    return 0\n", r"no module-level function reference_policy\(\)"),
    "syntax": ("def reference_policy(x)\n    return 0\n", r"cannot parse"),
}


@pytest.mark.parametrize("case", sorted(POLICY_REJECTIONS))
def test_reference_policy_rejections(case, tmp_path, packet_dir):
    src, pattern = POLICY_REJECTIONS[case]
    author = make_author(tmp_path, policy_src=src)
    with pytest.raises(BuildError, match=r"reference_policy\.py.*" + pattern):
        build_phase("phase-a-strict", author, packet_dir, tmp_path / "out", rebuild_check=False)
    _assert_no_archive(tmp_path / "out")


def test_reference_policy_allows_standard_library(tmp_path):
    p = tmp_path / "reference_policy.py"
    p.write_text(
        "from __future__ import annotations\nimport math\nfrom fractions import Fraction\nimport itertools, functools\n\n"
        "THRESHOLD = 0.5\n\ndef reference_policy(x: tuple[float, ...]) -> int:\n"
        "    return 2 if math.fsum(x[8::15]) > THRESHOLD else 0\n",
        encoding="utf-8",
    )
    assert check_reference_policy_source(p) == []


def test_static_findings_have_no_absolute_paths(tmp_path):
    p = tmp_path / "build_weights.py"
    p.write_text("import os\n\ndef make():\n    open('x')\n", encoding="utf-8")
    findings = check_build_source(p)
    assert findings == [
        "build_weights.py:0: no module-level function build() is defined",
        "build_weights.py:1: import of banned module 'os'",
        "build_weights.py:4: call to builtin open() (file, console or dynamic-code access)",
    ]
    assert check_build_source(tmp_path.joinpath("build_weights.py")) == findings


def test_missing_circuits(tmp_path, packet_dir):
    author = make_author(tmp_path, circuits=False)
    with pytest.raises(BuildError, match=r"required file.*CIRCUITS\.md"):
        build_phase("phase-a-strict", author, packet_dir, tmp_path / "out", rebuild_check=False)
    _assert_no_archive(tmp_path / "out")


# -- rejections: the guarded run -----------------------------------------------------

RUNTIME_REJECTIONS = {
    "float64_state_dict": (_build_body("m = m.double()\n"), r"state_dict validation.*float32"),
    "wrong_shape_state_dict": (
        _build_body("m.fc3.bias.data = torch.zeros(5)\n"),
        r"state_dict validation.*fc3\.bias: shape",
    ),
    "non_finite": (_build_body("m.fc1.bias[3] = float('inf')\n"), r"state_dict validation.*non-finite"),
    "returns_state_dict": (
        "import torch\nfrom model import zero_model\n\ndef build():\n    return zero_model().state_dict()\n",
        r"build\(\) returned OrderedDict, not a DoomLoopMLP",
    ),
    "build_raises": (_build_body("raise ValueError('nope')\n"), r"build_weights\.py: build\(\): ValueError: nope"),
    "build_takes_args": (
        "from model import zero_model\n\ndef build(scale):\n    return zero_model()\n",
        r"build\(\): TypeError",
    ),
    # Evades the static scan with a computed name; the runtime guard still refuses the optimiser.
    "evasive_optimiser_swallowed": (
        _build_body(
            "name = 'op' + 'tim'\ntry:\n    getattr(torch, name).SGD([torch.zeros(1)], lr=0.1)\nexcept Exception:\n    pass\n"
        ),
        r"anti-training guard.*SGD",
    ),
    # Evades the static rules; the runtime I/O tripwire refuses the file read (and survives being swallowed).
    "evasive_file_read": (
        _build_body("name = 'lo' + 'ad'\ntry:\n    getattr(torch, name)('weights.pt')\nexcept Exception:\n    pass\n"),
        r"I/O tripwire.*weights\.pt",
    ),
    "policy_returns_bool": (HONEST_BUILD, r"must return an int in 0\.\.3, got bool"),
    "policy_out_of_range": (HONEST_BUILD, r"must return an int in 0\.\.3, got int"),
    "policy_raises": (HONEST_BUILD, r"reference_policy\(all-zero input\): IndexError"),
}
POLICIES = {
    "policy_returns_bool": "def reference_policy(x):\n    return False\n",
    "policy_out_of_range": "def reference_policy(x):\n    return 4\n",
    "policy_raises": "def reference_policy(x):\n    return x[120]\n",
}


@pytest.mark.parametrize("case", sorted(RUNTIME_REJECTIONS))
def test_runtime_rejections(case, tmp_path, packet_dir):
    src, pattern = RUNTIME_REJECTIONS[case]
    author = make_author(tmp_path, src, POLICIES.get(case, HONEST_POLICY))
    out = tmp_path / "out"
    with pytest.raises(BuildError, match=pattern):
        build_phase("phase-a-strict", author, packet_dir, out, rebuild_check=False)
    _assert_no_archive(out)


GUARD_ONLY = {
    # Sources that the static checks reject; run straight through the guarded build to show the
    # runtime layer catches them on its own.
    "module_level_optim": (
        "import torch\nfrom model import zero_model\n\nOPT = torch.optim.SGD([torch.zeros(1)], lr=0.1)\n\n"
        "def build():\n    return zero_model()\n",
        r"module-level code.*anti-training guard.*SGD",
    ),
    "backward": (
        _build_body("x = torch.ones(1)\nx.backward()\n"),
        r"anti-training guard.*backward",
    ),
    "thread": (
        _build_body("threading.Thread(target=lambda: None).start()\n", "import threading\n"),
        r"anti-training guard.*thread creation",
    ),
    "open_file": (
        _build_body("open({tmp!r} + '/weights.txt')\n"),
        r"I/O tripwire|file access",
    ),
    "write_file": (
        _build_body("from model import save_artifact\nsave_artifact(m, {tmp!r} + '/x.py')\n"),
        r"file access.*x\.py', mode 'w",
    ),
    "missing_build": ("x = 1\n", r"does not define a callable build\(\)"),
}


@pytest.mark.parametrize("case", sorted(GUARD_ONLY))
def test_guarded_build_rejects_without_static_checks(case, tmp_path, packet_dir):
    src, pattern = GUARD_ONLY[case]
    author = make_author(tmp_path, src.replace("{tmp!r}", repr(str(tmp_path))))
    (tmp_path / "weights.txt").write_text("1.0\n", encoding="utf-8")
    with pytest.raises(BuildError, match=pattern):
        runner._guarded_build(author, packet_dir)
    assert not (tmp_path / "x.py").exists()


def test_rebuild_check_detects_non_determinism(tmp_path, packet_dir):
    # str hashes are salted per interpreter, so this differs between the two builds.
    src = _build_body("m.fc1.bias[0] = float(hash('doom-loop') % 1000003)\n")
    author = make_author(tmp_path, src)
    with pytest.raises(BuildError, match="rebuild check.*not deterministic"):
        build_phase("phase-a-strict", author, packet_dir, tmp_path / "out")
    _assert_no_archive(tmp_path / "out")


def _record_steps(monkeypatch) -> list[str]:
    labels: list[str] = []
    real_step = runner._step

    def step(label, fn):
        labels.append(label)
        return real_step(label, fn)

    monkeypatch.setattr(runner, "_step", step)
    return labels


SMOKE_LABEL = "reference_policy.py: reference_policy(all-zero input)"


def test_reference_policy_is_called_exactly_once_per_build_phase(tmp_path, packet_dir, monkeypatch):
    """The packet text says the harness calls reference_policy exactly once; the rebuild must not call it again."""
    author = make_author(tmp_path)
    labels = _record_steps(monkeypatch)
    runner._guarded_build(author, packet_dir)
    assert labels.count(SMOKE_LABEL) == 1
    labels.clear()
    # The fresh-interpreter rebuild's entry point, run in-process: builds the weights, never touches the policy.
    assert runner._child_main(["phase-a-strict", str(author), str(packet_dir)]) == 0
    assert not any(label.startswith("reference_policy.py") for label in labels), labels


def test_rebuild_child_does_not_load_or_call_reference_policy(tmp_path, packet_dir):
    """A policy that would fail its smoke call does not fail the rebuild: the child never runs it."""
    author = make_author(tmp_path, policy_src="def reference_policy(x):\n    raise RuntimeError('called twice')\n")
    expected = runner._guarded_build(author, packet_dir, smoke_policy=False)
    again = runner._rebuild_in_fresh_interpreter("phase-a-strict", author, packet_dir)
    data, meta = deserialize_state_dict(again)
    assert meta["phase"] == "phase-a-strict"
    for name, tensor in expected.state_dict().items():
        assert torch.equal(data[name], tensor), name


def test_policy_smoke_call_still_guarded_and_checked(tmp_path, packet_dir):
    with pytest.raises(BuildError, match=r"reference_policy\(all-zero input\): RuntimeError: called twice"):
        runner._guarded_build(
            make_author(tmp_path, policy_src="def reference_policy(x):\n    raise RuntimeError('called twice')\n"),
            packet_dir,
        )


# -- packet must match the harness model ------------------------------------------------


def test_refuses_packet_built_from_a_different_model(tmp_path, packet_dir):
    """A packet built from another model.py (stale or edited) is refused: the author would design for another net."""
    alt_model = tmp_path / "model.py"
    text = MODEL_PATH.read_text(encoding="utf-8")
    assert "return self.fc3(h2)" in text
    alt_model.write_text(text.replace("return self.fc3(h2)", "return self.fc3(h2) * 2.0"), encoding="utf-8")
    alt_packet = tmp_path / "packet_alt"
    build_packet(SPEC_PATH, alt_model, alt_packet)
    assert verify_packet(alt_packet) == [], "the stale packet is internally consistent"
    out = tmp_path / "out"
    with pytest.raises(BuildError, match=r"different model\.py than the harness uses.*source_model_sha256"):
        build_phase("phase-a-strict", make_author(tmp_path), alt_packet, out, rebuild_check=False)
    assert sorted(p.name for p in out.iterdir()) == ["attempts"]


# -- archive immutability and inputs --------------------------------------------------


def test_existing_archive_is_refused_unless_forced(tmp_path, packet_dir):
    author = make_author(tmp_path)
    out = tmp_path / "out"
    first = build_phase("phase-a-strict", author, packet_dir, out, rebuild_check=False)
    before = (out / ARTIFACT_FILE).read_bytes()
    (author / "CIRCUITS.md").write_text("# Circuits, edited\n", encoding="utf-8")
    with pytest.raises(BuildError, match="archives are immutable"):
        build_phase("phase-a-strict", author, packet_dir, out, rebuild_check=False)
    assert (out / ARTIFACT_FILE).read_bytes() == before
    assert json.loads((out / RECORD_FILE).read_bytes()) == first
    forced = build_phase("phase-a-strict", author, packet_dir, out, rebuild_check=False, force=True)
    assert forced["circuits_sha256"] != first["circuits_sha256"]
    assert (out / "CIRCUITS.md").read_text(encoding="utf-8") == "# Circuits, edited\n"


def test_archive_in_author_dir(tmp_path, packet_dir):
    author = make_author(tmp_path)
    record = build_phase("phase-a-strict", author, packet_dir, author, rebuild_check=False)
    assert (author / ARTIFACT_FILE).exists() and record["files"]["build_weights.py"] == _sha(author / "build_weights.py")


def test_refuses_dirs_inside_packet(tmp_path, packet_dir):
    author = make_author(tmp_path)
    with pytest.raises(BuildError, match="outside the authoring packet"):
        build_phase("phase-a-strict", author, packet_dir, packet_dir / "out", rebuild_check=False)
    assert not (packet_dir / "out").exists()
    # The author directory alone inside the packet is refused too (checked before it is even looked at).
    with pytest.raises(BuildError, match="outside the authoring packet"):
        build_phase("phase-a-strict", packet_dir / "auth", packet_dir, tmp_path / "out", rebuild_check=False)
    assert not (packet_dir / "auth").exists()
    _assert_no_archive(tmp_path / "out")


def _wrap_guarded_build(monkeypatch, after) -> None:
    real = runner._guarded_build

    def wrapped(author_dir, packet_dir, **kwargs):
        harness = real(author_dir, packet_dir, **kwargs)
        after()
        return harness

    monkeypatch.setattr(runner, "_guarded_build", wrapped)


def test_author_file_edited_during_build_is_refused(tmp_path, packet_dir, monkeypatch):
    """An author file that changes while the build runs would make the record's hashes lie; nothing is written."""
    author = make_author(tmp_path)
    out = tmp_path / "archive" / "phase-a-strict"
    _wrap_guarded_build(monkeypatch, lambda: (author / "CIRCUITS.md").write_text("# edited mid-build\n", encoding="utf-8"))
    with pytest.raises(BuildError, match="author files changed during the build"):
        build_phase("phase-a-strict", author, packet_dir, out, rebuild_check=False)
    assert sorted(p.name for p in out.iterdir()) == ["attempts"]
    assert _attempts(out)[-1]["outcome"] == "rejected"


def test_archive_appearing_during_build_is_not_overwritten(tmp_path, packet_dir, monkeypatch):
    """M0_CONTRACTS §E step 7: refuse before building and again before writing."""
    author = make_author(tmp_path)
    out = tmp_path / "out"
    foreign = b"another build's artefact"

    def write_foreign():
        out.mkdir(parents=True, exist_ok=True)
        (out / ARTIFACT_FILE).write_bytes(foreign)

    _wrap_guarded_build(monkeypatch, write_foreign)
    with pytest.raises(BuildError, match="archives are immutable"):
        build_phase("phase-a-strict", author, packet_dir, out, rebuild_check=False)
    assert (out / ARTIFACT_FILE).read_bytes() == foreign
    assert sorted(p.name for p in out.iterdir()) == [ARTIFACT_FILE, "attempts"]
    assert _attempts(out)[-1]["error"] == "<out>/artifact.dlw already exists; archives are immutable (use --force to overwrite)"


def test_refuses_unverified_packet(tmp_path, packet_dir):
    bad = tmp_path / "packet"
    shutil.copytree(packet_dir, bad)
    p = bad / "AUTHOR_INSTRUCTIONS.md"
    p.write_bytes(p.read_bytes() + b"\nhint\n")
    with pytest.raises(BuildError, match="packet failed verification.*hash mismatch"):
        build_phase("phase-a-strict", make_author(tmp_path), bad, tmp_path / "out", rebuild_check=False)


def test_missing_author_dir(tmp_path, packet_dir):
    with pytest.raises(BuildError, match="author directory not found"):
        build_phase("phase-a-strict", tmp_path / "nope", packet_dir, tmp_path / "out", rebuild_check=False)


# -- CLI --------------------------------------------------------------------------------


def test_cli(tmp_path, packet_dir):
    author = make_author(tmp_path)
    out = tmp_path / "weight_program" / "phase-a-strict"
    cmd = [sys.executable, str(SCRIPT), "phase-a-strict", "--author-dir", str(author), "--packet-dir", str(packet_dir),
           "--out-dir", str(out), "--no-registry"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    sha = _sha(out / ARTIFACT_FILE)
    assert sha in r.stdout and "byte-identical" in r.stdout
    r = subprocess.run(cmd + ["--no-rebuild-check"], capture_output=True, text=True)
    assert r.returncode == 1 and "immutable" in r.stderr
    r = subprocess.run(cmd + ["--no-rebuild-check", "--force"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert _sha(out / ARTIFACT_FILE) == sha
    r = subprocess.run([sys.executable, str(SCRIPT), "phase-z", "--author-dir", "a", "--out-dir", "b"],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "invalid choice" in r.stderr


# -- compute ban (spec §0.2: no forward passes, matrix products or linear algebra in build()) -----

# Each case: (build_weights.py source, pattern). Runtime-only cases pass the static checks and are
# caught by the tripwire; static cases are caught before anything runs.
COMPUTE_BAN_RUNTIME = {
    "F_linear": (
        _build_body("y = F.linear(torch.zeros(1, 120), m.fc1.weight)\n", "import torch.nn.functional as F\n"),
        r"compute ban.*torch\.nn\.functional\.linear",
    ),
    "model_call": (_build_body("y = m(torch.zeros(1, 120))\n"), r"compute ban.*calling an nn\.Module"),
    "layer_call": (_build_body("y = m.fc1(torch.zeros(1, 120))\n"), r"compute ban.*calling an nn\.Module"),
    "torch_mm": (_build_body("y = torch.mm(m.fc2.weight, m.fc1.weight)\n"), r"compute ban.*torch\.mm"),
    "tensor_mm_method": (_build_body("y = m.fc2.weight.mm(m.fc1.weight)\n"), r"compute ban.*Tensor\.mm"),
    "tensor_mv_method": (_build_body("h = torch.relu(m.fc1.weight.mv(torch.ones(120)) + m.fc1.bias)\n"), r"compute ban.*Tensor\.mv"),
    "torch_outer": (_build_body("y = torch.outer(m.fc1.bias, m.fc2.bias)\n"), r"compute ban.*torch\.outer"),
    "swallowed": (
        _build_body("try:\n    m(torch.zeros(1, 120))\nexcept Exception:\n    pass\n"),
        r"compute ban.*calling an nn\.Module",
    ),
    "module_level": (
        "import torch\nfrom model import zero_model\n\nP = torch.outer(torch.ones(2), torch.ones(2))\n\n"
        "def build():\n    return zero_model()\n",
        r"compute ban.*torch\.outer",
    ),
}
COMPUTE_BAN_STATIC = {
    "matmul_operator": (_build_body("y = m.fc2.weight @ m.fc1.weight\n"), r"harness build rules.*@ operator"),
    "matmul_inplace_operator": (_build_body("w = m.fc1.weight.clone()\nw @= torch.eye(120)\n"), r"@ operator"),
    "torch_matmul": (_build_body("y = torch.matmul(m.fc2.weight, m.fc1.weight)\n"), r"harness build rules.*'matmul'"),
    "torch_linalg_lstsq": (
        _build_body("w = torch.linalg.lstsq(torch.eye(3), torch.ones(3, 1)).solution\n"),
        r"harness build rules.*'linalg'",
    ),
    "np_linalg_lstsq": (
        _build_body("w = np.linalg.lstsq(np.eye(3), np.ones(3), rcond=None)[0]\n", "import numpy as np\n"),
        r"harness build rules.*'linalg'",
    ),
    "forward_call": (_build_body("y = m.forward(torch.zeros(1, 120))\n"), r"harness build rules.*\.forward\(\)"),
    "pinverse": (_build_body("w = torch.pinverse(torch.eye(3))\n"), r"'pinv'"),
    "einsum_import": (_build_body("pass\n", "from torch import einsum\n"), r"'einsum'"),
    # numpy.ndarray cannot be patched at run time, so the method form ndarray.dot is refused statically.
    "ndarray_dot": (
        _build_body("W = m.fc1.weight.detach().numpy()\nv = W.dot(np.ones(120, dtype=np.float32))\n", "import numpy as np\n"),
        r"harness build rules.*\.dot\(\)",
    ),
}


@pytest.mark.parametrize("case", sorted(COMPUTE_BAN_RUNTIME))
def test_compute_ban_runtime_rejections(case, tmp_path, packet_dir):
    src, pattern = COMPUTE_BAN_RUNTIME[case]
    author = make_author(tmp_path, src)
    assert check_build_source(author / "build_weights.py") == [], "runtime-only case must pass the static rules"
    out = tmp_path / "out"
    with pytest.raises(BuildError, match=pattern):
        build_phase("phase-a-strict", author, packet_dir, out, rebuild_check=False)
    _assert_no_archive(out)
    assert _attempts(out)[-1]["outcome"] == "rejected"


@pytest.mark.parametrize("case", sorted(COMPUTE_BAN_STATIC))
def test_compute_ban_static_rejections(case, tmp_path, packet_dir):
    src, pattern = COMPUTE_BAN_STATIC[case]
    author = make_author(tmp_path, src)
    out = tmp_path / "out"
    with pytest.raises(BuildError, match=pattern):
        build_phase("phase-a-strict", author, packet_dir, out, rebuild_check=False)
    _assert_no_archive(out)


COMPUTE_BAN_RUNTIME_ONLY = {
    # Statically rejected sources, run straight through the guarded build: the tripwire catches them too.
    "matmul_operator": (_build_body("y = m.fc2.weight @ m.fc1.weight\n"), r"Tensor\.__matmul__"),
    "rmatmul": (_build_body("y = m.fc2.weight.__rmatmul__(m.fc1.weight)\n"), r"Tensor\.__rmatmul__"),
    "torch_matmul": (_build_body("y = torch.matmul(m.fc2.weight, m.fc1.weight)\n"), r"torch\.matmul"),
    "torch_linalg_lstsq": (_build_body("w = torch.linalg.lstsq(torch.eye(3), torch.ones(3, 1))\n"), r"torch\.linalg\.lstsq"),
    "np_linalg_lstsq": (
        _build_body("w = np.linalg.lstsq(np.eye(3), np.ones(3), rcond=None)\n", "import numpy as np\n"),
        r"numpy\.linalg\.lstsq",
    ),
    "np_dot": (_build_body("v = np.dot(np.ones(3), np.ones(3))\n", "import numpy as np\n"), r"numpy\.dot"),
    "tensor_dot_method": (_build_body("s = m.fc1.weight[0].dot(torch.ones(120))\n"), r"Tensor\.dot"),
    "np_einsum": (_build_body("w = np.einsum('i,i', np.ones(2), np.ones(2))\n", "import numpy as np\n"), r"numpy\.einsum"),
    "forward_call": (_build_body("y = m.forward(torch.zeros(1, 120))\n"), r"calling an nn\.Module"),
    "linear_forward": (_build_body("y = torch.nn.Linear.forward(m.fc1, torch.zeros(1, 120))\n"), r"nn\.Linear\.forward"),
}


@pytest.mark.parametrize("case", sorted(COMPUTE_BAN_RUNTIME_ONLY))
def test_compute_ban_tripwire_without_static_checks(case, tmp_path, packet_dir):
    src, pattern = COMPUTE_BAN_RUNTIME_ONLY[case]
    with pytest.raises(BuildError, match=r"compute ban.*" + pattern):
        runner._guarded_build(make_author(tmp_path, src), packet_dir)


def test_compute_ban_is_lifted_after_the_build(tmp_path, packet_dir):
    import numpy as np
    import torch.nn.functional as F

    before = [(owner, name, getattr(owner, name), name in vars(owner)) for owner, name, _ in runner._compute_targets()]
    for src in (HONEST_BUILD, COMPUTE_BAN_RUNTIME["F_linear"][0]):
        try:
            runner._guarded_build(make_author(tmp_path / str(len(src)), src), packet_dir)
        except BuildError:
            pass
    for owner, name, value, own in before:
        assert (name in vars(owner)) == own, name
        assert getattr(owner, name) is value or getattr(owner, name) == value, name
    a = torch.ones(2, 2)
    assert torch.equal(a @ a, torch.full((2, 2), 2.0)) and torch.equal(torch.matmul(a, a), a.mm(a))
    assert float(np.dot(np.ones(3), np.ones(3))) == 3.0 and np.linalg.norm(np.ones(4)) == 2.0
    lin = torch.nn.Linear(2, 1)
    with torch.no_grad():
        assert lin(torch.zeros(1, 2)).shape == (1, 1) and F.linear(a, a).shape == (2, 2)
    assert torch.linalg.det(torch.eye(2)) == 1.0


def test_compute_ban_allows_elementwise_arithmetic(tmp_path, packet_dir):
    src = _build_body(
        """\
        with torch.no_grad():
            m.fc1.weight[0] = torch.ones(120) * 0.5 + 1.0
            m.fc1.weight[1, 3:9] = torch.arange(6, dtype=torch.float32) / 4.0 - 0.5
            m.fc1.bias.fill_(-0.25)
            m.fc2.weight[0, :2] = torch.tensor([1.0, -1.0]) * 2.0
            m.fc2.weight.mul_(0.5)
            m.fc3.weight[2].copy_(torch.from_numpy((np.arange(12) % 3 == 0).astype(np.float32) * 1.5))
            m.fc3.bias[0] = float(np.float32(0.125) * 2) + abs(-0.5) ** 2
            m.fc2.bias[:] = torch.where(m.fc2.bias > 0, m.fc2.bias, torch.full((12,), 0.1))
            m.fc1.weight[2] = m.fc1.weight[0] * m.fc1.weight[1] - m.fc1.weight[0].sum() / 120.0
        """,
        "import numpy as np\n",
    )
    author = make_author(tmp_path, src)
    assert check_build_source(author / "build_weights.py") == []
    record = build_phase("phase-a-strict", author, packet_dir, tmp_path / "out", rebuild_check=False)
    assert record["checks"]["compute_ban"] == "enforced"
    model, _ = load_artifact(tmp_path / "out" / ARTIFACT_FILE)
    assert model.fc1.weight[0, 0].item() == 1.5 and model.fc2.weight[0, 1].item() == -1.0


def test_compute_ban_active_in_the_rebuild_subprocess(tmp_path, packet_dir):
    """The fresh-interpreter rebuild enforces the same ban (here: a swallowed forward pass)."""
    author = make_author(tmp_path, COMPUTE_BAN_RUNTIME["swallowed"][0])
    with pytest.raises(BuildError, match=r"rebuild check: fresh-interpreter build failed.*compute ban.*calling an nn\.Module"):
        runner._rebuild_in_fresh_interpreter("phase-a-strict", author, packet_dir)
    author2 = make_author(tmp_path / "b", COMPUTE_BAN_RUNTIME_ONLY["np_linalg_lstsq"][0])
    with pytest.raises(BuildError, match=r"fresh-interpreter build failed.*compute ban.*numpy\.linalg\.lstsq"):
        runner._rebuild_in_fresh_interpreter("phase-a-strict", author2, packet_dir)


def test_compute_ban_static_findings(tmp_path):
    p = tmp_path / "build_weights.py"
    p.write_text(
        "import torch\nimport numpy.linalg as la\nfrom model import zero_model\n\n"
        "def build():\n    m = zero_model()\n    y = m.fc1.weight @ m.fc1.weight.T\n    m.forward(y)\n"
        "    torch.einsum('ij->i', y)\n    return m\n",
        encoding="utf-8",
    )
    why = "build() may not run forward passes, matrix products or linear-algebra solvers"
    assert check_build_source(p) == [
        f"build_weights.py:2: use of 'numpy.linalg' ('linalg' is not allowed: {why})",
        f"build_weights.py:7: use of the @ operator (matrix product; {why})",
        f"build_weights.py:8: call to .forward() (forward pass; {why})",
        f"build_weights.py:9: use of 'einsum' ('einsum' is not allowed: {why})",
    ]


# -- attempt log (spec §0.2, §10.2) ------------------------------------------------------


def _assert_no_abs_paths(out: Path, *roots: Path) -> None:
    for p in (out / "attempts").iterdir():
        text = p.read_text(encoding="utf-8")
        for root in roots:
            assert str(root) not in text and str(root.resolve()) not in text, (p.name, text)


def test_attempt_record_on_success(tmp_path, packet_dir):
    author = make_author(tmp_path)
    out = tmp_path / "out"
    build_phase("phase-a-strict", author, packet_dir, out, rebuild_check=False)
    raw = (out / "attempts" / "0001.json").read_bytes()
    rec = json.loads(raw)
    manifest = json.loads((packet_dir / "MANIFEST.json").read_bytes())
    assert rec == {
        "attempt": 1,
        "phase": "phase-a-strict",
        "outcome": "built",
        "error": None,
        "author_files": {n: _sha(author / n) for n in ("CIRCUITS.md", "build_weights.py", "reference_policy.py")},
        "packet_sha256": manifest["packet_sha256"],
    }
    assert raw == json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    _assert_no_abs_paths(out, tmp_path, packet_dir, REPO_ROOT)


def test_attempt_record_on_rejection(tmp_path, packet_dir):
    author = make_author(tmp_path, STATIC_REJECTIONS["backward"][0])
    out = tmp_path / "out"
    with pytest.raises(BuildError) as ei:
        build_phase("phase-a-strict", author, packet_dir, out, rebuild_check=False)
    assert sorted(p.name for p in out.iterdir()) == ["attempts"], "only the attempt record is written"
    (rec,) = _attempts(out)
    assert rec["attempt"] == 1 and rec["outcome"] == "rejected" and rec["phase"] == "phase-a-strict"
    assert rec["error"] == str(ei.value).replace(str(author), "<author>") and "scan_source" in rec["error"]
    assert "<author>/build_weights.py:" in rec["error"]
    assert rec["author_files"]["build_weights.py"] == _sha(author / "build_weights.py")
    assert rec["packet_sha256"] == json.loads((packet_dir / "MANIFEST.json").read_bytes())["packet_sha256"]


def test_attempt_record_replaces_absolute_paths(tmp_path, packet_dir):
    out = tmp_path / "out"
    with pytest.raises(BuildError, match="author directory not found"):
        build_phase("phase-a-strict", tmp_path / "nope", packet_dir, out, rebuild_check=False)
    (rec,) = _attempts(out)
    assert rec["error"] == "author directory not found: <author>"
    assert rec["author_files"] == {}
    # An unverifiable packet: its hash is not known, and its path is replaced as well.
    bad = tmp_path / "packet"
    shutil.copytree(packet_dir, bad)
    (bad / "stray.txt").write_text("x\n", encoding="utf-8")
    with pytest.raises(BuildError, match="packet failed verification"):
        build_phase("phase-a-strict", make_author(tmp_path), bad, out, rebuild_check=False)
    rec = _attempts(out)[-1]
    assert rec["attempt"] == 2 and rec["packet_sha256"] is None
    _assert_no_abs_paths(out, tmp_path, packet_dir)


def test_attempt_records_number_sequentially_and_log_the_immutable_refusal(tmp_path, packet_dir):
    author = make_author(tmp_path)
    out = tmp_path / "out"
    build_phase("phase-a-strict", author, packet_dir, out, rebuild_check=False)
    with pytest.raises(BuildError, match="archives are immutable"):
        build_phase("phase-a-strict", author, packet_dir, out, rebuild_check=False)
    with pytest.raises(BuildError, match="unknown phase"):
        build_phase("phase-c", author, packet_dir, out, rebuild_check=False)
    build_phase("phase-a-strict", author, packet_dir, out, rebuild_check=False, force=True)
    (out / "attempts" / "0009.json").write_bytes(b"{}")  # a gap: the next index follows the highest
    with pytest.raises(BuildError, match="archives are immutable"):
        build_phase("phase-a-strict", author, packet_dir, out, rebuild_check=False)
    names = sorted(p.name for p in (out / "attempts").iterdir())
    assert names == ["0001.json", "0002.json", "0003.json", "0004.json", "0009.json", "0010.json"]
    recs = {p: json.loads((out / "attempts" / p).read_bytes()) for p in names if p != "0009.json"}
    assert [r["outcome"] for r in recs.values()] == ["built", "rejected", "rejected", "built", "rejected"]
    assert [r["attempt"] for r in recs.values()] == [1, 2, 3, 4, 10]
    immutable = "<out>/artifact.dlw already exists; archives are immutable (use --force to overwrite)"
    assert recs["0002.json"]["error"] == immutable and recs["0010.json"]["error"] == immutable
    assert recs["0003.json"]["phase"] == "phase-c" and recs["0003.json"]["error"].startswith("unknown phase")
    assert set(recs["0003.json"]["author_files"]) == {"CIRCUITS.md", "build_weights.py", "reference_policy.py"}
    _assert_no_abs_paths(out, tmp_path, packet_dir)


def test_attempt_record_for_a_rebuild_failure(tmp_path, packet_dir):
    author = make_author(tmp_path, _build_body("m.fc1.bias[0] = float(hash('doom-loop') % 1000003)\n"))
    out = tmp_path / "out"
    with pytest.raises(BuildError, match="not deterministic"):
        build_phase("phase-a-strict", author, packet_dir, out)
    (rec,) = _attempts(out)
    assert rec["outcome"] == "rejected" and "not deterministic" in rec["error"]
    _assert_no_abs_paths(out, tmp_path, packet_dir)


def test_no_attempt_record_inside_the_packet(tmp_path, packet_dir):
    with pytest.raises(BuildError, match="outside the authoring packet"):
        build_phase("phase-a-strict", make_author(tmp_path), packet_dir, packet_dir / "out", rebuild_check=False)
    assert not (packet_dir / "out").exists()


# -- registered packet (spec §8, §15.1) ----------------------------------------------------


def _write_registry(path: Path, packet_sha256: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# test registry\nfiles: {{}}\npacket_sha256: '{packet_sha256}'\nspec_version: '0.2'\n", encoding="utf-8")
    return path


def _packet_sha(packet_dir: Path) -> str:
    return json.loads((packet_dir / "MANIFEST.json").read_bytes())["packet_sha256"]


def test_auto_registry_path_is_the_repo_registry():
    from agent_loop_lab.registry import REGISTRY_RELPATH

    real = Path(runner.__file__).resolve().parents[2] / REGISTRY_RELPATH
    assert real == REPO_ROOT / "experiments" / "registry.yaml"


def test_registry_match(tmp_path, packet_dir):
    reg = _write_registry(tmp_path / "reg" / "registry.yaml", _packet_sha(packet_dir))
    record = build_phase("phase-a-strict", make_author(tmp_path), packet_dir, tmp_path / "out", rebuild_check=False,
                         registry_path=reg)
    assert record["checks"]["packet_registered"] is True
    assert str(tmp_path) not in (tmp_path / "out" / RECORD_FILE).read_text(encoding="utf-8")


def test_registry_mismatch(tmp_path, packet_dir):
    reg = _write_registry(tmp_path / "reg" / "registry.yaml", "0" * 64)
    out = tmp_path / "out"
    with pytest.raises(BuildError, match=r"not the registered one.*registry 0{64}"):
        build_phase("phase-a-strict", make_author(tmp_path), packet_dir, out, rebuild_check=False, registry_path=reg)
    _assert_no_archive(out)
    assert _attempts(out)[-1]["outcome"] == "rejected"


def test_registry_auto(tmp_path, packet_dir, monkeypatch):
    author = make_author(tmp_path)
    # No registry at the auto location: no check.
    assert not runner.AUTO_REGISTRY_PATH.exists()
    r = build_phase("phase-a-strict", author, packet_dir, tmp_path / "o1", rebuild_check=False)
    assert r["checks"]["packet_registered"] is False
    # A registry at the auto location is checked: match ...
    auto = _write_registry(tmp_path / "repo" / "experiments" / "registry.yaml", _packet_sha(packet_dir))
    monkeypatch.setattr(runner, "AUTO_REGISTRY_PATH", auto)
    r = build_phase("phase-a-strict", author, packet_dir, tmp_path / "o2", rebuild_check=False)
    assert r["checks"]["packet_registered"] is True
    # ... and mismatch.
    _write_registry(auto, "f" * 64)
    with pytest.raises(BuildError, match="not the registered one"):
        build_phase("phase-a-strict", author, packet_dir, tmp_path / "o3", rebuild_check=False)
    # registry_path=None disables the check even though a (mismatching) registry exists.
    r = build_phase("phase-a-strict", author, packet_dir, tmp_path / "o4", rebuild_check=False, registry_path=None)
    assert r["checks"]["packet_registered"] is False


def test_registry_explicit_missing_path(tmp_path, packet_dir):
    out = tmp_path / "out"
    missing = tmp_path / "nowhere" / "registry.yaml"
    with pytest.raises(BuildError, match="freeze registry not found"):
        build_phase("phase-a-strict", make_author(tmp_path), packet_dir, out, rebuild_check=False, registry_path=missing)
    _assert_no_archive(out)
    assert _attempts(out)[-1]["error"] == "freeze registry not found: <registry>"


def test_cli_registry_options(tmp_path, packet_dir):
    author = make_author(tmp_path)
    reg = _write_registry(tmp_path / "reg" / "registry.yaml", _packet_sha(packet_dir))
    base = [sys.executable, str(SCRIPT), "phase-a-strict", "--author-dir", str(author), "--packet-dir", str(packet_dir),
            "--no-rebuild-check"]
    r = subprocess.run(base + ["--out-dir", str(tmp_path / "o1"), "--registry", str(reg)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads((tmp_path / "o1" / RECORD_FILE).read_bytes())["checks"]["packet_registered"] is True
    r = subprocess.run(base + ["--out-dir", str(tmp_path / "o2"), "--registry", str(tmp_path / "missing.yaml")],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "freeze registry not found" in r.stderr
    assert (tmp_path / "o2" / "attempts" / "0001.json").exists()
    _write_registry(reg, "0" * 64)
    r = subprocess.run(base + ["--out-dir", str(tmp_path / "o3"), "--registry", str(reg)], capture_output=True, text=True)
    assert r.returncode == 1 and "not the registered one" in r.stderr
    r = subprocess.run(base + ["--out-dir", str(tmp_path / "o4"), "--no-registry"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads((tmp_path / "o4" / RECORD_FILE).read_bytes())["checks"]["packet_registered"] is False
    r = subprocess.run(base + ["--out-dir", str(tmp_path / "o5"), "--no-registry", "--registry", str(reg)],
                       capture_output=True, text=True)
    assert r.returncode == 2 and "not allowed with" in r.stderr
