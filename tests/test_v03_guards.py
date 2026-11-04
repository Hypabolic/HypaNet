"""v0.3 guard modes (spec v0.3 §9.1, §11 "Anti-training"): BUILD and AUTHOR_RUN."""

from __future__ import annotations

import random
import sys

import numpy as np
import pytest
import torch

from agent_loop_lab.v03 import model as M
from agent_loop_lab.v03.guards import TrainingViolation, author_run_guard, build_guard, v03_guard_active


def _snapshot():
    return {
        "torch.matmul": torch.matmul,
        "torch.rand": torch.rand,
        "torch.Generator": torch.Generator,
        "Tensor.__matmul__ own": "__matmul__" in vars(torch.Tensor),
        "Tensor.normal_ own": "normal_" in vars(torch.Tensor),
        "F.linear": torch.nn.functional.linear,
        "Module.__call__": vars(torch.nn.Module)["__call__"],
        "forward": vars(M.DoomLoopTransformer)["forward"],
        "random.random": random.random,
        "Random.__init__": vars(random.Random)["__init__"],
        "np.random.rand": np.random.rand,
        "monitoring": tuple(sys.monitoring.get_tool(i) for i in range(6)),
        "meta_path": tuple(sys.meta_path),
        "grad": torch.is_grad_enabled(),
    }


@pytest.fixture(autouse=True)
def _set_aside_build_forbidden():
    """The BUILD guard refuses to start while the renderer is imported (other test modules import it):
    set it aside for each test and restore the same module objects afterwards."""
    import agent_loop_lab.v03 as pkg

    names = ("agent_loop_lab.v03.renderer", "renderer", "reference_policy")
    saved = {n: sys.modules.pop(n) for n in names if n in sys.modules}
    attr = vars(pkg).pop("renderer", None)
    try:
        yield
    finally:
        for n in names:
            sys.modules.pop(n, None)
        sys.modules.update(saved)
        if attr is not None:
            pkg.renderer = attr


@pytest.fixture(autouse=True)
def _restored(_set_aside_build_forbidden):
    before = _snapshot()
    yield
    after = _snapshot()
    assert not v03_guard_active()
    for k, v in before.items():
        if isinstance(v, (bool, tuple)):
            assert after[k] == v, k
        else:
            assert after[k] is v, f"{k} not restored"


A = torch.ones(3, 3)
B = torch.ones(3)
TOKS, EORD, OFF = torch.tensor([1, 40, 2]), torch.tensor([0, 1, 1]), torch.tensor([0, 0, 1])


def _bg():
    return build_guard(model_classes=(M.DoomLoopTransformer,))


def _blocked(guard, fn) -> bool:
    """True if fn is refused inside the guard (raises, and the guard also fails on exit)."""
    raised = False
    try:
        with guard:
            try:
                fn()
            except TrainingViolation:
                raised = True
    except TrainingViolation:
        return raised
    assert not raised, "violation raised but the guard did not fail on exit"
    return False


def _imatmul():
    x = A.clone()
    x @= A
    return x


BUILD_BLOCKED = {
    "@": lambda: A @ A,
    "@=": lambda: _imatmul(),
    "torch.matmul": lambda: torch.matmul(A, A),
    "Tensor.matmul": lambda: A.matmul(A),
    "torch.mm": lambda: torch.mm(A, A),
    "torch.bmm": lambda: torch.bmm(A[None], A[None]),
    "torch.addmm": lambda: torch.addmm(A, A, A),
    "torch.addr": lambda: torch.addr(A, B, B),
    "torch.baddbmm": lambda: torch.baddbmm(A[None], A[None], A[None]),
    "torch.mv": lambda: torch.mv(A, B),
    "torch.dot": lambda: torch.dot(B, B),
    "torch.einsum": lambda: torch.einsum("ij,jk->ik", A, A),
    "torch.tensordot": lambda: torch.tensordot(A, A, dims=1),
    "torch.linalg.solve": lambda: torch.linalg.solve(A + torch.eye(3), B),
    "F.linear": lambda: torch.nn.functional.linear(A, A),
    "F.relu": lambda: torch.nn.functional.relu(A),
    "torch.softmax": lambda: torch.softmax(A, -1),
    "Tensor.softmax": lambda: A.softmax(-1),
    "torch.log_softmax": lambda: torch.log_softmax(A, -1),
    "VariableFunctions.matmul": lambda: torch._C._VariableFunctions.matmul(A, A),
    "TensorBase.mm": lambda: torch._C._TensorBase.mm(A, A),
    "torch.ops.aten.mm": lambda: torch.ops.aten.mm(A, A),
    "np.matmul": lambda: np.matmul(np.ones((2, 2)), np.ones((2, 2))),
    "model call": lambda: M.zero_model()(TOKS, EORD, OFF),
    "model.forward": lambda: M.zero_model().forward(TOKS, EORD, OFF),
    "bound forward": lambda: (lambda f: f(TOKS, EORD, OFF))(M.zero_model().forward),
    "submodule call": lambda: M.zero_model().head(torch.zeros(1, 128)),
    "torch.rand": lambda: torch.rand(3),
    "random.random": lambda: random.random(),
    "random.Random": lambda: random.Random(1),
    "np.random.rand": lambda: np.random.rand(2),
    "Tensor.normal_": lambda: torch.zeros(3).normal_(),
    "torch.Generator": lambda: torch.Generator(),
    "open": lambda: open("/nonexistent_v03_dir/x.txt", "w"),
    "backward": lambda: (torch.ones(2, requires_grad=True) * 2).sum().backward(),
    "optim": lambda: torch.optim.SGD([torch.zeros(1)], lr=0.1),
}


@pytest.mark.parametrize("name", sorted(BUILD_BLOCKED))
def test_build_guard_blocks(name):
    assert _blocked(_bg(), BUILD_BLOCKED[name]), f"{name} was not refused by the BUILD guard"


def _legit_build_ops():
    m = M.zero_model()
    u = torch.tensor([1.0, -1.0, 1.0, -1.0])
    t = torch.zeros(4, 4)
    t[0, 1] = 2.0
    t[1:3, :2] = torch.outer(u[:2], u[:2])
    t.index_put_((torch.tensor([3]), torch.tensor([3])), torch.tensor(5.0))
    k = torch.kron(torch.eye(2), torch.ones(2, 2))
    s = (t + k) * 2 - t / 3
    with torch.no_grad():
        m.head.weight[0, :4] = s[0]
        m.layers[0].attn.q.weight[:4, :4] = u.outer(u).clone()
        m.layers[1].attn.rel_bias[0, 64] = 3.0
        m.tok_emb.weight[7].fill_(0.5)
        m.off_emb.weight[:2, :2].copy_(torch.full((2, 2), 0.25))
    _ = m.state_dict()
    _ = float(s.sum()), torch.arange(4).tolist(), torch.zeros(2).abs().max()
    return m


def test_build_guard_allows_construction_ops():
    with _bg():
        m = _legit_build_ops()
    assert float(m.head.weight[0, 1]) != 0.0


def test_build_guard_allows_tokenizer_on_literals():
    from agent_loop_lab.v03 import tokenizer

    with _bg():
        ids = [tokenizer.word_id(w) for w in ("read_file", "zork")]
    assert ids[0] == tokenizer.RESERVED_IDS["read_file"]


def test_build_guard_swallowed_violation_fails_on_exit():
    with pytest.raises(TrainingViolation):
        with _bg():
            try:
                torch.matmul(A, A)
            except TrainingViolation:
                pass


def test_build_guard_forbids_renderer_import():
    import agent_loop_lab.v03 as pkg

    saved = sys.modules.pop("agent_loop_lab.v03.renderer", None)
    saved_attr = vars(pkg).pop("renderer", None)
    try:
        with pytest.raises(TrainingViolation):
            with _bg():
                import agent_loop_lab.v03.renderer  # noqa: F401
    finally:
        sys.modules.pop("agent_loop_lab.v03.renderer", None)
        if saved is not None:
            sys.modules["agent_loop_lab.v03.renderer"] = saved
        if saved_attr is not None:
            pkg.renderer = saved_attr


AUTHOR_ALLOWED = {
    "forward": lambda: M.zero_model()(TOKS, EORD, OFF),
    "@": lambda: A @ A,
    "softmax": lambda: torch.softmax(A, -1),
    "F.linear": lambda: torch.nn.functional.linear(A, A),
    "outer": lambda: torch.outer(B, B),
}
AUTHOR_BLOCKED = {
    "backward": BUILD_BLOCKED["backward"],
    "optim": BUILD_BLOCKED["optim"],
    "enable_grad": lambda: torch.enable_grad().__enter__(),
    "torch.rand": BUILD_BLOCKED["torch.rand"],
    "torch.randn": lambda: torch.randn(2),
    "torch.manual_seed": lambda: torch.manual_seed(0),
    "random.random": BUILD_BLOCKED["random.random"],
    "random.shuffle": lambda: random.shuffle([1, 2]),
    "random.Random (not the renderer)": BUILD_BLOCKED["random.Random"],
    "random.SystemRandom": lambda: random.SystemRandom(),
    "np.random.rand": BUILD_BLOCKED["np.random.rand"],
    "np.random.default_rng": lambda: np.random.default_rng(0),
    "Tensor.uniform_": lambda: torch.zeros(3).uniform_(),
    "dropout": lambda: torch.nn.functional.dropout(A, 0.5),
}


@pytest.mark.parametrize("name", sorted(AUTHOR_ALLOWED))
def test_author_run_allows(name):
    with author_run_guard():
        AUTHOR_ALLOWED[name]()


@pytest.mark.parametrize("name", sorted(AUTHOR_BLOCKED))
def test_author_run_blocks(name):
    assert _blocked(author_run_guard(), AUTHOR_BLOCKED[name]), f"{name} was not refused by AUTHOR_RUN"


def test_author_run_allows_random_only_from_listed_files(tmp_path):
    helper = tmp_path / "renderer.py"
    helper.write_text("import random\n\ndef rng(seed):\n    return random.Random(seed)\n")
    other = tmp_path / "other.py"
    other.write_text("import random\n\ndef rng(seed):\n    return random.Random(seed)\n")
    ns_ok: dict = {"__file__": str(helper)}
    exec(compile(helper.read_text(), str(helper), "exec"), ns_ok)
    ns_bad: dict = {"__file__": str(other)}
    exec(compile(other.read_text(), str(other), "exec"), ns_bad)
    with author_run_guard(rng_allowed_files=[helper]):
        r = ns_ok["rng"](7)
        r.random()
    assert _blocked(author_run_guard(rng_allowed_files=[helper]), lambda: ns_bad["rng"](7))


def test_author_run_io_policy(tmp_path):
    root = tmp_path / "author"
    root.mkdir()
    sibling = tmp_path / "secret.txt"
    sibling.write_text("x")
    (root / "mine.txt").write_text("y")
    import os

    with author_run_guard(io_root=root, io_deny=[tmp_path]):
        assert (root / "mine.txt").read_text() == "y"
        (root / "out.txt").write_text("z")
    assert _blocked(author_run_guard(io_root=root, io_deny=[tmp_path]), lambda: sibling.read_text())
    assert _blocked(author_run_guard(io_root=root, io_deny=[tmp_path]), lambda: os.listdir(tmp_path))


def test_guards_do_not_nest():
    with _bg():
        with pytest.raises(TrainingViolation):
            with author_run_guard():
                pass


# -- file readers that raise no open audit event (review findings: torch.from_file, ctypes, .py side tables) --


def _readers(path):
    import ctypes
    import mmap

    p = str(path)
    return {
        "torch.from_file": lambda: torch.from_file(p, size=4, dtype=torch.uint8),
        "UntypedStorage.from_file": lambda: torch.UntypedStorage.from_file(p, False, 4),
        "TypedStorage.from_file": lambda: torch.TypedStorage.from_file(p, False, 4),
        "StorageBase.from_file (reach-around)": lambda: torch._C.StorageBase.from_file(p, False, 4),
        "VariableFunctions.from_file": lambda: torch._C._VariableFunctions.from_file(p, size=4, dtype=torch.uint8),
        "torch.jit.load": lambda: torch.jit.load(p),
        "PyTorchFileReader(path)": lambda: torch._C.PyTorchFileReader(p),
        "ctypes.CDLL": lambda: ctypes.CDLL(None),
        "ctypes.string_at": lambda: ctypes.string_at(id(b"x"), 4),
        "mmap": lambda: mmap.mmap(-1, 16),
    }


READER_NAMES = sorted(_readers("x"))


@pytest.mark.parametrize("name", READER_NAMES)
def test_build_guard_refuses_unaudited_readers(tmp_path, name):
    f = tmp_path / "secret.bin"
    f.write_bytes(b"SECRETBYTES")
    assert _blocked(_bg(), _readers(f)[name]), f"{name} was not refused by the BUILD guard"


@pytest.mark.parametrize("name", READER_NAMES)
def test_author_run_refuses_unaudited_readers(tmp_path, name):
    root = tmp_path / "author"
    root.mkdir()
    f = tmp_path / "secret.bin"
    f.write_bytes(b"SECRETBYTES")
    guard = author_run_guard(io_root=root, io_deny=[tmp_path])
    assert _blocked(guard, _readers(f)[name]), f"{name} was not refused by AUTHOR_RUN"


@pytest.mark.parametrize("what", ["torch.load", "np.fromfile", "np.load", "np.memmap"])
def test_build_guard_refuses_loaders(tmp_path, what):
    f = tmp_path / "t.py"  # a code suffix: the open audit rule alone would not refuse it under a read root
    f.write_bytes(b"\x00" * 64)
    fn = {
        "torch.load": lambda: torch.load(str(f)),
        "np.fromfile": lambda: np.fromfile(str(f), dtype=np.uint8),
        "np.load": lambda: np.load(str(f)),
        "np.memmap": lambda: np.memmap(str(f), dtype=np.uint8, mode="r"),
    }[what]
    assert _blocked(build_guard(read_roots=[tmp_path]), fn)


def test_build_guard_reads_code_only_under_read_roots(tmp_path):
    side = tmp_path / "side"
    side.mkdir()
    table = side / "tbl.py"
    table.write_text("# 1.0 2.0 3.0\n")
    mine = tmp_path / "mine"
    mine.mkdir()
    (mine / "ok.py").write_text("x = 1\n")
    # a .py file outside the read roots (a side table, a harness source) is refused
    assert _blocked(build_guard(read_roots=[mine]), lambda: open(table).read())
    assert _blocked(build_guard(read_roots=[mine]), lambda: open(__file__).read())
    # relative paths resolve against the working directory, not around the rule
    import os

    cwd = os.getcwd()
    os.chdir(side)
    try:
        assert _blocked(build_guard(read_roots=[mine]), lambda: open("tbl.py").read())
    finally:
        os.chdir(cwd)
    with build_guard(read_roots=[mine]):
        assert open(mine / "ok.py").read() == "x = 1\n"
        import json  # noqa: F401 - stdlib imports still resolve (interpreter install)
        open(torch.__file__).close()  # site-packages code under sys.prefix


def test_violation_text_caps_long_paths(tmp_path):
    long = "/nonexistent_v03_dir/" + "A" * 5000 + ".txt"
    with pytest.raises(TrainingViolation) as ei:
        with _bg():
            try:
                open(long)
            except TrainingViolation:
                pass
    assert "A" * 200 not in str(ei.value)
