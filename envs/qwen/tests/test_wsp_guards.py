"""v0.4 author-side guards (docs/V04_WSP_KERNEL.md §5): KERNEL_CLIENT and BUILD_EDIT on torch 2.14.1.

Every banned path runs in its own fresh ``python -I -B`` child that installs the guard and attempts
the action. A refusal must (a) raise ``GuardViolation`` in the child, (b) print ``HN_GUARD: refused``
and (c) end the child with status 3 even though the script caught the exception. The paths that
would switch grad mode on from C must kill the child at once. Allowed paths must run with status 0
and no ``HN_GUARD`` line. The children run in parallel (they are independent processes).

The author directory lives under ``/tmp`` so the kernel's AF_UNIX path stays under macOS's 104-byte
limit; a socket server in this process plays the kernel. No model is loaded.
"""

from __future__ import annotations

import dis
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
GUARD_SRC = HERE.parent / "hypaqwen" / "wsp" / "author" / "hn_guard"
REPO_FILE = HERE.parents[2] / "agent_doom_loop_detector_plan_v0_4.md"  # a repo file outside the author dir
PHASE = "p1"
# The children's interpreter: this env by default; set it to an author venv's python to check that venv.
CHILD_PYTHON = os.environ.get("HN_GUARD_TEST_PYTHON", sys.executable)


def _load_guard_module() -> types.ModuleType:
    """Import guards_v04 in this process without installing anything (for the static checks)."""
    spec = importlib.util.spec_from_file_location("hn_guard_v04_static", GUARD_SRC / "guards_v04.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


G = _load_guard_module()


# -- the author directory and the fake kernel --------------------------------------------------------


class _Kernel:
    """An AF_UNIX server answering b"pong" to each connection (the stand-in for the model server)."""

    def __init__(self, path: str) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(path)
        self.sock.listen(64)
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                conn.recv(16)
                conn.sendall(b"pong")

    def close(self) -> None:
        self.sock.close()


@pytest.fixture(scope="module")
def env():
    import safetensors.torch

    root = Path(os.path.realpath(tempfile.mkdtemp(prefix="hnw", dir="/tmp")))
    ad, outside = root / "author", root / "outside"
    for d in (ad / "tools", ad / "output" / PHASE, ad / "output" / "p2", ad / "packet", ad / ".kernel", outside):
        d.mkdir(parents=True)
    shutil.copytree(GUARD_SRC, ad / "tools" / "hn_guard", ignore=shutil.ignore_patterns("__pycache__"))
    safetensors.torch.save_file({"w": torch.arange(6.0).reshape(2, 3)}, str(ad / "data.safetensors"))
    safetensors.torch.save_file({"u": torch.ones(4)}, str(ad / "output" / PHASE / "edit_constants.safetensors"))
    (ad / "output" / PHASE / "build_edit.py").write_text("def build(orig, consts):\n    return {}\n")
    (ad / "output" / PHASE / "a.txt").write_text("a")
    (ad / "packet" / "renderer.py").write_text("import random\n\ndef draw(seed):\n    return random.Random(seed).random()\n")
    (outside / "secret.txt").write_text("sealed")
    safetensors.torch.save_file({"w": torch.ones(2)}, str(outside / "x.safetensors"))
    torch.save({"w": torch.ones(2)}, str(outside / "x.pt"), _use_new_zipfile_serialization=False)
    np.save(outside / "x.npy", np.ones(2))
    (ad / "output" / PHASE / "link").symlink_to(outside / "secret.txt")
    kernel = _Kernel(str(ad / ".kernel" / "k.sock"))
    other = _Kernel(str(outside / "o.sock"))
    try:
        yield {"root": root, "ad": ad, "outside": outside}
    finally:
        kernel.close()
        other.close()
        shutil.rmtree(root, ignore_errors=True)


PREAMBLE = r'''
import json, os, sys
AD, OUTSIDE, MODE, PHASE = {ad!r}, {outside!r}, {mode!r}, {phase!r}
OUT = os.path.join(AD, "output", PHASE)
SOCK = os.path.join(AD, ".kernel", "k.sock")
OTHER_SOCK = os.path.join(OUTSIDE, "o.sock")
CONSTS = os.path.join(OUT, "edit_constants.safetensors")
REPO_FILE = {repo_file!r}
sys.path.insert(0, os.path.join(AD, "tools"))
import torch, numpy as np, safetensors, safetensors.torch, tokenizers
TOK = tokenizers.Tokenizer(tokenizers.models.WordLevel({{"a": 0, "[UNK]": 1}}, unk_token="[UNK]"))
from hn_guard import guards_v04 as g
if MODE == "KERNEL_CLIENT":
    g.install(MODE, author_dir=AD, phase=PHASE, kernel_endpoint=SOCK)
else:
    g.install(MODE, author_dir=AD, phase=PHASE, consts_path=CONSTS)

def attempt(fn):
    try:
        r = fn()
    except g.GuardViolation as e:
        out = {{"refused": True, "type": type(e).__name__, "msg": str(e)[:300]}}
    except BaseException as e:
        out = {{"refused": False, "error": type(e).__name__, "msg": str(e)[:300]}}
    else:
        out = {{"refused": False, "value": repr(r)[:2000]}}
    print("RESULT " + json.dumps(out), flush=True)
'''


def _script(env: dict, mode: str, body: str) -> str:
    pre = PREAMBLE.format(ad=str(env["ad"]), outside=str(env["outside"]), mode=mode, phase=PHASE, repo_file=str(REPO_FILE))
    return pre + "\n" + body + "\n"


def _run(env: dict, mode: str, body: str) -> dict:
    proc = subprocess.run(
        [CHILD_PYTHON, "-I", "-B", "-c", _script(env, mode, body)],
        cwd=str(env["ad"]), capture_output=True, text=True, timeout=300,
    )
    results = [json.loads(line[7:]) for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    return {"rc": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr, "results": results}


# -- cases ---------------------------------------------------------------------------------------------
# name -> (mode, body). Bodies see AD, OUT, OUTSIDE, SOCK, OTHER_SOCK, CONSTS, REPO_FILE, TOK, g, attempt.

KC, BE = "KERNEL_CLIENT", "BUILD_EDIT"

REFUSED: dict[str, tuple[str, str]] = {
    # autograd: backward, autograd.grad, the engine
    "kc_tensor_backward": (KC, "x = torch.ones(3, requires_grad=True)\nattempt(lambda: (x * 2).sum().backward())"),
    "kc_autograd_backward": (KC, "attempt(lambda: torch.autograd.backward([torch.ones(1)]))"),
    "kc_autograd_grad": (KC, "attempt(lambda: torch.autograd.grad([torch.ones(1)], [torch.ones(1)]))"),
    "kc_engine_proxy": (KC, "from torch.autograd.variable import Variable\nattempt(lambda: Variable._execution_engine.run_backward)"),
    "kc_engine_new": (KC, "attempt(lambda: torch._C._ImperativeEngine())"),
    "kc_engine_graph_entry": (KC, "import torch.autograd.graph as G\nattempt(lambda: G._engine_run_backward([torch.ones(1)]))"),
    "kc_autograd_functional_jacobian": (KC, "attempt(lambda: torch.autograd.functional.jacobian(lambda t: t * 2, torch.ones(2)))"),
    "kc_autograd_functional_vjp": (KC, "attempt(lambda: torch.autograd.functional.vjp(lambda t: t * 2, torch.ones(2), torch.ones(2)))"),
    "kc_requires_grad_": (KC, "attempt(lambda: torch.ones(2).requires_grad_())"),
    # grad mode
    "kc_enable_grad_ctx": (KC, "def f():\n    with torch.enable_grad():\n        return torch.is_grad_enabled()\nattempt(f)"),
    "kc_enable_grad_decorator": (KC, "@torch.enable_grad()\ndef f():\n    return torch.is_grad_enabled()\nattempt(f)"),
    "kc_set_grad_enabled_call": (KC, "attempt(lambda: torch.set_grad_enabled(True))"),
    "kc_set_grad_enabled_ctx": (KC, "def f():\n    with torch.set_grad_enabled(True):\n        return torch.is_grad_enabled()\nattempt(f)"),
    "kc_inference_mode_false": (KC, "def f():\n    with torch.inference_mode(False):\n        return torch.is_grad_enabled()\nattempt(f)"),
    "kc_C_set_grad_enabled": (KC, "attempt(lambda: torch._C._set_grad_enabled(True))"),
    "kc_ops_set_grad_enabled": (KC, "attempt(lambda: torch.ops.aten.set_grad_enabled(True))"),
    "kc_jit_op_set_grad_enabled": (KC, "attempt(lambda: torch._C._jit_get_operation('aten::set_grad_enabled')[0](True))"),
    "kc_forward_ad_ctx": (KC, "import torch.autograd.forward_ad as fwAD\ndef f():\n    with fwAD.dual_level():\n        return 1\nattempt(f)"),
    "kc_forward_ad_enter": (KC, "import torch.autograd.forward_ad as fwAD\nattempt(lambda: fwAD.enter_dual_level())"),
    "kc_forward_ad_C": (KC, "attempt(lambda: torch._C._enter_dual_level())"),
    "kc_vmap": (KC, "attempt(lambda: torch.vmap(lambda t: t * 2)(torch.ones(2, 3)))"),
    # torch.func and the functorch internals behind it
    "kc_import_torch_func": (KC, "attempt(lambda: __import__('torch.func'))"),
    "kc_from_torch_import_func": (KC, "def f():\n    from torch import func\n    return func\nattempt(f)"),
    "kc_functorch_grad": (KC, "import torch._functorch.apis as A\nattempt(lambda: A.grad(lambda t: (t * t).sum())(torch.ones(3)))"),
    "kc_functorch_vjp": (KC, "import torch._functorch.eager_transforms as E\nattempt(lambda: E.vjp(lambda t: t * 2, torch.ones(3)))"),
    "kc_functorch_jvp": (KC, "import torch._functorch.eager_transforms as E\nattempt(lambda: E.jvp(lambda t: t * 2, (torch.ones(3),), (torch.ones(3),)))"),
    "kc_functorch_jacrev": (KC, "import torch._functorch.eager_transforms as E\nattempt(lambda: E.jacrev(lambda t: t * 2)(torch.ones(3)))"),
    "kc_functorch_jacfwd": (KC, "import torch._functorch.eager_transforms as E\nattempt(lambda: E.jacfwd(lambda t: t * 2)(torch.ones(3)))"),
    "kc_vmap_of_grad": (KC, "import torch._functorch.apis as A\nattempt(lambda: torch.vmap(A.grad(lambda t: (t * t).sum()))(torch.ones(2, 3)))"),
    # optimisers
    "kc_import_torch_optim": (KC, "attempt(lambda: __import__('torch.optim'))"),
    "kc_importlib_torch_optim": (KC, "import importlib\nattempt(lambda: importlib.import_module('torch.optim'))"),
    "kc_optim_without_finder": (KC, "sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != '_ForbiddenImportFinder']\nimport importlib\nattempt(lambda: importlib.import_module('torch.optim'))"),
    "kc_optim_source_read": (KC, "p = os.path.join(os.path.dirname(torch.__file__), 'optim', 'sgd.py')\nattempt(lambda: open(p).read())"),
    # compilers
    "kc_torch_compile": (KC, "attempt(lambda: torch.compile(lambda t: t))"),
    "kc_import_dynamo": (KC, "attempt(lambda: __import__('torch._dynamo'))"),
    "kc_lazy_dynamo_attr": (KC, "attempt(lambda: torch._dynamo)"),
    "kc_import_export": (KC, "attempt(lambda: __import__('torch.export'))"),
    "kc_jit_script": (KC, "attempt(lambda: torch.jit.script(lambda t: t * 2))"),
    "kc_jit_trace": (KC, "attempt(lambda: torch.jit.trace(lambda t: t * 2, torch.ones(2)))"),
    "kc_jit_compilation_unit": (KC, "attempt(lambda: torch.jit.CompilationUnit('def f(x):\\n    return x\\n'))"),
    "kc_jit_existing_cu_define": (KC, "import torch.jit._state as S\nattempt(lambda: S._python_cu.define('def f(x):\\n    return x\\n', [], None))"),
    # lstsq / pinv
    "kc_linalg_lstsq": (KC, "attempt(lambda: torch.linalg.lstsq(torch.eye(3), torch.ones(3, 1)))"),
    "kc_linalg_pinv": (KC, "attempt(lambda: torch.linalg.pinv(torch.eye(3)))"),
    "kc_torch_pinverse": (KC, "attempt(lambda: torch.pinverse(torch.eye(3)))"),
    "kc_tensor_pinverse": (KC, "attempt(lambda: torch.eye(3).pinverse())"),
    "kc_C_linalg_pinv": (KC, "attempt(lambda: torch._C._linalg.linalg_pinv(torch.eye(3)))"),
    "kc_ops_linalg_pinv": (KC, "attempt(lambda: torch.ops.aten.linalg_pinv(torch.eye(3)))"),
    "kc_numpy_lstsq": (KC, "attempt(lambda: np.linalg.lstsq(np.eye(3), np.ones(3), rcond=None))"),
    "kc_numpy_pinv": (KC, "attempt(lambda: np.linalg.pinv(np.eye(3)))"),
    # RNG
    "kc_torch_randn": (KC, "attempt(lambda: torch.randn(2))"),
    "kc_torch_rand": (KC, "attempt(lambda: torch.rand(2))"),
    "kc_tensor_normal_": (KC, "attempt(lambda: torch.ones(2).normal_())"),
    "kc_torch_generator": (KC, "attempt(lambda: torch.Generator())"),
    "kc_C_variable_functions_randn": (KC, "attempt(lambda: torch._C._VariableFunctions.randn(2))"),
    "kc_ops_randn": (KC, "attempt(lambda: torch.ops.aten.randn([2]))"),
    "kc_dropout": (KC, "attempt(lambda: torch.nn.functional.dropout(torch.ones(4)))"),
    "kc_random_random": (KC, "import random\nattempt(lambda: random.random())"),
    "kc_random_Random": (KC, "import random\nattempt(lambda: random.Random(1).random())"),
    "kc_random_hidden_instance": (KC, "import random\nattempt(lambda: random._inst.random())"),
    "kc_numpy_default_rng": (KC, "attempt(lambda: np.random.default_rng(0))"),
    "kc_numpy_rand": (KC, "attempt(lambda: np.random.rand(2))"),
    "kc_numpy_hidden_randomstate": (KC, "attempt(lambda: np.random.mtrand._rand.random_sample())"),
    "kc_os_urandom": (KC, "attempt(lambda: os.urandom(4))"),
    # sockets
    "kc_socket_inet": (KC, "import socket\nattempt(lambda: socket.socket(socket.AF_INET, socket.SOCK_STREAM))"),
    "kc_socket_other_unix_path": (KC, "import socket\nattempt(lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).connect(OTHER_SOCK))"),
    "kc_socket_getaddrinfo": (KC, "import socket\nattempt(lambda: socket.getaddrinfo('localhost', 80))"),
    # processes, threads, signals
    "kc_subprocess_run": (KC, "import subprocess\nattempt(lambda: subprocess.run(['/usr/bin/true']))"),
    "kc_os_system": (KC, "attempt(lambda: os.system('true'))"),
    "kc_os_fork": (KC, "attempt(lambda: os.fork())"),
    "kc_os_posix_spawn": (KC, "attempt(lambda: os.posix_spawn('/usr/bin/true', ['true'], {}))"),
    "kc_fork_exec_direct": (KC, "import _posixsubprocess\nattempt(lambda: _posixsubprocess.fork_exec())"),
    "kc_thread": (KC, "import threading\nattempt(lambda: threading.Thread(target=print).start())"),
    "kc_multiprocessing": (KC, "import multiprocessing\nattempt(lambda: multiprocessing.get_context('fork').Process(target=print).start())"),
    "kc_os_kill": (KC, "attempt(lambda: os.kill(os.getppid(), 0))"),
    "kc_setrlimit": (KC, "import resource\nattempt(lambda: resource.setrlimit(resource.RLIMIT_CPU, resource.getrlimit(resource.RLIMIT_CPU)))"),
    "kc_ctypes": (KC, "import ctypes\nattempt(lambda: ctypes.CDLL(None))"),
    "kc_mmap": (KC, "import mmap\nattempt(lambda: mmap.mmap(-1, 16))"),
    # reads outside the allow-list
    "kc_read_outside": (KC, "attempt(lambda: open(os.path.join(OUTSIDE, 'secret.txt')).read())"),
    "kc_read_repo_file": (KC, "attempt(lambda: open(REPO_FILE).read())"),
    "kc_pathlib_read_outside": (KC, "import pathlib\nattempt(lambda: pathlib.Path(OUTSIDE, 'secret.txt').read_text())"),
    "kc_fileio_outside": (KC, "import io\nattempt(lambda: io.FileIO(os.path.join(OUTSIDE, 'secret.txt')).read())"),
    "kc_listdir_outside": (KC, "attempt(lambda: os.listdir(OUTSIDE))"),
    "kc_symlink_read": (KC, "attempt(lambda: open(os.path.join(OUT, 'link')).read())"),
    "kc_custom_pathlike": (KC, "class P:\n    def __fspath__(self):\n        return OUTSIDE\nattempt(lambda: os.listdir(P()))"),
    "kc_directory_fd": (KC, "attempt(lambda: os.open(AD, os.O_RDONLY))"),
    "kc_safetensors_outside": (KC, "attempt(lambda: safetensors.torch.load_file(os.path.join(OUTSIDE, 'x.safetensors')))"),
    "kc_safe_open_outside": (KC, "attempt(lambda: safetensors.safe_open(os.path.join(OUTSIDE, 'x.safetensors'), 'pt'))"),
    "kc_torch_load_outside": (KC, "attempt(lambda: torch.load(os.path.join(OUTSIDE, 'x.pt')))"),
    "kc_numpy_load_outside": (KC, "attempt(lambda: np.load(os.path.join(OUTSIDE, 'x.npy')))"),
    "kc_torch_from_file": (KC, "attempt(lambda: torch.from_file(os.path.join(OUTSIDE, 'secret.txt'), size=1))"),
    "kc_storage_from_file": (KC, "attempt(lambda: torch.UntypedStorage.from_file(os.path.join(OUTSIDE, 'secret.txt'), False, 1))"),
    "kc_file_reader_path": (KC, "attempt(lambda: torch._C.PyTorchFileReader(os.path.join(OUTSIDE, 'x.pt')))"),
    "kc_tokenizer_from_file": (KC, "attempt(lambda: tokenizers.Tokenizer.from_file(os.path.join(OUTSIDE, 'secret.txt')))"),
    "kc_tokenizer_save": (KC, "attempt(lambda: TOK.save(os.path.join(OUT, 'tok.json')))"),
    # writes outside output/<phase>/
    "kc_write_author_dir": (KC, "attempt(lambda: open(os.path.join(AD, 'x.txt'), 'w'))"),
    "kc_write_other_phase": (KC, "attempt(lambda: open(os.path.join(AD, 'output', 'p2', 'x.txt'), 'w'))"),
    "kc_write_outside": (KC, "attempt(lambda: open(os.path.join(OUTSIDE, 'x.txt'), 'a'))"),
    "kc_remove_outside_phase": (KC, "attempt(lambda: os.remove(os.path.join(AD, 'data.safetensors')))"),
    "kc_rename_out_of_phase": (KC, "attempt(lambda: os.rename(os.path.join(OUT, 'a.txt'), os.path.join(AD, 'a.txt')))"),
    "kc_mkdir_outside_phase": (KC, "attempt(lambda: os.mkdir(os.path.join(AD, 'newdir')))"),
    "kc_symlink_create": (KC, "attempt(lambda: os.symlink(OUTSIDE, os.path.join(OUT, 'l2')))"),
    "kc_safetensors_save_outside": (KC, "attempt(lambda: safetensors.torch.save_file({'w': torch.ones(1)}, os.path.join(AD, 'y.safetensors')))"),
    "kc_torch_save_outside": (KC, "attempt(lambda: torch.save(torch.ones(1), os.path.join(AD, 'y.pt')))"),
    "kc_file_writer_path": (KC, "attempt(lambda: torch._C.PyTorchFileWriter(os.path.join(OUT, 'z.pt')))"),
    # imports
    "kc_import_transformers": (KC, "attempt(lambda: __import__('transformers'))"),
    "kc_import_hypaqwen": (KC, "attempt(lambda: __import__('hypaqwen'))"),
    "kc_import_agent_loop_lab": (KC, "attempt(lambda: __import__('agent_loop_lab'))"),
    "kc_import_subinterpreters": (KC, "attempt(lambda: __import__('_xxsubinterpreters'))"),
    "kc_import_readline": (KC, "attempt(lambda: __import__('readline'))"),
    "kc_reimport_gc": (KC, "del sys.modules['gc']\nattempt(lambda: __import__('gc'))"),
    "kc_reimport_torch_C": (KC, "del sys.modules['torch._C']\nimport importlib\nattempt(lambda: importlib.import_module('torch._C'))"),
    # tampering with the guard itself
    "kc_monitoring_set_events": (KC, "attempt(lambda: sys.monitoring.set_events(3, 0))"),
    "kc_monitoring_free_tool": (KC, "attempt(lambda: sys.monitoring.free_tool_id(3))"),
    "kc_settrace": (KC, "attempt(lambda: sys.settrace(lambda *a: None))"),
    "kc_setprofile": (KC, "attempt(lambda: sys.setprofile(lambda *a: None))"),
    "kc_gc_get_objects": (KC, "import gc\nattempt(lambda: gc.get_objects())"),
    "kc_gc_get_referents": (KC, "import gc\nattempt(lambda: gc.get_referents(g))"),
    "kc_code_assignment": (KC, "def a():\n    return 1\ndef b():\n    return 2\ndef f():\n    a.__code__ = b.__code__\nattempt(f)"),
    "kc_subclasses": (KC, "attempt(lambda: object.__subclasses__())"),
    "kc_atexit_clear": (KC, "import atexit\nattempt(lambda: atexit._clear())"),
    "kc_install_twice": (KC, "attempt(lambda: g.install('BUILD_EDIT', author_dir=AD, phase=PHASE))"),
    # tampering with the evidence: stderr (fd 2 and the guard's private copy) and the exit check
    "kc_dup2_over_stderr": (KC, "attempt(lambda: os.dup2(os.open(os.path.join(OUT, 'sink.log'), os.O_WRONLY | os.O_CREAT), 2))"),
    "kc_dup2_elsewhere": (KC, "attempt(lambda: os.dup2(1, 9))"),
    "kc_closerange": (KC, "attempt(lambda: os.closerange(0, 64))"),
    "kc_close_stderr": (KC, "attempt(lambda: os.close(2))"),
    "kc_posix_close_stderr": (KC, "import posix\nattempt(lambda: posix.close(2))"),
    "kc_socket_close_stderr": (KC, "import socket\nattempt(lambda: socket.close(2))"),
    "kc_fileio_on_stderr": (KC, "import _io\nattempt(lambda: _io.FileIO(2, 'w', closefd=True))"),
    "kc_fdopen_stderr": (KC, "attempt(lambda: os.fdopen(2, 'w'))"),
    "kc_recreate_atexit": (KC, "import _imp, importlib.machinery as m\nattempt(lambda: _imp.create_builtin(m.ModuleSpec('atexit', None))._clear())"),
    "kc_recreate_posix": (KC, "import _imp, importlib.machinery as m\nattempt(lambda: _imp.create_builtin(m.ModuleSpec('posix', None)))"),
    # BUILD_EDIT: model / module / functional calls
    "be_module_call": (BE, "class M(torch.nn.Module):\n    def forward(self, x):\n        return x\nm = M()\nattempt(lambda: m(torch.ones(1)))"),
    "be_module_call_impl": (BE, "class M(torch.nn.Module):\n    def forward(self, x):\n        return x\nm = M()\nattempt(lambda: m._call_impl(torch.ones(1)))"),
    "be_functional_relu": (BE, "attempt(lambda: torch.nn.functional.relu(torch.ones(2)))"),
    "be_functional_linear": (BE, "attempt(lambda: torch.nn.functional.linear(torch.ones(2), torch.ones(2, 2)))"),
    "be_softmax": (BE, "attempt(lambda: torch.softmax(torch.ones(2), -1))"),
    "be_tensor_softmax": (BE, "attempt(lambda: torch.ones(2).softmax(-1))"),
    # BUILD_EDIT: RNG, autograd, optimisers, compilers
    "be_torch_randn": (BE, "attempt(lambda: torch.randn(2))"),
    "be_random_Random": (BE, "import random\nattempt(lambda: random.Random(1).random())"),
    "be_backward": (BE, "attempt(lambda: torch.ones(1).backward())"),
    "be_enable_grad": (BE, "def f():\n    with torch.enable_grad():\n        return 1\nattempt(f)"),
    "be_import_optim": (BE, "attempt(lambda: __import__('torch.optim'))"),
    "be_torch_compile": (BE, "attempt(lambda: torch.compile(lambda t: t))"),
    "be_lstsq": (BE, "attempt(lambda: torch.linalg.lstsq(torch.eye(3), torch.ones(3, 1)))"),
    # BUILD_EDIT: I/O other than the constants file
    "be_read_author_file": (BE, "attempt(lambda: open(os.path.join(AD, 'data.safetensors'), 'rb').read())"),
    "be_read_packet_code": (BE, "attempt(lambda: open(os.path.join(AD, 'packet', 'renderer.py')).read())"),
    "be_safetensors_author_file": (BE, "attempt(lambda: safetensors.torch.load_file(os.path.join(AD, 'data.safetensors')))"),
    "be_listdir_author_dir": (BE, "attempt(lambda: os.listdir(AD))"),
    "be_write_output": (BE, "attempt(lambda: open(os.path.join(OUT, 'x.txt'), 'w'))"),
    "be_mkdir": (BE, "attempt(lambda: os.mkdir(os.path.join(OUT, 'd')))"),
    "be_socket_unix": (BE, "import socket\nattempt(lambda: socket.socket(socket.AF_UNIX, socket.SOCK_STREAM))"),
    "be_subprocess": (BE, "import subprocess\nattempt(lambda: subprocess.run(['/usr/bin/true']))"),
    "be_thread": (BE, "import threading\nattempt(lambda: threading.Thread(target=print).start())"),
    "be_dup2_over_stderr": (BE, "attempt(lambda: os.dup2(1, 2))"),
    "be_close_stderr": (BE, "attempt(lambda: os.close(2))"),
}
if sys.platform == "darwin":  # kqueue.fromfd adopts a descriptor and closes it on close()
    REFUSED["kc_kqueue_fromfd_stderr"] = (KC, "import select\nattempt(lambda: select.kqueue.fromfd(2))")

# Grad mode switched on from C, around every Python-level patch: the child must die at once.
FATAL: dict[str, tuple[str, str]] = {
    "kc_trampoline_set_grad_enabled": (KC, "list(map(torch.ops.aten.set_grad_enabled.default._op, [True]))\nprint('ALIVE', flush=True)"),
    "be_trampoline_set_grad_enabled": (BE, "list(map(torch.ops.aten.set_grad_enabled.default._op, [True]))\nprint('ALIVE', flush=True)"),
}

ALLOWED: dict[str, tuple[str, str]] = {
    "kc_matmul_and_linalg": (KC, (
        "A = torch.arange(9.0).reshape(3, 3)\n"
        "attempt(lambda: [(A @ A).sum().item(), torch.matmul(A, A).shape, torch.mm(A, A).shape, torch.einsum('ij,jk->ik', A, A).shape,"
        " torch.linalg.svd(A).S.shape, torch.linalg.eigh(A @ A.T).eigenvalues.shape, torch.linalg.qr(A).Q.shape,"
        " round(torch.linalg.norm(A).item(), 4), torch.linalg.solve(A + 10 * torch.eye(3), torch.ones(3)).shape,"
        " torch.outer(A[0], A[1]).shape, np.linalg.svd(np.eye(3))[1].tolist(), repr(A)[:6], str(A.dtype)])"
    )),
    "kc_grad_contexts_harmless": (KC, (
        "def f():\n"
        "    with torch.no_grad():\n        a = torch.ones(2) * 2\n"
        "    with torch.inference_mode():\n        b = torch.ones(2) + 1\n"
        "    with torch.set_grad_enabled(False):\n        c = torch.ones(2)\n"
        "    return [torch.is_grad_enabled(), a.sum().item(), b.sum().item(), c.sum().item()]\n"
        "attempt(f)"
    )),
    "kc_requires_grad_ineffective": (KC, (
        "x = torch.ones(3, requires_grad=True)\ny = x * 2\n"
        "attempt(lambda: [y.requires_grad, y.grad_fn is None, torch.is_grad_enabled(), x.detach().requires_grad_(False).requires_grad])"
    )),
    "kc_safetensors_read_author_dir": (KC, (
        "def f():\n"
        "    d = safetensors.torch.load_file(os.path.join(AD, 'data.safetensors'))\n"
        "    with safetensors.safe_open(os.path.join(AD, 'data.safetensors'), 'pt') as h:\n"
        "        ks, t = h.keys(), h.get_tensor('w')\n"
        "    return [d['w'].sum().item(), ks, t.shape]\n"
        "attempt(f)"
    )),
    "kc_reads_and_listing_author_dir": (KC, (
        "import pathlib\n"
        "attempt(lambda: [sorted(os.listdir(AD)), pathlib.Path(OUT, 'a.txt').read_text(), sorted(p.name for p in pathlib.Path(AD, 'packet').iterdir())])"
    )),
    "kc_writes_under_output_phase": (KC, (
        "import pathlib\n"
        "def f():\n"
        "    with open(os.path.join(OUT, 'w.txt'), 'w') as fh:\n        fh.write('x')\n"
        "    pathlib.Path(OUT, 'w2.txt').write_text('y')\n"
        "    os.mkdir(os.path.join(OUT, 'sub'))\n"
        "    os.rename(os.path.join(OUT, 'w2.txt'), os.path.join(OUT, 'sub', 'w3.txt'))\n"
        "    os.remove(os.path.join(OUT, 'sub', 'w3.txt'))\n"
        "    safetensors.torch.save_file({'v': torch.ones(3)}, os.path.join(OUT, 'v.safetensors'))\n"
        "    torch.save({'v': torch.ones(3)}, os.path.join(OUT, 'v.pt'))\n"
        "    np.save(os.path.join(OUT, 'v.npy'), np.ones(3))\n"
        "    return [safetensors.torch.load_file(os.path.join(OUT, 'v.safetensors'))['v'].sum().item(),\n"
        "            torch.load(os.path.join(OUT, 'v.pt'))['v'].sum().item(), float(np.load(os.path.join(OUT, 'v.npy')).sum())]\n"
        "attempt(f)"
    )),
    "kc_kernel_socket": (KC, (
        "import socket\n"
        "def f():\n"
        "    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        "    s.connect(SOCK)\n    s.sendall(b'ping')\n    r = s.recv(16)\n    s.close()\n    return r\n"
        "attempt(f)"
    )),
    "kc_tokenizer_in_memory": (KC, "attempt(lambda: TOK.encode('a').ids)"),
    "kc_packet_renderer_rng": (KC, "sys.path.insert(0, os.path.join(AD, 'packet'))\nimport renderer\nattempt(lambda: renderer.draw(5))"),
    "kc_control_normal": (KC, "attempt(lambda: g.control_normal(7, (3, 5)).tolist())"),
    "kc_status": (KC, "attempt(lambda: [g.installed_mode(), g.violation_count()])"),
    "kc_os_close_and_dup": (KC, (
        "def f():\n"
        "    fd = os.open(os.path.join(OUT, 'c.txt'), os.O_WRONLY | os.O_CREAT)\n"
        "    fd2 = os.dup(fd)\n    os.close(fd)\n    os.close(np.int64(fd2))\n"
        "    try:\n        os.close(fd)\n    except OSError as e:\n        return [e.errno, type(e).__name__]\n"
        "attempt(f)"
    )),
    "be_matmul_and_linalg": (BE, (
        "A = torch.arange(9.0).reshape(3, 3)\n"
        "attempt(lambda: [(A @ A).sum().item(), torch.linalg.svd(A).S.shape, torch.linalg.eigh(A @ A.T).eigenvalues.shape,"
        " torch.linalg.qr(A).Q.shape, round(torch.linalg.norm(A).item(), 4), (A.double() @ A.double()).float().dtype, A.clone().shape])"
    )),
    "be_read_consts": (BE, (
        "def f():\n"
        "    d = safetensors.torch.load_file(CONSTS)\n"
        "    raw = open(CONSTS, 'rb').read(8)\n"
        "    return [d['u'].sum().item(), len(raw)]\n"
        "attempt(f)"
    )),
    "be_read_build_file": (BE, "attempt(lambda: open(os.path.join(OUT, 'build_edit.py')).read().split()[0])"),
    "be_lazy_stdlib_import": (BE, "attempt(lambda: __import__('fractions').Fraction(1, 3))"),
    "be_control_normal": (BE, "attempt(lambda: g.control_normal(7, (3, 5)).tolist())"),
}


@pytest.fixture(scope="module")
def runs(env):
    cases = {**REFUSED, **FATAL, **ALLOWED}
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
        futs = {name: pool.submit(_run, env, mode, body) for name, (mode, body) in cases.items()}
        return {name: f.result() for name, f in futs.items()}


def _show(r: dict) -> str:
    return f"rc={r['rc']}\nstdout:\n{r['stdout'][-1500:]}\nstderr:\n{r['stderr'][-1500:]}"


@pytest.mark.parametrize("name", sorted(REFUSED))
def test_banned_path_is_refused(runs, name):
    r = runs[name]
    assert len(r["results"]) == 1, _show(r)
    res = r["results"][0]
    assert res["refused"] is True, f"{res}\n{_show(r)}"
    assert "HN_GUARD: refused" in r["stderr"], _show(r)
    assert r["rc"] == 3, "a caught violation must still fail the process\n" + _show(r)


@pytest.mark.parametrize("name", sorted(FATAL))
def test_grad_mode_from_c_kills_the_process(runs, name):
    r = runs[name]
    assert r["rc"] == 3, _show(r)
    assert "ALIVE" not in r["stdout"], _show(r)
    assert "grad mode was found enabled" in r["stderr"], _show(r)


@pytest.mark.parametrize("name", sorted(ALLOWED))
def test_allowed_path_works(runs, name):
    r = runs[name]
    assert r["rc"] == 0, _show(r)
    assert "HN_GUARD" not in r["stderr"], _show(r)
    assert len(r["results"]) == 1 and r["results"][0]["refused"] is False and "value" in r["results"][0], _show(r)


def _value(runs, name):
    return runs[name]["results"][0]["value"]


def test_allowed_values(runs, env):
    assert _value(runs, "kc_grad_contexts_harmless") == "[False, 4.0, 4.0, 2.0]"
    assert _value(runs, "kc_requires_grad_ineffective") == "[False, True, False, False]"
    assert _value(runs, "kc_kernel_socket") == "b'pong'"
    assert _value(runs, "kc_tokenizer_in_memory") == "[0]"
    assert _value(runs, "kc_status") == "['KERNEL_CLIENT', 0]"
    assert _value(runs, "kc_os_close_and_dup") == "[9, 'OSError']"
    assert _value(runs, "kc_writes_under_output_phase") == "[3.0, 3.0, 3.0]"
    assert _value(runs, "be_read_consts") == "[4.0, 8]"
    assert _value(runs, "kc_safetensors_read_author_dir") == "[15.0, ['w'], torch.Size([2, 3])]"


def test_control_normal_is_the_same_in_both_modes_and_unguarded(runs):
    want = repr(G.control_normal(7, (3, 5)).tolist())
    assert _value(runs, "kc_control_normal") == want
    assert _value(runs, "be_control_normal") == want


def test_refused_actions_left_no_trace(runs, env):
    ad, outside = env["ad"], env["outside"]
    for p in (ad / "x.txt", ad / "output" / "p2" / "x.txt", outside / "x.txt", ad / "newdir", ad / "y.safetensors", ad / "y.pt",
              ad / "output" / PHASE / "l2", ad / "output" / PHASE / "tok.json", ad / "output" / PHASE / "x.txt"):
        assert not os.path.lexists(p), p
    assert (ad / "data.safetensors").exists() and (ad / "output" / PHASE / "a.txt").exists()


def test_swallowed_violation_fails_finish(env):
    r = _run(env, KC, "try:\n    torch.randn(1)\nexcept Exception:\n    pass\ng.finish(0)")
    assert r["rc"] == 3, _show(r)
    r = _run(env, KC, "g.finish(5)")
    assert r["rc"] == 5 and "HN_GUARD" not in r["stderr"], _show(r)


# A caught violation followed by an attempt to end the run without evidence. Each must still exit with
# status 3 and leave the refusal's HN_GUARD line on the stderr the launcher reads.
SILENCERS: dict[str, str] = {
    # the review repro: fd 2 redirected into an allowed file, then _exit skipping atexit
    "dup2_then_os_exit": (
        "fd = os.open(os.path.join(OUT, 'sink.log'), os.O_WRONLY | os.O_CREAT | os.O_TRUNC)\n"
        "try:\n    os.dup2(fd, 2)\nexcept g.GuardViolation:\n    pass\n"
        "{violation}\nos._exit(0)"
    ),
    "os_exit": "{violation}\nos._exit(0)",
    "posix_exit": "import posix\n{violation}\nposix._exit(0)",
    "os_exit_trampoline": "{violation}\nlist(map(os._exit, [0]))",
    "os_abort": "{violation}\nos.abort()",
    "atexit_registered_exit": "import atexit\n{violation}\natexit.register(os._exit, 0)",
    "exit_through_the_wrappers_closure": "{violation}\naudit = os._exit.__closure__[0].cell_contents\naudit('hn_guard.exit', 0)",
    "fresh_posix_exit": (
        "import _imp, importlib.machinery as m\n"
        "p = list(map(_imp.create_builtin, [m.ModuleSpec('posix', None)]))[0]\n{violation}\np._exit(0)"
    ),
    "close_every_fd": (
        "for fd in range(0, 256):\n    try:\n        os.close(fd)\n    except OSError:\n        pass\n"
        "{violation}\nos._exit(0)"
    ),
    "close_trampoline": "try:\n    list(map(os.close, [2]))\nexcept OSError:\n    pass\n{violation}\nos._exit(0)",
}
VIOLATION = "try:\n    torch.randn(3)\nexcept g.GuardViolation:\n    pass"


@pytest.fixture(scope="module")
def silencer_runs(env):
    bodies = {n: b.replace("{violation}", VIOLATION) for n, b in SILENCERS.items()}
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
        futs = {n: pool.submit(_run, env, KC, b) for n, b in bodies.items()}
        return {n: f.result() for n, f in futs.items()}


@pytest.mark.parametrize("name", sorted(SILENCERS))
def test_violation_cannot_be_silenced(silencer_runs, env, name):
    r = silencer_runs[name]
    assert r["rc"] == 3, _show(r)
    assert "HN_GUARD: refused: torch.randn" in r["stderr"], _show(r)
    assert "violation(s) during this run; exit status 3" in r["stderr"], _show(r)
    assert not G.clean_run(r["rc"], r["stderr"])
    sink = env["ad"] / "output" / PHASE / "sink.log"
    assert not sink.exists() or "HN_GUARD" not in sink.read_text(), _show(r)


def test_exit_without_violation_keeps_its_status(env):
    for body, rc in (("os._exit(0)", 0), ("os._exit(7)", 7), ("list(map(os._exit, [4]))", 4), ("g.finish(0)", 0)):
        r = _run(env, KC, "print('x', flush=True)\n" + body)
        assert r["rc"] == rc and r["stdout"] == "x\n" and "HN_GUARD" not in r["stderr"], (body, _show(r))
        assert G.clean_run(r["rc"], r["stderr"]) is (rc == 0)


def test_clean_run_needs_status_0_and_no_marker():
    assert G.clean_run(0, "")
    assert G.clean_run(0, b"some warning\n")
    assert not G.clean_run(3, "")
    assert not G.clean_run(1, "")
    assert not G.clean_run(0, "x\nHN_GUARD: refused: torch.randn\n")
    assert not G.clean_run(0, b"HN_GUARD: 1 violation(s) during this run; exit status 3\n")


def test_overwritten_module_globals_do_not_reopen_anything(env):
    body = (
        "secret = os.path.join(OUTSIDE, 'secret.txt')\n"
        "for k in [k for k in vars(g) if not k.startswith('__')]:\n    setattr(g, k, None)\n"
        "import builtins\nreal_isinstance = builtins.isinstance\nbuiltins.isinstance = lambda *a: True\n"
        "out = []\n"
        "for fn in (lambda: open(secret).read(), lambda: torch._C._set_grad_enabled(True), lambda: torch.ones(1).backward()):\n"
        "    try:\n        out.append(repr(fn()))\n    except BaseException as e:\n        out.append(type(e).__name__)\n"
        "builtins.isinstance = real_isinstance\n"
        "out.append(torch.is_grad_enabled())\n"
        "print('TAMPER ' + repr(out), flush=True)"
    )
    r = _run(env, KC, body)
    lines = [ln for ln in r["stdout"].splitlines() if ln.startswith("TAMPER ")]
    assert len(lines) == 1, _show(r)
    vals = lines[0]
    assert "sealed" not in vals and vals.endswith("False]"), _show(r)
    assert "IOViolation" in vals, _show(r)  # the audit hook does not depend on module globals


def test_install_validates_arguments(tmp_path):
    with pytest.raises(ValueError):
        G._validate("NOPE", tmp_path, "p1", None, None)
    with pytest.raises(ValueError):
        G._validate(G.KERNEL_CLIENT, tmp_path, "../x", None, None)
    with pytest.raises(ValueError):
        G._validate(G.KERNEL_CLIENT, tmp_path, "p1", None, tmp_path / "c")
    with pytest.raises(ValueError):
        G._validate(G.BUILD_EDIT, tmp_path, "p1", tmp_path / "k.sock", None)
    with pytest.raises(ValueError):
        G._validate(G.KERNEL_CLIENT, "/", "p1", None, None)


# -- static properties (this process, nothing installed) ---------------------------------------------


def _nested_codes(fn: types.FunctionType) -> dict[str, types.CodeType]:
    return {c.co_name: c for c in fn.__code__.co_consts if isinstance(c, types.CodeType)}


ENFORCEMENT_CODES = {
    **{n: c for n, c in _nested_codes(G._make_audit_hook).items() if n in ("hook", "at_exit")},
    **{n: c for n, c in _nested_codes(G._make_actions_hook).items() if n == "actions"},
    **{n: c for n, c in _nested_codes(G._make_fd_wrappers).items() if n in ("_exit", "abort", "close")},
    **{n: c for n, c in _nested_codes(G._make_monitor).items() if n in ("on_call", "on_c_return", "on_py_start")},
    **{f.__name__: f.__code__ for f in (G._hn_realpath, G._hn_under, G._hn_path, G._hn_refuse, G._hn_mro_ids)},
}


def test_enforcement_code_has_no_global_lookups():
    """The hook, the monitor callbacks and their helpers read no module globals or builtins by name and
    build no functions (which would share their cells), so tampering with names cannot change them."""
    assert set(ENFORCEMENT_CODES) == {"hook", "at_exit", "actions", "_exit", "abort", "close", "on_call", "on_c_return", "on_py_start",
                                      "_hn_realpath", "_hn_under", "_hn_path", "_hn_refuse", "_hn_mro_ids"}
    for name, code in ENFORCEMENT_CODES.items():
        ops = {ins.opname for ins in dis.get_instructions(code)}
        assert not ops & {"LOAD_GLOBAL", "LOAD_NAME", "STORE_GLOBAL", "DELETE_GLOBAL", "MAKE_FUNCTION"}, (name, ops)
        assert not any(isinstance(c, types.CodeType) for c in code.co_consts), name


def test_pure_helpers_have_no_closure():
    for f in (G._hn_realpath, G._hn_under, G._hn_path, G._hn_refuse, G._hn_mro_ids):
        assert f.__closure__ is None and f.__defaults__ is None, f.__name__


def test_realpath_helper_matches_posixpath(tmp_path):
    import posix

    (tmp_path / "d" / "e").mkdir(parents=True)
    (tmp_path / "l1").symlink_to(tmp_path / "d")
    (tmp_path / "d" / "rel").symlink_to("e")
    (tmp_path / "loop").symlink_to(tmp_path / "loop")
    cases = [str(tmp_path / "l1" / "rel" / "f"), str(tmp_path / "l1" / ".." / "x"), str(tmp_path / "d" / "e" / ".." / "e"),
             "relative/../x", str(tmp_path) + "//d/./e/", "/"]
    for p in cases:
        assert G._hn_realpath(p, os.getcwd(), posix.lstat, posix.readlink) == os.path.realpath(p), p
    assert G._hn_realpath(str(tmp_path / "loop"), "/", posix.lstat, posix.readlink) is None


def test_path_coercion(tmp_path):
    import pathlib

    t = G._type_table()
    assert G._hn_path("a", t) == "a" and G._hn_path(b"a", t) == "a" and G._hn_path(3, t) == 3
    assert G._hn_path(pathlib.Path("/x/y"), t) == "/x/y"

    class S(str):
        def __fspath__(self):
            return "/elsewhere"

    assert G._hn_path(S("/here"), t) == "/here"

    class P:
        def __fspath__(self):
            return "/x"

    assert G._hn_path(P(), t) is None

    class Q(pathlib.PosixPath):
        def __fspath__(self):
            return "/elsewhere"

    assert G._hn_path(Q("/here"), t) is None
    entry = next(os.scandir(str(tmp_path.parent)))
    assert G._hn_path(entry, t) == entry.path


def test_control_normal_statistics_and_chunks():
    z = G.control_normal(1, (1000, 1000))
    assert z.dtype == torch.float32 and z.shape == (1000, 1000)
    assert abs(z.mean().item()) < 0.01 and abs(z.std().item() - 1.0) < 0.01
    assert torch.equal(z, G.control_normal(1, (1000, 1000)))
    assert not torch.equal(z, G.control_normal(2, (1000, 1000)))
    big = G.control_normal(3, (2 * G._CTRL_CHUNK + 3,))  # crosses chunk boundaries; odd length
    assert torch.equal(big[:5], G.control_normal(3, (5,))) and torch.isfinite(big).all()
    with pytest.raises(TypeError):
        G.control_normal(1.5, (2,))


def test_lists_cover_the_section_5_imports():
    for name in ("transformers", "hypaqwen", "peft", "accelerate", "agent_loop_lab", "sklearn", "torch._dynamo",
                 "torch.compile", "torch.export", "torch.func", "torch.optim"):
        assert name in G.FORBIDDEN_IMPORTS
    assert torch.__version__.startswith(G.VERIFIED_TORCH)


def test_guard_package_is_self_contained():
    """It ships alone into the author dir: only stdlib, torch, numpy and the optional safetensors / tokenizers."""
    import ast

    allowed = set(sys.stdlib_module_names) | {"torch", "numpy", "safetensors", "tokenizers"}
    for path in GUARD_SRC.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                roots = [(node.module or "").split(".")[0]]
            else:
                continue
            for r in roots:
                assert r in allowed, (path.name, r)


# -- the author venv script ------------------------------------------------------------------------------

VENV_SCRIPT = HERE.parent / "scripts" / "make_author_venv.sh"


def test_author_venv_pins_match_the_lock_and_the_doc():
    import re

    text = VENV_SCRIPT.read_text()
    pins = dict(re.findall(r"^(TORCH|NUMPY|SAFETENSORS|TOKENIZERS)_V=(\S+)$", text, re.M))
    lock = (HERE.parent / "uv.lock").read_text()
    doc = (GUARD_SRC.parent / "AUTHOR_VENV.md").read_text()
    for var, pkg in (("TORCH", "torch"), ("NUMPY", "numpy"), ("SAFETENSORS", "safetensors"), ("TOKENIZERS", "tokenizers")):
        assert f'name = "{pkg}"\nversion = "{pins[var]}"' in lock, pkg
        assert f"| `{pkg}` | {pins[var]} |" in doc, pkg
    assert pins["TORCH"] == G.VERIFIED_TORCH
    assert "transformers" not in re.findall(r'"(\w+)==', text)


@pytest.mark.skipif(os.environ.get("HN_TEST_AUTHOR_VENV") != "1", reason="builds a venv with torch (set HN_TEST_AUTHOR_VENV=1)")
def test_make_author_venv_builds_a_clean_venv():
    target = HERE.parent / ".test_author_venv"
    shutil.rmtree(target, ignore_errors=True)
    try:
        r = subprocess.run(["sh", str(VENV_SCRIPT), str(target)], capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stderr
        assert "author venv ready" in r.stdout
        again = subprocess.run(["sh", str(VENV_SCRIPT), str(target)], capture_output=True, text=True)
        assert again.returncode == 2 and "already exists" in again.stderr
    finally:
        shutil.rmtree(target, ignore_errors=True)
