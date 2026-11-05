"""WSP wire format, forward-request schema, author helpers and ``./fwd`` validation, without the model."""

from __future__ import annotations

import ast
import json
import os
import socket
import struct
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

from hypaqwen.wsp.author import client as C
from hypaqwen.wsp.author import fwd
from hypaqwen.wsp.author import protocol as P
from hypaqwen.wsp import install as I

AUTHOR_DIR = Path(C.__file__).parent
ALLOWED_AUTHOR_IMPORTS = {"__future__", "argparse", "atexit", "json", "os", "re", "sys", "contextlib", "socket",
                          "struct", "pathlib", "typing", "numpy", "torch", "tokenizers", "math", "stat", "importlib",
                          "runpy", "traceback", "safetensors"}  # the last three: run_py.py, the ./py launcher


def test_author_side_imports_only_stdlib_numpy_torch():
    for path in AUTHOR_DIR.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [(node.module or "").split(".")[0]] if node.level == 0 else []
            else:
                continue
            for m in mods:
                assert m in ALLOWED_AUTHOR_IMPORTS, (path.name, m)


def test_author_side_never_extends_sys_path():
    """Nothing in tools/hn_kernel may make its own (author-writable) directory importable."""
    for path in AUTHOR_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute) and node.attr in ("insert", "append", "extend") \
                    and isinstance(node.value, ast.Attribute) and node.value.attr == "path":
                raise AssertionError(f"{path.name}:{node.lineno} changes sys.path")


PLANTED = ("socket", "select", "json", "struct", "contextlib", "math", "stat", "re", "argparse", "typing",
           "numpy", "importlib", "__future__", "protocol", "client", "torch")


def _plant(dirs, marker):
    for d in dirs:
        for name in PLANTED:
            p = d / f"{name}.py"
            if not p.exists():
                p.write_text(f"open({str(marker)!r}, 'a').write({name!r} + '\\n')\n")


@pytest.fixture
def planted(tmp_path):
    """An author dir whose tools/hn_kernel/ (a writable copy) and root are full of planted stdlib/numpy modules."""
    author = tmp_path / "author"
    kdir = author / "tools" / "hn_kernel"
    kdir.mkdir(parents=True)
    for name in I.KERNEL_FILES:
        (kdir / name).write_bytes((I.SOURCE_DIR / name).read_bytes())
    marker = tmp_path / "PLANTED_CODE_RAN"
    _plant([kdir, author], marker)
    return author, kdir, marker


def _env(author):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "HN_"))}
    env["HN_AUTHOR_DIR"] = str(author)
    return env


@pytest.mark.parametrize("flags", [["-I", "-B"], ["-B"]])
def test_fwd_never_runs_planted_modules(planted, flags):
    author, kdir, marker = planted
    req = author / "r.json"
    req.write_text(json.dumps({"model": "bf16", "sessions": []}))
    for argv in (["--help"], ["r.json"]):
        p = subprocess.run([sys.executable, *flags, str(kdir / "fwd.py"), *argv], cwd=author, env=_env(author),
                           capture_output=True, text=True, timeout=60)
        assert p.returncode in (0, 2), p.stderr
    assert not marker.exists(), marker.read_text()


def test_pre_guard_client_import_never_runs_planted_modules(planted):
    """What the ./py wrapper does before the guard: load client.py by path, build the tokenizer, import wsp."""
    author, kdir, marker = planted
    code = ("import importlib.util, sys\n"
            f"spec = importlib.util.spec_from_file_location('hn_kernel_client', {str(kdir / 'client.py')!r})\n"
            "m = importlib.util.module_from_spec(spec); sys.modules['hn_kernel_client'] = m\n"
            "spec.loader.exec_module(m); m.init_tok()\n"
            f"spec = importlib.util.spec_from_file_location('wsp', {str(kdir / 'wsp.py')!r})\n"
            "w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)\n"
            "assert w.save_edit is m.save_edit and w._P is m._P\n"
            "print('ok')\n")
    p = subprocess.run([sys.executable, "-I", "-B", "-c", code], cwd=author, env=_env(author),
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and p.stdout.strip() == "ok", p.stderr
    assert not marker.exists(), marker.read_text()


def test_install_is_read_only_and_verified(tmp_path):
    dest = tmp_path / "author" / "tools" / "hn_kernel"
    try:
        assert I.install(dest) == I.source_manifest()
        I.verify(dest)
        assert sorted(os.listdir(dest)) == sorted(I.KERNEL_FILES)
        with pytest.raises(PermissionError):
            (dest / "socket.py").write_text("x")
        with pytest.raises(PermissionError):
            (dest / "fwd.py").write_text("x")
        os.chmod(dest, 0o755)  # what a session could do if the permissions were misconfigured
        with pytest.raises(I.KernelTampered, match="writable"):
            I.verify(dest)
        (dest / "socket.py").write_text("x")
        os.chmod(dest, 0o555)
        with pytest.raises(I.KernelTampered, match="unexpected entries"):
            I.verify(dest)
        os.chmod(dest, 0o755)
        (dest / "socket.py").unlink()
        os.chmod(dest / "protocol.py", 0o644)
        (dest / "protocol.py").write_text((dest / "protocol.py").read_text() + "\n# tampered\n")
        os.chmod(dest / "protocol.py", 0o444)
        os.chmod(dest, 0o555)
        with pytest.raises(I.KernelTampered, match="does not match"):
            I.verify(dest)
        os.chmod(dest, 0o755)
        (dest / "protocol.py").unlink()
        (dest / "protocol.py").symlink_to(I.SOURCE_DIR / "protocol.py")
        os.chmod(dest, 0o555)
        with pytest.raises(I.KernelTampered, match="regular file"):
            I.verify(dest)
    finally:
        I.uninstall(dest)
    assert not dest.exists()


def test_frame_round_trip():
    a, b = socket.socketpair()
    with a, b:
        t = {"x": np.arange(12, dtype=np.float32).reshape(3, 4), "empty": np.zeros((0, 5), np.float32)}
        P.send_msg(a, {"op": "save_edit", "args": {"name": "n"}}, t)
        header, got = P.recv_msg(b)
        assert header == {"op": "save_edit", "args": {"name": "n"}}
        assert np.array_equal(got["x"], t["x"]) and got["x"].dtype == np.float32 and got["empty"].shape == (0, 5)
        got["x"][0, 0] = 7.0  # writable
        P.send_msg(a, {"ok": True, "result": {"v": float("inf")}})
        assert P.recv_msg(b)[0]["result"]["v"] == float("inf")


def test_frame_refuses_non_fp32_and_garbage():
    a, b = socket.socketpair()
    with a, b:
        with pytest.raises(TypeError):
            P.send_msg(a, {"op": "x"}, {"t": np.zeros(3, np.float64)})
        a.sendall(b"XXXX" + b"\0" * 8)
        with pytest.raises(P.ProtocolError):
            P.recv_msg(b)
    a, b = socket.socketpair()
    with a, b:
        a.close()
        with pytest.raises(EOFError):
            P.recv_msg(b)


def _frame(header) -> bytes:
    head = json.dumps(header).encode() if not isinstance(header, bytes) else header
    return P.MAGIC + struct.pack("<Q", len(head)) + head


@pytest.mark.parametrize("header", [
    {"op": "x", "tensors": [{"key": "t", "shape": [2 ** 62, 4]}]},  # np.prod would overflow int64 to 0
    {"op": "x", "tensors": [{"key": "t", "shape": [2 ** 32] * 3}]},
    {"op": "x", "tensors": [{"key": "t", "shape": [10 ** 30]}]},
    {"op": "x", "tensors": [{"key": "t", "shape": [-1]}]},
    {"op": "x", "tensors": [{"key": "t", "shape": [True]}]},
    {"op": "x", "tensors": [{"key": "t", "shape": [1] * 9}]},
    {"op": "x", "tensors": [1]},
    {"op": "x", "tensors": [None]},
    {"op": "x", "tensors": [{"key": "t", "shape": [1]}, {"key": "t", "shape": [1]}]},
    {"op": "x", "tensors": "nope"},
    [1, 2],
    b"[" * 200_000 + b"]" * 200_000,  # RecursionError in json
    b"\xff\xfe",
    b"{bad json",
])
def test_recv_refuses_hostile_headers_as_protocol_errors(header):
    a, b = socket.socketpair()
    with a, b:
        t = threading.Thread(target=a.sendall, args=(_frame(header),), daemon=True)  # larger than the buffer
        t.start()
        with pytest.raises(P.ProtocolError):
            P.recv_msg(b)
        t.join(10)


def test_recv_bounds_total_tensor_bytes():
    a, b = socket.socketpair()
    with a, b:
        a.sendall(_frame({"tensors": [{"key": "a", "shape": [1000]}, {"key": "b", "shape": [1000]}]}))
        with pytest.raises(P.ProtocolError, match="tensor bytes"):
            P.recv_msg(b, max_total_bytes=4000)


def _req(**kw):
    r = {"model": "both", "sessions": [{"id": "a", "blocks": ["x", "y"], "windows": [0, 1]}],
         "return": {"hidden": [24, 3], "mlp_act": [0], "topk": 5}}
    r.update(kw)
    return r


def test_forward_schema_accepts_and_normalises():
    r = P.validate_forward_request(_req())
    assert r["return"] == {"hidden": [3, 24], "mlp_act": [0], "topk": 5}
    r = P.validate_forward_request({"model": "edited", "sessions": [{"id": "a", "blocks": ["x"], "windows": [0]}]})
    assert r["return"] == {"hidden": [], "mlp_act": [], "topk": 0}


@pytest.mark.parametrize("bad", [
    "not a dict",
    _req(model="bf16"),
    _req(sessions=[]),
    _req(extra=1),
    _req(sessions=[{"id": "a", "blocks": ["x"], "windows": [1]}]),  # window past the last block
    _req(sessions=[{"id": "a", "blocks": ["x", "y"], "windows": [0]}]),  # a block after the last window never runs
    _req(sessions=[{"id": "a", "blocks": ["x", "y"], "windows": [1, 0]}]),  # not increasing
    _req(sessions=[{"id": "a", "blocks": ["x"], "windows": [True]}]),
    _req(sessions=[{"id": "a", "blocks": ["x"], "windows": [0]}, {"id": "a", "blocks": ["x"], "windows": [0]}]),
    _req(sessions=[{"id": "a", "blocks": [1], "windows": [0]}]),
    _req(sessions=[{"id": "a", "blocks": ["x"], "windows": [0], "labels": [0]}]),
    _req(**{"return": {"hidden": [25]}}),
    _req(**{"return": {"mlp_act": [24]}}),
    _req(**{"return": {"topk": 21}}),
    _req(**{"return": {"hidden": [1, 1]}}),
    _req(**{"return": {"logits": True}}),
])
def test_forward_schema_refuses(bad):
    with pytest.raises(P.RequestError):
        P.validate_forward_request(bad)


def test_disclosed_schema():
    assert P.validate_disclosed_request({"disclosed_windows": ["V001"], "model": "edited"})["model"] == "edited"
    for bad in ({"disclosed_windows": [], "model": "edited"}, {"disclosed_windows": ["V1", "V1"], "model": "both"},
                {"disclosed_windows": ["V1"], "model": "edited", "return": {"hidden": [3]}}):
        with pytest.raises(P.RequestError):
            P.validate_disclosed_request(bad)


@pytest.fixture
def author(tmp_path, monkeypatch):
    a = tmp_path / "author"
    a.mkdir()
    monkeypatch.setenv(fwd.AUTHOR_ENV, str(a))
    monkeypatch.setenv(fwd.SOCKET_ENV, str(tmp_path / "none.sock"))
    monkeypatch.chdir(a)
    return a


def test_fwd_refuses_invalid_request_without_connecting(author, capsys):
    req = author / "request.json"
    req.write_text(json.dumps(_req(model="bf16")))
    assert fwd.main([str(req)]) == 2
    assert "invalid request" in capsys.readouterr().err
    req.write_text("{not json")
    assert fwd.main(["request.json"]) == 2


def test_fwd_takes_no_author_dir_or_socket_arguments(author, tmp_path):
    for extra in (["--author-dir", str(tmp_path)], ["--socket", str(tmp_path / "x.sock")], ["--author", "x"]):
        with pytest.raises(SystemExit) as e:
            fwd.main(["request.json", *extra])
        assert e.value.code == 2


def test_fwd_reads_only_regular_files_inside_the_author_dir(author, tmp_path, capsys):
    secret = tmp_path / "harness_secret.json"
    secret.write_text(json.dumps({"labels_of_val": 1, "hidden_key": 2}))
    (author / "sub").mkdir()
    (author / "link.json").symlink_to(secret)
    (author / "sub" / "dirlink").symlink_to(tmp_path, target_is_directory=True)
    (author / ".kernel").mkdir()
    (author / ".kernel" / "r.json").write_text(json.dumps(_req()))
    (author / "big.json").write_text(" " * (fwd.MAX_REQUEST_BYTES + 1))
    os.mkfifo(author / "fifo.json")
    for bad in (secret, author / ".." / "harness_secret.json", author / "link.json",
                author / "sub" / "dirlink" / "harness_secret.json", author / ".kernel" / "r.json",
                author / "big.json", author / "sub", author / "fifo.json", author / "missing.json"):
        assert fwd.main([str(bad)]) == 2, bad
        err = capsys.readouterr().err
        assert "invalid request" in err and "labels_of_val" not in err and "hidden_key" not in err, err


def test_wsp_alias_imports_without_connecting():
    from hypaqwen.wsp.author import wsp

    assert wsp.save_edit is C.save_edit and wsp.project_out is C.project_out
    assert C._client is None and "NAMES" not in vars(wsp)


def test_client_refuses_non_fp32_before_sending():
    with pytest.raises(C.EditRefused):
        C._wire(torch.zeros(3, dtype=torch.float64), "x")
    with pytest.raises(C.EditRefused):
        C._wire([1.0], "x")


def test_project_out_and_rank1():
    g = torch.Generator().manual_seed(0)
    W = torch.randn(6, 5, generator=g)
    u = torch.randn(6, generator=g)
    P0 = C.project_out(W, u)
    assert torch.allclose((u / u.norm()) @ P0, torch.zeros(5), atol=1e-5)
    P1 = C.project_out(W, u, cols=[1, 3])
    assert torch.equal(P1[:, [0, 2, 4]], W[:, [0, 2, 4]])
    assert torch.allclose((u / u.norm()) @ P1[:, [1, 3]], torch.zeros(2), atol=1e-5)
    a, b = torch.randn(6, generator=g), torch.randn(5, generator=g)
    assert torch.allclose(C.rank1(W, a, b) - W, torch.outer(a, b))
    assert torch.allclose(C.diff_means(W, W + 1.0), -torch.ones(5))


def test_control_direction_is_seeded_and_norm_matched():
    like = torch.full((4, 3), 2.0)
    a, b, c = C.control_direction(7, like), C.control_direction(7, like), C.control_direction(8, like)
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert torch.isclose(a.norm(), like.norm())


ARCH = {"layer_types": ["linear_attention", "full_attention"], "num_attention_heads": 8, "num_key_value_heads": 2,
        "head_dim": 256, "attn_output_gate": True, "linear_num_key_heads": 16, "linear_num_value_heads": 16,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128}


def test_head_slice_layout(monkeypatch):
    monkeypatch.setitem(C._cache, "ARCH", ARCH)
    q = torch.arange(4096 * 2, dtype=torch.float32).reshape(4096, 2)
    s = C.head_slice(1, 3, "q", q)
    assert s.name.endswith("layers.1.self_attn.q_proj.weight") and (s.start, s.stop) == (1536, 1792)
    assert torch.equal(s.weight, q[1536:1792])
    assert (C.head_slice(1, 3, "gate", q).start, C.head_slice(1, 5, "k", torch.zeros(512, 2)).start) == (1792, 256)
    o = C.head_slice(1, 2, "o", torch.zeros(1024, 2048))
    assert (o.dim, o.start, o.stop, tuple(o.weight.shape)) == (1, 512, 768, (1024, 256))
    v = C.head_slice(0, 3, "v", torch.zeros(6144, 4))
    assert (v.start, v.stop) == (4096 + 384, 4096 + 512)
    assert C.head_slice(0, 3, "k", torch.zeros(6144, 4)).start == 2048 + 384
    with pytest.raises(ValueError):
        C.head_slice(0, 3, "gate", torch.zeros(6144, 4))
    with pytest.raises(ValueError):
        C.head_slice(1, 8, "q", q)
