"""WSP kernel wire format and request schema (docs/V04_WSP_KERNEL.md §3), shared by server and clients.

Self-contained: imports only the standard library and numpy, because it is copied into the author
directory (``tools/hn_kernel/``) and runs in the author's venv. The model server imports it through
``hypaqwen.wsp.protocol``, so both ends always speak the same format.

A frame is ``MAGIC`` (4 bytes), the header length (uint64, little-endian), a UTF-8 JSON header, then
the raw bytes of each tensor listed in ``header["tensors"]``, in order, as C-contiguous little-endian
float32. A request header is ``{"op", "args", "tensors"}``; a response header is
``{"ok": true, "result", "tensors"}`` or ``{"ok": false, "error": {"type", "message"}}``.

The forward-request schema is validated here, so ``./fwd`` refuses a malformed request before it
reaches the server, and the server applies exactly the same rules again.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import socket
import struct
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

MAGIC = b"HNK1"
MAX_HEADER_BYTES = 64 << 20  # JSON headers carry texts and logits, never weights
MAX_TENSOR_BYTES = 4 << 30  # the largest tensor is the 248320 x 1024 embedding (1.0 GB)
MAX_TENSORS = 1 << 16  # tensors per frame (a forward response carries one per session, layer and model)
MAX_NDIM = 8
MAX_DIM = (1 << 31) - 1
WIRE_DTYPE = np.dtype("<f4")
_SUN_PATH_MAX = 100  # sockaddr_un.sun_path is 104 bytes on macOS, 108 on Linux

MODELS = ("edited", "original", "both")
MAX_TOPK = 20
N_LAYERS = 24  # Qwen3.5-0.8B decoder blocks; hidden layers run 0..N_LAYERS, MLP layers 0..N_LAYERS-1


class ProtocolError(RuntimeError):
    """A malformed frame. The connection cannot be trusted after one and is closed."""


class RequestError(ValueError):
    """A request that breaks the schema. Nothing ran and nothing was charged."""


# ---------------------------------------------------------------- framing


def _recv_exact(sock: socket.socket, n: int) -> bytearray:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], min(n - got, 1 << 24))
        if k == 0:
            raise EOFError("connection closed")
        got += k
    return buf


def send_msg(sock: socket.socket, header: Mapping[str, Any], tensors: Mapping[str, np.ndarray] | None = None) -> None:
    """Send one frame. Tensors must already be float32; nothing is cast silently."""
    arrays = []
    specs = []
    for key, arr in (tensors or {}).items():
        arr = np.asarray(arr)
        if arr.dtype != np.float32:
            raise TypeError(f"tensor {key!r} is {arr.dtype}, the wire carries float32 only")
        arr = np.ascontiguousarray(arr, dtype=WIRE_DTYPE)
        specs.append({"key": key, "shape": list(arr.shape)})
        arrays.append(arr)
    # NaN/inf are allowed: a large edit can overflow a logit, and both ends are Python's json.
    head = json.dumps({**header, "tensors": specs}, separators=(",", ":")).encode("utf-8")
    if len(head) > MAX_HEADER_BYTES:
        raise ProtocolError(f"header of {len(head)} bytes exceeds {MAX_HEADER_BYTES}")
    sock.sendall(MAGIC + struct.pack("<Q", len(head)) + head)
    for arr in arrays:
        if arr.nbytes:
            sock.sendall(memoryview(arr.reshape(-1)).cast("B"))


def recv_msg(sock: socket.socket, max_total_bytes: int | None = None) -> tuple[dict, dict[str, np.ndarray]]:
    """Receive one frame. Raises EOFError on a clean close before a frame, ProtocolError on garbage.

    Every malformed header (bad JSON, too deep, a non-object tensor spec, a huge or overflowing shape)
    is a ProtocolError, so a hostile frame can only cost its own connection. ``max_total_bytes``
    bounds the frame's tensor bytes before any buffer is allocated (the server passes it).
    """
    first = _recv_exact(sock, 12)
    if bytes(first[:4]) != MAGIC:
        raise ProtocolError("bad magic")
    (n,) = struct.unpack("<Q", first[4:])
    if n > MAX_HEADER_BYTES:
        raise ProtocolError(f"header of {n} bytes exceeds {MAX_HEADER_BYTES}")
    try:
        header = json.loads(bytes(_recv_exact(sock, n)).decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as e:  # JSONDecodeError is a ValueError
        raise ProtocolError(f"bad header: {type(e).__name__}") from None
    if not isinstance(header, dict) or not isinstance(header.get("tensors", []), list):
        raise ProtocolError("header is not an object with a tensor list")
    specs = header.pop("tensors", [])
    if len(specs) > MAX_TENSORS:
        raise ProtocolError(f"{len(specs)} tensors in one frame exceeds {MAX_TENSORS}")
    plan, seen = [], set()
    for spec in specs:
        if not isinstance(spec, dict):
            raise ProtocolError("a tensor spec must be an object")
        key, shape = spec.get("key"), spec.get("shape")
        if not isinstance(key, str) or len(key) > 256 or key in seen \
                or not isinstance(shape, list) or len(shape) > MAX_NDIM \
                or not all(type(d) is int and 0 <= d <= MAX_DIM for d in shape):
            raise ProtocolError("bad tensor spec")
        nbytes = math.prod(shape) * WIRE_DTYPE.itemsize  # Python ints: no overflow
        if nbytes > MAX_TENSOR_BYTES:
            raise ProtocolError(f"tensor {key!r} of {nbytes} bytes exceeds {MAX_TENSOR_BYTES}")
        seen.add(key)
        plan.append((key, tuple(shape), nbytes))
    total = sum(nb for _, _, nb in plan)
    if max_total_bytes is not None and total > max_total_bytes:
        raise ProtocolError(f"frame carries {total} tensor bytes, over {max_total_bytes}")
    tensors: dict[str, np.ndarray] = {}
    for key, shape, nbytes in plan:
        buf = _recv_exact(sock, nbytes) if nbytes else bytearray()
        tensors[key] = np.frombuffer(buf, dtype=WIRE_DTYPE).reshape(shape)  # writable: backed by a bytearray
    return header, tensors


def error_header(kind: str, message: str) -> dict:
    return {"ok": False, "error": {"type": kind, "message": message}}


# ---------------------------------------------------------------- Unix sockets


@contextlib.contextmanager
def _sun_path(path: Path) -> Iterator[str]:
    """Yield an address for ``path`` that fits sun_path, by binding relative to its directory if needed.

    The fallback changes the working directory briefly, so it is only safe in a single-threaded
    process (the server's start-up, or an author client, which may not start threads).
    """
    path = Path(path)
    if len(os.fsencode(str(path))) < _SUN_PATH_MAX:
        yield str(path)
        return
    if len(os.fsencode(path.name)) >= _SUN_PATH_MAX:
        raise ValueError(f"socket name too long: {path.name!r}")
    here = os.open(".", os.O_RDONLY)
    try:
        os.chdir(path.parent)
        yield path.name
    finally:
        os.fchdir(here)
        os.close(here)


def connect(path: Path | str, timeout: float | None = None) -> socket.socket:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        with _sun_path(Path(path)) as addr:
            s.connect(addr)
    except BaseException:
        s.close()
        raise
    return s


def listen(path: Path | str) -> socket.socket:
    """Bind a listening socket at ``path`` (owner-only), replacing a stale socket file."""
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.is_socket():
        path.unlink()
    elif path.exists():
        raise FileExistsError(f"{path} exists and is not a socket")
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        with _sun_path(path) as addr:
            s.bind(addr)
        os.chmod(path, 0o600)
        s.listen(1)
    except BaseException:
        s.close()
        raise
    return s


# ---------------------------------------------------------------- request schema


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _layers(x: Any, hi: int, what: str) -> list[int]:
    if not isinstance(x, list) or not all(_is_int(v) for v in x):
        raise RequestError(f"return.{what} must be a list of ints")
    if len(set(x)) != len(x) or any(v < 0 or v > hi for v in x):
        raise RequestError(f"return.{what} must be distinct layers in 0..{hi}")
    return sorted(x)


def _model(x: Any) -> str:
    if x not in MODELS:
        raise RequestError(f"model must be one of {list(MODELS)}, not {x!r}")
    return x


def validate_forward_request(req: Any, n_layers: int = N_LAYERS) -> dict:
    """Check a ``./fwd`` request and return it normalised (defaults filled, layers sorted).

    ``{"model": "edited"|"original"|"both",
       "sessions": [{"id": str, "blocks": [str, ...], "windows": [int, ...]}, ...],
       "return": {"hidden": [layer], "mlp_act": [layer], "topk": int <= 20}}``  (``return`` optional)

    ``windows`` are block indices, strictly increasing, each ending a window to read out. The last
    window must end at the last block: blocks after it would never run, so they are refused rather
    than ignored (the ledger hashes exactly what runs).
    ``hidden`` layers run 0..n_layers (the residual stream after that many blocks, 0 = embeddings,
    n_layers = before the final norm); ``mlp_act`` layers run 0..n_layers-1.
    """
    if not isinstance(req, dict):
        raise RequestError("request must be a JSON object")
    extra = set(req) - {"model", "sessions", "return"}
    if extra:
        raise RequestError(f"unknown request keys {sorted(extra)}")
    model = _model(req.get("model"))
    sessions = req.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise RequestError("sessions must be a non-empty list")
    out_sessions, seen = [], set()
    for i, s in enumerate(sessions):
        if not isinstance(s, dict) or set(s) != {"id", "blocks", "windows"}:
            raise RequestError(f"sessions[{i}] must have exactly the keys id, blocks, windows")
        sid, blocks, windows = s["id"], s["blocks"], s["windows"]
        if not isinstance(sid, str) or not 0 < len(sid) <= 128 or sid in seen:
            raise RequestError(f"sessions[{i}].id must be a distinct non-empty string of at most 128 characters")
        seen.add(sid)
        if not isinstance(blocks, list) or not blocks or not all(isinstance(b, str) for b in blocks):
            raise RequestError(f"sessions[{i}].blocks must be a non-empty list of strings")
        if not isinstance(windows, list) or not windows or not all(_is_int(w) for w in windows):
            raise RequestError(f"sessions[{i}].windows must be a non-empty list of ints")
        if any(b <= a for a, b in zip(windows, windows[1:])) or windows[0] < 0 or windows[-1] >= len(blocks):
            raise RequestError(f"sessions[{i}].windows must be strictly increasing block indices in 0..{len(blocks) - 1}")
        if windows[-1] != len(blocks) - 1:
            raise RequestError(f"sessions[{i}]: blocks {windows[-1] + 1}..{len(blocks) - 1} come after the last window "
                               "and would never run; drop them")
        out_sessions.append({"id": sid, "blocks": list(blocks), "windows": list(windows)})
    ret = req.get("return", {})
    if not isinstance(ret, dict) or set(ret) - {"hidden", "mlp_act", "topk"}:
        raise RequestError("return must be an object with keys among hidden, mlp_act, topk")
    topk = ret.get("topk", 0)
    if not _is_int(topk) or not 0 <= topk <= MAX_TOPK:
        raise RequestError(f"return.topk must be an int in 0..{MAX_TOPK}")
    return {
        "model": model,
        "sessions": out_sessions,
        "return": {"hidden": _layers(ret.get("hidden", []), n_layers, "hidden"),
                   "mlp_act": _layers(ret.get("mlp_act", []), n_layers - 1, "mlp_act"),
                   "topk": topk},
    }


def validate_disclosed_request(req: Any) -> dict:
    """``{"disclosed_windows": [window id, ...], "model": ...}`` for WSP-B's ``forward_disclosed``."""
    if not isinstance(req, dict) or set(req) != {"disclosed_windows", "model"}:
        raise RequestError("a disclosed request has exactly the keys disclosed_windows, model")
    ids = req["disclosed_windows"]
    if not isinstance(ids, list) or not ids or not all(isinstance(v, str) for v in ids) or len(set(ids)) != len(ids):
        raise RequestError("disclosed_windows must be a non-empty list of distinct window ids")
    return {"disclosed_windows": list(ids), "model": _model(req["model"])}
