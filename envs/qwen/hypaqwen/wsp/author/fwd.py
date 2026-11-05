"""``./fwd request.json``: one metered forward pass (docs/V04_WSP_KERNEL.md §1 principle 2, §3 ``forward``).

Installed as ``tools/hn_kernel/fwd.py`` (read-only, checked by ``hypaqwen.wsp.install.verify_tools``
before every session and in the audit; the ``./fwd`` wrapper refuses to run if ``tools/`` is
writable); imports only the standard library and numpy. ``./fwd`` runs without the ``./py`` guard,
so it never imports anything from an author-writable directory: the kernel dir and the author dir
are removed from ``sys.path`` before the first non-builtin import, and ``protocol`` is loaded by
explicit file path from this file's own directory after its standard-library imports are loaded.
The interpreter install stays on ``sys.path`` even inside the author dir (the read-only author venv,
``AUTHOR_DIR/.venv``). The ``./fwd`` wrapper runs it as ``.venv/bin/python -I -B``.

1. Reads ``request.json``, which must be a regular file (no symlink anywhere below the author dir)
   under the author dir and at most ``MAX_REQUEST_BYTES``, and validates it with the server's own
   schema (``protocol.validate_forward_request``; or ``validate_disclosed_request`` for a WSP-B
   ``{"disclosed_windows": [...], "model": ...}`` request). A malformed request never reaches the
   server and costs nothing.
2. Sends one ``forward`` (or ``forward_disclosed``) op to the forward socket.
3. Writes ``output/<phase>/fwd/<call id>/``: ``response.json`` (label logits, logsumexp, argmax and
   top-k token ids, deltas for ``both``, the budget), ``request.json`` (a copy) and one ``.npy``
   per requested activation, ``<model>.s<session index>.<hidden|mlp_act>.L<layer>.npy`` of shape
   ``[n_windows, d]``.
4. Prints a short summary: counts, class indices and token totals. It never prints vocabulary strings.

The only argument is the request path. The author dir and the socket come from the launcher's
environment (``HN_AUTHOR_DIR``, ``HN_FWD_SOCKET``; the author's Bash hook cannot set variables), else
from the install layout (``AUTHOR_DIR/tools/hn_kernel/fwd.py``, ``AUTHOR_DIR/.kernel/fwd.sock``).

Exit status: 0 on success, 1 if the server refused the request (nothing charged), 2 if the request
was invalid before sending.
"""

# No ``from __future__`` import: it would be the first module looked up on sys.path.
import os  # frozen, already loaded at start-up
import sys  # builtin

_HERE = os.path.dirname(os.path.realpath(__file__))


def _author_dir() -> str:
    env = os.environ.get("HN_AUTHOR_DIR")
    return os.path.realpath(env) if env else os.path.dirname(os.path.dirname(_HERE))


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _scrub_sys_path() -> None:
    """Drop '' (the cwd), the kernel dir and the author dir from sys.path: nothing author-writable is importable.

    The interpreter install is kept even when it lies inside the author dir: the author venv is
    ``AUTHOR_DIR/.venv`` (read-only), and its site-packages holds numpy.
    """
    author = _author_dir()
    installs = [os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix)
                if p and os.path.realpath(p) != os.sep]

    def keep(p: str) -> bool:
        r = os.path.realpath(p)
        if _inside(r, _HERE):
            return False
        return not _inside(r, author) or any(_inside(r, i) for i in installs)

    sys.path[:] = [p for p in sys.path if p and keep(p)]


if not __package__:  # run as the ./fwd script (as a package module, the harness's own sys.path is left alone)
    _scrub_sys_path()

import argparse  # noqa: E402
import contextlib  # noqa: E402,F401  (protocol's imports, loaded before protocol itself)
import json  # noqa: E402
import math  # noqa: E402,F401
import re  # noqa: E402
import socket  # noqa: E402,F401
import stat  # noqa: E402
import struct  # noqa: E402,F401
import typing  # noqa: E402,F401
from pathlib import Path  # noqa: E402
from typing import Sequence  # noqa: E402

import numpy as np  # noqa: E402


def _sibling(name: str):
    """``<this dir>/<name>.py`` loaded by explicit path, never through sys.path (shared via sys.modules)."""
    import importlib.util

    key = f"hn_kernel_{name}"
    mod = sys.modules.get(key)
    if mod is None:
        spec = importlib.util.spec_from_file_location(key, os.path.join(_HERE, f"{name}.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        try:
            spec.loader.exec_module(mod)
        except BaseException:
            del sys.modules[key]
            raise
    return mod


if __package__:
    from . import protocol as P
else:
    P = _sibling("protocol")

AUTHOR_ENV = "HN_AUTHOR_DIR"
SOCKET_ENV = "HN_FWD_SOCKET"
MAX_REQUEST_BYTES = 8 << 20  # the per-request token cap (50,000) is far below this


class _BadRequest(Exception):
    pass


def _relative_inside(path: Path, author: Path) -> Path:
    """``path`` relative to ``author``; raises if it is not lexically inside (``..`` collapsed)."""
    lex = Path(os.path.abspath(path))
    for base in (Path(os.path.abspath(author)), Path(os.path.realpath(author))):
        try:
            rel = lex.relative_to(base)
        except ValueError:
            continue
        if ".." not in rel.parts:
            return rel
    raise _BadRequest("the request file must be inside the author dir")


def _no_links(author: Path, rel: Path) -> Path:
    """``realpath(author) / rel`` with every component below the author dir checked not to be a symlink."""
    root = Path(os.path.realpath(author))
    cur = root
    for part in rel.parts:
        cur = cur / part
        try:
            st = os.lstat(cur)
        except FileNotFoundError:
            break  # nothing below exists yet, so nothing below is a symlink
        if stat.S_ISLNK(st.st_mode):
            raise _BadRequest(f"{rel} goes through a symlink")
    return root / rel


def read_request(path: Path, author: Path) -> tuple[str, str]:
    """``(text, path relative to the author dir)`` of a request file, or ``_BadRequest``.

    The file must be a regular file inside the author dir (not in ``.kernel/``), reached without
    symlinks, at most MAX_REQUEST_BYTES. Error messages never echo the file's contents.
    """
    rel = _relative_inside(path, author)
    if rel.parts and rel.parts[0] == ".kernel":
        raise _BadRequest("the request file may not be in .kernel/")
    full = _no_links(author, rel)
    try:
        fd = os.open(full, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except OSError as e:
        raise _BadRequest(f"cannot open {rel}: {e.strerror}") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise _BadRequest(f"{rel} is not a regular file")
        if st.st_size > MAX_REQUEST_BYTES:
            raise _BadRequest(f"{rel} is {st.st_size} bytes, over the {MAX_REQUEST_BYTES}-byte limit")
        f = os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise
    with f:
        data = f.read(MAX_REQUEST_BYTES + 1)
    if len(data) > MAX_REQUEST_BYTES:
        raise _BadRequest(f"{rel} is over the {MAX_REQUEST_BYTES}-byte limit")
    try:
        return data.decode("utf-8"), str(rel)
    except UnicodeDecodeError:
        raise _BadRequest(f"{rel} is not UTF-8") from None


def _send(sock_path: Path, req: dict, request_rel: str, disclosed: bool) -> tuple[dict, dict[str, np.ndarray]]:
    if disclosed:
        header = {"op": "forward_disclosed", "args": {"window_ids": req["disclosed_windows"], "model": req["model"]}}
    else:
        header = {"op": "forward", "args": {"request": req, "request_path": request_rel}}
    sock = P.connect(sock_path)
    try:
        P.send_msg(sock, header)
        return P.recv_msg(sock)
    finally:
        sock.close()


def _out_dir(author: Path, phase: str, call_id: int) -> Path:
    """``author/output/<phase>/fwd/<call id>``, refusing a symlinked component on the way."""
    rel = Path("output") / phase / "fwd" / str(call_id)
    try:
        return _no_links(author, rel)
    except _BadRequest:
        raise ValueError(f"{rel.parent} goes through a symlink; refusing to write there") from None


def _write(out_dir: Path, result: dict, arrays: dict[str, np.ndarray], request_text: str, request_rel: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=False)  # a call id is never reused
    files = {}
    for key, a in arrays.items():
        if not re.fullmatch(r"[A-Za-z0-9_.]+", key):
            raise ValueError(f"bad array key from server {key!r}")
        np.save(out_dir / f"{key}.npy", a)
        files[key] = {"file": f"{key}.npy", "shape": list(a.shape)}
    response = {**result, "request_file": request_rel, "arrays": files}
    (out_dir / "response.json").write_text(json.dumps(response, indent=1) + "\n", encoding="utf-8")
    (out_dir / "request.json").write_text(request_text, encoding="utf-8")
    return response


def summary(response: dict, out_dir: Path) -> str:
    b = response["budget"]
    lines = [f"fwd call {response['call_id']}: model={response['model']} edit={response['edit_hash'][:12]} "
             f"tokens={response['tokens']} used={b['tokens_used']} left={b['tokens_left']} -> {out_dir}"]
    kinds = ("edited", "original") if response["model"] == "both" else (response["model"],)
    for s in response["sessions"][:20]:
        argmax = {k: s[k]["label_argmax"] for k in kinds}
        part = " ".join(f"{k}={v}" for k, v in argmax.items())
        if "delta" in s:
            part += f" label_flips={s['delta']['label_argmax_flips']} token_flips={s['delta']['argmax_flips']}"
        lines.append(f"  {s['id']}: label argmax (class index) {part}")
    if len(response["sessions"]) > 20:
        lines.append(f"  ... {len(response['sessions']) - 20} more sessions in response.json")
    if response["arrays"]:
        lines.append(f"  arrays: {len(response['arrays'])} .npy files")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="./fwd", description="Run one metered forward pass request.",
                                 allow_abbrev=False)
    ap.add_argument("request", type=Path)
    args = ap.parse_args(argv)
    author = Path(_author_dir())
    sock_path = Path(os.environ.get(SOCKET_ENV) or author / ".kernel" / "fwd.sock")
    try:
        text, request_rel = read_request(args.request, author)
        raw = json.loads(text)
        disclosed = isinstance(raw, dict) and "disclosed_windows" in raw
        req = P.validate_disclosed_request(raw) if disclosed else P.validate_forward_request(raw)
    except (_BadRequest, json.JSONDecodeError, RecursionError, P.RequestError) as e:
        print(f"fwd: invalid request {args.request}: {e}", file=sys.stderr)
        return 2
    header, arrays = _send(sock_path, req, request_rel, disclosed)
    if not header.get("ok"):
        err = header.get("error") or {}
        print(f"fwd: refused ({err.get('type')}): {err.get('message')}", file=sys.stderr)
        return 1
    result = header["result"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(result["phase"])) or not isinstance(result["call_id"], int):
        raise ValueError("bad phase or call id from server")
    out_dir = _out_dir(author, result["phase"], result["call_id"])
    response = _write(out_dir, result, arrays, text, request_rel)
    print(summary(response, out_dir.relative_to(os.path.realpath(author))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
