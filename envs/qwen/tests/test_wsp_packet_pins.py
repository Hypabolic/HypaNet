"""The v0.4 packet's pinned interface token ids equal the real tokenizer's (src/agent_loop_lab/v04/packet.py).

The root environment builds the packet and has no tokenizer library, so it pins the prefix, suffix and
label ids and checks only that they decode to the interface strings. This test checks the pins against
``hypaqwen.interface.build_interface`` on the pinned ``tokenizer.json``, as the model server tokenises.
It reads packet.py as text (this environment never imports ``agent_loop_lab``). No model is loaded.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from hypaqwen import interface as I
from hypaqwen import model as M
from hypaqwen.wsp import server

PACKET_PY = Path(__file__).resolve().parents[3] / "src" / "agent_loop_lab" / "v04" / "packet.py"


def _pins() -> dict:
    tree = ast.parse(PACKET_PY.read_text())
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try:
                out[n.targets[0].id] = ast.literal_eval(n.value)
            except ValueError:
                pass
    return out


@pytest.fixture(scope="module")
def tok():
    tokenizers = pytest.importorskip("tokenizers")
    try:
        root = M.snapshot_dir()
    except Exception as exc:  # noqa: BLE001 - not in the local cache
        pytest.skip(f"pinned snapshot unavailable: {exc}")

    class Tok:  # the encode() of the server's tokenizer, through the tokenizers library
        def __init__(self, t):
            self.t = t

        def encode(self, text, add_special_tokens=False):
            return self.t.encode(text, add_special_tokens=add_special_tokens).ids

    return Tok(tokenizers.Tokenizer.from_file(str(root / "tokenizer.json")))


def test_pinned_ids_equal_the_tokenizer(tok):
    pins = _pins()
    iface = I.build_interface(tok)
    assert pins["PREFIX_IDS"] == iface.prefix_ids
    assert pins["SUFFIX_IDS"] == iface.suffix_ids
    assert pins["LABEL_IDS"] == iface.label_ids
    assert pins["INTERFACE_SPEC_SHA256"] == I.spec_sha256()
    assert not set(iface.label_ids) & set(iface.prefix_ids + iface.suffix_ids)


def test_pinned_budget_and_quota_equal_the_server():
    pins = _pins()
    assert pins["PHASE_BUDGET"] == server.PHASE_BUDGET and pins["REQUEST_CAP"] == server.REQUEST_CAP
    assert server.LOAD_QUOTA_FRACTION == 0.5
    assert pins["LOAD_QUOTA_BYTES"] == int(server.LOAD_QUOTA_FRACTION * 4 * 498_112_320)
