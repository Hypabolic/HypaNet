"""v0.3 inspector CLI (spec v0.3 §12)."""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import torch

from agent_loop_lab.v03 import inspect_cli
from agent_loop_lab.v03.artifact import save_artifact
from agent_loop_lab.v03.model import BOS, EOE, R, zero_model


def _encoded_session() -> dict:
    # BOS, event 0 = "read_file path=a" EOE, event 1 = "run_tests FAILED" EOE
    surface = ["<BOS>", "read_file", "path", "=", "a", "<EOE>", "run_tests", "FAILED", "<EOE>"]
    tokens = [BOS, 7, 8, 140, 3000, EOE, 9, 10, EOE]
    event_ord = [0, 1, 1, 1, 1, 1, 2, 2, 2]
    offset = [0, 0, 1, 2, 3, 4, 0, 1, 2]
    return {
        "tokens": tokens,
        "event_ord": event_ord,
        "offset": offset,
        "eoe_index": [5, 8],
        "surface": surface,
        "blocks": ["read_file path=a", "run_tests FAILED"],
    }


def _run(argv):
    out, err = io.StringIO(), io.StringIO()
    code = inspect_cli.main(argv, out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def _model_with_head():
    m = zero_model()
    with torch.no_grad():
        m.head.bias[2] = 1.0
        m.tok_emb.weight[3000, 5] = 1.0
        m.layers[1].attn.rel_bias[3, R] = 2.0
        m.layers[1].attn.rel_bias[3, 1] = 4.0
    return m


def test_inspect_log_encoded(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(_encoded_session()))
    code, out, err = _run(["inspect", "log", str(p)])
    assert code == 0, err
    assert "events: 2" in out and "== event 1" in out
    assert "3000:a" in out and "| run_tests FAILED" in out
    assert "-> PRODUCTIVE" in out
    art = tmp_path / "m.dlw"
    save_artifact(_model_with_head(), art)
    code, out, err = _run(["inspect", "log", str(p), "--artifact", str(art)])
    assert code == 0, err
    assert "-> STALLED" in out and "sha256" in out


def test_inspect_log_blocks_file_uses_tokenizer(tmp_path, monkeypatch):
    enc = _encoded_session()
    seen = {}

    def fake_encode(blocks):
        seen["blocks"] = blocks
        return SimpleNamespace(**{k: enc[k] for k in ("tokens", "event_ord", "offset", "eoe_index", "surface")})

    monkeypatch.setattr(inspect_cli, "encode_blocks", fake_encode)
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(b) for b in enc["blocks"]) + "\n")
    code, out, err = _run(["inspect", "log", str(p)])
    assert code == 0, err
    assert seen["blocks"] == enc["blocks"]
    assert "| read_file path=a" in out


def test_inspect_heads(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(_encoded_session()))
    art = tmp_path / "m.dlw"
    save_artifact(_model_with_head(), art)
    code, out, err = _run(["inspect", "heads", str(p), "--event", "1", "--artifact", str(art), "--layer", "1"])
    assert code == 0, err
    assert "EOE of event 1 at position 8" in out
    head3 = [line for line in out.splitlines() if line.strip().startswith("head 3:")][0]
    # bucket 1 (event 0) dominates, BOS next
    assert head3.split("events ")[1].startswith("e0:")
    assert "BOS:" in head3
    code, _, err = _run(["inspect", "heads", str(p), "--event", "2"])
    assert code == 2 and "outside" in err


def test_inspect_model(tmp_path):
    code, out, _ = _run(["inspect", "model"])
    assert code == 0 and "non-zero" in out and ": 0 /" in out
    art = tmp_path / "m.dlw"
    save_artifact(_model_with_head(), art)
    code, out, _ = _run(["inspect", "model", str(art)])
    assert code == 0
    assert "tok_emb.weight (10240, 128): 1 non-zero" in out
    assert "rows 3000: cols 5 (1)" in out
    assert "layers.1.attn.rel_bias (8, 65): 2 non-zero" in out
    assert "rows 3: cols 1,64 (2)" in out
    assert "head.bias (4,): 1 non-zero" in out and "idx 2" in out


def test_inspect_errors(tmp_path):
    code, _, err = _run(["inspect", "log", str(tmp_path / "missing.json")])
    assert code == 2 and "cannot read" in err
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    code, _, err = _run(["inspect", "log", str(bad)])
    assert code == 2
    art = tmp_path / "x.dlw"
    art.write_bytes(b"DLW1junk")
    code, _, err = _run(["inspect", "model", str(art)])
    assert code == 2 and "invalid artefact" in err
    code, _, _ = _run(["inspect"])
    assert code == 2
