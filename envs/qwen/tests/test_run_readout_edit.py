"""``hypaqwen.run_readout --edit ARCHIVE`` (the WSP-A column of the WSP-B disclosure, PREREGISTRATION §7.1).

CPU only, no checkpoint: a toy tied model and a fake ``load_edit``; the real ``load_edit`` is the runner's
(``tests/test_wsp_runner.py``). Checks that the edit reaches the model through ``runner.apply_edit`` and that
``meta["edit"]`` names the archive the way ``agent_loop_lab.v04.disclosure.check_column`` requires."""

from __future__ import annotations

import hashlib
import json

import pytest
import torch

from hypaqwen import run_readout
from hypaqwen.wsp import runner

EMBED = runner.EMBED
PROJ = runner.TEXT_PREFIX + "proj.weight"


class _Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embed_tokens = torch.nn.Embedding(8, 4)
        self.model.proj = torch.nn.Linear(4, 4, bias=False)
        self.lm_head = torch.nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def _archive(tmp_path):
    rec = {"phase": "wsp-a", "edit_state_sha256": "e" * 64, "files": {}}
    d = tmp_path / "wsp-a"
    d.mkdir()
    (d / "BUILD_RECORD.json").write_text(json.dumps(rec))
    return d


def test_edit_is_applied_and_named_in_meta(tmp_path):
    d = _archive(tmp_path)
    model = _Toy()
    edit = {PROJ: torch.full((4, 4), 0.25)}
    seen = []
    meta = run_readout.apply_archive_edit(model, d, dtype="fp32", load=lambda a: seen.append(a) or edit,
                                          apply=runner.apply_edit)
    assert seen == [d] and torch.equal(model.model.proj.weight, edit[PROJ])
    assert meta == {"archive": "wsp-a", "phase": "wsp-a", "edit_state_sha256": "e" * 64,
                    "build_record_sha256": hashlib.sha256((d / "BUILD_RECORD.json").read_bytes()).hexdigest()}


def test_edit_needs_fp32(tmp_path):
    with pytest.raises(SystemExit, match="fp32"):
        run_readout.apply_archive_edit(_Toy(), _archive(tmp_path), dtype="bf16", load=dict, apply=runner.apply_edit)
    with pytest.raises(SystemExit):  # refused by the CLI before any model is loaded
        run_readout.main(["--split", "val", "--dtype", "bf16", "--edit", str(tmp_path), "--out", str(tmp_path / "x.npz")])


def test_default_loader_is_the_runners_checked_load_edit(tmp_path, monkeypatch):
    """Without fakes, the edit comes from runner.load_edit (record-checked, journal-checked builds only)."""
    calls = []
    monkeypatch.setattr(runner, "load_edit", lambda a: calls.append(("load", a)) or {})
    monkeypatch.setattr(runner, "apply_edit", lambda m, e: calls.append(("apply", e)) or {})
    d = _archive(tmp_path)
    run_readout.apply_archive_edit(_Toy(), d, dtype="fp32")
    assert calls == [("load", d), ("apply", {})]
