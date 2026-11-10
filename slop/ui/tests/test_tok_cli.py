"""The ./tok CLI (ui/tok.py) and the compile CLI."""

from __future__ import annotations

import subprocess
import sys

import pytest

from conftest import SCREENS_DIR, SLOP
from ui import tok


def test_text_mode(capsys):
    assert tok.main(["--text", "OPERATION ON HOLD … 01"]) == 0
    out = capsys.readouterr().out
    assert "flags  caps" in out and "stem 'oper'" in out and "ELLIPSIS" in out and "LZ" in out and "EON id 2" in out


def test_screen_and_words_modes(tmp_path, capsys):
    (tmp_path / "vocab.txt").write_text("save\nset\n")
    assert tok.main([str(SCREENS_DIR / "notification_settings.json"), "--vocab-file", str(tmp_path / "vocab.txt")]) == 0
    out = capsys.readouterr().out
    assert "vocab line 2" in out and "window 0" in out and "role=label" in out
    (tmp_path / "words.txt").write_text("settings news\ngeneral\n")
    assert tok.main(["--words", str(tmp_path / "words.txt")]) == 0
    out = capsys.readouterr().out
    assert "'settings' -> 'set'" in out and "'news' -> 'new'" in out and "'general' -> 'gener'" in out


def test_vocab_mode(tmp_path, capsys):
    (tmp_path / "vocab.txt").write_text("new\nsettings\nnew\nPaused\ngener\n")
    (tmp_path / "lexicon").mkdir()
    (tmp_path / "lexicon" / "ui.txt").write_text("new news\ngeneral generate generic\n")
    assert tok.main(["--vocab", str(tmp_path / "vocab.txt")]) == 0
    out = capsys.readouterr().out
    assert "duplicate: line 3 'new'" in out
    assert "changes when stemmed again: line 2 'settings'" in out
    assert "not a clean stem" in out and "'Paused'" in out
    assert "merge: new news -> 'new'" in out and "merge: general generate generic -> 'gener'" in out


def test_root_and_cap(tmp_path, capsys):
    with pytest.raises(SystemExit):
        tok.main([str(SCREENS_DIR / "toasts.json"), "--root", str(tmp_path)])
    assert tok.main([str(SCREENS_DIR / "unicode_mix.json"), "--cap", "300"]) == 0
    assert "[output truncated at 300 characters" in capsys.readouterr().out


def test_scripts_run_standalone(tmp_path):
    r = subprocess.run([sys.executable, "ui/tok.py", "--text", "Save settings"], cwd=SLOP, capture_output=True, text=True)
    assert r.returncode == 0 and "stem 'save'" in r.stdout
    r = subprocess.run([sys.executable, "validate/build_floor_net.py", str(tmp_path / "art")], cwd=SLOP, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, "lint/compile.py", str(tmp_path / "art"), "-o", str(tmp_path / "n.bin")], cwd=SLOP, capture_output=True, text=True)
    assert r.returncode == 0 and (tmp_path / "n.bin").read_bytes()[:8] == b"SLOPNET1", r.stderr
