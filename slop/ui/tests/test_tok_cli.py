"""The ./tok CLI (ui/tok.py) and the compile CLI."""

from __future__ import annotations

import subprocess
import sys

import pytest

from conftest import SCREENS_DIR, SLOP
from ui import tok
from ui import tokenizer as tk


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


def test_id_layout_2(tmp_path, capsys):
    (tmp_path / "vocab.txt").write_text("".join(f"w{i}\n" for i in range(3999)) + "resum\n")
    assert tok.main(["--text", "Resume configure", "--vocab-file", str(tmp_path / "vocab.txt"), "--ids", "2"]) == 0
    out = capsys.readouterr().out
    assert "id  4511  vocab line 4000" in out and f"id {tk.hash_bucket('configur', 2):>5}  hash bucket" in out
    assert tok.main(["--vocab", str(tmp_path / "vocab.txt")]) == 0
    assert "4000 lines (max 3584, id layout 1)" in capsys.readouterr().out.splitlines()[0]
    (tmp_path / "meta.json").write_text('{"format": "slop.artifact/1", "pointer_head": [4, 0], "ids": 2}')
    assert tok.main(["--vocab", str(tmp_path / "vocab.txt")]) == 0
    out = capsys.readouterr().out
    assert "4000 lines (max 16384, id layout 2)" in out.splitlines()[0]
    assert tok.main(["--text", "configure", "--vocab-file", str(tmp_path / "vocab.txt")]) == 0  # layout from meta.json
    assert f"id {tk.hash_bucket('configur', 2):>5}" in capsys.readouterr().out


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


def test_vocab_restem_is_a_note_not_a_problem(tmp_path, capsys):
    # Porter is not idempotent: 'paus' (the stem of pause) stems again to 'pau', yet text produces it
    (tmp_path / "vocab.txt").write_text("paus\n")
    (tmp_path / "lexicon").mkdir()
    (tmp_path / "lexicon" / "ui.txt").write_text("pause paused pausing\n")
    assert tok.main(["--vocab", str(tmp_path / "vocab.txt")]) == 0
    out = capsys.readouterr().out
    assert "changes when stemmed again: line 1 'paus' -> 'pau' (lexicon words stem to it: pause paused pausing)" in out
    assert "never produces" not in out and out.splitlines()[1] == "0 problem(s)"


def test_vocab_mode_puts_the_count_first_and_pages_long_lists(tmp_path, capsys):
    stems = [f"zq{chr(97 + i // 26)}{chr(97 + i % 26)}t" for i in range(120)]  # zqaat, zqabt, ...: clean stems
    (tmp_path / "vocab.txt").write_text("\n".join(stems + ["Bad"] * 3) + "\n")
    (tmp_path / "lexicon").mkdir()
    (tmp_path / "lexicon" / "ui.txt").write_text("\n".join(f"{s} {s}s" for s in stems) + "\n")
    assert tok.main(["--vocab", str(tmp_path / "vocab.txt")]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[1] == "5 problem(s)"  # three unclean lines and two duplicates, counted before any list
    merges = next(x for x, ln in enumerate(out) if ln.startswith("merges"))
    assert out[merges].startswith("merges (stems more than one lexicon word maps to): 120; showing 1-50 (--offset N")
    assert sum(ln.startswith("  merge: ") for ln in out) == 50
    assert tok.main(["--vocab", str(tmp_path / "vocab.txt"), "--offset", "100"]) == 0
    out = capsys.readouterr().out
    assert "120; showing 101-120" in out and f"  merge: {stems[100]} {stems[100]}s -> '{stems[100]}'" in out
    assert out.splitlines()[1] == "5 problem(s)" and "problems: 5; showing none" in out
