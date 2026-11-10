"""§12.3 interpreter test network against the plain-code exact-stem floor."""

from __future__ import annotations

import pytest

from conftest import SCREENS_DIR
from ui import model as mdl
from ui import tokenizer as tk
from validate import build_floor_net, check_floor
from validate.floor import PRECEDENCE, floor_screen, precedence
from validate.random_screens import random_screens


@pytest.fixture(scope="module")
def floor_art(tmp_path_factory):
    d = tmp_path_factory.mktemp("floor_net")
    build_floor_net.build(d)
    return mdl.load_artifact(d)


def _screen(nodes):
    return {"schema": "slop.screen/1", "id": "t", "nodes": [{"i": i, "loc": [1, 1, 1, 1], "flags": [], **n} for i, n in enumerate(nodes)]}


def _calls(screen):
    return [(c.cls, c.pointer) for c in floor_screen(tk.encode_screen(screen))]


def test_floor_rules():
    assert set(PRECEDENCE) == set(tk.ROLES)
    assert precedence("label", ["in_control"]) == 8 and precedence("label", []) == 2
    # nested: the eyebrow's stems are inside the heading's; the eyebrow ranks lower and is flagged
    assert _calls(_screen([{"k": 0, "role": "eyebrow", "text": "Features"}, {"k": 0, "role": "h2", "text": "Features for teams"}])) == [
        ("ECHO", 1), ("KEEP", None)]
    # equal sets at equal rank: the later node is flagged
    assert _calls(_screen([{"k": 0, "role": "body", "text": "Saved"}, {"k": 0, "role": "body", "text": "SAVE"}])) == [
        ("KEEP", None), ("ECHO", 0)]
    # different clusters, context-only roles, sr_only and word-less nodes take no part
    assert _calls(_screen([{"k": 0, "role": "body", "text": "Saved"}, {"k": 1, "role": "body", "text": "Saved"}])) == [("KEEP", None)] * 2
    assert _calls(_screen([{"k": 0, "role": "aria", "text": "Close"}, {"k": 0, "role": "button", "text": "Close"}])) == [("KEEP", None)] * 2
    assert _calls(_screen([{"k": 0, "role": "body", "text": "Close", "flags": ["sr_only"]}, {"k": 0, "role": "body", "text": "Close"}])) == [("KEEP", None)] * 2
    assert _calls(_screen([{"k": 0, "role": "stat", "text": "42"}, {"k": 0, "role": "stat", "text": "42 …"}])) == [("KEEP", None)] * 2
    # several partners: point at the highest-ranked, earlier first
    calls = _calls(_screen([{"k": 0, "role": "caption", "text": "Plan"}, {"k": 0, "role": "h3", "text": "Plan"},
                            {"k": 0, "role": "button", "text": "Choose plan"}, {"k": 0, "role": "link", "text": "Plan"}]))
    assert calls == [("ECHO", 2), ("ECHO", 2), ("KEEP", None), ("ECHO", 2)]


def test_network_equals_floor_on_test_screens(floor_art, test_screens):
    rep = check_floor.compare_floor(floor_art, test_screens)
    print(f"\ntest screens: {rep['nodes']} nodes; agreement with the floor {rep['agree']}/{rep['nodes']} "
          f"({rep['agreement']:.2%}); within {build_floor_net.RANGE} nodes {rep['agree_within_range']}/{rep['nodes']}; "
          f"flags floor {rep['floor_flags']}, network {rep['network_flags']}")
    for m in rep["disagreements"]:
        print("  disagreement:", m)
    assert rep["agree_within_range"] == rep["nodes"]
    assert rep["agreement"] >= 0.99


def test_network_equals_floor_within_range_on_random_screens(floor_art):
    screens = random_screens(300, seed=20261110)
    rep = check_floor.compare_floor(floor_art, screens)
    print(f"\nrandom screens: {len(screens)} screens, {rep['nodes']} nodes; agreement with the floor {rep['agree']}/{rep['nodes']} "
          f"({rep['agreement']:.2%}); within {build_floor_net.RANGE} nodes {rep['agree_within_range']}/{rep['nodes']}; "
          f"flags floor {rep['floor_flags']}, network {rep['network_flags']}")
    assert rep["agree_within_range"] == rep["nodes"]


def test_check_floor_cli(tmp_path, capsys):
    rc = check_floor.main([str(SCREENS_DIR), "--js", "--json", str(tmp_path / "r.json")])
    out = capsys.readouterr().out
    assert rc == 0 and "PASS" in out
