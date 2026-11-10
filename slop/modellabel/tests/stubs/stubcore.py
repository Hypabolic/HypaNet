"""Shared behaviour of the stub `claude` and `codex` executables used by the modellabel tests. They never call a
model. Each call is logged to $STUB_LOG (one JSON line: argv, cwd, selected env, stdin) and answered according to:

- STUB_SEQ_<NAME> (NAME = CLAUDE or CODEX): comma-separated actions consumed one per call (a counter file in
  $STUB_STATE), then "label" for every later call. Actions: label, fence (label in a ```json block), badjson,
  schema (leaves a node out), refuse, error, tools (codex: a command_execution item before the answer), wont (a
  broken reply that starts "I won't"), precedence (a heading labelled ECHO of its eyebrow, LABELLING.md §5), and for
  claude only: garbled (truncated stdout, exit 1), hang (sleeps past any test timeout), nocost (a valid reply with no
  total_cost_usd), maxtokens (stop_reason max_tokens), weird (safety_stops as a list), badutf8 (invalid UTF-8 in
  stdout); for codex only: unknown_item (an item.started of an unknown type, never completed).
- STUB_SLEEP: seconds each call sleeps before answering (concurrency tests).
- `--version` prints a version line (the harness records it).
- STUB_FLAVOUR_<NAME>: "a" or "b". Flavour b labels every subtitle KEEP; flavour a labels a subtitle NARRATE with
  the first button as target (so a and b disagree on subtitles).
- STUB_ADJ: "agree" (both adjudicators choose the redundant label of A and B) or "split" (claude takes A's label,
  codex takes B's, so they still disagree).
- STUB_COST: the claude stub's total_cost_usd per call (default 0.10).
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


def log_call(name: str, argv: list, stdin: str):
    p = os.environ.get("STUB_LOG")
    if not p:
        return
    env_keys = ["CLAUDE_CODE_EFFORT_LEVEL", "CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK", "CLAUDE_CODE_NO_MODEL_FALLBACK",
                "CLAUDE_CODE_DISABLE_AUTO_MEMORY", "CLAUDE_CODE_DISABLE_CLAUDE_MDS", "CLAUDECODE"]
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps({"name": name, "argv": argv, "cwd": os.getcwd(), "stdin": stdin,
                            "env": {k: os.environ.get(k) for k in env_keys}}) + "\n")


def next_action(name: str) -> str:
    seq = [x.strip() for x in os.environ.get(f"STUB_SEQ_{name.upper()}", "").split(",") if x.strip()]
    state = Path(os.environ.get("STUB_STATE", "/nonexistent"))
    n = 0
    counter = state / f"{name}.count"
    if state.is_dir():
        try:
            n = int(counter.read_text())
        except (OSError, ValueError):
            n = 0
        counter.write_text(str(n + 1))
    return seq[n] if n < len(seq) else "label"


def parse_nodes(text: str) -> list:
    m = re.search(r"<nodes>\n(.*?)\n</nodes>", text, re.S)
    return [json.loads(x) for x in m.group(1).splitlines()] if m else []


def screen_id(text: str) -> str:
    m = re.search(r'<screen id="([^"]+)">', text)
    return m.group(1) if m else "?"


def label_reply(text: str, flavour: str) -> dict:
    nodes = parse_nodes(text)
    out = []
    heading = next((n["i"] for n in nodes if n["role"] in ("h1", "h2", "dialog_title")), None)
    button = next((n["i"] for n in nodes if n["role"] == "button"), None)
    for n in nodes:
        if n.get("context_only"):
            continue
        lab = {"i": n["i"], "class": "KEEP", "target": None, "pattern": None, "conf": "clear", "partial": False,
               "ct": False, "equiv": [], "aside": [], "role_ok": True, "role_fix": None, "why": "step 6: carries a fact"}
        if n.get("no_literal_words"):
            lab.update({"class": None, "aside": ["UNRESOLVED"], "why": "no literal words"})
        elif n["role"] == "eyebrow" and heading is not None and flavour != "precedence":
            lab.update({"class": "ECHO", "target": heading, "pattern": "R1", "why": "step 2: restates the heading"})
        elif flavour == "precedence" and n["i"] == heading:
            eyebrow = next((x["i"] for x in nodes if x["role"] == "eyebrow"), None)
            if eyebrow is not None:
                lab.update({"class": "ECHO", "target": eyebrow, "pattern": "R1", "why": "step 2: wrong way round"})
        elif n["role"] == "subtitle" and flavour == "a" and button is not None:
            lab.update({"class": "NARRATE", "target": button, "pattern": "R6", "ct": True, "conf": "judgement",
                        "why": "step 4: explains the button"})
        out.append(lab)
    return {"screen": screen_id(text), "nodes": out, "missed": [], "aside_screen": []}


def adjudicate_reply(text: str, who: str) -> dict:
    m = re.search(r"<disputed>\n(.*?)\n</disputed>", text, re.S)
    items = [json.loads(x) for x in m.group(1).splitlines()] if m else []
    mode = os.environ.get("STUB_ADJ", "agree")
    out = []
    for it in items:
        a, b = it["A"], it["B"]
        if mode == "split":
            pick = a if who == "claude" else b
        else:
            pick = a if a["class"] != "KEEP" else b
        out.append({"i": it["i"], "class": pick["class"], "target": pick["target"], "pattern": pick["pattern"],
                    "conf": pick.get("conf", "judgement"), "partial": pick["partial"], "ct": pick["ct"], "equiv": pick["equiv"],
                    "aside": pick["aside"], "reason": "stub: picked one side"})
    return {"screen": screen_id(text), "nodes": out}


def answer(name: str, text: str, action: str) -> str:
    """The model's final message for an action (or raises for actions handled by the caller)."""
    flavour = os.environ.get(f"STUB_FLAVOUR_{name.upper()}", "a")
    if "<disputed>" in text:
        obj = adjudicate_reply(text, name)
    else:
        obj = label_reply(text, "precedence" if action == "precedence" else flavour)
    if action == "schema" and obj["nodes"]:
        obj["nodes"] = obj["nodes"][1:]
    if action == "badjson":
        return "Here are the labels: {not json"
    if action == "refuse":
        return "I can't help with that request."
    if action == "wont":
        return "I won't restate node 2's label; here it is: {\"screen\": "
    s = json.dumps(obj)
    return f"```json\n{s}\n```" if action == "fence" else s


def arg_value(argv: list, *names):
    for n, a in enumerate(argv):
        if a in names and n + 1 < len(argv):
            return argv[n + 1]
    return None


def read_stdin() -> str:
    return sys.stdin.read()


def maybe_sleep():
    import time
    t = float(os.environ.get("STUB_SLEEP") or 0)
    if t:
        time.sleep(t)
