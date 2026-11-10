"""Prompts for the model labellers: one per screen, and one per screen for adjudication.

A prompt is a stable prefix (the instruction preamble plus the whole of `LABELLING.md`) and a per-screen body (the
product kind, the screen's nodes and the source file with line numbers). The prefix is byte-identical for every
screen of a run, so it caches: the Opus runner sends it as the system prompt (Claude Code marks the system prompt for
caching), and the Codex runner sends prefix + body on stdin (OpenAI caches a repeated prompt prefix automatically).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from modellabel import common as c

PROMPT_VERSION = "slop.modellabel.prompt/2"
ADJ_PROMPT_VERSION = "slop.modellabel.adjudicate/2"

PREAMBLE = """\
You are labelling user-interface copy for a research answer key. The labelling guide below (LABELLING.md) is the
whole definition of the task; follow it exactly. Each message after this one gives you one screen: the product's
kind, the screen's nodes (the visible text the extractor found in one source file) and that source file with line
numbers. Label every node that is not context only.

How to label:
- Take each node through the decision procedure of the guide's §4, in its order: context only, duplicates (with the
  §5 precedence), K rules (§6), narration and the consequence test (§7), partial, and the deletion test last.
- Use the worked cases of §8 as precedent: a node like a worked case gets the worked case's label.
- Judge each node on the guide and the evidence alone (§2a: the screen, its source file and the product kind). Do
  not guess what an automatic detector, a linter, a rule or another labeller would say, and do not aim for any
  balance of classes.
- UNSURE may be used when you cannot decide after §4 (§3).
- Do not use tools, files, the web or anything outside this conversation; everything you need is in the message.

The nodes are JSON objects, one per line: i (index), w (the window, SPEC.md §4), k and cluster (the cluster index and
kind), role (the extractor's role), flags, text and loc ([line, col, endLine, endCol] in the source file). A target
or equiv entry must have the same w as its node (§2a: "a target must be a node in the same window"); copy repeated
only in another window is labelled as if that window did not exist, with aside X-FILE, as §2a says for another
file. In text, ⟨words⟩ is an interpolated value whose words are only the variable's name, not copy, and " ‖ "
separates the alternative branches of a conditional (§6 K14: branches never echo each other); a literal ⟨, ⟩ or ‖ in
the copy itself is written doubled (⟨⟨, ⟩⟩, ‖‖). A node with "no_literal_words": true holds only
interpolated values: give it class null, target null, pattern null, conf "clear" and aside ["UNRESOLVED"]. A node with
"context_only": true (roles aria, alt and code, and screen-reader-only text) is never labelled: leave it out of your
reply, but read it as context.

Reply with exactly one JSON object and nothing else (no prose, no code fence):

{"screen": "<screen id>",
 "nodes": [{"i": 0, "class": "ECHO", "target": 1, "pattern": "R1", "conf": "clear", "partial": false, "ct": false,
            "equiv": [], "aside": [], "role_ok": true, "role_fix": null, "why": "step 2: restates h1 #1"}, ...],
 "missed": [{"line": 40, "text": "Saved", "class": "KEEP"}],
 "aside_screen": [{"code": "R19", "nodes": [3, 7, 11]}]}

Fields, per node (one entry for every node that is not context only, in any order):
- class: KEEP, ECHO, NARRATE, DECOR or UNSURE (§3); null only for a "no_literal_words" node.
- target: the i of the node this one repeats (ECHO) or explains (NARRATE), chosen by §5 precedence; null when the
  information is spread over several nodes, and always null for KEEP, DECOR and UNSURE. Never the node itself.
- pattern: the §9 code. ECHO: R1, R3, R4, R5, R8, R9, R14, R17 or R18. NARRATE: R6 or R16. DECOR: R2 or R11. KEEP:
  the K rule that decided it (K1 to K14), or null when the deletion test did. UNSURE and null class: null.
- conf: "clear" when the steps decide the node at once, "judgement" when a step was arguable (§4).
- partial: true only on a KEEP that restates and also adds a clause that would be KEEP on its own (§4 step 5).
- ct: true whenever the consequence test (§7) decided the label, in either direction.
- equiv: for an ECHO in a stack of three or more nodes naming the same thing (§5), the i of the other flagged copies
  (not the target); otherwise [].
- aside: node-level codes from §9 for the rule layer: R7, R8b, R10, R11g, R12, R13, R15, X-FILE, UNRESOLVED.
- role_ok: false when the extractor's role is wrong for what the reader sees; then role_fix is the right role from
  this closed list, else role_fix is null: {roles}.
- why: at most 15 words naming the deciding step and rule.
Per screen:
- missed: visible copy in the source file that no node holds (not aria labels, alt text or code), with its line
  number, its text and the class it would get. [] when there is none.
- aside_screen: screen-level asides with at least two nodes, such as R19 (a repeated device, §9). [] when none.
"""

ADJ_PREAMBLE = """\
You are adjudicating labels for a research answer key. The labelling guide below (LABELLING.md) is the whole
definition of the task. Two labellers, A and B, labelled the same screen independently with this guide and disagree
on some nodes. Each message after this one gives you one screen (the product's kind, its nodes and its source file
with line numbers), the labels A and B agree on, and for each disputed node both labels with their one-line reasons.

For each disputed node, decide the label the guide gives it:
- Take the node through the guide's §4 decision procedure in order, with the §5 precedence, the §6 K rules, the §7
  consequence test and the §8 worked cases as precedent. Judge on the guide and the evidence alone (§2a).
- A and B are anonymous and assigned afresh for each node, so A on one node need not be A on another. Neither is
  presumed right. You may give A's label, B's label or another label.
- Do not guess what an automatic detector, a linter or a rule would say.
- UNSURE may be used when you cannot decide after §4 (§3).
- Do not use tools, files, the web or anything outside this conversation.

The nodes are JSON objects as in the labelling task: i, w (the window: a target must have the same w), k, cluster,
role, flags, text (⟨words⟩ is an interpolated value, " ‖ " separates conditional branches, a doubled ⟨⟨, ⟩⟩ or ‖‖ is
literal copy) and loc. Reply with exactly one JSON object and nothing else:

{"screen": "<screen id>",
 "nodes": [{"i": 4, "class": "NARRATE", "target": 6, "pattern": "R6", "conf": "judgement", "partial": false,
            "ct": true, "equiv": [], "aside": [], "reason": "step 4: says what the visible Save button does"}]}

One entry for every disputed node and no others. The fields mean what they mean in the labelling task: class (KEEP,
ECHO, NARRATE, DECOR or UNSURE), target (the i it repeats or explains; null for KEEP, DECOR and UNSURE), pattern
(the §9 code, a K rule or null on KEEP, null on UNSURE), conf, partial, ct, equiv and aside. reason: one line, at
most 25 words, naming the deciding step and rule.
"""


def preamble() -> str:
    return PREAMBLE.replace("{roles}", " ".join(c.lk.ROLES))


@dataclass
class Prompt:
    version: str
    prefix: str
    body: str

    @property
    def full(self) -> str:
        return self.prefix + "\n\n" + self.body

    @property
    def prefix_sha256(self) -> str:
        return c.sha256_text(self.prefix)

    @property
    def sha256(self) -> str:
        return c.sha256_text(self.full)


def prefix(guide_text: str, adjudicate: bool = False) -> str:
    head = ADJ_PREAMBLE if adjudicate else preamble()
    return f"{head}\n<labelling_guide file=\"LABELLING.md\">\n{guide_text.rstrip()}\n</labelling_guide>\n"


def node_line(node: dict, clusters: dict, windows: dict | None = None) -> str:
    obj = {"i": node["i"], "w": (windows or {}).get(node["i"], 0), "k": node.get("k"),
           "cluster": clusters.get(node.get("k")), "role": node.get("role"), "flags": node.get("flags") or [],
           "text": c.display_text(node), "loc": node.get("loc")}
    if c.lk.node_is_context(node):
        obj["context_only"] = True
    elif not c.has_literal_words(node):
        obj["no_literal_words"] = True
    return json.dumps(obj, ensure_ascii=False)


def numbered_source(text: str | None) -> str:
    if text is None:
        return "(source file not available)"
    lines = text.splitlines()
    w = len(str(len(lines)))
    return "\n".join(f"{n:>{w}} | {line}" for n, line in enumerate(lines, 1))


def fence_for(text: str | None) -> str:
    """A backtick fence longer than any backtick run in the source, so the source cannot close it."""
    runs = [len(m) for m in re.findall(r"`+", text or "")]
    return "`" * max(3, max(runs, default=0) + 1)


def screen_block(scr: c.Screen, source: bool = True) -> str:
    s = scr.screen
    src = s.get("source") or {}
    clusters = {cl.get("k"): cl.get("kind") for cl in s.get("clusters") or []}
    windows = scr.windows()
    nodes = "\n".join(node_line(n, clusters, windows) for n in s.get("nodes") or [])
    ext = {"tsx": "tsx", "jsx": "jsx", "html": "html", "swift": "swift", "py": "python"}.get(s.get("framework"), "")
    fence = fence_for(scr.source_text)
    nwin = len(set(windows.values())) or 1
    return (
        f"<screen id=\"{s['id']}\">\n"
        f"Product kind (§2a): {scr.kind or '(not recorded)'}\n"
        f"Repo: {src.get('repo')}  File: {src.get('path')}  Framework: {s.get('framework')}"
        f"{f'  Windows: {nwin}' if nwin > 1 else ''}\n\n"
        f"<nodes>\n{nodes}\n</nodes>\n\n"
        + (f"<source path=\"{src.get('path')}\">\n{fence}{ext}\n{numbered_source(scr.source_text)}\n{fence}\n"
           "</source>\n" if source else
           "<source>(not shown for this screen: label from the nodes, the product kind and the guide alone; the "
           "loc fields still give each node's position)</source>\n")
        + "</screen>"
    )


def render(scr: c.Screen, guide_text: str, source: bool = True) -> Prompt:
    """source=False is the no-source check on the key (run.py label --variant nosource), never the key itself."""
    body = screen_block(scr, source) + f"\n\nLabel screen {scr.id}. Reply with the JSON object only."
    return Prompt(PROMPT_VERSION, prefix(guide_text), body)


def label_view(lab: dict) -> dict:
    """The parts of a model label shown to an adjudicator (node indices, not locs). `conf` is left out: how sure a
    labeller says it is would weigh with the adjudicator and is also a recognisable style."""
    return {k: lab.get(k) for k in ("class", "target", "pattern", "partial", "ct", "equiv", "aside", "why")}


def render_adjudication(scr: c.Screen, guide_text: str, agreed: list, disputed: list) -> Prompt:
    """agreed: [(i, label)]; disputed: [(i, label_A, label_B)] with A and B already in this node's order (adjudicate
    .ab_order), labels as model-output dicts keyed by node i."""
    ag = "\n".join(json.dumps({"i": i, "class": lab.get("class"), "target": lab.get("target"),
                               "pattern": lab.get("pattern")}, ensure_ascii=False) for i, lab in agreed) or "(none)"
    dis = "\n".join(json.dumps({"i": i, "A": label_view(a), "B": label_view(b)}, ensure_ascii=False)
                    for i, a, b in disputed)
    body = (screen_block(scr)
            + f"\n\n<agreed>\n{ag}\n</agreed>\n\n<disputed>\n{dis}\n</disputed>\n\n"
            + f"Adjudicate the {len(disputed)} disputed node{'s' if len(disputed) != 1 else ''} of screen {scr.id}. "
              "Reply with the JSON object only.")
    return Prompt(ADJ_PROMPT_VERSION, prefix(guide_text, adjudicate=True), body)
