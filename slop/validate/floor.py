"""The exact-stem floor (SPEC.md §11.4) in plain Python, on the tokeniser's windows.

Within one window, two nodes are a pair when they share a cluster index, both take part (role not ``aria``, ``alt``,
``code`` or ``comment``, no ``sr_only`` flag, at least one word token) and their stem sets are equal or one contains
the other. A node's stem set is the set of its word token ids (ids 512 and up: vocabulary rows and hash buckets), so
two stems that share a hash bucket count as equal, exactly as the network sees them. The lower-precedence node of a
pair is flagged ``ECHO`` and points at the other. Precedence follows ``LABELLING.md`` §5 (:data:`PRECEDENCE`, higher
wins); a ``label`` with the ``in_control`` flag counts as the label of an input (rank 1); at equal rank the earlier
node wins. A node in several pairs points at the partner that ranks highest under the same order.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from ui import tokenizer as tk
except ImportError:  # pragma: no cover
    import tokenizer as tk

# LABELLING.md §5: rank 1 = 8 (controls and main headings), h2..h6 = 7..3, rank 3 = 2, rank 4 = 1, rank 5 = 0.
PRECEDENCE: dict[str, int] = {
    "button": 8, "link": 8, "nav": 8, "tab": 8, "menu_item": 8, "breadcrumb": 8,
    "h1": 8, "dialog_title": 8, "empty_title": 8,
    "h2": 7, "h3": 6, "h4": 5, "h5": 4, "h6": 3,
    "body": 2, "list_item": 2, "dialog_body": 2, "empty_body": 2, "toast": 2, "error": 2, "stat": 2, "cell": 2,
    "input_value": 2, "label": 2, "other": 2,
    "subtitle": 1, "helper": 1, "caption": 1,
    "eyebrow": 0, "badge": 0, "tooltip": 0, "placeholder": 0,
    "aria": 0, "alt": 0, "code": 0, "comment": 0,
}
LABEL_OF_INPUT = 8
assert set(PRECEDENCE) == set(tk.ROLES)


def precedence(role: str, flags: list[str] | set[str]) -> int:
    if role == "label" and "in_control" in flags:
        return LABEL_OF_INPUT
    return PRECEDENCE[role]


def stem_set(node: tk.NodeEnc) -> frozenset[int] | None:
    """Word token ids of a participating node, or None."""
    if node.role in tk.CONTEXT_ROLES or "sr_only" in node.flags:
        return None
    s = frozenset(t for t in node.text.ids if t >= tk.VOCAB_BASE)
    return s or None


@dataclass
class FloorCall:
    index: int  # screen node list index
    cls: str  # KEEP or ECHO
    pointer: int | None  # screen node list index of the target


def floor_window(nodes: list[tk.NodeEnc], idx: list[int], max_distance: int | None = None) -> dict[int, FloorCall]:
    """Floor calls for the nodes of one window. ``max_distance`` limits pairs to that many nodes apart."""
    sets = [stem_set(nodes[x]) for x in idx]
    prio = [(precedence(nodes[x].role, nodes[x].flags), -m) for m, x in enumerate(idx)]
    out: dict[int, FloorCall] = {}
    for a, xa in enumerate(idx):
        best = None
        if sets[a] is not None:
            for b, xb in enumerate(idx):
                if b == a or sets[b] is None or nodes[xb].k != nodes[xa].k:
                    continue
                if max_distance is not None and abs(b - a) > max_distance:
                    continue
                if not (sets[a] <= sets[b] or sets[b] <= sets[a]):
                    continue
                if prio[b] > prio[a] and (best is None or prio[b] > prio[best]):
                    best = b
        out[xa] = FloorCall(xa, "ECHO", idx[best]) if best is not None else FloorCall(xa, "KEEP", None)
    return out


def floor_screen(enc: tk.EncodedScreen, max_distance: int | None = None) -> list[FloorCall]:
    calls: dict[int, FloorCall] = {}
    for w in enc.windows:
        calls.update(floor_window(enc.nodes, w.nodes, max_distance))
    return [calls[x] for x in range(len(enc.nodes))]
