"""Seeded random screens for stress-testing the floor network and the JS runtime (harness code, not data).

Words come from a short list so that equal and nested stem sets are common; roles, flags (``sr_only``,
``in_control``), cluster sizes (1-14 nodes) and screen sizes (1-90 nodes) vary, so long clusters and window splits
occur too.
"""

from __future__ import annotations

import random

try:
    from ui import tokenizer as tk
except ImportError:  # pragma: no cover
    import tokenizer as tk

WORDS = ["save", "saved", "settings", "account", "delete", "project", "projects", "team", "new", "invite", "members",
         "billing", "plan", "search", "results", "open", "close", "share", "file", "files", "upload", "export",
         "report", "weekly", "email", "notifications", "\ue000", "\ue001", "userName", "createdAt"]
EXTRAS = ["…", "→", "●", "✓", "3", "2026", "01", "1,000", "!", "?", ":", "—", "✨", "👍"]


def random_screen(rng: random.Random, sid: str) -> dict:
    n_nodes = rng.choice([1, 2, 3, 5, 8, 12, 20, 30, 45, 66, 90])
    nodes = []
    k = 0
    left = 0
    for i in range(n_nodes):
        if left == 0:
            k += 1 if nodes else 0
            left = rng.choice([1, 2, 3, 4, 6, 9, 14])
        left -= 1
        n_words = rng.choice([0, 1, 1, 2, 2, 3, 4, 6, 40]) if rng.random() > 0.05 else 0
        parts = [rng.choice(WORDS) for _ in range(n_words)]
        if rng.random() < 0.3:
            parts.insert(rng.randrange(len(parts) + 1), rng.choice(EXTRAS))
        text = " ".join(parts)
        if rng.random() < 0.2:
            text = text.upper()
        elif rng.random() < 0.2:
            text = text.title()
        flags = []
        if rng.random() < 0.05:
            flags.append("sr_only")
        if rng.random() < 0.3:
            flags.append("in_control")
        if rng.random() < 0.2:
            flags.append(rng.choice(tk.FLAGS))
        nodes.append({"i": i, "k": k, "role": rng.choice(tk.ROLES), "text": text, "flags": sorted(set(flags)),
                      "loc": [i + 1, 1, i + 1, 1 + len(text)]})
    clusters = [{"k": c, "kind": rng.choice(tk.IDS["cluster_kinds"]), "line": 1} for c in range(k + 1)]
    return {"schema": "slop.screen/1", "id": sid, "framework": "tsx",
            "source": {"repo": "random", "path": f"{sid}.tsx", "commit": "0" * 40, "licence": "MIT"},
            "clusters": clusters, "nodes": nodes}


def random_screens(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    return [random_screen(rng, f"random-{seed}-{j}") for j in range(n)]
