"""F2 (post Figure 4): "Every wire", the warm-up MLP Fable wrote, every non-zero parameter.

Run from the repo root:  uv run python scripts/blog/figures/f2.py

Reads the two committed artefacts (the revised network, v0.2 B, and the zero-label network, v0.2 A-selfcheck)
with the repo loader (agent_loop_lab.model.deserialize_state_dict), their CIRCUITS.md (unit names, circuits and
formulas) and BUILD_RECORD.json (sparsity counts, checked), and the committed v0.2 sealed report for the two
headline scores. The four example windows are invented here, one per class; the figure runs both networks on
them with plain numpy matrix arithmetic (the forward pass is three matrix products and two ReLUs) and checks
each prediction against the author's own reference_policy.py. No dataset, evaluator or scorer is opened.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from agent_loop_lab.model import deserialize_state_dict  # noqa: E402

OUT = ROOT / "blog_figures" / "F2.json"
GENERATED_BY = "scripts/blog/figures/f2.py"
SEALED = ROOT / "artifacts" / "reports" / "m4_sealed_v02.json"
CLASSES = ("PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING")
FEATURES = ("INS", "MUT", "VAL", "EXT", "CTL", "SUC", "ERR", "NEU", "RI", "RF", "WC", "SR", "VC", "VD", "TC")
FEATURE_NAMES = ("cat_INSPECT", "cat_MUTATE", "cat_VALIDATE", "cat_EXTERNAL", "cat_CONTROL", "out_SUCCESS",
                 "out_ERROR", "out_NEUTRAL", "repeated_intent", "repeated_failure", "workspace_changed",
                 "state_revisited", "validation_comparable", "validation_delta", "relative_token_cost")
NETWORKS = {
    "B": {"dir": ROOT / "weight_program" / "phase-b-revision", "commit": "0fc235d", "label": "Revised (v0.2 B)",
          "sealed_key": "B network"},
    "A": {"dir": ROOT / "weight_program" / "phase-a-selfcheck", "commit": "f8e0aae", "label": "Zero-label (v0.2 A)",
          "sealed_key": "A-selfcheck network"},
}
DP = 5


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def r(x: float) -> float:
    return round(float(x), DP)


def plain(md: str) -> str:
    """Markdown cell -> plain text (no backticks or bold)."""
    return md.replace("`", "").replace("**", "").strip()


# ---------------------------------------------------------------------------------------------------- windows --

def event(cat: str, out: str, *flags: str, vd: float = 0.0, tc: float = 0.4) -> list[float]:
    v = [0.0] * 15
    v[FEATURES.index(cat)] = 1.0
    v[FEATURES.index(out)] = 1.0
    for f in flags:
        v[FEATURES.index(f)] = 1.0
    if cat == "MUT" or "SR" in flags:
        v[FEATURES.index("WC")] = 1.0
    v[FEATURES.index("VD")] = vd
    if vd:
        v[FEATURES.index("VC")] = 1.0
    v[FEATURES.index("TC")] = tc
    return v


# Invented windows (oldest event first). Each is internally consistent with the feature invariants of spec v0.2 §4.
WINDOWS = [
    {
        "id": "steady",
        "name": "Steady progress",
        "expect": "PRODUCTIVE",
        "events": [
            ("read a file", event("INS", "SUC")),
            ("edit a file", event("MUT", "SUC")),
            ("run tests for the first time: 6 of 9 pass", event("VAL", "ERR")),
            ("read another file", event("INS", "SUC")),
            ("edit it", event("MUT", "SUC")),
            ("re-run the tests: 8 of 9 pass", event("VAL", "ERR", "VC", vd=0.222)),
            ("edit again", event("MUT", "SUC")),
            ("re-run: all 9 pass", event("VAL", "SUC", "VC", vd=0.111)),
        ],
    },
    {
        "id": "retry",
        "name": "A retry that worked",
        "expect": "RECOVERY",
        "events": [
            ("read a file", event("INS", "SUC")),
            ("edit it", event("MUT", "SUC")),
            ("run tests for the first time: all pass", event("VAL", "SUC")),
            ("call an API: error", event("EXT", "ERR")),
            ("same call, same error", event("EXT", "ERR", "RI", "RF")),
            ("same call again: success", event("EXT", "SUC", "RI")),
            ("edit a file", event("MUT", "SUC")),
            ("re-run the tests: still all pass", event("VAL", "SUC", "VC")),
        ],
    },
    {
        "id": "stuck",
        "name": "Same failure, five times",
        "expect": "STALLED",
        "events": [
            ("edit a file", event("MUT", "SUC")),
            ("re-run the tests: 7 of 9 pass, as before", event("VAL", "ERR", "VC")),
            ("same run, same failure", event("VAL", "ERR", "RI", "RF", "VC")),
            ("same run, same failure", event("VAL", "ERR", "RI", "RF", "VC")),
            ("same run, same failure", event("VAL", "ERR", "RI", "RF", "VC")),
            ("same run, same failure", event("VAL", "ERR", "RI", "RF", "VC")),
            ("read the same file again", event("INS", "SUC", "RI")),
            ("same run, same failure", event("VAL", "ERR", "RI", "RF", "VC")),
        ],
    },
    {
        "id": "undo",
        "name": "Undo, redo, undo",
        "expect": "OSCILLATING",
        "events": [
            ("edit a file", event("MUT", "SUC")),
            ("re-run the tests: 7 of 9 pass, as before", event("VAL", "ERR", "VC")),
            ("edit it another way", event("MUT", "SUC")),
            ("put the first version back", event("MUT", "SUC", "RI", "SR")),
            ("re-run: same failure", event("VAL", "ERR", "RI", "RF", "VC")),
            ("switch to the other version again", event("MUT", "SUC", "RI", "SR")),
            ("re-run: same failure", event("VAL", "ERR", "RI", "RF", "VC")),
            ("back to the first version", event("MUT", "SUC", "RI", "SR")),
        ],
    },
]


# ---------------------------------------------------------------------------------------------- circuits.md --

def parse_circuits(text: str) -> dict:
    """Layer-1 circuits (with units), layer-2 rows and layer-3 rows from a CIRCUITS.md."""
    circuits = []
    l1 = {}
    for m in re.finditer(r"^### Circuit (\w) — (.+?) \(units? ([^)]+)\)\n\n\| Unit \| Computes \| Meaning \|\n\|[-|]+\|\n((?:\|.*\|\n?)+)",
                         text, re.M):
        cid, name, _units, table = m.groups()
        units = []
        for row in table.strip().split("\n"):
            cells = [c.strip() for c in row.strip().strip("|").split("|")]
            spec, computes, meaning = cells[0], cells[1], "|".join(cells[2:])
            span = [int(x) for x in re.findall(r"\d+", spec)]
            members = list(range(span[0], span[1] + 1)) if ("–" in spec or "-" in spec) and len(span) == 2 else span
            for u in members:
                name_m = re.match(r"`?([A-Za-z0-9_]+)", computes)
                short = name_m.group(1) if name_m else f"u{u}"
                comp = plain(computes)
                if short.endswith("_s"):
                    if re.search(r"s = unit \+ 1", computes):
                        s = u + 1
                    elif re.search(r"unit `s` reads slot `s`", computes):
                        s = u
                    elif re.search(r"s = 4, 5, 6, 7", computes):
                        s = 4 + (u - members[0])
                    else:
                        raise SystemExit(f"F2: cannot work out the slot of unit {u} ({short})")
                    short = f"{short[:-2]}{s}"
                    comp = f"{comp}  [this unit: s = {s}]"
                l1[u] = {"unit": u, "circuit": cid, "name": short, "formula": comp,
                         "meaning": plain(meaning).split(". ")[0].rstrip(".") + "."}
            units.extend(members)
        circuits.append({"id": cid, "name": name.strip(), "units": units})

    def table_after(heading: str) -> list[list[str]]:
        i = text.index(heading)
        block = text[i:].split("\n\n")
        for chunk in block[1:]:
            if chunk.startswith("|"):
                rows = [ln for ln in chunk.split("\n") if ln.startswith("|")][2:]
                return [[c.strip() for c in row.strip().strip("|").split("|")] for row in rows]
        raise SystemExit(f"F2: no table after {heading}")

    l2 = {}
    for cells in table_after("## Layer 2 (fc2)"):
        u = int(cells[0])
        name_m = re.match(r"`?([A-Za-z0-9_]+)", cells[1])
        l2[u] = {"unit": u, "name": name_m.group(1), "formula": plain(cells[1]),
                 "meaning": plain("|".join(cells[2:])).split(". ")[0].rstrip(".") + "."}
    l3 = {}
    for cells in table_after("## Layer 3 (fc3)"):
        k = int(cells[0].split()[0])
        l3[k] = {"unit": k, "name": CLASSES[k], "formula": plain(cells[1]), "meaning": plain("|".join(cells[2:]))}
    return {"circuits": circuits, "l1": l1, "l2": l2, "l3": l3}


def load_policy(path: Path):
    spec = importlib.util.spec_from_file_location(f"policy_{path.parent.name.replace('-', '_')}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.reference_policy


def argmax_first(v: np.ndarray) -> int:
    best = 0
    for i in range(1, len(v)):
        if v[i] > v[best]:
            best = i
    return best


# ------------------------------------------------------------------------------------------------------ build --

def build() -> dict:
    sealed = json.loads(SEALED.read_text())
    windows_x = []
    for w in WINDOWS:
        x = [f for _, ev in w["events"] for f in ev]
        if len(x) != 120:
            raise SystemExit("F2: a window must have 8 events of 15 features")
        windows_x.append(np.array(x, dtype=np.float64))

    sources: list[Path] = [SEALED]
    nets = {}
    for key, cfg in NETWORKS.items():
        d = cfg["dir"]
        art, circ, rec, pol = d / "artifact.dlw", d / "CIRCUITS.md", d / "BUILD_RECORD.json", d / "reference_policy.py"
        sources += [art, circ, rec, pol]
        sd, _meta = deserialize_state_dict(art.read_bytes())
        W = {k: v.numpy().astype(np.float64) for k, v in sd.items()}
        record = json.loads(rec.read_text())

        edges, biases = [], []
        for layer, name in ((1, "fc1"), (2, "fc2"), (3, "fc3")):
            w = W[f"{name}.weight"]
            for dst, src in zip(*np.nonzero(w)):
                edges.append([layer, int(src), int(dst), r(w[dst, src])])
            b = W[f"{name}.bias"]
            for u in np.nonzero(b)[0]:
                biases.append([layer, int(u), r(b[u])])
            for part in ("weight", "bias"):
                nz = int(np.count_nonzero(W[f"{name}.{part}"]))
                if nz != record["sparsity"][f"{name}.{part}"]["nonzero"]:
                    raise SystemExit(f"F2: {key} {name}.{part} has {nz} non-zero, BUILD_RECORD says otherwise")
        total = len(edges) + len(biases)
        if total != record["sparsity"]["overall"]["nonzero"]:
            raise SystemExit(f"F2: {key} non-zero count mismatch")
        mags = [abs(e[3]) for e in edges]

        meta = parse_circuits(circ.read_text())
        l1_used = sorted({e[2] for e in edges if e[0] == 1} | {e[1] for e in edges if e[0] == 2})
        missing = [u for u in l1_used if u not in meta["l1"]]
        if missing:
            raise SystemExit(f"F2: {key} layer-1 units {missing} have no CIRCUITS.md row")

        policy = load_policy(pol)
        runs = []
        for w, x in zip(WINDOWS, windows_x):
            h1 = np.maximum(0.0, W["fc1.weight"] @ x + W["fc1.bias"])
            h2 = np.maximum(0.0, W["fc2.weight"] @ h1 + W["fc2.bias"])
            logits = W["fc3.weight"] @ h2 + W["fc3.bias"]
            pred = argmax_first(logits)
            ref = policy(tuple(float(v) for v in x))
            if pred != ref:
                raise SystemExit(f"F2: {key} network ({CLASSES[pred]}) and its reference policy ({CLASSES[ref]}) "
                                 f"disagree on window {w['id']}")
            runs.append({"window": w["id"], "h1": [r(v) for v in h1], "h2": [r(v) for v in h2],
                         "logits": [r(v) for v in logits], "pred": CLASSES[pred]})

        scores = {}
        for split in ("test", "ood"):
            m = sealed["splits"][split]["scores"]["overall"]["columns"][cfg["sealed_key"]]["macro_f1"]
            scores[split] = {"v": r(m["value"]), "ci": [r(m["ci95"][0]), r(m["ci95"][1])]}

        nets[key] = {
            "key": key,
            "label": cfg["label"],
            "commit": cfg["commit"],
            "dir": rel(d),
            "slots": record["sparsity"]["overall"]["slots"],
            "nonzero": total,
            "nonzero_weights": len(edges),
            "nonzero_biases": len(biases),
            "abs_w_min": r(min(mags)),
            "abs_w_max": r(max(mags)),
            "edges": edges,
            "biases": biases,
            "circuits": meta["circuits"],
            "l1": [meta["l1"].get(u, {"unit": u, "circuit": None, "name": f"u{u}", "formula": "all zero (unused)",
                                      "meaning": "Unused."}) for u in range(24)],
            "l2": [meta["l2"].get(u, {"unit": u, "name": f"u{u}", "formula": "all zero (unused)", "meaning": "Unused."})
                   for u in range(12)],
            "out": [meta["l3"][k] for k in range(4)],
            "runs": runs,
            "sealed_macro_f1": scores,
        }

    for w, run_b in zip(WINDOWS, nets["B"]["runs"]):
        if run_b["pred"] != w["expect"]:
            raise SystemExit(f"F2: window {w['id']} was built as {w['expect']} but B predicts {run_b['pred']}")

    return {
        "_provenance": {
            "source": [rel(p) for p in sources],
            "sha256": {rel(p): sha256(p) for p in sources},
            "source_commits": {k: v["commit"] for k, v in NETWORKS.items()},
            "git_commit": git_head(),
            "generated_by": GENERATED_BY,
            "note": "windows are invented; activations are a numpy forward pass checked against each reference_policy.py",
        },
        "figure": "F2",
        "title": "Every wire",
        "features": list(FEATURES),
        "feature_names": list(FEATURE_NAMES),
        "classes": list(CLASSES),
        "windows": [{"id": w["id"], "name": w["name"], "events": [e for e, _ in w["events"]],
                     "x": [r(v) for v in x]} for w, x in zip(WINDOWS, windows_x)],
        "networks": nets,
        "log10_range": [math.log10(1 / 32), math.log10(10_000)],
    }


if __name__ == "__main__":
    data = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")
    for k, n in data["networks"].items():
        print(k, n["nonzero"], "of", n["slots"], "|w|", n["abs_w_min"], "..", n["abs_w_max"],
              [(rn["window"], rn["pred"], rn["logits"]) for rn in n["runs"]])
