"""Export compact JSON data for every HypaNet blog post figure.

Run from the repo root:  uv run python scripts/blog/export_figures.py

F6, F12 and F14 are built here. Every other figure has its own module in scripts/blog/figures/fNN.py exposing
build() -> dict; this script imports each one and writes its result. F7 and F13 come from their modules, which
reuse the f7() and f13() builders below unchanged (F7 adds the early-stopping inset).

Reads only committed reports, committed v0.4 val-quick readout logits (label_logits, label-free) and the
committed WSP-A / WSP-B author notes. Never opens test / OOD / counterfactual datasets or fresh_test.
Writes blog_figures/F*.json; each file carries a "_provenance" object.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "blog_figures"
REPORTS = ROOT / "artifacts" / "reports"
GENERATED_BY = "scripts/blog/export_figures.py"
CLASSES = ("PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING")
DP = 5  # decimal places kept for every exported metric

V03 = REPORTS / "v03_sealed.json"
V02 = REPORTS / "m4_sealed_v02.json"
Q0_FULL = REPORTS / "v04_val_qwen0.json"
Q0_QUICK = REPORTS / "v04_val_quick_qwen0.json"
WSPA_QUICK = REPORTS / "v04_val_quick_wspa.json"
WSPB_QUICK = REPORTS / "v04_val_quick_wspb.json"
COLS = ROOT / "experiments" / "v0_4" / "columns"
NPZ = {"Qwen-0": COLS / "qwen0.val_quick.npz", "WSP-A": COLS / "wspa.val_quick.npz", "WSP-B": COLS / "wspb.val_quick.npz"}
ITER_A = ROOT / "weight_program" / "v0_4" / "wsp-a" / "inputs" / "ITERATIONS.md"
REV_B = ROOT / "weight_program" / "v0_4" / "wsp-b" / "inputs" / "REVISION.md"

FIGURES_DIR = ROOT / "scripts" / "blog" / "figures"
# One module per figure, in post order; each exposes build() -> dict and an OUT path.
FIGURE_MODULES = ("f0", "f28", "f1", "f3", "f2", "f4c", "f4b", "f5", "f8", "f4", "f7", "f18", "f9", "f26", "f10", "f11",
                  "f27", "f13", "f23", "f24", "f25", "f19", "f19b", "f21", "f22", "f15")

QUICK_NOTE = ("quick look = 1,000 val sessions (VAL_QUICK.txt), excluding_disclosed; reported beside, never instead "
              "of, the full-val result (PREREGISTRATION §3.5)")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def r(x):
    return None if x is None else round(float(x), DP)


def ci(pair):
    return None if pair is None else [r(pair[0]), r(pair[1])]


def load(p: Path):
    return json.loads(p.read_text())


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def provenance(sources: list[Path], extra: dict | None = None) -> dict:
    out = {
        "source": [str(p.relative_to(ROOT)) for p in sources],
        "sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "git_commit": git_head(),
        "generated_by": GENERATED_BY,
    }
    if extra:
        out.update(extra)
    return out


def write(name: str, obj: dict) -> None:
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def vci(pair):
    """summary-style [value, [lo, hi]] -> {"v", "ci"}"""
    return {"v": r(pair[0]), "ci": ci(pair[1])}


def mean_block(m):
    """summary-style mean±sd block -> {"mean", "sd", "ci", "seeds"}"""
    return {"mean": r(m["mean"]), "sd": r(m["sd"]), "ci": ci(m["seed_mean_ci95"]), "seeds": [r(x) for x in m["per_seed"]]}


def split_size(rep: dict, split: str) -> dict:
    pop = rep["splits"][split]["populations"]["all"]["population"]
    return {"sessions": pop["sessions"], "scored_windows": pop["scored_windows"]}


# ---------------------------------------------------------------------------
# F6  Off the map
# ---------------------------------------------------------------------------

C_BUDGETS = ("32", "128", "512", "2048", "all")


def f6(v03: dict, v02: dict) -> dict:
    s = v03["summary"]
    groups = v03["phase_c_groups"]

    def single(key):
        return {sp: {"macro_f1": vci(s[sp]["all"][key]["macro_f1"]),
                     "false_alarm": vci(s[sp]["all"][key]["false_alarm_rate"])} for sp in ("test", "ood")}

    rows = [
        {"id": "prior", "label": "Prior (always PRODUCTIVE)", "key": "prior dev-majority", "labels": 0, **single("prior dev-majority")},
        {"id": "A", "label": "v0.3 A (hand-written, no labels)", "key": "v0.3 A network", "labels": 0, **single("v0.3 A network")},
        {"id": "B", "label": "v0.3 B (hand-written, revised on 200 labels)", "key": "v0.3 B network", "labels": 200, **single("v0.3 B network")},
    ]
    for b in C_BUDGETS:
        g = f"v0.3 C-{b}"
        seeds = groups[g]
        row = {"id": f"C-{b}", "label": f"v0.3 C-{b} (trained)", "key": g, "n_seeds": len(seeds),
               "labelled_examples": v03["columns"][seeds[0]]["labelled_examples"]}
        for sp in ("test", "ood"):
            m = s[sp]["all"][f"{g} mean±sd"]
            row[sp] = {
                "seeds": [{"seed": k.rsplit("seed", 1)[1], "key": k,
                           "macro_f1": vci(s[sp]["all"][k]["macro_f1"]),
                           "false_alarm": vci(s[sp]["all"][k]["false_alarm_rate"])} for k in seeds],
                "mean": {"macro_f1": mean_block(m["macro_f1"]), "false_alarm": mean_block(m["false_alarm_rate"])}
                if len(seeds) == 3 else None,
            }
        rows.append(row)

    s2 = v02["summary"]

    def single2(key):
        return {sp: {"macro_f1": vci(s2[sp][key]["macro_f1"]), "false_alarm": vci(s2[sp][key]["false_alarm_rate"])}
                for sp in ("test", "ood")}

    ghosts = [
        {"id": "prior", "label": "v0.2 prior", "key": "prior dev-majority", **single2("prior dev-majority")},
        {"id": "rules", "label": "v0.2 rules", "key": "rules", **single2("rules")},
        {"id": "A", "label": "v0.2 A", "key": "A-strict network", **single2("A-strict network")},
        {"id": "B", "label": "v0.2 B", "key": "B network", **single2("B network")},
    ]
    for b in C_BUDGETS:
        key = f"C-{b} mean±sd"
        ghosts.append({"id": f"C-{b}", "label": f"v0.2 C-{b} (mean of 3 seeds)", "key": key,
                       **{sp: {"macro_f1": mean_block_nos(s2[sp][key]["macro_f1"]),
                               "false_alarm": mean_block_nos(s2[sp][key]["false_alarm_rate"])} for sp in ("test", "ood")}})

    return {
        "_provenance": provenance([V03, V02]),
        "figure": "F6", "title": "Off the map",
        "metrics": "macro-F1 and false-alarm rate (PRODUCTIVE/RECOVERY windows predicted STALLED/OSCILLATING), "
                   "session-bootstrap 95% CI; C means: mean, sd and seed-mean CI over seeds",
        "splits": {"test": {"name": "v0.3 sealed test", **split_size(v03, "test")},
                   "ood": {"name": "v0.3 sealed OOD", **split_size(v03, "ood")}},
        "notes": [
            "test and OOD are sealed splits, scored once (v03_sealed.json, official=true)",
            "C-2048 and C-all have one seed in v0.3; their mean is null and the single seed is the point",
            "prior false-alarm rate is exactly 0 (it never predicts STALLED/OSCILLATING): cannot sit on a log axis",
            "v0.2 ghosts are a different (easier) dataset; v0.2 A-strict and A-selfcheck are identical, A-strict shown",
        ],
        "v03": rows,
        "v02_ghosts": ghosts,
    }


def mean_block_nos(m):
    return {"mean": r(m["mean"]), "sd": r(m["sd"]), "ci": ci(m["seed_mean_ci95"])}


# ---------------------------------------------------------------------------
# F7  Labels vs F1
# ---------------------------------------------------------------------------

def f7(v03: dict, v02: dict) -> dict:
    s = v03["summary"]
    cols = v03["columns"]
    seeds_out, means = [], []
    for b in C_BUDGETS:
        g = f"v0.3 C-{b}"
        ks = v03["phase_c_groups"][g]
        for k in ks:
            c = cols[k]
            seeds_out.append({
                "key": k, "budget": b, "seed": k.rsplit("seed", 1)[1],
                "labelled_examples": c["labelled_examples"], "budget_sessions": c["budget_sessions"],
                "pool_sessions": c["pool_sessions"],
                "test": r(s["test"]["all"][k]["macro_f1"][0]), "ood": r(s["ood"]["all"][k]["macro_f1"][0]),
            })
        means.append({
            "key": g, "budget": b, "n_seeds": len(ks), "labelled_examples": cols[ks[0]]["labelled_examples"],
            "pool_sessions": cols[ks[0]]["pool_sessions"],
            "test": r(s["test"]["all"][f"{g} mean±sd"]["macro_f1"]["mean"]),
            "ood": r(s["ood"]["all"][f"{g} mean±sd"]["macro_f1"]["mean"]),
        })
    points = []
    for pid, key in (("A", "v0.3 A network"), ("B", "v0.3 B network")):
        points.append({"id": pid, "key": key, "labelled_examples": cols[key]["labelled_examples"],
                       "test": vci(s["test"]["all"][key]["macro_f1"]), "ood": vci(s["ood"]["all"][key]["macro_f1"])})
    s2 = v02["summary"]
    ghosts = []
    for b in C_BUDGETS:
        k0 = f"C-{b} seed0"
        ghosts.append({"key": f"C-{b} mean±sd", "budget": b, "labelled_examples": v02["columns"][k0]["labelled_examples"],
                       "test": r(s2["test"][f"C-{b} mean±sd"]["macro_f1"]["mean"]),
                       "ood": r(s2["ood"][f"C-{b} mean±sd"]["macro_f1"]["mean"])})
    return {
        "_provenance": provenance([V03, V02]),
        "figure": "F7", "title": "Labels vs F1",
        "x": "labelled_examples = labelled training windows drawn from the budget sessions (log scale)",
        "y": "macro-F1 on the v0.3 sealed test / OOD split",
        "splits": {"test": split_size(v03, "test"), "ood": split_size(v03, "ood")},
        "notes": [
            "budget_sessions is the nominal session budget; pool_sessions is the sessions actually available "
            "(C-all: budget 12,000, pool 11,864, 223,930 windows)",
            "A has 0 labelled examples; B was revised after seeing 200 disclosed labelled val examples",
            "v0.2 ghosts: different dataset, mean of 3 seeds per budget",
        ],
        "c_seeds": seeds_out,
        "c_means": means,
        "hand_written": points,
        "v02_ghost_means": ghosts,
    }


# ---------------------------------------------------------------------------
# F12  Scoreboard
# ---------------------------------------------------------------------------

def col_entry(col: dict, population: str, sessions: int, source: str, key: str) -> dict:
    return {
        "key": key, "population": population, "source": source, "sessions": sessions,
        "scored_windows": col["scored_windows"],
        "macro_f1": {"v": r(col["macro_f1"]["value"]), "ci": ci(col["macro_f1"]["ci95"])},
        "false_alarm": {"v": r(col["false_alarm_rate"]["value"]), "ci": ci(col["false_alarm_rate"]["ci95"])},
        "accuracy": {"v": r(col["accuracy"]["value"]), "ci": ci(col["accuracy"]["ci95"])},
        "per_class_f1": {c: {"v": r(col["per_class"][c]["f1"]["value"]), "ci": ci(col["per_class"][c]["f1"]["ci95"])}
                         for c in CLASSES},
    }


def f12(v03, q0f, q0q, wa, wb) -> dict:
    def ed(rep):
        return rep["excluding_disclosed"]["overall"]

    rows = []
    src = lambda p: str(p.relative_to(ROOT))  # noqa: E731
    rows.append({"id": "prior", "label": "Prior (always PRODUCTIVE)",
                 **col_entry(ed(q0f)["columns"]["prior dev-majority"], "full val", ed(q0f)["sessions"], src(Q0_FULL), "prior dev-majority")})
    rows.append({"id": "prior-quick", "label": "Prior (always PRODUCTIVE)",
                 **col_entry(ed(q0q)["columns"]["prior dev-majority"], "quick look", ed(q0q)["sessions"], src(Q0_QUICK), "prior dev-majority")})
    rows.append({"id": "qwen0", "label": "Qwen-0 (base Qwen3.5-0.8B)",
                 **col_entry(ed(q0f)["columns"]["Qwen-0"], "full val", ed(q0f)["sessions"], src(Q0_FULL), "Qwen-0")})
    rows.append({"id": "qwen0-quick", "label": "Qwen-0 (base Qwen3.5-0.8B)",
                 **col_entry(ed(q0q)["columns"]["Qwen-0"], "quick look", ed(q0q)["sessions"], src(Q0_QUICK), "Qwen-0")})
    rows.append({"id": "wspa-quick", "label": "v0.4 WSP-A (Qwen edited by Fable, no labels)",
                 **col_entry(ed(wa)["columns"]["WSP-A"], "quick look", ed(wa)["sessions"], src(WSPA_QUICK), "WSP-A")})
    rows.append({"id": "wspb-quick", "label": "v0.4 WSP-B (revised on 200 disclosed windows)",
                 **col_entry(ed(wb)["columns"]["WSP-B"], "quick look", ed(wb)["sessions"], src(WSPB_QUICK), "WSP-B")})
    v = v03["splits"]["val"]["populations"]["excluding_disclosed"]["overall"]
    for rid, key, label in (("v03-A", "v0.3 A network", "v0.3 A (from scratch, no labels)"),
                            ("v03-B", "v0.3 B network", "v0.3 B (from scratch, 200 labels)"),
                            ("v03-Call", "v0.3 C-all seed0", "v0.3 C-all (trained, 223,930 windows)")):
        rows.append({"id": rid, "label": label, **col_entry(v["columns"][key], "full val", v["sessions"], src(V03), key)})

    def paired(rep, name, population):
        p = rep["excluding_disclosed"]["paired_macro_f1"][name]
        return {"name": name, "population": population, "v": r(p["value"]), "ci": ci(p["ci95"]),
                "ci95_above_0": p["ci95_above_0"]}

    pairs = [paired(wb, "WSP-A - Qwen-0", "quick look"), paired(wb, "WSP-B - Qwen-0", "quick look"),
             paired(wb, "WSP-B - WSP-A", "quick look"), paired(wb, "Qwen-0 - prior dev-majority", "quick look"),
             paired(q0f, "Qwen-0 - prior dev-majority", "full val")]
    return {
        "_provenance": provenance([Q0_FULL, Q0_QUICK, WSPA_QUICK, WSPB_QUICK, V03]),
        "figure": "F12", "title": "The scoreboard",
        "population_defs": {
            "full val": "all val sessions excluding the 200 disclosed sessions (3,804 sessions, 69,974 scored windows); "
                        "same population for v0.3 and v0.4",
            "quick look": QUICK_NOTE + " (18,222 scored windows)",
        },
        "notes": [
            "WSP-A and WSP-B numbers are val QUICK-LOOK numbers (1,000 sessions); their full-val runs are pending",
            "v0.3 C-all val is optimistic: Phase C early-stopped on 500 val sessions",
            "paired diffs: macro-F1(minuend) - macro-F1(subtrahend) over shared session-bootstrap draws",
            "per_class_f1 in ClassLabel order PRODUCTIVE, RECOVERY, STALLED, OSCILLATING",
        ],
        "rows": rows,
        "paired_macro_f1": pairs,
    }


# ---------------------------------------------------------------------------
# F13  More certain, not more right
# ---------------------------------------------------------------------------

EDGES = [0.0] + [10 ** (e / 4) for e in range(-12, 9)] + [float("inf")]  # 0, 1e-3 .. 1e2 (4/decade), inf


def f13(q0q, wa, wb) -> dict:
    reports = {"Qwen-0": (q0q, Q0_QUICK), "WSP-A": (wa, WSPA_QUICK), "WSP-B": (wb, WSPB_QUICK)}
    models = []
    for name, path in NPZ.items():
        with np.load(path, allow_pickle=False) as z:
            lg = z["label_logits"].astype(np.float64)  # label-free; no labels are loaded
        top2 = np.sort(lg, axis=1)[:, -2:]
        margin = top2[:, 1] - top2[:, 0]
        counts, _ = np.histogram(margin, bins=np.array(EDGES))
        rep, rpath = reports[name]
        diag = rep["readout_diagnostics"][name]["top2_margin"]
        mine = {f"p{p}": r(np.percentile(margin, p)) for p in (5, 25, 50, 75, 95)}
        for k, v in mine.items():  # cross-check against the committed report
            assert abs(v - diag[k]) < 1e-4, (name, k, v, diag[k])
        acc = rep["excluding_disclosed"]["overall"]["columns"][name]["accuracy"]
        models.append({
            "name": name, "population": "quick look", "windows": int(margin.size),
            "counts": counts.tolist(), "median": r(np.median(margin)), "percentiles": mine,
            "margin_below_1e-3": int((margin < 1e-3).sum()),
            "accuracy": {"v": r(acc["value"]), "ci": ci(acc["ci95"]), "population": "quick look",
                         "scored_windows": rep["excluding_disclosed"]["overall"]["columns"][name]["scored_windows"],
                         "source": str(rpath.relative_to(ROOT))},
        })
    return {
        "_provenance": provenance([*NPZ.values(), Q0_QUICK, WSPA_QUICK, WSPB_QUICK]),
        "figure": "F13", "title": "More certain, not more right",
        "margin": "top-2 margin = largest minus second-largest of the four label-word logits (nats); label-free",
        "bin_edges": [0.0] + [r(e) for e in EDGES[1:-1]] + ["inf"],
        "bins": "counts[i] covers [bin_edges[i], bin_edges[i+1]); first bin is [0, 0.001), last is [100, inf); "
                "interior edges are log-spaced, 4 per decade",
        "notes": [
            QUICK_NOTE,
            "margins: every window in the 1,000 quick-look sessions (21,308 rows, incl. gate-rejected and ambiguous); "
            "accuracy: scored windows only (18,222), from the committed reports",
            "percentiles were recomputed here and checked against readout_diagnostics in each report",
        ],
        "models": models,
    }


# ---------------------------------------------------------------------------
# F14  Synthetic vs real
# ---------------------------------------------------------------------------

def bullet(path: Path, needle: str) -> dict:
    """The line containing ``needle`` plus its wrapped continuation lines, verbatim."""
    lines = path.read_text().splitlines()
    hits = [i for i, ln in enumerate(lines) if needle in ln]
    assert len(hits) == 1, (path, needle, hits)
    i = hits[0]
    out = [lines[i]]
    j = i + 1
    while j < len(lines) and lines[j].startswith("  ") and not lines[j].lstrip().startswith("- "):
        out.append(lines[j])
        j += 1
    return {"file": str(path.relative_to(ROOT)), "line": i + 1, "verbatim": out}


def est(value, original, text_set, edit_state, kind, path, needle, check):
    src = bullet(path, needle)
    joined = " ".join(src["verbatim"])
    assert check in joined, (needle, check)
    return {"accuracy": value, "original_accuracy": original, "text_set": text_set, "edit_state": edit_state,
            "kind": kind, **src}


def f14(q0f, q0q, wa, wb) -> dict:
    A, B = ITER_A, REV_B
    syn, real_disc = "synthetic (Fable's own texts)", "real disclosed val windows (200, 50 per class)"
    wspa = [
        est(0.40, None, "set B (90 sessions)", "original 938f4bc9", syn, A, "**fwd calls 2-4**", "baseline acc 0.40"),
        est(0.49, 0.40, "set E (60 sessions)", "v1c 91863b69", syn, A, "**fwd calls 12-14**", "acc 0.49"),
        est(0.59, None, "set E", "v2b 551152d6", syn, A, "**fwd calls 15-17**", "acc 0.59"),
        est(0.62, None, "set E", "v2c 3ef7a5ee", syn, A, "**fwd calls 18-20**", "acc 0.62"),
        est(0.57, 0.38, "set F (54 sessions)", "v2c 3ef7a5ee", syn, A, "**fwd calls 21-23**", "acc 0.57 vs"),
        est(0.56, 0.38, "set F (54 sessions)", "v3b 74576466 (final, = WSP-A)", syn, A, "**fwd calls 26-28**",
            "acc 0.56 vs 0.38 original"),
        est(0.66, None, "sets B+D, predicted (direct path, no forward pass)", "v3b 74576466 (final)", "prediction",
            A, "Predicted acc on B+D: 0.66", "0.66"),
    ]
    wspb = [
        est(0.315, 0.240, real_disc, "WSP-A 74576466 (before revision)", real_disc, B,
            "- WSP-A on the 200 windows: accuracy 0.315", "accuracy 0.315"),
        est(0.42, 0.19, "set R (64 realistic sessions)", "ac483550 (pre-rescale)", syn, B, "| 24-25 |", "acc 0.42 vs 0.19"),
        est(0.425, None, real_disc, "ac483550 (pre-rescale)", real_disc, B, "| 26-38 |", "acc 0.425"),
        est(0.440, None, real_disc, "0ea2eb77 (final, = WSP-B), offline rescale on the same logits", real_disc, B,
            "the factor. Offline on the same logits", "0.425 → 0.440"),
        est(0.44, 0.19, "set R (64 realistic sessions)", "0ea2eb77 (final, = WSP-B)", syn, B, "| 39-40 |",
            "acc 0.44 vs 0.19"),
    ]
    # the "original model: 0.240" sentence sits on the continuation of the WSP-A-on-disclosed bullet
    assert "model: 0.240" in " ".join(wspb[0]["verbatim"])
    a_fin, b_fin_r, b_fin_d = wspa[5], wspb[4], wspb[3]

    def meas(rep, col, population, path):
        a = rep["excluding_disclosed"]["overall"]["columns"][col]["accuracy"]
        return {"accuracy": r(a["value"]), "ci": ci(a["ci95"]), "population": population,
                "source": str(path.relative_to(ROOT))}

    prior_q = q0q["excluding_disclosed"]["overall"]["columns"]["prior dev-majority"]["accuracy"]["value"]
    return {
        "_provenance": provenance([A, B, Q0_FULL, Q0_QUICK, WSPA_QUICK, WSPB_QUICK]),
        "figure": "F14", "title": "Synthetic vs real",
        "metric": "ACCURACY (not macro-F1): correct windows / windows",
        "slope": [
            {"model": "Qwen-0 (original)",
             "fable_estimate": {"WSP-A notebook, set F (synthetic)": a_fin["original_accuracy"],
                                "WSP-B notebook, set R (synthetic)": b_fin_r["original_accuracy"],
                                "WSP-B notebook, disclosed real windows": wspb[0]["original_accuracy"]},
             "measured": [meas(q0q, "Qwen-0", "quick look", Q0_QUICK), meas(q0f, "Qwen-0", "full val", Q0_FULL)]},
            {"model": "WSP-A", "fable_estimate": {"final v3b, set F (synthetic)": a_fin["accuracy"]},
             "measured": [meas(wa, "WSP-A", "quick look", WSPA_QUICK)]},
            {"model": "WSP-B", "fable_estimate": {"final, set R (synthetic)": b_fin_r["accuracy"],
                                                  "final, disclosed real windows (offline rescale)": b_fin_d["accuracy"]},
             "measured": [meas(wb, "WSP-B", "quick look", WSPB_QUICK)]},
        ],
        "notes": [
            "Fable's figures are its own self-reported accuracies on its own labelled texts; they are copied "
            "verbatim from the notes, not recomputed",
            "WSP-A/WSP-B measured numbers are val QUICK-LOOK numbers (1,000 sessions, excluding_disclosed, "
            "18,222 scored windows); Qwen-0 also has a full-val value",
            "disclosed real windows are class-balanced (50 per class), so accuracy there equals mean recall; "
            f"val is not balanced (always-PRODUCTIVE scores accuracy {r(prior_q)} on the quick look)",
            "Fable's sets were labelled by Fable reading the oracle rules; grey-zone windows were dropped",
        ],
        "wspa_notebook": wspa,
        "wspb_notebook": wspb,
    }


# ---------------------------------------------------------------------------

def load_figure_module(name: str):
    """Import scripts/blog/figures/<name>.py by path (the folder is not a package)."""
    path = FIGURES_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"hypanet_blog_figure_{name}", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "build", None)) or not hasattr(mod, "OUT"):
        raise SystemExit(f"{path} must expose build() and OUT")
    return mod


def main() -> None:
    v03, v02 = load(V03), load(V02)
    q0f, q0q, wa, wb = load(Q0_FULL), load(Q0_QUICK), load(WSPA_QUICK), load(WSPB_QUICK)
    figs = {"F6": f6(v03, v02), "F7": f7(v03, v02), "F12": f12(v03, q0f, q0q, wa, wb),
            "F13": f13(q0q, wa, wb), "F14": f14(q0f, q0q, wa, wb)}
    # Per-figure modules; F7 and F13 replace the plain builders' output with their modules' (same builders, plus
    # F7's early-stopping inset), so the JSON matches what each module writes on its own.
    for mod_name in FIGURE_MODULES:
        mod = load_figure_module(mod_name)
        figs[Path(mod.OUT).stem] = mod.build()
    for name, obj in figs.items():
        write(name, obj)
    total = 0
    for name in figs:
        n = (OUT / f"{name}.json").stat().st_size
        total += n
        print(f"{name}.json {n:,} bytes")
    print(f"total {total:,} bytes")


if __name__ == "__main__":
    main()
