"""Agreement between the two model labellers, before and after adjudication.

Binary agreement is REDUNDANT (ECHO, NARRATE, DECOR) against KEEP, on nodes neither labeller calls UNSURE; four-class
agreement is over KEEP, ECHO, NARRATE and DECOR on the same nodes; a five-category κ keeps UNSURE as its own
category. Each has Cohen's κ; the binary view also has Gwet's AC1 and positive and negative agreement, and both κs a
95% interval from resampling screens (2,000 resamples, seed 20261115). Pointer agreement is the share of nodes both
call ECHO whose targets match (exactly, and counting a target in the other's equiv list). "After" replaces each
disputed node's label with that labeller's second-pass label where there is one. Nodes with no literal words
(class null) are left out. Also: per extractor role and per repo, the contested rate of the final key, each
labeller's UNSURE share, and each labeller's role_ok share (SPEC.md §12.1 gate: both at least 90% on reference_real).

Also (SPEC.md §5): binary agreement with UNSURE kept as a third category; the subset of nodes whose role is not a
control (button, link, nav, tab, menu_item); the clear and judgement subsets by each labeller's conf; each labeller's
REDUNDANT F1 and false-alarm rate against the other on the headline nodes (neither UNSURE nor ct), the inter-labeller
ceiling; 95% intervals for κ, AC1 and positive and negative agreement; ct disagreement; screens without a usable
label from both, by status and repo; one-sided missed copy and screen asides; on adjudicated nodes, how often each
model moved to the other's label (by its A/B position) and how often the key took each labeller's round-1 label;
and, when present, test-retest and no-source agreement (`label --variant`). Screens whose answer the guide works
through (the pause-menu fixture, LABELLING.md §8) are left out of every statistic and counted. The κ < 0.6 gate
(§5) reads the "before" (round-1) binary κ.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from modellabel import common as c
from modellabel.adjudicate import (ORDER, ab_order, binary, load_adjudications, match_missed, round1,
                                   dispute_reasons, round1_status, same_target, screen_disputes)
from modellabel.to_records import normalise_reply, raw_path

lk = c.lk
CLASSES4 = ["KEEP", "ECHO", "NARRATE", "DECOR"]
BOOT_SEED = 20261115
BOOT_N = 2000
ROLE_GATE = 0.90
KAPPA_GATE = 0.60
CONTROL_ROLES = {"button", "link", "nav", "tab", "menu_item"}
CONCESSION_FLAG = 0.20  # flag when the two models' concession rates differ by more than this (each n >= 10)


def cohen_kappa(pairs, cats=None):
    """Cohen's κ for a list of (a, b) category pairs. None when there are no pairs or chance agreement is 1."""
    n = len(pairs)
    if not n:
        return None
    cats = cats or sorted({x for p in pairs for x in p}, key=str)
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in cats) / (n * n)
    if pe >= 1.0:
        return None
    return (po - pe) / (1 - pe)


def gwet_ac1(pairs, cats):
    n = len(pairs)
    if not n or len(cats) < 2:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pi = {k: (ca[k] + cb[k]) / (2 * n) for k in cats}
    pe = sum(p * (1 - p) for p in pi.values()) / (len(cats) - 1)
    return None if pe >= 1.0 else (po - pe) / (1 - pe)


def pos_neg_agreement(pairs, pos="REDUNDANT"):
    a = sum(1 for x, y in pairs if x == pos and y == pos)
    d = sum(1 for x, y in pairs if x != pos and y != pos)
    bc = sum(1 for x, y in pairs if (x == pos) != (y == pos))
    pa = 2 * a / (2 * a + bc) if (2 * a + bc) else None
    na = 2 * d / (2 * d + bc) if (2 * d + bc) else None
    return pa, na


def metrics(items):
    """items: dicts {screen, i, role, repo, a, b} with a and b model labels (class, target, equiv)."""
    rows = [x for x in items if x["a"]["class"] is not None and x["b"]["class"] is not None]
    sure = [x for x in rows if "UNSURE" not in (x["a"]["class"], x["b"]["class"])]
    bin_pairs = [(binary(x["a"]["class"]), binary(x["b"]["class"])) for x in sure]
    c4 = [(x["a"]["class"], x["b"]["class"]) for x in sure]
    c5 = [(x["a"]["class"], x["b"]["class"]) for x in rows]
    echo = [x for x in rows if x["a"]["class"] == "ECHO" and x["b"]["class"] == "ECHO"]
    exact = sum(1 for x in echo if x["a"].get("target") == x["b"].get("target"))
    lenient = sum(1 for x in echo if same_target(x["a"], x["b"]))
    pa, na = pos_neg_agreement(bin_pairs)
    c3 = [(binary(x["a"]["class"]), binary(x["b"]["class"])) for x in rows]
    head = [x for x in sure if not (x["a"].get("ct") or x["b"].get("ct"))]
    hp = [(binary(x["a"]["class"]), binary(x["b"]["class"])) for x in head]
    out = {
        "nodes": len(rows), "nodes_sure": len(sure),
        "unsure_any": len(rows) - len(sure),
        "binary": {"kappa": cohen_kappa(bin_pairs, ["KEEP", "REDUNDANT"]), "ac1": gwet_ac1(bin_pairs, ["KEEP",
                   "REDUNDANT"]), "pos_agreement": pa, "neg_agreement": na,
                   "raw": _raw(bin_pairs),
                   "table": {f"{a}|{b}": n for (a, b), n in sorted(Counter(bin_pairs).items())}},
        "class4": {"kappa": cohen_kappa(c4, CLASSES4), "raw": _raw(c4),
                   "table": {f"{a}|{b}": n for (a, b), n in sorted(Counter(c4).items())}},
        "class5": {"kappa": cohen_kappa(c5, CLASSES4 + ["UNSURE"]), "raw": _raw(c5)},
        "binary_unsure": {"kappa": cohen_kappa(c3, ["KEEP", "REDUNDANT", "UNSURE"]), "raw": _raw(c3)},
        "ceiling": ceiling(hp),
        "ct_disagree": sum(1 for x in rows if bool(x["a"].get("ct")) != bool(x["b"].get("ct"))),
        "pointer": {"both_echo": len(echo), "exact": _share(exact, len(echo)), "with_equiv": _share(lenient,
                                                                                                    len(echo))},
    }
    return out


def ceiling(pairs):
    """Each labeller scored against the other on headline nodes: REDUNDANT F1 (the same either way) and the false-alarm
    rate (REDUNDANT where the other says KEEP, over the other's KEEP nodes)."""
    tp = sum(1 for a, b in pairs if a == b == "REDUNDANT")
    fa = sum(1 for a, b in pairs if a == "REDUNDANT" and b == "KEEP")
    fb = sum(1 for a, b in pairs if b == "REDUNDANT" and a == "KEEP")
    keep_b = sum(1 for _, b in pairs if b == "KEEP")
    keep_a = sum(1 for a, _ in pairs if a == "KEEP")
    f1 = 2 * tp / (2 * tp + fa + fb) if (2 * tp + fa + fb) else None
    return {"nodes": len(pairs), "redundant_f1": f1, f"false_alarm_{ORDER[0]}_vs_{ORDER[1]}": _share(fa, keep_b),
            f"false_alarm_{ORDER[1]}_vs_{ORDER[0]}": _share(fb, keep_a)}


def _raw(pairs):
    return _share(sum(1 for a, b in pairs if a == b), len(pairs))


def _share(k, n):
    return k / n if n else None


def bootstrap(items, n=BOOT_N, seed=BOOT_SEED):
    """95% intervals resampling screens: binary and four-class κ, AC1, positive and negative agreement."""
    by = defaultdict(list)
    for x in items:
        by[x["screen"]].append(x)
    screens = sorted(by)
    if len(screens) < 2:
        return {}
    rng = random.Random(seed)
    keys = {"binary_kappa": ("binary", "kappa"), "class4_kappa": ("class4", "kappa"), "ac1": ("binary", "ac1"),
            "pos_agreement": ("binary", "pos_agreement"), "neg_agreement": ("binary", "neg_agreement")}
    vals = {k: [] for k in keys}
    for _ in range(n):
        pick = [x for s in (rng.choice(screens) for _ in screens) for x in by[s]]
        m = metrics(pick)
        for k, (g, f) in keys.items():
            if m[g][f] is not None:
                vals[k].append(m[g][f])

    def ci(v):
        if len(v) < 20:
            return None
        v = sorted(v)
        return [v[int(0.025 * len(v))], v[min(len(v) - 1, int(0.975 * len(v)))]]

    return {**{f"{k}_ci95": ci(v) for k, v in vals.items()}, "resamples": n, "seed": seed}


def collect(ls: c.LabelSet, labels_dir: Path):
    r1 = round1(ls, labels_dir)
    adj = load_adjudications(ls, labels_dir, r1)
    excluded = ls.guide_example_ids
    before, after = [], []
    disputed = 0
    for sid, sides in r1.items():
        if sid in excluded:
            continue
        scr = ls.get(sid)
        nodes = scr.by_i()
        disp = screen_disputes(sides)
        la, lb = sides[ORDER[0]]["labels"], sides[ORDER[1]]["labels"]
        xa, xb = (adj.get(sid) or {}).get(ORDER[0]) or {}, (adj.get(sid) or {}).get(ORDER[1]) or {}
        for i in sorted(la):
            base = {"screen": sid, "i": i, "role": nodes[i].get("role"), "repo": scr.repo}
            before.append({**base, "a": la[i], "b": lb[i]})
            a2 = xa.get(i, la[i]) if i in disp["adjudicate"] else la[i]
            b2 = xb.get(i, lb[i]) if i in disp["adjudicate"] else lb[i]
            after.append({**base, "a": a2, "b": b2})
        disputed += len(disp["adjudicate"])
    return r1, before, after, disputed


def subsets(items) -> dict:
    """Binary κ and raw agreement on the §5 subsets."""
    groups = {"not a control role": [x for x in items if x["role"] not in CONTROL_ROLES]}
    for k, lab in (("a", ORDER[0]), ("b", ORDER[1])):
        for conf in ("clear", "judgement"):
            groups[f"{conf} by {lab}"] = [x for x in items if x[k].get("conf") == conf]
    out = {}
    for g, xs in groups.items():
        m = metrics(xs)
        out[g] = {"nodes": m["nodes"], "binary_kappa": m["binary"]["kappa"], "binary_raw": m["binary"]["raw"]}
    return out


def concession(ls: c.LabelSet, labels_dir: Path, r1: dict) -> dict:
    """On disputed nodes with both second passes: for each model, the share of nodes where it moved to the other
    model's round-1 label (same tests), split by whether its own label was shown as A or as B; and the share of
    adjudicated nodes where the key equals each labeller's round-1 label."""
    adj = load_adjudications(ls, labels_dir, r1)
    rows = {lab: {"A": [0, 0], "B": [0, 0]} for lab in ORDER}  # [moved, n]
    key_eq = {lab: 0 for lab in ORDER}
    adjudicated = 0
    for sid, sides in r1.items():
        if sid in ls.guide_example_ids:
            continue
        per = adj.get(sid) or {}
        if len(per) < 2:
            continue
        disp = screen_disputes(sides)
        for i in disp["adjudicate"]:
            first, _ = ab_order(sid, i)
            for lab in ORDER:
                other = ORDER[1] if lab == ORDER[0] else ORDER[0]
                own, oth, fin = sides[lab]["labels"][i], sides[other]["labels"][i], per[lab].get(i)
                if fin is None:
                    continue
                pos = "A" if lab == first else "B"
                rows[lab][pos][1] += 1
                if dispute_free(fin, oth) and not dispute_free(fin, own):
                    rows[lab][pos][0] += 1
            xa, xb = per[ORDER[0]].get(i), per[ORDER[1]].get(i)
            if xa is not None and xb is not None and dispute_free(xa, xb) and xa["class"] != "UNSURE":
                adjudicated += 1
                for lab in ORDER:
                    key_eq[lab] += dispute_free(xa, sides[lab]["labels"][i])
    out = {"by_model": {}, "adjudicated_nodes": adjudicated,
           "key_equals_round1": {lab: _share(v, adjudicated) for lab, v in key_eq.items()}}
    rates = {}
    for lab, r in rows.items():
        moved, n = r["A"][0] + r["B"][0], r["A"][1] + r["B"][1]
        rates[lab] = (_share(moved, n), n)
        out["by_model"][lab] = {"moved_to_other": _share(moved, n), "nodes": n,
                                "as_A": _share(*r["A"]), "n_as_A": r["A"][1], "as_B": _share(*r["B"]),
                                "n_as_B": r["B"][1]}
    (ra, na), (rb, nb) = rates[ORDER[0]], rates[ORDER[1]]
    out["asymmetric"] = bool(ra is not None and rb is not None and na >= 10 and nb >= 10
                             and abs(ra - rb) > CONCESSION_FLAG)
    return out


def dispute_free(x: dict, y: dict) -> bool:
    return not dispute_reasons(x, y)


def coverage(ls: c.LabelSet, labels_dir: Path, r1: dict) -> dict:
    """Screens without a usable round-1 label from both labellers, by labeller status and by repo (refusals correlate
    with content, so a gap is a selection effect, SPEC.md §5a)."""
    by_status, by_repo, screens = Counter(), Counter(), {}
    for scr in ls.screens:
        if scr.id in r1:
            continue
        st = {lab: round1_status(labels_dir, ls.name, lab, scr) for lab in ORDER}
        screens[scr.id] = {"repo": scr.repo, "nodes": len(scr.labelable()), **st}
        for lab, v in st.items():
            if v != "ok":
                by_status[f"{lab} {v}"] += 1
        by_repo[scr.repo] += 1
    return {"screens_missing": len(screens), "by_status": dict(by_status), "by_repo": dict(by_repo),
            "screens": screens}


def one_sided(r1: dict, excluded: set) -> dict:
    """Missed copy and screen asides raised by one labeller only (not in the key)."""
    out = {"missed_both": 0, "aside_screen_both": 0}
    for lab in ORDER:
        out[f"missed_only_{lab}"] = 0
        out[f"aside_screen_only_{lab}"] = 0
    for sid, sides in r1.items():
        if sid in excluded:
            continue
        ra, rb = sides[ORDER[0]]["reply"], sides[ORDER[1]]["reply"]
        pairs, oa, ob = match_missed(ra.get("missed") or [], rb.get("missed") or [])
        out["missed_both"] += len(pairs)
        out[f"missed_only_{ORDER[0]}"] += len(oa)
        out[f"missed_only_{ORDER[1]}"] += len(ob)
        ca = {it["code"] for it in ra.get("aside_screen") or []}
        cb = {it["code"] for it in rb.get("aside_screen") or []}
        out["aside_screen_both"] += len(ca & cb)
        out[f"aside_screen_only_{ORDER[0]}"] += len(ca - cb)
        out[f"aside_screen_only_{ORDER[1]}"] += len(cb - ca)
    return out


VARIANTS = ("retest", "nosource")


def variant_id(labeller: str, variant: str) -> str:
    return f"{labeller}-{variant}"


def variants(ls: c.LabelSet, labels_dir: Path) -> dict:
    """Each labeller against its own variant run (`label --variant retest|nosource`), on the screens both cover:
    test-retest (the same prompt again) and no-source (the screen JSON without its source file)."""
    out = {}
    for lab in ORDER:
        for v in VARIANTS:
            vid = variant_id(lab, v)
            items = []
            for scr in ls.screens:
                if scr.id in ls.guide_example_ids:
                    continue
                pair = []
                for who in (lab, vid):
                    raw = c.read_json(raw_path(labels_dir, ls.name, who, scr.id))
                    if not raw or raw.get("status") != "ok" or raw.get("screen_sha256") != scr.sha256:
                        break
                    labels, errs, _ = normalise_reply(raw["reply"], scr)
                    if errs:
                        break
                    pair.append(labels)
                if len(pair) == 2:
                    nodes = scr.by_i()
                    items += [{"screen": scr.id, "i": i, "role": nodes[i].get("role"), "repo": scr.repo,
                               "a": pair[0][i], "b": pair[1][i]} for i in sorted(pair[0])]
            if items:
                m = metrics(items)
                out[vid] = {"screens": len({x["screen"] for x in items}), "nodes": m["nodes"],
                            "binary_kappa": m["binary"]["kappa"], "binary_raw": m["binary"]["raw"],
                            "class4_kappa": m["class4"]["kappa"]}
    return out


def labeller_stats(r1: dict):
    out = {}
    for lab in ORDER:
        labs = [lab_ for sides in r1.values() for lab_ in sides[lab]["labels"].values()]
        labelled = [x for x in labs if x["class"] is not None]
        out[lab] = {"nodes": len(labs), "classes": dict(Counter(x["class"] or "(unresolved)" for x in labs)),
                    "unsure_share": _share(sum(1 for x in labelled if x["class"] == "UNSURE"), len(labelled)),
                    "role_ok_share": _share(sum(1 for x in labs if x.get("role_ok", True)), len(labs)),
                    "ct": sum(1 for x in labelled if x.get("ct"))}
    return out


def spend(labels_dir: Path, set_name: str) -> dict:
    """Opus spend on this set from labels/raw/spend.jsonl, by phase."""
    out = {}
    p = Path(labels_dir) / "raw" / "spend.jsonl"
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(r, dict) and r.get("event") != "launch" and r.get("set") == set_name:
                out[r.get("phase", "?")] = out.get(r.get("phase", "?"), 0.0) + float(r.get("cost_usd") or 0)
    return out


def final_stats(labels_dir: Path, set_name: str):
    lf = lk.LabelFile.load(lk.label_path(labels_dir, set_name, c.FINAL_LABELLER))
    nodes = [r for k, r in lf.latest.items() if k[1] == "node"]
    src = Counter(r.get("source") for r in nodes)
    scored = sum(v for k, v in src.items() if k != "unresolved")
    cont = [r for r in nodes if r.get("source") == "contested"]
    return {"nodes": len(nodes), "sources": dict(src),
            "contested_rate": _share(src.get("contested", 0), scored),
            "contested_with_binary": sum(1 for r in cont if r.get("binary")),
            "binary_contested_rate": _share(sum(1 for r in cont if not r.get("binary")), scored),
            "classes": dict(Counter(r.get("class") or r.get("source") for r in nodes))}


def report(ls: c.LabelSet, labels_dir: Path = c.LABELS_DIR) -> dict:
    r1, before, after, disputed = collect(ls, labels_dir)
    excluded = ls.guide_example_ids
    res = {"set": ls.name, "screens_both": len(r1), "screens_total": len(ls.screens), "labellers": list(ORDER),
           "excluded_screens": sorted(excluded & set(r1)), "disputed_nodes": disputed,
           "before": metrics(before), "after": metrics(after),
           "before_ci": bootstrap(before), "after_ci": bootstrap(after),
           "subsets_before": subsets(before), "per_role": {}, "per_repo": {},
           "labellers_stats": labeller_stats({k: v for k, v in r1.items() if k not in excluded}),
           "coverage": coverage(ls, labels_dir, r1), "one_sided": one_sided(r1, excluded),
           "concession": concession(ls, labels_dir, r1), "variants": variants(ls, labels_dir),
           "final": final_stats(labels_dir, ls.name)}
    for key, field in (("per_role", "role"), ("per_repo", "repo")):
        groups = sorted({x[field] for x in before})
        for g in groups:
            mb = metrics([x for x in before if x[field] == g])
            ma = metrics([x for x in after if x[field] == g])
            res[key][g] = {"nodes": mb["nodes"], "binary_kappa_before": mb["binary"]["kappa"],
                           "binary_kappa_after": ma["binary"]["kappa"], "class4_kappa_before": mb["class4"]["kappa"],
                           "class4_kappa_after": ma["class4"]["kappa"], "binary_raw_before": mb["binary"]["raw"]}
    res["spend_usd"] = spend(labels_dir, ls.name)
    st = res["labellers_stats"]
    res["role_gate"] = {"threshold": ROLE_GATE, "applies": ls.name == "reference_real",
                        "pass": all((st[x]["role_ok_share"] or 0) >= ROLE_GATE for x in ORDER) if r1 else None}
    kb = res["before"]["binary"]["kappa"]
    res["kappa_gate"] = {"threshold": KAPPA_GATE, "reads": "before (round 1) binary κ",
                         "pass": None if kb is None else kb >= KAPPA_GATE}
    return res


def _f(v, pct=False):
    if v is None:
        return "–"
    return f"{v:.1%}" if pct else f"{v:.3f}"


def markdown(res: dict) -> str:
    b, a = res["before"], res["after"]
    bc, ac = res["before_ci"], res["after_ci"]
    ci = lambda x: f" [{x[0]:.2f}, {x[1]:.2f}]" if x else ""  # noqa: E731
    L = [f"# Model-labeller agreement: {res['set']}", "",
         f"Labellers {ORDER[0]} and {ORDER[1]}; {res['screens_both']} of {res['screens_total']} screens labelled by "
         f"both; {b['nodes']} nodes with literal words; {res['disputed_nodes']} disputed nodes went to a second pass.",
         "", "| Measure | Before adjudication | After adjudication |", "|---|---|---|",
         f"| Binary κ (REDUNDANT vs KEEP), 95% CI | {_f(b['binary']['kappa'])}{ci(bc.get('binary_kappa_ci95'))} | "
         f"{_f(a['binary']['kappa'])}{ci(ac.get('binary_kappa_ci95'))} |",
         f"| Binary raw agreement | {_f(b['binary']['raw'], 1)} | {_f(a['binary']['raw'], 1)} |",
         f"| Gwet's AC1 (binary) | {_f(b['binary']['ac1'])} | {_f(a['binary']['ac1'])} |",
         f"| Positive / negative agreement | {_f(b['binary']['pos_agreement'])} / {_f(b['binary']['neg_agreement'])}"
         f" | {_f(a['binary']['pos_agreement'])} / {_f(a['binary']['neg_agreement'])} |",
         f"| Four-class κ, 95% CI | {_f(b['class4']['kappa'])}{ci(bc.get('class4_kappa_ci95'))} | "
         f"{_f(a['class4']['kappa'])}{ci(ac.get('class4_kappa_ci95'))} |",
         f"| AC1 95% CI; positive / negative agreement 95% CI | {ci(bc.get('ac1_ci95'))}; "
         f"{ci(bc.get('pos_agreement_ci95'))} / {ci(bc.get('neg_agreement_ci95'))} | {ci(ac.get('ac1_ci95'))}; "
         f"{ci(ac.get('pos_agreement_ci95'))} / {ci(ac.get('neg_agreement_ci95'))} |",
         f"| Binary κ with UNSURE as a third category | {_f(b['binary_unsure']['kappa'])} | "
         f"{_f(a['binary_unsure']['kappa'])} |",
         f"| Five-category κ (UNSURE kept) | {_f(b['class5']['kappa'])} | {_f(a['class5']['kappa'])} |",
         f"| Inter-labeller ceiling: REDUNDANT F1 (headline nodes) | {_f(b['ceiling']['redundant_f1'])} | "
         f"{_f(a['ceiling']['redundant_f1'])} |",
         f"| False-alarm rate {ORDER[0]} vs {ORDER[1]} / {ORDER[1]} vs {ORDER[0]} | "
         f"{_f(b['ceiling'][f'false_alarm_{ORDER[0]}_vs_{ORDER[1]}'], 1)} / "
         f"{_f(b['ceiling'][f'false_alarm_{ORDER[1]}_vs_{ORDER[0]}'], 1)} | "
         f"{_f(a['ceiling'][f'false_alarm_{ORDER[0]}_vs_{ORDER[1]}'], 1)} / "
         f"{_f(a['ceiling'][f'false_alarm_{ORDER[1]}_vs_{ORDER[0]}'], 1)} |",
         f"| Nodes where ct differs | {b['ct_disagree']} | {a['ct_disagree']} |",
         f"| Nodes with UNSURE from either | {b['unsure_any']} | {a['unsure_any']} |",
         f"| Pointer agreement, both ECHO (exact / with equiv) | {_f(b['pointer']['exact'], 1)} / "
         f"{_f(b['pointer']['with_equiv'], 1)} (n={b['pointer']['both_echo']}) | {_f(a['pointer']['exact'], 1)} / "
         f"{_f(a['pointer']['with_equiv'], 1)} (n={a['pointer']['both_echo']}) |", ""]
    g = res["kappa_gate"]
    L += [f"κ gate (SPEC.md §5, round-1 binary κ ≥ {g['threshold']}): "
          f"{'pass' if g['pass'] else 'FAIL' if g['pass'] is False else 'not measured'}."
          + (f" Left out of every statistic (the guide works their answer): {', '.join(res['excluded_screens'])}."
             if res["excluded_screens"] else ""), ""]
    fs = res["final"]
    L += [f"Final key: {fs['nodes']} node records; sources {json.dumps(fs['sources'])}; contested rate "
          f"{_f(fs['contested_rate'], 1)} of scored nodes ({fs['contested_with_binary']} contested nodes keep a binary "
          f"consensus; binary contested rate {_f(fs['binary_contested_rate'], 1)}).", ""]
    cv = res["coverage"]
    if cv["screens_missing"]:
        L += [f"Screens without a usable label from both: {cv['screens_missing']}; by status "
              f"{json.dumps(cv['by_status'])}; by repo {json.dumps(cv['by_repo'])}.", ""]
    L += ["One-sided screen-level items (not keyed): " + json.dumps(res["one_sided"]), ""]
    co = res["concession"]
    if co["adjudicated_nodes"] or any(v["nodes"] for v in co["by_model"].values()):
        L += ["| Second pass | Moved to the other's label | shown as A | shown as B | Key = its round-1 label |",
              "|---|---|---|---|---|"]
        for lab, v in co["by_model"].items():
            L.append(f"| {lab} | {_f(v['moved_to_other'], 1)} (n={v['nodes']}) | {_f(v['as_A'], 1)} (n={v['n_as_A']})"
                     f" | {_f(v['as_B'], 1)} (n={v['n_as_B']}) | {_f(co['key_equals_round1'][lab], 1)} |")
        if co["asymmetric"]:
            L.append(f"\nASYMMETRIC CONCESSION: the two models' rates differ by more than {CONCESSION_FLAG:.0%}; the "
                     "key leans towards the less deferential model on disputed nodes.")
        L.append("")
    L += ["Subsets (round 1):", "", "| Subset | Nodes | Binary κ | Binary raw |", "|---|---|---|---|"]
    for gname, m in res["subsets_before"].items():
        L.append(f"| {gname} | {m['nodes']} | {_f(m['binary_kappa'])} | {_f(m['binary_raw'], 1)} |")
    L.append("")
    if res["variants"]:
        L += ["| Variant | Screens | Nodes | Binary κ | Binary raw | Four-class κ |", "|---|---|---|---|---|---|"]
        for vid, v in res["variants"].items():
            L.append(f"| {vid} | {v['screens']} | {v['nodes']} | {_f(v['binary_kappa'])} | {_f(v['binary_raw'], 1)} |"
                     f" {_f(v['class4_kappa'])} |")
        L.append("")
    if res.get("spend_usd"):
        L += ["Opus spend on this set (US$): " + ", ".join(f"{k} {v:.2f}" for k, v in res["spend_usd"].items())
              + f"; total {sum(res['spend_usd'].values()):.2f}.", ""]
    L += ["| Labeller | Nodes | UNSURE share | role_ok share | ct labels |", "|---|---|---|---|---|"]
    for lab, s in res["labellers_stats"].items():
        L.append(f"| {lab} | {s['nodes']} | {_f(s['unsure_share'], 1)} | {_f(s['role_ok_share'], 1)} | {s['ct']} |")
    g = res["role_gate"]
    if g["applies"]:
        L += ["", f"Role gate (SPEC.md §12.1, both role_ok ≥ {g['threshold']:.0%}): "
                  f"{'pass' if g['pass'] else 'FAIL' if g['pass'] is False else 'not measured'}."]
    for key, title in (("per_role", "extractor role"), ("per_repo", "repo")):
        L += ["", f"Per {title}:", "", f"| {title.capitalize()} | Nodes | Binary κ before | after | Four-class κ before"
              " | after |", "|---|---|---|---|---|---|"]
        for gname, m in res[key].items():
            L.append(f"| {gname} | {m['nodes']} | {_f(m['binary_kappa_before'])} | {_f(m['binary_kappa_after'])} | "
                     f"{_f(m['class4_kappa_before'])} | {_f(m['class4_kappa_after'])} |")
    return "\n".join(L) + "\n"


def write_report(ls: c.LabelSet, labels_dir: Path = c.LABELS_DIR) -> tuple:
    res = report(ls, labels_dir)
    jp = Path(labels_dir) / f"{ls.name}.agreement.json"
    mp = Path(labels_dir) / f"{ls.name}.agreement.md"
    c.write_json(jp, res)
    md = markdown(res)
    mp.write_text(md, encoding="utf-8")
    return res, md, jp, mp
