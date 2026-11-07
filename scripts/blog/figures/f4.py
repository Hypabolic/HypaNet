"""F4 (post Figure 9): "A transformer, written by hand", the named wiring of both v0.3 written transformers.

Run from the repo root:  uv run python scripts/blog/figures/f4.py

Parses the committed circuit documents Fable wrote with each network:
- weight_program/v0_3/phase-b/CIRCUITS.md (revised transformer): section 3 (residual bands and who writes them),
  section 5 (every head: q, k, v -> out; every MLP unit: expression -> target) and section 6 (head readout).
- weight_program/v0_3/phase-a/CIRCUITS.md (zero-label transformer): the same sections.

Each head row and each MLP target becomes a "chip". A chip reads the bands named in its q / k / value expressions and
writes the bands on the right of "->". A band read by a chip is linked to its latest earlier writer (embeddings,
an earlier layer, or for an MLP the heads of its own layer). The class paths are the chips upstream of each logit's
positive terms; the negative terms (vetoes such as PROG) are listed but not traced. The roles are short plain-English
glosses of the CIRCUITS.md content column, written for the figure. Sealed scores are copied from
artifacts/reports/v03_sealed.json. No evaluator, scorer or dataset is opened.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "blog_figures" / "F4.json"
GENERATED_BY = "scripts/blog/figures/f4.py"
WP = ROOT / "weight_program" / "v0_3"
V03 = ROOT / "artifacts" / "reports" / "v03_sealed.json"
CLASSES = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]

# Short glosses for the chips, from the CIRCUITS.md content column (section 3) and section 5.
ROLES = {
    "b": {
        "E": "token flags, id bits", "P": "position in block",
        "H0.0": "tool of this block", "H0.1": "latest status word", "H0.2": "previous token",
        "H0.3": "has a hunk line", "H0.4": "intent token", "MLP0:VM": "validation or hunk",
        "H1.0": "undo match", "H1.1": "passed count", "H1.2": "nearest same key",
        "H1.3": "same key within 7", "H1.4": "counts failures", "H1.5": "last check or edit",
        "MLP1:NOTIDST": "status changed", "MLP1:CNT3": "failures ≤ 3", "MLP1:CNT5": "failures ≤ 5",
        "MLP1:REVTOK": "reverting token", "MLP1:NOOP": "re-applied edit", "MLP1:REDINSP": "repeated read",
        "H2.0": "prev same-key block", "H2.1": "best count so far", "H2.2": "nearest validation",
        "H2.3": "block reverts", "MLP2:IMPROVED": "new best pass count", "MLP2:SI": "ineffective block",
        "MLP2:INPOL": "retry within policy", "MLP2:RESOLV": "failure resolved",
        "H3.0": "window averages", "H3.1": "active retry", "H3.2": "resolved in last 4",
        "MLP3:V2": "≥ 2 reversions", "MLP3:I5": "≥ 5 ineffective", "MLP3:PROG": "progress in window",
        "MLP3:RECB": "recovering",
    },
    "a": {
        "E": "token flags, id bits", "P": "position in block",
        "H0.0": "old id slot", "H0.1": "new id slot", "H0.2": "tool of this block", "H0.3": "latest status word",
        "H0.4": "where the count is", "H0.5": "has a hunk line",
        "MLP0:SLOT": "intent slot", "MLP0:VM": "validation or hunk",
        "H1.0": "undo match", "H1.1": "intent bits to end", "H1.2": "passed count", "H1.3": "last check or edit",
        "MLP1:REVTOK": "reverting token", "MLP1:REDVAL": "redundant validation",
        "H2.0": "counts same intent", "H2.1": "block reverts",
        "MLP2:SI": "ineffective block", "MLP2:KILL": "retry beyond policy", "MLP2:INPOL": "failure in policy",
        "MLP2:RESOLV": "failure resolved",
        "H3.0": "window averages", "H3.1": "recovery sign", "H3.2": "latest validation",
        "H3.3": "earliest validation",
        "MLP3:V2": "≥ 2 reversions", "MLP3:I5": "≥ 5 ineffective", "MLP3:A4": "recovering",
        "MLP3:PROGN": "passed count rose", "MLP3:PROGF": "FAILED then OK", "MLP3:SIT": "ineffective now",
    },
}

NETS = {
    "b": {"dir": WP / "phase-b", "name": "Revised transformer", "key": "v0.3 B network",
          "labels": "200 labelled windows"},
    "a": {"dir": WP / "phase-a", "name": "Zero-label transformer", "key": "v0.3 A network",
          "labels": "zero labels"},
}

TOKEN = re.compile(r"[A-Z][A-Z0-9]*(?:\.\.[A-Z][A-Z0-9]*)?")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _section(md: str, num: int) -> str:
    m = re.search(rf"^## {num}\. .*?$(.*?)(?=^## \d+\. |\Z)", md, re.S | re.M)
    assert m, f"section {num} not found"
    return m.group(1)


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _strip_paren(s: str) -> str:
    prev = None
    while prev != s:
        prev, s = s, re.sub(r"\([^()]*\)", " ", s)
    return s


def parse_bands(md: str) -> tuple[list[dict], dict[str, list[str]]]:
    """Section 3: one band per named entry; ranges such as BTINSP..BTPOLL stay one band."""
    bands = []
    for line in _section(md, 3).splitlines():
        if not line.startswith("|") or line.startswith("|---") or "| dims |" in line:
            continue
        dims, names, writer, content = _cells(line)[:4]
        for n in [x.strip() for x in names.split(",")]:
            bands.append({"name": n, "dims": dims, "writer": writer, "content": content})
    order = [b["name"] for b in bands]
    # Members of each range band (e.g. BTINSP..BTPOLL = B + each of TINSP..TPOLL), so a read of BTVAL resolves to it.
    members: dict[str, list[str]] = {}
    for b in bands:
        if ".." in b["name"]:
            lo, hi = b["name"].split("..")
            base = None
            for pre in ("B", "P", ""):
                if lo.startswith(pre) and lo[len(pre):] in order and hi[len(pre):] in order:
                    base = order[order.index(lo[len(pre):]): order.index(hi[len(pre):]) + 1]
                    members[b["name"]] = [pre + x for x in base]
                    break
            assert base, f"cannot expand {b['name']}"
    return bands, members


def make_resolver(bands: list[dict], members: dict[str, list[str]]):
    names = {b["name"] for b in bands}
    order = [b["name"] for b in bands]
    member_of = {m: rng for rng, ms in members.items() for m in ms}

    def resolve(text: str, allow_bos: bool) -> list[str]:
        out: list[str] = []
        text = text.replace("intent match", " IBITS BTINSP..BTPOLL ")
        for tok in TOKEN.findall(text):
            cands: list[str] = []
            if ".." in tok:
                lo, hi = tok.split("..")
                if tok in names:
                    cands = [tok]
                elif lo in order and hi in order:
                    cands = order[order.index(lo): order.index(hi) + 1]
                elif lo in member_of:
                    cands = [member_of[lo]]
            elif tok in names:
                cands = [tok]
            elif tok in member_of:
                cands = [member_of[tok]]
            for c in cands:
                if c == "BOS" and not allow_bos:
                    continue
                if c not in out:
                    out.append(c)
        return out

    return resolve


def parse_chips(md: str, net: str, bands: list[dict], members: dict[str, list[str]]) -> list[dict]:
    resolve = make_resolver(bands, members)
    sec = _section(md, 5)
    chips: list[dict] = []
    # Embeddings.
    for wid, name in (("E", "Token embedding"), ("P", "Offset embedding")):
        writes = [b["name"] for b in bands if b["writer"].split(",")[0].strip() == wid]
        chips.append({"id": wid, "kind": "embed", "layer": -1, "label": name, "reads": [], "writes": writes,
                      "reads_by_out": {w: [] for w in writes},
                      "note": f"{name} ({wid}): writes " + ", ".join(writes)})
    layer = None
    lines = sec.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^### Layer (\d)", line)
        if m:
            layer = int(m.group(1))
        hm = re.match(r"^\| (H\d\.\d) \|", line)
        if hm:
            c = _cells(line)
            q, k, v = c[1], c[2], c[-1]
            qk = resolve(_strip_paren(q) + " " + _strip_paren(k), allow_bos=False)
            reads = list(qk)
            writes: list[str] = []
            by_out: dict[str, list[str]] = {}
            for part in v.split(";"):
                if "->" not in part:
                    continue
                lhs, rhs = part.split("->", 1)
                ls = resolve(_strip_paren(lhs), allow_bos=True)
                rs = resolve(_strip_paren(rhs), allow_bos=False)
                for r in ls:
                    if r not in reads:
                        reads.append(r)
                # "A, B -> C, D" pairs up; "A + B - C -> D" or a range into one band feeds every output.
                for n, w in enumerate(rs):
                    src = [ls[n]] if len(ls) == len(rs) else ls
                    by_out[w] = qk + [x for x in src if x not in qk]
                    if w not in writes:
                        writes.append(w)
            note = f"{hm.group(1)} · q: {q} · k: {k} · v → out: {v}"
            chips.append({"id": hm.group(1), "kind": "head", "layer": layer, "reads": reads, "writes": writes,
                          "reads_by_out": by_out, "note": note.replace("`", "")})
        mm = re.match(r"^MLP (\d) \(units ([\d-]+)[^)]*\)(.*)$", line)
        if mm:
            L = int(mm.group(1))
            block = [mm.group(3)]
            j = i + 1
            while j < len(lines) and not lines[j].startswith(("###", "## ", "| head", "Because", "MLP ")):
                block.append(lines[j])
                j += 1
            units: list[tuple[str, str, str]] = []  # (unit range, expression, target text)
            for b in block:
                if b.startswith("|"):
                    c = _cells(b)
                    if c[0] in ("units", "---") or c[0].startswith("---"):
                        continue
                    expr_cell = next(x for x in c if "->" in x)
                    units.append((c[0], expr_cell, ""))
                else:
                    for expr in re.findall(r"`([A-Z0-9]+ = [^`]+)`", b):
                        units.append(("", expr, ""))
            per_target: dict[str, dict] = {}
            for ur, expr, _ in units:
                if "->" in expr:
                    parts = expr.split("->")
                    targets: list[str] = []
                    reads_txt = parts[0]
                    for seg in parts[1:]:
                        head, _, rest = seg.partition(";")
                        targets += resolve(_strip_paren(head), allow_bos=False)
                        reads_txt += " " + rest
                else:
                    tgt, rhs = expr.split(" = ", 1)
                    targets, reads_txt = [tgt.strip()], rhs
                reads = [r for r in resolve(reads_txt.replace("`", " "), allow_bos=False) if r not in targets]
                for t in targets:
                    e = per_target.setdefault(t, {"reads": [], "units": [], "exprs": []})
                    e["reads"] += [r for r in reads if r not in e["reads"]]
                    if ur:
                        e["units"].append(ur)
                    e["exprs"].append(expr.replace("`", ""))
            for t, e in per_target.items():
                cid = f"MLP{L}:{t}"
                units_txt = f"units {', '.join(e['units'])}" if e["units"] else f"units {mm.group(2)}"
                chips.append({"id": cid, "kind": "mlp", "layer": L, "reads": e["reads"], "writes": [t],
                              "reads_by_out": {t: e["reads"]},
                              "note": f"MLP {L} ({units_txt}) → {t} · " + " · ".join(e["exprs"])})
            i = j
            continue
        i += 1
    for c in chips:
        c["role"] = ROLES[net].get(c["id"], "")
        assert c["role"], f"{net}: no role for {c['id']}"
        assert c["writes"], f"{net}: {c['id']} writes nothing"
        assert len(c["role"]) <= 20, f"{net}: role too long for a chip: {c['role']}"
    assert set(ROLES[net]) == {c["id"] for c in chips}, set(ROLES[net]) ^ {c["id"] for c in chips}
    return chips


def parse_layers(md: str) -> list[dict]:
    sec = _section(md, 5)
    out = []
    for m in re.finditer(r"^### Layer (\d) \(([^;)]+)", sec, re.M):
        out.append({"layer": int(m.group(1)), "title": m.group(2).strip()})
    for m in re.finditer(r"^MLP (\d) \(units (\d+)-(\d+)", sec, re.M):
        out[int(m.group(1))]["mlp_units"] = int(m.group(3)) - int(m.group(2)) + 1
    assert [x["layer"] for x in out] == [0, 1, 2, 3] and all("mlp_units" in x for x in out), out
    return out


def parse_logits(md: str, bands, members) -> list[dict]:
    resolve = make_resolver(bands, members)
    out = []
    for line in _section(md, 6).splitlines():
        m = re.match(r"^\| (\w+) \((\d)\) \|", line)
        if not m:
            continue
        c = _cells(line)
        expr = c[1].replace("`", "")
        terms = []
        for sign, coef, name in re.findall(r"([+-]?)\s*(\d+)\s*\*?\s*([A-Z][A-Z0-9]+)", expr):
            if name in resolve(name, False):
                terms.append({"band": name, "w": (-1 if sign == "-" else 1) * int(coef)})
        out.append({"class": m.group(1), "index": int(m.group(2)), "expr": expr,
                    "active": int(c[2]) if c[2].strip().lstrip("-").isdigit() else 0, "terms": terms})
    out.sort(key=lambda x: x["index"])
    assert [x["class"] for x in out] == CLASSES
    return out


def _stage(c: dict) -> float:
    """Order of execution: heads read the residual before their layer; an MLP also reads its own layer's heads."""
    return {"embed": -1.0, "head": c["layer"] + 0.0, "mlp": c["layer"] + 0.5}[c["kind"]]


def _writers(chips: list[dict], band: str, before: float) -> list[dict]:
    """Latest writer(s) of a band before a stage (a band rewritten later, like A's BITS at the EOE, keeps both)."""
    ws = [w for w in chips if band in w["writes"] and _stage(w) < before]
    assert ws, f"no writer of {band} before stage {before}"
    return ws


def link(chips: list[dict]) -> list[dict]:
    """Edges writer -> reader, each carrying the bands passed."""
    edges: dict[tuple[str, str], list[str]] = {}
    for r in chips:
        for band in r["reads"]:
            for w in _writers(chips, band, _stage(r)):
                edges.setdefault((w["id"], r["id"]), []).append(band)
    return [{"from": a, "to": b, "bands": bs} for (a, b), bs in edges.items()]


def paths(chips, logits) -> dict[str, dict]:
    """Trace each logit's positive terms upstream band by band: a chip contributes only the reads that feed the
    band asked of it (its value path plus its query / key match and gate), not all of its reads."""
    out = {}
    for lg in logits:
        pos = [t["band"] for t in lg["terms"] if t["w"] > 0]
        neg = [t["band"] for t in lg["terms"] if t["w"] < 0]
        seen_chips: set[str] = set()
        used: set[tuple[str, str]] = set()
        done: set[tuple[str, float]] = set()
        stack = [(b, 99.0, "logit") for b in pos]
        while stack:
            band, before, reader = stack.pop()
            if (band, before) in done:
                continue
            done.add((band, before))
            for w in _writers(chips, band, before):
                seen_chips.add(w["id"])
                if reader != "logit":
                    used.add((w["id"], reader))
                for r in w["reads_by_out"].get(band, []):
                    stack.append((r, _stage(w), w["id"]))
        direct = sorted({w["id"] for b in pos for w in _writers(chips, b, 99.0)})
        vetoes = sorted({w["id"] for b in neg for w in _writers(chips, b, 99.0)})
        out[lg["class"]] = {
            "positive": pos, "vetoes": neg, "direct": direct, "veto_chips": vetoes,
            "chips": sorted(seen_chips), "edges": sorted(f"{a}>{b}" for a, b in used),
        }
    return out


def build() -> dict:
    rep = json.loads(V03.read_text())
    nets = {}
    sources = []
    for nid, n in NETS.items():
        path = n["dir"] / "CIRCUITS.md"
        sources.append(path)
        md = path.read_text()
        bands, members = parse_bands(md)
        chips = parse_chips(md, nid, bands, members)
        logits = parse_logits(md, bands, members)
        edges = link(chips)
        set_params = int(re.search(r"sets ([\d,]+) of the ([\d,]+) parameters", md).group(1).replace(",", ""))
        total = int(re.search(r"sets [\d,]+ of the ([\d,]+) parameters", md).group(1).replace(",", ""))
        score = {sp: round(rep["summary"][sp]["all"][n["key"]]["macro_f1"][0], 5) for sp in ("test", "ood")}
        nets[nid] = {
            "name": n["name"], "labels": n["labels"], "source": str(path.relative_to(ROOT)),
            "params_total": total, "params_set_by_builder": set_params, "layers": parse_layers(md),
            "macro_f1": score,
            "bands": [{"name": b["name"], "dims": b["dims"], "content": b["content"].replace("`", "")} for b in bands],
            "chips": chips, "edges": edges, "logits": logits, "paths": paths(chips, logits),
        }
    b = nets["b"]
    assert [lg["active"] for lg in b["logits"]] == [0, 10, 20, 55]
    assert b["params_total"] == 2128932
    assert (round(b["macro_f1"]["test"], 3), round(b["macro_f1"]["ood"], 3)) == (0.957, 0.908)
    sources.append(V03)
    rels = [str(p.relative_to(ROOT)) for p in sources]
    return {
        "_provenance": {
            "source": rels,
            "sha256": {r: hashlib.sha256((ROOT / r).read_bytes()).hexdigest() for r in rels},
            "source_commit": {r: _git("log", "-1", "--format=%H", "--", r) for r in rels},
            "git_commit": _git("rev-parse", "HEAD"),
            "generated_by": GENERATED_BY,
        },
        "figure": "F4",
        "title": "A transformer, written by hand",
        "classes": CLASSES,
        "nets": nets,
        "notes": [
            "macro_f1: v0.3 sealed test / sealed OOD, scored once (v03_sealed.json summary)",
            "params_set_by_builder is the CIRCUITS.md count; a |w| > 1e-9 count of the B artefact gives 156,501 (manifest)",
            "paths trace the positive terms of each logit upstream; negative terms are listed as vetoes, not traced",
        ],
    }


def main() -> None:
    obj = build()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"{OUT.relative_to(ROOT)} {OUT.stat().st_size:,} bytes")
    for nid, n in obj["nets"].items():
        print(nid, len(n["chips"]), "chips", len(n["edges"]), "edges",
              {k: (len(v["chips"]), len(v["edges"])) for k, v in n["paths"].items()})


if __name__ == "__main__":
    sys.exit(main())
