"""SPEC.md §12.3 interpreter check: the floor test network against the plain-code floor, and JS against numpy.

    uv run python validate/check_floor.py [--art DIR] [--min 0.99] [--js] [--random N --seed S] [--json OUT] INPUT...

INPUT is a screen JSON file, a directory (every ``*.json`` in it, sorted) or a ``.jsonl`` file (one screen per
line). Without ``--art`` the network is built afresh by ``validate/build_floor_net.py`` in a temporary directory.

Reported: nodes, the share whose class and pointer equal the floor's (the §12.3 pass mark is ``--min``, default 0.99,
on ``val``), the same against the floor limited to pairs at most 7 nodes apart (the network's reach; expected 100%),
and the first disagreements. With ``--js`` the artefact is compiled and every screen is also run through
``lint/runtime.js``: argmax and pointer must be identical on every node and logits within 1e-6 x max(1, |logit|).
Exit status 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lint import compile as net_compile  # noqa: E402
from ui import model as mdl  # noqa: E402
from ui import tokenizer as tk  # noqa: E402
from validate import build_floor_net  # noqa: E402
from validate.floor import floor_screen  # noqa: E402
from validate.random_screens import random_screens  # noqa: E402

SLOP = Path(__file__).resolve().parents[1]
NET_RUN = SLOP / "lint" / "tests" / "net_run.js"
LOGIT_TOL = 1e-6


def load_inputs(paths: list[str]) -> list[dict]:
    screens: list[dict] = []
    for p in map(Path, paths):
        if p.is_dir():
            screens.extend(json.loads(f.read_text(encoding="utf-8")) for f in sorted(p.glob("*.json")))
        elif p.suffix == ".jsonl":
            screens.extend(json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
        else:
            screens.append(json.loads(p.read_text(encoding="utf-8")))
    return screens


def js_predict(net_path: Path, screens: list[dict], batch: int = 100) -> list[list[dict]]:
    out: list[list[dict]] = []
    for s in range(0, len(screens), batch):
        req = json.dumps({"net": str(net_path), "screens": screens[s : s + batch]})
        res = subprocess.run(["node", str(NET_RUN)], input=req, capture_output=True, text=True, check=True)
        out.extend(json.loads(res.stdout)["screens"])
    return out


def compare_js(art: mdl.Artifact, art_dir: Path, screens: list[dict], work: Path) -> dict:
    """JS against numpy on every node: identical argmax and pointer, logits within tolerance."""
    net_path = work / "slop-net.bin"
    net_path.write_bytes(net_compile.compile_artifact(art, (art_dir / "vocab.txt").read_bytes()))
    js = js_predict(net_path, screens)
    nodes = cls_diff = ptr_diff = logit_bad = 0
    worst = 0.0
    examples = []
    for s, js_preds in zip(screens, js):
        for a, b in zip(mdl.predict_screen(art, s), js_preds):
            nodes += 1
            rel = max(abs(x - y) / max(1.0, abs(x)) for x, y in zip(a.logits, b["logits"]))
            worst = max(worst, rel)
            bad = False
            if a.cls != b["cls"]:
                cls_diff += 1
                bad = True
            if a.pointer != b["pointer"]:
                ptr_diff += 1
                bad = True
            if rel > LOGIT_TOL:
                logit_bad += 1
                bad = True
            if bad and len(examples) < 10:
                examples.append({"screen": s.get("id"), "node": a.index, "numpy": [a.cls, a.pointer], "js": [b["cls"], b["pointer"]], "rel": rel})
    return {"nodes": nodes, "class_differs": cls_diff, "pointer_differs": ptr_diff, "logits_out_of_tolerance": logit_bad,
            "worst_relative_logit_difference": worst, "examples": examples,
            "pass": cls_diff == 0 and ptr_diff == 0 and logit_bad == 0}


def compare_floor(art: mdl.Artifact, screens: list[dict]) -> dict:
    nodes = agree = agree_near = 0
    floor_flags = net_flags = 0
    misses = []
    for s in screens:
        enc = tk.encode_screen(s, art.vocab)
        preds = mdl.predict_encoded(art, enc)
        full = floor_screen(enc)
        near = floor_screen(enc, build_floor_net.RANGE)
        for p, f, g in zip(preds, full, near):
            nodes += 1
            floor_flags += f.cls != "KEEP"
            net_flags += p.cls != "KEEP"
            ok = p.cls == f.cls and p.pointer == f.pointer
            agree += ok
            agree_near += p.cls == g.cls and p.pointer == g.pointer
            if not ok and len(misses) < 20:
                n = enc.nodes[p.index]
                misses.append({"screen": s.get("id"), "node": p.index, "role": n.role, "text": s["nodes"][p.index]["text"][:60],
                               "network": [p.cls, p.pointer], "floor": [f.cls, f.pointer]})
    return {"nodes": nodes, "agree": agree, "agreement": agree / nodes if nodes else 1.0,
            "agree_within_range": agree_near, "agreement_within_range": agree_near / nodes if nodes else 1.0,
            "floor_flags": floor_flags, "network_flags": net_flags, "disagreements": misses}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="*")
    ap.add_argument("--art", help="artefact directory (default: build the floor network afresh)")
    ap.add_argument("--min", type=float, default=0.99, help="required agreement with the floor (default 0.99)")
    ap.add_argument("--js", action="store_true", help="also check lint/runtime.js against numpy on every node")
    ap.add_argument("--random", type=int, default=0, help="add N seeded random screens")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="write the report here as JSON")
    args = ap.parse_args(argv)
    screens = load_inputs(args.inputs) + random_screens(args.random, args.seed)
    if not screens:
        ap.error("no screens")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        art_dir = Path(args.art) if args.art else work / "floor_net"
        if not args.art:
            build_floor_net.build(art_dir)
        art = mdl.load_artifact(art_dir)
        report: dict = {"artifact_sha256": art.sha256, "screens": len(screens), "floor": compare_floor(art, screens)}
        if args.js:
            report["js"] = compare_js(art, art_dir, screens, work)
    fl = report["floor"]
    ok = fl["agreement"] >= args.min and (not args.js or report["js"]["pass"])
    report["pass"] = ok
    print(f"artefact {report['artifact_sha256'][:16]}  screens {len(screens)}  nodes {fl['nodes']}")
    print(f"floor agreement (class and pointer): {fl['agree']}/{fl['nodes']} = {fl['agreement']:.4%}  (min {args.min:.0%})")
    print(f"within {build_floor_net.RANGE} nodes: {fl['agree_within_range']}/{fl['nodes']} = {fl['agreement_within_range']:.4%}")
    print(f"flags: floor {fl['floor_flags']}, network {fl['network_flags']}")
    for m in fl["disagreements"]:
        print(f"  {m['screen']} node {m['node']} {m['role']} {m['text']!r}: network {m['network']}, floor {m['floor']}")
    if args.js:
        js = report["js"]
        print(f"JS vs numpy: {js['nodes']} nodes, class differs {js['class_differs']}, pointer differs {js['pointer_differs']}, "
              f"logits out of tolerance {js['logits_out_of_tolerance']}, worst relative difference {js['worst_relative_logit_difference']:.3e}")
    print("PASS" if ok else "FAIL")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
