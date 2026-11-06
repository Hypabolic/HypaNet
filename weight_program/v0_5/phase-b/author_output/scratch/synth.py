"""Synthetic sanity checks of the revised network on hand-built RawOp sessions rendered by the packet renderer.
Prints the class at every event's EOE and the key event features. Usage: ./py synth.py [edits]"""
import sys

import torch

import model
import renderer
import v05_load

CN = model.CLASS_NAMES
TS = "2026-03-01T09:{:02d}:{:02d}.000+00:00"


def op(family, target=None, status="SUCCESS", args=None, validation=None, changed=None, err=None, i=0):
    d = {"family": family, "target": target, "args": args or {}, "outcome": status, "validation": validation,
         "changed": changed or [], "input_tokens": 1000 + 7 * i, "output_tokens": 50 + i,
         "timestamp": TS.format(i // 60, i % 60), "dropped": [], "extra": None,
         "error_class": None, "error_location": None}
    if err:
        d["error_class"], d["error_location"] = err
    return d


def chg(path, before, after):
    return {"path": path, "before": before, "after": after}


def run(name, ops, sid, edits):
    m = v05_load.apply_edits_file(edits)[0]
    blocks = renderer.render_session(ops, sid)
    act = v05_load.activations(m, blocks, query_positions=[])
    eoe = act["eoe_index"]
    r2 = act["resid_mlp"][2]
    r3 = act["resid_mlp"][3]
    print(f"=== {name}")
    for i, p in enumerate(eoe):
        hdr = blocks[i].split("\n")[0].split(" ", 1)[1][:60]
        lg = act["logits"][p]
        feats = f"INEFF={float(r2[p, 75]):.2f} RES={float(r2[p, 76]):.2f} INPOL={float(r2[p, 77]):.2f} " \
                f"A1={float(r3[p, 79]):.2f} A2={float(r3[p, 80]):.2f} act={float(r3[p, 81]):.2f} resw={float(r3[p, 82]):.2f}"
        print(f" {i:2d} {CN[int(torch.argmax(lg))][:4]} {hdr:60s} {feats}")


def main():
    edits = sys.argv[1] if len(sys.argv) > 1 else None
    A = "line1\nline2\nline3\n"
    B = "line1\nchanged\nline3\n"
    C = "line1\nother\nline3\n"
    # 1. A/B/A/B oscillation across two files, with a 7-file revert (TRUNC line) and tests between
    ops = [op("read_file", "src/a.py", i=0), op("read_file", "src/b.py", i=1)]
    files = [f"src/f{k}.py" for k in range(7)]
    for i in range(2, 14):
        if i % 2 == 0:
            ops.append(op("edit_file", "src/a.py", changed=[chg("src/a.py", A if i % 4 == 2 else B, B if i % 4 == 2 else A)], i=i))
        else:
            ops.append(op("run_tests", "unit", status="ERROR", validation={"suite": "unit", "passed": 3, "total": 8},
                          err=("AssertionError", "tests/test_a.py::test_x"), i=i))
    ops.append(op("revert", files, changed=[chg(f, B, A) for f in files], i=14))
    ops.append(op("revert", files, changed=[chg(f, A, B) for f in files], i=15))
    ops.append(op("revert", files, changed=[chg(f, B, A) for f in files], i=16))
    run("oscillation + multi-file reverts", ops, "s1", edits)
    # 2. long stalled session: same failure 12 times with reads between, then exploration
    ops = []
    for i in range(30):
        if i % 3 == 2:
            ops.append(op("read_file", "src/x.py", i=i))
        else:
            ops.append(op("run_tests", "unit", status="ERROR", validation={"suite": "unit", "passed": 5, "total": 9},
                          err=("TypeError", "tests/test_x.py::test_y"), i=i))
    for i in range(30, 38):
        ops.append(op("read_file", f"src/new{i}.py", i=i))
    run("long stall then fresh reads", ops, "s2", edits)
    # 3. http retries with a port in the URL, resolution, then a search query containing 'error'
    ops = [op("list_dir", "src", i=0), op("search", "src", args={"query": "error handling"}, i=1)]
    for i in range(2, 5):
        ops.append(op("http_call", {"method": "GET", "host": "localhost:8080", "path_template": "/v1/jobs/{id}"},
                      status="ERROR", err=("HTTP502", "localhost:8080/v1/jobs/{id}"), i=i))
    ops.append(op("http_call", {"method": "GET", "host": "localhost:8080", "path_template": "/v1/jobs/{id}"}, i=5))
    ops += [op("read_file", "src/y.py", i=6), op("edit_file", "src/y.py", changed=[chg("src/y.py", A, B)], i=7),
            op("run_tests", "unit", validation={"suite": "unit", "passed": 9, "total": 9}, i=8),
            op("search", "src", args={"query": "error handling"}, i=9),
            op("read_file", "src/z.py", i=10), op("read_file", "src/w.py", i=11)]
    run("http retries with port, resolution, 'error' in a query", ops, "s3", edits)
    # 4. productive repair: failing counts improve, then green; then 6 polls (in policy) and ready
    ops = [op("read_file", "src/p.py", i=0)]
    for k, passed in enumerate((2, 5, 7)):
        ops.append(op("edit_file", "src/p.py", changed=[chg("src/p.py", A + str(k), A + str(k + 1))], i=1 + 2 * k))
        ops.append(op("run_tests", "unit", status="ERROR", validation={"suite": "unit", "passed": passed, "total": 9},
                      err=("ValueError", f"tests/test_p.py::test_{passed}"), i=2 + 2 * k))
    ops.append(op("edit_file", "src/p.py", changed=[chg("src/p.py", A + "3", A + "4")], i=7))
    ops.append(op("run_tests", "unit", validation={"suite": "unit", "passed": 9, "total": 9}, i=8))
    for i in range(9, 15):
        ops.append(op("poll_status", "deploy/job/1", status="NEUTRAL", args={"backoff": 2 ** (i - 9)}, i=i))
    ops.append(op("poll_status", "deploy/job/1", i=15))
    ops.append(op("read_file", "docs/x.md", i=16))
    run("productive repair, then polls within policy, ready", ops, "s4", edits)
    # 5. polls beyond policy: 9 pendings
    ops = [op("read_file", "a.md", i=0)]
    for i in range(1, 10):
        ops.append(op("poll_status", "ci/run/9", status="NEUTRAL", args={"backoff": 30}, i=i))
    run("polls beyond policy", ops, "s5", edits)


if __name__ == "__main__":
    main()
