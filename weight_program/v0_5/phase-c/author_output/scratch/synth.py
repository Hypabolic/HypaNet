"""Synthetic sanity checks of the revised network on hand-built RawOp sessions rendered by the packet renderer.
Prints the class at every event's EOE and the key event features. Usage: ./py synth.py [edits|-]"""
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


def tf(passed, total, loc, i, suite="unit"):
    return op("run_tests", suite, status="ERROR", validation={"suite": suite, "passed": passed, "total": total},
              err=("AssertionError", loc), i=i)


OUTF = None


def emit(line):
    print(line)
    if OUTF is not None:
        OUTF.write(line + "\n")


def run(name, ops, sid, m):
    blocks = renderer.render_session(ops, sid)
    act = v05_load.activations(m, blocks, query_positions=[])
    eoe = act["eoe_index"]
    r2 = act["resid_mlp"][2]
    r3 = act["resid_mlp"][3]
    emit(f"=== {name}")
    for i, p in enumerate(eoe):
        hdr = blocks[i].split("\n")[0].split(" ", 1)[1][:58]
        body = " ".join(ln.strip()[:30] for ln in blocks[i].split("\n")[1:2])
        lg = act["logits"][p]
        feats = (f"INEFF={float(r2[p, 75]):.1f} RES={float(r2[p, 76]):.1f} INPOL={float(r2[p, 77]):.1f} FRESH={float(r2[p, 94]):.1f} "
                 f"A1={float(r3[p, 79]):.2f} A2={float(r3[p, 80]):.2f} act={float(r3[p, 81]):.1f} resw={float(r3[p, 82]):.1f} fresh={float(r3[p, 112]):.1f}")
        emit(f" {i:2d} {CN[int(torch.argmax(lg))][:4]} {hdr:58s} {body:30s} {feats}")


def main():
    global OUTF
    edits = sys.argv[1] if len(sys.argv) > 1 else "-"
    m = v05_load.load_b() if edits == "-" else v05_load.apply_edits_file(edits)[0]
    import os
    OUTF = open(os.path.join(v05_load.OUTPUT, "scratch", "synth_out.txt"), "w")
    A = "line1\nline2\nline3\n"
    B = "line1\nchanged\nline3\n"
    # 1. W-style incremental repair: write contract_k (real edit) -> FAILED with the count increasing, same error line
    ops = [op("read_file", "lib/c0.ex", i=0)]
    for k in range(1, 8):
        ops.append(op("write_file", f"lib/c{k}.ex", changed=[chg(f"lib/c{k}.ex", A, B)], i=2 * k - 1))
        ops.append(tf(k, 12, "lib/c0.ex", 2 * k))
    run("W-style repair (counts 1..7, same line): expect PRODUCTIVE throughout", ops, "s1", m)
    # 2. W-style stall: append-writes (h0) to one file, FAILED 1/12 unchanged
    ops = [op("read_file", "lib/c3.ex", i=0)]
    for k in range(1, 8):
        ops.append(op("write_file", "lib/c3.ex", changed=[chg("lib/c3.ex", A + "x" * k + "\n", A + "x" * k + "\n" + "y" * k + "\n")], i=2 * k - 1))
        ops.append(tf(1, 12, "lib/c0.ex", 2 * k))
    run("W-style append stall (h0 writes, count unchanged): expect STALLED from ~event 9", ops, "s2", m)
    # 3. W-style: progress failure, fresh read, then 4 identical re-runs, then 4 more
    ops = [op("write_file", "lib/c1.ex", changed=[chg("lib/c1.ex", A, B)], i=0), tf(2, 12, "lib/c0.ex", 1),
           op("write_file", "lib/c2.ex", changed=[chg("lib/c2.ex", A, B)], i=2), tf(3, 12, "lib/c0.ex", 3),
           op("read_file", "lib/c2.ex", i=4)]
    for i in range(5, 13):
        ops.append(tf(3, 12, "lib/c0.ex", i))
    run("W-style retries after a progress failure: events 5-8 RECOVERY (retries 1-3 at 6-8), STALLED from 12", ops, "s3", m)
    # 4. handoff plans after progress, then retries
    ops = [op("write_file", "lib/c1.ex", changed=[chg("lib/c1.ex", A, B)], i=0), tf(3, 12, "lib/c0.ex", 1)]
    for i in range(2, 7):
        ops.append(op("plan", None, args={"action_type": f"handoff:{i}:{i - 2}"}, i=i))
    for i in range(7, 11):
        ops.append(tf(3, 12, "lib/c0.ex", i))
    run("handoff plans then 4 re-runs: handoffs not ineffective; 8-10 RECOVERY", ops, "s4", m)
    # 5. record:/isolate: plans with paths, fresh reads, real edits (productive exploration)
    ops = []
    for k in range(8):
        ops.append(op("read_file", f"lib/c{k}.ex", i=3 * k))
        ops.append(op("plan", None, args={"action_type": f"isolate:lib/c{k}.ex"}, i=3 * k + 1))
        ops.append(op("write_file", f"lib/c{k}.ex", changed=[chg(f"lib/c{k}.ex", A, B)], i=3 * k + 2))
    run("read/isolate/write of fresh files: expect PRODUCTIVE", ops, "s5", m)
    # 6. V-style A/B/A/B oscillation with tests between, then a 7-file revert
    ops = [op("read_file", "src/a.py", i=0), op("read_file", "src/b.py", i=1)]
    for i in range(2, 14):
        if i % 2 == 0:
            ops.append(op("edit_file", "src/a.py", changed=[chg("src/a.py", A if i % 4 == 2 else B, B if i % 4 == 2 else A)], i=i))
        else:
            ops.append(tf(3, 8, "tests/test_a.py::test_x", i))
    files = [f"src/f{k}.py" for k in range(7)]
    ops.append(op("revert", files, changed=[chg(f, B, A) for f in files], i=14))
    ops.append(op("revert", files, changed=[chg(f, A, B) for f in files], i=15))
    ops.append(op("revert", files, changed=[chg(f, B, A) for f in files], i=16))
    run("V-style oscillation + multi-file reverts: OSCILLATING from the second reversion", ops, "s6", m)
    # 7. V-style long stall: same failure 12 times with re-reads between, then 8 fresh reads
    ops = []
    for i in range(30):
        if i % 3 == 2:
            ops.append(op("read_file", "src/x.py", i=i))
        else:
            ops.append(tf(5, 9, "tests/test_x.py::test_y", i))
    for i in range(30, 38):
        ops.append(op("read_file", f"src/new{i}.py", i=i))
    run("V-style long stall (re-reads between), then fresh reads: RECOVERY 1-3, STALLED later, PRODUCTIVE after reads", ops, "s7", m)
    # 8. http retries with a port, resolution, search with 'error'
    ops = [op("list_dir", "src", i=0), op("search", "src", args={"query": "error handling"}, i=1)]
    for i in range(2, 5):
        ops.append(op("http_call", {"method": "GET", "host": "localhost:8080", "path_template": "/v1/jobs/{id}"},
                      status="ERROR", err=("HTTP502", "localhost:8080/v1/jobs/{id}"), i=i))
    ops.append(op("http_call", {"method": "GET", "host": "localhost:8080", "path_template": "/v1/jobs/{id}"}, i=5))
    ops += [op("read_file", "src/y.py", i=6), op("edit_file", "src/y.py", changed=[chg("src/y.py", A, B)], i=7),
            op("run_tests", "unit", validation={"suite": "unit", "passed": 9, "total": 9}, i=8),
            op("search", "src", args={"query": "error handling"}, i=9),
            op("read_file", "src/z.py", i=10), op("read_file", "src/w.py", i=11)]
    run("http retries with port, resolution, 'error' in a query: RECOVERY 3-9, then PRODUCTIVE", ops, "s8", m)
    # 9. productive repair then polls within policy, ready
    ops = [op("read_file", "src/p.py", i=0)]
    for k, passed in enumerate((2, 5, 7)):
        ops.append(op("edit_file", "src/p.py", changed=[chg("src/p.py", A + str(k), A + str(k + 1))], i=1 + 2 * k))
        ops.append(tf(passed, 9, f"tests/test_p.py::test_{passed}", 2 + 2 * k))
    ops.append(op("edit_file", "src/p.py", changed=[chg("src/p.py", A + "3", A + "4")], i=7))
    ops.append(op("run_tests", "unit", validation={"suite": "unit", "passed": 9, "total": 9}, i=8))
    for i in range(9, 15):
        ops.append(op("poll_status", "deploy/job/1", status="NEUTRAL", args={"backoff": 2 ** (i - 9)}, i=i))
    ops.append(op("poll_status", "deploy/job/1", i=15))
    ops.append(op("read_file", "docs/x.md", i=16))
    run("V-style repair, six polls in policy, ready: PRODUCTIVE, RECOVERY during polls and after ready", ops, "s9", m)
    # 10. polls beyond policy
    ops = [op("read_file", "a.md", i=0)]
    for i in range(1, 12):
        ops.append(op("poll_status", "ci/run/9", status="NEUTRAL", args={"backoff": 30}, i=i))
    run("polls beyond policy: RECOVERY up to the 6th pending, then not RECOVERY; STALLED once 5 are out of policy", ops, "s10", m)
    # 11. V-style scenario 6: edits that change files but the failure stays identical (same line and count)
    ops = [op("read_file", "src/q.py", i=0), tf(8, 11, "tests/test_q.py::test_z", 1)]
    for k in range(2, 14, 2):
        ops.append(op("edit_file", "src/q.py", changed=[chg("src/q.py", A + str(k), A + str(k + 1))], i=k))
        ops.append(tf(8, 11, "tests/test_q.py::test_z", k + 1))
    run("V-style scenario 6 (edit, same failure): STALLED from ~event 9", ops, "s11", m)
    # 12. repeated plan actions (V-style single-token plans) + re-reads
    ops = [op("read_file", "src/r.py", i=0), op("read_file", "src/r.py", i=1)]
    for i in range(2, 10):
        ops.append(op("plan", None, args={"action_type": "checkpoint" if i % 2 else "review_diff"}, i=i))
    run("repeated single-token plans: STALLED once 5 repeats accumulate", ops, "s12", m)
    # 13. writes to many different files after a repeated failure (no append, no earlier mutation): EDR only within 3 events
    ops = [tf(2, 9, "tests/test_m.py::test_a", 0), tf(2, 9, "tests/test_m.py::test_a", 1)]
    for k in range(2, 10):
        ops.append(op("write_file", f"src/m{k}.py", changed=[chg(f"src/m{k}.py", A, B)], i=k))
    run("fresh writes after a repeated failure: EDR for the first 2 writes only, then PRODUCTIVE", ops, "s13", m)
    # 14. a revert of an edited file (real hunk) and an edit, no h0: REPMUT but no H0 -> not ineffective by itself
    ops = [op("read_file", "src/r.py", i=0), op("edit_file", "src/r.py", changed=[chg("src/r.py", A, B)], i=1),
           op("run_tests", "unit", validation={"suite": "unit", "passed": 9, "total": 9}, i=2),
           op("revert", ["src/r.py"], changed=[chg("src/r.py", B, A)], i=3),
           op("read_file", "src/s.py", i=4), op("edit_file", "src/s.py", changed=[chg("src/s.py", A, B)], i=5),
           op("edit_file", "src/r.py", changed=[chg("src/r.py", A, B)], i=6),
           op("run_tests", "unit", validation={"suite": "unit", "passed": 9, "total": 9}, i=7),
           op("read_file", "src/t.py", i=8)]
    run("edit, ok, revert, edit again (one reversion), repeated ok count: PRODUCTIVE", ops, "s14", m)
    OUTF.close()


if __name__ == "__main__":
    main()
