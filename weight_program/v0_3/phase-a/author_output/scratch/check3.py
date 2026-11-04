"""Iteration 3 inputs: very long session (> 64 events, several 128-row blocks), maximal blocks
(long header with TRUNC, 5 hunk lines + error line), missing timestamp, reserved words inside a
search query, over-budget revert paths, and policy CPU time on a long prefix."""
import sys
import time

import tokenizer
import reference_policy as rp
from check1 import SESSIONS as S1
from check2 import SESSIONS as S2
from common import compare, edit, encode, http, op, read, revert, search, tests

A = "x = 1\ny = 2\n"
B = "x = 1\ny = 3\n"

SESSIONS = {}

# ~90 events: all family sessions back to back (window logic must only use the last 8 / 15 blocks)
very_long = []
for name in ("repair", "explore", "transient", "backoff", "stall_tests", "abab", "edit_revert", "benign_runs",
             "polls_long", "http_many", "reads_repeat", "misc"):
    very_long += S1[name]
very_long += S2["long"]
SESSIONS["very_long"] = very_long

# maximal blocks: run_tests with a huge tests= list (header TRUNC), 7 changed files and a failure
files = [f"pkg/deep/dir/file{i}.qx" for i in range(7)]
maximal = op("run_tests", "integration", {"tests": [f"integration/test_case_{i}" for i in range(30)]},
             outcome="ERROR", error_class="AssertionError", error_location="pkg/deep/dir/file3.qx",
             validation={"suite": "integration", "passed": 11, "total": 40},
             changed=[{"path": p, "before": A, "after": B} for p in files])
maximal_ok = dict(maximal)
maximal_ok.update(outcome="SUCCESS", error_class=None, error_location=None,
                  validation={"suite": "integration", "passed": 40, "total": 40},
                  changed=[{"path": p, "before": B, "after": A} for p in files])
SESSIONS["maximal"] = [maximal, maximal, maximal_ok, maximal, maximal_ok, maximal, read("pkg/a.qx"), maximal]

# missing timestamp on some events (renders as "?", shifting header offsets by 6)
nots = [read("pkg/a.qx"), tests("core", 5, 9), edit("pkg/a.qx", A, B), tests("core", 7, 9)] * 2
for o in nots[1::2]:
    o["_nostamp"] = True
SESSIONS["no_timestamp"] = nots

# reserved words inside search queries / tests lists / actions
SESSIONS["reserved_words"] = [
    search("pkg", "passed error ok FAILED"), search("pkg", "12 passed"), search("pkg", "pending ready"),
    op("plan", None, {"action_type": "error"}), op("ask", None, {"action_type": "passed"}),
    search("pkg", "passed error ok FAILED"), search("pkg", "12 passed"), search("pkg", "passed error ok FAILED"),
    tests("core", 9, 9),
]

# over-budget revert paths (TRUNC inside paths=), repeated
many = [f"pkg/p{i}.qx" for i in range(12)]
rv = revert(many, [{"path": p, "before": B, "after": A} for p in many])
ed = op("edit_file", "pkg/p0.qx", changed=[{"path": p, "before": A, "after": B} for p in many])
SESSIONS["revert_trunc"] = [ed, rv, ed, rv, ed, rv]

if __name__ == "__main__":
    args = sys.argv[1:]
    verbose = True if "-v" in args else "dis"
    only = [a for a in args if a != "-v"] or list(SESSIONS)
    tot_d = tot_n = 0
    for name in only:
        print("=" * 100)
        print("SESSION", name)
        d, n = compare(SESSIONS[name], sid="sid-" + name, verbose=verbose)
        tot_d += d
        tot_n += n
    print("=" * 100)
    print(f"disagreements {tot_d} / {tot_n}")
    # policy CPU time on the longest prefix
    _, enc = encode(SESSIONS["very_long"], "sid-very_long")
    t0 = time.process_time()
    for _ in range(20):
        rp.reference_policy(enc.surface, enc.event_ord)
    print(f"policy cpu per call on {len(enc.eoe_index)} blocks / {len(enc.tokens)} tokens: "
          f"{(time.process_time() - t0) / 20 * 1000:.2f} ms")
