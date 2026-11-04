"""Iteration 1 inputs: one rendered session per scenario family, network vs reference policy."""
import sys

from common import (ask, build, compare, edit, http, lint, list_dir, plan, poll, read, revert, search,
                    tests, write)

A = "x = 1\ny = 2\n"
B = "x = 1\ny = 3\n"
C = "x = 1\ny = 4\n"
D = "x = 2\ny = 4\n"

SESSIONS = {}

# 1 incremental productive repair
SESSIONS["repair"] = [
    tests("core", 5, 9), read("pkg/core.qx"), edit("pkg/core.qx", A, B), tests("core", 7, 9),
    read("pkg/util.qx"), edit("pkg/util.qx", A, B), tests("core", 9, 9), lint("core"),
]
# 2 productive exploration
SESSIONS["explore"] = [
    read("pkg/a.qx"), read("pkg/b.qx"), search("pkg", "frobnicate widget"), list_dir("pkg/sub"),
    read("pkg/sub/c.qx"), edit("pkg/sub/c.qx", A, B), build("pkg"), tests("core", 9, 9),
]
# 3 transient external failure
SESSIONS["transient"] = [
    read("pkg/a.qx"), http("GET", "api.example", "/v1/items", ok=False), http("GET", "api.example", "/v1/items", ok=False),
    http("GET", "api.example", "/v1/items", ok=True), read("pkg/b.qx"), edit("pkg/b.qx", A, B), tests("core", 9, 9),
    read("pkg/c.qx"), edit("pkg/c.qx", A, B),
]
# 4 bounded backoff
SESSIONS["backoff"] = [
    http("POST", "ci.example", "/jobs", ok=True), poll("job-17", "pending", 1), poll("job-17", "pending", 2),
    poll("job-17", "pending", 4), poll("job-17", "ready"), read("pkg/a.qx"), edit("pkg/a.qx", A, B),
    tests("core", 9, 9), read("pkg/b.qx"), read("pkg/c.qx"),
]
# 5 repeated failed test invocation
SESSIONS["stall_tests"] = [tests("core", 5, 9)] * 9
# 5b with reads of the same file between
SESSIONS["stall_tests_reads"] = [tests("core", 5, 9), read("pkg/core.qx")] * 5
# 6 repeated edits that don't improve validation (distinct contents)
SESSIONS["edits_noimprove"] = [
    edit("pkg/core.qx", A, B), tests("core", 5, 9), edit("pkg/core.qx", B, C), tests("core", 5, 9),
    edit("pkg/core.qx", C, D), tests("core", 5, 9), edit("pkg/core.qx", D, "z\n"), tests("core", 5, 9),
]
# 7 A/B/A/B oscillation
SESSIONS["abab"] = [
    edit("pkg/core.qx", A, B), tests("core", 5, 9), edit("pkg/core.qx", B, A), tests("core", 5, 9),
    edit("pkg/core.qx", A, B), tests("core", 5, 9), edit("pkg/core.qx", B, A), tests("core", 5, 9),
]
# 8 tool oscillation: edit / revert
SESSIONS["edit_revert"] = [
    edit("pkg/core.qx", A, B), revert(["pkg/core.qx"], [{"path": "pkg/core.qx", "before": B, "after": A}]),
    edit("pkg/core.qx", A, B), revert(["pkg/core.qx"], [{"path": "pkg/core.qx", "before": B, "after": A}]),
    edit("pkg/core.qx", A, B), tests("core", 9, 9),
]
# 9 benign lookalike: repeated runs with improvement
SESSIONS["benign_runs"] = [
    tests("core", 3, 9), edit("pkg/core.qx", A, B), tests("core", 5, 9), edit("pkg/core.qx", B, C),
    tests("core", 7, 9), edit("pkg/core.qx", C, D), tests("core", 9, 9), read("pkg/x.qx"),
]
# 10 polls while a dependency legitimately recovers, then too many polls
SESSIONS["polls_long"] = [http("POST", "ci.example", "/jobs", ok=True)] + [poll("job-9", "pending", 2)] * 9 + [poll("job-9", "ready")]
# out-of-policy http retries
SESSIONS["http_many"] = [http("GET", "api.example", "/v1/items", ok=False)] * 10
# 11 rising cost without useful work: repeated reads
SESSIONS["reads_repeat"] = [read("pkg/core.qx")] * 8
# misc: plan / ask / write / no-op edit
SESSIONS["misc"] = [
    plan(), read("pkg/a.qx"), write("pkg/new.qx", None, A), edit("pkg/new.qx", A, A), edit("pkg/new.qx", A, A),
    edit("pkg/new.qx", A, A), edit("pkg/new.qx", A, A), edit("pkg/new.qx", A, A), ask(),
]

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
