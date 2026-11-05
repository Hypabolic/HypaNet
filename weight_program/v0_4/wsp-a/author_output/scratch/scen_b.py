"""Scenario set B: 45 recipes x 2 name tables = 90 sessions, with class transitions and balanced nuisance
(each class with and without error lines / hunk lines)."""
from mk2 import (Session, Ctx, explore, fix_and_pass, incremental_repair, transient_http, polls_ok, diagnosed_repair,
                 retry_active, stall_tests, stall_reads, stall_search, stall_noop_edits, stall_http, stall_polls,
                 stall_build, osc_ab, osc_revert, osc_write2, osc_abc, flaky_once, plan)

RECIPES = {
    # productive-dominant
    "P1": lambda c: (explore(c, 4), fix_and_pass(c, with_build=True), explore(c, 2, j0=4)),
    "P2": lambda c: (incremental_repair(c, 2), explore(c, 2), fix_and_pass(c, j=1)),
    "P3": lambda c: (explore(c, 2), fix_and_pass(c), fix_and_pass(c, j=1), plan(c), explore(c, 2, j0=2)),
    "P4": lambda c: (explore(c, 6), plan(c, 1)),
    "P5": lambda c: (transient_http(c, 1), explore(c, 5), fix_and_pass(c)),
    "P6": lambda c: (diagnosed_repair(c, "build"), explore(c, 4), fix_and_pass(c, j=1)),
    "P7": lambda c: (stall_tests(c, 5), explore(c, 3), fix_and_pass(c), explore(c, 2, j0=3)),
    "P8": lambda c: (osc_ab(c, 4, "tests"), explore(c, 3), fix_and_pass(c, with_build=True)),
    "P9": lambda c: (polls_ok(c, 2), explore(c, 3), fix_and_pass(c), explore(c, 2, j0=3)),
    "P10": lambda c: (fix_and_pass(c, with_build=True), fix_and_pass(c, j=1, with_build=True), fix_and_pass(c, j=2)),
    "P11": lambda c: (explore(c, 3), incremental_repair(c, 3, j=1), explore(c, 2, j0=3)),
    # recovery-dominant
    "R1": lambda c: (explore(c, 2), transient_http(c, 2), explore(c, 2, j0=2)),
    "R2": lambda c: (explore(c, 1), fix_and_pass(c), polls_ok(c, 4), explore(c, 1, j0=1)),
    "R3": lambda c: (explore(c, 2), diagnosed_repair(c, "build"), fix_and_pass(c, j=1)),
    "R4": lambda c: (explore(c, 1), diagnosed_repair(c, "tests"), explore(c, 2, j0=1)),
    "R5": lambda c: (explore(c, 3), retry_active(c, 2, "http")),
    "R6": lambda c: (explore(c, 2), retry_active(c, 3, "poll")),
    "R7": lambda c: (fix_and_pass(c), flaky_once(c), explore(c, 2)),
    "R8": lambda c: (explore(c, 2), diagnosed_repair(c, "lint"), explore(c, 1, j0=2)),
    "R9": lambda c: (stall_reads(c, 6), transient_http(c, 2), explore(c, 1)),
    "R10": lambda c: (explore(c, 2), retry_active(c, 2, "read")),
    "R11": lambda c: (explore(c, 2), transient_http(c, 3, "GET", j=1), fix_and_pass(c)),
    # stalled-dominant
    "S1": lambda c: (explore(c, 2), stall_tests(c, 7)),
    "S2": lambda c: (explore(c, 1), stall_reads(c, 8)),
    "S3": lambda c: (explore(c, 2), stall_noop_edits(c, 4, "tests")),
    "S4": lambda c: (explore(c, 1), stall_http(c, 9)),
    "S5": lambda c: (fix_and_pass(c), stall_polls(c, 11)),
    "S6": lambda c: (explore(c, 2), stall_build(c, 7, True)),
    "S7": lambda c: (explore(c, 1), stall_search(c, 7)),
    "S8": lambda c: (explore(c, 2), stall_reads(c, 8, two=True)),
    "S9": lambda c: (explore(c, 1), stall_noop_edits(c, 4, "build")),
    "S10": lambda c: (explore(c, 2), stall_tests(c, 7, with_read=True)),
    "S11": lambda c: (transient_http(c, 1), explore(c, 1), stall_tests(c, 8)),
    "S12": lambda c: (osc_ab(c, 2, "tests"), stall_tests(c, 7)),
    "S13": lambda c: (explore(c, 3), stall_build(c, 7, False, "lint")),
    # oscillating-dominant
    "O1": lambda c: (explore(c, 2), osc_ab(c, 5, "tests")),
    "O2": lambda c: (explore(c, 1), osc_revert(c, 4, None)),
    "O3": lambda c: (explore(c, 2), osc_write2(c, 3)),
    "O4": lambda c: (explore(c, 1), osc_ab(c, 5, "build_ok")),
    "O5": lambda c: (explore(c, 2), osc_abc(c)),
    "O6": lambda c: (explore(c, 1), osc_revert(c, 3, "build_ok")),
    "O7": lambda c: (explore(c, 2), osc_ab(c, 5, "lint")),
    "O8": lambda c: (stall_tests(c, 4), osc_ab(c, 4, "tests")),
    "O9": lambda c: (explore(c, 1), osc_revert(c, 4, "tests")),
    "O10": lambda c: (fix_and_pass(c), osc_ab(c, 5, None)),
    "O11": lambda c: (diagnosed_repair(c, "build"), osc_revert(c, 4, None)),
}


def build(sid):
    name, _ = sid.split("_")
    s = Session(sid, start=["2026-03-01T09:14:03.250+00:00", "2026-03-02T14:02:11.000+00:00", "2026-03-03T22:40:59.500+00:00"][len(sid) % 3])
    c = Ctx(s)
    RECIPES[name](c)
    return s


def all_sessions(variants=("a", "b")):
    return [build(f"{name}_{v}") for name in RECIPES for v in variants]
