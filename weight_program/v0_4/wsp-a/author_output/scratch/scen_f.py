"""Scenario set F: second fresh evaluation set (new vocabularies, new recipes, longer sessions), modelled on the
spec's scenario families 1-11."""
from mk2 import (Session, Ctx, explore, fix_and_pass, incremental_repair, transient_http, polls_ok, diagnosed_repair,
                 retry_active, stall_tests, stall_reads, stall_search, stall_noop_edits, stall_http, stall_polls,
                 stall_build, osc_ab, osc_revert, osc_write2, osc_abc, flaky_once, plan)
from mk1 import hnum

PROJECTS_F = [
    {"paths": ["kernel/sched/rt.c", "kernel/sched/fair.c", "kernel/sched/core.c", "include/sched.h", "kernel/time/tick.c", "kernel/sched/debug.c"],
     "dirs": ["kernel/sched", "kernel"], "suites": ["sched_kunit", "boot"], "targets": ["vmlinux"], "host": "kbuild.farm.example",
     "errs": ["LockdepWarning", "NullPtrDeref", "KasanReport"], "res": "boot_vm_12"},
    {"paths": ["site/pages/index.tsx", "site/pages/blog/[slug].tsx", "site/components/Nav.tsx", "site/lib/md.ts", "site/styles/base.css", "site/next.config.js"],
     "dirs": ["site/pages", "site"], "suites": ["site_unit", "lighthouse"], "targets": ["site_build"], "host": "cdn.deploy.example",
     "errs": ["HydrationError", "ModuleNotFound", "CSSParseError"], "res": "preview_deploy"},
    {"paths": ["data/etl/extract.py", "data/etl/transform.py", "data/etl/load.py", "data/schemas/orders.json", "data/etl/dq.py", "data/etl/cli.py"],
     "dirs": ["data/etl", "data"], "suites": ["etl", "dq_checks"], "targets": ["etl_image"], "host": "warehouse.query.example",
     "errs": ["SchemaDrift", "DuplicateKey", "NullInNotNull"], "res": "nightly_run_88"},
    {"paths": ["fw/src/uart.c", "fw/src/spi.c", "fw/src/main.c", "fw/include/board.h", "fw/src/dma.c", "fw/src/clock.c"],
     "dirs": ["fw/src", "fw"], "suites": ["hw_in_loop", "unit_host"], "targets": ["fw_bin"], "host": "flash.rig.example",
     "errs": ["HardFault", "WatchdogReset", "BrownOut"], "res": "rig_slot_3"},
    {"paths": ["plib/zim/norg.pl", "plib/zim/vex.pl", "plib/crun.pl", "plib/zim/tarn.pl", "plib/hask.pl", "plib/zim/oblo.pl"],
     "dirs": ["plib/zim", "plib"], "suites": ["epsilon", "zeta"], "targets": ["zim_bundle"], "host": "norghost.zim",
     "errs": ["VexError", "TarnFault", "ObloPanic"], "res": "zim_churn"},
    {"paths": ["game/ai/pathfind.cs", "game/ai/behaviour.cs", "game/net/replicate.cs", "game/core/tick.cs", "game/ai/sensor.cs", "game/ui/hud.cs"],
     "dirs": ["game/ai", "game"], "suites": ["ai_sim", "playtest"], "targets": ["game_client"], "host": "match.lobby.example",
     "errs": ["NavMeshMissing", "DesyncDetected", "StackOverflow"], "res": "build_farm_7"},
]

RECIPES = {
    "F1": lambda c: (explore(c, 2), incremental_repair(c, 4), explore(c, 3, j0=2), fix_and_pass(c, with_build=True)),
    "F2": lambda c: (explore(c, 6), plan(c), explore(c, 3, j0=6), fix_and_pass(c)),
    "F3": lambda c: (fix_and_pass(c), transient_http(c, 3), explore(c, 4), fix_and_pass(c, j=1)),
    "F4": lambda c: (explore(c, 3), polls_ok(c, 2), explore(c, 2, j0=3), retry_active(c, 2, "http")),
    "F5": lambda c: (explore(c, 2), fix_and_pass(c), stall_tests(c, 9), explore(c, 3, j0=2)),
    "F6": lambda c: (explore(c, 2), stall_noop_edits(c, 6, "tests")),
    "F7": lambda c: (explore(c, 3), osc_ab(c, 8, "tests"), fix_and_pass(c, j=2)),
    "F8": lambda c: (explore(c, 2), osc_revert(c, 5, "tests"), explore(c, 2, j0=2)),
    "F9": lambda c: (explore(c, 1), incremental_repair(c, 2), incremental_repair(c, 3, j=1)),
    "F10": lambda c: (explore(c, 2), polls_ok(c, 5), explore(c, 2, j0=2), polls_ok(c, 3)),
    "F11": lambda c: (explore(c, 4), stall_search(c, 9), stall_reads(c, 8)),
    "F12": lambda c: (diagnosed_repair(c, "lint"), osc_ab(c, 6, "lint"), explore(c, 2)),
    "F13": lambda c: (explore(c, 2), stall_http(c, 12), explore(c, 3, j0=2)),
    "F14": lambda c: (fix_and_pass(c), osc_write2(c, 4), fix_and_pass(c, j=1), explore(c, 2)),
    "F15": lambda c: (explore(c, 3), stall_build(c, 9, False), diagnosed_repair(c, "build"), explore(c, 2, j0=3)),
    "F16": lambda c: (explore(c, 2), flaky_once(c), explore(c, 2, j0=2), osc_abc(c), explore(c, 2, j0=4)),
    "F17": lambda c: (explore(c, 3), osc_ab(c, 6, "build_ok"), stall_tests(c, 7)),
    "F18": lambda c: (explore(c, 2), diagnosed_repair(c, "tests"), explore(c, 5, j0=2), diagnosed_repair(c, "build", j=1)),
}


def build(sid):
    name, v = sid.split("_")
    s = Session(sid, start=["2026-05-11T07:30:00.000+00:00", "2026-05-12T13:05:45.000+00:00", "2026-05-13T20:50:10.000+00:00"][ord(v) % 3])
    c = Ctx(s)
    c.P = PROJECTS_F[hnum(sid, "projF", hi=len(PROJECTS_F))]
    RECIPES[name](c)
    return s


def all_sessions(variants=("a", "b", "c")):
    return [build(f"{name}_{v}") for name in RECIPES for v in variants]
