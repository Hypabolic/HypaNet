"""Scenario set E: fresh evaluation sessions with new vocabularies and new multi-phase recipes (not used for
deriving any direction)."""
from mk2 import (Session, Ctx, explore, fix_and_pass, incremental_repair, transient_http, polls_ok, diagnosed_repair,
                 retry_active, stall_tests, stall_reads, stall_search, stall_noop_edits, stall_http, stall_polls,
                 stall_build, osc_ab, osc_revert, osc_write2, osc_abc, flaky_once, plan)
from mk1 import hnum

PROJECTS_E = [
    {"paths": ["engine/render/mesh.cpp", "engine/render/shader.cpp", "engine/render/camera.cpp", "engine/core/math.hpp", "engine/render/texture.cpp", "engine/core/log.cpp"],
     "dirs": ["engine/render", "engine"], "suites": ["render", "render_gl"], "targets": ["engine_lib"], "host": "assets.studio.example",
     "errs": ["ShaderCompileError", "NullDeref", "GLContextError"], "res": "asset_bake_31"},
    {"paths": ["backend/billing/invoice.py", "backend/billing/tax.py", "backend/billing/models.py", "backend/billing/api.py", "backend/common/db.py", "backend/billing/pdf.py"],
     "dirs": ["backend/billing", "backend"], "suites": ["billing", "billing_api"], "targets": ["billing_svc"], "host": "pay.gateway.example",
     "errs": ["DecimalError", "IntegrityError", "TaxRateMissing"], "res": "invoice_batch_9"},
    {"paths": ["ml/train/loop.py", "ml/train/data.py", "ml/models/unet.py", "ml/train/config.yaml", "ml/eval/metrics.py", "ml/train/ckpt.py"],
     "dirs": ["ml/train", "ml"], "suites": ["train_fast", "eval"], "targets": ["ml_wheel"], "host": "hub.weights.example",
     "errs": ["ShapeMismatch", "OOMError", "NaNLoss"], "res": "gpu_job_7781"},
    {"paths": ["mobile/ui/LoginView.swift", "mobile/ui/HomeView.swift", "mobile/net/Session.swift", "mobile/store/Keychain.swift", "mobile/ui/Theme.swift", "mobile/app/Router.swift"],
     "dirs": ["mobile/ui", "mobile"], "suites": ["ui_tests", "unit"], "targets": ["app_ios"], "host": "push.notify.example",
     "errs": ["TypeCheckError", "UnwrapNil", "SigningError"], "res": "testflight_build"},
    {"paths": ["gronk/blat/fizzle.qz", "gronk/blat/wump.qz", "gronk/snarf.qz", "gronk/blat/throb.qz", "gronk/quib.qz", "gronk/blat/oont.qz"],
     "dirs": ["gronk/blat", "gronk"], "suites": ["gamma", "delta"], "targets": ["blat_bundle"], "host": "snarfhost.gronk",
     "errs": ["WumpError", "ThrobFault", "QuibPanic"], "res": "gronk_mill"},
    {"paths": ["infra/tf/network.tf", "infra/tf/compute.tf", "infra/scripts/apply.sh", "infra/tf/vars.tf", "infra/tf/outputs.tf", "infra/scripts/plan.sh"],
     "dirs": ["infra/tf", "infra"], "suites": ["tf_validate", "infra_smoke"], "targets": ["tf_plan"], "host": "cloud.provider.example",
     "errs": ["QuotaExceeded", "StateLocked", "InvalidCIDR"], "res": "apply_run_204"},
]

RECIPES = {
    "E1": lambda c: (explore(c, 3), stall_tests(c, 8), explore(c, 2, j0=3), fix_and_pass(c)),
    "E2": lambda c: (fix_and_pass(c), osc_ab(c, 6, "tests"), explore(c, 2)),
    "E3": lambda c: (explore(c, 2), transient_http(c, 3), explore(c, 3, j0=2), stall_reads(c, 7)),
    "E4": lambda c: (incremental_repair(c, 3), polls_ok(c, 5), explore(c, 2)),
    "E5": lambda c: (explore(c, 4), osc_revert(c, 4, "build_ok"), fix_and_pass(c, j=1)),
    "E6": lambda c: (diagnosed_repair(c, "tests"), stall_noop_edits(c, 5, "tests")),
    "E7": lambda c: (explore(c, 1), stall_http(c, 10), transient_http(c, 1, j=2), explore(c, 2, j0=1)),
    "E8": lambda c: (explore(c, 2), osc_write2(c, 3), stall_tests(c, 6)),
    "E9": lambda c: (explore(c, 3), fix_and_pass(c), retry_active(c, 3, "poll")),
    "E10": lambda c: (fix_and_pass(c), fix_and_pass(c, j=1), osc_abc(c), explore(c, 3)),
    "E11": lambda c: (stall_search(c, 7), fix_and_pass(c), explore(c, 2)),
    "E12": lambda c: (explore(c, 3), stall_polls(c, 12)),
    "E13": lambda c: (explore(c, 2), flaky_once(c), osc_ab(c, 5, "lint")),
    "E14": lambda c: (diagnosed_repair(c, "build"), explore(c, 3), stall_build(c, 7, True)),
    "E15": lambda c: (incremental_repair(c, 2), explore(c, 3), retry_active(c, 2, "http")),
    "E16": lambda c: (explore(c, 5), plan(c), fix_and_pass(c, with_build=True), explore(c, 3, j0=5)),
    "E17": lambda c: (stall_reads(c, 6, two=True), diagnosed_repair(c, "lint"), explore(c, 2)),
    "E18": lambda c: (explore(c, 2), osc_ab(c, 6, None), stall_build(c, 5, False)),
    "E19": lambda c: (transient_http(c, 2, "GET"), explore(c, 2), osc_revert(c, 4, "tests")),
    "E20": lambda c: (explore(c, 1), stall_tests(c, 6, with_read=True), transient_http(c, 1), explore(c, 3)),
}


def build(sid):
    name, v = sid.split("_")
    s = Session(sid, start=["2026-04-01T10:00:00.000+00:00", "2026-04-02T18:45:30.000+00:00", "2026-04-03T06:12:00.000+00:00"][ord(v) % 3])
    c = Ctx(s)
    c.P = PROJECTS_E[hnum(sid, "projE", hi=len(PROJECTS_E))]
    RECIPES[name](c)
    return s


def all_sessions(variants=("a", "b", "c")):
    return [build(f"{name}_{v}") for name in RECIPES for v in variants]
