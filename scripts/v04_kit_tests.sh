#!/bin/sh
# v0.4 author-kit tests (experiments/v0_4/KIT_TESTS.md), run on the PREPARED author dir before the WSP-A launch.
#
#   scripts/v04_kit_tests.sh [AUTHOR_DIR]
#
# AUTHOR_DIR defaults to ../HypaNet-v04-author. The kit logs go to $V04_KIT_LOGDIR (default
# experiments/v0_4/kit_tests/logs; it must not exist yet), never to experiments/v0_4/logs, so the real
# session's meter, journal and ledger are untouched. Steps:
#   1. refuse unless output/wsp-a/ holds nothing but an empty scratch/ (a pristine prepared dir);
#   2. write the kit scripts and requests into output/wsp-a/ (agent_loop_lab.v04.kit write-kit-files) and
#      the kit prompt into the kit log dir;
#   3. one throwaway session through the real launcher (scripts/v04_launch.sh wsp-a with V04_KIT=1):
#      claude-haiku-5-5, the rendered settings, the hook self-test, the model server with a 2,000-token phase
#      budget and a 1,500-token request cap, at most US$1 (V04_KIT_MAX_USD);
#   4. check the transcript (agent_loop_lab.v04.kit check-kit-session: init tool list, allowed and refused
#      probes, the guard's refusals inside ./py, the meter's refusal) and the audit (no executed violation; the
#      kit guard script's forbidden imports found by the static scan; one ok and one refused ledger request;
#      the edit journalled);
#   5. the fake build_edit.py through scripts/v04_run_build.py against the kit journal: rebuilt
#      byte-identically and equal to the journal;
#   6. remove everything the kit tests wrote under output/wsp-a/ (scratch/ is left empty) and verify tools/.
# Writes $V04_KIT_LOGDIR/KIT_RESULT.json; exits 0 only if every step passed. V04_DEVICE as for the launcher.
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd -P)"
AUTHOR_DIR="${1:-$(dirname "$REPO")/HypaNet-v04-author}"
AUTHOR_DIR="$(cd "$AUTHOR_DIR" && pwd -P)"
KIT="$REPO/experiments/v0_4/author_kit"
PHASE=wsp-a
OUT="$AUTHOR_DIR/output/$PHASE"
KITLOG="${V04_KIT_LOGDIR:-$REPO/experiments/v0_4/kit_tests/logs}"
die() { echo "refusing: $*" >&2; exit 2; }
fail() { echo "KIT TEST FAILED: $*" >&2; FAILED="$FAILED $1"; }
FAILED=""

[ -e "$KITLOG" ] && die "$KITLOG exists (kit logs are never reused)"
[ "$(ls -A "$OUT")" = scratch ] && [ -z "$(ls -A "$OUT/scratch")" ] \
  || die "$OUT is not pristine (expected only an empty scratch/)"
mkdir -p "$KITLOG"
KITLOG="$(cd "$KITLOG" && pwd -P)"
cd "$REPO"

cleanup() {
  # everything the kit tests wrote under output/wsp-a/: scratch/ is emptied, fwd/ and the deliverables removed
  for p in "$OUT"/* "$OUT"/.[!.]*; do
    [ -e "$p" ] || continue
    [ "$p" = "$OUT/scratch" ] && continue
    rm -rf "$p"
  done
  for p in "$OUT/scratch"/* "$OUT/scratch"/.[!.]*; do [ -e "$p" ] && rm -rf "$p"; done
  return 0
}
trap cleanup EXIT

uv run python -m agent_loop_lab.v04.kit write-kit-files --phase "$PHASE" --author-dir "$AUTHOR_DIR"
uv run python -m agent_loop_lab.v04.kit kit-prompt --phase "$PHASE" --author-dir "$AUTHOR_DIR" > "$KITLOG/kit_prompt.md"

LAUNCH_RC=0
V04_KIT=1 V04_LOGDIR="$KITLOG" V04_PROMPT_FILE="$KITLOG/kit_prompt.md" V04_BUDGET_TOKENS=2000 V04_REQUEST_CAP=1500 \
  V04_MAX_BUDGET_USD="${V04_KIT_MAX_USD:-1.00}" V04_MODEL="${V04_KIT_MODEL:-claude-haiku-5-5}" \
  sh scripts/v04_launch.sh wsp-a "$AUTHOR_DIR" || LAUNCH_RC=$?
LOG="$KITLOG/$PHASE.stream.jsonl"
[ -s "$LOG" ] || die "no kit transcript (launcher exit $LAUNCH_RC)"

uv run python -m agent_loop_lab.v04.kit check-kit-session "$LOG" --phase "$PHASE" --author-dir "$AUTHOR_DIR" \
  --out "$KITLOG/kit_session_check.json" || fail session
uv run python - "$LOG" "$KITLOG" "$PHASE" <<'EOF' || fail audit
import json, sys
from pathlib import Path
log, kitlog, phase = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
a = json.loads(log.with_suffix(".audit.json").read_text())
problems = []
if a["executed_violations"]:
    problems.append(f"executed violations: {[v for v in a['violations'] if v.startswith('[EXECUTED]')]}")
need = ("kt_guard.py", "subprocess")
if not any(all(n in f for n in need) for f in a["file_findings"]):
    problems.append("the static scan did not flag kt_guard.py's subprocess import (positive control)")
led = a["ledger"] or {}
if led.get("status", {}).get("ok") != 1 or led.get("status", {}).get("refused") != 1:
    problems.append(f"ledger statuses {led.get('status')} (expected 1 ok, 1 refused)")
if led.get("kernel_ops", {}).get("save_edit") != 1:
    problems.append(f"kernel ops {led.get('kernel_ops')} (expected one save_edit)")
if not a["permission_denials"]:
    problems.append("no permission denials recorded")
for p in problems:
    print("AUDIT CHECK:", p, file=sys.stderr)
print(f"audit check: {'OK' if not problems else 'FAILED'} ({len(a['file_findings'])} file findings, "
      f"{len(a['permission_denials'])} denials, ledger {led.get('status')})")
sys.exit(1 if problems else 0)
EOF

BUILD_RC=0
uv run python scripts/v04_run_build.py "$PHASE" --author-dir "$AUTHOR_DIR" --logs "$KITLOG" --archive "$KITLOG/archive" \
  > "$KITLOG/build.out" 2> "$KITLOG/build.err" || BUILD_RC=$?
cat "$KITLOG/build.out"
if [ "$BUILD_RC" = 0 ] && uv run python - "$KITLOG/archive/BUILD_RECORD.json" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))["checks"]
sys.exit(0 if c["journal_checked"] and c["rebuild_check"].startswith("byte-identical") else 1)
EOF
then
  echo "fake build: rebuilt byte-identically and equal to the kit journal"
else
  cat "$KITLOG/build.err" >&2
  fail build
fi

cleanup
trap - EXIT
[ "$(ls -A "$OUT")" = scratch ] && [ -z "$(ls -A "$OUT/scratch")" ] || fail cleanup
(cd "$REPO/envs/qwen" && uv run python -m hypaqwen.wsp.install verify "$AUTHOR_DIR/tools" --hook "$KIT/hook_bash.py") || fail tools
for s in "$AUTHOR_DIR/.kernel/kernel.sock" "$AUTHOR_DIR/.kernel/fwd.sock"; do [ -e "$s" ] && fail sockets-left; done

uv run python - "$KITLOG" "$FAILED" "$LAUNCH_RC" <<'EOF'
import json, sys
from pathlib import Path
kitlog, failed, launch_rc = Path(sys.argv[1]), sys.argv[2].split(), int(sys.argv[3])
rec = {"ok": not failed, "failed": failed, "launcher_exit": launch_rc,
       "session_check": json.loads((kitlog / "kit_session_check.json").read_text()) if (kitlog / "kit_session_check.json").exists() else None}
(kitlog / "KIT_RESULT.json").write_text(json.dumps(rec, indent=2) + "\n")
print(f"kit tests: {'ALL PASSED' if not failed else 'FAILED: ' + ' '.join(failed)} -> {kitlog / 'KIT_RESULT.json'}")
EOF
[ -z "$FAILED" ]
