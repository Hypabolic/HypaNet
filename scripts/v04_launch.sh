#!/bin/sh
# Launch the v0.4 weight-author session (fresh claude-fable-5-1; claude-opus-5-5 fallback), docs/V04_WSP_KERNEL.md §7.
#
#   scripts/v04_launch.sh wsp-a                 [AUTHOR_DIR]   # new session (refuses if one exists)
#   scripts/v04_launch.sh continue K            [AUTHOR_DIR]   # technical continuation K (1..9), fixed prompt
#   scripts/v04_launch.sh resubmit N ERROR_FILE [AUTHOR_DIR]   # build rejection N (1..3): runner error text only
#   scripts/v04_launch.sh wsp-b                 [AUTHOR_DIR]   # refused until the WSP-B disclosure exists
#
# continue / resubmit apply to the current phase: wsp-b once LOGDIR/wsp-b.stream.jsonl exists, else wsp-a
# (override with V04_PHASE). AUTHOR_DIR defaults to ../HypaNet-v04-author (scripts/v04_prepare_author_dir.sh).
#
# Before claude: the cost cap (scripts/v04_cost.py, by session-id delta: WSP-A US$500, WSP-B US$250), no
# harness scratch (pytest-of-$USER) in the temp roots, the freeze registry and the handover record, the tools
# and wrappers (hypaqwen.wsp.install verify, kit hashes), the settings rendered for this launch
# (LOGDIR/NAME.settings.json, never overwritten) and the hook self-test on them. Then the model server starts
# in the background (python -m hypaqwen.wsp.server, envs/qwen, sockets in AUTHOR_DIR/.kernel/, ledger and
# journal in LOGDIR) and the launcher waits for its ready line; caffeinate keeps the Mac awake while it runs.
#
# claude -p, headless: --model/--fallback-model/--effort max, --session-id (new) or --resume, --settings <rendered>
# --setting-sources project, --restricted (file tools confined to the working directory; the settings' hook
# still applies), --permission-mode dontAsk, --permission-prompts none, --strict-mcp-config (no MCP servers),
# --tools Read,Glob,Grep,Write,Edit,Bash (an allow-list: no Agent, ListAgents, SendMessage, WebFetch, Skill...),
# --disable-slash-commands, --max-budget-usd <remaining cap>, --output-format stream-json --verbose. The
# environment is scrubbed of the variables that would link the session to a parent Claude Code session;
# CLAUDE_CODE_DISABLE_AUTO_MEMORY=1, CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000, HN_PHASE=<phase>.
#
# After claude exits: SIGTERM to the server (exit 143 expected) and wait, the session audit
# (scripts/v04_audit_session.py) and the cost (scripts/v04_cost.py). The build is NOT run: the build command is
# printed. Transcripts go to LOGDIR (never overwritten); every launch is appended to LOGDIR/LAUNCHES.jsonl.
#
# Kit tests only (scripts/v04_kit_tests.sh): V04_KIT=1 requires V04_LOGDIR (not experiments/v0_4/logs) and then
# honours V04_MODEL, V04_FALLBACK_MODEL, V04_EFFORT, V04_PROMPT_FILE, V04_BUDGET_TOKENS, V04_REQUEST_CAP,
# V04_MAX_BUDGET_USD and an unfrozen author dir; it never writes experiments/v0_4/SESSION_ID. V04_DEVICE
# (default mps) and V04_SERVER_TIMEOUT (seconds, default 900) apply to every launch.
set -eu
MODE="${1:?usage: v04_launch.sh wsp-a|continue K|resubmit N ERROR_FILE|wsp-b [AUTHOR_DIR]}"
REPO="$(cd "$(dirname "$0")/.." && pwd -P)"
KIT="$REPO/experiments/v0_4/author_kit"
QWEN="$REPO/envs/qwen"
REAL_LOGDIR="$REPO/experiments/v0_4/logs"
RESUBMIT_CAP=3
die() { echo "refusing: $*" >&2; exit 2; }
DEFAULT_AUTHOR="$(dirname "$REPO")/HypaNet-v04-author"
case "$MODE" in
  continue) K="${2:?K}"; AUTHOR_DIR="${3:-$DEFAULT_AUTHOR}";;
  resubmit) N="${2:?N}"; ERROR_FILE="${3:?ERROR_FILE}"; AUTHOR_DIR="${4:-$DEFAULT_AUTHOR}";;
  wsp-a|wsp-b) AUTHOR_DIR="${2:-$DEFAULT_AUTHOR}";;
  *) die "unknown mode $MODE";;
esac
AUTHOR_DIR="$(cd "$AUTHOR_DIR" 2>/dev/null && pwd -P)" || die "author dir missing"
for f in packet/PACKET_MANIFEST.json py fwd .venv/bin/python tools/hn_kernel/run_py.py tools/hn_hook/hook_bash.py; do
  [ -e "$AUTHOR_DIR/$f" ] || die "$AUTHOR_DIR is not a prepared v0.4 author dir ($f missing)"
done

KIT_MODE="${V04_KIT:-0}"
if [ "$KIT_MODE" = 1 ]; then
  LOGDIR="${V04_LOGDIR:?V04_KIT=1 needs V04_LOGDIR}"
  mkdir -p "$LOGDIR"
  LOGDIR="$(cd "$LOGDIR" && pwd -P)"
  [ "$LOGDIR" = "$REAL_LOGDIR" ] && die "V04_KIT=1 may not use the real log dir"
  case "$LOGDIR" in "$AUTHOR_DIR"|"$AUTHOR_DIR"/*) die "the log dir must be outside the author dir";; esac
  SID_FILE="$LOGDIR/SESSION_ID"
  MODEL="${V04_MODEL:-claude-haiku-5-5}"
  FALLBACK="${V04_FALLBACK_MODEL:-claude-sonnet-5-5}"
  EFFORT="${V04_EFFORT:-low}"
  BUDGET_TOKENS="${V04_BUDGET_TOKENS:-2000000}"
  REQUEST_CAP="${V04_REQUEST_CAP:-50000}"
else
  LOGDIR="$REAL_LOGDIR"
  SID_FILE="$REPO/experiments/v0_4/SESSION_ID"
  MODEL=claude-fable-5-1
  FALLBACK=claude-opus-5-5
  EFFORT=max
  BUDGET_TOKENS=2000000
  REQUEST_CAP=50000
  [ -f "$REPO/experiments/registry_v0_4.yaml" ] || die "experiments/registry_v0_4.yaml missing (G1 not frozen)"
  (cd "$REPO" && uv run python scripts/v04_freeze.py --check) || die "freeze registry check failed"
  uv run --project "$REPO" python - "$REPO/experiments/v0_4/HANDOVER.json" "$AUTHOR_DIR" <<'EOF' || die "handover record does not match this author dir"
import json, sys
h = json.load(open(sys.argv[1]))
assert h["author_dir"] == sys.argv[2] and not h["unfrozen_kit_test_dir"], h["author_dir"]
EOF
fi
DEVICE="${V04_DEVICE:-mps}"
SERVER_TIMEOUT="${V04_SERVER_TIMEOUT:-900}"
mkdir -p "$LOGDIR"

# Harness scratch (pytest's pytest-of-$USER holds rendered data and packets) must not exist while a session runs.
# The real per-user temp dir is checked too, so a changed TMPDIR cannot hide it. Kit sessions only warn.
USER_NAME="${USER:-$(id -un)}"
SYS_TMP="$(cd "$REPO" && uv run python -c 'import os, tempfile; print(os.path.realpath(tempfile.gettempdir()))')"
USER_TMP="$(getconf DARWIN_USER_TEMP_DIR 2>/dev/null || echo /tmp)"
for t in "$SYS_TMP" "${TMPDIR:-/tmp}" "$USER_TMP" /tmp /private/tmp; do
  if [ -e "$t/pytest-of-$USER_NAME" ]; then
    msg="harness scratch $t/pytest-of-$USER_NAME exists; remove it (rm -rf \"$t/pytest-of-$USER_NAME\") and do not run the test suite while the author session runs"
    if [ "$KIT_MODE" = 1 ]; then echo "WARNING (kit session): $msg" >&2; else die "$msg"; fi
  fi
done

current_phase() {
  if [ -n "${V04_PHASE:-}" ]; then echo "$V04_PHASE"
  elif [ -e "$LOGDIR/wsp-b.stream.jsonl" ]; then echo "wsp-b"
  else echo "wsp-a"; fi
}
COST_ARGS=""
[ "${V04_ACCEPT_UNRESOLVED_COST:-0}" = 1 ] && COST_ARGS="--accept-unresolved"
remaining_cap() {
  # shellcheck disable=SC2086
  (cd "$REPO" && uv run python scripts/v04_cost.py --logs "$LOGDIR" --phase "$1" --remaining $COST_ARGS) \
    || die "cost check refused a launch of $1 (scripts/v04_cost.py: cap reached, or a launch without a result event)"
}

case "$MODE" in
  wsp-a)
    TARGET="wsp-a"; NAME="$TARGET"
    [ -e "$SID_FILE" ] && die "a v0.4 weight-author session already exists ($SID_FILE)"
    SID="$(uuidgen | tr 'A-Z' 'a-z')"
    PROMPT_FILE="$KIT/prompt_wsp_a.md"
    [ "$KIT_MODE" = 1 ] && PROMPT_FILE="${V04_PROMPT_FILE:-$PROMPT_FILE}"
    PROMPT="$(cat "$PROMPT_FILE")"
    SESSION_ARGS="--session-id $SID"
    ;;
  wsp-b)
    die "WSP-B is not ready: it needs the WSP-A archive, the disclosure copy (disclosure/ in the author dir, the server's --disclosed file), prompt_wsp_b.md and a prepare step for output/wsp-b/; none exists yet"
    ;;
  continue)
    TARGET="$(current_phase)"
    case "$K" in [1-9]) ;; *) die "continue: K must be 1..9";; esac
    [ -s "$SID_FILE" ] || die "no session to continue ($SID_FILE)"
    SID="$(cat "$SID_FILE")"
    PROMPT="$(sed "s/{PHASE}/$TARGET/g" "$KIT/prompt_continue.md")"
    SESSION_ARGS="--resume $SID"
    NAME="$TARGET.c$K"
    ;;
  resubmit)
    TARGET="$(current_phase)"
    case "$N" in 1|2|3) ;; *) die "resubmit: N must be 1..$RESUBMIT_CAP";; esac
    [ -s "$ERROR_FILE" ] || die "resubmit: error file missing or empty"
    # Disclosed verbatim: the runner's capped author-facing text (BuildError / attempts/NNNN.json "error",
    # at most 2000 characters, scripts/v04_run_build.py's last_error.txt), never "operator_detail".
    [ "$(wc -c < "$ERROR_FILE")" -le 2100 ] || die "resubmit: $ERROR_FILE is longer than the runner's error cap (2000 characters)"
    echo "--- error text to be disclosed (review it before the session starts) ---" >&2
    cat "$ERROR_FILE" >&2
    echo "---" >&2
    [ -s "$SID_FILE" ] || die "no session to resubmit to ($SID_FILE)"
    SID="$(cat "$SID_FILE")"
    RESUB="$REPO/experiments/v0_4/resubmissions"
    [ "$KIT_MODE" = 1 ] && RESUB="$LOGDIR/resubmissions"
    mkdir -p "$RESUB"
    [ -e "$RESUB/$TARGET.r$N.error.txt" ] && die "resubmission $N already recorded"
    cp "$ERROR_FILE" "$RESUB/$TARGET.r$N.error.txt"
    PROMPT="$(uv run --project "$REPO" python - "$KIT/prompt_resubmit.md" "$TARGET" "$N" "$RESUBMIT_CAP" "$ERROR_FILE" <<'PYEOF'
import sys
t, phase, n, cap, err = sys.argv[1:6]
print(open(t).read().replace("{PHASE}", phase).replace("{N}", n).replace("{CAP}", cap).replace("{ERROR_TEXT}", open(err).read().strip()))
PYEOF
)"
    SESSION_ARGS="--resume $SID"
    NAME="$TARGET.r$N"
    ;;
esac
[ -d "$AUTHOR_DIR/output/$TARGET" ] || die "$AUTHOR_DIR/output/$TARGET missing"
LOG="$LOGDIR/$NAME.stream.jsonl"
[ -e "$LOG" ] && die "$LOG exists (sessions are not re-run silently)"
REMAINING="$(remaining_cap "$TARGET")"
if [ "$KIT_MODE" = 1 ] && [ -n "${V04_MAX_BUDGET_USD:-}" ]; then
  REMAINING="$(python3 -c "import sys; print(f'{min(float(sys.argv[1]), float(sys.argv[2])):.2f}')" "$REMAINING" "$V04_MAX_BUDGET_USD")"
fi
python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > 0 else 1)" "$REMAINING" || die "no budget left for $TARGET"

# tools, wrappers and settings
(cd "$QWEN" && uv run python -m hypaqwen.wsp.install verify "$AUTHOR_DIR/tools" --hook "$KIT/hook_bash.py") \
  || die "tools/ in $AUTHOR_DIR does not match the harness copy"
cmp -s "$KIT/py" "$AUTHOR_DIR/py" && cmp -s "$KIT/fwd" "$AUTHOR_DIR/fwd" || die "./py or ./fwd differs from the kit copy"
for p in "$AUTHOR_DIR" "$AUTHOR_DIR/py" "$AUTHOR_DIR/fwd" "$AUTHOR_DIR/packet" "$AUTHOR_DIR/.venv" "$AUTHOR_DIR/output"; do
  [ -w "$p" ] && die "$p is writable (the prepare script makes it read-only)"
done
[ -w "$AUTHOR_DIR/output/$TARGET" ] || die "$AUTHOR_DIR/output/$TARGET is not writable"
SETTINGS="$LOGDIR/$NAME.settings.json"
(cd "$REPO" && uv run python -m agent_loop_lab.v04.kit render-settings --phase "$TARGET" --author-dir "$AUTHOR_DIR" --out "$SETTINGS") \
  || die "settings render failed"
(cd "$REPO" && uv run python -m agent_loop_lab.v04.kit hook-selftest --phase "$TARGET" --author-dir "$AUTHOR_DIR" --settings "$SETTINGS") \
  || die "the PreToolUse hook self-test failed"

# model server
QPY="$(cd "$QWEN" && uv run python -c 'import sys; print(sys.executable)')"
READY="$LOGDIR/$NAME.server.out"
SERVER_ERR="$LOGDIR/$NAME.server.log"
[ -e "$READY" ] && die "$READY exists"
SERVER_PID=""
CAFF_PID=""
stop_server() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    SERVER_RC=0
    wait "$SERVER_PID" 2>/dev/null || SERVER_RC=$?
    echo "model server stopped (exit $SERVER_RC; 143 is the SIGTERM exit)"
  fi
  SERVER_PID=""
  [ -n "$CAFF_PID" ] && kill "$CAFF_PID" 2>/dev/null || true
  CAFF_PID=""
}
trap stop_server EXIT
trap 'stop_server; exit 130' INT TERM
(cd "$QWEN" && exec env -u PYTHONPATH "$QPY" -m hypaqwen.wsp.server --phase "$TARGET" --author-dir "$AUTHOR_DIR" \
  --log-dir "$LOGDIR" --device "$DEVICE" --budget "$BUDGET_TOKENS" --request-cap "$REQUEST_CAP") \
  > "$READY" 2> "$SERVER_ERR" &
SERVER_PID=$!
if command -v caffeinate > /dev/null 2>&1; then
  caffeinate -i -w "$SERVER_PID" &
  CAFF_PID=$!
fi
echo "model server pid $SERVER_PID (phase $TARGET, device $DEVICE); waiting up to ${SERVER_TIMEOUT}s for its ready line"
waited=0
until grep -q '"ready": true' "$READY" 2>/dev/null; do
  kill -0 "$SERVER_PID" 2>/dev/null || { cat "$SERVER_ERR" >&2; die "the model server exited before it was ready"; }
  [ "$waited" -ge "$SERVER_TIMEOUT" ] && die "the model server was not ready after ${SERVER_TIMEOUT}s"
  sleep 2
  waited=$((waited + 2))
done
head -n 1 "$READY"
[ -S "$AUTHOR_DIR/.kernel/kernel.sock" ] && [ -S "$AUTHOR_DIR/.kernel/fwd.sock" ] || die "the server's sockets are missing"

# a clean environment for the session: nothing that links it to a parent Claude Code session
for v in CLAUDECODE CLAUDE_CODE_ENTRYPOINT CLAUDE_CODE_CHILD_SESSION CLAUDE_CODE_MESSAGING_SOCKET \
         CLAUDE_CODE_MESSAGING_TOKEN CLAUDE_CODE_BRIDGE_SESSION_ID CLAUDE_CODE_SESSION_ID CLAUDE_PID \
         CLAUDE_PROJECT_DIR CLAUDE_EFFORT CLAUDE_CODE_SESSION_ATTENDED CLAUDE_CODE_EXECPATH CLAUDE_CODE_SSE_PORT \
         CLAUDE_CODE_SIMPLE CLAUDE_CODE_SAFE_MODE PYTHONPATH PYTHONHOME PYTHONSTARTUP VIRTUAL_ENV \
         HN_AUTHOR_DIR HN_KERNEL_SOCKET HN_FWD_SOCKET HN_TOKENIZER; do
  unset "$v" 2>/dev/null || true
done
export CLAUDE_CODE_DISABLE_AUTO_MEMORY=1
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${V04_MAX_OUTPUT_TOKENS:-128000}"
export HN_PHASE="$TARGET"
START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
[ "$MODE" = wsp-a ] && echo "$SID" > "$SID_FILE"
cd "$AUTHOR_DIR"
echo "launching $NAME: model=$MODEL fallback=$FALLBACK effort=$EFFORT max_budget_usd=$REMAINING session=$SID dir=$AUTHOR_DIR"
CLAUDE_RC=0
# shellcheck disable=SC2086
claude -p "$PROMPT" \
  --model "$MODEL" --fallback-model "$FALLBACK" --effort "$EFFORT" \
  $SESSION_ARGS \
  --settings "$SETTINGS" --setting-sources project --restricted \
  --permission-mode dontAsk --permission-prompts none \
  --strict-mcp-config \
  --tools Read,Glob,Grep,Write,Edit,Bash \
  --disable-slash-commands \
  --max-budget-usd "$REMAINING" \
  --output-format stream-json --verbose \
  < /dev/null > "$LOG" 2> "$LOGDIR/$NAME.stderr.log" || CLAUDE_RC=$?
[ "$CLAUDE_RC" = 0 ] || echo "claude exited $CLAUDE_RC (see $LOGDIR/$NAME.stderr.log)" >&2
cd "$REPO"
stop_server
trap - EXIT INT TERM
END="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '{"name": "%s", "phase": "%s", "mode": "%s", "session_id": "%s", "model": "%s", "fallback": "%s", "effort": "%s", "max_budget_usd": %s, "utc_start": "%s", "utc_end": "%s", "claude_rc": %s, "log": "%s", "kit": %s}\n' \
  "$NAME" "$TARGET" "$MODE" "$SID" "$MODEL" "$FALLBACK" "$EFFORT" "$REMAINING" "$START" "$END" "$CLAUDE_RC" "$(basename "$LOG")" \
  "$([ "$KIT_MODE" = 1 ] && echo true || echo false)" >> "$LOGDIR/LAUNCHES.jsonl"
echo "transcript: $LOG"
(cd "$QWEN" && uv run python -m hypaqwen.wsp.install verify "$AUTHOR_DIR/tools" --hook "$KIT/hook_bash.py") \
  || echo "TOOLS TAMPERED after the session (see above)" >&2
AUDIT_RC=0
uv run python scripts/v04_audit_session.py "$LOG" "$AUTHOR_DIR" "$TARGET" --logs "$LOGDIR" \
  --budget "$BUDGET_TOKENS" --request-cap "$REQUEST_CAP" || AUDIT_RC=$?
[ "$AUDIT_RC" = 0 ] || echo "AUDIT FOUND VIOLATIONS (see the .audit.json next to $LOG)" >&2
uv run python scripts/v04_cost.py --logs "$LOGDIR" --phase "$TARGET" | sed -n '1,200p' > "$LOGDIR/$NAME.cost.json"
uv run python scripts/v04_cost.py --logs "$LOGDIR" --phase "$TARGET" --check $COST_ARGS || true
echo "build (not run): uv run python scripts/v04_run_build.py $TARGET --author-dir $AUTHOR_DIR --logs $LOGDIR"
exit "$AUDIT_RC"
