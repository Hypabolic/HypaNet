#!/bin/sh
# Launch the HypaNet Clouds author session (SPEC section 7): claude-fable-5-1, max effort, no fallback model.
#
#   clouds/scripts/launch.sh start       [AUTHOR_DIR]   # the session (refuses if one exists)
#   clouds/scripts/launch.sh continue K  [AUTHOR_DIR]   # technical continuation K (1..30) after an API/limit error
#   clouds/scripts/launch.sh check       [AUTHOR_DIR]   # every pre-launch check; launches nothing
#
# AUTHOR_DIR defaults to ../HypaNet-clouds-author (scripts/prepare_author_dir.sh). start sends kit/prompt.md under a
# new --session-id (recorded in clouds/runs/r1/SESSION_ID); continue resumes that session with the fixed text
# kit/prompt_continue.md. Matthew approves the prompt before the first launch (SPEC section 7).
#
# Flags (as scripts/v05c_launch.sh): claude -p, --model claude-fable-5-1, --effort max, no --fallback-model and
# CLAUDE_CODE_NO_MODEL_FALLBACK=1, --settings kit/settings.json --setting-sources project, --permission-mode dontAsk
# --permission-prompts none, --strict-mcp-config, --tools Read,Glob,Grep,Write,Edit,Bash, --disallowedTools Agent
# Workflow WebFetch WebSearch Artifact Skill, --disable-slash-commands, --max-budget-usd <remaining of the US$500
# cap> (recorded in runs/r1/logs/NAME.launch.json before claude starts), --output-format stream-json --verbose;
# CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000, CLAUDE_CODE_DISABLE_AUTO_MEMORY=1, CLAUDE_CODE_DISABLE_CLAUDE_MDS=1; the
# environment scrubbed of variables that link it to a parent session. cwd = AUTHOR_DIR. No time limit: long thinking
# is never a reason to stop; only the cost cap, an error or NETWORK COMPLETE ends a launch.
#
# start also refuses unless kit/PROMPT_APPROVED.sha256 (written by Matthew when he approves the prompt) holds the
# sha256 of kit/prompt.md, and if the session's Claude Code project dir
# (~/.claude/projects/<author dir as a name>/, which the settings let the author Read) exists and is not empty;
# check reports both without refusing.
#
# --max-budget-usd: start passes the remaining cap. Claude Code's total_cost_usd is cumulative over --resume, and
# whether its budget check compares the flag with that restored total or with this launch's own spend is untested;
# continue therefore also passes only the remaining cap, which can never overspend the US$500 (the worst case is a
# continue that stops early; the launch record keeps session_total_before_usd). After a launch, a result with subtype
# error_max_budget_usd while scripts/cost.py still shows money left is explained: the check is then cumulative, and
# a later continue may pass session total + remaining by hand.
#
# Refuses before claude: an unknown mode; an author dir that does not verify against runs/r1/HANDOVER.json
# (kit/kit_build.py verify: every handed-over file unchanged, nothing added outside work/, the private venv and the
# harness files the tools run unchanged; for start also work/ empty); a test hand-over record; settings that do not
# deny reads of the repo and of every HypaNet* sibling; a HypaNet* sibling the settings do not cover; a prompt with a
# placeholder; the US$500 cap reached or a killed launch of unknown cost (scripts/cost.py); a launch that already
# ended with NETWORK COMPLETE (continue); an existing transcript of the same name; claude, uv or the Playwright
# Chromium missing; the GPU smoke render (the author's ./render code path on a tiny harness network, output in
# runs/r1/smoke/) failing or running on SwiftShader.
# A launch with no init event is moved aside as NAME.noinit.<UTC>.* and charged nothing (as v0.5).
# Afterwards: scripts/audit.py on the transcript, the hand-over verify again, and the cost summary.
set -eu
MODE="${1:?usage: launch.sh start|continue K|check [AUTHOR_DIR]}"
CLOUDS="$(cd "$(dirname "$0")/.." && pwd -P)"
REPO="$(cd "$CLOUDS/.." && pwd -P)"
PARENT="$(dirname "$REPO")"
KIT="$CLOUDS/kit"
RUN="$CLOUDS/runs/r1"
LOGDIR="$RUN/logs"
SID_FILE="$RUN/SESSION_ID"
RECORD="$RUN/HANDOVER.json"
SETTINGS="$KIT/settings.json"
DEFAULT_DIR="$PARENT/HypaNet-clouds-author"
MODEL="claude-fable-5-1"
EFFORT="max"
die() { echo "refusing: $*" >&2; exit 2; }
case "$MODE" in
  continue) K="${2:?K}"; AUTHOR_DIR="${3:-$DEFAULT_DIR}"
            case "$K" in [1-9]|[1-2][0-9]|30) ;; *) die "continue: K must be 1..30";; esac;;
  start|check) AUTHOR_DIR="${2:-$DEFAULT_DIR}";;
  *) die "unknown mode $MODE";;
esac
[ -d "$AUTHOR_DIR" ] && [ -f "$AUTHOR_DIR/HANDOVER.json" ] || die "$AUTHOR_DIR is not a prepared author dir (scripts/prepare_author_dir.sh)"
AUTHOR_DIR="$(cd "$AUTHOR_DIR" && pwd -P)"
case "$AUTHOR_DIR" in "$REPO"|"$REPO"/*) die "author dir must be outside the repo";; esac
UV="$(command -v uv)" || die "uv not on PATH"
cd "$CLOUDS"
[ -f "$RECORD" ] || die "$RECORD missing (scripts/prepare_author_dir.sh writes it)"
grep -q '"test": false' "$RECORD" || die "$RECORD is a TEST hand-over; re-run scripts/prepare_author_dir.sh --force without --test-root"

# 1. the hand-over
PRISTINE=""
if [ "$MODE" != continue ]; then PRISTINE="--pristine"; fi
# shellcheck disable=SC2086
"$UV" run --no-sync --quiet python kit/kit_build.py verify "$AUTHOR_DIR" "$RECORD" $PRISTINE \
  || die "the author dir does not verify against $RECORD (see PROBLEM lines above)"

# 2. settings: deny reads of the repo and of every HypaNet* sibling except the author dir
[ -f "$SETTINGS" ] || die "$SETTINGS missing"
"$UV" run --no-sync --quiet python -c 'import json,sys; json.load(open(sys.argv[1]))' "$SETTINGS" || die "$SETTINGS is not valid JSON"
grep -qF "\"Read(/$REPO/**)\"" "$SETTINGS" || die "$SETTINGS does not deny reads of $REPO"
"$UV" run --no-sync --quiet python kit/make_settings.py --check "$AUTHOR_DIR" \
  || die "$SETTINGS does not cover every current entry of ~/.claude (re-run: uv run python kit/make_settings.py)"
grep -qF '"Bash(./render:*)"' "$SETTINGS" && grep -qF '"Write(./work/**)"' "$SETTINGS" || die "$SETTINGS lacks the kit allow rules"
for d in "$PARENT"/HypaNet*; do
  [ -e "$d" ] || continue
  b="$(basename "$d")"
  case "$b" in
    HypaNet) ;;
    "$(basename "$AUTHOR_DIR")") ;;
    HypaNet-v0*) grep -qF "\"Read(/$PARENT/HypaNet-v0*/**)\"" "$SETTINGS" || die "settings do not deny ../$b";;
    HypaNet-v1*) grep -qF "\"Read(/$PARENT/HypaNet-v1*/**)\"" "$SETTINGS" || die "settings do not deny ../$b";;
    HypaNet-m*) grep -qF "\"Read(/$PARENT/HypaNet-m*/**)\"" "$SETTINGS" || die "settings do not deny ../$b";;
    *) die "unexpected sibling directory $d (not covered by the settings deny list; extend it first)";;
  esac
done

# 3. prompts
for f in prompt.md prompt_continue.md; do
  [ -s "$KIT/$f" ] || die "kit/$f missing"
  if grep -q '{[A-Z_][A-Z_0-9]*}' "$KIT/$f"; then die "unfilled placeholder in kit/$f"; fi
done
cmp -s "$KIT/prompt.md" "$AUTHOR_DIR/PROMPT.md" || die "AUTHOR_DIR/PROMPT.md differs from kit/prompt.md"
APPROVED_FILE="$KIT/PROMPT_APPROVED.sha256"
PROMPT_MD_SHA="$(shasum -a 256 "$KIT/prompt.md" | cut -d' ' -f1)"
APPROVED=no
if [ -s "$APPROVED_FILE" ] && [ "$(head -1 "$APPROVED_FILE" | cut -d' ' -f1 | tr -d '[:space:]')" = "$PROMPT_MD_SHA" ]; then
  APPROVED=yes
fi
if [ "$MODE" = start ] && [ "$APPROVED" != yes ]; then
  die "kit/prompt.md is not approved: kit/PROMPT_APPROVED.sha256 must hold its sha256 $PROMPT_MD_SHA (Matthew: shasum -a 256 kit/prompt.md > kit/PROMPT_APPROVED.sha256)"
fi
# the session's own Claude Code project dir: the settings allow the author to Read it, so it must start empty
OWN_PROJECT="$HOME/.claude/projects/$(printf '%s' "$AUTHOR_DIR" | sed 's/[^A-Za-z0-9]/-/g')"
OWN_PROJECT_STATE="absent"
if [ -d "$OWN_PROJECT" ]; then
  if [ -n "$(find "$OWN_PROJECT" -mindepth 1 -print -quit)" ]; then OWN_PROJECT_STATE="NOT EMPTY"; else OWN_PROJECT_STATE="empty"; fi
elif [ -e "$OWN_PROJECT" ]; then
  OWN_PROJECT_STATE="NOT EMPTY"
fi
if [ "$MODE" = start ] && [ "$OWN_PROJECT_STATE" = "NOT EMPTY" ]; then
  die "$OWN_PROJECT exists and is not empty (earlier transcripts the author could Read); move it aside first"
fi

# 4. cost
mkdir -p "$LOGDIR"
("$UV" run --no-sync --quiet python scripts/cost.py --logs "$LOGDIR" --check > /dev/null) \
  || die "the US\$500 cap is reached or a launch is unresolved (scripts/cost.py)"
REMAINING="$("$UV" run --no-sync --quiet python scripts/cost.py --logs "$LOGDIR" --remaining 2>/dev/null)"
case "$REMAINING" in ''|*[!0-9.]*) die "could not read the remaining budget";; esac
"$UV" run --no-sync --quiet python -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= 1 else 1)' "$REMAINING" \
  || die "less than US\$1 of the cap remains"

# 5. session and transcript names
case "$MODE" in
  start|check)
    NAME="r1"
    if [ "$MODE" = start ]; then
      if [ -e "$SID_FILE" ]; then die "a session already exists ($SID_FILE); use continue"; fi
      if [ -n "$(find "$LOGDIR" -name '*.stream.jsonl' -print -quit)" ]; then die "transcripts exist in $LOGDIR"; fi
    fi
    SID="$(uuidgen | tr 'A-Z' 'a-z')"
    PROMPT_FILE="$KIT/prompt.md"
    SESSION_ARGS="--session-id $SID"
    ;;
  continue)
    [ -s "$SID_FILE" ] || die "no session yet ($SID_FILE)"
    SID="$(cat "$SID_FILE")"
    if "$UV" run --no-sync --quiet python scripts/audit.py --complete "$LOGDIR" > /dev/null; then
      die "a launch already ended with NETWORK COMPLETE; the run is over"
    fi
    PROMPT_FILE="$KIT/prompt_continue.md"
    SESSION_ARGS="--resume $SID"
    NAME="r1.c$K"
    ;;
esac
# the --max-budget-usd flag (see the header): the remaining cap, on start and on continue
SESSION_TOTAL="0"
BUDGET_FLAG="$REMAINING"
if [ "$MODE" = continue ]; then
  SESSION_TOTAL="$("$UV" run --no-sync --quiet python scripts/cost.py --logs "$LOGDIR" --session-total "$SID")"
fi
case "$BUDGET_FLAG" in ''|*[!0-9.]*) die "could not compute the --max-budget-usd flag";; esac
LOG="$LOGDIR/$NAME.stream.jsonl"
if [ "$MODE" != check ] && [ -e "$LOG" ]; then die "$LOG exists (launches are not re-run silently)"; fi
PROMPT="$(cat "$PROMPT_FILE")"
PROMPT_SHA="$(shasum -a 256 "$PROMPT_FILE" | cut -d' ' -f1)"

# 6. programs and the GPU: the author's ./render code path on a tiny harness network, output in runs/r1/smoke/
CLAUDE_BIN="$(command -v claude)" || die "claude not on PATH"
SMOKE="$RUN/smoke"
rm -rf "$SMOKE"
mkdir -p "$SMOKE"
cat > "$SMOKE/smoke_net.json" <<'EOF'
{"_note": "launch.sh smoke test network (harness only)",
 "probes": [{"type": "density", "frame": "light", "dir": [1, 0, 0], "dist": 30, "blur": 30}, {"type": "const", "value": 1}],
 "layers": [{"W": [[-3, 0]], "b": [0], "act": "exp"},
            {"W": [[0.3], [0.3], [0.3], [0], [0], [0], [0], [0], [0]], "b": [0, 0, 0, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2], "act": "linear"}]}
EOF
if ! "$UV" run --no-sync --quiet python -I -B kit/tools/author_tool.py "$AUTHOR_DIR" "$SMOKE" render "$SMOKE/smoke_net.json" \
     --shot front_noon --frames 2 --res 320x200 --out "$SMOKE/smoke.png" > "$SMOKE/smoke.log" 2>&1; then
  cat "$SMOKE/smoke.log" >&2
  die "the smoke render failed"
fi
if grep -qi swiftshader "$SMOKE/smoke.log"; then die "WebGL runs on SwiftShader, not the GPU (see $SMOKE/smoke.log)"; fi
grep -q "ANGLE Metal Renderer" "$SMOKE/smoke.log" || die "the smoke render did not report the Metal GPU (see $SMOKE/smoke.log)"
[ -s "$SMOKE/smoke.png" ] && [ -s "$SMOKE/smoke.npy" ] || die "the smoke render wrote no image"
"$UV" run --no-sync --quiet python kit/kit_build.py verify "$AUTHOR_DIR" "$RECORD" $PRISTINE > /dev/null \
  || die "the smoke render changed the author dir"

if [ "$MODE" = check ]; then
  echo "check passed: would launch $NAME model=$MODEL fallback=none effort=$EFFORT max_budget_usd=$BUDGET_FLAG"
  echo "  author dir $AUTHOR_DIR (verified against $RECORD)"
  echo "  prompt approved (kit/PROMPT_APPROVED.sha256 = sha256 of kit/prompt.md): $APPROVED$([ "$APPROVED" = yes ] || echo ' -- start refuses until Matthew approves it')"
  echo "  session project dir $OWN_PROJECT: $OWN_PROJECT_STATE$([ "$OWN_PROJECT_STATE" != "NOT EMPTY" ] || echo ' -- start refuses')"
  echo "  prompt $PROMPT_FILE sha256 $PROMPT_SHA (${#PROMPT} chars); settings sha256 $(shasum -a 256 "$SETTINGS" | cut -d' ' -f1)"
  echo "  claude $CLAUDE_BIN; GPU smoke render OK ($(grep -o 'ANGLE Metal Renderer: [^,)]*' "$SMOKE/smoke.log" | head -1))"
  exit 0
fi

START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
# the launch record: written before claude starts, so a killed launch is still charged its ceiling
printf '{"name": "%s", "mode": "%s", "session_id": "%s", "model": "%s", "fallback": null, "effort": "%s", "max_budget_usd": %s, "max_budget_usd_flag": %s, "session_total_before_usd": %s, "utc_start": "%s", "settings_sha256": "%s", "prompt_sha256": "%s", "handover_sha256": "%s", "author_dir": "%s"}\n' \
  "$NAME" "$MODE" "$SID" "$MODEL" "$EFFORT" "$REMAINING" "$BUDGET_FLAG" "$SESSION_TOTAL" "$START" "$(shasum -a 256 "$SETTINGS" | cut -d' ' -f1)" \
  "$PROMPT_SHA" "$(shasum -a 256 "$RECORD" | cut -d' ' -f1)" "$AUTHOR_DIR" > "$LOGDIR/$NAME.launch.json"
if [ "$MODE" = start ]; then echo "$SID" > "$SID_FILE"; fi
for v in CLAUDECODE CLAUDE_CODE_ENTRYPOINT CLAUDE_CODE_CHILD_SESSION CLAUDE_CODE_MESSAGING_SOCKET \
         CLAUDE_CODE_MESSAGING_TOKEN CLAUDE_CODE_BRIDGE_SESSION_ID CLAUDE_CODE_SESSION_ID CLAUDE_PID \
         CLAUDE_PROJECT_DIR CLAUDE_EFFORT CLAUDE_CODE_SESSION_ATTENDED CLAUDE_CODE_EXECPATH CLAUDE_CODE_SSE_PORT \
         CLAUDE_CODE_SIMPLE CLAUDE_CODE_SAFE_MODE CLAUDE_CODE_EFFORT_LEVEL CLAUDE_CODE_SUBAGENT_MODEL ANTHROPIC_MODEL \
         ANTHROPIC_SMALL_FAST_MODEL CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD PYTHONPATH PYTHONHOME PYTHONSTARTUP \
         VIRTUAL_ENV UV_PROJECT_ENVIRONMENT; do
  unset "$v" 2>/dev/null || true
done
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000
export CLAUDE_CODE_DISABLE_AUTO_MEMORY=1
export CLAUDE_CODE_NO_MODEL_FALLBACK=1
export CLAUDE_CODE_DISABLE_CLAUDE_MDS=1
cd "$AUTHOR_DIR"
echo "launching $NAME: model=$MODEL fallback=none effort=$EFFORT max_output=$CLAUDE_CODE_MAX_OUTPUT_TOKENS max_budget_usd=$BUDGET_FLAG (remaining cap $REMAINING, session total so far $SESSION_TOTAL) session=$SID dir=$AUTHOR_DIR"
CLAUDE_RC=0
# shellcheck disable=SC2086
"$CLAUDE_BIN" -p "$PROMPT" \
  --model "$MODEL" --effort "$EFFORT" \
  $SESSION_ARGS \
  --settings "$SETTINGS" --setting-sources project \
  --permission-mode dontAsk --permission-prompts none \
  --strict-mcp-config \
  --tools Read,Glob,Grep,Write,Edit,Bash \
  --disallowedTools Agent Workflow WebFetch WebSearch Artifact Skill \
  --disable-slash-commands \
  --max-budget-usd "$BUDGET_FLAG" \
  --output-format stream-json --verbose \
  < /dev/null > "$LOG" 2> "$LOGDIR/$NAME.stderr.log" || CLAUDE_RC=$?
cd "$CLOUDS"
END="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STARTED=false
grep -q '"type":"system","subtype":"init"' "$LOG" 2>/dev/null && STARTED=true
printf '{"name": "%s", "mode": "%s", "session_id": "%s", "utc_start": "%s", "utc_end": "%s", "claude_rc": %s, "started": %s}\n' \
  "$NAME" "$MODE" "$SID" "$START" "$END" "$CLAUDE_RC" "$STARTED" >> "$LOGDIR/LAUNCHES.jsonl"
[ "$CLAUDE_RC" = 0 ] || echo "claude exited $CLAUDE_RC (see $LOGDIR/$NAME.stderr.log)" >&2
if [ "$STARTED" != true ]; then
  # no init event: no session was created and nothing was spent; keep the evidence out of the cost accounting
  ATTEMPT="$NAME.noinit.$(date -u +%Y%m%dT%H%M%SZ)"
  mv "$LOG" "$LOGDIR/$ATTEMPT.stream.log" 2>/dev/null || true
  mv "$LOGDIR/$NAME.stderr.log" "$LOGDIR/$ATTEMPT.stderr.log" 2>/dev/null || true
  mv "$LOGDIR/$NAME.launch.json" "$LOGDIR/$ATTEMPT.launch.json"
  if [ "$MODE" = start ]; then mv "$SID_FILE" "$LOGDIR/$ATTEMPT.SESSION_ID"; fi
  echo "NO INIT EVENT: claude exited $CLAUDE_RC without starting a session (see $LOGDIR/$ATTEMPT.stderr.log)." >&2
  echo "Moved aside as $ATTEMPT.*; nothing is charged. Fix the cause and re-run the same command." >&2
  exit 1
fi
echo "transcript: $LOG"
"$UV" run --no-sync --quiet python scripts/audit.py "$LOG" "$AUTHOR_DIR" || echo "AUDIT FOUND VIOLATIONS (see $LOGDIR/$NAME.audit.json)" >&2
"$UV" run --no-sync --quiet python kit/kit_build.py verify "$AUTHOR_DIR" "$RECORD" > /dev/null \
  || echo "HAND-OVER CHANGED during the session (run kit/kit_build.py verify to see what)" >&2
"$UV" run --no-sync --quiet python scripts/cost.py --logs "$LOGDIR" --check || true
if "$UV" run --no-sync --quiet python scripts/cost.py --logs "$LOGDIR" --budget-stops | grep -qx "$NAME"; then
  LEFT="$("$UV" run --no-sync --quiet python scripts/cost.py --logs "$LOGDIR" --remaining 2>/dev/null || echo 0)"
  if "$UV" run --no-sync --quiet python -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= 1 else 1)' "${LEFT:-0}" 2>/dev/null; then
    echo "BUDGET STOP WITH MONEY LEFT: Claude Code ended $NAME with error_max_budget_usd (flag US\$$BUDGET_FLAG), but" >&2
    echo "  scripts/cost.py still shows US\$$LEFT of the US\$500 cap. Its budget check must count something other than" >&2
    echo "  what launch.sh assumed (session total before this launch: US\$$SESSION_TOTAL; remaining then: US\$$REMAINING)." >&2
    echo "  Compare total_cost_usd in the last result event of $LOG with $LOGDIR/$NAME.launch.json, record the finding" >&2
    echo "  in kit/NOTES.md, and adjust the flag computation before 'continue <K+1>'. Nothing beyond the cap was spent." >&2
  fi
fi
if "$UV" run --no-sync --quiet python scripts/audit.py --complete "$LOGDIR" > /dev/null; then
  echo "NETWORK COMPLETE: the run is over. Archive work/ to runs/r1/ and score it."
else
  echo "The session ended without NETWORK COMPLETE. If it stopped on an API or limit error, continue with: scripts/launch.sh continue <K>"
fi
