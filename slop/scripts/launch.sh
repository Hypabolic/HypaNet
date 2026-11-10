#!/bin/sh
# Launch the HypaNet Slop author session (SPEC.md §12.7, §13): claude-fable-5-2, effort xhigh, no fallback model.
#
#   slop/scripts/launch.sh smoke        [AUTHOR_DIR]   # the one smoke launch (../HypaNet-slop-smoke, runs/smoke/, US$10)
#   slop/scripts/launch.sh start        [AUTHOR_DIR]   # the r1 session (refuses if one exists)
#   slop/scripts/launch.sh continue K   [AUTHOR_DIR]   # technical continuation K (1..30) after an API/limit error
#                                                      # or a reply that ended the session without MODEL COMPLETE
#   slop/scripts/launch.sh check [smoke|continue K] [AUTHOR_DIR]   # every pre-launch check; launches nothing and
#                                                      # writes nothing
#   slop/scripts/launch.sh --run r2 start|continue K|check [continue K] [AUTHOR_DIR]   # round 2 (below)
#
# Round 2 (--run r2; without --run everything is r1 as before): AUTHOR_DIR defaults to ../HypaNet-slop-author-r2 and
# its name must start with HypaNet-slop-author-r2 (an r1 dir is refused, and the reverse); record runs/r2/HANDOVER.json,
# logs runs/r2/logs, session runs/r2/SESSION_ID, launches r2 and r2.c<K>; settings kit/settings.r2.json; prompt
# kit/prompt_r2.md (= AUTHOR_DIR/PROMPT.md), approved by kit/PROMPT_R2_APPROVED.sha256; continue uses the same
# kit/prompt_continue.md. Its own cap: the US$ figure in runs/r2/CAP_USD, which Matthew writes (one figure and nothing
# else; every r2 mode refuses without it; nothing writes it); r1's spend is not charged to it. The whole figure, less
# r2's spend so far, is the --max-budget-usd flag, so like r1's it can be passed by about one response (up to about
# US$21): write about US$21 less than the most r2 may spend. No smoke launch: `smoke` is refused for r2, and
# start needs no smoke transcript. start verifies --pristine, which for r2 means work/ is exactly the recorded seed
# (r1's work/ without scratch/).
#
# AUTHOR_DIR defaults to ../HypaNet-slop-author (../HypaNet-slop-smoke for smoke), both built by
# scripts/prepare_author_dir.sh, which writes the hand-over record runs/r1/HANDOVER.json (runs/smoke/HANDOVER.json).
# start and smoke send kit/prompt.md under a new --session-id (recorded in runs/<run>/SESSION_ID); continue resumes
# the r1 session with the fixed technical text kit/prompt_continue.md. `check` alone checks a start.
#
# Flags (SPEC.md §13): claude -p, --model claude-fable-5-2, --effort xhigh, no --fallback-model,
# --settings kit/settings.json (kit/settings.smoke.json for smoke) --setting-sources project,
# --permission-mode dontAsk --permission-prompts none,
# --strict-mcp-config, --tools Read,Glob,Grep,Write,Edit,Bash, --disallowedTools Agent Workflow WebFetch WebSearch
# Artifact Skill, --disable-slash-commands, --max-budget-usd <flag>, --output-format stream-json --verbose.
# Environment: CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK=1, CLAUDE_CODE_NO_MODEL_FALLBACK=1,
# CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000, CLAUDE_CODE_DISABLE_AUTO_MEMORY=1, CLAUDE_CODE_DISABLE_CLAUDE_MDS=1; every
# other CLAUDECODE / CLAUDE_* / CLAUDE_CODE_* variable that links it to a parent session is unset (the Clouds list,
# CLAUDE_CODE_EFFORT_LEVEL included: it outranks --effort and Claude Code's own Fable 5.2 default is medium), as are
# the Python and model-override variables. cwd = AUTHOR_DIR. No time limit: long thinking is never a reason to stop;
# only the cap, an error or MODEL COMPLETE (a line of its own at the end of the author's last message) ends a launch.
#
# --max-budget-usd: the remaining part of the one US$200 cap (scripts/cost.py over runs/smoke/logs and
# runs/r1/logs, rounded down to the cent) on start and continue; min(10, remaining) on smoke. Claude Code does not
# count totals restored by --resume toward the flag (research/fable-5-2.md §6, §8 item 12), so passing the remaining
# cap on a continue neither overspends nor stops early. Spend can pass the flag by about one response (up to about
# US$21); the cap allows for one such overshoot (SPEC.md §13). The flag is recorded in runs/<run>/logs/NAME.launch.json
# before claude starts, so a killed launch is charged its ceiling.
#
# Refuses before claude: an unknown mode; Claude Code older than 2.1.321 (the first with claude-fable-5-2);
# CLAUDE_CONFIG_DIR set (the settings enumerate ~/.claude); an author dir inside the repo or not prepared; a TEST
# hand-over record; an author dir that does not verify against its record (kit/kit_build.py verify; for smoke and
# start also --pristine: work/ empty for r1 and smoke, exactly the recorded seed for r2); kit/leakage.py --require-sealed failing on the author dir (SPEC §9: the sealed
# n-gram check must have run, or be waived by name in kit/leakage_waivers.json); settings that kit/make_settings.py --check rejects, or that miss a rule
# scripts/launch_checks.py requires (dontAsk, autoMemoryEnabled false, switchModelsOnFlag false, the repo, temp,
# private-note and sealed-repo denies, a deny for every HypaNet* sibling, the tool allows); a prompt with a
# {PLACEHOLDER}; AUTHOR_DIR/PROMPT.md that differs from kit/prompt.md; for smoke and start, kit/PROMPT_APPROVED.sha256
# not holding the sha256 of kit/prompt.md (Matthew writes it: SPEC.md says he approves the prompt before any launch,
# smoke included) or a session project dir (~/.claude/projects/<author dir as a name>/) that exists and is not empty;
# the cap reached, less than US$1 left, or a killed launch of unknown cost (scripts/cost.py); start with an r1 session
# or transcript already there, or with no smoke transcript in runs/smoke/logs (SPEC.md §12.7: smoke first) unless
# kit/SMOKE_SKIPPED exists (Matthew skipped the smoke launch, 2026-11-11; the file holds the reason); smoke with a smoke transcript already there (one smoke launch); continue K with no r1
# session, a run that already ended with MODEL COMPLETE, or K not the next continuation; an existing transcript of
# the same name; claude or uv missing; the author's ./tok failing on a fixed string.
# A launch with no init event is moved aside as NAME.noinit.<UTC>.* and charged nothing (as Clouds).
# Afterwards: scripts/audit.py on the transcript, the hand-over verify again, and the cost summary.
set -eu
SLOP="$(cd "$(dirname "$0")/.." && pwd -P)"
die() { echo "refusing: $*" >&2; exit 2; }
RUN_NAME=r1
if [ "${1:-}" = "--run" ]; then
  RUN_NAME="${2:-}"
  shift 2 2>/dev/null || die "--run needs r1 or r2"
fi
case "$RUN_NAME" in r1|r2) ;; *) die "--run must be r1 or r2";; esac
MODE="${1:?usage: launch.sh [--run r2] smoke|start|continue K|check [smoke|continue K] [AUTHOR_DIR]}"
REPO="$(cd "$SLOP/.." && pwd -P)"
PARENT="$(dirname "$REPO")"
KIT="$SLOP/kit"
MODEL="claude-fable-5-2"
EFFORT="xhigh"
MIN_CLAUDE="2.1.321"

CHECK=0
if [ "$MODE" = check ]; then
  CHECK=1
  shift
  case "${1:-}" in
    smoke) MODE=smoke; shift;;
    continue) MODE=continue; shift; K="${1:?check continue K}"; shift;;
    *) MODE=start;;
  esac
  ARG_DIR="${1:-}"
else
  shift
  case "$MODE" in
    continue) K="${1:?continue K}"; shift; ARG_DIR="${1:-}";;
    smoke|start) ARG_DIR="${1:-}";;
    *) die "unknown mode $MODE";;
  esac
fi
if [ "$MODE" = continue ]; then
  case "$K" in [1-9]|[1-2][0-9]|30) ;; *) die "continue: K must be 1..30";; esac
fi
PROMPT_MAIN="$KIT/prompt.md"
APPROVED_FILE="$KIT/PROMPT_APPROVED.sha256"
if [ "$RUN_NAME" = r2 ]; then
  [ "$MODE" != smoke ] || die "run r2 has no smoke launch (start it with: launch.sh --run r2 start)"
  RUN="$SLOP/runs/r2"; DEFAULT_DIR="$PARENT/HypaNet-slop-author-r2"; SETTINGS="$KIT/settings.r2.json"
  PROMPT_MAIN="$KIT/prompt_r2.md"; APPROVED_FILE="$KIT/PROMPT_R2_APPROVED.sha256"
  ALL_LOGS="--run r2"
elif [ "$MODE" = smoke ]; then
  RUN="$SLOP/runs/smoke"; DEFAULT_DIR="$PARENT/HypaNet-slop-smoke"; SETTINGS="$KIT/settings.smoke.json"
  ALL_LOGS="--logs $SLOP/runs/smoke/logs --logs $SLOP/runs/r1/logs"
else
  RUN="$SLOP/runs/r1"; DEFAULT_DIR="$PARENT/HypaNet-slop-author"; SETTINGS="$KIT/settings.json"
  ALL_LOGS="--logs $SLOP/runs/smoke/logs --logs $SLOP/runs/r1/logs"
fi
LOGDIR="$RUN/logs"
SID_FILE="$RUN/SESSION_ID"
RECORD="$RUN/HANDOVER.json"
AUTHOR_DIR="${ARG_DIR:-$DEFAULT_DIR}"
PROMPT_NAME="kit/$(basename "$PROMPT_MAIN")"
APPROVED_NAME="kit/$(basename "$APPROVED_FILE")"
if [ "$RUN_NAME" = r2 ]; then
  [ -s "$RUN/CAP_USD" ] || die "runs/r2/CAP_USD missing: Matthew writes r2's cap there (a US\$ figure); nothing else does"
fi

# 0. programs and Claude Code's version
CLAUDE_BIN="$(command -v claude)" || die "claude not on PATH"
UV="$(command -v uv)" || die "uv not on PATH"
PY() { "$UV" run --project "$SLOP" --no-sync --quiet python "$@"; }
CLAUDE_VERSION="$("$CLAUDE_BIN" --version 2>/dev/null || true)"
PY scripts/launch_checks.py version "$CLAUDE_VERSION" "$MIN_CLAUDE" >/dev/null 2>&1 \
  || die "Claude Code '$CLAUDE_VERSION' is older than $MIN_CLAUDE or unreadable (claude-fable-5-2 needs $MIN_CLAUDE+)"
[ -z "${CLAUDE_CONFIG_DIR:-}" ] || die "CLAUDE_CONFIG_DIR is set; the settings enumerate ~/.claude (unset it)"

# 1. the author dir and its hand-over
[ -d "$AUTHOR_DIR" ] && [ -f "$AUTHOR_DIR/HANDOVER.json" ] || die "$AUTHOR_DIR is not a prepared author dir (scripts/prepare_author_dir.sh)"
AUTHOR_DIR="$(cd "$AUTHOR_DIR" && pwd -P)"
case "$AUTHOR_DIR" in "$REPO"|"$REPO"/*) die "author dir must be outside the repo";; esac
case "$(basename "$AUTHOR_DIR")" in
  HypaNet-slop-author-r2*) [ "$RUN_NAME" = r2 ] || die "$AUTHOR_DIR is an r2 author dir (use --run r2)";;
  *) [ "$RUN_NAME" = r1 ] || die "an r2 author dir's name starts with HypaNet-slop-author-r2 ($AUTHOR_DIR)";;
esac
cd "$SLOP"
[ -f "$RECORD" ] || die "$RECORD missing (scripts/prepare_author_dir.sh writes it)"
grep -q '"test": false' "$RECORD" || die "$RECORD is a TEST hand-over; re-run scripts/prepare_author_dir.sh --force without --test-root"
PRISTINE=""
if [ "$MODE" != continue ]; then PRISTINE="--pristine"; fi
if [ "$RUN_NAME" = r2 ]; then PRISTINE_MEANS="work/ must be exactly the recorded seed, r1's work/ without scratch/"
else PRISTINE_MEANS="work/ must be empty"; fi
# shellcheck disable=SC2086
PY kit/kit_build.py verify "$AUTHOR_DIR" "$RECORD" $PRISTINE \
  || die "the author dir does not verify against $RECORD$([ -z "$PRISTINE" ] || echo " (pristine: $PRISTINE_MEANS)") (see PROBLEM lines above)"

# the leakage check, sealed n-grams required (SPEC §9); the hand-over is fixed, so it gives the same answer every time
PY kit/leakage.py "$AUTHOR_DIR" --require-sealed --run "$RUN_NAME" > /dev/null \
  || die "kit/leakage.py --require-sealed fails on $AUTHOR_DIR (run it to see why; a sealed-ngrams not-run problem clears once scripts/build_sealed.py has written the n-grams, or by a named waiver in kit/leakage_waivers.json)"

# 2. settings
[ -f "$SETTINGS" ] || die "$SETTINGS missing"
PY kit/make_settings.py --check "$AUTHOR_DIR" --out "$SETTINGS" \
  || die "$SETTINGS does not match kit/make_settings.py (re-run: uv run python kit/make_settings.py $AUTHOR_DIR)"
PY scripts/launch_checks.py settings "$SETTINGS" "$AUTHOR_DIR" || die "$SETTINGS lacks a required rule (PROBLEM lines above)"

# 3. prompts
for f in "$(basename "$PROMPT_MAIN")" prompt_continue.md; do
  [ -s "$KIT/$f" ] || die "kit/$f missing"
  if grep -q '{[A-Z_][A-Z_0-9]*}' "$KIT/$f"; then die "unfilled placeholder in kit/$f"; fi
done
cmp -s "$PROMPT_MAIN" "$AUTHOR_DIR/PROMPT.md" || die "AUTHOR_DIR/PROMPT.md differs from $PROMPT_NAME"
PROMPT_MD_SHA="$(shasum -a 256 "$PROMPT_MAIN" | cut -d' ' -f1)"
APPROVED=no
if [ -s "$APPROVED_FILE" ] && [ "$(head -1 "$APPROVED_FILE" | cut -d' ' -f1 | tr -d '[:space:]')" = "$PROMPT_MD_SHA" ]; then
  APPROVED=yes
fi
if [ "$CHECK" = 0 ] && [ "$MODE" != continue ] && [ "$APPROVED" != yes ]; then
  die "$PROMPT_NAME is not approved: $APPROVED_NAME must hold its sha256 $PROMPT_MD_SHA (Matthew: shasum -a 256 $PROMPT_NAME > $APPROVED_NAME)"
fi
# the session's own Claude Code project dir: the settings allow the author to Read it, so it must start empty
OWN_PROJECT="$HOME/.claude/projects/$(printf '%s' "$AUTHOR_DIR" | sed 's/[^A-Za-z0-9]/-/g')"
OWN_PROJECT_STATE="absent"
if [ -d "$OWN_PROJECT" ]; then
  if [ -n "$(find "$OWN_PROJECT" -mindepth 1 -print -quit)" ]; then OWN_PROJECT_STATE="NOT EMPTY"; else OWN_PROJECT_STATE="empty"; fi
elif [ -e "$OWN_PROJECT" ]; then
  OWN_PROJECT_STATE="NOT EMPTY"
fi
if [ "$CHECK" = 0 ] && [ "$MODE" != continue ] && [ "$OWN_PROJECT_STATE" = "NOT EMPTY" ]; then
  die "$OWN_PROJECT exists and is not empty (earlier transcripts the author could Read); move it aside first"
fi

# 4. cost: one cap over smoke and r1; r2 has its own (runs/r2/CAP_USD)
# shellcheck disable=SC2086
CAP="$(PY scripts/cost.py $ALL_LOGS --cap 2>/dev/null)" || die "could not read the $RUN_NAME cap (scripts/cost.py $ALL_LOGS --cap)"
# shellcheck disable=SC2086
(PY scripts/cost.py $ALL_LOGS --check > /dev/null) || die "the US\$$CAP cap is reached or a launch is unresolved (scripts/cost.py)"
# shellcheck disable=SC2086
REMAINING="$(PY scripts/cost.py $ALL_LOGS --remaining 2>/dev/null)"
case "$REMAINING" in ''|*[!0-9.]*) die "could not read the remaining budget";; esac
BUDGET_FLAG="$(PY scripts/launch_checks.py budget "$MODE" "$REMAINING")" || die "less than US\$1 of the cap remains"
case "$BUDGET_FLAG" in ''|*[!0-9.]*) die "could not compute the --max-budget-usd flag";; esac

# 5. session and transcript names
SESSION_TOTAL="0"
case "$MODE" in
  smoke|start)
    if [ "$MODE" = smoke ]; then
      NAME="smoke"
      if [ -n "$(find "$LOGDIR" -name 'smoke*.stream.jsonl' -print -quit 2>/dev/null)" ]; then die "a smoke transcript exists in $LOGDIR (one smoke launch, SPEC.md §12.7)"; fi
    else
      NAME="$RUN_NAME"
      if [ -n "$(find "$LOGDIR" -name '*.stream.jsonl' -print -quit 2>/dev/null)" ]; then die "transcripts exist in $LOGDIR"; fi
      # r2 has no smoke requirement (Matthew skipped the smoke launch for the informal runs, kit/SMOKE_SKIPPED)
      if [ "$RUN_NAME" = r1 ] && [ "$CHECK" = 0 ] && [ ! -e "$KIT/SMOKE_SKIPPED" ] && [ -z "$(find "$SLOP/runs/smoke/logs" -name 'smoke*.stream.jsonl' -print -quit 2>/dev/null)" ]; then
        die "no smoke transcript in runs/smoke/logs: SPEC.md §12.7 runs the smoke launch first (scripts/launch.sh smoke)"
      fi
    fi
    if [ -e "$SID_FILE" ]; then die "a session already exists ($SID_FILE)$([ "$MODE" = smoke ] || echo '; use continue')"; fi
    SID="$(uuidgen | tr 'A-Z' 'a-z')"
    PROMPT_FILE="$PROMPT_MAIN"
    SESSION_ARGS="--session-id $SID"
    ;;
  continue)
    [ -s "$SID_FILE" ] || die "no $RUN_NAME session yet ($SID_FILE)"
    SID="$(cat "$SID_FILE")"
    if PY scripts/audit.py --complete "$LOGDIR" > /dev/null; then
      die "a launch already ended with MODEL COMPLETE; the run is over"
    fi
    [ -f "$LOGDIR/$RUN_NAME.stream.jsonl" ] || die "continue: no $RUN_NAME transcript in $LOGDIR"
    i=1
    while [ "$i" -lt "$K" ]; do
      [ -f "$LOGDIR/$RUN_NAME.c$i.stream.jsonl" ] || die "continue $K: $RUN_NAME.c$i has not run (continuations are numbered in order)"
      i=$((i + 1))
    done
    PROMPT_FILE="$KIT/prompt_continue.md"
    SESSION_ARGS="--resume $SID"
    NAME="$RUN_NAME.c$K"
    # shellcheck disable=SC2086
    SESSION_TOTAL="$(PY scripts/cost.py $ALL_LOGS --session-total "$SID")"
    ;;
esac
LOG="$LOGDIR/$NAME.stream.jsonl"
if [ -e "$LOG" ]; then die "$LOG exists (launches are not re-run silently)"; fi
PROMPT="$(cat "$PROMPT_FILE")"
PROMPT_SHA="$(shasum -a 256 "$PROMPT_FILE" | cut -d' ' -f1)"

# 6. the author's tools run (./tok on a fixed string), and leave the hand-over unchanged (check writes nothing: its
# tool output goes to a temporary file)
if [ "$CHECK" = 1 ]; then
  TOOLCHECK="$(mktemp "${TMPDIR:-/tmp}/slop-toolcheck.XXXXXX")"
  trap 'rm -f "$TOOLCHECK"' EXIT
else
  mkdir -p "$LOGDIR"
  TOOLCHECK="$LOGDIR/$NAME.toolcheck.log"
fi
if ! (cd "$AUTHOR_DIR" && ./tok --text "Resume recaptures the mouse." > "$TOOLCHECK" 2>&1); then
  cat "$TOOLCHECK" >&2
  die "the author's ./tok failed (output above)"
fi
# shellcheck disable=SC2086
PY kit/kit_build.py verify "$AUTHOR_DIR" "$RECORD" $PRISTINE > /dev/null || die "the tool check changed the author dir"

if [ "$CHECK" = 1 ]; then
  echo "check passed: would launch $NAME model=$MODEL fallback=none effort=$EFFORT max_budget_usd=$BUDGET_FLAG (remaining cap $REMAINING of US\$$CAP)"
  echo "  claude $CLAUDE_BIN ($CLAUDE_VERSION)"
  echo "  author dir $AUTHOR_DIR (verified against $RECORD)"
  echo "  prompt approved ($APPROVED_NAME = sha256 of $PROMPT_NAME): $APPROVED$([ "$APPROVED" = yes ] || echo ' -- smoke and start refuse until Matthew approves it')"
  echo "  session project dir $OWN_PROJECT: $OWN_PROJECT_STATE$([ "$OWN_PROJECT_STATE" != "NOT EMPTY" ] || echo ' -- smoke and start refuse')"
  echo "  prompt $PROMPT_FILE sha256 $PROMPT_SHA (${#PROMPT} chars); settings sha256 $(shasum -a 256 "$SETTINGS" | cut -d' ' -f1)"
  if [ "$RUN_NAME" = r2 ]; then
    echo "  smoke launch: not required for r2"
  elif [ -n "$(find "$SLOP/runs/smoke/logs" -name 'smoke*.stream.jsonl' -print -quit 2>/dev/null)" ]; then
    echo "  smoke launch: done (runs/smoke/logs)"
  else
    echo "  smoke launch: not yet run -- start refuses until it has (SPEC.md §12.7: one smoke launch before start)"
  fi
  exit 0
fi

START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
# the launch record: written before claude starts, so a killed launch is still charged its ceiling
printf '{"name": "%s", "mode": "%s", "session_id": "%s", "model": "%s", "fallback": null, "effort": "%s", "max_budget_usd": %s, "max_budget_usd_flag": %s, "remaining_cap_usd": %s, "session_total_before_usd": %s, "utc_start": "%s", "claude_version": "%s", "settings_sha256": "%s", "prompt_sha256": "%s", "prompt_approved": "%s", "handover_sha256": "%s", "author_dir": "%s"}\n' \
  "$NAME" "$MODE" "$SID" "$MODEL" "$EFFORT" "$BUDGET_FLAG" "$BUDGET_FLAG" "$REMAINING" "$SESSION_TOTAL" "$START" \
  "$CLAUDE_VERSION" "$(shasum -a 256 "$SETTINGS" | cut -d' ' -f1)" "$PROMPT_SHA" "$APPROVED" \
  "$(shasum -a 256 "$RECORD" | cut -d' ' -f1)" "$AUTHOR_DIR" > "$LOGDIR/$NAME.launch.json"
if [ "$MODE" != continue ]; then echo "$SID" > "$SID_FILE"; fi
# the parent-session scrub: the Clouds list, then every remaining CLAUDECODE / CLAUDE_* name except the auth
# and provider switches, then the Python and model-override variables
for v in CLAUDECODE CLAUDE_CODE_ENTRYPOINT CLAUDE_CODE_CHILD_SESSION CLAUDE_CODE_MESSAGING_SOCKET \
         CLAUDE_CODE_MESSAGING_TOKEN CLAUDE_CODE_BRIDGE_SESSION_ID CLAUDE_CODE_SESSION_ID CLAUDE_PID \
         CLAUDE_PROJECT_DIR CLAUDE_EFFORT CLAUDE_CODE_SESSION_ATTENDED CLAUDE_CODE_EXECPATH CLAUDE_CODE_SSE_PORT \
         CLAUDE_CODE_SIMPLE CLAUDE_CODE_SAFE_MODE CLAUDE_CODE_EFFORT_LEVEL CLAUDE_CODE_SUBAGENT_MODEL \
         CLAUDE_CODE_SUBAGENT_EFFORT ANTHROPIC_MODEL ANTHROPIC_SMALL_FAST_MODEL ANTHROPIC_DEFAULT_OPUS_MODEL \
         ANTHROPIC_DEFAULT_SONNET_MODEL ANTHROPIC_DEFAULT_HAIKU_MODEL ANTHROPIC_DEFAULT_FABLE_MODEL \
         CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD MAX_THINKING_TOKENS PYTHONPATH PYTHONHOME PYTHONSTARTUP \
         VIRTUAL_ENV UV_PROJECT_ENVIRONMENT; do
  unset "$v" 2>/dev/null || true
done
for v in $(env | sed -n 's/^\(CLAUDE[A-Za-z0-9_]*\)=.*/\1/p'); do
  case "$v" in
    CLAUDE_CODE_OAUTH_TOKEN|CLAUDE_CODE_USE_BEDROCK|CLAUDE_CODE_USE_VERTEX|CLAUDE_CODE_USE_FOUNDRY) ;;
    *) unset "$v" 2>/dev/null || true;;
  esac
done
export CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK=1
export CLAUDE_CODE_NO_MODEL_FALLBACK=1
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000
export CLAUDE_CODE_DISABLE_AUTO_MEMORY=1
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
cd "$SLOP"
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
  if [ "$MODE" != continue ]; then mv "$SID_FILE" "$LOGDIR/$ATTEMPT.SESSION_ID"; fi
  echo "NO INIT EVENT: claude exited $CLAUDE_RC without starting a session (see $LOGDIR/$ATTEMPT.stderr.log)." >&2
  echo "Moved aside as $ATTEMPT.*; nothing is charged. Fix the cause and re-run the same command." >&2
  exit 1
fi
echo "transcript: $LOG"
PY scripts/audit.py "$LOG" "$AUTHOR_DIR" || echo "AUDIT FOUND VIOLATIONS (see $LOGDIR/$NAME.audit.json)" >&2
PY kit/kit_build.py verify "$AUTHOR_DIR" "$RECORD" > /dev/null \
  || echo "HAND-OVER CHANGED during the session (run kit/kit_build.py verify to see what)" >&2
# shellcheck disable=SC2086
PY scripts/cost.py $ALL_LOGS --check || true
# shellcheck disable=SC2086
if PY scripts/cost.py $ALL_LOGS --budget-stops | grep -qx "$NAME"; then
  # shellcheck disable=SC2086
  LEFT="$(PY scripts/cost.py $ALL_LOGS --remaining 2>/dev/null || echo 0)"
  if [ "$MODE" != smoke ] && PY -c 'import sys; sys.exit(0 if float(sys.argv[1]) >= 1 else 1)' "${LEFT:-0}" 2>/dev/null; then
    echo "BUDGET STOP WITH MONEY LEFT: Claude Code ended $NAME with error_max_budget_usd (flag US\$$BUDGET_FLAG), but" >&2
    echo "  scripts/cost.py still shows US\$$LEFT of the US\$$CAP cap. Its budget check must count something other than" >&2
    echo "  what launch.sh assumed (session total before this launch: US\$$SESSION_TOTAL; remaining then: US\$$REMAINING)." >&2
    echo "  Compare total_cost_usd in the last result event of $LOG with $LOGDIR/$NAME.launch.json, record the finding" >&2
    echo "  in kit/NOTES.md, and adjust the flag computation before 'continue <K+1>'. Nothing beyond the cap was spent." >&2
  fi
fi
if [ "$MODE" = smoke ]; then
  echo "SMOKE LAUNCH DONE. Read $LOGDIR/smoke.audit.json (tokens per turn, refusals, thinking blocks with text), then"
  echo "move $AUTHOR_DIR and $OWN_PROJECT aside before 'start' (SPEC.md §12.7)."
elif PY scripts/audit.py --complete "$LOGDIR" > /dev/null; then
  echo "MODEL COMPLETE: the run is over. Archive work/ to runs/$RUN_NAME/ and score it."
else
  echo "The session ended without MODEL COMPLETE (an API or limit error, or a reply with no tool call). Read the end of $LOG; to resume, run: scripts/launch.sh$([ "$RUN_NAME" = r1 ] || echo " --run $RUN_NAME") continue <K>"
fi
