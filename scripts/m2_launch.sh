#!/bin/sh
# Launch the weight-author session (fresh Claude Fable 5.1; Opus 5.5 fallback) for one Phase A or B step.
#
#   scripts/m2_launch.sh phase-a-strict    [AUTHOR_DIR]
#   scripts/m2_launch.sh phase-a-selfcheck [AUTHOR_DIR]   # resumes the same session (spec §9.3)
#   scripts/m2_launch.sh phase-b-revision  [AUTHOR_DIR]   # resumes the same session (spec §9.4, experiments/m3/)
#       Needs the archived A-selfcheck artefact and the disclosure handed over by
#       scripts/m3_prepare_author_dir.sh. Settings experiments/m3/settings_phase_b.json; logs in experiments/m3/logs/.
#   scripts/m2_launch.sh resubmit PHASE N ERROR_FILE [AUTHOR_DIR]
#       After the runner rejects a submission (BuildError), resume the same session with ONLY the
#       runner's error text (ERROR_FILE). N is 1..3 per phase (pre-registered cap, experiments/m2/README.md).
#   scripts/m2_launch.sh continue PHASE K [AUTHOR_DIR]
#       Technical continuation after an API/limit error ended a response before anything was submitted
#       (no feedback; fixed prompt experiments/m2/prompt_continue.md). Logged as PHASE.cK.
#   PHASE for resubmit/continue: phase-a-strict, phase-a-selfcheck or phase-b-revision.
#
# Headless (-p). Tools are limited by experiments/m2/settings_<phase>.json, or
# experiments/m3/settings_phase_b.json for phase-b-revision (dontAsk: anything not
# allowed is refused). User-level settings, MCP servers, subagents and the web are off. The full
# stream-json transcript is saved to experiments/m2/logs/ (experiments/m3/logs/ for phase-b-revision)
# and audited with scripts/m2_audit_session.py.
set -eu
PHASE="$1"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RESUBMIT_CAP=3
if [ "$PHASE" = "continue" ]; then
  TARGET="$2"; K="$3"; AUTHOR_DIR="${4:-$(dirname "$REPO")/HypaNet-m2-author}"
elif [ "$PHASE" = "resubmit" ]; then
  TARGET="$2"; N="$3"; ERROR_FILE="$4"; AUTHOR_DIR="${5:-$(dirname "$REPO")/HypaNet-m2-author}"
else
  AUTHOR_DIR="${2:-$(dirname "$REPO")/HypaNet-m2-author}"
fi
MODEL="${M2_MODEL:-claude-fable-5-1}"
FALLBACK="${M2_FALLBACK_MODEL:-claude-opus-5-5}"
EFFORT="${M2_EFFORT:-max}"
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${M2_MAX_OUTPUT_TOKENS:-128000}"
LOGDIR="$REPO/experiments/m2/logs"
RESUBDIR="$REPO/experiments/m2/resubmissions"
# Settings file of a phase (phase-b-revision lives in experiments/m3/).
settings_for() {
  case "$1" in
    phase-b-revision) echo "$REPO/experiments/m3/settings_phase_b.json";;
    *) echo "$REPO/experiments/m2/settings_$(echo "$1" | tr '-' '_').json";;
  esac
}
# First field of a sha256 file ("HEX  name" or bare "HEX"); refuses unless it is 64 lowercase hex digits.
sha_field() {
  [ -s "$1" ] || { echo "refusing: $1 missing or empty" >&2; exit 2; }
  _h="$(head -n 1 "$1" | cut -d' ' -f1)"
  case "$_h" in *[!0-9a-f]*|"") echo "refusing: $1 does not hold a sha256" >&2; exit 2;; esac
  [ "${#_h}" -eq 64 ] || { echo "refusing: $1 does not hold a sha256" >&2; exit 2; }
  echo "$_h"
}
sha_of() { shasum -a 256 "$1" | cut -d' ' -f1; }
# Phase B preconditions: archived A-selfcheck artefact and the handed-over disclosure, all hashes matching.
check_phase_b_inputs() {
  A_SHA="$(sha_field "$REPO/weight_program/phase-a-selfcheck/artifact.sha256")" || exit 2
  [ -f "$REPO/weight_program/phase-a-selfcheck/artifact.dlw" ] || { echo "refusing: archived A-selfcheck artifact.dlw missing" >&2; exit 2; }
  [ "$(sha_of "$REPO/weight_program/phase-a-selfcheck/artifact.dlw")" = "$A_SHA" ] || { echo "refusing: A-selfcheck artifact.dlw does not match artifact.sha256" >&2; exit 2; }
  D_SHA="$(sha_field "$REPO/experiments/m3/disclosure/DISCLOSURE.sha256")" || exit 2
  [ -f "$REPO/experiments/m3/disclosure/DISCLOSURE.json" ] || { echo "refusing: experiments/m3/disclosure/DISCLOSURE.json missing" >&2; exit 2; }
  [ "$(sha_of "$REPO/experiments/m3/disclosure/DISCLOSURE.json")" = "$D_SHA" ] || { echo "refusing: experiments/m3/disclosure/DISCLOSURE.json does not match its .sha256" >&2; exit 2; }
  [ -f "$AUTHOR_DIR/disclosure/DISCLOSURE.json" ] || { echo "refusing: $AUTHOR_DIR/disclosure/DISCLOSURE.json missing (run scripts/m3_prepare_author_dir.sh)" >&2; exit 2; }
  [ "$(sha_of "$AUTHOR_DIR/disclosure/DISCLOSURE.json")" = "$D_SHA" ] || { echo "refusing: author-dir DISCLOSURE.json sha differs from experiments/m3/disclosure/DISCLOSURE.sha256" >&2; exit 2; }
  [ "$(sha_field "$AUTHOR_DIR/disclosure/DISCLOSURE.sha256")" = "$D_SHA" ] || { echo "refusing: author-dir DISCLOSURE.sha256 differs from the repo copy" >&2; exit 2; }
  [ -d "$AUTHOR_DIR/output/phase-b-revision" ] || { echo "refusing: $AUTHOR_DIR/output/phase-b-revision missing (run scripts/m3_prepare_author_dir.sh)" >&2; exit 2; }
  [ -f "$REPO/experiments/m3/HANDOVER.json" ] && grep -q "$D_SHA" "$REPO/experiments/m3/HANDOVER.json" && grep -q "$A_SHA" "$REPO/experiments/m3/HANDOVER.json" \
    || { echo "refusing: experiments/m3/HANDOVER.json missing or does not record these hashes" >&2; exit 2; }
}
case "$PHASE" in
  phase-b-revision) LOGDIR="$REPO/experiments/m3/logs"; RESUBDIR="$REPO/experiments/m3/resubmissions";;
  continue|resubmit) if [ "$TARGET" = "phase-b-revision" ]; then LOGDIR="$REPO/experiments/m3/logs"; RESUBDIR="$REPO/experiments/m3/resubmissions"; fi;;
esac
mkdir -p "$LOGDIR"
case "$PHASE" in
  phase-a-strict)
    SETTINGS="$REPO/experiments/m2/settings_phase_a_strict.json"
    [ -e "$REPO/experiments/m2/SESSION_ID" ] && { echo "refusing: a weight-author session already exists" >&2; exit 2; }
    SID="$(uuidgen | tr 'A-Z' 'a-z')"
    echo "$SID" > "$REPO/experiments/m2/SESSION_ID"
    PROMPT="$(cat "$REPO/experiments/m2/prompt_phase_a_strict.md")"
    SESSION_ARGS="--session-id $SID"
    ;;
  phase-a-selfcheck)
    SETTINGS="$REPO/experiments/m2/settings_phase_a_selfcheck.json"
    SID="$(cat "$REPO/experiments/m2/SESSION_ID")"
    STRICT_SHA="$(cut -d' ' -f1 "$REPO/weight_program/phase-a-strict/artifact.sha256")"
    PROMPT="$(sed "s/{A_STRICT_SHA256}/$STRICT_SHA/" "$REPO/experiments/m2/prompt_phase_a_selfcheck.md")"
    SESSION_ARGS="--resume $SID"
    ;;
  phase-b-revision)
    SETTINGS="$REPO/experiments/m3/settings_phase_b.json"
    SID="$(cat "$REPO/experiments/m2/SESSION_ID")"
    check_phase_b_inputs
    PROMPT="$(sed -e "s/{A_SELFCHECK_SHA256}/$A_SHA/g" -e "s/{DISCLOSURE_SHA256}/$D_SHA/g" "$REPO/experiments/m3/prompt_phase_b.md")"
    SESSION_ARGS="--resume $SID"
    ;;
  continue)
    case "$TARGET" in phase-a-strict|phase-a-selfcheck|phase-b-revision) ;; *) echo "continue: bad phase $TARGET" >&2; exit 2;; esac
    case "$K" in [1-9]) ;; *) echo "continue: K must be 1..9" >&2; exit 2;; esac
    if [ "$TARGET" = "phase-b-revision" ]; then check_phase_b_inputs; fi
    SETTINGS="$(settings_for "$TARGET")"
    SID="$(cat "$REPO/experiments/m2/SESSION_ID")"
    PROMPT="$(sed "s/{PHASE}/$TARGET/" "$REPO/experiments/m2/prompt_continue.md")"
    SESSION_ARGS="--resume $SID"
    PHASE="$TARGET.c$K"
    ;;
  resubmit)
    case "$TARGET" in phase-a-strict|phase-a-selfcheck|phase-b-revision) ;; *) echo "resubmit: bad phase $TARGET" >&2; exit 2;; esac
    case "$N" in 1|2|3) ;; *) echo "resubmit: N must be 1..$RESUBMIT_CAP" >&2; exit 2;; esac
    [ -s "$ERROR_FILE" ] || { echo "resubmit: error file missing or empty" >&2; exit 2; }
    if [ "$TARGET" = "phase-b-revision" ]; then check_phase_b_inputs; fi
    SETTINGS="$(settings_for "$TARGET")"
    SID="$(cat "$REPO/experiments/m2/SESSION_ID")"
    mkdir -p "$RESUBDIR"
    cp "$ERROR_FILE" "$RESUBDIR/$TARGET.r$N.error.txt"
    PROMPT="$(uv run --project "$REPO" python - "$REPO/experiments/m2/prompt_resubmit.md" "$TARGET" "$N" "$RESUBMIT_CAP" "$ERROR_FILE" <<'PYEOF'
import sys
t, phase, n, cap, err = sys.argv[1:6]
print(open(t).read().replace("{PHASE}", phase).replace("{N}", n).replace("{CAP}", cap).replace("{ERROR_TEXT}", open(err).read().strip()))
PYEOF
)"
    SESSION_ARGS="--resume $SID"
    PHASE="$TARGET.r$N"
    ;;
  *) echo "unknown phase: $PHASE" >&2; exit 2;;
esac
LOG="$LOGDIR/$PHASE.stream.jsonl"
[ -e "$LOG" ] && { echo "refusing: $LOG exists (sessions are not re-run silently)" >&2; exit 2; }
cd "$AUTHOR_DIR"
echo "launching $PHASE: model=$MODEL fallback=$FALLBACK effort=$EFFORT max_output=$CLAUDE_CODE_MAX_OUTPUT_TOKENS session=$SID dir=$AUTHOR_DIR"
# shellcheck disable=SC2086
claude -p "$PROMPT" \
  --model "$MODEL" --fallback-model "$FALLBACK" --effort "$EFFORT" \
  $SESSION_ARGS \
  --settings "$SETTINGS" --setting-sources project --permission-mode dontAsk \
  --strict-mcp-config \
  --disallowedTools Agent Workflow WebFetch WebSearch Artifact Skill \
  --output-format stream-json --verbose \
  < /dev/null > "$LOG" 2> "$LOGDIR/$PHASE.stderr.log" || echo "claude exited non-zero (see logs)" >&2
echo "transcript: $LOG"
AUDIT_PHASE="${PHASE%%.[rc]*}"
cd "$REPO" && uv run python scripts/m2_audit_session.py "$LOG" "$AUTHOR_DIR" "$AUDIT_PHASE"
