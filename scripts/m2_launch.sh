#!/bin/sh
# Launch the weight-author session (fresh Claude Fable 5.1; Opus 5.5 fallback) for one Phase A step.
#
#   scripts/m2_launch.sh phase-a-strict    [AUTHOR_DIR]
#   scripts/m2_launch.sh phase-a-selfcheck [AUTHOR_DIR]   # resumes the same session (spec §9.3)
#
# Headless (-p). Tools are limited by experiments/m2/settings_<phase>.json (dontAsk: anything not
# allowed is refused). User-level settings, MCP servers, subagents and the web are off. The full
# stream-json transcript is saved to experiments/m2/logs/ and audited with scripts/m2_audit_session.py.
set -eu
PHASE="$1"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
AUTHOR_DIR="${2:-$(dirname "$REPO")/HypaNet-m2-author}"
MODEL="${M2_MODEL:-claude-fable-5-1}"
FALLBACK="${M2_FALLBACK_MODEL:-claude-opus-5-5}"
EFFORT="${M2_EFFORT:-max}"
LOGDIR="$REPO/experiments/m2/logs"
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
  *) echo "unknown phase: $PHASE" >&2; exit 2;;
esac
LOG="$LOGDIR/$PHASE.stream.jsonl"
[ -e "$LOG" ] && { echo "refusing: $LOG exists (sessions are not re-run silently)" >&2; exit 2; }
cd "$AUTHOR_DIR"
echo "launching $PHASE: model=$MODEL fallback=$FALLBACK effort=$EFFORT session=$SID dir=$AUTHOR_DIR"
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
cd "$REPO" && uv run python scripts/m2_audit_session.py "$LOG" "$AUTHOR_DIR" "$PHASE"
