#!/bin/sh
# Launch the v0.3 weight-author session (fresh claude-fable-5-1; claude-opus-5-5 fallback), spec v0.3 §8-§9.
#
#   scripts/v03_launch.sh phase-a                 [AUTHOR_DIR]   # new session (refuses if one exists)
#   scripts/v03_launch.sh continue K              [AUTHOR_DIR]   # technical continuation K (1..9), fixed prompt
#   scripts/v03_launch.sh resubmit N ERROR_FILE   [AUTHOR_DIR]   # build rejection N (1..3): runner error text only
#   scripts/v03_launch.sh phase-b                 [AUTHOR_DIR]   # resumes the same session (spec §9.3)
#
# continue / resubmit apply to the current phase: v03-phase-b once experiments/v0_3/logs/v03-phase-b.stream.jsonl
# exists, otherwise v03-phase-a (override with V03_PHASE=v03-phase-a|v03-phase-b).
# AUTHOR_DIR defaults to ../HypaNet-v03-author (scripts/v03_prepare_author_dir.sh).
#
# Headless (-p), with exactly the flags of scripts/m2_launch.sh: --settings <phase file> with
# --setting-sources project, --permission-mode dontAsk (anything not allowed is refused), --strict-mcp-config,
# --disallowedTools Agent Workflow WebFetch WebSearch Artifact Skill, --effort max. Auto-memory is off
# (CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 and "autoMemoryEnabled": false in the settings files).
# No launch once the phase's cost cap is reached (scripts/v03_cost.py: A US$500, B US$250).
# Transcripts go to experiments/v0_3/logs/ (never overwritten) and are audited with scripts/v03_audit_session.py.
set -eu
MODE="${1:?usage: v03_launch.sh phase-a|continue K|resubmit N ERROR_FILE|phase-b [AUTHOR_DIR]}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
KIT="$REPO/experiments/v0_3/author_kit"
LOGDIR="$REPO/experiments/v0_3/logs"
SID_FILE="$REPO/experiments/v0_3/SESSION_ID"
RESUBMIT_CAP=3
die() { echo "refusing: $*" >&2; exit 2; }
case "$MODE" in
  continue) K="${2:?K}"; AUTHOR_DIR="${3:-$(dirname "$REPO")/HypaNet-v03-author}";;
  resubmit) N="${2:?N}"; ERROR_FILE="${3:?ERROR_FILE}"; AUTHOR_DIR="${4:-$(dirname "$REPO")/HypaNet-v03-author}";;
  phase-a|phase-b) AUTHOR_DIR="${2:-$(dirname "$REPO")/HypaNet-v03-author}";;
  *) die "unknown mode $MODE";;
esac
[ -d "$AUTHOR_DIR/packet" ] && [ -x "$AUTHOR_DIR/py" ] || die "$AUTHOR_DIR is not a prepared v0.3 author dir"
# Harness scratch in the system temp roots (pytest's pytest-of-$USER holds spec copies, rendered data and
# packets) must not exist while an author session runs, even though the settings and ./py deny those roots.
USER_NAME="${USER:-$(id -un)}"
SYS_TMP="$(cd "$REPO" && uv run python -c 'import os, tempfile; print(os.path.realpath(tempfile.gettempdir()))')"
for t in "$SYS_TMP" "${TMPDIR:-/tmp}" /tmp /private/tmp; do
  [ -e "$t/pytest-of-$USER_NAME" ] && die "harness scratch $t/pytest-of-$USER_NAME exists; remove it (rm -rf \"$t/pytest-of-$USER_NAME\") and do not run the test suite while the author session runs"
done
MODEL="${V03_MODEL:-claude-fable-5-1}"
FALLBACK="${V03_FALLBACK_MODEL:-claude-opus-5-5}"
EFFORT="${V03_EFFORT:-max}"
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${V03_MAX_OUTPUT_TOKENS:-128000}"
export CLAUDE_CODE_DISABLE_AUTO_MEMORY=1
mkdir -p "$LOGDIR"

current_phase() {
  if [ -n "${V03_PHASE:-}" ]; then echo "$V03_PHASE"
  elif [ -e "$LOGDIR/v03-phase-b.stream.jsonl" ]; then echo "v03-phase-b"
  else echo "v03-phase-a"; fi
}
settings_for() { echo "$KIT/settings_$(echo "$1" | tr '-' '_').json"; }
sha_of() { shasum -a 256 "$1" | cut -d' ' -f1; }
sha_field() {
  [ -s "$1" ] || die "$1 missing or empty"
  _h="$(head -n 1 "$1" | cut -d' ' -f1)"
  case "$_h" in *[!0-9a-f]*|"") die "$1 does not hold a sha256";; esac
  [ "${#_h}" -eq 64 ] || die "$1 does not hold a sha256"
  echo "$_h"
}
check_phase_b_inputs() {
  A_SHA="$(sha_field "$REPO/weight_program/v0_3/phase-a/artifact.sha256")"
  [ "$(sha_of "$REPO/weight_program/v0_3/phase-a/artifact.dlw")" = "$A_SHA" ] || die "archived Phase A artifact.dlw does not match its artifact.sha256"
  D_SHA="$(sha_field "$REPO/experiments/v0_3/disclosure/DISCLOSURE.sha256")"
  [ "$(sha_of "$REPO/experiments/v0_3/disclosure/DISCLOSURE.json")" = "$D_SHA" ] || die "experiments/v0_3/disclosure/DISCLOSURE.json does not match its .sha256"
  [ -f "$AUTHOR_DIR/disclosure/DISCLOSURE.json" ] || die "$AUTHOR_DIR/disclosure/DISCLOSURE.json missing"
  [ "$(sha_of "$AUTHOR_DIR/disclosure/DISCLOSURE.json")" = "$D_SHA" ] || die "author-dir DISCLOSURE.json differs from the registered disclosure"
  [ -d "$AUTHOR_DIR/output/v03-phase-b" ] || die "$AUTHOR_DIR/output/v03-phase-b missing"
  [ -f "$KIT/prompt_v03_phase_b.md" ] || die "$KIT/prompt_v03_phase_b.md missing (written at V3 with the disclosure)"
}
check_cap() {
  uv run --project "$REPO" python "$REPO/scripts/v03_cost.py" --phase "$1" --check > /dev/null \
    || die "cost cap reached for $1 (scripts/v03_cost.py)"
}

case "$MODE" in
  phase-a)
    TARGET="v03-phase-a"; NAME="$TARGET"
    [ -e "$SID_FILE" ] && die "a v0.3 weight-author session already exists ($SID_FILE)"
    check_cap "$TARGET"
    SID="$(uuidgen | tr 'A-Z' 'a-z')"
    echo "$SID" > "$SID_FILE"
    PROMPT="$(cat "$KIT/prompt_v03_phase_a.md")"
    SESSION_ARGS="--session-id $SID"
    ;;
  phase-b)
    TARGET="v03-phase-b"; NAME="$TARGET"
    check_phase_b_inputs
    check_cap "$TARGET"
    SID="$(cat "$SID_FILE")"
    PROMPT="$(sed -e "s/{A_SHA256}/$A_SHA/g" -e "s/{DISCLOSURE_SHA256}/$D_SHA/g" "$KIT/prompt_v03_phase_b.md")"
    SESSION_ARGS="--resume $SID"
    ;;
  continue)
    TARGET="$(current_phase)"
    case "$K" in [1-9]) ;; *) die "continue: K must be 1..9";; esac
    [ "$TARGET" = "v03-phase-b" ] && check_phase_b_inputs
    check_cap "$TARGET"
    SID="$(cat "$SID_FILE")"
    PROMPT="$(sed "s/{PHASE}/$TARGET/g" "$KIT/prompt_continue.md")"
    SESSION_ARGS="--resume $SID"
    NAME="$TARGET.c$K"
    ;;
  resubmit)
    TARGET="$(current_phase)"
    case "$N" in 1|2|3) ;; *) die "resubmit: N must be 1..$RESUBMIT_CAP";; esac
    [ -s "$ERROR_FILE" ] || die "resubmit: error file missing or empty"
    # The error text is disclosed verbatim: it must be the runner's capped error text (BuildError /
    # attempts/NNNN.json "error", at most 2000 characters), never a child's raw output ("operator_detail").
    [ "$(wc -c < "$ERROR_FILE")" -le 2100 ] || die "resubmit: $ERROR_FILE is longer than the runner's error cap (2000 characters); pass only the runner's error text"
    echo "--- error text to be disclosed (review it before the session starts) ---" >&2
    cat "$ERROR_FILE" >&2
    echo "---" >&2
    [ "$TARGET" = "v03-phase-b" ] && check_phase_b_inputs
    check_cap "$TARGET"
    SID="$(cat "$SID_FILE")"
    mkdir -p "$REPO/experiments/v0_3/resubmissions"
    [ -e "$REPO/experiments/v0_3/resubmissions/$TARGET.r$N.error.txt" ] && die "resubmission $N already recorded"
    cp "$ERROR_FILE" "$REPO/experiments/v0_3/resubmissions/$TARGET.r$N.error.txt"
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
SETTINGS="$(settings_for "$TARGET")"
[ -f "$SETTINGS" ] || die "settings file $SETTINGS missing"
LOG="$LOGDIR/$NAME.stream.jsonl"
[ -e "$LOG" ] && die "$LOG exists (sessions are not re-run silently)"
cd "$AUTHOR_DIR"
echo "launching $NAME: model=$MODEL fallback=$FALLBACK effort=$EFFORT max_output=$CLAUDE_CODE_MAX_OUTPUT_TOKENS session=$SID dir=$AUTHOR_DIR"
# shellcheck disable=SC2086
claude -p "$PROMPT" \
  --model "$MODEL" --fallback-model "$FALLBACK" --effort "$EFFORT" \
  $SESSION_ARGS \
  --settings "$SETTINGS" --setting-sources project --permission-mode dontAsk \
  --strict-mcp-config \
  --disallowedTools Agent Workflow WebFetch WebSearch Artifact Skill \
  --output-format stream-json --verbose \
  < /dev/null > "$LOG" 2> "$LOGDIR/$NAME.stderr.log" || echo "claude exited non-zero (see logs)" >&2
echo "transcript: $LOG"
cd "$REPO"
uv run python scripts/v03_audit_session.py "$LOG" "$AUTHOR_DIR" "$TARGET" || echo "AUDIT FOUND VIOLATIONS (see the .audit.json next to $LOG)" >&2
uv run python scripts/v03_cost.py --phase "$TARGET"
