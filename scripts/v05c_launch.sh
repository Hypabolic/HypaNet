#!/bin/sh
# Launch the v0.5 Phase C author session: a NEW claude-fable-5-1 session (no fallback model) that improves Phase B's
# network (weight_program/v0_5/phase-b/artifact.dlw) from two fixed disclosures: Phase B's 200 labelled validation
# windows (V001..V200, network_class = Phase B's class) and 200 labelled windows from the second (OOD) generator
# (W001..W200).
#
#   scripts/v05c_launch.sh phase-c              [AUTHOR_DIR]   # the new session (refuses if one exists)
#   scripts/v05c_launch.sh continue K           [AUTHOR_DIR]   # technical continuation K (1..9), fixed prompt
#   scripts/v05c_launch.sh resubmit N ERROR_FILE [AUTHOR_DIR]  # build rejection N (1..3): runner error text only
#   scripts/v05c_launch.sh check                [AUTHOR_DIR]   # every pre-launch check; launches nothing
#
# Kit mode (tests only): V05C_KIT=1 with V05C_RECORD, V05C_R1_DIR, V05C_B_DIR, V05C_DISCLOSURE_DIR and V05C_LOGDIR
# runs `check` against a kit hand-over (scripts/v05c_prepare_author_dir.sh in kit mode); every other mode is refused.
#
# AUTHOR_DIR defaults to ../HypaNet-v05c-author (scripts/v05c_prepare_author_dir.sh). phase-c sends
# prompt_v05_phase_c.md with {R1_SHA256}, {B_SHA256}, {DISCLOSURE_SHA256} and {DISCLOSURE_W_SHA256} filled in under
# a new --session-id; continue and resubmit resume that session (experiments/v0_5/phase_c/SESSION_ID) with
# experiments/v0_5/phase_c/author_kit/prompt_continue.md or prompt_resubmit.md (the runner's error text from
# scripts/v05c_run_build.py, at most 2000 characters, verbatim).
#
# Flags: exactly Phase B's (scripts/v05b_launch.sh): claude -p, --model claude-fable-5-1, --effort max, no
# --fallback-model and CLAUDE_CODE_NO_MODEL_FALLBACK=1, --settings <kit settings> --setting-sources project,
# --permission-mode dontAsk --permission-prompts none, --strict-mcp-config, --tools Read,Glob,Grep,Write,Edit,Bash,
# --disallowedTools Agent Workflow WebFetch WebSearch Artifact Skill, --disable-slash-commands, --max-budget-usd
# <remaining Phase C cap> (recorded in experiments/v0_5/phase_c/logs/NAME.launch.json before claude starts),
# --output-format stream-json --verbose; CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000, CLAUDE_CODE_DISABLE_AUTO_MEMORY=1,
# CLAUDE_CODE_DISABLE_CLAUDE_MDS=1; the environment scrubbed of variables that link it to a parent session.
#
# Refuses before claude: an unknown mode; an author dir that does not verify against
# experiments/v0_5/phase_c/HANDOVER.json (agent_loop_lab.v05.kit_c verify: packet/, tools/, r1/, phase_b/,
# disclosure/, py unchanged and read-only, the kit files unchanged since the hand-over, the repo's r1 and Phase B
# artefacts and both disclosures still the ones handed over); the packet manifest not matching the v0.3 registry;
# (real mode) the held-out evaluation not frozen, since the held-out sets are fixed before launch: heldout/README.md,
# exactly heldout/HELDOUT_MANIFEST.json (official, passing phase_c_data.manifest_problems, with the scoring code's
# sha256 matching what it pinned), experiments/v0_5/phase_c/DISCLOSURE_OOD_PRIVATE.json and the scoring code
# (phase_c_data.SCORING_CODE_FILES) not all committed unchanged (phase_c_eval prelaunch-check);
# the US$250 Phase C cap reached or a killed launch of unknown cost (scripts/v05c_cost.py); harness scratch
# pytest-of-$USER in the system temp roots; an existing transcript of the same name; a resubmission error text
# longer than the runner's cap. A launch with no init event is moved aside as NAME.noinit.<UTC>.*, as Phase B.
# Afterwards: the audit (scripts/v05c_audit_session.py), kit_c verify again and the cost summary.
set -eu
MODE="${1:?usage: v05c_launch.sh phase-c|continue K|resubmit N ERROR_FILE|check [AUTHOR_DIR]}"
REPO="$(cd "$(dirname "$0")/.." && pwd -P)"
ROOT="$REPO/experiments/v0_5/phase_c"
KIT="$ROOT/author_kit"
LOGDIR="$ROOT/logs"
SID_FILE="$ROOT/SESSION_ID"
RESUB="$ROOT/resubmissions"
RECORD="$ROOT/HANDOVER.json"
R1_DIR="$REPO/experiments/v0_5/runs/r1"
B_DIR="$REPO/weight_program/v0_5/phase-b"
HELDOUT="$ROOT/heldout"
DISC_DIR="$ROOT/disclosure"
PHASE="v05-phase-c"
RESUBMIT_CAP=3
DEFAULT_DIR="$(dirname "$REPO")/HypaNet-v05c-author"
die() { echo "refusing: $*" >&2; exit 2; }
case "$MODE" in
  continue) K="${2:?K}"; AUTHOR_DIR="${3:-$DEFAULT_DIR}";;
  resubmit) N="${2:?N}"; ERROR_FILE="${3:?ERROR_FILE}"; AUTHOR_DIR="${4:-$DEFAULT_DIR}";;
  phase-c|check) AUTHOR_DIR="${2:-$DEFAULT_DIR}";;
  *) die "unknown mode $MODE";;
esac
case "$MODE" in
  continue) case "$K" in [1-9]) ;; *) die "continue: K must be 1..9";; esac;;
  resubmit) case "$N" in 1|2|3) ;; *) die "resubmit: N must be 1..$RESUBMIT_CAP";; esac;;
esac
KIT_MODE="${V05C_KIT:-0}"
if [ "$KIT_MODE" = 1 ]; then
  # kit / dry runs (tests): every check against a kit hand-over; nothing can be launched from one
  [ "$MODE" = check ] || die "kit mode (V05C_KIT=1) runs only 'check'"
  RECORD="${V05C_RECORD:?kit mode needs V05C_RECORD}"
  R1_DIR="${V05C_R1_DIR:?kit mode needs V05C_R1_DIR}"
  B_DIR="${V05C_B_DIR:?kit mode needs V05C_B_DIR}"
  DISC_DIR="${V05C_DISCLOSURE_DIR:?kit mode needs V05C_DISCLOSURE_DIR}"
  LOGDIR="${V05C_LOGDIR:?kit mode needs V05C_LOGDIR}"
  SID_FILE="$LOGDIR/SESSION_ID"
fi
[ -d "$AUTHOR_DIR/packet" ] && [ -x "$AUTHOR_DIR/py" ] && [ -d "$AUTHOR_DIR/r1" ] && [ -d "$AUTHOR_DIR/phase_b" ] \
  && [ -f "$AUTHOR_DIR/tools/lib/hn_v05/revision_c.py" ] \
  || die "$AUTHOR_DIR is not a prepared v0.5 Phase C author dir (scripts/v05c_prepare_author_dir.sh)"
AUTHOR_DIR="$(cd "$AUTHOR_DIR" && pwd -P)"
case "$AUTHOR_DIR" in "$REPO"|"$REPO"/*) die "author dir must be outside the repo";; esac
[ -f "$RECORD" ] || die "$RECORD missing (scripts/v05c_prepare_author_dir.sh writes it)"

# Harness scratch in the system temp roots (pytest's pytest-of-$USER) must not exist while an author session runs.
USER_NAME="${USER:-$(id -un)}"
SYS_TMP="$(cd "$REPO" && uv run python -c 'import os, tempfile; print(os.path.realpath(tempfile.gettempdir()))')"
USER_TMP="$(getconf DARWIN_USER_TEMP_DIR 2>/dev/null || echo /tmp)"
for t in "$SYS_TMP" "${TMPDIR:-/tmp}" "$USER_TMP" /tmp /private/tmp; do
  [ -e "$t/pytest-of-$USER_NAME" ] && die "harness scratch $t/pytest-of-$USER_NAME exists; remove it (rm -rf \"$t/pytest-of-$USER_NAME\") and do not run the test suite while the author session runs"
done

sha_of() { shasum -a 256 "$1" | cut -d' ' -f1; }
sha_field() {
  [ -s "$1" ] || die "$1 missing or empty"
  _h="$(head -n 1 "$1" | cut -d' ' -f1)"
  case "$_h" in *[!0-9a-f]*|"") die "$1 does not hold a sha256";; esac
  [ "${#_h}" -eq 64 ] || die "$1 does not hold a sha256"
  echo "$_h"
}
cd "$REPO"
P_SHA="$(sed -n 's/^packet_sha256:[[:space:]]*\([0-9a-f]\{64\}\)[[:space:]]*$/\1/p' experiments/registry_v0_3.yaml)"
[ "$(sha_of "$AUTHOR_DIR/packet/PACKET_MANIFEST.json")" = "$P_SHA" ] || die "author-dir packet does not match the registry packet_sha256"
if [ "$KIT_MODE" = 1 ]; then
  grep -q '"kit": true' "$RECORD" || die "kit mode needs a kit record"
else
  grep -q '"kit": false' "$RECORD" || die "$RECORD is a kit (dry-run) record; the real hand-over needs a real record"
fi
uv run python -m agent_loop_lab.v05.kit_c verify "$AUTHOR_DIR" "$RECORD" --r1-dir "$R1_DIR" --b-dir "$B_DIR" \
  --disclosure-dir "$DISC_DIR" > /dev/null \
  || die "the author dir does not verify against $RECORD (see PROBLEM lines above)"
R1_SHA="$(sha_field "$AUTHOR_DIR/r1/artifact.sha256")"
B_SHA="$(sha_field "$AUTHOR_DIR/phase_b/artifact.sha256")"
[ "$(sha_of "$AUTHOR_DIR/phase_b/artifact.dlw")" = "$B_SHA" ] || die "author-dir phase_b/artifact.dlw differs from its sha256"
D_SHA="$(sha_field "$AUTHOR_DIR/disclosure/DISCLOSURE.sha256")"
[ "$(sha_of "$AUTHOR_DIR/disclosure/DISCLOSURE.json")" = "$D_SHA" ] || die "author-dir DISCLOSURE.json differs from its sha256"
W_SHA="$(sha_field "$AUTHOR_DIR/disclosure/DISCLOSURE_W.sha256")"
[ "$(sha_of "$AUTHOR_DIR/disclosure/DISCLOSURE_W.json")" = "$W_SHA" ] || die "author-dir DISCLOSURE_W.json differs from its sha256"
if [ "$KIT_MODE" != 1 ]; then
  [ "$B_SHA" = "223d074372bf726a0c89f28afe1e1fcf9b53e3bb0ff3973f0cc3acd932661001" ] \
    || die "author-dir phase_b/artifact.dlw is not v0.5 Phase B's artefact 223d0743..."
  # the held-out sets are generated, hashed and pre-registered before launch (experiments/v0_5/phase_c/heldout/)
  [ -f "$HELDOUT/README.md" ] || die "$HELDOUT/README.md missing (pre-register the held-out evaluation before launch)"
  git ls-files --error-unmatch "$HELDOUT/README.md" > /dev/null 2>&1 || die "$HELDOUT/README.md is not committed"
  git diff --quiet HEAD -- "$HELDOUT" || die "$HELDOUT has uncommitted changes"
  # exactly heldout/HELDOUT_MANIFEST.json (official, pre-registered seeds and sizes, scoring code as pinned), the
  # OOD disclosure private record and the scoring code, all committed unchanged (phase_c_eval.prelaunch_problems)
  uv run python -m agent_loop_lab.v05.phase_c_eval prelaunch-check \
    || die "the held-out evaluation is not frozen yet (see PROBLEM lines above)"
fi
[ -d "$AUTHOR_DIR/output/$PHASE" ] || die "$AUTHOR_DIR/output/$PHASE missing"
SETTINGS="$KIT/settings_v05_phase_c.json"
for f in "$SETTINGS" "$KIT/prompt_v05_phase_c.md" "$KIT/prompt_continue.md" "$KIT/prompt_resubmit.md"; do
  [ -f "$f" ] || die "$f missing"
done
grep -q "\"Read(/$REPO/\*\*)\"" "$SETTINGS" || die "$SETTINGS does not deny reads of $REPO (settings name this machine's paths)"
for sib in HypaNet-m2-author HypaNet-v03-author HypaNet-v04-author HypaNet-v05b-author 'HypaNet-v05-*'; do
  grep -qF "\"Read(/$(dirname "$REPO")/$sib/**)\"" "$SETTINGS" || die "$SETTINGS does not deny reads of ../$sib"
done
(uv run python scripts/v05c_cost.py --logs "$LOGDIR" --check > /dev/null) || die "Phase C cost cap reached or unresolved (scripts/v05c_cost.py)"
REMAINING="$(uv run python scripts/v05c_cost.py --logs "$LOGDIR" --remaining 2>/dev/null)"
case "$REMAINING" in ''|*[!0-9.]*) die "could not read the remaining Phase C budget";; esac

MODEL="claude-fable-5-1"
EFFORT="max"
case "$MODE" in
  phase-c|check)
    NAME="v05c"
    if [ "$MODE" = phase-c ]; then [ -e "$SID_FILE" ] && die "a Phase C session already exists ($SID_FILE)"; fi
    SID="$(uuidgen | tr 'A-Z' 'a-z')"
    PROMPT="$(sed -e "s/{R1_SHA256}/$R1_SHA/g" -e "s/{B_SHA256}/$B_SHA/g" -e "s/{DISCLOSURE_SHA256}/$D_SHA/g" \
      -e "s/{DISCLOSURE_W_SHA256}/$W_SHA/g" "$KIT/prompt_v05_phase_c.md")"
    SESSION_ARGS="--session-id $SID"
    ;;
  continue)
    [ -s "$SID_FILE" ] || die "no Phase C session yet ($SID_FILE)"
    SID="$(cat "$SID_FILE")"
    PROMPT="$(sed "s/{PHASE}/$PHASE/g" "$KIT/prompt_continue.md")"
    SESSION_ARGS="--resume $SID"
    NAME="v05c.c$K"
    ;;
  resubmit)
    [ -s "$SID_FILE" ] || die "no Phase C session yet ($SID_FILE)"
    [ -s "$ERROR_FILE" ] || die "resubmit: error file missing or empty"
    # The error text is disclosed verbatim: it must be the runner's capped error text
    # (weight_program/v0_5/phase-c/attempts/NNNN.error.txt, at most 2000 characters), nothing else.
    # characters, not bytes: the runner caps characters and echoes the author's (possibly non-ASCII) text back
    ERR_CHARS="$(uv run python -c 'import sys; print(len(open(sys.argv[1], encoding="utf-8").read()))' "$ERROR_FILE")" \
      || die "resubmit: $ERROR_FILE is not UTF-8 text"
    [ "$ERR_CHARS" -le 2001 ] || die "resubmit: $ERROR_FILE has $ERR_CHARS characters, more than the runner's error cap (2000 + newline)"
    [ -e "$RESUB/$PHASE.r$N.error.txt" ] && die "resubmission $N already recorded"
    echo "--- error text to be disclosed (review it before the session starts) ---" >&2
    cat "$ERROR_FILE" >&2
    echo "---" >&2
    SID="$(cat "$SID_FILE")"
    PROMPT="$(uv run python - "$KIT/prompt_resubmit.md" "$PHASE" "$N" "$RESUBMIT_CAP" "$ERROR_FILE" <<'PYEOF'
import sys
t, phase, n, cap, err = sys.argv[1:6]
print(open(t).read().replace("{PHASE}", phase).replace("{N}", n).replace("{CAP}", cap).replace("{ERROR_TEXT}", open(err).read().strip()))
PYEOF
)"
    SESSION_ARGS="--resume $SID"
    NAME="v05c.r$N"
    ;;
esac
case "$PROMPT" in *"{R1_SHA256}"*|*"{B_SHA256}"*|*"{DISCLOSURE_SHA256}"*|*"{DISCLOSURE_W_SHA256}"*|*"{PHASE}"*|*"{ERROR_TEXT}"*|*"{N}"*|*"{CAP}"*) die "unfilled placeholder in the prompt";; esac
LOG="$LOGDIR/$NAME.stream.jsonl"
[ -e "$LOG" ] && die "$LOG exists (sessions are not re-run silently)"
PROMPT_SHA="$(printf '%s' "$PROMPT" | shasum -a 256 | cut -d' ' -f1)"
if [ "$MODE" = check ]; then
  echo "check passed: would launch $NAME model=$MODEL fallback=none effort=$EFFORT max_budget_usd=$REMAINING dir=$AUTHOR_DIR"
  echo "prompt sha256 $PROMPT_SHA (${#PROMPT} chars); r1 $R1_SHA; base (Phase B) $B_SHA; disclosure $D_SHA; disclosure_w $W_SHA"
  exit 0
fi

command -v claude > /dev/null || die "claude not on PATH"
mkdir -p "$LOGDIR"
START="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
# the launch record: written before claude starts, so a killed launch is still charged its ceiling
printf '{"name": "%s", "phase": "%s", "mode": "%s", "session_id": "%s", "model": "%s", "fallback": null, "effort": "%s", "max_budget_usd": %s, "utc_start": "%s", "settings_sha256": "%s", "prompt_sha256": "%s", "r1_artifact_sha256": "%s", "b_artifact_sha256": "%s", "disclosure_sha256": "%s", "disclosure_w_sha256": "%s", "handover_sha256": "%s"}\n' \
  "$NAME" "$PHASE" "$MODE" "$SID" "$MODEL" "$EFFORT" "$REMAINING" "$START" "$(sha_of "$SETTINGS")" "$PROMPT_SHA" \
  "$R1_SHA" "$B_SHA" "$D_SHA" "$W_SHA" "$(sha_of "$RECORD")" > "$LOGDIR/$NAME.launch.json"
if [ "$MODE" = phase-c ]; then echo "$SID" > "$SID_FILE"; fi
if [ "$MODE" = resubmit ]; then mkdir -p "$RESUB"; cp "$ERROR_FILE" "$RESUB/$PHASE.r$N.error.txt"; fi
for v in CLAUDECODE CLAUDE_CODE_ENTRYPOINT CLAUDE_CODE_CHILD_SESSION CLAUDE_CODE_MESSAGING_SOCKET \
         CLAUDE_CODE_MESSAGING_TOKEN CLAUDE_CODE_BRIDGE_SESSION_ID CLAUDE_CODE_SESSION_ID CLAUDE_PID \
         CLAUDE_PROJECT_DIR CLAUDE_EFFORT CLAUDE_CODE_SESSION_ATTENDED CLAUDE_CODE_EXECPATH CLAUDE_CODE_SSE_PORT \
         CLAUDE_CODE_SIMPLE CLAUDE_CODE_SAFE_MODE CLAUDE_CODE_EFFORT_LEVEL CLAUDE_CODE_SUBAGENT_MODEL ANTHROPIC_MODEL \
         ANTHROPIC_SMALL_FAST_MODEL CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD PYTHONPATH PYTHONHOME PYTHONSTARTUP \
         VIRTUAL_ENV; do
  unset "$v" 2>/dev/null || true
done
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=128000
export CLAUDE_CODE_DISABLE_AUTO_MEMORY=1
export CLAUDE_CODE_NO_MODEL_FALLBACK=1
export CLAUDE_CODE_DISABLE_CLAUDE_MDS=1
cd "$AUTHOR_DIR"
echo "launching $NAME: model=$MODEL fallback=none effort=$EFFORT max_output=$CLAUDE_CODE_MAX_OUTPUT_TOKENS max_budget_usd=$REMAINING session=$SID dir=$AUTHOR_DIR"
CLAUDE_RC=0
# shellcheck disable=SC2086
claude -p "$PROMPT" \
  --model "$MODEL" --effort "$EFFORT" \
  $SESSION_ARGS \
  --settings "$SETTINGS" --setting-sources project \
  --permission-mode dontAsk --permission-prompts none \
  --strict-mcp-config \
  --tools Read,Glob,Grep,Write,Edit,Bash \
  --disallowedTools Agent Workflow WebFetch WebSearch Artifact Skill \
  --disable-slash-commands \
  --max-budget-usd "$REMAINING" \
  --output-format stream-json --verbose \
  < /dev/null > "$LOG" 2> "$LOGDIR/$NAME.stderr.log" || CLAUDE_RC=$?
cd "$REPO"
END="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STARTED=false
grep -q '"type":"system","subtype":"init"' "$LOG" 2>/dev/null && STARTED=true
printf '{"name": "%s", "mode": "%s", "session_id": "%s", "utc_start": "%s", "utc_end": "%s", "claude_rc": %s, "started": %s}\n' \
  "$NAME" "$MODE" "$SID" "$START" "$END" "$CLAUDE_RC" "$STARTED" >> "$LOGDIR/LAUNCHES.jsonl"
[ "$CLAUDE_RC" = 0 ] || echo "claude exited $CLAUDE_RC (see $LOGDIR/$NAME.stderr.log)" >&2
if [ "$STARTED" != true ]; then
  # no init event: no session was created and nothing was spent (as v0.4); keep the evidence under a .noinit name,
  # out of the cost accounting (*.stream.jsonl), and free the launch name, the session id and the resubmission slot
  ATTEMPT="$NAME.noinit.$(date -u +%Y%m%dT%H%M%SZ)"
  mv "$LOG" "$LOGDIR/$ATTEMPT.stream.log" 2>/dev/null || true
  mv "$LOGDIR/$NAME.stderr.log" "$LOGDIR/$ATTEMPT.stderr.log" 2>/dev/null || true
  mv "$LOGDIR/$NAME.launch.json" "$LOGDIR/$ATTEMPT.launch.json"
  if [ "$MODE" = phase-c ]; then mv "$SID_FILE" "$LOGDIR/$ATTEMPT.SESSION_ID"; fi
  if [ "$MODE" = resubmit ]; then mv "$RESUB/$PHASE.r$N.error.txt" "$RESUB/$ATTEMPT.error.txt"; fi
  echo "NO INIT EVENT: claude exited $CLAUDE_RC without starting a session (see $LOGDIR/$ATTEMPT.stderr.log)." >&2
  echo "Moved aside as $ATTEMPT.*; nothing is charged. Fix the cause and re-run the same command." >&2
  exit 1
fi
echo "transcript: $LOG"
uv run python scripts/v05c_audit_session.py "$LOG" "$AUTHOR_DIR" || echo "AUDIT FOUND VIOLATIONS (see the .audit.json next to $LOG)" >&2
uv run python -m agent_loop_lab.v05.kit_c verify "$AUTHOR_DIR" "$RECORD" --r1-dir "$R1_DIR" --b-dir "$B_DIR" \
  --disclosure-dir "$DISC_DIR" > /dev/null \
  || echo "HAND-OVER TAMPERED after the session (see PROBLEM lines above)" >&2
uv run python scripts/v05c_cost.py --logs "$LOGDIR"
