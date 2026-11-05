#!/bin/sh
# Hand the WSP-B disclosure to the weight author's existing directory (PREREGISTRATION §6, §7.1;
# docs/V04_WSP_KERNEL.md §7; modelled on scripts/v03_prepare_phase_b.sh).
#
#   scripts/v04_prepare_phase_b.sh [AUTHOR_DIR]
#
# Run it only after the WSP-A session has ended, WSP-A has been archived (scripts/v04_run_build.py wsp-a, into
# weight_program/v0_4/wsp-a/) and scripts/v04_build_disclosure.py has written experiments/v0_4/disclosure/,
# DISCLOSED_WINDOWS.json and DISCLOSURE_PRIVATE.json (committed). AUTHOR_DIR defaults to ../HypaNet-v04-author
# (prepared by scripts/v04_prepare_author_dir.sh; never inside the repo). Steps:
#   1. the real handover needs experiments/registry_v0_4.yaml and refuses unless experiments/v0_4/HANDOVER.json
#      names AUTHOR_DIR as the (non-kit) author dir, before anything is written. Kit mode (V04_KIT=1, or
#      V04_PREPARE_UNFROZEN=1; kit tests only, with or without the registry) needs V04_WSP_A_ARCHIVE,
#      V04_DISCLOSURE_ROOT and V04_LOGDIR (existing directories, never the real archive, disclosure root or log
#      dir) and writes its handover record to V04_HANDOVER (default AUTHOR_DIR.wsp_b.handover.json, never
#      inside the repo). Whenever the registry exists: scripts/v04_freeze.py --check, and the author dir's
#      packet must be the registered one;
#   2. check the WSP-A archive and the disclosure files against each other (agent_loop_lab.v04.handover
#      wsp-b-inputs --registry, or --kit-unfrozen in kit mode without a registry: BUILD_RECORD.sha256, a
#      journal-checked build, FREEZE.json frozen and consistent under the registry's G1 kernel freeze,
#      DISCLOSURE.sha256, the private record naming that archive and its FREEZE.json, DISCLOSED_WINDOWS.json);
#   3. seed the WSP-B journal LOGDIR/wsp-b/journal/ with a byte-identical copy of LOGDIR/wsp-a/journal/, whose
#      final edit hash must be the archived edit's (handover seed-journal): the model server replays it, so WSP-B
#      starts from the archived WSP-A edit; refused once WSP-B has a ledger or kernel log, idempotent before;
#   4. copy DISCLOSURE.json and DISCLOSURE.sha256 (nothing else) into AUTHOR_DIR/disclosure/ (an existing copy
#      must be byte-identical, otherwise refuse), files 0444, directory 0555;
#   5. make output/wsp-a/ read-only (chmod -R a-w) and check it; create output/wsp-b/scratch/ (refuse if
#      output/wsp-b/ exists and holds anything but an empty scratch/); AUTHOR_DIR and output/ go back to 0555;
#   6. verify the tools (hypaqwen.wsp.install verify: the same files as WSP-A, never reinstalled, since every
#      phase runs the same kernel) and the ./py and ./fwd wrappers; render the WSP-B settings once and run the
#      hook self-test on them;
#   7. write experiments/v0_4/wsp_b/HANDOVER.json (the hashes handed over, for the commit).
# These are the inputs `scripts/v04_launch.sh wsp-b` checks again before every WSP-B launch. Nothing is launched.
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd -P)"
AUTHOR_DIR="${1:-$(dirname "$REPO")/HypaNet-v04-author}"
KIT="$REPO/experiments/v0_4/author_kit"
QWEN="$REPO/envs/qwen"
REGISTRY="$REPO/experiments/registry_v0_4.yaml"
die() { echo "refusing: $*" >&2; exit 2; }
sha_of() { shasum -a 256 "$1" | cut -d' ' -f1; }

AUTHOR_DIR="$(cd "$AUTHOR_DIR" 2>/dev/null && pwd -P)" || die "author dir missing"
case "$AUTHOR_DIR" in "$REPO"|"$REPO"/*) die "author dir must be outside the repo";; esac
for f in packet/PACKET_MANIFEST.json py fwd .venv/bin/python tools/hn_kernel/run_py.py tools/hn_hook/hook_bash.py; do
  [ -e "$AUTHOR_DIR/$f" ] || die "$AUTHOR_DIR is not a prepared v0.4 author dir ($f missing)"
done
[ -d "$AUTHOR_DIR/output/wsp-a" ] || die "$AUTHOR_DIR/output/wsp-a missing"
for f in "$KIT/prompt_wsp_b.md" "$KIT/settings_wsp_b.json" "$KIT/hook_bash.py" "$KIT/py" "$KIT/fwd"; do
  [ -f "$f" ] || die "kit file $f missing"
done
CALLER_PWD="$(pwd -P)"
cd "$REPO"

REAL_ARCHIVE="$REPO/weight_program/v0_4/wsp-a"
REAL_DISC_ROOT="$REPO/experiments/v0_4"
REAL_LOGDIR="$REPO/experiments/v0_4/logs"
# real_path DIR: DIR resolved (a relative DIR against the caller's directory), or empty if it does not exist
real_path() { (cd "$CALLER_PWD" && cd "$1" 2>/dev/null && pwd -P) || true; }

# 1. mode, registry and packet. Kit mode (V04_KIT=1, or V04_PREPARE_UNFROZEN=1) works with or without the
# registry and never uses the real archive, disclosure, log dir or handover record; otherwise the registry is
# required and the author dir must be the one experiments/v0_4/HANDOVER.json hands over (before any write).
if [ "${V04_KIT:-0}" = 1 ] || [ "${V04_PREPARE_UNFROZEN:-0}" = 1 ]; then
  KIT_MODE=1
  A_ARCHIVE="$(real_path "${V04_WSP_A_ARCHIVE:?kit mode needs V04_WSP_A_ARCHIVE}")"
  DISC_ROOT="$(real_path "${V04_DISCLOSURE_ROOT:?kit mode needs V04_DISCLOSURE_ROOT}")"
  LOGDIR="$(real_path "${V04_LOGDIR:?kit mode needs V04_LOGDIR}")"
  [ -n "$A_ARCHIVE" ] && [ -n "$DISC_ROOT" ] && [ -n "$LOGDIR" ] \
    || die "kit mode: V04_WSP_A_ARCHIVE, V04_DISCLOSURE_ROOT and V04_LOGDIR must be existing directories"
  [ "$A_ARCHIVE" = "$(real_path "$REAL_ARCHIVE")" ] && die "kit mode may not use the real WSP-A archive"
  [ "$DISC_ROOT" = "$(real_path "$REAL_DISC_ROOT")" ] && die "kit mode may not use the real disclosure root"
  [ "$LOGDIR" = "$(real_path "$REAL_LOGDIR")" ] && die "kit mode may not use the real log dir"
  HANDOVER="${V04_HANDOVER:-$AUTHOR_DIR.wsp_b.handover.json}"
  case "$HANDOVER" in /*) ;; *) HANDOVER="$CALLER_PWD/$HANDOVER";; esac
  H_DIR="$(real_path "$(dirname "$HANDOVER")")"
  [ -n "$H_DIR" ] || die "kit mode: the directory of $HANDOVER does not exist"
  case "$H_DIR/" in "$REPO"/*) die "a kit handover record may not be written into the repo";; esac
  if [ -f "$REGISTRY" ]; then
    UNFROZEN=0
    REG_ARGS="--registry $REGISTRY"
  else
    echo "WARNING: no $REGISTRY: preparing an UNFROZEN WSP-B kit handover (kit tests only)" >&2
    UNFROZEN=1
    REG_ARGS="--kit-unfrozen"
  fi
elif [ -f "$REGISTRY" ]; then
  KIT_MODE=0
  UNFROZEN=0
  [ -f "$REPO/experiments/v0_4/HANDOVER.json" ] || die "experiments/v0_4/HANDOVER.json missing (scripts/v04_prepare_author_dir.sh)"
  uv run python - "$REPO/experiments/v0_4/HANDOVER.json" "$AUTHOR_DIR" <<'EOF' || die "experiments/v0_4/HANDOVER.json does not hand over $AUTHOR_DIR (a kit dir needs V04_KIT=1)"
import json, sys
h = json.load(open(sys.argv[1]))
assert h["author_dir"] == sys.argv[2] and h["unfrozen_kit_test_dir"] is False, h["author_dir"]
EOF
  A_ARCHIVE="$REAL_ARCHIVE"
  DISC_ROOT="$REAL_DISC_ROOT"
  LOGDIR="$REAL_LOGDIR"
  HANDOVER="$REPO/experiments/v0_4/wsp_b/HANDOVER.json"
  REG_ARGS="--registry $REGISTRY"
else
  die "experiments/registry_v0_4.yaml does not exist (G1 not frozen); set V04_KIT=1 for kit tests only"
fi
case "$REGISTRY" in *" "*) die "the registry path may not contain a space";; esac
case "$LOGDIR" in "$AUTHOR_DIR"|"$AUTHOR_DIR"/*) die "the log dir must be outside the author dir";; esac
if [ -f "$REGISTRY" ]; then
  uv run python scripts/v04_freeze.py --check || die "freeze registry check failed"
  P_SHA="$(sed -n 's/^packet_sha256:[[:space:]]*\([0-9a-f]\{64\}\)[[:space:]]*$/\1/p' "$REGISTRY")"
  [ "${#P_SHA}" -eq 64 ] || die "no packet_sha256 in $REGISTRY"
  [ "$(sha_of "$AUTHOR_DIR/packet/PACKET_MANIFEST.json")" = "$P_SHA" ] \
    || die "the author dir's packet is not the registered one ($P_SHA)"
fi

# 2. archive and disclosure
# shellcheck disable=SC2086
INPUTS="$(uv run python -m agent_loop_lab.v04.handover wsp-b-inputs --archive "$A_ARCHIVE" --disclosure-root "$DISC_ROOT" \
  $REG_ARGS)" \
  || die "WSP-B inputs missing or inconsistent (see above)"
A_EDIT="$(echo "$INPUTS" | sed -n 's/^ *"A_EDIT_HASH": "\([0-9a-f]\{64\}\)",*$/\1/p')"
D_SHA="$(echo "$INPUTS" | sed -n 's/^ *"DISCLOSURE_SHA256": "\([0-9a-f]\{64\}\)",*$/\1/p')"
[ "${#A_EDIT}" -eq 64 ] && [ "${#D_SHA}" -eq 64 ] || die "WSP-B inputs: unreadable hashes"
SRC="$DISC_ROOT/disclosure"

# 3. the WSP-B journal starts as the WSP-A journal
JOURNAL_JSON="$(uv run python -m agent_loop_lab.v04.handover seed-journal --logs "$LOGDIR" --edit-hash "$A_EDIT")" \
  || die "the WSP-B journal could not be seeded (see above)"
echo "journal: $JOURNAL_JSON"

# 4. disclosure copy: idempotent if byte-identical, refused otherwise
DST="$AUTHOR_DIR/disclosure"
# the author dir and output/ are 0555 again however this script ends
trap 'chmod 555 "$AUTHOR_DIR/output" "$AUTHOR_DIR" 2>/dev/null || true' EXIT
chmod u+w "$AUTHOR_DIR"
if [ -e "$DST" ]; then
  [ -d "$DST" ] && [ ! -L "$DST" ] || die "$DST exists and is not a directory"
  for f in "$DST"/* "$DST"/.[!.]*; do
    [ -e "$f" ] || continue
    case "$(basename "$f")" in DISCLOSURE.json|DISCLOSURE.sha256) ;; *) die "$DST holds an unexpected file: $f";; esac
  done
  for f in DISCLOSURE.json DISCLOSURE.sha256; do
    [ -f "$DST/$f" ] || die "$DST exists but $f is missing"
    cmp -s "$SRC/$f" "$DST/$f" || die "$DST/$f differs from $SRC/$f"
  done
  echo "disclosure already handed over (byte-identical): $DST"
else
  mkdir "$DST"
  cp "$SRC/DISCLOSURE.json" "$SRC/DISCLOSURE.sha256" "$DST/"
fi
chmod 444 "$DST/DISCLOSURE.json" "$DST/DISCLOSURE.sha256"
chmod 555 "$DST"
[ "$(sha_of "$DST/DISCLOSURE.json")" = "$D_SHA" ] || die "copied DISCLOSURE.json does not match $D_SHA"

# 5. WSP-A output read-only; WSP-B output created
chmod -R a-w "$AUTHOR_DIR/output/wsp-a"
[ -z "$(find "$AUTHOR_DIR/output/wsp-a" -perm -u+w -print -quit)" ] || die "$AUTHOR_DIR/output/wsp-a is still writable"
OUT_B="$AUTHOR_DIR/output/wsp-b"
if [ -e "$OUT_B" ]; then
  [ -z "$(find "$OUT_B" -mindepth 1 ! -path "$OUT_B/scratch" -print -quit)" ] || die "$OUT_B exists and is not empty"
fi
chmod u+w "$AUTHOR_DIR/output"
mkdir -p "$OUT_B/scratch"
chmod 555 "$AUTHOR_DIR/output"
chmod 555 "$AUTHOR_DIR"
for p in "$AUTHOR_DIR" "$AUTHOR_DIR/output" "$AUTHOR_DIR/packet" "$AUTHOR_DIR/py" "$AUTHOR_DIR/fwd" "$DST"; do
  [ -w "$p" ] && die "$p is writable"
done
[ -w "$OUT_B" ] && [ -w "$OUT_B/scratch" ] || die "$OUT_B is not writable"

# 6. tools (the same files as WSP-A: verified, never reinstalled), wrappers, settings and hook self-test
(cd "$QWEN" && uv run python -m hypaqwen.wsp.install verify "$AUTHOR_DIR/tools" --hook "$KIT/hook_bash.py") \
  || die "tools/ in $AUTHOR_DIR does not match the harness copy"
cmp -s "$KIT/py" "$AUTHOR_DIR/py" && cmp -s "$KIT/fwd" "$AUTHOR_DIR/fwd" || die "./py or ./fwd differs from the kit copy"
TMP_DIR="$(mktemp -d)"
TMP_SETTINGS="$TMP_DIR/settings_wsp_b.json"
uv run python -m agent_loop_lab.v04.kit render-settings --phase wsp-b --author-dir "$AUTHOR_DIR" --out "$TMP_SETTINGS" \
  || { rm -rf "$TMP_DIR"; die "the WSP-B settings could not be rendered"; }
uv run python -m agent_loop_lab.v04.kit hook-selftest --phase wsp-b --author-dir "$AUTHOR_DIR" --settings "$TMP_SETTINGS" \
  || { rm -rf "$TMP_DIR"; die "the WSP-B hook self-test failed"; }
rm -rf "$TMP_DIR"

# 7. handover record (hashes only; committed before the launch)
mkdir -p "$(dirname "$HANDOVER")"
uv run python - "$AUTHOR_DIR" "$HANDOVER" "$KIT_MODE" "$UNFROZEN" "$INPUTS" "$JOURNAL_JSON" <<'EOF'
import hashlib, json, sys
from pathlib import Path
d, out, kit_mode, unfrozen = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] == "1", sys.argv[4] == "1"
inputs, journal = json.loads(sys.argv[5]), json.loads(sys.argv[6])
kit = Path("experiments/v0_4/author_kit")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
reg = Path("experiments/registry_v0_4.yaml")
rec = {
    "author_dir": str(d),
    "phase": "wsp-b",
    "unfrozen_kit_test_dir": kit_mode,
    "unfrozen": unfrozen,
    "packet_sha256": sha(d / "packet" / "PACKET_MANIFEST.json"),
    "registry_sha256": sha(reg) if reg.is_file() else None,
    "wsp_a_build_record_sha256": inputs["A_SHA256"],
    "wsp_a_edit_state_sha256": inputs["A_EDIT_HASH"],
    "wsp_a_freeze_sha256": inputs["wsp_a"]["freeze_sha256"],
    "wsp_a_freeze_json_sha256": inputs["wsp_a"]["freeze_json_sha256"],
    "registry_freeze_sha256": inputs["freeze_sha256"],
    "disclosure_dir": "disclosure",
    "disclosure_sha256": inputs["DISCLOSURE_SHA256"],
    "disclosure_sha256_file_sha256": sha(d / "disclosure" / "DISCLOSURE.sha256"),
    "disclosed_windows_sha256": inputs["disclosed_windows_sha256"],
    "output_dir": "output/wsp-b",
    "journal": journal,
    "wrappers": {"py": sha(d / "py"), "fwd": sha(d / "fwd")},
    "tools": {p.relative_to(d).as_posix(): sha(p) for p in sorted((d / "tools").rglob("*")) if p.is_file()},
    "kit": {n: sha(kit / n) for n in ("prompt_wsp_b.md", "settings_wsp_b.json", "hook_bash.py", "prompt_continue.md",
                                      "prompt_resubmit.md")},
}
out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: rec[k] for k in ("author_dir", "wsp_a_build_record_sha256", "disclosure_sha256",
                                      "unfrozen_kit_test_dir")}, indent=2))
print(f"handover record: {out}")
EOF
echo "WSP-B handover ready: $DST (read-only), $OUT_B; launch with scripts/v04_launch.sh wsp-b"
