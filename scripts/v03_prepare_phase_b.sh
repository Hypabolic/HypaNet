#!/bin/sh
# Hand the v0.3 Phase B disclosure to the weight author's existing directory (spec v0.3 §9.3,
# experiments/v0_3/phase_b/README.md §2 Records and Handover).
#
#   scripts/v03_prepare_phase_b.sh [AUTHOR_DIR]
#
# Run it only after the Phase A session has ended, Phase A has been archived in weight_program/v0_3/phase-a/
# and scripts/v03_build_disclosure.py has written experiments/v0_3/disclosure/ (committed). AUTHOR_DIR
# defaults to ../HypaNet-v03-author (prepared by the frozen scripts/v03_prepare_author_dir.sh; never inside
# the repo). Steps:
#   1. verify experiments/v0_3/disclosure/DISCLOSURE.json against DISCLOSURE.sha256, the archived Phase A
#      artifact.dlw against its artifact.sha256, and sha256(AUTHOR_DIR/packet/PACKET_MANIFEST.json) against
#      the registry packet_sha256 (the same packet check as v03_prepare_author_dir.sh);
#   2. copy DISCLOSURE.json and DISCLOSURE.sha256 (nothing else) into AUTHOR_DIR/disclosure/ (an existing
#      copy must be byte-identical, otherwise refuse), files chmod 444, directory chmod 555;
#   3. make output/v03-phase-a, packet and tools read-only (chmod -R a-w): ./py scripts are not bound by the
#      settings deny rules;
#   4. create AUTHOR_DIR/output/v03-phase-b/ (refuse if it exists and is not empty);
#   5. write experiments/v0_3/phase_b/HANDOVER.json with the hashes handed over, for the commit
#      (experiments/v0_3/HANDOVER.json is the Phase A record and is not touched).
# These are the inputs `scripts/v03_launch.sh phase-b` checks (check_phase_b_inputs). The launcher also needs
# experiments/v0_3/author_kit/prompt_v03_phase_b.md, which this script does not write. Nothing is launched.
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
AUTHOR_DIR="${1:-$(dirname "$REPO")/HypaNet-v03-author}"
SRC="$REPO/experiments/v0_3/disclosure"
ARCH="$REPO/weight_program/v0_3/phase-a"
KIT="$REPO/experiments/v0_3/author_kit"
DST="$AUTHOR_DIR/disclosure"
OUT_B="$AUTHOR_DIR/output/v03-phase-b"
RECORD="$REPO/experiments/v0_3/phase_b/HANDOVER.json"
die() { echo "refusing: $*" >&2; exit 2; }
sha_of() { shasum -a 256 "$1" | cut -d' ' -f1; }
# First field of a sha256 file ("HEX  name" or bare "HEX"), which must be 64 lowercase hex digits.
sha_field() {
  [ -s "$1" ] || die "$1 missing or empty"
  _h="$(head -n 1 "$1" | cut -d' ' -f1)"
  case "$_h" in *[!0-9a-f]*|"") die "$1 does not hold a sha256";; esac
  [ "${#_h}" -eq 64 ] || die "$1 does not hold a sha256"
  echo "$_h"
}

case "$AUTHOR_DIR" in "$REPO"|"$REPO"/*) die "author dir must be outside the repo";; esac
[ -d "$AUTHOR_DIR/packet" ] && [ -x "$AUTHOR_DIR/py" ] || die "$AUTHOR_DIR is not a prepared v0.3 author dir"
[ -d "$AUTHOR_DIR/output/v03-phase-a" ] || die "$AUTHOR_DIR/output/v03-phase-a missing"

# 1. Verify everything that is handed over or referenced.
[ -f "$SRC/DISCLOSURE.json" ] || die "$SRC/DISCLOSURE.json missing (run scripts/v03_build_disclosure.py and commit it)"
D_SHA="$(sha_field "$SRC/DISCLOSURE.sha256")" || exit 2
[ "$(sha_of "$SRC/DISCLOSURE.json")" = "$D_SHA" ] || die "$SRC/DISCLOSURE.json does not match DISCLOSURE.sha256"
A_SHA="$(sha_field "$ARCH/artifact.sha256")" || exit 2
[ -f "$ARCH/artifact.dlw" ] || die "archived Phase A artifact.dlw missing"
[ "$(sha_of "$ARCH/artifact.dlw")" = "$A_SHA" ] || die "archived Phase A artifact.dlw does not match artifact.sha256"
P_SHA="$(sed -n 's/^packet_sha256:[[:space:]]*\([0-9a-f]\{64\}\)[[:space:]]*$/\1/p' "$REPO/experiments/registry_v0_3.yaml")"
[ "${#P_SHA}" -eq 64 ] || die "no packet_sha256 in experiments/registry_v0_3.yaml"
[ -f "$AUTHOR_DIR/packet/PACKET_MANIFEST.json" ] || die "$AUTHOR_DIR/packet/PACKET_MANIFEST.json missing"
[ "$(sha_of "$AUTHOR_DIR/packet/PACKET_MANIFEST.json")" = "$P_SHA" ] \
  || die "author-dir packet PACKET_MANIFEST.json does not match the registry packet_sha256 $P_SHA"

# 2. Disclosure copy: idempotent if byte-identical, refused otherwise.
if [ -e "$DST" ]; then
  [ -d "$DST" ] || die "$DST exists and is not a directory"
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
cmp -s "$SRC/DISCLOSURE.sha256" "$DST/DISCLOSURE.sha256" || die "copied DISCLOSURE.sha256 differs from $SRC"

# 3. Phase A material, tools and packet: read-only at the OS level as well.
for d in output/v03-phase-a tools packet; do
  if [ -e "$AUTHOR_DIR/$d" ]; then
    chmod -R a-w "$AUTHOR_DIR/$d"
    [ -z "$(find "$AUTHOR_DIR/$d" -perm -u+w -print -quit)" ] || die "$AUTHOR_DIR/$d is still writable"
  fi
done

# 4. Phase B output directory.
if [ -e "$OUT_B" ] && [ -n "$(ls -A "$OUT_B" 2>/dev/null)" ]; then
  die "$OUT_B exists and is not empty"
fi
mkdir -p "$OUT_B"

# 5. Handover record (hashes only; committed before the launch).
file_sha() { if [ -f "$1" ]; then sha_of "$1"; else echo "missing"; fi; }
mkdir -p "$(dirname "$RECORD")"
cat > "$RECORD" <<EOF
{
  "author_dir": "$AUTHOR_DIR",
  "disclosure_dir": "disclosure",
  "disclosure_sha256": "$D_SHA",
  "disclosure_sha256_file_sha256": "$(sha_of "$SRC/DISCLOSURE.sha256")",
  "output_dir": "output/v03-phase-b",
  "packet_sha256": "$P_SHA",
  "phase_a_artifact_sha256": "$A_SHA",
  "prompt_v03_phase_b_sha256": "$(file_sha "$KIT/prompt_v03_phase_b.md")",
  "settings_v03_phase_b_sha256": "$(file_sha "$KIT/settings_v03_phase_b.json")"
}
EOF
cat "$RECORD"
echo "phase B handover ready: $DST (read-only), $OUT_B"
