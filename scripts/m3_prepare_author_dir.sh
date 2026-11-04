#!/bin/sh
# Hand the Phase B disclosure to the weight author's existing working directory (spec v0.2 §9.4,
# experiments/m3/README.md §2-§4).
#
#   scripts/m3_prepare_author_dir.sh [AUTHOR_DIR]
#
# AUTHOR_DIR defaults to ../HypaNet-m2-author (prepared by scripts/m2_prepare_author_dir.sh; never inside
# the repo). Steps:
#   1. verify experiments/m3/disclosure/DISCLOSURE.json against DISCLOSURE.sha256, the archived
#      A-selfcheck artefact against its artifact.sha256, and the author-dir packet against the registry;
#   2. copy DISCLOSURE.json and DISCLOSURE.sha256 into AUTHOR_DIR/disclosure/ (an existing copy must be
#      byte-identical, otherwise refuse), files chmod 444, directory chmod 555; then make
#      output/phase-a-strict, output/phase-a-selfcheck, output/selfcheck_scripts, tools and packet
#      read-only (chmod -R a-w), since ./py scripts are not bound by the settings deny rules;
#   3. create AUTHOR_DIR/output/phase-b-revision/ (refuse if it exists and is not empty);
#   4. write experiments/m3/HANDOVER.json with the hashes handed over, for the commit.
# Nothing is launched.
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
AUTHOR_DIR="${1:-$(dirname "$REPO")/HypaNet-m2-author}"
SRC="$REPO/experiments/m3/disclosure"
DST="$AUTHOR_DIR/disclosure"
REV="$AUTHOR_DIR/output/phase-b-revision"
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
[ -d "$AUTHOR_DIR/packet" ] && [ -d "$AUTHOR_DIR/output/phase-a-selfcheck" ] \
  || die "$AUTHOR_DIR is not a prepared M2 author dir (packet/ or output/phase-a-selfcheck/ missing)"

# 1. Verify everything that is handed over or referenced.
[ -f "$SRC/DISCLOSURE.json" ] || die "$SRC/DISCLOSURE.json missing (run scripts/m3_build_disclosure.py and commit it)"
D_SHA="$(sha_field "$SRC/DISCLOSURE.sha256")" || exit 2
[ "$(sha_of "$SRC/DISCLOSURE.json")" = "$D_SHA" ] || die "$SRC/DISCLOSURE.json does not match DISCLOSURE.sha256"
A_SHA="$(sha_field "$REPO/weight_program/phase-a-selfcheck/artifact.sha256")" || exit 2
[ -f "$REPO/weight_program/phase-a-selfcheck/artifact.dlw" ] || die "archived A-selfcheck artifact.dlw missing"
[ "$(sha_of "$REPO/weight_program/phase-a-selfcheck/artifact.dlw")" = "$A_SHA" ] || die "A-selfcheck artifact.dlw does not match artifact.sha256"
P_SHA="$(sed -n 's/^packet_sha256:[[:space:]]*\([0-9a-f]\{64\}\)[[:space:]]*$/\1/p' "$REPO/experiments/registry.yaml")"
[ "${#P_SHA}" -eq 64 ] || die "no packet_sha256 in experiments/registry.yaml"
[ -f "$AUTHOR_DIR/packet/MANIFEST.json" ] || die "$AUTHOR_DIR/packet/MANIFEST.json missing"
grep -q "\"packet_sha256\": *\"$P_SHA\"" "$AUTHOR_DIR/packet/MANIFEST.json" \
  || die "author-dir packet MANIFEST.json does not carry the registry packet_sha256 $P_SHA"

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

# 2b. Archived Phase A material, tools and packet: read-only at the OS level as well (README §4).
#     The settings deny rules only bind Claude Code's own tools; a ./py script runs outside them.
for d in output/phase-a-strict output/phase-a-selfcheck output/selfcheck_scripts tools packet; do
  if [ -e "$AUTHOR_DIR/$d" ]; then
    chmod -R a-w "$AUTHOR_DIR/$d"
    [ -z "$(find "$AUTHOR_DIR/$d" -perm -u+w -print -quit)" ] || die "$AUTHOR_DIR/$d is still writable"
  fi
done

# 3. Phase B output directory.
if [ -e "$REV" ] && [ -n "$(ls -A "$REV" 2>/dev/null)" ]; then
  die "$REV exists and is not empty"
fi
mkdir -p "$REV"

# 4. Handover record (hashes only; committed before the launch).
file_sha() { if [ -f "$1" ]; then sha_of "$1"; else echo "missing"; fi; }
cat > "$REPO/experiments/m3/HANDOVER.json" <<EOF
{
  "a_selfcheck_artifact_sha256": "$A_SHA",
  "author_dir": "$AUTHOR_DIR",
  "disclosure_sha256": "$D_SHA",
  "disclosure_sha256_file_sha256": "$(sha_of "$SRC/DISCLOSURE.sha256")",
  "output_dir": "output/phase-b-revision",
  "packet_sha256": "$P_SHA",
  "prompt_phase_b_sha256": "$(file_sha "$REPO/experiments/m3/prompt_phase_b.md")",
  "settings_phase_b_sha256": "$(file_sha "$REPO/experiments/m3/settings_phase_b.json")"
}
EOF
cat "$REPO/experiments/m3/HANDOVER.json"
echo "phase B handover ready: $DST (read-only), $REV"
