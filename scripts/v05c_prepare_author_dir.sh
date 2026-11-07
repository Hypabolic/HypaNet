#!/bin/sh
# Prepare the sealed v0.5 Phase C author directory (a NEW session that improves Phase B's network).
#
#   scripts/v05c_prepare_author_dir.sh [AUTHOR_DIR]
#
# AUTHOR_DIR defaults to ../HypaNet-v05c-author; it must be outside the repo and must not exist yet. Run it only
# after the Phase C disclosures exist (experiments/v0_5/phase_c/disclosure/DISCLOSURE.json + DISCLOSURE.sha256 and
# DISCLOSURE_W.json + DISCLOSURE_W.sha256, committed). Steps:
#   1. refuse unless the only sibling HypaNet* directories are the ones the settings deny-lists name (the repo,
#      HypaNet-m2-author, HypaNet-v03-author, HypaNet-v04-author, HypaNet-v05b-author, HypaNet-v05-*) and the new
#      author dir;
#   2. build the packet into AUTHOR_DIR/packet with the frozen scripts/v03_build_packet.py and require
#      sha256(PACKET_MANIFEST.json) = packet_sha256 of experiments/registry_v0_3.yaml (as Phase B);
#   3. tools/: the v0.3 AUTHOR_RUN guard, experiments/v0_5/phase_c/author_kit/launch.py (Phase B's launcher, writes
#      only in output/v05-phase-c/) and tools/lib/ (the Phase C v05_load.py and hn_v05/: vendored copies of the
#      frozen v0.5 edits reader, Phase B's revision applier, revision_c.py and the DLW3 reader;
#      agent_loop_lab.v05.kit_c); ./py is the v0.3 wrapper, byte for byte;
#   4. r1/: exactly as Phase B (experiments/v0_5/PROMPT.txt checked against PROMPT.sha256, r1's replies and
#      INDEX.json, artifact.dlw rebuilt byte for byte from the replies, artifact.sha256, R1_MANIFEST.json);
#   5. phase_b/: Phase B's artefact (weight_program/v0_5/phase-b/artifact.dlw + artifact.sha256, which must be
#      r1's weights + Phase B's REVISION.edits, rebuilt by the revision applier) and Phase B's own output as its
#      build archived it (author_output/: REVISION.edits, CIRCUITS.md, REVISION.md, ITERATIONS.md, scratch/; checked
#      against BUILD_RECORD.json's output manifest), plus PHASE_B_MANIFEST.json. Never apply.json, BUILD_RECORD.json,
#      attempts/ or any score of Phase B's network;
#   6. disclosure/: both disclosures, screened (exactly id, prefix, oracle_label, network_class per window; ids
#      V001..V200 and W001..W200; 50 per class; no family name or master seed in a header; no second-generator
#      family name in a prefix), every network_class recomputed from Phase B's artefact, the V windows identical
#      to Phase B's disclosure (id, prefix, oracle_label) and no prefix shared between the two;
#   7. the private venv (Python 3.12, torch and numpy at this repo's versions, as v0.3);
#   8. packet/, tools/, r1/, phase_b/, disclosure/ and py read-only, and the private .venv, output/ and the author
#      dir itself; output/v05-phase-c/ empty and the only writable directory;
#   9. experiments/v0_5/phase_c/HANDOVER.json: every hash handed over, the kit files' hashes, the .venv inventory,
#      the author dir's entries, the checks. In real mode the script refuses if HANDOVER.json, SESSION_ID or a
#      Phase C transcript already exists.
# Nothing is launched. Nothing from experiments/v0_5/ood_analysis/, artifacts/reports/, experiments/v0_5/STATUS.md,
# the sealed evaluation or the held-out sets is read or copied.
#
# Kit / dry runs (tests): V05C_KIT=1 with V05C_R1_DIR (a run dir: replies/, artifact.dlw, artifact.sha256),
# V05C_B_DIR (a Phase B archive: artifact.dlw, artifact.sha256, BUILD_RECORD.json, author_output/),
# V05C_DISCLOSURE_DIR (both disclosures), V05C_B_DISCLOSURE (Phase B's DISCLOSURE.json, for the same-windows check),
# V05C_RECORD (where HANDOVER.json goes; not under experiments/) and optionally V05C_NO_REBUILD_CHECK=1 (the
# stand-in r1 artefact is not the replies' rebuild), V05C_DISCLOSURE_EXPECT=N and V05C_VENV_FROM=DIR (symlink
# AUTHOR_DIR/.venv to an existing venv instead of creating one). In kit mode r1 need not have a final run.json and
# Phase B's artefact need not be 223d0743....
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd -P)"
PARENT="$(dirname "$REPO")"
AUTHOR_DIR="${1:-$PARENT/HypaNet-v05c-author}"
KIT="$REPO/experiments/v0_5/phase_c/author_kit"
die() { echo "refusing: $*" >&2; exit 2; }
sha_of() { shasum -a 256 "$1" | cut -d' ' -f1; }

# v0.5 Phase B's network, the base of Phase C (STATUS.md: artifact 223d0743...)
B_SHA_EXPECTED="223d074372bf726a0c89f28afe1e1fcf9b53e3bb0ff3973f0cc3acd932661001"
KIT_MODE="${V05C_KIT:-0}"
R1_DIR="$REPO/experiments/v0_5/runs/r1"
B_DIR="$REPO/weight_program/v0_5/phase-b"
DISC_DIR="$REPO/experiments/v0_5/phase_c/disclosure"
B_DISC="$REPO/experiments/v0_5/phase_b/disclosure/DISCLOSURE.json"
RECORD="$REPO/experiments/v0_5/phase_c/HANDOVER.json"
EXPECT=200
if [ "$KIT_MODE" = 1 ]; then
  R1_DIR="${V05C_R1_DIR:?kit mode needs V05C_R1_DIR}"
  B_DIR="${V05C_B_DIR:?kit mode needs V05C_B_DIR}"
  DISC_DIR="${V05C_DISCLOSURE_DIR:?kit mode needs V05C_DISCLOSURE_DIR}"
  B_DISC="${V05C_B_DISCLOSURE:?kit mode needs V05C_B_DISCLOSURE}"
  RECORD="${V05C_RECORD:?kit mode needs V05C_RECORD}"
  EXPECT="${V05C_DISCLOSURE_EXPECT:-200}"
  B_SHA_EXPECTED=""
  case "$RECORD" in "$REPO"/experiments/*) die "kit mode: V05C_RECORD may not be under experiments/";; esac
else
  for v in V05C_R1_DIR V05C_B_DIR V05C_DISCLOSURE_DIR V05C_B_DISCLOSURE V05C_RECORD V05C_NO_REBUILD_CHECK \
           V05C_DISCLOSURE_EXPECT V05C_VENV_FROM; do
    eval "[ -z \"\${$v:-}\" ]" || die "$v is set but V05C_KIT is not 1"
  done
  [ -e "$RECORD" ] && die "$RECORD exists (a Phase C hand-over was already recorded; move it aside deliberately first)"
  [ -e "$REPO/experiments/v0_5/phase_c/SESSION_ID" ] && die "experiments/v0_5/phase_c/SESSION_ID exists (a Phase C session was launched)"
  for t in "$REPO"/experiments/v0_5/phase_c/logs/v05c*.stream.jsonl; do
    [ -e "$t" ] && die "$t exists (a Phase C session was launched)"
  done
fi

mkdir -p "$(dirname "$AUTHOR_DIR")"
AUTHOR_DIR="$(cd "$(dirname "$AUTHOR_DIR")" && pwd -P)/$(basename "$AUTHOR_DIR")"
case "$AUTHOR_DIR" in "$REPO"|"$REPO"/*) die "author dir must be outside the repo";; esac
[ -e "$AUTHOR_DIR" ] && die "$AUTHOR_DIR already exists (the Phase C author directory must be fresh)"
for d in "$(dirname "$AUTHOR_DIR")"/HypaNet*; do
  [ -e "$d" ] || continue
  case "$(basename "$d")" in
    HypaNet|HypaNet-m2-author|HypaNet-v03-author|HypaNet-v04-author|HypaNet-v05b-author|HypaNet-v05-*|"$(basename "$AUTHOR_DIR")") ;;
    *) die "unexpected sibling directory $d (not covered by the settings deny-lists; extend them first)";;
  esac
done
[ -f "$KIT/prompt_v05_phase_c.md" ] && [ -f "$KIT/settings_v05_phase_c.json" ] || die "author kit incomplete in $KIT"

cd "$REPO"
P_SHA="$(sed -n 's/^packet_sha256:[[:space:]]*\([0-9a-f]\{64\}\)[[:space:]]*$/\1/p' experiments/registry_v0_3.yaml)"
[ "${#P_SHA}" -eq 64 ] || die "no packet_sha256 in experiments/registry_v0_3.yaml"
WANT="$(head -n 1 experiments/v0_5/PROMPT.sha256 | cut -d' ' -f1)"
[ "$(sha_of experiments/v0_5/PROMPT.txt)" = "$WANT" ] || die "experiments/v0_5/PROMPT.txt does not match PROMPT.sha256"
for pair in DISCLOSURE DISCLOSURE_W; do
  [ -f "$DISC_DIR/$pair.json" ] || die "$DISC_DIR/$pair.json missing"
  [ -s "$DISC_DIR/$pair.sha256" ] || die "$DISC_DIR/$pair.sha256 missing"
  [ "$(sha_of "$DISC_DIR/$pair.json")" = "$(head -n 1 "$DISC_DIR/$pair.sha256" | cut -d' ' -f1)" ] \
    || die "$DISC_DIR/$pair.json does not match $pair.sha256"
done
# exactly the four disclosure files: nothing else in the disclosure dir (a private record must stay in the repo)
[ "$(ls -A "$DISC_DIR" | LC_ALL=C sort | tr '\n' ' ')" = "DISCLOSURE.json DISCLOSURE.sha256 DISCLOSURE_W.json DISCLOSURE_W.sha256 " ] \
  || die "$DISC_DIR holds other files than the two disclosures and their sha256 files"
D_SHA="$(sha_of "$DISC_DIR/DISCLOSURE.json")"
W_SHA="$(sha_of "$DISC_DIR/DISCLOSURE_W.json")"
[ -f "$R1_DIR/artifact.dlw" ] || die "$R1_DIR/artifact.dlw missing"
[ -f "$B_DIR/artifact.dlw" ] || die "$B_DIR/artifact.dlw missing"
[ -f "$B_DISC" ] || die "$B_DISC (Phase B's disclosure) missing"
if [ -n "$B_SHA_EXPECTED" ]; then
  [ "$(sha_of "$B_DIR/artifact.dlw")" = "$B_SHA_EXPECTED" ] || die "$B_DIR/artifact.dlw is not Phase B's artefact $B_SHA_EXPECTED"
fi
# screen both disclosures and recompute every network_class from Phase B's artefact before anything is written
uv run python -m agent_loop_lab.v05.kit_c check-disclosure "$DISC_DIR/DISCLOSURE.json" --ids V --expect "$EXPECT" \
  --artifact "$B_DIR/artifact.dlw" > /dev/null || die "DISCLOSURE.json check failed"
uv run python -m agent_loop_lab.v05.kit_c check-disclosure "$DISC_DIR/DISCLOSURE_W.json" --ids W --expect "$EXPECT" \
  --artifact "$B_DIR/artifact.dlw" > /dev/null || die "DISCLOSURE_W.json check failed"
uv run python -m agent_loop_lab.v05.kit_c check-same-windows "$DISC_DIR/DISCLOSURE.json" "$B_DISC" > /dev/null \
  || die "DISCLOSURE.json does not hold Phase B's windows"
uv run python -m agent_loop_lab.v05.kit_c check-disjoint "$DISC_DIR/DISCLOSURE.json" "$DISC_DIR/DISCLOSURE_W.json" \
  > /dev/null || die "the two disclosures overlap"

mkdir -p "$AUTHOR_DIR"
TMPD="$(mktemp -d "$(dirname "$AUTHOR_DIR")/.v05c-handover-tmp.XXXXXX")"   # outside the author dir; removed at exit
trap 'rm -rf "$TMPD"' EXIT
# 2. packet
uv run python scripts/v03_build_packet.py --out "$AUTHOR_DIR/packet" > /dev/null
uv run python scripts/v03_build_packet.py --out "$AUTHOR_DIR/packet" --verify-only > /dev/null || die "packet verify failed"
GOT="$(sha_of "$AUTHOR_DIR/packet/PACKET_MANIFEST.json")"
[ "$GOT" = "$P_SHA" ] || die "built packet $GOT != registry $P_SHA"
# 3. tools and ./py
uv run python -m agent_loop_lab.v05.kit_c install-tools "$AUTHOR_DIR/tools" > /dev/null
cp experiments/v0_3/author_kit/py "$AUTHOR_DIR/py" && chmod 755 "$AUTHOR_DIR/py"
# 4. r1
STAGE_FLAGS=""
[ "$KIT_MODE" = 1 ] && STAGE_FLAGS="--no-final-check"
[ "${V05C_NO_REBUILD_CHECK:-0}" = 1 ] && STAGE_FLAGS="$STAGE_FLAGS --no-rebuild-check"
# shellcheck disable=SC2086
uv run python -m agent_loop_lab.v05.kit_c stage-r1 "$R1_DIR" experiments/v0_5/PROMPT.txt "$AUTHOR_DIR/r1" $STAGE_FLAGS \
  > "$TMPD/stage-r1.json" || die "r1 hand-over failed"
# 5. phase_b
B_FLAGS=""
[ -n "$B_SHA_EXPECTED" ] && B_FLAGS="--expect-sha $B_SHA_EXPECTED"
# shellcheck disable=SC2086
uv run python -m agent_loop_lab.v05.kit_c stage-b "$B_DIR" "$R1_DIR/artifact.dlw" "$AUTHOR_DIR/phase_b" $B_FLAGS \
  > "$TMPD/stage-b.json" || die "Phase B hand-over failed"
# 6. disclosures
mkdir "$AUTHOR_DIR/disclosure"
cp "$DISC_DIR/DISCLOSURE.json" "$DISC_DIR/DISCLOSURE.sha256" "$DISC_DIR/DISCLOSURE_W.json" \
  "$DISC_DIR/DISCLOSURE_W.sha256" "$AUTHOR_DIR/disclosure/"
[ "$(sha_of "$AUTHOR_DIR/disclosure/DISCLOSURE.json")" = "$D_SHA" ] || die "copied DISCLOSURE.json differs"
[ "$(sha_of "$AUTHOR_DIR/disclosure/DISCLOSURE_W.json")" = "$W_SHA" ] || die "copied DISCLOSURE_W.json differs"
# 7. venv
if [ -n "${V05C_VENV_FROM:-}" ]; then
  [ -x "$V05C_VENV_FROM/bin/python" ] || die "V05C_VENV_FROM has no bin/python"
  ln -s "$V05C_VENV_FROM" "$AUTHOR_DIR/.venv"
  TORCH_V="$("$AUTHOR_DIR/.venv/bin/python" -I -B -c 'import torch; print(torch.__version__)')"
  NUMPY_V="$("$AUTHOR_DIR/.venv/bin/python" -I -B -c 'import numpy; print(numpy.__version__)')"
  VENV_KIND="kit symlink to $V05C_VENV_FROM"
else
  TORCH_V="$(uv run python -c 'import torch; print(torch.__version__)')"
  NUMPY_V="$(uv run python -c 'import numpy; print(numpy.__version__)')"
  uv venv -q --python 3.12 "$AUTHOR_DIR/.venv"
  uv pip install -q --python "$AUTHOR_DIR/.venv/bin/python" "torch==$TORCH_V" "numpy==$NUMPY_V"
  VENV_KIND="private"
fi
# 8. permissions
mkdir -p "$AUTHOR_DIR/output/v05-phase-c"
chmod -R a-w "$AUTHOR_DIR/packet" "$AUTHOR_DIR/tools" "$AUTHOR_DIR/r1" "$AUTHOR_DIR/phase_b" "$AUTHOR_DIR/disclosure"
chmod 555 "$AUTHOR_DIR/py"
[ -L "$AUTHOR_DIR/.venv" ] || chmod -R a-w "$AUTHOR_DIR/.venv"
chmod 555 "$AUTHOR_DIR/output" "$AUTHOR_DIR"
for d in packet tools r1 phase_b disclosure; do
  [ -z "$(find "$AUTHOR_DIR/$d" -perm -u+w -print -quit)" ] || die "$AUTHOR_DIR/$d is still writable"
done
if [ ! -L "$AUTHOR_DIR/.venv" ]; then
  [ -z "$(find "$AUTHOR_DIR/.venv" -perm -u+w -not -type l -print -quit)" ] || die "$AUTHOR_DIR/.venv is still writable"
fi
# 9. handover record
EXTRA="$TMPD/handover-extra.json"
uv run python - "$TMPD/stage-r1.json" "$TMPD/stage-b.json" "$EXTRA" "$TORCH_V" "$NUMPY_V" "$VENV_KIND" "$R1_DIR" \
  "$B_DIR" "$DISC_DIR" "$B_DISC" "$REPO" "$WANT" <<'EOF'
import json, os, sys
stage, stage_b, extra, torch_v, numpy_v, venv_kind, r1_dir, b_dir, disc_dir, b_disc, repo, prompt_sha = sys.argv[1:13]
s = json.load(open(stage))
b = json.load(open(stage_b))
rel = lambda p: os.path.relpath(p, repo) if os.path.realpath(p).startswith(repo + os.sep) else p
rec = {
    "venv": {"torch": torch_v, "numpy": numpy_v, "kind": venv_kind},
    "r1_source": rel(r1_dir),
    "r1_checks": {k: s[k] for k in ("artifact_sha256", "reading", "reading_units", "status", "rebuild_identical")},
    "b_source": rel(b_dir),
    "b_checks": {k: b[k] for k in ("artifact_sha256", "base_r1_sha256", "revision_sha256", "rebuild_identical",
                                   "author_output_files")},
    "disclosure_source": rel(disc_dir),
    "disclosure_network_class_checked": "both disclosures, against Phase B's artefact",
    "disclosure_same_windows_as": rel(b_disc),
    "prompt_txt_sha256": prompt_sha,
    "reference_policy_class": "absent: no session in the v0.5 lineage wrote a reference policy",
}
json.dump(rec, open(extra, "w"), indent=1, sort_keys=True)
EOF
KITFLAG=""
[ "$KIT_MODE" = 1 ] && KITFLAG="--kit"
# shellcheck disable=SC2086
uv run python -m agent_loop_lab.v05.kit_c record "$AUTHOR_DIR" "$RECORD" $KITFLAG --extra "$EXTRA" > /dev/null
[ -z "$(ls -A "$AUTHOR_DIR/output/v05-phase-c")" ] || die "output/v05-phase-c is not empty"
uv run python -m agent_loop_lab.v05.kit_c verify "$AUTHOR_DIR" "$RECORD" --r1-dir "$R1_DIR" --b-dir "$B_DIR" \
  --disclosure-dir "$DISC_DIR" > /dev/null || die "the fresh author dir does not verify against $RECORD"
"$AUTHOR_DIR/.venv/bin/python" -I -B -c "import torch, numpy; print('author venv:', torch.__version__, numpy.__version__)"
echo "handover record: $RECORD"
echo "author dir ready: $AUTHOR_DIR"
