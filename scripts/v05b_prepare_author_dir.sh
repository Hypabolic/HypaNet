#!/bin/sh
# Prepare the sealed v0.5 Phase B author directory (a NEW session that improves the network r1 wrote).
#
#   scripts/v05b_prepare_author_dir.sh [AUTHOR_DIR]
#
# AUTHOR_DIR defaults to ../HypaNet-v05b-author; it must be outside the repo and must not exist yet. Run it only
# after r1 has finished (experiments/v0_5/runs/r1/run.json has a final status and scripts/v05_apply_reply.py has
# written artifact.dlw / artifact.sha256) and the Phase B disclosure exists (experiments/v0_5/phase_b/disclosure/
# DISCLOSURE.json + DISCLOSURE.sha256, committed). Steps:
#   1. refuse unless the only sibling HypaNet* directories are the ones the settings deny-lists name (the repo,
#      HypaNet-m2-author, HypaNet-v03-author, HypaNet-v04-author) and the new author dir;
#   2. build the packet into AUTHOR_DIR/packet with the frozen scripts/v03_build_packet.py, verify every file
#      against its manifest and require sha256(PACKET_MANIFEST.json) = packet_sha256 of experiments/registry_v0_3.yaml
#      (exactly as scripts/v03_prepare_author_dir.sh; scripts/v03_freeze.py --check is not run: it hashes
#      dataset files this kit must not open);
#   3. tools/: the v0.3 AUTHOR_RUN guard, experiments/v0_5/phase_b/author_kit/launch.py (the v0.3 launcher plus
#      tools/lib on the module path, every read outside the author dir and the interpreter denied, writes only in
#      output/v05-phase-b/, low-level file modules refused) and tools/lib/ (v05_load.py and hn_v05/: vendored copies of the frozen v0.5
#      edits reader, the Phase B revision applier and the DLW3 reader; agent_loop_lab.v05.kit_b); ./py is the
#      v0.3 wrapper, byte for byte;
#   4. r1/: experiments/v0_5/PROMPT.txt (checked against PROMPT.sha256), r1's replies as the r1 harness wrote them
#      (NNN.md, NNN.p<K>.md, INDEX.json; INDEX.json checked against the files and their sha256s), artifact.dlw
#      (checked against artifact.sha256 and rebuilt byte for byte from the replies by the frozen reader) and
#      artifact.sha256, plus R1_MANIFEST.json; nothing else from the run (no run.json, apply.json or logs);
#   5. disclosure/: DISCLOSURE.json + DISCLOSURE.sha256, screened (exactly id, prefix, oracle_label,
#      network_class per window: no reference_policy_class, since r1 wrote no reference policy; V001..V200; 50 per
#      class) and every network_class recomputed from r1's artefact;
#   6. the private venv (Python 3.12, torch and numpy at this repo's versions, as v0.3);
#   7. packet/, tools/, r1/, disclosure/ and py read-only, and (as v0.4) the private .venv, output/ and the author
#      dir itself, so nothing can be added next to them (a .pth in .venv would run before the guard);
#      output/v05-phase-b/ empty and the only writable directory;
#   8. experiments/v0_5/phase_b/HANDOVER.json: every hash handed over, the kit files' hashes, the .venv inventory
#      (and the hashes of its .pth / sitecustomize / usercustomize / pyvenv.cfg), the author dir's entries, the
#      checks. In real mode the script refuses if HANDOVER.json, SESSION_ID or a Phase B transcript already exists:
#      the record later launches verify against is never replaced silently.
# Nothing is launched.
#
# Kit / dry runs (tests): V05B_KIT=1 with V05B_R1_DIR (a run dir: replies/, artifact.dlw, artifact.sha256),
# V05B_DISCLOSURE_DIR, V05B_RECORD (where HANDOVER.json goes; not under experiments/) and optionally
# V05B_NO_REBUILD_CHECK=1 (the stand-in artefact is not the replies' rebuild), V05B_DISCLOSURE_EXPECT=N and
# V05B_VENV_FROM=DIR (symlink AUTHOR_DIR/.venv to an existing venv instead of creating one). In kit mode r1 need
# not have a final run.json.
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd -P)"
PARENT="$(dirname "$REPO")"
AUTHOR_DIR="${1:-$PARENT/HypaNet-v05b-author}"
KIT="$REPO/experiments/v0_5/phase_b/author_kit"
die() { echo "refusing: $*" >&2; exit 2; }
sha_of() { shasum -a 256 "$1" | cut -d' ' -f1; }

KIT_MODE="${V05B_KIT:-0}"
R1_DIR="$REPO/experiments/v0_5/runs/r1"
DISC_DIR="$REPO/experiments/v0_5/phase_b/disclosure"
RECORD="$REPO/experiments/v0_5/phase_b/HANDOVER.json"
EXPECT=200
if [ "$KIT_MODE" = 1 ]; then
  R1_DIR="${V05B_R1_DIR:?kit mode needs V05B_R1_DIR}"
  DISC_DIR="${V05B_DISCLOSURE_DIR:?kit mode needs V05B_DISCLOSURE_DIR}"
  RECORD="${V05B_RECORD:?kit mode needs V05B_RECORD}"
  EXPECT="${V05B_DISCLOSURE_EXPECT:-200}"
  case "$RECORD" in "$REPO"/experiments/*) die "kit mode: V05B_RECORD may not be under experiments/";; esac
else
  for v in V05B_R1_DIR V05B_DISCLOSURE_DIR V05B_RECORD V05B_NO_REBUILD_CHECK V05B_DISCLOSURE_EXPECT V05B_VENV_FROM; do
    eval "[ -z \"\${$v:-}\" ]" || die "$v is set but V05B_KIT is not 1"
  done
  [ -e "$RECORD" ] && die "$RECORD exists (a Phase B hand-over was already recorded; move it aside deliberately first)"
  [ -e "$REPO/experiments/v0_5/phase_b/SESSION_ID" ] && die "experiments/v0_5/phase_b/SESSION_ID exists (a Phase B session was launched)"
  for t in "$REPO"/experiments/v0_5/phase_b/logs/v05b*.stream.jsonl; do
    [ -e "$t" ] && die "$t exists (a Phase B session was launched)"
  done
fi

mkdir -p "$(dirname "$AUTHOR_DIR")"
AUTHOR_DIR="$(cd "$(dirname "$AUTHOR_DIR")" && pwd -P)/$(basename "$AUTHOR_DIR")"
case "$AUTHOR_DIR" in "$REPO"|"$REPO"/*) die "author dir must be outside the repo";; esac
[ -e "$AUTHOR_DIR" ] && die "$AUTHOR_DIR already exists (the Phase B author directory must be fresh)"
for d in "$(dirname "$AUTHOR_DIR")"/HypaNet*; do
  [ -e "$d" ] || continue
  case "$(basename "$d")" in
    HypaNet|HypaNet-m2-author|HypaNet-v03-author|HypaNet-v04-author|"$(basename "$AUTHOR_DIR")") ;;
    *) die "unexpected sibling directory $d (not covered by the settings deny-lists; extend them first)";;
  esac
done
[ -f "$KIT/prompt_v05_phase_b.md" ] && [ -f "$KIT/settings_v05_phase_b.json" ] || die "author kit incomplete in $KIT"

cd "$REPO"
P_SHA="$(sed -n 's/^packet_sha256:[[:space:]]*\([0-9a-f]\{64\}\)[[:space:]]*$/\1/p' experiments/registry_v0_3.yaml)"
[ "${#P_SHA}" -eq 64 ] || die "no packet_sha256 in experiments/registry_v0_3.yaml"
WANT="$(head -n 1 experiments/v0_5/PROMPT.sha256 | cut -d' ' -f1)"
[ "$(sha_of experiments/v0_5/PROMPT.txt)" = "$WANT" ] || die "experiments/v0_5/PROMPT.txt does not match PROMPT.sha256"
[ -f "$DISC_DIR/DISCLOSURE.json" ] || die "$DISC_DIR/DISCLOSURE.json missing"
D_SHA="$(head -n 1 "$DISC_DIR/DISCLOSURE.sha256" | cut -d' ' -f1)"
[ "$(sha_of "$DISC_DIR/DISCLOSURE.json")" = "$D_SHA" ] || die "$DISC_DIR/DISCLOSURE.json does not match DISCLOSURE.sha256"
[ -f "$R1_DIR/artifact.dlw" ] || die "$R1_DIR/artifact.dlw missing"
# screen the disclosure and recompute every network_class from r1's artefact before anything is written
uv run python -m agent_loop_lab.v05.kit_b check-disclosure "$DISC_DIR/DISCLOSURE.json" --expect "$EXPECT" \
  --artifact "$R1_DIR/artifact.dlw" > /dev/null || die "disclosure check failed"

mkdir -p "$AUTHOR_DIR"
TMPD="$(mktemp -d "$(dirname "$AUTHOR_DIR")/.v05b-handover-tmp.XXXXXX")"   # outside the author dir; removed at exit
trap 'rm -rf "$TMPD"' EXIT
# 2. packet
uv run python scripts/v03_build_packet.py --out "$AUTHOR_DIR/packet" > /dev/null
uv run python scripts/v03_build_packet.py --out "$AUTHOR_DIR/packet" --verify-only > /dev/null || die "packet verify failed"
GOT="$(sha_of "$AUTHOR_DIR/packet/PACKET_MANIFEST.json")"
[ "$GOT" = "$P_SHA" ] || die "built packet $GOT != registry $P_SHA"
# 3. tools and ./py
uv run python -m agent_loop_lab.v05.kit_b install-tools "$AUTHOR_DIR/tools" > /dev/null
cp experiments/v0_3/author_kit/py "$AUTHOR_DIR/py" && chmod 755 "$AUTHOR_DIR/py"
# 4. r1
STAGE_FLAGS=""
[ "$KIT_MODE" = 1 ] && STAGE_FLAGS="--no-final-check"
[ "${V05B_NO_REBUILD_CHECK:-0}" = 1 ] && STAGE_FLAGS="$STAGE_FLAGS --no-rebuild-check"
# shellcheck disable=SC2086
uv run python -m agent_loop_lab.v05.kit_b stage-r1 "$R1_DIR" experiments/v0_5/PROMPT.txt "$AUTHOR_DIR/r1" $STAGE_FLAGS \
  > "$TMPD/stage-r1.json" || die "r1 hand-over failed"
# 5. disclosure
mkdir "$AUTHOR_DIR/disclosure"
cp "$DISC_DIR/DISCLOSURE.json" "$DISC_DIR/DISCLOSURE.sha256" "$AUTHOR_DIR/disclosure/"
[ "$(sha_of "$AUTHOR_DIR/disclosure/DISCLOSURE.json")" = "$D_SHA" ] || die "copied DISCLOSURE.json differs"
# 6. venv
if [ -n "${V05B_VENV_FROM:-}" ]; then
  [ -x "$V05B_VENV_FROM/bin/python" ] || die "V05B_VENV_FROM has no bin/python"
  ln -s "$V05B_VENV_FROM" "$AUTHOR_DIR/.venv"
  TORCH_V="$("$AUTHOR_DIR/.venv/bin/python" -I -B -c 'import torch; print(torch.__version__)')"
  NUMPY_V="$("$AUTHOR_DIR/.venv/bin/python" -I -B -c 'import numpy; print(numpy.__version__)')"
  VENV_KIND="kit symlink to $V05B_VENV_FROM"
else
  TORCH_V="$(uv run python -c 'import torch; print(torch.__version__)')"
  NUMPY_V="$(uv run python -c 'import numpy; print(numpy.__version__)')"
  uv venv -q --python 3.12 "$AUTHOR_DIR/.venv"
  uv pip install -q --python "$AUTHOR_DIR/.venv/bin/python" "torch==$TORCH_V" "numpy==$NUMPY_V"
  VENV_KIND="private"
fi
# 7. permissions
mkdir -p "$AUTHOR_DIR/output/v05-phase-b"
chmod -R a-w "$AUTHOR_DIR/packet" "$AUTHOR_DIR/tools" "$AUTHOR_DIR/r1" "$AUTHOR_DIR/disclosure"
chmod 555 "$AUTHOR_DIR/py"
# the private venv too (a kit symlink points at a venv this script does not own: left alone)
[ -L "$AUTHOR_DIR/.venv" ] || chmod -R a-w "$AUTHOR_DIR/.venv"
chmod 555 "$AUTHOR_DIR/output" "$AUTHOR_DIR"
for d in packet tools r1 disclosure; do
  [ -z "$(find "$AUTHOR_DIR/$d" -perm -u+w -print -quit)" ] || die "$AUTHOR_DIR/$d is still writable"
done
if [ ! -L "$AUTHOR_DIR/.venv" ]; then
  [ -z "$(find "$AUTHOR_DIR/.venv" -perm -u+w -not -type l -print -quit)" ] || die "$AUTHOR_DIR/.venv is still writable"
fi
# 8. handover record
EXTRA="$TMPD/handover-extra.json"
uv run python - "$TMPD/stage-r1.json" "$EXTRA" "$TORCH_V" "$NUMPY_V" "$VENV_KIND" "$R1_DIR" "$DISC_DIR" "$REPO" "$WANT" <<'EOF'
import json, os, sys
stage, extra, torch_v, numpy_v, venv_kind, r1_dir, disc_dir, repo, prompt_sha = sys.argv[1:10]
s = json.load(open(stage))
rel = lambda p: os.path.relpath(p, repo) if os.path.realpath(p).startswith(repo + os.sep) else p
rec = {
    "venv": {"torch": torch_v, "numpy": numpy_v, "kind": venv_kind},
    "r1_source": rel(r1_dir),
    "r1_checks": {k: s[k] for k in ("artifact_sha256", "reading", "reading_units", "status", "rebuild_identical")},
    "disclosure_source": rel(disc_dir),
    "disclosure_network_class_checked": True,
    "prompt_txt_sha256": prompt_sha,
    "reference_policy_class": "absent: r1 wrote no reference policy (v0.5 has none)",
}
json.dump(rec, open(extra, "w"), indent=1, sort_keys=True)
EOF
KITFLAG=""
[ "$KIT_MODE" = 1 ] && KITFLAG="--kit"
# shellcheck disable=SC2086
uv run python -m agent_loop_lab.v05.kit_b record "$AUTHOR_DIR" "$RECORD" $KITFLAG --extra "$EXTRA" > /dev/null
[ -z "$(ls -A "$AUTHOR_DIR/output/v05-phase-b")" ] || die "output/v05-phase-b is not empty"
uv run python -m agent_loop_lab.v05.kit_b verify "$AUTHOR_DIR" "$RECORD" --r1-dir "$R1_DIR" --disclosure-dir "$DISC_DIR" > /dev/null \
  || die "the fresh author dir does not verify against $RECORD"
"$AUTHOR_DIR/.venv/bin/python" -I -B -c "import torch, numpy; print('author venv:', torch.__version__, numpy.__version__)"
echo "handover record: $RECORD"
echo "author dir ready: $AUTHOR_DIR"
