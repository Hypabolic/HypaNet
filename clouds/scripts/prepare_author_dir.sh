#!/bin/sh
# Prepare the HypaNet Clouds author directory (SPEC section 7). Nothing is launched.
#
#   clouds/scripts/prepare_author_dir.sh [--force] [--test-root DIR] [AUTHOR_DIR]
#
# AUTHOR_DIR defaults to ../HypaNet-clouds-author (a sibling of the HypaNet repo). It must be outside the repo and
# must not exist, unless --force is given and it is an earlier HypaNet Clouds hand-over (its HANDOVER.json says so):
# then it is deleted and rebuilt. Refuses if a session was already launched (clouds/runs/r1/SESSION_ID or a
# transcript in clouds/runs/r1/logs/): the record later launches verify against is never replaced after that.
#
# --test-root DIR (tests only): the author's ./render and ./inspect may also read networks from DIR and write
# output there, and the record is marked "test": true, which scripts/launch.sh refuses. Re-run without it (and with
# --force) to leave a pristine real hand-over.
#
# Steps:
#   1. kit/build/scene/: the sanitised scene the tools serve (kit/kit_build.py scene): density, sky maps and a
#      shots.json with only the reference shots, presets and pairs; kit/settings.json regenerated
#      (kit/make_settings.py: it enumerates ~/.claude, see there);
#   2. AUTHOR_DIR: KIT.md, PROMPT.md (the launch prompt, kit/prompt.md), references/ (the 12 reference pairs: PNG,
#      linear .npy, parameters .json; scene.json), tools/py_guard.py, the wrappers ./render and ./inspect (call
#      kit/tools/author_tool.py on the harness side with uv) and ./python3 (the private venv under the guard),
#      .venv/ (Python 3.12 + numpy, private), work/ (empty: the only writable place);
#   3. permissions: everything read-only except work/ (the author dir itself too, so nothing can be added beside);
#   4. HANDOVER.json (sha256 of every handed-over file, the venv, and the harness files the tools run) in AUTHOR_DIR
#      and in clouds/runs/r1/HANDOVER.json (the record launch.sh verifies against).
set -eu
CLOUDS="$(cd "$(dirname "$0")/.." && pwd -P)"
REPO="$(cd "$CLOUDS/.." && pwd -P)"
PARENT="$(dirname "$REPO")"
RUN="$CLOUDS/runs/r1"
RECORD="$RUN/HANDOVER.json"
die() { echo "refusing: $*" >&2; exit 2; }

FORCE=0
TEST_ROOT=""
AUTHOR_DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --force) FORCE=1;;
    --test-root) TEST_ROOT="${2:?--test-root DIR}"; shift;;
    -*) die "unknown option $1";;
    *) [ -z "$AUTHOR_DIR" ] || die "one AUTHOR_DIR only"; AUTHOR_DIR="$1";;
  esac
  shift
done
AUTHOR_DIR="${AUTHOR_DIR:-$PARENT/HypaNet-clouds-author}"
mkdir -p "$(dirname "$AUTHOR_DIR")"
AUTHOR_DIR="$(cd "$(dirname "$AUTHOR_DIR")" && pwd -P)/$(basename "$AUTHOR_DIR")"
case "$AUTHOR_DIR" in "$REPO"|"$REPO"/*) die "author dir must be outside the repo";; esac
case "$(basename "$AUTHOR_DIR")" in HypaNet-clouds-author*) ;; *) die "author dir name must start with HypaNet-clouds-author (the settings allow only that)";; esac
if [ -n "$TEST_ROOT" ]; then
  [ -d "$TEST_ROOT" ] || die "--test-root $TEST_ROOT is not a directory"
  TEST_ROOT="$(cd "$TEST_ROOT" && pwd -P)"
  case "$TEST_ROOT" in "$AUTHOR_DIR"|"$AUTHOR_DIR"/*|"$REPO"|"$REPO"/*) die "--test-root must be outside the author dir and the repo";; esac
fi
if [ -e "$RUN/SESSION_ID" ]; then die "$RUN/SESSION_ID exists (a session was launched; the hand-over is frozen)"; fi
for t in "$RUN"/logs/*.stream.jsonl; do
  if [ -e "$t" ]; then die "$t exists (a session was launched; the hand-over is frozen)"; fi
done
if [ -e "$AUTHOR_DIR" ]; then
  [ "$FORCE" = 1 ] || die "$AUTHOR_DIR already exists (use --force to rebuild an earlier hand-over)"
  grep -q '"kind": "hypanet-clouds-author"' "$AUTHOR_DIR/HANDOVER.json" 2>/dev/null \
    || die "$AUTHOR_DIR exists but is not a HypaNet Clouds hand-over (no HANDOVER.json); not deleting it"
  chmod -R u+w "$AUTHOR_DIR"
  rm -rf "$AUTHOR_DIR"
fi
for f in KIT.md prompt.md prompt_continue.md make_settings.py tools/author_tool.py tools/py_guard.py kit_build.py; do
  [ -f "$CLOUDS/kit/$f" ] || die "kit/$f missing"
done
for f in density.u8 shots.json; do [ -f "$CLOUDS/scene/out/$f" ] || die "scene/out/$f missing (scene/build_scene.py)"; done
UV="$(command -v uv)" || die "uv not on PATH"
NUMPY_V="$(cd "$CLOUDS" && "$UV" run --no-sync --quiet python -c 'import numpy; print(numpy.__version__)')"
EXTRA="${TEST_ROOT:--}"

cd "$CLOUDS"
# 1. sanitised scene
"$UV" run --no-sync --quiet python kit/kit_build.py scene
"$UV" run --no-sync --quiet python kit/make_settings.py "$AUTHOR_DIR"
# 2. author dir
mkdir -p "$AUTHOR_DIR/tools" "$AUTHOR_DIR/work"
"$UV" run --no-sync --quiet python kit/kit_build.py references "$AUTHOR_DIR"
cp kit/KIT.md "$AUTHOR_DIR/KIT.md"
cp kit/prompt.md "$AUTHOR_DIR/PROMPT.md"
cp kit/tools/py_guard.py "$AUTHOR_DIR/tools/py_guard.py"
for tool in render inspect; do
  cat > "$AUTHOR_DIR/$tool" <<EOF
#!/bin/sh
# HypaNet Clouds author tool: ./$tool (see KIT.md). Runs the harness renderer; do not modify.
here="\$(cd "\$(dirname "\$0")" && pwd -P)"
exec "$UV" run --project "$CLOUDS" --no-sync --quiet python -I -B "$CLOUDS/kit/tools/author_tool.py" "\$here" "$EXTRA" $tool "\$@"
EOF
done
cat > "$AUTHOR_DIR/python3" <<'EOF'
#!/bin/sh
# HypaNet Clouds author tool: ./python3 = this directory's private Python (numpy only) under tools/py_guard.py.
here="$(cd "$(dirname "$0")" && pwd -P)"
PYTHONDONTWRITEBYTECODE=1 exec "$here/.venv/bin/python" -I -B "$here/tools/py_guard.py" "$@"
EOF
"$UV" venv -q --python 3.12 "$AUTHOR_DIR/.venv"
"$UV" pip install -q --python "$AUTHOR_DIR/.venv/bin/python" "numpy==$NUMPY_V"
# 3. permissions: read-only except work/
chmod 555 "$AUTHOR_DIR/render" "$AUTHOR_DIR/inspect" "$AUTHOR_DIR/python3"
chmod -R a-w "$AUTHOR_DIR/references" "$AUTHOR_DIR/tools" "$AUTHOR_DIR/.venv" "$AUTHOR_DIR/KIT.md" "$AUTHOR_DIR/PROMPT.md"
chmod 755 "$AUTHOR_DIR/work"
# 4. record (HANDOVER.json is written before the author dir itself becomes read-only)
if [ -n "$TEST_ROOT" ]; then
  "$UV" run --no-sync --quiet python kit/kit_build.py record "$AUTHOR_DIR" "$RECORD" --test-root "$TEST_ROOT"
else
  "$UV" run --no-sync --quiet python kit/kit_build.py record "$AUTHOR_DIR" "$RECORD"
fi
chmod 444 "$AUTHOR_DIR/HANDOVER.json"
chmod 555 "$AUTHOR_DIR"
"$UV" run --no-sync --quiet python kit/kit_build.py verify "$AUTHOR_DIR" "$RECORD" --pristine
for d in references tools .venv; do
  [ -z "$(find "$AUTHOR_DIR/$d" -perm -u+w -not -type l -print -quit)" ] || die "$AUTHOR_DIR/$d is still writable"
done
echo "author dir ready: $AUTHOR_DIR"
if [ -n "$TEST_ROOT" ]; then echo "TEST hand-over (extra root $TEST_ROOT): launch.sh will refuse it; re-run with --force and no --test-root"; fi
exit 0
