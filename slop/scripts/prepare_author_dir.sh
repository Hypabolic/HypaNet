#!/bin/sh
# Prepare the HypaNet slop author directory (SPEC §13). Nothing is launched. From clouds/scripts/prepare_author_dir.sh.
#
#   slop/scripts/prepare_author_dir.sh [--force] [--test-root DIR] [AUTHOR_DIR]
#
# AUTHOR_DIR defaults to ../HypaNet-slop-author (a sibling of the HypaNet repo); the smoke launch's directory,
# ../HypaNet-slop-smoke (SPEC §12.7), and round 2's, ../HypaNet-slop-author-r2, are prepared the same way by naming
# them. It must be outside the repo, its name must start with HypaNet-slop-author-r2 (run r2), HypaNet-slop-author
# (run r1) or HypaNet-slop-smoke (run smoke, with r1's kit), and it must not exist, unless
# --force is given and it is an earlier slop hand-over (its HANDOVER.json says so): then it is deleted and rebuilt.
# Refuses if that run's session was already launched (slop/runs/<run>/SESSION_ID or a transcript in
# slop/runs/<run>/logs/): the record later launches verify against is never replaced after that.
#
# --test-root DIR (tests only): the author's tools may also read from DIR and write there; KIT.md and PROMPT.md may be
# placeholders while kit/KIT.md or kit/prompt.md is unwritten; the settings go to DIR/settings.json and the record to
# DIR/HANDOVER.record.json (the repo's kit/settings*.json and runs/ are left alone); the record is marked
# "test": true, which the launcher refuses; the sealed n-gram leakage check may be "not run". Re-run without it (and
# with --force) for a real hand-over.
#
# Steps:
#   1. checks: kit/lib equals ui/ and tools/py_guard.py equals clouds' (kit/kit_build.py lib --check); the dev and
#      reference sets exist; the run's settings (kit/settings.json for r1, kit/settings.r2.json for r2,
#      kit/settings.smoke.json for smoke) regenerated for AUTHOR_DIR (kit/make_settings.py enumerates ~/.claude) and
#      checked;
#   2. AUTHOR_DIR: kit/kit_build.py tree writes KIT.md, PROMPT.md, LABELLING.md (the author's copy of the guide),
#      kit/lib/, tools/py_guard.py and examples/ (the 400 disclosed dev screens, the reference screens and their
#      sources, INDEX.txt, README.md); kit/leakage.py must pass on it, with --require-sealed on a real hand-over (the
#      sealed n-gram check of SPEC §9 must have run, or be waived by name in kit/leakage_waivers.json); the wrappers ./tok, ./lint, ./score, ./inspect (kit/tools/author_tool.py on
#      the harness side, with uv and node) and ./python3 (the private venv under tools/py_guard.py); .venv/ (Python
#      3.12 + numpy, private); work/ (empty: the only writable place). Run r2 (kit/kit_build.py RUNS): 1,200 dev
#      screens, PROMPT.md from kit/prompt_r2.md, resources/ (codex, thesaurus, r1 results; read-only) and work/
#      seeded with runs/r1/work minus scratch/ (writable; the record keeps the seed's hashes);
#   3. permissions: everything read-only except work/ (the author dir itself too, so nothing can be added beside);
#   4. the hand-over record slop/runs/<run>/HANDOVER.json (sha256 of every handed-over file, the venv, node, the
#      harness files the tools run and the data the kit was built from; the record the launcher verifies against),
#      and AUTHOR_DIR/HANDOVER.json, the author's view of it (file hashes, venv and node only).
set -eu
SLOP="$(cd "$(dirname "$0")/.." && pwd -P)"
REPO="$(cd "$SLOP/.." && pwd -P)"
PARENT="$(dirname "$REPO")"
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
AUTHOR_DIR="${AUTHOR_DIR:-$PARENT/HypaNet-slop-author}"
mkdir -p "$(dirname "$AUTHOR_DIR")"
AUTHOR_DIR="$(cd "$(dirname "$AUTHOR_DIR")" && pwd -P)/$(basename "$AUTHOR_DIR")"
case "$AUTHOR_DIR" in "$REPO"|"$REPO"/*) die "author dir must be outside the repo";; esac
case "$(basename "$AUTHOR_DIR")" in
  HypaNet-slop-author-r2*) RUN="$SLOP/runs/r2"; RUN_SETTINGS="$SLOP/kit/settings.r2.json"; KIT_RUN=r2;;
  HypaNet-slop-author*) RUN="$SLOP/runs/r1"; RUN_SETTINGS="$SLOP/kit/settings.json"; KIT_RUN=r1;;
  HypaNet-slop-smoke*) RUN="$SLOP/runs/smoke"; RUN_SETTINGS="$SLOP/kit/settings.smoke.json"; KIT_RUN=r1;;
  *) die "author dir name must start with HypaNet-slop-author-r2, HypaNet-slop-author or HypaNet-slop-smoke";;
esac
PROMPT_SRC=prompt.md
if [ "$KIT_RUN" = r2 ]; then PROMPT_SRC=prompt_r2.md; fi
if [ -n "$TEST_ROOT" ]; then
  [ -d "$TEST_ROOT" ] || die "--test-root $TEST_ROOT is not a directory"
  TEST_ROOT="$(cd "$TEST_ROOT" && pwd -P)"
  case "$TEST_ROOT" in "$AUTHOR_DIR"|"$AUTHOR_DIR"/*|"$REPO"|"$REPO"/*) die "--test-root must be outside the author dir and the repo";; esac
  case "$AUTHOR_DIR" in "$TEST_ROOT"/*) die "--test-root must not contain the author dir (the tools would read beside it)";; esac
  RECORD="$TEST_ROOT/HANDOVER.record.json"
  SETTINGS="$TEST_ROOT/settings.json"
else
  RECORD="$RUN/HANDOVER.json"
  SETTINGS="$RUN_SETTINGS"
fi
if [ -e "$RUN/SESSION_ID" ]; then die "$RUN/SESSION_ID exists (a session was launched; the hand-over is frozen)"; fi
for t in "$RUN"/logs/*.stream.jsonl; do
  if [ -e "$t" ]; then die "$t exists (a session was launched; the hand-over is frozen)"; fi
done
if [ -e "$AUTHOR_DIR" ]; then
  [ "$FORCE" = 1 ] || die "$AUTHOR_DIR already exists (use --force to rebuild an earlier hand-over)"
  grep -q '"kind": "hypanet-slop-author"' "$AUTHOR_DIR/HANDOVER.json" 2>/dev/null \
    || die "$AUTHOR_DIR exists but is not a HypaNet slop hand-over (no HANDOVER.json); not deleting it"
  chmod -R u+w "$AUTHOR_DIR"
  rm -rf "$AUTHOR_DIR"
fi
for f in make_settings.py kit_build.py leakage.py tools/author_tool.py tools/lint_run.mjs tools/py_guard.py; do
  [ -f "$SLOP/kit/$f" ] || die "kit/$f missing"
done
DOCS_FLAG=""
for f in KIT.md "$PROMPT_SRC"; do
  if [ ! -f "$SLOP/kit/$f" ]; then
    [ -n "$TEST_ROOT" ] || die "kit/$f missing (a real hand-over needs it)"
    DOCS_FLAG="--allow-missing-docs"
  fi
done
for f in datasets/dev/screens.jsonl datasets/dev/dev.gen.jsonl datasets/dev/dev.meta.jsonl \
         datasets/reference_real/manifest.json datasets/kinds.json; do
  [ -f "$SLOP/$f" ] || die "$f missing (build the sets first)"
done
UV="$(command -v uv)" || die "uv not on PATH"
NODE="$(command -v node)" || die "node not on PATH"
EXTRA="${TEST_ROOT:--}"

cd "$SLOP"
NUMPY_V="$("$UV" run --no-sync --quiet python -c 'import numpy; print(numpy.__version__)')"
PY_V="$("$UV" run --no-sync --quiet python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
# 1. checks and settings
"$UV" run --no-sync --quiet python kit/kit_build.py lib --check || die "kit/lib or kit/tools/py_guard.py is stale (uv run python kit/kit_build.py lib)"
"$UV" run --no-sync --quiet python kit/make_settings.py "$AUTHOR_DIR" --out "$SETTINGS"
"$UV" run --no-sync --quiet python kit/make_settings.py --check "$AUTHOR_DIR" --out "$SETTINGS"
# 2. author dir
mkdir -p "$AUTHOR_DIR/work"
"$UV" run --no-sync --quiet python kit/kit_build.py tree "$AUTHOR_DIR" --run "$KIT_RUN" $DOCS_FLAG
if [ -n "$TEST_ROOT" ]; then
  "$UV" run --no-sync --quiet python kit/leakage.py "$AUTHOR_DIR" --run "$KIT_RUN" || die "the leakage check failed on $AUTHOR_DIR (see above)"
else
  "$UV" run --no-sync --quiet python kit/leakage.py "$AUTHOR_DIR" --run "$KIT_RUN" --require-sealed \
    || die "the leakage check failed on $AUTHOR_DIR (see above; a real hand-over needs the sealed n-gram check)"
fi
for tool in tok lint score inspect; do
  cat > "$AUTHOR_DIR/$tool" <<EOF
#!/bin/sh
# HypaNet slop author tool: ./$tool (see KIT.md). Runs the harness's tokeniser, extractor, linter and model; do not modify.
here="\$(cd "\$(dirname "\$0")" && pwd -P)"
exec "$UV" run --project "$SLOP" --no-sync --quiet python -I -B "$SLOP/kit/tools/author_tool.py" "\$here" "$EXTRA" "$NODE" $tool "\$@"
EOF
done
cat > "$AUTHOR_DIR/python3" <<'EOF'
#!/bin/sh
# HypaNet slop author tool: ./python3 = this directory's private Python (numpy only) under tools/py_guard.py.
here="$(cd "$(dirname "$0")" && pwd -P)"
PYTHONDONTWRITEBYTECODE=1 exec "$here/.venv/bin/python" -I -B "$here/tools/py_guard.py" "$@"
EOF
"$UV" venv -q --python "$PY_V" "$AUTHOR_DIR/.venv"
"$UV" pip install -q --python "$AUTHOR_DIR/.venv/bin/python" "numpy==$NUMPY_V"
# 3. permissions: read-only except work/
chmod 555 "$AUTHOR_DIR/tok" "$AUTHOR_DIR/lint" "$AUTHOR_DIR/score" "$AUTHOR_DIR/inspect" "$AUTHOR_DIR/python3"
chmod -R a-w "$AUTHOR_DIR/examples" "$AUTHOR_DIR/kit" "$AUTHOR_DIR/tools" "$AUTHOR_DIR/.venv" \
  "$AUTHOR_DIR/KIT.md" "$AUTHOR_DIR/PROMPT.md" "$AUTHOR_DIR/LABELLING.md"
if [ -d "$AUTHOR_DIR/resources" ]; then chmod -R a-w "$AUTHOR_DIR/resources"; fi
chmod 755 "$AUTHOR_DIR/work"
# the r2 seed (r1's work/) stays writable: directories 755, files 644
find "$AUTHOR_DIR/work" -type d -exec chmod 755 {} +
find "$AUTHOR_DIR/work" -type f -exec chmod 644 {} +
# 4. record (HANDOVER.json is written before the author dir itself becomes read-only)
if [ -n "$TEST_ROOT" ]; then
  "$UV" run --no-sync --quiet python kit/kit_build.py record "$AUTHOR_DIR" "$RECORD" --node "$NODE" --run "$KIT_RUN" --test-root "$TEST_ROOT"
else
  "$UV" run --no-sync --quiet python kit/kit_build.py record "$AUTHOR_DIR" "$RECORD" --node "$NODE" --run "$KIT_RUN"
fi
chmod 444 "$AUTHOR_DIR/HANDOVER.json"
chmod 555 "$AUTHOR_DIR"
"$UV" run --no-sync --quiet python kit/kit_build.py verify "$AUTHOR_DIR" "$RECORD" --pristine
for d in examples kit tools .venv resources; do
  [ -e "$AUTHOR_DIR/$d" ] || continue
  [ -z "$(find "$AUTHOR_DIR/$d" -perm -u+w -not -type l -print -quit)" ] || die "$AUTHOR_DIR/$d is still writable"
done
echo "author dir ready ($KIT_RUN kit): $AUTHOR_DIR (record $RECORD)"
if [ -n "$TEST_ROOT" ]; then echo "TEST hand-over (extra root $TEST_ROOT): the launcher will refuse it; re-run with --force and no --test-root"; fi
exit 0
