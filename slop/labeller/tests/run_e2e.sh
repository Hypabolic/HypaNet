#!/usr/bin/env bash
# End-to-end: serve the fixtures into a temporary label dir, label all of them in headless Chromium with real key
# events (drive_browser.mjs), check the guide-change alarm against a scratch copy of the guide, then run check.py.
# Never writes to slop/labels/ and never edits LABELLING.md.
#   slop/labeller/tests/run_e2e.sh [OUT_DIR]       CHROME=/path/to/chrome to override the browser
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LAB="$(dirname "$HERE")"
SLOP="$(dirname "$LAB")"
OUT="${1:-$(mktemp -d)}"
mkdir -p "$OUT/labels"
CHROME="${CHROME:-$(ls -d "$HOME"/Library/Caches/ms-playwright/chromium_headless_shell-*/chrome-headless-shell-*/chrome-headless-shell 2>/dev/null | tail -1)}"
[ -x "$CHROME" ] || { echo "no headless Chromium found; set CHROME"; exit 2; }
PORT=$((8800 + RANDOM % 400))
python3 "$LAB/serve.py" --set fixtures --screens "$LAB/fixtures" --source-root "$LAB/fixtures/src" --labeller e2e \
  --labels-dir "$OUT/labels" --port "$PORT" > "$OUT/serve.log" 2>&1 &
S1=$!
cp "$SLOP/LABELLING.md" "$OUT/guide_copy.md"
python3 "$LAB/serve.py" --set guidetest --screens "$LAB/fixtures" --source-root "$LAB/fixtures/src" --labeller e2e \
  --labels-dir "$OUT/labels" --guide "$OUT/guide_copy.md" --port "$((PORT + 1))" > "$OUT/serve_guide.log" 2>&1 &
S2=$!
trap 'kill $S1 $S2 2>/dev/null || true' EXIT
sleep 1.5
node "$HERE/drive_browser.mjs" --url "http://127.0.0.1:$PORT/" --chrome "$CHROME" --out "$OUT/shots"
node "$HERE/drive_browser.mjs" --url "http://127.0.0.1:$((PORT + 1))/" --chrome "$CHROME" --out "$OUT/shots_guide" \
  --only guide --guide-copy "$OUT/guide_copy.md"
python3 "$LAB/check.py" "$OUT/labels/fixtures.e2e.jsonl" --export "$OUT/fixtures.e2e.latest.jsonl"
echo "artefacts in $OUT"
