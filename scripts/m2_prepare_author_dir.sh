#!/bin/sh
# Prepare the isolated weight-author working directory for M2 (spec v0.2 §8, §9).
#
#   scripts/m2_prepare_author_dir.sh [AUTHOR_DIR]
#
# AUTHOR_DIR defaults to ../HypaNet-m2-author (a sibling of the repo, never inside it).
# Contents: packet/ (fresh build, verified against the freeze registry), output/, tools/shape_check.py,
# the ./py wrapper and a private .venv (Python 3.12, torch 2.2.2, numpy<2). A copy of the exact packet
# handed over is kept in experiments/m2/packet_handed_over/ with HANDOVER.json for the commit.
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
AUTHOR_DIR="${1:-$(dirname "$REPO")/HypaNet-m2-author}"
case "$AUTHOR_DIR" in "$REPO"|"$REPO"/*) echo "refusing: author dir must be outside the repo" >&2; exit 2;; esac
if [ -e "$AUTHOR_DIR" ] && [ -n "$(ls -A "$AUTHOR_DIR" 2>/dev/null)" ]; then
  echo "refusing: $AUTHOR_DIR exists and is not empty" >&2; exit 2
fi
cd "$REPO"
uv run python scripts/freeze.py --check
mkdir -p "$AUTHOR_DIR/output/phase-a-strict" "$AUTHOR_DIR/output/phase-a-selfcheck" "$AUTHOR_DIR/tools"
uv run python scripts/build_authoring_packet.py --out "$AUTHOR_DIR/packet"
uv run python - "$AUTHOR_DIR/packet" <<'EOF'
import json, sys, yaml
from agent_loop_lab.packet import verify_packet
pkt = sys.argv[1]
problems = verify_packet(pkt)
assert not problems, problems
got = json.load(open(f"{pkt}/MANIFEST.json"))["packet_sha256"]
want = yaml.safe_load(open("experiments/registry.yaml"))["packet_sha256"]
assert got == want, f"packet {got} != registry {want}"
print("packet verified, sha256 matches registry:", got)
EOF
cp experiments/m2/shape_check.py "$AUTHOR_DIR/tools/shape_check.py"
cp experiments/m2/py "$AUTHOR_DIR/py" && chmod +x "$AUTHOR_DIR/py"
uv venv -q --python 3.12 "$AUTHOR_DIR/.venv"
uv pip install -q --python "$AUTHOR_DIR/.venv/bin/python" "torch==2.2.2" "numpy>=1.26,<2"
rm -rf experiments/m2/packet_handed_over && cp -R "$AUTHOR_DIR/packet" experiments/m2/packet_handed_over
uv run python - "$AUTHOR_DIR" <<'EOF'
import hashlib, json, sys
from pathlib import Path
d = Path(sys.argv[1])
rec = {
    "author_dir": str(d),
    "packet_sha256": json.loads((d / "packet/MANIFEST.json").read_text())["packet_sha256"],
    "tools": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [d / "tools/shape_check.py", d / "py"]},
    "prompts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path("experiments/m2").glob("prompt_*.md"))},
    "settings": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path("experiments/m2").glob("settings_*.json"))},
}
Path("experiments/m2/HANDOVER.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
print(json.dumps(rec, indent=2, sort_keys=True))
EOF
PYTHONDONTWRITEBYTECODE=1 "$AUTHOR_DIR/.venv/bin/python" -c "import torch, numpy; print('author venv:', torch.__version__, numpy.__version__)"
echo "author dir ready: $AUTHOR_DIR"
