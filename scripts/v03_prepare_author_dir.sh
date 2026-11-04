#!/bin/sh
# Prepare the sealed v0.3 weight-author directory (spec v0.3 §8, E6).
#
#   scripts/v03_prepare_author_dir.sh [AUTHOR_DIR]
#
# AUTHOR_DIR defaults to ../HypaNet-v03-author and must not exist yet (a fresh directory, never reused).
# Steps:
#   1. refuse unless the only sibling HypaNet* directories are the repo, HypaNet-m2-author and the new
#      author dir (the settings deny-lists name exactly those);
#   2. check the v0.3 freeze registry (scripts/v03_freeze.py --check);
#   3. build the packet into AUTHOR_DIR/packet and require its sha256 to equal the registry's;
#   4. install tools/ (standalone AUTHOR_RUN guard + launcher) and the ./py wrapper;
#   5. create the private venv (Python 3.12, torch and numpy at the versions of this repo's uv.lock env);
#   6. make packet/, tools/ and py read-only; create output/v03-phase-a/scratch/;
#   7. keep a copy of the packet in experiments/v0_3/packet_handed_over/ and write
#      experiments/v0_3/HANDOVER.json with every hash handed over, for the commit.
# Nothing is launched.
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PARENT="$(dirname "$REPO")"
AUTHOR_DIR="${1:-$PARENT/HypaNet-v03-author}"
KIT="$REPO/experiments/v0_3/author_kit"
die() { echo "refusing: $*" >&2; exit 2; }

case "$AUTHOR_DIR" in "$REPO"|"$REPO"/*) die "author dir must be outside the repo";; esac
[ -e "$AUTHOR_DIR" ] && die "$AUTHOR_DIR already exists (the v0.3 author directory must be fresh)"
for d in "$PARENT"/HypaNet*; do
  [ -e "$d" ] || continue
  case "$(basename "$d")" in
    HypaNet|HypaNet-m2-author|"$(basename "$AUTHOR_DIR")") ;;
    *) die "unexpected sibling directory $d (not covered by the settings deny-lists; extend them first)";;
  esac
done

cd "$REPO"
uv run python scripts/v03_freeze.py --check
P_SHA="$(sed -n 's/^packet_sha256:[[:space:]]*\([0-9a-f]\{64\}\)[[:space:]]*$/\1/p' experiments/registry_v0_3.yaml)"
[ "${#P_SHA}" -eq 64 ] || die "no packet_sha256 in experiments/registry_v0_3.yaml"

mkdir -p "$AUTHOR_DIR/output/v03-phase-a/scratch"
uv run python scripts/v03_build_packet.py --out "$AUTHOR_DIR/packet"
GOT="$(shasum -a 256 "$AUTHOR_DIR/packet/PACKET_MANIFEST.json" | cut -d' ' -f1)"
[ "$GOT" = "$P_SHA" ] || die "built packet $GOT != registry $P_SHA"
uv run python -m agent_loop_lab.v03.kit install-tools "$AUTHOR_DIR/tools"
cp "$KIT/py" "$AUTHOR_DIR/py" && chmod 755 "$AUTHOR_DIR/py"

TORCH_V="$(uv run python -c 'import torch; print(torch.__version__)')"
NUMPY_V="$(uv run python -c 'import numpy; print(numpy.__version__)')"
uv venv -q --python 3.12 "$AUTHOR_DIR/.venv"
uv pip install -q --python "$AUTHOR_DIR/.venv/bin/python" "torch==$TORCH_V" "numpy==$NUMPY_V"

chmod -R a-w "$AUTHOR_DIR/packet" "$AUTHOR_DIR/tools"
chmod 555 "$AUTHOR_DIR/py"

rm -rf experiments/v0_3/packet_handed_over
cp -R "$AUTHOR_DIR/packet" experiments/v0_3/packet_handed_over
chmod -R u+w experiments/v0_3/packet_handed_over
uv run python - "$AUTHOR_DIR" "$P_SHA" "$TORCH_V" "$NUMPY_V" <<'EOF'
import hashlib, json, sys
from pathlib import Path
d, p_sha, torch_v, numpy_v = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
kit = Path("experiments/v0_3/author_kit")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rec = {
    "author_dir": str(d),
    "packet_sha256": p_sha,
    "registry_sha256": sha(Path("experiments/registry_v0_3.yaml")),
    "tools": {p.relative_to(d).as_posix(): sha(p) for p in sorted((d / "tools").rglob("*")) if p.is_file()},
    "py": sha(d / "py"),
    "prompts": {p.name: sha(p) for p in sorted(kit.glob("prompt_*.md"))},
    "settings": {p.name: sha(p) for p in sorted(kit.glob("settings_*.json"))},
    "venv": {"torch": torch_v, "numpy": numpy_v},
}
Path("experiments/v0_3/HANDOVER.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
print(json.dumps(rec, indent=2, sort_keys=True))
EOF
"$AUTHOR_DIR/.venv/bin/python" -I -B -c "import torch, numpy; print('author venv:', torch.__version__, numpy.__version__)"
echo "author dir ready: $AUTHOR_DIR"
