#!/bin/sh
# Build the v0.4 weight author's private venv (envs/qwen/hypaqwen/wsp/author/AUTHOR_VENV.md) with uv.
#
#   envs/qwen/scripts/make_author_venv.sh DIR
#
# DIR must not exist yet. The venv gets Python 3.12 and exactly torch, numpy, safetensors and tokenizers
# at the versions of envs/qwen/uv.lock (plus torch's own dependencies), and never transformers, peft,
# accelerate, sklearn or any repo code. The script refuses if a pin below disagrees with the lock, and
# checks the result in isolated mode: versions, the guard's verified torch, no forbidden package.
# Nothing is launched and no author directory is prepared here (that is the session launcher's job).
set -eu
die() { echo "refusing: $*" >&2; exit 2; }

[ "$#" -eq 1 ] || die "usage: $0 DIR"
DIR="$1"
[ -e "$DIR" ] && die "$DIR already exists (the author venv is built fresh)"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
LOCK="$HERE/uv.lock"

TORCH_V=2.14.1
NUMPY_V=2.5.4
SAFETENSORS_V=0.8.0
TOKENIZERS_V=0.23.3

locked() {  # version of package $1 in envs/qwen/uv.lock
  awk -v n="$1" '$0 == "name = \"" n "\"" { getline; gsub(/version = |"/, ""); print; exit }' "$LOCK"
}
for pv in "torch $TORCH_V" "numpy $NUMPY_V" "safetensors $SAFETENSORS_V" "tokenizers $TOKENIZERS_V"; do
  set -- $pv
  [ "$(locked "$1")" = "$2" ] || die "pin $1==$2 differs from envs/qwen/uv.lock ($(locked "$1"))"
done

uv venv -q --python 3.12 "$DIR"
uv pip install -q --python "$DIR/bin/python" \
  "torch==$TORCH_V" "numpy==$NUMPY_V" "safetensors==$SAFETENSORS_V" "tokenizers==$TOKENIZERS_V"

"$DIR/bin/python" -I -B - "$TORCH_V" "$NUMPY_V" "$SAFETENSORS_V" "$TOKENIZERS_V" <<'EOF'
import importlib.util, sys
import numpy, safetensors, tokenizers, torch
want = dict(zip(("torch", "numpy", "safetensors", "tokenizers"), sys.argv[1:]))
got = {"torch": torch.__version__, "numpy": numpy.__version__, "safetensors": safetensors.__version__,
       "tokenizers": tokenizers.__version__}
assert sys.version_info[:2] == (3, 12), sys.version
assert all(got[k].split("+")[0] == v for k, v in want.items()), (got, want)
bad = [m for m in ("transformers", "peft", "accelerate", "sklearn", "scipy", "hypaqwen", "agent_loop_lab")
       if importlib.util.find_spec(m) is not None]
assert not bad, f"forbidden packages in the author venv: {bad}"
print("author venv:", sys.version.split()[0], got)
EOF
echo "author venv ready: $DIR"
