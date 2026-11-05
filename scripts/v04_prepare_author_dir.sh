#!/bin/sh
# Prepare the sealed v0.4 weight-author directory (docs/V04_WSP_KERNEL.md §4-§7; plan v0.4 "Rules for Fable's
# weight-space programming").
#
#   scripts/v04_prepare_author_dir.sh [AUTHOR_DIR]
#
# AUTHOR_DIR defaults to ../HypaNet-v04-author (a sibling of the repo) and must not exist yet: a fresh
# directory, never reused. Steps:
#   1. check the G1 freeze registry (scripts/v04_freeze.py --check) when experiments/registry_v0_4.yaml
#      exists; without it, refuse unless V04_PREPARE_UNFROZEN=1 (kit tests only: no packet hash to check,
#      and the handover record goes to $V04_HANDOVER, default AUTHOR_DIR.handover.json, not into the repo);
#   2. build the packet with scripts/v04_build_packet.py --out AUTHOR_DIR/packet and require its sha256
#      (sha256 of PACKET_MANIFEST.json) to equal the builder's printed hash and the registry's;
#      packet/qwen/ must hold config.json, tokenizer.json and tokenizer_config.json with the pinned
#      hashes (hypaqwen.model.FILE_SHA256) and no weight file may be anywhere in the packet;
#   3. install tools/ read-only with hypaqwen.wsp.install (hn_kernel/, hn_guard/, hn_hook/hook_bash.py);
#   4. install the ./py and ./fwd wrappers (0555);
#   5. build the author venv (envs/qwen/scripts/make_author_venv.sh AUTHOR_DIR/.venv);
#   6. create output/wsp-a/scratch/ and .kernel/ (0700, for the model server's sockets); make packet/,
#      tools/, .venv/, the wrappers, output/ and AUTHOR_DIR itself read-only (only output/wsp-a/ stays
#      writable);
#   7. render the WSP-A settings once and run the hook self-test against the installed hook;
#   8. keep a copy of the packet in experiments/v0_4/packet_handed_over/ and write
#      experiments/v0_4/HANDOVER.json with every hash handed over, for the commit.
# Nothing is launched. V04_PACKET_BUILDER (a different builder script) is honoured only with
# V04_PREPARE_UNFROZEN=1.
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd -P)"
PARENT="$(dirname "$REPO")"
AUTHOR_DIR="${1:-$PARENT/HypaNet-v04-author}"
KIT="$REPO/experiments/v0_4/author_kit"
QWEN="$REPO/envs/qwen"
PHASE=wsp-a
REGISTRY="$REPO/experiments/registry_v0_4.yaml"
die() { echo "refusing: $*" >&2; exit 2; }
sha_of() { shasum -a 256 "$1" | cut -d' ' -f1; }

case "$AUTHOR_DIR" in /*) ;; *) AUTHOR_DIR="$(pwd -P)/$AUTHOR_DIR";; esac
case "$AUTHOR_DIR" in "$REPO"|"$REPO"/*) die "author dir must be outside the repo";; esac
[ -e "$AUTHOR_DIR" ] && die "$AUTHOR_DIR already exists (the v0.4 author directory must be fresh)"
[ -d "$(dirname "$AUTHOR_DIR")" ] || die "$(dirname "$AUTHOR_DIR") does not exist"
for f in "$KIT/py" "$KIT/fwd" "$KIT/hook_bash.py" "$KIT/settings_wsp_a.json" "$KIT/prompt_wsp_a.md"; do
  [ -f "$f" ] || die "kit file $f missing"
done
# Claude Code loads CLAUDE.md files from the session directory's ancestors: none may exist.
d="$(dirname "$AUTHOR_DIR")"
while :; do
  for f in CLAUDE.md CLAUDE.local.md .claude/CLAUDE.md; do [ -e "$d/$f" ] && die "$d/$f would be loaded into the session"; done
  [ "$d" = / ] && break
  d="$(dirname "$d")"
done
[ -e "$HOME/.claude/CLAUDE.md" ] && die "$HOME/.claude/CLAUDE.md would be loaded into the session"

cd "$REPO"
UNFROZEN=0
BUILDER="$REPO/scripts/v04_build_packet.py"
if [ -f "$REGISTRY" ]; then
  uv run python scripts/v04_freeze.py --check
  P_SHA="$(sed -n 's/^packet_sha256:[[:space:]]*\([0-9a-f]\{64\}\)[[:space:]]*$/\1/p' "$REGISTRY")"
  [ "${#P_SHA}" -eq 64 ] || die "no packet_sha256 in $REGISTRY"
  HANDOVER="$REPO/experiments/v0_4/HANDOVER.json"
elif [ "${V04_PREPARE_UNFROZEN:-0}" = 1 ]; then
  echo "WARNING: no $REGISTRY: preparing an UNFROZEN author dir (kit tests only; never launch WSP-A from it)" >&2
  UNFROZEN=1
  P_SHA=""
  HANDOVER="${V04_HANDOVER:-$AUTHOR_DIR.handover.json}"
  BUILDER="${V04_PACKET_BUILDER:-$BUILDER}"
  case "$HANDOVER" in "$REPO"/*) die "an unfrozen handover record may not be written into the repo";; esac
else
  die "experiments/registry_v0_4.yaml does not exist (G1 not frozen); set V04_PREPARE_UNFROZEN=1 for kit tests only"
fi
[ -f "$BUILDER" ] || die "packet builder $BUILDER missing"

mkdir -p "$AUTHOR_DIR/output/$PHASE/scratch"
mkdir -m 700 "$AUTHOR_DIR/.kernel"

# 2. packet
BUILD_OUT="$(uv run python "$BUILDER" --out "$AUTHOR_DIR/packet")" || die "packet build failed"
echo "$BUILD_OUT"
PRINTED="$(echo "$BUILD_OUT" | sed -n 's/^packet sha256:[[:space:]]*\([0-9a-f]\{64\}\)[[:space:]]*$/\1/p' | tail -n 1)"
GOT="$(sha_of "$AUTHOR_DIR/packet/PACKET_MANIFEST.json")"
[ "$GOT" = "$PRINTED" ] || die "PACKET_MANIFEST.json sha256 $GOT != the builder's printed $PRINTED"
[ -z "$P_SHA" ] || [ "$GOT" = "$P_SHA" ] || die "built packet $GOT != registry $P_SHA"
for f in SPEC.md AUTHOR_INSTRUCTIONS.md qwen/config.json qwen/tokenizer.json qwen/tokenizer_config.json; do
  [ -f "$AUTHOR_DIR/packet/$f" ] || die "packet/$f missing"
done
WEIGHTS="$(find "$AUTHOR_DIR/packet" -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o -name '*.pth' -o -name '*.gguf' -o -name '*.npz' -o -name '*.npy' \) | head -n 1)"
[ -z "$WEIGHTS" ] || die "the packet holds a weight or array file: $WEIGHTS"
LINKS="$(find "$AUTHOR_DIR/packet" -type l | head -n 1)"
[ -z "$LINKS" ] || die "the packet holds a symlink: $LINKS"
(cd "$QWEN" && uv run python - "$AUTHOR_DIR/packet/qwen") <<'EOF'
import hashlib, sys
from pathlib import Path
from hypaqwen.model import FILE_SHA256
d = Path(sys.argv[1])
for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
    got = hashlib.sha256((d / name).read_bytes()).hexdigest()
    if got != FILE_SHA256[name]:
        sys.exit(f"refusing: packet/qwen/{name} sha256 {got} != pinned {FILE_SHA256[name]}")
print("packet/qwen: config and tokenizer match the pinned snapshot")
EOF

# 3-5. tools, wrappers, venv
TOOLS_JSON="$(cd "$QWEN" && uv run python -m hypaqwen.wsp.install install "$AUTHOR_DIR/tools" --hook "$KIT/hook_bash.py")"
cp "$KIT/py" "$AUTHOR_DIR/py"
cp "$KIT/fwd" "$AUTHOR_DIR/fwd"
chmod 555 "$AUTHOR_DIR/py" "$AUTHOR_DIR/fwd"
sh "$QWEN/scripts/make_author_venv.sh" "$AUTHOR_DIR/.venv"
VENV_JSON="$("$AUTHOR_DIR/.venv/bin/python" -I -B -c 'import json, sys, numpy, safetensors, tokenizers, torch; print(json.dumps({"python": sys.version.split()[0], "torch": torch.__version__, "numpy": numpy.__version__, "safetensors": safetensors.__version__, "tokenizers": tokenizers.__version__}))')"

# 6. read-only everywhere except output/wsp-a/
chmod -R a-w "$AUTHOR_DIR/packet" "$AUTHOR_DIR/.venv"
chmod 555 "$AUTHOR_DIR/output"
chmod 555 "$AUTHOR_DIR"
(cd "$QWEN" && uv run python -m hypaqwen.wsp.install verify "$AUTHOR_DIR/tools" --hook "$KIT/hook_bash.py")

# 7. settings render and hook self-test (the rendered copy is thrown away; the launcher renders its own)
TMP_SETTINGS="$(mktemp -d)/settings_$PHASE.json"
uv run python -m agent_loop_lab.v04.kit render-settings --phase "$PHASE" --author-dir "$AUTHOR_DIR" --out "$TMP_SETTINGS"
uv run python -m agent_loop_lab.v04.kit hook-selftest --phase "$PHASE" --author-dir "$AUTHOR_DIR" --settings "$TMP_SETTINGS"
rm -f "$TMP_SETTINGS" && rmdir "$(dirname "$TMP_SETTINGS")"

# 8. handover record
if [ "$UNFROZEN" = 0 ]; then
  rm -rf experiments/v0_4/packet_handed_over
  cp -R "$AUTHOR_DIR/packet" experiments/v0_4/packet_handed_over
  chmod -R u+w experiments/v0_4/packet_handed_over
fi
uv run python - "$AUTHOR_DIR" "$GOT" "$UNFROZEN" "$HANDOVER" "$TOOLS_JSON" "$VENV_JSON" <<'EOF'
import hashlib, json, sys
from pathlib import Path
d, p_sha, unfrozen, out, tools, venv = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1", Path(sys.argv[4]), \
    json.loads(sys.argv[5]), json.loads(sys.argv[6])
kit = Path("experiments/v0_4/author_kit")
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
reg = Path("experiments/registry_v0_4.yaml")
rec = {
    "author_dir": str(d),
    "phase": "wsp-a",
    "unfrozen_kit_test_dir": unfrozen,
    "packet_sha256": p_sha,
    "packet_files": {p.relative_to(d / "packet").as_posix(): sha(p) for p in sorted((d / "packet").rglob("*")) if p.is_file()},
    "registry_sha256": sha(reg) if reg.is_file() else None,
    "tools": {f"tools/{k}": v for k, v in sorted(tools.items())},
    "wrappers": {"py": sha(d / "py"), "fwd": sha(d / "fwd")},
    "kit": {p.name: sha(p) for p in sorted(kit.iterdir()) if p.is_file()},
    "venv": venv,
}
out.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: rec[k] for k in ("author_dir", "packet_sha256", "registry_sha256", "unfrozen_kit_test_dir", "venv")}, indent=2))
print(f"handover record: {out}")
EOF
echo "author dir ready: $AUTHOR_DIR"
