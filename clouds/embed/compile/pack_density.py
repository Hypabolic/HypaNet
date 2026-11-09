"""Density grid for the embed page (DESIGN.md "Layout": web/data/).

    uv run python embed/compile/pack_density.py [--scene-dir scene/out] [--out embed/web/data]

Writes web/data/density.u8.gz (gzip -9 of scene/out/density.u8; the page inflates it with DecompressionStream) and
web/data/density.json (a copy of scene/out/density.json). The grid's sha256 is checked against density.json and the
compiler's manifest (web/gen/manifest.json: density_sha256), so the page never pairs a bake with another grid.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

CLOUDS = Path(__file__).resolve().parents[2]
WEB = CLOUDS / "embed" / "web"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-dir", type=Path, default=CLOUDS / "scene" / "out")
    ap.add_argument("--out", type=Path, default=WEB / "data")
    a = ap.parse_args(argv)
    meta = json.loads((a.scene_dir / "density.json").read_text())
    raw = (a.scene_dir / meta["file"]).read_bytes()
    nx, ny, nz = meta["dims"]
    if len(raw) != nx * ny * nz:
        raise SystemExit(f"{meta['file']}: {len(raw)} bytes, expected {nx * ny * nz}")
    sha = hashlib.sha256(raw).hexdigest()
    if sha != meta["sha256"]:
        raise SystemExit(f"{meta['file']}: sha256 {sha} does not match density.json {meta['sha256']}")
    man = WEB / "gen" / "manifest.json"
    if man.is_file() and json.loads(man.read_text()).get("density_sha256") not in (None, sha):
        print(f"WARNING: {man} was compiled for another density grid; rerun compile_net.py", file=sys.stderr)
    a.out.mkdir(parents=True, exist_ok=True)
    # mtime=0: the same grid always gives the same bytes
    gz = gzip.compress(raw, compresslevel=9, mtime=0)
    (a.out / "density.u8.gz").write_bytes(gz)
    (a.out / "density.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"density.u8.gz: {len(gz):,} bytes ({len(raw):,} raw, {nx}x{ny}x{nz}); density.json copied to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
