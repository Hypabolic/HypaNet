"""Contact sheet of the reference renders: reference pairs first, then held-back (labelled).

    uv run python reference/contact_sheet.py [--dir reference/out] [--out reference/out/contact_sheet.png] [--scale 0.5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE / "out"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--cols", type=int, default=3)
    args = ap.parse_args()
    d = Path(args.dir)
    shots = json.loads((HERE.parent / "scene" / "out" / "shots.json").read_text())
    ids = [(p["id"], g) for g in ("reference", "heldback") for p in shots["pairs"][g]]
    ids = [(i, g) for i, g in ids if (d / f"{i}.png").exists()]
    W, H = int(640 * args.scale), int(400 * args.scale)
    rows = (len(ids) + args.cols - 1) // args.cols
    sheet = Image.new("RGB", (W * args.cols, H * rows), (0, 0, 0))
    dr = ImageDraw.Draw(sheet)
    for k, (i, g) in enumerate(ids):
        im = Image.open(d / f"{i}.png").convert("RGB").resize((W, H), Image.LANCZOS)
        x, y = (k % args.cols) * W, (k // args.cols) * H
        sheet.paste(im, (x, y))
        info = json.loads((d / f"{i}.json").read_text()) if (d / f"{i}.json").exists() else {}
        label = f"{i}{' (held-back)' if g == 'heldback' else ''}  {info.get('spp', '?')} spp"
        dr.text((x + 5, y + 4), label, fill=(0, 0, 0)); dr.text((x + 4, y + 3), label, fill=(255, 255, 255))
    out = Path(args.out) if args.out else d / "contact_sheet.png"
    sheet.save(out)
    print(out)


if __name__ == "__main__":
    main()
