"""Score HypaNet Clouds r1 renders against the Mitsuba references (SPEC section 7, afterwards).

    uv run python scripts/score.py [RENDER_DIR TAG]...    # default: runs/r1/score/net network, runs/r1/score/single single

For every pair of the reference (12) and held-back (6) sets, from the linear .npy files:

- full: mean |render - ref| / mean ref over the whole image and all channels (what render.py --compare prints; the
  exact sky dilutes it);
- cloud: the same on luminance Y over cloud pixels only (reference relative SE > 1e-3, as reference/NOTES.md);
- block8: luminance error on 8x8-pixel block means over blocks that are mostly cloud (Monte Carlo grain averages out,
  so this is closest to "does it look right");
- ratio R,G,B: sum render / sum ref per channel over cloud pixels (colour and brightness bias);
- floor: the error a perfect renderer would score per pixel against the noisy reference,
  sqrt(2/pi) * sum se_Y / sum Y_ref over cloud pixels (the block8 floor is about an eighth of it).

Writes runs/r1/score/scores.json and prints a table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CLOUDS = Path(__file__).resolve().parents[1]
REF = CLOUDS / "reference" / "out"
OUT = CLOUDS / "runs" / "r1" / "score"
LUM = np.array([0.2126, 0.7152, 0.0722])


def blocks(a: np.ndarray, n: int = 8) -> np.ndarray:
    h, w = a.shape[0] // n * n, a.shape[1] // n * n
    return a[:h, :w].reshape(h // n, n, w // n, n).mean(axis=(1, 3))


def score(render: np.ndarray, ref: np.ndarray, se: np.ndarray) -> dict:
    mask = se.max(-1) > 1e-3
    y, yr, ys = render @ LUM, ref @ LUM, se @ LUM
    by, byr, bm = blocks(y * mask), blocks(yr * mask), blocks(mask.astype(float)) > 0.5
    return {
        "full": float(np.abs(render - ref).mean() / ref.mean()),
        "cloud": float(np.abs(y - yr)[mask].sum() / yr[mask].sum()),
        "block8": float(np.abs(by - byr)[bm].sum() / byr[bm].sum()),
        "ratio_rgb": [float(render[..., c][mask].sum() / ref[..., c][mask].sum()) for c in range(3)],
        "floor": float(np.sqrt(2 / np.pi) * ys[mask].sum() / yr[mask].sum()),
        "cloud_fraction": float(mask.mean()),
    }


def main(argv: list[str]) -> int:
    runs = list(zip(argv[::2], argv[1::2])) or [(OUT / "net", "network"), (OUT / "single", "single")]
    pairs = json.loads((CLOUDS / "scene" / "out" / "shots.json").read_text())["pairs"]
    result = {}
    for d, tag in runs:
        rows = {}
        for split in ("reference", "heldback"):
            for pid in (p["id"] for p in pairs[split]):
                p = Path(d) / f"{pid}_{tag}.npy"
                if not p.is_file():
                    print(f"missing {p}", file=sys.stderr)
                    continue
                rows[pid] = {"split": split, **score(np.load(p), np.load(REF / f"{pid}.npy"), np.load(REF / f"{pid}_se.npy"))}
        result[tag] = rows
        print(f"\n{tag} ({d})")
        print(f"{'pair':20s} {'split':9s} {'full':>6s} {'cloud':>6s} {'block8':>6s} {'floor':>6s}   ratio R,G,B")
        for pid, r in rows.items():
            print(f"{pid:20s} {r['split']:9s} {r['full']:6.3f} {r['cloud']:6.3f} {r['block8']:6.3f} {r['floor']:6.3f}   "
                  + ", ".join(f"{x:.2f}" for x in r["ratio_rgb"]))
        for split in ("reference", "heldback"):
            v = [r for r in rows.values() if r["split"] == split]
            if v:
                print(f"{'mean ' + split:30s} {np.mean([r['full'] for r in v]):6.3f} {np.mean([r['cloud'] for r in v]):6.3f} "
                      f"{np.mean([r['block8'] for r in v]):6.3f} {np.mean([r['floor'] for r in v]):6.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scores.json").write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
