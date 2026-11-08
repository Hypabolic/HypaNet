"""Measure linear radiance in reference and render .npy files.

./python3 work/measure.py grid PAIR [RENDER.npy]      block means on a coarse grid (ref, and render/ref ratio)
./python3 work/measure.py box PAIR x0 y0 x1 y1 [RENDER.npy]   mean RGB in a box
./python3 work/measure.py cloudstats PAIR [RENDER.npy]  stats restricted to cloud pixels (where ref differs from sky)
"""
import sys
import numpy as np

REF = "references/"


def load(pair):
    return np.load(REF + pair + ".npy")


def fmt(v):
    return "(" + ", ".join(f"{x:.4f}" for x in v) + ")"


def cloud_mask(ref):
    # the background sky is smooth; the cloud differs from a heavily blurred version... simpler: use the
    # reference compare image idea: pixels whose value differs from the sky model are unknown, so we use
    # brightness relative to a horizontal-line median (sky varies slowly horizontally).
    med = np.median(ref, axis=1, keepdims=True)
    return np.abs(ref - med).max(axis=2) > 0.04 * ref.max()


if __name__ == "__main__":
    cmd, pair = sys.argv[1], sys.argv[2]
    ref = load(pair)
    rnd = None
    rest = sys.argv[3:]
    if rest and rest[-1].endswith(".npy"):
        rnd = np.load(rest[-1]); rest = rest[:-1]
    if cmd == "grid":
        H, W, _ = ref.shape
        nb = 10
        for j in range(nb):
            row = []
            for i in range(nb):
                blk = ref[j * H // nb:(j + 1) * H // nb, i * W // nb:(i + 1) * W // nb]
                m = blk.reshape(-1, 3).mean(0)
                if rnd is None:
                    row.append(f"{m[0]:.2f}/{m[1]:.2f}/{m[2]:.2f}")
                else:
                    r = rnd[j * H // nb:(j + 1) * H // nb, i * W // nb:(i + 1) * W // nb].reshape(-1, 3).mean(0)
                    row.append(f"{m[1]:.2f}:{r[1] / max(m[1], 1e-6):.2f}")
            print(" ".join(f"{s:>12s}" for s in row))
    elif cmd == "box":
        x0, y0, x1, y1 = map(int, rest[:4])
        m = ref[y0:y1, x0:x1].reshape(-1, 3).mean(0)
        print("ref ", fmt(m), " ratio G/R %.3f B/R %.3f" % (m[1] / m[0], m[2] / m[0]))
        if rnd is not None:
            r = rnd[y0:y1, x0:x1].reshape(-1, 3).mean(0)
            print("rnd ", fmt(r), " ratio G/R %.3f B/R %.3f" % (r[1] / r[0], r[2] / r[0]), " rnd/ref", fmt(r / m))
    elif cmd == "cloudstats":
        m = cloud_mask(ref)
        print("cloud pixels:", m.sum(), "mean ref", fmt(ref[m].mean(0)), "p95", fmt(np.percentile(ref[m], 95, axis=0)),
              "max", fmt(ref[m].max(0)))
        if rnd is not None:
            print("mean rnd", fmt(rnd[m].mean(0)), "p95", fmt(np.percentile(rnd[m], 95, axis=0)))
            d = np.abs(rnd[m] - ref[m]).mean() / ref[m].mean()
            print("cloud-only mean|rnd-ref|/mean(ref) = %.3f" % d)
