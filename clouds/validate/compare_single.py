"""SPEC 6.1: renderer exact single scattering vs Mitsuba single-sun mode, on linear images.

    uv run python validate/compare_single.py [--render] [--frames 64] [pairs ...]

Inputs:
  renderer  validate/out/exact/<pair>_exact_single.npy       (--mode exact-single: sun single scattering + T*sky)
            validate/out/exact_nobg/<pair>_exact_single.npy  (--mode exact-single --no-background: sun part only)
  Mitsuba   reference/out/single/<pair>.npy, <pair>_sun.npy, <pair>_bg.npy, <pair>_se.npy
--render re-renders the renderer side first.

Masks (from the renderer, which is noise-free in T): camera-ray transmittance T = (total - sun part) / sky(v).
  cloud      = T < 0.995  (the ray meets cloud)
  background = T > 0.99999 and sun part == 0 (clear sky)

Writes validate/out/single/<pair>_compare.png (rows: total | sun part; columns: renderer | Mitsuba | signed relative
difference, smoothed 4x4), validate/out/single/summary.json and prints a table.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from common import CLOUDS, OUT, REF_OUT, block, diverging, exposure, labelled, lum, render, row, shots, stack, tonemap

sys.path.insert(0, str(CLOUDS / "reference"))
from render_ref import camera_rays, load_sky  # noqa: E402

PAIRS = ["front_noon", "side_noon", "side_sunset", "backlit_golden", "below_noon", "distant_sunset"]
K2PI = np.sqrt(2 / np.pi)  # E|N(0, s)| = s * sqrt(2/pi)


def l1w(r, m, mask, se=None, k=1):
    """Luminance-weighted L1: sum |Y_r - Y_m| / sum Y_m over `mask`, after k x k block averaging (blocks are kept
    when fully inside the mask). With se, also the expected value of that ratio from Mitsuba noise alone."""
    Yr, Ym = lum(r), lum(m)
    if k > 1:
        Yr, Ym = block(Yr, k), block(Ym, k)
        mask = block(mask.astype(float), k) > 0.999
    d = np.abs(Yr - Ym)[mask].sum() / max(Ym[mask].sum(), 1e-12)
    if se is None:
        return float(d)
    s = lum(se) if k == 1 else np.sqrt(block(lum(se) ** 2, k)) / k
    return float(d), float(K2PI * s[mask].sum() / max(Ym[mask].sum(), 1e-12))


def mean_rel(r, m, mask, k=1, floor=0.02):
    """Plain mean over pixels (or k x k blocks) of |Y_r - Y_m| / max(Y_m, floor * mean Y_m over the mask).
    The floor keeps black pixels (fully shadowed and opaque, where single scattering is ~0 in both images) from
    turning the mean into noise / 0."""
    Yr, Ym = lum(r), lum(m)
    if k > 1:
        Yr, Ym = block(Yr, k), block(Ym, k)
        mask = block(mask.astype(float), k) > 0.999
    den = np.maximum(Ym[mask], floor * Ym[mask].mean())
    return float(np.mean(np.abs(Yr - Ym)[mask] / den))


def compare(pid):
    S = shots()
    shot, preset = pid.split("_", 1)
    r_tot = np.load(OUT / "exact" / f"{pid}_exact_single.npy").astype(np.float64)
    r_sun = np.load(OUT / "exact_nobg" / f"{pid}_exact_single.npy").astype(np.float64)
    single = REF_OUT / "single"
    m_tot = np.load(single / f"{pid}.npy").astype(np.float64)
    m_sun = np.load(single / f"{pid}_sun.npy").astype(np.float64)
    m_bg = np.load(single / f"{pid}_bg.npy").astype(np.float64)
    se = np.load(single / f"{pid}_se.npy").astype(np.float64)
    meta = json.loads((single / f"{pid}.json").read_text())
    H, W, _ = r_tot.shape

    from render_ref import sky_lookup
    sky = sky_lookup(load_sky(S["presets"][preset]), camera_rays(S["shots"][shot], W, H))
    r_bg = r_tot - r_sun
    T = lum(r_bg) / np.maximum(lum(sky), 1e-12)
    # cloud: the ray meets medium. (T alone misfires on the horizon row, where the renderer's box-filtered sky and
    # the pixel-centre lookup differ; so require sun light or clearly reduced T.)
    cloud = (lum(r_sun) > 0) | (T < 0.5)
    bgmask = (T > 0.99999) & (lum(r_sun) == 0)
    # The renderer stops at T < 0.002 and then adds T * sky (SPEC 4.2-4.3); Mitsuba keeps going, so opaque pixels
    # differ by up to 0.002 * sky. "noterm" removes that term from the renderer side to show the rest.
    term = T < 0.0021
    r_tot_nt = r_tot - np.where(term[..., None], r_bg, 0.0)

    out = {"pair": pid, "mitsuba_spp_sun": meta["parts"]["sun_single"]["spp"],
           "cloud_px": int(cloud.sum()), "background_px": int(bgmask.sum())}
    # sun single-scattering part (no background): the physics check
    rs = r_sun[cloud].sum(0); ms = m_sun[cloud].sum(0)
    out["sun_ratio_of_sums_rgb"] = [float(a / b) for a, b in zip(rs, ms)]
    out["sun_ratio_of_sums_lum"] = float(lum(rs[None])[0] / lum(ms[None])[0])
    for k in (1, 4, 8):
        d, n = l1w(r_sun, m_sun, cloud, se, k)
        out[f"sun_l1w_k{k}"] = d
        out[f"sun_l1w_k{k}_noise"] = n
    for k in (1, 4, 8):
        out[f"sun_meanrel_k{k}"] = mean_rel(r_sun, m_sun, cloud, k)
    # total image (sun + T*sky): what the comparison of linear images sees
    for k in (1, 4):
        d, n = l1w(r_tot, m_tot, cloud, se, k)
        out[f"tot_l1w_cloud_k{k}"] = d
        out[f"tot_l1w_cloud_k{k}_noise"] = n
        out[f"tot_meanrel_cloud_k{k}"] = mean_rel(r_tot, m_tot, cloud, k)
    for k in (1, 4):
        out[f"tot_noterm_l1w_cloud_k{k}"] = l1w(r_tot_nt, m_tot, cloud, None, k)
        out[f"tot_noterm_meanrel_cloud_k{k}"] = mean_rel(r_tot_nt, m_tot, cloud, k)
    out["terminated_px"] = int(term.sum())
    out["tot_ratio_of_sums_cloud_rgb"] = [float(a / b) for a, b in zip(r_tot[cloud].sum(0), m_tot[cloud].sum(0))]
    # camera-ray transmittance part T*sky on the cloud pixels
    out["bg_part_ratio_cloud"] = float(lum(r_bg[cloud].sum(0)[None])[0] / lum(m_bg[cloud].sum(0)[None])[0])
    out["bg_part_l1w_cloud_k4"] = l1w(r_bg, m_bg, cloud, None, 4)
    # clear-sky pixels
    out["background_meanrel"] = mean_rel(r_tot, m_tot, bgmask)
    out["background_maxrel"] = float(np.max(np.abs(lum(r_tot) - lum(m_tot))[bgmask] / lum(m_tot)[bgmask]))
    out["background_ratio_rgb"] = [float(a / b) for a, b in zip(r_tot[bgmask].sum(0), m_tot[bgmask].sum(0))]

    # diagnostic variants from diag_cutoffs.py (patched copy of the shader: no early stop / no density threshold)
    variants = {}
    for tag in ("term", "term_nodmin"):
        f = OUT / tag / f"{pid}_exact_single.npy"
        if not f.exists():
            continue
        rv = np.load(f).astype(np.float64)
        variants[tag] = rv
        cv = cloud | (lum(rv) > 0)
        out[f"{tag}_sun_ratio_of_sums_lum"] = float(lum(rv).sum() / lum(m_sun).sum())
        out[f"{tag}_sun_l1w_k8"] = l1w(rv, m_sun, cv, None, 8)
        for k in (1, 4, 8):
            out[f"{tag}_sun_meanrel_k{k}"] = mean_rel(rv, m_sun, cv, k)
        # what the SPEC cut-offs lose, relative to the full-mode Mitsuba reference (all orders of scattering)
        full = REF_OUT / f"{pid}.npy"
        if full.exists():
            Yf = lum(np.load(full).astype(np.float64))
            lost = lum(rv) - lum(r_sun)
            out[f"{tag}_lost_vs_full_sum"] = float(lost[cv].sum() / Yf[cv].sum())
            out[f"{tag}_lost_vs_full_max8x8"] = float((block(lost, 8) / np.maximum(block(Yf, 8), 1e-9)).max())

    # images
    ex = exposure(pid)
    # sun part shown at an exposure that puts its 99th percentile near 1 (it is dark without skylight)
    p99 = np.percentile(lum(m_sun)[cloud], 99) if cloud.any() else 1.0
    ex_sun = 1.0 / max(p99, 1e-6)

    def reldiff(r, m, scale):
        Yr, Ym = lum(r), lum(m)
        k = 4
        br, bm = block(Yr, k), block(Ym, k)
        rel = (br - bm) / np.maximum(bm, 1e-3 * bm.max())
        rel = np.repeat(np.repeat(rel, k, 0), k, 1)
        return diverging(rel, cloud, scale)

    top = row([labelled(tonemap(r_tot, ex), f"{pid}  renderer exact-single  (exp {ex:g})"),
               labelled(tonemap(m_tot, ex), f"Mitsuba single, {meta['spp']} spp"),
               labelled(reldiff(r_tot, m_tot, 0.10), "(renderer - Mitsuba)/Mitsuba, 4x4, +-10%")])
    bot = row([labelled(tonemap(r_sun, ex_sun), f"sun part only (exp {ex_sun:.2g})"),
               labelled(tonemap(m_sun, ex_sun), "Mitsuba sun part"),
               labelled(reldiff(r_sun, m_sun, 0.20), "sun part rel. diff, 4x4, +-20%")],
              footer=(f"sun: ratio of sums {out['sun_ratio_of_sums_lum']:.4f}, L1w {out['sun_l1w_k1']:.3f} "
                      f"(noise {out['sun_l1w_k1_noise']:.3f}), 8x8 {out['sun_l1w_k8']:.3f} | total cloud L1w "
                      f"{out['tot_l1w_cloud_k1']:.3f} | sky px rel {out['background_meanrel']:.4f}"))
    rows = [top, bot]
    if "term_nodmin" in variants:
        rv = variants["term_nodmin"]
        rows.append(row([labelled(tonemap(rv, ex_sun), "diagnostic: no T-stop, no d threshold (patched copy)"),
                         labelled(tonemap(m_sun, ex_sun), "Mitsuba sun part"),
                         labelled(reldiff(rv, m_sun, 0.20), "diagnostic rel. diff, 4x4, +-20%")],
                        footer=(f"without the SPEC cut-offs: ratio {out['term_nodmin_sun_ratio_of_sums_lum']:.4f}, "
                                f"mean rel 4x4 {out['term_nodmin_sun_meanrel_k4']:.4f} (SPEC renderer "
                                f"{out['sun_meanrel_k4']:.4f}); cut-offs lose {out.get('term_nodmin_lost_vs_full_sum', 0):.4f}"
                                f" of the full-mode image (max 8x8 block {out.get('term_nodmin_lost_vs_full_max8x8', 0):.3f})")))
    (OUT / "single").mkdir(parents=True, exist_ok=True)
    stack(rows).save(OUT / "single" / f"{pid}_compare.png")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="*", default=PAIRS)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--frames", type=int, default=64)
    a = ap.parse_args()
    if a.render:
        render(["--mode", "exact-single", "--frames", str(a.frames), "--pair", *a.pairs, "--out", str(OUT / "exact") + "/"])
        render(["--mode", "exact-single", "--no-background", "--frames", str(a.frames), "--pair", *a.pairs,
                "--out", str(OUT / "exact_nobg") + "/"])
    res = [compare(p) for p in a.pairs]
    (OUT / "single" / "summary.json").write_text(json.dumps(res, indent=1))
    print(f"{'pair':16s} {'sun ratio':>9s} {'sun L1w':>8s} {'(noise)':>8s} {'sun 4x4':>8s} {'sun 8x8':>8s} {'(noise)':>8s}"
          f" {'sunrel':>7s} {'sunrel4':>7s} {'sunrel8':>7s}"
          f" {'tot L1w':>8s} {'tot rel':>8s} {'tot rel4':>8s} {'nt rel':>7s} {'nt rel4':>7s} {'T*sky':>7s} {'sky rel':>8s}")
    for o in res:
        print(f"{o['pair']:16s} {o['sun_ratio_of_sums_lum']:9.4f} {o['sun_l1w_k1']:8.4f} {o['sun_l1w_k1_noise']:8.4f}"
              f" {o['sun_l1w_k4']:8.4f} {o['sun_l1w_k8']:8.4f} {o['sun_l1w_k8_noise']:8.4f}"
              f" {o['sun_meanrel_k1']:7.4f} {o['sun_meanrel_k4']:7.4f} {o['sun_meanrel_k8']:7.4f}"
              f" {o['tot_l1w_cloud_k1']:8.4f} {o['tot_meanrel_cloud_k1']:8.4f} {o['tot_meanrel_cloud_k4']:8.4f}"
              f" {o['tot_noterm_meanrel_cloud_k1']:7.4f} {o['tot_noterm_meanrel_cloud_k4']:7.4f}"
              f" {o['bg_part_ratio_cloud']:7.4f} {o['background_meanrel']:8.5f}")
    print("\ndiagnostic variants (diag_cutoffs.py): sun ratio of sums, mean rel k1/k4/k8, lost vs full (sum, max 8x8)")
    for o in res:
        for tag in ("term", "term_nodmin"):
            if f"{tag}_sun_ratio_of_sums_lum" in o:
                print(f"{o['pair']:16s} {tag:12s} {o[tag + '_sun_ratio_of_sums_lum']:.4f}  "
                      f"{o[tag + '_sun_meanrel_k1']:.4f} {o[tag + '_sun_meanrel_k4']:.4f} {o[tag + '_sun_meanrel_k8']:.4f}"
                      f"  lost {o.get(tag + '_lost_vs_full_sum', float('nan')):.4f} "
                      f"{o.get(tag + '_lost_vs_full_max8x8', float('nan')):.4f}")


if __name__ == "__main__":
    main()
