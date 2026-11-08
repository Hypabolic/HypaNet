# NETWORK.md: a hand-written light-transport network for the cumulus renderer

Rebuild: `./python3 work/build_network.py` (writes `work/network.json`; reads no image).

Every weight in `network.json` is a physics formula (phase-function prefactors, extinction × segment length,
logs of chosen constants) or a literal chosen by reasoning in `build_network.py`. Nothing was fitted. The
constants come from (a) the medium's physics, (b) my own Monte Carlo simulations of this medium in slabs,
spheres and a puck (`work/mc.py`, `work/mc_puck.py`; results in `work/mc_out/`, tables printed by
`work/mc_show.py`, `work/mc_sphere_feat.py`, `work/model_eval.py`), and (c) looking at my renders beside the
references, at linear block averages and colour ratios (`work/measure.py`) and at `./inspect` traces.
41 probes, 5 layers (29, 26, 23, 13, 9 units), 3057 parameters; a 640×400, 16-frame render takes ≈3 s.

## 1. What the network computes

The renderer needs, at every sample point x, `S = a_sun·E_sun + a_sky·L_sky + a_gnd·L_gnd`, the phase-weighted
radiance arriving at x. The medium is grey, so every `a` is the same in R, G and B; all colour comes from the
renderer's multiplication by `E_sun`, `L_sky`, `L_gnd` (the sunset colour change needs no colour in the network).

Per unit irradiance the network models (σ = 0.04 d, τ = optical depths from the probes, μ = v·s):

```
a_sun = T_sun·p(μ)                                   direct sun, single scattering (exact form)
      + depth^0.7 e^{-0.3 τ_f} [0.033 + 0.21 HG(μ;-0.3)]   "fresh" light, orders 1-5, broad part
      + 0.52 HG(μ;0.8) τ_f e^{-0.55 τ_f (1 + 0.6 leak)}    "fresh" forward glow (silver lining, light seen from below)
      + (Φ_lit + Φ_deep)/(4π) · B(v)                 multiply scattered (orders ≥ 6) light, a fluence over 4π
a_sky = 0.45 e^{+0.62 v_y} open·skyside·e^{-1.3 τ_up}  + (Φ_sky,lit + Φ_sky,deep)/(4π) · e^{0.5 G_v}
a_gnd = 0.45 e^{-0.62 v_y} open·gndside·e^{-1.3 τ_dn}  + (Φ_gnd,lit + Φ_gnd,deep)/(4π) · e^{0.5 G_v}

Φ_lit     = 1.3 E600^0.6 e^{2(G_s-1)} e^{-0.15 τ_c}                 light injected locally on the sun-lit cap
Φ_deep    = depth E600^0.3 g_c,   g_c = 1 / (1 + γ ((τ_c+0.5)/11.5)^2.77),  γ = 1 + 0.6 (1 - s_y)
B(v)      = e^{0.5 min(1,2 relu(G_v)) min(1,2|G_s|) + 0.11 μ}         outflow seen by the camera, flow along -s
Φ_sky,lit = 5.45 open·skyside·E600^0.6,   Φ_sky,deep = 10 depth E600^0.6 g_u,  g_u = 1/(1 + ((τ_up+0.5)/11.5)^2.77)
Φ_gnd,lit = 10.8 open·gndside·E600^0.6,   Φ_gnd,deep = 8.6 depth E600^0.6 g_d  (same with τ_dn)
leak      = |G_up| (1 - s_y) (1 - |G_s|)
```

Features: `τ_f` fine optical depth toward the sun (10 probes, reach 1150 m), `τ_c` diffusion-scale optical depth
toward the sun (6 probes, blurs 150–700 m, reach 1400 m), `τ_up`, `τ_dn` straight up / down (6 probes each),
`E150`, `E300`, `E600` the density box-filtered over 150/300/600 m cubes at x (E300 is carried but unused),
`G_s` the 200 m-scale density gradient along s (≈ cosine of incidence on a lit face, −1 on the far side),
`G_v` the same along the view ray (+1 when looking into the cloud from outside), `G_up` the 300 m-scale vertical
gradient (+1 on tops, −1 on the base), `depth = relu(1.5 E150 + 0.25)`, `open = 1 − relu(2 E600 − 1)`,
`skyside = min(1, 0.7 + 0.7 G_up)`, `gndside = relu(0.5 − 0.5 G_up)`, `s_y` = sun_up, `v_y` = view_up.

## 2. Probes (41)

| # | probe | role |
|---|---|---|
| 0–9 | light frame, dir +s; segments [0,10],[10,25],[25,50],[50,90],[90,150],[150,240],[240,370],[370,550],[550,800],[800,1150] m, probe at the midpoint, blur = 0.55 × length (≥ 7 m) | `τ_f = Σ σ·len·d` by the midpoint rule. Narrow blur so that a ray skimming past a billow is not polluted by the air beside it; long reach so the glow is not switched on where the beam has really gone through the whole base |
| 10–15 | light frame, dir +s; segments [0,60],[60,160],[160,320],[320,560],[560,900],[900,1400]; blurs 150,200,280,380,500,700 | `τ_c`: the sun exposure averaged over the neighbourhood (a few transport mean free paths ≈ 150–300 m) that feeds x by diffusion |
| 16,17,18 | world frame, dist 0, blur 150 / 300 / 600 | enclosure `E150`, `E300`, `E600` |
| 19,20 | light frame, dir −s / +s, dist 50, blur 200 | `G_s = 2(ρ(x−50s) − ρ(x+50s))` |
| 21,22 | view frame, dir +v / −v, dist 50, blur 200 | `G_v = 2(ρ(x+50v) − ρ(x−50v))` |
| 23–34 | world up / down, segments [0,15],[15,40],[40,90],[90,180],[180,350],[350,650], blurs 15,25,50,90,170,300 | `τ_up`, `τ_dn` |
| 35,36 | world down / up, dist 75, blur 300 | `G_up = 2(ρ(x−75ŷ) − ρ(x+75ŷ))` |
| 37 | cos_view_sun | μ |
| 38 | view_up | v_y |
| 39 | sun_up | s_y (only inside smooth physical factors: γ, leak) |
| 40 | const 1 | unused by any weight (kept so biases are explicit in the file) |

## 3. Layers, unit by unit

Layer 1 (29 units, from the probes; `GRAD = blur/(2·dist) = 2` turns a probe difference into a gradient ≈ cos)
- `tau_f`, `tau_c`, `tau_up`, `tau_dn` (linear): σ_max × segment length × probe, summed.
- `log_tf = log(τ_f + 0.01)`, `log_tc = log(τ_c + 0.5)`, `log_tu`, `log_td` (log): for the powers formed later.
- `lg9 = log(1.81 − 1.8μ)`, `lgm3 = log(1.09 + 0.6μ)`, `lg8 = log(1.64 − 1.6μ)` (log): HG denominators for
  g = 0.9, −0.3, 0.8; later `HG(μ;g) = exp(log((1−g²)/4π) − 1.5·lg)`.
- `mu`, `vy` (linear).
- `Gs` (linear), `aGs = |G_s|` (abs), `rGs_p = relu(2G_s − 1)`, `rGs_m = relu(−2G_s − 1)`: pieces of
  `h = min(1, 2|G_s|) = 2 aGs − rGs_p − rGs_m` (1 on lit and dark faces, 0 at the terminator).
- `Gv` (linear), `rGv = relu(G_v)`, `rGv2 = relu(2G_v − 1)`: pieces of `min(1, 2 relu(G_v)) = 2 rGv − rGv2`.
- `logE600 = log(E600 + 10⁻³)`.
- `lgr = log(1 + 0.6(1 − s_y))` (the grazing factor γ), `log_1msy = log(1 − s_y)`, `aGup = |G_up|` (abs).
- `depth = relu(1.5 E150 + 0.25)`: boundary-sink factor (fluence rises inward from a free surface).
- `clos = relu(2 E600 − 1)`: 0 at any surface, 1 when > 300 m inside.
- `sky_a = relu(0.7 + 0.7 G_up)`, `sky_b = relu(0.7 G_up − 0.3)`: `skyside = sky_a − sky_b` = 1 on tops, 0.7 on
  vertical flanks, 0 on the base. `gndside = relu(0.5 − 0.5 G_up)`: 1 on the base, 0.5 on flanks, 0 on tops.

Layer 2 (26 units)
- `dir9 = e^{-τ_f} HG(μ;0.9)`, `dirm3 = e^{-τ_f} HG(μ;−0.3)`: exact single scattering, combined 0.8/0.2 in the output.
- `log_aGup = log(|G_up| + ε)`, `log_1mGs = log(1 − |G_s| + ε)`, `log_rGv = log(min(1,2relu G_v) + ε)`,
  `log_h = log(min(1,2|G_s|) + ε)`, `log_depth`, `log_open = log(1 − clos + ε)`, `log_skyside`, `log_gndside`:
  logs of the factors multiplied inside later exp units.
- `pw_c = γ ((τ_c+0.5)/11.5)^2.77` (exp of 2.77·log_tc + lgr − 2.77·log 11.5), `pw_u`, `pw_d` (without γ).
- pass-through (linear ×1): `tau_f, log_tf, lg8, log_1msy, tau_c, tau_up, tau_dn, lgm3, mu, vy, Gs, Gv, logE600`.

Layer 3 (23 units)
- `bvp = exp(log_rGv + log_h) = min(1,2relu G_v)·min(1,2|G_s|)`: the outflow-boost argument.
- `tleak = exp(log_tf + log_aGup + log_1msy + log_1mGs) = τ_f·|G_up|(1−s_y)(1−|G_s|)`.
- `lg1p_c = log(1 + pw_c)`, `lg1p_u`, `lg1p_d`: so that `exp(−lg1p) = g`, the diffusion decay.
- pass-throughs of everything needed in layer 4.

Layer 4 (13 units; every exp unit is a product of powers of the features)
- `fr_iso = 0.033 e^{-0.3τ_f} depth^0.7`, `fr_back = 0.21 e^{-0.3τ_f} HG(μ;−0.3) depth^0.7`
- `lobe = 0.52 HG(μ;0.8) τ_f e^{-0.55 τ_f − 0.33 tleak}`
- `sun_lit = 1.3 E600^0.6 e^{2(G_s−1)} e^{-0.15τ_c} e^{0.5 bvp + 0.11μ}`
- `sun_deep = depth E600^0.3 e^{-lg1p_c} e^{0.5 bvp + 0.11μ}`
- `sky_single = 0.45 e^{0.62 v_y} open·skyside·e^{-1.3τ_up}`, `gnd_single = 0.45 e^{-0.62 v_y} open·gndside·e^{-1.3τ_dn}`
- `sky_lit = 5.45 open·skyside·E600^0.6 e^{0.5G_v}`, `sky_deep = 10 depth E600^0.6 e^{-lg1p_u} e^{0.5G_v}`
- `gnd_lit = 10.8 open·gndside·E600^0.6 e^{0.5G_v}`, `gnd_deep = 8.6 depth E600^0.6 e^{-lg1p_d} e^{0.5G_v}`
- pass-throughs of `dir9, dirm3`.

Layer 5 (output, 9 linear units): `a_sun = 0.8 dir9 + 0.2 dirm3 + fr_iso + fr_back + lobe + (sun_lit + sun_deep)/4π`,
`a_sky = sky_single + (sky_lit + sky_deep)/4π`, `a_gnd = gnd_single + (gnd_lit + gnd_deep)/4π`, each repeated for R, G, B.

## 4. The physical reasoning behind each term

**Single scattering.** Exact: `T_sun·p(v·s)` with `T_sun = exp(−τ_f)`.

**Fresh light (orders 1–5).** Slab MC (τ = 20, normal and 60° incidence): near the lit surface the low orders are
≈ 0.065 per unit irradiance for a camera on the sun side, decaying ≈ e^{-0.3τ}, keeping the phase function's
backscatter bump (0.034 at μ=−0.5 vs 0.065 at μ=−1), independent of the incidence angle at a given μ. They need
optical thickness to build up, hence `depth^0.7` (a thin wisp has none). The forward glow along the beam
(all orders still within ~20° of it) measured at μ = 1 is 0.43, 1.17, 1.25, 0.83, 0.55, 0.43, 0.16, 0.033 at
τ = 0.2, 1, 1.8, 3.4, 5, 6.6, 9.8, 13; `1.86 τ e^{-0.55τ}` gives 0.33, 1.1, 1.27, 0.99, 0.61, 0.44, 0.16, 0.019.
Its angular width matches HG(0.8) (ratios 1 : 0.08 : 0.01 at μ = 1, 0.866, 0.5). The glow is the silver lining
(the backlit cameras are 5–16° from the sun: HG(0.96; 0.8) = 0.86) and the bright tower flanks seen from below at
noon (μ ≈ 0.95). A beam running *along* a thin layer near a boundary (the flat base under a low sun) loses its
glow sideways: `leak = |G_up|(1−s_y)(1−|G_s|)` is ≈ 0.7 on the base's underside at sunset and ≈ 0 inside a tower
or on a lit face, and multiplies the decay rate by 1 + 0.6 leak.

**Multiply scattered light as a fluence.** Deep inside, S = Φ/4π − (3g/4π) J·v (P1 diffusion), checked against the
MC at τ = 13 (0.110 / 0.175 vs predicted 0.109 / 0.170). The sphere MC (τ_R = 4, 10, 20, 40):
- lit cap: Φ(≥6) ≈ 0.25, 0.7, 1.2, 1.6 per unit E for τ_R = 4, 10, 20, 40 (slab ≈ 1.3–1.6): grows with the body
  size → `E600^0.6`; follows a cosine law (60° slab: 0.59 × normal) → `e^{2(G_s−1)}`. `G_s` at 200 m blur is
  ≈ cos(incidence) at a sharp face and ≈ 0 for bodies much smaller than 200 m, which reproduces the MC finding
  that a τ_R = 4 sphere is lit almost uniformly all round.
- dark side of a sphere: Φ ≈ 2.6/τ_s at 25 m depth for all sizes τ_R ≥ 10: a power law, not an exponential
  (conservative diffusion obeys Laplace's equation); the fluence rises inward from a non-lit surface as
  (δ+z0)/z0 with z0 ≈ 55 m → the `depth` factor, made gentler because the cloud's boundary is fuzzy and a
  100 m billow (τ ≈ 4) is lit nearly uniformly.
- the terminator and the cloud's shadow flanks sit at Φ ≈ 0.6 (τ ≈ 5–6).
- a wide slab of τ ≈ 14 lit from above transmits Φ ≈ 0.5 to its underside (puck MC at 55°); lit from the side at
  8°, its underside carries 0.37 at 350 m from the lit edge and 0.1 at 650 m (`work/mc_puck.py`).
The real cloud (measured in the references: lit face ≈ 0.8, shadow flank ≈ 0.57, base underside ≈ 0.6 at
τ_c ≈ 10, far side of the big tower ≈ 0.1 at τ_c ≈ 27) is brighter at moderate τ_c than a convex sphere because
its lumps reflect light into each other and its base is slab-like. Hence `g_c = 1/(1 + γ((τ_c+0.5)/11.5)^2.77)`:
noon (γ = 1.11) 0.57 at τ_c = 10, 0.13 at 22; afternoon (γ = 1.35) 0.2 at 17, 0.065 at 27; golden (1.52) 0.43 at
11; sunset (1.58) 0.63 at 8. γ (the "grazing" factor) encodes the puck result that light travelling along the
flat base under a low sun leaks through both faces; it is a smooth function of s_y, not a preset switch.
`τ_c` is measured with blurs of 150–700 m because diffusion averages the sun exposure over the neighbourhood (a
crevice between two lit billows is nearly as bright as the billows).

**View dependence B(v).** At any free surface the diffuse light flows outward, so a camera outside looking in sees
more than Φ/4π: ×1.6 on the lit cap (MC), ×1.5 already at 60° incidence, ×2.0 on the dark bottom, but ×1.0 at the
terminator (there the flow is tangential, along −s). `min(1, 2|G_s|)` is 1 on lit and dark faces and 0 at the
terminator; `min(1, 2 relu(G_v))` is 1 whenever the camera looks into the cloud within ~60° of the normal (the MC
boost is flat out to 60° and gone at 90°). `e^{0.11μ}` adds the weak flow-along-s asymmetry of the interior.

**Sky and ground.** Extended sources: the sky's irradiance on an open surface is ≈ 2.5·L_sky (E_sky/(πL_sky) =
0.6–0.8; the horizon band is 4× the mean in R, so vertical flanks get 0.6–0.85 of a top's irradiance: `skyside`
= 1 / 0.7 / 0 on top / flank / base); the ground gives π·L_gnd on the base and π/2 on flanks (`gndside` = 1 / 0.5 /
0). The slab MC says the fluence just inside a surface lit by an extended source is ≈ 3× the irradiance. The
colour decomposition of the reference's shadow flank at noon (B/R = 0.96, between the sun's 0.77 and the sky's
3.2) requires a_sky ≈ 0.57 there, so the sky light must be at full strength. `open` says whether the point is at
a surface; the deep terms carry the light diffused through the body (a base under 250 m of cloud still gets
≈ 0.25 of the top's sky light), scaling as `E600^0.6` because a thin wisp has no multiple-scattering build-up.
The single-scattering part uses the hemispheric transmittance ≈ e^{-1.3τ} (2E₃(τ)) and the phase integral over a
hemisphere (0.84 forward, 0.16 back, written as 0.45·e^{±0.62 v_y}).

## 5. Revisions after looking at renders (in order)

1. First full network: right structure, 20–50 % too dark, blotchy. Inspect: the deep term used fine-scale optical
   depth (crevices as dark as the far side); sky and ground 5× too weak (shadow-flank colour decomposition).
   Fix: `τ_c` at the diffusion scale, extended-source terms at the slab-MC level, flanks see the sky.
2. Base underside 0.4×, backlit body blotchy. Puck MC showed slab transmission (0.5 at τ = 14) that the sphere
   law (0.2) misses. Fix: a decay nearly flat to τ ≈ 8, `open` on E600 (no penalty for concave gaps), gentler depth.
3. Far side 1.4× too bright, sunlit flanks seen from below 0.7×, silver lining 0.8×. Inspect: the glow's
   e^{-0.3τ} tail gave half the sun light at the backlit bottom (τ_f = 13) and too little at τ_f = 6, and my
   build-up factor had halved its peak. Fix: glow = 0.52 HG(0.8) τ e^{-0.55τ} (the MC profile), steeper decay,
   sky/ground light shifted from the surface term to the diffused term (bluer bases, as the golden-hour colour
   decomposition requires).
4. `below_golden` too red: the excess had the sun's colour; the puck lit from the side at 8° explains it (light
   along the base leaks). Fix: the grazing factor γ. Thin wisps too bright: no build-up → `E600^0.6`, `depth^0.7`.
5. Noon shadow flank 15 % bright with a ground-like colour: a vertical flank sees half the ground, 0.7 of the sky.
   Fix: `gndside`, `skyside`.
6. Lower front of the base at noon 0.8×: at the base's lower edge the 300 m gradients are dominated by the vertical
   density change, so `G_s` read −0.5 and `G_v` 0.2 on a lit face. Fix: gradients at 200 m, `min(1, 2relu(G_v))`,
   and the boost tied to |G_s| (how strongly lit/dark the face is) instead of |μ|.
7. Cellular texture on the backlit body: two neighbouring pixels with τ_c 16.8 and 21 differed 2.6× under an
   exponential decay while the sphere MC shows a gentle power law there. Fix: the rational decay
   `1/(1+γ(τ/11.5)^2.77)`, constants read off the reference-derived points (lit face, flank, base, far side).
8. Orange glow spreading over the base's underside at sunset: the forward glow at τ_f ≈ 8 under grazing light,
   which lateral leakage must suppress; and the fine ray's 550 m reach truncated τ_f along the base. Fix: reach
   1150 m and the `leak` factor (a plain sun-elevation factor on the glow was tried first and rejected: it also
   killed the real glow through the thin tower tops in `backlit_golden`).
9. Tried and rejected: a fan of cones tilted 45° from s (to find nearer sun-lit boundaries, e.g. the lit top ridge
   above the dark face of a tower). The lit ridge is real physics and the term worked where intended, but the
   boundaries the cones reach are often not lit (the base's underside, terminator flanks, a base top shadowed by a
   neighbouring tower) and a single straight cone cannot tell; the net effect was negative (below_noon 0.085 →
   0.10) for a small gain on the backlit body (0.066 → 0.063). Removed.

## 6. Results and what still differs from the references

Linear error mean|render − ref| / mean(ref) over the whole image (sky included), and the per-channel ratio of
means, 16 frames:

| pair | error | ratio R, G, B |
|---|---|---|
| front_noon | 0.059 | 1.02, 1.01, 1.00 |
| front_golden | 0.041 | 1.01, 1.00, 0.99 |
| front_sunset | 0.035 | 1.01, 0.99, 0.99 |
| side_noon | 0.058 | 1.01, 1.01, 1.00 |
| side_afternoon | 0.063 | 1.00, 1.00, 1.00 |
| side_sunset | 0.057 | 1.03, 1.01, 1.00 |
| backlit_afternoon | 0.066 | 0.97, 0.97, 0.98 |
| backlit_golden | 0.054 | 0.95, 0.97, 0.98 |
| backlit_sunset | 0.045 | 0.95, 0.98, 0.99 |
| below_noon | 0.085 | 0.94, 0.94, 0.95 |
| below_golden | 0.065 | 1.00, 0.99, 0.98 |
| distant_sunset | 0.017 | 1.01, 1.00, 1.00 |

Colour: the sunset lit face matches to 4 % (G/R 0.269 vs 0.266, B/R 0.094 vs 0.087); the golden base to 8 %;
the noon shadow flank's B/R 0.97 vs 0.99.

What still differs, and why:
- **Backlit mid-body 0.6–0.85×** (afternoon and golden). The dark face of the big tower is fed by lit surfaces
  that are much nearer in directions tilted from the sun (the lit top ridge, the lit side rims) than along s,
  which a single straight cone cannot see; the tilted-cone attempt (revision 9) could not distinguish lit from
  unlit boundaries. The sphere MC itself gives only 0.09 at τ_s = 27, so a convex-body model is intrinsically
  2–3× dark there; the real cloud's lumps re-radiate.
- **Thin wisps at the far edge of the base seen from below ≈ 1.2×**: the multiple-scattering terms still do not
  vanish completely in optically thin cloud.
- **Billow texture**: the renders are a little smoother than the references on lit faces (the 200 m gradients
  and 150 m-blurred τ_c smooth the billow-scale shading) and a little more cellular on the backlit body (the
  boundary-sink factor varies at the billow scale, the real fluence less so).
- **Sky colour**: the sky light is modelled with the mean sky radiance. The horizon is whiter/warmer and the zenith
  bluer, so flanks should be slightly warmer and tops slightly bluer than rendered; at sunset the flank facing
  the sun sees an orange horizon the mean cannot express. The effect is a few percent in the colour ratios.
- **Renderer cut-offs** (≤ 3 % at thin rims) are outside the network.

Free views (`work/renders/free_*.png`: sun from the south-east, sun behind the backlit camera, camera above the
cloud, below camera at 5° sun, front camera side-lit from the SSE) show no artifacts; every term is a smooth
function of local density structure, sun elevation and view geometry and makes physical sense for any camera and
sun.
