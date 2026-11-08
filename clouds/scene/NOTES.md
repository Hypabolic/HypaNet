# scene/ notes (SPEC §2 project setup, §3 scene)

Build everything (deterministic, about 50 s on the M4 Pro):

```
cd clouds && uv run python scene/build_scene.py            # density + skies + shots.json
uv run python scene/build_scene.py --no-sky                 # density + shots only (reuses sky_*.json stats)
uv run python scene/preview_fast.py side_noon --zoom 1.6    # noise-free quick look of the shape (~35 s)
uv run python scene/preview_mitsuba.py front_noon --spp 1024 # quick path-traced look (NOT the reference)
uv run python scene/choose_exposure.py                      # suggests EXPOSURE table (paste into build_scene.py)
```

Files: `build_scene.py` (entry point, presets, shots, exposures, pairs), `density.py` (cloud), `noise.py`
(Perlin / Worley), `sky.py` (sun + sky), `tonemap.py` (SPEC tone map, numpy; GLSL twin in its docstring),
`preview_fast.py`, `preview_mitsuba.py`, `choose_exposure.py` (harness helpers only).

## Output formats (scene/out/)

All in world units: metres, y up, north = -z, east = +x.

### density.u8 + density.json
- `density.u8`: 256 × 160 × 256 = 10,485,760 bytes, uint8, **x fastest, then y, then z**:
  `index = x + 256 * (y + 160 * z)`; `d = byte / 255`. In numpy:
  `np.fromfile(p, np.uint8).reshape(nz=256, ny=160, nx=256)` gives `[z, y, x]`.
- Box `box_min = (-900, 1450, -900)`, `box_max = (900, 2550, 900)`; voxel 7.03 × 6.875 × 7.03 m.
  Cell-centred: voxel (i, j, k) centre = `box_min + (i + 0.5, j + 0.5, k + 0.5) * voxel_size`, i.e. the GL
  `Data3DTexture` convention `texcoord = (p - box_min) / (box_max - box_min)` and Mitsuba `gridvolume` with
  `to_world = translate(box_min) * scale(box_max - box_min)` (as in `preview_mitsuba.py`).
- Two empty voxels on every box face (so clamp-to-edge sampling gives 0 at the box).
- `density.json`: `dims` [nx, ny, nz], `box_min`, `box_max`, `voxel_size`, `sigma_t_max` (0.04), `albedo`,
  `phase`, `seed`, `base_y`, `occupied_bounds_m`, `occupied_fraction`, `sha256` of the raw bytes.

### sky_<preset>.f32 + sky_<preset>.json (+ sky_<preset>.png preview)
- 1024 × 512 × 3 float32 little-endian, RGB interleaved, row-major, **row 0 = zenith (top)**; 6,291,456 bytes.
  `np.fromfile(p, "<f4").reshape(512, 1024, 3)`.
- Mapping (pixel centres): `u = (col + 0.5)/1024`, `v = (row + 0.5)/512`, `phi = 2πu`, `theta = πv`,
  direction `d = (sin θ sin φ, cos θ, −sin θ cos φ)`. Inverse: `θ = acos(d.y)`, `φ = atan2(d.x, −d.z) mod 2π`.
  So u = 0 looks north (−z), u = 0.25 east (+x), u = 0.5 south (+z), u = 0.75 west (−x).
  This is meant to be Mitsuba 3's `envmap` convention with identity `to_world` (Mitsuba's envmap uses
  `phi = atan2(d.x, -d.z)`, `theta = acos(d.y)`); the reference agent must still verify it with the §6.2
  background check (e.g. render the sunset map and check the bright horizon glow lands at the sun azimuth 250°).
- Rows 256..511 (θ > 90°) hold the constant ground radiance `L_gnd`.
- Sidecar: `width, height, toward_sun, E_sun, L_sky, L_gnd, E_sky, observer_altitude_m, sha256`.
- The PNG is a 512 × 256 tone-mapped preview (preset exposure noon/afternoon 0.25, golden 0.35, sunset 1.0).

### shots.json
- `presets.<name>`: `elevation_deg, azimuth_deg (250, clockwise from north), toward_sun, air_mass, T_atm, E_sun,
  L_sky, L_gnd, E_sky, sky_map, sky_json`.
- `shots.<id>`: `position, look_at, up [0,1,0], fov_y_deg (vertical), width 640, height 400, description`.
  Pinhole camera, `forward = normalize(look_at − position)`, `right = normalize(cross(forward, up))`,
  `cam_up = cross(right, forward)`; pixel (i, j), j = 0 at the top:
  `dir ∝ forward + ((2(i+0.5)/W − 1) · tan(fov/2) · W/H) · right + ((1 − 2(j+0.5)/H) · tan(fov/2)) · cam_up`.
  (Mitsuba `perspective` with `fov_axis: "y"` and `look_at` gives the same rays.)
- `pairs.reference` (12) and `pairs.heldback` (6): `{id: "<shot>_<preset>", shot, preset, exposure}`.
- `exposure_table`: exposure for all 24 shot × preset combinations (viewer, previews).
- `medium`, `sun_model` (constants, formulas, ozone depths), `units`, `camera_model`.

## Sun model
`E_sun = 20 · T_atm(h)`, `T_atm = exp(−(β_R · 8000 + β_M · 1.11 · 1200 + τ_O3) · m(h))`, Kasten–Young air mass,
β_R = (5.802, 13.558, 33.100)e-6 m⁻¹, β_M(sca) = 3.996e-6 m⁻¹ (extinction × 1.11). Same TOA irradiance (20,
spectrally flat in RGB) lights the sky.

**Ozone** (my choice; SPEC leaves it open): absorption coefficient at the layer peak
σ_O3 = (0.650, 1.881, 0.085) × 10⁻⁶ m⁻¹ for (680, 550, 440) nm (the values used by Bruneton 2017 / Hillaire 2020),
tent profile 0 at 10 km → 1 at 25 km → 0 at 40 km. Zenith optical depth from the ground
**τ_O3 = (0.00975, 0.02822, 0.00128)** (≈ 300 DU of ozone: Chappuis band, strongest in green). It is used both
in T_atm (scaled by the Kasten–Young air mass, as SPEC says) and in the sky integration (along the actual
spherical paths). Its effect: slightly bluer sky at low sun; small on E_sun.

| preset | elev | air mass | E_sun (RGB) | L_sky | L_gnd |
|---|---|---|---|---|---|
| noon | 55° | 1.22 | 18.55, 16.82, 14.36 | 0.175, 0.322, 0.557 | 0.494, 0.460, 0.418 |
| afternoon | 25° | 2.36 | 17.30, 14.31, 10.55 | 0.191, 0.328, 0.524 | 0.243, 0.212, 0.179 |
| golden | 8° | 6.86 | 13.12, 7.55, 3.11 | 0.179, 0.248, 0.327 | 0.067, 0.048, 0.037 |
| sunset | 2° | 19.43 | 6.05, 1.27, 0.102 | 0.133, 0.126, 0.124 | 0.0134, 0.0090, 0.0101 |

## Sky model (sky.py)
- Single scattering, Nishita 1993 style, spherical planet (R = 6360 km, atmosphere top 6460 km). Rayleigh
  (H = 8 km, β as above, phase 3/(16π)(1+ν²)), Mie (H = 1.2 km, β_sca = 3.996e-6, β_ext = 1.11 β_sca,
  Cornette–Shanks phase with g = 0.8), ozone absorption (above).
- Observer at **2,000 m** (middle of the cloud layer). 160 quadratic-spaced steps per view ray to the top of the
  atmosphere; sun transmittance from a precomputed optical-depth table (192 altitudes × 2048 zenith cosines,
  400 steps each); sunlight is 20 at the TOA (not attenuated in the sky integral except by the atmosphere along the
  path). Samples whose sun ray hits the planet get no light.
- No sun disk (the sun is the directional light). No multiple scattering in the sky (per SPEC); the consequence is
  a sky that is somewhat dark and greyish at sunset, away from the horizon.
- Lower hemisphere: Lambertian ground, albedo 0.1, `L_gnd = 0.1/π · (E_sun · sin h + E_sky)`, where `E_sky` is the
  cosine-weighted integral of the upper hemisphere of the map itself (cloud shadowing of the ground ignored). The
  ground fills every direction below the mathematical horizon (θ > 90°); the true horizon dip from 2 km (1.4°) is
  ignored.
- `L_sky` = plain (not cosine-weighted) mean radiance over the upper hemisphere's solid angle, as SPEC says.
- Looks: noon deep blue (from 2 km it is darker than at sea level) with a pale horizon; afternoon a soft aureole
  toward the sun; golden a warm glow at the horizon under the sun; sunset an orange horizon (3.0, 1.1, 0.15 at
  the horizon under the sun, orange-red (0.68, 0.28, 0.04) opposite) under a grey-blue sky. With per-channel ACES
  at exposure 1 the brightest horizon near the sun clips toward yellow in the PNG; the linear data is orange.

## Density field (density.py)
SPEC asks for "displaced metaballs + Perlin-Worley FBM erosion, height profile, fixed seed" (seed 20261108).
1. **Skeleton.** A low "base slab" of 10 spheres (r 160–300 m, centred 0.25 r above the base, so the cut base is
   wide and flat) plus **7 towers** of different heights (crowns at 2470, 2230, 2070, 1950, 1870, 1820, 1760 m),
   each a chain of spheres that narrows upward (r_base 140–270 → r_top 90–165), leans slightly downwind
   (0.10 m/m toward +x, 0.03 toward +z) and is jittered. The towers are 350–500 m apart so they read as separate
   turrets with dips between them.
2. **Billows**: 3 generations (110 × r 75–130 m, 420 × r 38–68 m, 1500 × r 17–30 m) seeded on the current
   iso-surface, biased to upward-facing surface and to height, sunk 0.40 / 0.34 / 0.28 r into the parent so they
   bulge (cauliflower). Billow radius grows with height (×0.6 at the base → ×1.15 at +800 m).
3. Smooth union by log-sum-exp with k = 6 m (sharp creases), evaluated at domain-warped coordinates (30 m,
   500 m wavelength), plus 10 m of FBM displacement. The base is cut flat at 1500 m ± 6 m (ragged undulation).
4. **Erosion**: inward carve of the distance field by Perlin-Worley noise: 7 m of fine billowy cracks at the tops,
   up to 40 m of wispy erosion on the lower flanks; then a smoothstep edge ramp 7 m wide at the tops, 15 m low down;
   thin wisps (d ≤ 0.25) just outside the lower flanks.
5. Interior profile 0.85 at the base → 1.0 at +500 m, × (0.9 ± 0.1) low-frequency variation.

Result: occupied x −759…745, y 1498…2509, z −591…654 m (well inside the box, nothing touches a face), 17 % of
voxels occupied. Previous attempt (a few huge overlapping skeleton spheres + 15 % sunk billows) read as one
rounded blob; the tower skeleton and bulging billows fixed that.

## Shots and presets (build_scene.py)
Sun azimuth 250° (WSW) for every preset; camera azimuths are set relative to it so each shot keeps its lighting
geometry at every sun elevation. Cloud centre (0, 1900, 0).

| shot | camera | lighting |
|---|---|---|
| front | 3.3 km away on the sun side, 1150 m altitude, fov 30° | sun behind the camera |
| side | 3.3 km, sun azimuth + 90°, 1150 m, fov 30° | sun from the left at 90° (terminator) |
| backlit | 3.3 km opposite the sun, 1200 m, fov 32° | looking toward the sun (silver lining) |
| below | 950 m from the axis, 600 m altitude, looking up at (0, 1700, 0), fov 65° | view of the flat base |
| distant | 14 km, 400 m altitude, looking 5° up, fov 12° | whole cloud low in the sky, horizon at the bottom |
| quarter | 2.8 km at sun azimuth + 135°, 2000 m altitude, fov 36° | three-quarter backlit close-up |

Reference pairs (12): front × {noon, golden, sunset}, side × {noon, afternoon, sunset}, backlit × {afternoon,
golden, sunset}, below × {noon, golden}, distant × sunset. Held-back (6): front_afternoon, side_golden,
backlit_noon, below_sunset, distant_golden, quarter_afternoon (the quarter shot is never shown to the author).

## Exposures
Per pair, from `choose_exposure.py`: full-scene Mitsuba render at 160 × 100, 512 spp; cloud pixels = optical depth
along the pixel ray > 1; exposure = 1 / p90(cloud luminance), rounded **down** to a 1/3-stop step (2^(k/3)), so the
cloud's bright parts sit at ACES input ≤ 1 (display ~0.8) and nothing important clips. Backlit pairs get a low
exposure because of the very bright silver lining (p99 up to 25); their dark body sits around mid-grey, which is how
a backlit cumulus photographs. Sunset pairs need 2–4 (the cloud is lit by E_sun ≈ (6, 1.3, 0.1)); the low sky near
the sun then saturates, which is fine.

| shot | noon | afternoon | golden | sunset |
|---|---|---|---|---|
| front | 0.315 | 0.315 | 0.5 | 2.0 |
| side | 0.315 | 0.397 | 0.794 | 2.52 |
| backlit | 0.397 | 0.198 | 0.315 | 1.587 |
| below | 0.25 | 0.397 | 1.0 | 4.0 |
| distant | 0.315 | 0.397 | 0.63 | 2.52 |
| quarter | 0.397 | 0.397 | 0.63 | 2.52 |

## Gotchas for the renderer / reference agents
- **Mitsuba `perspective` has `far_clip = 10,000 m` by default**; the distant camera is 14 km from the cloud, so the
  cloud silently disappears. Set `far_clip` (e.g. 1e6), as `preview_mitsuba.py` now does. The same applies to any
  far plane in the three.js renderer (it ray-marches analytically, so it should be fine).
- Mitsuba directional emitter: `direction` is the light's travel direction = `−toward_sun`; `irradiance` = E_sun.
- Path-traced noise is heavy for this medium (albedo 0.999, τ up to ~40, g = 0.9 lobe): 4096 spp at 320 × 200
  is still visibly grainy after tone mapping. The reference renders need far more samples (or accumulation over
  several runs with different seeds) to be "noise invisible".
- `out/previews/`: quick looks, NOT references: `<pair>.png` (Mitsuba metal_ad_rgb volpath, 320 × 200, 1024 spp, ~10 s each; linear `.npy` alongside, gitignored), `contact_sheet.png` (all 18 pairs, reference first), `*_fast.png` (noise-free approximate raymarch from `preview_fast.py`, zoom 1.6, for judging shape).

## Deviations / open choices
- scipy is a dependency, used only by the helper scripts (`preview_fast.py`, `choose_exposure.py`), not by
  `build_scene.py`, which needs only numpy + pillow.
- Ozone values, observer altitude 2 km, Mie g = 0.8 (Cornette–Shanks), sun azimuth 250°, the sixth "quarter" shot
  (held-back only) and the exposure rule are my choices; SPEC leaves them open.
- The sky's lower hemisphere starts at the mathematical horizon (θ = 90°), not the 1.4° geometric dip from 2 km.
