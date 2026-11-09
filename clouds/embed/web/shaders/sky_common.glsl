// HypaNet Clouds embed: the atmosphere of scene/sky.py on the GPU (shared by the sky passes and the march).
// Include after the precision statements. Defines the transmittance-LUT sampler uTransLut, the atmosphere
// constants, skyExtinction(h), skyTau(h, mu) (bilinear lookup, identical to sky.TransmittanceTable.lookup) and
// sunDiskRadiance(d, s, camH) (the visible sun disk).
//
// Transmittance LUT (built once by web/sky.js from shaders/sky_lut.frag.glsl): RGBA32F, SKY_TL_NMU x SKY_TL_NH
// texels, x = mu index, y = altitude index; rgb = optical depth from altitude h along cos-zenith mu to space,
// 1e4 when the ray hits the planet. Nodes (not cells): h_i = HMAX (i / (NH - 1))^2, mu_j = -1 + 2 j / (NMU - 1).
// Reads use texelFetch with in-shader bilinear weights, so no float-linear filtering is needed and the weights
// are exact (hardware filtering keeps only ~8 fractional bits, too coarse at the horizon).

uniform highp sampler2D uTransLut;

#define SKY_PI 3.14159265358979
const float SKY_R_EARTH = 6360000.0;
const float SKY_HMAX = 100000.0;          // R_TOP - R_EARTH
const float SKY_E_TOA = 20.0;
const int SKY_TL_NH = 192;
const int SKY_TL_NMU = 2048;
const vec3 SKY_BETA_R = vec3(5.802e-6, 13.558e-6, 33.100e-6);
const float SKY_BETA_M_SCA = 3.996e-6;
const float SKY_BETA_M_EXT = 3.996e-6 * 1.11;
const vec3 SKY_SIGMA_O3 = vec3(0.650e-6, 1.881e-6, 0.085e-6);
const float SKY_H_R = 8000.0;
const float SKY_H_M = 1200.0;
const float SKY_O3_CENTER = 25000.0;
const float SKY_O3_HALF = 15000.0;
// sun: 0.5332 degree diameter; sin of the angular radius and the disk's solid angle 2 pi (1 - cos R)
const float SUN_SIN_R = 0.004653030995425583;
const float SUN_OMEGA = 6.801804019774604e-05;

vec3 skyExtinction(float h) {
  float hp = max(h, 0.0);
  float o3 = max(0.0, 1.0 - abs(h - SKY_O3_CENTER) / SKY_O3_HALF);
  return SKY_BETA_R * exp(-hp / SKY_H_R) + SKY_BETA_M_EXT * exp(-hp / SKY_H_M) + SKY_SIGMA_O3 * o3;
}

// altitude of the point at distance t along cos-zenith mu from altitude h, without the cancellation of
// |p| - R_EARTH in float32: (|p|^2 - R^2) / (|p| + R), |p|^2 - R^2 = h (2R + h) + t (t + 2 r mu)
float skyAltitudeAlong(float h, float mu, float t) {
  float r = SKY_R_EARTH + h;
  float num = h * (2.0 * SKY_R_EARTH + h) + t * (t + 2.0 * r * mu);
  float rr = sqrt(max(r * r + t * (t + 2.0 * r * mu), 0.0));
  return num / (rr + SKY_R_EARTH);
}

// distance from altitude h along cos-zenith mu to the top of the atmosphere (sky.ray_to_top), float32-stable
float skyRayToTop(float h, float mu) {
  float r = SKY_R_EARTH + h;
  float x = max(SKY_HMAX - h, 0.0) * (2.0 * SKY_R_EARTH + SKY_HMAX + h);   // R_TOP^2 - r^2
  float sq = sqrt(max(r * r * mu * mu + x, 0.0));
  return mu > 0.0 ? x / (r * mu + sq) : sq - r * mu;
}

// optical depth (RGB) from altitude h along cos-zenith mu to space (sky.TransmittanceTable.lookup)
vec3 skyTau(float h, float mu) {
  float uh = sqrt(clamp(h, 0.0, SKY_HMAX) / SKY_HMAX) * float(SKY_TL_NH - 1);
  float um = (clamp(mu, -1.0, 1.0) + 1.0) * 0.5 * float(SKY_TL_NMU - 1);
  int i0 = clamp(int(floor(uh)), 0, SKY_TL_NH - 2);
  int j0 = clamp(int(floor(um)), 0, SKY_TL_NMU - 2);
  float fi = uh - float(i0);
  float fj = um - float(j0);
  vec3 t00 = texelFetch(uTransLut, ivec2(j0, i0), 0).rgb;
  vec3 t01 = texelFetch(uTransLut, ivec2(j0 + 1, i0), 0).rgb;
  vec3 t10 = texelFetch(uTransLut, ivec2(j0, i0 + 1), 0).rgb;
  vec3 t11 = texelFetch(uTransLut, ivec2(j0 + 1, i0 + 1), 0).rgb;
  return mix(mix(t00, t01, fj), mix(t10, t11, fj), fi);
}

// sun disk radiance along view direction d (unit) for toward-sun s, camera at altitude camH (m):
// E_TOA exp(-tau(camH, d.y)) / Omega_sun, limb-darkened (1 - 0.6 (1 - mu)) / 0.8 (mean 1 over the disk).
// Only above the horizon (d.y > 0); the host also gates it by the march's transmittance (T >= 0.002).
// The angle to the sun comes from |d x s|, not from dot(d, s): near 1 a float32 cosine resolves the
// 0.27 degree radius only to ~7 %.
vec3 sunDiskRadiance(vec3 d, vec3 s, float camH) {
  if (d.y <= 0.0 || dot(d, s) <= 0.0) return vec3(0.0);
  float r = length(cross(d, s)) / SUN_SIN_R;
  if (r >= 1.0) return vec3(0.0);
  float limb = (1.0 - 0.6 * (1.0 - sqrt(1.0 - r * r))) / 0.8;
  return SKY_E_TOA * exp(-min(skyTau(camH, d.y), vec3(700.0))) / SUN_OMEGA * limb;
}
