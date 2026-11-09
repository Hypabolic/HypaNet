// Sky-view pass (port of scene/sky.py sky_radiance + build_sky_map's lower hemisphere): the equirect sky map seen
// from the camera's altitude, for the march's sky() lookup (u = phi / 2pi, phi = atan2(d.x, -d.z); texel row 0 =
// zenith). Upper rows: single-scattering Rayleigh + Mie with ozone, 160 quadratically spaced steps to the top of
// the atmosphere, sun transmittance from the LUT. Lower rows (theta >= pi/2): the constant ground radiance uLgnd.
// Prepended by sky.js: precision statements + sky_common.glsl.
uniform vec3 uSunDir;                 // toward-sun unit vector s
uniform float uCamH;                  // observer altitude (m), sky.H_OBS
uniform vec3 uLgnd;                   // lower hemisphere radiance
uniform int uSteps;                   // 160, as sky_radiance
uniform ivec2 uSize;                  // map W x H
out vec4 fragColor;

float phaseRayleigh(float nu) { return 3.0 / (16.0 * SKY_PI) * (1.0 + nu * nu); }
float phaseMie(float nu) {
  const float g = 0.8;
  return 3.0 / (8.0 * SKY_PI) * (1.0 - g * g) * (1.0 + nu * nu) / ((2.0 + g * g) * pow(1.0 + g * g - 2.0 * g * nu, 1.5));
}

void main() {
  ivec2 px = ivec2(gl_FragCoord.xy);
  if (2 * px.y >= uSize.y) {
    fragColor = vec4(uLgnd, 1.0);
    return;
  }
  float theta = SKY_PI * gl_FragCoord.y / float(uSize.y);
  float phi = 2.0 * SKY_PI * gl_FragCoord.x / float(uSize.x);
  vec3 d = vec3(sin(theta) * sin(phi), cos(theta), -sin(theta) * cos(phi));
  vec3 s = uSunDir;
  float r0 = SKY_R_EARTH + uCamH;
  float mu = d.y;
  float L = skyRayToTop(uCamH, mu);
  float nu = dot(d, s);
  float pr = phaseRayleigh(nu), pm = phaseMie(nu);
  float n = float(uSteps);
  vec3 tauV = vec3(0.0), sum = vec3(0.0);
  for (int k = 0; k < uSteps; k++) {
    float kk = (float(k) + 0.5) / n;
    float t = L * kk * kk;
    float dt = L * 2.0 * kk / n;
    vec3 p = vec3(d.x * t, r0 + d.y * t, d.z * t);  // relative to the planet centre, observer at (0, r0, 0)
    float rr = length(p);
    float hh = skyAltitudeAlong(uCamH, mu, t);
    float muS = dot(p, s) / rr;
    vec3 e = skyExtinction(hh) * dt;
    tauV += e;                                       // observer -> sample, midpoint: cumsum - half the own step
    vec3 T = exp(-(tauV - 0.5 * e + skyTau(hh, muS)));
    float hp = max(hh, 0.0);
    vec3 sca = SKY_BETA_R * (exp(-hp / SKY_H_R) * pr) + vec3(SKY_BETA_M_SCA * exp(-hp / SKY_H_M) * pm);
    sum += sca * T * dt;
  }
  fragColor = vec4(SKY_E_TOA * sum, 1.0);
}
