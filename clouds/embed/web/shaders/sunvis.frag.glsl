// HypaNet Clouds embed: how much of the sun disk the camera sees (for the glare in tonemap.frag.glsl). Writes one
// texel: the mean, over 13 points of the disk, of the march's view transmittance T toward that point, counted only
// where the march would draw the disk (T >= 0.002 and the point above the horizon). Density only (no network): T
// along a view ray depends on the density alone. Deterministic midpoint steps on the march's step grid.
// clouds.js assembles: precision statements, shaders/density.glsl, sky_common.glsl, then this file.
uniform int uMaxSteps;
uniform float uSigmaMax;
uniform float uStep;
uniform vec3 uCamPos;
out vec4 fragColor;

vec2 boxHitV(vec3 ro, vec3 rd) {
  vec3 r = vec3(abs(rd.x) < 1e-12 ? 1e-12 : rd.x, abs(rd.y) < 1e-12 ? 1e-12 : rd.y, abs(rd.z) < 1e-12 ? 1e-12 : rd.z);
  vec3 ta = (uBoxMin - ro) / r;
  vec3 tb = (uBoxMax - ro) / r;
  vec3 tmin = min(ta, tb);
  vec3 tmax = max(ta, tb);
  return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

float viewT(vec3 rd) {
  vec2 tb = boxHitV(uCamPos, rd);
  float t0 = max(tb.x, 0.0), t1 = tb.y;
  if (t1 <= t0) return 1.0;
  float dt = max(uStep, (t1 - t0) / float(uMaxSteps));
  float T = 1.0;
  for (int i = 0; i < uMaxSteps; i++) {
    float t = t0 + (float(i) + 0.5) * dt;
    if (t >= t1) break;
    float d = dens0(uCamPos + t * rd);
    if (d > 0.002) {
      T *= exp(-d * uSigmaMax * dt);
      if (T < 0.002) return 0.0;
    }
  }
  return T;
}

void main() {
  vec3 s = uSun;
  vec3 a = normalize(abs(s.y) < 0.9 ? cross(s, vec3(0.0, 1.0, 0.0)) : cross(s, vec3(1.0, 0.0, 0.0)));
  vec3 b = cross(s, a);
  float acc = 0.0;
  for (int k = 0; k < 13; k++) {
    // centre, then 6 points at 0.5 and 6 at 0.85 of the radius (rotated by half a step)
    float r = k == 0 ? 0.0 : (k < 7 ? 0.5 : 0.85);
    float ang = (float(k) + (k < 7 ? 0.0 : 0.5)) * (SKY_PI / 3.0);
    vec3 d = normalize(s + SUN_SIN_R * r * (cos(ang) * a + sin(ang) * b));
    if (d.y > 0.0) acc += viewT(d);
  }
  fragColor = vec4(acc / 13.0, 0.0, 0.0, 1.0);
}
