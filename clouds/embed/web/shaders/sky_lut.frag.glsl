// Transmittance LUT pass (port of scene/sky.py TransmittanceTable): one texel per node (mu_j, h_i), x = j, y = i.
// rgb = optical depth to space along a quadratically spaced 400-step midpoint rule (denser near the start), 1e4 when
// the ray hits the planet. Prepended by sky.js: precision statements + sky_common.glsl.
uniform int uSteps;                   // 400, as the Python table
out vec4 fragColor;

void main() {
  int j = int(gl_FragCoord.x);
  int i = int(gl_FragCoord.y);
  float uh = float(i) / float(SKY_TL_NH - 1);
  float h = SKY_HMAX * uh * uh;
  float mu = -1.0 + 2.0 * float(j) / float(SKY_TL_NMU - 1);
  // hits the ground: mu < 0 and r^2 (mu^2 - 1) + R^2 >= 0, written as r^2 mu^2 - h (2R + h) to avoid cancellation
  float r = SKY_R_EARTH + h;
  if (mu < 0.0 && r * r * mu * mu - h * (2.0 * SKY_R_EARTH + h) >= 0.0) {
    fragColor = vec4(vec3(1e4), 1.0);
    return;
  }
  float L = skyRayToTop(h, mu);
  float n = float(uSteps);
  vec3 tau = vec3(0.0);
  for (int k = 0; k < uSteps; k++) {
    float sk = (float(k) + 0.5) / n;
    float t = L * sk * sk;
    float dt = L * 2.0 * sk / n;
    tau += skyExtinction(skyAltitudeAlong(h, mu, t)) * dt;
  }
  fragColor = vec4(tau, 1.0);
}
