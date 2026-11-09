// HypaNet Clouds embed: the ray march, one sample per pixel per frame, blended into the previous frame.
// The march is the research harness's (renderer/shaders/march.frag.glsl, MODE_NET + FINALIZE) with the compiled
// network: the same pixel jitter, jittered step, d > 0.002 gate, T < 0.002 stop and L += T albedo S a, and the same
// background term T sky(v). Two optional terms of the video path are on for the page: E_sun by altitude (uEsunLut) and
// the sun disk (sunDiskRadiance, only through rays that left the box with T >= 0.002).
//
// clouds.js assembles the source: precision statements, shaders/density.glsl, shaders/sky_common.glsl,
// gen/bake.glsl, gen/net_eval.glsl, then this file. Output: prev + (L - prev) / uMeanN (uMeanN = n + 1 for a running
// mean over n + 1 frames, exactly the harness's FINALIZE; a constant 1 / alpha while the sun moves; <= 1 restarts).
#define PI 3.14159265358979

// Loop bound is a uniform (512), not a constant: with constant bounds the Metal compiler behind ANGLE tries to
// unroll the loop (renderer/shaders/march.frag.glsl).
uniform int uMaxSteps;
uniform float uSigmaMax;              // sigma_t_max (1/m)
uniform float uAlbedo;

// ---- camera / frame ----
uniform vec3 uCamPos;
uniform vec3 uCamFwd;
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec2 uTan;                    // (tan(fov_y/2) * aspect, tan(fov_y/2))
uniform vec2 uRes;                    // full image size in pixels
uniform int uFrame;                   // frame index (jitter seed)
uniform int uPixelJitter;             // 1: jitter the sub-pixel position per frame (box filter)
uniform float uStep;                  // base step along the view ray (m)
uniform sampler2D uPrev;              // previous output (same size)
uniform float uMeanN;                 // this frame's sample gets weight 1 / uMeanN
uniform int uRejectHist;              // 1 (temporal blend while the sun moves): drop history far brighter than the sample

// ---- light ----
uniform vec3 uEsun;                   // E_sun when uEsunLutOn == 0 (the reference presets)
uniform int uEsunLutOn;               // 1: E_sun at a sample = uEsunLut interpolated at its altitude y
uniform vec3 uEsunLut[32];            // E_sun at y = uEsunLutY.x + k / uEsunLutY.y, k = 0..31 (Earth's shadow -> 0)
uniform vec2 uEsunLutY;               // (y0, 1 / dy)
uniform vec3 uLsky;
uniform vec3 uLgnd;
uniform sampler2D uSkyMap;            // equirect RGBA32F sky (web/sky.js SkyGPU), row 0 = zenith, nearest
uniform int uSunDiskOn;               // 1: draw the sun disk (sunDiskRadiance) where the march left T >= 0.002

out vec4 fragColor;

// ------------------------------------------------------------------------------------------------
uint pcg(uint v) {
  uint s = v * 747796405u + 2891336453u;
  uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
float rnd(ivec2 p, int frame, uint stream) {
  uint h = pcg(uint(p.x) + pcg(uint(p.y) + pcg(uint(frame) * 8u + stream)));
  return float(h >> 8) * (1.0 / 16777216.0);
}

vec2 boxHit(vec3 ro, vec3 rd) {
  vec3 r = vec3(abs(rd.x) < 1e-12 ? 1e-12 : rd.x, abs(rd.y) < 1e-12 ? 1e-12 : rd.y, abs(rd.z) < 1e-12 ? 1e-12 : rd.z);
  vec3 inv = 1.0 / r;
  vec3 ta = (uBoxMin - ro) * inv;
  vec3 tb = (uBoxMax - ro) * inv;
  vec3 tmin = min(ta, tb);
  vec3 tmax = max(ta, tb);
  return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

// ---- sky: equirect lookup, u = phi / 2pi with phi = atan2(d.x, -d.z), v = theta / pi; bilinear, u wraps ----
vec3 skyTex(sampler2D tex, vec2 uv) {
  ivec2 sz = textureSize(tex, 0);
  vec2 pc = uv * vec2(sz) - 0.5;
  vec2 f = floor(pc);
  vec2 w = pc - f;
  int x0 = int(f.x);
  int y0 = int(f.y);
  int xa = ((x0 % sz.x) + sz.x) % sz.x;
  int xb = (xa + 1) % sz.x;
  int ya = clamp(y0, 0, sz.y - 1);
  int yb = clamp(y0 + 1, 0, sz.y - 1);
  vec3 c00 = texelFetch(tex, ivec2(xa, ya), 0).rgb;
  vec3 c10 = texelFetch(tex, ivec2(xb, ya), 0).rgb;
  vec3 c01 = texelFetch(tex, ivec2(xa, yb), 0).rgb;
  vec3 c11 = texelFetch(tex, ivec2(xb, yb), 0).rgb;
  return mix(mix(c00, c10, w.x), mix(c01, c11, w.x), w.y);
}
// the map is rendered for the true sun vector, so no rotation (the harness's uSkyRot = 0, uSkyMix = 0)
vec3 sky(vec3 d) {
  float theta = acos(clamp(d.y, -1.0, 1.0));
  float phi = atan(d.x, -d.z);
  return skyTex(uSkyMap, vec2(fract(phi / (2.0 * PI)), theta / PI));
}

vec3 esunAt(float y) {
  if (uEsunLutOn == 0) return uEsun;
  float u = clamp((y - uEsunLutY.x) * uEsunLutY.y, 0.0, 31.0);
  int k = min(int(u), 30);
  return mix(uEsunLut[k], uEsunLut[k + 1], u - float(k));
}

// ---- camera ray for pixel (column, row from top) ----
vec3 rayDir(ivec2 pix, int frame) {
  vec2 o = uPixelJitter == 1 ? vec2(rnd(pix, frame, 0u), rnd(pix, frame, 1u)) : vec2(0.5);
  float sx = (2.0 * (float(pix.x) + o.x) / uRes.x - 1.0) * uTan.x;
  float sy = (1.0 - 2.0 * (float(pix.y) + o.y) / uRes.y) * uTan.y;
  return normalize(uCamFwd + sx * uCamRight + sy * uCamUp);
}

// ---- the march ----
// Sample k of a ray sits at t0 + (k + jitter) dt, dt = max(uStep, box chord / uMaxSteps).
struct Ray { vec3 ro; vec3 rd; vec3 e2; vec3 e3; float t0; float t1; float dt; float jit; bool hit; };

Ray makeRay(ivec2 pix, int frame) {
  Ray R;
  R.ro = uCamPos;
  R.rd = rayDir(pix, frame);
  R.e2 = normalize(cross(R.rd, uCamUp));   // view frame: (v, e2 ~ camera right, e3 ~ camera up), orthonormal
  R.e3 = cross(R.e2, R.rd);
  vec2 tb = boxHit(R.ro, R.rd);
  R.t0 = max(tb.x, 0.0);
  R.t1 = tb.y;
  R.hit = R.t1 > R.t0;
  R.dt = R.hit ? max(uStep, (R.t1 - R.t0) / float(uMaxSteps)) : uStep;
  R.jit = rnd(pix, frame, 2u);
  return R;
}

// in-scattered radiance L and transmittance T along the ray (the harness's marchSegment without the pass cap)
void march(Ray R, inout float T, inout vec3 L) {
  if (!R.hit) return;
  for (int i = 0; i < uMaxSteps; i++) {
    float t = R.t0 + (float(i) + R.jit) * R.dt;
    if (t >= R.t1) return;
    vec3 x = R.ro + t * R.rd;
    float d = dens0(x);
    if (d > 0.002) {
      float tr = exp(-d * uSigmaMax * R.dt);
      float a = 1.0 - tr;
      vec3 aSun, aSky, aGnd;
      evalNet(x, R.rd, R.e2, R.e3, aSun, aSky, aGnd);
      vec3 S = aSun * esunAt(x.y) + aSky * uLsky + aGnd * uLgnd;
      L += T * uAlbedo * S * a;
      T *= tr;
      if (T < 0.002) return;
    }
  }
}

bool finite3(vec3 v) { return !any(isnan(v)) && !any(isinf(v)); }

void main() {
  ivec2 fc = ivec2(gl_FragCoord.xy);
  ivec2 pix = ivec2(fc.x, int(uRes.y) - 1 - fc.y);
  Ray R = makeRay(pix, uFrame);
  float T = 1.0;
  vec3 L = vec3(0.0);
  march(R, T, L);
  // the sun disk only through rays that left the box (a ray cut at T < 0.002 hit opaque cloud)
  vec3 bg = sky(R.rd) + (uSunDiskOn == 1 && T >= 0.002 ? sunDiskRadiance(R.rd, uSun, uCamPos.y) : vec3(0.0));
  L += T * bg;
  if (uMeanN <= 1.0) {
    fragColor = vec4(finite3(L) ? L : vec3(0.0), 1.0);
  } else {
    vec3 prev = texelFetch(uPrev, fc, 0).rgb;
    // the sun disk (~1e5) would leave a trail of ghost suns behind a moving sun for dozens of frames; a pixel's
    // 1-sample noise stays far below a 16x jump in luminance
    if (uRejectHist == 1) {
      vec3 Y = vec3(0.2126, 0.7152, 0.0722);
      if (dot(prev, Y) > 16.0 * dot(L, Y) + 1.0) prev = L;
    }
    // a non-finite sample would stay in the running mean for good: skip it
    fragColor = vec4(finite3(L) ? prev + (L - prev) / uMeanN : prev, 1.0);
  }
}
