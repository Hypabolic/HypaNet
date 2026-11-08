// HypaNet Clouds ray marcher (SPEC section 4). One source, three variants chosen by defines:
//   MODE_NET    in-scattered radiance S from the author's MLP (generic interpreter; weights in uNet)
//   MODE_EXACT  HARNESS ONLY: exact sun single scattering, a_sun = T_sun(x) p(dot(v, s)), a_sky = a_gnd = 0
//   DEBUG_RAY   (with MODE_NET) the inspect path: fragment (col, row) re-marches ONE pixel's ray and writes
//               record values 4*col .. 4*col+3 of accepted sample row-1 (row 0 = ray summary)
//   DEBUG_EXACT (with DEBUG_RAY) HARNESS ONLY: also records the exact T_sun and phase at each sample
//   SEGMENT     (with MODE_NET or MODE_EXACT) march at most uEvalCap evaluations per pixel, saving the march
//               state (step index, T, L, nacc) in two float targets; the next pass resumes it. A single GPU
//               thread that runs too long (~100+ ms) makes macOS drop the whole draw, so heavy networks are
//               evaluated over several passes. Results are identical to an uninterrupted march.
//   FINALIZE    adds T * sky(v) to the finished state and folds the frame into the running mean (uPrev)
// Frames accumulate as a running mean in an RGBA32F target (uPrev -> output).
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp sampler3D;

#define PI 3.14159265358979
// Loop bounds are uniforms (512 and 1024), not constants: with constant bounds the Metal compiler behind
// ANGLE tries to unroll the nested loops and a single compile took two minutes.
uniform int uMaxSteps;
uniform int uMaxShadowSteps;
#define N_LEVELS 8

// ---- scene ----
uniform sampler3D uDensity;           // level 0: raw grid, R8, linear filtering, clamp to edge
uniform sampler3D uAtlas;             // blur levels 1..7 (see pyramid.js), R8, linear
uniform vec4 uLvlScale[N_LEVELS];     // atlas tc = q * scale + offset, q in voxel units
uniform vec4 uLvlOffset[N_LEVELS];
uniform vec3 uBoxMin;
uniform vec3 uBoxMax;
uniform vec3 uVoxel;                  // voxel size (m)
uniform vec3 uDims;                   // grid dims (nx, ny, nz)
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
uniform int uFrame0;                  // first frame index of this accumulation
uniform int uPixelJitter;             // 1: jitter the sub-pixel position per frame (box filter)
uniform float uStep;                  // base step along the view ray (m)
uniform float uShadowStep;            // exact mode: step of the shadow march (m)
uniform sampler2D uPrev;              // running mean so far
uniform float uStamp;                 // written to alpha: lets the host verify that a draw really ran

// ---- light ----
uniform vec3 uSun;                    // toward-sun unit vector s
uniform vec3 uLu;                     // light frame u = normalize(cross(s, world_up))
uniform vec3 uLw;                     // light frame w = cross(u, s)
uniform vec3 uEsun;
uniform vec3 uLsky;
uniform vec3 uLgnd;
uniform sampler2D uSkyA;              // equirect RGB(A) float maps (row 0 = zenith), nearest; bilinear in-shader
uniform sampler2D uSkyB;
uniform float uSkyMix;                // sky = mix(A, B, uSkyMix)
uniform float uSkyRot;                // radians: map azimuth is shifted by this (viewer sun-azimuth slider)
uniform float uBgScale;               // 1; HARNESS ONLY: 0 drops the background term L += T sky(v)

// ---- network ----
uniform sampler2D uNet;               // RGBA32F, width 1024 (see net.js)
uniform int uNumProbes;
uniform int uNumLayers;
uniform vec4 uProbeA[64];             // (type, frame, blur level lo, blur level t)
uniform vec4 uProbeB[64];             // density: (dir.xyz in frame, dist); const: (value, 0, 0, 0)
uniform vec4 uLayer[6];               // (texel offset, nin, nout, ceil(nin/4))

// ---- debug ----
uniform vec2 uDebugPixel;             // (column, row from the top)

#if defined(SEGMENT)
uniform highp sampler2D uState0;      // (next step index, or -1 when the ray is finished; T; nacc; stamp)
uniform highp sampler2D uState1;      // (L.rgb accumulated so far, 0)
uniform int uSegFirst;                // 1: start the ray (ignore uState*)
uniform int uEvalCap;                 // max network evaluations per pixel in this pass
layout(location = 0) out vec4 outState;
layout(location = 1) out vec4 outL;
#else
out vec4 fragColor;
#endif
#if defined(FINALIZE)
uniform highp sampler2D uState0;
uniform highp sampler2D uState1;
#endif

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

vec3 toVox(vec3 x) { return (x - uBoxMin) / uVoxel; }

// Raw density, trilinear. The grid has two empty voxels on every face, so clamp-to-edge gives 0 outside.
float dens0(vec3 x) { return texture(uDensity, toVox(x) / uDims).r; }

float densLevel(vec3 q, int k) {
  if (k == 0) return texture(uDensity, q / uDims).r;
  return texture(uAtlas, q * uLvlScale[k].xyz + uLvlOffset[k].xyz).r;
}

// density probe value: 0 outside the box; blur between pyramid levels lo and lo+1 with weight t
float densProbe(vec3 x, int lo, float t) {
  vec3 q = toVox(x);
  if (any(lessThan(q, vec3(0.0))) || any(greaterThan(q, uDims))) return 0.0;
  float a = densLevel(q, lo);
  if (t > 0.0) a = mix(a, densLevel(q, min(lo + 1, N_LEVELS - 1)), t);
  return a;
}

float hg(float mu, float g) {
  float d = 1.0 + g * g - 2.0 * g * mu;
  return (1.0 - g * g) / (4.0 * PI * d * sqrt(d));
}
float phaseFn(float mu) { return 0.8 * hg(mu, 0.9) + 0.2 * hg(mu, -0.3); }

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
vec3 sky(vec3 d) {
  float theta = acos(clamp(d.y, -1.0, 1.0));
  float phi = atan(d.x, -d.z) - uSkyRot;
  vec2 uv = vec2(fract(phi / (2.0 * PI)), theta / PI);
  vec3 a = skyTex(uSkyA, uv);
  if (uSkyMix <= 0.0) return a;
  return mix(a, skyTex(uSkyB, uv), uSkyMix);
}

// ---- camera ray for pixel (column, row from top) ----
vec3 rayDir(ivec2 pix, int frame) {
  vec2 o = uPixelJitter == 1 ? vec2(rnd(pix, frame, 0u), rnd(pix, frame, 1u)) : vec2(0.5);
  float sx = (2.0 * (float(pix.x) + o.x) / uRes.x - 1.0) * uTan.x;
  float sy = (1.0 - 2.0 * (float(pix.y) + o.y) / uRes.y) * uTan.y;
  return normalize(uCamFwd + sx * uCamRight + sy * uCamUp);
}

// ---- exact single scattering (harness only) ----
float sunTransmittance(vec3 x, float jit) {
  float t1 = boxHit(x, uSun).y;
  if (t1 <= 0.0) return 1.0;
  float dt = max(uShadowStep, t1 / float(uMaxShadowSteps));
  float acc = 0.0;
  float t = jit * dt;
  float stopAt = 18.0 / (uSigmaMax * dt);   // optical depth 18: T < 2e-8
  for (int i = 0; i < uMaxShadowSteps; i++) {
    if (t >= t1) break;
    acc += dens0(x + t * uSun);
    if (acc > stopAt) break;
    t += dt;
  }
  return exp(-acc * uSigmaMax * dt);
}

// ---- MLP interpreter ----
float act(float x, int c) {
  if (c == 0) return x;                                         // linear
  if (c == 1) return max(x, 0.0);                               // relu
  if (c == 2) return exp(clamp(x, -80.0, 80.0));                // exp
  if (c == 3) return log(max(x, 1e-8));                         // log
  if (c == 4) return tanh(clamp(x, -15.0, 15.0));               // tanh (exact to float precision)
  if (c == 5) return 1.0 / (1.0 + exp(-clamp(x, -80.0, 80.0))); // sigmoid
  if (c == 6) return max(x, 0.0) + log(1.0 + exp(-abs(x)));     // softplus (stable form)
  if (c == 7) return abs(x);                                    // abs
  if (c == 8) return sin(x);                                    // sin
  if (c == 9) return x * x;                                     // square
  if (c == 10) return sqrt(max(x, 0.0));                        // sqrt
  return 1.0 / max(x, 1e-6);                                    // recip
}

void setc(inout vec4 v, int c, float x) {
  if (c == 0) v.x = x; else if (c == 1) v.y = x; else if (c == 2) v.z = x; else v.w = x;
}

vec4 netFetch(int i) { return texelFetch(uNet, ivec2(i & 1023, i >> 10), 0); }

#ifdef DEBUG_RAY
ivec4 gCap;          // the 4 record indices this fragment writes
vec4 gCapV;
void capture(int idx, float v) {
  if (gCap.x == idx) gCapV.x = v;
  if (gCap.y == idx) gCapV.y = v;
  if (gCap.z == idx) gCapV.z = v;
  if (gCap.w == idx) gCapV.w = v;
}
#define HEADER 8
#endif

float probeValue(int i, vec3 x, vec3 v, vec3 e2, vec3 e3) {
  vec4 A = uProbeA[i];
  vec4 B = uProbeB[i];
  int type = int(A.x);
  if (type == 0) {
    int fr = int(A.y);
    vec3 d = fr == 0 ? B.x * uSun + B.y * uLu + B.z * uLw : (fr == 1 ? B.xyz : B.x * v + B.y * e2 + B.z * e3);
    return densProbe(x + B.w * d, int(A.z), A.w);
  }
  if (type == 1) return dot(v, uSun);
  if (type == 2) return uSun.y;
  if (type == 3) return v.y;
  if (type == 4) return (x.y - uBoxMin.y) / (uBoxMax.y - uBoxMin.y);
  return B.x;   // const
}

void evalNet(vec3 x, vec3 v, vec3 e2, vec3 e3, out vec3 aSun, out vec3 aSky, out vec3 aGnd) {
  vec4 h[32];
  vec4 g[32];
  for (int i = 0; i < 32; i++) h[i] = vec4(0.0);
  for (int i = 0; i < uNumProbes; i++) {
    float p = probeValue(i, x, v, e2, e3);
    setc(h[i >> 2], i & 3, p);
#ifdef DEBUG_RAY
    capture(HEADER + i, p);
#endif
  }
#ifdef DEBUG_RAY
  int base0 = HEADER + uNumProbes;
#endif
  for (int l = 0; l < uNumLayers; l++) {
    vec4 Ld = uLayer[l];
    int off = int(Ld.x);
    int nout = int(Ld.z);
    int nin4 = int(Ld.w);
    int nout4 = (nout + 3) >> 2;
    int stride = 4 * nin4 + 2;
    for (int j4 = 0; j4 < nout4; j4++) {
      int base = off + j4 * stride;
      vec4 acc = netFetch(base + 4 * nin4);            // biases of units 4*j4 .. 4*j4+3
      vec4 codes = netFetch(base + 4 * nin4 + 1);      // their activation codes
      for (int i = 0; i < nin4; i++) {
        int q = base + 4 * i;
        acc += h[i] * mat4(netFetch(q), netFetch(q + 1), netFetch(q + 2), netFetch(q + 3));
      }
      vec4 y = vec4(act(acc.x, int(codes.x)), act(acc.y, int(codes.y)), act(acc.z, int(codes.z)), act(acc.w, int(codes.w)));
      g[j4] = y;
#ifdef DEBUG_RAY
      for (int c = 0; c < 4; c++) if (4 * j4 + c < nout) capture(base0 + 4 * j4 + c, y[c]);
#endif
    }
    for (int j = 0; j < nout4; j++) h[j] = g[j];
#ifdef DEBUG_RAY
    base0 += nout;
#endif
  }
  aSun = h[0].xyz;
  aSky = vec3(h[0].w, h[1].xy);
  aGnd = vec3(h[1].zw, h[2].x);
}

// ---- the march ----
// Sample k of a ray sits at t0 + (k + jitter) dt (computed from the index, so a march split over several
// passes gives exactly the same samples).
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

#ifdef DEBUG_RAY
float gT0, gT1, gDt, gTfinal;
vec3 gSky, gDir;
#endif

// March from step index i with state (T, L, nacc); at most `cap` evaluations. Returns true when the ray is
// finished (left the box, T < 0.002 or 512 steps); false when it stopped at the cap (resume at the same i).
bool marchSegment(Ray R, ivec2 pix, int frame, inout int i, inout float T, inout vec3 L, inout int nacc, int cap
#ifdef DEBUG_RAY
                  , int want, int sBase
#endif
) {
  if (!R.hit) return true;
  int evals = 0;
  float mu = dot(R.rd, uSun);
  float sj = rnd(pix, frame, 3u);
  for (; i < uMaxSteps; i++) {
    float t = R.t0 + (float(i) + R.jit) * R.dt;
    if (t >= R.t1) return true;
    vec3 x = R.ro + t * R.rd;
    float d = dens0(x);
    if (d > 0.002) {
      if (evals >= cap) return false;
      evals++;
      float tr = exp(-d * uSigmaMax * R.dt);
      float a = 1.0 - tr;
      vec3 aSun = vec3(0.0), aSky = vec3(0.0), aGnd = vec3(0.0);
#if defined(DEBUG_RAY)
      if (want >= 0 && nacc > want) return true;
      if (nacc == want) {
        capture(0, t); capture(1, x.x); capture(2, x.y); capture(3, x.z);
        capture(4, d); capture(5, T); capture(6, R.dt); capture(7, a);
        evalNet(x, R.rd, R.e2, R.e3, aSun, aSky, aGnd);
        vec3 S = aSun * uEsun + aSky * uLsky + aGnd * uLgnd;
        capture(sBase, S.r); capture(sBase + 1, S.g); capture(sBase + 2, S.b);
#ifdef DEBUG_EXACT
        float Ts = sunTransmittance(x, fract(sj + float(nacc) * 0.6180339887));
        capture(sBase + 3, Ts); capture(sBase + 4, phaseFn(mu));
#endif
      }
#elif defined(MODE_EXACT)
      aSun = vec3(sunTransmittance(x, fract(sj + float(nacc) * 0.6180339887)) * phaseFn(mu));
#elif defined(MODE_NET)
      evalNet(x, R.rd, R.e2, R.e3, aSun, aSky, aGnd);
#endif
#if !defined(DEBUG_RAY)
      vec3 S = aSun * uEsun + aSky * uLsky + aGnd * uLgnd;
      L += T * uAlbedo * S * a;
#endif
      T *= tr;
      nacc++;
      if (T < 0.002) { i++; return true; }
    }
  }
  return true;
}

void main() {
  ivec2 fc = ivec2(gl_FragCoord.xy);
#if defined(DEBUG_RAY)
  // fragment (col, row): row 0 = summary, row r >= 1 = accepted sample r - 1; values 4*col .. 4*col + 3
  gCap = ivec4(4 * fc.x, 4 * fc.x + 1, 4 * fc.x + 2, 4 * fc.x + 3);
  gCapV = vec4(0.0);
  int nUnits = 0;
  for (int l = 0; l < uNumLayers; l++) nUnits += int(uLayer[l].z);
  int sBase = HEADER + uNumProbes + nUnits;
  ivec2 pix = ivec2(uDebugPixel);
  Ray R = makeRay(pix, uFrame);
  int i = 0, nacc = 0;
  float T = 1.0;
  vec3 L = vec3(0.0);
  if (fc.y == 0) {
    marchSegment(R, pix, uFrame, i, T, L, nacc, 1 << 30, -1, sBase);
    vec3 bg = sky(R.rd);
    float s[12] = float[12](float(nacc), R.t0, R.t1, R.dt, T, bg.r, bg.g, bg.b, R.rd.x, R.rd.y, R.rd.z, 0.0);
    vec4 o = vec4(0.0);
    for (int c = 0; c < 4; c++) {
      int k = 4 * fc.x + c;
      if (k < 12) setc(o, c, s[k]);
    }
    fragColor = o;
  } else {
    marchSegment(R, pix, uFrame, i, T, L, nacc, 1 << 30, fc.y - 1, sBase);
    fragColor = gCapV;
  }
#else
  ivec2 pix = ivec2(fc.x, int(uRes.y) - 1 - fc.y);
  Ray R = makeRay(pix, uFrame);
#if defined(SEGMENT)
  vec4 s0 = vec4(0.0, 1.0, 0.0, 0.0);
  vec3 L = vec3(0.0);
  if (uSegFirst == 0) {
    s0 = texelFetch(uState0, fc, 0);
    L = texelFetch(uState1, fc, 0).rgb;
  }
  if (s0.x < 0.0) {          // finished in an earlier pass: carry the state over
    outState = vec4(s0.xyz, uStamp);
    outL = vec4(L, 0.0);
    return;
  }
  int i = int(s0.x);
  float T = s0.y;
  int nacc = int(s0.z);
  bool done = marchSegment(R, pix, uFrame, i, T, L, nacc, uEvalCap);
  outState = vec4(done ? -1.0 : float(i), T, float(nacc), uStamp);
  outL = vec4(L, 0.0);
#elif defined(FINALIZE)
  vec4 s0 = texelFetch(uState0, fc, 0);
  vec3 L = texelFetch(uState1, fc, 0).rgb + s0.y * uBgScale * sky(R.rd);
  // frames are numbered from uFrame0 within this accumulation; alpha carries the draw stamp (main.js)
  int n = uFrame - uFrame0;
  if (n == 0) {
    fragColor = vec4(L, uStamp);
  } else {
    vec3 prev = texelFetch(uPrev, fc, 0).rgb;
    fragColor = vec4(prev + (L - prev) / float(n + 1), uStamp);
  }
#endif
#endif
}
