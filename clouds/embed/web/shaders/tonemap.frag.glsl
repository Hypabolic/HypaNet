// HypaNet Clouds embed: display pass. Upsamples the linear image (bilinear, texelFetch: RGBA32F need not be
// filterable), adds the sun glare, applies the auto-exposure and the SPEC tone map (renderer/shaders/tonemap.frag.glsl:
// ACES fit clamped to [0, 1], sRGB OETF), with +-0.5 LSB of dither instead of the reference's rounding.
//
// Glare (video/make_video.py's three Gaussians, 0.8 %, 0.3 % and 0.1 % of the energy at 0.125, 0.83 and 5 degrees)
// is drawn analytically around the sun from the irradiance of the visible part of the disk: uSunE (disk-averaged E
// at the camera's altitude) times the visible fraction from sunvis.frag.glsl. So it fades as the cloud covers the
// sun and still glows when the sun is just off screen.
uniform sampler2D uImage;             // current linear image (RGBA32F)
uniform sampler2D uHold;              // previous image, cross-faded out over the first frames of an accumulation
uniform float uHoldMix;               // weight of uHold (0: current only)
uniform sampler2D uExpo;              // 1 x 1, r = log exposure (exposure.frag.glsl)
uniform sampler2D uSunVis;            // 1 x 1, r = visible fraction of the sun disk
uniform vec3 uSunE;                   // sun irradiance at the camera (disk-averaged, RGB)
uniform float uGlare;                 // 1: glare on
uniform vec3 uSun;
uniform vec3 uCamFwd;
uniform vec3 uCamRight;
uniform vec3 uCamUp;
uniform vec2 uTan;
uniform vec2 uOutRes;                 // canvas size in pixels
uniform int uFrame;
out vec4 fragColor;

vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }

vec3 bilinear(sampler2D tex, vec2 uv) {
  ivec2 sz = textureSize(tex, 0);
  vec2 pc = uv * vec2(sz) - 0.5;
  vec2 f = floor(pc);
  vec2 w = pc - f;
  ivec2 a = clamp(ivec2(f), ivec2(0), sz - 1);
  ivec2 b = clamp(ivec2(f) + 1, ivec2(0), sz - 1);
  vec3 c00 = texelFetch(tex, a, 0).rgb;
  vec3 c10 = texelFetch(tex, ivec2(b.x, a.y), 0).rgb;
  vec3 c01 = texelFetch(tex, ivec2(a.x, b.y), 0).rgb;
  vec3 c11 = texelFetch(tex, b, 0).rgb;
  return mix(mix(c00, c10, w.x), mix(c01, c11, w.x), w.y);
}

float hash(vec2 p, float k) {
  vec3 q = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973) + k);
  q += dot(q, q.yzx + 33.33);
  return fract((q.x + q.y) * q.z);
}

float finite0(float v) { return (isnan(v) || isinf(v)) ? 0.0 : v; }

void main() {
  vec2 uv = gl_FragCoord.xy / uOutRes;
  vec3 L = bilinear(uImage, uv);
  if (uHoldMix > 0.0) L = mix(L, bilinear(uHold, uv), uHoldMix);
  L = vec3(finite0(L.r), finite0(L.g), finite0(L.b));
  if (uGlare > 0.0) {
    vec3 d = normalize(uCamFwd + (2.0 * uv.x - 1.0) * uTan.x * uCamRight + (2.0 * uv.y - 1.0) * uTan.y * uCamUp);
    float th = 2.0 * asin(min(1.0, 0.5 * length(d - uSun)));
    const vec3 SIG = vec3(0.125, 0.83, 5.0) * 0.017453292519943295;
    const vec3 AMP = vec3(0.008, 0.003, 0.001);
    vec3 g = AMP / (6.283185307179586 * SIG * SIG) * exp(-0.5 * th * th / (SIG * SIG));
    float vis = texelFetch(uSunVis, ivec2(0), 0).r;
    L += uGlare * vis * (g.x + g.y + g.z) * uSunE;
  }
  float e = exp(texelFetch(uExpo, ivec2(0), 0).r);
  vec3 c = min(max(e * L, vec3(0.0)), vec3(1e4));
  vec3 a = aces(c);
  vec3 lo = 12.92 * a;
  vec3 hi = 1.055 * pow(max(a, vec3(0.0031308)), vec3(1.0 / 2.4)) - 0.055;
  vec3 y = vec3(a.r <= 0.0031308 ? lo.r : hi.r, a.g <= 0.0031308 ? lo.g : hi.g, a.b <= 0.0031308 ? lo.b : hi.b);
  // triangular dither of one 8-bit step (sky gradients band without it)
  float n = hash(gl_FragCoord.xy, float(uFrame & 63) * 0.618) + hash(gl_FragCoord.yx + 17.0, float(uFrame & 63) * 0.382) - 1.0;
  fragColor = vec4(clamp(y + n / 255.0, 0.0, 1.0), 1.0);
}
