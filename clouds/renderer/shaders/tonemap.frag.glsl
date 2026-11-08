// SPEC section 3 tone map, identical to scene/tonemap.py:
//   c = exposure * L; ACES fit (Narkowicz) clamped to [0, 1]; sRGB OETF; 8-bit round half up.
// Non-finite radiance is treated as 0 (tonemap.py: nan_to_num(nan=0, posinf=0, neginf=0)).
precision highp float;
precision highp sampler2D;
uniform sampler2D uAccum;   // linear RGB radiance (RGBA32F), running mean of frames
uniform float uExposure;
out vec4 fragColor;

vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }

float finite0(float v) { return (isnan(v) || isinf(v)) ? 0.0 : v; }

void main() {
  vec3 L = texelFetch(uAccum, ivec2(gl_FragCoord.xy), 0).rgb;
  L = vec3(finite0(L.r), finite0(L.g), finite0(L.b));
  // clamp at 1e4 (ACES is already clamped to 1 there) so x (2.51 x + 0.03) cannot overflow float32
  vec3 c = min(max(uExposure * L, vec3(0.0)), vec3(1e4));
  vec3 a = aces(c);
  // sRGB OETF exactly as tonemap.py: x <= 0.0031308 -> 12.92 x, else 1.055 x^(1/2.4) - 0.055
  vec3 lo = 12.92 * a;
  vec3 hi = 1.055 * pow(max(a, vec3(0.0031308)), vec3(1.0 / 2.4)) - 0.055;
  vec3 y = vec3(a.r <= 0.0031308 ? lo.r : hi.r, a.g <= 0.0031308 ? lo.g : hi.g, a.b <= 0.0031308 ? lo.b : hi.b);
  fragColor = vec4(clamp(floor(y * 255.0 + 0.5), 0.0, 255.0) / 255.0, 1.0);
}
