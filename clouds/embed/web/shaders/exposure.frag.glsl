// HypaNet Clouds embed: auto-exposure on the GPU (no readback). One texel in, one texel out (ping-pong):
//   key   = exp(mean log(Y + 1e-6)) over a 48 x 30 grid of the current image, Y = luminance clamped at 100 (the sun
//           disk is a handful of pixels at ~1e5; video/make_video.py drops the brightest 0.5 % instead);
//   logE* = log(0.5) - 0.75 (log key - log 0.12), as make_video.py's post: follows 3/4 of a change in brightness,
//           so dusk still darkens; a golden-hour frame lands near the references' exposure;
//   out.r = mix(previous log exposure, clamp(logE*), uAdapt), uAdapt = 1 - exp(-dt / 0.4 s) (1 to snap).
// out.r = uFixedLogE instead when uFixed == 1.
uniform sampler2D uImage;             // linear RGB, RGBA32F
uniform sampler2D uPrevExpo;          // 1 x 1, r = log exposure
uniform float uAdapt;
uniform float uBias;                  // log exposure offset (viewer's choice; 0)
uniform int uFixed;
uniform float uFixedLogE;
out vec4 fragColor;

void main() {
  if (uFixed == 1) {
    fragColor = vec4(uFixedLogE, uFixedLogE, 0.0, 1.0);
    return;
  }
  ivec2 sz = textureSize(uImage, 0);
  float s = 0.0;
  for (int j = 0; j < 30; j++) {
    for (int i = 0; i < 48; i++) {
      ivec2 p = ivec2((vec2(float(i), float(j)) + 0.5) / vec2(48.0, 30.0) * vec2(sz));
      vec3 c = texelFetch(uImage, p, 0).rgb;
      float y = dot(c, vec3(0.2126, 0.7152, 0.0722));
      y = (isnan(y) || isinf(y)) ? 0.0 : clamp(y, 0.0, 100.0);
      s += log(y + 1e-6);
    }
  }
  float logKey = s / 1440.0;
  float target = clamp(log(0.5) - 0.75 * (logKey - log(0.12)) + uBias, log(0.03), log(60.0));
  float prev = texelFetch(uPrevExpo, ivec2(0), 0).r;
  fragColor = vec4(mix(prev, target, uAdapt), target, 0.0, 1.0);
}
