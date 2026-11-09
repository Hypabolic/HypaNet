// HypaNet Clouds embed: bake driver. One draw per z layer of a 3D RGBA target; texel (i, j, layer) holds the baked
// first-layer combination(s) BAKE_FN (gen/bake.glsl: bake0, bake1, ...) at its cell centre,
// boxMin + (i + 0.5, j + 0.5, layer + 0.5) * size / dims, the grid gen/net_eval.glsl reads with texture().
// clouds.js assembles: precision statements, shaders/density.glsl, gen/bake.glsl, then this file (BAKE_FN defined).
// The same function is baked at a coarser grid while the sun is being dragged (the reads are resolution-free).
uniform vec3 uBakeDims;
uniform float uBakeLayer;
out vec4 fragColor;

void main() {
  vec3 x = uBoxMin + vec3(gl_FragCoord.xy, uBakeLayer + 0.5) / uBakeDims * (uBoxMax - uBoxMin);
  fragColor = BAKE_FN(x);
}
