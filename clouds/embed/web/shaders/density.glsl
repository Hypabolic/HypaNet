// HypaNet Clouds embed: the scene density and the blurred-density probe, shared by the march and the bake passes.
// Identical to renderer/shaders/march.frag.glsl (dens0, densLevel, densProbe): the compiled network and the bake
// functions (gen/net_eval.glsl, gen/bake.glsl) call densProbe(x, lo, t) with the harness's semantics.
// Also declares the light frame (uSun, uLu, uLw) the generated code reads. Include after the precision statements.
#define N_LEVELS 8

uniform sampler3D uDensity;           // level 0: raw grid, R8, linear filtering, clamp to edge
uniform sampler3D uAtlas;             // blur levels 1..7 (pyramid.js), R8, linear
uniform vec4 uLvlScale[N_LEVELS];     // atlas tc = q * scale + offset, q in voxel units
uniform vec4 uLvlOffset[N_LEVELS];
uniform vec3 uBoxMin;
uniform vec3 uBoxMax;
uniform vec3 uVoxel;                  // voxel size (m)
uniform vec3 uDims;                   // grid dims (nx, ny, nz)

uniform vec3 uSun;                    // toward-sun unit vector s
uniform vec3 uLu;                     // light frame u = normalize(cross(s, world_up)) (fallback (1, 0, 0))
uniform vec3 uLw;                     // light frame w = cross(u, s)

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
