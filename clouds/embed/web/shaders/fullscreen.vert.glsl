// Full-screen pass: a 2 x 2 plane in clip space (three.js PlaneGeometry(2, 2)).
precision highp float;
in vec3 position;
void main() {
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
