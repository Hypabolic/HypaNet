// Interactive viewer: orbit camera, sun sliders, exposure, progressive accumulation, network.json loading.
// URL parameters: ?net=URL (network to load), ?shot=front_noon, ?scene=URL of scene/out/, ?harness=1 (adds
// the harness-only exact single-scattering mode).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CloudRenderer } from './main.js';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const harness = params.get('harness') === '1';
const canvas = $('c');
const msg = (text) => { $('msg').style.display = text ? 'flex' : 'none'; $('msg').textContent = text || ''; };
const MAX_FRAMES = 256;

const r = new CloudRenderer({ canvas, harness, sceneBase: params.get('scene') || '../scene/out/' });
let cam, controls;
const state = { mode: 'net', exposure: 0.315, scale: 1, dirty: true, shown: -1, toneDirty: false, error: null,
  netName: null, frameMs: 0, t0: 0 };

async function init() {
  const info = await r.load();
  const shots = r.shots;
  if (harness) $('mode').insertAdjacentHTML('beforeend', '<option value="exact">exact single scattering (harness)</option>');
  for (const id of Object.keys(shots.exposure_table)) $('shot').insertAdjacentHTML('beforeend', `<option value="${id}">${id.replace('_', ' · ')}</option>`);
  for (const p of Object.keys(shots.presets)) {
    const b = document.createElement('button');
    b.textContent = `${p} ${shots.presets[p].elevation_deg}°`;
    b.onclick = () => setPreset(p);
    $('presets').appendChild(b);
  }
  cam = new THREE.PerspectiveCamera(30, 1.6, 1, 1e6);
  controls = new OrbitControls(cam, canvas);
  controls.enableDamping = false;
  controls.addEventListener('change', () => { state.dirty = true; });
  setResolution(1);
  await applyShot(params.get('shot') || 'front_noon');
  $('shot').value = params.get('shot') || 'front_noon';
  bindUI();
  if (params.get('net')) {
    try { const t = await (await fetch(params.get('net'))).text(); loadNet(t, params.get('net').split('/').pop()); }
    catch (e) { msg(`could not load ${params.get('net')}: ${e.message}`); }
  }
  $('stats').textContent = `${info.gl.renderer}`;
  requestAnimationFrame(loop);
}

function setResolution(scale) {
  state.scale = scale;
  const w = Math.round(640 * scale), h = Math.round(400 * scale);
  r.setResolution(w, h);
  canvas.width = w; canvas.height = h;
  state.dirty = true;
}

async function applyShot(id) {
  const [shot, preset] = id.split('_');
  const s = r.shots.shots[shot];
  cam.fov = s.fov_y_deg;
  cam.aspect = 1.6;
  cam.near = 1; cam.far = 1e6;
  cam.position.set(...s.position);
  cam.up.set(...s.up);
  controls.target.set(...s.look_at);
  cam.updateProjectionMatrix();
  controls.update();
  setExposure(r.shots.exposure_table[id]);
  await setPreset(preset);
}

async function setPreset(p) {
  const P = r.shots.presets[p];
  $('elev').value = P.elevation_deg; $('azim').value = P.azimuth_deg;
  await r.setPreset(p);
  showSun();
  state.dirty = true;
}

async function setSunFromSliders() {
  await r.setSun(+$('elev').value, +$('azim').value);
  showSun();
  state.dirty = true;
}

function showSun() {
  $('elevv').textContent = `${(+$('elev').value).toFixed(1)}°`;
  $('azimv').textContent = `${(+$('azim').value).toFixed(0)}°`;
}

function setExposure(e) {
  state.exposure = e;
  $('expo').value = Math.log2(e);
  $('expov').textContent = e.toFixed(3);
  state.toneDirty = true;
}

function loadNet(text, name) {
  try {
    const info = r.setNetwork(text);
    state.netName = `${name}: ${info.probes} probes, layers [${info.layers}], ${info.params} params`;
    state.error = null;
    msg('');
    state.dirty = true;
  } catch (e) {
    msg(`network error:\n${e.message}`);
  }
}

function bindUI() {
  $('shot').onchange = () => applyShot($('shot').value);
  $('elev').oninput = setSunFromSliders;
  $('azim').oninput = setSunFromSliders;
  $('expo').oninput = () => setExposure(Math.pow(2, +$('expo').value));
  $('res').onchange = () => setResolution(+$('res').value);
  $('mode').onchange = () => { state.mode = $('mode').value; state.dirty = true; msg(''); };
  $('file').onchange = async () => { const f = $('file').files[0]; if (f) loadNet(await f.text(), f.name); };
  canvas.addEventListener('dragover', (e) => e.preventDefault());
  canvas.addEventListener('drop', async (e) => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f) loadNet(await f.text(), f.name);
  });
}

function applyCamera() {
  cam.updateMatrixWorld();
  const m = cam.matrixWorld;
  const right = new THREE.Vector3().setFromMatrixColumn(m, 0).normalize();
  const up = new THREE.Vector3().setFromMatrixColumn(m, 1).normalize();
  const fwd = new THREE.Vector3().setFromMatrixColumn(m, 2).normalize().negate();
  r.setCameraBasis(cam.position, fwd, right, up, cam.fov);
}

function loop() {
  requestAnimationFrame(loop);
  if (state.error) return;
  if (state.mode === 'net' && !r.net) {
    msg('Load a network.json (file picker or drop it here).' + (harness ? '\nOr pick the exact single-scattering mode.' : ''));
    return;
  }
  try {
    let reset = false;
    if (state.dirty) { applyCamera(); reset = true; state.dirty = false; state.t0 = performance.now(); state.shown = 0; }
    let n = state.shown;
    if (reset || n < MAX_FRAMES) n = r.renderProgressive(state.mode, reset, 20);
    if ((n !== state.shown && n > 0) || state.toneDirty) {
      if (n > 0) r.toneMap(state.exposure, true);
      if (n !== state.shown) state.frameMs = (performance.now() - state.t0) / n;
      state.shown = n;
      state.toneDirty = false;
    }
    const label = state.mode === 'exact' ? 'exact single scattering' : state.netName;
    $('stats').textContent = `${label} · ${n} frame${n === 1 ? '' : 's'} (${state.frameMs.toFixed(0)} ms/frame)` +
      (n >= MAX_FRAMES ? ' · converged' : '');
  } catch (e) {
    state.error = e;
    msg(`render error:\n${e.message}\nReload the page.`);
  }
}

init().catch((e) => msg(`failed to start: ${e.message}`));
