// Harness side of the author's ./lint (and of ./tok, ./score and ./inspect on a source file): the real extractor,
// the real rule layer and the real JS runtime, unchanged. Called only by kit/tools/author_tool.py, which has already
// checked every path against the author directory.
//
// stdin: {"file": abs path | null, "root": dir for stylesheets and slop.config.json, "relpath": path recorded in the
//         screen, "screen": a slop.screen/1 object | null, "net": abs path of a slop-net.bin | null, "rules": bool}
// stdout: {"screen", "findings" (rule layer), "preds" (network, or null), "network" (sha256 or null), "ms"}

import { readFileSync } from 'node:fs';
import { extractSource } from '../../extract/src/extract.ts';
import { loadConfig, loadRepoCss } from '../../extract/src/style.ts';
import { runRules } from '../../rules/index.js';
import { loadNet, predictScreen } from '../../lint/runtime.js';

const req = JSON.parse(readFileSync(0, 'utf8'));
let screen = req.screen;
if (!screen) {
  const src = readFileSync(req.file, 'utf8');
  screen = await extractSource(src, {
    repo: 'author', path: req.relpath, commit: null, licence: null,
    css: loadRepoCss(req.root), config: loadConfig(req.root),
  });
}
const findings = req.rules ? runRules(screen) : [];
let preds = null;
let network = null;
let ms = 0;
if (req.net) {
  const net = loadNet(readFileSync(req.net));
  network = net.sha256;
  const t0 = process.hrtime.bigint();
  preds = predictScreen(net, screen).map((p) => ({
    index: p.index, cls: p.cls, logits: p.logits, pointer: p.pointer, bos: p.bosWeight, window: p.window,
  }));
  ms = Number(process.hrtime.bigint() - t0) / 1e6;
}
process.stdout.write(JSON.stringify({ screen, findings, preds, network, ms }));
