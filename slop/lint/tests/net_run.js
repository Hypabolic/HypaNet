// Test helper: reads {"net": path to slop-net.bin, "screens": [...]} as JSON on stdin and prints the JS runtime's
// predictions per screen (class, logits, pointer, BOS weight per node) as JSON, for the Python parity tests.
import { readFileSync } from 'node:fs';
import { loadNet, predictScreen } from '../runtime.js';

const req = JSON.parse(readFileSync(0, 'utf8'));
const net = loadNet(readFileSync(req.net));
const t0 = process.hrtime.bigint();
const screens = req.screens.map((s) => predictScreen(net, s).map((p) => ({
  index: p.index, cls: p.cls, logits: p.logits, pointer: p.pointer, bos: p.bosWeight,
})));
const ms = Number(process.hrtime.bigint() - t0) / 1e6;
process.stdout.write(JSON.stringify({ sha256: net.sha256, pointerHead: net.pointerHead, screens, ms }));
