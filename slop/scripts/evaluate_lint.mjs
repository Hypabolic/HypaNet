// Node side of scripts/evaluate.py: walk the owner's repos, run the real extractor, the real rule layer and the real
// JS runtime (one or two compiled nets) on every UI file, in one process, the way kit/tools/lint_run.mjs does for one
// file. Owner-side only: it is never part of an author kit.
//
// stdin:  {"repos": [{"name", "root", "skip": [path relative to root, ...]}], "nets": [abs path of a slop-net.bin, ...],
//          "rules": bool}
// stdout: one JSON object per line:
//   {"kind": "repo", "name", "root", "files": N, "skipped": S, "skip_missing": [path]}   before each repo's files;
//    N counts the files linted (the skipped ones, the disclosed reference files, not included)
//   {"kind": "file", "repo", "path", "nodes": [{i, k, role, text, loc, flags}], "findings": [...], "preds": [[{index,
//    cls, logits, pointer}] per net]}                                             one per file with at least one node
//   {"kind": "error", "repo", "path", "error"}                                     a file the extractor could not read
//   {"kind": "nets", "sha256": [...]}                                              first
// File selection: .tsx .jsx .swift .html; never node_modules, .git, dist, build, .next, vendor, Pods, blog-figures,
// test or preview folders, secrets (.env*, *secret*, *credential*), or test, spec, stories and preview files.

import { lstatSync, readdirSync, readFileSync } from 'node:fs';
import { join, relative, sep } from 'node:path';
import { extractSource, isSecretPath } from '../extract/src/extract.ts';
import { loadConfig, loadRepoCss } from '../extract/src/style.ts';
import { loadNet, predictScreen } from '../lint/runtime.js';
import { runRules } from '../rules/index.js';

export const EXTS = /\.(tsx|jsx|swift|html)$/;
export const SKIP_DIRS = new Set(['node_modules', '.git', 'dist', 'build', '.next', 'vendor', 'Pods', 'blog-figures', 'out', 'coverage',
  '.turbo', '.vercel', 'DerivedData', '.build', '.expo', '.svelte-kit', '__tests__', '__mocks__', 'test', 'tests', 'spec', 'specs', 'e2e',
  'Preview Content', 'Previews', 'stories', '.storybook']);
const SKIP_DIR_RE = /(?:Tests|UITests|\.worktrees)$/;
const SKIP_FILE_RE = /(?:\.(?:test|spec|stories|story|e2e)\.[a-z]+$|(?:Tests?|Spec|Previews?)\.swift$|preview)/i;

export function listFiles(root) {
  const out = [];
  const walk = (dir) => {
    let names;
    try { names = readdirSync(dir).sort(); } catch { return; }
    for (const n of names) {
      if (SKIP_DIRS.has(n) || isSecretPath(n) || n.startsWith('.')) continue;
      const p = join(dir, n);
      let st;
      try { st = lstatSync(p); } catch { continue; }
      if (st.isDirectory()) {
        if (!SKIP_DIR_RE.test(n)) walk(p);
      } else if (st.isFile() && EXTS.test(n) && !SKIP_FILE_RE.test(n) && st.size < 2_000_000) out.push(p);
    }
  };
  walk(root);
  return out;
}

async function main() {
  const req = JSON.parse(readFileSync(0, 'utf8'));
  const nets = (req.nets || []).map((p) => loadNet(readFileSync(p)));
  const put = (o) => process.stdout.write(`${JSON.stringify(o)}\n`);
  put({ kind: 'nets', sha256: nets.map((n) => n.sha256) });
  for (const repo of req.repos) {
    const skip = new Set(repo.skip || []);
    const all = listFiles(repo.root);
    const rel = (f) => relative(repo.root, f).split(sep).join('/');
    const files = all.filter((f) => !skip.has(rel(f)));
    const found = new Set(all.map(rel));
    put({ kind: 'repo', name: repo.name, root: repo.root, files: files.length, skipped: all.length - files.length,
      skip_missing: [...skip].filter((p) => !found.has(p)) });
    const css = loadRepoCss(repo.root);
    const config = loadConfig(repo.root);
    for (const f of files) {
      const path = rel(f);
      let screen;
      try {
        screen = await extractSource(readFileSync(f, 'utf8'), { repo: repo.name, path, commit: null, licence: null, css, config });
      } catch (e) {
        put({ kind: 'error', repo: repo.name, path, error: String(e && e.message ? e.message : e).split('\n')[0] });
        continue;
      }
      if (!screen.nodes.length) continue;
      const findings = req.rules === false ? [] : runRules(screen);
      const preds = nets.map((net) => predictScreen(net, screen).map((p) => ({ index: p.index, cls: p.cls, logits: p.logits, pointer: p.pointer })));
      const nodes = screen.nodes.map((n) => ({ i: n.i, k: n.k, role: n.role, text: n.text, loc: n.loc, flags: n.flags }));
      put({ kind: 'file', repo: repo.name, path, nodes, findings, preds });
    }
  }
}

if (process.argv[1] && import.meta.url === new URL(`file://${process.argv[1]}`).href) await main();
