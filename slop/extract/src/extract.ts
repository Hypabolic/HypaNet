// Library entry points: one source file in, one slop.screen/1 object out.

import { existsSync, lstatSync, readdirSync, readFileSync } from 'node:fs';
import { dirname, extname, join, relative, resolve, sep } from 'node:path';
import { htmlToIR } from './html.ts';
import { infer } from './infer.ts';
import type { El } from './ir.ts';
import { resetIds } from './ir.ts';
import { jsxToIR } from './jsx.ts';
import type { Framework } from './schema.ts';
import type { Screen } from './screen.ts';
import { buildScreen } from './screen.ts';
import type { CssMap, SlopConfig } from './style.ts';
import { emptyConfig, loadConfig, loadRepoCss } from './style.ts';
import { swiftToIR } from './swift.ts';

export { serialise } from './screen.ts';
export type { Screen } from './screen.ts';

export interface ExtractOptions {
  repo: string;
  path: string; // path recorded in the screen, relative to the repo root, "/" separators
  commit?: string | null;
  licence?: string | null;
  css?: CssMap;
  config?: SlopConfig;
}

export function frameworkOf(path: string): Framework | null {
  const ext = extname(path).toLowerCase();
  if (ext === '.tsx' || ext === '.jsx' || ext === '.js' || ext === '.mjs') return 'tsx';
  if (ext === '.html' || ext === '.htm') return 'html';
  if (ext === '.swift') return 'swift';
  return null;
}

export async function extractSource(src: string, opts: ExtractOptions): Promise<Screen> {
  const fw = frameworkOf(opts.path);
  if (!fw) throw new Error(`unsupported file type: ${opts.path}`);
  resetIds();
  let css = opts.css ?? new Map();
  let root: El;
  if (fw === 'tsx') root = jsxToIR(src, opts.path);
  else if (fw === 'html') {
    const r = htmlToIR(src, css);
    root = r.root;
    css = r.css;
  } else root = await swiftToIR(src);
  const { nodes, clusters } = infer(root, { fw, css, cfg: opts.config ?? emptyConfig() });
  return buildScreen(src, fw, {
    repo: opts.repo, path: opts.path, commit: opts.commit ?? null, licence: opts.licence ?? null,
  }, nodes, clusters);
}

// ---------------------------------------------------------------------------------------------------------------
// Walking a repo

const SKIP_DIRS = new Set(['node_modules', '.next', 'dist', 'build', '.git', 'out', 'coverage', '.turbo', '.vercel', 'Pods', 'DerivedData', '.build', 'vendor', '__tests__', '__mocks__', '.cloudflare', '.expo', '.svelte-kit']);

/** Never read: secrets and credentials. */
export function isSecretPath(name: string): boolean {
  return /^\.env/.test(name) || /secret|credential/i.test(name) || name === '.cloudflare';
}

export function listSourceFiles(root: string): string[] {
  const out: string[] = [];
  const walk = (dir: string): void => {
    let names: string[];
    try {
      names = readdirSync(dir).sort();
    } catch {
      return;
    }
    for (const n of names) {
      if (SKIP_DIRS.has(n) || n.endsWith('.worktrees') || isSecretPath(n)) continue;
      const p = join(dir, n);
      let st;
      try {
        st = lstatSync(p);
      } catch {
        continue;
      }
      if (st.isDirectory()) {
        const rel = relative(root, p).split(sep).join('/');
        if (/(^|\/)components\/ui$/.test(rel)) continue;
        walk(p);
      } else if (frameworkOf(n) && !/\.(test|spec|stories|story|d)\.[a-z]+$/.test(n) && !/\.min\.js$/.test(n) && st.size < 2_000_000) {
        out.push(p);
      }
    }
  };
  walk(root);
  return out;
}

/** The repo root for a single file: the nearest ancestor holding .git or package.json (else its directory). */
export function findRoot(file: string): string {
  let d = dirname(resolve(file));
  for (;;) {
    if (existsSync(join(d, '.git')) || existsSync(join(d, 'package.json')) || existsSync(join(d, 'Package.swift'))) return d;
    const up = dirname(d);
    if (up === d) return dirname(resolve(file));
    d = up;
  }
}

export interface RepoContext {
  root: string;
  css: CssMap;
  config: SlopConfig;
}

export function repoContext(root: string): RepoContext {
  return { root, css: loadRepoCss(root), config: loadConfig(root) };
}

export async function extractFile(file: string, ctx: RepoContext, meta: { repo: string; commit?: string | null; licence?: string | null }): Promise<Screen> {
  const src = readFileSync(file, 'utf8');
  const path = relative(ctx.root, resolve(file)).split(sep).join('/');
  return extractSource(src, { ...meta, path, css: ctx.css, config: ctx.config });
}
