// slop-extract: node slop/extract/cli.js <file|dir> --repo NAME [--commit SHA] [--licence L] --out DIR
//   [--root DIR] [--keep-empty] [--summary FILE]

import { mkdirSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { extractFile, findRoot, isSecretPath, listSourceFiles, repoContext, serialise } from './extract.ts';
import { ROLES } from './schema.ts';

interface Args {
  input: string;
  repo: string;
  commit: string | null;
  licence: string | null;
  out: string;
  root: string | null;
  keepEmpty: boolean;
  summary: string | null;
}

function usage(msg?: string): never {
  if (msg) process.stderr.write(`slop-extract: ${msg}\n`);
  process.stderr.write('usage: node slop/extract/cli.js <file|dir> --repo NAME [--commit SHA] [--licence L] --out DIR [--root DIR] [--keep-empty] [--summary FILE]\n');
  process.exit(2);
}

function parseArgs(argv: string[]): Args {
  const a: Partial<Args> = { commit: null, licence: null, root: null, keepEmpty: false, summary: null };
  for (let i = 0; i < argv.length; i++) {
    const k = argv[i];
    const val = (): string => {
      const v = argv[++i];
      if (v === undefined) usage(`${k} needs a value`);
      return v;
    };
    if (k === '--repo') a.repo = val();
    else if (k === '--commit') a.commit = val();
    else if (k === '--licence' || k === '--license') a.licence = val();
    else if (k === '--out') a.out = val();
    else if (k === '--root') a.root = val();
    else if (k === '--summary') a.summary = val();
    else if (k === '--keep-empty') a.keepEmpty = true;
    else if (k === '-h' || k === '--help') usage();
    else if (k.startsWith('--')) usage(`unknown option ${k}`);
    else if (a.input === undefined) a.input = k;
    else usage(`unexpected argument ${k}`);
  }
  if (!a.input) usage('no input');
  if (!a.repo) usage('--repo is required');
  if (!a.out) usage('--out is required');
  return a as Args;
}

const args = parseArgs(process.argv.slice(2));
const input = resolve(args.input);
let isDir: boolean;
try {
  isDir = statSync(input).isDirectory();
} catch {
  usage(`cannot read ${args.input}`);
}
const root = args.root ? resolve(args.root) : isDir ? input : findRoot(input);
const files = isDir ? listSourceFiles(input) : [input];
const ctx = repoContext(root);
const roles: Record<string, number> = Object.fromEntries(ROLES.map((r) => [r, 0]));
const summary = {
  repo: args.repo, files: files.length, parsed: 0, failed: 0, empty: 0, written: 0, nodes: 0, roles,
  failures: [] as { path: string; error: string }[],
};
for (const f of files) {
  const base = f.split('/').pop() ?? f;
  if (isSecretPath(base)) continue;
  try {
    const screen = await extractFile(f, ctx, { repo: args.repo, commit: args.commit, licence: args.licence });
    summary.parsed++;
    if (screen.nodes.length === 0) {
      summary.empty++;
      if (!args.keepEmpty) continue;
    }
    for (const n of screen.nodes) roles[n.role]++;
    summary.nodes += screen.nodes.length;
    const dest = join(resolve(args.out), `${screen.source.path}.json`);
    mkdirSync(dirname(dest), { recursive: true });
    writeFileSync(dest, serialise(screen));
    summary.written++;
  } catch (err) {
    summary.failed++;
    const rel = f.startsWith(root) ? f.slice(root.length + 1) : f;
    summary.failures.push({ path: rel, error: String((err as Error)?.message ?? err).split('\n')[0] });
  }
}
for (const k of Object.keys(roles)) if (roles[k] === 0) delete roles[k];
const line = `${args.repo}: ${summary.files} files, ${summary.parsed} parsed, ${summary.failed} failed, ${summary.written} screens, ${summary.nodes} nodes\n`;
process.stderr.write(line);
for (const f of summary.failures) process.stderr.write(`  FAIL ${f.path}: ${f.error}\n`);
if (args.summary) writeFileSync(args.summary, `${JSON.stringify(summary, null, 1)}\n`);
