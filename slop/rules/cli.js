#!/usr/bin/env node
// Rule-layer CLI (SPEC.md §8).
//
//   node slop/rules/cli.js <screen.json|dir> [...] [--format text|json] [--rules id,id]
//
// A directory is walked recursively for `.json` files; files whose `schema` is not slop.screen/1 are skipped and
// counted. `--format json` prints one document on stdout:
//   {"schema": "slop.rules/1", "files": [{"file", "screen", "findings": [...]}], "skipped": N, "errors": [{"file", "error"}]}
// with each finding {"rule", "code", "scope", "condition"?, "nodes", "locs", "target"?, "message"}. Exit status: 0 no
// findings, 1 findings, 2 a file could not be read or checked (findings for the other files are still printed).

import { readdirSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { RULE_IDS, runRules } from './index.js';

function usage(msg) {
  if (msg) process.stderr.write(`${msg}\n`);
  process.stderr.write('usage: node rules/cli.js <screen.json|dir> [...] [--format text|json] [--rules id,id]\n');
  process.exit(2);
}

function parseArgs(argv) {
  const opts = { inputs: [], format: 'text', only: null };
  for (let a = 0; a < argv.length; a++) {
    const arg = argv[a];
    if (arg === '--format') opts.format = argv[++a];
    else if (arg.startsWith('--format=')) opts.format = arg.slice(9);
    else if (arg === '--rules') opts.only = (argv[++a] || '').split(',').filter(Boolean);
    else if (arg.startsWith('--rules=')) opts.only = arg.slice(8).split(',').filter(Boolean);
    else if (arg === '--list') opts.list = true;
    else if (arg === '-h' || arg === '--help') usage();
    else if (arg.startsWith('-')) usage(`unknown option ${arg}`);
    else opts.inputs.push(arg);
  }
  if (!['text', 'json'].includes(opts.format)) usage(`unknown format ${opts.format}`);
  if (opts.only) for (const r of opts.only) if (!RULE_IDS.includes(r)) usage(`unknown rule ${r} (rules: ${RULE_IDS.join(', ')})`);
  if (!opts.inputs.length && !opts.list) usage('no input');
  return opts;
}

function walk(p, out) {
  const st = statSync(p);
  if (st.isDirectory()) {
    for (const name of readdirSync(p).sort()) {
      if (name.startsWith('.') || name === 'node_modules') continue;
      walk(path.join(p, name), out);
    }
  } else if (p.endsWith('.json')) out.push(p);
  return out;
}

const opts = parseArgs(process.argv.slice(2));
if (opts.list) {
  for (const id of RULE_IDS) process.stdout.write(`${id}\n`);
  process.exit(0);
}

const report = { schema: 'slop.rules/1', files: [], skipped: 0, errors: [] };
for (const input of opts.inputs) {
  let files;
  try {
    files = statSync(input).isDirectory() ? walk(input, []) : [input];
  } catch (e) {
    report.errors.push({ file: input, error: e.message });
    continue;
  }
  for (const file of files) {
    let screen;
    try {
      screen = JSON.parse(readFileSync(file, 'utf8'));
    } catch (e) {
      report.errors.push({ file, error: `cannot read: ${e.message}` });
      continue;
    }
    if (!screen || screen.schema !== 'slop.screen/1') {
      report.skipped++;
      continue;
    }
    try {
      report.files.push({ file, screen: screen.id ?? null, findings: runRules(screen, { only: opts.only }) });
    } catch (e) {
      report.errors.push({ file, error: e.message });
    }
  }
}

const total = report.files.reduce((s, f) => s + f.findings.length, 0);
if (opts.format === 'json') {
  process.stdout.write(`${JSON.stringify(report)}\n`);
} else {
  const lines = [];
  for (const f of report.files) {
    if (!f.findings.length) continue;
    lines.push(f.file);
    for (const x of f.findings) {
      const at = x.locs[0] ? `${x.locs[0][0]}:${x.locs[0][1]}` : '?';
      const more = x.locs.length > 1 ? ` (+${x.locs.length - 1} more)` : '';
      lines.push(`  ${at.padEnd(7)} ${x.code.padEnd(4)} ${x.message}${more}   ${x.rule}`);
    }
  }
  const withFindings = report.files.filter((f) => f.findings.length).length;
  lines.push(`${total} finding${total === 1 ? '' : 's'} in ${withFindings} of ${report.files.length} screen${report.files.length === 1 ? '' : 's'}`
    + (report.skipped ? `, ${report.skipped} non-screen file${report.skipped === 1 ? '' : 's'} skipped` : ''));
  for (const e of report.errors) lines.push(`error: ${e.file}: ${e.error}`);
  process.stdout.write(`${lines.join('\n')}\n`);
}
process.exit(report.errors.length ? 2 : total ? 1 : 0);
