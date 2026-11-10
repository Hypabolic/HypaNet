// Batch front end to the real extractor for the dev generator: reads JSONL requests on stdin, one per line,
// {"repo", "path", "src", "css"?}, and writes one JSONL line per request in the same order: the screen
// (slop.screen/1, compact, markers escaped as in the extractor's own files) or {"error": "..."}.
// `css` is the text of the file's companion stylesheet (a CSS module), parsed exactly as the extractor parses a
// repo's CSS.
//
//   node slop/generator/extract_batch.ts < requests.jsonl > screens.jsonl

import { createInterface } from 'node:readline';
import { extractSource } from '../extract/src/extract.ts';
import type { CssMap } from '../extract/src/style.ts';
import { parseCss } from '../extract/src/style.ts';

interface Req {
  repo: string;
  path: string;
  src: string;
  css?: string | null;
}

const esc = (s: string): string => s.replace(/[]/g, (c) => `\\u${c.charCodeAt(0).toString(16)}`);

const rl = createInterface({ input: process.stdin, crlfDelay: Infinity });
const out: string[] = [];
const flush = (): void => {
  if (out.length) {
    process.stdout.write(out.join(''));
    out.length = 0;
  }
};
for await (const line of rl) {
  if (!line.trim()) continue;
  let res: string;
  try {
    const req = JSON.parse(line) as Req;
    const css: CssMap = new Map();
    if (req.css) parseCss(req.css, css);
    const screen = await extractSource(req.src, { repo: req.repo, path: req.path, commit: null, licence: 'generated', css });
    res = esc(JSON.stringify(screen));
  } catch (e) {
    res = JSON.stringify({ error: String((e as Error)?.stack ?? e) });
  }
  out.push(res + '\n');
  if (out.length >= 200) flush();
}
flush();
