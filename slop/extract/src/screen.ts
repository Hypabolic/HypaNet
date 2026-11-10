// Screen JSON (slop.screen/1): locations, id and a fixed byte-for-byte serialisation.

import { createHash } from 'node:crypto';
import type { OutCluster, OutNode } from './infer.ts';
import type { ClusterKind, Flag, Framework, Role } from './schema.ts';
import { SCHEMA } from './schema.ts';

export interface Source {
  repo: string;
  path: string;
  commit: string | null;
  licence: string | null;
}

export interface ScreenNode {
  i: number;
  k: number;
  role: Role;
  text: string;
  flags: Flag[];
  loc: [number, number, number, number];
}

export interface Screen {
  schema: string;
  id: string;
  framework: Framework;
  source: Source;
  clusters: { k: number; kind: ClusterKind; line: number }[];
  nodes: ScreenNode[];
}

/**
 * Converts UTF-16 offsets to 1-based [line, column]. Lines end at "\n"; columns count Unicode code points, so a
 * Python labeller can slice str directly. The end column is exclusive (one past the last character).
 */
export class LineMap {
  private starts: number[] = [0];
  private src: string;
  constructor(src: string) {
    this.src = src;
    for (let i = 0; i < src.length; i++) if (src.charCodeAt(i) === 10) this.starts.push(i + 1);
  }
  pos(off: number): [number, number] {
    let lo = 0;
    let hi = this.starts.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (this.starts[mid] <= off) lo = mid;
      else hi = mid - 1;
    }
    const lineStart = this.starts[lo];
    let col = 1;
    for (let i = lineStart; i < off; i++) {
      const c = this.src.charCodeAt(i);
      if (c >= 0xd800 && c <= 0xdbff && i + 1 < off) i++;
      col++;
    }
    return [lo + 1, col];
  }
}

export function screenId(repo: string, path: string, src: string): string {
  const h = createHash('sha256');
  h.update(`${SCHEMA}\n${repo}\n${path}\n`);
  h.update(createHash('sha256').update(src, 'utf8').digest('hex'));
  return h.digest('hex').slice(0, 16);
}

export function buildScreen(
  src: string, framework: Framework, source: Source, nodes: OutNode[], clusters: OutCluster[],
): Screen {
  const lm = new LineMap(src);
  return {
    schema: SCHEMA,
    id: screenId(source.repo, source.path, src),
    framework,
    source,
    clusters: clusters.map((c) => ({ k: c.k, kind: c.kind, line: lm.pos(c.s)[0] })),
    nodes: nodes.map((n) => {
      const [l1, c1] = lm.pos(n.s);
      const [l2, c2] = lm.pos(n.e);
      return { i: n.i, k: n.k, role: n.role, text: n.text, flags: n.flags, loc: [l1, c1, l2, c2] };
    }),
  };
}

/** JSON, with the VAR and ALT markers written as \ue000 and \ue001 escapes so that they are visible in the file. */
const j = (v: unknown): string => JSON.stringify(v).replace(/[\uE000\uE001]/g, (c) => `\\u${c.charCodeAt(0).toString(16)}`);
/** An object from [key, already-serialised value] pairs. */
const obj = (pairs: [string, string][]): string => `{${pairs.map(([k, v]) => `${j(k)}: ${v}`).join(', ')}}`;
const arr = (xs: unknown[]): string => `[${xs.map(j).join(', ')}]`;

/** One cluster and one node per line, ", " and ": " separators, UTF-8, trailing newline. */
export function serialise(s: Screen): string {
  const src = obj([['repo', j(s.source.repo)], ['path', j(s.source.path)], ['commit', j(s.source.commit)], ['licence', j(s.source.licence)]]);
  const cl = s.clusters.map((c) => obj([['k', j(c.k)], ['kind', j(c.kind)], ['line', j(c.line)]]));
  const nd = s.nodes.map((n) => obj([
    ['i', j(n.i)], ['k', j(n.k)], ['role', j(n.role)], ['text', j(n.text)], ['flags', arr(n.flags)], ['loc', arr(n.loc)],
  ]));
  const list = (xs: string[]): string => (xs.length ? `[\n  ${xs.join(',\n  ')}]` : '[]');
  return `{"schema": ${j(s.schema)}, "id": ${j(s.id)}, "framework": ${j(s.framework)},\n "source": ${src},\n "clusters": ${list(cl)},\n "nodes": ${list(nd)}}\n`;
}
