// One finding: the rule id, the pattern or aside code, whether the code is a node or a screen aside, the flagged
// nodes (their `i` and `loc`), an optional target (the kept copy) and a message.

export function finding(rule, code, screen, xs, message, extra = {}) {
  const nodes = screen.nodes;
  const out = { rule, code, scope: code === 'R19' ? 'screen' : 'node' };
  if (extra.condition) out.condition = extra.condition;
  out.nodes = xs.map((x) => nodes[x].i ?? x);
  out.locs = xs.map((x) => nodes[x].loc ?? null);
  if (extra.target !== undefined) out.target = { node: nodes[extra.target].i ?? extra.target, loc: nodes[extra.target].loc ?? null };
  out.message = message;
  return out;
}
