// Test helper: reads {"vocab": text|null, "texts": [...], "screens": [...]} as JSON on stdin and prints the JS
// tokeniser's output as JSON, for the Python parity tests (ui/tests/test_tokenizer_parity.py).
import { readFileSync } from 'node:fs';
import { Vocab, EMPTY_VOCAB, tokenizeText, encodeScreen } from '../tokenizer.js';

const req = JSON.parse(readFileSync(0, 'utf8'));
const vocab = req.vocab == null ? EMPTY_VOCAB : Vocab.fromText(req.vocab);
const texts = (req.texts || []).map((t) => {
  const tt = tokenizeText(t, vocab);
  return { ids: tt.ids, flags: tt.flags, dropped: tt.dropped, stems: tt.tokens.map((x) => x.stem ?? null) };
});
const screens = (req.screens || []).map((s) => {
  try {
    const enc = encodeScreen(s, vocab);
    return {
      nodes: enc.nodes.map((n) => ({ ids: n.text.ids, flags: n.flags, mask: n.flagMask })),
      windows: enc.windows,
    };
  } catch (e) {
    return { error: String(e.message) };
  }
});
process.stdout.write(JSON.stringify({ texts, screens }));
