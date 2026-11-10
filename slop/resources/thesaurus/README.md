# Thesaurus resource (slop r2)

A full open English thesaurus, reformatted so the author can look up synonyms and related words **by the stems the
slop tokeniser produces**. r1's hand-made lexicon had about 170 synonym groups over 1,077 stems and missed most
paraphrase echoes. This resource gives the author two complete sources to draw on when writing r2's concept codes.
It is reference data for reasoning, not a model input: nothing here is fitted or searched by a program.

```sh
cd slop
uv run python -I resources/thesaurus/fetch.py           # download into raw/ (gitignored), verify pinned sha256
uv run python -I resources/thesaurus/fetch.py --verify  # verify only
uv run python -I resources/thesaurus/build.py           # raw/ -> out/*.tsv + MANIFEST.json (about 10 s)
uv run python -I resources/thesaurus/coverage.py --top-missing 40   # coverage over dev and reference_real
uv run pytest resources/thesaurus/tests
```

`fetch.py` refuses (exit 2) on any size or digest mismatch, deletes the partial download, and never overwrites an
existing file that fails the check. The downloads are data only: `build.py` parses them (XML iterparse, refusing
entity declarations; comma-split text) and runs with `python -I`. Licences and attribution: `ATTRIBUTION.md`.

## Sources and why

| Source | Licence | What it gives |
| --- | --- | --- |
| **Open English WordNet 2025** (`english-wordnet-2025.xml.gz`, 11.4 MB) | CC BY 4.0, plus the Princeton WordNet licence notice | 107,519 synsets (sense-level synonym sets with a gloss and part of speech), 135,969 lemmas, 4,473 irregular inflections, and typed links: hypernym, similar-to, also-see, antonym, derivation, pertainym, OEWN's morphosemantic links, and more |
| **Moby Thesaurus II** (`mthesaur.txt`, 24.9 MB) | public domain (author's grant, 2001) | 30,260 headwords with 2.52 million synonyms and related terms, no senses or parts of speech: broad near-synonym coverage, and noisy |

- **OEWN first.** It is the maintained successor to Princeton WordNet 3.0/3.1, under a clean attribution licence.
  Synsets separate senses (`save` = rescue vs. keep vs. store), and antonym links matter for the linter (`enable`
  and `disable` are related but are not an echo).
- **Moby for breadth.** WordNet's synsets are strict. Moby lists the looser substitutes a copywriter actually uses
  (`delete` -> `remove`, `save` -> `store`). Its lists are long and mixed, so `stem_links.tsv` keeps only
  **mutual** Moby links (A lists B and B lists A), which are far more precise. The full lists stay in `moby.tsv`.
- **Not used.** Princeton WordNet 3.1 is the same data, older, and OEWN already includes it. Roget 1911
  (Gutenberg, public domain) adds a category structure, but its vocabulary is from 1911 and Moby covers its
  words. The OEWN "plus" edition was left out because its additions were not checked for licence terms.
- The OEWN 2025 base edition has **no proper nouns** (no instance synsets): place, person and brand names are
  not in it.

## Files

All tables are UTF-8 TSV with one header row, `\n` line ends, no quoting, and no tab or newline inside a field.
Lists use `|` between words that may contain spaces, `,` between synset ids, and spaces between stems. A
`stem:n` item means "linked by n distinct synsets or links". Synset ids are OEWN's without the `oewn-` prefix
(`00001740-v`); the last letter is the part of speech: `n`oun, `v`erb, `a`djective, `s` adjective satellite,
`r` adverb. `MANIFEST.json` (committed) records each output's sha256, bytes, rows and whether it is committed,
plus the sources' pins, the `ui/porter.py` hash and the build statistics.

| File | Rows | Size | In git | Columns |
| --- | ---: | ---: | --- | --- |
| `out/stems.tsv` | 64,632 | 3.4 MB | committed | `stem`, `wn_words` (OEWN single-token words with this stem), `wn_synsets` (their synsets, commonest sense first where OEWN orders them), `moby_words` (Moby single-token terms with this stem), `moby_root` (1 if a Moby headword has this stem) |
| `out/stem_links.tsv` | 51,618 | 11.9 MB | ignored | `stem`, `wn_syn` (stems sharing a synset), `wn_similar` (similar-to / also-see), `wn_hyper` (stems of direct hypernyms: "is a kind of"), `wn_deriv` (derivation, pertainym, participle, morphosemantic), `wn_ant` (antonyms), `moby_mutual` |
| `out/synsets.tsv` | 107,519 | 11.1 MB | ignored | `synset`, `pos`, `lexfile` (WordNet's coarse topic, e.g. `verb.change`), `lemmas` (`\|`, OEWN member order), `gloss` (first definition) |
| `out/words.tsv` | 140,382 | 6.8 MB | ignored | `word` (as written, case kept), `pos`, `toks` (tokeniser rendering, below), `n_words`, `synsets`, `source` (`lemma` or `form:<lemma>` for an irregular inflection) |
| `out/related.tsv` | 236,461 | 9.7 MB | ignored | `synset`, `rel`, `target`, `word`, `target_word` (the last two only for sense-level links such as antonym or derivation) |
| `out/moby.tsv` | 30,260 | 25.2 MB | ignored | `root`, `toks`, `n`, `related` (`\|`, Moby's order, duplicates and the root removed) |

Files over 5 MB are gitignored and rebuilt by `build.py`; `tests/` checks every file against the manifest when it
is present. `related.tsv` relation counts: hypernym 88,075, derivation 73,540, similar 23,178, antonym 7,988,
pertainym 7,372, domain_topic 6,433, mero_part 5,387, also 4,098, attribute 1,260, entails 407, causes 221,
participle 73, morphosemantic `morph.*` 16,881 (event, agent, result, instrument and so on), meronyms
(member/substance) 1,548. Inverse duplicates (hyponym and the like) are not stored: look the link up from the
other end.

### Stems and the tokeniser

Every word is run through `ui/tokenizer.py`'s `tokenize_text` (NFKC, letter runs, camel-case split, case fold,
accent strip, `ui/porter.py`), so a stem here is exactly the string the model's vocabulary keys on (`deleted`,
`delete` and `deletion` all give `delet`). `toks` is the tokeniser's view of a word or phrase: word tokens as
stems, punctuation as its character, a number as `#`, separated by spaces:

| Written | `toks` | Indexed by stem? |
| --- | --- | --- |
| `Settings` | `set` | yes |
| `log in`, `sign up` | `log in`, `sign up` | no: two word tokens |
| `e-mail`, `Sign-in` | `e - mail`, `sign - in` | no: the hyphen is its own token |
| `WiFi` | `wi fi` | no: camel case splits |

Only **single-token** words (one word token, nothing else) appear in `stems.tsv` and `stem_links.tsv`. Multi-word
and hyphenated lemmas (58,956 of OEWN's 135,969; 41,057 of Moby's 103,305 distinct terms) keep their `toks` in
`words.tsv` and `moby.tsv`, and they still appear in `synsets.tsv` lemma lists. So `log in | log on | sign in |
sign on` is one synset that can be found from any member. To reach a phrase from a stem, grep `words.tsv` on `toks`.

Things to keep in mind:

- **Porter merges words.** One stem row holds every word with that stem, across parts of speech and senses
  (`gener` covers general, generate, generic and generous). The links are unions over all of them.
- **WordNet senses are broad.** `remove` shares a synset with `murder` (the "kill" sense). Read the gloss in
  `synsets.tsv` before you trust a link. For a precise synonym, look for a stem that is in both `wn_syn` and
  `moby_mutual`: `start`/`begin`, `quick`/`fast`, `choose`/`select`, `help`/`support` and `photo`/`picture` all
  pass. Looser UI pairs are often only in Moby (`save`/`store`) or only as a hypernym (`delete` is a kind of
  `remove`). Some are missing from both (`search`/`find`, `error`/`fail`).
- **No closed-class words.** Neither source lists `the`, `your`, `we` and other closed-class words as headwords.

## Coverage

Coverage from `coverage.py`, over word stems in screen text tokenised as the model sees it. A stem is covered if it
has a row in `stems.tsv` from that source. "Content" leaves out 92 closed-class stems (`the`, `your`,
`which` and so on; listed in `coverage.py`). `datasets/val` was not read.

| Set | Distinct stems | WordNet | Moby | Either | Has a link (`wn_syn` or `moby_mutual`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| dev, 20,000 generated screens, 972,735 word tokens | 2,854 | 68.0% | 64.0% | 70.0% | 64.8% |
| dev, content stems only (2,774; 703,215 tokens) | 2,774 | 68.3% | 63.6% | 69.6% | 64.6% |
| dev, content, token-weighted | | 93.1% | 88.9% | 94.4% | 90.9% |
| reference_real, 22 real screens, 659 word tokens | 292 | 90.4% | 90.4% | 94.5% | 90.1% |
| reference_real, content stems only (258; 534 tokens) | 258 | 93.4% | 90.3% | 94.6% | 91.5% |
| reference_real, content, token-weighted | | 94.9% | 92.9% | 96.1% | 93.8% |

Most of the dev stems not covered are rare: the long tail of generated names and places. The commonest content
stems that neither source has are software and product vocabulary: `app`, `csv`, `login`, `offline`, `username`,
`gameplay`, `pdf`, `webhook`, `reconnect`, `multiplayer`, `changelog`, `admin`, `api`, `rss`. After those come
currency codes, language names (`cymraeg`, `deutsch`), short codes and abbreviations, and the names of people,
places and services. Covering this UI vocabulary needs a
small hand-written addition; this resource does not include one.
