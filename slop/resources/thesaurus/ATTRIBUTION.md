# Thesaurus sources: licences and attribution

The tables in `out/` are derived from the two sources below. `fetch.py` downloads them unchanged into `raw/`
(gitignored); `build.py` reformats them (see "Changes made"). The code in this directory (`fetch.py`, `build.py`,
`coverage.py`, `tests/`) is part of HypaNet and under the repository's licence.

## Open English WordNet 2025

- Title: Open English WordNet, 2025 edition (`english-wordnet-2025.xml.gz`, WN-LMF 1.3 XML)
- Creators: the Open English WordNet team (John P. McCrae and contributors), building on Princeton WordNet
- Source: https://github.com/globalwordnet/english-wordnet (release `2025-edition`, tag commit
  `dc343f2683279ecbb13fab4e2fd778d7b162d287`)
- File URL: https://github.com/globalwordnet/english-wordnet/releases/download/2025-edition/english-wordnet-2025.xml.gz
  (sha256 `9ca6d1dcb75f822fdd66617f7d9da48142ace38dd544d6ad5e2feca1674ad3fe`)
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0),
  https://creativecommons.org/licenses/by/4.0/ ; the underlying Princeton WordNet data is under the WordNet
  licence, reproduced below as the OEWN distribution requires.
- Citation: McCrae, J. P., Rademaker, A., Bond, F., Rudnicka, E. and Fellbaum, C. (2019). English WordNet 2019 – An
  Open-Source WordNet for English. Proceedings of the 10th Global WordNet Conference.
- Changes made: `out/synsets.tsv`, `out/words.tsv`, `out/related.tsv`, `out/stems.tsv` and `out/stem_links.tsv`
  are reformatted extracts: synset ids lose their `oewn-` prefix; only the first definition is kept and its white
  space collapsed; examples, pronunciations, ILI links and some relation types are dropped; inverse relations
  (hyponym and the like) are dropped; words are additionally keyed by Porter stems from the slop tokeniser, and
  stem-to-stem link counts are derived. No synset content is edited.

The OEWN `LICENSE.md` states: "You may share and adapt this resource providing attribution is given to both
Princeton WordNet and the Open English Wordnet team."

### WordNet licence (Princeton WordNet), as distributed with OEWN (`WNDB_License.txt`)

```
This software and database is being provided to you, the LICENSEE, by
the Open English Wordnet team under the Creative Commons Attribution 4.0
International License (CC-BY 4.0).

Open English Wordnet 2023 Copyright 2023 by the Open English Wordnet team.

Permission to use, copy, modify and distribute this software and
database and its documentation for any purpose and without fee or
royalty is hereby granted, provided that you agree to comply with
the following copyright notice and statements, including the disclaimer,
and that the same appear on ALL copies of the software, database and
documentation, including modifications that you make for internal
use or for distribution.

WordNet 3.1 Copyright 2011 by Princeton University.  All rights reserved.

THIS SOFTWARE AND DATABASE IS PROVIDED "AS IS" AND PRINCETON
UNIVERSITY MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR
IMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, PRINCETON
UNIVERSITY MAKES NO REPRESENTATIONS OR WARRANTIES OF MERCHANT-
ABILITY OR FITNESS FOR ANY PARTICULAR PURPOSE OR THAT THE USE
OF THE LICENSED SOFTWARE, DATABASE OR DOCUMENTATION WILL NOT
INFRINGE ANY THIRD PARTY PATENTS, COPYRIGHTS, TRADEMARKS OR
OTHER RIGHTS.

The name of Princeton University or Princeton may not be used in
advertising or publicity pertaining to distribution of the software
and/or database.  Title to copyright in this software, database and
any associated documentation shall at all times remain with
Princeton University and LICENSEE agrees to preserve same.
```

## Moby Thesaurus II

- Title: Moby Thesaurus II (`mthesaur.txt`), by Grady Ward
- Source: Project Gutenberg eBook #3202, https://www.gutenberg.org/ebooks/3202 ; fetched from the PGLAF mirror
  https://aleph.pglaf.org/3/2/0/3202/files/mthesaur.txt (same bytes as
  https://www.gutenberg.org/files/3202/files/mthesaur.txt; sha256
  `7c9742b1ed94435a893c0719b426725edb8a5242f8c526a75461bd6cee2dfd32`)
- Licence: public domain. The Moby documentation shipped with the eBook says: "Public Domain material by grant from
  the author, January, 2001."
- `mthesaur.txt` itself carries no Project Gutenberg header or licence text, and nothing here uses the Project
  Gutenberg name or trademark beyond naming where the file was obtained.
- Changes made: `out/moby.tsv` is the list reformatted (one root per row, duplicates and the root removed from its
  own list, the tokeniser's rendering of the root added); `out/stems.tsv` and `out/stem_links.tsv` add Moby-derived
  stem columns. Moby has no parts of speech, and its lists mix synonyms with loosely related terms.

Moby is a trademark of Grady Ward.
