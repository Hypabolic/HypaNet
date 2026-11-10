# Third-party material in the `ui_*.txt` lists

The `ui_*.txt` lists in this directory are HypaNet's own work (MIT, the repository licence) except for the entries
named below, which come from permissively licensed projects. Each list's header says which block is imported. Style
guides (Microsoft, Apple, Google, GOV.UK, Mailchimp, Shopify, Atlassian, NN/g) are quoted only in short cited phrases
in `CODEX.md` and the list headers; no word list was copied from any of them.

| List | Imported entries | Source | Licence |
|---|---|---|---|
| `ui_filler_words.txt`, Block B | 26 words, a subset of the 83-word `fillers` list | `fillers` 2.0.1, https://github.com/words/fillers (npm `fillers`) | MIT, Copyright (c) 2014 Titus Wormer |
| `ui_padding_phrases.txt`, Block B | 8 phrases, a subset of the patterns marked `omit: true` | `retext-simplify` 8.0.0, https://github.com/retextjs/retext-simplify, `lib/patterns.js` | MIT, Copyright (c) 2016 Titus Wormer |
| `ui_tech_pleonasms.txt` | 9 phrases that also appear in proselint's RAS syndrome check (pin number, pdf format, gui interface, lcd display, gps system, isbn number, upc codes, atm machine, please rsvp) | proselint, https://github.com/amperser/proselint, `proselint/checks/redundancy/ras_syndrome.py` | BSD-3-Clause, Copyright © 2014–2015 Jordan Suchow, Michael Pacer and Lara A. Ross |

Checked and not imported: the Vale packages for Microsoft and Google (https://github.com/errata-ai/Microsoft,
https://github.com/errata-ai/Google; MIT) and for write-good and proselint (MIT, BSD-3-Clause); `write-good`, `alex`,
`retext-equality`, `retext-intensify`, `hedges` and `weasels` (all MIT). Their rules are described in
`CODEX.md` §10; their lists target general prose, and several of their entries carry the consequence or scope
in UI copy (see the caution in `ui_consequence_markers.txt`).

## MIT licence (fillers; retext-simplify)

```
(The MIT License)

Copyright (c) 2014 Titus Wormer <tituswormer@gmail.com>   [fillers]
Copyright (c) 2016 Titus Wormer <tituswormer@gmail.com>   [retext-simplify]

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
'Software'), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## BSD 3-Clause licence (proselint)

```
Copyright © 2014–2015, Jordan Suchow, Michael Pacer, and Lara A. Ross
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
   disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
   following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
   products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
