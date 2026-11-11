# Slop: handoff (2026-11-11)

Fable 5.2 (`claude-fable-5-2`, effort xhigh) hand-writes, by reasoning and with no training, fitting or search, the
weights of a small transformer that flags redundant UI copy (`ECHO`, `NARRATE`, `DECOR` versus `KEEP`) for a
linter. `SPEC.md` is the design, `LABELLING.md` the definition of the labels, `kit/KIT.md` the author's brief.

The owner reframed this on 2026-11-11 as an informal run ("just for fun to see if it works"), not a formal
experiment. As a result, the smoke launch, the freeze (`scripts/freeze.py --write`) and the sealed selection were
skipped. The sealed n-gram leakage gate is waived by name in `kit/leakage_waivers.json`. The owner delegated prompt
approval. `sealed_real` has never been selected or opened.

## Where it stands

Two rounds are done. Both networks, their `build.py`, lexicons, `MODEL.md` and audits are committed.

| | r1 (`runs/r1/work`, 3e826f5) | r2 (`runs/r2/work`, 0087d60) |
|---|---|---|
| Fable spend | US$59.52 (2 launches) | US$79.04 (3 launches) |
| Vocabulary | 1,077 stems, id layout 1 | 5,336 stems, id layout 2 (Open English WordNet + own lexicon) |
| Val, 2,000 unseen generated screens: F1 / P / R / false alarms | 0.731 / 0.908 / 0.611 / 0.4% | **0.785 / 0.952 / 0.667 / 0.2%** |
| Disclosed dev (seen by Fable) | F1 0.836 on 400 | (1,200 disclosed in r2) |
| Calibration, 25 hidden real screens, 12 redundant nodes | 1 caught, 3 false alarms in 380 | 0 caught, 2 false alarms in 380 |
| Owner's repos, ~1,800 real nodes | 22 flagged | 14 flagged |

The finding so far is clear: generated-screen performance improves round on round, but on real copy the network
barely fires. Real redundancy in the owner's repos is mostly small, odd cases: a one-word eyebrow, a ♥ glyph, an
emoji empty-state title, the app name repeated, tooltips narrating toggles. The generator does not produce screens
like that. r2's goal 1, stopping flags on sentences that add a consequence or instruction, is not met on the
motivating real case: "All pages in this scan will be deleted." under "Discard this scan?" is still flagged. The
calibration figure rests on 12 positives, so treat it as a direction, not a measurement.

### Spend to date
- **Fable:** r1 US$59.52, r2 US$79.04.
- **Labelling:** US$6.78 of Opus 5.5 calls, plus Codex usage on the owner's plan.
- **Build sessions and agent workflows:** a few million tokens of Claude Code session usage.

Every token counts. Do not describe agent work as free.

### Real answer key
`modellabel/`: Claude Opus 5.5 and Codex (gpt-6.1-sol) label independently, then adjudicate. Contested nodes are not
scored. The files are in `labels/`, which is gitignored because it holds private copy.

| Set | Binary κ (before → after adjudication) |
|---|---|
| reference_real, 22 screens, disclosed to Fable in r2 | 0.56 → 0.85 |
| calibration, 25 screens, never shown to Fable | 0.65 → 0.88 |

## Layout (what is where)
- **`extract/`:** TS extractor (JSX/TSX, HTML, SwiftUI) → `slop.screen/1`.
- **`ui/`:** Python tokeniser, Porter stemmer and numpy model.
- **`lint/`:** JS tokeniser, runtime and compiler (`slop.net/2`, sparse rows; id layouts 1 and 2).
- **`rules/`:** the plain-code rule layer. It includes the codex rules (chatbot residue, placeholder copy, empty
  claims, LLM constructions, stock CTAs and others).
- **`generator/`:** generates the 20,000 dev and 2,000 val screens.
- **`codex/`:** slop codex. `CODEX.md` holds patterns L01–L52 (LLM markers) and U01–U52 (UI guidance), with
  `lists/` and `SOURCES.md` (licences per list).
- **`resources/thesaurus/`:** Open English WordNet (CC BY 4.0) plus Moby, keyed to Porter stems. `fetch.py` and
  `build.py` regenerate it.
- **`kit/`:** the author kit. KIT.md, the r1 and r2 prompts, tools (`tok`, `lint`, `score`, `inspect`, guarded
  `python3`), `kit_build.py`, settings, leakage check.
- **`scripts/`:**
  - `prepare_author_dir.sh` and `launch.sh` (`--run r1|r2`; `start`, `continue K`, `check`)
  - `cost.py` and `audit.py`
  - `evaluate.py`: val score and a real-repo lint report into the gitignored `runs/<run>/eval/`
  - `build_sets.py`, `build_sealed.py` and `freeze.py` (not run)
- **`runs/r1`, `runs/r2`:** hand-over records, audits, console logs and each round's `work/`. Transcripts
  (`*.stream.jsonl`) and `work/scratch/` stay local because they contain real reference-screen copy.

## Commands (from `slop/`)
```
npm test && npm run test:rules && uv run pytest && uv run pytest labeller/tests
uv run python scripts/evaluate.py runs/r2/work --against runs/r1/work      # val + owner's repos, local report
uv run python modellabel/run.py label|adjudicate|report --set reference_real|calibration ...
sh scripts/prepare_author_dir.sh ~/Development/hypabolic/HypaNet-slop-author-r2
sh scripts/launch.sh --run r2 check | start | continue K
```
Score a network on the calibration key the way it was done here:
1. Build a temporary root with `examples/dev/` holding the calibration screens and their labels from
   `labels/calibration.final.jsonl`, written with `kit_build.labels_jsonl`.
2. Copy the network into the root's `work/`.
3. Run `kit/tools/author_tool.py ROOT - $(which node) score work --set dev`.

`runs/r2/eval/cal_root` is the last such root. Folding this into `evaluate.py` is a small open task.

## Known issues
- **Usage limits cut Fable runs.** A Fable run uses the account's Claude Code usage. The 5-hour limit cut r1 once
  and r2 twice. Resume with `launch.sh --run rN continue K`, which continues the same session.
- **`audit.py` false positives.** It reports "reads examples/" when a script's docstring says it never reads them.
  Both r2 findings were exactly this. Make it ignore docstrings and comments.
- **Stale `launch.sh check` wording.** It still says r1 `start` needs a smoke launch even when `kit/SMOKE_SKIPPED`
  exists. `start` itself honours the marker.
- **Guide question (codex L27).** Should K11 ("Ready to get started?" over [Get started] is KEEP) also cover
  marketing CTA bands? This is undecided, and `LABELLING.md` is unchanged.
- **Reference set size.** reference_real is 22 screens since the blog-figures exclusion, a shortfall of 29 against
  the caps.
- **Stale meta file.** `labels/reference_real.mg.meta.json` is left over from a human labelling session that never
  happened. Ignore it.

## Rules that still apply
- **Sealed repos.** Never open or read them: Atomic-Private, MemoryOS, SandyScouts, hypabolic-homepage,
  fpp-quilting-generator, HypaAlarm, CoastalQuiltsWebsite, odysseus. Never read
  `~/Development/hypabolic/slop-private/`.
- **The repo is public.** Nothing committed may quote the owner's private repos. Before committing a round, check
  `MODEL.md` and the probes for copied reference or calibration text; two sentences were replaced in r2.
  `.cloudflare/` stays uncommitted.
- **Money.** Every Fable launch, labelling run or other paid model call needs the owner's approval of its cost and
  cap first. Never stop a Fable run for long thinking.
- **Writing results.** Every number carries its split (val, dev, calibration or real repos). Name Fable and Codex.
  Use British spelling.

## Next steps (proposed, each needs a cost go-ahead)
1. **Close the generated/real gap.** Teach the generator the real-app cases listed above: short labels, glyph
   decor, app-name repeats, tooltip narration, consequence sentences that are KEEP. Then run an r3 in which Fable
   revises r2. Generator work, plus about US$50–80 of Fable.
2. **A real answer key big enough to steer by.** Label 100–200 real screens from repos that are not sealed. The
   Opus calls would cost roughly US$15–30, plus Codex usage on the owner's plan. Keep calibration hidden as the test.
3. **Optional.** Fix the audit's docstring matching and fold the calibration score into `evaluate.py`.
