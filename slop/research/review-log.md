# Review log: SPEC.md and LABELLING.md, 2026-11-10

64 reviewer issues (21 validity, 11 feasibility, 14 definition, 18 conventions). Each was checked against the files
before it was applied. Results: 51 accepted as proposed (merges included), 12 accepted with changes, 1 rejected in part (F7's optional
node embedding). Where two issues cover the same point, the merge is noted. No paid API was called, and no sealed
HypaNet data was opened. Example strings were checked by fixed-string grep over the eight sealed source repos, which
reported file counts only.

Checks run for this log:

- `git check-ignore` matches nothing under `slop/` (rc=1). The remote is the public `Hypabolic/HypaNet`.
- 54 PersonalHomepage UI files are byte-identical to files in sealed repos (sha256, with node_modules, .next, .git,
  dist and build excluded).
- `packet.py` `leakage_problems` checks only sentinels and generator words.
- In the Clouds files, `audit.py:49,199` and `cost.py:31-36` hold the 5.1 constants, `launch.sh:86-96,141` refuses on
  the conditions given, and `settings.json` denies the HypaNet project directory.
- The pricing page (fetched 2026-11-10) gives Opus 5.5 at US$4 input, US$0.20 cache hit and US$20 output per MTok,
  and Haiku 5.5 at US$0.10, US$0.01 and US$0.50.
- Seed 20261116 is not used in src/, scripts/, experiments/, clouds/ or the README.

| # | Lens | Sev. | Section | Decision | What changed, or why rejected |
|---|---|---|---|---|---|
| V1 | validity | blocker | SPEC §5 agreement on sealed screens | Accepted, modified | Calibration pool = all of reference_real + up to 40 screens from repos in no split. Guide revised only on the pool. A second labeller labels 40 sealed screens after the freeze, for measurement only. guide_sha256 enforced by freeze.py --check. aitutaki-game dropped from the pool because it is Unreal C++, which the extractor does not read. Merged with D12. |
| V2 | validity | blocker | LABELLING §8 sealed strings | Accepted, modified | All four rows confirmed as sealed copy (Atomic A4, A7; fpp F1; odysseus O4) and replaced. Several of the proposed replacements also occur in sealed repos (one in 28 sealed files; string withheld), so other invented strings were used, each with 0 sealed matches. Three more rows matched sealed text and were replaced: three strings withheld (2, 1 and 2 sealed files). The leakage check now covers LABELLING.md, KIT.md and examples/ (SPEC §9). |
| V3 | validity | major | corpus-scout and the harness session | Accepted | SPEC §9 "Firewall": §3 of the note moves out of the repo and is replaced by a redacted version. The generator, rules, KIT.md and briefs are written in a fresh session. Labellers do not read the note. LABELLING.md is declared as drafted by a session that read it. The move is not yet done (it is a pre-build step). |
| V4 | validity | major | SPEC §9 real-set duplicates | Accepted | Exclusions (components/ui, worktrees, DesignRefs copies, mirrored prototypes), cross-set dedup by hash and Jaccard ≥ 0.8, and per-rule drop counts in the manifest. 54 duplicate files confirmed. |
| V5 | validity | major | SPEC §9 leakage check | Accepted, modified | New slop/kit/leakage.py with salted hashes of 2+-word whole nodes and 3-word runs, which opens no sealed file. Added a second waiver ground: phrases quoted from the public sources in research/, because generic genre words such as "How it works" are in the K1 list. |
| V6 | validity | major | SPEC §12.4 generated splits | Accepted | test_gen regenerated on a hash or Jaccard match with dev/val. ood_codex drops screens that contain disclosed strings, and the build refuses above 2%. |
| V7 | validity | major | SPEC §2, §11.1 shared priors | Accepted, modified | Both commissions stay with Codex (house pattern). The paired written-vs-rules result is headline on sealed_real only, ood_codex carries the shared-model note, and test_gen carries "same model family". The OOD generator's brief is now specified in §9. |
| V8 | validity | major | SPEC §12.4 generator labels | Accepted | Matthew labels 30 + 30 generator screens blind. κ and F1 are reported per generator, and a generator below κ 0.6 is revised. The Codex sample uses seed 20261114, namespace `check`, because ood_codex has no val namespace. |
| V9 | validity | major | SPEC §11.1 rules-engine information | Accepted | Same examples, tools, no-fitting rule and stem budget as Fable. Spend and revisions are reported. |
| V10 | validity | major | SPEC §11.2 "400 matches" | Accepted | T-400m trains on the 400 disclosed screens + reference_real with a fixed epoch count. The scaling curve is nested. Hyperparameters are frozen. Merged with C13. |
| V11 | validity | major | SPEC §11.3, §16 Q2 Opus 5.5 | Accepted | The second labeller is a person (Q2). LLMs get the reference few-shot prefix. Opus 5.5 is marked as the guide's drafter. Prompts are hashed. Refusals are scored as KEEP and counted. |
| V12 | validity | major | SPEC §10 operating points | Accepted | One primary (F1) and one co-primary (false alarms). Precision at matched recall and PR curves for scored systems. Linter view per file. |
| V13 | validity | major | SPEC §12.6 registry | Accepted | The registry adds the scoring and bootstrap code, hyperparameters, prompts and the comparison list, and is extended once with baseline artefacts. All systems are scored in the single opening. |
| V14 | validity | major | SPEC §5, §12.1 extractor | Accepted | End-to-end recall. Extractor frozen before sealed labelling. Labels now key on `loc`, not node index. |
| V15 | validity | major | SPEC §9 public_clean | Accepted | public_to_screen.py with a frozen role map, reported as a separate input path. Blind pooled adjudication of every system's flags. |
| V16 | validity | major | SPEC §9, §10, §14 pause fixture | Accepted, modified | Fixture labelled "disclosed to Fable" and shown beside a sealed sibling fixture. The sibling is obtained from a coding agent, not written by Matthew by hand, so that it is AI-built like the original. §14 says the output is on a disclosed fixture. |
| V17 | validity | major | SPEC §10 lexicon overlap | Accepted | ECHO recall split by stem-pair overlap with the disclosed screens. Vocabulary overlap compared with 400 undisclosed dev screens. |
| V18 | validity | minor | SPEC §9, §10 Atomic units | Accepted | Atomic-Private counts as one repo. The 25% cap also applies to comments_real. Per-repo and leave-one-out tables added. |
| V19 | validity | minor | SPEC §9 seed reuse | Accepted | New seed 20261116 with namespaces reference, sealed, public, comments, calibration and agreement. 20261113 is used for test_gen only. Merged with C12. |
| V20 | validity | minor | SPEC §10 pointer accuracy | Accepted | Denominator defined. Equivalent copies counted, via the new `equiv` field in label records and LABELLING §5. Intersection view added. |
| V21 | validity | minor | SPEC §5 agreement statistics | Accepted, modified | κ, AC1, positive and negative agreement with intervals, all nodes and non-control nodes, a human ceiling, and UNSURE shares. The 5% UNSURE limit applies to the calibration pool, because the guide cannot be revised from sealed labels. The sealed share is reported only. |
| F1 | feasibility | major | SPEC §1, §10, §14; LABELLING §7 R6 | Accepted, modified | Implemented as a `ct: true` field on every label the consequence test decides, in either direction, rather than "R6 judgement". The symmetric KEEP side (HypaChef) is also excluded from the headline. Additions: stretch column in §10, risk in §15, note in §14, KIT.md statement in §13, and §16 Q7. |
| F2 | feasibility | major | SPEC §6 camelCase | Accepted | Split before folding. Flags computed first. 10 golden identifier cases in §12.2. |
| F3 | feasibility | major | SPEC §3 Python parser | Accepted (option 1) | tree-sitter-python pinned, Python comment fixtures, and the generator emits Python pairs. |
| F4 | feasibility | major | SPEC §7, §12.3 precision | Accepted | float64 compute in both runtimes, relative 1e-6 tolerance, and parity on Fable's artefact before the sealed run. |
| F5 | feasibility | major | SPEC §13 context budget | Accepted | Compact per-node text and INDEX.txt. ./score capped at 40 wrong nodes with filters. ./inspect defaults to the EON row. 20,000-character cap on tool output. |
| F6 | feasibility | major | SPEC §6, §13 stems | Accepted | kit/lib read-only stemmer, tokeniser and model. build.py may stem a word lexicon. ./tok --words and --vocab (merges shown). |
| F7 | feasibility | minor | SPEC §7 pointer ties | Tie rule accepted; node_emb rejected | Ties go to the earlier node. node_emb is not adopted because rel_bias already lets a pointer head prefer the nearest of several equal-content nodes, so it adds 8,192 parameters for no new capability. The shape stays at 2,128,900. |
| F8 | feasibility | minor | SPEC §4 kinds and flags | Accepted | Closed kind list (7). Six flags added (28 of 32 slots). |
| F9 | feasibility | minor | SPEC §6 hash buckets | Accepted | Vocabulary 512–4,095 (3,584 stems), buckets mod 6,144, per-pair collision 1/6,144. Trained-baseline vocabulary changed to 3,584 to match. |
| F10 | feasibility | minor | SPEC §13 tools | Accepted | directions.json, --files probes, --against diff. |
| F11 | feasibility | minor | SPEC §4 windows | Accepted | Positions include BOS and EON. An oversized cluster is split at node boundaries and keeps its index. |
| D1 | definition | blocker | LABELLING §4 procedure | Accepted, reworded | The new order is duplicates, K rules, narration, partial, deletion last. Step wording tidied (the proposal's "go to step 3 only to check for a partial" was unclear). §8 re-derived with step numbers. |
| D2 | definition | major | LABELLING §2a evidence | Accepted | Evidence = screen JSON + source + README product kind. Visibility by branch and state. X-FILE aside. Subtitle "Why" rewritten. |
| D3 | definition | major | LABELLING §5 ranks | Accepted | Controls and h1/dialog_title/empty_title share rank 1. K11 added. "Delete playlist?" case added (a "project" variant occurs in sealed repos). |
| D4 | definition | major | LABELLING K1 genre words | Accepted, modified | Genre list frozen with the guide. Example changed to "Features" / "Features that scale", because "Pricing" over a pricing heading is an exact sealed pattern. |
| D5 | definition | major | LABELLING K2, dots; SPEC §4/§7 | Accepted, modified | A text-less dot is a flag on the next text node, and the §7 bare-dot sentence is deleted. Static status claims and the R11g glyph aside added. The rule for constant text beside a bound dot is simplified to KEEP (the node carries live state). The proposed "ECHO → null only if…" wording was circular. |
| D6 | definition | major | LABELLING §9 repetition; SPEC §8 | Accepted | Screen aside R19 with a record format in SPEC §5. repeated-chrome is scored against it. Dot rule corrected to v2-experimental:705. Eyebrow quota marked as taste. |
| D7 | definition | major | LABELLING K12 | Accepted | K12 added. sr_only nodes are context only. |
| D8 | definition | major | LABELLING K13/K14 | Accepted, modified | K13 and K14 added. Nodes with no literal words are context only (aside UNRESOLVED) rather than UNSURE, so i18n'd repos do not inflate the UNSURE limit. |
| D9 | definition | major | LABELLING §7 (b) | Accepted, modified | Hidden-input clause and the platform-behaviour sentence added. The proposed short case is sealed copy (string withheld), so "Hold Shift to select a range." is used instead. |
| D10 | definition | major | LABELLING R2, DECOR, K3, R12 | Accepted | R2 redefined. Meta strips narrowed, and version numbers on about or settings screens are KEEP. K3 rewritten. R12 moved to aside (rule layer), with the 38% vs 21% source. |
| D11 | definition | major | LABELLING counts | Accepted | "A literal count of items that are all visible". Data-bound counts are KEEP ("340 invoices" case). |
| D12 | definition | major | SPEC §5 agreement | Accepted (merged into V1) | Pool includes all of reference_real. κ with and without UNSURE. Subsets by Matthew's conf. Model-labeller conflict covered in §16 Q2. |
| D13 | definition | minor | LABELLING §5 unranked roles | Accepted | breadcrumb at rank 1. cell, input_value, unattached label and other at rank 3. comment always the flagged side. R14 sentence fixed. |
| D14 | definition | minor | LABELLING §1, K9, codes | Accepted | GOV.UK attribution replaced by the Microsoft Writing Style Guide, Shopify and frontend-design. K9 uses a co-visible pair. Code-numbering note added. |
| C1 | conventions | major | SPEC §3 gitignore | Accepted | slop/.gitignore patterns specified. Labels committed as hashes only. Merged with C2 and C11. |
| C2 | conventions | major | SPEC §3 research/ | Accepted (merged into V3) | The unredacted note leaves the repo. Only the redacted version is committed. |
| C3 | conventions | major | SPEC §3 isolation | Accepted | make_settings.py with dontAsk. Deny list covers ~/.claude (HypaNet project dir included), /tmp/claude-501, the private note directory, sealed and public source repos and the smoke dir. Tools refuse paths outside the author dir. |
| C4 | conventions | major | SPEC §12.7 smoke launch | Accepted | Separate smoke dir and logs. cost.py charges smoke* against the US$200. Overshoot to about US$21 stated. Clean-up before start. |
| C5 | conventions | major | SPEC §13 archive | Accepted | Archive redacted unless Matthew clears the repos. §16 Q5 extended. |
| C6 | conventions | minor | SPEC §13 cost arithmetic | Accepted | US$45.26–60.62 band, with the abort charge named. Sum 130 + 21 + 21 = 172 shown. |
| C7 | conventions | minor | SPEC §13 launch flags | Accepted | stream-json, env exports, scrub, launch record and approval gate added. |
| C8 | conventions | minor | SPEC §13 audit/cost constants | Accepted | MODEL, the MODEL COMPLETE regex, CAP_USD 200 and the 5.2 price row (8/40/10/16/0.20 from [PZ]) stated. |
| C9 | conventions | minor | SPEC §12.7 approval | Accepted | The smoke launch needs approval of the prompt, the cap and the effort. The opening Decisions line also names the cap and effort. |
| C10 | conventions | minor | SPEC §3, §9 paths | Accepted | Full slop/ paths, new files marked. freeze.py modelled on scripts/v04_freeze.py (exists). Marker at the repo root. |
| C11 | conventions | minor | SPEC §3 labels/ vs label/ | Accepted | labels/ in the layout. Tool directory renamed labeller/. |
| C12 | conventions | minor | SPEC §9 seed namespaces | Accepted (merged into V19) | Namespaces given. README Sealed-table and used-seed-paragraph placement stated. |
| C13 | conventions | minor | SPEC §11.2 "400 matches" | Accepted (merged into V10) | The claim is removed. T-400m includes reference_real. |
| C14 | conventions | minor | SPEC §11.3 LLM cost and privacy | Accepted | Arm marked optional. Estimate from list prices: about US$80–90 Opus 5.5 and about US$3 Haiku 5.5, for about 4,750 screens, a 25K cached prefix, 1K input and 400 output each. This is a harness estimate. §16 Q2 and Q6 added. |
| C15 | conventions | minor | SPEC §13 thinking visibility | Accepted | The smoke launch tests showThinkingSummaries. INTEGRITY.md reports the share of thinking blocks with text. |
| C16 | conventions | minor | SPEC §3 Clouds edit | Accepted | Clouds files are not edited. Its launcher's refusal is left in place (verified launch.sh:95). |
| C17 | conventions | minor | SPEC §13 py_guard | Accepted | Copied into slop/kit/tools and hashed in HANDOVER.json. |
| C18 | conventions | minor | SPEC §2 cross-reference | Accepted | "OOD generator (§9); rules-engine baseline (§11)". |

Items left for action outside the spec text:

- Write `slop/.gitignore` before any data or label file exists.
- Move and redact `research/corpus-scout.md` §3. It is still in place, and `slop/` is untracked, so nothing has been
  committed.
- Run the sealed n-gram check on the revised `LABELLING.md` once `sealed_ngrams.json` exists. The grep checks here
  cover only the example strings.
