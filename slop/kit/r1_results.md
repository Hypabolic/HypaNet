# Round 1 results

What your round-1 network (the artefact in `work/` as handed over, `e3e0b1d08f24…`, id layout 1, 1,077 stems) scored,
and where it went wrong on real copy. This is all the feedback from round 1 you are given.

## Scores

- **Dev examples** (the 400 screens of round 1, all of them still in `examples/dev/`): REDUNDANT F1 0.836, precision
  0.984, recall 0.726, 5 false alarms in 6,785 KEEP nodes; four-class macro-F1 0.897; pointer accuracy 94%. This kit
  has 1,200 dev screens (the same 400 and 800 more chosen by the same rule); `./score work --set dev` gives the
  round-1 network's score on all of them.
- **Held-out generated screens** (2,000 screens from the same generator, never shown): F1 0.731, precision 0.908,
  recall 0.611.
- **Real screens** (1,817 scored nodes from the owner's projects, all in files you were never shown): it flagged 22
  nodes, about 1% of them, and about half of those calls were right. 328 of the nodes come from projects none of whose
  files you have seen (6 flagged); the other 1,489 come from the undisclosed files of the three projects behind
  `examples/reference/` (16 flagged; the 21 disclosed files themselves were left out). Its precision on generated
  screens did not carry over to real copy, and most real redundancy went unflagged.

## Real copy: the false alarms

Every wrong call on real copy was a full sentence that shares a word or a concept with a nearby heading, title or
control but adds something. The examples below are invented; they show the categories, not the real strings. Each
one gets the same wrong call from your round-1 network (`./score work --files` on a screen you write shows it).

1. **A consequence under a confirmation title.** Dialog title "Remove this list?" with body "All items in this list
   will also be removed." and the buttons "Remove" and "Cancel". The body repeats the title's verb and noun, but "all
   items" and "also" are new: removing the list removes what is in it, which the title does not say. `KEEP` (K7);
   round 1 calls it `ECHO`. Restating a verb is not restating the sentence.
2. **A short instruction.** An empty detail pane beside a list titled "Playlists" that says "Pick a playlist." It
   shares the list's noun, but it tells the reader the next action in a place that is otherwise empty: `KEEP` (K8);
   round 1 calls it `ECHO`. An imperative is information unless its object is already a visible control's label
   (`LABELLING.md` §4).
3. **A help sentence that gives a reason or an effect.** Under a setting "Public profile": "Your profile is public,
   so anyone can find it." It names the setting's words, then says what follows from them: `KEEP` (K4); round 1
   calls it `ECHO`.

The pattern behind all three: a concept match with the anchor was treated as evidence, and the rest of the sentence
was not weighed. Words such as "will", "also", "all", "unless", "until", "because", a number or a time, and an
imperative with a new object, are what make these sentences worth keeping.

## Real copy: the misses

Each invented example below is a restatement that your round-1 network calls `KEEP`.

1. **Paraphrase with no shared stem**, by far the largest group: a subtitle "Everything you've saved, in one spot."
   under "Library"; an eyebrow "THE LOWDOWN" over "Overview"; an empty-state body "Your inbox is clear." under "No
   messages"; a helper "The name people see." under a field "Display name". The synonym groups (about 170 groups over
   1,077 stems) did not reach the words real copy uses, and an unknown word counts as information.
2. **Domain synonyms**, the same gap in the product's own nouns: a subtitle restating its page title with a
   near-synonym ("Members" and "The people on your team", "Orders" and "Things you've bought", "Recipes" and "Dishes
   you've saved").
