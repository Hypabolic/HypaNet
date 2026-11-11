#!/usr/bin/env python3
"""Selects thesaurus entries for the lexicon and writes work/thesaurus_groups.tsv (read by build.py).

    ./python3 work/select_thesaurus.py [--show]

Reads resources/thesaurus/out/{stems,stem_links}.tsv, the hand lexicon (work/lexicon/*.txt) and
work/thesaurus_exclude.txt (stems I struck out by reading the output; one per line). It never reads examples/,
scores or the network's outputs. The rule (also in MODEL.md section 2):

A stem x that is NOT in the hand lexicon may be taken when it is a *specific* word: at most MAXSYN = 3 WordNet
synsets (words with many senses are exactly the ones whose thesaurus neighbours wander off: "clear", "set",
"order"; those I group by hand). A *strict synonym* pair is two stems that share a WordNet synset AND are mutual
Moby links (both thesauri agree, README "Things to keep in mind"), and are not WordNet antonyms.

  Rule A (extension of hand groups). x joins the synonym group of a hand-lexicon stem m when x and m are strict
  synonyms and m is itself specific (at most MAXSYN synsets: a first version let any hand word attract members
  through its three commonest senses, and words such as "kid", "fair", "firm", "void" and "deck" pulled in
  "banter", "carnival", "unwavering", "excrete" and "bedeck"). m must be a word that carries a concept
  code and may have synonyms (not a function word, particle, marker, scope word, number word or single letter).
  If x qualifies for several groups it goes to the one with the most qualifying members; a tie drops x. x must
  not be an antonym of any member of the group.

  Rule B (general vocabulary). Among the remaining specific stems, strict-synonym pairs form star clusters:
  stems are visited in order of how many strict partners they have (most first, then alphabetically); an
  unassigned stem with unassigned partners becomes a head and takes those partners. Nothing is chained through
  a member, so every member of a cluster is a strict synonym of its head.

Output: one line per taken stem, "stem <tab> head <tab> via" (via = the hand stem it is a strict synonym of, or
"-" for rule B). build.py gives a rule-A stem the weight class (frame / half weight) of `via` and no other class.
"""
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK.parent / "kit" / "lib"))
import tokenizer as tk  # noqa: E402

TH = WORK.parent / "resources" / "thesaurus" / "out"
LEX = WORK / "lexicon"
MAXSYN = 3
GROUP_FILES = ["groups_actions.txt", "groups_objects.txt", "groups_states.txt", "groups_more.txt",
               "groups_general.txt"]
NO_GROUP_FILES = ["function.txt", "particle.txt", "marker.txt", "scope.txt"]  # words that never take synonyms


def stem_of(word):
    toks = tk.tokenize_text(word).tokens
    assert len(toks) == 1 and toks[0].kind == "word", word
    return toks[0].stem


def parse(field):
    out = {}
    for it in field.split():
        if ":" in it:
            s, n = it.rsplit(":", 1)
            out[s] = int(n)
        else:
            out[it] = 1
    return out


# ---- hand lexicon (the same reading as build.py) ---------------------------------------------------------------
hand = set()
closed = set()
for f in sorted(LEX.glob("*.txt")):
    for w in f.read_text(encoding="utf-8").split():
        st = stem_of(w)
        hand.add(st)
        if f.name in NO_GROUP_FILES:
            closed.add(st)
for w in ("two three four five six seven eight nine ten eleven twelve second third fourth fifth "
          "a b c d e f g h i j k l m n o p q r s t u v w x y z").split():
    st = stem_of(w)
    hand.add(st)
    closed.add(st)
group_of = {}
order = {}
for fname in GROUP_FILES:
    for line in (LEX / fname).read_text(encoding="utf-8").splitlines():
        ws = line.split()
        head = None
        for w in ws:
            st = stem_of(w)
            order.setdefault(st, len(order))
            if head is None:
                head = group_of.get(st, st)
            if st in group_of and group_of[st] != head:
                continue
            group_of[st] = head
members = {}
for st in hand:
    members.setdefault(group_of.get(st, st), set()).add(st)
excl_file = WORK / "thesaurus_exclude.txt"
exclude = set(excl_file.read_text(encoding="utf-8").split()) if excl_file.exists() else set()

# ---- thesaurus -------------------------------------------------------------------------------------------------
synsets = {}
with open(TH / "stems.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        p += [""] * (5 - len(p))
        synsets[p[0]] = [x for x in p[2].split(",") if x]
links = {}
with open(TH / "stem_links.tsv", encoding="utf-8") as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        p += [""] * (7 - len(p))
        links[p[0]] = (parse(p[1]), set(parse(p[5])), set(p[6].split()))  # syn, ant, mutual moby


def specific(x):
    return x.isalpha() and len(x) >= 3 and 1 <= len(synsets.get(x, ())) <= MAXSYN and x not in exclude


def strict(x):
    """Strict synonyms of x: shared synset and mutual Moby link, not antonyms."""
    if x not in links:
        return []
    syn, ant, moby = links[x]
    return [y for y in sorted(syn) if y in moby and y not in ant and y != x and x not in links.get(y, ({}, set(), set()))[1]]


out = []
taken = set()
# ---- rule A ----------------------------------------------------------------------------------------------------
for x in sorted(links):
    if x in hand or not specific(x):
        continue
    votes = {}
    for m in strict(x):
        if m not in hand or m in closed or not specific(m):
            continue
        votes.setdefault(group_of.get(m, m), []).append(m)
    if not votes:
        continue
    best = sorted(votes.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    if len(best) > 1 and len(best[0][1]) == len(best[1][1]):
        continue  # ambiguous between two groups
    head, via = best[0]
    ants = links[x][1]
    if ants & members.get(head, {head}):
        continue
    out.append((x, head, sorted(via, key=lambda m: (order.get(m, 10 ** 9), m))[0]))
    taken.add(x)
n_a = len(out)
# ---- rule B ----------------------------------------------------------------------------------------------------
cand = [x for x in sorted(links) if x not in hand and x not in taken and specific(x)]
cset = set(cand)
partners = {x: [y for y in strict(x) if y in cset] for x in cand}
assigned = set()
for h in sorted(cand, key=lambda x: (-len(partners[x]), x)):
    if h in assigned:
        continue
    free = [y for y in partners[h] if y not in assigned]
    if not free:
        continue
    assigned.add(h)
    for y in free:
        assigned.add(y)
        out.append((y, h, "-"))
(WORK / "thesaurus_groups.tsv").write_text(
    "# written by work/select_thesaurus.py; stem, head of its group, hand stem it is a strict synonym of\n"
    + "".join(f"{x}\t{h}\t{v}\n" for x, h, v in out), encoding="utf-8")
heads_b = {h for _, h, v in out if v == "-"}
print(f"rule A: {n_a} stems into {len({h for _, h, v in out[:n_a]})} hand groups; "
      f"rule B: {len(out) - n_a} stems under {len(heads_b)} heads ({len(out) - n_a + len(heads_b)} stems in all)")
if "--show" in sys.argv:
    by = {}
    for x, h, v in out[:n_a]:
        by.setdefault(h, []).append(x)
    lo = int(sys.argv[sys.argv.index("--show") + 1]) if len(sys.argv) > sys.argv.index("--show") + 1 else 0
    for h in sorted(by)[lo:lo + 400]:
        print(f"{h}: {' '.join(by[h])}")
