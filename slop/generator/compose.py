"""Screen composers for the dev generator (v2): one per screen kind. Each builds an abstract tree (``core.E``) and
decides every slot's label as it goes, planting the scored patterns of LABELLING.md §9 with their exact class,
pattern and target, and the K rules of §6 as hard negatives (KEEP records naming the rule in ``pattern``).

Every screen is about one topic: one product (``Ctx``: its domain, two nouns with synonyms, its people, its group).
Copy that is not planted adds a fact about that topic, so a KEEP subtitle talks about what its heading names. Both
sides of each decision come with and without shared stems: a restatement may use a synonym or a pronoun, and an
informative line may repeat the heading's noun. Copy is built from slots (``lexicon.py``), so that no frequent
string decides the class.

A clean screen (``Ctx.clean``) plants nothing redundant but keeps its hard negatives.
"""

from __future__ import annotations

import re

from . import lexicon as L
from . import tables as T
from .core import E, Page, S, article, cap, ctx, dyn, lower_first, plural, stack

KINDS = ["marketing", "app", "settings", "dialog", "empty", "dashboard", "game"]
KIND_WEIGHTS = [("marketing", 22), ("app", 16), ("settings", 16), ("dialog", 10), ("empty", 10), ("dashboard", 12),
                ("game", 14)]
FRAMEWORK_WEIGHTS = {
    "marketing": [("tsx", 70), ("html", 26), ("swift", 4)],
    "app": [("tsx", 62), ("html", 13), ("swift", 25)],
    "settings": [("tsx", 55), ("html", 20), ("swift", 25)],
    "dialog": [("tsx", 65), ("html", 10), ("swift", 25)],
    "empty": [("tsx", 60), ("html", 10), ("swift", 30)],
    "dashboard": [("tsx", 70), ("html", 12), ("swift", 18)],
    "game": [("tsx", 60), ("html", 15), ("swift", 25)],
}
CLEAN_SHARE = 0.15


class Noun:
    """A countable noun of the topic with synonyms that share no stem with it."""

    __slots__ = ("sg", "pl", "alts")

    def __init__(self, sg, pl, alts):
        self.sg, self.pl, self.alts = sg, pl, alts

    @property
    def Sg(self):
        return cap(self.sg)

    @property
    def Pl(self):
        return cap(self.pl)


class Ctx:
    def __init__(self, rng, fw: str, clean: bool):
        self.rng = rng
        self.fw = fw
        self.clean = clean
        self.dom = rng.pick(T.DOMAINS)
        ex = L.DOM_EXTRA[self.dom["key"]]
        self.x = Noun(*self.dom["obj"], ex["x"])
        self.y = Noun(*self.dom["obj2"], ex["y"])
        self.who = ex["who"]
        self.group = ex["group"]
        self.role = ex["role"]
        self.product = L.product_name(rng, self.dom)
        self.place = rng.pick(L.PLACES)
        self.r19: list = []
        self.props: list = []
        self.notice = None  # the first global notice (K6), for later repeats (R14)
        self.notice_alts: list = []
        self.cta = None  # the first call to action, for later repeats (R14)
        self.hero_h = None  # the hero's h1, for a closing section that repeats it (R9)
        self.nav_items: set = set()
        self.used: set = set()

    def want(self, p: float) -> bool:
        """Plant a redundant pattern with probability p (never on a clean screen)."""
        return (not self.clean) and self.rng.random() < p

    def chance(self, p: float) -> bool:
        return self.rng.random() < p

    def pick(self, xs):
        return self.rng.pick(xs)

    def fresh(self, xs):
        """Pick an item not used yet on this screen (copy never repeats by accident)."""
        cand = [x for x in xs if _key(x) not in self.used]
        x = self.rng.pick(cand or xs)
        self.used.add(_key(x))
        return x

    def fresh_n(self, xs, n):
        out = []
        for x in self.rng.shuffled(xs):
            if _key(x) in self.used:
                continue
            out.append(x)
            self.used.add(_key(x))
            if len(out) == n:
                break
        return out

    def prop(self, name: str) -> str:
        if name not in self.props:
            self.props.append(name)
        return name

    def noun(self) -> Noun:
        return self.x if self.chance(0.6) else self.y

    def alt(self, noun: Noun):
        return self.rng.pick(noun.alts)

    def f(self, t: str, **kw) -> str:
        """Fill a template with this topic's words and fresh numbers."""
        rng = self.rng
        n = rng.pick([2, 3, 4, 5, 6, 8, 10, 12, 14, 20, 30])
        vals = {
            "p": self.product, "x": self.x.pl, "x1": self.x.sg, "X": self.x.Pl, "X1": self.x.Sg,
            "y": self.y.pl, "y1": self.y.sg, "Y": self.y.Pl, "Y1": self.y.Sg,
            "who": self.who[1], "who1": self.who[0], "Who": cap(self.who[1]), "group": self.group,
            "Group": cap(self.group), "groups": plural(self.group), "Groups": cap(plural(self.group)), "role": self.role,
            "roles": plural(self.role), "Roles": cap(plural(self.role)), "place": self.place, "n": n,
            "N": f"{rng.randrange(120, 9800):,}", "d": rng.randrange(1, 9), "price": L.money(rng, 3, 40),
            "plan": rng.pick(L.PLANS[1:6]), "clock": L.clock(rng), "day": rng.pick(L.DAYS), "date": L.date(rng),
            "person": L.person(rng), "zone": rng.pick(L.ZONES),
        }
        vals.update(kw)
        for k in ("group", "x1", "y1", "who1", "role", "o"):
            if k in vals and isinstance(vals[k], str):
                vals["a_" + k] = article(vals[k])
                vals["A_" + k] = cap(article(vals[k]))
        return t.format(**vals)


def _key(x):
    return x if isinstance(x, str) else repr(x)


def words(s: str) -> set:
    return {w.strip(".,:;!?’'—-()").lower() for w in s.split()} - {""}


def shares_stem(a: str, b: str) -> bool:
    """A crude stem test (first five letters) for choosing between templates."""
    wb = words(b)
    for w in words(a):
        k = w[:5] if len(w) > 4 else w
        if len(k) > 2 and any(x.startswith(k) for x in wb):
            return True
    return False


def jitter(rng, value: str) -> str:
    """The same shape of figure with different digits ("46 kg" -> "31 kg", "£4,826" -> "£2,113")."""

    def one(m):
        s = m.group(0)
        digits = s.replace(",", "")
        if "." in digits:
            a, b = digits.split(".")
            v = rng.randrange(1, max(2, 10 ** len(a)))
            return (f"{v:,}" if "," in s else str(v)) + f".{rng.randrange(0, 10 ** len(b)):0{len(b)}d}"
        lo = 10 ** (len(digits) - 1) if len(digits) > 1 else 1
        v = rng.randrange(lo, 10 ** len(digits))
        return f"{v:,}" if "," in s else str(v)

    return re.sub(r"\d[\d,]*(?:\.\d+)?", one, value)


# ---------------------------------------------------------------------------------------------------------------
# copy builders shared by the composers


def about(c: Ctx, noun: Noun, stem=None) -> str:
    """A KEEP line about a listed noun: scope, timing, retention, limits. ``stem`` names the noun or not."""
    if stem is None:
        stem = c.chance(0.5)
    ctx_ = {"role": c.role, "group": c.group, "who": c.who[1]}
    if stem and c.chance(0.3):
        # a fact that opens with the same verbs as a restatement ("Browse ...", "Keep track of ...")
        return L.fill_fact(c.rng, c.pick(L.VERB_FACTS), noun.pl, noun.sg, ctx_)
    if stem:
        return L.fill_fact(c.rng, c.pick(L.LIST_FACTS), noun.pl, noun.sg, ctx_)
    return L.fill_fact(c.rng, c.pick(L.LIST_FACTS_ALT), noun.pl, noun.sg, ctx_)


def restate(c: Ctx, noun: Noun, stem=None) -> str:
    """A line that only says the screen holds ``noun`` (R3/R4 candidate)."""
    if stem is None:
        stem = c.chance(0.5)
    return L.echo_list(c.rng, noun.pl, None if stem else c.alt(noun)[1])


PRODUCT_FACTS = ["Every {x1} keeps a full history of who changed what, and when.", "{X} sync to every phone in your {group} within a minute.",
                 "Reminders go out {n} days before each {y1} is due.", "The free plan covers up to {n} {x}.",
                 "Works on iPhone, Android and the web, with one login.", "{X} you delete stay in the bin for {n} days.",
                 "Each {y1} can carry photos, notes and a location.", "{Who} in {place} keep about {N} {x} with us.",
                 "Exports open in any spreadsheet and keep your own column names.", "Only you and your {roles} can edit {x}."]


def product_fact(c: Ctx) -> str:
    """A KEEP sentence about the product: one of its own facts, or a fact about one of its nouns."""
    if c.chance(0.4):
        return c.fresh(c.dom["facts"])
    return c.f(c.fresh(PRODUCT_FACTS))


# v3: figures for each entry of lexicon.FACTS, in the same order
FACT_K = [[3, 5, 10, 25, 50], [7, 14, 21, 30], [2, 3, 5, 7], [0], [2, 4, 12, 24, 48], [2, 4, 6, 8, 12], [2, 3, 5],
          [0], [6, 12, 18, 24, 36], [12, 24, 48, 72], [2, 3, 5, 10], [4, 6, 8, 10, 12], [5, 10, 50, 100, 200],
          [10, 20, 25, 50], [0], [5, 10, 15, 20, 25]]


def num_word(k: int) -> str:
    """A figure at the start of a sentence: its word when short, else its digits."""
    return cap(L.NUMBER_WORDS[k]) if k in L.NUMBER_WORDS else str(k)


# facts that suit only some products (index into lexicon.FACTS -> domain keys); the rest suit every product
VENUES = {"pottery", "climb", "museum", "kayak", "tutor", "music", "vet", "bikes", "sail", "chess", "library", "ferry"}
FACT_DOMAINS = {3: VENUES - {"tutor", "ferry"}, 5: VENUES, 6: {"farm", "roast", "bikes", "photos", "brew"},
                9: VENUES, 11: {"pottery", "climb", "museum", "kayak", "tutor", "music", "birds", "sail", "rota"},
                14: {"sail", "chess", "birds", "photos", "podcast", "museum"}, 15: {"farm", "roast", "bikes", "vet", "rota", "fleet"}}


def fact_pick(c: Ctx, avoid=()):
    """A fact (lexicon.FACTS) not used yet on this screen and suited to the product, with the values its heading
    and restatements share."""
    ok = [i for i in range(len(L.FACTS)) if c.dom["key"] in FACT_DOMAINS.get(i, (c.dom["key"],))]
    idx = [i for i in ok if i not in avoid and ("fact", i) not in c.used] or ok
    i = c.pick(idx)
    c.used.add(("fact", i))
    k = c.pick(FACT_K[i])
    o1, o2 = c.pick(L.OPEN_TIMES)
    kw = {"k": k, "K": num_word(k), "k1": c.pick([1, 2]), "m": c.pick([3, 6, 12, 20]), "price": f"£{c.rng.randrange(3, 60)}",
          "o1": o1, "o2": o2, "date": L.date(c.rng), "day": c.pick(L.DAYS)}
    return i, L.FACTS[i], kw


def fact_text(c: Ctx, t: str, kw: dict) -> str:
    return cap(c.f(t, **kw))


def short_keep(c: Ctx, *kinds) -> str:
    """A short plain line that adds one fact and carries no figure (KEEP)."""
    pool = [t for k in (kinds or tuple(L.SHORT_KEEP)) for t in L.SHORT_KEEP[k]]
    return c.f(c.fresh(pool))


def narrate_control(c: Ctx, label: str) -> str:
    """R6: what pressing a visible control does, in words its label already gives."""
    r = c.rng.random()
    v = c.pick(L.TAP_VERBS)
    if r < 0.6:
        return f"{v} {label} {c.pick(L.NARRATE_PURPOSE)}."
    if r < 0.8:
        return f"Use the {label} button {c.pick(['below', 'underneath', 'at the bottom'])} {c.pick(L.NARRATE_PURPOSE)}."
    return f"{c.pick(L.NARRATE_FORM_OPEN)} {v.lower()} {label}."


def scroll_line(c: Ctx, noun: Noun | None = None) -> str:
    if noun is None:
        return c.pick(L.SCROLL_GENERIC)
    return c.pick(L.SCROLL_LINES).format(x=noun.pl, x1=noun.sg, a_x1=article(noun.sg))


COUNT_KEEP = {
    "features": [("{N} of the {m} features, picked for {who}; the rest are in the guide.", False),
                 ("These {n} come with every plan; {m} more are on Pro.", False),
                 ("Every one of the {n} works offline.", True), ("All {n} are included on the free plan.", True)],
    "steps": [("Most {groups} finish all {n} steps in a day.", True), ("All {n} steps work offline.", True),
              ("The {n} steps take about {k} minutes in all.", True), ("Steps 2 to {n} can wait until later.", False)],
    "pricing": [("All {n} plans include VAT.", True), ("Every one of the {n} plans has a {k}-day trial.", True),
                ("Schools get {k}% off all {n} plans.", True), ("{N} plans here; charities ask us for a fourth.", False)],
    "dash": [("{N} figures from the last {k} days; older ones are in Reports.", True), ("These {n} figures exclude drafts.", True),
             ("{N} of the {m} figures you follow; edit the list in Settings.", False)],
}


def count_keep(c: Ctx, n: int, kind: str):
    """A line that names the count of the items shown and adds a fact (KEEP, partial when the count is all it
    narrates), or names a count of items that are not all shown (KEEP)."""
    t, part = c.pick(COUNT_KEEP[kind])
    w = L.NUMBER_WORDS.get(n, str(n))
    s = S(c.f(t, n=w, N=cap(w), m=n + c.pick([3, 5, 8, 12]), k=c.pick([10, 14, 20, 30])), "subtitle")
    return s.keep(partial=True, conf="judgement") if part else s


def count_line(c: Ctx, n: int, noun: Noun | None = None) -> str:
    noun = noun or c.x
    t = c.pick(L.COUNT_LINES)
    w = L.NUMBER_WORDS.get(n, str(n))
    return t.format(n=w if c.chance(0.6) else str(n), N=cap(w), x1=noun.sg, p=c.product)


def layout_line(c: Ctx, noun: Noun) -> str:
    return c.pick(L.LAYOUT_LINES).format(x=noun.pl, x1=noun.sg)


def closing_spread(c: Ctx) -> str:
    a = c.pick(L.CLOSE_SPREAD_A)
    b = c.pick(L.CLOSE_SPREAD_B)
    return (a + b).format(p=c.product, x=c.x.pl)


def closing_head(c: Ctx, htext: str) -> str:
    h = lower_first(htext.rstrip("."))
    return c.pick(L.CLOSE_HEAD).format(h=h, H=cap(h), p=c.product)


def static_status(c: Ctx) -> str:
    return c.pick(L.STATUS_GLYPH) + c.pick(L.STATUS_WORDS + ["Now open", "Live now", "Available today", "Open now"])


def pill(c: Ctx) -> str:
    return c.pick(L.PILL_GLYPH) + c.pick(L.PILL_WORDS)


def nothing_line(c: Ctx) -> str:
    return (f"{c.pick(['Nothing', 'Nothing at all', 'Not a thing'])} {c.pick(['to show', 'here', 'to see', 'in here'])}"
            f" {c.pick(['yet', 'so far', 'for now', 'just now', 'at the moment'])}.")


def notice(c: Ctx):
    group = c.pick(L.NOTICE_KINDS)[0]
    kw = {"n": c.pick([7, 14, 21, 30, 60]), "d": c.rng.randrange(1, 9), "N": f"{c.rng.randrange(120, 9800):,}",
          "price": L.money(c.rng, 3, 40)}
    t0 = c.pick(group)
    first = c.f(t0, **kw)
    s = S(first, "caption").keep("K6")
    c.notice, c.notice_alts = s, [c.f(x, **kw) for x in group if x != t0] + [first]
    return E("note", s=s)


def repeat_notice(c: Ctx):
    """A later copy of the global notice (R14), or None."""
    if c.notice is None:
        return None
    return E("note", s=S(c.pick(c.notice_alts), "caption").echo(c.notice, "R14", conf="clear"))


def cta_buttons(c: Ctx, first=True):
    ctas = c.dom["cta"]
    b = S(ctas[0] if first else c.pick(ctas[1:]), "button")
    kids = [E("button", s=b)]
    if first:
        c.cta = b
        c.used.add(b.text)
    if c.chance(0.6):
        if c.chance(0.15):
            kids.append(E("link", s=S(c.pick(T.VACUOUS), "link").keep(aside=["R7"]), arrow=c.chance(0.4)))
        else:
            kids.append(E("link", s=S(c.fresh([x for x in ctas if x != b.text]), "link"), arrow=c.chance(0.4)))
    return E("actions", *kids), b


def item_names(c: Ctx) -> list:
    """Names of individual things of this topic (the first table column)."""
    return list(c.dom["table"][1][0])


# ---------------------------------------------------------------------------------------------------------------
# eyebrows


def genre_group(word: str):
    for g in L.GENRE_GROUPS:
        if word in g:
            return g
    return None


def eyebrow(c: Ctx, h, htext: str, concept=None, genre=None, p_r1=0.3, p_r2=0.12, genres=("Overview", "Products"),
            allow_k1=True, syn=None, cats_ok=True):
    """An eyebrow over a heading, or None. ``concept`` gives paraphrase eyebrows (R1); ``genre`` is a section-genre
    word the heading contains (R1); ``syn`` lists restatements of a page title (R1); ``genres`` are genre words
    that suit the section (R2 when the heading does not contain them)."""
    if concept is not None and c.want(p_r1):
        return S(c.pick(concept["eb"]), "eyebrow").echo(h, "R1", conf="judgement")
    if genre is not None and c.want(p_r1):
        return S(genre, "eyebrow").echo(h, "R1", conf="clear")
    if syn and c.want(p_r1):
        return S(c.pick(syn), "eyebrow").echo(h, "R1", conf="judgement")
    if c.want(p_r2):
        # genre words that suit the section, and their variants ("Word of mouth" for Testimonials): DECOR when the
        # heading neither shares the word's stem nor is a genre word of the same kind (LABELLING.md §6 K1, §8)
        words_ = list(dict.fromkeys(w for g in genres for w in (genre_group(g) or [g])))
        cand = [w for w in words_ if not shares_stem(w, htext) and not (genre_group(w) and genre_group(w) == genre_group(htext))
                and not any(shares_stem(x, htext) for x in (genre_group(w) or []))]
        if cand and c.chance(0.6):
            w = c.pick(cand)
            return S(w, "eyebrow").decor("R2", conf="clear" if w in L.GENRE_HEADING or w in genres else "judgement")
        return S(c.pick(L.NEXT_HEAD), "eyebrow").decor("R2")
    if allow_k1 and c.chance(0.3):
        r = c.rng.random()
        if r < 0.45 and cats_ok:
            text = c.fresh(c.dom["cats"])
        elif r < 0.55:
            # a content category (K1)
            text = c.pick(["Field guide", "Member offer", "Product update", "Release notes", "Season guide", "Workshop",
                           "Tutorial", "Guide", "Event", "Update", "Spring update", "Explainer"])
        elif r < 0.7:
            text = c.product
        else:
            text = c.pick([f"Issue {c.rng.randrange(2, 60)}", f"{c.pick(['Spring', 'Summer', 'Autumn', 'Winter'])} 2026",
                           f"{c.pick(L.MONTHS)} 2026", f"Release 4.{c.rng.randrange(1, 12)}",
                           f"Issue {c.rng.randrange(2, 30)} · {c.pick(L.MONTHS)} 2026"])
        if text.lower() == htext.lower():
            return None
        return S(text, "eyebrow").keep("K1")
    return None


# ---------------------------------------------------------------------------------------------------------------
# marketing

HERO_HEADS = ["Keep every {x1} on track", "{X}, sorted", "Run your {y} from one screen", "The calm way to manage {x}",
              "Your {x}, without the paperwork", "Spend less time on {y}", "Where {who} keep their {x}",
              "Every {x1} and every {y1}, in one app", "{p} for {groups}", "Less admin for {who}"]
CAT_HEADS = ["{cat}, without the guesswork", "New in {cat} this month", "{cat}: the {p} guide", "{cat} at {p}",
             "{cat}, done properly"]
H2_HEADS = ["What {p} does", "Made for {x}", "Everything {a_x1} needs", "Why {who} switch", "Less admin, more {x}",
            "Built around your {x}", "The tools behind every {x1}", "What you get with {p}", "A closer look at {p}",
            "Inside {p}"]


def m_hero(c: Ctx):
    r = c.rng.random()
    concept = cat = None
    if r < 0.14:
        return m_facts(c, hero=True)
    if r < 0.45:
        concept = c.pick(L.CONCEPTS)
        htext = c.f(c.pick(concept["head"]))
    elif r < 0.6:
        cat = c.fresh(c.dom["cats"])
        htext = c.f(c.pick(CAT_HEADS), cat=cat)
    else:
        htext = c.f(c.pick(HERO_HEADS))
    h1 = S(htext, "h1")
    c.hero_h = h1
    kids, top = [], []
    if c.want(0.06):
        top.append(E("meta", items=[S(x, "caption").decor("R2") for x in L.meta_strip(c.rng)]))
    elif c.chance(0.06):
        top.append(E("meta", items=[S(x, "caption").keep("K1" if not x.startswith("v") else None) for x in L.keep_meta(c.rng)]))
    if c.want(0.1):
        top.append(E("badge", s=S(static_status(c), "badge").decor("R11")))
    elif c.chance(0.05):
        top.append(E("badge", s=S(c.pick(["Early access", "Now on Android", "New in 4." + str(c.rng.randrange(1, 9))]),
                                  "badge").keep(conf="judgement")))
    if cat is not None:
        eb = S(cat, "eyebrow").keep("K1")  # a category that the heading also names (K1)
    else:
        eb = eyebrow(c, h1, htext, concept=concept, p_r1=0.35, genres=("Overview", "Get started", "Products"))
    if eb is None and c.chance(0.08):
        eb = S("New: " + lower_first(c.pick(c.dom["features"])[0]), "eyebrow").keep()
    if eb:
        kids.append(E("eyebrow", s=eb))
    kids.append(E("h", s=h1, level=1))
    sub = None
    if concept and c.want(0.3):
        sub = S(c.f(c.pick(concept["echo"])), "subtitle").echo(h1, "R3", conf="judgement")
    elif c.want(0.05):
        sub = S(scroll_line(c), "subtitle").narrate("R6", None, ct=True)
    elif concept and c.chance(0.14):
        sub = S(c.f(c.pick(concept["echo"])) + " " + c.f(c.pick(concept["keep"])), "subtitle").keep(partial=True, conf="judgement")
    elif concept and c.chance(0.75):
        sub = S(c.f(c.pick(concept["keep"])), "subtitle")
    elif c.chance(0.8):
        if cat is None and concept is None and c.chance(0.5):
            noun = c.x if c.x.pl in htext.lower() or c.x.sg in htext.lower() else c.y
            t = c.pick(["{P} sync to every phone in your {group} within a minute.", "Every {s} keeps a full history of who changed what.",
                        "The free plan covers up to {n} {pl}.", "{P} you delete stay in the bin for {n} days.",
                        "{P} can carry photos, notes and a location.", "Share {pl} with your {group} by link; no account needed."])
            sub = S(c.f(t, P=noun.Pl, pl=noun.pl, s=noun.sg), "subtitle")
        else:
            sub = S(product_fact(c), "subtitle")
    if sub is not None:
        kids.append(E("sub", s=sub))
    acts, _ = cta_buttons(c)
    body = top + [E("head", *kids), acts]
    if c.chance(0.4):
        body.append(notice(c))
    return E("section", *body, tone="hero")


def m_facts(c: Ctx, hero=False):
    """A section about one stated fact (a limit, a price, a time, a figure). An eyebrow, the subtitle or the first
    line may repeat the heading's figure and add nothing (R1, R3, R4), or add another fact (KEEP); cards below state
    further facts, and a card body may repeat its own title's figure (R5)."""
    i, f, kw = fact_pick(c)
    topic = not hero and c.chance(0.35)
    # a topic heading carries no figure, so a line that states the figure adds it (KEEP); under a heading that
    # states the figure, the same line only repeats it (R3)
    h = S(fact_text(c, c.pick(f["topic"] if topic else f["head"]), kw), "h1" if hero else "h2")
    if hero:
        c.hero_h = h
    if h.text in c.nav_items:
        h.keep("K11")
    kids = []
    if topic and c.chance(0.2):
        kids.append(E("eyebrow", s=S(fact_text(c, c.pick(f["eb"]), kw), "eyebrow").keep(conf="judgement")))
    elif not topic and c.want(0.22):
        kids.append(E("eyebrow", s=S(fact_text(c, c.pick(f["eb"]), kw), "eyebrow").echo(h, "R1", conf="clear")))
    elif c.chance(0.25):
        kids.append(E("eyebrow", s=S(c.pick([c.product, c.fresh(c.dom["cats"])]), "eyebrow").keep("K1")))
    kids.append(E("h", s=h, level=1 if hero else 2))
    sub_echo = not topic and c.want(0.4)
    if sub_echo:
        kids.append(E("sub", s=S(fact_text(c, c.fresh(f["echo"]), kw), "subtitle").echo(h, "R3")))
    elif topic and c.chance(0.75):
        kids.append(E("sub", s=S(fact_text(c, c.fresh(f["echo"]), kw), "subtitle")))
    elif c.chance(0.7):
        kids.append(E("sub", s=S(fact_text(c, c.fresh(f["keep"]), kw), "subtitle")))
    out = [E("head", *kids)]
    if hero:
        acts, _ = cta_buttons(c)
        out.append(acts)
        if c.chance(0.3):
            out.append(notice(c))
        return E("section", *out, tone="hero")
    if c.chance(0.65):
        cards = []
        for _ in range(c.pick([2, 3, 3, 4])):
            _, g, kw2 = fact_pick(c, avoid=(i,))
            if c.chance(0.4):
                t = S(fact_text(c, c.pick(g["topic"]), kw2), "h3")
                b = S(fact_text(c, c.pick(g["echo"] if c.chance(0.7) else g["keep"]), kw2), "body")
            else:
                t = S(fact_text(c, c.pick(g["head"]), kw2), "h3")
                if c.want(0.3):
                    b = S(fact_text(c, c.pick(g["echo"]), kw2), "body").echo(t, "R5")
                else:
                    b = S(fact_text(c, c.pick(g["keep"]), kw2), "body")
            cards.append(E("card", E("h", s=t, level=3), E("p", s=b)))
        out.append(E("grid", *cards, mappable=True, name="facts"))
    elif not sub_echo and c.want(0.35):
        out.append(E("p", s=S(fact_text(c, c.fresh(f["echo"]), kw), "body").echo(h, "R4")))
    else:
        out.append(E("p", s=S(fact_text(c, c.fresh(f["keep"]), kw), "body")))
    return E("section", *out)


CARD_ECHO_STEM = ["{t}, made simple.", "{t} for everyone.", "All about {tl}.", "{t}, right when you need it.",
                  "{t}, built in.", "Easy {tl}."]
CARD_ECHO_ALT = ["Exactly what it sounds like.", "Does what it says.", "Just as you would expect.", "Simple, and it works.",
                 "No surprises here.", "It is all in the name."]
BULLETS = ["Unlimited {x}", "{fmt} export", "Shared with your {group}", "Works offline for {n} days", "Reminders by {via}",
           "{n} years of history", "Photos on every {x1}", "Notes on every {x1}", "Works on iPhone and Android",
           "A printable summary each {period}", "Read-only links for {who}", "Up to {n} {who} per {group}",
           "History kept for {n} months", "{n} {x} a {period} on the free plan", "Comments on every {x1}"]
FMT = ["CSV and PDF", "CSV", "PDF", "Excel and CSV", "PDF and print"]
VIA = ["email or text", "email", "push or email", "text message"]


def card_grid(c: Ctx, n: int, name="features", lead=None):
    """Feature cards. With ``lead`` (the section heading, itself a feature title) the first card repeats the
    heading as its title (R9)."""
    feats = c.fresh_n(c.dom["features"], n)
    if lead is not None:
        feats = [f for f in feats if f[0] != lead.text][: max(1, n - 1)]
        feats.insert(0, next(f for f in c.dom["features"] if f[0] == lead.text))
    echo_at = c.rng.randrange(1 if lead is not None else 0, len(feats)) if c.want(0.3) and len(feats) > 1 else None
    device = c.rng.wpick([(None, 46), ("pill", 14), ("sibling", 12), ("index", 12), ("emoji", 6), ("bullets", 16)])
    if device == "emoji" and lead is not None:
        device = None
    if device == "pill" and c.clean:
        device = None
    pill_text = pill(c)
    sib = c.fresh_n(L.SIBLING_PILLS, len(feats))
    sib_glyph = c.pick(["", "", "✨ ", "🔒 ", "● ", "📱 "])
    style = c.pick(T.INDEX_STYLES)
    emojis = c.rng.shuffled(["✨", "⚡", "🌱", "🔒", "📦", "🧭", "🔔", "📈"])
    cards, devs = [], []
    for i, (title, body) in enumerate(feats):
        kids = []
        t = S(title, "h3")
        if lead is not None and i == 0:
            t.echo(lead, "R9", conf="clear")
        if i == echo_at:
            if c.chance(0.3):
                ce = c.pick(T.CARD_ECHO)
                t = S(ce[0], "h3")
                b = S(c.pick(ce[1]), "body").echo(t, "R5", conf="clear")
            elif c.chance(0.5):
                b = S(c.pick(CARD_ECHO_STEM).format(t=title, tl=lower_first(title)), "body").echo(t, "R5", conf="clear")
            else:
                b = S(c.pick(CARD_ECHO_ALT), "body").echo(t, "R5", conf="judgement")
        else:
            b = S(body, "body")
        if device == "index":
            d = S(style.format(n=i + 1, nn=f"{i + 1:02d}"), "eyebrow").keep("K3", aside=["R12"])
            kids.append(E("eyebrow", s=d))
            devs.append(d)
        elif device == "pill":
            d = S(pill_text, "badge").decor("R11")
            kids.append(E("badge", s=d))
            devs.append(d)
        elif device == "sibling" and i < len(sib):
            glyph = sib_glyph if sib_glyph and i % 2 == 0 else ""
            kids.append(E("badge", s=S(glyph + sib[i], "badge").keep("K2", aside=["R11g"] if glyph else None)))
        if device == "emoji":
            t.text = emojis[i % len(emojis)] + " " + t.text
            t.add_aside("R11g")
        kids.append(E("h", s=t, level=3))
        kids.append(E("p", s=b))
        if device == "bullets":
            items = []
            if c.want(0.25):
                if c.chance(0.5):
                    items.append(S(c.pick(["{t}, built in", "Full {tl}", "{t} included", "{t} as standard"]).format(
                        t=title, tl=lower_first(title)), "list_item").echo(t, "R4", conf="clear"))
                else:
                    items.append(S(c.pick(["Exactly as the name says", "Just what you would expect", "All of the above, and more"]),
                                   "list_item").echo(t, "R4", conf="judgement"))
            for bt in c.rng.some(BULLETS, c.rng.pick([2, 3])):
                items.append(S(c.f(bt, fmt=c.pick(FMT), via=c.pick(VIA), period=c.pick(["week", "month"])), "list_item"))
            if c.want(0.12):
                items.append(S(c.pick(["And everything else you need", "All in one place", "Everything, in one app",
                                       "Plus much more besides"]), "list_item").echo(None, "R17", conf="judgement"))
            kids.append(E("ul", items=items))
        cards.append(E("card", *kids))
    if device == "sibling" and len(sib) < len(feats):
        for cd in cards:
            cd.kids = [k for k in cd.kids if k.kind != "badge"]
    if device == "bullets":
        # restated bullets need distinct texts: drop duplicates across cards
        seen = set()
        for cd in cards:
            for k in cd.kids:
                if k.kind == "ul":
                    k.p["items"] = [s for s in k["items"] if not (s.text in seen or seen.add(s.text))] or k["items"][:1]
    if len(devs) >= 3:
        c.r19.append(devs)
    return E("grid", *cards, mappable=device != "bullets", name=name), len(cards)


def section_heading(c: Ctx, genres_ok: tuple, concept_ok=True, default_heads=H2_HEADS, cats_ok=False):
    """A marketing h2 and its eyebrow: a concept, a genre word inside a heading, a bare genre word (eyebrow of the
    same kind: ECHO; of another kind: DECOR), a category heading (K1 eyebrow sharing a stem) or a plain heading."""
    r = c.rng.random()
    concept = genre = None
    if concept_ok and r < 0.3:
        concept = c.fresh(L.CONCEPTS)
        htext = c.f(c.pick(concept["head"]))
    elif r < 0.48:
        genre = c.pick([g for g in genres_ok if g in L.GENRE_HEADING] or ["Features"])
        htext = c.f(c.pick(L.GENRE_HEADING[genre]))
    elif r < 0.6:
        # the heading is itself a section-genre word (LABELLING.md K1, 2026-11-10)
        grp = genre_group(c.pick(genres_ok)) or L.GENRE_GROUPS[0]
        htext = c.pick(grp)
        h2 = S(htext, "h2")
        if htext in c.nav_items:
            h2.keep("K11")
        eb = None
        if c.want(0.4):
            same = [w for w in grp if w != htext and not shares_stem(w, htext)] or [w for w in grp if w != htext]
            eb = S(c.pick(same), "eyebrow").echo(h2, "R1", conf="clear")
        elif c.want(0.2):
            other = [w for g in L.GENRE_GROUPS if g is not grp for w in g]
            eb = S(c.pick(other), "eyebrow").decor("R2")
        elif c.chance(0.25):
            eb = S(c.product, "eyebrow").keep("K1")
        return h2, eb, None
    elif r < 0.7 and cats_ok and c.dom["cats"]:
        cat = c.fresh(c.dom["cats"])
        htext = c.f(c.pick(CAT_HEADS), cat=cat)
        h2 = S(htext, "h2")
        return h2, S(cat, "eyebrow").keep("K1") if c.chance(0.7) else None, None
    else:
        htext = c.f(c.pick(default_heads))
    h2 = S(htext, "h2")
    if htext in c.nav_items:
        h2.keep("K11")
    eb = eyebrow(c, h2, htext, concept=concept, genre=genre, p_r1=0.4, genres=genres_ok, cats_ok=cats_ok)
    return h2, eb, concept


def m_features(c: Ctx):
    lead = None
    avail = [ft for ft in c.dom["features"] if _key(ft) not in c.used]
    if len(avail) >= 3 and c.want(0.2):
        # the section is named after one feature and its first card repeats that name (R9, a title in a grid)
        ft = c.pick(avail)
        c.used.add(_key(ft))
        h2, eb, concept = S(ft[0], "h2"), None, None
        lead = h2
    else:
        h2, eb, concept = section_heading(c, ("Features", "Capabilities", "Why us", "Overview"), cats_ok=True)
    kids = [E("eyebrow", s=eb)] if eb else []
    kids.append(E("h", s=h2, level=2))
    n = c.rng.wpick([(2, 1), (3, 5), (4, 3), (5, 1), (6, 2)])
    grid, n = card_grid(c, max(n, 3) if lead is not None else n, lead=lead)
    if c.want(0.1):
        kids.append(E("sub", s=S(count_line(c, n), "subtitle").narrate("R16", None)))
    elif c.chance(0.07):
        kids.append(E("sub", s=count_keep(c, n, "features")))
    elif concept and c.want(0.3):
        kids.append(E("sub", s=S(c.f(c.pick(concept["echo"])), "subtitle").echo(h2, "R3", conf="judgement")))
    elif concept and c.chance(0.15):
        kids.append(E("sub", s=S(c.f(c.pick(concept["echo"])) + " " + c.f(c.pick(concept["keep"])), "subtitle")
                      .keep(partial=True, conf="judgement")))
    elif concept and c.chance(0.6):
        kids.append(E("sub", s=S(c.f(c.pick(concept["keep"])), "subtitle")))
    elif c.chance(0.5):
        kids.append(E("sub", s=S(product_fact(c), "subtitle")))
    out = [E("head", *kids), grid]
    if c.want(0.25):
        if concept is None or c.chance(0.5):
            line = S(closing_spread(c), "body").echo(None, "R17", conf="judgement")
        else:
            line = S(closing_head(c, h2.text), "body").echo(h2, "R17")
        out.append(E("p", s=line))
    elif c.chance(0.15):
        out.append(E("p", s=S(product_fact(c), "body")))
    return E("section", *out)


def m_spotlight(c: Ctx):
    """A section on one feature, shown as a single card. The card title may repeat the section heading (R9); an h4
    inside may repeat the card title (R9); either may instead share its words and add a fact (KEEP)."""
    title, body = c.fresh(c.dom["features"])
    extended = c.chance(0.35)
    h2 = S(c.f(f"{title}{c.pick([', on every plan', ' for {groups}', ', now on Android', ' in {place}', ', built in', ' for every {x1}'])}")
           if extended else title, "h2")
    head = []
    if extended and c.want(0.4):
        # an eyebrow that is the heading without its qualifier (R1)
        head.append(E("eyebrow", s=S(title, "eyebrow").echo(h2, "R1", conf="clear")))
    elif c.chance(0.25):
        eb = S(c.pick([c.product, c.fresh(c.dom["cats"])]), "eyebrow").keep("K1")
        if not shares_stem(eb.text, h2.text):
            head.append(E("eyebrow", s=eb))
    head.append(E("h", s=h2, level=2))
    if c.chance(0.4):
        # a fact about the feature itself (when and where it is available), not one borrowed from elsewhere
        head.append(E("sub", s=S(c.f(c.pick(["On every plan since {date}.", "Included on the {plan} plan and above.",
                                               "Used by {N} {who} last month.", "New in release 4.{d}.",
                                               "On iPhone, Android and the web.", "Turned on for every {group} by default.",
                                               "Off by default; {roles} can turn it on.", "Free during the beta."])),
                                   "subtitle")))
    ck = []
    r = c.rng.random()
    if c.want(0.4):
        h3 = S(title if c.chance(0.7) else c.pick([f"About {lower_first(title)}", f"{title}, explained", f"Introducing {lower_first(title)}"]),
               "h3").echo(h2, "R9", conf="clear")
        if h3.text != h2.text:
            h3.conf = "judgement"
    elif r < 0.5:
        # the heading plus a qualifier (LABELLING.md §8: KEEP, partial)
        h3 = S(c.f(c.pick([f"{title} on the {{plan}} plan", f"{title} in {{place}}", f"{title}: new in 4.{{d}}",
                           f"{title} for {{groups}}"])), "h3").keep(partial=True, conf="judgement")
    else:
        h3 = S(c.f(c.pick(["How {who} use it", "In practice", "Why {who} asked for it", "Since {date}"])), "h3")
    ck.append(E("h", s=h3, level=3))
    ck.append(E("p", s=S(body, "body")))
    if c.chance(0.45):
        if c.want(0.3):
            h4 = S(h3.text if c.chance(0.6) else f"More on {lower_first(h3.text)}", "h4").echo(h3, "R9", conf="clear")
            if h4.text != h3.text:
                h4.conf = "judgement"
        else:
            h4 = S(c.f(c.pick(["{n} {who} use it weekly", "Included from {date}", f"{h3.text.split(':')[0]} limits", "Who can see it"])), "h4")
            if h4.text.endswith(" limits"):
                h4.keep(partial=True, conf="judgement")
        ck.append(E("h", s=h4, level=4))
        ck.append(E("ul", items=[S(c.f(b, fmt=c.pick(FMT), via=c.pick(VIA), period=c.pick(["week", "month"])), "list_item")
                                 for b in dict.fromkeys(c.rng.some(BULLETS, c.pick([2, 3])))]))
    if c.chance(0.4):
        ck.append(E("link", s=S(c.pick([f"See {lower_first(title)}", "Read the guide", f"Try it on {article(c.x.sg)}"]), "link"), arrow=c.chance(0.5)))
    return E("section", E("head", *head), E("card", *ck, shadcn=False))


STEP_POOL = [
    ("Create your account", ["It takes a minute with an email address.", "Sign up with an email address or a Google account."]),
    ("Add your first {x1}", ["Most people start with the one they check most often.", "Give it a name; everything else can wait."]),
    ("Invite your {group}", ["They get a link that works for {n} days.", "Each person gets their own login."]),
    ("Bring in your {y}", ["Spreadsheets with a header row come in with their columns matched.", "Paste from a spreadsheet or upload a CSV."]),
    ("Set your reminders", ["Pick the day and time; reminders follow your time zone.", "Choose email, text or both."]),
    ("Add {a_y1}", ["From your phone, even with no signal.", "It takes about {n} taps."]),
    ("Share your {x}", ["Each person sees only what you choose.", "Read-only links need no account."]),
    ("Print a summary", ["One A4 page per {x1}, ready for the noticeboard.", "Or save it as a PDF."]),
]


def m_steps(c: Ctx):
    r = c.rng.random()
    if r < 0.25:
        genre = "How it works"
        htext = c.f(c.pick(L.GENRE_HEADING[genre]))
        h2 = S(htext, "h2")
        eb = S(genre, "eyebrow").echo(h2, "R1") if c.want(0.5) else None
    elif r < 0.4:
        h2, eb, _ = section_heading(c, ("How it works", "Get started"), concept_ok=False)
        htext = h2.text
    else:
        htext = c.f(c.pick(["From sign-up to your first {x1}", "Getting going takes an afternoon", "Set up {p} in an afternoon",
                            "Your first week with {p}", "Three steps to tidier {x}", "From spreadsheet to {p}"]))
        h2 = S(htext, "h2")
        eb = eyebrow(c, h2, htext, allow_k1=False, p_r1=0, p_r2=0.25, genres=("How it works", "Get started"))
    kids = [E("eyebrow", s=eb)] if eb else []
    kids.append(E("h", s=h2, level=2))
    n = c.pick([3, 3, 4, 5])
    picks = c.rng.some(STEP_POOL, n)
    if c.want(0.2):
        kids.append(E("sub", s=S(count_line(c, n), "subtitle").narrate("R16", None)))
    elif c.chance(0.1):
        kids.append(E("sub", s=count_keep(c, n, "steps")))
    elif c.chance(0.4):
        kids.append(E("sub", s=S(c.f(c.pick(["Most {groups} are set up within a day.", "You can stop after step 2 and come back later.",
                                               "Nothing is shared until you invite someone.", "Each step can be skipped."])), "subtitle")))
    style = c.pick(T.STEP_STYLES)
    numbered = c.chance(0.85)
    items = []
    for i, (t, bodies) in enumerate(picks):
        num = S(style.format(n=i + 1, total=n), "caption").keep("K3") if numbered else None
        items.append((num, S(c.f(t), "h3"), S(c.f(c.pick(bodies)), "body") if c.chance(0.8) else None))
    out = [E("head", *kids), E("steps", items=items)]
    if c.want(0.25):
        out.append(E("p", s=S(c.pick(L.CLOSE_STEPS).format(N=cap(L.NUMBER_WORDS[n]), x1=c.x.sg), "body")
                     .echo(None, "R17", conf="judgement")))
    return E("section", *out)


# Plan cards: each tier has a seat count, a site count and a limit on the main noun (None: unlimited); the card's
# subtitle and its list are both drawn from those numbers, so they never disagree.
TIER_SOLO = ["For one {who1} trying it out.", "For a single {who1} getting started.", "Just you, and up to {xl} {x}.",
             "Solo use, with up to {xl} {x}.", "One login and {xl} {x}.", "For one person keeping {xl} {x} or fewer.",
             "For {a_who1} on their own.", "A place to start, for one {who1}."]
TIER_TEAM = ["For {a_group} of up to {s} people.", "For up to {s} {who} sharing {x}.", "For {groups} of {s} or fewer.",
             "{S} people, up to {xl} {x}.", "For {a_group} that shares {x}, up to {s} people.", "For {who} who log {y} every day.",
             "For small {groups} that share {x}.", "Room for {s} {who}."]
TIER_BIG = ["For larger {groups} that need an audit trail of every {x1}.", "For any size of {group}, with no seat limit.",
            "Unlimited people, for {groups} that keep growing.", "For {groups} that want every {x1} on record.",
            "No limits on people or {x}."]
TIER_SITES = ["For busy {groups} with up to {sites} sites.", "For {groups} working across {sites} sites.",
              "For {groups} with more than one site."]
TIER_GENERIC = ["{fmt} export", "{support} support", "Two-step sign-in {scope}", "Backups kept for {n} days", "{n} GB of photos",
                "Reports by {period}", "API access", "Custom fields on {x}", "Unlimited {y}", "Approvals by {roles}",
                "{Y} on a shared calendar", "Printable {x1} sheets", "Audit log of every change", "Single sign-on",
                "Priority booking", "Bulk import from CSV", "Offline mode", "Photo uploads", "Comments on {y}", "Recurring {y}",
                "Shared folders", "Year-end summary", "Custom reminders", "Read-only guest links", "Tags and filters",
                "Templates for new {x}"]


TIER_ECHO = {"Free": ["Free of charge.", "Costs nothing, as the name says."], "Starter": ["For getting started.", "A plan to start with."],
             "Plus": ["A little extra.", "The plus plan, with more."], "Pro": ["For pros.", "The professional plan."],
             "Team": ["For teams.", "Built for teams, as the name says."], "Studio": ["For studios.", "The studio plan."],
             "Club": ["For clubs.", "Made for clubs."], "Society": ["For societies.", "The society plan."]}


def tier_ladder(c: Ctx, n: int) -> list:
    """(seats, sites, limit) per tier, growing from the first to the last; None means unlimited."""
    seats = [c.pick([1, 1, 2, 3])]
    sites = [1]
    lim = [c.pick([3, 5, 10, 20])]
    for i in range(1, n):
        last = i == n - 1
        seats.append(None if last and c.chance(0.5) else (seats[-1] or 1) * c.pick([3, 4, 5]))
        sites.append(None if last and c.chance(0.12) else (c.pick([3, 5]) if (last and c.chance(0.3)) else 1))
        if sites[-1] is not None and sites[-2] is not None and sites[-1] < sites[-2]:
            sites[-1] = sites[-2]
        lim.append(None if last and c.chance(0.6) else lim[-1] * c.pick([4, 5, 10]))
    return list(zip(seats, sites, lim))


def tier_copy(c: Ctx, seats, sites, lim) -> tuple[str, list]:
    kw = {"s": seats or 0, "S": num_word(seats) if seats else "", "sites": sites or 0, "xl": lim or 0}
    if sites is None or sites > 1:
        t = c.pick(TIER_SITES[2:] if sites is None else TIER_SITES)
    elif seats == 1:
        t = c.pick([x for x in TIER_SOLO if "{xl}" not in x] if lim is None else TIER_SOLO)
    elif seats is None:
        t = c.pick(TIER_BIG if lim is None else [x for x in TIER_BIG if not x.startswith("No limits")])
    else:
        t = c.pick([x for x in TIER_TEAM if "{xl}" not in x] if lim is None else TIER_TEAM)
    blurb = c.f(t, **kw)
    pts = []
    if c.chance(0.75):
        pts.append(f"Up to {lim} {c.x.pl}" if lim else f"Unlimited {c.x.pl}")
    if c.chance(0.6):
        pts.append(c.pick(["One login", "Single user", "1 member"]) if seats == 1 else
                   (c.pick([f"{seats} members", f"Up to {seats} people", f"{seats} logins"]) if seats else
                    c.pick(["Unlimited members", "No seat limit"])))
    if sites is None:
        pts.append("Unlimited sites")
    elif sites > 1:
        pts.append(f"{sites} sites")
    for p in c.rng.some(TIER_GENERIC, c.pick([1, 2, 2, 3])):
        pts.append(c.f(p, fmt=c.pick(FMT), support=c.pick(["Priority email", "Phone", "Email", "Same-day", "Weekend"]),
                       scope=c.pick(["for everyone", "for admins", "on every account"]), period=c.pick(["week", "month", "season"])))
    return blurb, list(dict.fromkeys(pts))
PERIODS = ["per month", "/month", "a month, billed yearly", "per user per month", "per {who1} per month", "a year", "/mo"]


def m_pricing(c: Ctx):
    r = c.rng.random()
    concept = None
    if r < 0.25:
        concept = L.CONCEPTS[4]
        htext = c.f(c.pick(concept["head"]))
        h2 = S(htext, "h2")
        eb = eyebrow(c, h2, htext, concept=concept, p_r1=0.45, allow_k1=False)
    else:
        h2, eb, concept = section_heading(c, ("Pricing",), concept_ok=False,
                                          default_heads=["Plans for every size of {group}", "Pay for what you use",
                                                         "One price per month, no extras", "What it costs"])
    kids = [E("eyebrow", s=eb)] if eb else []
    kids.append(E("h", s=h2, level=2))
    n = c.pick([2, 3, 3, 4])
    if c.want(0.18):
        kids.append(E("sub", s=S(count_line(c, n), "subtitle").narrate("R16", None)))
    elif c.chance(0.1):
        kids.append(E("sub", s=count_keep(c, n, "pricing")))
    elif concept and c.want(0.3):
        kids.append(E("sub", s=S(c.f(c.pick(concept["echo"])), "subtitle").echo(h2, "R3", conf="judgement")))
    elif c.chance(0.5):
        kids.append(E("sub", s=S(c.f(c.pick(L.CONCEPTS[4]["keep"])), "subtitle")))
    names = c.rng.some(L.PLANS, n)
    period = c.f(c.pick(PERIODS))
    popular = c.rng.randrange(n) if c.chance(0.5) else None
    pop_text = c.f(c.pick(L.POPULAR))
    ladder = tier_ladder(c, n)
    price_now = c.rng.randrange(3, 12)
    cards = []
    for i, name in enumerate(names):
        k = []
        if i == popular:
            # one pill on one plan card: the recommendation is the fact (LABELLING.md §8, K2, judgement)
            k.append(E("badge", s=S(pop_text, "badge").keep("K2", conf="judgement")))
        tier_h = S(name, "h3")
        k.append(E("h", s=tier_h, level=3))
        blurb, pts = tier_copy(c, *ladder[i])
        if c.want(0.12):
            k.append(E("sub", s=S(c.pick(TIER_ECHO[name]), "subtitle").echo(tier_h, "R3", conf="judgement"), card=True))
        else:
            k.append(E("sub", s=S(blurb, "subtitle"), card=True))
        price_now += c.rng.randrange(4, 15) if i else 0
        price = "£0" if (i == 0 and name == "Free") else f"£{price_now}"
        k.append(E("price", amount=S(price, "stat"), period=S(period, "caption").keep("K13")))
        items = [S(p, "list_item") for p in pts]
        if c.want(0.1):
            items.insert(0, S(c.pick([f"The {name} plan", f"{name} features", f"Everything {name} offers"]), "list_item").echo(tier_h, "R4"))
        k.append(E("ul", items=items, mark="check"))
        k.append(E("button", s=S(f"Choose {name}", "button"), primary=i == popular))
        cards.append(E("card", *k))
    out = [E("head", *kids), E("grid", *cards)]
    if c.notice is not None and c.want(0.5):
        out.append(repeat_notice(c))
    elif c.notice is None and c.chance(0.4):
        out.append(notice(c))
    return E("section", *out)


FAQ_POOL_MORE = [
    ("How many {x} can I add?", "As many as you like on paid plans; the free plan stops at {n}."),
    ("Can {who} outside my {group} see my {x}?", "Only if you send them a read-only link; links expire after {n} days."),
    ("Is there a discount for {groups}?", "Yes: {n}% off for {groups} with more than {m} members."),
    ("Can I print {a_x1}?", "Yes; each {x1} prints on one A4 page with its last {n} {y}."),
    ("How do reminders work?", "A reminder goes out {n} days before each {y1}, by email or push, never both."),
    ("What if I lose signal?", "Keep going; {y} are saved on the phone and sync within a minute of reconnecting."),
    ("Who can edit {a_x1}?", "Owners and {roles}; everyone else sees it read-only."),
    ("Can I move from a spreadsheet?", "Yes; paste or upload a CSV and we match up to {m} columns."),
    ("Is my card stored?", "No; payments go through Stripe and we only keep the last four digits."),
]
FAQ_POOL = [
    ("Can I cancel?", "Yes. Cancel from the billing page and keep access until the end of the month."),
    ("How long are {x} kept?", "That depends on the plan: the free one keeps {n} months, paid plans keep all of them."),
    ("Can I see everything at once?", "All of it, on one page, if you pick the list view; it is slower past {m} {x}."),
    ("Can I import my {x}?", "Yes, from CSV or from {n} other apps; we keep your column names."),
    ("Is there a free plan?", "Yes, for up to {n} {x}, with no time limit."),
    ("Where is my data stored?", "In data centres in London and Dublin."),
    ("Does it work offline?", "Yes; changes you make offline sync when you reconnect."),
    ("Can my {group} share one account?", "Each person gets their own login; the {plan} plan has no seat limit."),
    ("Do you offer discounts for charities?", "Registered charities get {n}% off any paid plan."),
    ("Does it work on Android?", "Yes, on Android 11 and later, and in any modern browser."),
    ("Can I export everything?", "Yes, as CSV or PDF, at any time, including after you cancel."),
    ("What happens to deleted {x}?", "They stay in the bin for {n} days, then they are gone for good."),
]


def m_faq(c: Ctx):
    h2, eb, concept = section_heading(c, ("FAQ",), concept_ok=False,
                                      default_heads=["Before you sign up", "Things people ask us", "The small print, in plain English",
                                                     "What {who} ask us"])
    kids = [E("eyebrow", s=eb)] if eb else []
    kids.append(E("h", s=h2, level=2))
    if c.chance(0.3):
        kids.append(E("sub", s=S(c.f(c.pick(["Can’t find an answer? Email us; we reply within {n} hours.",
                                               "Answers are checked every quarter.", "More in the help centre, updated weekly."])), "subtitle")))
    items = [(S(c.f(q), "button"), S(c.f(a, m=c.pick([12, 20, 25, 40, 50])), "body"))
             for q, a in c.rng.some(FAQ_POOL + FAQ_POOL_MORE, c.pick([3, 4, 5]))]
    out = [E("head", *kids), E("faq", items=items)]
    if c.want(0.2):
        out.append(E("p", s=S(c.f(c.pick(L.CLOSE_FAQ)), "body").echo(None, "R17", conf="judgement")))
    return E("section", *out)


QUOTES = ["We moved every {x1} over in an afternoon and haven’t looked back.", "The reminders alone save me an hour a week.",
          "Our {group} picked it up without any training.", "It replaced three spreadsheets and a whiteboard.",
          "Support answered on a Sunday, which I did not expect.", "I check our {x} from my phone now instead of the office computer.",
          "{n} of us share it and nobody double-books any more.", "I finally know what is happening with every {x1}.",
          "That reminder email has saved me twice this year.", "We all use it now, even the ones who hate apps.",
          "That’s the first app our {group} has kept using past a month.", "All {n} of our {roles} were set up in a morning."]


def m_quotes(c: Ctx):
    r = c.rng.random()
    concept = None
    if r < 0.35:
        concept = L.CONCEPTS[5]
        htext = c.f(c.pick(concept["head"]))
        h2 = S(htext, "h2")
        eb = eyebrow(c, h2, htext, concept=concept, p_r1=0.45, allow_k1=False)
    else:
        h2, eb, concept = section_heading(c, ("Testimonials",), concept_ok=False,
                                          default_heads=["What {p} users tell us", "In their words", "From people who use it daily"])
    kids = [E("eyebrow", s=eb)] if eb else []
    kids.append(E("h", s=h2, level=2))
    if concept and c.want(0.35):
        kids.append(E("sub", s=S(c.f(c.pick(concept["echo"])), "subtitle").echo(h2, "R3", conf="judgement")))
    elif concept and c.chance(0.5):
        kids.append(E("sub", s=S(c.f(c.pick(concept["keep"])), "subtitle")))
    n = c.pick([2, 3])
    qs = []
    for q in c.rng.some(QUOTES, n):
        who = f"{L.person(c.rng)}, {c.pick([c.who[0], c.role, c.group + ' secretary', 'owner', 'volunteer'])}"
        qs.append(E("quote", s=S(c.f(q), "body"), who=S(who, "caption")))
    out = [E("head", *kids), E("grid", *qs)]
    if c.want(0.2):
        if concept is not None:
            out.append(E("p", s=S(closing_head(c, h2.text), "body").echo(h2, "R17")))
        else:
            out.append(E("p", s=S(closing_spread(c), "body").echo(None, "R17", conf="judgement")))
    return E("section", *out)


def m_cta(c: Ctx):
    htext = c.f(c.pick(["Ready when you are", "See it for yourself", "Your {x} will thank you", "Give it a week",
                        "Try it on your next {x1}", "Bring your {group} along"]))
    h2 = S(htext, "h2")
    if c.hero_h is not None and c.hero_h.cls == "KEEP" and c.want(0.2):
        # the closing section repeats the page's main heading (R9)
        h2 = S(c.hero_h.text, "h2").echo(c.hero_h, "R9", conf="clear")
    kids = [E("h", s=h2, level=2)]
    if c.chance(0.6):
        kids.append(E("sub", s=S(product_fact(c), "subtitle")))
    out = [E("head", *kids)]
    if c.cta is not None and c.want(0.45):
        tag = "link" if c.chance(0.4) else "button"
        b = S(c.cta.text, tag).echo(c.cta, "R14")
        out.append(E("actions", E(tag, s=b)))
    else:
        rest = [x for x in c.dom["cta"] + T.GENERIC_CTA if x not in c.used]
        out.append(E("actions", E("button", s=S(c.fresh(rest), "button"))))
    if c.notice is not None and c.want(0.45):
        out.append(repeat_notice(c))
    elif c.notice is None and c.chance(0.35):
        out.append(notice(c))
    return E("section", *out, tone="cta")


def m_footer(c: Ctx, nav_items):
    cols = []
    seen = set()
    for heads, links in c.rng.some(L.FOOTER_POOL, c.pick([2, 3, 4])):
        head = c.f(c.pick([h for h in heads if h not in nav_items] or heads))
        ss = []
        for x in c.rng.some(links, c.pick([2, 3, 4])):
            x = c.f(x)
            if x in seen or x == head:
                continue
            seen.add(x)
            s = S(x, "link")
            if x in nav_items:
                s.keep("K5")
            ss.append(s)
        if ss:
            cols.append((S(head, "h4"), ss))
    if c.cta is not None and cols:
        if c.want(0.35):
            cols[0][1].append(S(c.cta.text, "link").echo(c.cta, "R14"))
        elif c.chance(0.2):
            cols[0][1].append(S(c.pick(c.dom["cta"][1:]), "link"))
    legal = S(L.legal_line(c.rng, c.product), "caption")
    out = [E("footer", cols=cols, legal=legal)]
    if c.fw != "swift" and c.want(0.1):
        out.append(E("status", s=S(static_status(c), "body").decor("R11")))
    if c.want(0.06):
        out.append(E("meta", items=[S(x, "caption").decor("R2") for x in L.meta_strip(c.rng)]))
    elif c.chance(0.12):
        out.append(E("meta", items=[S(x, "caption").keep("K1" if not x.startswith("v") else None) for x in L.keep_meta(c.rng)]))
    return out


NAV_EXTRA = ["How it works", "Products", "Services", "Our team", "Contact", "FAQ", "Overview", "Reviews", "Plans",
             "Get started", "What it does", "Why us", "Testimonials", "Our story"]


def marketing(c: Ctx) -> Page:
    rng = c.rng
    swift = c.fw == "swift"
    pool = ["features", "steps", "pricing", "faq", "quotes", "cta", "spotlight", "facts"]
    if swift:
        pool = ["features", "steps", "cta", "spotlight", "facts"]
    n = rng.wpick([(1, 40), (2, 28), (3, 22), (4, 10)])
    body = []
    nav_items = []
    if n == 1:
        kinds = [c.pick(["hero"] * 3 + pool)]
    else:
        kinds = ["hero"] + rng.some([p for p in pool if p != "cta"], n - 1)
        if n >= 3 and rng.chance(0.5):
            kinds[-1] = "cta"
    if n >= 2 and not swift and rng.chance(0.4):
        nav_items = rng.some(T.NAV_ITEMS + NAV_EXTRA, c.pick([3, 4, 5]))
        body.append(E("nav", items=[S(c.product, "link")] + [S(x, "nav") for x in nav_items]))
        c.used.update(nav_items)
        c.nav_items = set(nav_items)
    for k in kinds:
        body.append({"hero": m_hero, "features": m_features, "steps": m_steps, "pricing": m_pricing, "faq": m_faq,
                     "quotes": m_quotes, "cta": m_cta, "spotlight": m_spotlight, "facts": m_facts}[k](c))
    if n >= 2 and not swift and rng.chance(0.35):
        body.extend(m_footer(c, nav_items))
    ebs = [e["s"] for top in body for e in top.walk() if e.kind == "eyebrow"]
    if len(ebs) >= 3 and len(ebs) >= len(kinds) - 1:
        c.r19.append(ebs)
    name = {1: kinds[0].capitalize() + "Section"}.get(n, c.pick(["LandingPage", "HomePage", "Home", "Index"]))
    if kinds[0] == "hero" and n == 1:
        name = c.pick(["Hero", "HeroBanner", "Intro"])
    return Page(name, "marketing", body, product=c.product)


# ---------------------------------------------------------------------------------------------------------------
# tables, stats, charts and alerts (shared by app pages and dashboards)

EXTRA_COLS = {
    "Added": lambda c: L.date(c.rng),
    "Owner": lambda c: L.person(c.rng),
    "Updated": lambda c: c.pick(["2 hours ago", "Yesterday", "3 days ago", "Just now", "Last week", f"{c.rng.randrange(2, 50)} min ago"]),
    "Ref": lambda c: f"#{c.rng.randrange(100, 9999)}",
    "Location": lambda c: c.pick(L.PLACES),
    "Notes": lambda c: c.pick(["Check again next week", "Needs a photo", "Waiting on a reply", "None", "Moved from the old list",
                               "Ask " + c.pick(L.FIRST)]),
}


def fresh_value(c: Ctx, v: str) -> str:
    """The same kind of value with other figures: amounts, counts, dates and times."""
    if re.fullmatch(r"[£$€]?[\d,]+(\.\d+)?( kg| km| m| g| ml| L|%| min| h)?", v):
        return jitter(c.rng, v)
    if re.fullmatch(r"\d{1,2}:\d{2}", v):
        return L.clock(c.rng)
    m = re.fullmatch(r"(\d{1,2}) ([A-Z][a-z]+)", v)
    if m and m.group(2) in L.MONTHS:
        return f"{c.rng.randrange(1, 29)} {m.group(2)}"
    return v


def data_table(c: Ctx, n_rows=None, n_cols=None):
    cols, pools = c.dom["table"]
    idx = list(range(len(cols)))
    k = n_cols or c.rng.wpick([(2, 2), (3, 4), (4, 5), (5, 3), (6, 2), (7, 1)])
    keep = [0] + sorted(c.rng.some(idx[1:], min(k - 1, len(idx) - 1)))
    extras = c.rng.some(list(EXTRA_COLS), max(0, k - len(keep)))
    headers_t = [cols[j] for j in keep] + extras
    n_rows = min(n_rows or c.pick([2, 3, 4, 5, 6]), len(pools[0]))
    headers = [S(x, "label") for x in headers_t]
    rows = []
    seen = [set() for _ in headers_t]
    status_j = headers_t.index("Status") if "Status" in headers_t and c.chance(0.7) else None
    status_dot = c.chance(0.3)
    keys = c.rng.some(pools[0], n_rows)
    for r in range(n_rows):
        row = []
        for j, h in enumerate(headers_t):
            if j < len(keep):
                v = keys[r] if j == 0 else c.pick(pools[keep[j]])
                if j and c.chance(0.6):
                    v = fresh_value(c, v)
            else:
                v = EXTRA_COLS[h](c)
            if j == status_j:
                if c.fw == "html":
                    s = S(v, "cell")
                    if v in seen[j]:
                        s.keep("K5")
                    seen[j].add(v)
                    row.append(s)
                else:
                    glyph = "● " if status_dot else ""
                    row.append(E("badge", s=S(glyph + v, "badge").keep("K2", aside=["R11g"] if glyph else None),
                                 bound=c.prop("status") if c.fw == "tsx" else "isActive"))
                continue
            s = S(v, "cell")
            if v in seen[j]:
                s.keep("K5")
            seen[j].add(v)
            row.append(s)
        rows.append(row)
    return E("table", headers=headers, rows=rows)


def stat_caption(c: Ctx, lab, value: str, val=None):
    """A caption under a figure: a restatement of its label (R5), of its label and figure together (R5, pointing at
    the figure), a delta or a denominator that adds a fact, or a short plain line."""
    timed = bool(words(lab.text) & {"this", "today", "now", "week", "month", "year", "season", "day"})
    ll = lower_first(lab.text)
    if val is not None and c.want(0.07):
        t = c.pick([f"{value} {ll}", f"{lab.text}: {value}", f"{value}, {ll}", f"{lab.text} stands at {value}",
                    f"{value} {ll}, as shown", f"{lab.text}; {value} in all"])
        return S(t, "caption").echo(val, "R5", conf="clear")
    num = re.fullmatch(r"[£]?([\d,]+)", value)
    if num and c.chance(0.06):
        # a denominator or a target larger than the figure (KEEP)
        v = int(num.group(1).replace(",", ""))
        m = f"{v + c.rng.randrange(1, v + 10):,}"
        m = ("£" + m) if value.startswith("£") else m
        return S(c.pick([f"Out of {m} in total", f"Of {m} in total", f"{m} in total last year", f"Target: {m} in total"]), "caption")
    if c.chance(0.06):
        return S(short_keep(c, "stat"), "caption")
    if c.want(0.1):
        if c.chance(0.5):
            t = c.pick(["{L} right now", "{L}, as of today", "{L} at a glance", "{L} in total", "{L}, all told",
                        "The {l} figure", "{L} so far"] if not timed else ["{L}, at a glance", "{L}, all told", "{L} in total"])
            if lab.text.endswith(" in"):
                t = t.replace(" in total", ", all told")
            return S(t.format(L=lab.text, l=lower_first(lab.text)), "caption").echo(lab, "R5", conf="judgement")
        t = (f"{c.pick(['The', 'That', 'This'])} {c.pick(['figure', 'number', 'count', 'total'])}"
             f"{c.pick([' above', ', at a glance', ', in total', ' shown here', ' for you'])}.")
        return S(t, "caption").echo(lab, "R5", conf="judgement")
    if c.chance(0.4):
        if c.chance(0.4) and not timed:
            return S(f"{lab.text} last {c.pick(['week', 'month', 'season', 'year'])}: {jitter(c.rng, value)}", "caption")
        if c.chance(0.4) and timed:
            return S(f"{c.pick(['Target', 'Last time', 'Average', 'Best so far'])}: {jitter(c.rng, value)}", "caption")
        return S(c.pick([f"+{c.rng.randrange(2, 40)}% on last {c.pick(['week', 'month', 'year'])}",
                         f"−{c.rng.randrange(1, 20)}% on last {c.pick(['week', 'month', 'season'])}",
                         f"Up {c.rng.randrange(2, 30)} since {c.pick(L.DAYS)}", f"{c.pick(['Same as', 'Level with', 'Unchanged from'])} last {c.pick(['week', 'month', 'year'])}",
                         f"Down {c.rng.randrange(1, 12)} since {c.pick(L.MONTHS)}", f"{c.rng.randrange(2, 9)} more than {c.pick(L.MONTHS)}"]), "caption")
    return None


def stats_row(c: Ctx, n: int):
    out, labels, values = [], [], []
    for label, value in c.rng.some(c.dom["stats"], n):
        v = jitter(c.rng, value)
        lab = S(label, "label")
        val = S(v, "stat")
        out.append(E("stat", value=val, label=lab, note=stat_caption(c, lab, v, val)))
        labels.append(lab)
        values.append(val)
    return E("stats", *out), labels, values


SERIES = ["This year", "Last year", "Target", "Average", "Planned", "Actual", "Forecast"]


def chart_panel(c: Ctx, h1=None):
    """A panel with a heading, a chart (no text) and a legend whose entries name the series (K2)."""
    noun = c.noun()
    t = c.pick([f"{noun.Pl} by week", f"{noun.Pl} over time", f"{noun.Pl} this {c.pick(['month', 'season', 'year'])}",
                f"{noun.Pl} by {c.pick(['month', 'day', 'site', 'person'])}", f"Usage over {c.pick(['time', 'the year', 'the season'])}",
                f"{noun.Sg} trend"])
    title = S(t, "h2")
    kids = [E("h", s=title, level=2)]
    if c.chance(0.35):
        kids.append(E("sub", s=S(c.pick([f"Totals per {c.pick(['week', 'month'])}, excluding drafts.",
                                         f"Compared with the same weeks in {c.pick([2024, 2025])}.",
                                         f"{noun.Pl} still in draft are not counted.", f"Dotted bars are forecast {noun.pl}.",
                                         f"Figures for {c.group} members only.", f"{noun.Pl} are counted on the day they close."]), "subtitle")))
    kids.append(E("chart", series=noun.pl))
    if c.chance(0.7):
        legend = c.rng.some(SERIES, c.pick([2, 3])) if c.chance(0.5) else c.rng.some(item_names(c), c.pick([2, 3]))
        kids.append(E("legend", items=[S(x, "list_item").keep("K2") for x in legend]))
    return E("section", *kids)


ALERT_TITLES = ["Couldn’t sync your {x}", "{X} didn’t load", "Couldn’t save this {x1}", "Upload failed",
                "Couldn’t reach the server", "Payment failed", "{Y} are out of date"]
ALERT_ECHO_STEM = {"Couldn’t sync your {x}": "Your {x} couldn’t be synced.", "{X} didn’t load": "We couldn’t load your {x}.",
                   "Couldn’t save this {x1}": "This {x1} could not be saved.", "Upload failed": "Your upload failed.",
                   "Couldn’t reach the server": "The server could not be reached.", "Payment failed": "Your payment failed.",
                   "{Y} are out of date": "Your {y} are not up to date."}
ALERT_ECHO_ALT = ["Something went wrong.", "That didn’t work.", "An error occurred.", "Sorry, there was a problem.",
                  "It didn’t go through."]
ALERT_KEEP = ["The server didn’t answer; changes stay on this device and are retried every {n} minutes.",
              "{person} is editing this {x1}; try again after {clock}.", "Check your connection, then pull down to retry.",
              "Your card was declined; update it on the billing page before {date}.",
              "Files over {n} MB are not accepted; try a smaller photo.",
              "Your {x} will sync again once you are back online; nothing is lost."]


def error_alert(c: Ctx):
    t = c.pick(ALERT_TITLES)
    title = S(c.f(t), "error")
    if c.want(0.4):
        if c.chance(0.5):
            body = S(c.f(ALERT_ECHO_STEM[t]), "error").echo(title, "R4", conf="clear")
        else:
            body = S(c.pick(ALERT_ECHO_ALT), "error").echo(title, "R4", conf="judgement")
    else:
        body = S(c.f(c.pick(ALERT_KEEP)), "error").keep("K8")
    return E("alert", title=title, s=body, tone="error")


# ---------------------------------------------------------------------------------------------------------------
# app pages


# List titles that state a window or a limit, with subtitles that repeat it (R3) or add a fact (KEEP)
LIST_FACT_HEADS = [
    ("{X} due in the next {k} days", ["Each {x1} due within {k} days.", "Due within {k} days: these {x}, and only these.",
                                      "The next {k} days, {x} only; nothing later."],
     ["Overdue {x} are listed separately.", "Sorted by due date, soonest first.", "Only you can see this list."]),
    ("{X} from the last {k} days", ["Only the last {k} days of {x}.", "{X} added in the past {k} days, no older.",
                                    "The past {k} days; every {x1} from them."],
     ["Older {x} are in the archive.", "Drafts are not shown.", "Shared with your {group}."]),
    ("{X} over {k} days old", ["Every {x1} here is more than {k} days old.", "{K} days or older, each one.",
                               "Nothing younger than {k} days."],
     ["They move to the archive at {m} months.", "Your {role} reviews this list monthly.", "Hidden from everyone else."]),
]


def list_heading(c: Ctx, noun: Noun):
    """A list page title: the noun, or a subset of it."""
    if c.chance(0.6):
        return noun.Pl
    return c.pick([f"Overdue {noun.pl}", f"{noun.Pl} this week", f"Recent {noun.pl}", f"Shared {noun.pl}",
                   f"Archived {noun.pl}", f"{noun.Pl} in {c.place}", f"Upcoming {noun.pl}", f"Your {noun.pl}"])


def page_sub(c: Ctx, h, noun: Noun, allow_r16=True):
    """The subtitle under a list page's heading: a restatement (R3), layout narration (R16), a partial, or a fact
    about the noun (with or without its stem)."""
    if c.want(0.28):
        return S(restate(c, noun), "subtitle").echo(h, "R3", conf="clear")
    if allow_r16 and c.want(0.1):
        return S(layout_line(c, noun), "subtitle").narrate("R16", None)
    if c.chance(0.12):
        return S(restate(c, noun) + " " + about(c, noun), "subtitle").keep(partial=True)
    if c.chance(0.22):
        return S(short_keep(c, "list"), "subtitle")
    if c.chance(0.6):
        return S(about(c, noun, stem=c.chance(0.65)), "subtitle")
    return None


def title_syns(c: Ctx, noun: Noun) -> list:
    a = c.alt(noun)[1]
    return [cap(a), f"Your {noun.pl}", f"All {noun.pl}", f"Every {c.alt(noun)[0]}", f"All your {a}"]


def app(c: Ctx) -> Page:
    rng = c.rng
    swift = c.fw == "swift"
    noun = c.noun()
    htext = list_heading(c, noun)
    list_fact = None
    if c.chance(0.2):
        lf = c.pick(LIST_FACT_HEADS)
        kw = {"k": c.pick([3, 7, 14, 30, 60, 90]), "m": c.pick([3, 6, 12])}
        kw["K"] = num_word(kw["k"])
        htext = cap(c.f(lf[0], **kw))
        list_fact = (lf, kw)
    body = []
    crumb_last = None
    if not swift and c.chance(0.3):
        crumbs = [S(c.pick(["Home", c.product, "Workspace", c.group.capitalize()]), "breadcrumb")]
        crumb_last = S(htext, "breadcrumb")
        crumbs.append(crumb_last)
        body.append(E("crumbs", items=crumbs, aria=ctx("Breadcrumb", "aria") if c.chance(0.5) else None))
    h1 = S(htext, "h1")
    if crumb_last is not None:
        h1.keep("K11")
    nav_title = None
    top_h = None
    if swift and c.chance(0.6):
        nav_title = S(htext, "h1")
        if c.want(0.45):
            # the in-page title repeats the navigation title (R9)
            h1 = S(htext if (c.chance(0.6) or htext != noun.Pl) else title_syns(c, noun)[0], "h1").echo(nav_title, "R9", conf="clear")
            if h1.text != htext:
                h1.conf = "judgement"
        else:
            h1 = S(c.pick([f"Recent {noun.pl}", f"Shared {noun.pl}", f"{noun.Pl} this week", item_names(c)[0]]), "h2")
    elif not swift and c.chance(0.25):
        # an app bar with its own title over the page title
        if c.want(0.5):
            top_h = S(htext, "h1")
            h1 = S(htext if (c.chance(0.6) or htext != noun.Pl) else title_syns(c, noun)[0], "h1").echo(top_h, "R9", conf="clear")
            if h1.text != htext:
                h1.conf = "judgement"
            if crumb_last is not None:
                top_h.keep("K11")
        elif htext != noun.Pl and c.chance(0.6):
            top_h = S(noun.Pl, "h1")  # the whole list over a subset of it: both KEEP; the subset is the heading plus a qualifier
            h1.partial, h1.conf = True, "judgement"
        else:
            top_h = S(c.product, "h1")
        body.append(E("section", E("h", s=top_h, level=1)))
    kids = []
    plain = htext == noun.Pl
    if h1.cls == "KEEP":
        eb = None
        if plain and c.want(0.25):
            eb = S(c.pick(title_syns(c, noun)), "eyebrow").echo(h1, "R1", conf="judgement")
        elif c.chance(0.15):
            eb = S(c.pick([c.product, c.fresh(c.dom["cats"]), f"{c.group.capitalize()} {c.pick(['records', 'admin'])}"]),
                   "eyebrow").keep("K1")
            if eb.text.lower() == htext.lower():
                eb = None
        if eb:
            kids.append(E("eyebrow", s=eb))
    kids.append(E("h", s=h1, level=1 if h1.role == "h1" else 2))
    if h1.cls == "ECHO":
        sub_target = h1.target
    elif nav_title is not None:
        sub_target = nav_title
    else:
        sub_target = h1
    if list_fact is not None and c.want(0.45):
        sub = S(cap(c.f(c.pick(list_fact[0][1]), **list_fact[1])), "subtitle").echo(sub_target, "R3")
    elif list_fact is not None and c.chance(0.6):
        sub = S(cap(c.f(c.pick(list_fact[0][2]), **list_fact[1])), "subtitle")
    else:
        sub = page_sub(c, sub_target, noun)
    if sub is not None:
        kids.append(E("sub", s=sub))
    head = E("head", *kids)
    body.append(head)
    # a third copy of the same name: a stack (equiv)
    flagged = [x for x in [k["s"] for k in kids] if x.cls == "ECHO"]
    if len(flagged) >= 2:
        top = flagged[0].target
        restack(top, flagged)
    # actions
    acts = []
    btn = S(c.pick([f"New {noun.sg}", f"Add {noun.sg}", f"Add {article(noun.sg)}", f"Log {article(noun.sg)}"]), "button")
    if shares_stem(btn.text, htext):
        btn.keep("K9")
    tip = None
    if c.want(0.1):
        if c.chance(0.5):
            tip = S(btn.text, "tooltip").echo(btn, "R5", conf="clear")
        else:
            tip = S(c.pick([f"Add {article(c.alt(noun)[0])}", f"Create another {c.alt(noun)[0]}"]), "tooltip").echo(btn, "R5", conf="judgement")
    elif c.chance(0.12):
        # the button's label plus a qualifier (LABELLING.md §8: KEEP, partial)
        tip = S(c.pick([f"{btn.text} (N)", f"{btn.text} to {c.place}", f"{btn.text} from a template"]), "tooltip").keep(conf="judgement", partial=True)
    acts.append(E("button", s=btn, tip=tip))
    if c.chance(0.5):
        ia = c.pick(L.ICON_ACTIONS)
        tip_t = c.f(ia[0])
        acts.append(E("iconbtn", tip=S(tip_t, "tooltip").keep("K12"), aria=ctx(ia[1], "aria") if c.chance(0.7) else None,
                      icon=ia[2], html_icon=ia[3], sf=ia[4]))
    if c.want(0.08):
        acts.insert(0, E("badge", s=S(static_status(c), "badge").decor("R11")))
    body.append(E("actions", *acts))
    if c.chance(0.12):
        body.append(error_alert(c))
    has_tabs = c.chance(0.35)
    if has_tabs:
        tabs = rng.some(["All", "Active", "Archived", "Shared", "Mine", "Overdue", "Drafts", "This week", "Starred"], c.pick([2, 3, 4]))
        tab_slots = [S(t, "tab") for t in tabs]
        if htext not in tabs and htext != noun.Pl and c.chance(0.5):
            tab_slots.insert(rng.randrange(len(tab_slots) + 1), S(htext, "tab").keep("K11"))
        body.append(E("tabs", items=tab_slots))
    inner = []
    if nav_title is None and h1.role == "h1" and h1.cls == "KEEP" and c.want(0.1):
        r9 = S(c.pick([h1.text, f"All {noun.pl}", cap(c.alt(noun)[1])] if plain else [h1.text]), "h2").echo(h1, "R9", conf="clear")
        if r9.text != h1.text:
            r9.conf = "judgement"
        inner.append(E("h", s=r9, level=2))
    elif c.chance(0.45):
        h2 = S(c.pick([f"Needs attention", f"{noun.Pl} this week" if htext != f"{noun.Pl} this week" else "Shared with you",
                       "Shared with you", "Archived", f"Added in {c.pick(L.MONTHS)}"]), "h2")
        if plain and h2.text.startswith(noun.Pl + " "):
            h2.keep(partial=True, conf="judgement")
        inner.append(E("h", s=h2, level=2))
    table = data_table(c)
    if c.chance(0.35):
        et, eb_ = empty_copy(c, noun)
        if c.want(0.2) and h1.role == "h1" and h1.cls == "KEEP":
            es = S(h1.text, "empty_title").echo(h1, "R9", conf="clear")
            ebody = S(c.f(c.pick(["You haven’t added any {a} yet. " + "Add one from the button above, or import a CSV.",
                                  "Nothing yet: {a} you add appear here, newest first."]), a=noun.pl), "empty_body").keep("K8")
        else:
            es = S(et, "empty_title")
            if c.want(0.3):
                # a conditional empty state whose body only repeats its title (R4)
                t, conf = empty_echo(c, noun, et)
                ebody = S(t, "empty_body").echo(es, "R4", conf=conf)
            elif c.chance(0.12):
                ebody = S(empty_echo(c, noun, et)[0] + " " + c.f(c.pick(EMPTY_KEEP), a=noun.pl, A=noun.Pl), "empty_body").keep("K8", partial=True)
            else:
                ebody = S(eb_, "empty_body").keep("K8")
        cond = E("cond", a=[E("empty", title=es, body=ebody, action=None)], b=[table],
                 expr=f"{c.prop('count')} === 0", swift="count == 0")
        inner.append(cond if c.fw != "html" else table)
    else:
        inner.append(table)
    if c.fw == "tsx" and c.chance(0.25):
        pre, post = c.pick([("Signed in as ", "."), ("Viewing as ", ""), ("Shared by ", ".")])
        inner.append(E("dyn", s=S(f"{pre}VAR email{post}", "caption"), pre=pre, expr="user.email", post=post, look="note"))
        c.prop("user")
    elif c.fw != "html" and c.chance(0.15):
        inner.append(E("dyn", s=dyn("VAR updated at", "caption"), expr="updatedAt", look="note"))
    if c.chance(0.12):
        inner.append(E("note", s=S(c.f(c.pick(["Hold Shift to select a range of rows.", "Press / to jump to search.",
                                               "Drag rows to change the order; the new order is saved for everyone.",
                                               "Swipe left on {a_x1} to archive it.", "Long-press a row to select several.",
                                               "Tap {a_x1} twice to mark it done.", "Click a date to block it out for everyone.",
                                               "Press Enter to save and start the next {x1}."])),
                                   "caption").keep("K4", ct=True)))
    elif c.want(0.08):
        inner.append(E("note", s=S(scroll_line(c, noun) if c.chance(0.8) else "Use the tabs above to switch views." if has_tabs
                                   else scroll_line(c, noun), "caption").narrate("R6", None, ct=True, conf="judgement")))
    if c.chance(0.25):
        inner.append(detail_card(c))
    if c.chance(0.16):
        inner.append(figure_grid(c, h1 if h1.role == "h1" else None, noun))
    elif h1.role == "h1" and h1.cls == "KEEP" and c.chance(0.25):
        if c.want(0.4):
            ct = S(c.pick([cap(c.alt(noun)[1]), f"Your {c.alt(noun)[1]}", f"All {c.alt(noun)[1]}"] if plain else [h1.text]),
                   "h3").echo(h1, "R9", conf="judgement" if plain else "clear")
        else:
            ct = S(c.pick([f"{noun.Pl} added this week", f"{noun.Sg} of the month", f"Oldest {noun.sg}", f"{noun.Pl} with notes",
                           f"Busiest {noun.sg}"]), "h3")
            if plain and ct.text.startswith(noun.Pl + " "):
                ct.keep(partial=True, conf="judgement")
        inner.append(E("card", E("h", s=ct, level=3), E("p", s=S(c.f(c.pick(["{person} added {n} {y} on {day}.",
                                                                            "{n} {y} since {date}.", "Last checked {day} at {clock}.",
                                                                            "Shared with {n} {who}."])), "body")), shadcn=False))
    body.append(E("section", *inner))
    name = c.pick([f"{ident_words(htext)}Page", f"{ident_words(htext)}List", f"{ident_words(htext)}Screen", f"{ident_words(noun.pl)}Index"])
    return Page(name, "app", body, product=c.product, props=c.props, nav_title=nav_title)


def ident_words(s: str) -> str:
    return "".join(w[:1].upper() + w[1:] for w in re.split(r"[^A-Za-z0-9]+", s) if w)


def restack(top, copies):
    """Point every copy at ``top`` and list the other copies in each one's ``equiv`` (LABELLING.md §5)."""
    for x in copies:
        x.target = top
        x.equiv = [o for o in copies if o is not x] if len(copies) > 1 else []


# ---------------------------------------------------------------------------------------------------------------
# settings and forms

# Field labels -> restatements sharing no stem with the label (placeholder R8, helper R5)
FIELD_ALT = {
    "Hive name": ["What you call this colony"], "Apiary postcode": ["Where the bees live"],
    "Queen marking colour": ["The dot on her thorax"], "Plot number": ["Which allotment is yours"],
    "Site name": ["Where your allotment is"], "Last frost date": ["When the cold ends"],
    "Bike make and model": ["Brand and type of cycle"], "Drop-off date": ["When you bring it in"],
    "What needs doing": ["The work required"], "Client name": ["Who you are billing"], "Payment terms": ["How long they have to pay"],
    "VAT number": ["Your tax registration"], "Sail number": ["The digits on your mainsail"], "Class": ["Type of dinghy"],
    "Emergency contact": ["Who we call if something happens"], "Recipe name": ["What the dish is called"],
    "Servings": ["How many it feeds"], "Source link": ["Where you found it"], "Show title": ["What your podcast is called"],
    "Episode number": ["Which instalment this is"], "Author email": ["Where listeners can write"],
    "Experience": ["How long you have been throwing"], "Dietary needs": ["Foods you avoid"], "Locker number": ["Which cupboard is yours"],
    "Climbing name": ["What the wall knows you as"], "Shoe size": ["What fits your feet"], "Belay certificate": ["Proof you can hold a rope"],
    "Album title": ["What to call this collection"], "Date range": ["From when to when"], "People": ["Who is pictured"],
    "Vehicle registration": ["Your number plate"], "Vehicle length": ["How long your car is"], "Lead passenger": ["Who is travelling"],
    "Delivery address": ["Where we bring it"], "Safe place": ["Where to leave it"], "Allergies": ["Foods that make you ill"],
    "Visit date": ["When you are coming"], "Number of visitors": ["How many of you"], "Access needs": ["Anything that helps you get around"],
    "ECF code": ["Your federation rating ID"], "Preferred time control": ["How fast you like to play"], "Board name": ["What opponents see you as"],
    "Monthly income": ["What you earn each month"], "Payday": ["When the money arrives"], "Savings goal": ["What you are putting by for"],
    "Batch code": ["The brew’s reference"], "Original gravity": ["Starting density"], "Fermenter": ["Which vessel it is in"],
    "Group size": ["How many are coming"], "Swimming ability": ["How well you swim"], "Pick-up point": ["Where we collect you"],
    "Language": ["What you want to learn"], "Current level": ["How far you have got"], "Learning goal": ["What you are aiming for"],
    "Registration": ["Number plate"], "Next MOT": ["When the test is due"], "Driver phone": ["Mobile for whoever drives it"],
    "Sample ID": ["Its reference"], "Storage location": ["Where it is kept"], "Hazard class": ["How dangerous it is"],
    "Survey square": ["Which grid square you cover"], "Observers": ["Who is counting"], "Weather": ["Conditions on the day"],
    "Pupil name": ["Who is learning"], "Instrument": ["What they play"], "Current grade": ["Their latest exam level"],
    "Meter serial": ["The number on your meter"], "Tariff": ["What you pay per unit"], "Home size": ["How many bedrooms"],
    "Library card number": ["The digits on your card"], "PIN": ["Your four digits"], "Home branch": ["Where you usually borrow"],
    "Brew method": ["How you make your coffee"], "Bag size": ["How much coffee per delivery"], "Delivery day": ["When it arrives"],
    "Shift name": ["What to call this slot"], "Volunteers needed": ["How many helpers"], "Meeting point": ["Where everyone gathers"],
    "Pet’s name": ["What your animal is called"], "Species and breed": ["What kind of animal"], "Microchip number": ["The code under the skin"],
    "Full name": ["First and last"], "Display name": ["What others see you as"], "Bio": ["A line about yourself"],
    "Website": ["Your site’s URL"], "Email": ["Where we can reach you"], "Phone number": ["Mobile or landline"],
    "Password": ["Something only you know"], "Username": ["Your handle"], "Billing email": ["Where invoices go"],
    "Company name": ["Your business"], "Postcode": ["Your area code"], "Workspace name": ["What your team is called"],
    "Workspace URL": ["Your team’s web address"], "Default language": ["What members see first"], "Name": ["What to call it"],
}
# Labels that are not noun phrases (generic K4 templates do not fit them)
NOT_NP = {"What needs doing", "Experience", "Dietary needs", "Allergies", "Access needs", "Weather", "Servings", "Payday",
          "People", "Observers", "Language", "Class", "Volunteers needed", "Number of visitors", "Group size", "Swimming ability",
          "Species and breed", "Bio", "Tariff", "Instrument"}
REQUIRED_FIELDS = {"Password", "Email", "Username", "Display name", "Name", "Client name", "Recipe name", "Album title", "Show title",
                   "Pupil name", "Lead passenger", "Workspace name", "Workspace URL", "Billing email", "Company name"}
HELPER_ECHO = ["Enter your {l}.", "Your {l}.", "The {l} for this account.", "Add your {l} here.", "Type in your {l}.",
               "Please provide your {l}.", "The {l} goes here.", "Fill in the {l}."]
PLACEHOLDER_ECHO = ["{L}", "Enter {l}", "Enter your {l}", "Your {l}", "Type your {l}", "{L} here", "Add {l}", "{L}…"]
HELPER_KEEP_STEM = ["The {l} appears on {where}.", "Your {l} is never shown publicly.", "The {l} can be changed once a month.",
                    "With no {l}, the {group}’s default applies.", "We check the {l} before saving; it must be unique.",
                    "Only {roles} can see the {l}.", "The {l} is printed on {where}."]
WHERE = ["every invoice", "receipts", "shared links", "the summary sheet", "reminder emails", "the members list"]


def field_from(c: Ctx, label, example, helper, swift):
    c.used.add(("label", label.lower()))
    lab = S(label, "label")
    ll = lower_first(label)
    alts = FIELD_ALT.get(label)
    ph = None
    r = c.rng.random()
    if c.want(0.28):
        if alts and c.chance(0.4):
            ph = S(c.pick(alts), "placeholder").echo(lab, "R8", conf="judgement")
        elif c.chance(0.35) and label not in NOT_NP:
            # "e.g." that introduces no example, only the label again
            ph = S(c.pick(["e.g. your {l}", "e.g. the {l}", "e.g. {l}", "e.g. your {l} here"]).format(
                l=ll + (" address" if ll.endswith("email") else "")), "placeholder").echo(lab, "R8")
        else:
            ph = S(c.pick(PLACEHOLDER_ECHO).format(l=ll, L=label), "placeholder").echo(lab, "R8")
    elif r < 0.18 and example.startswith("e.g. "):
        # the label again, with an example: a partial (LABELLING.md §4 step 5)
        ph = S(f"{label}, {example}", "placeholder").keep(partial=True)
    elif r < 0.26 and label not in NOT_NP:
        # the label plus a qualifier (LABELLING.md §8: KEEP, partial)
        opts = ["{L}, up to {n} characters", "{L} as it appears on your {doc}"] + ([] if ll.startswith("full ") else ["Full {l}, no abbreviations"])
        ph = S(c.f(c.pick(opts), L=label, l=ll, n=c.pick([20, 30, 40, 60, 80]), doc=c.pick(["letter", "card", "account", "booking"])),
               "placeholder").keep(conf="judgement", partial=True)
    elif r < 0.32 and label not in NOT_NP and label not in REQUIRED_FIELDS and not ll.startswith("full "):
        ph = S(c.pick([f"Enter your {ll} or leave it empty", f"Your {ll}, if you have one", f"Enter the {ll} from your {c.pick(['letter', 'card', 'receipt'])}"]),
               "placeholder").keep(partial=True, conf="judgement")
    elif swift or r < 0.75:
        # a bare example is as common as one that says "e.g."
        bare = example.startswith("e.g. ") and c.chance(0.35)
        ph = S(cap(example[5:]) if bare else example, "placeholder")
    hl = None
    if c.want(0.16):
        if alts and c.chance(0.45):
            hl = S(c.pick(alts) + ".", "helper").echo(lab, "R5", conf="judgement")
        else:
            hl = S(c.pick(HELPER_ECHO).format(l=ll), "helper").echo(lab, "R5")
    elif c.chance(0.06):
        hl = S(c.pick(HELPER_ECHO).format(l=ll) + " " + helper, "helper").keep("K4", partial=True)
    elif c.chance(0.6):
        if c.chance(0.2):
            hl = S(short_keep(c, "field"), "helper").keep("K4")
        elif label not in NOT_NP and c.chance(0.4):
            hl = S(c.f(c.pick(HELPER_KEEP_STEM), l=ll, where=c.pick(WHERE)), "helper").keep("K4")
        else:
            hl = S(helper, "helper").keep("K4")
    return E("field", label=lab, ph=ph, helper=hl, type="password" if label == "Password" else "text"), lab


def constraint_field(c: Ctx, swift: bool):
    """A field whose label carries its own limit or format: a helper or placeholder that only repeats it (R5, R8),
    one that adds a fact (K4) or an example."""
    free = [j for j, p in enumerate(L.CONSTRAINT_PLAIN) if ("label", c.f(p).lower()) not in c.used]
    j = c.pick(free or [9])
    c.used.add(("label", c.f(L.CONSTRAINT_PLAIN[j]).lower()))
    label_t, ks, echoes, keeps, phs, examples = L.CONSTRAINT_FIELDS[j]
    plain = c.chance(0.4)
    if plain:
        # the label without its limit: a helper or placeholder that states the limit adds it (K4, KEEP)
        label_t = L.CONSTRAINT_PLAIN[j]
    k = c.pick(ks)
    kw = {"k": k, "K": num_word(k) if k else "", "k1": c.pick([1, 2]), "year": c.pick([2019, 2021, 2023]),
          "pin": "".join(str(c.rng.randrange(1, 10)) for _ in range(max(k, 1))), "item": c.pick(item_names(c))}
    label = cap(c.f(label_t, **kw))
    lab = S(label, "label")
    ph = None
    if plain and c.chance(0.3):
        ph = S(cap(c.f(c.pick(phs), **kw)), "placeholder")
    elif not plain and c.want(0.25):
        ph = S(cap(c.f(c.pick(phs), **kw)), "placeholder").echo(lab, "R8")
    elif c.chance(0.4) or swift:
        ph = S(c.f(c.pick(examples), **kw), "placeholder")
    hl = None
    if plain and c.chance(0.6):
        hl = S(cap(c.f(c.pick(echoes), **kw)), "helper").keep("K4")
    elif not plain and c.want(0.3):
        hl = S(cap(c.f(c.pick(echoes), **kw)), "helper").echo(lab, "R5")
    elif c.chance(0.7):
        hl = S(cap(c.f(c.pick(keeps), **kw)), "helper").keep("K4")
    return E("field", label=lab, ph=ph, helper=hl, type="password" if label.startswith("Password") else "text"), lab


INTEGRATIONS = [("Google Calendar", "{Y} appear in your calendar within a minute.", "Your schedule, in Google’s app."),
                ("Slack", "Posts a message in the channel you choose.", "Chat updates."),
                ("Xero", "Paid invoices are marked as paid in Xero overnight.", "Accounting sync."),
                ("Zapier", "Needs a Zapier account on any paid plan.", "Automations."),
                ("Mailchimp", "New members join the list you pick; existing ones are left alone.", "Newsletter sign-ups."),
                ("Outlook", "Two-way sync; edits in either place win by time.", "Your Microsoft diary.")]
# Toggles grouped by the heading they sit under: (label, helper, restatement sharing no stem)
TOGGLE_GROUPS = {
    "Email": [("Email notifications", "A digest at {clock}, never more than once a day.", "Messages in your inbox."),
              ("Weekly summary", "Sent on {day} mornings, covering the previous seven days.", "A recap every seven days."),
              ("Marketing emails", "About one a month. Unsubscribing does not affect receipts.", "News and offers.")],
    "On your phone": [("Push notifications", "Sent only between 8am and 9pm unless marked urgent.", "Alerts that pop up."),
                      ("Mention alerts", "Only for mentions in {x} you follow.", "When someone tags you."),
                      ("Sounds", "Follows your phone’s silent switch.", "Audible chimes.")],
    "Privacy": [("Public profile", "Anyone with the link can see your name and bio, but not your email.", "Others can view you."),
                ("Share usage data", "Anonymous counts only; no file names or contents.", "Help us improve."),
                ("Show me in search", "{Who} in other {groups} can find you by name.", "Be findable.")],
    "Sign-in": [("Two-step sign-in", "Asks for a code from your phone on a new device.", "Extra protection at login."),
                ("Sign-in alerts", "We email you when a new device signs in.", "Hear about logins."),
                ("Remember this device", "For {n} days; clearing cookies forgets it.", "No password next time.")],
    "Reminders": [("{Y1} reminders", "Sent {n} days before each {y1} is due.", "Nudges before the date."),
                  ("Auto-archive", "{X} untouched for {n} days move to the archive; nothing is deleted.", "Tidy up old entries."),
                  ("Daily digest", "One email at {clock} listing today’s {y}.", "A morning roundup.")],
    "Connected apps": INTEGRATIONS,
}
TOGGLE_ALT_MORE = {
    "Email notifications": ["Messages to your inbox.", "Updates by mail."], "Weekly summary": ["One recap a week.", "Your seven-day roundup."],
    "Marketing emails": ["Offers and announcements.", "Promotional mail."], "Push notifications": ["Pop-up alerts.", "Banners on your lock screen."],
    "Mention alerts": ["Hear when you are tagged.", "When people name you."], "Sounds": ["Chimes and beeps.", "Audio cues."],
    "Public profile": ["Let others view you.", "Visible to anyone."], "Share usage data": ["Send anonymous statistics.", "Help improve things."],
    "Show me in search": ["Let people find you.", "Appear in lookups."], "Two-step sign-in": ["A second check at login.", "Extra login protection."],
    "Sign-in alerts": ["Warnings about logins.", "Know when someone logs in."], "Remember this device": ["Fewer sign-ins on this one.", "Skip login next time."],
    "Auto-archive": ["Tidy away old entries.", "Move old things aside."], "Daily digest": ["A morning roundup.", "Your day in one mail."],
}
# more KEEP helpers per toggle, so that no one template recurs
TOGGLE_HELP_MORE = {
    "Email notifications": ["At most one email a day, at {clock}.", "Comments and mentions only; never marketing."],
    "Weekly summary": ["Covers Monday to Sunday; sent the next morning.", "Skipped in weeks with no changes."],
    "Marketing emails": ["Roughly monthly. Receipts still arrive.", "Unsubscribing here does not stop receipts."],
    "Push notifications": ["Silent between 9pm and 8am unless urgent.", "Only on phones signed in to this account."],
    "Mention alerts": ["Muted {x} never alert you.", "Mentions by {roles} are always sent."],
    "Public profile": ["Your email stays hidden either way.", "Search engines do not index it."],
    "Share usage data": ["Counts only, never names or notes.", "Sent once a week, anonymised."],
    "Show me in search": ["Only your name and {group} are shown.", "{Roles} can always find you."],
    "Two-step sign-in": ["A code by text or an authenticator app.", "Asked for once per new device."],
    "Sign-in alerts": ["Includes the town and the device.", "Sent to your sign-in email only."],
    "Remember this device": ["Clearing cookies forgets it.", "Up to {n} days on this browser."],
    "{Y1} reminders": ["{n} days ahead by default; change it for each {x1}.", "Skips {y} already marked done.",
                       "Push first, then email if unread."],
    "Auto-archive": ["After {n} quiet days, {x} are archived, not deleted.", "Pinned {x} are never archived.",
                     "Archived {x} still count in yearly totals."],
    "Daily digest": ["Arrives at {clock}; skipped on days with nothing due.", "Lists {y} for the next 24 hours.",
                     "Weekends included unless you turn them off."],
}
SELECTS = [
    ("Time zone", ["London (GMT+1)", "Lisbon (GMT+1)", "Paris (GMT+2)", "New York (GMT−4)"], "Due dates and reminders use this zone."),
    ("Language", ["English (UK)", "Cymraeg", "Français", "Deutsch", "Español"], "Some help articles are only in English."),
    ("Date format", ["DD/MM/YYYY", "YYYY-MM-DD", "MM/DD/YYYY"], "Applies to exports as well as the screen."),
    ("Currency", ["GBP (£)", "EUR (€)", "USD ($)"], "Existing invoices keep their original currency."),
    ("Week starts on", ["Monday", "Sunday", "Saturday"], "Changes the calendar and weekly reports."),
    ("Session length", ["1 day", "7 days", "30 days"], "After this long without use you are signed out."),
]
SETTINGS_PAGES = {
    # page title: (field sections, toggle groups, selects, synonyms for R1 eyebrows and R3 echoes sharing no stem)
    "Profile": (["Profile"], ["Privacy"], [], ["About you", "Who you are"]),
    "Account": (["Account"], ["Sign-in"], ["Language"], ["Your login", "Sign-in details"]),
    "Notifications": ([], ["Email", "On your phone", "Reminders"], [], ["Alerts", "What we tell you about"]),
    "Billing": (["Billing"], [], ["Currency"], ["Payments", "Plan and invoices"]),
    "Workspace": (["Workspace"], [], ["Time zone", "Week starts on"], ["Your team", "Team setup"]),
    "Security": (["Account"], ["Sign-in"], ["Session length"], ["Sign-in and safety", "Protection"]),
    "Integrations": ([], ["Connected apps"], [], ["Connected apps", "Other tools"]),
    "Preferences": ([], ["Reminders"], ["Time zone", "Date format", "Language", "Week starts on"], ["Options", "How it behaves"]),
}


def settings(c: Ctx) -> Page:
    rng = c.rng
    swift = c.fw == "swift"
    body = []
    multi = not swift and c.chance(0.15)
    if multi:
        k = c.pick([2, 3])
        total = c.pick([3, 4, 5])
        body.append(E("note", s=S(c.pick(["Step {k} of {t}", "Part {k} of {t}"]).format(k=k, t=total), "caption").keep("K3")))
    domain_page = c.chance(0.25)
    if domain_page:
        k = c.rng.randrange(4)
        alt = c.alt(c.x)[0]
        htext = [f"{c.x.Sg} settings", f"{c.x.Sg} details", f"Edit {c.x.sg}", f"{c.product} preferences"][k]
        syns = [[f"This {alt}", f"About this {alt}"], [f"About this {alt}", f"This {alt}"],
                [f"Change this {alt}", f"Update the {alt}"], ["Options", "How it behaves"]][k]
        plan = ([], ["Reminders"] if c.chance(0.5) else [], [], syns)
    else:
        htext = c.pick(list(SETTINGS_PAGES))
        plan = SETTINGS_PAGES[htext]
    sec_names, tog_names, sel_names, syns = plan
    h1 = S(htext, "h1")
    nav_title = None
    if swift:
        nav_title = h1
        if c.want(0.3):
            r9 = S(htext, "h2").echo(nav_title, "R9")
            body.append(E("section", E("h", s=r9, level=2)))
    else:
        kids = []
        eb = None
        if c.want(0.2):
            eb = S(c.pick(syns), "eyebrow").echo(h1, "R1", conf="judgement")
        elif c.chance(0.15):
            eb = S(c.pick([c.product, c.pick(["Settings", "Account settings", "Your account"])]), "eyebrow").keep("K1", conf="judgement")
            if shares_stem(eb.text, htext) or eb.text == htext:
                eb = S(c.product, "eyebrow").keep("K1")
        if eb:
            kids.append(E("eyebrow", s=eb))
        kids.append(E("h", s=h1, level=1))
        if c.want(0.25):
            if c.chance(0.5):
                sub = S(c.pick(["Manage your {n}.", "Manage your {n} here.", "View and change your {n}.", "Your {n}, all in one place.",
                                "Everything about your {n}.", "Update your {n}.", "{N}, in one place.", "All {n}; every one, here.",
                                "{N} for this account, listed.", "The {n}, from top to bottom."]).format(
                    n=lower_first(htext), N=htext), "subtitle").echo(h1, "R3")
            else:
                syn = c.pick(syns)
                pool = [f"{cap(syn)}, in one place.", f"{cap(syn)}, all together.", f"Review {lower_first(syn)}."]
                if not syn.lower().startswith("about"):
                    pool.append(f"Everything about {lower_first(syn)}.")
                if htext in ("Preferences",) or htext.endswith("settings"):
                    pool += ["Change how things work for you.", "Adjust things to suit you.", "Make it work your way."]
                sub = S(c.pick(pool), "subtitle").echo(h1, "R3", conf="judgement")
            kids.append(E("sub", s=sub))
            if eb is not None and eb.cls == "ECHO":
                stack(h1, [eb, sub], ["R1", "R3"], conf="judgement")
        elif c.chance(0.15):
            kids.append(E("sub", s=S(short_keep(c, "settings"), "subtitle")))
        elif c.chance(0.5):
            kids.append(E("sub", s=S(c.f(c.pick(["Changes apply straight away on every device.", "Only you can see these {n}.",
                                                   "{Role}s can change these for the whole {group}.", "Changes here take effect from your next sign-in.",
                                                   "These {n} apply to every {x1} in your {group}."]), n=lower_first(htext),
                                         Role=cap(c.role)), "subtitle")))
        body.append(E("head", *kids))
    tab_names = set()
    if not swift and not domain_page and c.chance(0.35):
        others = [p for p in SETTINGS_PAGES if p != htext]
        tabs = [S(t, "tab") for t in rng.some(others, 3)]
        tabs.insert(rng.randrange(len(tabs) + 1), S(htext, "tab").keep("K11"))
        tab_names = {t.text for t in tabs}
        body.insert(0 if c.chance(0.5) else len(body), E("tabs", items=tabs))
    by_name = dict(T.SETTINGS_SECTIONS)
    first_err = None
    form_kids = []
    sections = [(n, by_name[n]) for n in sec_names]
    if domain_page:
        sections = [(c.pick([f"{c.x.Sg} details", "Basics", c.x.Sg]), c.dom["fields"])]
    for name, fields in sections:
        r9 = False
        if name == htext:
            if not swift and c.want(0.3):
                r9 = True
            else:
                name = c.pick(["General", "Details", "Basics", "Your details"] + ([] if name.endswith("details") else [f"{name} details"]))
        h2 = S(name, "h2")
        if name in tab_names:
            h2.keep("K11")
        sk = [E("h", s=h2, level=2)]
        nounish = name in ("Profile", "Account", "Billing", "Workspace", c.x.Sg) or name.endswith(" details")
        if r9:
            h2.echo(h1, "R9")
        elif nounish and not name.lower().startswith("your") and c.want(0.12):
            base_n = lower_first(name).replace(" details", "")
            sk.append(E("sub", s=S(c.pick(["Manage your {n} details.", "Your {n} information.", "Edit your {n} details here.",
                                           "Details of your {n}.", "Change your {n} details."]).format(n=base_n), "subtitle").echo(h2, "R3")))
        elif c.chance(0.3):
            keep = {"Billing": ["Used on invoices and receipts.", "Billing details are checked against your card each month."],
                    "Profile": ["Shown to everyone in your {group}.", "Your profile is visible to {who} you share {x} with."],
                    "Account": ["Only you can see these.", "Account changes need your password."],
                    "Workspace": ["Shown in invitations and the browser tab.", "Workspace changes apply to all {n} members."]}
            sk.append(E("sub", s=S(c.f(c.pick(keep.get(name, ["Only you and {roles} can see these.", "Changes apply straight away.",
                                                               "Shown to everyone in your {group}.", "Checked before saving."]))),
                                   "subtitle")))
        for label, example, helper in rng.some(fields, c.pick([1, 2, 2, 3])):
            fe, lab = field_from(c, label, example, helper, swift)
            if not swift and first_err is None and c.chance(0.12):
                err = S(c.pick(["Enter " + lower_first(label), lower_first(label).capitalize() + " is required",
                                f"Check the {lower_first(label)} and try again"]), "error")
                fe.p["error"] = err
                fe.p["err_cond"] = "errors." + "".join(w.capitalize() if i else w.lower() for i, w in enumerate(re.split(r"[^A-Za-z]+", label)) if w)
                c.prop("errors")
                first_err = err
            sk.append(fe)
        if c.chance(0.35):
            sk.append(constraint_field(c, swift)[0])
        form_kids.append(E("section", *sk))
    for gname in tog_names:
        name = gname.format(Y=c.y.Pl)
        h2 = S(name, "h2")
        if name in tab_names:
            h2.keep("K11")
        sk = [E("h", s=h2, level=2)]
        k9 = False
        for label, helper, alt in rng.some(TOGGLE_GROUPS[gname], c.pick([1, 2, 3])):
            alt = c.pick([alt, *TOGGLE_ALT_MORE.get(label, [])])
            raw_label = label
            label = c.f(label)
            lab = S(label, "label")
            hl = None
            if c.want(0.18):
                r = c.rng.random()
                if r < 0.3:
                    hl = S(c.pick(["Receive {l}.", "Get {l}.", "{L} for your account.", "{L}; that is all."])
                           .format(l=lower_first(label), L=label), "helper").echo(lab, "R5", conf="judgement")
                elif r < 0.5:
                    # what the visible switch does (LABELLING.md §8: NARRATE R6, ct)
                    hl = S(c.pick(["Switch {l} on or off.", "Turn {l} on or off.", "Turn on {l}.", "Toggle {l}."])
                           .format(l=lower_first(label)), "helper").narrate("R6", lab, ct=True, conf="judgement")
                else:
                    hl = S(alt, "helper").echo(lab, "R5", conf="judgement")
            elif c.chance(0.2):
                hl = S(short_keep(c, "toggle"), "helper").keep("K4")
            elif c.chance(0.65):
                hl = S(c.f(c.pick([helper, *TOGGLE_HELP_MORE.get(raw_label, [])])), "helper").keep("K4")
            sk.append(E("toggle", label=lab, helper=hl))
            if shares_stem(name, label):
                k9 = True
        if c.chance(0.2):
            j = c.rng.randrange(len(L.FIGURE_TOGGLES))
            label_t, ks, echoes, keeps = L.FIGURE_TOGGLES[j]
            plain = c.chance(0.4)
            kw = {"k": c.pick(ks), "clock": L.clock(c.rng)}
            kw["K"] = num_word(kw["k"]) if kw["k"] else ""
            lab = S(cap(c.f(L.FIGURE_TOGGLES_PLAIN[j] if plain else label_t, **kw)), "label")
            if plain and c.chance(0.6):
                hl = S(cap(c.f(c.pick(echoes), **kw)), "helper").keep("K4")
            elif not plain and c.want(0.3):
                hl = S(cap(c.f(c.pick(echoes), **kw)), "helper").echo(lab, "R5")
            else:
                hl = S(cap(c.f(c.pick(keeps), **kw)), "helper").keep("K4")
            sk.append(E("toggle", label=lab, helper=hl))
        if k9 and h2.cls == "KEEP":
            h2.keep("K9")
        form_kids.append(E("section", *sk))
    if sel_names:
        h2 = S(c.pick(["Regional", "Display", "Defaults", "Formats"]) if len(sel_names) > 1 else c.pick(["Options", "More"]), "h2")
        sk = [E("h", s=h2, level=2)]
        sel = {s[0]: s for s in SELECTS}
        for nm in rng.some(sel_names, min(len(sel_names), c.pick([1, 2]))):
            label, opts, helper = sel[nm]
            sk.append(E("select", label=S(label, "label"), options=[S(o, "menu_item") for o in opts],
                        helper=S(helper, "helper").keep("K4") if c.chance(0.6) else None))
        form_kids.append(E("section", *sk))
    if not form_kids:
        fe, _ = field_from(c, *c.pick(c.dom["fields"]), swift)
        form_kids.append(E("section", fe))
    # a near-synonym h2 heading the page's only section repeats the page title (LABELLING.md §8: R9, judgement);
    # with other sections on the page it is KEEP
    secs = [k for k in form_kids if k.kind == "section" and k.kids and k.kids[0].kind == "h"]
    if len(secs) == 1 and secs[0].kids[0]["s"].text in syns and secs[0].kids[0]["s"].cls == "KEEP":
        if c.want(0.45):
            secs[0].kids[0]["s"].echo(h1, "R9", conf="judgement")
        else:
            fe, _ = field_from(c, "Webhook URL", "e.g. https://example.com/hook", c.f("We post an event each time {a_x1} changes."), swift)
            form_kids.append(E("section", E("h", s=S(c.pick(["Webhooks", "Developer access", "API"]), "h2"), level=2), fe))
    if first_err is not None and c.fw != "swift" and c.chance(0.6):
        summ = E("errsummary", title=S(c.pick(["There is a problem", "Check these fields", "Some details are missing"]), "h2"),
                 items=[S(first_err.text, "link")])
        first_err.keep("K12")
        form_kids.insert(0, summ)
    save = S(c.pick(["Save changes", "Save", "Save " + lower_first(htext) if htext != "Profile" else "Update profile", "Apply",
                     "Save settings" if not domain_page else f"Save {c.x.sg}"]), "button")
    acts = [E("button", s=S("Cancel", "button"), primary=False), E("button", s=save)] if c.chance(0.5) else [E("button", s=save)]
    if c.want(0.12):
        form_kids.append(E("note", s=S(narrate_control(c, save.text), "caption").narrate("R6", save, ct=True)))
    elif c.chance(0.15):
        form_kids.append(E("note", s=S(c.pick(["Changes to your email take effect after you confirm the link we send.",
                                              "Saving signs you out on other devices.", "Changes apply to new entries only."]),
                                       "caption").keep("K4", ct=True)))
    form_kids.append(E("actions", *acts))
    if swift:
        body.extend(form_kids)
    else:
        body.append(E("form", *form_kids))
    base = ident_words(htext.replace("Settings", "Prefs"))
    name = c.pick([f"{base}Form", f"{base}Page", f"{base}Screen", f"Edit{base}"])
    return Page(name, "settings", body, product=c.product, props=c.props, nav_title=nav_title)


# ---------------------------------------------------------------------------------------------------------------
# dialogs

ACKS = [("Export started", "Your export has started.", "The download has begun.",
         ["We will email the file when it is ready, usually within {n} minutes.", "Want it sooner? Shorter date ranges finish faster.",
          "Only you can open the link."]),
        ("Invite sent", "Your invite has been sent.", "They have been asked to join.",
         ["The link in the invite works for {n} days.", "Make sure they check their spam folder.", "Your {role} is copied in."]),
        ("Payment received", "We have received your payment.", "Thanks, it went through.",
         ["A receipt is on its way to your inbox.", "It shows on your statement as {p}.", "Your next payment is on {date}."]),
        ("{Y1} booked", "Your {y1} is booked.", "All confirmed.",
         ["You can change it until {clock} the day before.", "Confirm by email within 24 hours to keep the place.",
          "Shared with your {group}."]),
        ("{X1} saved", "This {x1} has been saved.", "It is stored safely.",
         ["Saved {x} are kept for {n} days in the archive.", "Synced to your other devices.", "Your {role} can see it now."]),
        ("{X} imported", "Your {x} have been imported.", "They are all in.",
         ["{n} rows were skipped because they had no date.", "Want to check them? The skipped rows are in a CSV.",
          "Duplicates were merged, {n} in all."])]
PAST = {"Delete": "deleted", "Remove": "removed", "Archive": "archived", "Cancel": "cancelled", "Withdraw": "withdrawn",
        "Unpublish": "unpublished", "Disconnect": "disconnected", "Reset": "reset", "Discard": "discarded"}


def dialog_echo(c: Ctx, verb: str, noun_text: str, alt_noun: str | None) -> tuple[str, str]:
    """A body that only asks the title's question again (R4): with the title's words, or with synonyms."""
    base, _, syns = L.VERBS[verb]
    if alt_noun is not None and c.chance(0.45):
        t = c.pick(["Are you sure you want to {v} this {o}?", "Do you really want to {v} it?", "Sure you want to {v} this {o}?",
                    "Please confirm you want to {v} it."])
        return t.format(v=c.pick(syns), o=alt_noun), "judgement"
    t = c.pick(["Are you sure you want to {v} this {o}?", "Do you really want to {v} this {o}?", "You are about to {v} this {o}.",
                "Please confirm that you want to {v} this {o}.", "This will {v} the {o}.", "This {o} will be {pp}.",
                "The {o} is about to be {pp}.", "Confirm: the {o} will be {pp}."])
    return t.format(v=base, o=noun_text, pp=PAST[verb]), "clear"


def dialog_keep(c: Ctx, verb: str, noun_text: str) -> str:
    """A consequence (K7), naming the action (shared stem) or not."""
    _, ger, _ = L.VERBS[verb]
    other = c.y.pl if noun_text != c.y.sg else c.x.pl
    if c.chance(0.22):
        # a consequence, then a question that offers something new (K7; the body ends with "?")
        t = c.pick(["Its {n} {other} go with it. Export them first?", "Anyone it is shared with loses access. Tell them first?",
                    "{N} people opened it this week. Check with them first?", "Its history goes too. Download a copy first?",
                    "Your {n} shared links stop working. Send a copy instead?", "The {role} is told by email. Add a note?"])
        return c.f(t, o=noun_text, other=other, N=num_word(c.pick([2, 3, 4, 5, 6])))
    if c.chance(0.15):
        return c.f(c.pick(["You can restore it from the bin for {n} days.", "Your {group} is told by email.",
                           "Your {n} shared links stop working.", "You keep the {other} linked to it."]), other=other)
    if c.chance(0.5):
        t = c.pick(["{G} this {o} also removes its {n} {other}.", "{G} {a_o} cannot be undone after {n} days in the bin.",
                    "{G} it notifies the {n} people it is shared with.", "{G} this {o} frees one of the {n} places on your plan.",
                    "{G} it now keeps the {n} {other} linked to it, but they lose their link."])
        return c.f(t, G=ger, o=noun_text, other=other)
    return c.f(c.pick(T.GENERIC_CONSEQUENCES + ["Its {n} {other} move to the archive.", "Anyone it is shared with loses access at once.",
                                               "The {n} {other} linked to it stay, but show it as removed."]), other=other)


def dialog(c: Ctx) -> Page:
    rng = c.rng
    swift = c.fw == "swift"
    r = c.rng.random()
    page_h1 = None
    if r < 0.55:
        # a confirmation
        if c.chance(0.4):
            btxt, noun_text, cons = c.dom["danger"]
            verb = btxt.split(" ")[0]
            alt_noun = None
            if noun_text in (c.x.sg, c.y.sg):
                alt_noun = c.alt(c.x if noun_text == c.x.sg else c.y)[0]
        else:
            verb = c.pick(["Delete", "Remove", "Archive", "Delete", "Cancel" if c.dom["key"] in ("bikes", "ferry", "vet", "kayak", "tutor", "music", "pottery", "climb") else "Archive"])
            noun = c.noun()
            noun_text, alt_noun = noun.sg, c.alt(noun)[0]
            cons = None
            btxt = f"{verb} {noun_text}"
        named = noun_text == c.x.sg and c.chance(0.3)
        item = c.pick(item_names(c)) if named else None
        many = None
        if named:
            title = S(f"{verb} {item}?", "dialog_title")
        elif noun_text in (c.x.sg, c.y.sg) and c.chance(0.1):
            # no count in the title: a body that states it adds a consequence (K7)
            many = (c.x if noun_text == c.x.sg else c.y, c.pick([3, 4, 5, 6, 8, 12, 14, 20]), False)
            title = S(f"{verb} selected {many[0].pl}?", "dialog_title")
            btxt = f"{verb} selected"
        elif noun_text in (c.x.sg, c.y.sg) and c.chance(0.25):
            # a count in the title, which a body may only repeat (R4) or add to (K7)
            many = (c.x if noun_text == c.x.sg else c.y, c.pick([3, 4, 5, 6, 8, 12, 14, 20]))
            title = S(f"{verb} {many[1]} {many[0].pl}?", "dialog_title")
            btxt = f"{verb} {many[1]}"
        else:
            title = S(f"{verb} {noun_text}?" if c.chance(0.7) else f"{verb} this {noun_text}?", "dialog_title")
        confirm = S(btxt if c.chance(0.7) else verb, "button").keep("K11")
        bodies = []
        if many is not None and len(many) == 3:
            pl, k = many[0].pl, many[1]
            bodies.append(S(c.pick([f"All {k} {pl} will be {PAST[verb]}.", f"You are about to {verb.lower()} {k} {pl}.",
                                    f"{num_word(k)} {pl} are about to be {PAST[verb]}."]), "dialog_body").keep("K7"))
        elif many is None and not swift and c.chance(0.1):
            t, _ = dialog_echo(c, verb, noun_text, alt_noun)
            bodies.append(S(t + " " + (cons if cons else dialog_keep(c, verb, noun_text)), "dialog_body").keep("K7", partial=True))
        elif c.want(0.35):
            if many is not None:
                pl, k = many[0].pl, many[1]
                t, conf = c.pick([f"All {k} {pl} will be {PAST[verb]}.", f"You are about to {verb.lower()} {k} {pl}.",
                                  f"{num_word(k)} {pl} are about to be {PAST[verb]}.", f"{num_word(k)} {pl}, {PAST[verb]} at once."]), "clear"
            else:
                t, conf = dialog_echo(c, verb, noun_text, alt_noun)
            bodies.append(S(t, "dialog_body").echo(title, "R4", conf=conf))
        if not (swift and bodies) and (not bodies or c.chance(0.5)):
            if c.want(0.2) and not swift:
                bodies.append(S(narrate_control(c, confirm.text), "dialog_body").narrate("R6", confirm, ct=True))
            elif not bodies or c.chance(0.6):
                if many is not None:
                    other = c.y.pl if many[0] is c.x else c.x.pl
                    kt = c.f(c.pick(["Their {n} {other} go too; this cannot be undone.", "Anyone they are shared with loses access at once.",
                                     "They stay in the bin for {n} days.", "Their history goes too. Download a copy first?",
                                     "You can restore them from the bin for {n} days.", "Your {group} is told by email."]), other=other)
                else:
                    kt = cons if cons and c.chance(0.5) else dialog_keep(c, verb, noun_text)
                bodies.append(S(kt, "dialog_body").keep("K7"))
        if swift and len(bodies) > 1:
            bodies = bodies[:1]
        cancel = S(c.pick(["Cancel", "Keep it", "Go back", "Not now"] if verb != "Cancel" else ["Keep it", "Go back", "Not now"]), "button")
        kids = [E("actions", E("button", s=cancel, primary=False), E("button", s=confirm, destructive=True))]
        d = E("dialog", *kids, title=title, body=bodies, confirm=True)
    elif r < 0.72 and not swift:
        # a form dialog: renaming takes a name; creating takes the domain's fields
        noun = c.x if c.chance(0.6) else c.y
        rename = c.chance(0.45)
        if rename:
            title = S(f"Rename {noun.sg}" if c.chance(0.6) else f"Rename this {noun.sg}", "dialog_title")
            label = c.pick(["Name", f"{noun.Sg} name", "New name"])
            example = f"e.g. {c.pick(item_names(c))}"
            helper = c.f(c.pick(["The new name shows everywhere this {a} appears, including shared links.",
                                 "Up to {n} characters; names must be unique in your {group}.",
                                 "Links people have saved keep working."]), a=noun.sg)
            fields = [(label, example, helper)]
        else:
            title = S(c.pick([f"New {noun.sg}", f"Add {article(noun.sg)}", f"Add {noun.sg}"]), "dialog_title")
            fields = rng.some(c.dom["fields"], c.pick([1, 2]))
        fkids = []
        for fld in fields:
            fe, _ = field_from(c, *fld, False)
            fkids.append(fe)
        if not rename and c.chance(0.25):
            fkids.append(constraint_field(c, False)[0])
        bodies = []
        if c.want(0.15):
            if rename:
                t = c.pick(["Give this {o} a new name.", "Enter a new name for the {o}.", "Choose a new name."]) if c.chance(0.5) \
                    else c.pick(["Choose what to call it.", "Pick something different to call it.", "Call it something else."])
            else:
                t = c.pick(["Add a new {o}.", "Create {a_o} here.", "Fill in the details of your new {o}."]) if c.chance(0.5) \
                    else c.pick([f"Set up another {c.alt(noun)[0]}.", "Start another one."])
            bodies.append(S(t.format(o=noun.sg, a_o=article(noun.sg)), "dialog_body").echo(title, "R4", conf="judgement"))
        elif c.chance(0.5):
            bodies.append(S(about(c, noun), "dialog_body"))
        save = S(c.pick(["Save", "Create", f"Create {noun.sg}", "Save name", "Add"]) if not rename else c.pick(["Save", "Rename", "Save name"]),
                 "button")
        if shares_stem(save.text, title.text):
            save.keep("K11")
        kids = fkids + [E("actions", E("button", s=S("Cancel", "button"), primary=False), E("button", s=save))]
        d = E("dialog", *kids, title=title, body=bodies)
    elif r < 0.85:
        # an action page and its dialog (R9 when the dialog repeats the page title)
        act = c.pick(["Export", "Import", "Share", "Print", "Download"])
        noun = c.noun()
        ptext = f"{act} {noun.pl}"
        page_h1 = S(ptext, "h1")
        if c.want(0.4):
            if c.chance(0.6):
                title = S(ptext, "dialog_title").echo(page_h1, "R9", conf="clear")
            else:
                syn = {"Export": "Download", "Import": "Bring in", "Share": "Send", "Print": "Print out", "Download": "Save"}[act]
                title = S(f"{syn} your {c.alt(noun)[1]}", "dialog_title").echo(page_h1, "R9", conf="judgement")
        else:
            title = S(c.pick([f"{act} {c.rng.randrange(2, 40)} {noun.pl}?", f"{act} as CSV", f"{act} {noun.pl} from {c.pick(L.MONTHS)}",
                              f"{act} to your {c.group}"]), "dialog_title")
        body_t = c.f(c.pick(["Files over {n} MB are split into parts.", "The file includes notes and photos; it may take {n} minutes.",
                             "Only {x} you can edit are included.", "Columns follow the order on screen."]))
        bodies = [S(body_t, "dialog_body").keep("K7" if c.chance(0.3) else None)]
        ok = S(c.pick([act, f"{act} now", "Start", "Continue"]), "button")
        if shares_stem(ok.text, title.text):
            ok.keep("K11")
        kids = [E("actions", E("button", s=S("Cancel", "button"), primary=False), E("button", s=ok))]
        d = E("dialog", *kids, title=title, body=bodies)
    else:
        # an acknowledgement
        t, echo_stem, echo_alt, keep_bodies = c.pick(ACKS)
        keep_body = c.pick(keep_bodies)
        title = S(c.f(t), "dialog_title")
        bodies = []
        if c.want(0.3):
            if c.chance(0.5):
                bodies.append(S(c.f(echo_stem), "dialog_body").echo(title, "R4", conf="judgement"))
            else:
                bodies.append(S(c.f(echo_alt), "dialog_body").echo(title, "R4", conf="judgement"))
        else:
            bodies.append(S(c.f(keep_body), "dialog_body"))
        ok = S(c.pick(["OK", "Done", "Close", "Got it"]), "button")
        kids = [E("actions", E("button", s=ok))]
        d = E("dialog", *kids, title=title, body=bodies, sheetlike=False)
    body = []
    if page_h1 is not None:
        act_facts = {"Export": ["Exports include notes and photos.", "Export files are kept for {n} days.", "Exports open in any spreadsheet."],
                     "Import": ["Imports accept CSV files up to {n} MB.", "Rows without a date are skipped and listed.", "Imports can be undone for {n} hours."],
                     "Share": ["Shared links are read-only.", "Links expire after {n} days.", "People you share with need no account."],
                     "Print": ["Each page fits one A4 sheet.", "Prints include the last {n} {y}.", "Photos print in black and white."],
                     "Download": ["Downloads are zipped when over {n} MB.", "Downloads include notes and photos.", "Files are named by date."]}
        body.append(E("section", E("h", s=page_h1, level=1), E("p", s=S(c.f(c.pick(act_facts[act])), "body"))))
    elif c.chance(0.4):
        body.append(E("section", E("h", s=S(c.noun().Pl, "h1"), level=1), E("p", s=S(product_fact(c), "body"))))
    body.append(d)
    if swift and not body[:-1]:
        body.insert(0, E("section", E("p", s=S(product_fact(c), "body"))))
    if c.fw == "tsx":
        c.prop("open")
    name = c.pick([f"Confirm{ident_words(title.text)}", f"{ident_words(title.text)}Modal", f"{ident_words(title.text)}Sheet",
                   f"{c.x.Sg}Actions"])
    return Page(name[:40], "dialog", body, product=c.product, props=c.props)


# ---------------------------------------------------------------------------------------------------------------
# empty states

EMPTY_TITLES = ["No {a} yet", "No {a} this {period}", "You have no {a}", "No {a} to show", "Nothing here yet", "No {a} found"]
# dated titles and bodies that repeat their date and add nothing (R4)
EMPTY_DATED = [("No {a} since {date}", ["Not a single {a1} since {date}.", "Since {date}, there have been none.",
                                        "There are no {a} dated after {date}."]),
               ("No {a} on {day}", ["{day} has none, not one {a1}.", "There are no {a} for {day}.", "Nothing on {day}; the day is clear."]),
               ("No {a} due before {date}", ["Nothing falls due until after {date}.", "Before {date}, no {a1} is due."]),
               ("No {a} in {month}", ["{month} has no {a}, not one.", "None at all in {month}.", "In {month}, zero {a}."])]
EMPTY_KEEP = ["{A} you add from any device appear here, newest first.", "Add one from the menu, or import a CSV.",
              "{A} shared with you by your {group} also show here.", "Try clearing the filters; {n} {a} are hidden by them.",
              "Your {role} adds {a} each {day}; check back then.", "Import last year’s {a} from a spreadsheet to get going.",
              "Ask your {role} to add one.", "Check back on {day}.", "Filters may be hiding some.", "New {a} land here first.",
              "Want one now? Start from a template.", "Make sure you are in the right {group}.", "Your {group} has not shared any.",
              "Clear the search to see all {a}.", "{A} older than {n} months are in the archive.", "Imported {a} appear within the hour."]
EMPTY_ECHO_STEM = ["There are no {a} yet.", "You don’t have any {a}.", "No {a} have been added.", "The list of {a} is empty.",
                   "It looks like there are no {a} here.", "Not one {a1} so far.", "No {a} so far; none at all.",
                   "Zero {a}, for now.", "The {a} list has nothing in it.", "None yet, not a single {a1}."]


def empty_copy(c: Ctx, noun: Noun):
    """An empty-state title and a K8 body (cause or next action) for ``noun``."""
    for et, eb in c.dom["empty"]:
        if noun.pl.lower() in et.lower() and c.chance(0.4):
            return et, eb
    kw = {"a": noun.pl, "a1": noun.sg, "A": noun.Pl, "period": c.pick(["week", "month", "season"]), "month": c.pick(L.MONTHS)}
    if c.chance(0.25):
        t = c.f(c.pick(EMPTY_DATED)[0], **kw)
    else:
        t = c.f(c.pick(EMPTY_TITLES), **kw)
    return t, c.f(c.pick(EMPTY_KEEP), **kw)


def empty_echo(c: Ctx, noun: Noun, title: str):
    """A body that only repeats the empty state's title (R4): its date when the title has one."""
    for tt, echoes in EMPTY_DATED:
        m = re.fullmatch(re.escape(tt).replace(r"\{a\}", "(?P<a>.+?)").replace(r"\{date\}", "(?P<date>.+)")
                         .replace(r"\{day\}", "(?P<day>.+)").replace(r"\{month\}", "(?P<month>.+)"), title)
        if m and m.groupdict().get("a") == noun.pl:
            return cap(c.pick(echoes).format(a=noun.pl, a1=noun.sg, **{k: v for k, v in m.groupdict().items() if k != "a"})), "clear"
    if c.chance(0.75):
        return c.pick(EMPTY_ECHO_STEM).format(a=noun.pl, a1=noun.sg), "clear"
    return nothing_line(c), "judgement"


def empty(c: Ctx) -> Page:
    noun = c.noun()
    et, eb = empty_copy(c, noun)
    page = []
    h = None
    if c.chance(0.45):
        h = S(noun.Pl, "h1")
        hk = [E("h", s=h, level=1)]
        if c.want(0.15):
            hk.append(E("sub", s=S(restate(c, noun), "subtitle").echo(h, "R3")))
        elif c.chance(0.3):
            hk.append(E("sub", s=S(about(c, noun), "subtitle")))
        page.append(E("head", *hk))
    action = None
    if c.chance(0.65):
        a = S(c.pick([f"New {noun.sg}", f"Add {noun.sg}", f"Create {noun.sg}", f"Add your first {noun.sg}", f"Log {article(noun.sg)}"]), "button")
        a.keep("K11")
        action = E("button", s=a)
    if h is not None and c.want(0.2):
        title = S(h.text, "empty_title").echo(h, "R9", conf="clear")
        body = S(eb, "empty_body").keep("K8")
    else:
        title = S(et, "empty_title")
        if c.want(0.4):
            t, conf = empty_echo(c, noun, et)
            body = S(t, "empty_body").echo(title, "R4", conf=conf)
        elif c.chance(0.2):
            # the restating clause and a next action: KEEP, partial (LABELLING.md §4 step 5)
            body = S(empty_echo(c, noun, et)[0] + " " + c.f(c.pick(EMPTY_KEEP), a=noun.pl, A=noun.Pl), "empty_body").keep("K8", partial=True)
        elif c.chance(0.12) and not re.search(r" (since|on|due before|in) ", et):
            # the same dated sentence as a restatement, under a title with no date: it adds the date (KEEP)
            tt, echoes = c.pick(EMPTY_DATED)
            body = S(cap(c.f(c.pick(echoes), a=noun.pl, a1=noun.sg, month=c.pick(L.MONTHS))), "empty_body").keep(conf="judgement")
        elif action is not None and c.want(0.2):
            body = S(c.pick([narrate_control(c, action["s"].text), f"Use the button below to create your first {noun.sg}.",
                             "Click the button below to add one."]), "empty_body").narrate("R6", action["s"], ct=True)
        else:
            body = S(eb, "empty_body").keep("K8")
    page.append(E("empty", title=title, body=body, action=action))
    if c.chance(0.25):
        page.append(E("note", s=S(c.f(c.pick(["Need a hand? Our help centre has a {n}-minute guide.",
                                              "{X} you delete stay in the bin for {n} days.", "Questions? Ask your {role}."])), "caption")))
    return Page(c.pick([f"No{ident_words(noun.pl)}", f"{ident_words(noun.pl)}Placeholder", f"{ident_words(noun.pl)}Blank",
                        f"Zero{ident_words(noun.pl)}"]), "empty", page, product=c.product)


# ---------------------------------------------------------------------------------------------------------------
# dashboards

DASH_HEADS = {
    "Overview": ["At a glance", "Summary"], "Dashboard": ["Home", "Your dashboard"], "This week": ["Weekly view", "Week so far"],
    "Today": ["Right now", "Today’s view"], "{p} at a glance": ["Overview", "Summary"], "{Group} overview": ["Summary", "At a glance"],
    "Your {group}": ["Members’ view", "Home"], "{X} overview": ["Summary", "At a glance"], "Home": ["Your dashboard", "Start"],
    "At a glance": ["Overview", "Summary"], "Summary": ["Overview", "At a glance"],
}


GROUP_LABELS = ["Details", "Summary", "Info", "Key facts", "More", "Overview", "At a glance", "Specifics", "The basics"]
FIELD_LABELS = [("Due", lambda c: L.date(c.rng)), ("Owner", lambda c: L.person(c.rng)), ("Location", lambda c: c.pick(L.PLACES)),
                ("Last {y1}", lambda c: c.pick(["Yesterday", "3 days ago", L.date(c.rng)])), ("Next {y1}", lambda c: f"{c.pick(L.DAYS)} {L.clock(c.rng)}"),
                ("Added", lambda c: L.date(c.rng, year=True)), ("Shared with", lambda c: f"{c.rng.randrange(2, 30)} people"),
                ("Reference", lambda c: f"#{c.rng.randrange(100, 9999)}")]


def detail_card(c: Ctx):
    """A card of field-style labels (K1) and their values, sometimes under a group label that only names what
    comes next (R2)."""
    kids = []
    if c.want(0.45):
        kids.append(E("eyebrow", s=S(c.pick(GROUP_LABELS), "label").decor("R2")))
    elif c.chance(0.65):
        kids.append(E("eyebrow", s=S(c.pick([c.pick(item_names(c)), f"Week {c.rng.randrange(2, 52)}", L.date(c.rng, year=True),
                                             f"{c.pick(L.MONTHS)} {c.pick([2025, 2026])}"]), "label").keep("K1", conf="judgement")))
    pairs = []
    for lab_t, fn in c.rng.some(FIELD_LABELS, c.pick([2, 3])):
        pairs.append(E("stat", label=S(c.f(lab_t), "label").keep("K1"), value=S(fn(c), "body"), note=None))
    kids.append(E("stats", *pairs))
    # a kicker that opens a card of three or more nodes reads as the card's title (h3), so a group label sits in a
    # plain block instead
    return E("div", *kids) if len(kids) > 1 else E("card", *kids, shadcn=False)


def figure_card(c: Ctx, noun=None, title=None):
    """A card whose heading names a figure and whose label restates it (R5), or narrows it (KEEP; the heading plus
    a qualifier is partial)."""
    noun = noun or c.noun()
    h3 = title if title is not None else S(noun.Pl, "h3")
    v = str(c.rng.randrange(2, 400))
    plain = h3.text == noun.Pl
    if h3.cls == "KEEP" and c.want(0.2):
        if plain and c.chance(0.6) or not plain and c.chance(0.8):
            lab = S(h3.text, "label").echo(h3, "R5", conf="clear")
        else:
            lab = S(c.pick([f"Total {noun.pl}", f"All {noun.pl}", f"Number of {noun.pl}", cap(c.alt(noun)[1])]), "label").echo(h3, "R5", conf="judgement")
    elif plain:
        lab = S(c.pick([f"{noun.Pl} this {c.pick(['week', 'month', 'season'])}", f"Active {noun.pl}", f"{noun.Pl} in {c.place}",
                        f"Overdue {noun.pl}", f"{noun.Pl} shared with you"]), "label")
        if lab.text.startswith(noun.Pl + " "):
            lab.keep(partial=True, conf="judgement")
    else:
        lab = S(c.pick(["This week", "Since Monday", "Last 30 days", f"In {c.place}", "Open now", "Today", "This month", "Year to date"]), "label")
    return E("card", E("h", s=h3, level=3), E("stats", E("stat", label=lab, value=S(v, "stat"), note=None)), shadcn=False)


def figure_grid(c: Ctx, page_h, noun: Noun):
    """Three or more figure cards; one card title may repeat the page title (R9, a title inside a grid)."""
    pool = [f"{noun.Pl} this week", f"Overdue {noun.pl}", f"Shared {noun.pl}", f"{noun.Pl} in {c.place}", f"Active {noun.pl}",
            f"New {noun.pl}", f"Archived {noun.pl}", f"Open {noun.pl}"]
    pool = [t for t in pool if page_h is None or t != page_h.text]
    titles = [S(t, "h3") for t in c.rng.some(pool, c.pick([3, 3, 4]))]
    plain = page_h is not None and page_h.text == noun.Pl
    for t in titles:
        if plain and t.text.startswith(noun.Pl + " "):
            t.keep(partial=True, conf="judgement")
    if page_h is not None and page_h.cls == "KEEP" and c.want(0.35):
        titles[c.rng.randrange(len(titles))] = S(page_h.text, "h3").echo(page_h, "R9", conf="clear")
    return E("grid", *[figure_card(c, noun, t) for t in titles])


def next_card(c: Ctx):
    """A field-style kicker over a value (K1, judgement when styled as a kicker)."""
    t = c.f(c.pick(L.FIELD_KICKERS))
    names = item_names(c)
    if t.startswith("Your plan"):
        val = f"{c.pick(L.PLANS[1:6])}, {L.money(c.rng, 4, 40)} a month"
    elif t in ("Owner", "Assigned to"):
        val = L.person(c.rng)
    elif t == "Location":
        val = c.pick(L.PLACES)
    elif t.startswith("Your " + c.group) or t == f"Your {c.group}":
        val = f"{c.place} {c.group}"
    else:
        val = f"{c.pick(names)}, {c.pick(L.DAYS)} {L.clock(c.rng)}"
    kick = S(t, "eyebrow").keep("K1", conf="judgement")
    h = S(val, "h3")
    kids = [E("eyebrow", s=kick), E("h", s=h, level=3)]
    if c.chance(0.6):
        if t.startswith("Your plan") or t == "Renews":
            line = c.pick(["Renews on {date}.", "Billed to the card ending {n}4.", "Moving to another plan is pro-rated to the day."])
        elif t in ("Owner", "Assigned to"):
            line = c.pick(["Since {date}.", "Ask them before changing it.", "Hand over from the menu."])
        else:
            line = c.pick(["{person} is leading.", "Reminder sent {n} days before.", "Changes allowed until {clock} the day before.",
                           "{n} people are going.", "Meet at {place}."])
        kids.append(E("p", s=S(c.f(line), "body")))
    return E("card", *kids)


def dashboard(c: Ctx) -> Page:
    rng = c.rng
    key = c.pick(list(DASH_HEADS))
    htext = c.f(key)
    syn = DASH_HEADS[key]
    h1 = S(htext, "h1")
    kids = []
    if c.want(0.2):
        same_kind = genre_group(htext)
        if same_kind and c.chance(0.5):
            kids.append(E("eyebrow", s=S(c.pick([w for w in same_kind if w != htext]), "eyebrow").echo(h1, "R1", conf="clear")))
        else:
            kids.append(E("eyebrow", s=S(c.pick(syn), "eyebrow").echo(h1, "R1", conf="judgement")))
    elif c.chance(0.25):
        kids.append(E("eyebrow", s=S(c.pick([f"Week {rng.randrange(1, 52)} · {c.pick(L.MONTHS)} 2026", f"{c.pick(L.DAYS)} {L.date(rng)}",
                                             f"Q{rng.randrange(1, 5)} 2026", f"{c.pick(L.MONTHS)} 2026"]), "eyebrow").keep("K1")))
    elif c.want(0.06):
        kids.append(E("eyebrow", s=S(c.pick(L.meta_strip(rng)), "eyebrow").decor("R2")))
    kids.append(E("h", s=h1, level=1))
    stats_n = c.pick([2, 3, 4])
    first_fact = None
    if c.want(0.12):
        kids.append(E("sub", s=S(c.pick(["Here are your {n} key numbers.", "The {n} cards below cover the basics.",
                                         "Your {n} main figures are shown below.", "{N} figures, below."])
                                 .format(n=L.NUMBER_WORDS[stats_n], N=cap(L.NUMBER_WORDS[stats_n])), "subtitle").narrate("R16", None)))
    elif c.chance(0.08):
        kids.append(E("sub", s=count_keep(c, stats_n, "dash")))
    elif c.want(0.2):
        if shares_stem("overview dashboard glance", htext) and c.chance(0.5):
            t = c.pick(["An overview of everything.", "Your dashboard, at a glance.", "An at-a-glance summary.", "Everything, at a glance."])
            kids.append(E("sub", s=S(t, "subtitle").echo(h1, "R3", conf="judgement")))
        else:
            lead = c.pick(['A summary of', 'A look at', 'Everything about', 'How things stand with'])
            obj = c.pick(['your ' + c.x.pl, 'your ' + c.group, 'the ' + c.x.pl] + ([] if lead == 'Everything about' else ['everything', 'it all']))
            t = f"{lead} {obj}{c.pick(['.', ', at a glance.', ' in one place.'])}"
            kids.append(E("sub", s=S(t, "subtitle").echo(h1, "R3", conf="judgement")))
    elif c.chance(0.5):
        first_fact = S(c.f(c.pick(["Figures update every {n} minutes.", "Numbers exclude test data.", "Times are shown in {zone}.",
                                   "Totals include {y} still in draft.", "Counts reset on the 1st of each month."])), "subtitle")
        kids.append(E("sub", s=first_fact))
    body = [E("head", *kids)]
    if c.chance(0.5):
        body.append(E("tabs", items=[S(t, "tab") for t in c.pick(L.RANGE_SETS)]))
    stats, labels, values = stats_row(c, stats_n)
    body.append(E("stats", *stats.kids))
    if c.chance(0.3):
        body.append(next_card(c))
    if c.chance(0.2):
        body.append(detail_card(c))
    if c.chance(0.3):
        if c.chance(0.5):
            body.append(figure_grid(c, None, c.noun()))
        else:
            body.append(E("grid", figure_card(c), *([figure_card(c)] if c.chance(0.4) else [])))
    # the same figure repeated in a later card (R14), or a different figure with the same value (KEEP)
    if c.want(0.16):
        j = rng.randrange(len(labels))
        lab2 = S(labels[j].text, "label").echo(labels[j], "R14", conf="clear")
        val2 = S(values[j].text, "stat").echo(values[j], "R14")
        body.append(E("section", E("h", s=S(c.pick(["Summary", "This period", "Key figure", "Highlights", "Worth knowing",
                                                     "Elsewhere", "One more figure"]), "h2"), level=2),
                      E("stats", E("stat", value=val2, label=lab2, note=None))))
    elif c.chance(0.16):
        # a different figure in a later card (KEEP), the same shape as the repeated one above
        j = rng.randrange(len(labels))
        others = [s for s, _ in c.dom["stats"] if s not in [x.text for x in labels]]
        if others:
            lab_o = c.pick(others)
            v_o = jitter(c.rng, dict(c.dom["stats"])[lab_o]) if c.chance(0.6) else values[j].text
            body.append(E("section", E("h", s=S(c.pick(["Also this week", "Elsewhere", "One more figure", "Worth knowing",
                                                         "Last period", "Compared with last year"]), "h2"), level=2),
                          E("stats", E("stat", value=S(v_o, "stat"), label=S(lab_o, "label"), note=None))))
    # a panel
    if c.chance(0.5):
        panel = chart_panel(c)
        if c.want(0.12):
            panel.kids.append(E("badge", s=S(static_status(c), "badge").decor("R11")))
        body.append(panel)
    else:
        panel = []
        if c.want(0.1):
            ptitle = S(htext, "h2").echo(h1, "R9")
        else:
            ptitle = S(c.pick([f"Recent {c.y.pl}", f"Latest {c.x.pl}", "Needs attention", f"{c.y.Pl} due soon", f"Busiest {c.x.pl}"]), "h2")
        panel.append(E("h", s=ptitle, level=2))
        panel.append(data_table(c, n_rows=c.pick([3, 4, 5])))
        body.append(E("section", *panel))
    tail = []
    if c.fw != "html" and c.chance(0.35):
        tail.append(E("status", s=S(c.pick(L.STATUS_WORDS), "body").keep("K2"), bound=c.prop("online") if c.fw == "tsx" else "isOnline"))
    elif c.want(0.1):
        tail.append(E("badge", s=S(static_status(c), "badge").decor("R11")))
    if first_fact is not None and c.want(0.25):
        tail.append(E("note", s=S(first_fact.text, "caption").echo(first_fact, "R14")))
    elif c.fw == "tsx" and c.chance(0.3):
        pre, post = c.pick([("Last updated ", "."), ("Updated ", ""), ("Synced ", " ago"), ("As of ", ""), ("Data from ", ".")])
        tail.append(E("dyn", s=S(f"{pre}VAR updated at{post}", "caption"), pre=pre, expr="updatedAt", post=post, look="note"))
    if tail:
        body.append(E("section", *tail))
    name = c.pick(["Overview", "HomeScreen", "Summary", f"{ident_words(c.product)}Home", "StatsPage", f"{ident_words(c.group)}Overview"])
    return Page(name, "dashboard", body, product=c.product, props=c.props)


# ---------------------------------------------------------------------------------------------------------------
# game menus and HUDs

GAME_SETTINGS = {
    "Video": [("Field of view", "Wider views show more but can cause motion sickness on large screens.", "How much you can see."),
              ("Motion blur", "Turning it off can help if you feel sick while turning quickly.", "Smearing when moving fast."),
              ("Brightness", "Set it so the symbol on the left is only just visible.", "How light the picture is."),
              ("Frame rate limit", "Lower limits save battery on handhelds.", "Cap on frames per second.")],
    "Audio": [("Master volume", "Voice chat has its own slider.", "Overall loudness."),
              ("Music", "Menus keep their music even at zero.", "Soundtrack level."),
              ("Voice chat", "Push-to-talk is on by default; change the key under Controls.", "Talking to your squad.")],
    "Controls": [("Mouse sensitivity", "Applies to aiming only; menus use your system setting.", "How fast the cursor turns."),
                 ("Invert Y axis", "Applies to mouse and controller.", "Flip up and down."),
                 ("Vibration", "Controllers only; does nothing on keyboard.", "Rumble feedback.")],
    "Gameplay": [("Difficulty", "Can be changed at any time; achievements need Normal or higher.", "How hard it is."),
                 ("Auto-save", "Saves every {n} minutes and at each checkpoint.", "Keeps progress automatically."),
                 ("Hints", "Shows a hint after {n} minutes stuck on a puzzle.", "Little nudges.")],
    "Accessibility": [("Subtitles", "Includes speaker names and important sound effects.", "Words on screen for speech."),
                      ("Colour-blind mode", "Changes enemy outlines and objective markers only.", "Palette for colour vision."),
                      ("Hold to press", "Turns rapid button presses into a single hold.", "Fewer repeated presses.")],
}
OVERLAY_EB = {"PAUSED": ["{M} ON HOLD", "{M} SUSPENDED", "TIME STOPPED", "SIMULATION FROZEN", "ON HOLD", "FROZEN IN PLACE", "BREAK"],
              "GAME OVER": ["RUN ENDED", "THE END", "DEFEATED", "ALL LIVES LOST", "{M} FAILED", "OUT OF LIVES"],
              "VICTORY": ["YOU WON", "{M} ACCOMPLISHED", "STAGE CLEARED", "SUCCESS", "{M} COMPLETE", "WELL DONE"],
              "LOADING": ["PLEASE WAIT", "STAND BY", "ONE MOMENT", "HOLD TIGHT", "NEARLY THERE"]}
OVERLAY_KEEP = {"PAUSED": ["Autosaved at the last checkpoint, {n} minutes ago.", "Progress since the last checkpoint is not saved.",
                           "Online matches keep running while you are paused.", "Your squad has been told you are away."],
                "GAME OVER": ["You reached wave {n}, two short of your best.", "Coins collected this run are kept.",
                              "Restarting resets this level’s timer but keeps your upgrades.", "{n} lives restore at {clock}."],
                "VICTORY": ["Three stars unlocks the bonus route on the map.", "Your time of {d}:{n:02d} is a new personal best.",
                            "Rewards not claimed now are lost when you quit.", "{n} crew levelled up."],
                "LOADING": ["Estimated wait {n}0 seconds, {d} players in your region.", "Cross-play is on: you may meet console players.",
                            "Leaving the queue three times in an hour adds a {n}-minute wait."]}


def game(c: Ctx) -> Page:
    g = c.pick(T.GAMES)
    c.product = g["title"]
    r = c.rng.random()
    if r < 0.45:
        return game_overlay(c, g)
    if r < 0.6:
        return game_main(c, g)
    if r < 0.8:
        return game_settings(c, g)
    return game_hud(c, g)


def _keep_sub_ct(text: str) -> bool:
    return any(w in text.lower() for w in ("resum", "restart", "leaving", "claimed", "quit", "paused", "queue", "squad"))


def game_overlay(c: Ctx, g) -> Page:
    st = c.pick(T.GAME_STATES)
    key = st["head"][0].upper()
    caps = c.chance(0.6)
    up = (lambda s: s.upper()) if caps else (lambda s: s[:1] + s[1:].lower() if s.isupper() else s)
    head_text = c.pick(st["head"])
    head_text = head_text.upper() if caps else (head_text.capitalize() if head_text.isupper() else head_text)
    h1 = S(head_text, "h1")
    kids = []
    mission = g["mission"]
    if c.want(0.4):
        ebt = c.pick(OVERLAY_EB[key] + st["eyebrow_echo"]).format(M=mission.upper())
        kids.append(E("eyebrow", s=S(up(ebt), "eyebrow").echo(h1, "R1", conf="judgement")))
    elif c.chance(0.3):
        kids.append(E("eyebrow", s=S(up(c.pick([g["title"], f"Chapter {c.rng.randrange(2, 9)} · {c.pick(['The docks', 'Low tide', 'Night run', 'The orchard'])}",
                                                f"{cap(mission)} {c.rng.randrange(2, 30)}", f"Act {c.pick(['I', 'II', 'III', 'IV'])}"])),
                                     "eyebrow").keep("K1")))
    elif c.want(0.08):
        kids.append(E("eyebrow", s=S(L.hud_meta(c.rng), "eyebrow").decor("R2")))
    kids.append(E("h", s=h1, level=1))
    action = S(up(c.pick(st["action"])), "button").keep("K11")
    menu = [S(up(x), "button") for x in c.rng.some(T.GAME_MENU, c.pick([1, 2, 3]))]
    buttons = [action] + menu
    sub = None
    if c.want(0.35):
        if c.chance(0.5):
            line = c.pick(st["narrate"])
        else:
            line = narrate_control(c, action.text.capitalize() if action.text.isupper() else action.text)
        sub = S(line, "subtitle").narrate("R6", action, ct=True, conf="judgement")
    elif c.want(0.2):
        sub = S(c.pick(st["sub_echo"]), "subtitle").echo(h1, "R3", conf="judgement")
    elif c.want(0.08):
        sub = S(c.pick(["Choose one of the {n} options below.", "Pick from the {n} options below.", "{N} choices, below."]).format(
            n=L.NUMBER_WORDS[len(buttons)], N=cap(L.NUMBER_WORDS[len(buttons)])), "subtitle").narrate("R16", None)
    elif c.chance(0.55):
        t = c.f(c.pick(OVERLAY_KEEP[key]))
        ct = _keep_sub_ct(t)
        sub = S(t, "subtitle").keep("K4" if ct else None, ct=ct, conf="judgement" if ct else "clear")
    if sub is not None:
        kids.append(E("sub", s=sub))
    body = [E("head", *kids), E("actions", *[E("button", s=b, primary=i == 0) for i, b in enumerate(buttons)])]
    if c.chance(0.2):
        body.append(E("note", s=S(c.pick(T.GAME_HINTS), "caption").keep("K4", ct=True)))
    if c.chance(0.2):
        body.append(keep_meta_el(c))
    return Page(c.pick(["PauseMenu", "OverlayScreen", "EndScreen", "StateOverlay", "RunOverPanel", "QueuePanel"]), "game", body,
                product=g["title"])


def game_main(c: Ctx, g) -> Page:
    h1 = S(g["title"].upper() if c.chance(0.5) else g["title"], "h1")
    kids = []
    if c.chance(0.4):
        kids.append(E("eyebrow", s=S(c.pick([f"Season {c.rng.randrange(2, 7)}", "Early access", "Director’s cut", "Chapter one",
                                             f"Update {c.rng.randrange(1, 4)}.{c.rng.randrange(0, 9)}"]), "eyebrow").keep("K1")))
    kids.append(E("h", s=h1, level=1))
    if c.chance(0.4):
        kids.append(E("sub", s=S(c.f(c.pick(["Your last save: {place}, {n} hours ago.", "Cross-play is on.",
                                             "Server maintenance on {day} from 06:00 to 08:00 UK time.", "{n} friends online."])), "subtitle")))
    items = c.rng.some(T.GAME_MAIN, c.pick([3, 4, 5])) + c.rng.some(g["items"], c.pick([0, 1, 2]))
    buttons = [S(x, "button") for x in items]
    body = [E("head", *kids), E("actions", *[E("button", s=b, primary=i == 0) for i, b in enumerate(buttons)])]
    if c.want(0.15):
        body.append(E("note", s=S(c.pick(["Use the menu to navigate.", "Select an option to begin.", "Click any option to continue.",
                                          narrate_control(c, buttons[0].text)]), "caption").narrate("R6", None, ct=True, conf="judgement")))
    if c.chance(0.5):
        body.append(E("note", s=S(c.pick([f"v1.{c.rng.randrange(0, 9)}.{c.rng.randrange(0, 20)}", f"Build 2026.{c.rng.randrange(1, 12):02d}.{c.rng.randrange(1, 28):02d}",
                                          f"v0.{c.rng.randrange(5, 9)}.{c.rng.randrange(0, 9)} early access"]), "caption").keep()))
    if c.want(0.12):
        body.append(E("meta", items=[S(L.hud_meta(c.rng), "caption").decor("R2")]))
    elif c.chance(0.3):
        body.append(keep_meta_el(c))
    return Page(c.pick(["MainMenu", "TitleScreen", "StartMenu"]), "game", body, product=g["title"])


def keep_meta_el(c: Ctx):
    """A mono, upper-case line that states a version, build, season or in-game date (KEEP; K1 for dates)."""
    items = []
    for x in L.keep_meta(c.rng, game=True):
        dated = x.startswith(("SEASON", "DAY"))
        items.append(S(x, "caption").keep("K1", conf="judgement") if dated else S(x, "caption").keep())
    return E("meta", items=items)


def game_settings(c: Ctx, g) -> Page:
    swift = c.fw == "swift"
    htext = c.pick(["SETTINGS", "Settings", "OPTIONS", "Options"])
    h1 = S(htext, "h1")
    body = []
    nav_title = None
    if swift and c.chance(0.5):
        nav_title = S(htext.capitalize(), "h1")
        if c.want(0.4):
            body.append(E("section", E("h", s=S(htext.capitalize(), "h2").echo(nav_title, "R9"), level=2)))
    else:
        kids = []
        if c.want(0.2):
            kids.append(E("eyebrow", s=S(c.pick(["CONFIGURATION", "PREFERENCES", "SYSTEM", "SETUP"]) if htext.isupper() else
                                         c.pick(["Configuration", "Preferences", "Setup"]), "eyebrow").echo(h1, "R1", conf="judgement")))
        elif c.chance(0.2):
            kids.append(E("eyebrow", s=S(g["title"].upper() if htext.isupper() else g["title"], "eyebrow").keep("K1")))
        kids.append(E("h", s=h1, level=1))
        if c.want(0.15):
            kids.append(E("sub", s=S(c.pick(["Change the game settings.", "Adjust your options here.", "Configure the game.",
                                             "Tweak how it plays."]), "subtitle").echo(h1, "R3", conf="judgement")))
        body.append(E("head", *kids))
    tabs = c.rng.some(list(GAME_SETTINGS), c.pick([3, 4, 5]))
    body.append(E("tabs", items=[S(t, "tab") for t in tabs]))
    sec = c.pick(tabs)
    h2 = S(sec, "h2").keep("K11")  # the open tab's name over its panel
    rows = [E("h", s=h2, level=2)]
    for label, helper, alt in c.rng.some(GAME_SETTINGS[sec], c.pick([2, 3])):
        lab = S(label, "label")
        if c.want(0.15):
            if c.chance(0.5):
                hl = S(c.pick(["Adjust the {l}.", "Change your {l}.", "Set the {l}.", "Turn {l} up or down."]).format(l=lower_first(label)),
                       "helper").echo(lab, "R5")
            else:
                hl = S(alt, "helper").echo(lab, "R5", conf="judgement")
        elif c.chance(0.7):
            hl = S(c.f(helper), "helper").keep("K4")
        else:
            hl = None
        rows.append(E("toggle", label=lab, helper=hl))
    if c.chance(0.3):
        rows.append(E("note", s=S(c.pick(T.GAME_HINTS), "caption").keep("K4", ct=True)))
    body.append(E("section", *rows))
    done = S(c.pick(["APPLY", "Apply", "BACK", "Back", "Save and exit", "Confirm"]), "button")
    if c.want(0.1):
        body.append(E("note", s=S(c.pick([f"Press {done.text.capitalize()} to keep your changes.", narrate_control(c, done.text.capitalize())]),
                                  "caption").narrate("R6", done, ct=True)))
    body.append(E("actions", E("button", s=done)))
    if c.chance(0.25):
        body.append(keep_meta_el(c))
    return Page(c.pick(["OptionsMenu", "GameOptions", "OptionsScreen"]), "game", body, product=g["title"], nav_title=nav_title)


def game_hud(c: Ctx, g) -> Page:
    body = []
    stats = []
    for label, values in c.rng.some(T.HUD_LABELS, c.pick([2, 3, 4])):
        v = c.pick(values)
        if "/" in v:
            b = c.rng.randrange(2, 9) if v.split("/")[1].strip().isdigit() and int(v.split("/")[1]) < 10 else c.rng.randrange(30, 200)
            v = f"{c.rng.randrange(0, b + 1)} / {b}"
        else:
            v = jitter(c.rng, v)
        stats.append(E("stat", value=S(v, "stat"), label=S(label.upper() if c.chance(0.5) else label, "label")))
    body.append(E("stats", *stats))
    obj = S(c.f(c.pick(T.HUD_OBJECTIVES + ["Hold the bridge for {n} minutes", "Find {n} supply crates", "Reach {place} before dawn"])), "body")
    if c.chance(0.6):
        kick = c.pick(["Objective", "OBJECTIVE", "Current task", "Next", "Target", "Due"])
        # a field-style kicker that names the value after it (K1); judgement when styled as a kicker
        body.append(E("section", E("eyebrow", s=S(kick, "eyebrow").keep("K1", conf="judgement")), E("p", s=obj)))
    else:
        body.append(E("section", E("p", s=obj)))
    if c.chance(0.5):
        wave = c.pick(["Wave {a} of {b}", "Lap {a} / {b}", "Round {a} of {b}", "Stage {a}/{b}"])
        body.append(E("note", s=S(wave.format(a=c.pick([2, 3, 4]), b=c.pick([5, 8, 10])), "caption").keep("K3")))
    if c.fw != "html" and c.chance(0.4):
        body.append(E("status", s=S(c.pick(["Shield low", "Signal", "Squad connected", "Online", "Live"]), "body").keep("K2"),
                      bound=c.prop("health") if c.fw == "tsx" else "lowShield"))
    if c.want(0.25):
        body.append(E("meta", items=[S(L.hud_meta(c.rng), "caption").decor("R2")]))
    elif c.chance(0.3):
        body.append(keep_meta_el(c))
    if c.want(0.15):
        body.append(E("badge", s=S(c.pick(["● REC", "● LIVE", "◆ ONLINE", "● ONLINE", "LIVE"]), "badge").decor("R11")))
    if c.chance(0.3):
        body.append(E("note", s=S(c.pick(T.GAME_HINTS), "caption").keep("K4", ct=True)))
    return Page(c.pick(["Hud", "HeadsUp", "Overlay", "StatusBar"]), "game", body, product=g["title"], props=c.props)


COMPOSERS = {"marketing": marketing, "app": app, "settings": settings, "dialog": dialog, "empty": empty,
             "dashboard": dashboard, "game": game}


def k14(c: Ctx, role="body"):
    """Two branches of one conditional that read alike (K14), or None where the framework has no conditional."""
    if c.fw == "html":
        return None
    a, b = c.pick(T.K14_GAME if c.product in [g["title"] for g in T.GAMES] else T.K14_PAIRS)
    flag = c.prop(c.pick(["saved", "online", "paid", "verified", "selected"])) if c.fw == "tsx" else "isDone"
    kind = "note" if role == "caption" else "p"
    return E("cond", a=[E(kind, s=S(a, role))], b=[E(kind, s=S(b, role).keep("K14"))], expr=flag, swift=flag)


def link_stacks(page: Page):
    """Echoes that repeat the same node form a stack: each lists the other flagged copies in ``equiv``
    (LABELLING.md §5)."""
    groups: dict = {}
    for s in page.slots():
        if s.cls == "ECHO" and s.target is not None:
            groups.setdefault(id(s.target), []).append(s)
    for g in groups.values():
        for s in g:
            s.equiv = [o for o in g if o is not s]


def compose(rng, kind: str, fw: str, clean: bool) -> tuple[Page, Ctx]:
    c = Ctx(rng, fw, clean)
    page = COMPOSERS[kind](c)
    if kind in ("app", "dashboard", "settings", "game") and fw != "html" and rng.random() < 0.18:
        cond = k14(c, c.pick(["body", "caption"]))
        if cond is not None:
            page.body.append(E("section", cond) if fw == "swift" else cond)
    page.r19 = c.r19
    if c.props:
        page.props = c.props
    link_stacks(page)
    return page, c
