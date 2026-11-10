"""Screen composers for the dev generator: one per screen kind. Each builds an abstract tree (``core.E``) and
decides every slot's label as it goes, planting the scored patterns of LABELLING.md §9 with their exact class,
pattern and target, and the K rules of §6 as hard negatives (KEEP records naming the rule in ``pattern``).

Planting is local and by construction: a restating subtitle is written as a restatement of the heading it sits
under and points at it; copy that is not planted is drawn from tables of facts that each add information. A clean
screen (``Ctx.clean``) plants nothing redundant but keeps its hard negatives.
"""

from __future__ import annotations

from . import tables as T
from .core import E, Page, S, cap, ctx, dyn, lower_first, stack

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


class Ctx:
    def __init__(self, rng, fw: str, clean: bool):
        self.rng = rng
        self.fw = fw
        self.clean = clean
        self.dom = rng.pick(T.DOMAINS)
        self.product = rng.pick(self.dom["names"])
        self.r19: list = []
        self.props: list = []
        self.notice = None  # the first global notice (K6), for later repeats (R14)
        self.notice_alts: list = []
        self.cta_button = None  # the first call to action, for later repeats (R14)
        self.used: set = set()

    def want(self, p: float) -> bool:
        """Plant a redundant pattern with probability p (never on a clean screen)."""
        return (not self.clean) and self.rng.random() < p

    def chance(self, p: float) -> bool:
        return self.rng.random() < p

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

    @property
    def obj(self):
        return self.dom["obj"][0]

    @property
    def objs(self):
        return self.dom["obj"][1]


def _key(x):
    return x if isinstance(x, str) else repr(x)


def words(s: str) -> set:
    return {w.strip(".,:;!?’'—-").lower() for w in s.split()} - {""}


def shares_stem(a: str, b: str) -> bool:
    """A crude stem test for genre words (K1): any word of ``a`` whose first five letters start a word of ``b``."""
    wb = words(b)
    for w in words(a):
        k = w[:5] if len(w) > 4 else w
        if any(x.startswith(k) for x in wb):
            return True
    return False


# ---------------------------------------------------------------------------------------------------------------
# shared pieces


def eyebrow_for(c: Ctx, head_text: str, head_slot, concept=None, genre=None, allow_k1=True, p_r1=0.3, p_r2=0.15,
                genres=("Overview", "Products", "Services", "Get started")):
    """An eyebrow over a heading, or None. ``concept`` gives paraphrase eyebrows; ``genre`` is a section-genre word
    the heading contains; ``genres`` are the section-genre words that suit this kind of section (R2 when the heading
    does not contain them)."""
    if (concept or genre) and c.want(p_r1):
        text = genre if genre else c.rng.pick(concept)
        return S(text, "eyebrow").echo(head_slot, "R1", conf="clear" if genre else "judgement")
    if c.want(p_r2):
        cand = [w for w in genres if not shares_stem(w, head_text)]
        if cand and c.chance(0.6):
            return S(c.rng.pick(cand), "eyebrow").decor("R2")
        return S(c.rng.pick(T.NEXT_LABELS), "eyebrow").decor("R2")
    if allow_k1 and c.chance(0.35):
        kind = c.rng.pick(["cat", "product", "date"])
        if kind == "cat":
            text = c.fresh(c.dom["cats"])
        elif kind == "product":
            text = c.product
        else:
            text = c.rng.pick(["Issue 14", "Spring 2026", "June 2026", "Issue 3 · May 2026", "Release 4.2", "Autumn edition"])
        if shares_stem(text, head_text):
            return None
        return S(text, "eyebrow").keep("K1")
    return None


def notice(c: Ctx):
    n = c.rng.pick(T.NOTICES)
    s = S(n[0], "caption").keep("K6")
    c.notice, c.notice_alts = s, n[1]
    return E("note", s=s)


def cta_buttons(c: Ctx, first=True):
    ctas = c.dom["cta"]
    b = S(ctas[0] if first else c.rng.pick(ctas[1:]), "button")
    kids = [E("button", s=b)]
    if first:
        c.cta_button = b
        c.used.add(b.text)
    if c.chance(0.6):
        if c.chance(0.15):
            kids.append(E("link", s=S(c.rng.pick(T.VACUOUS), "link").keep(aside=["R7"]), arrow=c.chance(0.4)))
        else:
            kids.append(E("link", s=S(c.fresh([x for x in ctas if x != b.text]), "link"), arrow=c.chance(0.4)))
    return E("actions", *kids), b


def fact(c: Ctx) -> str:
    return c.fresh(c.dom["facts"])


H2_TEMPLATES = ["What {product} does", "Made for {objs}", "Everything a {obj} needs", "Why keepers of {objs} switch",
                "Less admin, more {objs}", "Built around your {objs}", "The tools behind every {obj}",
                "What you get with {product}", "A closer look at {product}"]


def fill(t: str, c: Ctx, **kw) -> str:
    vals = {"product": c.product, "obj": c.obj, "objs": c.objs, "Obj": cap(c.obj), "Objs": cap(c.objs),
            "obj2": c.dom["obj2"][0], "obj2s": c.dom["obj2"][1]}
    vals.update(kw)
    return t.format(**vals)


def count_line(c: Ctx, n: int, obj=None) -> str:
    t = c.rng.pick([x for x in T.LAYOUT_COUNT if ("{obj}" not in x) or obj])
    return t.format(n=T.NUMBER_WORDS.get(n, str(n)), N=cap(T.NUMBER_WORDS.get(n, str(n))), obj=obj or c.obj)


# ---------------------------------------------------------------------------------------------------------------
# marketing


def m_hero(c: Ctx):
    concept = c.rng.pick(T.HEAD_CONCEPTS) if c.chance(0.55) else None
    if concept:
        htext = concept[0]
    else:
        htext = fill(c.rng.pick(["Keep every {obj} on track", "{Objs}, sorted", "Run your {obj2s} from one screen",
                                 "The calm way to manage {objs}", "Your {objs}, without the paperwork",
                                 "Spend less time on {obj2s}"]), c)
    h1 = S(htext, "h1")
    kids = []
    top = []
    if c.want(0.07):
        top.append(E("meta", items=[S(x, "caption").decor("R2") for x in c.rng.pick(T.META_DECOR)]))
    if c.want(0.12):
        top.append(E("badge", s=S(c.rng.pick(["● Now in beta", "Live", "● Available now", "✨ New", "Open"]), "badge").decor("R11")))
    elif c.chance(0.06):
        top.append(E("badge", s=S("Early access", "badge").keep(conf="judgement")))
    eb = eyebrow_for(c, htext, h1, concept=concept[1] if concept else None, p_r1=0.35)
    if eb is None and c.chance(0.1):
        eb = S("New: " + lower_first(c.rng.pick(c.dom["features"])[0]), "eyebrow").keep()
    if eb:
        kids.append(E("eyebrow", s=eb))
    kids.append(E("h", s=h1, level=1))
    r = c.rng.random()
    if concept and c.want(0.3):
        kids.append(E("sub", s=S(c.rng.pick(concept[2]), "subtitle").echo(h1, "R3", conf="judgement")))
    elif c.want(0.06):
        kids.append(E("sub", s=S(c.rng.pick(["Scroll to explore.", "Scroll down to see more."]), "subtitle").narrate("R6", None, ct=True)))
    elif concept and c.chance(0.07):
        kids.append(E("sub", s=S(c.rng.pick(concept[2]) + " " + fact(c), "subtitle").keep(partial=True, conf="judgement")))
    elif r < 0.85:
        kids.append(E("sub", s=S(fact(c), "subtitle")))
    head = E("head", *kids)
    acts, _ = cta_buttons(c)
    body = top + [head, acts]
    if c.chance(0.4):
        body.append(notice(c))
    return E("section", *body, tone="hero")


def card_grid(c: Ctx, n: int, name="features"):
    feats = c.fresh_n(c.dom["features"], n)
    echo_at = c.rng.randrange(len(feats)) if c.want(0.3) else None
    device = c.rng.wpick([(None, 55), ("pill", 15), ("sibling", 10), ("index", 12), ("emoji", 8)])
    if device == "pill" and c.clean:
        device = None
    pill = c.rng.pick(T.REPEATED_PILLS)
    sib = c.fresh_n(T.SIBLING_PILLS, len(feats))
    style = c.rng.pick(T.INDEX_STYLES)
    emojis = c.rng.shuffled(["✨", "⚡", "🌱", "🔒", "📦", "🧭", "🔔", "📈"])
    cards, devs = [], []
    for i, (title, body) in enumerate(feats):
        kids = []
        if i == echo_at:
            ce = c.fresh(T.CARD_ECHO)
            t = S(ce[0], "h3")
            b = S(c.rng.pick(ce[1]), "body").echo(t, "R5", conf="clear")
        else:
            t = S(title, "h3")
            b = S(body, "body")
        if device == "index":
            d = S(style.format(n=i + 1, nn=f"{i + 1:02d}"), "eyebrow").keep("K3", aside=["R12"])
            kids.append(E("eyebrow", s=d))
            devs.append(d)
        elif device == "pill":
            d = S(pill, "badge").decor("R11")
            kids.append(E("badge", s=d))
            devs.append(d)
        elif device == "sibling" and i < len(sib):
            kids.append(E("badge", s=S(sib[i], "badge").keep("K2")))
        if device == "emoji":
            t.text = emojis[i % len(emojis)] + " " + t.text
            t.add_aside("R11g")
        kids.append(E("h", s=t, level=3))
        kids.append(E("p", s=b))
        cards.append(E("card", *kids))
    if device == "sibling" and len(sib) < len(feats):
        # sibling pills must cover every card for the grid to stay uniform
        for cd in cards:
            cd.kids = [k for k in cd.kids if k.kind != "badge"]
    if len(devs) >= 3:
        c.r19.append(devs)
    return E("grid", *cards, mappable=True, name=name), len(cards)


def m_features(c: Ctx):
    concept = genre = None
    r = c.rng.random()
    if r < 0.35:
        concept = c.fresh(T.HEAD_CONCEPTS)
        htext = concept[0]
    elif r < 0.55:
        g = c.rng.pick([x for x in T.GENRE_IN_HEADING if x[0] in ("Features", "Capabilities", "Overview", "Why us", "Services", "Products")])
        genre, htext = g[0], c.rng.pick(g[1])
    else:
        htext = fill(c.rng.pick(H2_TEMPLATES), c)
    h2 = S(htext, "h2")
    kids = []
    eb = eyebrow_for(c, htext, h2, concept=concept[1] if concept else None, genre=genre, p_r1=0.4,
                     genres=("Features", "Capabilities", "Why us", "Overview"))
    if eb:
        kids.append(E("eyebrow", s=eb))
    kids.append(E("h", s=h2, level=2))
    n = c.rng.wpick([(2, 1), (3, 5), (4, 3), (5, 1), (6, 2)])
    sub_kind = c.rng.random()
    grid, n = card_grid(c, n)
    if c.want(0.12):
        kids.append(E("sub", s=S(count_line(c, n), "subtitle").narrate("R16", None)))
    elif concept and c.want(0.3):
        kids.append(E("sub", s=S(c.rng.pick(concept[2]), "subtitle").echo(h2, "R3", conf="judgement")))
    elif sub_kind < 0.55:
        kids.append(E("sub", s=S(fact(c), "subtitle")))
    out = [E("head", *kids), grid]
    if c.want(0.3):
        if c.chance(0.5):
            line = S(c.rng.pick(T.CLOSING_SPREAD).format(product=c.product), "body").echo(None, "R17", conf="judgement")
        else:
            line = S(c.rng.pick(T.CLOSING_HEAD).format(h=lower_first(htext)), "body").echo(h2, "R17")
        out.append(E("p", s=line))
    elif c.chance(0.15):
        out.append(E("p", s=S(fact(c), "body")))
    return E("section", *out)


STEP_POOL = [
    ("Create your account", "It takes a minute with an email address and a password."),
    ("Add your first {obj}", "Most people start with the one they check most often."),
    ("Invite the people you work with", "They get an email link that stays valid for seven days."),
    ("Check your {obj2s} anywhere", "The app runs on iPhone, Android and the web."),
    ("Import what you already have", "Spreadsheets with a header row come in with their columns matched."),
    ("Set your reminders", "Choose the day and time; reminders follow your time zone."),
]


def m_steps(c: Ctx):
    r = c.rng.random()
    genre = None
    if r < 0.3:
        genre, htext = "How it works", c.rng.pick(dict(T.GENRE_IN_HEADING)["How it works"])
    else:
        htext = fill(c.rng.pick(["From sign-up to your first {obj}", "Getting going takes an afternoon",
                                 "Set up {product} in an afternoon", "Your first week with {product}"]), c)
    h2 = S(htext, "h2")
    kids = []
    eb = eyebrow_for(c, htext, h2, genre=genre, allow_k1=False, p_r1=0.5, p_r2=0.25, genres=("How it works", "Get started"))
    if eb:
        kids.append(E("eyebrow", s=eb))
    kids.append(E("h", s=h2, level=2))
    n = c.rng.pick([3, 3, 4])
    picks = c.rng.some(STEP_POOL, n)
    if c.want(0.2):
        kids.append(E("sub", s=S(count_line(c, n), "subtitle").narrate("R16", None)))
    elif c.chance(0.4):
        kids.append(E("sub", s=S(fact(c), "subtitle")))
    style = c.rng.pick(T.STEP_STYLES)
    numbered = c.chance(0.85)
    items = []
    for i, (t, b) in enumerate(picks):
        num = None
        if numbered:
            num = S(style.format(n=i + 1, total=n), "caption").keep("K3")
        items.append((num, S(fill(t, c), "h3"), S(fill(b, c), "body") if c.chance(0.8) else None))
    out = [E("head", *kids), E("steps", items=items)]
    if c.want(0.25):
        out.append(E("p", s=S(c.rng.pick(["That’s all it takes to get started.", "And that’s it: you’re set up.",
                                         "Three simple steps, and you’re ready."]) if n == 3 else
                                "That’s all it takes to get started.", "body").echo(None, "R17", conf="judgement")))
    return E("section", *out)


TIERS = [("Free", "£0"), ("Starter", "£6"), ("Plus", "£12"), ("Pro", "£19"), ("Team", "£29"), ("Studio", "£49")]
TIER_BLURBS = ["For one person trying it out.", "For a small team that meets weekly.", "For busy teams with several sites.",
               "For organisations that need an audit trail.", "For clubs and community groups."]
TIER_POINTS = ["Up to 4 {objs}", "Unlimited {objs}", "CSV and PDF export", "Priority email support", "5 team members",
               "Unlimited team members", "Two-step sign-in for everyone", "Daily backups kept for 90 days"]


def m_pricing(c: Ctx):
    r = c.rng.random()
    concept = genre = None
    if r < 0.3:
        concept = T.HEAD_CONCEPTS[5]
        htext = concept[0]
    elif r < 0.55:
        genre, htext = "Pricing", c.rng.pick(dict(T.GENRE_IN_HEADING)["Pricing"])
    else:
        htext = fill(c.rng.pick(["Plans for every size of {obj} list", "Pay for what you use", "One price per month, no extras"]), c)
    h2 = S(htext, "h2")
    kids = []
    eb = eyebrow_for(c, htext, h2, concept=concept[1] if concept else None, genre=genre, allow_k1=False, p_r1=0.45, p_r2=0.3,
                     genres=("Pricing",))
    if eb:
        kids.append(E("eyebrow", s=eb))
    kids.append(E("h", s=h2, level=2))
    n = c.rng.pick([2, 3, 3, 4])
    if c.want(0.2):
        kids.append(E("sub", s=S(c.rng.pick(["Below are our {n} plans.", "Choose from {n} plans below.",
                                             "Here are our {n} plans, side by side."]).format(n=T.NUMBER_WORDS[n]),
                                 "subtitle").narrate("R16", None)))
    elif concept and c.want(0.3):
        kids.append(E("sub", s=S(c.rng.pick(concept[2]), "subtitle").echo(h2, "R3", conf="judgement")))
    elif c.chance(0.5):
        kids.append(E("sub", s=S(c.rng.pick(["Prices include VAT.", "Every plan includes unlimited storage.",
                                             "Switch plans whenever you like; we pro-rate the difference."]), "subtitle")))
    start = c.rng.randrange(0, len(TIERS) - n + 1)
    tiers = TIERS[start:start + n]
    period = c.rng.pick(["per month", "/month", "a month, billed yearly", "per user per month"])
    popular = c.rng.randrange(n) if c.chance(0.5) else None
    cards = []
    blurbs = c.rng.some(TIER_BLURBS, n)
    for i, (name, price) in enumerate(tiers):
        k = []
        if i == popular:
            k.append(E("badge", s=S("Most popular", "badge").keep("K2")))
        k.append(E("h", s=S(name, "h3"), level=3))
        k.append(E("sub", s=S(blurbs[i % len(blurbs)], "subtitle"), card=True))
        k.append(E("price", amount=S(price, "stat"), period=S(period, "caption").keep("K13")))
        pts = c.rng.some(TIER_POINTS, 2)
        k.append(E("ul", items=[S(fill(p, c), "list_item") for p in pts], mark="check"))
        k.append(E("button", s=S(f"Choose {name}", "button"), primary=i == popular))
        cards.append(E("card", *k))
    out = [E("head", *kids), E("grid", *cards)]
    if c.notice is not None and c.want(0.5):
        out.append(E("note", s=S(c.rng.pick(c.notice_alts), "caption").echo(c.notice, "R14")))
    elif c.notice is None and c.chance(0.4):
        out.append(notice(c))
    return E("section", *out)


FAQ_POOL = [
    ("Can I cancel?", "Yes. Cancel from the billing page and you keep access until the end of the month."),
    ("Is there a free trial?", "The free plan has no time limit; paid plans start with 14 days free."),
    ("Where is my data stored?", "In data centres in London and Dublin."),
    ("Can I export everything?", "Yes, as CSV or PDF, at any time, including after you cancel."),
    ("Do you offer discounts for charities?", "Registered charities get 50% off any paid plan."),
    ("Does it work on Android?", "Yes, on Android 11 and later, and in any modern browser."),
    ("Can several people share an account?", "Each person gets their own login; the Team plan has no seat limit."),
]


def m_faq(c: Ctx):
    r = c.rng.random()
    concept = genre = None
    if r < 0.4:
        concept = T.HEAD_CONCEPTS[13]
        htext = concept[0]
    elif r < 0.6:
        genre, htext = "FAQ", c.rng.pick(dict(T.GENRE_IN_HEADING)["FAQ"])
    else:
        htext = c.rng.pick(["Before you sign up", "Things people ask us", "The small print, in plain English"])
    h2 = S(htext, "h2")
    kids = []
    eb = eyebrow_for(c, htext, h2, concept=concept[1] if concept else None, genre=genre, allow_k1=False, p_r1=0.45, p_r2=0.3,
                     genres=("FAQ",))
    if eb:
        kids.append(E("eyebrow", s=eb))
    kids.append(E("h", s=h2, level=2))
    if concept and c.want(0.3):
        kids.append(E("sub", s=S(c.rng.pick(concept[2]), "subtitle").echo(h2, "R3", conf="judgement")))
    items = [(S(q, "button"), S(a, "body")) for q, a in c.rng.some(FAQ_POOL, c.rng.pick([3, 4, 5]))]
    out = [E("head", *kids), E("faq", items=items)]
    if c.want(0.2):
        out.append(E("p", s=S(c.rng.pick(T.CLOSING_FAQ), "body").echo(None, "R17", conf="judgement")))
    return E("section", *out)


QUOTES = ["We moved every {obj} over in an afternoon and haven’t looked back.",
          "The reminders alone save me an hour a week.", "Our volunteers picked it up without any training.",
          "It replaced three spreadsheets and a whiteboard.", "Support answered on a Sunday, which I did not expect."]
PEOPLE = ["Priya Shah, operations lead", "Tom Evans, owner", "Mei Lin, club secretary", "Rhys Jones, volunteer coordinator",
          "Ana Costa, studio manager", "Kofi Mensah, head of finance"]


def m_quotes(c: Ctx):
    concept = genre = None
    if c.chance(0.4):
        concept = T.HEAD_CONCEPTS[6]
        htext = concept[0]
    elif c.chance(0.4):
        genre, htext = "Testimonials", c.rng.pick(dict(T.GENRE_IN_HEADING)["Testimonials"])
    else:
        htext = fill(c.rng.pick(["What {product} users tell us", "In their words", "From people who use it daily"]), c)
    h2 = S(htext, "h2")
    kids = []
    eb = eyebrow_for(c, htext, h2, concept=concept[1] if concept else None, genre=genre, allow_k1=False, p_r1=0.45, p_r2=0.3,
                     genres=("Testimonials",))
    if eb:
        kids.append(E("eyebrow", s=eb))
    kids.append(E("h", s=h2, level=2))
    if concept and c.want(0.35):
        kids.append(E("sub", s=S(c.rng.pick(concept[2]), "subtitle").echo(h2, "R3", conf="judgement")))
    n = c.rng.pick([2, 3])
    qs = [E("quote", s=S(fill(q, c), "body"), who=S(p, "caption")) for q, p in zip(c.rng.some(QUOTES, n), c.rng.some(PEOPLE, n))]
    out = [E("head", *kids), E("grid", *qs)]
    if c.want(0.2):
        out.append(E("p", s=S(c.rng.pick(T.CLOSING_HEAD).format(h=lower_first(htext)), "body").echo(h2, "R17")))
    return E("section", *out)


def m_cta(c: Ctx):
    htext = fill(c.rng.pick(["Ready when you are", "See it for yourself", "Your {objs} will thank you",
                             "Give it a week", "Try it on your next {obj}"]), c)
    h2 = S(htext, "h2")
    kids = [E("h", s=h2, level=2)]
    if c.chance(0.6):
        kids.append(E("sub", s=S(fact(c), "subtitle")))
    out = [E("head", *kids)]
    if c.cta_button is not None and c.want(0.45):
        b = S(c.cta_button.text, "button").echo(c.cta_button, "R14")
        out.append(E("actions", E("button", s=b)))
    else:
        rest = [x for x in c.dom["cta"] + T.GENERIC_CTA if x not in c.used]
        out.append(E("actions", E("button", s=S(c.fresh(rest), "button"))))
    if c.notice is not None and c.want(0.45):
        out.append(E("note", s=S(c.rng.pick(c.notice_alts), "caption").echo(c.notice, "R14")))
    elif c.notice is None and c.chance(0.35):
        out.append(notice(c))
    return E("section", *out, tone="cta")


def m_footer(c: Ctx, nav_items):
    cols = []
    for head, links in c.rng.some(T.FOOTER_COLS, c.rng.pick([2, 3, 4])):
        ss = []
        for x in links[: c.rng.pick([2, 3, 4])]:
            s = S(x, "link")
            if x in nav_items:
                s.keep("K5")
            ss.append(s)
        cols.append((S(head, "h4"), ss))
    legal = S(c.rng.pick(T.LEGAL).format(product=c.product), "caption")
    out = [E("footer", cols=cols, legal=legal)]
    if c.fw != "swift" and c.want(0.1):
        out.append(E("status", s=S(c.rng.pick(["All systems normal", "All systems operational", "Online"]), "body").decor("R11")))
    if c.want(0.06):
        out.append(E("meta", items=[S(x, "caption").decor("R2") for x in c.rng.pick(T.META_DECOR)]))
    return out


def marketing(c: Ctx) -> Page:
    rng = c.rng
    swift = c.fw == "swift"
    pool = ["features", "steps", "pricing", "faq", "quotes", "cta"]
    if swift:
        pool = ["features", "steps", "cta"]
    n = rng.wpick([(1, 40), (2, 28), (3, 22), (4, 10)])
    body = []
    nav_items = []
    if n == 1:
        kinds = [rng.pick(["hero"] * 3 + pool)]
    else:
        kinds = ["hero"] + rng.some([p for p in pool if p != "cta"], n - 1)
        if n >= 3 and rng.chance(0.5):
            kinds[-1] = "cta"
    if n >= 2 and not swift and rng.chance(0.4):
        nav_items = rng.some(T.NAV_ITEMS, rng.pick([3, 4, 5]))
        body.append(E("nav", items=[S(c.product, "link")] + [S(x, "nav") for x in nav_items]))
        c.used.update(nav_items)
    for k in kinds:
        body.append({"hero": m_hero, "features": m_features, "steps": m_steps, "pricing": m_pricing, "faq": m_faq,
                     "quotes": m_quotes, "cta": m_cta}[k](c))
    if n >= 2 and not swift and rng.chance(0.35):
        body.extend(m_footer(c, nav_items))
    # one eyebrow device on most sections (screen aside R19)
    ebs = [e["s"] for top in body for e in top.walk() if e.kind == "eyebrow"]
    if len(ebs) >= 3 and len(ebs) >= len(kinds) - 1:
        c.r19.append(ebs)
    name = {1: kinds[0].capitalize() + "Section"}.get(n, "LandingPage")
    if kinds[0] == "hero" and n == 1:
        name = "Hero"
    return Page(name, "marketing", body, product=c.product)


# ---------------------------------------------------------------------------------------------------------------
# app pages


def page_header(c: Ctx, htext: str, synonyms, noun: str, level=1, allow_r16=True):
    """Eyebrow, heading and subtitle for an app page."""
    h = S(htext, "h1" if level == 1 else f"h{level}")
    kids = []
    eb = None
    if c.want(0.22) and synonyms:
        eb = S(c.rng.pick(synonyms), "eyebrow").echo(h, "R1", conf="judgement")
    elif c.chance(0.15):
        eb = S(c.rng.pick([c.product, c.fresh(c.dom["cats"])]), "eyebrow").keep("K1")
        if shares_stem(eb.text, htext):
            eb = None
    if eb:
        kids.append(E("eyebrow", s=eb))
    kids.append(E("h", s=h, level=level))
    sub = None
    r = c.rng.random()
    if c.want(0.28):
        sub = S(c.rng.pick(T.SUB_ECHO).format(n=noun), "subtitle").echo(h, "R3")
    elif allow_r16 and c.want(0.1):
        sub = S(c.rng.pick(T.LAYOUT_PLAIN).format(obj=c.obj, objs=c.objs), "subtitle").narrate("R16", None)
    elif c.chance(0.06):
        sub = S(c.rng.pick(T.SUB_PARTIAL).format(n=noun), "subtitle").keep(partial=True)
    elif r < 0.6:
        sub = S(fact(c), "subtitle")
    if sub is not None:
        kids.append(E("sub", s=sub))
    # a third copy of the same name: a stack (equiv)
    if eb is not None and eb.cls == "ECHO" and sub is not None and sub.cls == "ECHO":
        stack(h, [eb, sub], ["R1", "R3"], conf="judgement")
    return E("head", *kids), h, sub


def data_table(c: Ctx, n_rows=None):
    cols, pools = c.dom["table"]
    n_rows = min(n_rows or c.rng.pick([2, 3, 4, 5]), len(pools[0]))
    headers = [S(x, "label") for x in cols]
    rows = []
    seen = [set() for _ in cols]
    status_col = len(cols) - 1 if cols[-1] == "Status" and c.chance(0.7) else None
    keys = c.rng.some(pools[0], n_rows)
    for r in range(n_rows):
        row = []
        for j, pool in enumerate(pools):
            v = keys[r] if j == 0 else c.rng.pick(pool)
            if j == status_col:
                s = S(v, "badge").keep("K2")
                row.append(E("badge", s=s, bound=c.prop("status") if c.fw == "tsx" else "isActive"))
                continue
            s = S(v, "cell")
            if v in seen[j]:
                s.keep("K5")
            seen[j].add(v)
            row.append(s)
        rows.append(row)
    if status_col is not None and c.fw == "html":
        for row in rows:
            row[status_col] = row[status_col]["s"]
            row[status_col].role = "cell"
            if row[status_col].pattern == "K2":
                row[status_col].pattern = None
    return E("table", headers=headers, rows=rows)


def app(c: Ctx) -> Page:
    rng = c.rng
    concept = None
    if rng.chance(0.3):
        concept = rng.pick(T.PAGE_CONCEPTS)
        htext, syn, noun = concept
    else:
        htext, syn, noun = cap(c.objs), [f"Your {c.objs}", f"All {c.objs}"], c.objs
    body = []
    if c.fw != "swift" and rng.chance(0.35):
        crumbs = [S(rng.pick(["Home", c.product, "Workspace"]), "breadcrumb")]
        last = S(htext, "breadcrumb")
        crumbs.append(last)
        body.append(E("crumbs", items=crumbs, aria=ctx("Breadcrumb", "aria") if rng.chance(0.5) else None))
        crumb_last = last
    else:
        crumb_last = None
    head, h1, sub = page_header(c, htext, syn, noun)
    if crumb_last is not None:
        h1.keep("K11")
    nav_title = None
    if c.fw == "swift" and rng.chance(0.6):
        # the title moves to the navigation bar; the in-page heading becomes an h2 that repeats it (R9) or names
        # a subset of the page
        nav_title = S(htext, "h1")
        h1.role = "h2"
        for e in head.walk():
            if e.kind == "h":
                e.p["level"] = 2
        if c.want(0.5):
            h1.echo(nav_title, "R9")
            restack(nav_title, [x for x in _head_slots(head) if x.cls == "ECHO"])
        else:
            h1.text = rng.pick([f"Recent {noun}", f"Shared {noun}", f"{cap(noun)} this week"])
            _retarget(head, h1, nav_title)
    body.append(head)
    acts = []
    btn = S(f"New {c.obj}", "button")
    if htext.lower() == c.objs:
        btn.keep("K9")
    tip = None
    if c.fw != "swift" and c.want(0.1):
        tip = S(btn.text, "tooltip").echo(btn, "R5")
    elif c.fw == "swift" and c.want(0.1):
        tip = S(btn.text, "tooltip").echo(btn, "R5")
    acts.append(E("button", s=btn, tip=tip))
    if rng.chance(0.5):
        tip, aria = rng.pick(T.ICON_ACTIONS)
        acts.append(E("iconbtn", tip=S(tip, "tooltip").keep("K12"), aria=ctx(aria, "aria") if rng.chance(0.7) else None))
    if c.want(0.08):
        acts.insert(0, E("badge", s=S(rng.pick(["● Live", "Live", "Active", "● Online"]), "badge").decor("R11")))
    body.append(E("actions", *acts))
    has_tabs = rng.chance(0.4)
    if has_tabs:
        tabs = rng.some(T.TABS_BY_PAGE, rng.pick([3, 4]))
        tab_slots = [S(t, "tab") for t in tabs]
        if concept and rng.chance(0.6) and htext not in tabs:
            ts = S(htext, "tab").keep("K11")
            tab_slots.insert(rng.randrange(len(tab_slots) + 1), ts)
        body.append(E("tabs", items=tab_slots))
    if nav_title is None and c.want(0.12):
        r9 = S(rng.pick([htext, f"All {noun}", f"Your {noun}"]), "h2").echo(h1, "R9")
        inner = [E("h", s=r9, level=2)]
    else:
        inner = [E("h", s=S(rng.pick(["Recent", "This week", "Needs attention", "Shared with you", "Archived"]), "h2"), level=2)] \
            if rng.chance(0.5) else []
    table = data_table(c)
    if rng.chance(0.35):
        et, eb = rng.pick(c.dom["empty"])
        es = S(et, "empty_title")
        ebody = S(eb, "empty_body").keep("K8")
        cond = E("cond", a=[E("empty", title=es, body=ebody, action=None)], b=[table],
                 expr=f"{c.prop('count')} === 0", swift="count == 0")
        inner.append(cond if c.fw != "html" else table)
    else:
        inner.append(table)
    if c.fw == "tsx" and rng.chance(0.3):
        inner.append(E("dyn", s=S("Signed in as VAR email.", "caption"), pre="Signed in as ", expr="user.email", post=".", look="note"))
        c.prop("user")
    elif c.fw != "html" and rng.chance(0.2):
        inner.append(E("dyn", s=dyn("VAR updated at", "caption"), expr="updatedAt" if c.fw == "tsx" else "updatedAt", look="note"))
    if rng.chance(0.15):
        inner.append(E("note", s=S(rng.pick(["Hold Shift to select a range of rows.", "Press / to jump to search.",
                                            "Drag rows to change the order; the new order is saved for everyone."]),
                                   "caption").keep("K4", ct=True)))
    elif c.want(0.08):
        lines = ["Scroll down to see more.", "Tap an item to see its details.", "Hover over a row to see more options."]
        if has_tabs:
            lines.append("Use the tabs above to switch views.")
        inner.append(E("note", s=S(rng.pick(lines), "caption").narrate("R6", None, ct=True, conf="judgement")))
    body.append(E("section", *inner))
    return Page(f"{cap(htext)}Page", "app", body, product=c.product, props=c.props, nav_title=nav_title)


def _head_slots(head):
    return [e["s"] for e in head.walk() if e["s"] is not None]


def restack(top, copies):
    """Point every copy at ``top`` and list the other copies in each one's ``equiv`` (LABELLING.md §5)."""
    for x in copies:
        x.target = top
        x.equiv = [o for o in copies if o is not x] if len(copies) > 1 else []


def _retarget(head, old, new):
    """After moving a page title into the navigation bar, echoes that pointed at the old heading point at the
    navigation title (the heading itself is now the R9 copy)."""
    for e in head.walk():
        s = e["s"]
        if s is not None and s is not old and s.target is old:
            s.target = new
            if s.equiv:
                s.equiv = [x for x in s.equiv if x is not old]


# ---------------------------------------------------------------------------------------------------------------
# settings and forms


def field_from(c: Ctx, label, example, helper, swift):
    lab = S(label, "label")
    ll = lower_first(label)
    ph = None
    if c.want(0.3):
        ph = S(c.rng.pick(T.PLACEHOLDER_ECHO).format(l=ll, L=label), "placeholder").echo(lab, "R8")
    elif swift or c.chance(0.6):
        ph = S(example, "placeholder")
    hl = None
    if c.want(0.16):
        hl = S(c.rng.pick(T.HELPER_ECHO).format(l=ll), "helper").echo(lab, "R5")
    elif c.chance(0.6):
        hl = S(helper, "helper").keep("K4")
    return E("field", label=lab, ph=ph, helper=hl, type="password" if label == "Password" else "text"), lab


INTEGRATIONS = [("Google Calendar", "Bookings appear in your calendar within a minute."),
                ("Slack", "Posts a message in the channel you choose."),
                ("Xero", "Paid invoices are marked as paid in Xero overnight."),
                ("Zapier", "Needs a Zapier account on any paid plan."),
                ("Mailchimp", "New members join the list you pick; existing ones are left alone.")]
SETTINGS_PLAN = {
    "Settings": {"sections": None, "toggles": None, "toggle_names": None},
    "Notifications": {"sections": [], "toggles": T.TOGGLES, "toggle_names": ["Email", "On your phone", "Digest"]},
    "Billing": {"sections": ["Billing"], "toggles": None, "toggle_names": ["Invoices", "Receipts"]},
    "Team": {"sections": ["Workspace"], "toggles": None, "toggle_names": ["Members", "Permissions"]},
    "Security": {"sections": ["Account"], "toggles": [x for x in T.TOGGLES if x[0] in ("Two-step sign-in", "Public profile",
                                                                                         "Share usage data")],
                 "toggle_names": ["Sign-in", "Privacy"]},
    "Integrations": {"sections": [], "toggles": INTEGRATIONS, "toggle_names": ["Connected apps", "Calendars and chat"]},
    "Profile": {"sections": ["Profile", "Account"], "toggles": None, "toggle_names": ["Privacy", "Preferences"]},
}


def settings(c: Ctx) -> Page:
    rng = c.rng
    swift = c.fw == "swift"
    body = []
    multi = not swift and rng.chance(0.15)
    if multi:
        k = rng.pick([2, 3])
        total = rng.pick([3, 4, 5])
        body.append(E("note", s=S(rng.pick(["Step {k} of {t}", "Part {k} of {t}"]).format(k=k, t=total), "caption").keep("K3")))
    concept = rng.pick(T.PAGE_CONCEPTS[:7])
    htext, syn, noun = concept
    nav_title = None
    head, h1, sub = page_header(c, htext, syn, noun, allow_r16=False)
    if swift:
        nav_title = h1
        h1.role = "h1"
        head = None
        if c.want(0.35):
            r9 = S(htext, "h2").echo(nav_title, "R9")
            body.append(E("section", E("h", s=r9, level=2)))
    else:
        body.append(head)
    tab_slot = None
    if not swift and rng.chance(0.35):
        tabs = [S(t, "tab") for t in rng.some(["General", "Notifications", "Billing", "Security", "Team", "Integrations"], 3)]
        if htext not in [t.text for t in tabs]:
            tab_slot = S(htext, "tab").keep("K11")
            tabs.insert(rng.randrange(len(tabs) + 1), tab_slot)
        else:
            for t in tabs:
                if t.text == htext:
                    t.keep("K11")
        body.insert(0 if rng.chance(0.5) else len(body), E("tabs", items=tabs))
    # the sections a page of this name holds
    by_name = dict(T.SETTINGS_SECTIONS)
    plan = SETTINGS_PLAN[htext]
    if plan["sections"] is None:
        pool = [x for x in T.SETTINGS_SECTIONS if x[0] != htext]
        sections = rng.some(pool, rng.pick([1, 2, 2, 3]))
    else:
        sections = [(n, by_name[n]) for n in plan["sections"] if n != htext or rng.chance(0)]
        sections = sections[: rng.pick([1, len(sections)])]
    tab_names = {t.text for t in body[0]["items"]} if body and body[0].kind == "tabs" else set()
    for e in body:
        if e.kind == "tabs":
            tab_names = {t.text for t in e["items"]}
    first_err = None
    form_kids = []
    for name, fields in sections:
        h2 = S(name, "h2")
        if name in tab_names:
            h2.keep("K11")
        sk = [E("h", s=h2, level=2)]
        if c.want(0.12):
            sk.append(E("sub", s=S(rng.pick(T.SUB_ECHO).format(n=lower_first(name) + " details"), "subtitle").echo(h2, "R3")))
        elif rng.chance(0.3):
            sk.append(E("sub", s=S(rng.pick(["Only you and workspace owners can see these.", "Changes apply straight away.",
                                             "Used on invoices and receipts.", "Shown to everyone in your workspace."]),
                                   "subtitle")))
        for label, example, helper in rng.some(fields, rng.pick([1, 2, 2, 3])):
            f, lab = field_from(c, label, example, helper, swift)
            if not swift and first_err is None and rng.chance(0.12):
                err = S(rng.pick(["Enter " + lower_first(label), lower_first(label).capitalize() + " is required"]), "error")
                f.p["error"] = err
                f.p["err_cond"] = "errors." + "".join(w.capitalize() if i else w.lower() for i, w in enumerate(label.split()))
                c.prop("errors")
                first_err = err
            sk.append(f)
        sk_section = E("section", *sk) if not swift else E("section", *sk)
        form_kids.append(sk_section)
    # toggles and selects
    if plan["toggles"] is not None or rng.chance(0.45):
        names = plan["toggle_names"] or ["Notifications", "Preferences", "Privacy"]
        name = rng.pick([x for x in names if x != htext])
        h2 = S(name, "h2")
        if name in tab_names:
            h2.keep("K11")
        sk = [E("h", s=h2, level=2)]
        k9 = False
        for label, helper in rng.some(plan["toggles"] or T.TOGGLES, rng.pick([1, 2, 3])):
            lab = S(label, "label")
            hl = None
            if c.want(0.18):
                hl = S(rng.pick(T.TOGGLE_ECHO).format(l=lower_first(label), L=label), "helper").echo(lab, "R5", conf="judgement")
            elif rng.chance(0.65):
                hl = S(helper, "helper").keep("K4")
            sk.append(E("toggle", label=lab, helper=hl))
            if shares_stem(name, label):
                k9 = True
        if k9:
            h2.keep("K9")
        if rng.chance(0.5):
            label, opts, helper = rng.pick(T.SELECTS)
            sk.append(E("select", label=S(label, "label"), options=[S(o, "menu_item") for o in opts],
                        helper=S(helper, "helper").keep("K4") if rng.chance(0.6) else None))
        form_kids.append(E("section", *sk))
    if first_err is not None and c.fw != "swift" and rng.chance(0.6):
        summ = E("errsummary", title=S("There is a problem", "h2"), items=[S(first_err.text, "link")])
        first_err.keep("K12")
        form_kids.insert(0, summ)
    save = S(rng.pick(["Save changes", "Save", "Update profile", "Save settings"]), "button")
    acts = [E("button", s=S("Cancel", "button"), primary=False), E("button", s=save)] if rng.chance(0.5) else [E("button", s=save)]
    if c.want(0.12):
        form_kids.append(E("note", s=S(rng.pick(T.NARRATE_FORM).format(b=save.text), "caption").narrate("R6", save, ct=True)))
    elif rng.chance(0.15):
        form_kids.append(E("note", s=S(rng.pick(["Changes to your email take effect after you confirm the link we send.",
                                                "Saving signs you out on other devices."]), "caption").keep("K4", ct=True)))
    form_kids.append(E("actions", *acts))
    if swift:
        body.extend(form_kids)
    else:
        body.append(E("form", *form_kids))
    return Page(f"{htext}Settings", "settings", body, product=c.product, props=c.props, nav_title=nav_title)


# ---------------------------------------------------------------------------------------------------------------
# dialogs


ACKS = [("Export started", "Your export has started.", "We will email the file when it is ready, usually within ten minutes."),
        ("Invite sent", "Your invite has been sent.", "The link in the invite works for seven days."),
        ("Payment received", "We have received your payment.", "A receipt is on its way to your inbox."),
        ("Booking confirmed", "Your booking is confirmed.", "You can change it until 6pm the day before."),
        ("Saved for later", "This has been saved for later.", "Saved items are kept for 90 days.")]


def dialog(c: Ctx) -> Page:
    rng = c.rng
    swift = c.fw == "swift"
    r = rng.random()
    if r < 0.65:
        # a confirmation
        if rng.chance(0.5):
            btxt, noun, cons = c.dom["danger"]
            verb = btxt.split(" ")[0]
        else:
            verb, _ = rng.pick(T.DIALOG_VERBS)
            noun = c.obj
            cons = rng.pick(T.GENERIC_CONSEQUENCES)
            btxt = f"{verb} {noun}"
        title = S(f"{verb} {noun}?" if rng.chance(0.7) else f"{verb} this {noun}?", "dialog_title")
        confirm = S(btxt if rng.chance(0.7) else verb, "button").keep("K11")
        bodies = []
        planted = False
        if c.want(0.35):
            bodies.append(S(rng.pick(T.DIALOG_ECHO).format(v=verb.lower(), obj=noun), "dialog_body").echo(title, "R4"))
            planted = True
        if not (swift and bodies) and (not planted or rng.chance(0.5)):
            if c.want(0.2) and not swift:
                bodies.append(S(rng.pick(T.DIALOG_NARRATE).format(b=confirm.text), "dialog_body").narrate("R6", confirm, ct=True))
            elif not bodies or rng.chance(0.6):
                bodies.append(S(cons, "dialog_body").keep("K7"))
        if swift and len(bodies) > 1:
            bodies = bodies[:1]
        cancel = S(rng.pick(["Cancel", "Keep it", "Go back"]), "button")
        kids = [E("actions", E("button", s=cancel, primary=False), E("button", s=confirm, destructive=True))]
        d = E("dialog", *kids, title=title, body=bodies, confirm=True)
    elif r < 0.85 and not swift:
        # a form dialog
        noun = c.obj
        title = S(f"Rename {noun}" if rng.chance(0.5) else f"New {noun}", "dialog_title")
        label, example, helper = rng.pick(c.dom["fields"])
        f, lab = field_from(c, label, example, helper, False)
        bodies = []
        if c.want(0.15):
            bodies.append(S(rng.pick(["Give this {o} a new name.", "Enter a new name for the {o}.", "Name your {o}."]).format(o=noun),
                            "dialog_body").echo(title, "R4", conf="judgement"))
        elif rng.chance(0.5):
            bodies.append(S(rng.pick(c.dom["facts"]), "dialog_body"))
        save = S(rng.pick(["Save", "Create", f"Create {noun}", "Save name"]), "button")
        if save.text == title.text:
            save.keep("K11")
        kids = [f, E("actions", E("button", s=S("Cancel", "button"), primary=False), E("button", s=save))]
        d = E("dialog", *kids, title=title, body=bodies)
    else:
        # an acknowledgement
        t, echo_body, keep_body = rng.pick(ACKS)
        title = S(t, "dialog_title")
        bodies = []
        if c.want(0.3):
            bodies.append(S(echo_body, "dialog_body").echo(title, "R4", conf="judgement"))
        else:
            bodies.append(S(keep_body, "dialog_body"))
        ok = S(rng.pick(["OK", "Done", "Close"]), "button")
        kids = [E("actions", E("button", s=ok))]
        d = E("dialog", *kids, title=title, body=bodies, sheetlike=False)
    body = []
    if rng.chance(0.4):
        body.append(E("section", E("h", s=S(cap(c.objs), "h1"), level=1),
                      E("p", s=S(fact(c), "body"))))
    body.append(d)
    if swift and not body[:-1]:
        body.insert(0, E("section", E("p", s=S(fact(c), "body"))))
    if c.fw == "tsx":
        c.prop("open")
    return Page(f"{cap(c.obj)}Dialog", "dialog", body, product=c.product, props=c.props)


# ---------------------------------------------------------------------------------------------------------------
# empty states


def empty(c: Ctx) -> Page:
    rng = c.rng
    et, eb = rng.pick(c.dom["empty"])
    title = S(et, "empty_title")
    action = None
    if rng.chance(0.65):
        a = S(rng.pick([f"New {c.obj}", f"Add {c.obj}", f"Create {c.obj}", f"Add your first {c.obj}"]), "button")
        a.keep("K11")
        action = E("button", s=a)
    if c.want(0.4):
        body = S(rng.pick(T.EMPTY_ECHO).format(objs=c.objs if not et.lower().startswith("no ") else et[3:].replace(" yet", "").lower()),
                 "empty_body").echo(title, "R4")
    elif action is not None and c.want(0.2):
        body = S(rng.pick(T.EMPTY_NARRATE).format(obj=c.obj, b=action["s"].text), "empty_body").narrate("R6", action["s"], ct=True)
    else:
        body = S(eb, "empty_body").keep("K8")
    page = []
    if rng.chance(0.4):
        h = S(cap(c.objs), "h1")
        hk = [E("h", s=h, level=1)]
        if c.want(0.15):
            hk.append(E("sub", s=S(rng.pick(T.SUB_ECHO).format(n=c.objs), "subtitle").echo(h, "R3")))
        page.append(E("head", *hk))
    page.append(E("empty", title=title, body=body, action=action))
    if rng.chance(0.25):
        page.append(E("note", s=S(rng.pick(["Need a hand? Our help centre has a five-minute guide.",
                                           "Items you delete stay in the bin for 30 days."]), "caption")))
    return Page(f"Empty{cap(c.objs)}", "empty", page, product=c.product)


# ---------------------------------------------------------------------------------------------------------------
# dashboards


def dashboard(c: Ctx) -> Page:
    rng = c.rng
    htext = rng.pick(["Overview", "Dashboard", "This week", "Today", f"{c.product} at a glance"])
    syn = {"Overview": ["At a glance", "Summary"], "Dashboard": ["Your dashboard", "Home"], "This week": ["Weekly view", "Week so far"],
           "Today": ["Today’s view", "Right now"]}.get(htext, ["At a glance"])
    h1 = S(htext, "h1")
    kids = []
    if c.want(0.2):
        kids.append(E("eyebrow", s=S(rng.pick(syn), "eyebrow").echo(h1, "R1", conf="judgement")))
    elif rng.chance(0.25):
        kids.append(E("eyebrow", s=S(rng.pick(["Week 23 · June 2026", "Tuesday 9 June", "Q2 2026", "May 2026"]), "eyebrow").keep("K1")))
    elif c.want(0.06):
        kids.append(E("eyebrow", s=S(rng.pick(["Local time 09:41", "UTC 14:02", "51.45° N · 2.59° W"]), "eyebrow").decor("R2")))
    kids.append(E("h", s=h1, level=1))
    stats_n = rng.pick([2, 3, 4])
    first_fact = None
    if c.want(0.12):
        kids.append(E("sub", s=S(rng.pick(["Here are your {n} key numbers.", "The {n} cards below cover the basics.",
                                           "Your {n} main figures are shown below."]).format(n=T.NUMBER_WORDS[stats_n]),
                                 "subtitle").narrate("R16", None)))
    elif htext in ("Overview", "Dashboard") and c.want(0.2):
        kids.append(E("sub", s=S(rng.pick(["An overview of everything.", "Your dashboard, at a glance.",
                                           "An at-a-glance summary of everything."]), "subtitle").echo(h1, "R3", conf="judgement")))
    elif rng.chance(0.5):
        first_fact = S(rng.pick(["Figures update every 15 minutes.", "Numbers exclude test data.", "Times are shown in UK time."]),
                       "subtitle")
        kids.append(E("sub", s=first_fact))
    body = [E("head", *kids)]
    if rng.chance(0.5):
        body.append(E("tabs", items=[S(t, "tab") for t in T.TIME_RANGES]))
    stats = []
    for label, value in rng.some(c.dom["stats"], stats_n):
        lab = S(label, "label")
        note = None
        if c.want(0.08):
            note = S(rng.pick(["Total {l}", "{L} so far", "Your {l}"]).format(l=lower_first(label), L=label), "caption").echo(lab, "R5", conf="judgement")
        elif rng.chance(0.35):
            note = S(rng.pick(["+12% on last week", "−3% on last month", "Up 4 since Monday", "Same as last week"]), "caption")
        stats.append(E("stat", value=S(value, "stat"), label=lab, note=note))
    body.append(E("stats", *stats))
    # a panel
    panel = []
    ptitle = None
    if c.want(0.1):
        ptitle = S(htext, "h2").echo(h1, "R9")
    else:
        ptitle = S(rng.pick(["Usage over time", "Recent activity", "By category", f"Latest {c.objs}", "Needs attention"]), "h2")
    panel.append(E("h", s=ptitle, level=2))
    legend = ptitle.text in ("Usage over time", "By category", htext) and rng.chance(0.6)
    if legend:
        panel.append(E("legend", items=[S(x, "list_item").keep("K2") for x in rng.pick(T.LEGENDS)]))
    if not legend or rng.chance(0.4):
        panel.append(data_table(c, n_rows=rng.pick([3, 4, 5])))
    if c.fw != "html" and rng.chance(0.35):
        panel.append(E("status", s=S(rng.pick(["Connected", "Syncing", "Up to date"]), "body").keep("K2"),
                       bound=c.prop("online") if c.fw == "tsx" else "isOnline"))
    elif c.want(0.12):
        panel.append(E("badge", s=S(rng.pick(["● Live", "Live", "● Real-time"]), "badge").decor("R11")))
    if first_fact is not None and c.want(0.25):
        alt = {"Figures update every 15 minutes.": "Updated every 15 minutes.", "Numbers exclude test data.":
               "Test data is not included.", "Times are shown in UK time.": "All times are UK time."}[first_fact.text]
        panel.append(E("note", s=S(rng.pick([first_fact.text, alt]), "caption").echo(first_fact, "R14")))
    elif c.fw == "tsx" and rng.chance(0.3):
        panel.append(E("dyn", s=S("Last updated VAR updated at.", "caption"), pre="Last updated ", expr="updatedAt", post=".", look="note"))
    body.append(E("section", *panel))
    return Page("Dashboard" if htext == "Dashboard" else f"{c.product}Dashboard", "dashboard", body, product=c.product,
                props=c.props)


# ---------------------------------------------------------------------------------------------------------------
# game menus and HUDs


def game(c: Ctx) -> Page:
    rng = c.rng
    g = rng.pick(T.GAMES)
    c.product = g["title"]
    r = rng.random()
    if r < 0.45:
        return game_overlay(c, g)
    if r < 0.6:
        return game_main(c, g)
    if r < 0.8:
        return game_settings(c, g)
    return game_hud(c, g)


def _keep_sub_ct(text: str) -> bool:
    return any(w in text.lower() for w in ("resum", "restart", "leaving", "claimed", "quit", "paused", "queue"))


def game_overlay(c: Ctx, g) -> Page:
    rng = c.rng
    st = rng.pick(T.GAME_STATES)
    caps = rng.chance(0.6)
    up = (lambda s: s.upper()) if caps else (lambda s: s)
    head_text = rng.pick(st["head"])
    if caps:
        head_text = head_text.upper()
    h1 = S(head_text, "h1")
    kids = []
    if c.want(0.4):
        kids.append(E("eyebrow", s=S(up(rng.pick(st["eyebrow_echo"])), "eyebrow").echo(h1, "R1", conf="judgement")))
    elif rng.chance(0.3):
        kids.append(E("eyebrow", s=S(up(rng.pick([g["title"], "Chapter 3 · The docks", f"{cap(g['mission'])} 4", "Act II"])),
                                     "eyebrow").keep("K1")))
    elif c.want(0.08):
        kids.append(E("eyebrow", s=S(rng.pick(T.HUD_DECOR_META), "eyebrow").decor("R2")))
    kids.append(E("h", s=h1, level=1))
    action = S(up(rng.pick(st["action"])), "button").keep("K11")
    menu = [S(up(x), "button") for x in rng.some(T.GAME_MENU, rng.pick([1, 2, 3]))]
    buttons = [action] + menu
    sub = None
    if c.want(0.35):
        line = rng.pick(st["narrate"])
        sub = S(line, "subtitle").narrate("R6", action, ct=True, conf="judgement")
    elif c.want(0.2):
        sub = S(rng.pick(st["sub_echo"]), "subtitle").echo(h1, "R3", conf="judgement")
    elif c.want(0.08):
        sub = S(rng.pick(["Choose one of the {n} options below.", "Pick from the {n} options below."]).format(
            n=T.NUMBER_WORDS[len(buttons)]), "subtitle").narrate("R16", None)
    elif rng.chance(0.5):
        t = rng.pick(st["keep_sub"])
        ct = _keep_sub_ct(t)
        sub = S(t, "subtitle").keep("K4" if ct else None, ct=ct, conf="judgement" if ct else "clear")
    if sub is not None:
        kids.append(E("sub", s=sub))
    body = [E("head", *kids), E("actions", *[E("button", s=b, primary=i == 0) for i, b in enumerate(buttons)])]
    if rng.chance(0.2):
        body.append(E("note", s=S(rng.pick(T.GAME_HINTS), "caption").keep("K4", ct=True)))
    return Page(rng.pick(["PauseMenu", "OverlayScreen", "EndScreen", "StateOverlay"]), "game", body, product=g["title"])


def game_main(c: Ctx, g) -> Page:
    rng = c.rng
    h1 = S(g["title"].upper() if rng.chance(0.5) else g["title"], "h1")
    kids = []
    if rng.chance(0.4):
        kids.append(E("eyebrow", s=S(rng.pick(["Season 2", "Early access", "Director’s cut", "Chapter one"]), "eyebrow").keep("K1")))
    kids.append(E("h", s=h1, level=1))
    if rng.chance(0.4):
        kids.append(E("sub", s=S(rng.pick(["Your last save: Harbour, 2 hours ago.", "Cross-play is on.",
                                           "Server maintenance on Thursday from 06:00 to 08:00 UK time."]), "subtitle")))
    items = rng.some(T.GAME_MAIN, rng.pick([3, 4, 5])) + rng.some(g["items"], rng.pick([0, 1, 2]))
    buttons = [S(x, "button") for x in items]
    body = [E("head", *kids), E("actions", *[E("button", s=b, primary=i == 0) for i, b in enumerate(buttons)])]
    if c.want(0.15):
        body.append(E("note", s=S(rng.pick(["Use the menu to navigate.", "Select an option to begin.", "Click any option to continue."]),
                                  "caption").narrate("R6", None, ct=True, conf="judgement")))
    if rng.chance(0.5):
        body.append(E("note", s=S(rng.pick(["v1.0.4", "Build 2026.06.11", "v0.9.2 early access"]), "caption").keep()))
    if c.want(0.12):
        body.append(E("meta", items=[S(rng.pick(T.HUD_DECOR_META), "caption").decor("R2")]))
    return Page("MainMenu", "game", body, product=g["title"])


def game_settings(c: Ctx, g) -> Page:
    rng = c.rng
    swift = c.fw == "swift"
    htext = rng.pick(["SETTINGS", "Settings", "OPTIONS", "Options"])
    h1 = S(htext, "h1")
    body = []
    nav_title = None
    if swift and rng.chance(0.5):
        nav_title = S(htext.capitalize(), "h1")
        if c.want(0.4):
            body.append(E("section", E("h", s=S(htext.capitalize(), "h2").echo(nav_title, "R9"), level=2)))
    else:
        kids = []
        if c.want(0.2):
            kids.append(E("eyebrow", s=S(rng.pick(["CONFIGURATION", "PREFERENCES", "SYSTEM"]) if htext.isupper() else
                                         rng.pick(["Configuration", "Preferences"]), "eyebrow").echo(h1, "R1", conf="judgement")))
        kids.append(E("h", s=h1, level=1))
        if c.want(0.15):
            kids.append(E("sub", s=S(rng.pick(["Change the game settings.", "Adjust your options here.", "Configure the game."]),
                                     "subtitle").echo(h1, "R3")))
        body.append(E("head", *kids))
    tabs = rng.some(T.GAME_TABS, rng.pick([3, 4, 5]))
    body.append(E("tabs", items=[S(t, "tab") for t in tabs]))
    sec = rng.pick(tabs)
    h2 = S(sec, "h2").keep("K11")  # the open tab's name over its panel
    rows = [E("h", s=h2, level=2)]
    for label, helper in rng.some(T.GAME_SETTINGS, rng.pick([2, 3, 4])):
        lab = S(label, "label")
        if c.want(0.15):
            hl = S(rng.pick(["Adjust the {l}.", "Change your {l}.", "Set the {l}."]).format(l=lower_first(label)), "helper").echo(lab, "R5")
        elif rng.chance(0.7):
            hl = S(helper, "helper").keep("K4")
        else:
            hl = None
        rows.append(E("toggle", label=lab, helper=hl))
    if rng.chance(0.3):
        rows.append(E("note", s=S(rng.pick(T.GAME_HINTS), "caption").keep("K4", ct=True)))
    body.append(E("section", *rows))
    done = S(rng.pick(["APPLY", "Apply", "BACK", "Back", "Save and exit"]), "button")
    acts = [E("button", s=done)]
    if c.want(0.1):
        body.append(E("note", s=S(f"Press {done.text.capitalize()} to keep your changes.", "caption").narrate("R6", done, ct=True)))
    body.append(E("actions", *acts))
    return Page("SettingsMenu", "game", body, product=g["title"], nav_title=nav_title)


def game_hud(c: Ctx, g) -> Page:
    rng = c.rng
    body = []
    stats = []
    for label, values in rng.some(T.HUD_LABELS, rng.pick([2, 3, 4])):
        stats.append(E("stat", value=S(rng.pick(values), "stat"), label=S(label.upper() if rng.chance(0.5) else label, "label")))
    body.append(E("stats", *stats))
    obj = S(rng.pick(T.HUD_OBJECTIVES), "body")
    ok = [E("eyebrow", s=S(rng.pick(["Objective", "OBJECTIVE", "Current task"]), "eyebrow").keep())] if rng.chance(0.6) else []
    body.append(E("section", *ok, E("p", s=obj)))
    if rng.chance(0.5):
        wave = rng.pick(["Wave {a} of {b}", "Lap {a} / {b}", "Round {a} of {b}", "Stage {a}/{b}"])
        body.append(E("note", s=S(wave.format(a=rng.pick([2, 3, 4]), b=rng.pick([5, 8, 10])), "caption").keep("K3")))
    if c.fw != "html" and rng.chance(0.4):
        body.append(E("status", s=S(rng.pick(["Shield low", "Signal", "Squad connected"]), "body").keep("K2"),
                      bound=c.prop("health") if c.fw == "tsx" else "lowShield"))
    if c.want(0.25):
        body.append(E("meta", items=[S(rng.pick(T.HUD_DECOR_META), "caption").decor("R2")]))
    if c.want(0.15):
        body.append(E("badge", s=S(rng.pick(["● REC", "● LIVE", "◆ ONLINE"]), "badge").decor("R11")))
    if rng.chance(0.3):
        body.append(E("note", s=S(rng.pick(T.GAME_HINTS), "caption").keep("K4", ct=True)))
    return Page("Hud", "game", body, product=g["title"], props=c.props)


COMPOSERS = {"marketing": marketing, "app": app, "settings": settings, "dialog": dialog, "empty": empty,
             "dashboard": dashboard, "game": game}


def k14(c: Ctx, role="body"):
    """Two branches of one conditional that read alike (K14), or None where the framework has no conditional."""
    if c.fw == "html":
        return None
    a, b = c.rng.pick(T.K14_GAME if c.product in [g["title"] for g in T.GAMES] else T.K14_PAIRS)
    flag = c.prop(c.rng.pick(["saved", "online", "paid", "verified", "selected"])) if c.fw == "tsx" else "isDone"
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
        cond = k14(c, rng.pick(["body", "caption"]))
        if cond is not None:
            page.body.append(E("section", cond) if fw == "swift" else cond)
    page.r19 = c.r19
    if c.props:
        page.props = c.props
    link_stacks(page)
    return page, c
