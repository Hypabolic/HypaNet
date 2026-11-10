"""Compositional copy for the dev generator (v2): per-domain synonyms and people, generated names, numbers and
dates, and phrase builders for both sides of every decision (copy that adds a fact about a topic, and copy that only
restates, narrates or decorates it).

Every string and template here was invented for the generator; none is taken from a repo, and nothing here enters
the author kit (SPEC.md §9). Copy avoids double quotes, backslashes, braces and angle brackets so that it can be
written unescaped into TSX, HTML and Swift; apostrophes are typographic (’).

Two rules shape the tables:

* Overlap must not decide the class. Every restating template comes in a form that shares a stem with what it
  restates and a form that does not (a synonym or a pronoun); every informative template comes in both forms too.
* A frequent string must not decide the class. Copy is built from slots (nouns, numbers, times, names, places), so
  that a phrase seldom repeats, and the short words that must repeat (status words, genre words) are used on both
  sides.
"""

from __future__ import annotations

from .core import article, cap, plural

# ---------------------------------------------------------------------------------------------------------------
# Per-domain supplement: synonyms for the two nouns (sharing no stem with them), the people who use the product, the
# group they belong to and a role with extra rights.

DOM_EXTRA = {
    "bees": {"x": [("colony", "colonies")], "y": [("check", "checks"), ("visit", "visits")],
             "who": ("beekeeper", "beekeepers"), "group": "association", "role": "mentor"},
    "allot": {"x": [("patch", "patches")], "y": [("planting", "plantings")],
              "who": ("plot holder", "plot holders"), "group": "allotment society", "role": "site rep"},
    "bikes": {"x": [("fix", "fixes"), ("job", "jobs")], "y": [("slot", "slots"), ("appointment", "appointments")],
              "who": ("rider", "riders"), "group": "cycling club", "role": "mechanic"},
    "invoice": {"x": [("bill", "bills")], "y": [("customer", "customers")],
                "who": ("freelancer", "freelancers"), "group": "studio", "role": "bookkeeper"},
    "sail": {"x": [("regatta", "regattas")], "y": [("dinghy", "dinghies"), ("craft", "craft")],
             "who": ("sailor", "sailors"), "group": "club", "role": "race officer"},
    "meals": {"x": [("dish", "dishes")], "y": [("dinner", "dinners")],
              "who": ("cook", "cooks"), "group": "household", "role": "planner"},
    "podcast": {"x": [("instalment", "instalments")], "y": [("series", "series")],
                "who": ("host", "hosts"), "group": "network", "role": "producer"},
    "pottery": {"x": [("course", "courses"), ("session", "sessions")], "y": [("kiln load", "kiln loads")],
                "who": ("potter", "potters"), "group": "studio", "role": "tutor"},
    "climb": {"x": [("problem", "problems"), ("line", "lines")], "y": [("visit", "visits")],
              "who": ("climber", "climbers"), "group": "club", "role": "setter"},
    "photos": {"x": [("collection", "collections")], "y": [("picture", "pictures"), ("snap", "snaps")],
               "who": ("relative", "relatives"), "group": "family", "role": "editor"},
    "ferry": {"x": [("crossing", "crossings"), ("departure", "departures")], "y": [("fare", "fares"), ("pass", "passes")],
              "who": ("passenger", "passengers"), "group": "travelling party", "role": "agent"},
    "farm": {"x": [("delivery", "deliveries")], "y": [("hamper", "hampers")],
             "who": ("customer", "customers"), "group": "household", "role": "packer"},
    "museum": {"x": [("show", "shows"), ("display", "displays")], "y": [("trip", "trips")],
               "who": ("visitor", "visitors"), "group": "school party", "role": "curator"},
    "chess": {"x": [("contest", "contests")], "y": [("fixture", "fixtures")],
              "who": ("player", "players"), "group": "club", "role": "captain"},
    "budget": {"x": [("spending plan", "spending plans")], "y": [("payment", "payments"), ("purchase", "purchases")],
               "who": ("saver", "savers"), "group": "household", "role": "partner"},
    "brew": {"x": [("brew", "brews")], "y": [("barrel", "barrels"), ("firkin", "firkins")],
             "who": ("brewer", "brewers"), "group": "brewery", "role": "head brewer"},
    "kayak": {"x": [("outing", "outings")], "y": [("kayak", "kayaks"), ("canoe", "canoes")],
              "who": ("paddler", "paddlers"), "group": "group", "role": "guide"},
    "tutor": {"x": [("class", "classes"), ("session", "sessions")], "y": [("teacher", "teachers")],
              "who": ("learner", "learners"), "group": "class", "role": "coordinator"},
    "fleet": {"x": [("van", "vans")], "y": [("operator", "operators")],
              "who": ("fleet manager", "fleet managers"), "group": "depot", "role": "dispatcher"},
    "lab": {"x": [("study", "studies"), ("run", "runs")], "y": [("specimen", "specimens"), ("aliquot", "aliquots")],
            "who": ("scientist", "scientists"), "group": "lab", "role": "witness"},
    "birds": {"x": [("count", "counts")], "y": [("record", "records"), ("observation", "observations")],
              "who": ("recorder", "recorders"), "group": "county group", "role": "verifier"},
    "music": {"x": [("session", "sessions"), ("class", "classes")], "y": [("student", "students"), ("learner", "learners")],
              "who": ("teacher", "teachers"), "group": "music school", "role": "head of department"},
    "energy": {"x": [("appliance", "appliances"), ("gadget", "gadgets")], "y": [("measurement", "measurements")],
               "who": ("householder", "householders"), "group": "household", "role": "bill payer"},
    "library": {"x": [("borrowed item", "borrowed items")], "y": [("title", "titles"), ("novel", "novels")],
                "who": ("reader", "readers"), "group": "branch", "role": "librarian"},
    "roast": {"x": [("plan", "plans")], "y": [("pack", "packs")],
              "who": ("coffee drinker", "coffee drinkers"), "group": "office", "role": "account holder"},
    "rota": {"x": [("slot", "slots")], "y": [("helper", "helpers")],
             "who": ("organiser", "organisers"), "group": "charity", "role": "coordinator"},
    "vet": {"x": [("visit", "visits"), ("consultation", "consultations")], "y": [("animal", "animals")],
            "who": ("owner", "owners"), "group": "household", "role": "nurse"},
}

# ---------------------------------------------------------------------------------------------------------------
# Names, numbers, dates

NAME_A = ["North", "Fern", "Cobble", "Tide", "Elm", "Kiln", "Brook", "Moss", "Quill", "Lark", "Wren", "Pebble",
          "Copper", "Hazel", "Juniper", "Marlow", "Otter", "Saffron", "Thistle", "Willow", "Bramble", "Linden",
          "Slate", "Heron", "Rowan", "Tansy", "Sorrel", "Ember", "Harbour", "Meadow"]
NAME_B = ["line", "wise", "ly", "box", "book", "base", "dock", "yard", "path", "nest", "works", "kit", "desk", "hub",
          "tab", "log", "mark", "field", "stack", "post"]
FIRST = ["Priya", "Tom", "Mei", "Rhys", "Ana", "Kofi", "Hannah", "Owen", "Sara", "Jamal", "Ines", "Luca", "Niamh",
         "Ravi", "Zoe", "Dafydd", "Amara", "Felix", "Grace", "Hamid", "Isla", "Joon", "Kat", "Leon", "Maya", "Nico",
         "Oona", "Pavel", "Ruth", "Sol", "Tariq", "Uma", "Vik", "Wren", "Yusuf"]
LAST = ["Shah", "Evans", "Lin", "Jones", "Costa", "Mensah", "Brook", "Price", "Okafor", "Novak", "Reid", "Haddad",
        "Byrne", "Patel", "Moreau", "Lloyd", "Adeyemi", "Kerr", "Silva", "Takahashi", "Walsh", "Hughes", "Iqbal",
        "Fraser", "Doyle", "Kowalski", "Mills", "Ng", "Pryce", "Quinn"]
PLACES = ["Bristol", "Leeds", "Cardiff", "Norwich", "Exeter", "Dundee", "York", "Bath", "Hull", "Derby", "Swansea",
          "Galway", "Lancaster", "Truro", "Inverness", "Brighton", "Stroud", "Kendal", "Whitby", "Aberystwyth"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
                10: "ten", 12: "twelve"}
ZONES = ["UK time", "Lisbon time", "Central European Time", "your own time zone", "the time zone of the site"]
PLANS = ["Free", "Starter", "Plus", "Pro", "Team", "Studio", "Club", "Society"]


def product_name(rng, dom) -> str:
    if rng.chance(0.45):
        return rng.pick(dom["names"])
    return rng.pick(NAME_A) + rng.pick(NAME_B)


def person(rng) -> str:
    return f"{rng.pick(FIRST)} {rng.pick(LAST)}"


def clock(rng) -> str:
    return f"{rng.randrange(6, 23):02d}:{rng.pick(['00', '05', '10', '15', '20', '30', '40', '45', '50', '55'])}"


def date(rng, year=False) -> str:
    d = f"{rng.randrange(1, 29)} {rng.pick(MONTHS)}"
    return d + (f" {rng.pick([2025, 2026])}" if year else "")


def money(rng, lo=3, hi=400) -> str:
    v = rng.randrange(lo, hi)
    return f"£{v:,}" if rng.chance(0.7) else f"£{v:,}.{rng.pick(['00', '50', '20', '95', '40'])}"


def word_or_num(n: int, rng) -> str:
    return NUMBER_WORDS[n] if n in NUMBER_WORDS and rng.chance(0.5) else str(n)


# ---------------------------------------------------------------------------------------------------------------
# Generic section concepts (marketing). Each concept has a heading, eyebrows that restate it (R1; none shares a
# stem with the heading), subtitles that restate it (R3, with and without shared stems) and subtitles that add a
# fact about it (KEEP; {x} is the product's main plural noun, {y} the second, {p} the product, {n} a number).

CONCEPTS = [
    {"head": ["Up and running in minutes", "Set up before your coffee cools", "Ready in an afternoon"],
     "eb": ["Quick setup", "Fast start", "No waiting", "Easy start", "Speedy onboarding", "Instant start", "Zero to go"],
     "echo": ["Getting started takes no time at all.", "You will be going in no time.", "Starting is quick and painless.",
              "Set up in minutes, start right away."],
     "keep": ["Most {groups} are up and running within {n} minutes of signing up.",
              "Setup takes about {n} minutes if your {x} are already in a spreadsheet.",
              "Import a spreadsheet of your {x} and the columns are matched for you.",
              "Most {who} add their first {x1} within {n} minutes of signing up.",
              "Bring your {y} over from a CSV; anything we cannot read is listed, not dropped.",
              "Setup asks {n} questions, and you can skip all of them."]},
    {"head": ["Built for teams of every size", "For one person or a whole {group}", "Room for the whole {group}"],
     "eb": ["Any team size", "Small or large", "Grows with you", "Scales up", "Big or small", "Every headcount"],
     "echo": ["Whether your team is small or large, it fits.", "It works for small groups and big ones alike.",
              "However many of you there are, it scales."],
     "keep": ["Teams of up to {n} share one bill; larger teams get a monthly invoice.",
              "Each member gets their own login; {roles} can edit, everyone else can view.",
              "Up to {n} people on the free plan, with no limit on paid plans.",
              "Invite links stay valid for {n} days and can be revoked at any time.",
              "{A_group} of {N} shares one bill, paid by whoever owns the account."]},
    {"head": ["Your data stays yours", "Your records, your rules", "Nobody else reads your {x}"],
     "eb": ["Privacy first", "Ownership", "Kept private", "Private by design", "Under lock", "Belongs to you"],
     "echo": ["We believe what you store belongs to you.", "You own it, always.", "It remains yours, and only yours."],
     "keep": ["Your data is stored in London and Dublin and never sold.",
              "Your {x} stay yours: export them as CSV at any time, even after you cancel.",
              "Export every {x1} as CSV at any time, including after you cancel.",
              "Stored in London and Dublin; nothing leaves the UK and EU.",
              "Deleted {y} are erased from backups after {n} days.",
              "We never sell or share your {y}, and staff cannot open them without your say-so."]},
    {"head": ["Works offline", "No signal, no problem", "Keeps going without a connection"],
     "eb": ["Offline ready", "Anywhere", "Off the grid", "No Wi-Fi needed", "Dead zones welcome", "Unplugged"],
     "echo": ["It keeps working when the internet does not.", "Use it even when you have no connection.",
              "Whether online or offline, it carries on."],
     "keep": ["Offline edits sync within a minute of reconnecting; clashes are flagged, never overwritten.",
              "Works offline for up to {n} days; after that, sign in once to carry on.",
              "{Y} you add without signal sync within a minute of reconnecting.",
              "Edits made offline on two phones are merged; clashes are flagged, never overwritten.",
              "The last {n} days of {x} stay on your phone for offline use.",
              "Photos taken offline upload on Wi-Fi only, unless you say otherwise."]},
    {"head": ["Simple, honest pricing", "One price, no surprises", "Pay for what you use"],
     "eb": ["Fair prices", "Transparent costs", "Nothing hidden", "Clear costs", "Honest rates", "Fair and square",
            "Straight talk on cost"],
     "echo": ["Our prices are simple and honest.", "Costs are clear, with nothing hidden.", "What you see is what you pay."],
     "keep": ["Pricing is per {group}, not per person; charities pay half.",
              "Prices include VAT; {groups} registered as charities pay half.",
              "Move up or down a plan on any day; the difference is worked out to the day.",
              "Paid plans start with {n} days free and need no card until then.",
              "The free plan covers up to {n} {x}, with no time limit."]},
    {"head": ["Loved by {who}", "Trusted across the country", "Chosen by {N} {who}"],
     "eb": ["Customer favourite", "Popular choice", "Much loved", "Highly rated", "Crowd pleaser", "Well liked"],
     "echo": ["Lots of people really like using it.", "A firm favourite with its users.", "People love it, and say so."],
     "keep": ["Rated 4.{d} out of 5 from {N} reviews on the App Store.",
              "{N} {who} used {p} last month.",
              "Used by {n} {groups} in {place} alone.",
              "Most {who} who try it for a month stay for a year or more."]},
    {"head": ["Secure by default", "Locked down from day one", "Safe hands for your {x}"],
     "eb": ["Security first", "Protected", "Built-in safety", "Safety", "Locked tight", "Guarded"],
     "echo": ["Safety is built in from the start.", "We take protecting you seriously.", "Safe and sound, as standard."],
     "keep": ["Security reviews by an outside firm every {n} months; the last report is on request.",
              "Secure sign-in links expire after {n} minutes.",
              "Two-step sign-in is on for every account and cannot be turned off by members.",
              "Data is encrypted on our servers and on its way to them; keys change every {n} days.",
              "Every change to {a_x1} is logged with who made it and when.",
              "Sessions end after {n} days without use."]},
    {"head": ["Made for your phone", "In your pocket", "Built for the field"],
     "eb": ["Mobile first", "On the go", "Handheld", "Pocket sized", "Phone ready", "Portable"],
     "echo": ["Designed for the device in your pocket.", "Use it on the go, wherever you are.", "Built with phones in mind."],
     "keep": ["The iPhone and Android apps work one-handed, with buttons sized for gloves.",
              "Open your next {x1} from the lock screen in {n} taps.",
              "Runs on iOS 16 and Android 11 or later; tablets get a two-column layout.",
              "Widgets show your next {x1} without opening the app."]},
    {"head": ["Help when you need it", "Real people on the other end", "Ask us anything"],
     "eb": ["Real support", "Always here", "Human help", "Support", "On hand", "Here for you"],
     "echo": ["Our team is here to help whenever you need us.", "Get help whenever you need it.", "Assistance is always on hand."],
     "keep": ["Help is a reply away: email support answers within {n} working hours.",
              "Email support answers within {n} working hours, Monday to Saturday.",
              "Pro plans get a phone line staffed until 8pm UK time.",
              "Our {n}-minute video guides cover every screen.",
              "Moving from another app? We import your {x} for you, free."]},
    {"head": ["Faster than ever", "Quicker every release", "No more waiting"],
     "eb": ["Speed", "Lightning quick", "Snappy", "Quick", "Rapid", "Zippy"],
     "echo": ["It is quicker than it has ever been.", "Speed you can feel.", "Everything happens at pace now."],
     "keep": ["Faster sync: changes now reach other phones in about {n} seconds.",
              "Lists of {n},000 {x} open in under a second.",
              "Search covers every {y1} as you type, including notes.",
              "Release 4.{d} halved the time it takes to open {a_x1}.",
              "Reports that took a minute now take {n} seconds."]},
    {"head": ["Plays well with others", "Connects to the tools you use", "Fits into your week"],
     "eb": ["Integrations", "Connected", "Joined up", "Works together", "Linked", "Compatible"],
     "echo": ["It connects with the software you already rely on.", "Hooks into your favourite apps.",
              "Works alongside everything else you use."],
     "keep": ["Sync {y} to Google Calendar or Outlook; changes flow both ways.",
              "A Zapier connector covers {n} triggers, on any paid plan.",
              "Paid invoices are marked as paid in Xero overnight.",
              "A read-only API lets your {group} build its own reports."]},
    {"head": ["Start for free", "Nothing to pay to begin", "Free to try, free to stay"],
     "eb": ["No cost", "Zero to begin", "On the house", "Gratis", "Costs nothing"],
     "echo": ["Getting going costs nothing.", "Begin without paying a penny.", "It is free to get going."],
     "keep": ["Free for up to {n} {x}, for as long as you like.",
              "Start with the free plan; paid features unlock for {n} days, then switch off without charging you.",
              "The free plan keeps up to {n} {x} for as long as you like.",
              "Paid features unlock for {n} days, then switch off without charging you.",
              "No card is needed until you choose a paid plan.",
              "{Groups} with fewer than {n} members never pay."]},
]

# Section-genre words (LABELLING.md §6 K1) in groups of the same kind, and headings that contain each one.
GENRE_GROUPS = [
    ["Overview", "Summary", "At a glance"],
    ["Pricing", "Plans", "Plans and pricing"],
    ["FAQ", "Questions", "Common questions"],
    ["Contact", "Get in touch", "Contact us"],
    ["About", "About us", "Our story"],
    ["Features", "Capabilities", "What it does"],
    ["Testimonials", "Reviews", "What people say", "Word of mouth", "Kind words"],
    ["How it works", "The process", "Step by step"],
    ["Get started", "Getting started", "Start here"],
    ["Our team", "The team", "Meet the team"],
    ["Services", "What we offer"],
    ["Why us", "Why choose us"],
]
GENRE_HEADING = {
    "Features": ["Features that grow with you", "Features for busy weeks", "The features you will use daily",
                 "Features {who} asked for"],
    "Capabilities": ["Capabilities at a glance", "Capabilities built for {who}"],
    "Pricing": ["Pricing for every stage", "Pricing that scales with your {group}", "Straightforward pricing"],
    "FAQ": ["FAQ: your questions answered", "The {p} FAQ"],
    "How it works": ["How it works, step by step", "Here is how it works", "How it works for {who}"],
    "Testimonials": ["Testimonials from {who}", "Testimonials from our customers"],
    "Services": ["Services we offer", "Our services, explained"],
    "Overview": ["An overview of {p}", "Product overview"],
    "Why us": ["Why {who} choose us", "Why us, and not the rest"],
    "About us": ["About us and our story", "A little about us"],
    "Our team": ["Meet our team", "Our team in {place}"],
}

# Labels that only name what comes next (R2). Built from a verb and an optional object.
NEXT_HEAD = ["Introducing", "Discover", "Read on", "Take a look", "Up next", "In this section", "The details",
             "A closer look", "What’s inside", "Here’s the thing", "Explore", "See inside"]

# Decorative meta strips (R2): coordinates, local time, founding year on a new product, edition labels.


def meta_strip(rng) -> list:
    kind = rng.randrange(5)
    if kind == 0:
        return [f"{rng.randrange(35, 60)}.{rng.randrange(1000, 9999)}° N", f"{rng.randrange(0, 9)}.{rng.randrange(1000, 9999)}° W"]
    if kind == 1:
        return [f"Local time {clock(rng)}", rng.pick(["GMT", "GMT+1", "BST", "UTC"])]
    if kind == 2:
        return [f"Founded {rng.pick([2025, 2026])}", f"Vol. {rng.randrange(1, 9):02d}"]
    if kind == 3:
        return [f"{rng.pick(PLACES)} — {rng.pick(PLACES)}", f"Since {rng.pick([2025, 2026])}"]
    return [f"Index {rng.randrange(1, 40):02d}", f"Edition {rng.pick([2025, 2026])}", f"N {rng.randrange(50, 58)}.{rng.randrange(10, 99)}"]


def hud_meta(rng) -> str:
    return rng.pick([
        f"SECTOR {rng.randrange(1, 12)}{rng.pick('ABCDEFG')} // {rng.randrange(10, 60)}.{rng.randrange(1, 9)}N {rng.randrange(10, 140)}.{rng.randrange(1, 9)}W",
        f"GRID {rng.randrange(1, 20):02d}-{rng.randrange(1, 40):02d} · ALT {rng.randrange(1, 90) * 10}M",
        f"SYS.OK · v{rng.randrange(1, 5)}.{rng.randrange(0, 20):02d} · NODE {rng.randrange(1, 9)}",
        f"CAM {rng.randrange(1, 12):02d} · REC", f"FREQ {rng.randrange(100, 180)}.{rng.randrange(1, 9)} MHZ",
        f"LAT {rng.randrange(40, 60)}.{rng.randrange(10, 99)} · LON −{rng.randrange(0, 9)}.{rng.randrange(10, 99)}",
    ])


# ---------------------------------------------------------------------------------------------------------------
# Restating copy built from parts. ``x`` is a plural noun phrase; ``alt`` is a synonym phrase (no shared stem).

ECHO_VERB = ["Manage", "View and manage", "Keep track of", "See", "Browse", "Review", "Find", "Look through",
             "Check on", "Organise"]
ECHO_TAIL = ["", " here", " in one place", ", all in one place", " at a glance", " from one screen", " in one list",
             ", all together", "; every one of them", ", one by one", "; nothing else", ", listed"]


def echo_list(rng, x: str, alt: str | None) -> str:
    """A subtitle that only says the page lists ``x`` (R3). With ``alt`` it uses the synonym or a pronoun. Half of
    them do not address the reader."""
    if alt is None:
        obj = rng.pick([f"your {x}", f"all your {x}", f"every one of your {x}", f"the {x} you have", f"all {x}",
                        f"the {x}", f"each of the {x}", f"every one of the {x}", f"all the {x} on record"])
    else:
        obj = rng.pick([f"your {alt}", f"all your {alt}", "everything you have added", "all of them",
                        "everything added so far", "what has been added", f"the {alt}", f"all the {alt}", "everything on record"])
    if rng.chance(0.2):
        return cap(obj) + rng.pick([", in one list.", "; all of them, listed.", ", every one, listed here.",
                                    ", from first to last.", ", nothing more."])
    return rng.pick(ECHO_VERB) + " " + obj + rng.pick(ECHO_TAIL) + "."


# ---------------------------------------------------------------------------------------------------------------
# Informative copy about a listed noun: scope, timing, limits, retention, sharing. ``{X}``/``{x}`` the plural,
# ``{x1}`` the singular; the ``alt`` forms refer to the list without naming it.

LIST_FACTS = [
    "{X} older than {n} months move to the archive; nothing is deleted.",
    "Your {x} older than {n} months move to the archive.",
    "Your {group} sees the same {x} as you.",
    "You can undo changes to {x} for {n} minutes.",
    "New {x} show up here within {n} minutes of being saved.",
    "Only owners and {roles} can change {x}; everyone else can view them.",
    "The {plan} plan keeps up to {n} {x}; larger plans have no limit.",
    "{X} you delete stay in the bin for {n} days.",
    "Each {x1} keeps a history of who changed it and when.",
    "{X} are shared with everyone in your {group}.",
    "Times are shown in {zone}.",
    "{X} sync to your other devices within a minute.",
    "Your {x} are backed up every night; the last {n} copies are kept.",
    "The total counts {x} in every state except draft.",
    "Each {x1} keeps its photos, notes and receipts together.",
    "Browse {x} by {group}; filters are kept for next time.",
    "Review {x} before {date}; anything unreviewed is archived.",
    "Find any {x1} by name, date or {role}.",
    "Keep track of {x} offline; changes sync when you reconnect.",
    "Manage who can edit {x} from the Members page.",
    "Look through {x} from before {year} in the archive.",
    "Check on {x} shared with you; they are read-only.",
    "Want them on paper? Each {x1} prints on one A4 sheet.",
    "Look for the coloured edge: those {x} are shared.",
    "{X} below the line are waiting for approval.",
    "Times below are in {zone}.",
    "Figures below exclude {x} still in draft.",
    "{X} from your {group} appear here too.",
    "Only {x} you can edit show here.",
    "Reminders go out {n} days before each {x1} is due.",
    "{X} added by {who} outside your {group} need approval first.",
    "{X} from before {year} are kept read-only.",
    "{X} marked private are hidden from this list for everyone but you.",
    "Totals exclude {x} still in draft.",
]
VERB_FACTS = [
    "Browse {x} by {group}; filters are kept for next time.", "Review {x} before {date}; anything unreviewed is archived.",
    "Find any {x1} by name, date or {role}.", "Keep track of {x} offline; changes sync when you reconnect.",
    "Manage who can edit {x} from the Members page.", "Look through {x} from before {year} in the archive.",
    "Check on {x} shared with you; they are read-only.", "See {x} from other {groups} only when they share them.",
    "View {x} on a map from the menu.", "Organise {x} into folders; each {x1} can sit in one.",
    "Manage {x} for your whole {group} from one login.", "Browse {x} offline for up to {n} days.",
]
LIST_FACTS_ALT = [
    "Anything older than {n} months moves to the archive; nothing is deleted.",
    "Items you remove stay in the bin for {n} days.",
    "Only owners and {roles} can make changes here; everyone else can view.",
    "Shared with everyone in your {group}.",
    "Times are shown in {zone}.",
    "Changes reach your other devices within a minute.",
    "Updated every {n} minutes; drafts are not counted.",
    "Entries from before {year} are kept read-only.",
    "Private entries are hidden from everyone but you.",
    "Approval from {a_role} is needed before anything new appears.",
    "Everything here is backed up every night.",
    "The total excludes anything still in draft.",
    "Make sure each entry has a date; undated ones are skipped.",
    "Only {roles} can confirm new entries.",
]


def fill_fact(rng, t: str, x: str, x1: str, ctx: dict) -> str:
    n = rng.pick([2, 3, 5, 6, 7, 10, 12, 14, 18, 21, 24, 30, 45, 60, 90])
    vals = {"X": x[:1].upper() + x[1:], "x": x, "x1": x1, "n": n, "year": rng.pick([2019, 2020, 2021, 2022, 2023, 2024]), "date": date(rng),
            "zone": rng.pick(ZONES), "plan": rng.pick(PLANS[:5])}
    vals.update(ctx)
    if "role" in vals:
        vals["roles"] = plural(vals["role"])
    if "group" in vals:
        vals["groups"] = plural(vals["group"])
    for k in ("x1", "role", "group"):
        if k in vals:
            vals["a_" + k] = article(vals[k])
    return t.format(**vals)


# ---------------------------------------------------------------------------------------------------------------
# Narration (R6, R16) and closing summaries (R17), built from parts.

TAP_VERBS = ["Click", "Tap", "Press", "Select", "Hit", "Choose"]
NARRATE_PURPOSE = ["to continue", "to get started", "when you are ready", "to go ahead", "to carry on", "to proceed",
                   "once you are happy", "to finish"]
NARRATE_FORM_OPEN = ["Fill in the form, then", "Enter your details and", "Complete the fields above, then",
                     "Make your changes, then", "Once you have filled everything in,", "When you are done,"]
NARRATE_DONE_OPEN = ["Make your changes, then", "When you are done,", "When you are ready,", "Once you have decided,"]
SCROLL_LINES = ["Scroll down to see more {x}.", "Scroll to see all your {x}.", "Swipe up for older {x}.",
                "Tap any {x1} to open it.", "Click a row to open that {x1}.", "Hover over {a_x1} to see more options.",
                "Use the arrows to page through your {x}.", "Pull down to refresh the list of {x}."]
SCROLL_GENERIC = ["Scroll to explore.", "Scroll down to read on.", "Keep scrolling to learn more.",
                  "Scroll for the full story.", "Swipe to see more."]
COUNT_LINES = ["Below are our {n} plans.", "Here are the {n} steps to get started.", "Choose from {n} options below.",
               "We’ve listed {n} reasons below.", "Below you’ll find {n} key features.", "The {n} cards below cover the basics.",
               "{N} quick stages to your first {x1}.", "{N} things {p} does for you, below.", "Here are {n} ways {p} helps.",
               "Pick one of the {n} plans below.", "These {n} cards explain the essentials."]
LAYOUT_LINES = ["The table below lists all your {x}.", "This page lists your {x}.", "Below you’ll find your recent {x}.",
                "On this screen you can see every {x1}.", "Here is a list of your {x}.", "The section below shows your latest {x}.",
                "This page contains everything about your {x}.", "The list underneath shows each {x1} in turn.",
                "Every {x1} appears as a row below."]
CLOSE_SPREAD_A = ["That’s everything you need", "All of this", "Put together, that’s {p}", "In short, everything",
                  "Together, these make {p}", "And that’s just the start", "That covers it", "Taken together, that is {p}"]
CLOSE_SPREAD_B = [", all in one place.", ", in one simple app.", " in a nutshell.", " to get it done.",
                  " the complete package.", " of what {p} can do.", ", from start to finish.", " for your {x}."]
CLOSE_HEAD = ["In short: {h}.", "{H}: that’s the promise.", "The upshot: {h}.", "Bottom line: {h}.",
              "{H}. That’s {p} in one line.", "One more time: {h}.", "{H}, every time."]
CLOSE_STEPS = ["That’s all it takes to get started.", "And that’s it: you’re set up.", "{N} steps, and you’re ready.",
               "That’s the whole process.", "Done: your first {x1} is in.", "That’s all there is to it."]
# v4: closing lines in the same frames that add a fact (KEEP)
CLOSE_KEEP = ["And that’s free for up to {n} {x}.", "And that’s on every plan, with no setup fee.",
              "That’s everything on the free plan; {plan} adds reports.", "That’s all included in the price, VAT too.",
              "That covers up to {n} {x}; beyond that, talk to us.", "All of this works offline for up to {n} days.",
              "In short, everything here works on Android too.", "Put together, that’s about {n} minutes saved per {x1}.",
              "Together, these cost {price} a month for the whole {group}.", "And that’s just the start: exports arrive next month."]
CLOSE_STEPS_KEEP = ["That’s all it takes; most {groups} finish in a day.", "And that’s it: your first {x1} is free.",
                    "That’s the whole process, and it works offline.", "That’s all there is to it; your {role} gets a copy.",
                    "And that’s it: you can undo any step for {n} days."]
CLOSE_FAQ_KEEP = ["That’s everything people usually ask; billing questions are on the pricing page.",
                  "That covers the questions we hear most; for anything else, phone us on weekdays.",
                  "Those are the common questions; {roles} have their own guide.",
                  "And that’s the lot: answers are reviewed every month."]
CLOSE_FAQ = ["That covers the questions we hear most.", "We hope that answers your questions.",
             "Those are the common questions, answered.", "That’s everything people usually ask.",
             "Those are the questions {who} ask most."]

# Static status claims and decorative pills (R11). The same words are used, bound to data, as KEEP (K2).
STATUS_WORDS = ["Live", "Online", "Active", "Open", "Available", "Synced", "Connected", "Running", "Up to date", "Ready"]
STATUS_GLYPH = ["● ", "", "", "◆ "]
PILL_WORDS = ["New", "Popular", "Featured", "Hot", "Beta", "Updated", "Pro", "Trending", "Top pick", "Fresh"]
PILL_GLYPH = ["", "", "✨ ", "🔥 ", "● "]
SIBLING_PILLS = ["All plans", "Pro plan", "Team plan", "Add-on", "Beta", "Coming in July", "iOS only", "New in 4.2",
                 "Web only", "Included free", "New", "Updated", "Popular", "Coming soon", "Android only", "Paid add-on"]
POPULAR = ["Most popular", "Best value", "Recommended", "Our pick", "Most chosen", "Staff pick", "Popular"]

# Notices (K6 once; later copies R14), compositional
NOTICE_KINDS = [
    (["No credit card required.", "No card needed.", "No card details needed to start."],),
    (["Free for {n} days.", "Try it free for {n} days.", "{n} days free, then {price} a month."],),
    (["Cancel any time.", "Cancel whenever you like.", "Leave any time, no notice period."],),
    (["Rated 4.{d} out of 5 by {N} customers.", "4.{d} stars from {N} reviews."],),
    (["Set up in under {n} minutes.", "{n} minutes to set up."],),
]

# Generic verbs for confirmation dialogs: (button verb, base, gerund, synonym phrase sharing no stem)
VERBS = {
    "Delete": ("delete", "Deleting", ["get rid of", "erase", "wipe"]),
    "Remove": ("remove", "Removing", ["take away", "drop", "get rid of"]),
    "Archive": ("archive", "Archiving", ["put away", "file away", "shelve"]),
    "Cancel": ("cancel", "Cancelling", ["call off", "scrap", "drop"]),
    "Withdraw": ("withdraw", "Withdrawing", ["pull out of", "back out of"]),
    "Unpublish": ("unpublish", "Unpublishing", ["take down", "pull"]),
    "Disconnect": ("disconnect", "Disconnecting", ["unlink", "cut off"]),
    "Reset": ("reset", "Resetting", ["wipe", "start over with"]),
    "Discard": ("discard", "Discarding", ["throw away", "drop"]),
}

# Icon-only buttons: tooltip, accessible name, and the icon each framework draws (lucide, HTML sprite id, SF Symbol).
ICON_ACTIONS = [
    ("Export as CSV", "Export", "Download", "download", "square.and.arrow.down"),
    ("Download {x} as CSV", "Download", "Download", "download", "arrow.down.circle"),
    ("Filter {x}", "Filter", "Filter", "filter", "line.3.horizontal.decrease.circle"),
    ("Print this page", "Print", "Printer", "print", "printer"),
    ("Copy share link", "Share", "Link", "link", "link"),
    ("Share with your {group}", "Share", "Share2", "share", "square.and.arrow.up"),
    ("Archive selected", "Archive", "Archive", "archive", "archivebox"),
    ("Refresh", "Refresh", "RefreshCw", "refresh", "arrow.clockwise"),
    ("Edit {x1}", "Edit", "Pencil", "edit", "pencil"),
    ("Delete selected", "Delete", "Trash2", "trash", "trash"),
    ("Search {x}", "Search", "Search", "search", "magnifyingglass"),
    ("Sort by date", "Sort", "ArrowUpDown", "sort", "arrow.up.arrow.down"),
    ("Add to favourites", "Favourite", "Star", "star", "star"),
    ("More options", "More", "MoreHorizontal", "more", "ellipsis.circle"),
]

# Time-range tabs for dashboards: varied sets
RANGE_SETS = [
    ["Today", "7 days", "30 days", "12 months"], ["Day", "Week", "Month", "Year"], ["This week", "Last week"],
    ["Week", "Month", "Quarter"], ["24 hours", "7 days", "90 days"], ["Today", "Yesterday", "This month"],
    ["Month", "Season", "All time"], ["Last 7 days", "Last 28 days", "Last 12 months", "All time"],
]

# Field-style kickers (LABELLING.md K1): a label that names the kind of value after it.
FIELD_KICKERS = ["Next {x1}", "Due", "Last {y1}", "Your plan", "Your next {x1}", "Your {group}", "Renews", "Owner",
                 "Booked for", "Status", "Your balance", "Your rota", "Started", "Location", "Assigned to"]


# ---------------------------------------------------------------------------------------------------------------
# v3: facts that a heading, label or figure states, with restatements that carry the same figure (R1, R3, R4, R5)
# and lines that add a different fact (KEEP). A restatement repeats the heading's figure, date, limit or condition
# and nothing else; it may use commas and semicolons, and it seldom says "you". {k} and {K} are the fact's own
# figure (digits, or its word), {m} a second figure, {price} a price, {o1}/{o2} opening times.

FACTS = [
    {"topic": ["Free plan", "The free tier"], "head": ["Free for up to {k} {x}", "Up to {k} {x}, free", "{k} {x} free, for good"],
     "echo": ["Keep up to {k} {x} without paying anything.", "The first {k} {x} cost nothing, on any account.",
              "Up to {k} {x}; no charge for any of them.", "{K} {x} at no cost, whoever you are."],
     "keep": ["Above {k}, it is {price} a month per {group}.", "Charities can keep twice as many.",
              "Past that, each extra {x1} is {price} a year."],
     "eb": ["{k} free", "Free to {k}"]},
    {"topic": ["Free trial", "Trying it out"], "head": ["{k} days free", "Try it free for {k} days", "A {k}-day free trial"],
     "echo": ["For the first {k} days, the bill is zero.", "{K} days, free of charge, from sign-up.",
              "Nothing to pay for {k} days; not a penny."],
     "keep": ["No card is needed, and the trial ends without charging you.", "The trial includes every paid feature.",
              "Want longer? Ask and we add a week."],
     "eb": ["{k} days free"]},
    {"topic": ["Reminders", "Never miss a {y1}"], "head": ["Reminders {k} days ahead", "A nudge {k} days before", "Reminded {k} days early"],
     "echo": ["A reminder arrives {k} days before, every time.", "You hear about each {y1} {k} days in advance.",
              "{K} days before each {y1}, a reminder goes out."],
     "keep": ["By email or push, never both; change it for each {x1}.", "Quiet hours, 9pm to 8am, hold them back.",
              "Your {role} gets a copy."],
     "eb": ["{k} days ahead"]},
    {"topic": ["Opening hours", "When we are open"], "head": ["Open {o1} to {o2}, every day", "Open daily, {o1} to {o2}", "{o1} to {o2}, seven days a week"],
     "echo": ["From {o1} until {o2}, seven days a week.", "Every day of the week, {o1} to {o2}.",
              "Doors open at {o1} and close at {o2}, daily."],
     "keep": ["Bank holidays: 10am to 4pm.", "Last entry is an hour before closing.", "Quietest before 11am."],
     "eb": ["Daily, {o1}–{o2}"]},
    {"topic": ["Support", "Talk to a person"], "head": ["Replies within {k} hours", "An answer in {k} hours", "Help within {k} hours"],
     "echo": ["Every email is answered in {k} hours or less.", "Write to us and hear back inside {k} hours.",
              "Within {k} hours, someone replies; always."],
     "keep": ["Messages sent at the weekend are answered on Monday.", "Pro plans also get a phone line until 8pm.",
              "A real person, never a bot."],
     "eb": ["{k}-hour replies"]},
    {"topic": ["Booking", "Plan ahead"], "head": ["Book up to {k} weeks ahead", "Slots open {k} weeks out", "{k} weeks of dates to choose from"],
     "echo": ["Slots open {k} weeks in advance.", "You can reserve as far as {k} weeks out.",
              "Dates appear {k} weeks ahead, on a rolling basis."],
     "keep": ["Members get a two-day head start on new dates.", "Cancel free until 24 hours before.",
              "Only you can see your bookings."],
     "eb": ["{k} weeks ahead"]},
    {"topic": ["Delivery", "Getting it to you"], "head": ["Delivered in {k} working days", "At your door in {k} working days", "{k}-day delivery"],
     "echo": ["Orders arrive within {k} working days.", "Expect it {k} working days after you order.",
              "{K} working days, door to door."],
     "keep": ["Orders placed after 2pm leave the next morning.", "Tracking links arrive by text.",
              "Shared with your {group}, so anyone can sign for it."],
     "eb": ["{k}-day delivery"]},
    {"topic": ["Price", "What it costs"], "head": ["{price} a month per {group}", "One price: {price} a month", "{price} a month, all in"],
     "echo": ["Every {group} pays {price} each month.", "Monthly, it is {price}, the same for everyone.",
              "{price} a month; that is the whole price."],
     "keep": ["Charities pay half; prices include VAT.", "Pay yearly and get two months free.",
              "Want a quote for a bigger {group}? Ask us."],
     "eb": ["{price} a month"]},
    {"topic": ["History", "Your records"], "head": ["{X} kept for {k} months", "{k} months of history", "Every {x1} kept {k} months"],
     "echo": ["We hold on to {x} for {k} months.", "Each {x1} stays with us for {k} months.",
              "For {k} months, nothing is thrown away."],
     "keep": ["After that they are deleted, backups included.", "Download them as CSV before then.",
              "Only you can see this history."],
     "eb": ["{k} months"]},
    {"topic": ["Cancellations", "Changing your plans"], "head": ["Cancel free up to {k} hours before", "Free changes until {k} hours before", "{k} hours to change your mind"],
     "echo": ["Cancelling costs nothing until {k} hours beforehand.", "Change your mind up to {k} hours ahead, free.",
              "Up to {k} hours before, a cancellation is free."],
     "keep": ["Later than that, the fee becomes credit for {m} months.", "Your {role} is told straight away.",
              "Swaps with another {who1} are always free."],
     "eb": ["{k} hours’ notice"]},
    {"topic": ["Sync", "Every device, up to date"], "head": ["Changes sync in {k} seconds", "Synced in {k} seconds", "{k}-second sync"],
     "echo": ["Edits reach every phone within {k} seconds.", "Updates show up elsewhere {k} seconds later.",
              "In {k} seconds or so, every device has the change."],
     "keep": ["Offline edits wait and sync when you reconnect.", "Clashes are flagged, never overwritten.",
              "Works without signal."],
     "eb": ["{k}-second sync"]},
    {"topic": ["Group size", "Who can come"], "head": ["Groups of up to {k}", "No more than {k} to a group", "Small groups, {k} at most"],
     "echo": ["Each session takes {k} people at most.", "No more than {k} in a group, ever.",
              "A group is capped at {k}; never more."],
     "keep": ["Under-16s come with an adult.", "Private bookings can go up to {m}.", "Bring your own {y}, or borrow ours."],
     "eb": ["Up to {k}"]},
    {"topic": ["Storage", "Room for photos"], "head": ["{k} GB of storage", "Room for {k} GB", "{k} GB for your files"],
     "echo": ["There is space for {k} GB of files.", "Store up to {k} GB, all told.", "{K} GB; that is the allowance."],
     "keep": ["Photos are kept at full size.", "Extra space is {price} per 100 GB.", "Deleted files stop counting at once."],
     "eb": ["{k} GB"]},
    {"topic": ["Charities", "Good causes"], "head": ["{k}% off for charities", "Charities pay {k}% less", "{k}% charity discount"],
     "echo": ["Registered charities pay {k}% less.", "Charities save {k}%, on any plan.", "For charities, {k}% comes off."],
     "keep": ["Send us your charity number once; it applies at renewal.", "The discount stacks with yearly billing.",
              "Schools count too."],
     "eb": ["{k}% off"]},
    {"topic": ["Entries", "How to enter"], "head": ["Entries close {day} {date}", "Last entries {day} {date}", "Closing date: {day} {date}"],
     "echo": ["The last day to enter is {day} {date}.", "After {day} {date}, entries are closed.",
              "{day} {date}; no entries after that."],
     "keep": ["Late entries go on a waiting list.", "Results are posted the following week.", "Your {group} is told by email."],
     "eb": ["Until {date}"]},
    {"topic": ["Coverage", "Where we go"], "head": ["Within {k} miles of {place}", "{k} miles around {place}", "Covering {place} and {k} miles out"],
     "echo": ["We cover a {k}-mile radius around {place}.", "Anywhere up to {k} miles from {place}, we come.",
              "{place}, plus {k} miles in every direction."],
     "keep": ["Further out, delivery is {price}.", "Postcodes outside it can join the waiting list.",
              "Islands included."],
     "eb": ["{k} miles"]},
]
OPEN_TIMES = [("7am", "10pm"), ("8am", "6pm"), ("9am", "5pm"), ("6am", "9pm"), ("10am", "4pm"), ("7:30am", "8pm")]

# Short plain lines that add one fact and carry no figure (KEEP), by the place they fill; each pool holds only lines
# that cannot contradict the copy around them.
SHORT_KEEP = {
    "list": ["Sorted by date, newest first.", "Drafts are not shown.", "Archived {x} are hidden.", "Updated every hour.",
             "Synced to your other devices.", "Read-only for {who} outside the {group}.", "Oldest at the bottom.",
             "Your {role} sees the same list.", "Shared with your {group}."],
    "settings": ["Changes apply straight away.", "Applies from tomorrow.", "Synced to your other devices.",
                 "Your {role} can see these too.", "Saved as you go."],
    "field": ["Can be changed later.", "Checked before saving.", "Saved as you type."],
    "toggle": ["Only on this device.", "Applies to every device you use.", "Changes apply straight away.", "Applies from tomorrow.",
               "Off by default.", "Your {role} can turn it off for everyone."],
    "stat": ["Excludes drafts.", "Updated every hour.", "Checked each morning.", "Test data left out."],
}

# Field labels that carry their own limit or format, a helper that only repeats it (R5) and one that adds a fact
# (K4); placeholders that repeat the limit (R8).
CONSTRAINT_FIELDS = [
    ("Bio (up to {k} characters)", [160, 200, 280], ["No more than {k} characters.", "{K} characters at most, spaces included."],
     ["Shown on your public profile.", "Other {who} see it beside your name."], ["Up to {k} characters", "Max. {k} characters"], ["e.g. Weekend {who1} in {place}", "e.g. Learning since {year}"]),
    ("Password, {k} characters or more", [10, 12, 14], ["Use at least {k} characters.", "{K} characters is the minimum."],
     ["Spaces are allowed; avoid one you use elsewhere.", "We never email it to you."], ["{k}+ characters", "At least {k} characters"], ["e.g. three random words"]),
    ("Group size ({k1} to {k})", [6, 8, 10, 12], ["Between {k1} and {k} people.", "From {k1} up to {k}, no more."],
     ["Bigger groups can book the whole room.", "Under-16s count as half."], ["{k1} to {k}", "{k1}–{k} people"], ["e.g. 4", "e.g. 5"]),
    ("Display name (max {k} characters)", [20, 24, 30], ["{K} characters at most.", "No longer than {k} characters."],
     ["Other members see it beside your comments.", "Must be unique in your {group}."], ["Max. {k} characters", "Up to {k} characters"], ["e.g. {person}"]),
    ("PIN ({k} digits)", [4, 6], ["Exactly {k} digits.", "{K} digits, no letters."],
     ["Staff can reset it at the desk.", "Asked for at the self-service kiosk."], ["{k} digits", "Enter {k} digits"], ["e.g. {pin}"]),
    ("Phone number (UK mobile)", [0], ["A UK mobile number.", "Only UK mobile numbers."],
     ["Only used for sign-in codes.", "We text you on the morning of each {y1}."], ["UK mobile", "07… UK mobile"], ["e.g. 07700 900123", "e.g. 07700 900456"]),
    ("Postcode (UK only)", [0], ["UK postcodes only.", "A postcode in the UK."],
     ["Used to work out delivery days.", "It sets which branch your {x} go to."], ["UK postcode", "UK only"], ["e.g. BS3 4QT", "e.g. LS6 2AB"]),
    ("Weight in kg", [0], ["In kilograms.", "Kilograms, not pounds."],
     ["Round to the nearest half kilo.", "Used for the ferry’s load sheet."], ["kg", "Weight, kg"], ["e.g. 62", "e.g. 4.5"]),
    ("Notes (optional)", [0], ["This one is optional.", "Not required."],
     ["Visible to your {role} only.", "Printed on the summary sheet."], ["Optional", "Optional notes"], ["e.g. Gate code is on the fence", "e.g. Back door sticks"]),
    ("{X1} name (up to {k} characters)", [30, 40, 60], ["Keep it to {k} characters or fewer.", "{K} characters, at most."],
     ["Must be unique in your {group}.", "Shown on every reminder."], ["Up to {k} characters", "{k} characters max."], ["e.g. {item}"]),
    ("Start time (24-hour clock)", [0], ["Use the 24-hour clock.", "In 24-hour time."],
     ["Times follow the {group}’s time zone.", "Your {role} is told if it moves."], ["24-hour clock", "HH:MM, 24-hour"], ["e.g. 14:30", "e.g. 09:15"]),
]

CONSTRAINT_PLAIN = ["Bio", "Password", "Group size", "Display name", "PIN", "Phone number", "Postcode", "Weight", "Notes",
                    "{X1} name", "Start time"]

# Toggles whose label carries a figure, with a helper that repeats it (R5) or adds a fact (K4).
FIGURE_TOGGLES = [
    ("Remind me {k} days before", [2, 3, 5, 7], ["A reminder goes out {k} days ahead.", "{K} days before, you get a nudge."],
     ["Reminders follow your time zone.", "Your {role} gets the same reminder."]),
    ("Daily digest at {clock}", [0], ["One email each day at {clock}.", "At {clock}, every day, one email."],
     ["Skipped on days with nothing new.", "Lists only {x} that changed."]),
    ("Sign out after {k} days", [7, 14, 30], ["After {k} days, you are signed out.", "{K} days, then a fresh sign-in."],
     ["Only on this device.", "Clearing cookies signs you out sooner."]),
    ("Archive after {k} months", [3, 6, 12], ["Things move to the archive after {k} months.", "{K} months on, it is archived."],
     ["Nothing is deleted.", "Archived {x} still count in reports."]),
]

FIGURE_TOGGLES_PLAIN = ["Reminders", "Daily digest", "Automatic sign-out", "Auto-archive"]


# Mono, upper-case meta lines that state a date, issue or version (KEEP; K1 for dates and issues).
def keep_meta(rng, game=False) -> list:
    m = rng.pick(MONTHS)
    if game:
        return [rng.pick([f"BUILD {rng.randrange(2024, 2027)}.{rng.randrange(1, 13):02d}.{rng.randrange(1, 29):02d}",
                          f"V{rng.randrange(0, 3)}.{rng.randrange(0, 20)}.{rng.randrange(0, 10)}",
                          f"PATCH {rng.randrange(1, 5)}.{rng.randrange(0, 10)} · {rng.randrange(1, 29)} {m[:3].upper()}",
                          f"SEASON {rng.randrange(1, 8)} · WEEK {rng.randrange(1, 13)}",
                          f"DAY {rng.randrange(2, 90)} · {clock(rng)}"])]
    return rng.pick([[f"Issue {rng.randrange(2, 80)}", f"{m} 2026"], [f"Release 4.{rng.randrange(1, 12)}", f"{rng.randrange(1, 29)} {m} 2026"],
                     [f"Updated {rng.randrange(1, 29)} {m} 2026"], [f"No. {rng.randrange(10, 99)}", f"{rng.pick(['Spring', 'Summer', 'Autumn', 'Winter'])} 2026"],
                     [f"v{rng.randrange(1, 6)}.{rng.randrange(0, 20)}.{rng.randrange(0, 10)}", f"{m} 2026"]])


# Footers: column headings vary by product, and each column draws its links from a pool.
FOOTER_POOL = [
    (["Product", "The app", "Explore", "Tour", "Inside"], ["Features", "Pricing", "Changelog", "Roadmap", "Download", "Integrations", "What’s new"]),
    (["Company", "Who we are", "The team", "Us", "Studio"], ["About", "Careers", "Press", "Contact", "Blog", "Partners"]),
    (["Resources", "Learn", "Guides", "Help", "Support"], ["Docs", "Guides", "Status", "Community", "Help centre", "Webinars", "Templates"]),
    (["Legal", "Policies", "Small print", "Trust"], ["Privacy", "Terms", "Cookies", "Licences", "Security", "Accessibility"]),
    (["For {who}", "Who it is for", "Use cases"], ["For {groups}", "For {roles}", "For schools", "For charities", "For clubs"]),
    (["Follow", "Elsewhere", "Social"], ["Newsletter", "Mastodon", "Instagram", "YouTube", "RSS"]),
    (["Get the app", "Apps", "Download"], ["iPhone", "Android", "Web app", "Mac", "Windows"]),
]
LEGAL_V3 = ["© {year} {p} Ltd. Registered in England and Wales, no. {cno}.", "© {year} {p}. All prices include VAT.",
            "{p} is a trading name of {firm}, company no. {cno}.", "© {year} {firm}. VAT GB{vat}.",
            "{p} Ltd, {num} {street}, {place}. Company {cno}.", "Made in {place} by {firm}.", "© {p} {year}. Registered charity {charity}.",
            "{firm}, registered in Scotland, SC{cno6}.", "© {year} {p}"]
STREETS = ["Mill Lane", "Quay Street", "Station Road", "Market Place", "Church Street", "Victoria Road", "Harbour Row", "King Street"]
FIRM_TAIL = ["Ltd", "Limited", "LLP", "Studio Ltd", "Labs Ltd", "& Co"]


def legal_line(rng, product: str) -> str:
    firm = f"{rng.pick(NAME_A)}{rng.pick(NAME_B)} {rng.pick(FIRM_TAIL)}"
    return rng.pick(LEGAL_V3).format(year=rng.pick([2025, 2026]), p=product, cno=f"{rng.randrange(10**7, 10**8)}", firm=firm,
                                     vat=f"{rng.randrange(10**8, 10**9)}", num=rng.randrange(1, 120), street=rng.pick(STREETS),
                                     place=rng.pick(PLACES), charity=f"{rng.randrange(10**6, 10**7)}", cno6=f"{rng.randrange(10**5, 10**6)}")
