"""TSX renderer for the dev generator, in four styling idioms: Tailwind, CSS modules, shadcn-like components and
inline styles. Writes the component through ``Writer`` objects so that every slot's span is known.

The file is assembled from three parts, imports, module-level constants (arrays for ``.map()``) and the component,
each written by its own writer; spans in the later parts are shifted by the line counts of the earlier ones.
"""

from __future__ import annotations

from .core import E, GenError, Page, Slot, Writer, ident, slug

ACCENTS = ["emerald", "sky", "indigo", "amber", "rose", "teal", "violet", "lime", "orange", "cyan"]
IDIOMS = ["tailwind", "cssmod", "shadcn", "inline"]

# CSS module class names: a look -> candidate class names. Names in the second list are not semantic, so their
# styling has to carry the role (uppercase and letter-spacing for an eyebrow, small type for a subtitle).
CSSMOD_NAMES = {
    "eyebrow": ["eyebrow", "kicker", "overline", "sectionLabel", "preheading"],
    "sub": ["subtitle", "lede", "intro", "summary", "standfirst"],
    "badge": ["badge", "pill", "tag", "chip"],
    "note": ["note", "smallPrint", "finePrint", "hint"],
    "card": ["card", "featureCard", "tile", "panelCard"],
}
CSSMOD_RULES = {
    "eyebrow": "text-transform: uppercase; letter-spacing: 0.14em; font-size: 0.75rem; font-weight: 600;",
    "sub": "font-size: 0.875rem; color: var(--muted); max-width: 40rem;",
    "badge": "display: inline-block; border-radius: 999px; padding: 2px 10px; font-size: 0.75rem;",
    "note": "font-size: 0.75rem; color: var(--muted);",
    "card": "border: 1px solid var(--line); border-radius: 12px; padding: 24px;",
    "h1": "font-size: 3rem; line-height: 1.1;", "h2": "font-size: 2rem;", "h3": "font-size: 1.125rem;", "h4": "font-size: 1rem;",
    "p": "line-height: 1.6;", "button": "border-radius: 8px; padding: 8px 16px; background: var(--accent); color: white;",
    "link": "color: var(--accent);", "section": "padding: 80px 24px;", "grid": "display: grid; gap: 24px;",
    "meta": "font-family: ui-monospace, monospace; font-size: 0.75rem; text-transform: uppercase;",
    "stat": "font-size: 2rem; font-weight: 700;", "dot": "width: 8px; height: 8px; border-radius: 50%;",
    "hero": "padding: 96px 24px; text-align: center;", "stack": "display: grid; gap: 8px;",
    "actions": "display: flex; gap: 12px; margin-top: 24px;", "row": "display: flex; gap: 12px;",
    "list": "display: grid; gap: 8px;", "secondary": "border: 1px solid var(--line); border-radius: 8px; padding: 8px 16px;",
}


def jsx_text(s: str) -> str:
    if any(c in s for c in "{}<>"):
        raise GenError(f"copy not writable as JSX text: {s!r}")
    return s


def js_str(s: str) -> str:
    if '"' in s or "\\" in s:
        raise GenError(f"copy not writable as a string literal: {s!r}")
    return s


class TsxRenderer:
    def __init__(self, rng, idiom: str):
        self.rng = rng
        self.idiom = idiom
        self.accent = rng.pick(ACCENTS)
        self.imports: dict[str, list[str]] = {}
        self.css_classes: dict[str, str] = {}  # class name -> rule
        self.consts = Writer()
        self.const_names: set[str] = set()
        self.w = Writer()
        self.ids = 0
        self.icons: list[str] = []
        self.wrap_long = rng.chance(0.3)
        self.eyebrow_var = rng.randrange(3)
        self.card_var = rng.randrange(3)
        self.cssnames = {k: rng.pick(v) for k, v in CSSMOD_NAMES.items()}
        self.use_map = rng.chance(0.45)

    # -- helpers -------------------------------------------------------------------------------------------------

    def imp(self, module: str, name: str):
        names = self.imports.setdefault(module, [])
        if name not in names:
            names.append(name)

    def icon(self, name: str) -> str:
        self.imp("lucide-react", name)
        return f'<{name} className="size-4" aria-hidden="true" />'

    def uid(self, base: str) -> str:
        self.ids += 1
        return f"{slug(base)[:18]}-{self.ids}"

    def look(self, kind: str, extra: str = "") -> str:
        """The class (or style) attribute for a look, in this file's idiom, with a leading space."""
        a = self.accent
        if self.idiom == "cssmod":
            name = self.cssnames.get(kind, {"h1": "title", "h2": "heading", "h3": "cardTitle", "h4": "minorTitle",
                                             "p": "body", "button": "button", "link": "link", "section": "section",
                                             "grid": "grid", "meta": "meta", "stat": "figure", "dot": "dot",
                                             "hero": "hero", "row": "row", "stack": "stack", "list": "list",
                                             "actions": "actions", "secondary": "secondaryButton"}.get(kind, kind))
            rule = CSSMOD_RULES.get(kind, "margin: 0;")
            self.css_classes.setdefault(name, rule)
            return f" className={{styles.{name}}}"
        if self.idiom == "inline":
            st = {
                "eyebrow": 'fontSize: 12, textTransform: "uppercase", letterSpacing: "0.14em", fontWeight: 600',
                "sub": 'fontSize: 14, color: "#6b7280", marginTop: 8',
                "note": 'fontSize: 12, color: "#6b7280"',
                "h1": "fontSize: 44, fontWeight: 700, margin: 0", "h2": "fontSize: 30, fontWeight: 600, margin: 0",
                "h3": "fontSize: 18, fontWeight: 600", "h4": "fontSize: 16, fontWeight: 600",
                "p": "lineHeight: 1.6, marginTop: 12", "stat": "fontSize: 32, fontWeight: 700",
                "meta": 'fontFamily: "ui-monospace, monospace", fontSize: 12, textTransform: "uppercase"',
                "section": 'padding: "80px 24px"', "grid": 'display: "grid", gap: 24', "row": 'display: "flex", gap: 12',
                "card": 'border: "1px solid #e5e7eb", borderRadius: 12, padding: 24',
                "button": 'padding: "8px 16px", borderRadius: 8, background: "#111", color: "#fff"',
                "secondary": 'padding: "8px 16px", borderRadius: 8, border: "1px solid #d4d4d8"',
                "link": 'color: "#2563eb"', "stack": 'display: "flex", flexDirection: "column", gap: 8',
            }.get(kind)
            cls = {"badge": "badge", "card": "card", "dot": "dot"}.get(kind)
            out = ""
            if cls:
                out += f' className="{cls}"'
            if st:
                out += f" style={{{{ {st} }}}}"
            return out
        tw = {
            "eyebrow": [f"text-xs font-semibold uppercase tracking-widest text-{a}-600",
                        f"text-sm font-medium uppercase tracking-wide text-{a}-700",
                        "font-mono text-xs uppercase tracking-[0.2em] text-zinc-500"][self.eyebrow_var],
            "h1": "text-4xl font-bold tracking-tight sm:text-5xl", "h2": "text-3xl font-semibold tracking-tight",
            "h3": "text-lg font-semibold", "h4": "text-base font-medium",
            "sub": "mt-3 text-lg text-muted-foreground", "p": "mt-4 leading-7 text-zinc-700",
            "note": "mt-3 text-xs text-zinc-500",
            "button": f"rounded-md bg-{a}-600 px-4 py-2 text-sm font-medium text-white",
            "secondary": "rounded-md border px-4 py-2 text-sm font-medium",
            "link": f"text-sm font-medium text-{a}-700 hover:underline",
            "badge": f"rounded-full bg-{a}-100 px-2.5 py-0.5 text-xs font-medium text-{a}-800",
            "card": ["rounded-xl border p-6", "rounded-2xl border bg-white p-6 shadow-sm", "rounded-lg border p-5"][self.card_var],
            "grid": "mt-10 grid gap-6 md:grid-cols-3", "section": "mx-auto max-w-6xl px-6 py-20",
            "hero": "mx-auto max-w-3xl px-6 py-24 text-center", "row": "mt-8 flex flex-wrap items-center gap-3",
            "stack": "space-y-2", "list": "mt-6 space-y-2", "meta": "flex gap-6 font-mono text-xs uppercase text-zinc-500",
            "stat": "text-3xl font-bold", "dot": "h-2 w-2 rounded-full bg-emerald-500", "actions": "mt-6 flex gap-3",
        }.get(kind, "")
        if extra:
            tw = (tw + " " + extra).strip()
        return f' className="{tw}"' if tw else ""

    def put(self, s: Slot) -> tuple:
        return (s, jsx_text(s.text))

    def text_el(self, ind, tag, s: Slot, attrs=""):
        """<tag attrs>TEXT</tag>, wrapped over lines when the text is long."""
        if self.wrap_long and len(s.text) > 70 and " " in s.text[30:]:
            cut = s.text.index(" ", 30)
            a, b = s.text[:cut], s.text[cut + 1:]
            self.w.ln(ind, f"<{tag}{attrs}>")
            self.w.raw("  " * (ind + 1))
            self.w.put(s, jsx_text(a) + "\n" + "  " * (ind + 1) + jsx_text(b))
            self.w.raw("\n")
            self.w.ln(ind, f"</{tag}>")
        else:
            self.w.ln(ind, f"<{tag}{attrs}>", self.put(s), f"</{tag}>")

    # -- page ----------------------------------------------------------------------------------------------------

    def render(self, page: Page):
        name = ident(page.name)
        props = page.props
        if props:
            sig = "{ " + ", ".join(props) + " }: " + name + "Props"
            pre = f"type {name}Props = {{ " + "; ".join(f"{p}: {self._ptype(p)}" for p in props) + " };\n\n"
        else:
            sig, pre = "", ""
        default = self.rng.chance(0.5)
        self.w.raw(pre)
        self.w.ln(0, f"export {'default ' if default else ''}function {name}({sig}) {{")
        self.w.ln(1, "return (")
        wrap = self.rng.pick(["main", "div", "section", "fragment"]) if len(page.body) > 1 else "none"
        if wrap == "fragment":
            self.w.ln(2, "<>")
        elif wrap != "none":
            self.w.ln(2, f"<{wrap}{self.look('stack') if wrap == 'div' else ''}>")
        for e in page.body:
            self.el(e, 3 if wrap != "none" else 2)
        if wrap == "fragment":
            self.w.ln(2, "</>")
        elif wrap != "none":
            self.w.ln(2, f"</{wrap}>")
        self.w.ln(1, ");")
        self.w.ln(0, "}")
        # assemble: imports, constants, component
        head = Writer()
        if self.idiom == "cssmod":
            head.ln(0, f'import styles from "./{name}.module.css";')
        for mod in sorted(self.imports):
            names = self.imports[mod]
            if mod == "next/link":
                head.ln(0, 'import Link from "next/link";')
            else:
                head.ln(0, f"import {{ {', '.join(names)} }} from \"{mod}\";")
        if head.parts:
            head.raw("\n")
        if self.consts.parts:
            self.consts.raw("\n")
        h_lines = head.text().count("\n")
        c_lines = self.consts.text().count("\n")
        src = head.text() + self.consts.text() + self.w.text()
        self.consts.shift(h_lines)
        self.w.shift(h_lines + c_lines)
        css = None
        if self.idiom == "cssmod":
            css = "".join(f".{k} {{ {v} }}\n" for k, v in self.css_classes.items())
        return src, css

    def _ptype(self, p):
        return {"online": "boolean", "status": "string", "saving": "boolean", "count": "number", "user": "User",
                "open": "boolean", "health": "number", "error": "string | null"}.get(p, "boolean")

    # -- elements ------------------------------------------------------------------------------------------------

    def el(self, e: E, ind: int):
        fn = getattr(self, "k_" + e.kind, None)
        if fn is None:
            raise GenError(f"tsx renderer has no element {e.kind!r}")
        fn(e, ind)

    def kids(self, e: E, ind: int):
        for k in e.kids:
            self.el(k, ind)

    def k_section(self, e, ind):
        tone = e["tone"] or "plain"
        tag = "footer" if tone == "footer" else "section"
        look = "hero" if tone == "hero" else "section"
        self.w.ln(ind, f"<{tag}{self.look(look)}>")
        self.kids(e, ind + 1)
        self.w.ln(ind, f"</{tag}>")

    def k_div(self, e, ind):
        self.w.ln(ind, f"<div{self.look(e['look'] or 'stack')}>")
        self.kids(e, ind + 1)
        self.w.ln(ind, "</div>")

    k_head = k_div

    def k_eyebrow(self, e, ind):
        self.text_el(ind, "p", e["s"], self.look("eyebrow"))

    def k_h(self, e, ind):
        lv = e["level"] or 2
        self.text_el(ind, f"h{lv}", e["s"], self.look(f"h{min(lv, 4)}"))

    def k_sub(self, e, ind):
        if self.idiom == "shadcn" and e["card"]:
            self.imp("@/components/ui/card", "CardDescription")
            self.text_el(ind, "CardDescription", e["s"])
        else:
            self.text_el(ind, "p", e["s"], self.look("sub"))

    def k_p(self, e, ind):
        self.text_el(ind, "p", e["s"], self.look("p"))

    def k_note(self, e, ind):
        self.text_el(ind, "p", e["s"], self.look("note"))

    def k_actions(self, e, ind):
        self.w.ln(ind, f"<div{self.look('actions' if self.idiom != 'tailwind' else 'row')}>")
        self.kids(e, ind + 1)
        self.w.ln(ind, "</div>")

    def k_button(self, e, ind):
        s, tip = e["s"], e["tip"]
        icon = self.icon(e["icon"]) + " " if e["icon"] else ""
        if self.idiom == "shadcn":
            self.imp("@/components/ui/button", "Button")
            variant = "" if e["primary"] is not False else ' variant="outline"'
            if tip is not None:
                self.w.raw("  " * ind + f"<Button{variant} title=\"")
                self.w.put(tip, js_str(tip.text))
                self.w.raw('">' + icon)
            else:
                self.w.raw("  " * ind + f"<Button{variant}>{icon}")
            self.w.put(s, jsx_text(s.text))
            self.w.raw("</Button>\n")
            return
        look = self.look("button" if e["primary"] is not False else "secondary")
        typ = ' type="button"' if self.rng.chance(0.5) else ""
        self.w.raw("  " * ind + f"<button{typ}{look}")
        if tip is not None:
            self.w.raw(' title="')
            self.w.put(tip, js_str(tip.text))
            self.w.raw('"')
        self.w.raw(">" + icon)
        self.w.put(s, jsx_text(s.text))
        self.w.raw("</button>\n")

    def k_link(self, e, ind):
        s = e["s"]
        href = e["href"] or "/" + slug(s.text)[:20]
        arrow = ""
        if e["arrow"]:
            arrow = " " + self.icon(self.rng.pick(["ArrowRight", "ChevronRight", "ArrowUpRight"]))
        tag = "Link" if self.rng.chance(0.5) else "a"
        if tag == "Link":
            self.imp("next/link", "default")
        self.w.ln(ind, f'<{tag} href="{href}"{self.look("link")}>', self.put(s), f"{arrow}</{tag}>")

    def k_iconbtn(self, e, ind):
        icon = self.icon(e["icon"] or "MoreHorizontal")
        tip, aria = e["tip"], e["aria"]  # the icon matches the action the tooltip names
        self.w.raw("  " * ind + "<button")
        if aria is not None:
            self.w.raw(' aria-label="')
            self.w.put(aria, js_str(aria.text))
            self.w.raw('"')
        self.w.raw(' title="')
        self.w.put(tip, js_str(tip.text))
        self.w.raw(f'" className="rounded p-2 hover:bg-zinc-100">{icon}</button>\n')

    def k_badge(self, e, ind):
        s = e["s"]
        bound = e["bound"]
        if self.idiom == "shadcn":
            self.imp("@/components/ui/badge", "Badge")
            v = f' variant={{{bound}}}' if bound else self.rng.pick(["", ' variant="secondary"', ' variant="outline"'])
            self.w.ln(ind, f"<Badge{v}>", self.put(s), "</Badge>")
            return
        if bound:
            self.imp("@/lib/utils", "cn")
            look = f' className={{cn("rounded-full px-2 py-0.5 text-xs font-medium", {bound} ? "bg-emerald-100 text-emerald-800" : "bg-zinc-100 text-zinc-600")}}'
        else:
            look = self.look("badge")
        self.w.ln(ind, f"<span{look}>", self.put(s), "</span>")

    def k_status(self, e, ind):
        s = e["s"]
        bound = e["bound"]
        if bound:
            self.imp("@/lib/utils", "cn")
            dot = f'<span className={{cn("h-2 w-2 rounded-full", {bound} ? "bg-emerald-500" : "bg-zinc-400")}} />'
        elif self.idiom == "tailwind" or self.idiom == "shadcn":
            dot = f'<span className="h-2 w-2 rounded-full bg-{self.rng.pick(["emerald", "green", "lime", "sky"])}-500" />'
        else:
            dot = f"<span{self.look('dot')} />"
        self.w.ln(ind, '<div className="flex items-center gap-2 text-sm">')
        self.w.ln(ind + 1, dot)
        self.w.ln(ind + 1, "<span>", self.put(s), "</span>")
        self.w.ln(ind, "</div>")

    def k_legend(self, e, ind):
        colours = ["bg-emerald-500", "bg-amber-500", "bg-rose-500", "bg-sky-500", "bg-zinc-400"]
        self.w.ln(ind, '<ul className="flex gap-4 text-xs">')
        for n, s in enumerate(e["items"]):
            self.w.ln(ind + 1, '<li className="flex items-center gap-1.5">')
            self.w.ln(ind + 2, f'<span className="h-2 w-2 rounded-full {colours[n % len(colours)]}" />')
            self.w.ln(ind + 2, "<span>", self.put(s), "</span>")
            self.w.ln(ind + 1, "</li>")
        self.w.ln(ind, "</ul>")

    # cards ------------------------------------------------------------------------------------------------------

    def k_grid(self, e, ind):
        cards = e.kids
        if self.use_map and e["mappable"] and len(cards) >= 2 and self._uniform(cards):
            self._mapped_grid(e, ind)
            return
        self.w.ln(ind, f"<div{self.look('grid')}>")
        for c in cards:
            self.el(c, ind + 1)
        self.w.ln(ind, "</div>")

    def _uniform(self, cards):
        sig = [tuple(k.kind for k in c.kids) for c in cards]
        if len(set(sig)) != 1:
            return False
        allowed = {"eyebrow", "h", "sub", "p", "badge", "note", "link"}
        return all(k.kind in allowed for k in cards[0].kids)

    def _mapped_grid(self, e, ind):
        cards = e.kids
        fields = []
        for k in cards[0].kids:
            base = {"eyebrow": "kicker", "h": "title", "sub": "summary", "p": "body", "badge": "badge", "note": "note",
                    "link": "cta"}[k.kind]
            name = base
            n = 2
            while name in fields:
                name = f"{base}{n}"
                n += 1
            fields.append(name)
        arr = ident(e["name"] or "items", upper_first=False)
        while arr in self.const_names:
            arr += "List"
        self.const_names.add(arr)
        cw = self.consts
        cw.ln(0, f"const {arr} = [")
        for c in cards:
            cw.raw("  { ")
            for j, (f, k) in enumerate(zip(fields, c.kids)):
                if j:
                    cw.raw(", ")
                cw.raw(f'{f}: "')
                cw.put(k["s"], js_str(k["s"].text))
                cw.raw('"')
            cw.raw(" },\n")
        cw.ln(0, "];")
        v = arr[0] if arr[0] not in "ie" else "item"
        self.w.ln(ind, f"<div{self.look('grid')}>")
        self.w.ln(ind + 1, f"{{{arr}.map(({v}) => (")
        card_open, card_close = self._card_tags()
        self.w.ln(ind + 2, card_open.replace(">", f" key={{{v}.{fields[0]}}}>", 1))
        for f, k in zip(fields, cards[0].kids):
            tag, look = self._card_child_tag(k)
            self.w.ln(ind + 3, f"<{tag}{look}>{{{v}.{f}}}</{tag.split(' ')[0]}>")
        self.w.ln(ind + 2, card_close)
        self.w.ln(ind + 1, "))}")
        self.w.ln(ind, "</div>")

    def _card_tags(self):
        if self.idiom == "shadcn":
            self.imp("@/components/ui/card", "Card")
            return "<Card className=\"p-6\">", "</Card>"
        return f"<div{self.look('card')}>", "</div>"

    def _card_child_tag(self, k):
        kind = k.kind
        if kind == "h":
            lv = k["level"] or 3
            return f"h{lv}", self.look(f"h{min(lv, 4)}")
        if kind == "badge":
            if self.idiom == "shadcn":
                self.imp("@/components/ui/badge", "Badge")
                return "Badge", ""
            return "span", self.look("badge")
        if kind == "sub":
            return "p", self.look("sub")
        if kind == "eyebrow":
            return "p", self.look("eyebrow")
        if kind == "note":
            return "p", self.look("note")
        if kind == "link":
            return "a href=\"#\"", self.look("link")
        return "p", self.look("p")

    def k_card(self, e, ind):
        if self.idiom == "shadcn" and e["shadcn"] is not False:
            self.imp("@/components/ui/card", "Card")
            self.w.ln(ind, '<Card className="p-6">')
            self.kids(e, ind + 1)
            self.w.ln(ind, "</Card>")
            return
        tag = self.rng.pick(["div", "article"]) if self.idiom != "cssmod" else "div"
        self.w.ln(ind, f"<{tag}{self.look('card')}>")
        self.kids(e, ind + 1)
        self.w.ln(ind, f"</{tag}>")

    def k_ul(self, e, ind):
        items = e["items"]
        tag = "ol" if e["ordered"] else "ul"
        mark = e["mark"]
        if self.use_map and e["mappable"] and not mark and len(items) >= 2:
            arr = ident(e["name"] or "points", upper_first=False)
            while arr in self.const_names:
                arr += "List"
            self.const_names.add(arr)
            self.consts.ln(0, f"const {arr} = [")
            for s in items:
                self.consts.ln(1, '"', (s, js_str(s.text)), '",')
            self.consts.ln(0, "];")
            self.w.ln(ind, f"<{tag}{self.look('list')}>")
            self.w.ln(ind + 1, f"{{{arr}.map((line) => (")
            self.w.ln(ind + 2, "<li key={line}>{line}</li>")
            self.w.ln(ind + 1, "))}")
            self.w.ln(ind, f"</{tag}>")
            return
        self.w.ln(ind, f"<{tag}{self.look('list')}>")
        for s in items:
            if mark == "check":
                self.w.ln(ind + 1, '<li className="flex gap-2">')
                self.w.ln(ind + 2, self.icon("Check"))
                self.w.ln(ind + 2, "<span>", self.put(s), "</span>")
                self.w.ln(ind + 1, "</li>")
            else:
                self.w.ln(ind + 1, "<li>", self.put(s), "</li>")
        self.w.ln(ind, f"</{tag}>")

    def k_steps(self, e, ind):
        self.w.ln(ind, f'<ol{self.look("grid")}>')
        for num, title, body in e["items"]:
            self.w.ln(ind + 1, "<li>")
            if num is not None:
                self.w.ln(ind + 2, f"<span{self.look('meta') if self.idiom != 'tailwind' else ' className=\"font-mono text-sm text-zinc-400\"'}>",
                          self.put(num), "</span>")
            self.w.ln(ind + 2, f"<h3{self.look('h3')}>", self.put(title), "</h3>")
            if body is not None:
                self.w.ln(ind + 2, f"<p{self.look('p')}>", self.put(body), "</p>")
            self.w.ln(ind + 1, "</li>")
        self.w.ln(ind, "</ol>")

    def k_quote(self, e, ind):
        self.w.ln(ind, f"<figure{self.look('card')}>")
        self.w.ln(ind + 1, "<blockquote>", self.put(e["s"]), "</blockquote>")
        self.w.ln(ind + 1, "<figcaption>", self.put(e["who"]), "</figcaption>")
        self.w.ln(ind, "</figure>")

    def k_faq(self, e, ind):
        self.w.ln(ind, '<div className="mt-8 divide-y">')
        for q, a in e["items"]:
            self.w.ln(ind + 1, '<details className="py-4">')
            self.w.ln(ind + 2, '<summary className="font-medium">', self.put(q), "</summary>")
            self.w.ln(ind + 2, f"<p{self.look('p')}>", self.put(a), "</p>")
            self.w.ln(ind + 1, "</details>")
        self.w.ln(ind, "</div>")

    # forms ------------------------------------------------------------------------------------------------------

    def k_form(self, e, ind):
        self.w.ln(ind, '<form className="space-y-6">')
        self.kids(e, ind + 1)
        self.w.ln(ind, "</form>")

    def k_field(self, e, ind):
        fid = self.uid(e["label"].text if e["label"] else "field")
        lab, ph, helper, err = e["label"], e["ph"], e["helper"], e["error"]
        typ = e["type"] or "text"
        shad = self.idiom == "shadcn"
        self.w.ln(ind, '<div className="grid gap-2">' if shad else "<div>")
        if lab is not None:
            if shad:
                self.imp("@/components/ui/label", "Label")
                self.w.ln(ind + 1, f'<Label htmlFor="{fid}">', self.put(lab), "</Label>")
            else:
                self.w.ln(ind + 1, f'<label htmlFor="{fid}" className="block text-sm font-medium">', self.put(lab), "</label>")
        tag = "Input" if shad else "input"
        if shad:
            self.imp("@/components/ui/input", "Input")
        if typ == "textarea":
            tag = "Textarea" if shad else "textarea"
            if shad:
                self.imp("@/components/ui/textarea", "Textarea")
            typ_attr = ""
        else:
            typ_attr = f' type="{typ}"' if typ != "text" else ""
        if ph is not None:
            self.w.raw("  " * (ind + 1) + f'<{tag} id="{fid}"{typ_attr} placeholder="')
            self.w.put(ph, js_str(ph.text))
            self.w.raw('" />\n')
        else:
            self.w.ln(ind + 1, f'<{tag} id="{fid}"{typ_attr} />')
        if helper is not None:
            self.w.ln(ind + 1, '<p className="text-sm text-muted-foreground">', self.put(helper), "</p>")
        if err is not None:
            cond = e["err_cond"] or "errors." + ident(lab.text if lab else "field", upper_first=False)
            self.w.ln(ind + 1, f'{{{cond} && <p className="text-sm text-destructive">', self.put(err), "</p>}")
        self.w.ln(ind, "</div>")

    def k_select(self, e, ind):
        fid = self.uid(e["label"].text)
        shad = self.idiom == "shadcn" and self.rng.chance(0.6)
        self.w.ln(ind, "<div>")
        if shad:
            for n in ("Select", "SelectContent", "SelectItem", "SelectTrigger", "SelectValue"):
                self.imp("@/components/ui/select", n)
            self.imp("@/components/ui/label", "Label")
            self.w.ln(ind + 1, f'<Label htmlFor="{fid}">', self.put(e["label"]), "</Label>")
            self.w.ln(ind + 1, "<Select>")
            self.w.ln(ind + 2, f'<SelectTrigger id="{fid}">')
            self.w.ln(ind + 3, "<SelectValue />")
            self.w.ln(ind + 2, "</SelectTrigger>")
            self.w.ln(ind + 2, "<SelectContent>")
            for o in e["options"]:
                self.w.ln(ind + 3, f'<SelectItem value="{slug(o.text)}">', self.put(o), "</SelectItem>")
            self.w.ln(ind + 2, "</SelectContent>")
            self.w.ln(ind + 1, "</Select>")
        else:
            self.w.ln(ind + 1, f'<label htmlFor="{fid}" className="block text-sm font-medium">', self.put(e["label"]), "</label>")
            self.w.ln(ind + 1, f'<select id="{fid}" className="mt-1 rounded-md border px-3 py-2">')
            for o in e["options"]:
                self.w.ln(ind + 2, "<option>", self.put(o), "</option>")
            self.w.ln(ind + 1, "</select>")
        if e["helper"] is not None:
            self.w.ln(ind + 1, '<p className="text-sm text-muted-foreground">', self.put(e["helper"]), "</p>")
        self.w.ln(ind, "</div>")

    def k_toggle(self, e, ind):
        fid = self.uid(e["label"].text)
        if self.idiom == "shadcn":
            self.imp("@/components/ui/switch", "Switch")
            self.imp("@/components/ui/label", "Label")
            self.w.ln(ind, '<div className="flex items-center justify-between gap-4">')
            self.w.ln(ind + 1, "<div>")
            self.w.ln(ind + 2, f'<Label htmlFor="{fid}">', self.put(e["label"]), "</Label>")
            if e["helper"] is not None:
                self.w.ln(ind + 2, '<p className="text-sm text-muted-foreground">', self.put(e["helper"]), "</p>")
            self.w.ln(ind + 1, "</div>")
            self.w.ln(ind + 1, f'<Switch id="{fid}" />')
            self.w.ln(ind, "</div>")
            return
        self.w.ln(ind, "<div>")
        self.w.ln(ind + 1, '<label className="flex items-center gap-2">')
        self.w.ln(ind + 2, f'<input id="{fid}" type="checkbox" />')
        self.w.ln(ind + 2, "<span>", self.put(e["label"]), "</span>")
        self.w.ln(ind + 1, "</label>")
        if e["helper"] is not None:
            self.w.ln(ind + 1, '<p className="ml-6 text-sm text-muted-foreground">', self.put(e["helper"]), "</p>")
        self.w.ln(ind, "</div>")

    def k_errsummary(self, e, ind):
        self.w.ln(ind, '<div role="alert" className="rounded-md border border-red-300 p-4">')
        self.w.ln(ind + 1, '<h2 className="font-semibold">', self.put(e["title"]), "</h2>")
        self.w.ln(ind + 1, "<ul>")
        for s in e["items"]:
            self.w.ln(ind + 2, f'<li><a href="#{e["href"] or "field"}">', self.put(s), "</a></li>")
        self.w.ln(ind + 1, "</ul>")
        self.w.ln(ind, "</div>")

    # states -----------------------------------------------------------------------------------------------------

    def k_dialog(self, e, ind):
        title, body = e["title"], e["body"] or []
        if self.idiom == "shadcn":
            alert = e["confirm"]
            pre = "AlertDialog" if alert else "Dialog"
            mod = "@/components/ui/alert-dialog" if alert else "@/components/ui/dialog"
            for n in (pre, pre + "Content", pre + "Header", pre + "Title", pre + "Description", pre + "Footer"):
                self.imp(mod, n)
            self.w.ln(ind, f"<{pre} open={{open}}>")
            self.w.ln(ind + 1, f"<{pre}Content>")
            self.w.ln(ind + 2, f"<{pre}Header>")
            self.w.ln(ind + 3, f"<{pre}Title>", self.put(title), f"</{pre}Title>")
            for b in body:
                self.w.ln(ind + 3, f"<{pre}Description>", self.put(b), f"</{pre}Description>")
            self.w.ln(ind + 2, f"</{pre}Header>")
            for k in e.kids:
                if k.kind == "actions" and alert:
                    self.w.ln(ind + 2, f"<{pre}Footer>")
                    for b in k.kids:
                        btn = "AlertDialogCancel" if b["primary"] is False else "AlertDialogAction"
                        self.imp(mod, btn)
                        self.w.ln(ind + 3, f"<{btn}>", self.put(b["s"]), f"</{btn}>")
                    self.w.ln(ind + 2, f"</{pre}Footer>")
                elif k.kind == "actions":
                    self.w.ln(ind + 2, f"<{pre}Footer>")
                    for b in k.kids:
                        self.el(b, ind + 3)
                    self.w.ln(ind + 2, f"</{pre}Footer>")
                else:
                    self.el(k, ind + 2)
            self.w.ln(ind + 1, f"</{pre}Content>")
            self.w.ln(ind, f"</{pre}>")
            return
        tid = self.uid("dialog-title")
        self.w.ln(ind, f'<div role="dialog" aria-modal="true" aria-labelledby="{tid}" className="fixed inset-0 grid place-items-center bg-black/40">')
        self.w.ln(ind + 1, f"<div{self.look('card', 'w-full max-w-md bg-white')}>")
        self.w.ln(ind + 2, f'<h2 id="{tid}"{self.look("h3")}>', self.put(title), "</h2>")
        for b in body:
            self.w.ln(ind + 2, f"<p{self.look('p')}>", self.put(b), "</p>")
        self.kids(e, ind + 2)
        self.w.ln(ind + 1, "</div>")
        self.w.ln(ind, "</div>")

    def k_empty(self, e, ind):
        title, body, action = e["title"], e["body"], e["action"]
        if self.rng.chance(0.4) and (action is None or action.kind == "button"):
            self.imp("@/components/empty-state", "EmptyState")
            self.w.raw("  " * ind + "<EmptyState\n")
            self.w.ln(ind + 1, 'title="', (title, js_str(title.text)), '"')
            if body is not None:
                self.w.ln(ind + 1, 'description="', (body, js_str(body.text)), '"')
            if action is not None:
                self.imp("@/components/ui/button", "Button")
                self.w.ln(ind + 1, "action={<Button>", self.put(action["s"]), "</Button>}")
            self.w.ln(ind, "/>")
            return
        cls = self.rng.pick(["empty-state", "empty", "list-empty", "empty-state mx-auto max-w-sm text-center"])
        self.w.ln(ind, f'<div className="{cls}">')
        self.w.ln(ind + 1, f"<h3{self.look('h3')}>", self.put(title), "</h3>")
        if body is not None:
            self.w.ln(ind + 1, f"<p{self.look('sub')}>", self.put(body), "</p>")
        if action is not None:
            self.el(action, ind + 1)
        self.w.ln(ind, "</div>")

    def k_chart(self, e, ind):
        """A chart component fed by a prop: it renders no text of its own."""
        series = ident(e["series"] or "series", upper_first=False)
        comp = self.rng.pick(["BarChart", "LineChart", "AreaChart", "Sparkline"])
        self.imp("@/components/charts", comp)
        self.w.ln(ind, f"<{comp} data={{{series}}} height={{{self.rng.pick([120, 160, 200, 240])}}} />")
        if series not in self.const_names:
            self.const_names.add(series)
            self.consts.ln(0, f"const {series} = [{', '.join(str(self.rng.randrange(1, 60)) for _ in range(7))}];")

    def k_alert(self, e, ind):
        if e["tone"] == "error":
            self.w.ln(ind, '<div role="alert" className="rounded-md border border-red-300 bg-red-50 p-4 text-sm">')
            if e["title"] is not None:
                self.w.ln(ind + 1, '<p className="font-medium text-red-800">', self.put(e["title"]), "</p>")
            self.w.ln(ind + 1, '<p className="text-red-700">', self.put(e["s"]), "</p>")
            self.w.ln(ind, "</div>")
            return
        if self.idiom == "shadcn":
            for n in ("Alert", "AlertTitle", "AlertDescription"):
                self.imp("@/components/ui/alert", n)
            self.w.ln(ind, "<Alert>")
            if e["title"] is not None:
                self.w.ln(ind + 1, "<AlertTitle>", self.put(e["title"]), "</AlertTitle>")
            self.w.ln(ind + 1, "<AlertDescription>", self.put(e["s"]), "</AlertDescription>")
            self.w.ln(ind, "</Alert>")
            return
        self.w.ln(ind, '<div role="status" className="rounded-md border p-4 text-sm">')
        if e["title"] is not None:
            self.w.ln(ind + 1, '<p className="font-medium">', self.put(e["title"]), "</p>")
        self.w.ln(ind + 1, "<p>", self.put(e["s"]), "</p>")
        self.w.ln(ind, "</div>")

    def k_cond(self, e, ind):
        self.w.ln(ind, f"{{{e['expr']} ? (")
        self._branch(e["a"], ind + 1)
        self.w.ln(ind, ") : (")
        self._branch(e["b"], ind + 1)
        self.w.ln(ind, ")}")

    def _branch(self, els, ind):
        if len(els) == 1:
            self.el(els[0], ind)
        else:
            self.w.ln(ind, "<>")
            for x in els:
                self.el(x, ind + 1)
            self.w.ln(ind, "</>")

    def k_dyn(self, e, ind):
        """Interpolated text: ``pre{expr}post``. With no literal around it the span is the expression alone."""
        s = e["s"]
        tag = e["tag"] or "p"
        look = self.look(e["look"] or "p")
        pre, post = e["pre"] or "", e["post"] or ""
        self.w.raw("  " * ind + f"<{tag}{look}>" + ("" if pre else "{"))
        self.w.put(s, (pre + "{" if pre else "") + e["expr"] + ("}" + post if post else ""))
        self.w.raw(("" if post else "}") + f"</{tag}>\n")

    # navigation and data ----------------------------------------------------------------------------------------

    def k_nav(self, e, ind):
        self.w.ln(ind, '<nav className="flex items-center gap-6 text-sm">')
        for s in e["items"]:
            self.w.ln(ind + 1, f'<a href="/{slug(s.text)}">', self.put(s), "</a>")
        self.w.ln(ind, "</nav>")

    def k_tabs(self, e, ind):
        if self.idiom == "shadcn":
            for n in ("Tabs", "TabsList", "TabsTrigger"):
                self.imp("@/components/ui/tabs", n)
            self.w.ln(ind, f'<Tabs defaultValue="{slug(e["items"][0].text)}">')
            self.w.ln(ind + 1, "<TabsList>")
            for s in e["items"]:
                self.w.ln(ind + 2, f'<TabsTrigger value="{slug(s.text)}">', self.put(s), "</TabsTrigger>")
            self.w.ln(ind + 1, "</TabsList>")
            self.w.ln(ind, "</Tabs>")
            return
        self.w.ln(ind, '<div role="tablist" className="flex gap-2 border-b">')
        for n, s in enumerate(e["items"]):
            self.w.ln(ind + 1, f'<button role="tab" aria-selected={{{"true" if n == 0 else "false"}}}>', self.put(s), "</button>")
        self.w.ln(ind, "</div>")

    def k_crumbs(self, e, ind):
        if e["aria"] is not None:
            self.w.ln(ind, '<nav aria-label="', (e["aria"], js_str(e["aria"].text)), '">')
        else:
            self.w.ln(ind, "<nav>")
        self.w.ln(ind + 1, '<ol className="flex gap-2 text-sm text-zinc-500">')
        items = e["items"]
        for n, s in enumerate(items):
            if n == len(items) - 1:
                self.w.ln(ind + 2, '<li aria-current="page">', self.put(s), "</li>")
            else:
                self.w.ln(ind + 2, f'<li><a href="/{slug(s.text)}">', self.put(s), "</a></li>")
        self.w.ln(ind + 1, "</ol>")
        self.w.ln(ind, "</nav>")

    def k_table(self, e, ind):
        shad = self.idiom == "shadcn"
        if shad:
            for n in ("Table", "TableBody", "TableCell", "TableHead", "TableHeader", "TableRow"):
                self.imp("@/components/ui/table", n)
        T = (lambda a, b: b) if shad else (lambda a, b: a)
        self.w.ln(ind, f"<{T('table', 'Table')}{'' if shad else ' className=\"w-full text-sm\"'}>")
        self.w.ln(ind + 1, f"<{T('thead', 'TableHeader')}>")
        self.w.ln(ind + 2, f"<{T('tr', 'TableRow')}>")
        for s in e["headers"]:
            self.w.ln(ind + 3, f"<{T('th', 'TableHead')}>", self.put(s), f"</{T('th', 'TableHead')}>")
        self.w.ln(ind + 2, f"</{T('tr', 'TableRow')}>")
        self.w.ln(ind + 1, f"</{T('thead', 'TableHeader')}>")
        self.w.ln(ind + 1, f"<{T('tbody', 'TableBody')}>")
        for row in e["rows"]:
            self.w.ln(ind + 2, f"<{T('tr', 'TableRow')}>")
            for c in row:
                if isinstance(c, E):
                    self.w.ln(ind + 3, f"<{T('td', 'TableCell')}>")
                    self.el(c, ind + 4)
                    self.w.ln(ind + 3, f"</{T('td', 'TableCell')}>")
                else:
                    self.w.ln(ind + 3, f"<{T('td', 'TableCell')}>", self.put(c), f"</{T('td', 'TableCell')}>")
            self.w.ln(ind + 2, f"</{T('tr', 'TableRow')}>")
        self.w.ln(ind + 1, f"</{T('tbody', 'TableBody')}>")
        self.w.ln(ind, f"</{T('table', 'Table')}>")

    def k_stats(self, e, ind):
        self.w.ln(ind, f"<dl{self.look('grid')}>")
        for st in e.kids:
            self.el(st, ind + 1)
        self.w.ln(ind, "</dl>")

    def k_stat(self, e, ind):
        self.w.ln(ind, f"<div{self.look('card')}>")
        self.w.ln(ind + 1, '<dt className="text-sm text-zinc-500">', self.put(e["label"]), "</dt>")
        self.w.ln(ind + 1, f"<dd{self.look('stat')}>", self.put(e["value"]), "</dd>")
        if e["note"] is not None:
            self.w.ln(ind + 1, f"<dd{self.look('note')}>", self.put(e["note"]), "</dd>")
        self.w.ln(ind, "</div>")

    def k_price(self, e, ind):
        self.w.ln(ind, '<div className="mt-6">')
        self.w.ln(ind + 1, f"<p{self.look('stat')}>", self.put(e["amount"]), "</p>")
        self.w.ln(ind + 1, f"<p{self.look('note')}>", self.put(e["period"]), "</p>")
        self.w.ln(ind, "</div>")

    def k_meta(self, e, ind):
        self.w.ln(ind, f"<div{self.look('meta')}>")
        for s in e["items"]:
            self.w.ln(ind + 1, "<span>", self.put(s), "</span>")
        self.w.ln(ind, "</div>")

    def k_footer(self, e, ind):
        self.w.ln(ind, '<footer className="border-t py-12 text-sm">')
        self.w.ln(ind + 1, '<div className="grid grid-cols-2 gap-8 md:grid-cols-4">')
        for head, links in e["cols"]:
            self.w.ln(ind + 2, "<div>")
            self.w.ln(ind + 3, '<h4 className="font-semibold">', self.put(head), "</h4>")
            self.w.ln(ind + 3, '<ul className="mt-3 space-y-2">')
            for s in links:
                self.w.ln(ind + 4, f'<li><a href="/{slug(s.text)}">', self.put(s), "</a></li>")
            self.w.ln(ind + 3, "</ul>")
            self.w.ln(ind + 2, "</div>")
        self.w.ln(ind + 1, "</div>")
        if e["legal"] is not None:
            self.w.ln(ind + 1, '<p className="mt-10 text-xs text-zinc-500">', self.put(e["legal"]), "</p>")
        self.w.ln(ind, "</footer>")
