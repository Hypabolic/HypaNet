"""HTML renderer for the dev generator: whole documents or partial templates, styled with BEM classes and a
``<style>`` block, Tailwind classes, or inline ``style`` attributes."""

from __future__ import annotations

from .core import E, GenError, Page, Slot, Writer, slug

IDIOMS = ["bem", "tailwind", "inline"]

BEM_STYLE = {
    "eyebrow": "text-transform: uppercase; letter-spacing: .14em; font-size: 12px;",
    "kicker": "text-transform: uppercase; letter-spacing: .14em; font-size: 12px;",
    "overline": "text-transform: uppercase; letter-spacing: .2em; font-size: 11px;",
    "label-top": "text-transform: uppercase; letter-spacing: .12em; font-size: 12px;",
    "lede": "font-size: 14px; color: #666;", "intro": "font-size: 14px; color: #666;",
    "subtitle": "font-size: 14px; color: #666;", "note": "font-size: 12px; color: #777;",
    "badge": "border-radius: 999px; padding: 2px 8px; font-size: 12px;",
    "pill": "border-radius: 999px; padding: 2px 8px; font-size: 12px;",
    "card": "border: 1px solid #ddd; border-radius: 12px; padding: 24px;",
    "meta": "font-family: monospace; font-size: 11px; text-transform: uppercase;",
    "dot": "width: 8px; height: 8px; border-radius: 50%; background: #2a2;",
    "figure": "font-size: 32px; font-weight: 700;",
}


def html_text(s: str) -> str:
    if any(c in s for c in "<>"):
        raise GenError(f"copy not writable as HTML: {s!r}")
    return s.replace("&", "&amp;")


def attr_text(s: str) -> str:
    if '"' in s:
        raise GenError(f"copy not writable as an attribute: {s!r}")
    return html_text(s)


class HtmlRenderer:
    def __init__(self, rng, idiom: str):
        self.rng = rng
        self.idiom = idiom
        self.w = Writer()
        self.used: dict[str, str] = {}
        self.block = rng.pick(["hero", "section", "panel", "intro"])
        self.eyebrow_cls = rng.pick(["eyebrow", "kicker", "overline", "label-top"])
        self.sub_cls = rng.pick(["lede", "intro", "subtitle"])
        self.badge_cls = rng.pick(["badge", "pill"])
        self.ids = 0

    def uid(self, base):
        self.ids += 1
        return f"{slug(base)[:16]}-{self.ids}"

    def cls(self, look: str) -> str:
        if self.idiom == "tailwind":
            tw = {
                "eyebrow": "text-xs font-semibold uppercase tracking-widest text-teal-700",
                "sub": "mt-3 text-lg text-muted-foreground", "note": "mt-2 text-xs text-gray-500",
                "badge": "rounded-full bg-amber-100 px-2 py-0.5 text-xs", "card": "rounded-xl border p-6",
                "meta": "flex gap-4 font-mono text-xs uppercase text-gray-500", "stat": "text-3xl font-bold",
                "dot": "inline-block h-2 w-2 rounded-full bg-green-500", "h1": "text-5xl font-bold", "h2": "text-3xl font-semibold",
                "h3": "text-lg font-semibold", "button": "rounded bg-teal-700 px-4 py-2 text-white", "grid": "grid gap-6 md:grid-cols-3",
            }.get(look)
            return f' class="{tw}"' if tw else ""
        if self.idiom == "inline":
            st = {
                "eyebrow": "text-transform: uppercase; letter-spacing: 0.16em; font-size: 12px",
                "sub": "font-size: 14px; color: #666", "note": "font-size: 12px; color: #777",
                "meta": "font-family: monospace; font-size: 11px; text-transform: uppercase",
                "stat": "font-size: 32px", "h1": "font-size: 44px", "h2": "font-size: 28px",
            }.get(look)
            c = {"badge": "badge", "card": "card", "dot": "dot"}.get(look)
            out = f' class="{c}"' if c else ""
            if c:
                self.used[c] = BEM_STYLE.get(c, "")
            return out + (f' style="{st}"' if st else "")
        name = {
            "eyebrow": f"{self.block}__{self.eyebrow_cls}", "sub": f"{self.block}__{self.sub_cls}",
            "note": "note", "badge": self.badge_cls, "card": "card", "meta": "meta", "stat": "figure", "dot": "dot",
            "h1": f"{self.block}__title", "h2": "section__title", "h3": "card__title", "button": "btn btn--primary",
            "secondary": "btn btn--ghost", "grid": "grid", "p": "",
        }.get(look, "")
        if not name:
            return ""
        last = name.split("__")[-1].split(" ")[0]
        rule = BEM_STYLE.get(last)
        if rule:
            self.used[name.split(" ")[0]] = rule
        return f' class="{name}"'

    def put(self, s: Slot):
        return (s, html_text(s.text))

    # -- page ----------------------------------------------------------------------------------------------------

    def render(self, page: Page):
        body = Writer()
        self.w = body
        full = self.rng.chance(0.55)
        ind = 2 if full else 0
        if full:
            wrap = self.rng.pick(["main", "div", None])
        else:
            wrap = None
        if wrap:
            body.ln(1, f"<{wrap}>")
        for e in page.body:
            self.el(e, ind if not wrap else ind)
        if wrap:
            body.ln(1, f"</{wrap}>")
        if not full:
            src = body.text()
            if self.used and self.idiom != "tailwind":
                # partial templates carry their own small style block at the top
                head = Writer()
                head.ln(0, "<style>")
                for k, v in self.used.items():
                    head.ln(1, f".{k} {{ {v} }}")
                head.ln(0, "</style>")
                body.shift(head.text().count("\n"))
                src = head.text() + src
            return src, None
        head = Writer()
        head.ln(0, "<!doctype html>")
        head.ln(0, '<html lang="en-GB">')
        head.ln(0, "<head>")
        head.ln(1, '<meta charset="utf-8">')
        head.ln(1, '<meta name="viewport" content="width=device-width, initial-scale=1">')
        head.ln(1, f"<title>{html_text(page.product or page.name)}</title>")
        if self.idiom == "tailwind":
            head.ln(1, '<script src="https://cdn.tailwindcss.com"></script>')
        elif self.used:
            head.ln(1, "<style>")
            for k, v in self.used.items():
                head.ln(2, f".{k} {{ {v} }}")
            head.ln(1, "</style>")
        head.ln(0, "</head>")
        head.ln(0, "<body>")
        body.shift(head.text().count("\n"))
        tail = "</body>\n</html>\n"
        return head.text() + body.text() + tail, None

    # -- elements ------------------------------------------------------------------------------------------------

    def el(self, e: E, ind: int):
        fn = getattr(self, "k_" + e.kind, None)
        if fn is None:
            raise GenError(f"html renderer has no element {e.kind!r}")
        fn(e, ind)

    def kids(self, e, ind):
        for k in e.kids:
            self.el(k, ind)

    def k_section(self, e, ind):
        tone = e["tone"] or "plain"
        tag = "footer" if tone == "footer" else "section"
        c = f' class="{self.block}"' if tone == "hero" and self.idiom == "bem" else ""
        self.w.ln(ind, f"<{tag}{c}>")
        self.kids(e, ind + 1)
        self.w.ln(ind, f"</{tag}>")

    def k_div(self, e, ind):
        self.w.ln(ind, "<div>")
        self.kids(e, ind + 1)
        self.w.ln(ind, "</div>")

    k_head = k_div

    def k_eyebrow(self, e, ind):
        self.w.ln(ind, f"<p{self.cls('eyebrow')}>", self.put(e["s"]), "</p>")

    def k_h(self, e, ind):
        lv = e["level"] or 2
        self.w.ln(ind, f"<h{lv}{self.cls(f'h{lv}') if lv <= 3 else ''}>", self.put(e["s"]), f"</h{lv}>")

    def k_sub(self, e, ind):
        self.w.ln(ind, f"<p{self.cls('sub')}>", self.put(e["s"]), "</p>")

    def k_p(self, e, ind):
        self.w.ln(ind, "<p>", self.put(e["s"]), "</p>")

    def k_note(self, e, ind):
        tag = self.rng.pick(["p", "small"])
        self.w.ln(ind, f"<{tag}{self.cls('note')}>", self.put(e["s"]), f"</{tag}>")

    def k_actions(self, e, ind):
        self.w.ln(ind, '<div class="actions">')
        self.kids(e, ind + 1)
        self.w.ln(ind, "</div>")

    def k_button(self, e, ind):
        s, tip = e["s"], e["tip"]
        self.w.raw("  " * ind + f'<button type="button"{self.cls("button" if e["primary"] is not False else "secondary")}')
        if tip is not None:
            self.w.raw(' title="')
            self.w.put(tip, attr_text(tip.text))
            self.w.raw('"')
        self.w.raw(">")
        self.w.put(s, html_text(s.text))
        self.w.raw("</button>\n")

    def k_link(self, e, ind):
        s = e["s"]
        arrow = " &rarr;" if e["arrow"] else ""
        if arrow:
            self.w.ln(ind, f'<a href="/{slug(s.text)[:20]}">', (s, html_text(s.text) + arrow), "</a>")
        else:
            self.w.ln(ind, f'<a href="/{slug(s.text)[:20]}">', self.put(s), "</a>")

    def k_iconbtn(self, e, ind):
        tip, aria = e["tip"], e["aria"]
        self.w.raw("  " * ind + "<button")
        if aria is not None:
            self.w.raw(' aria-label="')
            self.w.put(aria, attr_text(aria.text))
            self.w.raw('"')
        self.w.raw(' title="')
        self.w.put(tip, attr_text(tip.text))
        self.w.raw(f'"><svg class="icon" aria-hidden="true"><use href="#i-{e["html_icon"] or "more"}"></use></svg></button>\n')

    def k_badge(self, e, ind):
        self.w.ln(ind, f"<span{self.cls('badge')}>", self.put(e["s"]), "</span>")

    def k_status(self, e, ind):
        if e["bound"]:
            raise GenError("bound status in static HTML")
        self.w.ln(ind, '<p class="status">')
        self.w.ln(ind + 1, f"<span{self.cls('dot')}></span>")
        self.w.ln(ind + 1, "<span>", self.put(e["s"]), "</span>")
        self.w.ln(ind, "</p>")

    def k_legend(self, e, ind):
        colours = ["#2a2", "#d90", "#c33", "#39c", "#999"]
        self.w.ln(ind, '<ul class="legend">')
        for n, s in enumerate(e["items"]):
            self.w.ln(ind + 1, f'<li><span class="dot" style="background: {colours[n % 5]}"></span>', self.put(s), "</li>")
        self.used.setdefault("dot", BEM_STYLE["dot"])
        self.w.ln(ind, "</ul>")

    def k_grid(self, e, ind):
        self.w.ln(ind, f"<div{self.cls('grid')}>")
        for c in e.kids:
            self.el(c, ind + 1)
        self.w.ln(ind, "</div>")

    def k_card(self, e, ind):
        self.w.ln(ind, f"<article{self.cls('card')}>")
        self.kids(e, ind + 1)
        self.w.ln(ind, "</article>")

    def k_ul(self, e, ind):
        tag = "ol" if e["ordered"] else "ul"
        # a tick list draws its marks with CSS (a glyph in the item would be part of the node's text)
        self.w.ln(ind, f"<{tag}{' class=\"ticks\"' if e['mark'] == 'check' else ''}>")
        for s in e["items"]:
            self.w.ln(ind + 1, "<li>", self.put(s), "</li>")
        self.w.ln(ind, f"</{tag}>")

    def k_steps(self, e, ind):
        self.w.ln(ind, '<ol class="steps">')
        for num, title, body in e["items"]:
            self.w.ln(ind + 1, "<li>")
            if num is not None:
                self.w.ln(ind + 2, f"<span{self.cls('meta')}>", self.put(num), "</span>")
            self.w.ln(ind + 2, "<h3>", self.put(title), "</h3>")
            if body is not None:
                self.w.ln(ind + 2, "<p>", self.put(body), "</p>")
            self.w.ln(ind + 1, "</li>")
        self.w.ln(ind, "</ol>")

    def k_quote(self, e, ind):
        self.w.ln(ind, "<figure>")
        self.w.ln(ind + 1, "<blockquote>", self.put(e["s"]), "</blockquote>")
        self.w.ln(ind + 1, "<figcaption>", self.put(e["who"]), "</figcaption>")
        self.w.ln(ind, "</figure>")

    def k_faq(self, e, ind):
        for q, a in e["items"]:
            self.w.ln(ind, "<details>")
            self.w.ln(ind + 1, "<summary>", self.put(q), "</summary>")
            self.w.ln(ind + 1, "<p>", self.put(a), "</p>")
            self.w.ln(ind, "</details>")

    def k_form(self, e, ind):
        self.w.ln(ind, '<form method="post">')
        self.kids(e, ind + 1)
        self.w.ln(ind, "</form>")

    def k_field(self, e, ind):
        lab, ph, helper, err = e["label"], e["ph"], e["helper"], e["error"]
        fid = self.uid(lab.text if lab else "field")
        self.w.ln(ind, '<div class="field">')
        if lab is not None:
            self.w.ln(ind + 1, f'<label for="{fid}">', self.put(lab), "</label>")
        typ = e["type"] or "text"
        if typ == "textarea":
            if ph is not None:
                self.w.ln(ind + 1, f'<textarea id="{fid}" placeholder="', (ph, attr_text(ph.text)), '"></textarea>')
            else:
                self.w.ln(ind + 1, f'<textarea id="{fid}"></textarea>')
        elif ph is not None:
            self.w.ln(ind + 1, f'<input id="{fid}" type="{typ}" placeholder="', (ph, attr_text(ph.text)), '">')
        else:
            self.w.ln(ind + 1, f'<input id="{fid}" type="{typ}">')
        if helper is not None:
            self.w.ln(ind + 1, f'<small class="{self.rng.pick(["hint", "field__hint", "help-text"])}">', self.put(helper), "</small>")
        if err is not None:
            self.w.ln(ind + 1, '<p class="error" role="alert">', self.put(err), "</p>")
        self.w.ln(ind, "</div>")

    def k_select(self, e, ind):
        fid = self.uid(e["label"].text)
        self.w.ln(ind, '<div class="field">')
        self.w.ln(ind + 1, f'<label for="{fid}">', self.put(e["label"]), "</label>")
        self.w.ln(ind + 1, f'<select id="{fid}">')
        for o in e["options"]:
            self.w.ln(ind + 2, "<option>", self.put(o), "</option>")
        self.w.ln(ind + 1, "</select>")
        if e["helper"] is not None:
            self.w.ln(ind + 1, '<small class="hint">', self.put(e["helper"]), "</small>")
        self.w.ln(ind, "</div>")

    def k_toggle(self, e, ind):
        fid = self.uid(e["label"].text)
        self.w.ln(ind, '<div class="field field--check">')
        self.w.ln(ind + 1, f'<input id="{fid}" type="checkbox">')
        self.w.ln(ind + 1, f'<label for="{fid}">', self.put(e["label"]), "</label>")
        if e["helper"] is not None:
            self.w.ln(ind + 1, '<small class="hint">', self.put(e["helper"]), "</small>")
        self.w.ln(ind, "</div>")

    def k_errsummary(self, e, ind):
        self.w.ln(ind, '<div class="error-summary" role="alert">')
        self.w.ln(ind + 1, "<h2>", self.put(e["title"]), "</h2>")
        self.w.ln(ind + 1, "<ul>")
        for s in e["items"]:
            self.w.ln(ind + 2, f'<li><a href="#{e["href"] or "field"}">', self.put(s), "</a></li>")
        self.w.ln(ind + 1, "</ul>")
        self.w.ln(ind, "</div>")

    def k_dialog(self, e, ind):
        self.w.ln(ind, "<dialog open>")
        self.w.ln(ind + 1, "<h2>", self.put(e["title"]), "</h2>")
        for b in e["body"] or []:
            self.w.ln(ind + 1, "<p>", self.put(b), "</p>")
        for k in e.kids:
            if k.kind == "actions":
                self.w.ln(ind + 1, '<form method="dialog">')
                for b in k.kids:
                    self.w.ln(ind + 2, f'<button value="{slug(b["s"].text)[:16]}">', self.put(b["s"]), "</button>")
                self.w.ln(ind + 1, "</form>")
            else:
                self.el(k, ind + 1)
        self.w.ln(ind, "</dialog>")

    def k_empty(self, e, ind):
        self.w.ln(ind, f'<div class="{self.rng.pick(["empty-state", "empty", "list__empty"])}">')
        self.w.ln(ind + 1, "<h3>", self.put(e["title"]), "</h3>")
        if e["body"] is not None:
            self.w.ln(ind + 1, "<p>", self.put(e["body"]), "</p>")
        if e["action"] is not None:
            self.el(e["action"], ind + 1)
        self.w.ln(ind, "</div>")

    def k_alert(self, e, ind):
        if e["tone"] == "error":
            if self.idiom == "tailwind":
                self.w.ln(ind, '<div class="rounded border border-red-300 bg-red-50 p-4" role="alert">')
                if e["title"] is not None:
                    self.w.ln(ind + 1, '<p class="font-medium text-red-800">', self.put(e["title"]), "</p>")
                self.w.ln(ind + 1, '<p class="text-sm text-red-700">', self.put(e["s"]), "</p>")
            else:
                self.w.ln(ind, '<div class="alert alert--error" role="alert">')
                if e["title"] is not None:
                    self.w.ln(ind + 1, '<p class="alert__title">', self.put(e["title"]), "</p>")
                self.w.ln(ind + 1, "<p>", self.put(e["s"]), "</p>")
            self.w.ln(ind, "</div>")
            return
        self.w.ln(ind, '<div class="callout" role="status">')
        if e["title"] is not None:
            self.w.ln(ind + 1, "<strong>", self.put(e["title"]), "</strong>")
        self.w.ln(ind + 1, "<p>", self.put(e["s"]), "</p>")
        self.w.ln(ind, "</div>")

    def k_chart(self, e, ind):
        """A chart with no text of its own (the data is drawn by script)."""
        self.ids += 1
        if self.rng.chance(0.5):
            self.w.ln(ind, f'<canvas id="chart-{self.ids}" height="{self.rng.pick([120, 160, 200])}" aria-hidden="true"></canvas>')
        else:
            self.w.ln(ind, f'<div class="chart" data-series="{slug(e["series"] or "series")}" style="height: 160px"></div>')

    def k_nav(self, e, ind):
        self.w.ln(ind, "<nav>")
        self.w.ln(ind + 1, "<ul>")
        for s in e["items"]:
            self.w.ln(ind + 2, f'<li><a href="/{slug(s.text)}">', self.put(s), "</a></li>")
        self.w.ln(ind + 1, "</ul>")
        self.w.ln(ind, "</nav>")

    def k_tabs(self, e, ind):
        self.w.ln(ind, '<div role="tablist">')
        for n, s in enumerate(e["items"]):
            self.w.ln(ind + 1, f'<button role="tab" aria-selected="{"true" if n == 0 else "false"}">', self.put(s), "</button>")
        self.w.ln(ind, "</div>")

    def k_crumbs(self, e, ind):
        if e["aria"] is not None:
            self.w.ln(ind, '<nav class="breadcrumbs" aria-label="', (e["aria"], attr_text(e["aria"].text)), '">')
        else:
            self.w.ln(ind, '<nav class="breadcrumbs">')
        self.w.ln(ind + 1, "<ol>")
        items = e["items"]
        for n, s in enumerate(items):
            if n == len(items) - 1:
                self.w.ln(ind + 2, '<li aria-current="page">', self.put(s), "</li>")
            else:
                self.w.ln(ind + 2, f'<li><a href="/{slug(s.text)}">', self.put(s), "</a></li>")
        self.w.ln(ind + 1, "</ol>")
        self.w.ln(ind, "</nav>")

    def k_table(self, e, ind):
        self.w.ln(ind, "<table>")
        self.w.ln(ind + 1, "<thead>")
        self.w.ln(ind + 2, "<tr>")
        for s in e["headers"]:
            self.w.ln(ind + 3, "<th>", self.put(s), "</th>")
        self.w.ln(ind + 2, "</tr>")
        self.w.ln(ind + 1, "</thead>")
        self.w.ln(ind + 1, "<tbody>")
        for row in e["rows"]:
            self.w.ln(ind + 2, "<tr>")
            for c in row:
                if isinstance(c, E):
                    self.w.ln(ind + 3, "<td>")
                    self.el(c, ind + 4)
                    self.w.ln(ind + 3, "</td>")
                else:
                    self.w.ln(ind + 3, "<td>", self.put(c), "</td>")
            self.w.ln(ind + 2, "</tr>")
        self.w.ln(ind + 1, "</tbody>")
        self.w.ln(ind, "</table>")

    def k_stats(self, e, ind):
        self.w.ln(ind, '<dl class="stats">')
        self.kids(e, ind + 1)
        self.w.ln(ind, "</dl>")

    def k_stat(self, e, ind):
        self.w.ln(ind, "<div>")
        self.w.ln(ind + 1, "<dt>", self.put(e["label"]), "</dt>")
        self.w.ln(ind + 1, f"<dd{self.cls('stat')}>", self.put(e["value"]), "</dd>")
        if e["note"] is not None:
            self.w.ln(ind + 1, f"<dd{self.cls('note')}>", self.put(e["note"]), "</dd>")
        self.w.ln(ind, "</div>")

    def k_price(self, e, ind):
        self.w.ln(ind, '<div class="price">')
        self.w.ln(ind + 1, f"<p{self.cls('stat')}>", self.put(e["amount"]), "</p>")
        self.w.ln(ind + 1, f"<p{self.cls('note')}>", self.put(e["period"]), "</p>")
        self.w.ln(ind, "</div>")

    def k_meta(self, e, ind):
        self.w.ln(ind, f"<div{self.cls('meta')}>")
        for s in e["items"]:
            self.w.ln(ind + 1, "<span>", self.put(s), "</span>")
        self.w.ln(ind, "</div>")

    def k_footer(self, e, ind):
        self.w.ln(ind, "<footer>")
        for head, links in e["cols"]:
            self.w.ln(ind + 1, "<div>")
            self.w.ln(ind + 2, "<h4>", self.put(head), "</h4>")
            self.w.ln(ind + 2, "<ul>")
            for s in links:
                self.w.ln(ind + 3, f'<li><a href="/{slug(s.text)}">', self.put(s), "</a></li>")
            self.w.ln(ind + 2, "</ul>")
            self.w.ln(ind + 1, "</div>")
        if e["legal"] is not None:
            self.w.ln(ind + 1, "<p><small>", self.put(e["legal"]), "</small></p>")
        self.w.ln(ind, "</footer>")
