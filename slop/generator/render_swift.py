"""SwiftUI renderer for the dev generator. Writes one ``View`` struct; ``@State`` properties and literal arrays
for ``ForEach`` go above ``body``, alerts and confirmation dialogs become modifiers on the outer stack, and
``.navigationTitle`` closes a ``NavigationStack``."""

from __future__ import annotations

from .core import E, GenError, Page, S, Slot, Writer, ident, slug

IDIOMS = ["stack", "form", "list"]
SYMBOLS = ["star", "bolt", "leaf", "clock", "lock", "bell", "tray", "folder", "map", "flag", "gearshape", "person.2"]


def sw_str(s: str) -> str:
    if '"' in s or "\\" in s:
        raise GenError(f"copy not writable as a Swift string: {s!r}")
    return s


class SwiftRenderer:
    def __init__(self, rng, idiom: str):
        self.rng = rng
        self.idiom = idiom
        self.w = Writer()
        self.props = Writer()
        self.state: list[str] = []
        self.extra: list[Slot] = []  # copy the renderer itself adds (picker titles)
        self.modifiers: list[E] = []
        self.ids = 0
        self.structs: list[str] = []
        self.const_names: set[str] = set()
        self.use_map = rng.chance(0.4)
        self.in_section = False

    def var(self, base: str, init: str) -> str:
        self.ids += 1
        name = ident(base, upper_first=False)[:20] + (str(self.ids) if self.ids > 1 else "")
        self.state.append(f"@State private var {name} = {init}")
        return name

    def q(self, s: Slot):
        return (s, sw_str(s.text))

    # -- page ----------------------------------------------------------------------------------------------------

    def render(self, page: Page):
        name = ident(page.name)
        if not name.endswith("View"):
            name += "View"
        body = self.w
        nav = page.nav_title is not None
        ind = 2
        if nav:
            body.ln(ind, "NavigationStack {")
            ind += 1
        container = "Form" if self.idiom == "form" else ("List" if self.idiom == "list" else "ScrollView")
        body.ln(ind, f"{container} {{")
        inner = ind + 1
        if container == "ScrollView":
            body.ln(inner, f"VStack(alignment: .leading, spacing: {self.rng.pick(['12', '16', '20', '24'])}) {{")
            inner += 1
        for e in page.body:
            self.el(e, inner)
        if container == "ScrollView":
            body.ln(inner - 1, "}")
            body.ln(inner - 1, ".padding()")
        body.ln(ind, "}")
        for m in self.modifiers:
            self.modifier(m, ind)
        if nav:
            body.ln(ind, ".navigationTitle(\"", self.q(page.nav_title), "\")")
            ind -= 1
            body.ln(ind, "}")
        body.ln(1, "}")
        body.ln(0, "}")
        for st in self.structs:
            body.raw("\n" + st)
        head = Writer()
        head.ln(0, "import SwiftUI")
        head.raw("\n")
        head.ln(0, f"struct {name}: View {{")
        for st in self.state:
            head.ln(1, st)
        h = head.text().count("\n")
        self.props.shift(h)
        mid = Writer()
        if self.props.parts:
            self.props.raw("\n")
        elif self.state:
            mid.raw("\n")
        mid.ln(1, "var body: some View {")
        p_lines = self.props.text().count("\n")
        body.shift(h + p_lines + mid.text().count("\n"))
        return head.text() + self.props.text() + mid.text() + body.text(), None

    # -- elements ------------------------------------------------------------------------------------------------

    def el(self, e: E, ind: int):
        fn = getattr(self, "k_" + e.kind, None)
        if fn is None:
            raise GenError(f"swift renderer has no element {e.kind!r}")
        fn(e, ind)

    def kids(self, e, ind):
        for k in e.kids:
            self.el(k, ind)

    def k_section(self, e, ind):
        if (self.idiom == "form" or self.idiom == "list") and not self.in_section:
            self.w.ln(ind, "Section {")
            self.in_section = True
            self.kids(e, ind + 1)
            self.in_section = False
            self.w.ln(ind, "}")
            return
        self.w.ln(ind, "VStack(alignment: .leading, spacing: 12) {")
        self.kids(e, ind + 1)
        self.w.ln(ind, "}")

    def k_div(self, e, ind):
        self.w.ln(ind, "VStack(alignment: .leading, spacing: 8) {")
        self.kids(e, ind + 1)
        self.w.ln(ind, "}")

    k_head = k_div

    def k_eyebrow(self, e, ind):
        self.w.ln(ind, 'Text("', self.q(e["s"]), '")')
        self.w.ln(ind + 1, ".font(.caption)")
        self.w.ln(ind + 1, ".textCase(.uppercase)")
        self.w.ln(ind + 1, f".tracking({self.rng.pick(['1.5', '2', '3'])})")
        if self.rng.chance(0.5):
            self.w.ln(ind + 1, ".foregroundStyle(.secondary)")

    def k_h(self, e, ind):
        lv = e["level"] or 2
        font = {1: self.rng.pick([".largeTitle", ".title"]), 2: ".title2", 3: ".title3"}.get(lv, ".headline")
        self.w.ln(ind, 'Text("', self.q(e["s"]), '")')
        self.w.ln(ind + 1, f".font({font})")
        if lv <= 2 and self.rng.chance(0.5):
            self.w.ln(ind + 1, ".bold()")

    def k_sub(self, e, ind):
        self.w.ln(ind, 'Text("', self.q(e["s"]), '")')
        self.w.ln(ind + 1, f".font({self.rng.pick(['.footnote', '.subheadline', '.callout'])})")
        self.w.ln(ind + 1, ".foregroundStyle(.secondary)")

    def k_p(self, e, ind):
        self.w.ln(ind, 'Text("', self.q(e["s"]), '")')

    def k_note(self, e, ind):
        self.w.ln(ind, 'Text("', self.q(e["s"]), '")')
        self.w.ln(ind + 1, ".font(.caption)")
        self.w.ln(ind + 1, ".foregroundStyle(.secondary)")

    def k_actions(self, e, ind):
        self.w.ln(ind, "HStack(spacing: 12) {")
        self.kids(e, ind + 1)
        self.w.ln(ind, "}")

    def k_button(self, e, ind):
        s, tip = e["s"], e["tip"]
        role = ", role: .destructive" if e["destructive"] else ""
        fn = ident(s.text, upper_first=False)[:24]
        if self.rng.chance(0.5):
            self.w.ln(ind, 'Button("', self.q(s), f'"{role}) {{ {fn}() }}')
        else:
            self.w.ln(ind, 'Button("', self.q(s), f'"{role}, action: {fn})')
        if e["primary"] is not False:
            self.w.ln(ind + 1, ".buttonStyle(.borderedProminent)")
        if tip is not None:
            self.w.ln(ind + 1, '.help("', self.q(tip), '")')

    def k_link(self, e, ind):
        s = e["s"]
        if self.rng.chance(0.5):
            self.w.ln(ind, 'Link("', self.q(s), f'", destination: URL(string: "https://example.com/{slug(s.text)[:20]}")!)')
        else:
            self.w.ln(ind, 'NavigationLink("', self.q(s), f'") {{ {ident(s.text)[:20]}Detail() }}')

    def k_iconbtn(self, e, ind):
        self.w.ln(ind, "Button(action: {}) {")
        self.w.ln(ind + 1, f'Image(systemName: "{self.rng.pick(["trash", "square.and.arrow.up", "ellipsis.circle", "pencil"])}")')
        self.w.ln(ind, "}")
        self.w.ln(ind + 1, '.help("', self.q(e["tip"]), '")')
        if e["aria"] is not None:
            self.w.ln(ind + 1, '.accessibilityLabel("', self.q(e["aria"]), '")')

    def k_badge(self, e, ind):
        self.w.ln(ind, 'Text("', self.q(e["s"]), '")')
        self.w.ln(ind + 1, ".font(.caption2)")
        self.w.ln(ind + 1, ".padding(.horizontal, 6)")
        self.w.ln(ind + 1, ".padding(.vertical, 2)")
        if e["bound"]:
            self.w.ln(ind + 1, f".background(Capsule().fill({ident(e['bound'], upper_first=False)} ? Color.green.opacity(0.2) : Color.gray.opacity(0.2)))")
        else:
            self.w.ln(ind + 1, f".background(Capsule().fill(.{self.rng.pick(['blue', 'orange', 'mint', 'purple'])}.opacity(0.15)))")

    def k_status(self, e, ind):
        self.w.ln(ind, "HStack(spacing: 6) {")
        self.w.ln(ind + 1, "Circle()")
        if e["bound"]:
            self.w.ln(ind + 2, f".fill({ident(e['bound'], upper_first=False)} ? Color.green : Color.gray)")
        else:
            self.w.ln(ind + 2, f".fill(Color.{self.rng.pick(['green', 'mint', 'blue'])})")
        self.w.ln(ind + 2, ".frame(width: 8, height: 8)")
        self.w.ln(ind + 1, 'Text("', self.q(e["s"]), '")')
        self.w.ln(ind, "}")

    def k_legend(self, e, ind):
        colours = ["green", "orange", "red", "blue", "gray"]
        self.w.ln(ind, "HStack(spacing: 16) {")
        for n, s in enumerate(e["items"]):
            self.w.ln(ind + 1, "HStack(spacing: 4) {")
            self.w.ln(ind + 2, f"Circle().fill(Color.{colours[n % 5]}).frame(width: 8, height: 8)")
            self.w.ln(ind + 2, 'Text("', self.q(s), '").font(.caption)')
            self.w.ln(ind + 1, "}")
        self.w.ln(ind, "}")

    def k_grid(self, e, ind):
        cards = e.kids
        if self.use_map and e["mappable"] and len(cards) >= 2 and self._uniform(cards):
            self._mapped(e, ind)
            return
        self.w.ln(ind, 'LazyVGrid(columns: [GridItem(.adaptive(minimum: 160))], spacing: 16) {')
        for c in cards:
            self.el(c, ind + 1)
        self.w.ln(ind, "}")

    def _uniform(self, cards):
        sig = [tuple(k.kind for k in c.kids) for c in cards]
        return len(set(sig)) == 1 and all(k.kind in ("h", "p", "sub", "note") for k in cards[0].kids)

    def _mapped(self, e, ind):
        cards = e.kids
        fields = []
        for k in cards[0].kids:
            base = {"h": "title", "p": "detail", "sub": "summary", "note": "note"}[k.kind]
            name, n = base, 2
            while name in fields:
                name, n = f"{base}{n}", n + 1
            fields.append(name)
        typ = ident(e["name"] or "item")
        if typ.endswith("s"):
            typ = typ[:-1]
        typ = typ or "Item"
        arr = ident(e["name"] or "items", upper_first=False)
        while arr in self.const_names:
            arr += "List"
        self.const_names.add(arr)
        self.props.ln(1, f"private let {arr} = [")
        for c in cards:
            self.props.raw("    " + typ + "(")
            for j, (f, k) in enumerate(zip(fields, c.kids)):
                if j:
                    self.props.raw(", ")
                self.props.raw(f'{f}: "')
                self.props.put(k["s"], sw_str(k["s"].text))
                self.props.raw('"')
            self.props.raw("),\n")
        self.props.ln(1, "]")
        self.structs.append(f"struct {typ} {{\n" + "".join(f"    let {f}: String\n" for f in fields) + "}\n")
        v = typ[0].lower()
        self.w.ln(ind, f"ForEach({arr}, id: \\.{fields[0]}) {{ {v} in")
        self.w.ln(ind + 1, "GroupBox {")
        self.w.ln(ind + 2, "VStack(alignment: .leading, spacing: 6) {")
        for f, k in zip(fields, cards[0].kids):
            mod = {"h": ".font(.headline)", "sub": ".font(.footnote).foregroundStyle(.secondary)",
                   "note": ".font(.caption)", "p": ""}[k.kind]
            self.w.ln(ind + 3, f"Text({v}.{f}){mod}")
        self.w.ln(ind + 2, "}")
        self.w.ln(ind + 1, "}")
        self.w.ln(ind, "}")

    def k_card(self, e, ind):
        self.w.ln(ind, "GroupBox {")
        self.w.ln(ind + 1, "VStack(alignment: .leading, spacing: 6) {")
        self.kids(e, ind + 2)
        self.w.ln(ind + 1, "}")
        self.w.ln(ind, "}")

    def k_ul(self, e, ind):
        items = e["items"]
        if self.use_map and e["mappable"] and len(items) >= 2:
            arr = ident(e["name"] or "points", upper_first=False)
            while arr in self.const_names:
                arr += "List"
            self.const_names.add(arr)
            self.props.ln(1, f"private let {arr} = [")
            for s in items:
                self.props.ln(2, '"', (s, sw_str(s.text)), '",')
            self.props.ln(1, "]")
            self.w.ln(ind, f"ForEach({arr}, id: \\.self) {{ line in")
            self.w.ln(ind + 1, 'Label(line, systemImage: "checkmark")')
            self.w.ln(ind, "}")
            return
        self.w.ln(ind, "VStack(alignment: .leading, spacing: 6) {")
        for s in items:
            if e["mark"] == "check" or self.rng.chance(0.5):
                self.w.ln(ind + 1, 'Label("', self.q(s), f'", systemImage: "{self.rng.pick(["checkmark", "checkmark.circle", "circle.fill"])}")')
            else:
                self.w.ln(ind + 1, 'Text("', self.q(s), '")')
        self.w.ln(ind, "}")

    def k_steps(self, e, ind):
        self.w.ln(ind, "VStack(alignment: .leading, spacing: 16) {")
        for num, title, body in e["items"]:
            self.w.ln(ind + 1, "HStack(alignment: .top, spacing: 12) {")
            if num is not None:
                self.w.ln(ind + 2, 'Text("', self.q(num), '")')
                self.w.ln(ind + 3, ".font(.caption.monospaced())")
            self.w.ln(ind + 2, "VStack(alignment: .leading, spacing: 4) {")
            self.w.ln(ind + 3, 'Text("', self.q(title), '").font(.headline)')
            if body is not None:
                self.w.ln(ind + 3, 'Text("', self.q(body), '")')
            self.w.ln(ind + 2, "}")
            self.w.ln(ind + 1, "}")
        self.w.ln(ind, "}")

    def k_quote(self, e, ind):
        self.w.ln(ind, "VStack(alignment: .leading, spacing: 4) {")
        self.w.ln(ind + 1, 'Text("', self.q(e["s"]), '").italic()')
        self.w.ln(ind + 1, 'Text("', self.q(e["who"]), '").font(.caption)')
        self.w.ln(ind, "}")

    def k_faq(self, e, ind):
        for q, a in e["items"]:
            self.w.ln(ind, 'DisclosureGroup("', self.q(q), '") {')
            self.w.ln(ind + 1, 'Text("', self.q(a), '")')
            self.w.ln(ind, "}")

    def k_form(self, e, ind):
        # inside a Form or List the fields already sit in sections; in a stack they get a plain group
        self.kids(e, ind)

    def _section(self, ind, footer: Slot | None, render):
        if self.in_section:
            # already inside a Section: the helper goes under the control as secondary text
            render(ind)
            if footer is not None:
                self.w.ln(ind, 'Text("', self.q(footer), '")')
                self.w.ln(ind + 1, ".font(.footnote)")
                self.w.ln(ind + 1, ".foregroundStyle(.secondary)")
            return
        if self.idiom in ("form", "list") or footer is not None:
            self.w.ln(ind, "Section {")
            render(ind + 1)
            if footer is not None:
                self.w.ln(ind, "} footer: {")
                self.w.ln(ind + 1, 'Text("', self.q(footer), '")')
            self.w.ln(ind, "}")
        else:
            render(ind)

    def k_field(self, e, ind):
        lab, ph, helper, err = e["label"], e["ph"], e["helper"], e["error"]
        name = self.var(lab.text if lab else (ph.text if ph else "value"), '""')
        kind = "SecureField" if e["type"] == "password" else "TextField"

        def body(i):
            if lab is not None and ph is not None:
                self.w.ln(i, f'{kind}("', self.q(lab), f'", text: ${name}, prompt: Text("', self.q(ph), '"))')
            elif lab is not None:
                raise GenError("swift field with a label needs a prompt (a bare title is a placeholder)")
            else:
                self.w.ln(i, f'{kind}("', self.q(ph), f'", text: ${name})')
            if err is not None:
                self.w.ln(i, f"if {name}.isEmpty && submitted {{")
                self.w.ln(i + 1, 'Text("', self.q(err), '")')
                self.w.ln(i + 2, ".foregroundStyle(.red)")
                self.w.ln(i, "}")

        if err is not None and "@State private var submitted = false" not in self.state:
            self.state.append("@State private var submitted = false")
        self._section(ind, helper, body)

    def k_select(self, e, ind):
        name = self.var(e["label"].text, f'"{sw_str(e["options"][0].text)}"')

        def body(i):
            self.w.ln(i, 'Picker("', self.q(e["label"]), f'", selection: ${name}) {{')
            for o in e["options"]:
                self.w.ln(i + 1, 'Text("', self.q(o), f'").tag("{sw_str(o.text)}")')
            self.w.ln(i, "}")

        self._section(ind, e["helper"], body)

    def k_toggle(self, e, ind):
        name = self.var(e["label"].text, self.rng.pick(["true", "false"]))
        self._section(ind, e["helper"], lambda i: self.w.ln(i, 'Toggle("', self.q(e["label"]), f'", isOn: ${name})'))

    def k_dialog(self, e, ind):
        # rendered later as a modifier on the outer container
        self.modifiers.append(e)

    def modifier(self, e: E, ind: int):
        flag = self.var("showing" + ident(e["title"].text)[:16], "false")
        kind = ".confirmationDialog" if e["sheetlike"] else ".alert"
        self.w.ln(ind, f'{kind}("', self.q(e["title"]), f'", isPresented: ${flag}) {{')
        for k in e.kids:
            if k.kind == "actions":
                for b in k.kids:
                    role = ", role: .cancel" if b["primary"] is False else (", role: .destructive" if b["destructive"] else "")
                    self.w.ln(ind + 1, 'Button("', self.q(b["s"]), f'"{role}) {{}}')
        body = e["body"] or []
        if body:
            self.w.ln(ind, "} message: {")
            self.w.ln(ind + 1, 'Text("', self.q(body[0]), '")')
            if len(body) > 1:
                raise GenError("swift alerts take one message")
        self.w.ln(ind, "}")

    def k_empty(self, e, ind):
        title, body, action = e["title"], e["body"], e["action"]
        sym = self.rng.pick(SYMBOLS)
        if action is None and body is not None and self.rng.chance(0.6):
            self.w.ln(ind, 'ContentUnavailableView("', self.q(title), f'", systemImage: "{sym}",')
            self.w.ln(ind + 2, 'description: Text("', self.q(body), '"))')
            return
        self.w.ln(ind, "ContentUnavailableView {")
        self.w.ln(ind + 1, 'Label("', self.q(title), f'", systemImage: "{sym}")')
        if body is not None:
            self.w.ln(ind, "} description: {")
            self.w.ln(ind + 1, 'Text("', self.q(body), '")')
        if action is not None:
            self.w.ln(ind, "} actions: {")
            self.el(action, ind + 1)
        self.w.ln(ind, "}")

    def k_alert(self, e, ind):
        self.w.ln(ind, "GroupBox {")
        if e["title"] is not None:
            self.w.ln(ind + 1, 'Label("', self.q(e["title"]), '", systemImage: "info.circle")')
        self.w.ln(ind + 1, 'Text("', self.q(e["s"]), '")')
        self.w.ln(ind + 2, ".font(.footnote)")
        self.w.ln(ind, "}")

    def k_cond(self, e, ind):
        self.w.ln(ind, f"if {e['swift'] or e['expr']} {{")
        for x in e["a"]:
            self.el(x, ind + 1)
        self.w.ln(ind, "} else {")
        for x in e["b"]:
            self.el(x, ind + 1)
        self.w.ln(ind, "}")

    def k_dyn(self, e, ind):
        s = e["s"]
        pre, post = e["pre"] or "", e["post"] or ""
        if pre or post:
            self.w.ln(ind, 'Text("', (s, sw_str(pre) + "\\(" + e["expr"] + ")" + sw_str(post)), '")')
        else:
            self.w.ln(ind, "Text(", (s, e["expr"]), ")")

    def k_tabs(self, e, ind):
        name = self.var("section", f'"{sw_str(e["items"][0].text)}"')
        title = S(self.rng.pick(["View", "Section", "Show"]), "label")
        self.extra.append(title)
        self.w.ln(ind, 'Picker("', self.q(title), f'", selection: ${name}) {{')
        for s in e["items"]:
            self.w.ln(ind + 1, 'Text("', self.q(s), f'").tag("{sw_str(s.text)}")')
        self.w.ln(ind, "}")
        self.w.ln(ind, ".pickerStyle(.segmented)")

    def k_nav(self, e, ind):
        self.w.ln(ind, "HStack {")
        for s in e["items"]:
            self.w.ln(ind + 1, 'NavigationLink("', self.q(s), f'") {{ {ident(s.text)[:16]}View() }}')
        self.w.ln(ind, "}")

    def k_table(self, e, ind):
        self.w.ln(ind, "Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {")
        self.w.ln(ind + 1, "GridRow {")
        for s in e["headers"]:
            self.w.ln(ind + 2, 'Text("', self.q(s), '").font(.caption).foregroundStyle(.secondary)')
        self.w.ln(ind + 1, "}")
        for row in e["rows"]:
            self.w.ln(ind + 1, "GridRow {")
            for c in row:
                if isinstance(c, E):
                    self.el(c, ind + 2)
                else:
                    self.w.ln(ind + 2, 'Text("', self.q(c), '")')
            self.w.ln(ind + 1, "}")
        self.w.ln(ind, "}")

    def k_stats(self, e, ind):
        self.w.ln(ind, "HStack(spacing: 24) {")
        self.kids(e, ind + 1)
        self.w.ln(ind, "}")

    def k_stat(self, e, ind):
        self.w.ln(ind, "VStack(alignment: .leading, spacing: 2) {")
        self.w.ln(ind + 1, 'Text("', self.q(e["label"]), '")')
        self.w.ln(ind + 2, ".font(.caption)")
        self.w.ln(ind + 1, 'Text("', self.q(e["value"]), '")')
        self.w.ln(ind + 2, ".monospacedDigit()")
        if e["note"] is not None:
            self.w.ln(ind + 1, 'Text("', self.q(e["note"]), '").font(.caption2)')
        self.w.ln(ind, "}")

    def k_price(self, e, ind):
        self.w.ln(ind, "VStack(alignment: .leading) {")
        self.w.ln(ind + 1, 'Text("', self.q(e["amount"]), '")')
        self.w.ln(ind + 1, 'Text("', self.q(e["period"]), '").font(.caption)')
        self.w.ln(ind, "}")

    def k_meta(self, e, ind):
        self.w.ln(ind, "HStack(spacing: 12) {")
        for s in e["items"]:
            self.w.ln(ind + 1, 'Text("', self.q(s), '")')
            self.w.ln(ind + 2, ".font(.caption.monospaced())")
        self.w.ln(ind, "}")
