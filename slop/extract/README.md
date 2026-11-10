# UI-copy extractor

The extractor reads one source file (JSX/TSX, HTML or SwiftUI) and writes one `slop.screen/1` JSON file: the
visible text of the file in reading order, split into nodes, each with a role, style and context flags, a cluster and
a source location. The format is defined in `SPEC.md` §4. The same code runs at labelling time and, later, inside
the linter.

## Running it

Node 22.18 or later is needed; the sources are TypeScript written in erasable syntax only and are run by Node's
built-in type stripping, so there is no build step. Install the pinned dependencies once with `npm ci` in `slop/`.

```
node slop/extract/cli.js <file|dir> --repo NAME [--commit SHA] [--licence L] --out DIR
                         [--root DIR] [--keep-empty] [--summary FILE]
```

- `<file|dir>`: a single source file, or a directory walked recursively. The walk reads `.tsx`, `.jsx`, `.js`,
  `.mjs`, `.html`, `.htm` and `.swift`; it skips `node_modules`, `.next`, `dist`, `build`, `out`, `.git`, `Pods`,
  `DerivedData`, `__tests__`, `*.worktrees`, `components/ui/**`, test, spec and story files, symbolic links, and any
  `.env*`, `*secret*`, `*credential*` or `.cloudflare` path.
- `--repo`, `--commit`, `--licence`: copied into `source`. `commit` and `licence` are `null` when not given.
- `--out`: each screen is written to `DIR/<path>.json`, where `<path>` is the file's path relative to the root.
- `--root`: the repo root, used for the recorded path, the repo's CSS and `slop.config.json`. It defaults to the
  input directory, or for a single file to the nearest ancestor holding `.git`, `package.json` or `Package.swift`.
- Files that yield no nodes are not written unless `--keep-empty` is given. A file that fails to parse is reported
  and skipped; the run carries on. `--summary` writes the counts, the role distribution and the failures as JSON.

The library entry points are `extractSource` and `extractFile` in `src/extract.ts`, with `serialise` for the
canonical bytes.

## What a node is

A node is a run of visible text. Inline elements inside a paragraph, heading, label, button or link (`strong`,
`em`, `code`, a link inside prose, a nested React Native `Text`) are merged into one node; block elements, styled
pills, eyebrows and anything with its own role start a new node. Attributes that show text become their own nodes
at the element's position: `placeholder`, `title` (tooltip), `aria-label` (aria), `alt`, an input's `value`, and the
text props of components (`title`, `description`, `eyebrow`, `label`, `cta` and so on, by prop name). A props object
passed to a component, written in place or bound to an in-file `const` (below), gives a node for each string under a
copy key (`head`, `label`, `title`, `heading`, `caption`, `description`, `text`), down to nested arrays: `<DataTable
columns={[{ head: "Year" }]} />` gives the column head "Year" (a `label`, inside a table), `<Legend items={[{ label:
"Train" }]} />` a `label` (inside a list when the array has two or more items), each at its literal; the other keys
take the roles of the props of the same name (`title` an `h3`, `heading` an `h2`, `caption` a `caption`,
`description` a `subtitle`, `text` a `body`). A navigator screen's `options={{ title }}` is its `tab`
(`Tabs.Screen`), drawer `nav` item (`Drawer.Screen`) or header `h1` (`Stack.Screen` and any other `*Screen`). In
SwiftUI,
`.navigationTitle` is placed first, a `Section` header before its rows and its footer after, and `.help`, `.badge`,
`.accessibilityLabel`, `.alert`, `.sheet` and `.toolbar` content after the view they modify.

Words keep the spacing a browser gives them. Text either side of an inline tag displayed as a block (`block`,
`flex`, `grid` or the repo CSS's `display`) is kept apart by a space, and so is text in two different inline elements
(also inside one branch of a conditional);
text and an inline child with no whitespace between them stay joined, as rendered (`Low<strong>tide</strong>` is
"Lowtide", and JSX drops whitespace that holds a line break, as React does). A short coloured word (one to three
words, a chromatic Tailwind text colour) inside a `flex` or `grid` row is an item of its own, not part of the row's
text. HTML character references are decoded everywhere, named (`&times;`, `&apos;`, `&pound;`), decimal and
hexadecimal, exactly as a browser decodes them (parse5 reads the text). SVG `<text>` and `<tspan>` are read (a
`<tspan>` merges into its `<text>`); an `<svg>` is an icon only when it draws no text.

SwiftUI specifics. A closure passed by label (`actions: { … }`, `message: { … }`) is read like a trailing closure, so
`.alert(_:isPresented:actions:message:)` gives its buttons and message. Properties of type `some ToolbarContent`
(`@ToolbarContentBuilder`) are read like `.toolbar { }` content. A custom view's first unlabelled argument is read when
it is copy (a string literal that is not an SF Symbol name, `String(localized:)`, `Text(…)`), so `PrimaryButton("Pay
now")` and `TagPill("Gift wrapped")` are nodes; a labelled argument that is an initialiser of a type that is not a view
(`copy: ReceiptCopy(…)`, by name: not `*View`, `*Button`, `*Label`, `*Text` and the like) is a value passed in and gives
no node, and `content:` and `copy:` arguments give text only when they are copy. A local view function (`func
point(_ icon: String, _ text: String) -> some View`, declared once in the file) is read at each call in view position,
its parameters bound to the call's arguments (a closure argument can be called as `content()`); its own body is then
not read again on its own. iOS 18 `Tab("…", systemImage:)` gives a tab, and `.labelsHidden()` turns the labels of the
controls it covers into `aria` nodes. A view that is not drawn gives no nodes: one under `.opacity(0)` or `.hidden()`,
or under `.accessibilityHidden(true)` when all its text is buttons with a `.keyboardShortcut` (the off-screen
shortcut buttons). `.accessibilityHidden(true)` alone hides a view from VoiceOver only; its text is still on screen
and is kept.

Interpolations become the VAR marker, the private-use character U+E000, followed by the words of the last
identifier segment (`{user.firstName}` gives `VAR first name`; `t("account.settingsTitle")` gives `VAR settings
title`, while `t("Save changes")` stays as copy). The string branches of a conditional are joined with the ALT marker,
U+E001 (`Online ALT Offline`). The letters "VAR" and "ALT" in copy (a key cap, say) are ordinary words; only the two
code points are markers, and the screen JSON writes them as the escapes `\ue000` and `\ue001`. Private-use characters
in source copy (U+E000–U+F8FF and the supplementary private-use planes, such as Material Icons ligature code points)
are dropped from the text before any marker is written, so a literal U+E000 or U+E001 in a string never reads as one. A node whose text
holds a VAR marker carries `vw`, one integer per marker in order: how many identifier words follow it (`Due in VAR eta
hours hours` has `"vw": [2]`); the rules layer reads it, the tokeniser ignores it. **Complete renderings.** When a conditional part of a longer
text can render nothing, or sits inside a template literal or concatenation, the node's text is every complete
rendering of the text, joined with ALT, so that an optional part does not read as always present and a fallback
keeps the words around it: `{n} file{n === 1 ? "" : "s"}` gives `VAR n file ALT VAR n files`; `{date}{md ? " ·
Markdown" : ""}` gives `VAR date · Markdown ALT VAR date`; `` ` · ${error ?? "failed"}` `` gives `· VAR error ALT ·
failed`; `Draft{dirty && <em> (unsaved)</em>}` gives `Draft (unsaved) ALT Draft`. Fragments shown under different
`&&` conditions in one run of text are one choice among them or none (`{f}{s === "a" && " · sending…"}{s === "b" &&
" · saved"}` gives `VAR f · sending… ALT VAR f · saved ALT VAR f`), and a nested `&&` is a choice inside its
fragment. Renderings are written in branch order (a ternary's consequent first, an `&&` fragment before its absence),
the first part varying slowest; empty and repeated renderings are dropped. Past eight renderings the branches stay
inline where they occur, as do two non-empty branches in a run with nothing optional (`Status: Online ALT Offline`).
An empty fallback (`x ?? ""`, `x || ""`, Swift `x ?? ""`) only guards a missing value and is not a rendering of its
own. `\(date, format: …)` in a Swift
string is one interpolation; `xs.join(", ")` over a literal string array is the joined copy (at the first string's
`loc`); `new Date(x.createdAt)` is named by its argument. A text-less dot
element, or an element holding only a dot glyph, is never a node: it sets `dot` on the next node in the same
container. Separators with no letters or digits (`·`, `/`) are dropped.

**In-file constants.** A name in text position that the same file binds to copy is read through to it when every path
yields a literal; anything else keeps the VAR behaviour. In JSX and TSX: an identifier bound by `const` (any scope,
resolved by scope, so a parameter or local of the same name hides it, and not changed in place) to a string or a
template literal without interpolation; a member of an object or array literal bound that way (`COPY.title`,
`COPY.footer.note`, `LABELS["a"]`; with a dynamic key, `LABELS[k]`, every value joined with ALT, at most 24); a
ternary whose branches are such; and `+` of such. `t(KEY)` with `KEY` such a constant reads like `t("…")`. In an
expanded array copy (below) a dynamic key that is the copy's own value picks that value (`LABELS[item.kind]`). In
SwiftUI: a `let` (local, a member, `static let`, or at the top level) whose value is a string literal or
`String(localized:)`; a computed property (`var title: String { switch kind { case .a: "A" … } }`) or a function
returning `String` (called in text position) whose every path returns one, through `switch`, `if`/`else`, early
returns and ternaries, joined with ALT (at most 24 branches; a `nil` branch renders nothing). A Swift member is found
on `self`, `Self`, a type the file declares or extends (`Kind.title`, `static let`), or a value whose type the file
states: a property or parameter annotated with it, or the closure parameter of a `ForEach` (or `.map`) over
`Kind.allCases`, plain or filtered. A Swift string literal may interpolate: the interpolations stay VAR (`"\(pages)
will be sent"` gives `VAR pages will be sent`), while a JSX template literal with an interpolation is not read
through. A stored `var` (`@State private var draft = "…"`), a `let` with no value (a view's parameter) and a body with
any other statement keep VAR. A node showing one constant literal sits at that literal; one showing several, or a
second node showing the same literal, sits at its use site.

**Literal arrays.** A `.map()` over an array literal is expanded once per element, at most 24 elements, in order,
instead of being read once as a template: the receiver is an inline array literal, or a name bound in the same file by
`const` (at the top level or inside the component, with or without `as const` or `satisfies`) to an array literal
whose elements are all string or object literals, optionally sliced with `.slice(a, b)`. The receiver is resolved by
scope, so a parameter or local of the same name hides the constant. In each copy the callback's parameter, plain or
destructured (also by a `const` at the top of the callback body), stands for the element: string, number, boolean and
null values are substituted, also inside template literals and string concatenation, through `toUpperCase()`,
`toLowerCase()` and `trim()`, as a translation key (`t(item.key)`), and as text props spread onto a component
(`{...item}`); a condition on them (`{item.badge && …}`, a ternary, `===` against a literal) is decided per copy; and
an array literal one level down (`col.links.map(…)`) is expanded the same way. The index parameter and anything that
is not a literal keep the VAR behaviour, and so does every other `.map()`. SwiftUI `ForEach(xs) { x in … }` and
`List(xs) { x in … }` are expanded the same way when `xs` is an array literal, or a `let` declared once in the file
bound to one, of string literals or of initialisers with labelled literal arguments (`Step(title: "…")`); the
parameter may be named or `$0`.

## How roles are chosen

Each node's role comes from the first of these four stages that gives one.

1. **Tag and attribute.** The nearest ancestor with a semantic tag or attribute: `h1`–`h6`, `label`, `legend`, `th`
   and `dt` (label), `button` and `summary`, `a` (link), `figcaption` and `caption`, `li`, `td` (cell), `option`
   (menu item), the ARIA `role` attribute, and the attribute nodes above. Links and buttons inside a navigation,
   tab list, menu or breadcrumb become `nav`, `tab`, `menu_item` or `breadcrumb`. In a card link (an `<a>` or
   `*Link` holding two or more texts), a heading, paragraph or other block element keeps its own role, decided by the
   later stages; only text directly in the link or in inline tags is `link` (JSX and HTML; a SwiftUI `NavigationLink`
   row is unchanged). A link holding a single text is a link whatever wraps the text.
2. **Component name.** `*Title` (dialog, empty-state, page and section titles have their own roles), `*Description`,
   `Badge|Chip|Tag|Pill`, `EmptyState` with `title`/`description`, `*Button`, `*Link`, `Tabs`, `Breadcrumb` and
   `DropdownMenu` parts (a project's own bare `<Dialog>`, `AlertDialog`, `Sheet`, `Drawer` or `Popover` that holds its
   `*Title` directly, with no `*Content`, `*Portal` or `*Trigger` part, is the dialog itself),
   `FormLabel`/`FormDescription`/`FormMessage`, `Typography variant`, `as="h2"`, and the
   SwiftUI views (`Button`, `Link`, `Toggle`, `Picker`, `Section`, `GroupBox`, `ContentUnavailableView`, `Menu`,
   `Tab`). The nearer of a tag and a component wins, so a `CardTitle` inside a card link is still a heading. A SwiftUI
   custom view's first unlabelled copy argument takes its role from the view's name: `*Button` a button,
   `Badge|Chip|Tag|Pill` a badge (a button when the view also takes an action closure or `action:`), `*Title`,
   `*Heading`, `*Header` an `h3`, `*Link` a link, `*Caption` a caption, `*Label` a label, otherwise no role. An
   icon-only button is `aria`: its `label` (or `text`) when the component's name is `*Icon*Button` (`IconButton`,
   `GlassIconButton`), or when a `*Button` component takes an icon (`icon`, `systemImage`, …) and a `label` but no
   visible title (JSX: no children; Swift: no unlabelled copy argument). Action-title props (`actionTitle`,
   `secondaryActionTitle`, `buttonTitle`, …) are buttons. A component defined in the file whose own name ends in
   `Badge|Chip|Tag|Pill` (or `Eyebrow|Kicker|Overline`) gives its markup that role: in JSX at the class stage, in
   SwiftUI when its `body` is a small leaf view (one to three texts, no control or container). A SwiftUI `List` with a
   `selection:` and the sidebar column of a `NavigationSplitView` are navigation, and their rows are `nav`.
3. **Class tokens**, used only when no ancestor gave a role at stage 1 or 2: first `slop.config.json`, then semantic
   class names (`hero__eyebrow`, `status-pill`), then Tailwind and the repo's own CSS: a large figure is a `stat`;
   `rounded-full` on short text is a `badge`; `uppercase` with positive `tracking-*` is a kicker; `text-destructive`,
   `text-red-*` and `text-rose-*` are an `error` (not on a dot-led status line), and so is text that is one bound value
   named as an error (`{uploadError}`, `model.errorMessage`) outside a dialog; `text-xs` is small;
   `text-muted-foreground` is muted. A short coloured word (one to three words, a chromatic text colour) in an inline
   element inside a row (`li`, `tr` or a template copy), after other text of that row, is a `badge`. Class names whose
   last word is `title`, `heading`, `headline` or `header` (`hn-panel__title`, `card-header`) make an `h3`. SVG
   `<text>` is never body: a data value is a `cell` (a class ending in `num`, `value`, `data`, `count`, `pct` and the
   like, a figure, or a bare value inside two nested templates) and everything else a `label` (axes, legends, row and
   column heads). The repo's CSS is read from simple class
   selectors (`.name { … }`) in every `.css`, `.scss` and `.less` file and in `<style>` blocks, for
   `text-transform`, `letter-spacing`, `font-size`, `font-family`, `border-radius`, `width`, `height` and `display`.
   SwiftUI fonts count here too (`.largeTitle` and `.title` give `h1`, `.title2` `h2`, `.title3` `h3`, `.headline`
   `h4`; `.caption` is small), whatever the modifier is called: any modifier whose only argument is a text style
   (`.font(.title)`, `.appFont(.title)`, `.hcFont(.caption)`) or a house style named after one (`.captionBold`,
   `.bodyEmphasis`) is read as a font. `.foregroundStyle(.red)` or a colour named `destructive`, `danger` or `error`
   (not a ternary) is error-coloured. A pill or kicker inside a list item or table cell overrides that weak tag role.
4. **Position**, for whatever is left. Neighbours are the nodes before and after in reading order that share the same
   group (the nearest ancestor holding more than one piece of text). First, an untagged heading: a line of one to ten
   words, not ending in `.`, `,`, `:` or `;` and not a figure, in a large bold type (Tailwind `text-lg` or larger, or
   an arbitrary size of 18px or more, with `font-semibold`, `font-bold`, `font-extrabold` or `font-black`; or the
   repo CSS's literal `font-size` of 18px or more with `font-weight` 600 or more; or the same as an inline style),
   with content after it inside its block's parent that is not a heading or a line in the same or a larger such type,
   is an `h2` at 24px or more and an `h3` below (CSS custom properties are not resolved). Then a group label in a
   sidebar (inside an `<aside>`, a `*Sidebar` component or a `sidebar` class): one to three words with no closing
   punctuation, not directly before a heading, whose element is followed by a container sibling holding more text
   ("Today" over a list), is a `label`. A kicker directly before a heading is an
   `eyebrow`; the only line after a heading in its group is its tagline (`subtitle`); the first node of a card,
   dialog, state or alert with at least three nodes after it is that panel's heading (`h3`); otherwise a kicker is a
   `label` (a group label, or a field label such as a HUD's "Objective"); small or muted text of up to eight words
   directly before a heading is an `eyebrow`; text directly after a heading is a `subtitle` when it is small, muted,
   or the only line before the next control or heading; small text elsewhere is a `caption`; everything else is
   `body`.

After that, headings and body text inside a dialog become `dialog_title` and `dialog_body`, inside an empty state
`empty_title` and `empty_body`, inside a toast `toast`; secondary text inside a form field becomes `helper`. Links
in a list inside a page header (`header > ul > li > a`) are `nav`. A second line of text inside a button (a
description under its label) is a `caption`; the first stays `button`, and texts in different branches of a condition
inside the button are alternatives, not lines. A hand-built empty state is the branch shown when a collection is
empty (`xs.length === 0 ? … : …`, `!xs.length && …`, `if (!xs.length) return …`, Swift `if xs.isEmpty { … }`, or the
other branch of `xs.length > 0` and `!xs.isEmpty`) when it holds one to four lines of text, at least one of them
literal copy, and no list or other container; with no heading of its own, its first line of up to eight words becomes
`empty_title` and the rest `empty_body`, also when it is the only line. The extractor never emits `comment` or `code` (track C is deferred).

## Flags and clusters

Style flags (`caps`, `title_case`, `tracked`, `small`, `mono`) follow the CSS cascade from every ancestor; `caps` and
`title_case` also come from the literal text, and hold for a conditional only if every branch has them. `pill`,
`truncated` and `stateful` read the node's own element and its inline wrappers. `stateful` means the text is one
bound expression, or its colour, class or dot is chosen by a condition. `conditional` covers ternaries, `&&`,
`if` and `switch` branches, early returns, dialogs and sheets. `repeated` marks nodes inside three or more sibling
subtrees with the same role sequence of two or more nodes, and nodes inside a `.map()` or `ForEach` template, or one
expanded copy of it, of two or more nodes. `section_first` and `section_last` mark the first and last node under each heading, up to the next
heading of the same or higher level and within the heading's own section (a sibling heading of the same level bounds
it). Tooltip, aria and alt nodes do not take their element's text styling.

Clusters are contiguous. A node belongs to the nearest card or state container (dialogs, alerts, toasts, empty
states, early-return branches); otherwise to the nearest form field (`control`: the largest ancestor of a single
input that holds no heading, no other action and at most five nodes); otherwise to the outermost list, table or
navigation (`list`). Free nodes are grouped by their group parent and split into `header` (eyebrow, heading,
subtitle, badges and actions up to the first body text or repeated child), `actions` (only buttons and links) and
`body`. Three or more repeated siblings that are not list rows become one `card` cluster each. A cluster's `line` is
the first line of the innermost element enclosing all its nodes.

## Locations, ids and bytes

`loc` is `[line, col, endLine, endCol]`, 1-based; lines end at `\n`, columns count Unicode code points, and the end
column is one past the last character. Locations point at the visible characters: inside quotes for attributes and
string literals, at the expression for an interpolation. In an expanded array copy a node's `loc` is the string
literal in the array that it shows (the first one, if it shows several). Because labels key on `loc`, no two nodes
share one: a second node showing the same literal (an `alt` or `aria-label` that repeats a title) takes the span of
the literal's property (`title: "…"`) instead; a node showing one in-file constant sits at the constant's literal, or at its
use site when that literal is taken or the node shows several; the first node in a copy with no literal of its own (a constant
label, or the VAR text of the index) takes the span of the whole array element; and copy text left without a free
span keeps its span in the callback in the first copy and is dropped from later ones. Text read from an inlined local
SwiftUI function sits at the argument it shows (the literal in the call), or else at its own span in the function,
and a later call that would repeat that span takes the span of the whole call. Visible nodes choose before
context-only ones. The screen `id` is the first 16 hex digits of
sha256(`slop.screen/1`, repo, path, sha256 of the source). The serialiser writes one cluster and one node per line
with fixed key order and flags in the order of `SPEC.md` §4, so the same input always gives the same bytes. Windows
(`SPEC.md` §4) need token counts and are cut later, by the tokeniser.

## `slop.config.json`

Put it at the repo root to map class tokens or component names to roles:

```json
{"classRoles": {"masthead-overline": "eyebrow"}, "componentRoles": {"Byline": "caption"}}
```

In SwiftUI a modifier with a single leading-dot argument becomes a class token named after both, so
`.brandFont(.display)` can be mapped with `"brandFont.display": "h1"`. Per-repo maps used for the experiment are
frozen before labelling.

## Tests and fixtures

`npm test` (in `slop/`) runs every fixture in `extract/fixtures/`, checks the output byte for byte against
`expected.json`, runs each one twice to check determinism, validates the schema (including `vw`: present exactly when
the text holds a VAR marker, one count per marker, never more than the words after it), and fails unless there are
at least 60 fixtures, every role bar `comment` and `code` and every flag and cluster kind appears somewhere, and the Swift
grammar's wasm matches its pinned sha256. `npm run typecheck` runs `tsc`.

To add a fixture:

1. Make a directory under `extract/fixtures/` named for what it tests, holding exactly one source file. A `.css`,
   `.scss` or `slop.config.json` beside it is read as that fixture's repo CSS and config. Write the copy yourself:
   never paste strings from a real repo, private or public.
2. Run `npm test -- --update NAME` to write `expected.json`.
3. Read the new `expected.json` node by node and check every role, flag, cluster and location by hand. The golden
   file is only as good as that review.
4. Run `npm test` and commit the source file and `expected.json` together.

Any change to the extractor that alters an existing `expected.json` must be reviewed the same way; after the freeze
(`SPEC.md` §12.1) the extractor and its fixtures do not change.

## Dependencies

Pinned exactly in `slop/package.json`: `@babel/parser` for JSX and TSX, `parse5` for HTML, and `web-tree-sitter`
with the tree-sitter-swift grammar. The grammar's wasm comes prebuilt from `@repomix/tree-sitter-wasms` (built from
tree-sitter-swift 0.7.1), because the `tree-sitter-swift` package ships native bindings and no wasm; its sha256 is
pinned in `test/run.ts`.
