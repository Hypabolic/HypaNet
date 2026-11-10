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
text props of components (`title`, `description`, `eyebrow`, `label`, `cta` and so on, by prop name). In SwiftUI,
`.navigationTitle` is placed first, a `Section` header before its rows and its footer after, and `.help`, `.badge`,
`.accessibilityLabel`, `.alert`, `.sheet` and `.toolbar` content after the view they modify.

Interpolations become the VAR marker, the private-use character U+E000, followed by the words of the last
identifier segment (`{user.firstName}` gives `VAR first name`; `t("account.settingsTitle")` gives `VAR settings
title`, while `t("Save changes")` stays as copy). The string branches of a conditional are joined with the ALT marker,
U+E001 (`Online ALT Offline`). The letters "VAR" and "ALT" in copy (a key cap, say) are ordinary words; only the two
code points are markers, and the screen JSON writes them as the escapes `\ue000` and `\ue001`. A text-less dot
element, or an element holding only a dot glyph, is never a node: it sets `dot` on the next node in the same
container. Separators with no letters or digits (`·`, `/`) are dropped.

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
   tab list, menu or breadcrumb become `nav`, `tab`, `menu_item` or `breadcrumb`.
2. **Component name.** `*Title` (dialog, empty-state, page and section titles have their own roles), `*Description`,
   `Badge|Chip|Tag|Pill`, `EmptyState` with `title`/`description`, `*Button`, `*Link`, `Tabs`, `Breadcrumb` and
   `DropdownMenu` parts, `FormLabel`/`FormDescription`/`FormMessage`, `Typography variant`, `as="h2"`, and the
   SwiftUI views (`Button`, `Link`, `Toggle`, `Picker`, `Section`, `GroupBox`, `ContentUnavailableView`, `Menu`).
   The nearer of a tag and a component wins, so a `CardTitle` inside a card link is still a heading.
3. **Class tokens**, used only when no ancestor gave a role at stage 1 or 2: first `slop.config.json`, then semantic
   class names (`hero__eyebrow`, `status-pill`), then Tailwind and the repo's own CSS: a large figure is a `stat`;
   `rounded-full` on short text is a `badge`; `uppercase` with positive `tracking-*` is a kicker; `text-destructive`
   is an `error`; `text-xs` is small; `text-muted-foreground` is muted. The repo's CSS is read from simple class
   selectors (`.name { … }`) in every `.css`, `.scss` and `.less` file and in `<style>` blocks, for
   `text-transform`, `letter-spacing`, `font-size`, `font-family`, `border-radius`, `width` and `height`. SwiftUI
   fonts count here too (`.largeTitle` and `.title` give `h1`, `.title2` `h2`, `.title3` `h3`, `.headline` `h4`;
   `.caption` is small). A pill or kicker inside a list item or table cell overrides that weak tag role.
4. **Position**, for whatever is left. Neighbours are the nodes before and after in reading order that share the same
   group (the nearest ancestor holding more than one piece of text). A kicker directly before a heading is an
   `eyebrow`, otherwise a `label`; small or muted text of up to eight words directly before a heading is an
   `eyebrow`; text directly after a heading is a `subtitle` when it is small, muted, or the only line before the next
   control or heading; small text elsewhere is a `caption`; everything else is `body`.

After that, headings and body text inside a dialog become `dialog_title` and `dialog_body`, inside an empty state
`empty_title` and `empty_body`, inside a toast `toast`; secondary text inside a form field becomes `helper`. The
extractor never emits `comment` or `code` (track C is deferred).

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
the literal's property (`title: "…"`) instead; the first node in a copy with no literal of its own (a constant
label, or the VAR text of the index) takes the span of the whole array element; and copy text left without a free
span keeps its span in the callback in the first copy and is dropped from later ones. Visible nodes choose before
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
`expected.json`, runs each one twice to check determinism, validates the schema, and fails unless there are at least
60 fixtures, every role bar `comment` and `code` and every flag and cluster kind appears somewhere, and the Swift
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
