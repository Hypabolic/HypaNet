// Role and container tables. Stage 1 (tag and attribute) and stage 2 (component name) live here; stage 3 (class
// tokens) is in style.ts and infer.ts; stage 4 (position) is in infer.ts.

import type { Box, El } from './ir.ts';
import type { Framework, Role } from './schema.ts';
import type { SlopConfig } from './style.ts';
import { baseToken, isCardSurface } from './style.ts';

export const TEXT_BLOCK_TAGS = new Set([
  'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th', 'dt', 'dd', 'label', 'legend', 'figcaption', 'caption',
  'button', 'a', 'summary', 'option', 'blockquote', 'q', 'pre', 'title', 'span', 'small', 'strong', 'em', 'b', 'i', 'text',
]);

export const INLINE_TAGS = new Set([
  'tspan', 'textPath', 'a', 'abbr', 'b', 'bdi', 'bdo', 'cite', 'code', 'data', 'dfn', 'em', 'i', 'kbd', 'mark', 'q', 's', 'samp', 'small',
  'span', 'strong', 'sub', 'sup', 'time', 'u', 'var', 'del', 'ins', 'wbr', 'br', 'font',
]);

export const MONO_TAGS = new Set(['code', 'kbd', 'pre', 'samp', 'tt']);

const HEADING_TAG = /^h([1-6])$/;

/** Stage 1: tag and attribute roles of an intrinsic element (attribute first, then tag). */
export function tagRole(e: El): Role | null {
  const role = e.attrs.role;
  if (role) {
    if (role === 'button' || role === 'switch') return 'button';
    if (role === 'link') return 'link';
    if (role === 'tab') return 'tab';
    if (/^menuitem/.test(role) || role === 'option') return 'menu_item';
    if (role === 'heading') {
      const lv = parseInt(e.attrs['aria-level'] ?? '2', 10);
      return (`h${Math.min(6, Math.max(1, lv || 2))}`) as Role;
    }
    if (role === 'tooltip') return 'tooltip';
    if (role === 'listitem') return 'list_item';
    if (role === 'cell' || role === 'gridcell') return 'cell';
    if (role === 'columnheader' || role === 'rowheader') return 'label';
  }
  if (!e.intrinsic) return null;
  const n = e.name;
  const m = n.match(HEADING_TAG);
  if (m) return (`h${m[1]}`) as Role;
  switch (n) {
    case 'label': case 'legend': case 'th': case 'dt': return 'label';
    case 'button': case 'summary': return 'button';
    case 'a': return 'link';
    case 'figcaption': case 'caption': return 'caption';
    case 'li': return 'list_item';
    case 'td': return 'cell';
    case 'dd': return 'body';
    case 'option': return 'menu_item';
    default: return null;
  }
}

const TYPO_VARIANT: Record<string, Role> = {
  h1: 'h1', h2: 'h2', h3: 'h3', h4: 'h4', h5: 'h5', h6: 'h6', overline: 'eyebrow', eyebrow: 'eyebrow',
  subtitle: 'subtitle', subtitle1: 'subtitle', subtitle2: 'subtitle', lead: 'subtitle', caption: 'caption',
  body: 'body', body1: 'body', body2: 'body', button: 'button', label: 'label', small: 'caption', muted: 'caption',
};

const AS_ROLE: Record<string, Role> = {
  h1: 'h1', h2: 'h2', h3: 'h3', h4: 'h4', h5: 'h5', h6: 'h6', label: 'label', button: 'button', a: 'link', li: 'list_item',
  figcaption: 'caption', legend: 'label', th: 'label', td: 'cell',
};

/** Component-name table, first match wins. Applied to non-intrinsic elements (JSX components, SwiftUI views). */
const COMPONENT_ROLES: [RegExp, Role][] = [
  [/^(AlertDialog|Dialog|Sheet|Drawer|Modal|Popover)(Action|Cancel|Close)$/, 'button'],
  [/^(AlertDialog|Dialog|Sheet|Drawer|Modal|Popover|Confirm)\w*Title$/, 'dialog_title'],
  [/^(AlertDialog|Dialog|Sheet|Drawer|Modal|Popover|Confirm)\w*(Description|Body|Message)$/, 'dialog_body'],
  [/^Empty(State)?(Title|Heading)$/, 'empty_title'],
  [/^Empty(State)?(Description|Body|Text|Message)$/, 'empty_body'],
  [/^Toast(Title|Description|Message)?$/, 'toast'],
  [/^(Form|Field)?(Message|Error)$|ErrorMessage$|^FieldError$|^ErrorText$|^InlineError$/, 'error'],
  [/^(Form|Field|Input)(Description|Hint|Helper|HelperText)$|HelperText$|^Hint$/, 'helper'],
  [/^AlertTitle$/, 'h4'],
  [/^AlertDescription$/, 'body'],
  [/^Tooltip(Content)?$|^Tip$/, 'tooltip'],
  [/^(Page|Hero|Screen)(Title|Heading)$/, 'h1'],
  [/^Section(Title|Heading)$/, 'h2'],
  [/Title$/, 'h3'],
  [/^Heading$|Heading$/, 'h2'],
  [/(Subtitle|Subheading|Subhead|Tagline|Lead|Lede)$/, 'subtitle'],
  [/(Eyebrow|Kicker|Overline|Pretitle)$/, 'eyebrow'],
  [/Description$/, 'subtitle'],
  [/^(TableHead|TableHeaderCell|Th)$/, 'label'],
  [/^(TableCell|Td)$/, 'cell'],
  [/Caption$/, 'caption'],
  [/(Badge|Chip|Tag|Pill)$/, 'badge'],
  [/^(Tabs?Trigger|Tab)$/, 'tab'],
  [/^Breadcrumb(Item|Link|Page)$/, 'breadcrumb'],
  [/^(Nav|NavBar|Navbar|Sidebar|SidebarMenu)(Item|Link|Button|MenuButton|MenuItem)$|^NavigationMenu(Link|Trigger)$/, 'nav'],
  [/^(DropdownMenu|ContextMenu|Menubar)?(Item|CheckboxItem|RadioItem)$|^(Select|Command|Combobox|Listbox)(Item|Option)$|^MenuItem$|^Option$/, 'menu_item'],
  [/(Button|Btn)$|^(Pressable|TouchableOpacity|TouchableHighlight|TouchableWithoutFeedback)$/, 'button'],
  [/Link$/, 'link'],
  [/^(Stat|Metric|Kpi|KPI)(Value|Number)?$|(Stat|Metric)Value$/, 'stat'],
  [/(Form|Field|Input)?Label$/, 'label'],
  [/^Kbd$/, 'other'],
];

/** SwiftUI view names that carry no role of their own (they take the role of their context). */
const SWIFT_NEUTRAL = new Set(['Text', 'Label', 'Section', 'Menu', 'Toggle', 'Picker', 'Stepper', 'GroupBox', 'ProgressView']);

/** Stage 2: component name, and the `as`/`variant`/`level` attributes of typography components. */
export function componentRole(e: El, fw: Framework, cfg: SlopConfig): Role | null {
  if (e.intrinsic) return null;
  const name = e.name;
  const c = cfg.componentRoles.get(name);
  if (c) return c;
  if (fw === 'swift' && SWIFT_NEUTRAL.has(name)) return null;
  const as = e.attrs.as;
  if (as && AS_ROLE[as]) return AS_ROLE[as];
  if (/^(Typography|Text|Heading|Title|Paragraph)$/.test(name)) {
    const v = e.attrs.variant ?? e.attrs.level ?? e.attrs.size;
    if (v && TYPO_VARIANT[v]) return TYPO_VARIANT[v];
    if (name === 'Heading' && e.attrs.level && /^[1-6]$/.test(e.attrs.level)) return (`h${e.attrs.level}`) as Role;
    if (name === 'Text' || name === 'Paragraph') return null;
  }
  if (e.attrs.role) {
    const r = tagRole({ ...e, intrinsic: false });
    if (r) return r;
  }
  for (const [re, role] of COMPONENT_ROLES) if (re.test(name)) return role;
  return null;
}

/** Strong role of an element: front-end hint, then tag/attribute, then component name. */
export function strongRole(e: El, fw: Framework, cfg: SlopConfig): Role | null {
  if (e.hint) return e.hint;
  return tagRole(e) ?? componentRole(e, fw, cfg);
}

// ---------------------------------------------------------------------------------------------------------------
// Props that carry visible text on components (JSX props, SwiftUI labelled arguments)

const DIALOGISH = /(Dialog|Modal|Sheet|Drawer|Popover|Confirm)/;
const EMPTYISH = /^Empty|EmptyState/;
const TOASTISH = /Toast|Snackbar|Notification/;
const BUTTONISH = /(Button|Btn|IconButton|Action)$/;

/** An icon-only button component: its label is read by assistive technology, not shown. */
export const ICON_BUTTON = /Icon\w*(Button|Btn)$|^IconButton/;

export function propRole(comp: string, prop: string): Role | null {
  const p = prop;
  if (p === 'aria-label' || p === 'ariaLabel' || p === 'accessibilityLabel') return 'aria';
  if (ICON_BUTTON.test(comp) && /^(label|text|labelText)$/.test(p)) return 'aria';
  if (p === 'alt') return 'alt';
  if (p === 'placeholder' || p === 'searchPlaceholder') return 'placeholder';
  if (/^Tooltip/.test(comp) && /^(content|label|title|text)$/.test(p)) return 'tooltip';
  if (/^(tooltip|tip|hint)$/.test(p) && !/^(Form|Field|Input)/.test(comp)) return p === 'hint' ? 'helper' : 'tooltip';
  if (/^(helperText|helper|help|helpText|description)$/.test(p) && /^(Form|Field|Input|TextField|Select|Textarea|Checkbox|Switch)/.test(comp)) return 'helper';
  if (/^(errorMessage|errorText|error)$/.test(p)) return 'error';
  if (p === 'title') {
    if (DIALOGISH.test(comp)) return 'dialog_title';
    if (EMPTYISH.test(comp)) return 'empty_title';
    if (TOASTISH.test(comp)) return 'toast';
    if (BUTTONISH.test(comp) || /Icon|Link$/.test(comp)) return 'tooltip';
    if (/^(Page|Hero|Screen)/.test(comp)) return 'h1';
    if (/^Section/.test(comp)) return 'h2';
    return 'h3';
  }
  if (/^(emptyTitle)$/.test(p)) return 'empty_title';
  if (/^(emptyText|emptyMessage|emptyDescription)$/.test(p)) return 'empty_body';
  if (/^(confirmTitle|dialogTitle)$/.test(p)) return 'dialog_title';
  if (/^(heading|headline)$/.test(p)) return /^(Page|Hero|Screen)/.test(comp) ? 'h1' : 'h2';
  if (/^(subtitle|subheading|subheadline|tagline|lead|lede|subtext|subTitle)$/.test(p)) return 'subtitle';
  if (/^(eyebrow|kicker|overline|pretitle|superTitle|supertitle)$/.test(p)) return 'eyebrow';
  if (p === 'description') {
    if (DIALOGISH.test(comp)) return 'dialog_body';
    if (EMPTYISH.test(comp)) return 'empty_body';
    if (TOASTISH.test(comp)) return 'toast';
    return 'subtitle';
  }
  if (/^(body|text|message|content|summary|blurb|copy)$/.test(p)) {
    if (DIALOGISH.test(comp)) return 'dialog_body';
    if (EMPTYISH.test(comp)) return 'empty_body';
    if (TOASTISH.test(comp)) return 'toast';
    return 'body';
  }
  if (p === 'label') {
    if (BUTTONISH.test(comp)) return 'button';
    if (/^Tab/.test(comp)) return 'tab';
    if (/Link$/.test(comp)) return 'link';
    if (/MenuItem$|Option$/.test(comp)) return 'menu_item';
    if (/(Badge|Chip|Tag|Pill)$/.test(comp)) return 'badge';
    return 'label';
  }
  if (/^(caption|footnote|note|footer)$/.test(p)) return 'caption';
  if (/^(badge|tag|chip|pill|badgeText|badgeLabel)$/.test(p)) return 'badge';
  if (/^(cta|ctaLabel|ctaText|ctaTitle|buttonText|buttonLabel|buttonTitle|actionLabel|actionText|actionTitle|confirmText|confirmLabel|cancelText|cancelLabel|cancelTitle|submitText|submitLabel|submitTitle|primaryAction|secondaryAction|primaryActionTitle|secondaryActionTitle|primaryActionLabel|secondaryActionLabel)$/.test(p)) return 'button';
  if (/^(linkText|linkLabel)$/.test(p)) return 'link';
  if (/^(value|stat|metric|figure)$/.test(p) && /(Stat|Metric|Kpi|KPI)/.test(comp)) return 'stat';
  return null;
}

// ---------------------------------------------------------------------------------------------------------------
// Containers

const ICON_SOURCES = /^(lucide-react|lucide-react-native|lucide-vue-next|react-icons(\/.*)?|@heroicons\/.*|@radix-ui\/react-icons|@phosphor-icons\/.*|phosphor-react|@tabler\/icons-react|react-feather|@mui\/icons-material(\/.*)?|@fortawesome\/react-fontawesome|@iconify\/react|@primer\/octicons-react|iconoir-react|@remixicon\/react|react-native-vector-icons(\/.*)?|@expo\/vector-icons)$/;

export function isIconImport(source: string | undefined): boolean {
  return source !== undefined && (ICON_SOURCES.test(source) || /(^|\/)icons?(\/|$)/i.test(source));
}

/** Container and shape decoration for intrinsic (HTML) elements. */
export function decorateIntrinsic(e: El): void {
  const n = e.name;
  const role = e.attrs.role;
  const toks = e.cls.map(baseToken).filter((t): t is string => t !== null).map((t) => t.toLowerCase());
  const has = (re: RegExp): boolean => toks.some((t) => re.test(t));
  if (TEXT_BLOCK_TAGS.has(n)) e.textBlock = true;
  if (INLINE_TAGS.has(n)) e.inline = true;
  if (n === 'svg') e.icon = true;
  if (n === 'i' && (has(/^(fa|fas|far|fab|fa-|bi|bi-|icon|icon-|ph|ph-|ri-|material-icons|material-symbols)/) || toks.length === 0)) e.icon = true;
  if ((n === 'span' || n === 'i') && has(/^material-(icons|symbols)/)) e.icon = true;
  if (e.icon && (has(/arrow|chevron-right|caret-right/))) e.arrowIcon = true;
  if (n === 'input') {
    const t = (e.attrs.type ?? 'text').toLowerCase();
    if (!/^(hidden|submit|button|reset|image)$/.test(t)) e.control = true;
  }
  if (n === 'textarea' || n === 'select') e.control = true;
  let box: Box | undefined;
  if (n === 'dialog') box = 'dialog';
  else if (n === 'nav') box = /breadcrumb/i.test(e.attrs['aria-label'] ?? '') ? 'breadcrumb' : 'nav';
  else if (n === 'ul' || n === 'ol' || n === 'dl' || n === 'menu') box = has(/breadcrumb/) || /breadcrumb/i.test(e.attrs['aria-label'] ?? '') ? 'breadcrumb' : 'list';
  else if (n === 'table') box = 'table';
  else if (n === 'select' || n === 'datalist') box = 'menu';
  else if (n === 'form' || n === 'fieldset') box = 'form';
  else if (n === 'header') box = 'header';
  else if (n === 'footer') box = 'footer';
  else if (n === 'section' || n === 'article' || n === 'main' || n === 'aside' || n === 'details') box = 'section';
  if (role) {
    if (role === 'dialog' || role === 'alertdialog') box = 'dialog';
    else if (role === 'alert' || role === 'status') box = 'alert';
    else if (role === 'navigation') box = 'nav';
    else if (role === 'tablist') box = 'tablist';
    else if (role === 'menu' || role === 'menubar' || role === 'listbox') box = 'menu';
    else if (role === 'list') box = 'list';
    else if (role === 'table' || role === 'grid') box = 'table';
    else if (role === 'banner') box = 'header';
    else if (role === 'contentinfo') box = 'footer';
    else if (role === 'region') box = 'section';
    else if (role === 'form' || role === 'search') box = 'form';
  }
  if (n === 'small') e.sty.small = true;
  if ((!box || box === 'section') && (!TEXT_BLOCK_TAGS.has(n) || n === 'li' || n === 'a')) {
    if (has(/(^|[-_])(empty-?state|empty)$/) || has(/^empty-/)) box = 'empty';
    else if (has(/(^|[-_])(toast|snackbar)$/)) box = 'toast';
    else if (has(/(^|[-_])(modal|dialog)$/)) box = 'dialog';
    else if (has(/^alert$|^alert-(danger|warning|info|success|error)$|^callout$/)) box = 'alert';
    else if (has(/^(tabs|tab-list|tablist)$/)) box = 'tablist';
    else if (has(/^breadcrumbs?$/)) box = 'breadcrumb';
    else if (isCardSurface(e.cls)) box = 'card';
  }
  if (box) e.box = box;
}

/** Container and shape decoration for components (JSX) by name. */
export function decorateComponent(e: El, importSource: string | undefined): void {
  const n = e.name;
  if (isIconImport(importSource) || /Icon$|^Icons?[A-Z]|^Icon$|^Lucide/.test(n)) {
    e.icon = true;
    if (/Arrow|ChevronRight|CaretRight|MoveRight|ExternalLink/.test(n)) e.arrowIcon = true;
  }
  if (/Link$/.test(n) || /^(Trans|Strong|Em|Code|Highlight|Mark)$/.test(n)) e.inline = true;
  if (n === 'Text') {
    // React Native: nested <Text> is one run of text
    e.inline = true;
    e.textBlock = true;
  }
  if (/^(Input|Textarea|TextArea|Select|Switch|Checkbox|Radio|RadioGroup|RadioGroupItem|Slider|Combobox|DatePicker|TextField|NumberInput|PasswordInput|InputOTP|OTPInput|SearchInput|TextInput)$/.test(n)) e.control = true;
  if (/(^|[a-z])Dot$|^Dot[A-Z]?|^StatusDot$|^(Status)?Indicator$|^Pulse$/.test(n)) e.dot = true;
  let box: Box | undefined;
  if (/^(AlertDialog|Dialog|Modal|Sheet|Drawer|Popover)(Content|Panel)$|(Dialog|Modal)$/.test(n) && !/^(AlertDialog|Dialog|Sheet|Drawer|Popover)$/.test(n)) box = 'dialog';
  else if (/^(Toast|Toaster|Snackbar)$/.test(n)) box = 'toast';
  else if (/^(Alert|Callout|Banner|Notice)$/.test(n)) box = 'alert';
  else if (/^Empty(State)?$|EmptyState$/.test(n)) box = 'empty';
  else if (/Card(Root)?$|Tile$/.test(n)) box = 'card';
  else if (/^Table$/.test(n)) box = 'table';
  else if (/^(DropdownMenuContent|ContextMenuContent|MenubarContent|SelectContent|CommandList|CommandGroup|Menu|MenuList|Listbox)$/.test(n)) box = 'menu';
  else if (/^(TabsList|TabList)$/.test(n)) box = 'tablist';
  else if (/^(Breadcrumb|BreadcrumbList|Breadcrumbs)$/.test(n)) box = 'breadcrumb';
  else if (/^(Nav|Navbar|NavBar|Navigation|NavigationMenu|NavigationMenuList|Sidebar|SidebarMenu|SidebarNav|MainNav|SiteNav|MobileNav|TopNav)$/.test(n)) box = 'nav';
  else if (/^(Header|SiteHeader|PageHeader|AppHeader|TopBar|Topbar|AppBar|MainHeader|GlobalHeader)$/.test(n)) box = 'header';
  else if (/^(Footer|SiteFooter|PageFooter|AppFooter)$/.test(n)) box = 'footer';
  else if (/^(Form)$/.test(n)) box = 'form';
  else if (/^(Section|PageSection)$/.test(n)) box = 'section';
  else if (/^(List|ItemList)$/.test(n)) box = 'list';
  if (!box && isCardSurface(e.cls)) box = 'card';
  if (box) e.box = box;
}

export const STATEISH: ReadonlySet<Box> = new Set<Box>(['dialog', 'state', 'empty', 'toast', 'alert']);
export const LISTISH: ReadonlySet<Box> = new Set<Box>(['list', 'nav', 'tablist', 'menu', 'breadcrumb', 'table']);
