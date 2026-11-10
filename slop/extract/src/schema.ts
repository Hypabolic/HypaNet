// Closed lists from SPEC.md §4. Order matters: ids are fixed by position (ui/tokenizer.py mirrors them),
// and flags are always written in this order.

export const ROLES = [
  'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'eyebrow', 'subtitle', 'body', 'list_item',
  'caption', 'label', 'helper', 'placeholder', 'input_value', 'button', 'link', 'nav', 'tab', 'breadcrumb',
  'menu_item', 'badge', 'tooltip', 'empty_title', 'empty_body', 'dialog_title', 'dialog_body', 'toast', 'error', 'stat',
  'cell', 'aria', 'alt', 'comment', 'code', 'other',
] as const;
export type Role = (typeof ROLES)[number];

export const FLAGS = [
  'caps', 'title_case', 'tracked', 'small', 'mono', 'pill', 'dot', 'icon', 'emoji_lead', 'arrow_tail',
  'dynamic', 'stateful', 'conditional', 'in_nav', 'in_table', 'in_form', 'in_list', 'in_dialog', 'in_header',
  'in_footer', 'in_card', 'in_state', 'in_control', 'section_first', 'section_last', 'repeated', 'sr_only', 'truncated',
] as const;
export type Flag = (typeof FLAGS)[number];

export const CLUSTER_KINDS = ['header', 'actions', 'control', 'state', 'card', 'list', 'body'] as const;
export type ClusterKind = (typeof CLUSTER_KINDS)[number];

export type Framework = 'tsx' | 'html' | 'swift';

export const SCHEMA = 'slop.screen/1';

/** Roles the extractor never emits in r1 (track C is deferred). */
export const TRACK_C_ROLES: ReadonlySet<Role> = new Set<Role>(['comment', 'code']);

export const HEADING_LEVEL: Partial<Record<Role, number>> = {
  h1: 1, h2: 2, h3: 3, h4: 4, h5: 5, h6: 6, dialog_title: 1, empty_title: 1,
};

/** Roles read as context only: never adjacent for position rules, never section_first/last. */
export const CONTEXT_ROLES: ReadonlySet<Role> = new Set<Role>(['aria', 'alt', 'tooltip']);

export function isRole(x: string): x is Role {
  return (ROLES as readonly string[]).includes(x);
}
