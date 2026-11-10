#!/usr/bin/env python3
"""Build the labeller's test screens (slop.screen/1) from the invented sources in fixtures/src/.

Hand-written stand-ins for extractor output: roles, flags and clusters are chosen by hand, and each node's `loc` is
found in the source, so the snippet pane highlights the right span. All copy is invented except the pause menu, which
is the motivating example of SPEC.md. Not labels: these files carry no classes.

    python3 slop/labeller/fixtures/make_fixtures.py
"""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "src"

# node: (cluster, role, text, flags[, find]) — `find` is the source text when it differs from `text`; U+E000 is the
# extractor's VAR marker (SPEC.md §4)
FIXTURES = [
    ("01_pause_menu", "game-ui", "src/PauseMenu.tsx", "tsx", ["header", "actions"], [
        (0, "eyebrow", "OPERATION ON HOLD", ["caps", "tracked", "small", "in_header", "section_first"]),
        (0, "h1", "PAUSED", ["caps", "in_header"]),
        (0, "subtitle", "Resume recaptures the mouse.", ["small", "in_header"]),
        (1, "button", "RESUME", ["caps", "in_nav"]),
        (1, "button", "SETTINGS", ["caps", "in_nav"]),
        (1, "button", "CONTROLS", ["caps", "in_nav"]),
        (1, "button", "RESTART MISSION", ["caps", "in_nav", "section_last"]),
    ]),
    ("02_profile_form", "invoicer", "web/settings/profile.html", "html",
     ["header", "control", "control", "control", "actions"], [
         (0, "eyebrow", "Account", ["small", "in_header", "section_first"]),
         (0, "h2", "Profile", ["in_header"]),
         (0, "subtitle", "Update your profile details here.", ["small", "in_header"]),
         (1, "label", "Display name", ["in_form", "in_control"]),
         (1, "placeholder", "Display name", ["in_form", "in_control"]),
         (2, "label", "Postcode", ["in_form", "in_control"]),
         (2, "placeholder", "e.g. SW1A 1AA", ["in_form", "in_control"]),
         (2, "helper", "Enter your postcode", ["small", "in_form", "in_control"]),
         (3, "label", "Billing email", ["in_form", "in_control"]),
         (3, "helper", "Invoices and receipts go to this address.", ["small", "in_form", "in_control"]),
         (4, "button", "Save profile", ["in_form"]),
         (4, "other", "Saving can take a few seconds", ["sr_only", "in_form", "section_last"]),
     ]),
    ("03_landing_features", "kitchenkit", "app/page.tsx", "tsx",
     ["header", "header", "card", "card", "card", "header", "actions"], [
         (0, "badge", "Now in beta", ["pill", "dot", "small"]),
         (0, "h1", "Plan a week of dinners in ten minutes", []),
         (0, "subtitle", "Kitchenkit turns what is in your fridge into a plan and a list.", []),
         (0, "link", "Start free trial", []),
         (0, "caption", "No card needed", ["small"]),
         (1, "eyebrow", "FEATURES", ["caps", "tracked", "small", "section_first"]),
         (1, "h2", "Tools for small kitchens", []),
         (1, "body", "The three tools below cover planning, shopping and cooking.", []),
         (2, "badge", "● New", ["pill", "dot", "small", "in_card", "repeated"]),
         (2, "caption", "01 /", ["mono", "in_card", "repeated"]),
         (2, "h3", "Fridge scan", ["in_card", "repeated"]),
         (2, "body", "Snap the shelves and get a list of what you have.", ["in_card", "repeated"]),
         (3, "badge", "● New", ["pill", "dot", "small", "in_card", "repeated"]),
         (3, "caption", "02 /", ["mono", "in_card", "repeated"]),
         (3, "h3", "Week planner", ["in_card", "repeated"]),
         (3, "body", "Seven dinners built from that list and your tastes.", ["in_card", "repeated"]),
         (4, "badge", "● New", ["pill", "dot", "small", "in_card", "repeated"]),
         (4, "caption", "03 /", ["mono", "in_card", "repeated"]),
         (4, "h3", "Shop once", ["in_card", "repeated"]),
         (4, "body", "One shopping list, grouped by aisle.", ["in_card", "repeated", "section_last"]),
         (5, "h2", "Ready to cook?", ["section_first"]),
         (5, "link", "Start free trial", []),
         (5, "body", "That is the whole of Kitchenkit: scan, plan, shop.", ["small", "section_last"]),
         (6, "link", "Start free trial", ["in_footer"]),
     ]),
    ("04_invoice_list", "invoicer-ios", "Invoicer/InvoiceListView.swift", "swift",
     ["header", "state", "state"], [
         (0, "h1", "Invoices", ["in_nav"]),
         (0, "button", "Delete all", ["in_nav"]),
         (1, "empty_title", "No invoices", ["conditional", "in_state"]),
         (1, "empty_body", "There are no invoices yet.", ["conditional", "in_state"]),
         (1, "button", "Create invoice", ["conditional", "in_state"]),
         (2, "dialog_title", "Delete all invoices?", ["in_dialog"]),
         (2, "button", "Delete all invoices", ["in_dialog"]),
         (2, "button", "Cancel", ["in_dialog"]),
         (2, "dialog_body", "This also deletes \ue000 export count saved exports.", ["dynamic", "in_dialog"],
          "This also deletes \\(exportCount) saved exports."),
     ]),
    ("05_status_dashboard", "opsboard", "src/pages/Status.tsx", "tsx",
     ["body", "header", "list", "list", "body"], [
         (0, "nav", "Overview", ["in_nav"]),
         (0, "nav", "Status", ["in_nav"]),
         (0, "nav", "Billing", ["in_nav"]),
         (1, "h1", "Status", ["in_header"]),
         (1, "subtitle", "Checks run every minute from three regions.", ["in_header"]),
         (1, "badge", "● Live", ["pill", "dot", "small", "in_header"]),
         (1, "tooltip", "Refresh now", ["in_header"]),
         (1, "aria", "Refresh service status", ["in_header"]),
         (2, "list_item", "Healthy", ["dot", "in_list"]),
         (2, "list_item", "Degraded", ["dot", "in_list"]),
         (2, "list_item", "Down", ["dot", "in_list"]),
         (3, "cell", "Service", ["in_table"]),
         (3, "cell", "State", ["in_table"]),
         (3, "cell", "Uptime (30 days)", ["in_table"]),
         (3, "cell", "\ue000 name", ["dynamic", "in_table"], "{s.name}"),
         (3, "cell", "\ue000 state label", ["dynamic", "stateful", "dot", "in_table"], "{s.stateLabel}"),
         (3, "cell", "\ue000 uptime %", ["dynamic", "in_table"], "{s.uptime}%"),
         (4, "caption", "Last updated \ue000 updated at. All times are UTC.", ["small", "dynamic"],
          "Last updated {updatedAt}. All times are UTC."),
         (4, "caption", "The table above lists every service with its state and uptime.", ["small"]),
     ]),
    ("06_onboarding_step", "invoicer", "web/onboarding/step2.html", "html",
     ["header", "list", "body", "actions", "body"], [
         (0, "caption", "Step 2 of 4", ["small", "section_first"]),
         (0, "h1", "Connect your bank", []),
         (0, "subtitle", "This page walks you through the second of four onboarding stages.", []),
         (1, "body", "Three simple steps to connect:", []),
         (1, "list_item", "Choose your bank", ["in_list"]),
         (1, "list_item", "Sign in with your bank", ["in_list"]),
         (1, "list_item", "Pick the accounts to sync", ["in_list"]),
         (2, "helper", "We only read transactions; we never move money.", ["small"]),
         (2, "helper", "Hold Shift to pick several accounts at once.", ["small"]),
         (3, "button", "Continue", []),
         (3, "link", "Back", []),
         (4, "caption", "Now simply click Continue to move on.", ["small", "section_last"]),
     ]),
]


def locate(lines, find, start):
    """First occurrence of `find` at or after (line, col) `start`, wrapping to the top of the file once (reading
    order can differ from source order, e.g. a SwiftUI navigationTitle). Returns loc and the new start."""
    for line, col in (start, (0, 0)):
        for ln in range(line, len(lines)):
            c = lines[ln].find(find, col if ln == line else 0)
            if c >= 0:
                return [ln + 1, c + 1, ln + 1, c + 1 + len(find)], (ln, c + len(find))
    raise SystemExit(f"not found: {find!r}")


def build(name, repo, path, framework, cluster_kinds, nodes):
    lines = (SRC / repo / path).read_text().splitlines()
    out_nodes, pos = [], (0, 0)
    first_line = {}
    for i, spec in enumerate(nodes):
        k, role, text, flags = spec[:4]
        find = spec[4] if len(spec) > 4 else text
        loc, pos = locate(lines, find, pos)
        first_line.setdefault(k, loc[0])
        out_nodes.append({"i": i, "k": k, "role": role, "text": text, "flags": flags, "loc": loc})
    sid = hashlib.sha256(f"labeller-fixture:{repo}/{path}".encode()).hexdigest()[:16]
    return {"schema": "slop.screen/1", "id": sid, "framework": framework,
            "source": {"repo": repo, "path": path, "commit": "fixture", "licence": "invented fixture"},
            "clusters": [{"k": k, "kind": kind, "line": first_line.get(k, 1)} for k, kind in enumerate(cluster_kinds)],
            "nodes": out_nodes}


def main():
    for fx in FIXTURES:
        screen = build(*fx)
        text = json.dumps(screen, ensure_ascii=False, indent=1).replace("\ue000", "\\ue000").replace("\ue001", "\\ue001")
        (HERE / f"{fx[0]}.json").write_text(text + "\n")
        print(f"{fx[0]}.json  {screen['id']}  {len(screen['nodes'])} nodes")


if __name__ == "__main__":
    main()
