# Revision request 2 (from the harness implementer)

Revision 1 is a big improvement. One defect remains against spec §7.3:

- `merge_ping_pong` and `lockfile_churn` are the same generator (`_negotiation`) with a different
  file path: for every seed 0-49 they produce identical (category, outcome) sequences and identical
  label counts. §7.3 requires **at least two families absent from the §7.1 list**, which must be two
  genuinely different mechanisms.

Please:
1. Give `lockfile_churn` its own latent mechanics, distinct from `merge_ping_pong` (for example a
   resolver/installer loop: an EXTERNAL package fetch or install step, a lockfile rewrite, a build or
   test VALIDATE whose outcome depends on the resolved pins, an uninstall/reinstall or revert that
   restores an earlier lock state — with both oscillating and genuinely-resolving variants). It should
   use a different mix of operation families and categories from `merge_ping_pong`.
2. Check the other families for the same problem: no two families may share their core generator.
3. Add a test: for seeds 0-49, no pair of families produces an identical (category, outcome) session,
   and each pair of families differs in the multiset of operation families used across those seeds.
4. Keep everything else (tests passing, structural variety, transitions, lookalikes) and update
   `OOD_GENERATOR.md`.

Same constraints: read only this directory; do not modify the shared `src/agent_loop_lab/` files.
