# Latent effect rules (shared by every generator)

Excerpt of the harness's M1 contract, section 1. Binding for the OOD generator.

## Causal rules

The engine decides each operation's effect from **latent state**, never from encoded features or fingerprints. It applies these rules in this order:

1. **Episodes.**
   - A failure *episode* opens at the first **exact retry**: the same operation on the same relevant state, immediately after it failed or returned not-ready. It does not open on the initial failure. That keeps an ordinary failing test in an edit-then-test cycle from turning a window into RECOVERY.
   - The initial failing operation has effect `NONE`.
   - Kind `retry` (ERROR outcomes): attempts 1..`R_max` are `IN_POLICY_RETRY`, and later attempts are `REDUNDANT`.
   - Kind `poll` (NEUTRAL not-ready outcomes): attempts 1..`P_max` are `IN_POLICY_POLL`, and later attempts are `REDUNDANT`.
   - The operation that succeeds ends the episode: effect `RESOLVES_EPISODE`, `resolved_episode_here = True`.
   - An episode is abandoned when the agent does something else and never retries. It is then closed silently (`open_episode_kind = None`) at the first non-retry operation that changes relevant state, or after 2 unrelated operations.
   - At most one episode is open at a time.
   - Re-running validation on an unchanged workspace after a deterministic failure is still a retry, so attempts within `R_max` are in policy. Retries beyond `R_max` are `REDUNDANT`.
2. **Progress.** An operation that raises latent progress (`progress_after > previous`) is `PROGRESS`. Progress is credited to the mutation that fixes a fault, not to the test run that reveals the fix.
3. **Facts.** An operation that adds facts to `K` (clues about a fault, revealed by inspecting the right location, a diagnostic, or a test run that newly reveals a failure location) is `NEW_FACT`, with `new_facts` = the number added.
4. **Redundancy (§6.1 with the r2 reading in §0.2).** `REDUNDANT` covers:
   - re-reading content whose facts are all already in `K`, or re-reading content already read that yielded no facts;
   - an edit whose **behavioural** change is already present: it changes text but not behaviour, such as reformatting, renaming locals or re-adding an existing guard;
   - validation re-run when the **behavioural** workspace is unchanged since the last run of the same suite and selection, unless it is an in-policy retry under rule 1;
   - out-of-policy retries and polls;
   - re-issuing an identical plan or control step with no new information.
5. **Reversion.** An agent mutation whose latent after-state equals a latent workspace state that existed earlier in the session, and that undoes an earlier agent mutation, sets `undoes_prior_mutation = True`. Its effect is `REVERSION` if the matching state lies within its own trailing 8-event window. The labeller recomputes reversions per window.
6. Everything else is `NONE`: a novel but fruitless inspection, a novel behaviour-changing edit that doesn't help, or a first plan.

If an operation qualifies for several effects, precedence is RESOLVES_EPISODE > PROGRESS > NEW_FACT > REVERSION > IN_POLICY_RETRY/POLL > REDUNDANT > NONE. `new_facts` and `progress_after` are always recorded as they are, whatever the precedence.

`ws_before` / `ws_after` are latent workspace content ids: sha256 of the true file contents, which are unaffected by dropped observations. `CanonicalEvent.workspace_state_*` uses `fingerprint.workspace_digest` over the same contents, so the two agree whenever the observation isn't dropped.
