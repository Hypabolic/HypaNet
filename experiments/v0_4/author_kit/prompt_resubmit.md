The harness rejected your **{PHASE}** submission. This is a build rejection, not behavioural
feedback: the error text below is the only thing disclosed to you, and nothing has been evaluated.

Harness error (verbatim):

```
{ERROR_TEXT}
```

Fix only what this error requires, in `output/{PHASE}/`, under exactly the same rules as before
(read only inside this directory; one `./py` or `./fwd` command per Bash call). Re-read the
`build_edit.py` rules in `packet/AUTHOR_INSTRUCTIONS.md` before you finish, make sure the kernel's
edit state is the one `build_edit.py` builds, and record the fix in your iteration log. This is
resubmission {N} of at most {CAP}. When you are done, state clearly that your {PHASE} submission is
final, with a one-paragraph summary of what you changed.
