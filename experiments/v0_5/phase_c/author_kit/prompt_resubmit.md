The harness rejected your **{PHASE}** submission. This is a build rejection, not behavioural
feedback: the error text below is the only thing disclosed to you, and nothing has been evaluated.

Harness error (verbatim):

```
{ERROR_TEXT}
```

Fix only what this error requires, in `output/{PHASE}/`, under exactly the same rules as before:
read only inside this directory, and run one `./py` command per Bash call. Check `REVISION.edits` with
`./py -m v05_load check` before you finish, and record the fix in `ITERATIONS.md` and `REVISION.md`.
This is resubmission {N} of at most {CAP}. When you are done, state clearly that your {PHASE}
submission is final, and give a one-paragraph summary of what you changed.
