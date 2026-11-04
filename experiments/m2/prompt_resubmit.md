The harness rejected your **{PHASE}** submission. This is a build/compile rejection, not behavioural
feedback: no forward pass was run, no data or labels exist, and the error text below is the only
thing disclosed to you (spec §8, §9.2).

Harness error (verbatim):

```
{ERROR_TEXT}
```

Fix only what this error requires, in `output/{PHASE}/`, under exactly the same rules as before
(same allowed commands, run each on its own; read only inside this working directory). Re-read
`packet/AUTHOR_INSTRUCTIONS.md` ("What the harness rejects") before you finish. This is resubmission
{N} of at most {CAP}. When done, print a one-paragraph summary of what you changed.
