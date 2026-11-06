"""Plan v0.5 (Exp-5-style tool-free weight writing) harness code.

The author model writes every weight of the unchanged v0.3 network as ``edits`` statements in its
replies, with no tools and no feedback. :mod:`.session`, :mod:`.runner`, :mod:`.audit` and
:mod:`.cost` run and check the author conversation; :mod:`.edits` reads its replies into a v0.3
state dict and an apply report. None of these imports the generators, the datasets or the oracle.
The one exception is :mod:`.scoring` (adjudicator only, never imported by the others), which scores
a finished network on the validation split through the v0.3 and v0.4 scoring code.
"""
