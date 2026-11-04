Created `rules.py`, `RULES.md`, and `test_rules.py`.

Precedence: OSCILLATING → STALLED → RECOVERY → PRODUCTIVE, using observable proxies for reversions, ineffective repetition, bounded retries, and validation progress. Assumptions and expected errors are documented.

All 13 tests passed with `python3 -m unittest test_rules -v`.