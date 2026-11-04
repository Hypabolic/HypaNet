Implemented Revision 2: independent lockfile fetch/install/build mechanics, causal ABI resolution, cross-family uniqueness and operation-multiset tests for seeds 0–49, and updated documentation. Existing variety, transitions, lookalikes, and telemetry checks pass. Shared contracts remain unchanged.

`PYTHONPATH=src .venv/bin/python -m pytest tests_ood -q`

**315 passed in 21.37s.**