"""Plan v0.4 (Qwen3.5-0.8B baselines and weight-space programming) harness code.

This package runs in the frozen v0.3 environment. It exports the v0.3 task (rendered sessions and
window rows) for the separate Qwen environment in ``envs/qwen``, and scores that environment's
predictions with the unchanged v0.3 metrics. The Qwen environment never imports the generators,
the oracle or this package.
"""
