"""v0.4 Qwen3.5-0.8B harness (plan v0.4). Runs only in the ``envs/qwen`` environment.

Reads the task exports in ``datasets/v0_4`` (rendered blocks, window positions). It never imports
``agent_loop_lab``, the generators or the oracle.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
V04_DATA = REPO / "datasets" / "v0_4"

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
