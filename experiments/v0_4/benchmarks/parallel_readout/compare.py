"""Check that parallel benchmark runs give the single-process logits: per session, max |logit difference| and
argmax changes against runs/mps1. Reads the label-free task export for window counts.

    envs/qwen/.venv/bin/python experiments/v0_4/benchmarks/parallel_readout/compare.py mps2 mps3 mps4
"""
import glob
import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def per_session(run: Path, npos: dict) -> dict:
    out = {}
    for f in sorted(glob.glob(str(run / "w*.npz"))):
        z, i = np.load(f), 0
        for s in z["session_ids"]:
            n = npos[str(s)]
            out[str(s)] = (z["label_logits"][i:i + n], z["row_pos"][i:i + n])
            i += n
        assert i == len(z["label_logits"]), f
    return out


def main() -> int:
    need = {str(s) for s in np.load(HERE / "runs" / "mps1" / "w0.npz")["session_ids"]}
    npos = {}
    with gzip.open(REPO / "datasets" / "v0_4" / "val.tasks.jsonl.gz", "rt") as f:
        for line in f:
            t = json.loads(line)
            if t["session_id"] in need:
                npos[t["session_id"]] = len(t["window_pos"])
    base = per_session(HERE / "runs" / "mps1", npos)
    for name in sys.argv[1:]:
        o = per_session(HERE / "runs" / name, npos)
        assert set(o) == set(base)
        md = max(float(np.abs(o[s][0] - base[s][0]).max()) for s in base)
        pos = all(np.array_equal(o[s][1], base[s][1]) for s in base)
        am = sum(int((o[s][0].argmax(1) != base[s][0].argmax(1)).sum()) for s in base)
        print(f"{name}: max |dlogit| {md}, row_pos equal {pos}, argmax changes {am} of {sum(len(v[0]) for v in base.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
