"""Parallel readout benchmark: run hypaqwen.run_readout as several worker processes over one fixed session set.

    cd envs/qwen && uv run python ../../experiments/v0_4/benchmarks/parallel_readout/bench.py NAME mps:0 [mps:0 ...]   # one worker per spec (device:threads; threads 0 = the readout's default)

Sessions (first N_SESS of VAL_QUICK.txt) are dealt round-robin to the workers. Prints and appends to results.jsonl:
wall time, per-worker loop seconds and tokens, aggregate tok/s, and min free memory % seen while running.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
QWEN = REPO / "envs" / "qwen"
HERE = Path(__file__).resolve().parent  # runs/ (gitignored outputs) and results.jsonl go here
N_SESS = 48


def free_pct() -> int:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "free percentage" in line:
            return int(line.rsplit(":", 1)[1].strip().rstrip("%"))
    return -1


def main() -> int:
    name, specs = sys.argv[1], sys.argv[2:]
    ids = (REPO / "datasets/v0_4/VAL_QUICK.txt").read_text().split()[:N_SESS]
    run = HERE / "runs" / name
    run.mkdir(parents=True, exist_ok=True)
    procs = []
    t0 = time.time()
    for w, spec in enumerate(specs):
        dev, thr = spec.split(":")
        sub = run / f"w{w}.txt"
        sub.write_text("".join(s + "\n" for s in ids[w::len(specs)]))
        out = run / f"w{w}.npz"
        out.unlink(missing_ok=True)
        cmd = ["uv", "run", "python", "-m", "hypaqwen.run_readout", "--split", "val", "--sessions", str(sub),
               "--device", dev, "--out", str(out)] + ([] if thr == "0" else ["--threads", thr])
        procs.append((spec, out, subprocess.Popen(cmd, cwd=QWEN, stdout=open(run / f"w{w}.log", "w"),
                                                  stderr=subprocess.STDOUT)))
    low = 100
    while any(p.poll() is None for _, _, p in procs):
        low = min(low, free_pct())
        time.sleep(3)
    wall = time.time() - t0
    workers = []
    for spec, out, p in procs:
        if p.returncode != 0:
            print(f"{spec}: exit {p.returncode}")
            workers.append({"spec": spec, "rc": p.returncode})
            continue
        meta = json.loads(str(np.load(out)["meta"]))
        workers.append({"spec": spec, "rc": 0, "seconds": meta["seconds"], "tokens": meta["tokens"],
                        "tok_s": meta["tokens_per_second"]})
    ok = [w for w in workers if w["rc"] == 0]
    tokens = sum(w["tokens"] for w in ok)
    loop = max(w["seconds"] for w in ok) if ok else float("nan")
    res = {"name": name, "specs": specs, "wall_s": round(wall, 1), "loop_s": round(loop, 1), "tokens": tokens,
           "agg_tok_s_loop": round(tokens / loop, 1), "agg_tok_s_wall": round(tokens / wall, 1),
           "min_free_mem_pct": low, "workers": workers}
    print(json.dumps(res))
    with open(HERE / "results.jsonl", "a") as f:
        f.write(json.dumps(res) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
