"""List every non-zero block of the revised network (Phase B + REVISION.edits) for CIRCUITS.md.
Writes scratch/nz_summary.txt. Rows of 2-D tensors are listed with their non-zero columns collapsed into ranges;
tok_emb is summarised per column (row ranges); rows with identical patterns are grouped."""
import os

import torch

import v05_load


def ranges(idx):
    idx = sorted(int(i) for i in idx)
    out = []
    for i in idx:
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in out)


def main():
    m, _ = v05_load.apply_edits_file()
    sd = m.state_dict()
    lines = []
    for name, t in sd.items():
        nz = int((t != 0).sum())
        lines.append(f"### {name} {tuple(t.shape)} nonzero={nz}")
        if nz == 0:
            continue
        if name == "tok_emb.weight":
            for col in range(t.shape[1]):
                rows = torch.nonzero(t[:, col]).flatten().tolist()
                if rows:
                    vals = sorted(set(round(float(v), 4) for v in t[rows, col].tolist()))
                    lines.append(f"  col {col}: {len(rows)} rows [{ranges(rows)[:160]}] values {vals[:6]}")
            continue
        if t.ndim == 1:
            idx = torch.nonzero(t).flatten().tolist()
            groups = {}
            for i in idx:
                groups.setdefault(round(float(t[i]), 4), []).append(i)
            for v, ii in sorted(groups.items()):
                lines.append(f"  value {v}: [{ranges(ii)}]")
            continue
        # 2-D: group rows by their (cols, values) signature
        sig = {}
        for r in range(t.shape[0]):
            cols = torch.nonzero(t[r]).flatten().tolist()
            if not cols:
                continue
            key = tuple((c, round(float(t[r, c]), 4)) for c in cols)
            sig.setdefault(key, []).append(r)
        for key, rows in sig.items():
            desc = " ".join(f"{c}:{v}" for c, v in key)
            lines.append(f"  rows [{ranges(rows)}] -> {desc[:200]}")
    path = os.path.join(v05_load.OUTPUT, "scratch", "nz_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", path, len(lines), "lines")


if __name__ == "__main__":
    main()
