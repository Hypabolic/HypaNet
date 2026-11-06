"""List the non-zero blocks of every tensor of the revised network (r1 + REVISION.edits), grouped by rows."""
import sys

import torch

import v05_load


def runs(idx):
    """Compress a sorted list of ints into 'a-b' ranges."""
    out = []
    start = prev = None
    for i in idx:
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            out.append(f"{start}-{prev}" if prev > start else f"{start}")
            start = prev = i
    if start is not None:
        out.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ",".join(out)


def main():
    import os
    path = sys.argv[1] if len(sys.argv) > 1 else None
    m = v05_load.apply_edits_file(path)[0]
    sd = m.state_dict()
    out_path = os.path.join(v05_load.OUTPUT, "scratch", "nz_summary.txt")
    lines = []
    global print
    _print = print

    def print(*a):  # noqa: A001 - collect instead of printing
        lines.append(" ".join(str(x) for x in a))

    for name, t in sd.items():
        nz = (t != 0)
        n = int(nz.sum())
        if n == 0:
            continue
        print(f"== {name} shape {tuple(t.shape)} nonzero {n}")
        if t.ndim == 1:
            idx = nz.nonzero().flatten().tolist()
            vals = sorted(set(round(float(v), 4) for v in t[nz]))
            print(f"   idx {runs(idx)} values {vals[:12]}{'...' if len(vals) > 12 else ''}")
        else:
            if name == "tok_emb.weight":
                # group by column: which token ids are set
                for col in range(t.shape[1]):
                    ids = nz[:, col].nonzero().flatten().tolist()
                    if ids:
                        vals = sorted(set(round(float(v), 4) for v in t[ids, col]))
                        print(f"   col {col}: {len(ids)} ids [{runs(ids)[:90]}{'...' if len(runs(ids)) > 90 else ''}] values {vals}")
                continue
            if name == "off_emb.weight":
                for col in range(t.shape[1]):
                    ids = nz[:, col].nonzero().flatten().tolist()
                    if ids:
                        print(f"   col {col}: rows {runs(ids)}")
                continue
            # rows -> cols listing, grouped by identical column sets
            groups = {}
            for r in range(t.shape[0]):
                cols = nz[r].nonzero().flatten().tolist()
                if not cols:
                    continue
                key = (runs(cols), tuple(sorted(set(round(float(v), 4) for v in t[r, cols]))))
                groups.setdefault(key, []).append(r)
            for (cols, vals), rows in groups.items():
                print(f"   rows {runs(rows)} -> cols {cols} values {list(vals)[:8]}{'...' if len(vals) > 8 else ''}")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    _print("wrote", out_path, "lines", len(lines))


if __name__ == "__main__":
    main()
