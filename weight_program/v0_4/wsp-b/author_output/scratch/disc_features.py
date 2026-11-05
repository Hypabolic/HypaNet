"""Surface features of the 200 disclosed windows (last 8 blocks), per oracle class and per network class.
Writes output/wsp-b/scratch/disc_feat.json (per window) and prints class tables."""
import json
import re
import numpy as np

CL = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
D = json.load(open("disclosure/DISCLOSURE.json"))
items = D["windows"] if "windows" in D else D
if isinstance(items, dict):
    items = list(items.values())


def parse_blocks(prefix):
    lines = [ln for ln in prefix.split("\n") if ln != "### window start"]
    blocks, cur = [], None
    for ln in lines:
        if ln and not ln.startswith(" "):
            cur = {"header": ln, "body": []}
            blocks.append(cur)
        elif cur is not None and ln:
            cur["body"].append(ln)
    out = []
    for b in blocks:
        h = b["header"]
        parts = h.split(" ")
        tool = parts[1] if len(parts) > 1 else "?"
        m = re.search(r" -> (\S+)( (\d+|\?)/(\d+|\?) passed)?", h)
        status = m.group(1) if m else "?"
        passed = int(m.group(3)) if (m and m.group(3) and m.group(3) != "?") else None
        total = int(m.group(4)) if (m and m.group(4) and m.group(4) != "?") else None
        fields = h.split(" -> ")[0]
        core = re.sub(r"^\d\d:\d\d:\d\d\.\d{3} ", "", fields)
        core = re.sub(r" req=[0-9a-f]{8}", "", core)
        target = None
        mt = re.search(r"(path|scope|suite|target|resource|action|paths)=(\S+)", h)
        if mt:
            target = mt.group(2)
        elif tool == "http_call":
            target = " ".join(parts[2:4])
        hunks = []
        err = None
        for ln in b["body"]:
            mh = re.match(r"\s+@ (\S+) (h[0-9a-f]+|h0) > (h[0-9a-f]+|h0)", ln)
            if mh:
                hunks.append((mh.group(1), mh.group(2), mh.group(3)))
            me = re.match(r"\s+! (\S+) at (\S+):(\S+)", ln)
            if me:
                err = (me.group(1), me.group(2), me.group(3))
        out.append({"tool": tool, "status": status, "passed": passed, "total": total, "core": core + " -> " + status + (f" {passed}/{total}" if passed is not None else "") + (f" !{err[0]}@{err[1]}:{err[2]}" if err else ""),
                    "target": target, "hunks": hunks, "err": err, "query": re.search(r'query="([^"]*)"', h).group(1) if 'query="' in h else None})
    return out


def feats(blocks):
    n = len(blocks)
    win = blocks[-8:]
    prev = blocks[:-8]
    f = {}
    f["n_blocks"] = n
    statuses = [b["status"] for b in win]
    f["n_err"] = sum(1 for s in statuses if s in ("error", "FAILED", "pending"))
    f["last_err"] = statuses[-1] in ("error", "FAILED", "pending")
    f["n_pending"] = sum(1 for s in statuses if s == "pending")
    f["n_failed_val"] = sum(1 for b in win if b["tool"] in ("run_tests", "build", "lint") and b["status"] == "FAILED")
    f["n_ok_val"] = sum(1 for b in win if b["tool"] in ("run_tests", "build", "lint") and b["status"] == "ok")
    # repeats: blocks whose core equals an earlier block's core in the window
    cores = [b["core"] for b in win]
    f["n_repeat"] = sum(1 for i, c in enumerate(cores) if c in cores[:i])
    from collections import Counter
    cnt = Counter(cores)
    f["max_run"] = max(cnt.values())
    f["n_repeat_fail"] = sum(1 for i, b in enumerate(win) if b["core"] in cores[:i] and b["status"] in ("error", "FAILED", "pending"))
    f["n_repeat_read"] = sum(1 for i, b in enumerate(win) if b["core"] in cores[:i] and b["tool"] in ("read_file", "search", "list_dir"))
    # edits
    edits = [b for b in win if b["tool"] in ("edit_file", "write_file", "revert")]
    f["n_edits"] = len(edits)
    f["n_revert"] = sum(1 for b in win if b["tool"] == "revert")
    f["n_edit_nohunk"] = sum(1 for b in win if b["tool"] in ("edit_file", "write_file") and not b["hunks"] and b["status"] == "ok")
    paths = Counter(h[0] for b in edits for h in b["hunks"])
    f["max_edits_same_path"] = max(paths.values()) if paths else 0
    # reversals: a hunk whose (path, new) equals an earlier (path, old) in the window (returns to an earlier state)
    seen_old, seen_new = [], []
    rev = 0
    for b in win:
        for (p, old, new) in b["hunks"]:
            if (p, new) in seen_old:
                rev += 1
            seen_old.append((p, old)); seen_new.append((p, new))
    f["reversals"] = rev
    # ids that reappear (any id seen earlier in the window on the same path)
    ids = []
    reuse = 0
    for b in win:
        for (p, old, new) in b["hunks"]:
            if (p, new) in ids:
                reuse += 1
            ids.append((p, old)); ids.append((p, new))
    f["id_reuse"] = reuse
    # new facts proxy: reads/searches/list_dir whose core is not seen earlier in the whole prefix
    allcores_prev = [b["core"] for b in prev]
    newf = 0
    seen = list(allcores_prev)
    for b in win:
        if b["tool"] in ("read_file", "search", "list_dir", "http_call") and b["status"] == "ok":
            key = (b["tool"], b["target"], b["query"])
            if key not in [(x["tool"], x["target"], x["query"]) for x in prev] and key not in [(x["tool"], x["target"], x["query"]) for x in win[:win.index(b)]]:
                newf += 1
    f["new_facts"] = newf
    f["n_reads"] = sum(1 for b in win if b["tool"] in ("read_file", "search", "list_dir"))
    # resolution: an error/pending/FAILED block followed later in the window by an ok/ready of the same tool and target
    res_at = -1
    for i, b in enumerate(win):
        if b["status"] in ("ok", "ready") and any(x["tool"] == b["tool"] and x["target"] == b["target"] and x["status"] in ("error", "FAILED", "pending") for x in win[:i]):
            res_at = i
    f["resolved_offset"] = (7 - res_at) if res_at >= 0 else -1  # blocks since the resolution (0 = last block)
    # validation improvement: passed fraction increases for the same suite within the window
    vals = [(b["target"], b["passed"], b["total"]) for b in win if b["passed"] is not None and b["total"]]
    improve = 0
    for i in range(1, len(vals)):
        if vals[i][0] == vals[i - 1][0] and vals[i][2] and vals[i - 1][2] and vals[i][1] / vals[i][2] > vals[i - 1][1] / vals[i - 1][2]:
            improve += 1
    f["val_improve"] = improve
    f["last_tool"] = win[-1]["tool"]
    f["tools"] = " ".join(b["tool"][:4] + ("!" if b["status"] in ("error", "FAILED", "pending") else "") for b in win)
    return f


rows = []
for w in items:
    blocks = parse_blocks(w["prefix"])
    f = feats(blocks)
    f.update({"id": w["id"], "oracle": w["oracle_label"], "net": w["network_class"]})
    rows.append(f)
json.dump(rows, open("output/wsp-b/scratch/disc_feat.json", "w"), indent=1)
keys = ["n_blocks", "n_err", "last_err", "n_pending", "n_failed_val", "n_ok_val", "n_repeat", "max_run", "n_repeat_fail", "n_repeat_read", "n_edits", "n_revert", "n_edit_nohunk", "max_edits_same_path", "reversals", "id_reuse", "new_facts", "n_reads", "resolved_offset", "val_improve"]
print("per oracle class: mean of each feature (resolved_offset: fraction with a resolution in window)")
print(f"{'feature':20s} " + " ".join(f"{c[:5]:>7s}" for c in CL))
for k in keys:
    vals = []
    for c in CL:
        v = np.array([r[k] for r in rows if r["oracle"] == c], dtype=float)
        if k == "resolved_offset":
            vals.append((v >= 0).mean())
        else:
            vals.append(v.mean())
    print(f"{k:20s} " + " ".join(f"{x:7.2f}" for x in vals))
print("\nfraction with reversals>=2 / id_reuse>=2 / n_repeat>=5 / new_facts>=2 / n_err==0 per class:")
for c in CL:
    R = [r for r in rows if r["oracle"] == c]
    print(f"  {c:12s} rev>=2 {np.mean([r['reversals'] >= 2 for r in R]):.2f}  reuse>=2 {np.mean([r['id_reuse'] >= 2 for r in R]):.2f}  repeat>=5 {np.mean([r['n_repeat'] >= 5 for r in R]):.2f}"
          f"  newfacts>=2 {np.mean([r['new_facts'] >= 2 for r in R]):.2f}  noerr {np.mean([r['n_err'] == 0 for r in R]):.2f}  last_err {np.mean([r['last_err'] for r in R]):.2f}  revert>0 {np.mean([r['n_revert'] > 0 for r in R]):.2f}")
print("\ntool sequences (last 8 blocks; '!' = error/FAILED/pending) by oracle class and network class:")
for c in CL:
    print(f"== oracle {c}")
    for r in sorted([r for r in rows if r["oracle"] == c], key=lambda r: r["net"]):
        print(f"  {r['id']} net={r['net'][:4]} rev={r['reversals']} rep={r['n_repeat']} nf={r['new_facts']} err={r['n_err']} res={r['resolved_offset']:2d} | {r['tools']}")
