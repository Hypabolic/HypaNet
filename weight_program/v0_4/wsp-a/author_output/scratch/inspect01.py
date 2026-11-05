"""Inspect metadata, label rows, final norm, embedding scale; count Qwen tokens of sample blocks."""
import json
import torch
from wsp import (ARCH, NAMES, SHAPES, LABEL_IDS, CLASS_NAMES, tok, label_rows, load, load_original,
                 budget, load_quota, edited, edit_hash)
import renderer

print("edited:", edited(), "hash:", edit_hash()[:12])
print("budget:", budget())
print("quota:", load_quota())
print("label ids:", LABEL_IDS, CLASS_NAMES)
arch = {k: v for k, v in ARCH.items() if k != "layer_types"}
print("arch:", json.dumps(arch)[:1500])
print("n names:", len(NAMES))
shapes_by_layer0 = {n: SHAPES[n] for n in NAMES if ".layers.0." in n or ".layers.3." in n or "embed" in n or n.endswith("norm.weight") and "layers" not in n}
for n, s in shapes_by_layer0.items():
    print(n, s)

rows = label_rows(original=True)
print("label row norms:", torch.linalg.vector_norm(rows, dim=1))
G = rows @ rows.T
print("label row gram:\n", G)
fn = load("model.language_model.norm.weight")
print("final norm w: mean", fn.mean().item(), "std", fn.std().item(), "min", fn.min().item(), "max", fn.max().item())
print("(1+w) rms:", torch.sqrt(((1 + fn) ** 2).mean()).item())

# token counts of grammar-sheet style blocks
samples = [
    "09:14:03.250 read_file path=zork/zeta/widget.qx lines=12-40 -> ok in=1840 out=96",
    "09:14:09.880 edit_file path=zork/zeta/widget.qx -> ok in=2214 out=310\n  @ zork/zeta/widget.qx h3f9a0c1d2e > h7b21e9c04a",
    "09:14:20.415 run_tests suite=alpha tests=alpha/spin_wobble -> FAILED 7/9 passed in=2611 out=188\n  ! QuuxError at zork/zeta/widget.qx:417 (pid 4821)",
    "09:14:40.118 http_call POST https://quuxhost.zeta/v9/gadgets/{gid} body={\"flavor\":\"quux\"} req=5d03ac91 -> error in=2810 out=33\n  ! QuuxError at zeta_portal:212 (pid 30177)",
    "09:14:52.733 poll_status resource=zeta_hopper backoff=4 -> pending in=2851 out=18",
]
for s in samples:
    ids = tok.encode(s + "\n", add_special_tokens=False).ids
    print(len(ids), "tokens:", s.split("\n")[0][:60])
# how are the label words tokenised in context
for w in [" productive", " recovering", " stalled", " looping", "State:", " error", " FAILED", " pending", " ok", " revert", "revert", " h0"]:
    print(repr(w), tok.encode(w, add_special_tokens=False).ids)
