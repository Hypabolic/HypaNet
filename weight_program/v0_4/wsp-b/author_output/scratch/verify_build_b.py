"""Check that output/wsp-b/build_edit.py reproduces the kernel's current edit exactly, and that two builds agree."""
from collections.abc import Mapping
import hashlib

import torch
from safetensors.torch import load_file
from wsp import NAMES, edited, load, load_original, edit_hash
import build_edit

torch.set_num_threads(1)


class Orig(Mapping):
    def __getitem__(self, name):
        if name not in NAMES:
            raise KeyError(name)
        return load_original(name)

    def __iter__(self):
        return iter(sorted(NAMES))

    def __len__(self):
        return len(NAMES)


consts = load_file("output/wsp-b/edit_constants.safetensors")
new1 = build_edit.build(Orig(), consts); new2 = build_edit.build(Orig(), consts)
ok = True
for name in sorted(set(new1) | set(edited())):
    if name not in new1:
        print("MISSING in build:", name); ok = False; continue
    if name not in edited():
        print("NOT EDITED in kernel:", name); ok = False; continue
    t1, t2 = new1[name], new2[name]
    cur = load(name)
    same = torch.equal(t1, cur) and t1.dtype == cur.dtype and t1.shape == cur.shape
    same12 = torch.equal(t1, t2)
    nchanged = int((t1 != load_original(name)).sum())
    print(f"{name.split('language_model.')[1]:36s} build==kernel={same} build1==build2={same12} changed_entries={nchanged} sha={hashlib.sha256(t1.contiguous().numpy().tobytes()).hexdigest()[:12]}")
    ok = ok and same and same12
print("ALL MATCH:", ok, " edit hash:", edit_hash())
print("constants:", {k: tuple(v.shape) for k, v in consts.items()})
