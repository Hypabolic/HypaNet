"""Apply output/wsp-b/build_edit.py to the kernel (the recommended pattern), print receipts and the edit hash."""
from collections.abc import Mapping

import torch
from safetensors.torch import load_file
from wsp import NAMES, edited, drop_edit, load_original, save_edit, save_label_rows, LABEL_IDS, edit_hash
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
new = build_edit.build(Orig(), consts)
for name in edited():
    if name not in new:
        print("dropping", name, drop_edit(name))
for name, t in new.items():
    if name == "model.language_model.embed_tokens.weight":
        r = save_label_rows(t[list(LABEL_IDS)])
    else:
        r = save_edit(name, t)
    print(f"{r.name.split('language_model.')[1]}: changed={r.changed_entries} rows={r.changed_rows} cols={r.changed_cols} relF={r.rel_frobenius:.2e} rank={r.delta_rank}")
print("edited:", [n.split("language_model.")[1] for n in edited()])
print("edit hash:", edit_hash())
