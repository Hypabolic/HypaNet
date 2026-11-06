"""Trace the layer-3 head-2 (resw) attention and the RES dim on a synthetic poll session."""
import sys

import torch

import renderer
import v05_load

TS = "2026-03-01T09:{:02d}:{:02d}.000+00:00"


def op(family, target=None, status="SUCCESS", args=None, i=0):
    return {"family": family, "target": target, "args": args or {}, "outcome": status, "validation": None,
            "changed": [], "input_tokens": 1000 + 7 * i, "output_tokens": 50 + i, "timestamp": TS.format(0, i),
            "dropped": [], "extra": None, "error_class": None, "error_location": None}


def main():
    edits = sys.argv[1] if len(sys.argv) > 1 else None
    m = v05_load.apply_edits_file(edits)[0]
    ops = [op("read_file", "a.md", i=0)] + [op("poll_status", "ci/run/9", status="NEUTRAL", args={"backoff": 30}, i=i)
                                             for i in range(1, 7)]
    blocks = renderer.render_session(ops, "s5")
    enc = v05_load.encode(blocks)
    qpos = enc.eoe_index[5]
    act = v05_load.activations(m, blocks, query_positions=[qpos])
    a = act["attn"][3][2, 0]
    r2 = act["resid_mlp"][2]
    print("top attention of layer-3 head 2 at EOE of event 5:")
    for j in torch.argsort(a, descending=True)[:10]:
        j = int(j)
        print(f"  w={float(a[j]):.4f} pos {j} ev {enc.event_ord[j] - 1} off {enc.offset[j]} '{enc.surface[j]}' d76={float(r2[j, 76]):.3f} d5={float(act['resid_mlp'][2][j, 5]):.2f}")
    print("d76 at EOEs:", [round(float(r2[p, 76]), 3) for p in enc.eoe_index])
    print("max |d76| over all tokens:", float(r2[:, 76].abs().max()))
    # which tokens have d76 != 0
    nz = [(j, enc.surface[j], round(float(r2[j, 76]), 3)) for j in range(len(enc.surface)) if abs(float(r2[j, 76])) > 1e-3]
    print("nonzero d76 tokens:", nz[:20])


if __name__ == "__main__":
    main()
