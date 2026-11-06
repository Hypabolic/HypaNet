"""Why does v0.5 B fail off distribution?  Post-hoc diagnostic, no sealed data.

Generates a FRESH OOD set from the same independent generator (``synth_ood``) with a new master seed
(never the sealed one), renders it with the v0.3 renderer, and runs v0.5 B and v0.3 B. For v0.5 B it reads
the residual stream at every EOE (dims documented in weight_program/v0_5/phase-b/CIRCUITS.md) and compares
the network's per-event INEFF flag, and its parts, with the oracle's latent per-event flags.

    uv run python experiments/v0_5/ood_analysis/analyse.py [--per-family 60] [--seed 20261107]

Writes experiments/v0_5/ood_analysis/rows.jsonl.gz (one row per window) and prints summaries.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from agent_loop_lab.datasets.ood import ood_seed, ood_session_id  # noqa: E402
from agent_loop_lab.oracle_config import load_oracle_config  # noqa: E402
from agent_loop_lab.schema import LatentEffect, OracleLabel  # noqa: E402
from agent_loop_lab.synth.labeller import window_stats  # noqa: E402
from agent_loop_lab.synth.latent import steps_from_record  # noqa: E402
from agent_loop_lab.synth_ood import FAMILIES  # noqa: E402
from agent_loop_lab.v03 import renderer, tokenizer  # noqa: E402
from agent_loop_lab.v03.artifact import load_artifact  # noqa: E402
from agent_loop_lab.v03.rawop import capture_raw  # noqa: E402
from agent_loop_lab.windows import window_endpoints  # noqa: E402

SEALED_MASTER_SEED = 20261104
CN = ["PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING"]
# v0.5 B residual dims at the EOE after layer 2 (CIRCUITS.md §1, §5-7)
L2 = dict(NOOP=49, muthunk=58, g_SAME=59, g_CNTOK=63, g_REV=66, g_REP=67, r_SAMEF=72, r_CNTOK=73, kpen_ev=74,
          INEFF=75, RES=76, INPOL=77)
L3 = dict(A2=80, act=81, resw=82, osc=83, sc=84, act_b=85, resw_b=86)
FAM_TOKEN = {"read_file", "list_dir", "search", "edit_file", "write_file", "revert", "build", "lint", "run_tests",
             "poll_status", "http_call", "plan", "ask", "yield"}


def ineff_parts(f: dict) -> list[str]:
    parts = []
    for k in ("g_REP", "g_CNTOK", "g_SAME", "NOOP"):
        if f[k] > 0.5:
            parts.append(k)
    if f["muthunk"] > 0.5 and (f["r_SAMEF"] > 0.5 or f["r_CNTOK"] > 0.5):
        parts.append("EDR")
    if f["g_REV"] > 0.5:
        parts.append("REV")
    if f["kpen_ev"] > 0.5 and not parts:
        parts.append("POLLOUT")
    return parts


def event_head(block: str) -> str:
    h = block.split("\n")[0]
    toks = h.split(" ")
    fam = next((t for t in toks if t in FAM_TOKEN), toks[1] if len(toks) > 1 else "?")
    status = h.split(" -> ")[1].split(" ")[0] if " -> " in h else "?"
    return f"{fam}:{status}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--out", default=None)
    ap.add_argument("--indist", action="store_true", help="the standard (in-distribution) generator instead of synth_ood")
    a = ap.parse_args()
    if a.seed == SEALED_MASTER_SEED:
        raise SystemExit("refusing the sealed master seed")
    torch.set_num_threads(4)
    cfg = load_oracle_config()
    v05b, _ = load_artifact(REPO / "weight_program/v0_5/phase-b/artifact.dlw")
    v03b, _ = load_artifact(REPO / "weight_program/v0_3/phase-b/artifact.dlw")
    out = a.out or str(Path(__file__).parent / ("rows_indist.jsonl.gz" if a.indist else "rows.jsonl.gz"))
    if a.indist:
        from agent_loop_lab.synth.families import FAMILY_NAMES, generate_session

        fams = {f: (lambda f: lambda seed, sid: generate_session(f, seed, sid, cfg))(f) for f in FAMILY_NAMES}
    else:
        fams = {f: (lambda g: lambda seed, sid: g(random.Random(seed), sid, cfg))(g) for f, g in FAMILIES.items()}
    rows = []
    for family, gen in fams.items():
        for k in range(a.per_family):
            seed = ood_seed(a.seed + (1 if a.indist else 0), family, k)
            sid = ood_session_id(seed)
            with capture_raw() as cap:
                events, record = gen(seed, sid)
                raw = cap.pop(sid, events)
            blocks = renderer.render_session([r.to_dict() for r in raw], sid)
            enc = tokenizer.encode_session(blocks)
            tok = torch.tensor(enc.tokens)
            eo = torch.tensor(enc.event_ord)
            off = torch.tensor(enc.offset)
            eoe = list(enc.eoe_index)
            with torch.no_grad():
                act = v05b.activations(tok, eo, off, query_positions=eoe[:1])
                lg5 = act["logits"][eoe]
                lg3 = v03b(tok, eo, off)[eoe]
            r2 = act["resid_mlp"][2][0][eoe] if act["resid_mlp"][2].dim() == 3 else act["resid_mlp"][2][eoe]
            r3 = act["resid_mlp"][3][0][eoe] if act["resid_mlp"][3].dim() == 3 else act["resid_mlp"][3][eoe]
            feats = []
            for i in range(len(eoe)):
                f = {k2: round(float(r2[i, d]), 3) for k2, d in L2.items()}
                f.update({k2: round(float(r3[i, d]), 3) for k2, d in L3.items()})
                f["parts"] = ineff_parts(f)
                feats.append(f)
            steps, p0 = steps_from_record(record)
            heads = [event_head(b) for b in blocks]
            for pos in window_endpoints(events):
                seq = events[pos].sequence_index
                lab = record.window_labels[seq]
                if lab.value not in CN:
                    continue
                st = window_stats(steps, p0, pos, cfg)
                lo = st.start
                win_ev = []
                for j in range(lo, pos + 1):
                    s = steps[j]
                    win_ev.append(dict(
                        head=heads[j], effect=s.effect.value, dp=round(s.progress_after - (steps[j - 1].progress_after if j else p0), 4),
                        facts=s.new_facts, net_ineff=feats[j]["INEFF"] > 0.5, parts=feats[j]["parts"],
                        oracle_red=s.effect is LatentEffect.REDUNDANT))
                rows.append(dict(
                    sid=sid, family=family, pos=pos, label=lab.value,
                    v05b=CN[int(torch.argmax(lg5[pos]))], v03b=CN[int(torch.argmax(lg3[pos]))],
                    logits=[round(float(x), 2) for x in lg5[pos]],
                    oracle=dict(dp=round(st.delta_p, 4), facts=st.facts, I=st.stall_ineffective, rev=st.reversions,
                                final_I=st.final_stall_ineffective, in_policy=st.episode_in_policy_at_end,
                                resolved=st.resolved_recently),
                    net={k2: feats[pos][k2] for k2 in ("INEFF", "A2", "sc", "act", "act_b", "resw", "osc")},
                    net_I=sum(e["net_ineff"] for e in win_ev), events=win_ev))
        print(family, len(rows), flush=True)
    with gzip.open(out, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    conf5, conf3 = Counter(), Counter()
    for r in rows:
        conf5[(r["label"], r["v05b"])] += 1
        conf3[(r["label"], r["v03b"])] += 1
    for name, conf in (("v0.5 B", conf5), ("v0.3 B", conf3)):
        print(name, [[conf[(o, p)] for p in CN] for o in CN])
    return 0


if __name__ == "__main__":
    sys.exit(main())
