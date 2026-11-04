"""Direct weight program for the doom-loop MLP (spec v0.2, Phase B revision).

Compiles the policy in reference_policy.py into
Linear(120, 24) -> ReLU -> Linear(24, 12) -> ReLU -> Linear(12, 4)
by literal index assignment only. See CIRCUITS.md for every unit.

Input layout: flat = 15 * slot + feature (slot 0 oldest, slot 7 endpoint).
"""

import torch
from model import zero_model, DoomLoopMLP

# Per-event feature offsets (feature_index.json).
INS, MUT, VAL, EXT, CTL = 0, 1, 2, 3, 4
SUC, ERR, NEU = 5, 6, 7
RI, RF, WC, SR, VC, VD, TC = 8, 9, 10, 11, 12, 13, 14

SLOTS = 8
RECENT = (4, 5, 6, 7)  # last k_rec = 4 events, endpoint included

# ---- layer-1 unit indices --------------------------------------------------
U_RED = (0, 1, 2, 3, 4, 5, 6)      # r_s for slots 1..7 (unit s-1): redundancy score
U_RESA = (7, 8, 9, 10)             # resA_s, s = 4..7: EXT / non-comparable VAL success repeat
U_RESB = (11, 12, 13, 14)          # resB_s, s = 4..7: comparable passing run after a repeated failure
U_RT_A = 15                        # relu(sum RF over slots 4..7)
U_RT_B = 16                        # relu(sum RF over slots 4..7 - 1)   (RT = u15 - u16)
U_NF = 17                          # relu(sum(ERR - VC) - 10 * sum_{4..7} VD): fresh-failure evidence
U_VM1 = 18                         # relu(V - 1), V = reversion count
U_PROG = 19                        # relu(100 * S - 1), S = net validation delta
U_NZ = 20                          # relu(-20 * S - 1)
U_OOP = 21                         # relu(RFsum - 3): retries out of policy
U_POLL = 22                        # relu(RI_7 + NEU_7 + EXT_7 - 2): endpoint poll
U_OVER = 23                        # relu(sum NEU + 3 * NEU_0 - 6): poll run over policy

# ---- layer-2 unit indices --------------------------------------------------
H_OSC = 0        # relu(u18)
H_SA_A = 1       # relu(pre_a),     pre_a = R + 2*OOP - 1 - 40*(PROG + NZ + VM1)
H_SA_B = 2       # relu(pre_a - 1)  (STALL_A = h1 - h2)
H_SB_A = 3       # relu(pre_b),     pre_b = 2*POLL + OVER - 2 - 40*(PROG + NZ + VM1)
H_SB_B = 4       # relu(pre_b - 1)  (STALL_B = h3 - h4)
H_ACT = 5        # relu(2*RT + NF/32 - 2 - 10*OOP)
H_RES_A = 6      # relu(8 * sum res)
H_RES_B = 7      # relu(8 * sum res - 1)   (RES = h6 - h7)
H_POLL = 8       # relu(POLL - OVER)
# units 9..11 unused

# ---- output rows -------------------------------------------------------------
PRODUCTIVE, RECOVERY, STALLED, OSCILLATING = 0, 1, 2, 3


def idx(slot: int, feature: int) -> int:
    return 15 * slot + feature


def build() -> DoomLoopMLP:
    m = zero_model()
    with torch.no_grad():
        w1 = m.fc1.weight
        b1 = m.fc1.bias
        w2 = m.fc2.weight
        b2 = m.fc2.bias
        w3 = m.fc3.weight
        b3 = m.fc3.bias

        # ================= layer 1: per-slot scores and window sums ==========

        # r_s = relu(RI + RF + SR - NEU - 3*EXT) for slots 1..7
        for s in range(1, SLOTS):
            u = U_RED[s - 1]
            w1[u, idx(s, RI)] = 1.0
            w1[u, idx(s, RF)] = 1.0
            w1[u, idx(s, SR)] = 1.0
            w1[u, idx(s, NEU)] = -1.0
            w1[u, idx(s, EXT)] = -3.0
            b1[u] = 0.0

        # resA_s = relu(RI + SUC + EXT + VAL - VC - 3 + (1/8) sum_{j<s} (ERR_j + NEU_j))
        for k, s in enumerate(RECENT):
            u = U_RESA[k]
            w1[u, idx(s, RI)] = 1.0
            w1[u, idx(s, SUC)] = 1.0
            w1[u, idx(s, EXT)] = 1.0
            w1[u, idx(s, VAL)] = 1.0
            w1[u, idx(s, VC)] = -1.0
            for j in range(s):
                w1[u, idx(j, ERR)] = 0.125
                w1[u, idx(j, NEU)] = 0.125
            b1[u] = -3.0

        # resB_s = relu(RI + SUC + VC - 3 + (1/8) sum_{j<s} RF_j)
        for k, s in enumerate(RECENT):
            u = U_RESB[k]
            w1[u, idx(s, RI)] = 1.0
            w1[u, idx(s, SUC)] = 1.0
            w1[u, idx(s, VC)] = 1.0
            for j in range(s):
                w1[u, idx(j, RF)] = 0.125
            b1[u] = -3.0

        # RT_a = relu(sum_{4..7} RF), RT_b = relu(sum_{4..7} RF - 1)
        for s in RECENT:
            w1[U_RT_A, idx(s, RF)] = 1.0
            w1[U_RT_B, idx(s, RF)] = 1.0
        b1[U_RT_A] = 0.0
        b1[U_RT_B] = -1.0

        # NF = relu(sum_s (ERR_s - VC_s) - 10 * sum_{4..7} VD_s)
        for s in range(SLOTS):
            w1[U_NF, idx(s, ERR)] = 1.0
            w1[U_NF, idx(s, VC)] = -1.0
        for s in RECENT:
            w1[U_NF, idx(s, VD)] = -10.0
        b1[U_NF] = 0.0

        # VM1 = relu(sum SR - 1)
        for s in range(SLOTS):
            w1[U_VM1, idx(s, SR)] = 1.0
        b1[U_VM1] = -1.0

        # PROG = relu(100 * sum VD - 1); NZ = relu(-20 * sum VD - 1)
        for s in range(SLOTS):
            w1[U_PROG, idx(s, VD)] = 100.0
            w1[U_NZ, idx(s, VD)] = -20.0
        b1[U_PROG] = -1.0
        b1[U_NZ] = -1.0

        # OOP = relu(sum RF - 3)
        for s in range(SLOTS):
            w1[U_OOP, idx(s, RF)] = 1.0
        b1[U_OOP] = -3.0

        # POLL = relu(RI_7 + NEU_7 + EXT_7 - 2)
        w1[U_POLL, idx(7, RI)] = 1.0
        w1[U_POLL, idx(7, NEU)] = 1.0
        w1[U_POLL, idx(7, EXT)] = 1.0
        b1[U_POLL] = -2.0

        # OVER = relu(sum NEU + 3 * NEU_0 - 6)
        for s in range(SLOTS):
            w1[U_OVER, idx(s, NEU)] = 1.0
        w1[U_OVER, idx(0, NEU)] = 4.0   # 1 (part of the sum) + 3 (slot-0 weight)
        b1[U_OVER] = -6.0

        # ================= layer 2: class conditions ==========================

        # OSC = relu(VM1)
        w2[H_OSC, U_VM1] = 1.0
        b2[H_OSC] = 0.0

        # STALL_A: pre_a = R + 2*OOP - 1 - 40*(PROG + NZ + VM1)
        for h, bias in ((H_SA_A, -1.0), (H_SA_B, -2.0)):
            for u in U_RED:
                w2[h, u] = 1.0
            w2[h, U_OOP] = 2.0
            w2[h, U_PROG] = -40.0
            w2[h, U_NZ] = -40.0
            w2[h, U_VM1] = -40.0
            b2[h] = bias

        # STALL_B: pre_b = 2*POLL + OVER - 2 - 40*(PROG + NZ + VM1)
        for h, bias in ((H_SB_A, -2.0), (H_SB_B, -3.0)):
            w2[h, U_POLL] = 2.0
            w2[h, U_OVER] = 1.0
            w2[h, U_PROG] = -40.0
            w2[h, U_NZ] = -40.0
            w2[h, U_VM1] = -40.0
            b2[h] = bias

        # ACT = relu(2*(RT_a - RT_b) + NF/32 - 2 - 10*OOP)
        w2[H_ACT, U_RT_A] = 2.0
        w2[H_ACT, U_RT_B] = -2.0
        w2[H_ACT, U_NF] = 0.03125
        w2[H_ACT, U_OOP] = -10.0
        b2[H_ACT] = -2.0

        # RES = clip01(8 * sum res)
        for h, bias in ((H_RES_A, 0.0), (H_RES_B, -1.0)):
            for u in U_RESA + U_RESB:
                w2[h, u] = 8.0
            b2[h] = bias

        # POLL_final = relu(POLL - OVER)
        w2[H_POLL, U_POLL] = 1.0
        w2[H_POLL, U_OVER] = -1.0
        b2[H_POLL] = 0.0

        # ================= layer 3: logits with precedence ====================
        # PRODUCTIVE: all zero (default by the first-index tie rule).

        # RECOVERY = 2000*ACT + 20*RES + 20*POLL_final - 10000*OSC
        w3[RECOVERY, H_ACT] = 2000.0
        w3[RECOVERY, H_RES_A] = 20.0
        w3[RECOVERY, H_RES_B] = -20.0
        w3[RECOVERY, H_POLL] = 20.0
        w3[RECOVERY, H_OSC] = -10000.0

        # STALLED = 20*(STALL_A + STALL_B) - 2000*ACT - 100*RES - 100*POLL_final - 1000*OSC
        w3[STALLED, H_SA_A] = 20.0
        w3[STALLED, H_SA_B] = -20.0
        w3[STALLED, H_SB_A] = 20.0
        w3[STALLED, H_SB_B] = -20.0
        w3[STALLED, H_ACT] = -2000.0
        w3[STALLED, H_RES_A] = -100.0
        w3[STALLED, H_RES_B] = 100.0
        w3[STALLED, H_POLL] = -100.0
        w3[STALLED, H_OSC] = -1000.0

        # OSCILLATING = 20*OSC
        w3[OSCILLATING, H_OSC] = 20.0

        b3[PRODUCTIVE] = 0.0
        b3[RECOVERY] = 0.0
        b3[STALLED] = 0.0
        b3[OSCILLATING] = 0.0
    return m


if __name__ == "__main__":
    from model import sparsity
    print(sparsity(build()))
