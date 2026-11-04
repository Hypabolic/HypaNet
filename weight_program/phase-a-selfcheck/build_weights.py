"""Direct weight program for the doom-loop MLP (spec v0.2, Phase A-strict).

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
U_INE = (0, 1, 2, 3, 4, 5, 6, 7)   # ine_s: stall-ineffective score of slot s
U_END2 = 8                         # clip helper for slot 7 (END = u7 - u8)
U_EE = (9, 10, 11, 12)             # ee_s, s = 4..7: episode-like error
U_RS = (13, 14, 15, 16)            # rs_s, s = 4..7: resolution-like retry
U_VM1 = 17                         # relu(V - 1)   (V = reversion count)
U_VM2 = 18                         # relu(V - 2)   (V2 = u17 - u18)
U_PROG_A = 19                      # relu(1e4 * S - 174)   (S = net validation delta)
U_PROG_B = 20                      # relu(1e4 * S - 175)   (PROG = u19 - u20)
U_OOP = 21                         # relu(RFsum - 2 * EXT_7 - 3): retries out of policy
U_POLL = 22                        # relu(RI_7 + NEU_7 + EXT_7 - 2): endpoint poll
# unit 23 unused

# ---- layer-2 unit indices --------------------------------------------------
H_OSC = 0        # relu(V2 - PROG)
H_STALL_A = 1    # relu(pre),     pre = C + 30*END - 34 - 30*PROG - 30*V2
H_STALL_B = 2    # relu(pre - 1)  (STALL' = h1 - h2)
H_ACT_A = 3      # relu(E - 10*OOP)
H_ACT_B = 4      # relu(E - 10*OOP - 1)   (ACT = h3 - h4)
H_RES_A = 5      # relu(8 * sum rs)
H_RES_B = 6      # relu(8 * sum rs - 1)   (RES = h5 - h6)
H_POLL = 7       # relu(u22)
# units 8..11 unused

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

        # ine_s = relu(RI + SR + RF - NEU), one unit per slot
        for s in range(SLOTS):
            u = U_INE[s]
            w1[u, idx(s, RI)] = 1.0
            w1[u, idx(s, SR)] = 1.0
            w1[u, idx(s, RF)] = 1.0
            w1[u, idx(s, NEU)] = -1.0
            b1[u] = 0.0

        # end2 = relu(RI_7 + SR_7 + RF_7 - NEU_7 - 1); END = ine_7 - end2
        w1[U_END2, idx(7, RI)] = 1.0
        w1[U_END2, idx(7, SR)] = 1.0
        w1[U_END2, idx(7, RF)] = 1.0
        w1[U_END2, idx(7, NEU)] = -1.0
        b1[U_END2] = -1.0

        # ee_s = relu(ERR - VC) for s in 4..7
        for k, s in enumerate(RECENT):
            u = U_EE[k]
            w1[u, idx(s, ERR)] = 1.0
            w1[u, idx(s, VC)] = -1.0
            b1[u] = 0.0

        # rs_s = relu(RI + SUC + MUT + VAL + EXT - VC - 3
        #             + (1/8) * sum_{j<s} (ERR_j - VC_j)) for s in 4..7
        for k, s in enumerate(RECENT):
            u = U_RS[k]
            w1[u, idx(s, RI)] = 1.0
            w1[u, idx(s, SUC)] = 1.0
            w1[u, idx(s, MUT)] = 1.0
            w1[u, idx(s, VAL)] = 1.0
            w1[u, idx(s, EXT)] = 1.0
            w1[u, idx(s, VC)] = -1.0
            for j in range(s):
                w1[u, idx(j, ERR)] = 0.125
                w1[u, idx(j, VC)] = -0.125
            b1[u] = -3.0

        # Vm1 = relu(sum SR - 1), Vm2 = relu(sum SR - 2)
        for s in range(SLOTS):
            w1[U_VM1, idx(s, SR)] = 1.0
            w1[U_VM2, idx(s, SR)] = 1.0
        b1[U_VM1] = -1.0
        b1[U_VM2] = -2.0

        # PROG_a = relu(1e4 * sum VD - 174), PROG_b = relu(1e4 * sum VD - 175)
        for s in range(SLOTS):
            w1[U_PROG_A, idx(s, VD)] = 10000.0
            w1[U_PROG_B, idx(s, VD)] = 10000.0
        b1[U_PROG_A] = -174.0
        b1[U_PROG_B] = -175.0

        # OOP = relu(sum RF - 2 * EXT_7 - 3)
        for s in range(SLOTS):
            w1[U_OOP, idx(s, RF)] = 1.0
        w1[U_OOP, idx(7, EXT)] = -2.0
        b1[U_OOP] = -3.0

        # POLL = relu(RI_7 + NEU_7 + EXT_7 - 2)
        w1[U_POLL, idx(7, RI)] = 1.0
        w1[U_POLL, idx(7, NEU)] = 1.0
        w1[U_POLL, idx(7, EXT)] = 1.0
        b1[U_POLL] = -2.0

        # ================= layer 2: class conditions ==========================

        # OSC = relu(V2 - PROG) = relu((u17 - u18) - (u19 - u20))
        w2[H_OSC, U_VM1] = 1.0
        w2[H_OSC, U_VM2] = -1.0
        w2[H_OSC, U_PROG_A] = -1.0
        w2[H_OSC, U_PROG_B] = 1.0
        b2[H_OSC] = 0.0

        # STALL': pre = C + 30 * END - 34 - 30 * PROG - 30 * V2,
        #         C = sum ine_s, END = ine_7 - end2
        for h, bias in ((H_STALL_A, -34.0), (H_STALL_B, -35.0)):
            for s in range(SLOTS):
                w2[h, U_INE[s]] = 1.0
            w2[h, U_INE[7]] = 31.0      # 1 (part of C) + 30 (END)
            w2[h, U_END2] = -30.0
            w2[h, U_PROG_A] = -30.0
            w2[h, U_PROG_B] = 30.0
            w2[h, U_VM1] = -30.0
            w2[h, U_VM2] = 30.0
            b2[h] = bias

        # ACT = clip01(E - 10 * OOP), E = sum ee_s
        for h, bias in ((H_ACT_A, 0.0), (H_ACT_B, -1.0)):
            for u in U_EE:
                w2[h, u] = 1.0
            w2[h, U_OOP] = -10.0
            b2[h] = bias

        # RES = clip01(8 * sum rs_s)
        for h, bias in ((H_RES_A, 0.0), (H_RES_B, -1.0)):
            for u in U_RS:
                w2[h, u] = 8.0
            b2[h] = bias

        # POLL pass-through
        w2[H_POLL, U_POLL] = 1.0
        b2[H_POLL] = 0.0

        # ================= layer 3: logits with precedence ====================
        # PRODUCTIVE: all zero (default by the first-index tie rule).

        # RECOVERY = 20*RES + 20*POLL + 200*ACT - 100*STALL' - 1000*OSC
        w3[RECOVERY, H_RES_A] = 20.0
        w3[RECOVERY, H_RES_B] = -20.0
        w3[RECOVERY, H_POLL] = 20.0
        w3[RECOVERY, H_ACT_A] = 200.0
        w3[RECOVERY, H_ACT_B] = -200.0
        w3[RECOVERY, H_STALL_A] = -100.0
        w3[RECOVERY, H_STALL_B] = 100.0
        w3[RECOVERY, H_OSC] = -1000.0

        # STALLED = 20*STALL' - 60*ACT
        w3[STALLED, H_STALL_A] = 20.0
        w3[STALLED, H_STALL_B] = -20.0
        w3[STALLED, H_ACT_A] = -60.0
        w3[STALLED, H_ACT_B] = 60.0

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
