"""Reference policy for the agent doom-loop detector (spec v0.2, Phase B revision).

The policy reads the 8 x 15 window (oldest event first, slot 7 = endpoint)
and returns the canonical class index:

    0 PRODUCTIVE, 1 RECOVERY, 2 STALLED, 3 OSCILLATING

It is written as the exact arithmetic of the compiled network (per-slot
clipped scores, window sums, clipped indicators, four logits, argmax with
ties to the first index) so that the network and this function agree on
every input. For binary features the arithmetic reduces to the rules below.

Signals (all window-level unless stated):

  redundancy R      sum over slots 1..7 of relu(repeated_intent +
                    repeated_failure + state_revisited - out_NEUTRAL
                    - cat_EXTERNAL): repeated work that is not an external
                    retry/poll and not a neutral-outcome repeat.
  reversions V      number of state_revisited events.
  net delta S       sum of validation_delta over the window.
  retry RT          at least one repeated_failure in the last 4 events.
  fresh failure NF  (non-comparable ERROR count - comparable SUCCESS count)
                    - 10 * (validation_delta sum over the last 4 events):
                    positive when the failure chain is new (a first result
                    of a suite, or a regression), zero or negative when the
                    chain merely repeats a pre-existing failure or when the
                    recent runs improved.
  out of policy OOP 4 or more repeated failures in the window (R_max = 3).
  resolution RES    in the last 4 events, a successful repeat of an EXTERNAL
                    call or a non-comparable VALIDATE op after an earlier
                    ERROR or NEUTRAL event, or a comparable passing run after
                    an earlier repeated failure.
  polling           the endpoint is a repeated NEUTRAL EXTERNAL op; the poll
                    run is "over" when the window holds 7+ neutral events or
                    the run already filled slot 0 (count + 3 * NEU_0 >= 7).

Classes in precedence order:

  OSCILLATING  V >= 2.
  RECOVERY     an in-policy retry episode is active (RT and NF > 0 and not
               OOP), or RES, or in-flight polling that is not over.
  STALLED      (R >= 2 or OOP, or over-polling) with no net validation change
               (S < 0.02 and S > -0.1) and V < 2, unless RECOVERY holds.
  PRODUCTIVE   otherwise.
"""

# Per-event feature offsets (feature_index.json): flat = 15 * slot + offset.
_INS, _MUT, _VAL, _EXT, _CTL = 0, 1, 2, 3, 4
_SUC, _ERR, _NEU = 5, 6, 7
_RI, _RF, _WC, _SR, _VC, _VD, _TC = 8, 9, 10, 11, 12, 13, 14

_SLOTS = 8
_RECENT = (4, 5, 6, 7)  # the last k_rec = 4 events, endpoint included


def _relu(v: float) -> float:
    return v if v > 0.0 else 0.0


def _clip01(v: float) -> float:
    # min(1, relu(v)) = relu(v) - relu(v - 1)
    return _relu(v) - _relu(v - 1.0)


def reference_policy(x: tuple[float, ...]) -> int:
    if len(x) != 15 * _SLOTS:
        raise ValueError("expected 120 features")

    def f(slot: int, off: int) -> float:
        return float(x[15 * slot + off])

    # ---- layer 1: per-slot scores and window sums --------------------------

    # redundancy per slot (slot 0 never repeats): r_s
    redundancy = sum(
        _relu(f(s, _RI) + f(s, _RF) + f(s, _SR) - f(s, _NEU) - 3.0 * f(s, _EXT))
        for s in range(1, _SLOTS)
    )

    # resolution A: successful repeat of EXTERNAL / non-comparable VALIDATE
    # after an earlier ERROR or NEUTRAL event (last 4 events)
    res_a = 0.0
    for s in _RECENT:
        prior = sum(f(j, _ERR) + f(j, _NEU) for j in range(s))
        res_a += _relu(
            f(s, _RI) + f(s, _SUC) + f(s, _EXT) + f(s, _VAL) - f(s, _VC) - 3.0
            + prior / 8.0
        )

    # resolution B: comparable passing run after an earlier repeated failure
    res_b = 0.0
    for s in _RECENT:
        prior = sum(f(j, _RF) for j in range(s))
        res_b += _relu(f(s, _RI) + f(s, _SUC) + f(s, _VC) - 3.0 + prior / 8.0)

    # recent retry: any repeated failure in the last 4 events (clipped to 1)
    recent_rf = sum(f(s, _RF) for s in _RECENT)
    retry = _clip01(recent_rf)                                            # RT

    # fresh failure evidence
    fresh = _relu(
        sum(f(s, _ERR) - f(s, _VC) for s in range(_SLOTS))
        - 10.0 * sum(f(s, _VD) for s in _RECENT)
    )                                                                     # NF

    reversions = sum(f(s, _SR) for s in range(_SLOTS))
    v_minus_1 = _relu(reversions - 1.0)                                   # >= 1 iff V >= 2

    net_delta = sum(f(s, _VD) for s in range(_SLOTS))                     # S
    progress = _relu(100.0 * net_delta - 1.0)                             # >= 1 iff S >= 0.02
    regression = _relu(-20.0 * net_delta - 1.0)                           # >= 1 iff S <= -0.1

    repeated_failures = sum(f(s, _RF) for s in range(_SLOTS))
    out_of_policy = _relu(repeated_failures - 3.0)                        # OOP

    poll_end = _relu(f(7, _RI) + f(7, _NEU) + f(7, _EXT) - 2.0)           # endpoint poll
    neutral_count = sum(f(s, _NEU) for s in range(_SLOTS))
    poll_over = _relu(neutral_count + 3.0 * f(0, _NEU) - 6.0)             # OVER

    # ---- layer 2: class conditions -----------------------------------------
    veto = 40.0 * (progress + regression + v_minus_1)

    osc = _relu(v_minus_1)                                                # OSC
    stall_a = _clip01(redundancy + 2.0 * out_of_policy - 1.0 - veto)      # R >= 2 or OOP
    stall_b = _clip01(2.0 * poll_end + poll_over - 2.0 - veto)            # over-polling
    active = _relu(2.0 * retry + fresh / 32.0 - 2.0 - 10.0 * out_of_policy)  # ACT
    resolved = _clip01(8.0 * (res_a + res_b))                             # RES
    polling = _relu(poll_end - poll_over)                                 # POLL

    # ---- logits with precedence OSCILLATING > RECOVERY > STALLED > PRODUCTIVE
    logit_productive = 0.0
    logit_recovery = (
        2000.0 * active + 20.0 * resolved + 20.0 * polling - 10000.0 * osc
    )
    logit_stalled = (
        20.0 * (stall_a + stall_b)
        - 2000.0 * active - 100.0 * resolved - 100.0 * polling - 1000.0 * osc
    )
    logit_oscillating = 20.0 * osc

    logits = (logit_productive, logit_recovery, logit_stalled, logit_oscillating)
    best = 0
    for i in range(1, 4):
        if logits[i] > logits[best]:
            best = i
    return int(best)
