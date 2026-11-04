"""Reference policy for the agent doom-loop detector (spec v0.2, Phase A-strict).

The policy reads the 8 x 15 window (oldest event first, slot 7 = endpoint)
and returns the canonical class index:

    0 PRODUCTIVE, 1 RECOVERY, 2 STALLED, 3 OSCILLATING

It is written as the exact arithmetic of the compiled network (per-slot
clipped scores, window sums, clipped indicators, four logits, argmax with
ties to the first index) so that the network and this function agree on
every input, including non-binary ones. For the binary features the
arithmetic reduces to the boolean rules stated next to each step.

Policy, in oracle precedence order (spec 6.3):

  OSCILLATING  V >= 2 reversions (state_revisited events) and no net
               validation progress in the window.
  STALLED      at least 5 units of stall-ineffective evidence (repeated
               intents, state reversions, repeated identical failures;
               neutral-outcome repeats such as polls do not count), the
               endpoint itself is stall-ineffective, no net validation
               progress, fewer than 2 reversions, and no in-policy episode
               is active (see RECOVERY).
  RECOVERY     an in-policy episode is active (an ERROR that is not a
               comparable validation result occurred in the last 4 events
               and the repeated-failure count is within the retry bound:
               <= 3, or <= 5 when the endpoint is EXTERNAL), or a
               successful retry of a MUTATE / VALIDATE / EXTERNAL operation
               happened in the last 4 events after an earlier episode-like
               error (episode resolved recently), or the endpoint is a
               repeated NEUTRAL EXTERNAL operation (in-flight polling).
  PRODUCTIVE   everything else (the default; novel activity is never a
               loop).
"""

# Per-event feature offsets (feature_index.json): flat = 15 * slot + offset.
_INS, _MUT, _VAL, _EXT, _CTL = 0, 1, 2, 3, 4
_SUC, _ERR, _NEU = 5, 6, 7
_RI, _RF, _WC, _SR, _VC, _VD, _TC = 8, 9, 10, 11, 12, 13, 14

_SLOTS = 8
_RECENT = (4, 5, 6, 7)  # the last k_rec = 4 events, endpoint included

# Progress gate: net validation delta >= 0.0175 counts as observed progress.
# The compiled gate is a ramp of width 1e-4 just below the threshold.
_PROG_GAIN = 10000.0
_PROG_BIAS = 174.0  # = gain * 0.0175 - 1


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

    # ---- per-slot stall-ineffective score ---------------------------------
    # ine_s = relu(repeated_intent + state_revisited + repeated_failure
    #              - out_NEUTRAL)            (0 for a novel or neutral event)
    ine = [
        _relu(f(s, _RI) + f(s, _SR) + f(s, _RF) - f(s, _NEU)) for s in range(_SLOTS)
    ]
    stall_evidence = sum(ine)            # C
    endpoint_ineffective = _clip01(      # END = min(1, ine_7)
        f(7, _RI) + f(7, _SR) + f(7, _RF) - f(7, _NEU)
    )

    # ---- reversions --------------------------------------------------------
    reversions = sum(f(s, _SR) for s in range(_SLOTS))      # V
    two_reversions = _clip01(reversions - 1.0)              # V2 = [V >= 2]

    # ---- net validation progress ------------------------------------------
    net_delta = sum(f(s, _VD) for s in range(_SLOTS))       # S
    progress = _clip01(_PROG_GAIN * net_delta - _PROG_BIAS)  # PROG = [S >= 0.0175]

    # ---- recent episode-like errors (ERROR that is not a comparable test) --
    recent_errors = sum(_relu(f(s, _ERR) - f(s, _VC)) for s in _RECENT)   # E

    # ---- retry bound: repeated failures beyond policy ----------------------
    repeated_failures = sum(f(s, _RF) for s in range(_SLOTS))
    out_of_policy = _relu(repeated_failures - 2.0 * f(7, _EXT) - 3.0)     # OOP
    active_episode = _clip01(recent_errors - 10.0 * out_of_policy)        # ACT

    # ---- resolution: successful retry of MUTATE/VALIDATE/EXTERNAL after an
    #      earlier episode-like error, within the last 4 events -------------
    resolution_sum = 0.0
    for s in _RECENT:
        prior = sum(f(j, _ERR) - f(j, _VC) for j in range(s))
        conj = (
            f(s, _RI) + f(s, _SUC) + f(s, _MUT) + f(s, _VAL) + f(s, _EXT)
            - f(s, _VC) - 3.0
        )
        resolution_sum += _relu(conj + prior / 8.0)
    resolved = _clip01(8.0 * resolution_sum)                              # RES

    # ---- endpoint is an in-flight poll (repeated NEUTRAL EXTERNAL) --------
    polling = _relu(f(7, _RI) + f(7, _NEU) + f(7, _EXT) - 2.0)            # POLL

    # ---- class conditions --------------------------------------------------
    osc = _relu(two_reversions - progress)                                # OSC
    stall_pre = (
        stall_evidence + 30.0 * endpoint_ineffective - 34.0
        - 30.0 * progress - 30.0 * two_reversions
    )
    stall_raw = _clip01(stall_pre)                 # STALL' (before the episode veto)

    # ---- logits with precedence OSCILLATING > STALLED > RECOVERY > PRODUCTIVE
    logit_productive = 0.0
    logit_recovery = (
        20.0 * resolved + 20.0 * polling + 200.0 * active_episode
        - 100.0 * stall_raw - 1000.0 * osc
    )
    logit_stalled = 20.0 * stall_raw - 60.0 * active_episode
    logit_oscillating = 20.0 * osc

    logits = (logit_productive, logit_recovery, logit_stalled, logit_oscillating)
    best = 0
    for i in range(1, 4):
        if logits[i] > logits[best]:
            best = i
    return int(best)
