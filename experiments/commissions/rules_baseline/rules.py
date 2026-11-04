"""Independent, hand-written baseline for the v0.2 authoring packet."""


def rules_classify(x: tuple[float, ...]) -> int:
    """Classify a gated 8 x 15 window; class order is P, R, S, O."""
    if len(x) != 120:
        raise ValueError("expected exactly 120 features")
    events = tuple(x[i:i + 15] for i in range(0, 120, 15))
    net_validation = sum(e[13] for e in events if e[12] == 1.0)
    progress = net_validation > 0.02
    reversions = sum(e[1] == 1.0 and e[11] == 1.0 for e in events)
    novel_inspections = sum(
        e[0] == 1.0 and e[6] == 0.0 and e[8] == 0.0 for e in events
    )

    # One visible failure episode at a time; category substitutes for identity.
    episode_category = None
    retries = 0
    resolved_at = -8
    for i, e in enumerate(events):
        category = e[:5].index(1.0)
        if e[6] == 1.0:
            if episode_category != category:
                episode_category = category
                retries = 0
            elif e[8] == 1.0:
                retries += 1
        elif episode_category is not None:
            if (category == episode_category and e[5] == 1.0 and e[8] == 1.0) or (
                e[12] == 1.0 and e[13] > 0.0
            ):
                episode_category = None
                retries = 0
                resolved_at = i
            elif category == episode_category and e[8] == 1.0:
                retries += 1

    active = episode_category is not None and retries <= (
        5 if episode_category == 3 else 3
    )
    # Neutral external polling can be a bounded episode without an ERROR.
    poll_start = 8
    while poll_start > 0:
        e = events[poll_start - 1]
        if e[3] != 1.0 or e[7] != 1.0:
            break
        poll_start -= 1
    polls = sum(e[8] == 1.0 for e in events[poll_start:])
    active = active or (poll_start < 8 and 1 <= polls <= 5)

    ineffective = tuple(
        (e[1] == 1.0 and e[11] == 1.0)
        or ((e[8] == 1.0 or e[9] == 1.0) and not (e[12] == 1.0 and e[13] > 0.0))
        for e in events
    )
    if reversions >= 2 and not progress:
        return 3
    if (not progress and novel_inspections == 0 and sum(ineffective) >= 5
            and ineffective[-1] and not active):
        return 2
    if active or resolved_at >= 4:
        return 1
    return 0
