"""Family 11, rising_cost (M1_CONTRACTS §3, §7)."""

from __future__ import annotations

import collections
import statistics

from agent_loop_lab.synth.families import rising_cost
from agent_loop_lab.synth.latent import steps_from_record
from tests import synth_helpers as H

FAMILY = "rising_cost"


def _variant(record) -> str:
    return record.latent_trace["meta"]["variant"]


def _tokens(events) -> list[int]:
    return [e.input_tokens + e.output_tokens for e in events]


def test_family_contract_and_plurality():
    mix = H.check_family(FAMILY, intended="STALLED")
    assert mix["RECOVERY"] == mix["OSCILLATING"] == 0
    assert mix["PRODUCTIVE"] > 0


def test_variant_label_mix():
    per: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for _, record in H.sessions(FAMILY):
        per[_variant(record)].update(v.value for v in record.window_labels.values())
    H.assert_plurality(per["redundant"], "STALLED")
    H.assert_plurality(per["productive"], "PRODUCTIVE")
    assert per["productive"]["STALLED"] == 0


def test_cost_rises_in_both_variants_and_does_not_separate_them():
    rising: dict[str, list[bool]] = collections.defaultdict(list)
    mean_tok: dict[str, list[float]] = collections.defaultdict(list)
    for events, record in H.sessions(FAMILY, n=200):
        t = _tokens(events)
        k = len(t) // 3
        rising[_variant(record)].append(statistics.mean(t[-k:]) > statistics.mean(t[:k]))
        mean_tok[_variant(record)].append(statistics.mean(t))
    for v in ("redundant", "productive"):
        assert sum(rising[v]) / len(rising[v]) >= 0.85, v
    # Best single threshold on a session's mean token cost barely beats the majority class.
    xs = sorted([(x, 0) for x in mean_tok["redundant"]] + [(x, 1) for x in mean_tok["productive"]])
    n = len(xs)
    ones = sum(y for _, y in xs)
    best, left0, left1 = max(ones, n - ones), 0, 0
    for _, y in xs:
        left0 += y == 0
        left1 += y == 1
        acc = left0 + (ones - left1)
        best = max(best, acc, n - acc)
    majority = max(ones, n - ones) / n
    assert best / n <= majority + 0.15, (best / n, majority)


def test_labels_and_effects_do_not_depend_on_cost(monkeypatch):
    """Flatten the ramp: same operations, effects and labels; only the tokens change."""
    sds = H.seeds(FAMILY, 20)
    ramped = [H.generate(FAMILY, sd) for sd in sds]
    monkeypatch.setattr(rising_cost._Ramp, "__call__", lambda self, k: 1.0)
    flat = [H.generate(FAMILY, sd) for sd in sds]
    differs = 0
    for (ev_r, rec_r), (ev_f, rec_f) in zip(ramped, flat):
        assert rec_r.latent_trace["ops"] == rec_f.latent_trace["ops"]
        assert rec_r.event_effects == rec_f.event_effects
        assert rec_r.window_labels == rec_f.window_labels
        differs += _tokens(ev_r) != _tokens(ev_f)
    assert differs == len(sds)


def test_redundant_variant_reacquires_known_content():
    for events, record in H.sessions(FAMILY):
        steps, _ = steps_from_record(record)
        assert all(s.open_episode_kind is None for s in steps)
        if _variant(record) == "redundant":
            tail = steps[len(steps) // 2 :]
            assert sum(s.effect.value == "REDUNDANT" for s in tail) >= len(tail) // 2, record.session_id
        else:
            assert steps[-1].progress_after > steps[0].progress_after


def test_sessions_vary():
    shapes, variants, families, langs = collections.Counter(), collections.Counter(), collections.Counter(), set()
    lengths: dict[str, list[int]] = collections.defaultdict(list)
    for events, record in H.sessions(FAMILY):
        meta = record.latent_trace["meta"]
        shapes[meta["ramp"]["shape"]] += 1
        variants[meta["variant"]] += 1
        lengths[meta["variant"]].append(len(events))
        families.update(e.operation_family for e in events)
        langs.add(record.latent_trace["world"]["language"])
    assert set(shapes) == {"geometric", "linear", "steps"}
    assert variants["redundant"] > variants["productive"] > 0
    assert len(langs) >= 4
    # Length ranges of the two variants overlap.
    assert min(lengths["redundant"]) <= max(lengths["productive"]) and min(lengths["productive"]) <= max(lengths["redundant"])
    for fam in ("read_file", "search", "list_dir", "run_tests", "plan", "edit_file", "http_call"):
        assert families[fam] > 0, fam
