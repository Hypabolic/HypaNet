import unittest

from rules import rules_classify


def event(category=4, outcome=7, *, repeat=False, failure=False,
          changed=False, revisit=False, comparable=False, delta=0.0, cost=0.0):
    e = [0.0] * 15
    e[category] = e[outcome] = 1.0
    e[8:15] = map(float, (repeat, failure, changed, revisit, comparable, delta, cost))
    return tuple(e)


def window(*tail):
    return sum((event(),) * (8 - len(tail)) + tail, ())


class RulesTests(unittest.TestCase):
    def test_productive_default_and_novel_exploration(self):
        self.assertEqual(rules_classify(window()), 0)
        self.assertEqual(rules_classify(window(*(event(0) for _ in range(8)))), 0)

    def test_success_changes_and_cost_do_not_prove_progress(self):
        tail = (event(1, 5, changed=True),) + (event(1, 5, repeat=True, changed=True),) * 7
        self.assertEqual(rules_classify(window(*tail)), 2)
        self.assertEqual(rules_classify(window(*(event(cost=1.0) for _ in range(8)))), 0)

    def test_stall_count_and_final_event(self):
        repeated = event(0, repeat=True)
        self.assertEqual(rules_classify(window(*([repeated] * 4))), 0)
        self.assertEqual(rules_classify(window(*([repeated] * 5))), 2)
        self.assertEqual(rules_classify(window(*([repeated] * 5), event())), 0)
        self.assertEqual(rules_classify(window(event(0), *([repeated] * 5))), 0)

    def test_repeated_failure_counts_without_intent_repeat(self):
        # A neutral external request followed by distinct failing intents can
        # share a failure fingerprint. Six same-category repeats exceed policy.
        errors = (event(2, 6),) + (event(2, 6, repeat=True, failure=True),) * 4
        extra = event(3, 6, failure=True)
        # New external episode would protect the endpoint; resolve it first.
        self.assertEqual(rules_classify(window(*errors)), 0)
        self.assertEqual(rules_classify(window(*errors, extra, event(3, 5, repeat=True),
                                             event(0, repeat=True))), 2)

    def test_retry_budget(self):
        first = event(2, 6)
        retry = event(2, 6, repeat=True, failure=True)
        self.assertEqual(rules_classify(window(first, *([retry] * 3))), 1)
        self.assertEqual(rules_classify(window(first, *([retry] * 4))), 0)
        self.assertEqual(rules_classify(window(first, *([retry] * 5))), 2)

    def test_external_budget_and_neutral_polls(self):
        for outcome in (6, 7):
            with self.subTest(outcome=outcome):
                first = event(3, outcome)
                repeat = event(3, outcome, repeat=True, failure=outcome == 6)
                self.assertEqual(rules_classify(window(first, *([repeat] * 5))), 1)
                self.assertEqual(rules_classify(window(first, *([repeat] * 6))), 2)
        self.assertEqual(rules_classify(window(event(3))), 0)

    def test_resolution_recency_boundary(self):
        first, resolved = event(3, 6), event(3, 5, repeat=True)
        self.assertEqual(rules_classify(window(first, resolved, *([event()] * 3))), 1)
        self.assertEqual(rules_classify(window(first, resolved, *([event()] * 4))), 0)

    def test_validation_resolution_and_recovery_over_progress(self):
        self.assertEqual(rules_classify(window(event(2, 6),
                                             event(2, 5, comparable=True, delta=0.1))), 1)
        self.assertEqual(rules_classify(window(event(2, 5, comparable=True, delta=0.1),
                                             event(3, 6))), 1)

    def test_oscillation_threshold_and_precedence(self):
        rev = event(1, 5, changed=True, revisit=True)
        self.assertEqual(rules_classify(window(rev)), 0)
        self.assertEqual(rules_classify(window(rev, rev)), 3)
        self.assertEqual(rules_classify(window(rev, rev, event(3, 6))), 3)
        self.assertEqual(rules_classify(window(rev, rev, *([event(0, repeat=True)] * 5))), 3)
        self.assertEqual(rules_classify(window(event(0), rev, rev)), 3)
        external = event(3, 5, changed=True, revisit=True)
        self.assertEqual(rules_classify(window(external, external)), 0)

    def test_progress_threshold_netting_and_positive_event_exemption(self):
        rev = event(1, 5, changed=True, revisit=True)
        self.assertEqual(rules_classify(window(rev, rev,
                                             event(2, comparable=True, delta=0.02))), 3)
        self.assertEqual(rules_classify(window(rev, rev,
                                             event(2, comparable=True, delta=0.021))), 0)
        self.assertEqual(rules_classify(window(rev, rev,
                                             event(2, comparable=True, delta=0.1),
                                             event(2, comparable=True, delta=-0.1))), 3)
        repeated = event(2, repeat=True, comparable=True)
        self.assertEqual(rules_classify(window(*([repeated] * 5),
                                             event(2, repeat=True, comparable=True, delta=0.01))), 0)

    def test_stall_over_recent_resolution(self):
        self.assertEqual(rules_classify(window(event(3, 6),
                                             *([event(0, repeat=True)] * 4),
                                             event(3, 5, repeat=True),
                                             event(0, repeat=True))), 2)

    def test_new_episode_category_and_determinism(self):
        x = window(event(2, 6), event(2, 6, repeat=True), event(3, 6))
        self.assertEqual(rules_classify(x), 1)
        self.assertEqual(rules_classify(x), rules_classify(x))
        self.assertIsInstance(rules_classify(x), int)

    def test_length_contract(self):
        with self.assertRaises(ValueError):
            rules_classify(())


if __name__ == '__main__':
    unittest.main()
