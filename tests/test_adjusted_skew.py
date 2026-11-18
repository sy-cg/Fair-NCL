import math
import unittest

from fairness.skew import (
    build_leave_two_out_train_events,
    build_train_event_sequences,
    compute_train_item_bias_statistics,
    prevalence_adjusted_jsd,
)


class PrevalenceAdjustedSkewTests(unittest.TestCase):
    def test_leave_two_out_events_exclude_validation_and_test_targets(self):
        sequences = {1: [1, 2, 3, 4], 2: [5, 6, 7], 3: [8, 9]}

        self.assertEqual(
            build_leave_two_out_train_events(sequences),
            {1: [1, 2], 2: [5]},
        )

    def test_train_events_fall_back_to_longest_training_prefix(self):
        train_data = [
            {"user_id": 1, "input_seq": [1], "target": 2},
            {"user_id": 1, "input_seq": [1, 2], "target": 3},
            {"user_id": 2, "input_seq": [4], "target": 5},
        ]

        self.assertEqual(
            build_train_event_sequences({}, train_data),
            {1: [1, 2, 3], 2: [4, 5]},
        )

    def test_train_event_fallback_stitches_truncated_prefixes(self):
        train_data = [
            {"user_id": 1, "input_seq": [1], "target": 2},
            {"user_id": 1, "input_seq": [1, 2], "target": 3},
            {"user_id": 1, "input_seq": [2, 3], "target": 4},
        ]

        self.assertEqual(
            build_train_event_sequences({}, train_data),
            {1: [1, 2, 3, 4]},
        )

    def test_population_matching_item_has_zero_adjusted_skew(self):
        prior = [0.15, 0.85]

        matching = prevalence_adjusted_jsd([15, 85], prior, prior_strength=5.0)
        artificially_balanced = prevalence_adjusted_jsd([50, 50], prior, prior_strength=5.0)

        self.assertAlmostEqual(matching, 0.0, places=12)
        self.assertGreater(artificially_balanced, 0.05)
        self.assertLessEqual(artificially_balanced, 1.0)

    def test_unique_training_sequences_prevent_prefix_double_counting(self):
        train_data = [
            {"user_id": 1, "input_seq": [1], "target": 2, "gender": 0},
            {"user_id": 1, "input_seq": [1, 2], "target": 3, "gender": 0},
            {"user_id": 2, "input_seq": [1], "target": 2, "gender": 1},
            {"user_id": 2, "input_seq": [1, 2], "target": 4, "gender": 1},
        ]
        duplicated_prefixes = train_data + [dict(train_data[0]), dict(train_data[2])]
        train_events = {1: [1, 2, 3], 2: [1, 2, 4]}

        original = compute_train_item_bias_statistics(
            train_data,
            sensitive_attrs=("gender",),
            train_event_sequences=train_events,
            metric="prior_jsd",
            min_count=1,
            prior_strength=2.0,
        )
        duplicated = compute_train_item_bias_statistics(
            duplicated_prefixes,
            sensitive_attrs=("gender",),
            train_event_sequences=train_events,
            metric="prior_jsd",
            min_count=1,
            prior_strength=2.0,
        )

        self.assertEqual(original["scores"], duplicated["scores"])
        self.assertEqual(original["item_support"], duplicated["item_support"])
        self.assertEqual(original["group_priors"], duplicated["group_priors"])

        submitted = compute_train_item_bias_statistics(
            train_data,
            sensitive_attrs=("gender",),
            train_event_sequences=train_events,
            metric="raw_range",
            min_count=1,
            event_source="prefix_samples",
        )
        self.assertEqual(submitted["num_unique_train_events"], 10)

    def test_low_support_items_are_not_eligible_low_skew_candidates(self):
        train_data = [
            {"user_id": 1, "input_seq": [1], "target": 2, "gender": 0},
            {"user_id": 2, "input_seq": [1], "target": 3, "gender": 1},
        ]
        train_events = {1: [1, 2], 2: [1, 3]}

        statistics = compute_train_item_bias_statistics(
            train_data,
            sensitive_attrs=("gender",),
            train_event_sequences=train_events,
            metric="prior_jsd",
            min_count=2,
            prior_strength=2.0,
        )

        self.assertIn(1, statistics["scores"])
        self.assertIn(1, statistics["per_attribute_scores"]["gender"])
        self.assertNotIn(2, statistics["scores"])
        self.assertNotIn(3, statistics["scores"])
        self.assertEqual(statistics["excluded_low_support_items"], 2)
        self.assertTrue(math.isclose(sum(statistics["group_priors"]["gender"].values()), 1.0))


if __name__ == "__main__":
    unittest.main()
