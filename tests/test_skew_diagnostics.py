import unittest

from analysis.skew_diagnostics import build_skew_diagnostics


class SkewDiagnosticsTests(unittest.TestCase):
    def test_diagnostics_compare_interaction_and_user_prevalence_priors(self):
        data = {
            "dataset": "toy",
            "sequences": {
                1: [1, 2, 3, 4, 5, 6, 7, 8],
                2: [1, 2, 9, 10],
                3: [1, 3, 11, 12],
                4: [2, 3, 13, 14],
            },
            "train_data": [
                {"user_id": 1, "input_seq": [1, 2, 3, 4, 5], "target": 6, "gender": 0, "age_group": 0},
                {"user_id": 2, "input_seq": [1], "target": 2, "gender": 0, "age_group": 1},
                {"user_id": 3, "input_seq": [1], "target": 3, "gender": 1, "age_group": 0},
                {"user_id": 4, "input_seq": [2], "target": 3, "gender": 1, "age_group": 1},
            ],
        }

        diagnostics = build_skew_diagnostics(data, min_count=1, permutations=0)
        summary = diagnostics["summary"]

        self.assertIn("interaction_prior", summary["prior_sensitivity"])
        self.assertIn("user_prior", summary["prior_sensitivity"])
        interaction_gender = summary["prior_sensitivity"]["interaction_prior"]["gender"]
        user_gender = summary["prior_sensitivity"]["user_prior"]["gender"]
        self.assertNotEqual(interaction_gender, user_gender)
        self.assertIn("interaction_vs_user_adjusted", summary["correlations"])
        self.assertIn("adjusted_vs_log_popularity", summary["correlations"])
        self.assertIn("interaction_vs_user_low_skew_decile_overlap", summary)
        self.assertIn("train_popularity", diagnostics["items"][0])


if __name__ == "__main__":
    unittest.main()
