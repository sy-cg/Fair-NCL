import unittest

from analysis.supplementary_statistics import (
    exact_paired_sign_flip_pvalue,
    holm_adjust,
    paired_hierarchical_bootstrap,
    paired_hierarchical_bootstrap_many,
    pareto_front,
    select_validation_configuration,
    summarize_group_metric,
)


class GroupMetricTests(unittest.TestCase):
    def test_summary_preserves_direction_and_worst_group(self):
        rows = [
            {"gender": 0, "NDCG@10": 0.8},
            {"gender": 0, "NDCG@10": 0.6},
            {"gender": 1, "NDCG@10": 0.2},
            {"gender": 1, "NDCG@10": 0.4},
        ]

        summary = summarize_group_metric(rows, "NDCG@10", "gender")

        self.assertAlmostEqual(summary["group_values"][0], 0.7)
        self.assertAlmostEqual(summary["group_values"][1], 0.3)
        self.assertAlmostEqual(summary["signed_gap"], 0.4)
        self.assertAlmostEqual(summary["absolute_gap"], 0.4)
        self.assertAlmostEqual(summary["worst_group"], 0.3)
        self.assertEqual(summary["worst_group_id"], 1)
        self.assertAlmostEqual(summary["relative_gap"], 0.4 / 0.7)


class BootstrapTests(unittest.TestCase):
    def test_batched_bootstrap_reuses_draws_for_multiple_estimands(self):
        baseline = []
        method = []
        for seed in (42, 2025, 2026, 2027, 2028):
            baseline.extend([
                {"seed": seed, "user_id": 1, "gender": 0, "NDCG@10": 0.6},
                {"seed": seed, "user_id": 2, "gender": 1, "NDCG@10": 0.2},
            ])
            method.extend([
                {"seed": seed, "user_id": 1, "gender": 0, "NDCG@10": 0.6},
                {"seed": seed, "user_id": 2, "gender": 1, "NDCG@10": 0.3},
            ])

        results = paired_hierarchical_bootstrap_many(
            method,
            baseline,
            specifications=[
                ("NDCG@10", "gender", "gap", None),
                ("NDCG@10", "gender", "group_mean", 1),
            ],
            iterations=50,
            random_seed=7,
        )

        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(results[0]["mean_delta"], -0.1)
        self.assertAlmostEqual(results[1]["mean_delta"], 0.1)
        self.assertEqual(results[0]["iterations"], 50)
    def test_paired_bootstrap_keeps_constant_user_improvement(self):
        baseline = []
        method = []
        for seed in (42, 2025, 2026, 2027, 2028):
            for user_id in range(8):
                baseline.append({"seed": seed, "user_id": user_id, "NDCG@10": 0.2})
                method.append({"seed": seed, "user_id": user_id, "NDCG@10": 0.3})

        result = paired_hierarchical_bootstrap(
            method,
            baseline,
            metric="NDCG@10",
            iterations=200,
            random_seed=7,
        )

        self.assertAlmostEqual(result["mean_delta"], 0.1)
        self.assertAlmostEqual(result["ci_lower"], 0.1)
        self.assertAlmostEqual(result["ci_upper"], 0.1)
        self.assertEqual(result["paired_seeds"], 5)
        self.assertEqual(result["paired_users"], 40)
        self.assertAlmostEqual(result["baseline_estimate"], 0.2)
        self.assertAlmostEqual(result["method_estimate"], 0.3)

    def test_paired_bootstrap_rejects_mismatched_user_sets(self):
        baseline = [
            {"seed": 42, "user_id": 1, "NDCG@10": 0.2},
            {"seed": 42, "user_id": 2, "NDCG@10": 0.3},
        ]
        method = [{"seed": 42, "user_id": 1, "NDCG@10": 0.4}]

        with self.assertRaisesRegex(ValueError, "identical user sets"):
            paired_hierarchical_bootstrap(method, baseline, metric="NDCG@10", iterations=10)

    def test_paired_bootstrap_reports_named_group_change(self):
        baseline = []
        method = []
        for seed in (42, 2025, 2026, 2027, 2028):
            baseline.extend([
                {"seed": seed, "user_id": 1, "gender": 0, "NDCG@10": 0.6},
                {"seed": seed, "user_id": 2, "gender": 1, "NDCG@10": 0.2},
            ])
            method.extend([
                {"seed": seed, "user_id": 1, "gender": 0, "NDCG@10": 0.6},
                {"seed": seed, "user_id": 2, "gender": 1, "NDCG@10": 0.3},
            ])

        result = paired_hierarchical_bootstrap(
            method,
            baseline,
            metric="NDCG@10",
            group_attribute="gender",
            group_value=1,
            estimand="group_mean",
            iterations=50,
        )

        self.assertAlmostEqual(result["baseline_estimate"], 0.2)
        self.assertAlmostEqual(result["method_estimate"], 0.3)
        self.assertAlmostEqual(result["mean_delta"], 0.1)


class MultipleComparisonTests(unittest.TestCase):
    def test_exact_five_seed_sign_flip_has_minimum_two_sided_p_value(self):
        self.assertAlmostEqual(exact_paired_sign_flip_pvalue([1, 1, 1, 1, 1]), 0.0625)

    def test_holm_adjustment_is_monotone_in_sorted_p_values(self):
        adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.2})

        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["b"], 0.06)
        self.assertAlmostEqual(adjusted["c"], 0.2)


class ParetoAndSelectionTests(unittest.TestCase):
    def test_pareto_front_maximizes_utility_and_worst_group(self):
        rows = [
            {"method": "a", "utility": 0.8, "fairness": 0.5},
            {"method": "b", "utility": 0.7, "fairness": 0.6},
            {"method": "c", "utility": 0.6, "fairness": 0.4},
        ]

        self.assertEqual(
            pareto_front(rows, utility_key="utility", fairness_key="fairness"),
            {"a", "b"},
        )

    def test_pareto_front_can_minimize_disparity(self):
        rows = [
            {"method": "a", "utility": 0.8, "gap": 0.2},
            {"method": "b", "utility": 0.7, "gap": 0.1},
            {"method": "c", "utility": 0.6, "gap": 0.3},
        ]

        self.assertEqual(
            pareto_front(
                rows,
                utility_key="utility",
                fairness_key="gap",
                fairness_higher_is_better=False,
            ),
            {"a", "b"},
        )

    def test_validation_selection_enforces_fifteen_percent_conservative_utility_constraint(self):
        candidates = [
            {
                "candidate": "high_fairness_but_unstable_utility",
                "NDCG@10": 0.90,
                "utility_retention_conservative": 0.84,
                "fairness_benefit": 0.20,
                "gap_risk": 0.05,
                "worst_group": 0.80,
            },
            {
                "candidate": "qualified",
                "NDCG@10": 0.90,
                "utility_retention_conservative": 0.86,
                "fairness_benefit": 0.10,
                "gap_risk": 0.06,
                "worst_group": 0.75,
            },
            {
                "candidate": "utility_only",
                "NDCG@10": 1.00,
                "utility_retention_conservative": 0.95,
                "fairness_benefit": 0.01,
                "gap_risk": 0.10,
                "worst_group": 0.70,
            },
        ]

        selected = select_validation_configuration(candidates, backbone_ndcg=1.0)

        self.assertEqual(selected["candidate"], "qualified")
        self.assertEqual(selected["selection_status"], "qualified")
        self.assertAlmostEqual(selected["utility_tolerance"], 0.15)
        self.assertAlmostEqual(selected["utility_retention_floor"], 0.85)

    def test_validation_selection_prioritizes_fixed_group_benefit_then_gap_risk(self):
        candidates = [
            {
                "candidate": "larger_benefit",
                "NDCG@10": 1.0,
                "utility_retention_conservative": 0.90,
                "fairness_benefit": 0.03,
                "gap_risk": 0.20,
                "worst_group": 0.7,
            },
            {
                "candidate": "smaller_gap",
                "NDCG@10": 1.0,
                "utility_retention_conservative": 0.90,
                "fairness_benefit": 0.02,
                "gap_risk": 0.01,
                "worst_group": 0.7,
            },
        ]

        selected = select_validation_configuration(candidates, backbone_ndcg=1.0)

        self.assertEqual(selected["candidate"], "larger_benefit")
        self.assertAlmostEqual(selected["fairness_risk_score"], 0.20)

    def test_validation_selection_falls_back_to_highest_ndcg(self):
        candidates = [
            {
                "candidate": "fairer",
                "NDCG@10": 0.80,
                "utility_retention_conservative": 0.84,
                "fairness_benefit": 0.20,
                "gap_risk": 0.01,
                "worst_group": 0.70,
            },
            {
                "candidate": "higher_utility",
                "NDCG@10": 0.82,
                "utility_retention_conservative": 0.83,
                "fairness_benefit": 0.10,
                "gap_risk": 0.02,
                "worst_group": 0.68,
            },
        ]

        selected = select_validation_configuration(candidates, backbone_ndcg=1.0)

        self.assertEqual(selected["candidate"], "higher_utility")
        self.assertEqual(selected["selection_status"], "no_qualified_candidate")


if __name__ == "__main__":
    unittest.main()
