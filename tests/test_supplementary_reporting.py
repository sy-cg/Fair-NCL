import json
import tempfile
import unittest
from pathlib import Path

from analysis.run_supplementary_analysis import build_supplementary_report


class SupplementaryReportingTests(unittest.TestCase):
    def test_report_discovers_runs_and_builds_paired_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for method, offset in (("baseline", 0.0), ("fair_ncl", 0.1)):
                for seed in (42, 2025, 2026, 2027, 2028):
                    run_dir = root / "ml-1m" / f"{method}_{seed}"
                    run_dir.mkdir(parents=True)
                    job = {
                        "job_id": f"{method}_{seed}",
                        "phase": "supplementary",
                        "dataset": "ml-1m",
                        "backbone": "sasrec",
                        "method": method,
                        "seed": seed,
                    }
                    users = [
                        {
                            "user_id": 1,
                            "gender": 0,
                            "age_group": 0,
                            "gender_age": "g0_a0",
                            "HitRate@10": 1.0,
                            "NDCG@10": 0.6 + offset,
                            "HitRate@20": 1.0,
                            "NDCG@20": 0.7 + offset,
                        },
                        {
                            "user_id": 2,
                            "gender": 1,
                            "age_group": 1,
                            "gender_age": "g1_a1",
                            "HitRate@10": 0.0,
                            "NDCG@10": 0.2 + offset,
                            "HitRate@20": 1.0,
                            "NDCG@20": 0.3 + offset,
                        },
                    ]
                    result = {
                        "utility": {
                            "HitRate@10": 0.5,
                            "NDCG@10": 0.4 + offset,
                            "HitRate@20": 1.0,
                            "NDCG@20": 0.5 + offset,
                        },
                        "selected": {},
                        "per_user": users,
                        "augmentation_diagnostics": {
                            "attempted_positions": 10,
                            "replaced_positions": 8,
                            "replacement_rate": 0.8,
                            "lower_skew_fraction": 0.75 if method == "fair_ncl" else 0.0,
                            "mean_skew_change": 0.05 if method == "fair_ncl" else 0.0,
                            "mean_similarity": 0.9 if method == "fair_ncl" else None,
                        },
                    }
                    (run_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
                    (run_dir / "test_results.json").write_text(json.dumps(result), encoding="utf-8")
                    (run_dir / "run_provenance.json").write_text(
                        json.dumps({
                            "environment": {"gpu_name": "Test GPU", "torch_version": "test"},
                            "code": {"git_commit": "abc123"},
                            "dataset": {"num_users": 2, "num_items": 10, "num_train_samples": 20},
                        }),
                        encoding="utf-8",
                    )

            report = build_supplementary_report(root, bootstrap_iterations=100)

        self.assertEqual(len(report["run_summary"]), 2)
        comparison = next(
            row for row in report["paired_comparisons"]
            if row["metric"] == "NDCG@10"
            and row["group_attribute"] is None
            and row["estimand"] == "mean"
        )
        self.assertAlmostEqual(comparison["mean_delta"], 0.1)
        self.assertAlmostEqual(comparison["ci_lower"], 0.1)
        self.assertEqual(comparison["paired_seeds"], 5)
        self.assertAlmostEqual(comparison["baseline_estimate"], 0.4)
        self.assertAlmostEqual(comparison["method_estimate"], 0.5)
        self.assertEqual(comparison["improvement_direction"], "higher")
        self.assertTrue(comparison["bootstrap_supported_improvement"])
        self.assertAlmostEqual(comparison["holm_p_value"], comparison["raw_p_value"])
        self.assertFalse(comparison["statistically_supported_improvement"])
        self.assertTrue(any(row["metric"] == "HitRate@10" for row in report["paired_comparisons"]))
        self.assertTrue(any(row["metric"] == "NDCG@20" for row in report["paired_comparisons"]))
        group_change = next(
            row for row in report["paired_comparisons"]
            if row["metric"] == "NDCG@10"
            and row["group_attribute"] == "gender"
            and row["group_value"] == 1
            and row["estimand"] == "group_mean"
        )
        self.assertAlmostEqual(group_change["mean_delta"], 0.1)
        gender_summary = next(
            row for row in report["group_summary"]
            if row["method"] == "fair_ncl"
            and row["attribute"] == "gender"
            and row["metric"] == "NDCG@10"
        )
        self.assertEqual(json.loads(gender_summary["group_counts"]), {"0": 1.0, "1": 1.0})
        fair_run = next(row for row in report["run_summary"] if row["method"] == "fair_ncl")
        self.assertAlmostEqual(fair_run["mean_skew_change_mean"], 0.05)
        self.assertEqual(fair_run["num_train_samples"], 20)
        self.assertEqual(fair_run["gpu_names"], "Test GPU")
        proxy_row = next(
            row for row in report["proxy_outcome"]
            if row["method"] == "fair_ncl"
            and row["seed"] == 42
            and row["attribute"] == "gender"
            and row["metric"] == "NDCG@10"
        )
        self.assertAlmostEqual(proxy_row["mean_skew_change"], 0.05)
        self.assertAlmostEqual(proxy_row["worst_group_delta"], 0.1)
        self.assertTrue(any(row["is_pareto"] for row in report["pareto"]))
        self.assertEqual(
            {row["fairness_axis"] for row in report["pareto"]},
            {"marginal_joint_worst", "intersectional_worst"},
        )
        self.assertEqual(report["manifest"]["incomplete_groups"], [])

    def test_report_rejects_a_missing_job_from_the_declared_plan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "ml-1m" / "baseline_42"
            run_dir.mkdir(parents=True)
            job = {
                "job_id": "baseline_42",
                "phase": "supplementary",
                "dataset": "ml-1m",
                "backbone": "sasrec",
                "method": "baseline",
                "seed": 42,
            }
            result = {
                "utility": {"HitRate@10": 0.5, "NDCG@10": 0.4},
                "per_user": [{"user_id": 1, "gender": 0, "age_group": 0, "NDCG@10": 0.4}],
            }
            (run_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
            (run_dir / "test_results.json").write_text(json.dumps(result), encoding="utf-8")
            plan_path = root / "expected.jsonl"
            expected = [job, {**job, "job_id": "fair_ncl_42", "method": "fair_ncl"}]
            plan_path.write_text("".join(json.dumps(row) + "\n" for row in expected), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing 1 expected jobs"):
                build_supplementary_report(
                    root,
                    bootstrap_iterations=10,
                    expected_jobs_path=plan_path,
                    require_complete=True,
                )


if __name__ == "__main__":
    unittest.main()
