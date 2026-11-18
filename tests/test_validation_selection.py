import json
import tempfile
import unittest
from pathlib import Path

from analysis.select_validation_configs import (
    build_method_override_config,
    build_validation_selection,
)


class ValidationSelectionTests(unittest.TestCase):
    def test_selection_uses_validation_only_and_exports_method_scopes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for seed in (42, 2025, 2026):
                self._write_run(root, "baseline", seed, "validation_reference", None, 1.0, 0.2)
                candidates = {
                    "c0": (0.995, 0.10),
                    "c1": (0.980, 0.01),
                    "c2": (0.992, 0.15),
                    "c3": (0.991, 0.16),
                    "c4": (1.000, 0.20),
                }
                for candidate, (utility, gap) in candidates.items():
                    self._write_run(root, "fair_ncl", seed, "validation_audit", candidate, utility, gap)
            self._write_run(root, "baseline", 42, "comparison", None, 100.0, 0.0)

            selections = build_validation_selection(root)
            config = build_method_override_config(selections)

        self.assertEqual(len(selections), 1)
        self.assertEqual(selections[0]["candidate"], "c1")
        self.assertEqual(len(selections[0]["all_candidates"]), 5)
        self.assertTrue(any(candidate["is_validation_pareto"] for candidate in selections[0]["all_candidates"]))
        self.assertTrue(all("fairness_risk_score" in candidate for candidate in selections[0]["all_candidates"]))
        self.assertEqual(
            selections[0]["baseline_disadvantaged_groups"],
            {"gender": "1", "age_group": "1"},
        )
        selected_candidate = next(
            candidate
            for candidate in selections[0]["all_candidates"]
            if candidate["candidate"] == "c1"
        )
        self.assertAlmostEqual(selected_candidate["utility_retention_conservative"], 0.98)
        self.assertAlmostEqual(selected_candidate["fairness_benefit"], 0.17)
        self.assertAlmostEqual(selected_candidate["gap_risk"], 0.01 / 0.98)
        self.assertTrue(selected_candidate["fairness_improving"])
        selected = config["__methods__"]["fair_ncl"]["__dataset_backbones__"]["ml-1m"]["sasrec"]
        self.assertEqual(selected["utility_beta"], 0.3)
        self.assertNotIn("audit_candidate", selected)

    def test_selection_rejects_incomplete_three_seed_candidate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for seed in (42, 2025, 2026):
                self._write_run(root, "baseline", seed, "validation_reference", None, 1.0, 0.2)
            for seed in (42, 2025):
                self._write_run(root, "fair_ncl", seed, "validation_audit", "c0", 0.995, 0.1)
            with self.assertRaisesRegex(ValueError, "Incomplete validation candidate"):
                build_validation_selection(root)

    def test_selection_rejects_missing_fairness_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for seed in (42, 2025, 2026):
                self._write_run(root, "baseline", seed, "validation_reference", None, 1.0, 0.2)
                for index in range(5):
                    self._write_run(
                        root,
                        "fair_ncl",
                        seed,
                        "validation_audit",
                        f"c{index}",
                        0.995,
                        0.1,
                        include_fairness=not (seed == 2026 and index == 4),
                    )

            with self.assertRaisesRegex(ValueError, "missing fairness evidence"):
                build_validation_selection(root)

    def test_selection_rejects_jobs_missing_from_declared_audit_plan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            expected_jobs = []
            for seed in (42, 2025, 2026):
                self._write_run(root, "baseline", seed, "validation_reference", None, 1.0, 0.2)
                for index in range(5):
                    self._write_run(root, "fair_ncl", seed, "validation_audit", f"c{index}", 0.995, 0.1)
                    expected_jobs.append({
                        "job_id": f"validation_audit_fair_ncl_c{index}_{seed}",
                        "phase": "validation_audit",
                    })
            expected_jobs.append({"job_id": "missing_job", "phase": "validation_audit"})
            plan_path = root / "audit.jsonl"
            plan_path.write_text(
                "".join(json.dumps(job) + "\n" for job in expected_jobs),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing 1 expected jobs"):
                build_validation_selection(root, audit_jobs_path=plan_path)

    @staticmethod
    def _write_run(root, method, seed, phase, candidate, utility, gap, include_fairness=True):
        run_dir = root / "ml-1m" / f"{phase}_{method}_{candidate}_{seed}"
        run_dir.mkdir(parents=True)
        params = {"utility_alpha": 0.7, "utility_beta": 0.3}
        if candidate is not None:
            params["audit_candidate"] = candidate
        job = {
            "job_id": f"{phase}_{method}_{candidate}_{seed}" if candidate is not None else f"{phase}_{method}_{seed}",
            "phase": phase,
            "dataset": "ml-1m",
            "backbone": "sasrec",
            "method": method,
            "seed": seed,
            "params": params,
        }
        fairness = {}
        if include_fairness:
            fairness = {
                "gender": {
                    "ndcg@10_gap": gap,
                    "ndcg@10_worst_group": utility - gap,
                    "group_ndcg@10": {"0": utility, "1": utility - gap},
                },
                "age_group": {
                    "ndcg@10_gap": gap,
                    "ndcg@10_worst_group": utility - gap,
                    "group_ndcg@10": {"0": utility, "1": utility - gap},
                },
            }
        validation = {
            "utility": {"NDCG@10": utility},
            "fairness": fairness,
        }
        (run_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
        (run_dir / "validation_results.json").write_text(json.dumps(validation), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
