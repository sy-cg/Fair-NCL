import json
from pathlib import Path
import tempfile
import unittest

from analysis.report_revision_completion import _expected_seed_count, _holm_adjust, _plan_backbones
from analysis.skew_diagnostics import _spearman
from experiments.revision_completion import (
    LEAN_SENSITIVITY_AXES,
    _fair_jobs,
    coserec_tune_jobs,
    factorial_jobs,
    finalize_coserec_jobs,
    finalize_jobs,
    mce_tune_jobs,
    random_pool_jobs,
    _select_validation,
    selection_manifest,
    sensitivity_jobs,
)


RESULTS_ROOT = Path("results_revision_kbs_v2")


class RevisionCompletionPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fair = _fair_jobs(RESULTS_ROOT)

    def test_completed_fair_reference_matrix_is_five_seed(self):
        self.assertEqual(len(self.fair), 40)
        self.assertEqual(
            {seed for _, _, seed in self.fair},
            {42, 2025, 2026, 2027, 2028},
        )

    def test_lean_reference_loading_requires_only_sasrec(self):
        fair = _fair_jobs(RESULTS_ROOT, required_backbones=("sasrec",))
        self.assertEqual(len(fair), 10)
        self.assertEqual({backbone for _, backbone, _ in fair}, {"sasrec"})
        self.assertEqual({dataset for dataset, _, _ in fair}, {"ml-1m", "taobao"})

    def test_lean_reference_can_load_from_plan_without_result_artifacts(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            plan = root / "reference.jsonl"
            jobs = [
                job for (dataset, backbone, seed), job in self.fair.items()
                if backbone == "sasrec"
            ]
            plan.write_text(
                "".join(json.dumps(job) + "\n" for job in jobs),
                encoding="utf-8",
            )
            fair = _fair_jobs(
                root / "missing-results",
                required_backbones=("sasrec",),
                reference_plan=plan,
            )
            self.assertEqual(len(fair), 10)
            self.assertEqual({backbone for _, backbone, _ in fair}, {"sasrec"})

    def test_plan_sizes_and_no_rerun_methods(self):
        plans = (
            factorial_jobs(self.fair)
            + random_pool_jobs(self.fair)
            + sensitivity_jobs(self.fair)
            + mce_tune_jobs(self.fair, RESULTS_ROOT)
            + coserec_tune_jobs(self.fair)
        )
        self.assertEqual(len(plans), 898)
        self.assertEqual(len({job.job_id for job in plans}), len(plans))
        self.assertFalse({"ncl_sr_mce", "a_fsr"} & {job.method for job in plans})
        for job in plans:
            self.assertEqual(job.params.get("bias_metric"), "prior_jsd")
            self.assertEqual(job.params.get("bias_event_source"), "unique_train_events")

    def test_factorial_reuses_full_cell_and_runs_other_cells(self):
        jobs = factorial_jobs(self.fair)
        cells = {job.params["factorial_cell"] for job in jobs}
        self.assertEqual(len(jobs), 600)
        self.assertEqual(len(cells), 15)
        self.assertNotIn("1111", cells)
        self.assertIn("0000", cells)

    def test_coserec_is_sasrec_only_and_validation_only(self):
        jobs = coserec_tune_jobs(self.fair)
        self.assertEqual(len(jobs), 18)
        self.assertEqual({job.backbone for job in jobs}, {"sasrec"})
        self.assertTrue(all(job.params["evaluate_test"] is False for job in jobs))
        self.assertTrue(all(job.params["coserec_max_insert_num_per_pos"] == 1 for job in jobs))

    def test_lean_sensitivity_is_three_axes_three_seeds_on_sasrec(self):
        jobs = sensitivity_jobs(
            self.fair,
            seeds=(42, 2025, 2026),
            axes=LEAN_SENSITIVITY_AXES,
        )
        self.assertEqual(len(jobs), 36)
        self.assertEqual({job.backbone for job in jobs}, {"sasrec"})
        self.assertEqual({job.seed for job in jobs}, {42, 2025, 2026})
        self.assertEqual(
            {job.params["sensitivity_axis"] for job in jobs},
            set(LEAN_SENSITIVITY_AXES),
        )

    def test_holm_is_monotone_and_bounded(self):
        rows = [
            {"dataset": "d", "backbone": "b", "stage": "s", "metric": "m", "paired_p_exact": value}
            for value in (0.01, 0.03, 0.2)
        ]
        _holm_adjust(rows)
        adjusted = [row["holm_p"] for row in rows]
        self.assertEqual(adjusted, [0.03, 0.06, 0.2])

    def test_lean_sensitivity_reporting_accepts_three_seeds(self):
        self.assertEqual(_expected_seed_count("sensitivity"), 3)
        self.assertEqual(_expected_seed_count("coserec_final"), 5)

    def test_report_scope_is_inferred_from_plan_backbones(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            plan = Path(directory) / "lean.jsonl"
            plan.write_text(
                json.dumps({"backbone": "sasrec"}) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(_plan_backbones([plan]), ("sasrec",))

    def test_dependency_free_spearman(self):
        self.assertAlmostEqual(_spearman([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertAlmostEqual(_spearman([1, 2, 3], [3, 2, 1]), -1.0)
        self.assertIsNone(_spearman([1, 1, 1], [2, 3, 4]))

    def test_validation_selection_generates_only_final_test_jobs(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            tune_jobs = mce_tune_jobs(self.fair, root) + coserec_tune_jobs(self.fair)
            for job in tune_jobs:
                candidate = job.params["tune_candidate"]
                score = float(candidate.split("_")[-1])
                output = root / job.dataset / job.job_id
                output.mkdir(parents=True)
                (output / "validation_results.json").write_text(
                    json.dumps({"utility": {"HitRate@10": score}}),
                    encoding="utf-8",
                )
                (output / "test_results.json").write_text(
                    json.dumps({"status": "not_run_by_protocol", "validation_only": True}),
                    encoding="utf-8",
                )
                (output / "history.json").write_text(
                    json.dumps({"train_loss": [1.0], "skipped_nonfinite_updates": [0]}),
                    encoding="utf-8",
                )
                (output / "best.pt").write_bytes(b"checkpoint-placeholder")
            final = finalize_jobs(self.fair, root)
            self.assertEqual(len(final["mce_final"]), 40)
            self.assertEqual(len(final["coserec_final"]), 10)
            self.assertTrue(all(job.params.get("evaluate_test", True) for jobs in final.values() for job in jobs))
            manifest = selection_manifest(self.fair, root, final)
            self.assertEqual(len(manifest["validation_selection"]["mce"]), 8)
            self.assertEqual(len(manifest["validation_selection"]["coserec"]), 2)
            self.assertFalse(manifest["excluded_mce_pairs"])
            self.assertFalse(manifest["excluded_coserec_pairs"])

    def test_coserec_only_finalize_does_not_require_mce_tuning(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            tune_jobs = coserec_tune_jobs(self.fair)
            for job in tune_jobs:
                candidate = job.params["tune_candidate"]
                score = float(candidate.split("_")[-1])
                output = root / job.dataset / job.job_id
                output.mkdir(parents=True)
                (output / "validation_results.json").write_text(
                    json.dumps({"utility": {"HitRate@10": score}}),
                    encoding="utf-8",
                )
                (output / "test_results.json").write_text(
                    json.dumps({"status": "not_run_by_protocol", "validation_only": True}),
                    encoding="utf-8",
                )
                (output / "history.json").write_text(
                    json.dumps({"train_loss": [1.0], "skipped_nonfinite_updates": [0]}),
                    encoding="utf-8",
                )
                (output / "best.pt").write_bytes(b"checkpoint-placeholder")

            final = finalize_coserec_jobs(self.fair, root)
            self.assertEqual(len(final), 10)
            self.assertEqual({job.backbone for job in final}, {"sasrec"})
            self.assertTrue(all(job.params["evaluate_test"] for job in final))

    def test_validation_selection_excludes_zero_update_candidates(self):
        with tempfile.TemporaryDirectory(dir=".") as directory:
            root = Path(directory)
            tune_jobs = [
                job for job in mce_tune_jobs(self.fair, root)
                if job.dataset == "ml-1m" and job.backbone == "caser"
            ]
            for job in tune_jobs:
                output = root / job.dataset / job.job_id
                output.mkdir(parents=True)
                (output / "validation_results.json").write_text(
                    json.dumps({"utility": {"HitRate@10": 0.1}}),
                    encoding="utf-8",
                )
                (output / "test_results.json").write_text(
                    json.dumps({"status": "not_run_by_protocol", "validation_only": True}),
                    encoding="utf-8",
                )
                (output / "history.json").write_text(
                    json.dumps({"skipped_nonfinite_updates": [10]}),
                    encoding="utf-8",
                )
                (output / "best.pt").write_bytes(b"checkpoint-placeholder")
            selected = _select_validation(tune_jobs, root)
            self.assertIsNone(selected[("ml-1m", "caser")])


if __name__ == "__main__":
    unittest.main()
