import json
import tempfile
import unittest
from pathlib import Path

from experiments.plan_status import is_job_done, summarize_plan


class PlanStatusTests(unittest.TestCase):
    @staticmethod
    def _write_complete_run(root, job, *, validation_only=False):
        run_dir = root / "results" / job["dataset"] / job["job_id"]
        run_dir.mkdir(parents=True)
        (run_dir / "validation_results.json").write_text(
            json.dumps({"selected": {"HitRate@10": 0.1}}),
            encoding="utf-8",
        )
        (run_dir / "best.pt").write_bytes(b"checkpoint")
        result = (
            {"status": "not_run_by_protocol", "validation_only": True}
            if validation_only else {"selected": {"HitRate@10": 0.1}}
        )
        (run_dir / "test_results.json").write_text(json.dumps(result), encoding="utf-8")

    def test_summarize_plan_reports_done_and_todo_indices(self):
        jobs = [
            {
                "job_id": "job-a",
                "phase": "validation_audit",
                "dataset": "ml-1m",
                "params": {"evaluate_test": False},
            },
            {
                "job_id": "job-b",
                "phase": "validation_audit",
                "dataset": "ml-1m",
                "params": {"evaluate_test": False},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "jobs.jsonl"
            plan.write_text(
                "".join(json.dumps(job) + "\n" for job in jobs),
                encoding="utf-8",
            )
            self._write_complete_run(root, jobs[0], validation_only=True)

            summary = summarize_plan(str(plan), str(root / "results"))

        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["done"], 1)
        self.assertEqual(summary["todo_indices"], [1])
        self.assertEqual(summary["phase"], "validation_audit")

    def test_is_job_done_uses_indexed_job_result_path(self):
        jobs = [
            {"job_id": "job-a", "phase": "comparison", "dataset": "taobao"},
            {"job_id": "job-b", "phase": "comparison", "dataset": "taobao"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "jobs.jsonl"
            plan.write_text(
                "".join(json.dumps(job) + "\n" for job in jobs),
                encoding="utf-8",
            )
            self._write_complete_run(root, jobs[1])

            self.assertFalse(is_job_done(str(plan), str(root / "results"), 0))
            self.assertTrue(is_job_done(str(plan), str(root / "results"), 1))

    def test_failed_or_partial_result_is_not_complete(self):
        job = {
            "job_id": "job-failed",
            "phase": "supplementary",
            "dataset": "ml-1m",
            "params": {"evaluate_test": True},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "jobs.jsonl"
            plan.write_text(json.dumps(job) + "\n", encoding="utf-8")
            run_dir = root / "results" / "ml-1m" / "job-failed"
            run_dir.mkdir(parents=True)
            (run_dir / "test_results.json").write_text(
                json.dumps({"status": "failed"}),
                encoding="utf-8",
            )

            self.assertFalse(is_job_done(str(plan), str(root / "results"), 0))


if __name__ == "__main__":
    unittest.main()
