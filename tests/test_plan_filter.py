import tempfile
import unittest
from pathlib import Path

from experiments.filter_plan import filter_jobs, load_plan, save_plan


class PlanFilterTests(unittest.TestCase):
    def test_filter_preserves_original_jobs_and_order(self):
        jobs = [
            {"method": "fair_ncl", "dataset": "ml-1m", "backbone": "sasrec", "seed": 42},
            {"method": "baseline", "dataset": "ml-1m", "backbone": "sasrec", "seed": 42},
            {"method": "baseline", "dataset": "taobao", "backbone": "caser", "seed": 2028},
        ]
        selected = filter_jobs(jobs, methods=["baseline"], seeds=[42])
        self.assertEqual(selected, [jobs[1]])

    def test_plan_round_trip_keeps_json_rows(self):
        jobs = [{"method": "baseline", "dataset": "ml-1m", "backbone": "sasrec", "seed": 42}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.jsonl"
            self.assertEqual(save_plan(jobs, str(path)), 1)
            self.assertEqual(load_plan(str(path)), jobs)


if __name__ == "__main__":
    unittest.main()
