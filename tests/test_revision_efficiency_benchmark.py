import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "benchmark_revision_efficiency.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("benchmark_revision_efficiency", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RevisionEfficiencyBenchmarkTests(unittest.TestCase):
    def test_taobao_efficiency_summary_reads_run_artifacts_without_processed_pickle(self):
        benchmark = _load_script()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            run_dir = root / "results" / "taobao" / "job_a"
            run_dir.mkdir(parents=True)
            (run_dir / "job.json").write_text(
                json.dumps({
                    "job_id": "job_a",
                    "phase": "supplementary",
                    "dataset": "taobao",
                    "method": "fair_ncl",
                    "backbone": "sasrec",
                    "seed": 42,
                }),
                encoding="utf-8",
            )
            (run_dir / "test_results.json").write_text(
                json.dumps({
                    "runtime": {
                        "training_seconds": 12.5,
                        "test_evaluation_seconds": 1.5,
                        "parameter_count": 1234,
                        "peak_cuda_memory_bytes": 4096,
                    },
                    "utility": {"NDCG@10": 0.25},
                }),
                encoding="utf-8",
            )
            (run_dir / "resource_diagnostics.json").write_text(
                json.dumps({"resource_build_seconds": 0.75, "cache_bytes": 2048}),
                encoding="utf-8",
            )
            (run_dir / "run_provenance.json").write_text(
                json.dumps({
                    "dataset": {
                        "num_users": 10,
                        "num_items": 20,
                        "num_train_samples": 30,
                    },
                    "environment": {"gpu_name": "Test GPU"},
                }),
                encoding="utf-8",
            )
            manifest = root / "manifest.csv"
            manifest.write_text(
                "dataset,method,backbone,seed,result_dir\n"
                f"taobao,fair_ncl,sasrec,42,{run_dir.as_posix()}\n",
                encoding="utf-8",
            )
            output = root / "efficiency.csv"
            audit = root / "efficiency_audit.json"

            result = benchmark.benchmark_manifest(manifest, output, audit)

            self.assertEqual(result["processed_rows"], 1)
            self.assertEqual(len(result["error_rows"]), 0)
            with output.open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["dataset"], "taobao")
            self.assertEqual(rows[0]["job_id"], "job_a")
            self.assertEqual(rows[0]["training_seconds"], "12.5")
            self.assertEqual(rows[0]["resource_build_seconds"], "0.75")
            self.assertEqual(rows[0]["gpu_name"], "Test GPU")
            audit_doc = json.loads(audit.read_text(encoding="utf-8"))
            self.assertEqual(audit_doc["missing_rows"], [])
            self.assertEqual(audit_doc["error_rows"], [])


if __name__ == "__main__":
    unittest.main()



