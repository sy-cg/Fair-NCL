import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.provenance import collect_run_provenance, write_run_provenance


class ProvenanceTests(unittest.TestCase):
    def test_provenance_records_environment_and_dataset_scale(self):
        config = SimpleNamespace(dataset="ml-1m", device="cpu", seed=42)
        data = {
            "num_users": 10,
            "num_items": 20,
            "train_data": [1, 2, 3],
            "val_data": [1],
            "test_data": [1],
            "metadata": {"sequence_semantics": "true_timestamp_sequence"},
        }

        record = collect_run_provenance(config, data, workspace=Path.cwd())

        self.assertIn("python_version", record["environment"])
        self.assertIn("torch_version", record["environment"])
        self.assertIn("git_commit", record["code"])
        self.assertEqual(record["dataset"]["num_train_samples"], 3)
        self.assertEqual(record["dataset"]["num_users"], 10)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run_provenance.json"
            write_run_provenance(record, output)
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
