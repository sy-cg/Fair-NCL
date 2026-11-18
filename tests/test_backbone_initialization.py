import json
import tempfile
import unittest
from pathlib import Path

from experiments.backbone_initialization import (
    enforce_frozen_backbone_eval,
    initialize_frozen_backbone,
)


class _FakeBackbone:
    def __init__(self):
        self.loaded_state = None
        self.strict = None
        self.eval_called = False

    def load_state_dict(self, state, strict=True):
        self.loaded_state = dict(state)
        self.strict = strict

    def eval(self):
        self.eval_called = True


class _FakeModel:
    def __init__(self, freeze_backbone=True):
        self.freeze_backbone = freeze_backbone
        self.backbone = _FakeBackbone()


class FrozenBackboneInitializationTests(unittest.TestCase):
    def test_frozen_afrl_loads_matched_validation_reference_backbone(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_job = self._write_reference(root, seed=42)
            checkpoint = {
                "job": source_job,
                "model_state_dict": {
                    "backbone.item_emb.weight": "item-weights",
                    "backbone.encoder.weight": "encoder-weights",
                },
            }
            model = _FakeModel()

            record = initialize_frozen_backbone(
                model,
                self._target_job(method="afrl", seed=42),
                root,
                checkpoint_loader=lambda _: checkpoint,
            )

        self.assertEqual(
            model.backbone.loaded_state,
            {
                "item_emb.weight": "item-weights",
                "encoder.weight": "encoder-weights",
            },
        )
        self.assertTrue(model.backbone.strict)
        self.assertEqual(record["source_job_id"], source_job["job_id"])
        self.assertEqual(record["source_phase"], "validation_reference")

    def test_frozen_pfrec_fails_if_matched_reference_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "matched validation_reference"):
                initialize_frozen_backbone(
                    _FakeModel(),
                    self._target_job(method="pfrec", seed=2025),
                    Path(directory),
                    checkpoint_loader=lambda _: {},
                )

    def test_unfrozen_model_does_not_require_reference_checkpoint(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            record = initialize_frozen_backbone(
                _FakeModel(freeze_backbone=False),
                self._target_job(method="afrl", seed=42),
                Path(directory),
                checkpoint_loader=lambda path: calls.append(path),
            )

        self.assertIsNone(record)
        self.assertEqual(calls, [])

    def test_frozen_backbone_is_kept_in_evaluation_mode(self):
        model = _FakeModel(freeze_backbone=True)

        enforce_frozen_backbone_eval(model)

        self.assertTrue(model.backbone.eval_called)

    @staticmethod
    def _target_job(method, seed):
        return {
            "job_id": f"validation_audit_ml-1m_{method}_caser_s{seed}",
            "phase": "validation_audit",
            "dataset": "ml-1m",
            "method": method,
            "backbone": "caser",
            "seed": seed,
            "params": {},
        }

    @staticmethod
    def _write_reference(root, seed):
        source_job = {
            "job_id": f"validation_reference_ml-1m_baseline_caser_s{seed}",
            "phase": "validation_reference",
            "dataset": "ml-1m",
            "method": "baseline",
            "backbone": "caser",
            "seed": seed,
            "params": {},
        }
        source_dir = root / "ml-1m" / source_job["job_id"]
        source_dir.mkdir(parents=True)
        (source_dir / "job.json").write_text(json.dumps(source_job), encoding="utf-8")
        (source_dir / "best.pt").write_bytes(b"checkpoint-placeholder")
        return source_job


if __name__ == "__main__":
    unittest.main()
