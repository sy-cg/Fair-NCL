import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


def _load_mce():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "methods" / "mce.py"
    spec = importlib.util.spec_from_file_location("mce_stability_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MCEStabilityTests(unittest.TestCase):
    def test_rank_deficient_embeddings_are_finite(self):
        mce = _load_mce()
        torch.manual_seed(0)

        base = torch.randn(128, 8)
        z = base.repeat(1, 8).requires_grad_(True)
        z_aug = (
            base + 1e-5 * torch.randn_like(base)
        ).repeat(1, 8).requires_grad_(True)

        losses = mce.mce_non_contrastive_losses(z, z_aug)
        loss = losses["mce_uniform_loss"] + losses["mce_align_loss"]

        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all().item())
        self.assertTrue(torch.isfinite(z_aug.grad).all().item())

    def test_eigh_solver_failure_uses_cpu_float64_fallback(self):
        mce = _load_mce()
        torch.manual_seed(1)

        matrix = torch.randn(16, 16)
        matrix = (matrix + matrix.T) / 2.0
        matrix.requires_grad_(True)

        original_eigh = torch.linalg.eigh
        calls = {"count": 0}

        def flaky_eigh(value, *args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 1:
                raise torch._C._LinAlgError("forced test failure")
            return original_eigh(value, *args, **kwargs)

        with patch.object(torch.linalg, "eigh", side_effect=flaky_eigh):
            log_matrix, stabilized = mce._principal_log_psd(matrix, eps=1e-4)
            loss = log_matrix.square().mean() + stabilized.square().mean()

        self.assertGreaterEqual(calls["count"], 2)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(torch.isfinite(matrix.grad).all().item())

    def test_non_finite_matrix_is_not_silently_repaired(self):
        mce = _load_mce()
        matrix = torch.eye(4)
        matrix[0, 0] = float("nan")

        with self.assertRaises(FloatingPointError):
            mce._robust_eigh(matrix, eps=1e-4)


if __name__ == "__main__":
    unittest.main()
