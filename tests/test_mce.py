import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def _load_mce_module():
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "mce_under_test",
        repo_root / "methods" / "mce.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeMatrix:
    def __init__(self, dtype, shape=(2, 2)):
        self.dtype = dtype
        self.shape = shape

    def dim(self):
        return len(self.shape)

    def size(self, index):
        return self.shape[index]

    def to(self, dtype):
        return FakeMatrix(dtype, self.shape)

    def __matmul__(self, other):
        if self.dtype != other.dtype:
            raise AssertionError(f"matmul dtype mismatch: {self.dtype} vs {other.dtype}")
        return FakeMatrix(self.dtype, self.shape)

    def __truediv__(self, other):
        return FakeMatrix(self.dtype, self.shape)


class MCENumericsTests(unittest.TestCase):
    def test_matrix_cross_entropy_promotes_half_inputs_before_principal_log(self):
        mce = _load_mce_module()
        fake_torch = types.SimpleNamespace(
            float16="float16",
            bfloat16="bfloat16",
            float32="float32",
            promote_types=lambda left, right: left if left == right else "float32",
        )
        log_input_dtypes = []

        def fake_principal_log(prediction, eps):
            log_input_dtypes.append(prediction.dtype)
            return FakeMatrix(prediction.dtype), FakeMatrix(prediction.dtype)

        with patch.dict(sys.modules, {"torch": fake_torch}):
            with patch.object(mce, "_principal_log_psd", side_effect=fake_principal_log):
                with patch.object(mce, "_trace", return_value=1.0):
                    result = mce.matrix_cross_entropy(
                        FakeMatrix("float16"),
                        FakeMatrix("float16"),
                    )

        self.assertEqual(log_input_dtypes, ["float32"])
        self.assertEqual(result, 0.0)

    def test_mce_normalizes_half_embeddings_in_stable_precision(self):
        mce = _load_mce_module()
        fake_torch = types.ModuleType("torch")
        fake_torch.float16 = "float16"
        fake_torch.bfloat16 = "bfloat16"
        fake_torch.float32 = "float32"
        fake_torch.promote_types = lambda left, right: left if left == right else "float32"
        fake_torch_nn = types.ModuleType("torch.nn")
        fake_functional = types.ModuleType("torch.nn.functional")
        normalize_calls = []

        def fake_normalize(value, dim, eps=None):
            normalize_calls.append((value.dtype, dim, eps))
            return value

        fake_functional.normalize = fake_normalize
        fake_torch_nn.functional = fake_functional
        fake_torch.nn = fake_torch_nn

        with patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "torch.nn": fake_torch_nn,
                "torch.nn.functional": fake_functional,
            },
        ):
            with patch.object(mce, "_raise_if_non_finite"):
                with patch.object(mce, "centered_covariance", return_value=FakeMatrix("float32")):
                    with patch.object(mce, "_identity", return_value=FakeMatrix("float32")):
                        with patch.object(mce, "matrix_cross_entropy", return_value=1.0):
                            with patch.object(mce, "_trace", return_value=0.5):
                                mce.mce_non_contrastive_losses(
                                    FakeMatrix("float16"),
                                    FakeMatrix("float16"),
                                    eps=1e-4,
                                )

        self.assertEqual(
            normalize_calls,
            [("float32", 1, 1e-4), ("float32", 1, 1e-4)],
        )

    def test_mce_rejects_non_finite_embeddings_before_covariance(self):
        mce = _load_mce_module()
        fake_torch = types.ModuleType("torch")
        fake_torch.float16 = "float16"
        fake_torch.bfloat16 = "bfloat16"
        fake_torch.float32 = "float32"
        fake_torch.promote_types = lambda left, right: left if left == right else "float32"
        fake_torch_nn = types.ModuleType("torch.nn")
        fake_functional = types.ModuleType("torch.nn.functional")
        fake_functional.normalize = lambda value, dim, eps=None: value
        fake_torch_nn.functional = fake_functional
        fake_torch.nn = fake_torch_nn

        with patch.dict(
            sys.modules,
            {
                "torch": fake_torch,
                "torch.nn": fake_torch_nn,
                "torch.nn.functional": fake_functional,
            },
        ):
            with patch.object(
                mce,
                "_raise_if_non_finite",
                side_effect=FloatingPointError("bad MCE input"),
            ) as guard:
                with self.assertRaisesRegex(FloatingPointError, "bad MCE input"):
                    mce._normalize_pair_for_mce(
                        FakeMatrix("float32"),
                        FakeMatrix("float32"),
                        eps=1e-4,
                    )

        guard.assert_called_once()
        self.assertEqual(guard.call_args.kwargs["subject"], "MCE input z")


if __name__ == "__main__":
    unittest.main()
