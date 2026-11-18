import unittest
import importlib.util
from pathlib import Path


def _load_training_safety():
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "training_safety_under_test",
        repo_root / "methods" / "training_safety.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


safe_backward_step = _load_training_safety().safe_backward_step


class FakeLoss:
    def __init__(self, finite=True):
        self.finite = finite
        self.backward_called = False

    def backward(self):
        self.backward_called = True


class FakeScaler:
    def __init__(self):
        self.scale_called = False
        self.unscale_called = False
        self.step_called = False
        self.update_called = False

    def scale(self, loss):
        self.scale_called = True
        return loss

    def unscale_(self, optimizer):
        self.unscale_called = True

    def step(self, optimizer):
        self.step_called = True
        optimizer.step()

    def update(self):
        self.update_called = True


class FakeOptimizer:
    def __init__(self):
        self.zero_called = False
        self.step_called = False

    def zero_grad(self, set_to_none=True):
        self.zero_called = True

    def step(self):
        self.step_called = True


class FakeValue:
    def __init__(self, finite=True, value=1.0):
        self.finite = finite
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def __float__(self):
        return float(self.value)


class TrainingSafetyTests(unittest.TestCase):
    def test_nonfinite_loss_skips_backward_and_optimizer_step(self):
        optimizer = FakeOptimizer()
        scaler = FakeScaler()
        loss = FakeLoss(finite=False)

        state = safe_backward_step(
            [],
            optimizer,
            scaler,
            loss,
            max_norm=5.0,
            is_finite_fn=lambda value: value.finite,
            clip_grad_norm_fn=lambda params, max_norm: FakeValue(),
        )

        self.assertFalse(state["stepped"])
        self.assertEqual(state["reason"], "nonfinite_loss")
        self.assertFalse(loss.backward_called)
        self.assertFalse(scaler.scale_called)
        self.assertFalse(scaler.step_called)
        self.assertTrue(optimizer.zero_called)

    def test_nonfinite_gradient_norm_skips_optimizer_step_after_backward(self):
        optimizer = FakeOptimizer()
        scaler = FakeScaler()
        loss = FakeLoss(finite=True)

        state = safe_backward_step(
            [],
            optimizer,
            scaler,
            loss,
            max_norm=5.0,
            is_finite_fn=lambda value: value.finite,
            clip_grad_norm_fn=lambda params, max_norm: FakeValue(finite=False),
        )

        self.assertFalse(state["stepped"])
        self.assertEqual(state["reason"], "nonfinite_grad_norm")
        self.assertTrue(loss.backward_called)
        self.assertTrue(scaler.unscale_called)
        self.assertFalse(scaler.step_called)
        self.assertTrue(scaler.update_called)
        self.assertTrue(optimizer.zero_called)

    def test_finite_gradient_norm_steps_optimizer(self):
        optimizer = FakeOptimizer()
        scaler = FakeScaler()
        loss = FakeLoss(finite=True)

        state = safe_backward_step(
            [],
            optimizer,
            scaler,
            loss,
            max_norm=5.0,
            is_finite_fn=lambda value: value.finite,
            clip_grad_norm_fn=lambda params, max_norm: FakeValue(finite=True, value=2.0),
        )

        self.assertTrue(state["stepped"])
        self.assertIsNone(state["reason"])
        self.assertEqual(state["grad_norm"], 2.0)
        self.assertTrue(loss.backward_called)
        self.assertTrue(scaler.step_called)
        self.assertTrue(optimizer.step_called)
        self.assertTrue(scaler.update_called)


if __name__ == "__main__":
    unittest.main()
