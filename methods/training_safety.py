"""Numerical safety helpers for optimizer updates."""

from __future__ import annotations

from typing import Callable, Iterable, Optional


def safe_backward_step(
    parameters: Iterable,
    optimizer,
    scaler,
    loss,
    max_norm: float,
    *,
    is_finite_fn: Optional[Callable] = None,
    clip_grad_norm_fn: Optional[Callable] = None,
) -> dict:
    """Run backward, clip gradients, and step only when values are finite."""
    is_finite = is_finite_fn or _is_finite_tensor
    clip_grad_norm = clip_grad_norm_fn or _clip_grad_norm_finite
    params = list(parameters)

    if not is_finite(loss):
        optimizer.zero_grad(set_to_none=True)
        return {
            "stepped": False,
            "reason": "nonfinite_loss",
            "grad_norm": None,
        }

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = clip_grad_norm(params, max_norm)
    if not is_finite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        scaler.update()
        return {
            "stepped": False,
            "reason": "nonfinite_grad_norm",
            "grad_norm": _safe_float(grad_norm),
        }

    scaler.step(optimizer)
    scaler.update()
    return {
        "stepped": True,
        "reason": None,
        "grad_norm": _safe_float(grad_norm),
    }


def _clip_grad_norm_finite(parameters, max_norm: float):
    import torch

    grads = [param.grad for param in parameters if getattr(param, "grad", None) is not None]
    if not grads:
        return torch.tensor(0.0)

    first_grad = grads[0]
    total_sq = torch.zeros((), device=first_grad.device, dtype=torch.float64)
    for grad in grads:
        grad = grad.detach()
        if not torch.isfinite(grad).all():
            return grad.new_tensor(float("nan"))
        total_sq = total_sq + grad.to(dtype=torch.float64).pow(2).sum()

    total_norm = total_sq.sqrt()
    if not torch.isfinite(total_norm):
        return total_norm.to(device=first_grad.device)

    clip_coef = float(max_norm) / (total_norm + 1e-6)
    if bool((clip_coef < 1.0).item()):
        for grad in grads:
            grad.mul_(clip_coef.to(device=grad.device, dtype=grad.dtype))
    return total_norm.to(device=first_grad.device)


def _is_finite_tensor(value) -> bool:
    import torch

    return bool(torch.isfinite(value.detach()).all().item())


def _safe_float(value):
    if value is None:
        return None
    try:
        return float(value.detach().cpu())
    except Exception:
        try:
            return float(value)
        except Exception:
            return None
