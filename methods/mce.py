"""NCL-SR-style matrix cross-entropy losses.

The ICLR 2025 NCL-SR formulation applies matrix cross entropy to centered
cross-covariances. Numerical noise can make a finite-batch cross-covariance
slightly non-symmetric or indefinite, so this implementation explicitly
projects matrices onto the symmetric positive-definite cone before taking the
principal logarithm.

Some CUDA eigensolver backends can still fail to converge for finite but
severely ill-conditioned matrices (for example, matrices with many clustered
or repeated eigenvalues). The normal path below is intentionally unchanged for
reproducibility. Only when ``torch.linalg.eigh`` actually fails do we retry the
same matrix on CPU in float64; a tiny deterministic diagonal regularizer is
used only as the final fallback.

NaN/Inf inputs are checked BEFORE eigendecomposition because some backends may
return non-finite eigenpairs instead of raising an exception.
"""

from __future__ import annotations

import warnings
from typing import Dict


MCE_NUMERICS_VERSION = "2026-09-05-v5"
_EIGH_FALLBACK_WARNED = False


def centered_covariance(left, right):
    if left.dim() != 2 or right.dim() != 2 or left.shape != right.shape:
        raise ValueError("MCE inputs must have the same [batch, dimension] shape")
    if left.size(0) <= 1:
        raise ValueError("MCE requires a batch containing at least two samples")
    left_centered = left - left.mean(dim=0, keepdim=True)
    right_centered = right - right.mean(dim=0, keepdim=True)
    return left_centered.transpose(0, 1) @ right_centered / float(left.size(0))


def matrix_cross_entropy(reference, prediction, eps: float = 1e-4):
    """Compute tr(-U log(V) + V) using a stable principal matrix log."""
    if reference.dim() != 2 or prediction.dim() != 2:
        raise ValueError("matrix cross entropy expects two matrices")
    if reference.shape != prediction.shape or reference.size(0) != reference.size(1):
        raise ValueError("matrix cross entropy expects equally sized square matrices")
    if eps <= 0:
        raise ValueError("eps must be strictly positive")
    compute_dtype = _stable_matrix_dtype(reference, prediction)
    reference = _to_dtype(reference, compute_dtype)
    prediction = _to_dtype(prediction, compute_dtype)
    log_prediction, stabilized_prediction = _principal_log_psd(prediction, eps)
    return -_trace(reference @ log_prediction) + _trace(stabilized_prediction)


def mce_non_contrastive_losses(
    z,
    z_aug,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> Dict[str, object]:
    """Return NCL-SR uniformity and alignment losses for two user views."""
    if eps <= 0:
        raise ValueError("eps must be strictly positive")

    with _autocast_disabled_for(z):
        z_normalized, z_aug_normalized = _normalize_pair_for_mce(z, z_aug, eps)
        cross_covariance = centered_covariance(z_normalized, z_aug_normalized)
        self_covariance = centered_covariance(z_normalized, z_normalized)
        augmented_covariance = centered_covariance(
            z_aug_normalized,
            z_aug_normalized,
        )

        dimension = int(z_normalized.size(1))
        identity = _identity(dimension, z_normalized) / float(dimension)
        uniform_loss = matrix_cross_entropy(
            identity,
            cross_covariance,
            eps=eps,
        )
        align_loss = -_trace(cross_covariance)
        align_loss = align_loss + float(gamma) * matrix_cross_entropy(
            self_covariance,
            augmented_covariance,
            eps=eps,
        )
    return {
        "mce_uniform_loss": uniform_loss,
        "mce_align_loss": align_loss,
    }


def _principal_log_psd(matrix, eps: float):
    matrix = _to_dtype(matrix, _stable_matrix_dtype(matrix))
    symmetric = 0.5 * (matrix + matrix.transpose(0, 1))
    eigenvalues, eigenvectors = _robust_eigh(symmetric, eps=eps)

    stabilized_values = eigenvalues.clamp_min(float(eps))
    stabilized = (
        eigenvectors * stabilized_values.unsqueeze(0)
    ) @ eigenvectors.transpose(0, 1)
    logarithm = (
        eigenvectors * stabilized_values.log().unsqueeze(0)
    ) @ eigenvectors.transpose(0, 1)
    return logarithm, stabilized


def _robust_eigh(symmetric, eps: float):
    """Eigendecompose a symmetric matrix with finite checks and fallbacks.

    Fast path:
        Use the original device/dtype.

    Recovery path for finite matrices:
        1. Retry the exact same matrix on CPU float64.
        2. If CPU float64 also fails, add a tiny deterministic diagonal
           regularizer and retry once.

    NaN/Inf inputs are rejected before the first eigensolver call.
    """
    import torch

    _raise_if_non_finite(symmetric, location="before eigendecomposition")

    try:
        eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
        if _eigh_outputs_are_finite(eigenvalues, eigenvectors):
            return eigenvalues, eigenvectors
        first_error = RuntimeError(
            "torch.linalg.eigh returned non-finite eigenpairs "
            "for a finite MCE covariance matrix"
        )
    except RuntimeError as error:
        first_error = error

    _warn_eigh_fallback_once(first_error)

    original_device = symmetric.device
    original_dtype = symmetric.dtype
    dimension = int(symmetric.size(0))

    symmetric_cpu = symmetric.to(device="cpu", dtype=torch.float64)

    try:
        eigenvalues_cpu, eigenvectors_cpu = torch.linalg.eigh(symmetric_cpu)
        if not _eigh_outputs_are_finite(eigenvalues_cpu, eigenvectors_cpu):
            raise RuntimeError(
                "CPU float64 torch.linalg.eigh returned non-finite eigenpairs"
            )
    except RuntimeError:
        machine_floor = float(torch.finfo(torch.float64).eps) * max(1, dimension)
        jitter = max(float(eps) * 0.1, machine_floor)
        diagonal_scale = torch.linspace(
            1.0,
            2.0,
            steps=dimension,
            dtype=torch.float64,
            device="cpu",
        )
        regularizer = torch.diag(diagonal_scale * jitter)

        try:
            eigenvalues_cpu, eigenvectors_cpu = torch.linalg.eigh(
                symmetric_cpu + regularizer
            )
            if not _eigh_outputs_are_finite(eigenvalues_cpu, eigenvectors_cpu):
                raise RuntimeError(
                    "regularized CPU float64 torch.linalg.eigh "
                    "returned non-finite eigenpairs"
                )
        except RuntimeError as final_error:
            raise RuntimeError(
                "MCE symmetric eigendecomposition failed on the original "
                "device and on CPU float64, including the regularized "
                "fallback. The matrix is finite but numerically pathological."
            ) from final_error

    return (
        eigenvalues_cpu.to(device=original_device, dtype=original_dtype),
        eigenvectors_cpu.to(device=original_device, dtype=original_dtype),
    )


def _raise_if_non_finite(matrix, location: str, subject: str = "MCE covariance"):
    import torch

    finite_mask = torch.isfinite(matrix)
    if bool(finite_mask.all().item()):
        return

    non_finite_count = int((~finite_mask).sum().item())
    raise FloatingPointError(
        f"{subject} contains non-finite values "
        f"{location} ({non_finite_count} NaN/Inf entries). "
        "This indicates upstream numerical instability and should not be "
        "masked by an eigensolver fallback."
    )


def _eigh_outputs_are_finite(eigenvalues, eigenvectors):
    import torch

    return bool(
        torch.isfinite(eigenvalues).all().item()
        and torch.isfinite(eigenvectors).all().item()
    )


def _warn_eigh_fallback_once(error):
    global _EIGH_FALLBACK_WARNED
    if _EIGH_FALLBACK_WARNED:
        return

    warnings.warn(
        "MCE torch.linalg.eigh failed to converge on the current device; "
        "retrying this exceptional batch on CPU float64. "
        f"Original error: {error}",
        RuntimeWarning,
        stacklevel=3,
    )
    _EIGH_FALLBACK_WARNED = True


def _trace(matrix):
    import torch

    return torch.trace(matrix)


def _identity(dimension: int, like):
    import torch

    return torch.eye(dimension, dtype=like.dtype, device=like.device)


def _normalize_pair_for_mce(z, z_aug, eps: float):
    import torch.nn.functional as functional

    compute_dtype = _stable_matrix_dtype(z, z_aug)
    z = _to_dtype(z, compute_dtype)
    z_aug = _to_dtype(z_aug, compute_dtype)
    _raise_if_non_finite(
        z,
        location="before normalization",
        subject="MCE input z",
    )
    _raise_if_non_finite(
        z_aug,
        location="before normalization",
        subject="MCE input z_aug",
    )
    return (
        functional.normalize(z, dim=1, eps=float(eps)),
        functional.normalize(z_aug, dim=1, eps=float(eps)),
    )


def _autocast_disabled_for(tensor):
    import contextlib
    import torch

    device_type = getattr(getattr(tensor, "device", None), "type", "cuda")
    amp_autocast = getattr(getattr(torch, "amp", None), "autocast", None)
    if amp_autocast is not None:
        return amp_autocast(device_type=device_type, enabled=False)

    cuda_amp_autocast = getattr(
        getattr(getattr(torch, "cuda", None), "amp", None),
        "autocast",
        None,
    )
    if device_type == "cuda" and cuda_amp_autocast is not None:
        return cuda_amp_autocast(enabled=False)

    return contextlib.nullcontext()


def _stable_matrix_dtype(*matrices):
    import torch

    dtype = matrices[0].dtype
    for matrix in matrices[1:]:
        dtype = torch.promote_types(dtype, matrix.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _to_dtype(matrix, dtype):
    if matrix.dtype == dtype:
        return matrix
    return matrix.to(dtype=dtype)
