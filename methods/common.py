from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def active_sensitive_attributes(config) -> List[str]:
    """Return attributes optimized during training.

    The project treats gender-age as an intersectional reporting group, not as
    an additional independent sensitive factor.
    """
    return list(getattr(config, "sensitive_attributes", ["gender", "age_group"]))


def reporting_sensitive_attributes(config) -> List[str]:
    return list(getattr(config, "report_sensitive_attributes", active_sensitive_attributes(config)))


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = _tensor_to_device(value, device) if isinstance(value, torch.Tensor) else value
    return moved


def _tensor_to_device(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    if value.device.type == device.type:
        if device.type != "cuda" or device.index is None or value.device.index == device.index:
            return value
    return value.to(device, non_blocking=True)


def last_sequence_representation(backbone: nn.Module, input_seq: torch.Tensor) -> torch.Tensor:
    """Extract a single user representation from any supported sequential backbone."""
    seq_output = backbone(input_seq)
    if isinstance(seq_output, tuple):
        seq_output = seq_output[0]

    if seq_output.dim() == 2:
        return seq_output

    if seq_output.dim() != 3:
        raise ValueError(f"Expected 2D or 3D sequence output, got {tuple(seq_output.shape)}")

    lengths = (input_seq > 0).sum(dim=1)
    batch_idx = torch.arange(input_seq.size(0), device=input_seq.device)
    last_idx = torch.clamp(lengths - 1, min=0)
    return seq_output[batch_idx, last_idx]


def score_items(backbone: nn.Module,
                user_repr: torch.Tensor,
                candidate_items: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Score items using the same head for full and sampled ranking.

    BERT4Rec uses an output projection head, while SASRec/GRU4Rec/Caser score
    by dot product with item embeddings.
    """
    if hasattr(backbone, "out"):
        full_logits = backbone.out(user_repr)
        if candidate_items is None:
            return full_logits
        if candidate_items.dim() == 1:
            candidate_items = candidate_items.unsqueeze(1)
        return torch.gather(full_logits, 1, candidate_items)

    item_emb = _item_embedding(backbone)
    if candidate_items is None:
        return torch.matmul(user_repr, item_emb.weight.transpose(0, 1))

    if candidate_items.dim() == 1:
        candidate_items = candidate_items.unsqueeze(1)
    cand_emb = item_emb(candidate_items)
    return torch.matmul(user_repr.unsqueeze(1), cand_emb.transpose(-1, -2)).squeeze(1)


def _item_embedding(backbone: nn.Module) -> nn.Embedding:
    if hasattr(backbone, "item_emb"):
        return backbone.item_emb
    if hasattr(backbone, "model") and hasattr(backbone.model, "item_emb"):
        return backbone.model.item_emb
    raise AttributeError("Backbone does not expose item_emb; cannot score candidates.")


def recommendation_ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.dim() > 1:
        target = target.squeeze(-1)
    return F.cross_entropy(logits, target.long(), ignore_index=0)


def variance_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))


def covariance_loss(z: torch.Tensor) -> torch.Tensor:
    if z.size(0) <= 1:
        return z.new_tensor(0.0)
    z = z - z.mean(dim=0, keepdim=True)
    cov = (z.T @ z) / (z.size(0) - 1)
    off_diag = cov.flatten()[:-1].view(cov.size(0) - 1, cov.size(0) + 1)[:, 1:].flatten()
    return (off_diag ** 2).sum() / z.size(1)


def non_contrastive_losses(z: torch.Tensor, z_aug: torch.Tensor) -> Dict[str, torch.Tensor]:
    z_n = F.normalize(z, dim=1)
    z_aug_n = F.normalize(z_aug, dim=1)
    return {
        "align_loss": F.mse_loss(z_n, z_aug_n),
        "var_loss": 0.5 * (variance_loss(z_n) + variance_loss(z_aug_n)),
        "cov_loss": covariance_loss(z_n) + covariance_loss(z_aug_n),
    }


def collect_sensitive_labels(batch: Dict[str, torch.Tensor],
                             attrs: Iterable[str]) -> Dict[str, torch.Tensor]:
    return {attr: batch[attr].long() for attr in attrs if attr in batch}
