import itertools
import random
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.afrl_components import (
    AFRL_CombineMLP,
    AFRL_Discriminator,
    AFRL_Generator,
    DebiasedCollaborativeEncoder,
)
from .augmentations import SequenceAugmenter
from .common import (
    active_sensitive_attributes,
    collect_sensitive_labels,
    covariance_loss,
    last_sequence_representation,
    non_contrastive_losses,
    recommendation_ce_loss,
    score_items,
    variance_loss,
)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.weight = weight
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.weight * grad_output, None


def grad_reverse(x: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, weight)


def make_mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class BackboneMethod(nn.Module):
    """Plain backbone wrapper with a stable method-level API."""

    def __init__(self, backbone: nn.Module, config, name: str = "baseline"):
        super().__init__()
        self.backbone = backbone
        self.config = config
        self.name = name
        self.hidden_units = config.hidden_units
        self.uses_custom_optimization = False

    def encode(self, input_seq: torch.Tensor) -> torch.Tensor:
        return last_sequence_representation(self.backbone, input_seq)

    def transform_representation(self, z: torch.Tensor, batch: Optional[Dict] = None) -> torch.Tensor:
        return z

    def predict(self, input_seq: torch.Tensor, candidate_items: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = self.transform_representation(self.encode(input_seq))
        return score_items(self.backbone, z, candidate_items)

    def compute_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        logits = self.predict(batch["input_seq"])
        rec_loss = recommendation_ce_loss(logits, batch["target"])
        return {"loss": rec_loss, "rec_loss": rec_loss}


class NCLMethod(BackboneMethod):
    """Non-contrastive learning wrapper and Fair-NCL ablation variants."""

    def __init__(self, backbone: nn.Module, config, augmenter: SequenceAugmenter, name: str):
        super().__init__(backbone, config, name=name)
        self.augmenter = augmenter
        self.aug_rec_weight = getattr(config, "fair_ncl_aug_rec_weight", 0.5)
        self.align_weight = getattr(config, "fair_ncl_align_weight", 1.0)
        self.var_weight = getattr(config, "fair_ncl_var_weight", 1.0)
        self.cov_weight = getattr(config, "fair_ncl_cov_weight", 0.04)

    def compute_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        input_seq = batch["input_seq"]
        target = batch["target"]
        aug_seq = self.augmenter.augment(input_seq)

        z = self.encode(input_seq)
        z_aug = self.encode(aug_seq)
        logits = score_items(self.backbone, z)
        logits_aug = score_items(self.backbone, z_aug)

        rec_loss = recommendation_ce_loss(logits, target)
        aug_rec_loss = recommendation_ce_loss(logits_aug, target)
        ncl_losses = non_contrastive_losses(z, z_aug)
        loss = (
            rec_loss
            + self.aug_rec_weight * aug_rec_loss
            + self.align_weight * ncl_losses["align_loss"]
            + self.var_weight * ncl_losses["var_loss"]
            + self.cov_weight * ncl_losses["cov_loss"]
        )
        return {
            "loss": loss,
            "rec_loss": rec_loss,
            "aug_rec_loss": aug_rec_loss,
            **ncl_losses,
        }


def _collect_optimizer_params(modules) -> List[torch.nn.Parameter]:
    params = []
    for module in modules:
        if module is None:
            continue
        for param in module.parameters():
            if param.requires_grad:
                params.append(param)
    return params


def _build_optimizer(modules, lr: float, weight_decay: float):
    params = _collect_optimizer_params(modules)
    if not params:
        return None
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def _optimizer_step(loss: torch.Tensor,
                    optimizer,
                    clip_val: float,
                    retain_graph: bool = False) -> None:
    if optimizer is None:
        return
    optimizer.zero_grad(set_to_none=True)
    loss.backward(retain_graph=retain_graph)
    params = [param for group in optimizer.param_groups for param in group["params"] if param.grad is not None]
    if params:
        torch.nn.utils.clip_grad_norm_(params, clip_val)
    optimizer.step()


def _mean_adv_loss(heads: nn.ModuleDict,
                   embeddings: torch.Tensor,
                   labels: Dict[str, torch.Tensor]) -> torch.Tensor:
    if not labels:
        return embeddings.new_tensor(0.0)
    losses = []
    for attr, target in labels.items():
        if attr in heads:
            losses.append(F.cross_entropy(heads[attr](embeddings), target.long()))
    if not losses:
        return embeddings.new_tensor(0.0)
    return torch.stack(losses).mean()


def _per_sample_ce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if target.dim() > 1:
        target = target.squeeze(-1)
    return F.cross_entropy(logits, target.long(), ignore_index=0, reduction="none")


class GRLMethod(BackboneMethod):
    """Canonical single-optimizer GRL baseline."""

    def __init__(self, backbone: nn.Module, config, name: str = "grl"):
        super().__init__(backbone, config, name=name)
        self.sensitive_attributes = active_sensitive_attributes(config)
        self.attribute_dims = config.attribute_dims
        self.adv_weight = getattr(config, "grl_weight", getattr(config, "adv_debias_weight", 0.2))
        self.adversaries = nn.ModuleDict({
            attr: make_mlp(self.hidden_units, self.hidden_units, self.attribute_dims[attr], config.dropout_rate)
            for attr in self.sensitive_attributes
        })

    def compute_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        z = self.encode(batch["input_seq"])
        logits = score_items(self.backbone, z)
        rec_loss = recommendation_ce_loss(logits, batch["target"])

        labels = collect_sensitive_labels(batch, self.sensitive_attributes)
        adv_loss = z.new_tensor(0.0)
        if labels:
            z_rev = grad_reverse(z, self.adv_weight)
            adv_loss = _mean_adv_loss(self.adversaries, z_rev, labels)

        loss = rec_loss + adv_loss
        return {"loss": loss, "rec_loss": rec_loss, "adv_loss": adv_loss}


class AdvDebiasMethod(BackboneMethod):
    """Alternating adversarial debiasing baseline.

    Unlike GRL, this baseline uses separate updates for the discriminator and
    the recommender, which makes it a cleaner approximation of standard
    adversarial debiasing pipelines.
    """

    def __init__(self, backbone: nn.Module, config, name: str = "adv_debias"):
        super().__init__(backbone, config, name=name)
        self.uses_custom_optimization = True
        self.sensitive_attributes = active_sensitive_attributes(config)
        self.attribute_dims = config.attribute_dims
        self.adv_weight = getattr(config, "adv_debias_weight", 0.2)
        self.disc_steps = int(getattr(config, "adv_disc_steps", 1))
        self.disc_lr = float(getattr(config, "adv_debias_disc_lr", config.learning_rate))
        self.weight_decay = float(getattr(config, "l2_emb", 1e-6))
        self.clip_val = float(getattr(config, "gradient_clip_val", 5.0))
        self.adversaries = nn.ModuleDict({
            attr: make_mlp(self.hidden_units, self.hidden_units, self.attribute_dims[attr], config.dropout_rate)
            for attr in self.sensitive_attributes
        })
        self.main_optimizer = None
        self.adv_optimizer = None

    def _ensure_optimizers(self):
        if self.main_optimizer is None:
            self.main_optimizer = _build_optimizer([self.backbone], self.config.learning_rate, self.weight_decay)
        if self.adv_optimizer is None:
            self.adv_optimizer = _build_optimizer([self.adversaries], self.disc_lr, self.weight_decay)

    def training_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        self._ensure_optimizers()
        labels = collect_sensitive_labels(batch, self.sensitive_attributes)

        disc_loss = self.backbone.item_emb.weight.new_tensor(0.0)
        if labels:
            for _ in range(self.disc_steps):
                z_detached = self.encode(batch["input_seq"]).detach()
                disc_loss = _mean_adv_loss(self.adversaries, z_detached, labels)
                _optimizer_step(disc_loss, self.adv_optimizer, self.clip_val)

        z = self.encode(batch["input_seq"])
        logits = score_items(self.backbone, z)
        rec_loss = recommendation_ce_loss(logits, batch["target"])
        confusion_loss = _mean_adv_loss(self.adversaries, z, labels) if labels else z.new_tensor(0.0)
        total_loss = rec_loss - self.adv_weight * confusion_loss
        _optimizer_step(total_loss, self.main_optimizer, self.clip_val)

        return {
            "loss": total_loss.detach(),
            "rec_loss": rec_loss.detach(),
            "disc_loss": disc_loss.detach() if isinstance(disc_loss, torch.Tensor) else z.new_tensor(float(disc_loss)),
            "adv_loss": confusion_loss.detach(),
        }


class SMPCFRMethod(BackboneMethod):
    """Selective-filter baseline inspired by SM/PCFR.

    The implementation keeps the core selective-fairness idea: a filter is
    trained for each sensitive-attribute combination, while a discriminator is
    updated only for the currently selected fairness requirement.
    """

    def __init__(self, backbone: nn.Module, config, name: str = "sm_pcfr"):
        super().__init__(backbone, config, name=name)
        self.uses_custom_optimization = True
        self.sensitive_attributes = active_sensitive_attributes(config)
        self.attribute_dims = config.attribute_dims
        self.adv_weight = getattr(config, "sm_lambda", 1.0)
        self.consistency_weight = getattr(config, "sm_consistency_weight", 0.1)
        self.disc_steps = int(getattr(config, "sm_disc_steps", 1))
        self.weight_decay = float(getattr(config, "l2_emb", 1e-6))
        self.clip_val = float(getattr(config, "gradient_clip_val", 5.0))

        self.filter_keys = self._build_filter_keys()
        self.filters = nn.ModuleDict({
            key: make_mlp(self.hidden_units, self.hidden_units * 2, self.hidden_units, config.dropout_rate)
            for key in self.filter_keys
        })
        self.adversaries = nn.ModuleDict({
            attr: make_mlp(self.hidden_units, self.hidden_units, self.attribute_dims[attr], config.dropout_rate)
            for attr in self.sensitive_attributes
        })
        self._mask_pointer = 0
        self.main_optimizer = None
        self.disc_optimizer = None

    def _ensure_optimizers(self):
        if self.main_optimizer is None:
            self.main_optimizer = _build_optimizer(
                [self.backbone, self.filters],
                self.config.learning_rate,
                self.weight_decay,
            )
        if self.disc_optimizer is None:
            self.disc_optimizer = _build_optimizer([self.adversaries], self.config.learning_rate, self.weight_decay)

    def _build_filter_keys(self) -> List[str]:
        keys = []
        for r in range(1, len(self.sensitive_attributes) + 1):
            for combo in itertools.combinations(self.sensitive_attributes, r):
                keys.append("__".join(combo))
        return keys

    def _next_mask(self) -> Dict[str, bool]:
        combo_key = self.filter_keys[self._mask_pointer % len(self.filter_keys)]
        self._mask_pointer += 1
        active = set(combo_key.split("__"))
        return {attr: attr in active for attr in self.sensitive_attributes}

    def _mask_key(self, mask: Dict[str, bool]) -> Optional[str]:
        active = [attr for attr in self.sensitive_attributes if mask.get(attr, False)]
        return "__".join(active) if active else None

    def _apply_filter(self, z: torch.Tensor, mask: Dict[str, bool]) -> torch.Tensor:
        key = self._mask_key(mask)
        return z if key is None else self.filters[key](z)

    def transform_representation(self, z: torch.Tensor, batch: Optional[Dict] = None) -> torch.Tensor:
        fair_mask = {attr: True for attr in self.sensitive_attributes}
        return self._apply_filter(z, fair_mask)

    def training_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        self._ensure_optimizers()
        mask = self._next_mask()
        active_attrs = [attr for attr, is_on in mask.items() if is_on and attr in batch]

        disc_loss = self.backbone.item_emb.weight.new_tensor(0.0)
        if active_attrs:
            for _ in range(self.disc_steps):
                z_detached = self.encode(batch["input_seq"]).detach()
                z_filtered_detached = self._apply_filter(z_detached, mask).detach()
                disc_losses = [
                    F.cross_entropy(self.adversaries[attr](z_filtered_detached), batch[attr].long())
                    for attr in active_attrs
                ]
                disc_loss = torch.stack(disc_losses).mean()
                _optimizer_step(disc_loss, self.disc_optimizer, self.clip_val)

        z = self.encode(batch["input_seq"])
        z_filtered = self._apply_filter(z, mask)
        logits = score_items(self.backbone, z_filtered)
        rec_loss = recommendation_ce_loss(logits, batch["target"])

        confusion_loss = z.new_tensor(0.0)
        if active_attrs:
            confusion_loss = torch.stack([
                F.cross_entropy(self.adversaries[attr](z_filtered), batch[attr].long())
                for attr in active_attrs
            ]).mean()
        consistency_loss = F.mse_loss(z_filtered, z.detach())
        total_loss = rec_loss - self.adv_weight * confusion_loss + self.consistency_weight * consistency_loss
        _optimizer_step(total_loss, self.main_optimizer, self.clip_val)

        return {
            "loss": total_loss.detach(),
            "rec_loss": rec_loss.detach(),
            "disc_loss": disc_loss.detach() if isinstance(disc_loss, torch.Tensor) else z.new_tensor(float(disc_loss)),
            "adv_loss": confusion_loss.detach(),
            "consistency_loss": consistency_loss.detach(),
        }


class AFRLMethod(BackboneMethod):
    """AFRL-style alternating information-alignment baseline."""

    def __init__(self, backbone: nn.Module, config, name: str = "afrl"):
        super().__init__(backbone, config, name=name)
        self.uses_custom_optimization = True
        self.sensitive_attributes = active_sensitive_attributes(config)
        self.attribute_dims = config.attribute_dims
        self.beta = getattr(config, "afrl_beta", 1.0)
        self.lambda_adv = getattr(config, "afrl_lambda", 1.0)
        self.recon_weight = getattr(config, "afrl_recon_weight", 0.01)
        self.disc_steps = int(getattr(config, "afrl_disc_steps", getattr(config, "disc_pretrain_steps", 5)))
        self.freeze_backbone = bool(getattr(config, "afrl_freeze_backbone", True))
        self.weight_decay = float(getattr(config, "l2_emb", 1e-6))
        self.clip_val = float(getattr(config, "gradient_clip_val", 5.0))
        self.rng = random.Random(getattr(config, "seed", 42))

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.debiased_generator = DebiasedCollaborativeEncoder(self.hidden_units, config.dropout_rate)
        self.attribute_generators = nn.ModuleDict({
            attr: AFRL_Generator(self.hidden_units, config.dropout_rate)
            for attr in self.sensitive_attributes
        })
        self.target_predictors = nn.ModuleDict({
            attr: nn.Linear(self.hidden_units, self.attribute_dims[attr])
            for attr in self.sensitive_attributes
        })
        self.fair_adversaries = nn.ModuleDict({
            attr: AFRL_Discriminator(self.hidden_units, self.attribute_dims[attr], config.dropout_rate)
            for attr in self.sensitive_attributes
        })
        self.combine = AFRL_CombineMLP(self.hidden_units, len(self.sensitive_attributes), config.dropout_rate)

        self.attr_optimizer = None
        self.debias_optimizer = None
        self.disc_optimizer = None
        self.combine_optimizer = None

    def _ensure_optimizers(self):
        if self.attr_optimizer is None:
            self.attr_optimizer = _build_optimizer(
                [self.attribute_generators, self.target_predictors],
                self.config.learning_rate,
                self.weight_decay,
            )
        if self.debias_optimizer is None:
            self.debias_optimizer = _build_optimizer(
                [self.debiased_generator],
                self.config.learning_rate,
                self.weight_decay,
            )
        if self.disc_optimizer is None:
            self.disc_optimizer = _build_optimizer(
                [self.fair_adversaries],
                self.config.learning_rate,
                self.weight_decay,
            )
        if self.combine_optimizer is None:
            self.combine_optimizer = _build_optimizer(
                [self.combine],
                self.config.learning_rate,
                self.weight_decay,
            )

    def _sample_mask(self) -> Dict[str, bool]:
        mask = {attr: self.rng.random() < 0.5 for attr in self.sensitive_attributes}
        if not any(mask.values()) and self.sensitive_attributes:
            mask[self.rng.choice(self.sensitive_attributes)] = True
        return mask

    def _compose(self,
                 z: torch.Tensor,
                 mask: Dict[str, bool],
                 detach_parts: bool = False) -> Dict[str, torch.Tensor]:
        z_debiased = self.debiased_generator(z)
        if detach_parts:
            z_debiased = z_debiased.detach()
        attr_reprs = {}
        combine_parts = [z_debiased]
        for attr in self.sensitive_attributes:
            z_attr = self.attribute_generators[attr](z)
            attr_reprs[attr] = z_attr
            cur = z_attr.detach() if detach_parts else z_attr
            combine_parts.append(torch.zeros_like(cur) if mask.get(attr, False) else cur)
        z_fair = self.combine.network(torch.cat(combine_parts, dim=1))
        return {
            "z_debiased": z_debiased,
            "attr_reprs": attr_reprs,
            "z_fair": z_fair,
        }

    def transform_representation(self, z: torch.Tensor, batch: Optional[Dict] = None) -> torch.Tensor:
        mask = {attr: True for attr in self.sensitive_attributes}
        return self._compose(z, mask, detach_parts=False)["z_fair"]

    def training_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        self._ensure_optimizers()
        base_z = self.encode(batch["input_seq"]).detach() if self.freeze_backbone else self.encode(batch["input_seq"])

        target_terms = []
        for attr in self.sensitive_attributes:
            if attr not in batch:
                continue
            z_attr = self.attribute_generators[attr](base_z.detach())
            cls_loss = F.cross_entropy(self.target_predictors[attr](z_attr), batch[attr].long())
            norm_loss = torch.norm(z_attr, dim=1).mean()
            target_terms.append(cls_loss + self.beta * norm_loss)
        target_loss = torch.stack(target_terms).mean() if target_terms else base_z.new_tensor(0.0)
        _optimizer_step(target_loss, self.attr_optimizer, self.clip_val)

        z_debiased = self.debiased_generator(base_z.detach())
        labels = collect_sensitive_labels(batch, self.sensitive_attributes)
        fair_adv_loss = _mean_adv_loss(self.fair_adversaries, z_debiased, labels) if labels else base_z.new_tensor(0.0)
        recon_loss = F.mse_loss(z_debiased, base_z.detach())
        debias_loss = self.recon_weight * recon_loss - self.lambda_adv * fair_adv_loss
        _optimizer_step(debias_loss, self.debias_optimizer, self.clip_val)

        disc_loss = base_z.new_tensor(0.0)
        if labels:
            for _ in range(self.disc_steps):
                z_debiased_detached = self.debiased_generator(base_z.detach()).detach()
                disc_loss = _mean_adv_loss(self.fair_adversaries, z_debiased_detached, labels)
                _optimizer_step(disc_loss, self.disc_optimizer, self.clip_val)

        mask = self._sample_mask()
        state = self._compose(base_z.detach(), mask, detach_parts=True)
        logits = score_items(self.backbone, state["z_fair"])
        rec_loss = recommendation_ce_loss(logits, batch["target"])
        _optimizer_step(rec_loss, self.combine_optimizer, self.clip_val)

        total_loss = rec_loss + target_loss + self.recon_weight * recon_loss
        return {
            "loss": total_loss.detach(),
            "rec_loss": rec_loss.detach(),
            "target_loss": target_loss.detach(),
            "adv_loss": fair_adv_loss.detach(),
            "recon_loss": recon_loss.detach(),
            "disc_loss": disc_loss.detach() if isinstance(disc_loss, torch.Tensor) else base_z.new_tensor(float(disc_loss)),
        }


class PFRecMethod(BackboneMethod):
    """Prompt-based selective fairness baseline.

    The backbone is frozen by default and fairness is injected through
    task-specific, user-conditioned, and attribute-combination prompts.
    """

    def __init__(self, backbone: nn.Module, config, name: str = "pfrec"):
        super().__init__(backbone, config, name=name)
        self.uses_custom_optimization = True
        self.sensitive_attributes = active_sensitive_attributes(config)
        self.attribute_dims = config.attribute_dims
        self.prompt_weight = getattr(config, "pfrec_prompt_weight", 0.05)
        self.adv_weight = getattr(config, "pfrec_adv_weight", 0.2)
        self.disc_steps = int(getattr(config, "pfrec_disc_steps", 1))
        self.freeze_backbone = bool(getattr(config, "pfrec_freeze_backbone", True))
        self.weight_decay = float(getattr(config, "l2_emb", 1e-6))
        self.clip_val = float(getattr(config, "gradient_clip_val", 5.0))
        self.prompt_keys = self._build_prompt_keys()
        self._prompt_pointer = 0

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.task_prompt = nn.Parameter(torch.zeros(1, self.hidden_units))
        self.combo_prompts = nn.ParameterDict({
            key: nn.Parameter(torch.zeros(self.hidden_units))
            for key in self.prompt_keys
        })
        self.user_prompt = make_mlp(self.hidden_units, self.hidden_units * 2, self.hidden_units, config.dropout_rate)
        self.adapters = nn.ModuleDict({
            key: make_mlp(self.hidden_units * 2, self.hidden_units * 2, self.hidden_units, config.dropout_rate)
            for key in self.prompt_keys
        })
        self.adversaries = nn.ModuleDict({
            attr: make_mlp(self.hidden_units, self.hidden_units, self.attribute_dims[attr], config.dropout_rate)
            for attr in self.sensitive_attributes
        })
        nn.init.normal_(self.task_prompt, std=0.02)
        for prompt in self.combo_prompts.values():
            nn.init.normal_(prompt, std=0.02)

        self.prompt_optimizer = None
        self.disc_optimizer = None

    def _ensure_optimizers(self):
        if self.prompt_optimizer is None:
            modules = [self.user_prompt, self.adapters]
            self.prompt_optimizer = _build_optimizer(modules, self.config.learning_rate, self.weight_decay)
            prompt_params = [self.task_prompt] + list(self.combo_prompts.parameters())
            if self.prompt_optimizer is not None:
                self.prompt_optimizer.add_param_group({"params": prompt_params})
            else:
                self.prompt_optimizer = torch.optim.AdamW(prompt_params, lr=self.config.learning_rate, weight_decay=self.weight_decay)
        if self.disc_optimizer is None:
            self.disc_optimizer = _build_optimizer([self.adversaries], self.config.learning_rate, self.weight_decay)

    def _build_prompt_keys(self) -> List[str]:
        keys = []
        for r in range(1, len(self.sensitive_attributes) + 1):
            for combo in itertools.combinations(self.sensitive_attributes, r):
                keys.append("__".join(combo))
        return keys

    def _next_mask(self) -> Dict[str, bool]:
        combo_key = self.prompt_keys[self._prompt_pointer % len(self.prompt_keys)]
        self._prompt_pointer += 1
        active = set(combo_key.split("__"))
        return {attr: attr in active for attr in self.sensitive_attributes}

    def _mask_key(self, mask: Dict[str, bool]) -> str:
        active = [attr for attr in self.sensitive_attributes if mask.get(attr, False)]
        return "__".join(active) if active else self.prompt_keys[0]

    def _apply_prompt(self, z: torch.Tensor, mask: Dict[str, bool]) -> torch.Tensor:
        key = self._mask_key(mask)
        task_prompt = self.task_prompt.expand(z.size(0), -1)
        combo_prompt = self.combo_prompts[key].unsqueeze(0).expand(z.size(0), -1)
        user_prompt = self.user_prompt(z)
        fused_prompt = task_prompt + combo_prompt + user_prompt
        adapter_out = self.adapters[key](torch.cat([z, fused_prompt], dim=1))
        return z + adapter_out

    def transform_representation(self, z: torch.Tensor, batch: Optional[Dict] = None) -> torch.Tensor:
        fair_mask = {attr: True for attr in self.sensitive_attributes}
        return self._apply_prompt(z, fair_mask)

    def training_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        self._ensure_optimizers()
        base_z = self.encode(batch["input_seq"]).detach() if self.freeze_backbone else self.encode(batch["input_seq"])
        mask = self._next_mask()
        active_attrs = [attr for attr, is_on in mask.items() if is_on and attr in batch]

        disc_loss = base_z.new_tensor(0.0)
        if active_attrs:
            for _ in range(self.disc_steps):
                z_prompt_detached = self._apply_prompt(base_z.detach(), mask).detach()
                disc_losses = [
                    F.cross_entropy(self.adversaries[attr](z_prompt_detached), batch[attr].long())
                    for attr in active_attrs
                ]
                disc_loss = torch.stack(disc_losses).mean()
                _optimizer_step(disc_loss, self.disc_optimizer, self.clip_val)

        z_prompt = self._apply_prompt(base_z, mask)
        logits = score_items(self.backbone, z_prompt)
        rec_loss = recommendation_ce_loss(logits, batch["target"])
        confusion_loss = base_z.new_tensor(0.0)
        if active_attrs:
            confusion_loss = torch.stack([
                F.cross_entropy(self.adversaries[attr](z_prompt), batch[attr].long())
                for attr in active_attrs
            ]).mean()
        prompt_reg = self.task_prompt.pow(2).mean()
        for prompt in self.combo_prompts.values():
            prompt_reg = prompt_reg + prompt.pow(2).mean()
        total_loss = rec_loss - self.adv_weight * confusion_loss + self.prompt_weight * prompt_reg
        _optimizer_step(total_loss, self.prompt_optimizer, self.clip_val)

        return {
            "loss": total_loss.detach(),
            "rec_loss": rec_loss.detach(),
            "adv_loss": confusion_loss.detach(),
            "prompt_reg": prompt_reg.detach(),
            "disc_loss": disc_loss.detach() if isinstance(disc_loss, torch.Tensor) else base_z.new_tensor(float(disc_loss)),
        }


class AFSRMethod(BackboneMethod):
    """A-FSR-style demographic-agnostic baseline.

    The method does not use sensitive attributes during training. Instead, it
    identifies high-salience local patterns with a gradient-based heuristic,
    smooths them using multi-hop item neighbors, and applies a robust objective
    over the hardest samples in the batch.
    """

    def __init__(self, backbone: nn.Module, config, resources: Dict, name: str = "a_fsr"):
        super().__init__(backbone, config, name=name)
        self.rng = random.Random(getattr(config, "seed", 42))
        self.similarity_candidates = resources.get("similarity_candidates", {})
        self.span_radius = int(getattr(config, "afsr_pattern_span", 2))
        self.dns_weight = float(getattr(config, "afsr_dns_weight", 0.5))
        self.dro_weight = float(getattr(config, "afsr_dro_weight", 0.5))
        self.dro_fraction = float(getattr(config, "afsr_dro_fraction", 0.3))
        self.dro_temperature = float(getattr(config, "afsr_dro_temperature", 5.0))
        self.num_items = int(getattr(config, "num_items", 0))

    def _sequence_outputs(self, input_seq: torch.Tensor) -> torch.Tensor:
        seq_output = self.backbone(input_seq)
        if isinstance(seq_output, tuple):
            seq_output = seq_output[0]
        return seq_output

    def _identify_stereotypical_positions(self, input_seq: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        seq_output = self._sequence_outputs(input_seq)
        if seq_output.dim() != 3:
            return torch.zeros(input_seq.size(0), dtype=torch.long, device=input_seq.device)

        seq_output.retain_grad()
        lengths = (input_seq > 0).sum(dim=1)
        batch_idx = torch.arange(input_seq.size(0), device=input_seq.device)
        last_idx = torch.clamp(lengths - 1, min=0)
        z = seq_output[batch_idx, last_idx]
        logits = score_items(self.backbone, z)
        per_sample_loss = _per_sample_ce_loss(logits, target)
        grads = torch.autograd.grad(per_sample_loss.sum(), seq_output, retain_graph=True, allow_unused=True)[0]
        if grads is None:
            return torch.zeros(input_seq.size(0), dtype=torch.long, device=input_seq.device)
        salience = grads.norm(dim=2) * (input_seq > 0).float()
        return salience.argmax(dim=1)

    def _multi_hop_candidates(self, item: int, hops: int = 2) -> List[int]:
        frontier = [item]
        visited = {item}
        collected = []
        for _ in range(hops):
            next_frontier = []
            for current in frontier:
                for cand in self.similarity_candidates.get(current, []):
                    cand = int(cand)
                    if cand <= 0 or cand > self.num_items or cand in visited:
                        continue
                    visited.add(cand)
                    collected.append(cand)
                    next_frontier.append(cand)
            frontier = next_frontier
            if not frontier:
                break
        return collected

    def _smooth_sequences(self, input_seq: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        out = input_seq.detach().clone()
        for row in range(out.size(0)):
            center = int(positions[row].item())
            if int(out[row, center].item()) <= 0:
                continue
            left = max(0, center - self.span_radius)
            right = min(out.size(1) - 1, center + self.span_radius)
            for pos in range(left, right + 1):
                item = int(out[row, pos].item())
                if item <= 0:
                    continue
                candidates = self._multi_hop_candidates(item, hops=2)
                if not candidates:
                    continue
                out[row, pos] = int(self.rng.choice(candidates))
        return out

    def compute_loss(self, batch: Dict) -> Dict[str, torch.Tensor]:
        input_seq = batch["input_seq"]
        target = batch["target"]

        positions = self._identify_stereotypical_positions(input_seq, target)
        smooth_seq = self._smooth_sequences(input_seq, positions)

        z = self.encode(input_seq)
        logits = score_items(self.backbone, z)
        per_loss = _per_sample_ce_loss(logits, target)
        rec_loss = per_loss.mean()

        z_smooth = self.encode(smooth_seq)
        logits_smooth = score_items(self.backbone, z_smooth)
        per_loss_smooth = _per_sample_ce_loss(logits_smooth, target)
        dns_loss = per_loss_smooth.mean()

        robust_losses = 0.5 * (per_loss + per_loss_smooth)
        keep = max(1, int(round(float(robust_losses.size(0)) * self.dro_fraction)))
        top_losses, _ = torch.topk(robust_losses, k=min(keep, robust_losses.size(0)))
        dro_weights = torch.softmax(self.dro_temperature * top_losses.detach(), dim=0)
        dro_loss = torch.sum(dro_weights * top_losses)

        total_loss = rec_loss + self.dns_weight * dns_loss + self.dro_weight * dro_loss
        return {
            "loss": total_loss,
            "rec_loss": rec_loss,
            "dns_loss": dns_loss,
            "dro_loss": dro_loss,
        }
