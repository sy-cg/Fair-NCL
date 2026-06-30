import os
import pickle
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from train import save_checkpoint, validate_model_optimized


def get_sequence_representation(model, input_seq):
    """Extract a [B, D] representation from any supported sequential backbone."""
    output = model(input_seq)
    if isinstance(output, tuple):
        output = output[0]

    if output.dim() == 2:
        return output

    seq_lengths = torch.sum(input_seq > 0, dim=1)
    batch_indices = torch.arange(input_seq.size(0), device=input_seq.device)
    last_indices = torch.clamp(seq_lengths - 1, min=0)
    return output[batch_indices, last_indices]


def alignment_loss(z, z_aug):
    z = F.normalize(z, dim=1)
    z_aug = F.normalize(z_aug, dim=1)
    return 2.0 - 2.0 * (z * z_aug).sum(dim=1).mean()


def variance_loss(z, eps=1e-4):
    z = z - z.mean(dim=0)
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))


def covariance_loss(z):
    if z.size(0) <= 1:
        return torch.tensor(0.0, device=z.device)

    z = z - z.mean(dim=0)
    cov = (z.T @ z) / (z.size(0) - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).sum() / z.size(1)


def fair_ncl_losses(z, z_aug, config):
    align = alignment_loss(z, z_aug)
    var = 0.5 * (variance_loss(z) + variance_loss(z_aug))
    cov = 0.5 * (covariance_loss(z) + covariance_loss(z_aug))

    total = (
        getattr(config, 'fair_ncl_align_weight', 1.0) * align +
        getattr(config, 'fair_ncl_var_weight', 1.0) * var +
        getattr(config, 'fair_ncl_cov_weight', 0.04) * cov
    )

    return total, {
        'align_loss': align,
        'var_loss': var,
        'cov_loss': cov
    }


def train_fair_ncl_model(model, train_loader, val_loader, config, augmenter):
    """Train a base sequential recommender with Fair-NCL augmentation and losses."""
    print(f"Training Fair-NCL with base model: {getattr(config, 'base_model_name', config.model_name)}")

    if config.use_data_parallel and config.num_gpus > 1:
        print(f"Using DataParallel with {config.num_gpus} GPUs")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.l2_emb,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    scaler = GradScaler(enabled=config.use_mixed_precision)

    best_val_recall = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    history = defaultdict(list)

    for epoch in range(config.num_epochs):
        start_time = time.time()
        model.train()
        train_losses = defaultdict(float)
        num_batches = 0

        pbar = tqdm(train_loader, desc=f'Fair-NCL Epoch {epoch + 1}/{config.num_epochs}')
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=config.use_mixed_precision):
                input_seq = batch['input_seq'].to(config.device)
                target = batch['target'].to(config.device)
                if target.dim() > 1:
                    target = target.squeeze(-1)

                aug_batch = augmenter.augment_batch_optimized(batch) if augmenter is not None else batch
                aug_seq = aug_batch['input_seq'].to(config.device)

                model_for_repr = model.module if hasattr(model, 'module') else model
                rec_loss = model_for_repr.compute_loss(input_seq, target)
                aug_rec_loss = model_for_repr.compute_loss(aug_seq, target)

                z = get_sequence_representation(model_for_repr, input_seq)
                z_aug = get_sequence_representation(model_for_repr, aug_seq)
                ncl_loss, ncl_parts = fair_ncl_losses(z, z_aug, config)

                total_loss = (
                    rec_loss +
                    getattr(config, 'fair_ncl_aug_rec_weight', 0.5) * aug_rec_loss +
                    ncl_loss
                )

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
            scaler.step(optimizer)
            scaler.update()

            train_losses['total_loss'] += total_loss.item()
            train_losses['rec_loss'] += rec_loss.item()
            train_losses['aug_rec_loss'] += aug_rec_loss.item()
            for name, value in ncl_parts.items():
                train_losses[name] += value.item()

            num_batches += 1
            if num_batches % 10 == 0:
                pbar.set_postfix({
                    'loss': f"{train_losses['total_loss'] / num_batches:.4f}",
                    'rec': f"{train_losses['rec_loss'] / num_batches:.4f}",
                    'align': f"{train_losses['align_loss'] / num_batches:.4f}"
                })

        for name, value in train_losses.items():
            history[f'train_{name}'].append(value / max(num_batches, 1))

        val_metrics = validate_model_optimized(model, val_loader, config)
        val_loss = val_metrics['loss']
        scheduler.step(val_loss)

        for name, value in val_metrics.items():
            history[f'val_{name}'].append(value)

        current_recall = val_metrics.get('recall@10')
        if current_recall is not None:
            is_best = current_recall > best_val_recall
            if is_best:
                best_val_recall = current_recall
        else:
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

        if is_best:
            patience_counter = 0
            save_checkpoint(model, config, epoch, val_metrics, is_best=True)
            print("  -> Best Fair-NCL model saved")
        else:
            patience_counter += 1

        elapsed = time.time() - start_time
        print(f"Epoch {epoch + 1}: train_loss={history['train_total_loss'][-1]:.4f}, "
              f"val_loss={val_loss:.4f}, val_recall@10={val_metrics.get('recall@10', 0):.4f}, "
              f"time={elapsed:.1f}s")

        if patience_counter >= config.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

        torch.cuda.empty_cache()

    history_path = os.path.join(config.processed_data_dir, f'{config.model_name}_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(dict(history), f)

    return dict(history)
