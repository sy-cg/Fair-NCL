"""
FFVAE模型训练函数
处理FFVAE的特殊训练流程（包括判别器训练）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import time
import os
import numpy as np
from collections import defaultdict

from evaluation.full_ranking import evaluate_full_ranking_loader


def train_ffvae_model(model, train_loader, val_loader, config):
    """FFVAE专用训练函数"""

    print("Starting FFVAE training...")
    print(f"FFVAE Configuration:")
    print(f"  z dimension: {model.ffvae.z_dim}")
    print(f"  Sensitive attributes: {config.sensitive_attributes}")
    print(f"  Alpha (predictiveness): {model.ffvae.alpha}")
    print(f"  Beta (disentanglement): {model.ffvae.beta}")

    # 主模型优化器
    main_optimizer = optim.Adam(
        list(model.ffvae.encoder.parameters()) +
        list(model.ffvae.decoder.parameters()) +
        list(model.recommendation_head.parameters()),
        lr=config.learning_rate
    )

    # 判别器优化器
    disc_optimizer = optim.Adam(
        model.ffvae.tc_discriminator.parameters(),
        lr=config.learning_rate * 2  # 判别器学习率更高
    )

    # 学习率调度器
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        main_optimizer, T_max=config.num_epochs
    )
    disc_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        disc_optimizer, T_max=config.num_epochs
    )

    # 混合精度训练
    scaler = GradScaler(enabled=config.use_mixed_precision)

    # 训练历史
    history = defaultdict(list)
    best_val_recall = 0
    best_epoch = 0
    patience_counter = 0

    # 判别器预训练轮数
    disc_pretrain_epochs = 2

    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()

        # 训练阶段
        model.train()
        train_losses = defaultdict(float)
        num_batches = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs}')

        for batch_idx, batch in enumerate(train_pbar):
            # 将批次数据移到设备
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(config.device)

            # 1. 训练判别器
            disc_optimizer.zero_grad()
            with autocast(enabled=config.use_mixed_precision):
                disc_loss, log_ratio = model.ffvae.compute_tc_discriminator_loss(batch)

            scaler.scale(disc_loss).backward()
            scaler.step(disc_optimizer)
            scaler.update()

            train_losses['disc_loss'] += disc_loss.item()

            # 2. 训练主模型（如果判别器已经预训练）
            if epoch >= disc_pretrain_epochs:
                main_optimizer.zero_grad()

                with autocast(enabled=config.use_mixed_precision):
                    # FFVAE损失
                    ffvae_loss, loss_dict = model.ffvae.compute_loss(batch, log_ratio.detach())

                    # 推荐损失
                    #input_seq = batch['input_seq']
                    #target = batch['target']
                    #rec_loss = model.compute_loss(input_seq, target)
                    # 给 rec_loss 一个默认值，防止报错，或者直接设为 0
                    input_seq = batch['input_seq']
                    target = batch['target']
                    rec_loss = model.compute_loss(input_seq, target)
                    # 总损失
                    total_loss = ffvae_loss #+ rec_loss  模型结果较不公平，进行纯表征预训练

                total_loss = ffvae_loss + getattr(config, 'ffvae_rec_weight', 1.0) * rec_loss
                scaler.scale(total_loss).backward()
                scaler.unscale_(main_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.ffvae.encoder.parameters()) +
                    list(model.ffvae.decoder.parameters()) +
                    list(model.recommendation_head.parameters()),
                    config.grad_clip
                )
                scaler.step(main_optimizer)
                scaler.update()

                # 记录损失
                for k, v in loss_dict.items():
                    train_losses[k] += v.item()
                train_losses['rec_loss'] += rec_loss.item()

            num_batches += 1

            # 更新进度条
            if batch_idx % 10 == 0:
                avg_losses = {k: v / num_batches for k, v in train_losses.items()}
                train_pbar.set_postfix(avg_losses)

        # 计算平均损失
        for k in train_losses:
            train_losses[k] /= num_batches
            history[f'train_{k}'].append(train_losses[k])

        # 验证阶段
        if (epoch + 1) % config.eval_interval == 0:
            val_metrics = validate_ffvae_model(model, val_loader, config)

            # 记录验证指标
            for k, v in val_metrics.items():
                history[f'val_{k}'].append(v)

            # 检查最佳模型
            current_recall = val_metrics.get('Recall@10', 0)
            if current_recall > best_val_recall:
                best_val_recall = current_recall
                best_epoch = epoch
                patience_counter = 0

                # 保存最佳模型
                save_ffvae_checkpoint(model, config, epoch, best_val_recall, is_best=True)
            else:
                patience_counter += 1

            print(f"\nEpoch {epoch + 1}:")
            print(f"  Val Recall@10: {current_recall:.4f} (Best: {best_val_recall:.4f})")
            print(f"  FFVAE Losses - Recon: {train_losses['recon_loss']:.4f}, "
                  f"Pred: {train_losses['pred_loss']:.4f}, TC: {train_losses['tc_loss']:.4f}")

        # 学习率调度
        main_scheduler.step()
        disc_scheduler.step()

        # 早停
        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s")

    print(f"\nTraining completed. Best Recall@10: {best_val_recall:.4f} at epoch {best_epoch + 1}")

    return history


@torch.no_grad()
def validate_ffvae_model(model, val_loader, config):
    """验证FFVAE模型"""
    model.eval()

    all_predictions = []
    all_targets = []

    for batch in tqdm(val_loader, desc='Validation'):
        # 将批次数据移到设备
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(config.device)

        input_seq = batch['input_seq']
        pos_items = batch['target'].unsqueeze(1)
        neg_items = batch['negative_items']
        candidate_items = torch.cat([pos_items, neg_items], dim=1)

        # 预测（使用完整表示）
        logits = model.predict(input_seq, candidate_items)

        all_predictions.append(logits.cpu())

        # 标签
        labels = torch.zeros_like(logits)
        labels[:, 0] = 1
        all_targets.append(labels.cpu())

    # 合并预测
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # 计算指标
    metrics = compute_ranking_metrics(predictions, targets, config.topk_list)

    return metrics


def validate_ffvae_fairness(model, val_loader, config):
    """验证FFVAE的公平性（使用不同的敏感属性掩码）"""
    model.eval()

    fairness_results = {}

    # 测试不同的公平性设置
    fairness_configs = [
        ('full', None),  # 完整表示
        ('fair_all', [True] * len(config.sensitive_attributes)),  # 去除所有敏感属性
        ('fair_gender', [i == 0 for i in range(len(config.sensitive_attributes))]),  # 只去除性别
        ('fair_age', [i == 1 for i in range(len(config.sensitive_attributes))]),  # 只去除年龄
    ]

    for name, mask in fairness_configs:
        all_predictions = []
        all_targets = []

        for batch in tqdm(val_loader, desc=f'Fairness eval {name}'):
            # 将批次数据移到设备
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(config.device)

            input_seq = batch['input_seq']
            pos_items = batch['target'].unsqueeze(1)
            neg_items = batch['negative_items']
            candidate_items = torch.cat([pos_items, neg_items], dim=1)

            # 使用公平性掩码预测
            logits = model.predict(input_seq, candidate_items, fairness_mask=mask)

            all_predictions.append(logits.cpu())

            labels = torch.zeros_like(logits)
            labels[:, 0] = 1
            all_targets.append(labels.cpu())

        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)

        metrics = compute_ranking_metrics(predictions, targets, config.topk_list)
        fairness_results[name] = metrics

    return fairness_results


def compute_ranking_metrics(predictions, targets, k_list):
    """计算排序指标"""
    metrics = {}

    _, indices = torch.sort(predictions, dim=1, descending=True)

    for k in k_list:
        top_k = indices[:, :k]
        hits = torch.gather(targets, 1, top_k).sum(dim=1)

        metrics[f'Recall@{k}'] = hits.float().mean().item()
        metrics[f'HitRate@{k}'] = (hits > 0).float().mean().item()

        # NDCG
        dcg = torch.zeros(predictions.size(0))
        idcg = torch.zeros(predictions.size(0))

        for i in range(min(k, predictions.size(1))):
            rel = torch.gather(targets, 1, indices[:, i:i + 1]).squeeze()
            dcg += rel / torch.log2(torch.tensor(i + 2.0))
            if i == 0:
                idcg += targets.max(dim=1)[0] / torch.log2(torch.tensor(2.0))

        metrics[f'NDCG@{k}'] = (dcg / (idcg + 1e-8)).mean().item()

    return metrics


def save_ffvae_checkpoint(model, config, epoch, best_metric, is_best=False):
    """保存FFVAE模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'best_metric': best_metric,
        'config_dict': config.__dict__ if hasattr(config, '__dict__') else config,
        'ffvae_config': {
            'z_dim': model.ffvae.z_dim,
            'b_dims': model.ffvae.b_dims,
            'alpha': model.ffvae.alpha,
            'beta': model.ffvae.beta
        }
    }

    filename = 'ffvae_checkpoint.pth'
    if is_best:
        filename = 'ffvae_best.pth'

    filepath = os.path.join(config.model_save_dir, filename)
    torch.save(checkpoint, filepath)

    if is_best:
        print(f"Best FFVAE model saved to {filepath}")


def load_ffvae_checkpoint(model, config):
    """加载FFVAE模型检查点"""
    filepath = os.path.join(config.model_save_dir, 'ffvae_best.pth')

    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best FFVAE model from {filepath}")

        # 打印FFVAE配置
        if 'ffvae_config' in checkpoint:
            print("FFVAE Configuration:")
            for k, v in checkpoint['ffvae_config'].items():
                print(f"  {k}: {v}")

        return model
    else:
        print(f"No FFVAE checkpoint found at {filepath}")
        return model


@torch.no_grad()
def validate_ffvae_model(model, val_loader, config):
    """Final FFVAE validation with full-ranking evaluation."""
    model.eval()

    ranking_results = evaluate_full_ranking_loader(
        data_loader=val_loader,
        config=config,
        predict_fn=lambda batch: model.predict(batch['input_seq']),
        sensitive_attributes=[],
        desc='Validation-FullRanking-FFVAE',
        legacy_output=False,
    )
    utility = ranking_results['utility']

    metrics = {}
    for k in getattr(config, 'topk_list', [5, 10, 20]):
        metrics[f'Recall@{k}'] = utility.get(f'Recall@{k}', 0.0)
        metrics[f'HitRate@{k}'] = utility.get(f'HitRate@{k}', 0.0)
        metrics[f'Precision@{k}'] = utility.get(f'Precision@{k}', 0.0)
        metrics[f'NDCG@{k}'] = utility.get(f'NDCG@{k}', 0.0)
        metrics[f'MRR@{k}'] = utility.get(f'MRR@{k}', 0.0)
    return metrics


def validate_ffvae_fairness(model, val_loader, config):
    """Final FFVAE fairness audit with full-ranking evaluation."""
    model.eval()

    fairness_results = {}
    fairness_configs = [
        ('full', None),
        ('fair_all', [True] * len(config.sensitive_attributes)),
        ('fair_gender', [i == 0 for i in range(len(config.sensitive_attributes))]),
        ('fair_age', [i == 1 for i in range(len(config.sensitive_attributes))]),
    ]

    for name, mask in fairness_configs:
        ranking_results = evaluate_full_ranking_loader(
            data_loader=val_loader,
            config=config,
            predict_fn=lambda batch, current_mask=mask: model.predict(
                batch['input_seq'],
                fairness_mask=current_mask,
            ),
            sensitive_attributes=config.sensitive_attributes,
            desc=f'Fairness-FullRanking-{name}',
            legacy_output=False,
        )
        fairness_results[name] = {
            'utility': ranking_results['utility'],
            'fairness': ranking_results['fairness'],
            'selected': ranking_results['selected'],
        }

    return fairness_results
