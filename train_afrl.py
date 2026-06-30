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


def train_afrl_model_optimized(model, train_loader, val_loader, config):
    """GPU优化的AFRL训练函数"""

    print("Starting AFRL optimized training...")

    # 设置优化器
    # 主模型参数
    main_params = []
    for name, param in model.named_parameters():
        if 'discriminator' not in name:
            main_params.append(param)

    # 判别器参数
    disc_params = []
    for name, param in model.named_parameters():
        if 'discriminator' in name:
            disc_params.append(param)

    # 创建优化器
    main_optimizer = optim.Adam(main_params, lr=config.learning_rate)
    disc_optimizer = optim.Adam(disc_params, lr=config.learning_rate * 2)  # 判别器学习率更高

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

    # 判别器预训练步数
    disc_pretrain_steps = getattr(config, 'disc_pretrain_steps', 5)

    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()

        # 训练阶段
        model.train()
        train_losses = defaultdict(float)

        # 训练进度条
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs}')

        for batch_idx, batch in enumerate(train_pbar):
            # 判别器预训练（前几个epoch）
            if epoch < disc_pretrain_steps:
                disc_losses = train_discriminators_step(
                    model, batch, disc_optimizer, scaler, config
                )
                for k, v in disc_losses.items():
                    train_losses[k] += v

            # 主模型训练
            main_losses = train_main_model_step(
                model, batch, main_optimizer, scaler, config
            )
            for k, v in main_losses.items():
                train_losses[k] += v

            # 更新进度条
            if batch_idx % 10 == 0:
                avg_losses = {k: v / (batch_idx + 1) for k, v in train_losses.items()}
                train_pbar.set_postfix(avg_losses)

        # 计算平均损失
        num_batches = len(train_loader)
        for k in train_losses:
            train_losses[k] /= num_batches
            history[f'train_{k}'].append(train_losses[k])

        # 验证阶段
        if (epoch + 1) % config.eval_interval == 0:
            val_metrics = validate_afrl_model(model, val_loader, config)

            # 记录验证指标
            for k, v in val_metrics.items():
                history[f'val_{k}'].append(v)

            # 检查最佳模型
            current_recall = val_metrics['Recall@10']
            if current_recall > best_val_recall:
                best_val_recall = current_recall
                best_epoch = epoch
                patience_counter = 0

                # 保存最佳模型
                save_checkpoint(model, config, epoch, best_val_recall, is_best=True)
            else:
                patience_counter += 1

            print(f"\nEpoch {epoch + 1} - Val Recall@10: {current_recall:.4f} "
                  f"(Best: {best_val_recall:.4f} at epoch {best_epoch + 1})")

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


def train_discriminators_step(model, batch, optimizer, scaler, config):
    """训练判别器的单个步骤"""
    losses = {}

    with autocast(enabled=config.use_mixed_precision):
        # 获取去偏嵌入
        _, _, debiased_emb, _ = model(batch['input_seq'])

        # 更新判别器
        disc_losses = model.update_discriminators(batch, debiased_emb)

        # 计算总判别器损失
        total_disc_loss = sum(disc_losses.values())

    # 反向传播
    optimizer.zero_grad()
    scaler.scale(total_disc_loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        [p for n, p in model.named_parameters() if 'discriminator' in n],
        config.grad_clip
    )
    scaler.step(optimizer)
    scaler.update()

    # 记录损失
    for k, v in disc_losses.items():
        losses[k] = v.item()

    return losses


def train_main_model_step(model, batch, optimizer, scaler, config):
    """训练主模型的单个步骤"""
    losses = {}

    with autocast(enabled=config.use_mixed_precision):
        # 前向传播
        fair_emb, attr_embs, debiased_emb, user_emb = model(batch['input_seq'])

        # 计算AFRL损失
        afrl_losses = model.compute_afrl_losses(
            batch, fair_emb, attr_embs, debiased_emb, user_emb
        )

        total_loss = afrl_losses['total_loss']

    # 反向传播
    optimizer.zero_grad()
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(
        [p for n, p in model.named_parameters() if 'discriminator' not in n],
        config.grad_clip
    )
    scaler.step(optimizer)
    scaler.update()

    # 记录损失
    for k, v in afrl_losses.items():
        losses[k] = v.item()

    return losses


@torch.no_grad()
def validate_afrl_model(model, val_loader, config):
    """验证AFRL模型"""
    model.eval()

    all_predictions = []
    all_targets = []

    for batch in tqdm(val_loader, desc='Validation'):
        # 获取候选物品
        pos_items = batch['target'].unsqueeze(1)
        neg_items = batch['negative_items']
        candidate_items = torch.cat([pos_items, neg_items], dim=1)

        # 预测（使用随机公平性掩码进行验证）
        batch_size = batch['input_seq'].size(0)
        num_attrs = len(config.sensitive_attributes)
        random_mask = torch.rand(batch_size, num_attrs, device=config.device) > 0.5

        logits = model.predict(batch['input_seq'], candidate_items, random_mask.float())

        all_predictions.append(logits.cpu())

        # 标签：第一个是正样本
        labels = torch.zeros_like(logits)
        labels[:, 0] = 1
        all_targets.append(labels.cpu())

    # 合并预测
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # 计算指标
    metrics = compute_ranking_metrics(predictions, targets, config.topk_list)

    return metrics


def compute_ranking_metrics(predictions, targets, k_list):
    """计算排序指标"""
    metrics = {}

    # 获取排序
    _, indices = torch.sort(predictions, dim=1, descending=True)

    for k in k_list:
        # Recall@K
        top_k = indices[:, :k]
        hits = torch.gather(targets, 1, top_k).sum(dim=1)
        recall = hits.float().mean().item()
        metrics[f'Recall@{k}'] = recall

        # HitRate@K (命中率：至少有一个正确推荐的用户比例)
        hit_users = (hits > 0).float().mean().item()
        metrics[f'HR@{k}'] = hit_users

        # Precision@K
        precision = (hits.float() / k).mean().item()
        metrics[f'Precision@{k}'] = precision

        # NDCG@K
        dcg = torch.zeros(predictions.size(0))
        idcg = torch.zeros(predictions.size(0))

        for i in range(min(k, predictions.size(1))):
            rel = torch.gather(targets, 1, indices[:, i:i + 1]).squeeze()
            dcg += rel / torch.log2(torch.tensor(i + 2.0))

            if i == 0:
                idcg += targets.max(dim=1)[0] / torch.log2(torch.tensor(2.0))

        ndcg = (dcg / (idcg + 1e-8)).mean().item()
        metrics[f'NDCG@{k}'] = ndcg

        # MRR@K
        mrr = torch.zeros(predictions.size(0))
        for i in range(min(k, predictions.size(1))):
            rel = torch.gather(targets, 1, indices[:, i:i + 1]).squeeze()
            mrr += rel / (i + 1)
        mrr = mrr.mean().item()
        metrics[f'MRR@{k}'] = mrr

    return metrics


def save_checkpoint(model, config, epoch, best_metric, is_best=False):
    """保存模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'best_metric': best_metric,
        'config_dict': config.__dict__ if hasattr(config, '__dict__') else config
    }

    # 保存路径
    model_name = config.model_name.replace('afrl_', '')
    filename = f'afrl_{model_name}_checkpoint.pth'
    if is_best:
        filename = f'afrl_{model_name}_best.pth'

    filepath = os.path.join(config.model_save_dir, filename)
    torch.save(checkpoint, filepath)

    if is_best:
        print(f"Best model saved to {filepath}")


def load_afrl_checkpoint(model, config):
    """加载AFRL模型检查点"""
    model_name = config.model_name.replace('afrl_', '')
    filepath = os.path.join(config.model_save_dir, f'afrl_{model_name}_best.pth')

    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from {filepath}")
        return model
    else:
        print(f"No checkpoint found at {filepath}")
        return model


@torch.no_grad()
def validate_afrl_model(model, val_loader, config):
    """Final AFRL validation with full-ranking evaluation."""
    model.eval()

    ranking_results = evaluate_full_ranking_loader(
        data_loader=val_loader,
        config=config,
        predict_fn=lambda batch: model.predict(
            batch['input_seq'],
            fairness_mask=(
                torch.rand(
                    batch['input_seq'].size(0),
                    len(config.sensitive_attributes),
                    device=batch['input_seq'].device,
                ) > 0.5
            ).float(),
        ),
        sensitive_attributes=[],
        desc='Validation-FullRanking-AFRL',
        legacy_output=False,
    )
    utility = ranking_results['utility']

    metrics = {}
    for k in getattr(config, 'topk_list', [5, 10, 20]):
        metrics[f'Recall@{k}'] = utility.get(f'Recall@{k}', 0.0)
        metrics[f'HR@{k}'] = utility.get(f'HitRate@{k}', 0.0)
        metrics[f'Precision@{k}'] = utility.get(f'Precision@{k}', 0.0)
        metrics[f'NDCG@{k}'] = utility.get(f'NDCG@{k}', 0.0)
        metrics[f'MRR@{k}'] = utility.get(f'MRR@{k}', 0.0)
    return metrics
