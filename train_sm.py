import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import time
import os
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any

from evaluation.full_ranking import evaluate_full_ranking_loader


def train_sm_model_optimized(model, train_loader, val_loader, config):
    """GPU优化的SM训练函数（修复版）"""

    print("Starting SM (Separate Method) optimized training...")
    print(f"SM Configuration:")
    print(f"  Lambda (adversarial): {getattr(config, 'sm_lambda', 1.0)}")
    print(f"  Sensitive attributes: {config.sensitive_attributes}")
    print(f"  Filter combinations: {len(model.filter_module.filters)}")

    # 设置优化器 - 修复：正确分离参数
    main_params = []
    disc_params = []

    for name, param in model.named_parameters():
        if 'discriminator' in name:
            disc_params.append(param)
        else:
            main_params.append(param)

    main_optimizer = optim.Adam(main_params, lr=config.learning_rate)
    disc_optimizer = optim.Adam(disc_params, lr=config.learning_rate * 2)

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

    disc_pretrain_steps = getattr(config, 'sm_pretrain_steps', 3)

    for epoch in range(config.num_epochs):
        epoch_start_time = time.time()

        # 训练阶段
        model.train()
        train_losses = defaultdict(float)

        train_pbar = tqdm(train_loader, desc=f'SM Epoch {epoch + 1}/{config.num_epochs}')

        for batch_idx, batch in enumerate(train_pbar):
            # 批处理训练（修复版）
            if epoch < disc_pretrain_steps:
                # 判别器预训练
                batch_loss = train_sm_batch_discriminators_fixed(
                    model, batch, disc_optimizer, disc_params, scaler, config
                )
            else:
                # 主模型训练
                batch_loss = train_sm_batch_main_fixed(
                    model, batch, main_optimizer, main_params,
                    disc_optimizer, disc_params, scaler, config
                )

            # 累积损失
            for k, v in batch_loss.items():
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
            val_metrics = validate_sm_model_batch(model, val_loader, config)

            # 记录验证指标
            for k, v in val_metrics.items():
                history[f'val_{k}'].append(v)

            # 检查最佳模型
            current_recall = val_metrics['Recall@10']
            if current_recall > best_val_recall:
                best_val_recall = current_recall
                best_epoch = epoch
                patience_counter = 0
                save_sm_checkpoint(model, config, epoch, best_val_recall, is_best=True)
            else:
                patience_counter += 1

            print(f"\nEpoch {epoch + 1}:")
            print(f"  Val Recall@10: {current_recall:.4f} (Best: {best_val_recall:.4f})")

        # 学习率调度
        main_scheduler.step()
        disc_scheduler.step()

        # 早停
        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1} completed in {epoch_time:.1f}s")

    print(f"\nTraining completed. Best Recall@10: {best_val_recall:.4f}")

    return history


def train_sm_batch_discriminators_fixed(model, batch, optimizer, disc_params, scaler, config):
    """修复版：批处理判别器训练"""
    losses = defaultdict(float)

    # 为整个批次生成随机掩码
    batch_size = batch['input_seq'].size(0)
    num_attrs = len(config.sensitive_attributes)
    random_masks = torch.rand(batch_size, num_attrs, device=config.device) > 0.5

    with autocast(enabled=config.use_mixed_precision):
        # 获取过滤后的嵌入（批处理）
        input_seq = batch['input_seq'].to(config.device)

        # 修复：使用统一的掩码格式
        mask_list = []
        for i in range(batch_size):
            mask = {}
            for j, attr in enumerate(config.sensitive_attributes):
                mask[attr] = bool(random_masks[i, j].item())
            mask_list.append(mask)

        filtered_emb, _ = model(input_seq, mask_list)

        # 计算判别器损失
        total_disc_loss = 0
        for j, attr in enumerate(config.sensitive_attributes):
            # 获取该属性的掩码
            attr_mask = random_masks[:, j]

            if attr_mask.any():
                # 只对需要判别的样本计算损失
                masked_emb = filtered_emb[attr_mask]
                masked_labels = batch[attr][attr_mask].to(config.device)

                disc_logits = model.discriminator_module.discriminators[attr](masked_emb)
                disc_loss = F.cross_entropy(disc_logits, masked_labels)

                losses[f'disc_{attr}_loss'] = disc_loss.item()
                total_disc_loss += disc_loss

    # 反向传播
    if total_disc_loss != 0:
        optimizer.zero_grad()
        scaler.scale(total_disc_loss).backward()
        scaler.unscale_(optimizer)
        # 修复：使用正确的梯度裁剪值
        grad_clip = getattr(config, 'gradient_clip_val', 5.0)
        torch.nn.utils.clip_grad_norm_(disc_params, grad_clip)
        scaler.step(optimizer)
        scaler.update()

    return losses


def train_sm_batch_main_fixed(model, batch, main_optimizer, main_params,
                              disc_optimizer, disc_params, scaler, config):
    """修复版：批处理主模型训练"""
    losses = defaultdict(float)

    # 生成批次掩码
    batch_size = batch['input_seq'].size(0)
    num_attrs = len(config.sensitive_attributes)
    random_masks = torch.rand(batch_size, num_attrs, device=config.device) > 0.5

    # 第一步：训练判别器
    disc_losses = update_discriminators_batch_fixed(
        model, batch, disc_optimizer, disc_params, scaler, config, random_masks
    )
    for k, v in disc_losses.items():
        losses[k] = v

    # 第二步：训练主模型
    with autocast(enabled=config.use_mixed_precision):
        # 前向传播
        input_seq = batch['input_seq'].to(config.device)

        # 修复：使用统一的掩码格式
        mask_list = []
        for i in range(batch_size):
            mask = {}
            for j, attr in enumerate(config.sensitive_attributes):
                mask[attr] = bool(random_masks[i, j].item())
            mask_list.append(mask)

        filtered_emb, original_emb = model(input_seq, mask_list)

        # 1. 推荐损失（批处理）
        pos_items = batch['target'].to(config.device)
        pos_emb = model.base_model.item_emb(pos_items)
        pos_scores = torch.sum(filtered_emb * pos_emb, dim=1)

        neg_items = batch.get('negative_items', None)
        if neg_items is not None and neg_items.size(1) > 0:
            neg_items = neg_items.to(config.device)
            neg_emb = model.base_model.item_emb(neg_items)
            neg_scores = torch.sum(
                filtered_emb.unsqueeze(1) * neg_emb, dim=2
            )
            neg_scores_max = torch.max(neg_scores, dim=1)[0]
            rec_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores_max) + 1e-8).mean()
        else:
            rec_loss = F.mse_loss(pos_scores, torch.ones_like(pos_scores))

        losses['rec_loss'] = rec_loss.item()

        # 2. 对抗损失（批处理）
        adv_loss = 0
        for j, attr in enumerate(config.sensitive_attributes):
            attr_mask = random_masks[:, j]

            if attr_mask.any():
                masked_emb = filtered_emb[attr_mask]
                gen_logits = model.discriminator_module.discriminators[attr](masked_emb)

                # 生成随机标签以混淆判别器
                batch_size_masked = masked_emb.size(0)
                num_classes = model.attribute_dims[attr]
                fake_labels = torch.randint(0, num_classes, (batch_size_masked,),
                                          device=masked_emb.device)

                gen_loss = F.cross_entropy(gen_logits, fake_labels)
                adv_loss += model.lambda_adv * gen_loss

        if isinstance(adv_loss, torch.Tensor):
            adv_loss = adv_loss / len(config.sensitive_attributes)
            losses['adv_loss'] = adv_loss.item()
        else:
            losses['adv_loss'] = 0.0

        # 总损失
        total_loss = rec_loss + (adv_loss if isinstance(adv_loss, torch.Tensor) else 0)
        losses['total_loss'] = total_loss.item()

    # 反向传播
    main_optimizer.zero_grad()
    scaler.scale(total_loss).backward()
    scaler.unscale_(main_optimizer)
    # 修复：使用正确的梯度裁剪值
    grad_clip = getattr(config, 'gradient_clip_val', 5.0)
    torch.nn.utils.clip_grad_norm_(main_params, grad_clip)
    scaler.step(main_optimizer)
    scaler.update()

    return losses


def update_discriminators_batch_fixed(model, batch, optimizer, disc_params,
                                      scaler, config, random_masks):
    """修复版：更新判别器"""
    losses = defaultdict(float)

    with autocast(enabled=config.use_mixed_precision):
        # 获取过滤后的嵌入
        input_seq = batch['input_seq'].to(config.device)

        # 修复：使用统一的掩码格式
        mask_list = []
        for i in range(random_masks.size(0)):
            mask = {}
            for j, attr in enumerate(config.sensitive_attributes):
                mask[attr] = bool(random_masks[i, j].item())
            mask_list.append(mask)

        filtered_emb, _ = model(input_seq, mask_list)
        filtered_emb = filtered_emb.detach()  # 不要传播梯度到生成器

        # 计算判别器损失
        total_disc_loss = 0
        for j, attr in enumerate(config.sensitive_attributes):
            attr_mask = random_masks[:, j]

            if attr_mask.any():
                masked_emb = filtered_emb[attr_mask]
                masked_labels = batch[attr][attr_mask].to(config.device)

                disc_logits = model.discriminator_module.discriminators[attr](masked_emb)
                disc_loss = F.cross_entropy(disc_logits, masked_labels)

                losses[f'disc_{attr}_loss'] = disc_loss.item()
                total_disc_loss += disc_loss

    # 反向传播
    if total_disc_loss != 0:
        optimizer.zero_grad()
        scaler.scale(total_disc_loss).backward()
        scaler.unscale_(optimizer)
        grad_clip = getattr(config, 'gradient_clip_val', 5.0)
        torch.nn.utils.clip_grad_norm_(disc_params, grad_clip)
        scaler.step(optimizer)
        scaler.update()

    return losses

@torch.no_grad()
def validate_sm_model_batch(model, val_loader, config):
    """批处理SM验证（改进版）"""
    model.eval()

    all_predictions = []
    all_targets = []

    for batch in tqdm(val_loader, desc='SM Validation'):
        # 生成批次掩码
        batch_size = batch['input_seq'].size(0)
        num_attrs = len(config.sensitive_attributes)
        random_masks = torch.rand(batch_size, num_attrs, device=config.device) > 0.5

        # 获取候选物品
        pos_items = batch['target'].unsqueeze(1).to(config.device)
        neg_items = batch['negative_items'].to(config.device)
        candidate_items = torch.cat([pos_items, neg_items], dim=1)

        # 批量预测
        input_seq = batch['input_seq'].to(config.device)
        torch.compiler.cudagraph_mark_step_begin()
        logits = model.predict(input_seq, candidate_items, random_masks.float())

        all_predictions.append(logits.cpu())

        # 标签
        labels = torch.zeros_like(logits)
        labels[:, 0] = 1
        all_targets.append(labels.cpu())

    # 合并预测
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # 计算指标
    metrics = compute_sm_ranking_metrics(predictions, targets, config.topk_list)

    return metrics


def compute_sm_ranking_metrics(predictions, targets, k_list):
    """计算排序指标"""
    metrics = {}

    _, indices = torch.sort(predictions, dim=1, descending=True)

    for k in k_list:
        top_k = indices[:, :k]
        hits = torch.gather(targets, 1, top_k).sum(dim=1)

        metrics[f'Recall@{k}'] = hits.float().mean().item()
        metrics[f'HitRate@{k}'] = (hits > 0).float().mean().item()
        metrics[f'Precision@{k}'] = (hits.float() / k).mean().item()

        # NDCG@K
        dcg = torch.zeros(predictions.size(0))
        idcg = torch.zeros(predictions.size(0))

        for i in range(min(k, predictions.size(1))):
            rel = torch.gather(targets, 1, indices[:, i:i+1]).squeeze()
            dcg += rel / torch.log2(torch.tensor(i + 2.0))
            if i == 0:
                idcg += targets.max(dim=1)[0] / torch.log2(torch.tensor(2.0))

        metrics[f'NDCG@{k}'] = (dcg / (idcg + 1e-8)).mean().item()

        # MRR@K
        mrr = torch.zeros(predictions.size(0))
        for i in range(min(k, predictions.size(1))):
            rel = torch.gather(targets, 1, indices[:, i:i+1]).squeeze()
            mrr += rel / (i + 1)
        metrics[f'MRR@{k}'] = mrr.mean().item()

    return metrics


def save_sm_checkpoint(model, config, epoch, best_metric, is_best=False):
    """保存SM模型检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'best_metric': best_metric,
        'config_dict': config.__dict__ if hasattr(config, '__dict__') else config,
        'sm_framework': True
    }

    model_name = config.model_name.replace('sm_', '')
    filename = f'sm_{model_name}_checkpoint.pth'
    if is_best:
        filename = f'sm_{model_name}_best.pth'

    filepath = os.path.join(config.model_save_dir, filename)
    torch.save(checkpoint, filepath)

    if is_best:
        print(f"Best SM model saved to {filepath}")


def load_sm_checkpoint(model, config):
    """加载SM模型检查点"""
    model_name = config.model_name.replace('sm_', '')
    filepath = os.path.join(config.model_save_dir, f'sm_{model_name}_best.pth')

    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best SM model from {filepath}")
        return model
    else:
        print(f"No SM checkpoint found at {filepath}")
        return model


# 定义全局变量（用于向后兼容）
main_params = []
disc_params = []


@torch.no_grad()
def validate_sm_model_batch(model, val_loader, config):
    """Final SM validation with full-ranking evaluation."""
    model.eval()

    ranking_results = evaluate_full_ranking_loader(
        data_loader=val_loader,
        config=config,
        predict_fn=lambda batch: model.predict(
            batch['input_seq'],
            sensitive_mask=(
                torch.rand(
                    batch['input_seq'].size(0),
                    len(config.sensitive_attributes),
                    device=batch['input_seq'].device,
                ) > 0.5
            ).float(),
        ),
        sensitive_attributes=[],
        desc='Validation-FullRanking-SM',
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
