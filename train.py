import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import time
from collections import defaultdict

from evaluation.full_ranking import evaluate_full_ranking_loader


def train_model_optimized(model, train_loader, val_loader, config, augmenter=None):
    """GPU优化的模型训练"""
    print(f"Training {config.model_name.upper()} model with GPU optimization...")

    # 使用DataParallel进行多GPU训练
    if config.use_data_parallel and config.num_gpus > 1:
        print(f"Using DataParallel with {config.num_gpus} GPUs")
        model = nn.DataParallel(model)

    # 优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.l2_emb,
        eps=1e-8,
        betas=(0.9, 0.999)
    )

    # 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8,
        min_lr=1e-6
    )

    # 混合精度训练
    scaler = GradScaler(enabled=config.use_mixed_precision)

    # 训练状态
    best_val_loss = float('inf')
    best_val_recall = 0.0  # 添加最佳召回率跟踪
    patience_counter = 0
    history = defaultdict(list)  # 使用defaultdict存储更多指标

    # 预热学习率
    warmup_steps = len(train_loader) * 2  # 前2个epoch预热

    # 记录上一个学习率
    last_lr = optimizer.param_groups[0]['lr']

    for epoch in range(config.num_epochs):
        start_time = time.time()

        # ========== 训练阶段 ==========
        model.train()
        epoch_train_loss = 0
        num_batches = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs} [Train]')

        for batch_idx, batch in enumerate(train_pbar):
            # 学习率预热
            if epoch * len(train_loader) + batch_idx < warmup_steps:
                lr_scale = min(1.0, (epoch * len(train_loader) + batch_idx + 1) / warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config.learning_rate * lr_scale

            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

            # 公平性增强（兼容性改进）
            if augmenter is not None and config.use_fairness_augmentation:
                with autocast(enabled=config.use_mixed_precision):
                    # 检查augmenter方法
                    if hasattr(augmenter, 'augment_batch_optimized'):
                        batch = augmenter.augment_batch_optimized(batch)
                    elif hasattr(augmenter, 'augment_batch'):
                        batch = augmenter.augment_batch(batch)

            # 前向传播（改进的维度处理）
            with autocast(enabled=config.use_mixed_precision):
                input_seq = batch['input_seq'].to(config.device)
                target = batch['target'].to(config.device)

                # 确保target是一维的
                if target.dim() > 1:
                    target = target.squeeze(-1)

                # 兼容不同的损失计算方式
                if hasattr(model, 'compute_loss'):
                    loss = model.compute_loss(input_seq, target)
                else:
                    # 基础的交叉熵损失
                    logits = model.predict(input_seq)
                    loss = nn.functional.cross_entropy(logits, target, ignore_index=0)

            # 反向传播
            scaler.scale(loss).backward()

            # 梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)

            # 优化器步骤
            scaler.step(optimizer)
            scaler.update()

            epoch_train_loss += loss.item()
            num_batches += 1

            # 更新进度条
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

            # 定期清理GPU缓存
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()

        avg_train_loss = epoch_train_loss / num_batches
        history['train_loss'].append(avg_train_loss)

        # ========== 验证阶段 ==========
        val_metrics = validate_model_optimized(model, val_loader, config)
        val_loss = val_metrics['loss']

        # 记录所有验证指标
        for k, v in val_metrics.items():
            history[f'val_{k}'].append(v)

        # 调整学习率
        scheduler.step(val_loss)

        # 检查并打印学习率变化
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != last_lr:
            print(f'  -> Learning rate reduced from {last_lr:.2e} to {current_lr:.2e}')
            last_lr = current_lr

        # 计算epoch时间
        epoch_time = time.time() - start_time

        print(f'Epoch {epoch + 1}/{config.num_epochs} ({epoch_time:.1f}s):')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        if 'recall@10' in val_metrics:
            print(f'  Val Recall@10: {val_metrics["recall@10"]:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')

        # 早停检查（支持基于loss或recall）
        current_metric = val_metrics.get('recall@10', -val_loss)  # 如果有recall用recall，否则用负loss
        is_best = False

        if 'recall@10' in val_metrics:
            # 基于recall的早停
            if val_metrics['recall@10'] > best_val_recall:
                best_val_recall = val_metrics['recall@10']
                patience_counter = 0
                is_best = True
        else:
            # 基于loss的早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                is_best = True

        if is_best:
            # 保存最佳模型
            save_checkpoint(model, config, epoch, val_metrics, is_best=True)
            print(f'  -> Best model saved!')
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

        print('-' * 60)

        # 清理GPU缓存
        torch.cuda.empty_cache()

    # 保存训练历史
    history_path = os.path.join(config.processed_data_dir, f'{config.model_name}_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(dict(history), f)
    print(f'Training history saved to {history_path}')

    return dict(history)


@torch.no_grad()
def validate_model_optimized(model, val_loader, config):
    """GPU优化的模型验证，返回多个指标"""
    model.eval()
    total_loss = 0
    num_batches = 0

    # 用于计算额外指标
    val_pbar = tqdm(val_loader, desc='Validation', leave=False)

    for batch in val_pbar:
        with autocast(enabled=config.use_mixed_precision):
            input_seq = batch['input_seq'].to(config.device)
            target = batch['target'].to(config.device)

            # 确保target是一维的
            if target.dim() > 1:
                target = target.squeeze(-1)

            # 计算损失
            if hasattr(model, 'compute_loss'):
                loss = model.compute_loss(input_seq, target)
            else:
                logits = model.predict(input_seq)
                loss = nn.functional.cross_entropy(logits, target, ignore_index=0)

            # 如果有候选物品，计算排序指标
            if 'negative_items' in batch and batch['negative_items'].size(1) > 0:
                pos_items = batch['target'].unsqueeze(1)
                neg_items = batch['negative_items']
                candidate_items = torch.cat([pos_items, neg_items], dim=1)

                # 获取预测分数
                if hasattr(model, 'predict') and model.predict.__code__.co_argcount > 2:
                    # 模型支持候选物品预测
                    scores = model.predict(input_seq, candidate_items)
                    all_predictions.append(scores.cpu())

                    # 标签：第一个是正样本
                    labels = torch.zeros_like(scores)
                    labels[:, 0] = 1
                    all_targets.append(labels.cpu())

        total_loss += loss.item()
        num_batches += 1

        val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    # 基础指标
    metrics = {'loss': total_loss / num_batches}
    ranking_results = evaluate_full_ranking_loader(
        data_loader=val_loader,
        config=config,
        predict_fn=lambda batch: model.predict(batch['input_seq']),
        sensitive_attributes=[],
        desc='Validation-FullRanking',
        legacy_output=False,
    )
    utility = ranking_results['utility']
    metrics['recall@10'] = utility.get('Recall@10', 0.0)
    metrics['hitrate@10'] = utility.get('HitRate@10', 0.0)
    metrics['ndcg@10'] = utility.get('NDCG@10', 0.0)
    return metrics

    # 如果有预测结果，计算排序指标
    if all_predictions:
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)

        # 计算Recall@10
        _, indices = torch.sort(predictions, dim=1, descending=True)
        top_10 = indices[:, :10]
        hits = torch.gather(targets, 1, top_10).sum(dim=1)
        recall_10 = hits.float().mean().item()
        metrics['recall@10'] = recall_10

        # 可以添加更多指标
        # metrics['ndcg@10'] = compute_ndcg(predictions, targets, 10)

    return metrics


def save_checkpoint(model, config, epoch, metrics, is_best=False):
    """保存模型检查点"""
    # 处理DataParallel
    model_to_save = model.module if hasattr(model, 'module') else model

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'metrics': metrics,
        'config': config.__dict__ if hasattr(config, '__dict__') else config
    }

    # 文件名
    if is_best:
        filename = f'best_{config.model_name}_{"with" if config.use_fairness_augmentation else "without"}_fairness.pt'
    else:
        filename = f'{config.model_name}_epoch_{epoch}.pt'

    filepath = os.path.join(config.processed_data_dir, filename)
    torch.save(checkpoint, filepath)

    # 如果是最佳模型，额外保存一份标准命名的版本（用于兼容性）
    if is_best:
        simple_filename = f'{config.model_name}_best.pth'
        simple_filepath = os.path.join(config.model_save_dir, simple_filename)
        os.makedirs(config.model_save_dir, exist_ok=True)
        torch.save(checkpoint, simple_filepath)


def load_best_model_optimized(model, config):
    """加载最佳模型（改进的兼容性）"""
    # 尝试多个可能的路径
    possible_paths = [
        os.path.join(config.processed_data_dir,
                     f'best_{config.model_name}_{"with" if config.use_fairness_augmentation else "without"}_fairness.pt'),
        os.path.join(config.model_save_dir, f'{config.model_name}_best.pth'),
        os.path.join(config.processed_data_dir, f'{config.model_name}_best.pt')
    ]

    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break

    if model_path:
        # 加载检查点
        checkpoint = torch.load(model_path, map_location=config.device)

        # 获取state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            print(f'Loaded checkpoint from {model_path} (epoch {checkpoint.get("epoch", "?")})')
        else:
            # 旧格式，直接是state_dict
            state_dict = checkpoint
            print(f'Loaded model state from {model_path}')

        # 处理DataParallel的状态字典
        if list(state_dict.keys())[0].startswith('module.'):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v  # 移除 'module.' 前缀
            state_dict = new_state_dict

        model.load_state_dict(state_dict)
    else:
        print(f'No saved model found. Searched paths: {possible_paths}')

    return model


def save_checkpoint(model, config, epoch, metrics, is_best=False):
    """保存模型检查点"""
    # 先定义文件路径
    if is_best:
        filename = f'best_{config.model_name}_{"with" if config.use_fairness_augmentation else "without"}_fairness.pt'
    else:
        filename = f'{config.model_name}_epoch_{epoch}.pt'

    filepath = os.path.join(config.processed_data_dir, filename)

    # 处理DataParallel
    model_to_save = model.module if hasattr(model, 'module') else model

    # 创建checkpoint字典
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_to_save.state_dict(),
        'metrics': metrics,
        'config': config.__dict__ if hasattr(config, '__dict__') else config
    }

    # 保存checkpoint
    try:
        torch.save(checkpoint, filepath)

        # 如果是最佳模型，额外保存一份标准命名的版本（用于兼容性）
        if is_best:
            simple_filename = f'{config.model_name}_best.pth'
            simple_filepath = os.path.join(config.model_save_dir, simple_filename)
            os.makedirs(config.model_save_dir, exist_ok=True)
            torch.save(checkpoint, simple_filepath)

    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        # 尝试保存到备用位置
        backup_path = filepath.replace('.pt', '_backup.pt')
        try:
            torch.save(checkpoint, backup_path)
            print(f"Checkpoint saved to backup location: {backup_path}")
        except Exception as e2:
            print(f"Failed to save to backup location: {e2}")


@torch.no_grad()
def validate_model_optimized(model, val_loader, config):
    """Final full-ranking validation entrypoint shared by vanilla/fair variants."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    val_pbar = tqdm(val_loader, desc='Validation', leave=False)
    for batch in val_pbar:
        with autocast(enabled=config.use_mixed_precision):
            input_seq = batch['input_seq'].to(config.device)
            target = batch['target'].to(config.device)
            if target.dim() > 1:
                target = target.squeeze(-1)

            if hasattr(model, 'compute_loss'):
                loss = model.compute_loss(input_seq, target)
            else:
                logits = model.predict(input_seq)
                loss = nn.functional.cross_entropy(logits, target, ignore_index=0)

        total_loss += loss.item()
        num_batches += 1
        val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    metrics = {'loss': total_loss / max(num_batches, 1)}
    ranking_results = evaluate_full_ranking_loader(
        data_loader=val_loader,
        config=config,
        predict_fn=lambda batch: model.predict(batch['input_seq']),
        sensitive_attributes=[],
        desc='Validation-FullRanking',
        legacy_output=False,
    )
    utility = ranking_results['utility']
    metrics['recall@10'] = utility.get('Recall@10', 0.0)
    metrics['hitrate@10'] = utility.get('HitRate@10', 0.0)
    metrics['ndcg@10'] = utility.get('NDCG@10', 0.0)
    return metrics
