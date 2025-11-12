import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import time


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

    # 修复：移除verbose参数，手动打印学习率变化
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8,
        min_lr=1e-6
    )

    # 混合精度训练
    scaler = GradScaler(enabled=config.use_mixed_precision)

    # 训练状态
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

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

            # 公平性增强
            if augmenter is not None and config.use_fairness_augmentation:
                with autocast(enabled=config.use_mixed_precision):
                    batch = augmenter.augment_batch_optimized(batch)

            # 前向传播
            with autocast(enabled=config.use_mixed_precision):
                input_seq = batch['input_seq']
                target = batch['target'].squeeze(-1)

                loss = model.compute_loss(input_seq, target)

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
        train_losses.append(avg_train_loss)

        # ========== 验证阶段 ==========
        val_loss = validate_model_optimized(model, val_loader, config)
        val_losses.append(val_loss)

        # 调整学习率
        scheduler.step(val_loss)
        
        # 手动检查并打印学习率变化
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != last_lr:
            print(f'  -> Learning rate reduced from {last_lr:.2e} to {current_lr:.2e}')
            last_lr = current_lr

        # 计算epoch时间
        epoch_time = time.time() - start_time

        print(f'Epoch {epoch + 1}/{config.num_epochs} ({epoch_time:.1f}s):')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # 保存最佳模型
            model_to_save = model.module if hasattr(model, 'module') else model
            model_name = f'best_{config.model_name}_{"with" if config.use_fairness_augmentation else "without"}_fairness.pt'
            torch.save(model_to_save.state_dict(), os.path.join(config.processed_data_dir, model_name))
            print(f'  -> Best model saved! (Loss: {best_val_loss:.4f})')
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break

        print('-' * 60)

        # 清理GPU缓存
        torch.cuda.empty_cache()

    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }

    return history


@torch.no_grad()
def validate_model_optimized(model, val_loader, config):
    """GPU优化的模型验证"""
    model.eval()
    total_loss = 0
    num_batches = 0

    val_pbar = tqdm(val_loader, desc='Validation', leave=False)

    for batch in val_pbar:
        with autocast(enabled=config.use_mixed_precision):
            input_seq = batch['input_seq']
            target = batch['target'].squeeze(-1)

            loss = model.compute_loss(input_seq, target)

        total_loss += loss.item()
        num_batches += 1

        val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    return total_loss / num_batches


def load_best_model_optimized(model, config):
    """加载最佳模型"""
    model_name = f'best_{config.model_name}_{"with" if config.use_fairness_augmentation else "without"}_fairness.pt'
    model_path = os.path.join(config.processed_data_dir, model_name)

    if os.path.exists(model_path):
        # 加载到正确的设备
        state_dict = torch.load(model_path, map_location=config.device)

        # 处理DataParallel的状态字典
        if list(state_dict.keys())[0].startswith('module.'):
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v  # 移除 'module.' 前缀
            state_dict = new_state_dict

        model.load_state_dict(state_dict)
        print(f'Loaded best model from {model_path}')
    else:
        print(f'No saved model found at {model_path}')

    return model