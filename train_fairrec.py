import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import time
import os
from collections import defaultdict


def train_fairrec_model(model, train_loader, val_loader, config):
    """
    FairRec 训练函数 (Alternating Optimization)
    Step A: 训练 Adversaries 识别 uf 中的敏感信息
    Step B: 训练 Encoder + Recommendation 欺骗 Adversaries 并完成推荐
    """
    print(
        f"\nStarting FairRec training (Base: {config.base_model_name if hasattr(config, 'base_model_name') else 'sasrec'})...")

    # 1. 参数分组
    # Group 1: Main Parameters (Base Model + Projection + Bias Predictor + Rec Head)
    main_params = list(model.base_model.parameters()) + \
                  list(model.fairrec.projection.parameters()) + \
                  list(model.fairrec.bias_predictors.parameters()) + \
                  list(model.recommendation_head.parameters())

    # Group 2: Adversary Parameters (Disciminators)
    adv_params = list(model.fairrec.adversaries.parameters())

    main_optimizer = optim.Adam(main_params, lr=config.learning_rate)
    adv_optimizer = optim.Adam(adv_params, lr=config.learning_rate)  # 通常不需要 *2，保持一致即可

    scaler = GradScaler(enabled=config.use_mixed_precision)

    # 学习率调度
    main_scheduler = optim.lr_scheduler.CosineAnnealingLR(main_optimizer, T_max=config.num_epochs)

    history = defaultdict(list)
    best_val_recall = 0
    patience_counter = 0

    # 预热：前几轮先不强加对抗约束，让推荐模型先收敛一点？
    # 或者先训练判别器？FairRec论文通常同步训练。

    for epoch in range(config.num_epochs):
        model.train()
        epoch_losses = defaultdict(float)

        pbar = tqdm(train_loader, desc=f'FairRec Epoch {epoch + 1}/{config.num_epochs}')

        for batch in pbar:
            # Move data
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(config.device)

            # --- Step A: Train Adversaries (Discriminators) ---
            # 目标：最大化判别准确率 (Minimize BCE vs True Labels)
            adv_optimizer.zero_grad()
            with autocast(enabled=config.use_mixed_precision):
                adv_loss = model.compute_adversary_loss(batch)

            scaler.scale(adv_loss).backward()
            scaler.step(adv_optimizer)
            # 注意：这里不 update scaler，因为还要做 step B

            epoch_losses['adv_disc'] += adv_loss.item()

            # --- Step B: Train Main Model ---
            # 目标：最小化推荐损失 + 最小化 Bias Pred 损失 + 最大化 Adversary 疑惑度 (Minimize BCE vs 0.5)
            main_optimizer.zero_grad()
            with autocast(enabled=config.use_mixed_precision):
                total_loss, loss_dict = model.compute_loss(batch)

            scaler.scale(total_loss).backward()

            # Gradient Clipping
            scaler.unscale_(main_optimizer)
            torch.nn.utils.clip_grad_norm_(main_params, config.grad_clip)

            scaler.step(main_optimizer)
            scaler.update()  # update scalar once per batch

            for k, v in loss_dict.items():
                epoch_losses[k] += v.item()

            # Update bar
            if pbar.n % 10 == 0:
                pbar.set_postfix({
                    'rec': f"{loss_dict['rec']:.4f}",
                    'adv': f"{loss_dict['adv']:.4f}"
                })

        # --- Validation ---
        main_scheduler.step()

        # 记录 Epoch 平均 Loss
        for k, v in epoch_losses.items():
            history[k].append(v / len(train_loader))

        if (epoch + 1) % config.eval_interval == 0:
            # 这里的 validate_model_optimized 需要能够处理 FairRec 的 predict 接口
            # 我们在 model.predict 里已经做了兼容
            from train import validate_model_optimized
            val_metrics = validate_model_optimized(model, val_loader, config)

            recall = val_metrics['recall@10']
            print(f" -> Val Recall@10: {recall:.4f}")

            if recall > best_val_recall:
                best_val_recall = recall
                torch.save(model.state_dict(), os.path.join(config.model_save_dir, 'fairrec_best.pth'))
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config.patience:
                print("Early stopping.")
                break

    return history