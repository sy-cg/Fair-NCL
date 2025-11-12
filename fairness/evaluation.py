import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.cuda.amp import autocast

# 在文件开头添加导入
from utils.metrics import MetricCalculator, FairnessMetrics


class OptimizedFairnessEvaluator:
    """GPU优化的公平性评估器"""

    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.sensitive_attributes = config.sensitive_attributes
        self.topk_list = config.topk_list

        # 使用新的指标计算器
        self.metric_calculator = MetricCalculator(device=config.device)

    @torch.no_grad()
    def evaluate_model_optimized(self, model, data_loader):
        """每用户 1 正 + 99 负 的 Top-K 推荐评估（最小化修改）"""
        print("Evaluating model with 1 pos + 99 neg items per user ...")
        model.eval()

        all_predictions = []
        all_targets = []
        all_user_ids = []
        user_groups = defaultdict(list)

        eval_pbar = tqdm(data_loader, desc='Evaluation')

        with autocast(enabled=self.config.use_mixed_precision):
            for batch in eval_pbar:
                input_seq = batch['input_seq'].to(self.device)
                pos_items = batch['target'].to(self.device)  # [B]
                neg_items = batch['negative_items'].to(self.device)  # [B, 99]

                candidate_items = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)  # [B, 100]

                # 模型预测：对每个用户的 100 个物品打分
                logits = model.predict(input_seq, candidate_items)  # [B, 100]
                assert logits.shape == candidate_items.shape, f"Expected logits of shape {candidate_items.shape}, got {logits.shape}"

                all_predictions.append(logits.cpu())

                # 构造标签：第一个位置为正样本，其余为负样本
                labels = torch.zeros_like(logits, dtype=torch.long)
                labels[:, 0] = 1
                all_targets.append(labels.cpu())

                all_user_ids.extend(batch['user_id'].cpu().numpy())

                # 收集敏感属性
                for attr in self.sensitive_attributes:
                    if attr in batch:
                        user_groups[attr].extend(batch[attr].cpu().numpy())

        # 合并预测和标签
        predictions = torch.cat(all_predictions, dim=0)  # [N, 100]
        targets = torch.cat(all_targets, dim=0)  # [N, 100]

        sensitive_attrs = {
            attr: np.array(user_groups[attr]) for attr in self.sensitive_attributes
        }

        results = self.metric_calculator.compute_all_metrics(
            predictions=predictions,
            targets=targets,
            sensitive_attrs=sensitive_attrs,
            k_list=self.config.topk_list
        )

        torch.cuda.empty_cache()
        return results

    def print_evaluation_results(self, results):
        """打印评估结果"""
        self.metric_calculator.print_metrics_summary(results)

    def compute_group_metrics(self, predictions, targets, group_indices, k_list):
        """计算特定群体的推荐指标"""
        group_predictions = predictions[group_indices]
        group_targets = targets[group_indices]

        if len(group_predictions) == 0:
            return {}

        # 使用指标计算器
        metrics = self.metric_calculator.rec_metrics.compute_topk_metrics(
            group_predictions, group_targets, k_list
        )

        return metrics

    # 在 evaluation.py 的 save_evaluation_results 方法中增强

    def save_evaluation_results(self, results, save_path):
        """保存评估结果为csv"""
        import pandas as pd
        import os

        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # 展平字典
        flat_results = {}

        # 展平实验参数
        if 'experiment_params' in results:
            for key, value in results['experiment_params'].items():
                flat_results[f'param_{key}'] = value

        # 展平性能指标
        for section, section_values in results.items():
            if section == 'experiment_params':
                continue

            if isinstance(section_values, dict):
                for key, value in section_values.items():
                    if isinstance(value, dict):
                        # 处理嵌套字典（如公平性指标）
                        for subkey, subvalue in value.items():
                            if not isinstance(subvalue, dict):  # 跳过group_rates等嵌套字典
                                flat_results[f"{section}.{key}.{subkey}"] = subvalue
                    else:
                        flat_results[f"{section}.{key}"] = value
            else:
                flat_results[section] = section_values

        # 创建DataFrame并保存
        df = pd.DataFrame([flat_results])
        df.to_csv(save_path, index=False)
        print(f"Evaluation results saved to {save_path}")

        # 同时保存一个更易读的版本
        readable_path = save_path.replace('.csv', '_readable.txt')
        with open(readable_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EXPERIMENT RESULTS SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            # 实验参数
            if 'experiment_params' in results:
                f.write("Experiment Parameters:\n")
                f.write("-" * 40 + "\n")
                for key, value in results['experiment_params'].items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")

            # 性能指标
            if 'performance' in results:
                f.write("Performance Metrics:\n")
                f.write("-" * 40 + "\n")
                for key, value in results['performance'].items():
                    f.write(f"  {key}: {value:.4f}\n")
                f.write("\n")

            # 公平性指标
            if 'fairness' in results:
                f.write("Fairness Metrics:\n")
                f.write("-" * 40 + "\n")
                for attr, metrics in results['fairness'].items():
                    f.write(f"\n  {attr.upper()}:\n")
                    for metric, value in metrics.items():
                        if not isinstance(value, dict):
                            f.write(f"    {metric}: {value:.4f}\n")
