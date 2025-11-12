import torch
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import warnings
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats


class RecommendationMetrics:
    """
    推荐系统评估指标计算类
    支持GPU加速的向量化计算
    """

    def __init__(self, device: torch.device = None):
        self.device = device if device else torch.device('cpu')

    def compute_topk_metrics(self,
                             predictions: Union[torch.Tensor, np.ndarray],
                             targets: Union[torch.Tensor, np.ndarray],
                             k_list: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        计算Top-K推荐指标
        """
        # 转换为torch tensor
        if isinstance(predictions, np.ndarray):
            predictions = torch.from_numpy(predictions).to(self.device)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets).to(self.device)

        predictions = predictions.to(self.device)
        targets = targets.to(self.device)

        # 检查预测维度
        assert predictions.ndim == 2, f"Predictions shape should be [B, N], but got {predictions.shape}"

        if targets.ndim == 1:
            targets = targets.unsqueeze(1)
        elif targets.ndim == 2 and targets.size(1) == 0:
            return {f'{m}@{k}': 0.0 for m in ['Recall', 'Precision', 'NDCG', 'HR', 'MRR'] for k in k_list}

        metrics = {}

        for k in k_list:
            recall_k = self._compute_recall_at_k(predictions, targets, k)
            metrics[f'Recall@{k}'] = recall_k

            precision_k = self._compute_precision_at_k(predictions, targets, k)
            metrics[f'Precision@{k}'] = precision_k

            ndcg_k = self._compute_ndcg_at_k(predictions, targets, k)
            metrics[f'NDCG@{k}'] = ndcg_k

            hr_k = self._compute_hit_rate_at_k(predictions, targets, k)
            metrics[f'HR@{k}'] = hr_k

            mrr_k = self._compute_mrr_at_k(predictions, targets, k)
            metrics[f'MRR@{k}'] = mrr_k

        return metrics

    def _compute_recall_at_k(self, predictions: torch.Tensor,
                             targets: torch.Tensor, k: int) -> float:
        batch_size = predictions.size(0)
        _, top_k_indices = torch.topk(predictions, k, dim=1)

        # 处理 one-hot 或 binary 标签矩阵
        if targets.size(1) == predictions.size(1):
            # 提取正样本索引位置
            true_indices = torch.argmax(targets, dim=1, keepdim=True)
            hits = torch.any(top_k_indices == true_indices, dim=1).float()
        else:
            hits = torch.any(top_k_indices == targets, dim=1).float()

        return torch.mean(hits).item()

    def _compute_precision_at_k(self, predictions: torch.Tensor,
                                targets: torch.Tensor, k: int) -> float:
        _, top_k_indices = torch.topk(predictions, k, dim=1)

        if targets.size(1) == predictions.size(1):
            true_indices = torch.argmax(targets, dim=1, keepdim=True)
            hits = torch.any(top_k_indices == true_indices, dim=1).float()
        else:
            hits = torch.any(top_k_indices == targets, dim=1).float()

        return torch.mean(hits).item()

    def _compute_ndcg_at_k(self, predictions: torch.Tensor,
                           targets: torch.Tensor, k: int) -> float:
        batch_size = predictions.size(0)
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        is_single_label = targets.size(1) == 1
        ndcg_scores = []

        for i in range(batch_size):
            if targets.size(1) == predictions.size(1):  # one-hot 情况
                user_targets = [torch.argmax(targets[i]).item()]
            else:
                user_targets = targets[i].tolist() if targets.size(1) > 1 else [targets[i].item()]

            user_targets = [t for t in user_targets if t >= 0]
            if not user_targets:
                continue
            user_top_k = top_k_indices[i].tolist()
            dcg = sum(1.0 / math.log2(rank + 2) for rank, item in enumerate(user_top_k) if item in user_targets)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(user_targets), k)))
            ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

        return np.mean(ndcg_scores) if ndcg_scores else 0.0

    def _compute_hit_rate_at_k(self, predictions: torch.Tensor,
                               targets: torch.Tensor, k: int) -> float:
        return self._compute_recall_at_k(predictions, targets, k)

    def _compute_mrr_at_k(self, predictions: torch.Tensor,
                          targets: torch.Tensor, k: int) -> float:
        batch_size = predictions.size(0)
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        is_single_label = targets.size(1) == 1
        mrr_scores = []

        for i in range(batch_size):
            if targets.size(1) == predictions.size(1):  # one-hot 情况
                user_targets = [torch.argmax(targets[i]).item()]
            else:
                user_targets = targets[i].tolist() if targets.size(1) > 1 else [targets[i].item()]
            user_targets = [t for t in user_targets if t >= 0]
            user_top_k = top_k_indices[i].tolist()
            rr = next((1.0 / (rank + 1) for rank, item in enumerate(user_top_k) if item in user_targets), 0.0)
            mrr_scores.append(rr)

        return np.mean(mrr_scores) if mrr_scores else 0.0


class FairnessMetrics:
    """
    公平性指标计算类
    """

    def __init__(self, sensitive_attributes: List[str]):
        self.sensitive_attributes = sensitive_attributes

    def compute_demographic_parity(self,
                                   predictions: np.ndarray,
                                   sensitive_attr: np.ndarray,
                                   k: int = 10) -> Dict[str, float]:
        unique_groups = np.unique(sensitive_attr)

        if len(unique_groups) < 2:
            return {'demographic_parity': 1.0, 'group_rates': {}}

        group_rates = {}

        for group in unique_groups:
            group_mask = (sensitive_attr == group)
            group_predictions = predictions[group_mask]

            if len(group_predictions) == 0:
                continue

            top_k_probs = []
            for pred in group_predictions:
                top_k_items = np.argsort(pred)[-k:]
                top_k_prob = np.mean(pred[top_k_items])
                top_k_probs.append(top_k_prob)

            group_rates[int(group)] = np.mean(top_k_probs)

        if len(group_rates) < 2:
            return {'demographic_parity': 1.0, 'group_rates': group_rates}

        rates = list(group_rates.values())
        max_diff = max(rates) - min(rates)

        # 新增指标
        spd = rates[0] - rates[1]  # Statistical Parity Difference
        di = rates[0] / (rates[1] + 1e-8)  # Disparate Impact

        return {
            'demographic_parity': max(0, 1 - max_diff),
            'group_rates': group_rates,
            'max_difference': max_diff,
            'statistical_parity_difference': spd,
            'disparate_impact': di
        }

    def compute_equalized_opportunity(self,
                                      predictions: np.ndarray,
                                      targets: np.ndarray,
                                      sensitive_attr: np.ndarray,
                                      k: int = 10) -> Dict[str, float]:
        """
        计算机会均等 (Equalized Opportunity)
        """
        unique_groups = np.unique(sensitive_attr)

        if len(unique_groups) < 2:
            return {'equalized_opportunity': 1.0, 'group_tpr': {}}

        group_tpr = {}

        for group in unique_groups:
            group_mask = (sensitive_attr == group)
            group_predictions = predictions[group_mask]
            group_targets = targets[group_mask]

            if len(group_predictions) == 0:
                continue

            hits = []
            for i, pred in enumerate(group_predictions):
                target_vec = group_targets[i]
                if len(target_vec.shape) > 0 and target_vec.shape[0] == pred.shape[0]:
                    target = int(np.argmax(target_vec))
                else:
                    target = int(target_vec)

                top_k_items = np.argsort(pred)[-k:]
                hit = 1 if target in top_k_items else 0
                hits.append(hit)

            group_tpr[int(group)] = np.mean(hits) if hits else 0.0

        if len(group_tpr) < 2:
            return {'equalized_opportunity': 1.0, 'group_tpr': group_tpr}

        tpr_values = list(group_tpr.values())
        max_diff = max(tpr_values) - min(tpr_values)
        fairness_score = 1 - max_diff

        return {
            'equalized_opportunity': max(0, fairness_score),
            'group_tpr': group_tpr,
            'max_difference': max_diff
        }

    def compute_individual_fairness(self,
                                    predictions: np.ndarray,
                                    user_similarities: np.ndarray) -> float:
        """
        计算个体公平性
        相似用户应该得到相似推荐
        """
        if predictions.shape[0] != user_similarities.shape[0]:
            raise ValueError("Predictions and similarities dimension mismatch")

        # 计算推荐相似度
        pred_similarities = np.corrcoef(predictions)

        # 计算用户相似度和推荐相似度的相关性
        # 展平上三角矩阵
        n = user_similarities.shape[0]
        user_sim_flat = []
        pred_sim_flat = []

        for i in range(n):
            for j in range(i + 1, n):
                user_sim_flat.append(user_similarities[i, j])
                pred_sim_flat.append(pred_similarities[i, j])

        # 计算皮尔逊相关系数
        if len(user_sim_flat) > 1:
            correlation, _ = stats.pearsonr(user_sim_flat, pred_sim_flat)
            return max(0, correlation)  # 确保非负
        else:
            return 0.0

    def compute_diversity_metrics(self,
                                  predictions: np.ndarray,
                                  item_features: Optional[np.ndarray] = None,
                                  k: int = 10) -> Dict[str, float]:
        """
        计算推荐多样性指标
        """
        metrics = {}

        # 1. Intra-list Diversity (ILD)
        if item_features is not None:
            ild_scores = []
            for pred in predictions:
                top_k_items = np.argsort(pred)[-k:]
                top_k_features = item_features[top_k_items]

                # 计算特征间的平均距离
                distances = []
                for i in range(len(top_k_features)):
                    for j in range(i + 1, len(top_k_features)):
                        dist = np.linalg.norm(top_k_features[i] - top_k_features[j])
                        distances.append(dist)

                ild = np.mean(distances) if distances else 0.0
                ild_scores.append(ild)

            metrics['intra_list_diversity'] = np.mean(ild_scores)

        # 2. Coverage
        all_recommended_items = set()
        for pred in predictions:
            top_k_items = np.argsort(pred)[-k:]
            all_recommended_items.update(top_k_items)

        total_items = predictions.shape[1]
        coverage = len(all_recommended_items) / total_items
        metrics['coverage'] = coverage

        # 3. Gini Index (衡量推荐分布的不平等程度)
        item_counts = defaultdict(int)
        for pred in predictions:
            top_k_items = np.argsort(pred)[-k:]
            for item in top_k_items:
                item_counts[item] += 1

        counts = list(item_counts.values())
        if len(counts) > 1:
            counts.sort()
            n = len(counts)
            gini = sum((2 * i - n - 1) * count for i, count in enumerate(counts, 1)) / (n * sum(counts))
            metrics['gini_index'] = gini
        else:
            metrics['gini_index'] = 0.0

        return metrics


class MetricCalculator:
    """
    综合指标计算器
    整合推荐性能指标和公平性指标
    """

    def __init__(self, device: torch.device = None):
        self.device = device if device else torch.device('cpu')
        self.rec_metrics = RecommendationMetrics(device)

    def compute_all_metrics(self,
                            predictions: Union[torch.Tensor, np.ndarray],
                            targets: Union[torch.Tensor, np.ndarray],
                            sensitive_attrs: Dict[str, np.ndarray] = None,
                            k_list: List[int] = [5, 10, 20]) -> Dict[str, Dict]:
        """
        计算所有指标

        Args:
            predictions: 预测矩阵
            targets: 目标矩阵
            sensitive_attrs: 敏感属性字典
            k_list: K值列表

        Returns:
            包含性能指标和公平性指标的字典
        """
        results = {}

        # 1. 计算推荐性能指标
        performance_metrics = self.rec_metrics.compute_topk_metrics(
            predictions, targets, k_list)
        results['performance'] = performance_metrics

        # 2. 计算公平性指标
        if sensitive_attrs is not None:
            fairness_results = {}

            # 转换预测为numpy数组
            if isinstance(predictions, torch.Tensor):
                pred_np = predictions.cpu().numpy()
            else:
                pred_np = predictions

            if isinstance(targets, torch.Tensor):
                targets_np = targets.cpu().numpy()
            else:
                targets_np = targets

            for attr_name, attr_values in sensitive_attrs.items():
                fairness_calc = FairnessMetrics([attr_name])

                # Demographic Parity
                dp_result = fairness_calc.compute_demographic_parity(
                    pred_np, attr_values, k=k_list[1])  # 使用中间的k值

                # Equalized Opportunity
                eo_result = fairness_calc.compute_equalized_opportunity(
                    pred_np, targets_np, attr_values, k=k_list[1])

                fairness_results[attr_name] = {
                    'demographic_parity': dp_result['demographic_parity'],
                    'equalized_opportunity': eo_result['equalized_opportunity'],
                    'statistical_parity_difference': dp_result.get('statistical_parity_difference', np.nan),
                    'disparate_impact': dp_result.get('disparate_impact', np.nan),
                    'group_rates': dp_result['group_rates'],
                    'group_tpr': eo_result['group_tpr']
                }

            results['fairness'] = fairness_results

        return results

    def print_metrics_summary(self, metrics: Dict[str, Dict]):
        """打印指标摘要"""
        print("\n" + "=" * 80)
        print("METRICS SUMMARY")
        print("=" * 80)

        # 性能指标
        if 'performance' in metrics:
            print("\nPerformance Metrics:")
            print("-" * 40)
            for metric, value in metrics['performance'].items():
                print(f"{metric:15s}: {value:.4f}")

        # 公平性指标
        if 'fairness' in metrics:
            print("\nFairness Metrics:")
            print("-" * 40)
            for attr, attr_metrics in metrics['fairness'].items():
                print(f"\n{attr.upper()} Fairness:")
                for metric, value in attr_metrics.items():
                    if isinstance(value, dict):
                        print(f"  {metric}: {value}")
                    else:
                        print(f"  {metric:25s}: {value:.4f}")

        print("=" * 80)


# 便捷函数
def calculate_metrics(predictions, targets, sensitive_attrs=None, k_list=[5, 10, 20], device=None):
    """
    便捷的指标计算函数
    """
    calculator = MetricCalculator(device)
    return calculator.compute_all_metrics(predictions, targets, sensitive_attrs, k_list)


def print_metrics(metrics):
    """
    便捷的指标打印函数
    """
    calculator = MetricCalculator()
    calculator.print_metrics_summary(metrics)