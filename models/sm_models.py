"""
SM model registration and evaluation adapters.
"""

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from evaluation.full_ranking import evaluate_full_ranking_loader
from models.base_model import ModelRegistry
from models.sm_framework import SMFramework


@ModelRegistry.register("sm_sasrec")
class SM_SASRec(SMFramework):
    def __init__(self, config: Any):
        from models.sasrec import OptimizedSASRec

        base_model = OptimizedSASRec(config)
        super().__init__(base_model, config)

        print("SM_SASRec initialized:")
        print("  Base model: SASRec")
        print(f"  Hidden units: {self.hidden_units}")
        print(f"  Sensitive attributes: {self.sensitive_attributes}")
        print(f"  Number of filter combinations: {len(self.filter_module.filters)}")


@ModelRegistry.register("sm_bert4rec")
class SM_BERT4Rec(SMFramework):
    def __init__(self, config: Any):
        from models.bert4rec import OptimizedBERT4Rec

        base_model = OptimizedBERT4Rec(config)
        super().__init__(base_model, config)

        print("SM_BERT4Rec initialized:")
        print("  Base model: BERT4Rec")
        print(f"  Hidden units: {self.hidden_units}")
        print(f"  Sensitive attributes: {self.sensitive_attributes}")
        print(f"  Number of filter combinations: {len(self.filter_module.filters)}")


@ModelRegistry.register("sm_caser")
class SM_Caser(SMFramework):
    def __init__(self, config: Any):
        from models.caser import Caser

        base_model = Caser(config)
        super().__init__(base_model, config)

        print("SM_Caser initialized (Native 3D Output):")
        print("  Base model: Caser")
        print(f"  Filters: {len(self.filter_module.filters)}")


@ModelRegistry.register("sm_gru4rec")
class SM_GRU4Rec(SMFramework):
    def __init__(self, config: Any):
        from models.gru4rec import GRU4Rec

        base_model = GRU4Rec(config)
        super().__init__(base_model, config)


class SMEvaluationAdapter:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.sensitive_attributes = config.sensitive_attributes
        self.topk_list = config.topk_list
        self.test_combinations = self._generate_test_combinations()

    def _generate_test_combinations(self):
        import itertools

        combinations = [{}]
        for attr in self.sensitive_attributes:
            combinations.append({attr: True})

        for r in range(2, len(self.sensitive_attributes) + 1):
            for combo in itertools.combinations(self.sensitive_attributes, r):
                combinations.append({attr: attr in combo for attr in self.sensitive_attributes})

        return combinations

    @torch.no_grad()
    def evaluate_sm_model(self, model, data_loader, eval_mode="average"):
        print(f"Evaluating SM model with mode: {eval_mode}")

        if eval_mode in ("average", "all"):
            all_results = {}
            for mask in self.test_combinations:
                mask_name = self._get_mask_name(mask)
                print(f"\nEvaluating with {mask_name} mask...")
                all_results[mask_name] = self._evaluate_with_mask(model, data_loader, mask, mask_name)
            return self._compute_average_results(all_results) if eval_mode == "average" else all_results

        if eval_mode == "detailed":
            return self._detailed_evaluation(model, data_loader)

        default_mask = self.test_combinations[0] if self.test_combinations else {}
        return self._evaluate_with_mask(model, data_loader, default_mask, "default")

    def _get_mask_name(self, mask):
        if not mask:
            return "no_sensitive"
        sensitive_attrs = [attr for attr, is_sensitive in mask.items() if is_sensitive]
        if not sensitive_attrs:
            return "no_sensitive"
        return "_".join(sorted(sensitive_attrs)) + "_sensitive"

    def _evaluate_with_mask(self, model, data_loader, mask, mask_name):
        results = evaluate_full_ranking_loader(
            data_loader=data_loader,
            config=self.config,
            predict_fn=lambda batch: model.predict(batch["input_seq"], sensitive_mask=mask),
            sensitive_attributes=self.sensitive_attributes,
            desc=f"Eval {mask_name}",
            legacy_output=True,
        )
        results["sm_info"] = {
            "mask_name": mask_name,
            "mask": mask,
        }
        return results

    def _compute_average_results(self, all_results):
        avg_results = {"performance": {}, "fairness": {}}

        perf_metrics = [
            "Recall@5", "Recall@10", "Recall@20",
            "HitRate@5", "HitRate@10", "HitRate@20",
            "NDCG@5", "NDCG@10", "NDCG@20",
            "Precision@5", "Precision@10", "Precision@20",
            "MRR@5", "MRR@10", "MRR@20",
        ]

        for metric in perf_metrics:
            values = [results["performance"][metric] for results in all_results.values() if metric in results.get("performance", {})]
            if values:
                avg_results["performance"][metric] = float(np.mean(values))
                avg_results["performance"][f"{metric}_std"] = float(np.std(values))

        for attr in self.sensitive_attributes:
            avg_results["fairness"][attr] = {}
            for metric in ("demographic_parity", "equalized_opportunity"):
                values = [
                    results["fairness"][attr][metric]
                    for results in all_results.values()
                    if attr in results.get("fairness", {}) and metric in results["fairness"][attr]
                ]
                if values:
                    avg_results["fairness"][attr][metric] = float(np.mean(values))
                    avg_results["fairness"][attr][f"{metric}_std"] = float(np.std(values))

        avg_results["detailed_results"] = all_results
        return avg_results

    def _detailed_evaluation(self, model, data_loader):
        print("Performing detailed SM evaluation...")
        return {
            "filter_analysis": self._analyze_filter_effectiveness(model, data_loader),
            "discriminator_analysis": self._analyze_discriminator_performance(model, data_loader),
            "fairness_analysis": self._analyze_fairness_improvement(model, data_loader),
        }

    def _analyze_filter_effectiveness(self, model, data_loader):
        import torch.nn.functional as F

        filter_stats = defaultdict(lambda: {"count": 0, "avg_score": 0.0})
        model.eval()

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= 10:
                    break

                for combo_key in model.filter_module.filters.keys():
                    attrs = model.filter_module._decode_combo_key(combo_key)
                    mask = {attr: attr in attrs for attr in self.sensitive_attributes}
                    filtered_emb, original_emb = model(batch["input_seq"], mask)
                    similarity = F.cosine_similarity(filtered_emb, original_emb, dim=1).mean().item()
                    filter_stats[combo_key]["count"] += 1
                    filter_stats[combo_key]["avg_score"] += similarity

        for combo_key in filter_stats:
            count = filter_stats[combo_key]["count"]
            if count > 0:
                filter_stats[combo_key]["avg_score"] /= count

        return dict(filter_stats)

    def _analyze_discriminator_performance(self, model, data_loader):
        return {}

    def _analyze_fairness_improvement(self, model, data_loader):
        return {}

    def print_sm_evaluation_results(self, results):
        if "detailed_results" not in results:
            return self._print_metrics_summary(results)

        print("\n" + "=" * 80)
        print("SM EVALUATION RESULTS (AVERAGED)")
        print("=" * 80)
        self._print_metrics_summary(results)

        print("\n" + "=" * 80)
        print("DETAILED RESULTS BY SENSITIVE MASK")
        print("=" * 80)

        for mask_name, mask_results in results["detailed_results"].items():
            print(f"\n{mask_name.upper()}:")
            print(f"Mask: {mask_results['sm_info']['mask']}")
            perf = mask_results.get("performance", {})
            print(f"  Recall@10: {perf.get('Recall@10', 0):.4f}")
            print(f"  HitRate@10: {perf.get('HitRate@10', 0):.4f}")
            print(f"  NDCG@10: {perf.get('NDCG@10', 0):.4f}")

            for attr in self.sensitive_attributes:
                if attr in mask_results.get("fairness", {}):
                    fair = mask_results["fairness"][attr]
                    print(f"  {attr} DP: {fair.get('demographic_parity', 0):.4f}")
                    print(f"  {attr} EO: {fair.get('equalized_opportunity', 0):.4f}")

    def _print_metrics_summary(self, results):
        performance = results.get("performance", {})
        fairness = results.get("fairness", {})

        print("\nPerformance Metrics:")
        print("-" * 40)
        for metric, value in performance.items():
            if isinstance(value, (int, float, np.floating)):
                print(f"{metric:15s}: {float(value):.4f}")

        print("\nFairness Metrics:")
        print("-" * 40)
        for attr, attr_metrics in fairness.items():
            print(f"\n{attr.upper()} Fairness:")
            for metric, value in attr_metrics.items():
                if isinstance(value, (int, float, np.floating)):
                    print(f"{metric:28s}: {float(value):.4f}")
                elif isinstance(value, dict):
                    print(f"{metric:28s}: {value}")

    def save_sm_evaluation_results(self, results, save_path):
        import os
        import pandas as pd
        import pickle

        pkl_path = os.path.splitext(save_path)[0] + ".pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(results, f)
        print(f"SM detailed results saved to {pkl_path}")

        if "performance" in results and "fairness" in results:
            csv_path = os.path.splitext(save_path)[0] + ".csv"
            flat_results = {}
            for key, value in results["performance"].items():
                flat_results[f"performance.{key}"] = value
            for attr, attr_metrics in results["fairness"].items():
                for metric, value in attr_metrics.items():
                    if not isinstance(value, dict):
                        flat_results[f"fairness.{attr}.{metric}"] = value
            pd.DataFrame([flat_results]).to_csv(csv_path, index=False)
            print(f"SM summary results saved to {csv_path}")


def compare_sm_with_baselines(sm_results, baseline_results, afrl_results=None):
    print("\n" + "=" * 100)
    print("SM FRAMEWORK COMPARISON")
    print("=" * 100)

    methods = ["Baseline", "SM"]
    results_list = [baseline_results, sm_results]

    if afrl_results:
        methods.append("AFRL")
        results_list.append(afrl_results)

    print(f"{'Method':<12} {'Recall@10':<12} {'HitRate@10':<13} {'NDCG@10':<12} {'G-DP':<8} {'A-DP':<8}")
    print("-" * 80)

    for method, results in zip(methods, results_list):
        perf = results.get("performance", {})
        fair = results.get("fairness", {})
        print(
            f"{method:<12} "
            f"{perf.get('Recall@10', 0):<12.4f} "
            f"{perf.get('HitRate@10', 0):<13.4f} "
            f"{perf.get('NDCG@10', 0):<12.4f} "
            f"{fair.get('gender', {}).get('demographic_parity', 0):<8.4f} "
            f"{fair.get('age_group', {}).get('demographic_parity', 0):<8.4f}"
        )

    print("=" * 100)


def get_sm_model_info(model):
    return {
        "framework": "Separate Method (SM)",
        "base_model": model.base_model.__class__.__name__,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "filter_combinations": len(model.filter_module.filters),
        "sensitive_attributes": model.sensitive_attributes,
        "hidden_units": model.hidden_units,
    }
