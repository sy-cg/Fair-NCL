import numpy as np

from evaluation.full_ranking import evaluate_full_ranking_loader


class OptimizedFairnessEvaluator:
    """Legacy evaluation adapter backed by full-ranking evaluation."""

    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.sensitive_attributes = config.sensitive_attributes
        self.topk_list = config.topk_list

    def evaluate_model_optimized(self, model, data_loader):
        print("Evaluating model with full ranking over the complete item space ...")
        return evaluate_full_ranking_loader(
            data_loader=data_loader,
            config=self.config,
            predict_fn=lambda batch: model.predict(batch["input_seq"]),
            sensitive_attributes=self.sensitive_attributes,
            desc="Evaluation",
            legacy_output=True,
        )

    def print_evaluation_results(self, results):
        print("\n" + "=" * 80)
        print("METRICS SUMMARY")
        print("=" * 80)

        performance = results.get("performance", {})
        if performance:
            print("\nPerformance Metrics:")
            print("-" * 40)
            for metric in sorted(performance.keys()):
                value = performance[metric]
                if isinstance(value, (int, float, np.floating)):
                    print(f"{metric:15s}: {float(value):.4f}")

        fairness = results.get("fairness", {})
        if fairness:
            print("\nFairness Metrics:")
            print("-" * 40)
            for attr, attr_metrics in fairness.items():
                print(f"\n{attr.upper()} Fairness:")
                for metric, value in attr_metrics.items():
                    if isinstance(value, (int, float, np.floating)):
                        print(f"{metric:28s}: {float(value):.4f}")
                    elif isinstance(value, dict):
                        print(f"{metric:28s}: {value}")

    def compute_group_metrics(self, predictions, targets, group_indices, k_list):
        raise NotImplementedError(
            "Legacy sampled-ranking helper has been removed. Use full-ranking evaluation instead."
        )

    def save_evaluation_results(self, results, save_path):
        import os
        import pandas as pd

        csv_path = os.path.splitext(save_path)[0] + ".csv"
        flat_results = {}

        for section, section_values in results.items():
            if isinstance(section_values, dict):
                for key, value in section_values.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            flat_results[f"{section}.{key}.{subkey}"] = subvalue
                    else:
                        flat_results[f"{section}.{key}"] = value
            else:
                flat_results[section] = section_values

        pd.DataFrame([flat_results]).to_csv(csv_path, index=False)
        print(f"Evaluation results saved to {csv_path}")
