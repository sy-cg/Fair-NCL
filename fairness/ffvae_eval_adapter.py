"""
FFVAE evaluation adapter built on full-ranking evaluation.
"""

import numpy as np
import torch
from tqdm import tqdm

from evaluation.full_ranking import evaluate_full_ranking_loader
from fairness.evaluation import OptimizedFairnessEvaluator


class FFVAEEvaluationAdapter(OptimizedFairnessEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self.fairness_configs = self._generate_fairness_configs()

    def _generate_fairness_configs(self):
        configs = {
            "full": None,
            "fair_all": [True] * len(self.sensitive_attributes),
        }

        for i, attr in enumerate(self.sensitive_attributes):
            mask = [False] * len(self.sensitive_attributes)
            mask[i] = True
            configs[f"fair_{attr}"] = mask

        import itertools

        for r in range(2, len(self.sensitive_attributes) + 1):
            for combo in itertools.combinations(enumerate(self.sensitive_attributes), r):
                indices, attrs = zip(*combo)
                mask = [False] * len(self.sensitive_attributes)
                for idx in indices:
                    mask[idx] = True
                configs[f"fair_{'_'.join(attrs)}"] = mask

        return configs

    @torch.no_grad()
    def evaluate_ffvae_model(self, model, data_loader, eval_mode="average"):
        print(f"Evaluating FFVAE model with mode: {eval_mode}")

        if eval_mode in ("average", "all"):
            all_results = {}
            for config_name, fairness_mask in self.fairness_configs.items():
                print(f"\nEvaluating with {config_name} configuration...")
                all_results[config_name] = self._evaluate_with_mask(model, data_loader, fairness_mask, config_name)
            return self._compute_average_results(all_results) if eval_mode == "average" else all_results

        if eval_mode in self.fairness_configs:
            return self._evaluate_with_mask(model, data_loader, self.fairness_configs[eval_mode], eval_mode)
        return self._evaluate_with_mask(model, data_loader, None, "full")

    def _evaluate_with_mask(self, model, data_loader, fairness_mask, config_name):
        results = evaluate_full_ranking_loader(
            data_loader=data_loader,
            config=self.config,
            predict_fn=lambda batch: model.predict(batch["input_seq"], fairness_mask=fairness_mask),
            sensitive_attributes=self.sensitive_attributes,
            desc=f"Eval {config_name}",
            legacy_output=True,
        )
        results["ffvae_info"] = {
            "config_name": config_name,
            "fairness_mask": fairness_mask,
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

    def evaluate_disentanglement(self, model, data_loader):
        print("Evaluating FFVAE disentanglement...")

        model.eval()
        disentanglement_scores = {}

        with torch.no_grad():
            all_z = []
            all_b = {i: [] for i in range(len(self.sensitive_attributes))}
            all_attrs = {attr: [] for attr in self.sensitive_attributes}

            for batch in tqdm(data_loader, desc="Collecting representations"):
                input_seq = batch["input_seq"].to(self.config.device)
                x = model.get_sequence_embedding(input_seq)

                z, b_list, _, _ = model.ffvae.encode(x)

                all_z.append(z.cpu())
                for i, b_i in enumerate(b_list):
                    all_b[i].append(b_i.cpu())

                for attr in self.sensitive_attributes:
                    if attr in batch:
                        all_attrs[attr].append(batch[attr].cpu())

            z_concat = torch.cat(all_z, dim=0)
            b_concat = {i: torch.cat(all_b[i], dim=0) for i in all_b}
            attrs_concat = {attr: torch.cat(all_attrs[attr], dim=0) for attr in all_attrs}

            from sklearn.feature_selection import mutual_info_regression

            for i in range(len(self.sensitive_attributes)):
                disentanglement_scores[f"MI(z, b_{i})"] = float(
                    mutual_info_regression(z_concat.numpy(), b_concat[i].numpy().ravel())[0]
                )

            for i in range(len(self.sensitive_attributes)):
                for j in range(i + 1, len(self.sensitive_attributes)):
                    disentanglement_scores[f"MI(b_{i}, b_{j})"] = float(
                        mutual_info_regression(b_concat[i].numpy().reshape(-1, 1), b_concat[j].numpy().ravel())[0]
                    )

            for i, attr in enumerate(self.sensitive_attributes):
                if attr in attrs_concat:
                    disentanglement_scores[f"MI(b_{i}, {attr})"] = float(
                        mutual_info_regression(b_concat[i].numpy().reshape(-1, 1), attrs_concat[attr].numpy())[0]
                    )

        return disentanglement_scores

    def print_ffvae_evaluation_results(self, results):
        if "detailed_results" not in results:
            return super().print_evaluation_results(results)

        print("\n" + "=" * 80)
        print("FFVAE EVALUATION RESULTS (AVERAGED)")
        print("=" * 80)
        super().print_evaluation_results(results)

        print("\n" + "=" * 80)
        print("DETAILED RESULTS BY FAIRNESS CONFIG")
        print("=" * 80)
        for config_name, config_results in results["detailed_results"].items():
            print(f"\n{config_name.upper()}:")
            perf = config_results.get("performance", {})
            print(f"  Recall@10: {perf.get('Recall@10', 0):.4f}")
            print(f"  NDCG@10: {perf.get('NDCG@10', 0):.4f}")
