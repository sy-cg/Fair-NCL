"""
AFRL evaluation adapter.

This module evaluates an AFRL-wrapped recommender by supplying a `fairness_mask`
to `model.predict(...)`. The mask controls which attribute-specific embeddings
are kept in the AFRL combiner (1 keeps, 0 removes).
"""

from __future__ import annotations

import numpy as np
import torch

from evaluation.full_ranking import evaluate_full_ranking_loader
from .evaluation import OptimizedFairnessEvaluator


class AFRLEvaluationAdapter(OptimizedFairnessEvaluator):
    """Evaluate AFRL models under different fairness masks."""

    def __init__(self, config):
        super().__init__(config)
        self.fairness_masks = self._generate_fairness_masks()

    def _generate_fairness_masks(self):
        num_attrs = len(self.sensitive_attributes)

        masks = {
            "all_nonsensitive": torch.ones(1, num_attrs, dtype=torch.float32),
            "all_sensitive": torch.zeros(1, num_attrs, dtype=torch.float32),
        }

        # Backward-compatible naming for the two active attributes.
        alias = {
            "gender": "gender_sensitive",
            "age_group": "age_sensitive",
        }

        for idx, attr in enumerate(self.sensitive_attributes):
            key = alias.get(attr, f"{attr}_sensitive")
            mask = torch.ones(1, num_attrs, dtype=torch.float32)
            mask[0, idx] = 0.0
            masks[key] = mask

        return masks

    @torch.no_grad()
    def evaluate_afrl_model(self, model, data_loader, eval_mode: str = "average"):
        """Evaluate an AFRL model.

        Args:
            model: AFRL model wrapper exposing `predict(input_seq, candidate_items, fairness_mask)`.
            data_loader: evaluation loader with negative candidates.
            eval_mode: "average" (default), "all", or a specific mask key.
        """

        print(f"Evaluating AFRL model with mode: {eval_mode}")

        if eval_mode in ("average", "all"):
            all_results = {}
            for mask_name, mask in self.fairness_masks.items():
                print(f"\nEvaluating with {mask_name} mask...")
                mask = mask.to(self.device)
                all_results[mask_name] = self._evaluate_with_mask(model, data_loader, mask, mask_name)

            return self._compute_average_results(all_results) if eval_mode == "average" else all_results

        if eval_mode in self.fairness_masks:
            mask = self.fairness_masks[eval_mode].to(self.device)
            return self._evaluate_with_mask(model, data_loader, mask, eval_mode)

        return self.evaluate_afrl_model(model, data_loader, "average")

    def _evaluate_with_mask(self, model, data_loader, fairness_mask, mask_name: str):
        results = evaluate_full_ranking_loader(
            data_loader=data_loader,
            config=self.config,
            predict_fn=lambda batch: model.predict(
                batch["input_seq"],
                fairness_mask=fairness_mask.expand(batch["input_seq"].size(0), -1),
            ),
            sensitive_attributes=self.sensitive_attributes,
            desc=f"Eval {mask_name}",
            legacy_output=True,
        )

        results["afrl_info"] = {
            "mask_name": mask_name,
            "mask": fairness_mask.detach().cpu().numpy().tolist(),
        }
        return results

    def _compute_average_results(self, all_results):
        avg_results = {"performance": {}, "fairness": {}}

        perf_metrics = [
            "Recall@5",
            "Recall@10",
            "Recall@20",
            "HitRate@5",
            "HitRate@10",
            "HitRate@20",
            "NDCG@5",
            "NDCG@10",
            "NDCG@20",
            "Precision@5",
            "Precision@10",
            "Precision@20",
            "MRR@5",
            "MRR@10",
            "MRR@20",
        ]

        for metric in perf_metrics:
            values = [
                r["performance"][metric]
                for r in all_results.values()
                if "performance" in r and metric in r["performance"]
            ]
            if values:
                avg_results["performance"][metric] = float(np.mean(values))
                avg_results["performance"][f"{metric}_std"] = float(np.std(values))

        for attr in self.sensitive_attributes:
            avg_results["fairness"][attr] = {}
            for metric in ("demographic_parity", "equalized_opportunity"):
                values = [
                    r["fairness"][attr][metric]
                    for r in all_results.values()
                    if "fairness" in r and attr in r["fairness"] and metric in r["fairness"][attr]
                ]
                if values:
                    avg_results["fairness"][attr][metric] = float(np.mean(values))
                    avg_results["fairness"][attr][f"{metric}_std"] = float(np.std(values))

        avg_results["detailed_results"] = all_results
        return avg_results

    def print_afrl_evaluation_results(self, results):
        if "detailed_results" not in results:
            return super().print_evaluation_results(results)

        print("\n" + "=" * 80)
        print("AFRL EVALUATION RESULTS (AVERAGED)")
        print("=" * 80)
        super().print_evaluation_results(results)

        print("\n" + "=" * 80)
        print("DETAILED RESULTS BY FAIRNESS MASK")
        print("=" * 80)

        for mask_name, mask_results in results["detailed_results"].items():
            print(f"\n{mask_name.upper()}:")
            print(f"Mask: {mask_results['afrl_info']['mask']}")

            perf = mask_results.get("performance", {})
            print(f"  Recall@10: {perf.get('Recall@10', 0):.4f}")
            print(f"  NDCG@10: {perf.get('NDCG@10', 0):.4f}")

            for attr in self.sensitive_attributes:
                fair = mask_results.get("fairness", {}).get(attr, {})
                if fair:
                    print(f"  {attr} DP: {fair.get('demographic_parity', 0):.4f}")
