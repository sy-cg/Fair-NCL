"""
FairRec evaluation adapter with full-ranking evaluation.
"""

from evaluation.full_ranking import evaluate_full_ranking_loader
from fairness.evaluation import OptimizedFairnessEvaluator


class FairRecEvaluationAdapter(OptimizedFairnessEvaluator):
    def __init__(self, config):
        super().__init__(config)
        self.eval_modes = ["full", "fair"]

    def evaluate_fairrec(self, model, data_loader):
        model.eval()
        all_results = {}

        print(f"\nEvaluating FairRec with modes: {self.eval_modes}")
        for mode in self.eval_modes:
            print(f"Running evaluation for mode: [{mode}]...")
            all_results[mode] = evaluate_full_ranking_loader(
                data_loader=data_loader,
                config=self.config,
                predict_fn=lambda batch, current_mode=mode: model.predict(
                    batch["input_seq"],
                    fairness_config=current_mode,
                ),
                sensitive_attributes=self.sensitive_attributes,
                desc=f"FairRec-{mode}",
                legacy_output=True,
            )

        return all_results

    def print_comparison(self, results):
        print("\n" + "=" * 30 + " FairRec Audit Report " + "=" * 30)
        for mode in self.eval_modes:
            perf = results[mode].get("performance", {})
            print(f"\nMode: [{mode.upper()}]")
            print(f"  Recall@10: {perf.get('Recall@10', 0):.4f} | NDCG@10: {perf.get('NDCG@10', 0):.4f}")
            for attr, f_metrics in results[mode].get("fairness", {}).items():
                print(f"  {attr} Demographic Parity: {f_metrics.get('demographic_parity', 0):.4f}")
        print("=" * 80)
