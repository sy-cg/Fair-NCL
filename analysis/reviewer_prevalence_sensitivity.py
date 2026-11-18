"""Check item-skew sensitivity to support and prior shrinkage without retraining."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from statistics import median

from analysis.skew_diagnostics import _spearman_maps
from fairness.skew import build_leave_two_out_train_events, compute_train_item_bias_statistics


def summarize(dataset: str) -> dict:
    with (Path("processed_data") / dataset / "processed_data.pkl").open("rb") as handle:
        data = pickle.load(handle)
    events = build_leave_two_out_train_events(data["sequences"])
    scores_by_strength = {
        strength: compute_train_item_bias_statistics(
            data["train_data"],
            train_event_sequences=events,
            metric="prior_jsd",
            min_count=5,
            prior_strength=strength,
        )
        for strength in (0.0, 5.0, 10.0)
    }
    reference = scores_by_strength[5.0]
    support = reference["item_support"]
    bins = []
    for label, lower, upper in (("5--9", 5, 10), ("10--49", 10, 50), ("50+", 50, None)):
        items = [
            item for item, count in support.items()
            if count >= lower and (upper is None or count < upper)
            and item in reference["scores"]
        ]
        bins.append({
            "support": label,
            "items": len(items),
            **{
                f"median_skew_tau_{int(strength)}": median(
                    scores_by_strength[strength]["scores"][item] for item in items
                ) if items else None
                for strength in (0.0, 5.0, 10.0)
            },
        })
    return {
        "dataset": dataset,
        "train_items": len(support),
        "items_below_five": sum(count < 5 for count in support.values()),
        "scored_items": len(reference["scores"]),
        "tau_0_vs_5_spearman": _spearman_maps(
            scores_by_strength[0.0]["scores"], reference["scores"]
        ),
        "tau_5_vs_10_spearman": _spearman_maps(
            reference["scores"], scores_by_strength[10.0]["scores"]
        ),
        "support_bins": bins,
    }


def main() -> None:
    for dataset in ("ml-1m", "taobao"):
        summary = summarize(dataset)
        output = Path("tables/supplementary/skew") / dataset / "prevalence_sensitivity.json"
        output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"{dataset}: wrote {output}")


if __name__ == "__main__":
    main()
