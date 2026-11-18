"""Export RQ4 representation diagnostics from completed checkpoints without retraining."""
import argparse
import json
from pathlib import Path

import torch

from analysis.mechanism import export_mechanism_analysis
from config import Config
from data.dataset import GPUOptimizedDataLoader, OptimizedMovieLensDataset
from experiments.runner import load_processed_dataset
from methods.registry import create_method
from utils.utils import set_seed


def export_one(job_dir, processed_root):
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    if job["method"] not in {"fair_ncl", "baseline", "ncl_sr_mce", "a_fsr", "coserec"}:
        return "skipped_method"
    output = job_dir / "mechanism_test_summary.json"
    if output.exists():
        return "already_done"
    checkpoint_path = job_dir / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = Config()
    config.dataset = job["dataset"]
    config.base_model_name = job["backbone"]
    config.model_name = job["backbone"]
    config.method = job["method"]
    config.seed = int(job["seed"])
    for key, value in job["params"].items():
        setattr(config, key, value)
    config.apply_memory_profile(
        dataset=config.dataset, method=config.method,
        backbone=config.base_model_name, explicit_keys=set(job["params"]),
    )
    set_seed(config.seed)
    data = load_processed_dataset(processed_root, config.dataset)
    config.num_items = data["num_items"]
    config.num_users = data["num_users"]
    config.cache_dir = str(Path("cache") / config.dataset)
    test_dataset = OptimizedMovieLensDataset(
        data["test_data"], config.max_seq_len, config.num_items,
        user_history_dict={}, num_negative=0, seed=config.seed + 2,
    )
    loader = GPUOptimizedDataLoader(
        test_dataset, config.eval_batch_size, shuffle=False, config=config,
    )
    # Diagnostics use only the trained encoder; augmentation resources are unused.
    resources = {"bias_statistics": {"scores": {}, "metric": None}}
    if job["method"] == "coserec":
        from methods.coserec import ItemCFIUFSimilarity
        resources["coserec_similarity"] = ItemCFIUFSimilarity({}, [1, 2])
    model = create_method(
        job["method"], job["backbone"], config, resources=resources,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(config.device)
    export_mechanism_analysis(model, loader, config, str(job_dir), split_name="test")
    return "exported"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="results_revision_kbs_v2")
    parser.add_argument("--processed-root", default="processed_data")
    parser.add_argument("--methods", nargs="+", default=["fair_ncl", "baseline"])
    args = parser.parse_args()
    counts = {}
    for job_path in sorted(Path(args.results_root).glob("*/supplementary_*/job.json")):
        job = json.loads(job_path.read_text(encoding="utf-8"))
        if job.get("method") not in args.methods or job.get("seed") not in (42, 2025, 2026, 2027, 2028):
            continue
        result = export_one(job_path.parent, args.processed_root)
        counts[result] = counts.get(result, 0) + 1
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == "__main__":
    main()

