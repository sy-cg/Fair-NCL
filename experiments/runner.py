import argparse
import gc
import json
import os
import pickle
import shutil
import time
from typing import Dict

import torch

from config import Config
from data.dataset import create_research_data_loaders
from fairness.skew import build_train_event_sequences
from .backbone_initialization import initialize_frozen_backbone
from .job_utils import result_artifact_is_complete, stable_job_id
from .provenance import collect_run_provenance, write_run_provenance
from methods.registry import build_method_resources, create_method
from methods.trainer import train_research_method
from utils.utils import set_seed


def load_processed_dataset(processed_root: str, dataset: str) -> Dict:
    path = os.path.join(processed_root, dataset, "processed_data.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed dataset not found: {path}. "
            f"Run preprocessing first with run_research_pipeline.py preprocess."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def run_experiment_job(job: Dict,
                       processed_root: str = "processed_data",
                       results_root: str = "results",
                       debug: bool = False,
                       skip_existing: bool = False):
    output_dir = os.path.join(results_root, job["dataset"], job["job_id"])
    result_path = os.path.join(output_dir, "test_results.json")
    if skip_existing and result_artifact_is_complete(job, results_root):
        history_path = os.path.join(output_dir, "history.json")
        history = _load_json(history_path) if os.path.exists(history_path) else {}
        test_results = _load_json(result_path)
        print(f"Skipping existing job: {job['job_id']}")
        return {"status": "skipped_existing", "history": history, "test": test_results}

    if job.get("reuse_from_job_id") or job.get("reuse_from_phase"):
        return _reuse_experiment_job(job, results_root)

    config = Config()
    config.dataset = job["dataset"]
    config.model_name = job["backbone"]
    config.base_model_name = job["backbone"]
    config.method = job["method"]
    config.phase = job.get("phase", "")
    config.seed = int(job.get("seed", 42))

    for key, value in job.get("params", {}).items():
        setattr(config, key, value)

    explicit_param_keys = set(job.get("params", {}).keys())
    if config.phase in {"backbone", "augment", "loss"} and "export_mechanism_analysis" not in explicit_param_keys:
        config.export_mechanism_analysis = False

    config.apply_memory_profile(
        dataset=config.dataset,
        method=config.method,
        backbone=config.base_model_name,
        explicit_keys=explicit_param_keys,
    )

    if debug:
        config.num_epochs = min(getattr(config, "num_epochs", 50), 2)
        config.batch_size = min(getattr(config, "batch_size", 256), 64)
        config.eval_batch_size = min(getattr(config, "eval_batch_size", 512), 128)
        if "export_mechanism_analysis" not in explicit_param_keys:
            config.export_mechanism_analysis = False

    set_seed(config.seed)
    data = load_processed_dataset(processed_root, config.dataset)
    config.num_items = data["num_items"]
    config.num_users = data["num_users"]
    config.processed_data_dir = os.path.join(processed_root, config.dataset)
    config.cache_dir = os.path.join("cache", config.dataset)
    config.model_save_dir = os.path.join(results_root, config.dataset, job["job_id"], "checkpoints")
    os.makedirs(config.cache_dir, exist_ok=True)
    os.makedirs(config.model_save_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "effective_config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(config), f, indent=2, ensure_ascii=False, default=str)

    write_run_provenance(
        collect_run_provenance(config, data, workspace=os.getcwd()),
        os.path.join(output_dir, "run_provenance.json"),
    )

    with open(os.path.join(output_dir, "job.json"), "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2, ensure_ascii=False)

    train_data = data["train_data"]
    val_data = data["val_data"]
    test_data = data["test_data"]
    train_loader, val_loader, test_loader = create_research_data_loaders(
        train_data,
        val_data,
        test_data,
        config,
    )
    item_metadata = data.get("items", data.get("movies"))
    train_event_sequences = build_train_event_sequences(
        data.get("sequences", {}),
        train_data,
    )
    if str(getattr(config, "bias_event_source", "unique_train_events")).lower() == "prefix_samples":
        train_event_sequences = None
    resource_started = time.perf_counter()
    resources = build_method_resources(
        config,
        train_data,
        item_metadata=item_metadata,
        train_event_sequences=train_event_sequences or None,
    )
    resource_seconds = time.perf_counter() - resource_started
    _save_resource_diagnostics(output_dir, resources, resource_seconds, config.cache_dir)
    model = create_method(
        method=job["method"],
        backbone_name=job["backbone"],
        config=config,
        train_data=None,
        resources=resources,
    )
    backbone_initialization = initialize_frozen_backbone(
        model,
        job,
        results_root,
        checkpoint_loader=lambda path: torch.load(str(path), map_location="cpu"),
    )
    if backbone_initialization is not None:
        config.backbone_initialization = backbone_initialization
        with open(os.path.join(output_dir, "backbone_initialization.json"), "w", encoding="utf-8") as f:
            json.dump(backbone_initialization, f, indent=2, ensure_ascii=False)
    del resources
    del train_data
    del val_data
    del test_data
    del data
    gc.collect()

    return train_research_method(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        output_dir=output_dir,
        job_info=job,
    )


def _reuse_experiment_job(job: Dict, results_root: str) -> Dict:
    dataset = job["dataset"]
    job_id = job["job_id"]
    source_job_id = _resolve_source_job_id(job, results_root)

    source_dir = os.path.join(results_root, dataset, source_job_id)
    output_dir = os.path.join(results_root, dataset, job_id)
    if not os.path.exists(source_dir):
        raise FileNotFoundError(
            f"Cannot reuse job '{job_id}' because source results were not found: {source_dir}"
        )

    source_best = os.path.join(source_dir, "best.pt")
    source_test_results = os.path.join(source_dir, "test_results.json")
    if not os.path.exists(source_best) or not os.path.exists(source_test_results):
        raise FileNotFoundError(
            f"Cannot reuse job '{job_id}' because source results are incomplete: {source_dir}"
        )

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(source_dir, output_dir)
    with open(os.path.join(output_dir, "job.json"), "w", encoding="utf-8") as f:
        json.dump(job, f, indent=2, ensure_ascii=False)

    checkpoint_path = os.path.join(output_dir, "best.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            checkpoint["job"] = job
            checkpoint["reused_from_job_id"] = source_job_id
            torch.save(checkpoint, checkpoint_path)

    history_path = os.path.join(output_dir, "history.json")
    test_results_path = os.path.join(output_dir, "test_results.json")
    history = _load_json(history_path) if os.path.exists(history_path) else {}
    test_results = _load_json(test_results_path) if os.path.exists(test_results_path) else {}
    return {"history": history, "test": test_results}


def load_job_from_jsonl(path: str, index: int) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise IndexError(f"No job at index {index} in {path}")


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_resource_diagnostics(output_dir: str,
                               resources: Dict,
                               elapsed_seconds: float,
                               cache_dir: str) -> None:
    statistics = resources.get("bias_statistics", {})
    scores = [float(value) for value in statistics.get("scores", {}).values()]
    supports = [int(value) for value in statistics.get("item_support", {}).values()]
    diagnostics = {
        key: value
        for key, value in statistics.items()
        if key not in {"scores", "per_attribute_scores", "item_support"}
    }
    diagnostics.update({
        "resource_build_seconds": float(elapsed_seconds),
        "similarity_items": len(resources.get("similarity_candidates", {})),
        "low_skew_pool_items": len(resources.get("low_skew_items", [])),
        "cache_bytes": _directory_size(cache_dir),
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "score_mean": sum(scores) / len(scores) if scores else None,
        "support_min": min(supports) if supports else None,
        "support_max": max(supports) if supports else None,
        "support_mean": sum(supports) / len(supports) if supports else None,
    })
    with open(os.path.join(output_dir, "resource_diagnostics.json"), "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, ensure_ascii=False)


def _directory_size(path: str) -> int:
    total = 0
    if not os.path.exists(path):
        return total
    for root, _, files in os.walk(path):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                total += os.path.getsize(file_path)
            except OSError:
                continue
    return int(total)


def _resolve_source_job_id(job: Dict, results_root: str) -> str:
    explicit_job_id = job.get("reuse_from_job_id")
    if explicit_job_id:
        return explicit_job_id

    source_phase = job.get("reuse_from_phase")
    if not source_phase:
        raise ValueError(f"Job '{job.get('job_id', '<unknown>')}' does not define a reuse source.")

    candidate_job_id = stable_job_id(
        source_phase,
        job["dataset"],
        job["method"],
        job["backbone"],
        int(job.get("seed", 42)),
        job.get("params", {}),
    )
    candidate_dir = os.path.join(results_root, job["dataset"], candidate_job_id)
    if os.path.exists(candidate_dir):
        return candidate_job_id

    matched_job_id = _match_job_id_from_saved_metadata(job, results_root, source_phase)
    return matched_job_id or candidate_job_id


def _match_job_id_from_saved_metadata(job: Dict, results_root: str, source_phase: str) -> str | None:
    dataset_dir = os.path.join(results_root, job["dataset"])
    if not os.path.exists(dataset_dir):
        return None

    expected_params = job.get("params", {})
    expected_method = job["method"]
    expected_backbone = job["backbone"]
    expected_seed = int(job.get("seed", 42))

    for entry in os.scandir(dataset_dir):
        if not entry.is_dir():
            continue
        job_path = os.path.join(entry.path, "job.json")
        if not os.path.exists(job_path):
            continue
        try:
            saved_job = _load_json(job_path)
        except Exception:
            continue

        if (
            saved_job.get("phase") == source_phase
            and saved_job.get("method") == expected_method
            and saved_job.get("backbone") == expected_backbone
            and int(saved_job.get("seed", 42)) == expected_seed
            and saved_job.get("params", {}) == expected_params
        ):
            return str(saved_job.get("job_id", entry.name))

    return None


def main():
    parser = argparse.ArgumentParser(description="Run one Fair-NCL research job.")
    parser.add_argument("--jobs", required=True, help="JSONL job file generated by experiments.specs.")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--processed-root", default="processed_data")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    job = load_job_from_jsonl(args.jobs, args.index)
    run_experiment_job(job, args.processed_root, args.results_root, args.debug, args.skip_existing)


if __name__ == "__main__":
    main()
