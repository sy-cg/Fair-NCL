import json
import os
import time
from collections import defaultdict
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from analysis.mechanism import export_mechanism_analysis
from evaluation.full_ranking import evaluate_full_ranking_loader
from experiments.backbone_initialization import enforce_frozen_backbone_eval
from .common import move_batch_to_device, reporting_sensitive_attributes
from .training_safety import safe_backward_step


def train_research_method(model,
                          train_loader,
                          val_loader,
                          test_loader,
                          config,
                          output_dir: str,
                          job_info: Optional[Dict] = None) -> Dict:
    """Train and evaluate any method wrapper under a common protocol."""
    os.makedirs(output_dir, exist_ok=True)
    model = model.to(config.device)
    use_custom_optimization = bool(getattr(model, "uses_custom_optimization", False))
    optimizer = None
    if not use_custom_optimization:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=getattr(config, "l2_emb", 1e-6),
        )
    scaler = GradScaler(enabled=getattr(config, "use_mixed_precision", False))

    best_metric = -float("inf")
    best_state = None
    best_epoch = -1
    best_val_results = None
    patience = getattr(config, "patience", 10)
    patience_counter = 0
    history = defaultdict(list)
    postfix_interval = max(1, int(getattr(config, "progress_postfix_interval", 1)))
    eval_interval = max(1, int(getattr(config, "eval_interval", 1)))

    training_started = time.perf_counter()
    if getattr(config.device, "type", "cpu") == "cuda":
        torch.cuda.reset_peak_memory_stats(config.device)

    for epoch in range(config.num_epochs):
        epoch_started = time.perf_counter()
        model.train()
        enforce_frozen_backbone_eval(model)
        train_losses = defaultdict(float)
        num_batches = 0
        skipped_nonfinite_updates = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.num_epochs} [train]")
        for batch_index, batch in enumerate(pbar, start=1):
            batch = move_batch_to_device(batch, config.device)
            if use_custom_optimization:
                loss_dict = model.training_step(batch)
                loss = loss_dict["loss"]
            else:
                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=getattr(config, "use_mixed_precision", False)):
                    loss_dict = model.compute_loss(batch)
                    loss = loss_dict["loss"]

                step_state = safe_backward_step(
                    model.parameters(),
                    optimizer,
                    scaler,
                    loss,
                    getattr(config, "gradient_clip_val", 5.0),
                )
                if not step_state["stepped"]:
                    skipped_nonfinite_updates += 1
                    if skipped_nonfinite_updates <= 3 or skipped_nonfinite_updates % 50 == 0:
                        print(
                            "[WARN] Skipping optimizer update with "
                            f"{step_state['reason']} at epoch {epoch + 1}, "
                            f"batch {batch_index}."
                        )
                    continue

            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    train_losses[key] += float(value.detach().cpu())
            num_batches += 1
            if batch_index == 1 or batch_index % postfix_interval == 0:
                pbar.set_postfix({"loss": f"{float(loss.detach().cpu()):.4f}"})

        for key, value in train_losses.items():
            history[f"train_{key}"].append(value / max(1, num_batches))
        history["skipped_nonfinite_updates"].append(skipped_nonfinite_updates)
        history["epoch_train_seconds"].append(time.perf_counter() - epoch_started)

        should_evaluate = ((epoch + 1) % eval_interval == 0) or (epoch + 1 == config.num_epochs)
        if not should_evaluate:
            continue

        val_results = evaluate_research_method(model, val_loader, config, export_per_user=False)
        for key, value in val_results["selected"].items():
            if isinstance(value, (int, float, np.floating)):
                history[f"val_{key}"].append(float(value))

        current_metric = val_results["utility"].get("HitRate@10", 0.0)
        print(f"Epoch {epoch + 1}: val HitRate@10={current_metric:.4f}, "
              f"NDCG@10={val_results['utility'].get('NDCG@10', 0.0):.4f}")

        if current_metric > best_metric:
            best_metric = current_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_val_results = val_results
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        _save_checkpoint(
            model,
            config,
            output_dir,
            "best.pt",
            best_epoch,
            best_val_results or {},
            job_info,
        )
        _save_json(
            os.path.join(output_dir, "validation_results.json"),
            best_val_results or {},
        )

    training_seconds = time.perf_counter() - training_started
    if not bool(getattr(config, "evaluate_test", True)):
        validation_only_result = {
            "status": "not_run_by_protocol",
            "validation_only": True,
            "runtime": {
                "training_seconds": float(training_seconds),
                "test_evaluation_seconds": 0.0,
                "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(config.device))
                if getattr(config.device, "type", "cpu") == "cuda" else 0,
            },
        }
        augmenter = getattr(model, "augmenter", None)
        if augmenter is not None and hasattr(augmenter, "diagnostic_summary"):
            validation_only_result["augmentation_diagnostics"] = _jsonable(augmenter.diagnostic_summary())
        _save_json(os.path.join(output_dir, "history.json"), dict(history))
        _save_json(os.path.join(output_dir, "test_results.json"), validation_only_result)
        return {
            "history": dict(history),
            "validation": best_val_results or {},
            "test": validation_only_result,
        }

    evaluation_started = time.perf_counter()
    test_results = evaluate_research_method(
        model,
        test_loader,
        config,
        export_per_user=bool(getattr(config, "export_per_user_metrics", False)),
    )
    evaluation_seconds = time.perf_counter() - evaluation_started
    test_results["runtime"] = {
        "training_seconds": float(training_seconds),
        "test_evaluation_seconds": float(evaluation_seconds),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(config.device))
        if getattr(config.device, "type", "cpu") == "cuda" else 0,
    }
    augmenter = getattr(model, "augmenter", None)
    if augmenter is not None and hasattr(augmenter, "diagnostic_summary"):
        test_results["augmentation_diagnostics"] = _jsonable(augmenter.diagnostic_summary())
    if getattr(config, "export_mechanism_analysis", True):
        try:
            test_results["mechanism"] = _jsonable(
                export_mechanism_analysis(model, test_loader, config, output_dir, split_name="test")
            )
        except Exception as exc:
            test_results["mechanism"] = {
                "status": "failed",
                "error": str(exc),
            }
    _save_json(os.path.join(output_dir, "history.json"), dict(history))
    _save_json(os.path.join(output_dir, "test_results.json"), test_results)
    return {"history": dict(history), "test": test_results}


@torch.inference_mode()
def evaluate_research_method(model, data_loader, config, export_per_user: bool = False) -> Dict:
    model.eval()
    results = evaluate_full_ranking_loader(
        data_loader=data_loader,
        config=config,
        predict_fn=lambda batch: model.predict(batch["input_seq"]),
        sensitive_attributes=reporting_sensitive_attributes(config),
        desc="evaluate",
        legacy_output=False,
        export_per_user=export_per_user,
        include_intersectional=bool(getattr(config, "report_intersectional_groups", False)),
    )
    output = {
        "utility": _jsonable(results["utility"]),
        "fairness": _jsonable(results["fairness"]),
        "selected": _jsonable(results["selected"]),
    }
    if export_per_user:
        output["per_user"] = _jsonable(results.get("per_user", []))
    return output


def _save_checkpoint(model, config, output_dir, filename, epoch, metrics, job_info):
    path = os.path.join(output_dir, filename)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "job": job_info or {},
        "config": _safe_config_dict(config),
    }, path)


def _save_json(path: str, data: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2, ensure_ascii=False)


def _safe_config_dict(config):
    result = {}
    for key, value in vars(config).items():
        if key.startswith("_"):
            continue
        if isinstance(value, torch.device):
            result[key] = str(value)
        elif isinstance(value, (str, int, float, bool, type(None), list, tuple, dict)):
            result[key] = value
        else:
            result[key] = str(value)
    return result


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
