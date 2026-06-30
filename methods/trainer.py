import json
import os
from collections import defaultdict
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from analysis.mechanism import export_mechanism_analysis
from evaluation.full_ranking import evaluate_full_ranking_loader
from .common import move_batch_to_device, reporting_sensitive_attributes


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

    for epoch in range(config.num_epochs):
        model.train()
        train_losses = defaultdict(float)
        num_batches = 0

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

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), getattr(config, "gradient_clip_val", 5.0))
                scaler.step(optimizer)
                scaler.update()

            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    train_losses[key] += float(value.detach().cpu())
            num_batches += 1
            if batch_index == 1 or batch_index % postfix_interval == 0:
                pbar.set_postfix({"loss": f"{float(loss.detach().cpu()):.4f}"})

        for key, value in train_losses.items():
            history[f"train_{key}"].append(value / max(1, num_batches))

        should_evaluate = ((epoch + 1) % eval_interval == 0) or (epoch + 1 == config.num_epochs)
        if not should_evaluate:
            continue

        val_results = evaluate_research_method(model, val_loader, config)
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

    test_results = evaluate_research_method(model, test_loader, config)
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
def evaluate_research_method(model, data_loader, config) -> Dict:
    model.eval()
    results = evaluate_full_ranking_loader(
        data_loader=data_loader,
        config=config,
        predict_fn=lambda batch: model.predict(batch["input_seq"]),
        sensitive_attributes=reporting_sensitive_attributes(config),
        desc="evaluate",
        legacy_output=False,
    )
    return {
        "utility": _jsonable(results["utility"]),
        "fairness": _jsonable(results["fairness"]),
        "selected": _jsonable(results["selected"]),
    }


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
