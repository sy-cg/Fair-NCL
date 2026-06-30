from __future__ import annotations

import json
import os
from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from methods.common import (
    covariance_loss,
    move_batch_to_device,
    reporting_sensitive_attributes,
    variance_loss,
)


def export_mechanism_analysis(model,
                              data_loader,
                              config,
                              output_dir: str,
                              split_name: str = "test") -> Dict[str, object]:
    """Export RQ4-oriented representation diagnostics for one trained model.

    The output consists of:
    - a JSON summary with probe leakage and group separability statistics
    - a compressed NPZ with a sampled subset of user representations for
      downstream visualization such as t-SNE / UMAP
    """
    payload = collect_representation_payload(model, data_loader, config, split_name=split_name)
    summary = build_mechanism_summary(payload, config)

    summary_filename = f"mechanism_{split_name}_summary.json"
    repr_filename = f"mechanism_{split_name}_repr_sample.npz"
    summary_path = os.path.join(output_dir, summary_filename)
    repr_path = os.path.join(output_dir, repr_filename)

    export_payload = sample_representation_payload(
        payload,
        max_users=int(getattr(config, "mechanism_export_max_users", 5000)),
        seed=int(getattr(config, "seed", 42)),
    )
    np.savez_compressed(repr_path, **export_payload)

    summary["files"] = {
        "summary_json": summary_filename,
        "representation_npz": repr_filename,
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(summary), handle, indent=2, ensure_ascii=False)

    return summary


@torch.inference_mode()
def collect_representation_payload(model,
                                   data_loader,
                                   config,
                                   split_name: str = "test") -> Dict[str, np.ndarray]:
    model.eval()
    attrs = reporting_sensitive_attributes(config)

    raw_repr_chunks = []
    fair_repr_chunks = []
    user_id_chunks = []
    attr_chunks = {attr: [] for attr in attrs}

    for raw_batch in tqdm(data_loader, desc=f"mechanism-{split_name}", leave=False):
        batch = move_batch_to_device(raw_batch, config.device)
        raw_repr = model.encode(batch["input_seq"])
        fair_repr = model.transform_representation(raw_repr, batch=batch)

        raw_repr_chunks.append(raw_repr.detach().float().cpu().numpy())
        fair_repr_chunks.append(fair_repr.detach().float().cpu().numpy())
        user_id_chunks.append(batch["user_id"].detach().long().cpu().numpy())
        for attr in attrs:
            if attr in batch:
                attr_chunks[attr].append(batch[attr].detach().long().cpu().numpy())

    payload = {
        "split": np.asarray([split_name]),
        "user_id": _concat_or_empty(user_id_chunks, dtype=np.int64),
        "raw_repr": _concat_or_empty(raw_repr_chunks, dtype=np.float32, ndim=2),
        "fair_repr": _concat_or_empty(fair_repr_chunks, dtype=np.float32, ndim=2),
    }
    for attr in attrs:
        payload[attr] = _concat_or_empty(attr_chunks[attr], dtype=np.int64)
    return payload


def build_mechanism_summary(payload: Dict[str, np.ndarray], config) -> Dict[str, object]:
    raw_repr = payload["raw_repr"]
    fair_repr = payload["fair_repr"]
    attrs = reporting_sensitive_attributes(config)

    summary: Dict[str, object] = {
        "split": str(payload.get("split", np.asarray(["test"]))[0]),
        "num_samples": int(raw_repr.shape[0]),
        "representation": {
            "raw": _representation_statistics(raw_repr),
            "fair": _representation_statistics(fair_repr),
            "transform": _transform_statistics(raw_repr, fair_repr),
        },
        "attributes": {},
    }

    for attr in attrs:
        labels = payload.get(attr)
        if labels is None or labels.size == 0:
            continue
        summary["attributes"][attr] = {
            "group_counts": _group_counts(labels),
            "probe_raw": _linear_probe_cv(
                raw_repr,
                labels,
                max_users=int(getattr(config, "mechanism_probe_max_users", 20000)),
                folds=int(getattr(config, "mechanism_probe_folds", 5)),
                seed=int(getattr(config, "seed", 42)),
            ),
            "probe_fair": _linear_probe_cv(
                fair_repr,
                labels,
                max_users=int(getattr(config, "mechanism_probe_max_users", 20000)),
                folds=int(getattr(config, "mechanism_probe_folds", 5)),
                seed=int(getattr(config, "seed", 42)),
            ),
            "separability_raw": _group_separability(raw_repr, labels),
            "separability_fair": _group_separability(fair_repr, labels),
        }

    return summary


def sample_representation_payload(payload: Dict[str, np.ndarray],
                                  max_users: int,
                                  seed: int) -> Dict[str, np.ndarray]:
    total = int(payload["user_id"].shape[0])
    if total <= max_users or max_users <= 0:
        return payload

    labels = _joint_group_labels(payload)
    indices = _stratified_sample_indices(labels, max_users, seed)
    return {key: value[indices] if isinstance(value, np.ndarray) and value.shape[0] == total else value
            for key, value in payload.items()}


def _representation_statistics(repr_array: np.ndarray) -> Dict[str, float]:
    if repr_array.size == 0:
        return {
            "mean_norm": np.nan,
            "std_norm": np.nan,
            "variance_loss": np.nan,
            "covariance_loss": np.nan,
        }

    tensor = torch.as_tensor(repr_array, dtype=torch.float32)
    normalized = F.normalize(tensor, dim=1)
    norms = np.linalg.norm(repr_array, axis=1)
    return {
        "mean_norm": float(norms.mean()),
        "std_norm": float(norms.std(ddof=0)),
        "variance_loss": float(variance_loss(normalized).item()),
        "covariance_loss": float(covariance_loss(normalized).item()),
    }


def _transform_statistics(raw_repr: np.ndarray, fair_repr: np.ndarray) -> Dict[str, float]:
    if raw_repr.size == 0 or fair_repr.size == 0:
        return {
            "delta_l2_mean": np.nan,
            "delta_l2_std": np.nan,
            "cosine_mean": np.nan,
            "cosine_std": np.nan,
        }

    delta = np.linalg.norm(fair_repr - raw_repr, axis=1)
    raw_norm = np.linalg.norm(raw_repr, axis=1)
    fair_norm = np.linalg.norm(fair_repr, axis=1)
    cosine = np.sum(raw_repr * fair_repr, axis=1) / np.clip(raw_norm * fair_norm, 1e-12, None)
    return {
        "delta_l2_mean": float(delta.mean()),
        "delta_l2_std": float(delta.std(ddof=0)),
        "cosine_mean": float(cosine.mean()),
        "cosine_std": float(cosine.std(ddof=0)),
    }


def _group_counts(labels: np.ndarray) -> Dict[str, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {str(int(group)): int(count) for group, count in zip(unique, counts)}


def _linear_probe_cv(repr_array: np.ndarray,
                     labels: np.ndarray,
                     max_users: int,
                     folds: int,
                     seed: int) -> Dict[str, object]:
    labels = np.asarray(labels)
    repr_array = np.asarray(repr_array, dtype=np.float32)

    result: Dict[str, object] = {
        "num_samples_total": int(labels.shape[0]),
        "num_samples_used": 0,
        "folds": 0,
        "accuracy": np.nan,
        "balanced_accuracy": np.nan,
        "macro_f1": np.nan,
        "roc_auc": np.nan,
        "status": "unavailable",
    }

    if repr_array.ndim != 2 or repr_array.shape[0] != labels.shape[0]:
        result["status"] = "invalid_input"
        return result

    unique, counts = np.unique(labels, return_counts=True)
    if unique.size < 2:
        result["status"] = "single_group"
        return result

    sample_idx = _stratified_sample_indices(labels, max_users, seed) if labels.shape[0] > max_users else np.arange(labels.shape[0])
    x = repr_array[sample_idx]
    y = labels[sample_idx]
    result["num_samples_used"] = int(y.shape[0])

    _, sample_counts = np.unique(y, return_counts=True)
    actual_folds = min(int(folds), int(sample_counts.min()))
    if actual_folds < 2:
        result["status"] = "insufficient_minority_samples"
        return result

    splitter = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=seed)
    classes = np.unique(y)
    class_to_pos = {cls: idx for idx, cls in enumerate(classes)}
    predictions = np.empty_like(y)
    probabilities = np.zeros((y.shape[0], classes.shape[0]), dtype=np.float64)

    try:
        for train_idx, test_idx in splitter.split(x, y):
            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    multi_class="auto",
                    random_state=seed,
                ),
            )
            classifier.fit(x[train_idx], y[train_idx])
            predictions[test_idx] = classifier.predict(x[test_idx])
            fold_proba = classifier.predict_proba(x[test_idx])
            fitted_classes = classifier.classes_
            for local_idx, cls in enumerate(fitted_classes):
                probabilities[test_idx, class_to_pos[cls]] = fold_proba[:, local_idx]
    except Exception as exc:
        result["status"] = f"probe_failed: {exc}"
        return result

    result["folds"] = int(actual_folds)
    result["accuracy"] = float(accuracy_score(y, predictions))
    result["balanced_accuracy"] = float(balanced_accuracy_score(y, predictions))
    result["macro_f1"] = float(f1_score(y, predictions, average="macro"))

    try:
        if classes.shape[0] == 2:
            positive_class = class_to_pos[classes[-1]]
            result["roc_auc"] = float(roc_auc_score(y, probabilities[:, positive_class]))
        else:
            result["roc_auc"] = float(roc_auc_score(y, probabilities, multi_class="ovr", average="macro"))
    except Exception:
        result["roc_auc"] = np.nan

    result["status"] = "ok"
    return result


def _group_separability(repr_array: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    labels = np.asarray(labels)
    repr_array = np.asarray(repr_array, dtype=np.float32)

    result: Dict[str, object] = {
        "num_groups": 0,
        "centroid_distance_mean": np.nan,
        "centroid_distance_min": np.nan,
        "centroid_distance_max": np.nan,
        "within_group_dispersion_mean": np.nan,
        "between_within_ratio": np.nan,
    }

    unique = np.unique(labels)
    if unique.size < 2 or repr_array.ndim != 2 or repr_array.shape[0] != labels.shape[0]:
        result["num_groups"] = int(unique.size)
        return result

    normalized = repr_array / np.clip(np.linalg.norm(repr_array, axis=1, keepdims=True), 1e-12, None)
    centroids = []
    within = []
    for group in unique:
        group_repr = normalized[labels == group]
        if group_repr.shape[0] == 0:
            continue
        centroid = group_repr.mean(axis=0)
        centroids.append(centroid)
        within.append(float(np.linalg.norm(group_repr - centroid, axis=1).mean()))

    if len(centroids) < 2:
        result["num_groups"] = len(centroids)
        return result

    distances = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            distances.append(float(np.linalg.norm(centroids[i] - centroids[j])))

    within_mean = float(np.mean(within)) if within else np.nan
    between_mean = float(np.mean(distances)) if distances else np.nan
    result.update({
        "num_groups": int(len(centroids)),
        "centroid_distance_mean": between_mean,
        "centroid_distance_min": float(np.min(distances)) if distances else np.nan,
        "centroid_distance_max": float(np.max(distances)) if distances else np.nan,
        "within_group_dispersion_mean": within_mean,
        "between_within_ratio": float(between_mean / (within_mean + 1e-12)) if distances and within else np.nan,
    })
    return result


def _joint_group_labels(payload: Dict[str, np.ndarray]) -> np.ndarray:
    keys = [key for key in ("gender", "age_group") if key in payload and payload[key].size]
    if not keys:
        return np.asarray(["all"] * int(payload["user_id"].shape[0]), dtype=object)
    if len(keys) == 1:
        return payload[keys[0]].astype(str)
    return np.asarray([f"{g}_{a}" for g, a in zip(payload["gender"], payload["age_group"])], dtype=object)


def _stratified_sample_indices(labels: np.ndarray, max_users: int, seed: int) -> np.ndarray:
    labels = np.asarray(labels)
    total = int(labels.shape[0])
    if max_users <= 0 or total <= max_users:
        return np.arange(total)

    rng = np.random.default_rng(seed)
    unique, counts = np.unique(labels, return_counts=True)

    target = np.floor(counts / counts.sum() * max_users).astype(int)
    target = np.maximum(target, 1)
    target = np.minimum(target, counts)

    # Rebalance after rounding.
    while target.sum() > max_users:
        idx = int(np.argmax(target))
        if target[idx] > 1:
            target[idx] -= 1
        else:
            break
    while target.sum() < max_users:
        deficits = counts - target
        idx = int(np.argmax(deficits))
        if deficits[idx] <= 0:
            break
        target[idx] += 1

    indices = []
    for group, group_target in zip(unique, target):
        group_idx = np.where(labels == group)[0]
        if group_idx.size <= group_target:
            indices.append(group_idx)
        else:
            indices.append(rng.choice(group_idx, size=int(group_target), replace=False))
    sampled = np.concatenate(indices) if indices else np.arange(total)
    rng.shuffle(sampled)
    return np.sort(sampled)


def _concat_or_empty(chunks: Iterable[np.ndarray], dtype, ndim: int = 1) -> np.ndarray:
    chunks = [np.asarray(chunk) for chunk in chunks if chunk is not None and np.asarray(chunk).size > 0]
    if chunks:
        return np.concatenate(chunks, axis=0).astype(dtype, copy=False)
    if ndim == 2:
        return np.empty((0, 0), dtype=dtype)
    return np.empty((0,), dtype=dtype)


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
