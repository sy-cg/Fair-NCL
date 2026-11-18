"""Initialize frozen fairness baselines from matched validation references."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional


FROZEN_PRETRAINED_METHODS = {"afrl", "pfrec"}


def initialize_frozen_backbone(model,
                               job: Mapping[str, Any],
                               results_root: Path | str,
                               checkpoint_loader: Callable[[Path], Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Load a seed-matched baseline backbone before AFRL/PFRec training."""
    method = str(job.get("method", "")).lower()
    if method not in FROZEN_PRETRAINED_METHODS or not bool(getattr(model, "freeze_backbone", False)):
        return None

    source_job, checkpoint_path = _find_matched_reference(job, Path(results_root))
    checkpoint = checkpoint_loader(checkpoint_path)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Baseline checkpoint is not a mapping: {checkpoint_path}")
    _validate_checkpoint_job(checkpoint.get("job"), source_job, checkpoint_path)

    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError(f"Baseline checkpoint has no model_state_dict: {checkpoint_path}")
    prefix = "backbone."
    backbone_state = {
        str(key)[len(prefix):]: value
        for key, value in state.items()
        if str(key).startswith(prefix)
    }
    if not backbone_state:
        raise ValueError(
            f"Baseline checkpoint contains no '{prefix}*' parameters: {checkpoint_path}"
        )
    model.backbone.load_state_dict(backbone_state, strict=True)

    relative_checkpoint = checkpoint_path.relative_to(Path(results_root)).as_posix()
    return {
        "strategy": "matched_validation_reference",
        "source_job_id": source_job["job_id"],
        "source_phase": source_job["phase"],
        "source_checkpoint": relative_checkpoint,
        "dataset": source_job["dataset"],
        "backbone": source_job["backbone"],
        "seed": int(source_job["seed"]),
    }


def enforce_frozen_backbone_eval(model) -> None:
    """Keep a frozen pretrained backbone deterministic during wrapper training."""
    if bool(getattr(model, "freeze_backbone", False)):
        model.backbone.eval()


def _find_matched_reference(job: Mapping[str, Any], results_root: Path):
    dataset = str(job["dataset"])
    dataset_root = results_root / dataset
    matches = []
    if dataset_root.exists():
        for job_path in sorted(dataset_root.glob("*/job.json")):
            source_job = json.loads(job_path.read_text(encoding="utf-8"))
            if (
                source_job.get("phase") == "validation_reference"
                and source_job.get("method") == "baseline"
                and source_job.get("dataset") == dataset
                and source_job.get("backbone") == job.get("backbone")
                and int(source_job.get("seed", -1)) == int(job.get("seed", -2))
            ):
                matches.append((source_job, job_path.with_name("best.pt")))

    identity = f"{dataset}:{job.get('backbone')}:seed={job.get('seed')}"
    if not matches:
        raise FileNotFoundError(
            f"No matched validation_reference baseline checkpoint for {identity}. "
            "Run experiments/revision_baseline_reference.jsonl in the same results root first."
        )
    if len(matches) != 1:
        source_ids = ", ".join(str(source["job_id"]) for source, _ in matches)
        raise ValueError(
            f"Found multiple matched validation_reference baselines for {identity}: {source_ids}"
        )
    source_job, checkpoint_path = matches[0]
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Matched validation_reference is missing best.pt: {checkpoint_path}"
        )
    return source_job, checkpoint_path


def _validate_checkpoint_job(checkpoint_job,
                             source_job: Mapping[str, Any],
                             checkpoint_path: Path) -> None:
    if not isinstance(checkpoint_job, Mapping):
        raise ValueError(f"Baseline checkpoint has no job metadata: {checkpoint_path}")
    for key in ("job_id", "phase", "dataset", "method", "backbone", "seed"):
        checkpoint_value = checkpoint_job.get(key)
        source_value = source_job.get(key)
        if str(checkpoint_value) != str(source_value):
            raise ValueError(
                f"Baseline checkpoint job metadata mismatch for {key}: "
                f"expected {source_value!r}, found {checkpoint_value!r} in {checkpoint_path}"
            )
