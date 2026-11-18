from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def merge_params(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge parameter dictionaries from low to high priority."""
    merged: Dict[str, Any] = {}
    for params in dicts:
        if not params:
            continue
        merged.update(params)
    return merged


def merge_nested_params(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge nested parameter dictionaries from low to high priority."""
    merged: Dict[str, Any] = {}
    for params in dicts:
        if not params:
            continue
        merged = _merge_nested_dicts(merged, params)
    return merged


def _merge_nested_dicts(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in incoming.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _merge_nested_dicts(result[key], value)
        else:
            result[key] = value
    return result


def stable_job_id(phase: str,
                  dataset: str,
                  method: str,
                  backbone: str,
                  seed: int,
                  params: Dict[str, Any]) -> str:
    """Build a deterministic job id that is stable across plan regeneration."""
    payload = json.dumps(
        params or {},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    return f"{phase}_{dataset}_{method}_{backbone}_s{int(seed)}_{digest}"


def result_artifact_is_complete(job: Dict[str, Any], results_root: str) -> bool:
    """Return whether a run has the artifacts required by its protocol."""
    run_dir = Path(results_root) / job["dataset"] / job["job_id"]
    result_path = run_dir / "test_results.json"
    validation_path = run_dir / "validation_results.json"
    checkpoint_path = run_dir / "best.pt"
    if not result_path.exists() or not validation_path.exists() or not checkpoint_path.exists():
        return False

    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    validation_selected = validation.get("selected")
    if not isinstance(validation_selected, dict) or not validation_selected:
        return False

    evaluate_test = bool(job.get("params", {}).get("evaluate_test", True))
    if evaluate_test:
        selected = result.get("selected")
        return isinstance(selected, dict) and bool(selected)
    return result.get("status") == "not_run_by_protocol"
