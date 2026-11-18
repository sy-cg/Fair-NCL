"""Summarize supplementary-run efficiency from saved run artifacts.

The benchmark intentionally reads only JSON artifacts produced by completed
runs.  It does not load processed_data/*.pkl, because those pickles may contain
pandas extension dtypes that are not portable across the AutoDL environments.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


RESULT_DIR_COLUMNS = ("result_dir", "run_dir", "run_path", "job_dir", "output_dir", "experiment_dir", "path")
RESULT_ROOT_COLUMNS = ("results_root", "result_root")
IDENTITY_FIELDS = (
    "manifest_index",
    "dataset",
    "backbone",
    "method",
    "seed",
    "phase",
    "job_id",
    "result_dir",
)
OUTPUT_FIELDS = IDENTITY_FIELDS + (
    "status",
    "validation_only",
    "training_seconds",
    "test_evaluation_seconds",
    "total_runtime_seconds",
    "resource_build_seconds",
    "total_wall_seconds",
    "parameter_count",
    "peak_cuda_memory_bytes",
    "peak_cuda_memory_gb",
    "cache_bytes",
    "similarity_items",
    "low_skew_pool_items",
    "num_users",
    "num_items",
    "num_train_samples",
    "training_seconds_per_train_sample",
    "wall_seconds_per_train_sample",
    "HitRate@10",
    "NDCG@10",
    "HitRate@20",
    "NDCG@20",
    "gpu_name",
    "torch_version",
    "git_commit",
)


def benchmark_manifest(manifest: Path | str,
                       output: Path | str,
                       audit: Path | str) -> Dict[str, Any]:
    manifest_path = Path(manifest)
    output_path = Path(output)
    audit_path = Path(audit)

    rows = _read_manifest(manifest_path)
    output_rows = []
    missing_rows = []
    error_rows = []

    for index, manifest_row in enumerate(rows):
        try:
            result_dir = _resolve_result_dir(manifest_row, manifest_path)
            if result_dir is None:
                missing_rows.append({
                    "manifest_index": index,
                    "reason": "cannot_resolve_result_dir",
                    "row": manifest_row,
                })
                continue

            job_path = result_dir / "job.json"
            result_path = result_dir / "test_results.json"
            if not job_path.exists() or not result_path.exists():
                missing_rows.append({
                    "manifest_index": index,
                    "reason": "missing_run_artifacts",
                    "result_dir": str(result_dir),
                    "missing_files": [
                        name
                        for name, path in (
                            ("job.json", job_path),
                            ("test_results.json", result_path),
                        )
                        if not path.exists()
                    ],
                    "row": manifest_row,
                })
                continue

            job = _read_json(job_path)
            result = _read_json(result_path)
            resource = _read_optional_json(result_dir / "resource_diagnostics.json")
            provenance = _read_optional_json(result_dir / "run_provenance.json")
            output_rows.append(
                _build_output_row(
                    manifest_index=index,
                    manifest_row=manifest_row,
                    result_dir=result_dir,
                    job=job,
                    result=result,
                    resource=resource,
                    provenance=provenance,
                )
            )
        except Exception as exc:  # keep one malformed row from losing all rows
            error_rows.append({
                "manifest_index": index,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "row": manifest_row,
            })

    _write_csv(output_path, output_rows)
    audit_doc = {
        "manifest": str(manifest_path),
        "output": str(output_path),
        "input_rows": len(rows),
        "processed_rows": len(output_rows),
        "missing_rows": missing_rows,
        "error_rows": error_rows,
        "datasets": sorted({str(row.get("dataset", "")) for row in output_rows if row.get("dataset")}),
        "note": "JSON-artifact summary only; processed_data pickle files are not loaded.",
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return audit_doc


def _read_manifest(path: Path) -> list[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _resolve_result_dir(row: Mapping[str, str], manifest_path: Path) -> Path | None:
    for column in RESULT_DIR_COLUMNS:
        value = _clean(row.get(column))
        if value:
            return _resolve_path(value, manifest_path)

    dataset = _clean(row.get("dataset"))
    job_id = _clean(row.get("job_id"))
    if not dataset or not job_id:
        return None

    result_root = None
    for column in RESULT_ROOT_COLUMNS:
        result_root = _clean(row.get(column))
        if result_root:
            break
    base = _resolve_path(result_root or "results", manifest_path)
    return base / dataset / job_id


def _resolve_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    cwd_path = path
    if cwd_path.exists():
        return cwd_path

    manifest_relative = manifest_path.parent / path
    if manifest_relative.exists():
        return manifest_relative

    return cwd_path


def _build_output_row(manifest_index: int,
                      manifest_row: Mapping[str, str],
                      result_dir: Path,
                      job: Mapping[str, Any],
                      result: Mapping[str, Any],
                      resource: Mapping[str, Any],
                      provenance: Mapping[str, Any]) -> Dict[str, Any]:
    runtime = _mapping(result.get("runtime"))
    utility = _mapping(result.get("utility"))
    dataset_info = _mapping(provenance.get("dataset"))
    environment = _mapping(provenance.get("environment"))
    code = _mapping(provenance.get("code"))

    training_seconds = _number(runtime.get("training_seconds"))
    evaluation_seconds = _number(runtime.get("test_evaluation_seconds"))
    resource_seconds = _number(resource.get("resource_build_seconds"))
    total_runtime = _sum_present(training_seconds, evaluation_seconds)
    total_wall = _sum_present(total_runtime, resource_seconds)
    peak_bytes = _number(runtime.get("peak_cuda_memory_bytes"))
    train_samples = _number(dataset_info.get("num_train_samples"))

    row: Dict[str, Any] = {
        "manifest_index": manifest_index,
        "dataset": _first_present(job.get("dataset"), manifest_row.get("dataset")),
        "backbone": _first_present(job.get("backbone"), manifest_row.get("backbone")),
        "method": _first_present(job.get("method"), manifest_row.get("method")),
        "seed": _first_present(job.get("seed"), manifest_row.get("seed")),
        "phase": _first_present(job.get("phase"), manifest_row.get("phase")),
        "job_id": _first_present(job.get("job_id"), manifest_row.get("job_id"), result_dir.name),
        "result_dir": str(result_dir),
        "status": result.get("status", "completed"),
        "validation_only": bool(result.get("validation_only", False)),
        "training_seconds": training_seconds,
        "test_evaluation_seconds": evaluation_seconds,
        "total_runtime_seconds": total_runtime,
        "resource_build_seconds": resource_seconds,
        "total_wall_seconds": total_wall,
        "parameter_count": _number(runtime.get("parameter_count")),
        "peak_cuda_memory_bytes": peak_bytes,
        "peak_cuda_memory_gb": peak_bytes / (1024 ** 3) if peak_bytes is not None else None,
        "cache_bytes": _number(resource.get("cache_bytes")),
        "similarity_items": _number(resource.get("similarity_items")),
        "low_skew_pool_items": _number(resource.get("low_skew_pool_items")),
        "num_users": _number(dataset_info.get("num_users")),
        "num_items": _number(dataset_info.get("num_items")),
        "num_train_samples": train_samples,
        "training_seconds_per_train_sample": _safe_divide(training_seconds, train_samples),
        "wall_seconds_per_train_sample": _safe_divide(total_wall, train_samples),
        "gpu_name": environment.get("gpu_name"),
        "torch_version": environment.get("torch_version"),
        "git_commit": code.get("git_commit"),
    }
    for metric in ("HitRate@10", "NDCG@10", "HitRate@20", "NDCG@20"):
        row[metric] = _number(utility.get(metric))
    return row


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_optional_json(path: Path) -> Dict[str, Any]:
    return _read_json(path) if path.exists() else {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _clean(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and str(value) != "":
            return value
    return None


def _number(value: Any) -> float | int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else number


def _sum_present(*values: float | int | None) -> float | int | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present)


def _safe_divide(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(_jsonable_row(row) for row in rows)


def _jsonable_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: "" if value is None else value for key, value in row.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Fair-NCL revision efficiency from saved run artifacts."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    args = parser.parse_args()

    audit_doc = benchmark_manifest(args.manifest, args.output, args.audit)
    print(json.dumps({
        "input_rows": audit_doc["input_rows"],
        "processed_rows": audit_doc["processed_rows"],
        "missing_rows": len(audit_doc["missing_rows"]),
        "error_rows": len(audit_doc["error_rows"]),
        "datasets": audit_doc["datasets"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

