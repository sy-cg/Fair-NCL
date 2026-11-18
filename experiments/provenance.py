"""Per-run environment and dataset provenance for reproducible experiments."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Mapping


def collect_run_provenance(config,
                           data: Mapping[str, Any],
                           workspace: Path | str = ".") -> Dict[str, Any]:
    return {
        "environment": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "torch_version": _package_version("torch"),
            "numpy_version": _package_version("numpy"),
            "pandas_version": _package_version("pandas"),
            "scipy_version": _package_version("scipy"),
            "tqdm_version": _package_version("tqdm"),
            "scikit_learn_version": _package_version("scikit-learn"),
            "matplotlib_version": _package_version("matplotlib"),
            "openpyxl_version": _package_version("openpyxl"),
            "transformers_version": _package_version("transformers"),
            **_torch_environment(getattr(config, "device", "cpu")),
        },
        "code": {
            "git_commit": _git_commit(Path(workspace)),
            "git_dirty": _git_dirty(Path(workspace)),
        },
        "run": {
            "dataset": str(getattr(config, "dataset", data.get("dataset", ""))),
            "seed": int(getattr(config, "seed", 42)),
        },
        "dataset": {
            "num_users": int(data.get("num_users", 0)),
            "num_items": int(data.get("num_items", 0)),
            "num_train_samples": len(data.get("train_data", [])),
            "num_validation_samples": len(data.get("val_data", [])),
            "num_test_samples": len(data.get("test_data", [])),
            "metadata": data.get("metadata", {}),
            "build_config": data.get("build_config", {}),
        },
    }


def write_run_provenance(record: Mapping[str, Any], path: Path | str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def _package_version(package: str):
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None


def _torch_environment(device) -> Dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "cuda_version": None,
            "cudnn_version": None,
            "device": str(device),
            "gpu_name": None,
        }
    device_text = str(device)
    gpu_name = None
    if device_text.startswith("cuda") and torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(device)
        except Exception:
            gpu_name = torch.cuda.get_device_name(0)
    return {
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device": device_text,
        "gpu_name": gpu_name,
    }


def _git_commit(workspace: Path):
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _git_dirty(workspace: Path):
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(completed.stdout.strip()) if completed.returncode == 0 else None
