"""Runtime tempfile configuration for long-running experiment jobs."""

from __future__ import annotations

import errno
import os
import tempfile
import warnings
from pathlib import Path


_ORIGINAL_TEMPORARY_DIRECTORY_RMTREE = None
_STALE_CLEANUP_GUARD_INSTALLED = False


def configure_runtime_tempdir(root_dir=None) -> str:
    """Use a project-local temp dir and tolerate ESTALE cleanup at exit."""
    temp_dir = _resolve_tempdir(root_dir)
    os.makedirs(temp_dir, exist_ok=True)

    for key in ("TMPDIR", "TEMP", "TMP"):
        os.environ[key] = temp_dir
    tempfile.tempdir = temp_dir

    install_stale_temp_cleanup_guard()
    return temp_dir


def install_stale_temp_cleanup_guard() -> None:
    """Prevent stale NFS/overlay temp cleanup from failing completed jobs."""
    global _ORIGINAL_TEMPORARY_DIRECTORY_RMTREE
    global _STALE_CLEANUP_GUARD_INSTALLED

    if _STALE_CLEANUP_GUARD_INSTALLED:
        return

    _ORIGINAL_TEMPORARY_DIRECTORY_RMTREE = tempfile.TemporaryDirectory._rmtree

    def _rmtree_ignore_stale(cls, name, ignore_errors=False, repeated=False):
        try:
            return _call_original_tempdir_rmtree(
                name,
                ignore_errors=ignore_errors,
                repeated=repeated,
            )
        except OSError as exc:
            if _is_stale_file_handle(exc):
                warnings.warn(
                    f"Ignoring stale temporary directory cleanup failure: {name}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None
            raise

    tempfile.TemporaryDirectory._rmtree = classmethod(_rmtree_ignore_stale)
    _STALE_CLEANUP_GUARD_INSTALLED = True


def _call_original_tempdir_rmtree(name, ignore_errors=False, repeated=False):
    try:
        return _ORIGINAL_TEMPORARY_DIRECTORY_RMTREE(
            name,
            ignore_errors=ignore_errors,
            repeated=repeated,
        )
    except TypeError as exc:
        if "unexpected keyword argument 'repeated'" not in str(exc):
            raise
        return _ORIGINAL_TEMPORARY_DIRECTORY_RMTREE(
            name,
            ignore_errors=ignore_errors,
        )


def _resolve_tempdir(root_dir=None) -> str:
    configured = os.environ.get("FAIR_NCL_TMPDIR")
    if configured:
        return str(Path(configured).expanduser().resolve())

    root = Path(root_dir) if root_dir is not None else Path.cwd()
    return str((root / "cache" / "tmp").resolve())


def _is_stale_file_handle(error: OSError) -> bool:
    return getattr(error, "errno", None) == errno.ESTALE
