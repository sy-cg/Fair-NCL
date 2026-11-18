import errno
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.runtime_temp import configure_runtime_tempdir, install_stale_temp_cleanup_guard


class RuntimeTempTests(unittest.TestCase):
    def test_configure_runtime_tempdir_uses_workspace_cache_tmp(self):
        original_tempdir = tempfile.tempdir
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            try:
                with patch.dict(os.environ, {}, clear=True):
                    temp_path = configure_runtime_tempdir(root)

                    self.assertEqual(Path(temp_path), root / "cache" / "tmp")
                    self.assertEqual(os.environ["TMPDIR"], str(root / "cache" / "tmp"))
                    self.assertEqual(os.environ["TEMP"], str(root / "cache" / "tmp"))
                    self.assertEqual(os.environ["TMP"], str(root / "cache" / "tmp"))
                    self.assertEqual(tempfile.tempdir, str(root / "cache" / "tmp"))
                    self.assertTrue((root / "cache" / "tmp").is_dir())
            finally:
                tempfile.tempdir = original_tempdir

    def test_stale_temp_cleanup_is_ignored_at_interpreter_exit(self):
        install_stale_temp_cleanup_guard()

        with patch("utils.runtime_temp._ORIGINAL_TEMPORARY_DIRECTORY_RMTREE") as original:
            original.side_effect = OSError(errno.ESTALE, "Stale file handle")
            with self.assertWarns(RuntimeWarning):
                tempfile.TemporaryDirectory._rmtree("stale-temp-dir")

        original.assert_called_once()

    def test_stale_temp_cleanup_supports_python310_rmtree_signature(self):
        install_stale_temp_cleanup_guard()
        calls = []

        def original_rmtree(name, ignore_errors=False):
            calls.append((name, ignore_errors))
            return "cleaned"

        with patch("utils.runtime_temp._ORIGINAL_TEMPORARY_DIRECTORY_RMTREE", original_rmtree):
            result = tempfile.TemporaryDirectory._rmtree("temp-dir", ignore_errors=True)

        self.assertEqual(result, "cleaned")
        self.assertEqual(calls, [("temp-dir", True)])


if __name__ == "__main__":
    unittest.main()
