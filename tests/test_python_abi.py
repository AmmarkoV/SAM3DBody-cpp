"""C/Python ABI and native loader checks, without downloading any models.

Pass --probe /path/to/test_python_abi_layout and optionally --lib-dir /build/dir.
Without --probe, only the failure guards and loader tests run.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import fsb_ctypes as fsb

PROBE = os.environ.get("FSB_ABI_PROBE")
LIB_DIR = os.environ.get("FSB_TEST_LIB_DIR")


class NativeAbiTests(unittest.TestCase):
    def test_c_struct_layout(self):
        if not PROBE:
            self.skipTest("Pass --probe for compiled C vs ctypes layout validation")
        layout = json.loads(subprocess.check_output([PROBE], text=True))
        self.assertEqual(layout["abi_version"], fsb.ABI_VERSION)
        for structure in (fsb.FsbConfig, fsb.FsbResult):
            native = dict(layout[structure.__name__])
            self.assertEqual(native.pop("size"), ctypes.sizeof(structure))
            self.assertEqual(set(native), {name for name, _ in structure._fields_})
            for name, field_type in structure._fields_:
                with self.subTest(structure=structure.__name__, field=name):
                    self.assertEqual(native[name], [getattr(structure, name).offset,
                                                    ctypes.sizeof(field_type)])

    def test_real_library_exports_and_lifecycle(self):
        if not LIB_DIR:
            self.skipTest("Pass --lib-dir for built DLL/shared-library validation")
        lib = fsb.load_library(LIB_DIR)
        handle = lib.fsb_create()
        self.assertTrue(handle)
        try:
            # Invalid inputs take the native guard path without loading models.
            self.assertEqual(lib.fsb_load(handle, None), 0)
            self.assertEqual(lib.fsb_process_bgr(handle, None, 0, 0, None, 0), 0)
        finally:
            lib.fsb_destroy(handle)


class AbiRejectionTests(unittest.TestCase):
    @staticmethod
    def library(**overrides):
        values = {
            "fsb_abi_version": fsb.ABI_VERSION,
            "fsb_config_size": ctypes.sizeof(fsb.FsbConfig),
            "fsb_result_size": ctypes.sizeof(fsb.FsbResult),
        }
        values.update(overrides)
        return SimpleNamespace(**{name: Mock(return_value=value)
                                  for name, value in values.items()})

    def test_old_library_without_version_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "missing fsb_abi_version"):
            fsb._check_abi(SimpleNamespace())

    def test_changed_version_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "fsb_abi_version"):
            fsb._check_abi(self.library(fsb_abi_version=fsb.ABI_VERSION + 1))

    def test_short_config_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "fsb_config_size"):
            fsb._check_abi(self.library(fsb_config_size=ctypes.sizeof(fsb.FsbConfig) - 8))

    def test_old_result_stride_is_rejected(self):
        # This is the previous frontend layout: it ended before skel_3d/has_skel.
        with self.assertRaisesRegex(RuntimeError, "fsb_result_size"):
            fsb._check_abi(self.library(fsb_result_size=fsb.FsbResult.skel_3d.offset))

    def test_windows_release_loader_keeps_search_handles(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            root = Path(folder)
            output = root / "Release"
            output.mkdir()
            library_path = output / "fast_sam_3dbody.dll"
            library_path.touch()
            lib = self.library()
            for name in ("fsb_create", "fsb_destroy", "fsb_load", "fsb_process_bgr"):
                setattr(lib, name, Mock())
            with patch.object(fsb.sys, "platform", "win32"), \
                 patch.dict(os.environ, {"PATH": ""}), \
                 patch.object(fsb.os, "add_dll_directory", create=True) as add_directory, \
                 patch.object(fsb.ctypes, "CDLL", return_value=lib) as cdll:
                handle = Mock()
                add_directory.return_value = handle
                self.assertIs(fsb.load_library(str(root)), lib)
                cdll.assert_called_once_with(str(library_path))
                self.assertEqual(add_directory.call_count, 2)
                self.assertEqual(len(lib._fsb_dll_directories), 2)
                handle.close.assert_not_called()
                self.assertEqual(lib.fsb_process_bgr.argtypes[-2],
                                 ctypes.POINTER(fsb.FsbResult))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe")
    parser.add_argument("--lib-dir")
    args, remainder = parser.parse_known_args()
    PROBE = args.probe or PROBE
    LIB_DIR = args.lib_dir or LIB_DIR
    unittest.main(argv=[sys.argv[0], *remainder])
