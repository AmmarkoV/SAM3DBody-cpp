"""Shared, ABI-checked ctypes binding for the SAM3DBody C API.

Keep these structs synchronized with fast_sam_3dbody_capi.h. The native ABI
version must increase even when fields are only appended: array strides change.
This module has no dependency on NumPy, OpenCV, or PyTorch.
"""

import ctypes
import os
from pathlib import Path
import sys


ABI_VERSION = 1


class FsbConfig(ctypes.Structure):
    _fields_ = [
        ("onnx_dir", ctypes.c_char_p),
        ("gguf_path", ctypes.c_char_p),
        ("yolo_path", ctypes.c_char_p),
        ("cuda_device", ctypes.c_int),
        ("skip_body_model", ctypes.c_int),
        ("person_thresh", ctypes.c_float),
        ("person_nms_iou", ctypes.c_float),
        ("max_persons", ctypes.c_int),
        ("focal_x", ctypes.c_float),
        ("focal_y", ctypes.c_float),
        ("principal_x", ctypes.c_float),
        ("principal_y", ctypes.c_float),
        ("zero_face_params", ctypes.c_int),
        ("detector", ctypes.c_int),
    ]


class FsbResult(ctypes.Structure):
    _fields_ = [
        ("bbox", ctypes.c_float * 4),
        ("focal_length", ctypes.c_float),
        ("pred_cam_t", ctypes.c_float * 3),
        ("global_rot", ctypes.c_float * 3),
        ("body_pose", ctypes.c_float * 133),
        ("shape", ctypes.c_float * 45),
        ("scale", ctypes.c_float * 28),
        ("hand_pose", ctypes.c_float * 108),
        ("face_params", ctypes.c_float * 72),
        ("yolo_kps", ctypes.c_float * 51),
        ("has_yolo_kps", ctypes.c_int),
        ("kps_3d", ctypes.c_float * 210),
        ("kps_2d", ctypes.c_float * 140),
        ("has_kps", ctypes.c_int),
        ("pred_pose_raw", ctypes.c_float * 266),
        ("pred_cam_raw", ctypes.c_float * 3),
        ("mhr_model_params", ctypes.c_float * 204),
        ("skel_3d", ctypes.c_float * 381),
        ("has_skel", ctypes.c_int),
    ]


def _check_abi(lib):
    """Reject stale/incompatible libraries before they can read or write structs."""
    checks = (
        ("fsb_abi_version", ctypes.c_uint, ABI_VERSION),
        ("fsb_config_size", ctypes.c_size_t, ctypes.sizeof(FsbConfig)),
        ("fsb_result_size", ctypes.c_size_t, ctypes.sizeof(FsbResult)),
    )
    for name, result_type, expected in checks:
        try:
            query = getattr(lib, name)
        except AttributeError as exc:
            raise RuntimeError(
                f"Native library is missing {name}; rebuild it from the same "
                "checkout as the Python frontend."
            ) from exc
        query.argtypes = []
        query.restype = result_type
        actual = query()
        if actual != expected:
            raise RuntimeError(
                f"Native ABI mismatch: {name} returned {actual}, expected "
                f"{expected}. Rebuild the library and use matching Python files."
            )


def load_library(lib_dir: str) -> ctypes.CDLL:
    """Load a CMake build directory (or a shared-library path) on Windows/Linux."""
    root = Path(lib_dir).expanduser().resolve()
    if sys.platform == "win32":
        names = ("fast_sam_3dbody.dll", "libfast_sam_3dbody.dll")
    elif sys.platform == "darwin":
        names = ("libfast_sam_3dbody.dylib",)
    else:
        names = ("libfast_sam_3dbody.so",)

    if root.is_file():
        lib_path = root
        root = root.parent
    else:
        directories = [root] + [root / config for config in (
            "Release", "RelWithDebInfo", "Debug", "MinSizeRel"
        )]
        candidates = [directory / name for directory in directories for name in names]
        lib_path = next((path for path in candidates if path.is_file()), None)
        if lib_path is None:
            raise FileNotFoundError(
                f"Native library not found under {root}; expected {names[0]}. "
                "Build the project first, or set --lib-dir to its output directory."
            )

    dll_directories = []
    if sys.platform == "win32":
        # Python 3.8+ ignores PATH for dependent DLL lookup unless directories are
        # explicitly registered. Retain handles for providers loaded after CDLL.
        search_dirs = [lib_path.parent, root]
        if not (lib_path.parent / "onnxruntime.dll").is_file():
            build_root = root.parent if root.name in (
                "Release", "RelWithDebInfo", "Debug", "MinSizeRel"
            ) else root
            cache_root = build_root / "onnxruntime_dl"
            packages = sorted(path for path in cache_root.glob("onnxruntime-win-x64*")
                              if path.is_dir() and any(
                                  (path / folder / "onnxruntime.dll").is_file()
                                  for folder in ("lib", "bin")))
            # Windows does not specify an order among AddDllDirectory entries.
            # Never expose both cached CPU and GPU ORT DLLs to that search.
            if len(packages) > 1:
                raise OSError(
                    "Multiple ONNX Runtime packages are cached, but the selected "
                    "output has no staged onnxruntime.dll. Rebuild the target "
                    "and load its output directory so the matching DLL is used."
                )
            ort_root = packages[0] if packages else cache_root  # legacy flat cache
            search_dirs += [ort_root / "lib", ort_root / "bin"]
        search_dirs += [Path(part) for part in os.environ.get("PATH", "").split(os.pathsep)
                        if part]
        seen = set()
        for directory in search_dirs:
            directory = directory.resolve()
            if directory.is_dir() and directory not in seen:
                dll_directories.append(os.add_dll_directory(str(directory)))
                seen.add(directory)

    try:
        # The exported C API uses cdecl, including on Windows: use CDLL, not WinDLL.
        lib = ctypes.CDLL(str(lib_path))
        _check_abi(lib)
        lib.fsb_create.restype = ctypes.c_void_p
        lib.fsb_create.argtypes = []
        lib.fsb_destroy.restype = None
        lib.fsb_destroy.argtypes = [ctypes.c_void_p]
        lib.fsb_load.restype = ctypes.c_int
        lib.fsb_load.argtypes = [ctypes.c_void_p, ctypes.POINTER(FsbConfig)]
        lib.fsb_process_bgr.restype = ctypes.c_int
        lib.fsb_process_bgr.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(FsbResult), ctypes.c_int,
        ]
    except Exception:
        for directory in dll_directories:
            directory.close()
        raise
    lib._fsb_dll_directories = dll_directories
    return lib
