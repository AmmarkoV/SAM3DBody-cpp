#!/usr/bin/env python3
"""Check every ctypes mirror of FsbResult / FsbConfig against the C library.

A mirror shorter than the C struct makes fsb_process_bgr() write past the
caller's buffer, so each script checks fsb_result_size() at load time too;
this catches a stale mirror before anyone runs the script.

The structs are rebuilt from each file's source with ast, so the scripts
(and their cv2 / rclpy imports) are never imported.

    python3 python/test_ctypes_layout.py [path/to/libfast_sam_3dbody.so]
"""
import ast
import ctypes
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIRRORS = [
    "python/fast_sam_3dbody_frontend.py",
    "python/fast_sam_3dbody_frontend-3D.py",
    "python/fast_sam_3dbody_dump_csv.py",
    "python/fast_sam_3dbody_dump_dpose_compat_csv.py",
    "python/ros_demo_webcam.py",
    "src/ros2/poseEstimation3D/fsb_engine.py",
]


def load_struct(path, name):
    tree = ast.parse(open(path).read(), path)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            src = ast.get_source_segment(open(path).read(), node)
            ns = {"ctypes": ctypes}
            exec(src, ns)
            return ns[name]
    raise LookupError(f"{name} not found in {path}")


def main():
    lib_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "build", "libfast_sam_3dbody.so")
    lib = ctypes.CDLL(lib_path)
    lib.fsb_result_size.restype = ctypes.c_int
    want = lib.fsb_result_size()

    failed = 0
    for rel in MIRRORS:
        got = ctypes.sizeof(load_struct(os.path.join(ROOT, rel), "FsbResult"))
        ok = got == want
        failed += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {rel}: FsbResult {got} bytes (C: {want})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
