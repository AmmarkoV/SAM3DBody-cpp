#!/usr/bin/env python3
"""Check every ctypes mirror of FsbResult / FsbConfig against the C library.

A mirror shorter than the C struct makes fsb_process_bgr() write past the
caller's buffer, so load_library() checks fsb_result_size() at load time too;
this catches a stale mirror before anyone runs a script.

The scripts in python/ import the structs from python/fsb_capi.py; the ROS 2
package keeps its own copy (it is installed on its own). A FsbResult class
defined anywhere else in python/, tools/ or src/ fails the test: import it
from fsb_capi instead.

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
    "python/fsb_capi.py",
    "src/ros2/poseEstimation3D/fsb_engine.py",
]


def stray_mirrors():
    """python/, tools/ and src/ files that define their own FsbResult but are not in MIRRORS"""
    out = []
    for top in ("python", "tools", "src"):
        for d, _, files in os.walk(os.path.join(ROOT, top)):
            for f in files:
                path = os.path.join(d, f)
                rel = os.path.relpath(path, ROOT)
                if not f.endswith(".py") or rel in MIRRORS:
                    continue
                try:
                    tree = ast.parse(open(path).read(), path)
                except (SyntaxError, UnicodeDecodeError):
                    continue
                if any(isinstance(n, ast.ClassDef) and n.name == "FsbResult" for n in ast.walk(tree)):
                    out.append(rel)
    return out


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
    for rel in stray_mirrors():
        failed += 1
        print(f"FAIL {rel}: defines its own FsbResult — import it from python/fsb_capi.py")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
