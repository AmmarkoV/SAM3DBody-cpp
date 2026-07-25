#!/usr/bin/env python3
# Consumer side of the live webcam -> robot shared-memory transport.
#
# The C++ binary (--bvh-shm) publishes each frame's BVH channel floats into a
# SharedMemoryVideoBuffers "generic" buffer, using the buffer's timestamp field
# as a MONOTONIC FRAME COUNTER (real timestamps are only seconds-resolution).
# This reader waits for the descriptor + feed to appear (the feed is created
# lazily on the first detected person), then blocks until the counter advances
# and returns the frame's floats as a 1-D np.float32 array.
#
# We bind the C functions directly rather than use the shipped SharedMemory
# Manager, whose read_from_shared_memory() force-casts to uint8 (which would
# destroy float data) and assumes the feed already exists at connect time.
import ctypes
import time
import numpy as np


class ShmBvhReader:
    def __init__(self, lib_path, descriptor, stream, connect_timeout=30.0, poll_s=0.001):
        self.poll_s = poll_s
        self._last_ts = 0
        self._lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        self._bind()
        L = self._lib

        desc = descriptor.encode(); strm = stream.encode()

        # 1) Wait for the descriptor (the C++ side creates it at startup).
        deadline = time.time() + connect_timeout
        self.ctx = None
        while time.time() < deadline:
            self.ctx = L.connectToSharedMemoryContextDescriptor(desc)
            if self.ctx:
                break
            time.sleep(0.02)
        if not self.ctx:
            raise RuntimeError(f"shm descriptor '{descriptor}' never appeared")

        # 2) Wait for the feed itself (created lazily on the first frame) and for
        #    its backing shared memory to be populated, then map it locally.
        self.item = -1
        while time.time() < deadline:
            self.item = L.resolveFeedNameToID(self.ctx, strm)
            if self.item >= 0 and L.remoteSharedMemoryContextVideoFrameIsPopulated(self.ctx, self.item):
                break
            time.sleep(0.02)
        if self.item < 0:
            raise RuntimeError(f"shm feed '{stream}' never appeared on '{descriptor}'")

        self.frame = L.getVideoBufferPointer(self.ctx, strm)
        if not self.frame:
            raise RuntimeError(f"could not get shm buffer pointer for '{stream}'")
        self.local_map = L.allocateLocalMapping()
        if not self.local_map or not L.mapRemoteToLocal(self.ctx, self.local_map, self.item):
            raise RuntimeError(f"could not map shm feed '{stream}' locally")

    def _bind(self):
        L = self._lib
        sig = [
            ("connectToSharedMemoryContextDescriptor", [ctypes.c_char_p], ctypes.c_void_p),
            ("resolveFeedNameToID", [ctypes.c_void_p, ctypes.c_char_p], ctypes.c_int),
            ("remoteSharedMemoryContextVideoFrameIsPopulated", [ctypes.c_void_p, ctypes.c_uint], ctypes.c_int),
            ("getVideoBufferPointer", [ctypes.c_void_p, ctypes.c_char_p], ctypes.c_void_p),
            ("allocateLocalMapping", [], ctypes.c_void_p),
            ("mapRemoteToLocal", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int], ctypes.c_int),
            ("getLocalMappingPointer", [ctypes.c_void_p, ctypes.c_int], ctypes.POINTER(ctypes.c_byte)),
            ("freeLocalMapping", [ctypes.c_void_p], ctypes.c_int),
            ("startReadingFromVideoBufferPointer", [ctypes.c_void_p], ctypes.c_int),
            ("stopReadingFromVideoBufferPointer", [ctypes.c_void_p], ctypes.c_int),
            ("getVideoFrameDataSize", [ctypes.c_void_p], ctypes.c_ulong),
            ("getVideoFrameTimestamp", [ctypes.c_void_p], ctypes.c_ulong),
        ]
        for name, args, res in sig:
            fn = getattr(L, name)
            fn.argtypes = args
            fn.restype = res

    def read(self, timeout=None):
        """Block until a new frame (counter advanced) and return its floats, or
        None if `timeout` seconds elapse with no new frame."""
        L = self._lib
        deadline = None if timeout is None else time.time() + timeout
        while True:
            if not L.startReadingFromVideoBufferPointer(self.frame):
                time.sleep(self.poll_s)
            else:
                ts = int(L.getVideoFrameTimestamp(self.frame))
                if ts != 0 and ts != self._last_ts:
                    self._last_ts = ts
                    size = int(L.getVideoFrameDataSize(self.frame))
                    n = size // 4
                    ptr = L.getLocalMappingPointer(self.local_map, self.item)
                    fptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))
                    arr = np.ctypeslib.as_array(fptr, shape=(n,)).astype(np.float64)
                    L.stopReadingFromVideoBufferPointer(self.frame)
                    return arr
                L.stopReadingFromVideoBufferPointer(self.frame)
            if deadline is not None and time.time() >= deadline:
                return None
            time.sleep(self.poll_s)

    def close(self):
        try:
            if getattr(self, "local_map", None):
                self._lib.freeLocalMapping(self.local_map)
                self.local_map = None
        except Exception:
            pass
