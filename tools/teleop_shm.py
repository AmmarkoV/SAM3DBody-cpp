#!/usr/bin/env python3
# Python side of robot/teleop_shm.h: publish the retargeted robot pose to a
# bridge process (robot/h1_bridge) over one POSIX shm object.
#
# Layout, seqlock and CRC rules are defined in robot/teleop_shm.h — this file is
# a byte-exact mirror (the offsets are asserted below and in the C header).
import ctypes, mmap, os, time, zlib
import numpy as np

MAGIC, VERSION = 0x314C5054, 1
MAX_JOINTS, NAME_LEN, HAND_CH = 64, 32, 12
F_TRACKING, F_HANDS_VALID, F_SHUTDOWN = 1 << 0, 1 << 1, 1 << 2
CRC_OFFSET = 16


class Header(ctypes.Structure):
    _fields_ = [("magic", ctypes.c_uint32), ("version", ctypes.c_uint32),
                ("n_joints", ctypes.c_uint32), ("n_hand_ch", ctypes.c_uint32),
                ("robot", ctypes.c_char * NAME_LEN),
                ("joint_names", (ctypes.c_char * NAME_LEN) * MAX_JOINTS),
                ("writer_pid", ctypes.c_int64)]


class Frame(ctypes.Structure):
    _fields_ = [("seq", ctypes.c_uint64), ("crc32", ctypes.c_uint32), ("_pad0", ctypes.c_uint32),
                ("counter", ctypes.c_uint64), ("t_mono_ns", ctypes.c_int64),
                ("flags", ctypes.c_uint32), ("_pad1", ctypes.c_uint32),
                ("root", ctypes.c_float * 7), ("_pad2", ctypes.c_float),
                ("q", ctypes.c_float * MAX_JOINTS),
                ("hand", (ctypes.c_float * HAND_CH) * 2)]


class Shm(ctypes.Structure):
    _fields_ = [("header", Header), ("frame", Frame)]


assert ctypes.sizeof(Header) == 2104 and ctypes.sizeof(Frame) == 424 and ctypes.sizeof(Shm) == 2528
assert Frame.counter.offset == 16 and Frame.flags.offset == 32 and Frame.root.offset == 40
assert Frame.q.offset == 72 and Frame.hand.offset == 328 and Shm.frame.offset == 2104


def _path(name):
    return "/dev/shm/" + name.lstrip("/")


class TeleopShmWriter:
    def __init__(self, name, robot, joint_names, n_hand_ch=0):
        if len(joint_names) > MAX_JOINTS:
            raise ValueError(f"{len(joint_names)} joints > {MAX_JOINTS}")
        self.path = _path(name)
        self.n = len(joint_names)
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, ctypes.sizeof(Shm))
            self._mm = mmap.mmap(fd, ctypes.sizeof(Shm))
        finally:
            os.close(fd)
        self._s = Shm.from_buffer(self._mm)
        h = self._s.header
        h.magic = 0                                    # readers ignore us until complete
        h.version, h.n_joints, h.n_hand_ch = VERSION, self.n, n_hand_ch
        h.robot = robot.encode()[:NAME_LEN - 1]
        for i, n in enumerate(joint_names):
            b = n.encode()
            if len(b) >= NAME_LEN:
                raise ValueError(f"joint name too long: {n}")
            h.joint_names[i].value = b
        h.writer_pid = os.getpid()
        f = self._s.frame
        f.seq, f.counter = 0, 0
        h.magic = MAGIC                                # header complete
        self._payload = (ctypes.c_char * (ctypes.sizeof(Frame) - CRC_OFFSET)).from_buffer(
            self._mm, Shm.frame.offset + CRC_OFFSET)

    def publish(self, q, root=None, flags=F_TRACKING, hand=None):
        q = np.asarray(q, dtype=np.float32)
        if q.shape != (self.n,):
            raise ValueError(f"q has shape {q.shape}, expected ({self.n},)")
        f = self._s.frame
        f.seq += 1                                     # odd: write in progress
        f.counter += 1
        f.t_mono_ns = time.monotonic_ns()              # CLOCK_MONOTONIC on Linux
        f.flags = flags | (F_HANDS_VALID if hand is not None else 0)
        ctypes.memmove(f.root, np.ascontiguousarray(
            np.zeros(7, np.float32) if root is None else np.asarray(root, np.float32)).ctypes.data, 28)
        ctypes.memmove(f.q, q.ctypes.data, 4 * self.n)
        if hand is not None:
            hv = np.zeros((2, HAND_CH), np.float32)
            hh = np.asarray(hand, np.float32); hv[:, :hh.shape[1]] = hh
            ctypes.memmove(f.hand, hv.ctypes.data, hv.nbytes)
        f.crc32 = zlib.crc32(bytes(self._payload))
        f.seq += 1                                     # even: consistent

    def close(self, unlink=True):
        """Publish a final SHUTDOWN frame (the bridge ramps out on it), then unmap."""
        try:
            last_q = np.frombuffer(bytes(self._s.frame.q), np.float32)[:self.n].copy()
            self.publish(last_q, flags=F_SHUTDOWN)
        except Exception:
            pass
        self._s = None; self._payload = None
        self._mm.close()
        if unlink:
            try: os.unlink(self.path)
            except FileNotFoundError: pass


class TeleopShmReader:
    """Reference reader (tests / debugging).  The real consumer is the C++ bridge."""
    def __init__(self, name):
        fd = os.open(_path(name), os.O_RDONLY)
        try: self._mm = mmap.mmap(fd, ctypes.sizeof(Shm), prot=mmap.PROT_READ)
        finally: os.close(fd)

    def read(self):
        for _ in range(100):
            raw = self._mm[:]
            s = Shm.from_buffer_copy(raw)
            if s.header.magic != MAGIC or s.frame.seq & 1:
                continue
            fb = raw[Shm.frame.offset:]
            if zlib.crc32(fb[CRC_OFFSET:]) != s.frame.crc32:
                continue
            n = s.header.n_joints
            return dict(counter=s.frame.counter, t_mono_ns=s.frame.t_mono_ns, flags=s.frame.flags,
                        robot=s.header.robot.decode(),
                        names=[s.header.joint_names[i].value.decode() for i in range(n)],
                        q=np.array(s.frame.q[:n]), root=np.array(s.frame.root[:]),
                        hand=np.array([list(r) for r in s.frame.hand]))
        return None
