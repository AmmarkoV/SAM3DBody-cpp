#!/usr/bin/env python3
# No-disk LAFAN forward kinematics for the live webcam -> robot pipeline.
#
# gmr_stream.py used to turn each streamed frame into GMR's {joint: (pos, quat)}
# dict by writing a 1-frame .bvh to a temp file and calling GMR's load_bvh_file,
# which RE-PARSES the whole HIERARCHY header AND touches disk every frame.
#
# This module removes both costs while reusing GMR's exact maths: it parses the
# template HIERARCHY once (via GMR's own read_bvh) to cache the static skeleton
# (parents / bone order / euler order / channel layout), then per frame decodes
# just the MOTION channel floats -> euler_to_quat -> GMR's quat_fk -> the same
# Y-up->Z-up + cm->m transform that load_bvh_file applies.  The output dict is
# byte-for-byte what load_bvh_file(format="lafan1") returned, so nothing
# downstream (retarget, despike, viewer) changes.
#
# The per-frame channel decode mirrors read_bvh()'s MOTION handling exactly; see
# GMR/general_motion_retargeting/utils/lafan_vendor/extract.py.
import numpy as np

from general_motion_retargeting.utils.lafan_vendor import utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh, channelmap

# Y-up (BVH) -> Z-up (GMR world) basis, identical to load_bvh_file's constant.
_ROT_M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)


class LafanFK:
    """Reusable, disk-free equivalent of load_bvh_file for a streamed skeleton."""

    def __init__(self, template_path, fmt="lafan1"):
        self.fmt = fmt
        # Parse the HIERARCHY (and its 1 template MOTION line) ONCE.  We keep only
        # the static skeleton; the template's pose values are irrelevant.
        anim = read_bvh(template_path)
        self.parents = anim.parents
        self.offsets = anim.offsets            # (N,3) local joint offsets
        self.bones   = anim.bones
        self.N       = len(self.parents)

        # Re-derive the euler channel order the way read_bvh does, since it does
        # not expose it on the Anim object.  read_bvh picks 'order' from the first
        # CHANNELS line it can map; every LAFAN joint shares that order.
        self.order, self.channels = self._read_channel_layout(template_path)

        # Rotation quat applied to every joint's global orientation (Y-up->Z-up).
        from scipy.spatial.transform import Rotation as R
        self._rot_quat = R.from_matrix(_ROT_M).as_quat(scalar_first=True)
        self._rot_mT   = _ROT_M.T

    @staticmethod
    def _read_channel_layout(path):
        """Return (euler order string e.g. 'zyx', channels-per-joint int)."""
        order = None
        channels = None
        with open(path) as f:
            for line in f:
                if "MOTION" in line:
                    break
                s = line.split()
                if s and s[0] == "CHANNELS":
                    channels = int(s[1])
                    if order is None:
                        # last 3 tokens are the rotation channels (pos precede them)
                        ci = 0 if channels == 3 else 3
                        parts = line.split()[2 + ci:2 + ci + 3]
                        if all(p in channelmap for p in parts):
                            order = "".join(channelmap[p] for p in parts)
        if order is None or channels is None:
            raise ValueError(f"could not read CHANNELS layout from {path}")
        return order, channels

    def _decode_motion(self, data_block):
        """MOTION floats -> (positions (N,3), rotations-euler-deg (N,3)).

        Exact port of read_bvh()'s per-line branch for channels in {3,6,9}."""
        N = self.N
        positions = self.offsets.copy()
        rotations = np.zeros((N, 3))
        c = self.channels
        if c == 3:
            positions[0:1] = data_block[0:3]
            rotations[:]   = data_block[3:].reshape(N, 3)
        elif c == 6:
            db = data_block.reshape(N, 6)
            positions[:]   = db[:, 0:3]
            rotations[:]   = db[:, 3:6]
        elif c == 9:
            positions[0]   = data_block[0:3]
            db = data_block[3:].reshape(N - 1, 9)
            rotations[1:]  = db[:, 3:6]
            positions[1:] += db[:, 0:3] * db[:, 6:9]
        else:
            raise ValueError(f"unsupported channels/joint: {c}")
        return positions, rotations

    def frame_from_channels(self, data_block):
        """One MOTION frame's floats (1-D np.float array) -> GMR frame dict.

        Mirrors load_bvh_file(format=self.fmt): global FK, Y-up->Z-up, cm->m,
        plus the LeftFootMod/RightFootMod synthetic joints GMR expects."""
        data_block = np.asarray(data_block, dtype=float)
        positions, rotations = self._decode_motion(data_block)

        # local eulers(deg) -> local quats, then GMR's forward kinematics.
        lrot = utils.euler_to_quat(np.radians(rotations)[None], order=self.order)  # (1,N,4)
        lpos = positions[None]                                                     # (1,N,3)
        grot, gpos = utils.quat_fk(lrot, lpos, self.parents)                       # (1,N,4),(1,N,3)
        grot = grot[0]
        gpos = gpos[0]

        result = {}
        for i, bone in enumerate(self.bones):
            orientation = utils.quat_mul(self._rot_quat, grot[i])
            position    = gpos[i] @ self._rot_mT / 100.0   # cm -> m
            result[bone] = [position, orientation]

        if self.fmt == "lafan1":
            result["LeftFootMod"]  = [result["LeftFoot"][0],  result["LeftToe"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightToe"][1]]
        elif self.fmt == "nokov":
            result["LeftFootMod"]  = [result["LeftFoot"][0],  result["LeftToeBase"][1]]
            result["RightFootMod"] = [result["RightFoot"][0], result["RightToeBase"][1]]
        return result
