#!/usr/bin/env python3
"""
Greedy 2D-bbox IoU tracker — stable person ids across frames.

The C engine returns per-frame detections without identity, so the ids the node
publishes (Skeleton.id, the human_<id> TF frames) come from here.  This is a
port of the tracker in src/net/sam_3dbody_net.cpp — same greedy matching, same
thresholds — so a person keeps the same id whether they are tracked through the
ROS node or through the net server's BVH files.

It replaces the SORT tracker magician_body_pose_estimation ran on the detector
output; there is no Kalman prediction, which for a 15-30 fps camera and people
walking around is not the part that matters.
"""


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return 0.0
    intersection = width * height
    union = ((a[2] - a[0]) * (a[3] - a[1]) +
             (b[2] - b[0]) * (b[3] - b[1]) - intersection)
    return intersection / union if union > 0 else 0.0


class IoUTracker:
    """Assign stable ids to this frame's bounding boxes."""

    def __init__(self, iou_threshold=0.10, retire_frames=90):
        self.iou_threshold = iou_threshold
        self.retire_frames = retire_frames      # ~3 s @ 30 fps
        self._tracks = []                       # [{'id', 'bbox', 'last_seen'}]
        self._next_id = 0
        self._frame = 0

    def assign(self, boxes):
        """boxes: list of (x1, y1, x2, y2) -> list of track ids, same order."""
        self._frame += 1
        ids = [-1] * len(boxes)

        candidates = [
            (_iou(box, track['bbox']), d, t)
            for d, box in enumerate(boxes)
            for t, track in enumerate(self._tracks)
        ]
        candidates = [c for c in candidates if c[0] >= self.iou_threshold]
        candidates.sort(key=lambda c: c[0], reverse=True)

        used_detections, used_tracks = set(), set()
        for _, d, t in candidates:
            if d in used_detections or t in used_tracks:
                continue
            used_detections.add(d)
            used_tracks.add(t)
            ids[d] = self._tracks[t]['id']
            self._tracks[t]['bbox'] = boxes[d]
            self._tracks[t]['last_seen'] = self._frame

        for d, box in enumerate(boxes):
            if ids[d] >= 0:
                continue
            ids[d] = self._next_id
            self._tracks.append({'id': self._next_id, 'bbox': box,
                                 'last_seen': self._frame})
            self._next_id += 1

        self._tracks = [t for t in self._tracks
                        if self._frame - t['last_seen'] <= self.retire_frames]
        return ids
