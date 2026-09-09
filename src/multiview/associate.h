// ════════════════════════════════════════════════════════════════════════════
//  associate.h — cross-view time alignment + identity association
//
//  Given the per-stream results (each lifted to world via T_world_cam and timed
//  by its QR fit), build a master timeline over the common overlap and group
//  the per-stream detections into global persons.
//
//  Time: every stream maps a master unix time t to a frame index via its
//  (t0_ms, fps_eff); the streams started at different wall-clock times, so each
//  uses a different absolute frame for the same t (start_frame + relative idx).
//
//  Identity: tracks are matched across streams by the mean world-space distance
//  of their root positions over the ticks where both are present (greedy per
//  stream pair, then union-find into global persons).  Robust enough for a
//  sparse scene; revisit for crowds.
// ════════════════════════════════════════════════════════════════════════════

#ifndef MULTIVIEW_ASSOCIATE_H_INCLUDED
#define MULTIVIEW_ASSOCIATE_H_INCLUDED

#include <vector>

#include "multiview_frontend.h"   // mv::StreamResult

namespace mv
{

// One camera's view of a person at one master tick.
struct ViewObs
{
    int stream    = -1;
    int frame_rel = -1;   // index into StreamResult::frames
    int det       = -1;   // index into frames[frame_rel].detections
};

struct PersonTick { std::vector<ViewObs> views; };   // empty if person absent this tick

struct Correspondences
{
    double t_start_ms = 0, dt_ms = 0;
    int    nticks     = 0;
    int    n_persons  = 0;

    std::vector<std::vector<PersonTick>> ticks;        // [tick][person]
    std::vector<std::vector<int>>        person_track; // [person][stream] = track id or -1
};

// `match_thresh_m` is the max mean world-root distance to call two tracks the
// same person.  Streams that produced no frames (e.g. a decode gap outside the
// window) are simply absent.
Correspondences associate(const std::vector<StreamResult>& streams,
                          double master_fps,
                          double match_thresh_m = 0.6);

} // namespace mv

#endif // MULTIVIEW_ASSOCIATE_H_INCLUDED
