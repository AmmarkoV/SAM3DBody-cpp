// ════════════════════════════════════════════════════════════════════════════
//  sync_io.h — read/write the session `.sync` file
//
//  `.sync` bundles, per camera, everything the multi-view fusion step needs:
//  the video path, intrinsics (optional), the extrinsic T_world←cam, and the
//  QR temporal fit (t0_ms + fps_eff).  It is a line-based, keyword-led text
//  format (like `.calib`/`.bvh`) so the C++ runtime parses it with a few
//  getline + istringstream calls.  See MULTIVIEW_PLAN.md.
//
//  Grammar (blank lines and '#' comments ignored; `video` takes the rest of
//  its line):
//      SESSION
//      master_fps <f>
//      CAMERA <index>
//        video <path>
//        size <w> <h>
//        calib <0|1>
//        K <fx> <fy> <cx> <cy>            (present iff calib==1)
//        D <k1> <k2> <p1> <p2> <k3>       (present iff calib==1)
//        placed <0|1>
//        T_world_cam <16 floats row-major> (present iff placed==1)
//        time <0|1> <t0_ms> <fps_eff> <resid_med_ms>
//      CAMERA <index> ...
// ════════════════════════════════════════════════════════════════════════════

#ifndef MULTIVIEW_SYNC_IO_H_INCLUDED
#define MULTIVIEW_SYNC_IO_H_INCLUDED

#include <string>
#include <vector>

#include "extrinsics.h"     // mv::Mat4

namespace mv
{

struct SyncCamera
{
    std::string video;
    int    width = 0, height = 0;

    bool   has_calib = false;
    double fx = 0, fy = 0, cx = 0, cy = 0;   // intrinsics at the video resolution
    double dist[5] = {0,0,0,0,0};            // k1 k2 p1 p2 k3

    bool   placed = false;
    Mat4   T_world_cam{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};

    bool   has_time = false;
    double t0_ms = 0, fps_eff = 0, resid_med_ms = 0;
};

struct SyncSession
{
    double                   master_fps = 0;
    std::vector<SyncCamera>  cameras;
};

bool write_sync(const std::string& path, const SyncSession& s);
bool read_sync (const std::string& path, SyncSession& s);

} // namespace mv

#endif // MULTIVIEW_SYNC_IO_H_INCLUDED
