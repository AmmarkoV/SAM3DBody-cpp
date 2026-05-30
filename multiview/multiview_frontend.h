// ════════════════════════════════════════════════════════════════════════════
//  multiview_frontend.h — per-stream inference for the multi-view fusion path
//
//  Step 2 front-end (see MULTIVIEW_PLAN.md): run the existing offline pipeline
//  (Pass 1 inference + Pass 2 tracking) on one camera's video, with that
//  camera's calibrated intrinsics from the `.sync` injected, so every stream's
//  poses share a consistent metric camera model.  The caller lifts each
//  stream's results into world coordinates with `T_world_cam` and then
//  time-aligns / associates / fuses across streams.
//
//  Loading a pipeline per stream (rather than reconfiguring one) keeps the
//  per-camera focal length simple; the cost is dwarfed by the per-video
//  inference itself.
// ════════════════════════════════════════════════════════════════════════════

#ifndef MULTIVIEW_FRONTEND_H_INCLUDED
#define MULTIVIEW_FRONTEND_H_INCLUDED

#include <vector>

#include "offline_passes.h"   // offline::FrameRecord / Track / Config
#include "sync_io.h"          // mv::SyncCamera (intrinsics + T_world_cam + time)

namespace mv
{

struct StreamResult
{
    int    width = 0, height = 0;
    double fps   = 0.0;

    // Geometry / time copied from the .sync camera (for the caller's lift +
    // alignment).
    Mat4   T_world_cam{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
    bool   placed   = false;
    bool   has_time = false;
    double t0_ms = 0, fps_eff = 0;

    std::vector<offline::FrameRecord> frames;   // Pass-1 output (mesh stripped)
    std::vector<offline::Track>       tracks;   // Pass-2 global identities
};

// Run Pass 1 + Pass 2 on `cam.video`.  `base_cfg` supplies the model paths
// (onnx_dir/gguf/yolo/lbs), scene/tracking knobs and `max_frames`; the camera's
// intrinsics (when present) are injected into the pipeline.  Returns false if
// the pipeline or video can't be opened.
bool run_stream(const SyncCamera& cam, const offline::Config& base_cfg,
                StreamResult& out);

} // namespace mv

#endif // MULTIVIEW_FRONTEND_H_INCLUDED
