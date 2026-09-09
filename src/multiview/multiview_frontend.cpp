// ════════════════════════════════════════════════════════════════════════════
//  multiview_frontend.cpp — see multiview_frontend.h.
// ════════════════════════════════════════════════════════════════════════════

#include "multiview_frontend.h"

#include "cli_common.h"        // apply_common_to_pipeline_cfg
#include "fast_sam_3dbody.h"   // fsb::Pipeline / PipelineConfig

#include <opencv2/videoio.hpp>

#include <cstdio>

namespace mv
{

bool run_stream(const SyncCamera& cam, const offline::Config& base_cfg, StreamResult& out)
{
    // Build the pipeline config from the shared paths, then inject this
    // camera's calibrated intrinsics (0 = pipeline default = image diagonal).
    fsb::PipelineConfig pcfg;
    apply_common_to_pipeline_cfg(base_cfg, pcfg);
    pcfg.skip_body_model = false;     // need keypoints_3d for fusion
    if (cam.has_calib)
    {
        pcfg.focal_x     = (float)cam.fx;
        pcfg.focal_y     = (float)cam.fy;
        pcfg.principal_x = (float)cam.cx;
        pcfg.principal_y = (float)cam.cy;
    }

    fsb::Pipeline pipeline;
    if (!pipeline.load(pcfg)) { fprintf(stderr,"[mv] pipeline load failed for '%s'\n", cam.video.c_str()); return false; }

    cv::VideoCapture cap(cam.video);
    if (!cap.isOpened()) { fprintf(stderr,"[mv] could not open '%s'\n", cam.video.c_str()); return false; }

    offline::Config cfg = base_cfg;
    cfg.from = cam.video;

    std::vector<int> scene_cuts;
    if (!offline::run_inference_pass(pipeline, cap, out.frames, scene_cuts, out.fps, cfg))
        return false;
    out.tracks = offline::build_global_tracks(out.frames, cfg);

    out.start_frame = base_cfg.start_frame;
    out.width       = cam.width  > 0 ? cam.width  : (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    out.height      = cam.height > 0 ? cam.height : (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    out.T_world_cam = cam.T_world_cam;
    out.placed      = cam.placed;
    out.has_time    = cam.has_time;
    out.t0_ms       = cam.t0_ms;
    out.fps_eff     = cam.fps_eff;
    return true;
}

} // namespace mv
