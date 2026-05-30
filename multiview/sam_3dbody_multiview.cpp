// ════════════════════════════════════════════════════════════════════════════
//  sam_3dbody_multiview — multi-view fusion driver (front-end stage).
//
//  Reads a session.sync, runs the offline per-stream pipeline on each camera
//  (calibrated intrinsics injected), and reports what each stream produced.
//  Cross-view association + fusion (AverageFuser) build on top of this — see
//  MULTIVIEW_PLAN.md.
//
//  Usage:
//    sam_3dbody_multiview --sync session.sync --onnx-dir DIR \
//        [--gguf F.gguf] [--yolo F.onnx] [--lbs F.lbs] [--cuda N] [--max-frames N]
// ════════════════════════════════════════════════════════════════════════════

#include "multiview_frontend.h"
#include "sync_io.h"
#include "cli_common.h"        // default_lbs_path

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char** argv)
{
    std::string sync_path;
    offline::Config cfg;
    cfg.scene_detection = false;   // each stream is a single continuous shot

    for (int i = 1; i < argc; ++i)
    {
        if      (!strcmp(argv[i],"--sync")       && i+1<argc) sync_path        = argv[++i];
        else if (!strcmp(argv[i],"--onnx-dir")   && i+1<argc) cfg.onnx_dir     = argv[++i];
        else if (!strcmp(argv[i],"--gguf")       && i+1<argc) cfg.gguf_path    = argv[++i];
        else if (!strcmp(argv[i],"--yolo")       && i+1<argc) cfg.yolo_path    = argv[++i];
        else if (!strcmp(argv[i],"--lbs")        && i+1<argc) cfg.lbs_path     = argv[++i];
        else if (!strcmp(argv[i],"--cuda")       && i+1<argc) cfg.cuda_device  = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--max-frames") && i+1<argc) cfg.max_frames   = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--help")||!strcmp(argv[i],"-h")) {
            printf("usage: %s --sync session.sync --onnx-dir DIR [--gguf F] [--yolo F] [--lbs F] [--cuda N] [--max-frames N]\n", argv[0]);
            return 0; }
        else { fprintf(stderr,"unknown argument: %s\n", argv[i]); return 2; }
    }
    if (sync_path.empty() || cfg.onnx_dir.empty())
    { fprintf(stderr,"--sync and --onnx-dir are required\n"); return 2; }

    if (cfg.gguf_path.empty()) cfg.gguf_path = cfg.onnx_dir + "/pipeline.gguf";
    if (cfg.yolo_path.empty()) cfg.yolo_path = cfg.onnx_dir + "/yolo.onnx";
    if (cfg.lbs_path.empty())  cfg.lbs_path  = default_lbs_path(cfg);

    mv::SyncSession sess;
    if (!mv::read_sync(sync_path, sess)) { fprintf(stderr,"could not read '%s'\n", sync_path.c_str()); return 1; }
    printf("[mv] %s: %zu cameras, master_fps=%.3f\n", sync_path.c_str(), sess.cameras.size(), sess.master_fps);

    for (size_t ci = 0; ci < sess.cameras.size(); ++ci)
    {
        const mv::SyncCamera& cam = sess.cameras[ci];
        printf("\n[mv] === camera %zu: %s (%dx%d, calib=%s, placed=%d, time=%d) ===\n",
               ci, cam.video.c_str(), cam.width, cam.height,
               cam.has_calib?"yes":"approx", cam.placed, cam.has_time);

        mv::StreamResult sr;
        if (!mv::run_stream(cam, cfg, sr)) { fprintf(stderr,"[mv] camera %zu failed\n", ci); continue; }

        long dets = 0;
        for (const auto& fr : sr.frames) dets += (long)fr.detections.size();
        printf("[mv] camera %zu: %zu frames, %ld detections, %zu track(s); world origin=[% .3f % .3f % .3f]\n",
               ci, sr.frames.size(), dets, sr.tracks.size(),
               sr.T_world_cam[3], sr.T_world_cam[7], sr.T_world_cam[11]);
        for (const auto& t : sr.tracks)
            printf("        track %d  frames [%d,%d]  (%zu detections)\n",
                   t.id, t.first_frame, t.last_frame, t.frame_to_det.size());
    }
    return 0;
}
