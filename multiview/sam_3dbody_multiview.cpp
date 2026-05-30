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
#include "associate.h"
#include "extrinsics.h"        // mat4_rotation_quat / mat4_transform_point
#include "sync_io.h"
#include "cli_common.h"        // default_lbs_path
#include "bvh_writer.h"

#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    std::string sync_path, bvh_out, bvh_template = "./body.bvh";
    offline::Config cfg;
    cfg.scene_detection = false;   // each stream is a single continuous shot
    double window_seconds = 3.0;   // length of the common window to process (0 = full overlap)
    double match_thresh_m = 0.6;

    for (int i = 1; i < argc; ++i)
    {
        if      (!strcmp(argv[i],"--sync")           && i+1<argc) sync_path       = argv[++i];
        else if (!strcmp(argv[i],"--onnx-dir")       && i+1<argc) cfg.onnx_dir    = argv[++i];
        else if (!strcmp(argv[i],"--gguf")           && i+1<argc) cfg.gguf_path   = argv[++i];
        else if (!strcmp(argv[i],"--yolo")           && i+1<argc) cfg.yolo_path   = argv[++i];
        else if (!strcmp(argv[i],"--lbs")            && i+1<argc) cfg.lbs_path    = argv[++i];
        else if (!strcmp(argv[i],"--cuda")           && i+1<argc) cfg.cuda_device = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--window-seconds") && i+1<argc) window_seconds  = atof(argv[++i]);
        else if (!strcmp(argv[i],"--match-thresh")   && i+1<argc) match_thresh_m  = atof(argv[++i]);
        else if (!strcmp(argv[i],"--bvh")            && i+1<argc) bvh_out         = argv[++i];
        else if (!strcmp(argv[i],"--bvh-template")   && i+1<argc) bvh_template    = argv[++i];
        else if (!strcmp(argv[i],"--help")||!strcmp(argv[i],"-h")) {
            printf("usage: %s --sync session.sync --onnx-dir DIR [--gguf F] [--yolo F] [--lbs F]\n"
                   "          [--cuda N] [--window-seconds S] [--match-thresh M]\n", argv[0]);
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
    const int S = (int)sess.cameras.size();
    printf("[mv] %s: %d cameras, master_fps=%.3f\n", sync_path.c_str(), S, sess.master_fps);

    // ── Common wall-clock overlap window across the timed cameras ────────────
    std::vector<double> nframes(S, 0);
    double t_lo = -1e300, t_hi = 1e300; int active = 0;
    for (int ci = 0; ci < S; ++ci)
    {
        const auto& cam = sess.cameras[ci];
        cv::VideoCapture cap(cam.video);
        nframes[ci] = cap.isOpened() ? cap.get(cv::CAP_PROP_FRAME_COUNT) : 0;
        if (!cam.has_time || cam.fps_eff <= 0 || nframes[ci] < 1) continue;
        double mspf = 1000.0 / cam.fps_eff;
        double a = cam.t0_ms, b = cam.t0_ms + (nframes[ci]-1)*mspf;
        t_lo = std::max(t_lo, a); t_hi = std::min(t_hi, b); ++active;
    }
    if (active == 0 || t_lo > t_hi) { fprintf(stderr,"[mv] no common time overlap across cameras\n"); return 1; }
    if (window_seconds > 0) t_hi = std::min(t_hi, t_lo + window_seconds*1000.0);
    printf("[mv] common window: unix_ms [%.0f .. %.0f]  (%.2f s)\n", t_lo, t_hi, (t_hi-t_lo)/1000.0);

    // ── Run each stream over its slice of the window ─────────────────────────
    std::vector<mv::StreamResult> streams(S);
    for (int ci = 0; ci < S; ++ci)
    {
        const auto& cam = sess.cameras[ci];
        offline::Config scfg = cfg;
        if (cam.has_time && cam.fps_eff > 0 && nframes[ci] >= 1)
        {
            int sf = (int)std::lround((t_lo - cam.t0_ms) * cam.fps_eff / 1000.0);
            int cnt = (int)std::lround((t_hi - t_lo) * cam.fps_eff / 1000.0) + 1;
            if (sf < 0) sf = 0;
            if (sf + cnt > (int)nframes[ci]) cnt = (int)nframes[ci] - sf;
            scfg.start_frame = sf; scfg.max_frames = cnt;
        }
        printf("\n[mv] === camera %d: %s  window frames [%d, +%d) ===\n",
               ci, cam.video.c_str(), scfg.start_frame, scfg.max_frames);
        if (!mv::run_stream(cam, scfg, streams[ci])) { fprintf(stderr,"[mv] camera %d failed/empty\n", ci); continue; }

        long dets = 0; for (const auto& fr : streams[ci].frames) dets += (long)fr.detections.size();
        printf("[mv] camera %d: %zu frames, %ld detections, %zu track(s); world origin=[% .3f % .3f % .3f]\n",
               ci, streams[ci].frames.size(), dets, streams[ci].tracks.size(),
               streams[ci].T_world_cam[3], streams[ci].T_world_cam[7], streams[ci].T_world_cam[11]);
    }

    // ── Time alignment + cross-view identity association ─────────────────────
    mv::Correspondences corr = mv::associate(streams, sess.master_fps, match_thresh_m);
    printf("\n[mv] association: %d global person(s) over %d master tick(s)\n", corr.n_persons, corr.nticks);
    for (int p = 0; p < corr.n_persons; ++p)
    {
        printf("  person %d: tracks", p);
        for (int s = 0; s < S; ++s) printf("  cam%d=%d", s, corr.person_track[p][s]);
        // count ticks seen + multi-view ticks
        int seen = 0, multi = 0;
        for (int k = 0; k < corr.nticks; ++k) {
            int v = (int)corr.ticks[k][p].views.size();
            if (v > 0) ++seen; if (v >= 2) ++multi;
        }
        printf("   | seen in %d ticks (%d multi-view)\n", seen, multi);
    }

    // ── AverageFuser → fused BVH per person ──────────────────────────────────
    if (!bvh_out.empty())
    {
        BVHWriter w;
        if (!w.open(bvh_template, bvh_out, 1.0f/(float)sess.master_fps, cfg.lbs_path))
        { fprintf(stderr,"[mv] BVHWriter open failed (template '%s')\n", bvh_template.c_str()); return 1; }

        std::set<int> seen;
        for (int k = 0; k < corr.nticks; ++k)
        {
            std::vector<BVHWriter::FusedPerson> fused;
            std::vector<int> present;
            for (int p = 0; p < corr.n_persons; ++p)
            {
                const auto& pt = corr.ticks[k][p];
                if (pt.views.empty()) continue;
                BVHWriter::FusedPerson fpn; fpn.track_id = p;
                for (const auto& vo : pt.views)
                {
                    const auto& sr = streams[vo.stream];
                    const fsb::MHRResult* m = &sr.frames[vo.frame_rel].detections[vo.det];
                    auto q  = mv::mat4_rotation_quat(sr.T_world_cam);
                    auto rw = mv::mat4_transform_point(sr.T_world_cam,
                                  { m->pred_cam_t[0], m->pred_cam_t[1], m->pred_cam_t[2] });
                    BVHWriter::FusedView fv;
                    fv.mhr        = m;
                    fv.q_world_cam = { (float)q[0],(float)q[1],(float)q[2],(float)q[3] };
                    fv.root_world  = { (float)rw[0],(float)rw[1],(float)rw[2] };
                    fpn.views.push_back(fv);
                }
                fused.push_back(std::move(fpn));
                present.push_back(p);
            }
            std::vector<int> pad;
            for (int id : seen)
                if (std::find(present.begin(), present.end(), id) == present.end()) pad.push_back(id);
            w.write_frame_fused(fused, pad);
            for (int id : present) seen.insert(id);
        }
        w.close();
        printf("[mv] wrote fused BVH base '%s' (%d persons, %d ticks @ %.2f fps)\n",
               bvh_out.c_str(), corr.n_persons, corr.nticks, sess.master_fps);
    }
    return 0;
}
