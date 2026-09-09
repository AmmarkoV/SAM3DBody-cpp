// ════════════════════════════════════════════════════════════════════════════
//  sam_3dbody_sync — produce a session.sync from several camera videos.
//
//  One sampled pass per video gathers BOTH the static-marker poses (for
//  extrinsics) and the QR timestamps (for the temporal fit).  It then solves
//  T_world<-cam (reference camera = world) and the per-stream (t0, fps_eff),
//  and writes them, with per-camera intrinsics, to a .sync file.
//
//  Usage:
//    sam_3dbody_sync --cam A.mp4 [--calib a.calib] --cam B.mp4 [...] \
//        --out session.sync [--aruco-dict NAME] [--marker-length M] \
//        [--dynamic-below 10 | --static-markers 11,14] \
//        [--step N] [--max-frames N] [--max-failures N] [--ref 0] [--master-fps F]
// ════════════════════════════════════════════════════════════════════════════

#include "aruco_qr.h"
#include "calib.h"
#include "extrinsics.h"
#include "sync_io.h"
#include "sync_time.h"

#include <opencv2/videoio.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <utility>
#include <vector>

struct CamSpec { std::string video, calib; };

int main(int argc, char** argv)
{
    std::vector<CamSpec> cin_;
    std::string dict = "DICT_6X6_250", out_path;
    double marker_len = 0.10, master_fps = 0.0;
    int    dynamic_below = 10, step = 5, max_frames = 0, max_failures = 30, ref = 0;
    std::set<int> static_set;

    for (int i = 1; i < argc; ++i)
    {
        if      (!strcmp(argv[i],"--cam")           && i+1<argc) cin_.push_back({argv[++i],""});
        else if (!strcmp(argv[i],"--calib")         && i+1<argc) { if(!cin_.empty()) cin_.back().calib=argv[++i]; else ++i; }
        else if (!strcmp(argv[i],"--out")           && i+1<argc) out_path = argv[++i];
        else if (!strcmp(argv[i],"--aruco-dict")    && i+1<argc) dict = argv[++i];
        else if (!strcmp(argv[i],"--marker-length") && i+1<argc) marker_len = atof(argv[++i]);
        else if (!strcmp(argv[i],"--dynamic-below") && i+1<argc) dynamic_below = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--static-markers")&& i+1<argc) { std::string s=argv[++i];
            for (size_t p=0;p<s.size();){ size_t c=s.find(',',p);
                static_set.insert(atoi(s.substr(p,c==std::string::npos?std::string::npos:c-p).c_str()));
                if(c==std::string::npos)break; p=c+1; } }
        else if (!strcmp(argv[i],"--step")          && i+1<argc) step = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--max-frames")    && i+1<argc) max_frames = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--max-failures")  && i+1<argc) max_failures = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--ref")           && i+1<argc) ref = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--master-fps")    && i+1<argc) master_fps = atof(argv[++i]);
        else if (!strcmp(argv[i],"--help")||!strcmp(argv[i],"-h")) {
            printf("usage: %s --cam VIDEO [--calib F] [--cam ...] --out session.sync\n"
                   "          [--aruco-dict NAME] [--marker-length M]\n"
                   "          [--dynamic-below N | --static-markers a,b,c]\n"
                   "          [--step N] [--max-frames N] [--max-failures N] [--ref N] [--master-fps F]\n", argv[0]);
            return 0; }
        else { fprintf(stderr,"unknown argument: %s\n", argv[i]); return 2; }
    }
    if (cin_.size() < 2) { fprintf(stderr,"need at least two --cam VIDEO inputs\n"); return 2; }
    if (out_path.empty()) { fprintf(stderr,"--out session.sync is required\n"); return 2; }
    if (step < 1) step = 1;

    auto is_static = [&](int id){ return static_set.empty() ? id >= dynamic_below : (bool)static_set.count(id); };

    const int N = (int)cin_.size();
    std::vector<mv::CameraStaticPoses>             poses(N);
    std::vector<mv::TimeFit>                       tfit(N);
    std::vector<int>                               vw(N), vh(N);
    std::vector<double>                            vnf(N), vcfps(N);
    std::vector<bool>                              has_calib(N,false);
    std::vector<struct calibration>                calib(N);

    for (int ci = 0; ci < N; ++ci)
    {
        cv::VideoCapture cap(cin_[ci].video);
        if (!cap.isOpened()) { fprintf(stderr,"cam %d: could not open '%s'\n", ci, cin_[ci].video.c_str()); return 1; }
        vw[ci]=(int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
        vh[ci]=(int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        vnf[ci]=cap.get(cv::CAP_PROP_FRAME_COUNT);
        vcfps[ci]=cap.get(cv::CAP_PROP_FPS);

        mv::ArucoQrDetector det(dict, marker_len);
        if (!cin_[ci].calib.empty()) {
            if (ReadCalibration(cin_[ci].calib.c_str(), 0, 0, &calib[ci])) { det.set_calibration(calib[ci]); has_calib[ci]=true; }
            else fprintf(stderr,"cam %d: could not read calib '%s' (approx)\n", ci, cin_[ci].calib.c_str());
        }

        int consec=0;
        auto grab_retry=[&]{ while(true){ if(cap.grab()){consec=0;return true;} if(++consec>max_failures)return false; } };
        auto read_retry=[&](cv::Mat&f){ while(true){ if(cap.read(f)&&!f.empty()){consec=0;return true;} if(++consec>max_failures)return false; } };

        mv::StaticPoseAccumulator acc;
        std::vector<std::pair<int,double>> samples;
        mv::FrameDetections d; cv::Mat frame; int processed=0;
        while (true)
        {
            bool eof=false;
            for (int s=0;s<step-1;++s) if(!grab_retry()){eof=true;break;}
            if (eof || !read_retry(frame)) break;
            int true_idx=(int)cap.get(cv::CAP_PROP_POS_FRAMES)-1;
            det.process(frame, d);
            for (const auto& m : d.markers) if (is_static(m.id)) acc.add(m.id, m.rvec.data(), m.tvec.data());
            for (const auto& q : d.qrs) if (!q.t_unix_ms.empty()) samples.push_back({true_idx, atof(q.t_unix_ms.c_str())});
            ++processed;
            if (max_frames>0 && processed>=max_frames) break;
        }

        poses[ci].pose = acc.poses();
        poses[ci].count = acc.counts();
        tfit[ci] = mv::fit_time(samples);

        printf("[cam %d] %s  %dx%d  calib=%s  scanned %d; static:", ci, cin_[ci].video.c_str(),
               vw[ci], vh[ci], has_calib[ci]?"yes":"approx", processed);
        for (auto& kv : poses[ci].pose) printf(" id%d(x%d)", kv.first, poses[ci].count[kv.first]);
        printf("  | time: %s fps_eff=%.3f (n=%d)\n",
               tfit[ci].ok?"ok":"FAIL", tfit[ci].fps_eff, tfit[ci].samples);
    }

    std::string report;
    auto ext = mv::solve_extrinsics(poses, ref, &report);
    printf("\n%s\n", report.c_str());

    // ── Assemble + write the session ─────────────────────────────────────────
    mv::SyncSession sess;
    for (int ci = 0; ci < N; ++ci)
    {
        mv::SyncCamera sc;
        sc.video = cin_[ci].video;
        sc.width = vw[ci]; sc.height = vh[ci];
        if (has_calib[ci])
        {
            double fx=calib[ci].intrinsic[CALIB_INTR_FX], fy=calib[ci].intrinsic[CALIB_INTR_FY];
            double cx=calib[ci].intrinsic[CALIB_INTR_CX], cy=calib[ci].intrinsic[CALIB_INTR_CY];
            if (calib[ci].width>0 && calib[ci].height>0 &&
                ((int)calib[ci].width!=vw[ci] || (int)calib[ci].height!=vh[ci]))
            {   // express K at the video resolution
                double sx=(double)vw[ci]/calib[ci].width, sy=(double)vh[ci]/calib[ci].height;
                fx*=sx; cx*=sx; fy*=sy; cy*=sy;
            }
            sc.has_calib=true; sc.fx=fx; sc.fy=fy; sc.cx=cx; sc.cy=cy;
            sc.dist[0]=calib[ci].k1; sc.dist[1]=calib[ci].k2; sc.dist[2]=calib[ci].p1;
            sc.dist[3]=calib[ci].p2; sc.dist[4]=calib[ci].k3;
        }
        sc.placed = ext[ci].placed;
        sc.T_world_cam = ext[ci].T_world_cam;
        sc.has_time = tfit[ci].ok;
        sc.t0_ms = tfit[ci].t0_ms; sc.fps_eff = tfit[ci].fps_eff; sc.resid_med_ms = tfit[ci].residual_med_ms;
        sess.cameras.push_back(sc);
    }
    sess.master_fps = master_fps > 0 ? master_fps
                    : (tfit[ref].ok ? tfit[ref].fps_eff : vcfps[ref]);

    if (!mv::write_sync(out_path, sess)) { fprintf(stderr,"failed to write '%s'\n", out_path.c_str()); return 1; }
    printf("wrote %s  (master_fps=%.3f, %d cameras)\n", out_path.c_str(), sess.master_fps, N);

    // Common time overlap across cameras that have a temporal fit.
    bool any=false; double tmax=-1e30, tmin=1e30;
    for (int ci=0; ci<N; ++ci) if (tfit[ci].ok) {
        double t0=tfit[ci].t0_ms, t1=tfit[ci].frame_to_unix_ms(vnf[ci]);
        if (t0>tmax) tmax=t0; if (t1<tmin) tmin=t1; any=true;
    }
    if (any) printf("common overlap: unix_ms [%.0f .. %.0f]  (%.2f s)\n", tmax, tmin, (tmin-tmax)/1000.0);
    return 0;
}
