// ════════════════════════════════════════════════════════════════════════════
//  sam_3dbody_extrinsics — recover camera extrinsics from shared static markers.
//
//  Runs the ArUco detector over each camera's video, accumulates the poses of
//  the STATIC markers (id >= --dynamic-below, default 10 — the moving object
//  markers are excluded), and solves T_world<-cam for every camera with the
//  first camera as the world origin (or --ref N).
//
//  Usage:
//    sam_3dbody_extrinsics --cam A.mp4 [--calib a.calib] --cam B.mp4 [...] \
//        [--aruco-dict DICT_6X6_250] [--marker-length M] \
//        [--dynamic-below 10 | --static-markers 10,11,14] \
//        [--step N] [--max-frames N] [--max-failures N] [--ref 0]
// ════════════════════════════════════════════════════════════════════════════

#include "aruco_qr.h"
#include "calib.h"
#include "extrinsics.h"

#include <opencv2/videoio.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

struct CamSpec { std::string video; std::string calib; };

static void euler_zyx_deg(const mv::Mat4& T, double& yaw, double& pitch, double& roll)
{
    // R = Rz(yaw)Ry(pitch)Rx(roll); extract from the row-major rotation block.
    double r00=T[0], r10=T[4], r20=T[8], r21=T[9], r22=T[10];
    pitch = std::asin(-std::max(-1.0,std::min(1.0,r20))) * 180.0/M_PI;
    yaw   = std::atan2(r10, r00) * 180.0/M_PI;
    roll  = std::atan2(r21, r22) * 180.0/M_PI;
}

int main(int argc, char** argv)
{
    std::vector<CamSpec> cams_in;
    std::string dict = "DICT_6X6_250";
    double marker_len = 0.10;
    int    dynamic_below = 10;
    std::set<int> static_set;            // explicit whitelist (overrides dynamic_below)
    int    step = 5, max_frames = 0, max_failures = 30, ref = 0;

    for (int i=1;i<argc;++i)
    {
        if      (!strcmp(argv[i],"--cam")           && i+1<argc) cams_in.push_back({argv[++i],""});
        else if (!strcmp(argv[i],"--calib")         && i+1<argc) { if(!cams_in.empty()) cams_in.back().calib=argv[++i]; else ++i; }
        else if (!strcmp(argv[i],"--aruco-dict")    && i+1<argc) dict = argv[++i];
        else if (!strcmp(argv[i],"--marker-length") && i+1<argc) marker_len = atof(argv[++i]);
        else if (!strcmp(argv[i],"--dynamic-below") && i+1<argc) dynamic_below = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--static-markers")&& i+1<argc) {
            std::string s=argv[++i]; for (size_t p=0;p<s.size();){ size_t c=s.find(',',p);
                static_set.insert(atoi(s.substr(p, c==std::string::npos?std::string::npos:c-p).c_str()));
                if(c==std::string::npos) break; p=c+1; } }
        else if (!strcmp(argv[i],"--step")          && i+1<argc) step = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--max-frames")    && i+1<argc) max_frames = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--max-failures")  && i+1<argc) max_failures = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--ref")           && i+1<argc) ref = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--help")||!strcmp(argv[i],"-h")) {
            printf("usage: %s --cam VIDEO [--calib F.calib] [--cam ...] [--aruco-dict NAME]\n"
                   "          [--marker-length M] [--dynamic-below N | --static-markers a,b,c]\n"
                   "          [--step N] [--max-frames N] [--max-failures N] [--ref N]\n", argv[0]);
            return 0; }
        else { fprintf(stderr,"unknown argument: %s\n", argv[i]); return 2; }
    }
    if (cams_in.size() < 2) { fprintf(stderr,"need at least two --cam VIDEO inputs\n"); return 2; }
    if (step < 1) step = 1;

    auto is_static = [&](int id){ return static_set.empty() ? id >= dynamic_below
                                                            : (bool)static_set.count(id); };

    std::vector<mv::CameraStaticPoses> cams(cams_in.size());

    for (size_t ci=0; ci<cams_in.size(); ++ci)
    {
        const auto& cs = cams_in[ci];
        cv::VideoCapture cap(cs.video);
        if (!cap.isOpened()) { fprintf(stderr,"cam %zu: could not open '%s'\n", ci, cs.video.c_str()); return 1; }

        mv::ArucoQrDetector det(dict, marker_len);
        if (!cs.calib.empty()) {
            struct calibration c;
            if (ReadCalibration(cs.calib.c_str(), 0, 0, &c)) det.set_calibration(c);
            else fprintf(stderr,"cam %zu: could not read calib '%s' (using approx)\n", ci, cs.calib.c_str());
        }

        // Retry past SD-card bad frames (single/few-frame glitches); sampled scan.
        int consec=0;
        auto grab_retry=[&]{ while(true){ if(cap.grab()){consec=0;return true;} if(++consec>max_failures)return false; } };
        auto read_retry=[&](cv::Mat&f){ while(true){ if(cap.read(f)&&!f.empty()){consec=0;return true;} if(++consec>max_failures)return false; } };

        mv::StaticPoseAccumulator acc;
        mv::FrameDetections d;
        cv::Mat frame;
        int processed=0;
        while (true)
        {
            bool eof=false;
            for (int s=0;s<step-1;++s) if(!grab_retry()){eof=true;break;}
            if (eof || !read_retry(frame)) break;
            det.process(frame, d);
            for (const auto& m : d.markers)
                if (is_static(m.id)) acc.add(m.id, m.rvec.data(), m.tvec.data());
            ++processed;
            if (max_frames>0 && processed>=max_frames) break;
        }

        cams[ci].pose  = acc.poses();
        cams[ci].count = acc.counts();

        printf("[cam %zu] %s  calib=%s  scanned %d frames; static markers:",
               ci, cs.video.c_str(), cs.calib.empty()?"approx":cs.calib.c_str(), processed);
        if (cams[ci].pose.empty()) printf(" (none)");
        for (auto& kv : cams[ci].pose) printf("  id%d(x%d)", kv.first, cams[ci].count[kv.first]);
        printf("\n");
    }

    std::string report;
    auto ext = mv::solve_extrinsics(cams, ref, &report);
    printf("\n%s\n", report.c_str());

    for (size_t i=0;i<ext.size();++i)
    {
        if (!ext[i].placed) { printf("cam %zu: NOT PLACED\n", i); continue; }
        const mv::Mat4& T=ext[i].T_world_cam;
        double yaw,pitch,roll; euler_zyx_deg(T,yaw,pitch,roll);
        printf("cam %zu: T_world_cam  pos(m)=[% .3f % .3f % .3f]  ypr(deg)=[% .1f % .1f % .1f]\n",
               i, T[3],T[7],T[11], yaw,pitch,roll);
    }
    return 0;
}
