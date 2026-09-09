// ════════════════════════════════════════════════════════════════════════════
//  sam_3dbody_synctime — fit a stream's QR timecodes to (t0, fps_eff).
//
//  Scans a video, decodes the QR clock per (sampled) frame, and fits the
//  frame↔unix-ms line.  The fitted fps_eff should match the camera's real
//  frame rate — a direct check that the temporal model is right.
//
//  Usage:
//    sam_3dbody_synctime --from VIDEO [--aruco-dict NAME] [--step N]
//                        [--max-frames N] [--max-failures N]
// ════════════════════════════════════════════════════════════════════════════

#include "aruco_qr.h"
#include "sync_time.h"

#include <opencv2/videoio.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

int main(int argc, char** argv)
{
    std::string from, dict = "DICT_6X6_250";
    int step = 5, max_frames = 0, max_failures = 30;

    for (int i = 1; i < argc; ++i)
    {
        if      (!strcmp(argv[i],"--from")         && i+1<argc) from         = argv[++i];
        else if (!strcmp(argv[i],"--aruco-dict")   && i+1<argc) dict         = argv[++i];
        else if (!strcmp(argv[i],"--step")         && i+1<argc) step         = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--max-frames")   && i+1<argc) max_frames   = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--max-failures") && i+1<argc) max_failures = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--help")||!strcmp(argv[i],"-h"))
        { printf("usage: %s --from VIDEO [--aruco-dict NAME] [--step N] [--max-frames N] [--max-failures N]\n", argv[0]); return 0; }
        else { fprintf(stderr,"unknown argument: %s\n", argv[i]); return 2; }
    }
    if (from.empty()) { fprintf(stderr,"--from VIDEO is required\n"); return 2; }
    if (step < 1) step = 1;

    cv::VideoCapture cap(from);
    if (!cap.isOpened()) { fprintf(stderr,"could not open '%s'\n", from.c_str()); return 1; }
    double container_fps = cap.get(cv::CAP_PROP_FPS);
    double nframes        = cap.get(cv::CAP_PROP_FRAME_COUNT);

    mv::ArucoQrDetector det(dict, 0.10);     // marker length irrelevant for QR

    int consec = 0;
    auto grab_retry = [&]{ while(true){ if(cap.grab()){consec=0;return true;} if(++consec>max_failures)return false; } };
    auto read_retry = [&](cv::Mat& f){ while(true){ if(cap.read(f)&&!f.empty()){consec=0;return true;} if(++consec>max_failures)return false; } };

    std::vector<std::pair<int,double>> samples;
    mv::FrameDetections d;
    cv::Mat frame;
    int processed = 0;
    while (true)
    {
        bool eof = false;
        for (int s = 0; s < step - 1; ++s) if (!grab_retry()) { eof = true; break; }
        if (eof || !read_retry(frame)) break;

        int true_idx = (int)cap.get(cv::CAP_PROP_POS_FRAMES) - 1;   // index of the frame just read
        det.process(frame, d);
        for (const auto& q : d.qrs)
            if (!q.t_unix_ms.empty())
                samples.push_back({ true_idx, atof(q.t_unix_ms.c_str()) });

        ++processed;
        if (max_frames > 0 && processed >= max_frames) break;
    }

    printf("[synctime] %s\n", from.c_str());
    printf("  scanned %d frames (step %d), %d QR timestamp samples; container fps=%.3f, frames=%.0f\n",
           processed, step, (int)samples.size(), container_fps, nframes);

    mv::TimeFit fit = mv::fit_time(samples);
    if (!fit.ok) { fprintf(stderr,"  fit FAILED (too few QR samples)\n"); return 1; }

    printf("  fit: t0_ms=%.0f  ms/frame=%.4f  fps_eff=%.4f  (container %.4f)\n",
           fit.t0_ms, fit.ms_per_frame, fit.fps_eff, container_fps);
    printf("  residuals: median=%.1f ms  max=%.1f ms\n", fit.residual_med_ms, fit.residual_max_ms);
    printf("  coverage: unix_ms [%.0f .. %.0f]  (%.2f s)\n",
           fit.t0_ms, fit.frame_to_unix_ms(nframes > 0 ? nframes : 0),
           nframes > 0 ? (fit.ms_per_frame * nframes) / 1000.0 : 0.0);
    return 0;
}
