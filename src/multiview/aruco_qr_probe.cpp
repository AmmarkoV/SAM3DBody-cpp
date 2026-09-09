// ════════════════════════════════════════════════════════════════════════════
//  aruco_qr_probe — drive ArucoQrDetector over a video and print what it finds.
//
//  A validation tool for the multi-view sync front-end: run it on real footage
//  to confirm ArUco poses and QR timecodes come out sane before building the
//  extrinsics / temporal solvers on top.
//
//  Usage:
//    aruco_qr_probe --from VIDEO [--calib FILE.calib] [--aruco-dict DICT_6X6_250]
//                   [--marker-length M] [--marker-lengths "1=0.06,3=0.12"]
//                   [--max-frames N]
// ════════════════════════════════════════════════════════════════════════════

#include "aruco_qr.h"
#include "calib.h"

#include <opencv2/calib3d.hpp>     // Rodrigues / RQDecomp3x3 for readable RPY
#include <opencv2/videoio.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void rvec_to_rpy_deg(const std::array<double,3>& rvec, double rpy[3])
{
    cv::Mat rv = (cv::Mat_<double>(3,1) << rvec[0], rvec[1], rvec[2]);
    cv::Mat R, mR, mQ;
    cv::Rodrigues(rv, R);
    cv::Vec3d e = cv::RQDecomp3x3(R, mR, mQ);   // degrees
    rpy[0] = e[0]; rpy[1] = e[1]; rpy[2] = e[2];
}

int main(int argc, char** argv)
{
    std::string from, calib_path, dict = "DICT_6X6_250", marker_lengths;
    double marker_len = 0.05;
    int    max_frames = 0;        // 0 = whole video
    int    step       = 1;        // process every Nth frame (sampling/reconnaissance)
    int    max_failures = 30;     // consecutive bad-frame reads tolerated (SD glitches)

    for (int i = 1; i < argc; ++i)
    {
        if      (!strcmp(argv[i], "--from")            && i+1 < argc) from           = argv[++i];
        else if (!strcmp(argv[i], "--calib")           && i+1 < argc) calib_path     = argv[++i];
        else if (!strcmp(argv[i], "--aruco-dict")      && i+1 < argc) dict           = argv[++i];
        else if (!strcmp(argv[i], "--marker-length")   && i+1 < argc) marker_len     = atof(argv[++i]);
        else if (!strcmp(argv[i], "--marker-lengths")  && i+1 < argc) marker_lengths = argv[++i];
        else if (!strcmp(argv[i], "--max-frames")      && i+1 < argc) max_frames     = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--step")            && i+1 < argc) step           = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-failures")    && i+1 < argc) max_failures   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
        {
            printf("usage: %s --from VIDEO [--calib F.calib] [--aruco-dict NAME]\n"
                   "          [--marker-length M] [--marker-lengths \"id=len,...\"] [--max-frames N]\n",
                   argv[0]);
            return 0;
        }
        else { fprintf(stderr, "unknown argument: %s\n", argv[i]); return 2; }
    }

    if (from.empty()) { fprintf(stderr, "--from VIDEO is required\n"); return 2; }
    if (mv::aruco_dict_id(dict) < 0)
        fprintf(stderr, "[warn] unknown --aruco-dict '%s'; falling back to DICT_6X6_250\n", dict.c_str());

    mv::ArucoQrDetector det(dict, marker_len);

    if (!calib_path.empty())
    {
        struct calibration c;
        if (ReadCalibration(calib_path.c_str(), 0, 0, &c)) { det.set_calibration(c); PrintCalibration(&c); }
        else fprintf(stderr, "[warn] could not read --calib '%s'; using approximate intrinsics\n", calib_path.c_str());
    }

    // Per-id marker length overrides: "1=0.06,3=0.12"
    for (size_t p = 0; p < marker_lengths.size(); )
    {
        size_t comma = marker_lengths.find(',', p);
        std::string tok = marker_lengths.substr(p, comma == std::string::npos ? std::string::npos : comma - p);
        size_t eq = tok.find('=');
        if (eq != std::string::npos)
            det.set_marker_length(atoi(tok.substr(0,eq).c_str()), atof(tok.substr(eq+1).c_str()));
        if (comma == std::string::npos) break;
        p = comma + 1;
    }

    cv::VideoCapture cap(from);
    if (!cap.isOpened()) { fprintf(stderr, "could not open video '%s'\n", from.c_str()); return 1; }

    printf("[probe] %s  dict=%s  marker_len=%.3f m  calib=%s\n",
           from.c_str(), dict.c_str(), marker_len,
           det.calibrated() ? calib_path.c_str() : "approx");

    if (step < 1) step = 1;

    // SD cards occasionally fail to write a frame, so a decode can fail mid-clip
    // even though there is good data after it.  Retry past such gaps (like
    // tracker.py's max_consecutive_failures) instead of stopping at the first
    // failure; only give up after `max_failures` consecutive bad frames.
    int consec_fail = 0, total_bad = 0;
    auto grab_retry = [&]() -> bool {
        while (true) {
            if (cap.grab()) { consec_fail = 0; return true; }
            if (++consec_fail > max_failures) return false;
            ++total_bad;
        }
    };
    auto read_retry = [&](cv::Mat& f) -> bool {
        while (true) {
            if (cap.read(f) && !f.empty()) { consec_fail = 0; return true; }
            if (++consec_fail > max_failures) return false;
            ++total_bad;
        }
    };

    cv::Mat frame;
    int fidx = 0, processed = 0, frames_with_marker = 0, frames_with_qr = 0, total_markers = 0;
    mv::FrameDetections d;
    while (true)
    {
        // Sampled scan: skip step-1 frames with grab() (reliable across codecs,
        // unlike POS_FRAMES seeking which mis-behaves on some containers).
        bool eof = false;
        for (int s = 0; s < step - 1; ++s) if (!grab_retry()) { eof = true; break; }
        if (eof || !read_retry(frame)) break;

        det.process(frame, d);
        ++processed;
        if (!d.markers.empty()) ++frames_with_marker;
        if (!d.qrs.empty())     ++frames_with_qr;
        total_markers += (int)d.markers.size();

        if (!d.markers.empty() || !d.qrs.empty())
        {
            printf("frame %d:\n", fidx);
            for (const auto& m : d.markers)
            {
                double rpy[3]; rvec_to_rpy_deg(m.rvec, rpy);
                printf("  aruco id=%-3d  T(m)=[% .3f % .3f % .3f]  RPY(deg)=[% .1f % .1f % .1f]\n",
                       m.id, m.tvec[0], m.tvec[1], m.tvec[2], rpy[0], rpy[1], rpy[2]);
            }
            for (const auto& q : d.qrs)
                printf("  qr    t_unix_ms=%s frame=%s hz=%s  raw=\"%s\"\n",
                       q.t_unix_ms.empty()?"-":q.t_unix_ms.c_str(),
                       q.frame.empty()?"-":q.frame.c_str(),
                       q.hz.empty()?"-":q.hz.c_str(), q.raw.c_str());
        }

        fidx += step;
        if (max_frames > 0 && processed >= max_frames) break;
    }

    printf("\n[probe] processed %d frames (step %d, %d bad frames skipped): "
           "%d with marker(s) (%d marker detections total), %d with QR\n",
           processed, step, total_bad, frames_with_marker, total_markers, frames_with_qr);
    return 0;
}
