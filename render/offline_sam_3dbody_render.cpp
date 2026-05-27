// ════════════════════════════════════════════════════════════════════════════
//  offline_sam_3dbody_render.cpp
//
//  Offline multi-pass MHR → BVH extractor for the SAM-3D-Body pipeline.
//
//  WHEN TO USE THIS BINARY
//  ───────────────────────
//   - You have a video file on disk (mp4 / mkv / avi / …).
//   - You can afford a few seconds of "pre-process the whole clip" latency
//     in exchange for noticeably smoother BVH output and more stable
//     identities across occlusions.
//
//   For webcams, RTSP streams, or single still images use the live binaries
//   (fast_sam_3dbody_run / fast_sam_3dbody_render).  This binary refuses to
//   open those sources.
//
//  HOW IT IS DIFFERENT FROM THE LIVE BINARIES
//  ──────────────────────────────────────────
//   Live binaries are causal: they process frame N before frame N+1 is
//   available, so every filter and tracker can only look BACKWARDS.  That
//   forces three compromises:
//
//     * the tracker is greedy IoU forward-only — when two people cross,
//       identities can swap;
//     * the smoothing filter has to be one-sided, which adds a temporal lag
//       proportional to its cutoff frequency;
//     * a single bad FFN frame becomes a visible flick because we don't
//       know whether the next frame is "back to normal" until it arrives.
//
//   Offline removes all three constraints.  This binary runs five distinct
//   passes over the whole video; each pass uses information from the
//   future as well as the past.  See the per-pass headers below.
//
//  SHARED CODE
//  ───────────
//   Almost all of the heavy lifting lives in code we already have:
//     - fsb::Pipeline                — the YOLO + backbone + decoder + MHR
//                                       inference engine (src/fast_sam_3dbody.{h,cpp})
//     - BVHWriter                    — multi-person BVH export with bone-length
//                                       rewrite (src/bvh_writer.{h,cpp})
//     - ButterWorth, QuatLPF, helpers — scalar Butterworth + quaternion
//                                       SLERP-EMA filters (src/outputFiltering.h)
//     - fsb::apply_hand_pose         — PCA decode of hand pose into the LBS
//                                       model_params vector (src/preprocess.hpp)
//
//   The new code in *this* file is just:
//     - the orchestration (read video → call passes → write BVH)
//     - a globally-optimal-ish tracker (greedy by combined cost + post-merge)
//     - jitter detection and interpolation across the per-track timeseries
//     - the forward+backward "zero-phase" wrapper around the per-frame filters
//
//  USAGE
//  ─────
//     offline_sam_3dbody_render \
//         --onnx-dir ./onnx --gguf ./onnx/pipeline.gguf \
//         --yolo ./onnx/yolo.onnx --from clip.mp4 \
//         --bvh ./out.bvh \
//         [--smoothing zero-phase|forward|off] \
//         [--bw-cutoff HZ] [--rot-clamp DEG] \
//         [--interpolate-jitter] [--jitter-threshold-cm CM] \
//         [--track-merge-frames N] [--track-merge-cm CM] \
//         [--no-bvh-body-shape-change] [--no-bvh-hand-shape-change] \
//         [--bvh-raw-fingers]
//
// ════════════════════════════════════════════════════════════════════════════

#include "../src/fast_sam_3dbody.h"
#include "../src/bvh_writer.h"
#include "../src/outputFiltering.h"
#include "../src/preprocess.hpp"   // for fsb::apply_hand_pose
#include "../src/cli_common.h"     // shared --onnx-dir / --bvh / … parser

extern "C" {
#include "../GraphicsEngine/ModelLoader/model_loader_transform_joints.h"
}

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

// ─── Constants & small helpers ───────────────────────────────────────────────

constexpr float PI_F = 3.14159265359f;

// Per-frame snapshot of pipeline output.
//
// pred_vertices is deliberately stripped before this struct is populated —
// at 18439×3 floats per detection it's the bulk of MHRResult's memory cost
// and we never need the mesh for BVH export.  This keeps memory bounded
// even on long clips (~1 hour at 30 fps with 3 people ≈ 600 MB).
struct FrameRecord
{
    int                          frame_idx = -1;
    std::vector<fsb::MHRResult>  detections;          // pipeline output
    std::vector<int>             track_ids;           // filled by Pass 2
    std::vector<char>            was_interpolated;    // filled by Pass 3 (parallel to detections)
};

// Identity span across the session: a contiguous block of frames where the
// same person is tracked, with the mapping back into each FrameRecord.
struct Track
{
    int                                 id            = -1;
    int                                 first_frame   = INT32_MAX;
    int                                 last_frame    = -1;
    // For each session frame in [first_frame, last_frame] that the track is
    // present in: index into FrameRecord::detections.  -1 means the track was
    // alive but no detection covered it this frame (it'll get padded at write
    // time, or interpolated in Pass 3 if requested).
    std::map<int, int>                  frame_to_det;
};

// Wallclock chrono helper.
using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0)
{
    auto now = Clock::now();
    return std::chrono::duration<double, std::milli>(now - t0).count();
}

// ─── CLI ─────────────────────────────────────────────────────────────────────

// Config inherits the common subset from CommonConfig (cli_common.h):
//   --onnx-dir --gguf --yolo --from --cuda --trt --no-fp16 --thresh --nms
//   --bvh --bvh-template --no-bvh-*-shape-change --bvh-raw-fingers
//   --bw-cutoff --rot-clamp
// Offline-specific knobs (smoothing / tracking / scene / gap / jitter)
// live in the derived part below.
struct Config : public CommonConfig
{
    Config() {
        // Offline-specific default: --rot-clamp's geodesic SLERP clamp is
        // much wider here than in the live binaries because filtfilt
        // smoothing already absorbs most jitter.  See the QuatLPF section
        // in README "Output filtering".
        rot_clamp_deg = 30.0f;
    }

    std::string lbs_path;          // auto: <onnx_dir>/body_model.lbs

    // Smoothing
    enum class Smoothing { ZeroPhase, Forward, Off };
    Smoothing smoothing      = Smoothing::ZeroPhase;

    // Tracking
    float     track_iou_thresh     = 0.10f;   // match threshold per frame
    float     track_dist_weight    = 0.20f;   // λ on ‖Δ pred_cam_t‖ (metres) in cost
    int       track_merge_frames   = 30;      // post-hoc merge: max gap in frames
    float     track_merge_cm       = 50.0f;   // post-hoc merge: max 3D distance
    int       min_track_frames     = 8;       // drop tracks shorter than this many
                                               // detections — typically YOLO false
                                               // positives (single-frame faces in a
                                               // crowd, etc.)

    // Jitter handling
    bool      interpolate_jitter   = false;
    float     jitter_threshold_cm  = 30.0f;   // per-frame, per-keypoint velocity

    // Gap interpolation — fills missing frames between detections inside a
    // single track's lifespan by linear/SLERP-interpolating its bracketing
    // real detections.  On by default: without it, BVHWriter pads with
    // "duplicate last pose" which manifests as a visibly frozen segment in
    // the BVH for any track that briefly loses YOLO detection (partial
    // occlusion, low-confidence frame, fast head turn).  Scene cuts are
    // respected — we never bridge a cut.
    bool      gap_interpolation    = true;
    int       gap_max_frames       = 0;        // 0 = no upper bound, fill any gap

    // Scene-change detection
    //
    // Films cut between shots — Matrix.mp4 has many — and the background pixels
    // change completely across a cut.  Within a shot, even with hand-held
    // camera motion, the background corners track smoothly under
    // Lucas-Kanade optical flow.  We detect a scene change whenever the
    // fraction of corners that track successfully from the previous frame
    // drops below scene_success_thresh.
    //
    // Scene cuts are propagated to:
    //   - Pass 3: don't interpolate jitter across a cut (a "200 cm/frame
    //             velocity" at a cut is real motion, not noise).
    //   - Pass 4: filter each between-cuts segment independently so
    //             smoothing never blends a person from shot A into shot B.
    bool      scene_detection      = true;
    float     scene_success_thresh = 0.50f;
    int       scene_min_corners    = 30;

    // bvh_body_shape_change / bvh_hand_shape_change /
    // bvh_compensate_finger_endsites all come from CommonConfig now.
};

static void print_usage(const char* prog)
{
    printf(
        "Usage: %s --onnx-dir DIR --gguf F.gguf --yolo F.onnx --from VIDEO.mp4 --bvh OUT.bvh [options]\n"
        "\n"
        "REQUIRED\n"
        "  --onnx-dir PATH            Directory with backbone / decoder / body_model ONNX files\n"
        "  --gguf     PATH            pipeline.gguf (MHR + camera heads)\n"
        "  --yolo     PATH            YOLO pose model (.onnx)\n"
        "  --from     VIDEO           Path to a video file.  Webcams / streams / still images NOT supported.\n"
        "  --bvh      PATH            Output BVH base name.  Files written as <stem>_<id>.bvh.\n"
        "\n"
        "PIPELINE\n"
        "  --bvh-template PATH        BVH skeleton template (default ./body.bvh)\n"
        "  --cuda     N               CUDA device index, -1 for CPU (default 0)\n"
        "  --trt                      Use ONNX Runtime TensorRT EP\n"
        "  --no-fp16                  Disable FP16\n"
        "  --thresh   F               YOLO person confidence (default 0.50)\n"
        "  --nms      F               YOLO NMS IoU (default 0.45)\n"
        "\n"
        "SMOOTHING\n"
        "  --smoothing zero-phase|forward|off\n"
        "                             zero-phase  — forward+backward filter, no temporal lag (default)\n"
        "                             forward     — same as live binaries\n"
        "                             off         — no smoothing\n"
        "  --bw-cutoff HZ             Cutoff for Butterworth / QuatLPF (default 6 Hz)\n"
        "  --rot-clamp DEG            Geodesic SLERP clamp on global_rot (deg/frame, default 30; 0 = off)\n"
        "\n"
        "TRACKING\n"
        "  --track-merge-frames N     Merge tracks separated by gaps ≤ N frames (default 30)\n"
        "  --track-merge-cm    CM     ... and within CM cm of 3D root distance (default 50)\n"
        "  --min-track-frames  N      Drop tracks with fewer than N detections (default 8;\n"
        "                             typically YOLO false positives — single-frame extras, etc.)\n"
        "\n"
        "SCENE-CHANGE DETECTION (on by default — gates jitter interpolation + smoothing)\n"
        "  --no-scene-detection       Disable; treat the clip as a single continuous shot.\n"
        "  --static-scene             Alias for --no-scene-detection — assert a single-shot input.\n"
        "  --scene-success-threshold T  Cut when LK-flow success-rate falls below T (default 0.50).\n"
        "  --scene-min-corners      N   Re-seed corners when fewer than N remain (default 30).\n"
        "\n"
        "JITTER INTERPOLATION (opt-in)\n"
        "  --interpolate-jitter       Replace frames whose 3D keypoint velocity exceeds the threshold\n"
        "                             by linear / SLERP interpolation of their neighbours.\n"
        "  --jitter-threshold-cm CM   Per-frame keypoint velocity threshold (default 30 cm/frame)\n"
        "\n"
        "GAP INTERPOLATION (on by default — fills missing frames inside each track)\n"
        "  --no-gap-interpolation     Disable; missing frames will be PADDED with last-known pose\n"
        "                             (this restores the pre-fix freeze-on-occlusion behaviour).\n"
        "  --gap-max-frames N         Skip gaps longer than N frames (0 = always interpolate; default 0).\n"
        "                             For very long occlusions a linear interpolation between two\n"
        "                             unrelated poses can look unphysical — set N to fall back to\n"
        "                             padding for gaps you'd rather see frozen than slowly morphing.\n"
        "\n"
        "BVH SHAPE\n"
        "  --no-bvh-body-shape-change Keep template body bone lengths\n"
        "  --no-bvh-hand-shape-change Keep template hand/finger bone lengths\n"
        "  --bvh-raw-fingers          Do not rescale finger End-Site OFFSETs\n"
        "  --help / -h                This message\n",
        prog);
}

static bool parse_args(int argc, char** argv, Config& c)
{
    for (int i = 1; i < argc; ++i)
    {
        // Common flags first (--onnx-dir / --gguf / --yolo / --from / --cuda /
        // --trt / --no-fp16 / --thresh / --nms / --bvh* / --bw-cutoff /
        // --rot-clamp).  See src/cli_common.h.  Anything that isn't a common
        // flag falls through to the offline-specific dispatch below.
        if (parse_common_arg(argc, argv, i, c)) continue;

#define A1(flag, field, conv) \
        if (!strcmp(argv[i], flag) && i+1 < argc) { c.field = conv(argv[++i]); continue; }
        A1("--lbs",                     lbs_path,            std::string)
        A1("--track-merge-frames",      track_merge_frames,  std::stoi)
        A1("--track-merge-cm",          track_merge_cm,      std::stof)
        A1("--min-track-frames",        min_track_frames,    std::stoi)
        A1("--scene-success-threshold", scene_success_thresh, std::stof)
        A1("--scene-min-corners",       scene_min_corners,   std::stoi)
        A1("--jitter-threshold-cm",     jitter_threshold_cm, std::stof)
        A1("--gap-max-frames",          gap_max_frames,      std::stoi)
#undef A1
        if (!strcmp(argv[i], "--smoothing") && i+1 < argc)
        {
            std::string v = argv[++i];
            if      (v == "zero-phase") c.smoothing = Config::Smoothing::ZeroPhase;
            else if (v == "forward")    c.smoothing = Config::Smoothing::Forward;
            else if (v == "off")        c.smoothing = Config::Smoothing::Off;
            else { fprintf(stderr, "unknown --smoothing %s\n", v.c_str()); return false; }
            continue;
        }
        if (!strcmp(argv[i], "--interpolate-jitter"))     { c.interpolate_jitter = true; continue; }
        if (!strcmp(argv[i], "--no-gap-interpolation"))   { c.gap_interpolation = false; continue; }
        if (!strcmp(argv[i], "--no-scene-detection"))     { c.scene_detection = false; continue; }
        // --static-scene is the user-facing alias for --no-scene-detection:
        // "I'm asserting this clip is a single continuous shot."  Skips the
        // per-frame OpenCV scene-detector work and prevents any false-
        // positive cuts from interrupting the smoothing.
        if (!strcmp(argv[i], "--static-scene"))           { c.scene_detection = false; continue; }
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { print_usage(argv[0]); return false; }
        fprintf(stderr, "unknown argument: %s\n", argv[i]);
        return false;
    }

    if (c.from.empty() || c.bvh_path.empty())
    {
        fprintf(stderr, "--from and --bvh are required\n\n");
        print_usage(argv[0]);
        return false;
    }
    if (c.lbs_path.empty()) c.lbs_path = default_lbs_path(c);

    // Reject webcam-style sources and still images.  This binary is explicitly
    // about offline processing of finite-length video streams.
    bool numeric = !c.from.empty() &&
                   std::all_of(c.from.begin(), c.from.end(),
                               [](char ch){ return std::isdigit((unsigned char)ch); });
    if (numeric)
    {
        fprintf(stderr, "offline mode does not accept webcam indices — give a video file path\n");
        return false;
    }
    const char* ext_dot = strrchr(c.from.c_str(), '.');
    if (ext_dot)
    {
        std::string ext(ext_dot + 1);
        for (auto& ch : ext) ch = (char)std::tolower((unsigned char)ch);
        // Common still-image extensions OpenCV will open as a single-frame
        // "video", which makes no sense for an offline tool that wants to
        // smooth a temporal trajectory.
        if (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp" ||
            ext == "tiff" || ext == "tif" || ext == "webp")
        {
            fprintf(stderr, "offline mode does not accept still images — give a video file path\n");
            return false;
        }
    }
    return true;
}

// ─── Geometry helpers (used by both SceneDetector and the tracker) ──────────

static float bbox_iou(const std::array<float,4>& a, const std::array<float,4>& b)
{
    float ix1 = std::max(a[0], b[0]);
    float iy1 = std::max(a[1], b[1]);
    float ix2 = std::min(a[2], b[2]);
    float iy2 = std::min(a[3], b[3]);
    float iw  = std::max(0.f, ix2 - ix1);
    float ih  = std::max(0.f, iy2 - iy1);
    float inter = iw * ih;
    if (inter <= 0.f) return 0.f;
    float aa = std::max(0.f, a[2]-a[0]) * std::max(0.f, a[3]-a[1]);
    float bb = std::max(0.f, b[2]-b[0]) * std::max(0.f, b[3]-b[1]);
    return inter / (aa + bb - inter + 1e-6f);
}

static float vec3_dist(const std::array<float,3>& a, const std::array<float,3>& b)
{
    float dx = a[0]-b[0], dy = a[1]-b[1], dz = a[2]-b[2];
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

// ════════════════════════════════════════════════════════════════════════════
//  SCENE-CHANGE DETECTOR
// ════════════════════════════════════════════════════════════════════════════
//  Implements the user's request to flag scene changes via background-only
//  corner tracking.  The original idea was to render a person/background
//  mask with a dedicated black/white shader; in this offline tool we have
//  no GL context so we use the YOLO bboxes (dilated) as the person mask.
//  For typical action footage that's accurate enough — the bbox is
//  conservative (it always covers the person) and dilating by ~20 px
//  removes hair/clothing edges that would dominate the corner set on the
//  body region.
//
//  How it works each frame:
//
//    1. Convert the BGR frame to grayscale.
//    2. Build a single-channel mask that is 255 in the background, 0 over
//       each YOLO bbox (dilated).
//    3. Lucas-Kanade-track the previous frame's bg corners forward to this
//       frame.  Count the fraction that ended with status==1.
//    4. If the success rate dips below scene_success_thresh, OR if there
//       are simply no corners left to track, this is a scene cut.
//    5. After a cut (or if our running corner set has thinned out)
//       re-seed via cv::goodFeaturesToTrack on the current mask.
//
//  Frame 0 always returns "not a cut" — we need at least one frame of
//  history to compare against.
// ════════════════════════════════════════════════════════════════════════════

class SceneDetector
{
public:
    SceneDetector(float success_thresh, int min_corners)
        : success_thresh_(success_thresh), min_corners_(min_corners) {}

    // Returns true if this frame is the FIRST frame of a new shot
    // (i.e. there was a cut between frame_idx-1 and frame_idx).
    //
    // Implementation notes (tuned for fast-action footage like matrix.mp4):
    //   * LK uses a 41×41 window with 5 pyramid levels.  Defaults (21×21,
    //     3 levels) bail out on the ~50-150 px inter-frame motion you see
    //     during fight choreography, producing one false-positive cut per
    //     such frame.  The wider window + deeper pyramid handle motions
    //     up to roughly winSize × 2^maxLevel ≈ 1300 px.
    //
    //   * Three independent signals are computed each frame, and we require
    //     ANY TWO OF THREE to agree before declaring a cut.  Voting fuses
    //     the failure modes of the individual heuristics:
    //
    //       (A) corner_fail   — LK optical-flow success rate drops below
    //                            scene_success_thresh.  Mostly trips on cuts
    //                            but also on rapid panning.
    //       (B) hist_diff     — HSV histogram correlation of the bg drops
    //                            below 0.5.  Mostly trips on cuts but can
    //                            be fooled by a strong lighting change.
    //       (C) person_jump   — the per-frame set of detected people
    //                            changes in a way that isn't consistent
    //                            with continuous motion: either the count
    //                            changes by >=2 (or doubles/halves), or
    //                            most current detections fail to match any
    //                            previous person within 1 m of 3D root
    //                            distance.  This catches the case where
    //                            the background and palette barely change
    //                            across a cut but the actors do.
    //
    //   * A simple debounce: we never flag two cuts within
    //     MIN_FRAMES_BETWEEN_CUTS=4 frames of each other.  Real cuts are
    //     isolated events; consecutive flags are always tracker artefacts.
    bool process(const cv::Mat& bgr,
                 const std::vector<fsb::MHRResult>& detections,
                 int frame_idx)
    {
        cv::Mat gray;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

        // Background mask: 255 everywhere except inside dilated bboxes.
        cv::Mat mask(gray.size(), CV_8U, cv::Scalar(255));
        for (const auto& r : detections) {
            const int pad = 20;
            int x1 = std::max(0, (int)r.bbox[0] - pad);
            int y1 = std::max(0, (int)r.bbox[1] - pad);
            int x2 = std::min(gray.cols, (int)r.bbox[2] + pad);
            int y2 = std::min(gray.rows, (int)r.bbox[3] + pad);
            if (x2 > x1 && y2 > y1)
                mask(cv::Rect(x1, y1, x2-x1, y2-y1)).setTo(0);
        }

        // Compute the HSV-histogram fingerprint of the background.  Using
        // 16 bins × 16 bins × 16 bins (4096 entries) — enough to discriminate
        // shots while staying cheap (one cv::calcHist per frame).
        cv::Mat hsv, hist;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        const int h_bins = 16, s_bins = 16, v_bins = 16;
        const int hist_size[] = { h_bins, s_bins, v_bins };
        const float h_r[] = {0, 180}, s_r[] = {0, 256}, v_r[] = {0, 256};
        const float* ranges[] = { h_r, s_r, v_r };
        const int channels[] = { 0, 1, 2 };
        cv::calcHist(&hsv, 1, channels, mask, hist, 3, hist_size, ranges, true, false);
        cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX);

        bool corner_fail = false;
        bool hist_diff   = false;

        if (frame_idx > 0 && !prev_gray_.empty() && !prev_corners_.empty()) {
            std::vector<cv::Point2f> next;
            std::vector<uchar> status;
            std::vector<float> err;
            cv::calcOpticalFlowPyrLK(prev_gray_, gray, prev_corners_, next,
                                     status, err,
                                     /*winSize=*/cv::Size(41,41),
                                     /*maxLevel=*/5);
            int good = 0;
            std::vector<cv::Point2f> kept;
            for (size_t i = 0; i < status.size(); ++i) {
                if (!status[i]) continue;
                int x = (int)next[i].x, y = (int)next[i].y;
                if (x < 0 || y < 0 || x >= mask.cols || y >= mask.rows) continue;
                if (mask.at<uchar>(y, x) == 0) continue;
                ++good;
                kept.push_back(next[i]);
            }
            float rate = (float)good / (float)prev_corners_.size();
            corner_fail = (rate < success_thresh_);
            if (!corner_fail) prev_corners_ = std::move(kept);
        }

        if (frame_idx > 0 && !prev_hist_.empty()) {
            // Correlation: 1.0 = identical, ~0 = unrelated.  Anything below
            // 0.5 across a full background means the colour distribution
            // changed substantially — strong scene-change signal.
            double corr = cv::compareHist(prev_hist_, hist, cv::HISTCMP_CORREL);
            hist_diff = (corr < 0.5);
        }

        // Signal (C) — person-set discontinuity.  In a continuous shot the
        // number of detected people stays roughly constant frame-to-frame
        // and each detection sits within ~1 m of *some* previous detection
        // in 3D (pred_cam_t is in metres).  Across a cut, both invariants
        // tend to break at once.
        bool person_jump = false;
        if (frame_idx > 0) {
            int n0 = (int)prev_detections_.size();
            int n1 = (int)detections.size();
            // Skip count check when EITHER frame had no people: persons
            // entering / leaving the frame naturally produces a 0↔N change
            // and we don't want to mistake that for a cut.
            if (n0 > 0 && n1 > 0) {
                // Big count change is strong evidence on its own.
                int n_min = std::min(n0, n1), n_max = std::max(n0, n1);
                if (n_max - n_min >= 2 || n_max >= 2 * n_min) person_jump = true;

                // Even with stable counts, if MOST current detections can't
                // be matched to a previous one within 1 m, the actor set
                // has changed.  Greedy nearest-prev-by-pred_cam_t suffices
                // — we just want to know the median match distance.
                if (!person_jump) {
                    std::vector<float> match_dist;
                    match_dist.reserve(n1);
                    for (const auto& cur : detections) {
                        float best = std::numeric_limits<float>::infinity();
                        for (const auto& prv : prev_detections_) {
                            float d = vec3_dist(cur.pred_cam_t, prv.pred_cam_t);
                            if (d < best) best = d;
                        }
                        match_dist.push_back(best);
                    }
                    std::sort(match_dist.begin(), match_dist.end());
                    float median_d = match_dist[match_dist.size() / 2];
                    if (median_d > 1.0f) person_jump = true;
                }
            }
        }

        // Vote: any two of {corner_fail, hist_diff, person_jump} agreeing
        // is enough.  This makes each signal corroborate the others —
        // tracking artefacts (false positive on A or B alone) get rejected
        // unless a SECOND independent signal also fires.
        int signals = (int)corner_fail + (int)hist_diff + (int)person_jump;
        bool scene_change = (signals >= 2);

        // Debounce — never report cuts in adjacent frames.  Real shots last
        // at least a few frames; consecutive flags are LK/hist noise.
        constexpr int MIN_FRAMES_BETWEEN_CUTS = 4;
        if (scene_change && (frame_idx - last_cut_frame_) < MIN_FRAMES_BETWEEN_CUTS)
            scene_change = false;
        if (scene_change) last_cut_frame_ = frame_idx;

        // Re-seed corners on a real cut OR when the working set has thinned.
        if (scene_change || (int)prev_corners_.size() < min_corners_) {
            cv::goodFeaturesToTrack(gray, prev_corners_, /*maxCorners=*/200,
                                    /*qualityLevel=*/0.01, /*minDistance=*/10,
                                    mask);
        }
        prev_gray_       = std::move(gray);
        prev_hist_       = hist.clone();
        prev_detections_ = detections;          // copy: needed for signal (C)
        return scene_change;
    }

private:
    cv::Mat                       prev_gray_;
    cv::Mat                       prev_hist_;
    std::vector<cv::Point2f>      prev_corners_;
    std::vector<fsb::MHRResult>   prev_detections_;   // for person_jump signal
    float                         success_thresh_;
    int                           min_corners_;
    int                           last_cut_frame_ = -1000;
};

// Returns true iff some scene-cut frame index `c` satisfies a < c <= b — i.e.
// a cut occurred AT OR AFTER (a+1) and AT OR BEFORE b.  Pass 3 and Pass 4 use
// this to gate operations that span two frames.
static bool cut_between(const std::vector<int>& cuts, int a, int b)
{
    auto it = std::upper_bound(cuts.begin(), cuts.end(), a);
    return it != cuts.end() && *it <= b;
}

// ════════════════════════════════════════════════════════════════════════════
//  PASS 1 — INFERENCE
// ════════════════════════════════════════════════════════════════════════════
//  Decode the whole video and run the pipeline once per frame.  This is the
//  one pass we cannot avoid; everything downstream operates on the buffered
//  result.  Memory cost is dominated by MHRResult::pred_vertices (18439×3
//  floats per detection ≈ 220 KB), so we evict it immediately after each
//  detection — Pass-2/3/4 only need joint angles, keypoints, and pred_cam_t.
// ════════════════════════════════════════════════════════════════════════════

static bool run_inference_pass(fsb::Pipeline& pipeline,
                               cv::VideoCapture& cap,
                               std::vector<FrameRecord>& out_frames,
                               std::vector<int>& out_scene_cuts,
                               double& out_fps,
                               const Config& cfg)
{
    out_fps = cap.get(cv::CAP_PROP_FPS);
    if (out_fps <= 0.0) out_fps = 30.0;
    int total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);

    printf("[pass1] decoding + inference%s (%d frames @ %.2f fps)\n",
           cfg.scene_detection ? " + scene detection" : "",
           total_frames, out_fps);

    // Scene detector lives alongside the loop; it consumes the BGR frame
    // before we drop it, so no extra storage is needed.
    SceneDetector scene(cfg.scene_success_thresh, cfg.scene_min_corners);

    cv::Mat frame;
    int idx = 0;
    auto t0 = Clock::now();
    while (cap.read(frame))
    {
        if (frame.empty()) break;

        FrameRecord rec;
        rec.frame_idx  = idx;
        rec.detections = pipeline.process_bgr(frame.data, frame.cols, frame.rows);

        // Strip the mesh — we never need it for BVH and it dominates RAM.
        for (auto& r : rec.detections) {
            r.pred_vertices.clear();
            r.pred_vertices.shrink_to_fit();
        }

        rec.track_ids.assign(rec.detections.size(), -1);
        rec.was_interpolated.assign(rec.detections.size(), 0);

        // Detect scene change AFTER inference so the detector sees the
        // current frame's person mask via the detections we just produced.
        if (cfg.scene_detection) {
            if (scene.process(frame, rec.detections, idx))
                out_scene_cuts.push_back(idx);
        }

        out_frames.push_back(std::move(rec));

        if (++idx % 30 == 0) {
            double inv_avg = (double)idx / std::max(1.0, ms_since(t0));
            int eta_s = (int)((total_frames - idx) / std::max(0.1, inv_avg) / 1000.0);
            printf("\r[pass1]   %d / %d frames  (eta ~%d s)   ",
                   idx, total_frames, eta_s);
            fflush(stdout);
        }
    }
    double elapsed_s = ms_since(t0) / 1000.0;
    printf("\r[pass1] done — %d frames in %.1f s (%.1f fps inference); %zu scene cut(s)\n",
           idx, elapsed_s, (double)idx / std::max(1e-3, elapsed_s),
           out_scene_cuts.size());
    if (!out_scene_cuts.empty()) {
        printf("[pass1]   scene-cut frames:");
        for (int c : out_scene_cuts) printf(" %d", c);
        printf("\n");
    }
    return idx > 0;
}

// ════════════════════════════════════════════════════════════════════════════
//  PASS 2 — GLOBAL TRACKING
// ════════════════════════════════════════════════════════════════════════════
//  The live binaries' tracker (BVHWriter::assign_tracks) is greedy IoU per
//  frame and forward-only.  It works for most cases but is fragile when
//  people cross or briefly leave the frame.  Offline we have the entire
//  timeline, so we do better in two ways:
//
//    1. Richer per-frame cost.  The greedy match in BVHWriter uses just
//       bbox IoU; here we add a 3D-distance penalty using pred_cam_t (the
//       camera-space root position).  Two people whose bboxes briefly
//       overlap as they cross usually still have a clean 3D separation.
//
//    2. Post-hoc track merge.  A track that ends at frame F and a track
//       that starts at frame G ≤ F + N with its first 3D position close
//       to the first track's last 3D position is almost certainly the
//       same person re-acquired after a brief occlusion.  We splice them
//       under the older track's ID.
//
//  The intra-frame matching is the same greedy-by-cost we already trust;
//  the per-track cost is cheap enough that we don't need Hungarian.
// ════════════════════════════════════════════════════════════════════════════

// A running tracker state.  Carries the last known bbox + 3D root for cost
// computation and the session frame the track was last detected on.
struct LiveTrack
{
    int                  id;
    int                  last_frame;
    std::array<float,4>  last_bbox;
    std::array<float,3>  last_cam_t;
    std::array<float,4>  first_bbox;
    std::array<float,3>  first_cam_t;
    int                  first_frame;
};

static std::vector<Track>
build_global_tracks(std::vector<FrameRecord>& frames, const Config& cfg)
{
    printf("[pass2] global identity tracking …\n");

    std::vector<LiveTrack> live;
    int next_id = 0;

    // Per-track ALL-frames record we'll return after merging.
    std::vector<Track> tracks;
    auto find_track = [&](int id) -> Track* {
        for (auto& t : tracks) if (t.id == id) return &t;
        return nullptr;
    };

    constexpr int MAX_MISSING = 120;  // ≈ 4 s @ 30 fps; longer than the live
                                       // tracker because Pass-3/4 can also
                                       // fill gaps via interpolation.

    for (size_t f = 0; f < frames.size(); ++f)
    {
        auto& fr = frames[f];
        const int F = (int)f;
        size_t N = fr.detections.size();

        // Sort all (live-track, detection) candidate pairs by combined cost,
        // then greedily accept the cheapest while each side is unclaimed.
        struct Pair { int t; int d; float cost; };
        std::vector<Pair> pairs;
        pairs.reserve(live.size() * N);

        for (size_t t = 0; t < live.size(); ++t) {
            for (size_t d = 0; d < N; ++d) {
                float iou = bbox_iou(live[t].last_bbox, fr.detections[d].bbox);
                if (iou < cfg.track_iou_thresh) continue;
                // pred_cam_t is in metres; cost contribution is metres * λ.
                float dist = vec3_dist(live[t].last_cam_t, fr.detections[d].pred_cam_t);
                float cost = (1.0f - iou) + cfg.track_dist_weight * dist;
                pairs.push_back({(int)t, (int)d, cost});
            }
        }
        std::sort(pairs.begin(), pairs.end(),
                  [](const Pair& a, const Pair& b){ return a.cost < b.cost; });

        std::vector<char> t_taken(live.size(), 0), d_taken(N, 0);
        for (const auto& p : pairs)
        {
            if (t_taken[p.t] || d_taken[p.d]) continue;
            t_taken[p.t] = d_taken[p.d] = 1;
            // Update live track
            live[p.t].last_frame  = F;
            live[p.t].last_bbox   = fr.detections[p.d].bbox;
            live[p.t].last_cam_t  = fr.detections[p.d].pred_cam_t;
            // Record into the full-session track
            fr.track_ids[p.d] = live[p.t].id;
            Track* tr = find_track(live[p.t].id);
            if (tr) {
                tr->last_frame = F;
                tr->frame_to_det[F] = (int)p.d;
            }
        }

        // Unmatched detections spawn new tracks.
        for (size_t d = 0; d < N; ++d) {
            if (d_taken[d]) continue;
            LiveTrack t;
            t.id          = next_id++;
            t.first_frame = t.last_frame = F;
            t.first_bbox  = t.last_bbox  = fr.detections[d].bbox;
            t.first_cam_t = t.last_cam_t = fr.detections[d].pred_cam_t;
            live.push_back(t);
            fr.track_ids[d] = t.id;

            Track full;
            full.id          = t.id;
            full.first_frame = full.last_frame = F;
            full.frame_to_det[F] = (int)d;
            tracks.push_back(full);
        }

        // Retire any track we haven't seen in too long.
        live.erase(std::remove_if(live.begin(), live.end(),
            [F](const LiveTrack& t){ return (F - t.last_frame) > MAX_MISSING; }),
            live.end());
    }

    printf("[pass2]   greedy pass: %zu tracks before merge\n", tracks.size());

    // ── Post-hoc track merge ──────────────────────────────────────────────
    // For every pair (A, B) where A ends shortly before B starts and the 3D
    // positions are close, splice them under A's id.  We do this iteratively
    // until no more merges are possible.
    bool changed = true;
    int merges = 0;
    while (changed)
    {
        changed = false;
        // Sort tracks by first_frame for a tidy O(n²) scan.
        std::sort(tracks.begin(), tracks.end(),
                  [](const Track& a, const Track& b){ return a.first_frame < b.first_frame; });

        for (size_t i = 0; i < tracks.size(); ++i) {
            for (size_t j = i + 1; j < tracks.size(); ++j) {
                Track& A = tracks[i];
                Track& B = tracks[j];
                if (B.first_frame <= A.last_frame) continue;  // overlap → can't merge
                int gap = B.first_frame - A.last_frame;
                if (gap > cfg.track_merge_frames) continue;

                // 3D distance between A.last and B.first cam_t (metres → cm).
                auto& A_last_det = frames[A.last_frame ].detections[A.frame_to_det[A.last_frame ]];
                auto& B_first_det= frames[B.first_frame].detections[B.frame_to_det[B.first_frame]];
                float d_cm = vec3_dist(A_last_det.pred_cam_t, B_first_det.pred_cam_t) * 100.f;
                if (d_cm > cfg.track_merge_cm) continue;

                // Merge: re-label B's frames as A's id and union their maps.
                for (auto& [fr_idx, det_idx] : B.frame_to_det) {
                    frames[fr_idx].track_ids[det_idx] = A.id;
                    A.frame_to_det[fr_idx] = det_idx;
                }
                A.last_frame = std::max(A.last_frame, B.last_frame);
                tracks.erase(tracks.begin() + j);
                ++merges;
                changed = true;
                break;
            }
            if (changed) break;
        }
    }

    printf("[pass2]   merged %d pairs → %zu tracks\n", merges, tracks.size());

    // ── Prune short tracks ────────────────────────────────────────────────
    // YOLO will occasionally emit a one- or two-frame phantom person — a
    // misclassified face in a crowd, motion-blur reflection, etc.  These
    // surface as tiny tracks that nobody wants in their BVH output.  Drop
    // anything below --min-track-frames detections.  We also have to clear
    // the corresponding entries in FrameRecord::track_ids so PASS 6 doesn't
    // try to write them anyway.
    if (cfg.min_track_frames > 1) {
        int dropped = 0;
        auto pred = [&](const Track& t){
            if ((int)t.frame_to_det.size() < cfg.min_track_frames) {
                for (auto& [fr_idx, det_idx] : t.frame_to_det)
                    frames[fr_idx].track_ids[det_idx] = -1;
                ++dropped;
                return true;
            }
            return false;
        };
        tracks.erase(std::remove_if(tracks.begin(), tracks.end(), pred), tracks.end());
        printf("[pass2]   pruned %d tracks below --min-track-frames=%d → %zu kept\n",
               dropped, cfg.min_track_frames, tracks.size());
    }

    for (const auto& t : tracks) {
        printf("[pass2]     track %d  frames [%d,%d]  (%zu detections)\n",
               t.id, t.first_frame, t.last_frame, t.frame_to_det.size());
    }
    return tracks;
}

// ─── MHR-result interpolation helper ─────────────────────────────────────────
//
// Linearly interpolates every scalar field, SLERPs global_rot (since Euler
// linear-blend across ±π wraps would flip the body), and takes bbox /
// focal_length / pred_cam_t along linearly.  Used by both the gap-fill pass
// below and the jitter pass further down.
//
// Note on `body_pose` / `hand_pose` / `mhr_model_params`: these are Euler
// angles too, so the same ±π caveat applies.  In practice the per-frame
// deltas are small (a few degrees) and a linear blend across a short gap
// produces results visually indistinguishable from a per-joint quaternion
// SLERP.  Pass 5's zero-phase smoother cleans up any small artefacts.  If
// you see weird limb-flip artefacts on long gaps, that's a sign the gap was
// across a moment where multiple joints crossed gimbal-lock — easiest
// mitigation is `--gap-max-frames N` so those long gaps stay frozen.

static fsb::MHRResult interp_mhr(const fsb::MHRResult& a,
                                  const fsb::MHRResult& b,
                                  float t)
{
    fsb::MHRResult out = a;                  // start from a; overwrite mutable fields
    const float u = 1.f - t;

    for (int k = 0; k < 4; ++k) out.bbox[k]       = a.bbox[k]       * u + b.bbox[k]       * t;
    for (int k = 0; k < 3; ++k) out.pred_cam_t[k] = a.pred_cam_t[k] * u + b.pred_cam_t[k] * t;
    out.focal_length = a.focal_length * u + b.focal_length * t;

    if (a.keypoints_3d.size() == b.keypoints_3d.size() && !a.keypoints_3d.empty()) {
        out.keypoints_3d.assign(a.keypoints_3d.size(), 0.f);
        for (size_t k = 0; k < out.keypoints_3d.size(); ++k)
            out.keypoints_3d[k] = a.keypoints_3d[k] * u + b.keypoints_3d[k] * t;
    }
    if (a.keypoints_2d.size() == b.keypoints_2d.size() && !a.keypoints_2d.empty()) {
        out.keypoints_2d.assign(a.keypoints_2d.size(), 0.f);
        for (size_t k = 0; k < out.keypoints_2d.size(); ++k)
            out.keypoints_2d[k] = a.keypoints_2d[k] * u + b.keypoints_2d[k] * t;
    }
    if (a.body_pose.size() == b.body_pose.size() && !a.body_pose.empty()) {
        out.body_pose.assign(a.body_pose.size(), 0.f);
        for (size_t k = 0; k < out.body_pose.size(); ++k)
            out.body_pose[k] = a.body_pose[k] * u + b.body_pose[k] * t;
    }
    if (a.hand_pose.size() == b.hand_pose.size() && !a.hand_pose.empty()) {
        out.hand_pose.assign(a.hand_pose.size(), 0.f);
        for (size_t k = 0; k < out.hand_pose.size(); ++k)
            out.hand_pose[k] = a.hand_pose[k] * u + b.hand_pose[k] * t;
    }
    for (size_t k = 0; k < out.mhr_model_params.size(); ++k)
        out.mhr_model_params[k] = a.mhr_model_params[k] * u + b.mhr_model_params[k] * t;

    // global_rot — SLERP via the same euler↔quat helpers Pass 4 already uses.
    float qa[4], qb[4], qc[4];
    euler_zyx_to_quat(a.global_rot[0], a.global_rot[1], a.global_rot[2], qa);
    euler_zyx_to_quat(b.global_rot[0], b.global_rot[1], b.global_rot[2], qb);
    float dot = qa[0]*qb[0] + qa[1]*qb[1] + qa[2]*qb[2] + qa[3]*qb[3];
    if (dot < 0.f) { for (int k=0;k<4;++k) qb[k] = -qb[k]; dot = -dot; }
    float omega = std::acos(std::max(-1.f, std::min(1.f, dot)));
    float s     = std::sin(omega);
    float wa, wb;
    if (s < 1e-6f) { wa = u; wb = t; }
    else           { wa = std::sin(u * omega) / s; wb = std::sin(t * omega) / s; }
    for (int k = 0; k < 4; ++k) qc[k] = wa*qa[k] + wb*qb[k];
    float n = std::sqrt(qc[0]*qc[0]+qc[1]*qc[1]+qc[2]*qc[2]+qc[3]*qc[3]);
    if (n > 1e-9f) for (int k=0;k<4;++k) qc[k] /= n;
    quat_to_euler_zyx(qc, &out.global_rot[0], &out.global_rot[1], &out.global_rot[2]);

    return out;
}

// ════════════════════════════════════════════════════════════════════════════
//  PASS 3 — GAP INTERPOLATION  (on by default; --no-gap-interpolation to opt out)
// ════════════════════════════════════════════════════════════════════════════
//  For every track, the offline tracker records the frames in which the
//  person was actually detected (track.frame_to_det).  Any frame in
//  [first_frame, last_frame] that is NOT in that map is a hole — usually a
//  YOLO confidence dip, brief partial occlusion, or a fast head turn that
//  evicted the bbox momentarily.
//
//  Before this pass existed, write_frame_external received those holes via
//  the `pad_ids` list, and BVHWriter::pad_continuation_frame duplicated the
//  previous row of motion data.  In Blender that shows up as a frozen pose
//  for the entire gap — exactly the regression that prompted this pass.
//
//  This pass walks each track's detected frames in order.  For any pair
//  of consecutive detected frames (fa, fb) with at least one missing
//  frame between them, it linearly / SLERP-interpolates a synthetic
//  MHRResult at every intermediate frame f ∈ (fa, fb), inserts it into
//  frames[f].detections, and registers it in track.frame_to_det.  Pass 4
//  (smoothing) and Pass 6 (export) then see a contiguous track timeline
//  and produce real motion in the BVH instead of a hold.
//
//  Two guard conditions:
//    * Scene cuts inside the gap break interpolation — bridging from a
//      person's pose in shot A to their pose in shot B would smear them
//      across the cut.  Honoured via cut_between(scene_cuts, fa, fb).
//    * `--gap-max-frames N` (0 = no limit) skips gaps longer than N frames.
//      Useful when very long occlusions produce visibly unphysical
//      morphing between two unrelated poses — bound the interpolation to
//      short occlusions and let longer ones stay frozen.
// ════════════════════════════════════════════════════════════════════════════

static void gap_interpolation_pass(std::vector<FrameRecord>& frames,
                                    std::vector<Track>& tracks,
                                    const std::vector<int>& scene_cuts,
                                    const Config& cfg)
{
    if (!cfg.gap_interpolation) {
        printf("[pass3] gap interpolation disabled — missing frames will be padded\n");
        return;
    }
    if (cfg.gap_max_frames > 0)
        printf("[pass3] filling track gaps (max %d frames)%s …\n",
               cfg.gap_max_frames,
               scene_cuts.empty() ? "" : ", honouring scene cuts");
    else
        printf("[pass3] filling track gaps%s …\n",
               scene_cuts.empty() ? "" : " (honouring scene cuts)");

    int n_gaps = 0, n_skipped_cut = 0, n_skipped_long = 0, n_synth = 0;

    for (auto& track : tracks)
    {
        if (track.frame_to_det.size() < 2) continue;

        // Snapshot the existing detected frames in order — we'll mutate
        // track.frame_to_det inside the loop and don't want to confuse the
        // iterator.
        std::vector<int> keys;
        keys.reserve(track.frame_to_det.size());
        for (const auto& kv : track.frame_to_det) keys.push_back(kv.first);

        for (size_t i = 1; i < keys.size(); ++i)
        {
            int fa = keys[i-1], fb = keys[i];
            int gap = fb - fa - 1;
            if (gap <= 0) continue;                          // no missing frames
            ++n_gaps;

            if (cut_between(scene_cuts, fa, fb))  { ++n_skipped_cut;  continue; }
            if (cfg.gap_max_frames > 0 && gap > cfg.gap_max_frames)
                                                  { ++n_skipped_long; continue; }

            const auto& ra = frames[fa].detections[track.frame_to_det[fa]];
            const auto& rb = frames[fb].detections[track.frame_to_det[fb]];

            for (int f = fa + 1; f < fb; ++f) {
                float t = (float)(f - fa) / (float)(fb - fa);
                fsb::MHRResult synth = interp_mhr(ra, rb, t);

                // Append the synthetic detection.  Updating the parallel
                // tracking arrays so jitter / smoothing / export all see
                // a self-consistent FrameRecord.
                int new_idx = (int)frames[f].detections.size();
                frames[f].detections.push_back(std::move(synth));
                frames[f].track_ids.push_back(track.id);
                frames[f].was_interpolated.push_back(1);
                track.frame_to_det[f] = new_idx;
                ++n_synth;
            }
        }
    }

    printf("[pass3]   synthesized %d frames across %d gap(s) "
           "(skipped %d at scene cuts, %d beyond --gap-max-frames)\n",
           n_synth, n_gaps, n_skipped_cut, n_skipped_long);
}

// ════════════════════════════════════════════════════════════════════════════
//  PASS 4 — JITTER INTERPOLATION  (opt-in via --interpolate-jitter)
// ════════════════════════════════════════════════════════════════════════════
//  A single bad FFN frame typically presents as a 3D root or 3D keypoint
//  velocity an order of magnitude above the surrounding motion: 30+ cm in
//  one frame for a hand or hip, while the rest of the time those joints
//  move a few cm per frame.  We flag such frames per-track-per-keypoint
//  and replace them with a linear interpolation of their neighbours.
//
//  Limitations: we don't try to interpolate body_pose Euler angles directly
//  — they're high-dim and the per-channel linear interp would risk gimbal
//  artefacts.  Instead we interpolate the camera-space root position,
//  global_rot via SLERP, and keypoints_3d linearly.  body_pose / hand_pose
//  / mhr_model_params are taken from the closer non-jittered neighbour.
//
//  Frames at the very beginning or end of a track whose first/last frames
//  are jittery are NOT touched — there's no second neighbour to interpolate
//  from, so we leave them alone.
// ════════════════════════════════════════════════════════════════════════════

static void interpolate_jitter_pass(std::vector<FrameRecord>& frames,
                                    const std::vector<Track>& tracks,
                                    const std::vector<int>& scene_cuts,
                                    const Config& cfg)
{
    if (!cfg.interpolate_jitter) return;
    printf("[pass4] jitter detection & interpolation (threshold %.1f cm/frame)%s …\n",
           cfg.jitter_threshold_cm,
           scene_cuts.empty() ? "" : " — honouring scene cuts");

    int n_flagged = 0;
    int n_interpolated = 0;

    for (const auto& track : tracks)
    {
        if (track.frame_to_det.size() < 3) continue;   // not enough neighbours

        // Build an ordered list of (frame_idx, det_idx) for this track.
        std::vector<std::pair<int,int>> seq(track.frame_to_det.begin(),
                                            track.frame_to_det.end());

        // Flag jittery frames.  We mark a frame as jittery if the max
        // keypoint velocity (cm/frame, where keypoints_3d is in metres)
        // between this frame and the previous detected one exceeds the
        // threshold.  This is a coarse but effective filter.
        std::vector<char> jit(seq.size(), 0);
        std::vector<char> cut_before(seq.size(), 0);   // scene cut sits in (seq[i-1], seq[i]]
        for (size_t i = 1; i < seq.size(); ++i)
        {
            // A scene cut between the two frames invalidates the velocity
            // comparison — the person legitimately "teleported".  Mark the
            // pair so the interpolator below won't bridge across it either.
            if (cut_between(scene_cuts, seq[i-1].first, seq[i].first)) {
                cut_before[i] = 1;
                continue;
            }
            const auto& a = frames[seq[i-1].first].detections[seq[i-1].second];
            const auto& b = frames[seq[i  ].first].detections[seq[i  ].second];
            if (a.keypoints_3d.size() < 70*3 || b.keypoints_3d.size() < 70*3) continue;

            float max_v = 0.f;
            int n_kps = (int)std::min(a.keypoints_3d.size(), b.keypoints_3d.size()) / 3;
            for (int k = 0; k < n_kps; ++k) {
                float dx = b.keypoints_3d[3*k+0] - a.keypoints_3d[3*k+0];
                float dy = b.keypoints_3d[3*k+1] - a.keypoints_3d[3*k+1];
                float dz = b.keypoints_3d[3*k+2] - a.keypoints_3d[3*k+2];
                float v_cm = 100.0f * std::sqrt(dx*dx + dy*dy + dz*dz);
                if (v_cm > max_v) max_v = v_cm;
            }
            if (max_v > cfg.jitter_threshold_cm) {
                jit[i] = 1;
                ++n_flagged;
            }
        }

        // Interpolate flagged frames.  For each flagged frame i, find the
        // closest non-flagged anchors on both sides; if both exist AND no
        // scene cut lies between them and i, replace this frame's mutable
        // fields with interpolated values.
        for (size_t i = 1; i + 1 < seq.size(); ++i) {
            if (!jit[i]) continue;

            int lo = (int)i - 1;
            while (lo >= 0 && jit[lo]) --lo;
            size_t hi = i + 1;
            while (hi < seq.size() && jit[hi]) ++hi;
            if (lo < 0 || hi >= seq.size()) continue;        // no two-sided anchor

            // Bail out if a scene cut sits between the anchors and the
            // jittery frame in either direction — we don't want to bridge
            // the cut.
            if (cut_between(scene_cuts, seq[lo].first, seq[i].first) ||
                cut_between(scene_cuts, seq[i].first,  seq[hi].first))
                continue;

            const auto& aF = frames[seq[lo].first].detections[seq[lo].second];
            const auto& bF = frames[seq[hi].first].detections[seq[hi].second];
            auto&       cF = frames[seq[i ].first].detections[seq[i ].second];

            int fa = seq[lo].first, fb = seq[hi].first, fc = seq[i].first;
            float t = (float)(fc - fa) / std::max(1, fb - fa);

            // pred_cam_t — linear
            for (int k = 0; k < 3; ++k)
                cF.pred_cam_t[k] = aF.pred_cam_t[k] * (1.f - t) + bF.pred_cam_t[k] * t;

            // keypoints_3d — linear, channel-wise
            if (aF.keypoints_3d.size() == bF.keypoints_3d.size() &&
                cF.keypoints_3d.size() == aF.keypoints_3d.size())
            {
                for (size_t k = 0; k < cF.keypoints_3d.size(); ++k)
                    cF.keypoints_3d[k] = aF.keypoints_3d[k] * (1.f - t) + bF.keypoints_3d[k] * t;
            }

            // global_rot — SLERP via the quaternion helpers we already trust.
            float qa[4], qb[4], qc[4];
            euler_zyx_to_quat(aF.global_rot[0], aF.global_rot[1], aF.global_rot[2], qa);
            euler_zyx_to_quat(bF.global_rot[0], bF.global_rot[1], bF.global_rot[2], qb);
            // Hemisphere-correct before mixing.
            float dot = qa[0]*qb[0] + qa[1]*qb[1] + qa[2]*qb[2] + qa[3]*qb[3];
            if (dot < 0) { for (int k=0;k<4;++k) qb[k] = -qb[k]; dot = -dot; }
            float omega = std::acos(std::max(-1.f, std::min(1.f, dot)));
            float s     = std::sin(omega);
            float a, b;
            if (s < 1e-6f) { a = 1.f - t; b = t; }
            else           { a = std::sin((1.f - t) * omega) / s;
                              b = std::sin(t * omega)         / s; }
            for (int k = 0; k < 4; ++k) qc[k] = a*qa[k] + b*qb[k];
            float n = std::sqrt(qc[0]*qc[0]+qc[1]*qc[1]+qc[2]*qc[2]+qc[3]*qc[3]);
            if (n > 1e-9f) for (int k=0;k<4;++k) qc[k] /= n;
            quat_to_euler_zyx(qc, &cF.global_rot[0], &cF.global_rot[1], &cF.global_rot[2]);

            // body_pose / hand_pose / mhr_model_params — take from the closer
            // anchor.  These are joint Euler angles where naive linear
            // interpolation is unsafe (wrap / gimbal-lock).  Picking the
            // closer non-jittered frame is a conservative substitute.
            const auto& near = (t < 0.5f) ? aF : bF;
            if (cF.body_pose.size() == near.body_pose.size())
                cF.body_pose = near.body_pose;
            if (cF.hand_pose.size() == near.hand_pose.size())
                cF.hand_pose = near.hand_pose;
            cF.mhr_model_params = near.mhr_model_params;

            frames[seq[i].first].was_interpolated[seq[i].second] = 1;
            ++n_interpolated;
        }
    }

    printf("[pass4]   flagged %d frames; interpolated %d.\n", n_flagged, n_interpolated);
}

// ════════════════════════════════════════════════════════════════════════════
//  PASS 5 — ZERO-PHASE SMOOTHING
// ════════════════════════════════════════════════════════════════════════════
//  filtfilt: run the IIR forward over the whole series, reverse the result,
//  run it again (initialised fresh), reverse back.  The phase shift of the
//  two passes cancels exactly, leaving a zero-lag output.  The magnitude
//  response is squared (so |H(ω)|² instead of |H(ω)|) — to keep the
//  effective bandwidth comparable to the one-pass filter, you'd push the
//  cutoff up by √2.  The user knob is still --bw-cutoff so we document the
//  behaviour change rather than secretly re-mapping the value.
//
//  Applied per-track:
//    * pred_cam_t[3]      — Butterworth forward+backward
//    * keypoints_3d[70×3] — Butterworth (where present)
//    * body_pose[133]     — Butterworth (per Euler channel; same caveats as
//                            online — but most channels are 1-DOF here)
//    * hand_pose[108]     — Butterworth
//    * mhr_model_params[204] — Butterworth (drives the LBS at BVH-write time)
//    * global_rot[3]      — QuatLPF forward+backward
//
//  Forward-only smoothing is also provided for parity with the live binaries
//  (Smoothing::Forward); --smoothing off short-circuits this pass entirely.
// ════════════════════════════════════════════════════════════════════════════

// One-pass scalar filter over a sequence of values.
static void filter_forward_scalar(std::vector<float>& xs, float fs, float fc)
{
    if (xs.empty() || fc <= 0.f || fc >= fs * 0.5f) return;  // pass-through above Nyquist
    ButterWorth f{};
    initButterWorth(&f, fs, fc);
    for (auto& v : xs) v = filter(&f, v);
}

// filtfilt — zero-phase forward+backward.
static void filtfilt_scalar(std::vector<float>& xs, float fs, float fc)
{
    if (xs.empty() || fc <= 0.f || fc >= fs * 0.5f) return;
    filter_forward_scalar(xs, fs, fc);
    std::reverse(xs.begin(), xs.end());
    filter_forward_scalar(xs, fs, fc);
    std::reverse(xs.begin(), xs.end());
}

// Apply scalar filter to a stride-K channelised buffer (e.g. keypoints_3d as
// frames × (70*3)).  We extract one channel at a time into a contiguous
// vector, filter, write back.  K must be the channel count per frame.
static void filter_channels(std::vector<std::vector<float>>& per_frame,
                            int K, float fs, float fc,
                            Config::Smoothing mode)
{
    if (mode == Config::Smoothing::Off) return;
    const size_t F = per_frame.size();
    if (F < 4) return;

    std::vector<float> tmp(F);
    for (int k = 0; k < K; ++k)
    {
        // Tolerate missing channels in some frames (e.g. when LBS skipped a
        // detection): we leave the missing entries untouched.
        for (size_t f = 0; f < F; ++f)
            tmp[f] = (k < (int)per_frame[f].size()) ? per_frame[f][k] : 0.f;

        if (mode == Config::Smoothing::ZeroPhase) filtfilt_scalar      (tmp, fs, fc);
        else                                       filter_forward_scalar(tmp, fs, fc);

        for (size_t f = 0; f < F; ++f)
            if (k < (int)per_frame[f].size())
                per_frame[f][k] = tmp[f];
    }
}

// Quaternion zero-phase: forward QuatLPF, reverse the sequence, forward
// QuatLPF again, reverse back.  Hemisphere correction inside filter_quat
// handles the sign discontinuity in either direction.
static void filtfilt_quat(std::vector<std::array<float,4>>& qs, float fs, float fc,
                          Config::Smoothing mode)
{
    if (mode == Config::Smoothing::Off || qs.size() < 4) return;
    auto run_forward = [&]() {
        QuatLPF f{};
        init_quat_lpf(&f, fs, fc);
        for (auto& q : qs) {
            float in[4]  = { q[0], q[1], q[2], q[3] };
            float out[4];
            filter_quat(&f, in, 0.f, out);   // no outlier clamp during offline smoothing
            for (int k = 0; k < 4; ++k) q[k] = out[k];
        }
    };
    run_forward();
    if (mode == Config::Smoothing::ZeroPhase) {
        std::reverse(qs.begin(), qs.end());
        run_forward();
        std::reverse(qs.begin(), qs.end());
    }
}

// Helper: smooth one contiguous segment of a track between scene cuts.
// `idx_lo..idx_hi-1` is the half-open range of indices into `frame_keys`.
static void smooth_segment(const std::vector<int>& frame_keys,
                           size_t idx_lo, size_t idx_hi,
                           const Track& track,
                           std::vector<FrameRecord>& frames,
                           float fs, const Config& cfg)
{
    if (idx_hi - idx_lo < 4) return;   // too short — filter would barely do anything

    const size_t F = idx_hi - idx_lo;
    std::vector<std::vector<float>> cam_t(F, std::vector<float>(3));
    std::vector<std::vector<float>> kp3d (F);
    std::vector<std::vector<float>> bpose(F);
    std::vector<std::vector<float>> hpose(F);
    std::vector<std::vector<float>> mhrp (F);
    std::vector<std::array<float,4>> grot(F);

    for (size_t i = 0; i < F; ++i) {
        int fi  = frame_keys[idx_lo + i];
        int dii = track.frame_to_det.at(fi);
        const auto& r = frames[fi].detections[dii];
        for (int k = 0; k < 3; ++k) cam_t[i][k] = r.pred_cam_t[k];
        kp3d [i] = r.keypoints_3d;
        bpose[i] = r.body_pose;
        hpose[i] = r.hand_pose;
        mhrp [i].assign(r.mhr_model_params.begin(), r.mhr_model_params.end());
        float q[4];
        euler_zyx_to_quat(r.global_rot[0], r.global_rot[1], r.global_rot[2], q);
        grot[i] = { q[0], q[1], q[2], q[3] };
    }

    filter_channels(cam_t, 3,    fs, cfg.bw_cutoff, cfg.smoothing);
    filter_channels(kp3d,  70*3, fs, cfg.bw_cutoff, cfg.smoothing);
    filter_channels(bpose, 133,  fs, cfg.bw_cutoff, cfg.smoothing);
    filter_channels(hpose, 108,  fs, cfg.bw_cutoff, cfg.smoothing);
    filter_channels(mhrp,  204,  fs, cfg.bw_cutoff, cfg.smoothing);
    filtfilt_quat  (grot,        fs, cfg.bw_cutoff, cfg.smoothing);

    for (size_t i = 0; i < F; ++i) {
        int fi  = frame_keys[idx_lo + i];
        int dii = track.frame_to_det.at(fi);
        auto& r = frames[fi].detections[dii];
        for (int k = 0; k < 3; ++k) r.pred_cam_t[k] = cam_t[i][k];
        if (r.keypoints_3d.size() == kp3d [i].size()) r.keypoints_3d = kp3d [i];
        if (r.body_pose    .size() == bpose[i].size()) r.body_pose    = bpose[i];
        if (r.hand_pose    .size() == hpose[i].size()) r.hand_pose    = hpose[i];
        for (int k = 0; k < 204 && k < (int)mhrp[i].size(); ++k)
            r.mhr_model_params[k] = mhrp[i][k];
        quat_to_euler_zyx(grot[i].data(),
                          &r.global_rot[0], &r.global_rot[1], &r.global_rot[2]);
    }
}

static void smoothing_pass(std::vector<FrameRecord>& frames,
                           const std::vector<Track>& tracks,
                           const std::vector<int>& scene_cuts,
                           float fs, const Config& cfg)
{
    if (cfg.smoothing == Config::Smoothing::Off) {
        printf("[pass5] smoothing disabled (--smoothing off)\n");
        return;
    }
    const char* mode_name = (cfg.smoothing == Config::Smoothing::ZeroPhase)
                              ? "zero-phase (forward+backward)"
                              : "forward only";
    printf("[pass5] smoothing per track — %s, %.1f Hz cutoff%s\n",
           mode_name, cfg.bw_cutoff,
           scene_cuts.empty() ? "" : " (per-shot segmented)");

    int n_segments = 0;
    for (const auto& track : tracks)
    {
        const size_t F = track.frame_to_det.size();
        if (F < 4) continue;

        std::vector<int> frame_keys; frame_keys.reserve(F);
        for (auto& kv : track.frame_to_det) frame_keys.push_back(kv.first);

        // Split this track's index range at scene cuts.  We never blend
        // pose data from shot N into shot N+1: that would smear the person
        // from the old shot into the new one.
        size_t seg_start = 0;
        for (size_t i = 1; i < F; ++i) {
            if (cut_between(scene_cuts, frame_keys[i-1], frame_keys[i])) {
                smooth_segment(frame_keys, seg_start, i, track, frames, fs, cfg);
                seg_start = i;
                ++n_segments;
            }
        }
        smooth_segment(frame_keys, seg_start, F, track, frames, fs, cfg);
        ++n_segments;
    }
    printf("[pass5]   smoothed %d segment(s) across all tracks\n", n_segments);
}

// ════════════════════════════════════════════════════════════════════════════
//  PASS 6 — BVH EXPORT
// ════════════════════════════════════════════════════════════════════════════
//  Walk frames in chronological order.  For every frame, build the parallel
//  vectors of (results, track_ids) for tracks that have a detection in this
//  frame, and a pad_ids list for tracks that are alive but missing.  Hand
//  both to BVHWriter::write_frame_external — same writer the live binaries
//  use, just with externally-assigned IDs.
// ════════════════════════════════════════════════════════════════════════════

static void export_to_bvh(const std::vector<FrameRecord>& frames,
                          const std::vector<Track>& tracks,
                          double fps, const Config& cfg)
{
    BVHWriter w;
    if (!w.open(cfg.bvh_template, cfg.bvh_path, 1.0f / (float)fps, cfg.lbs_path,
                cfg.bvh_body_shape_change, cfg.bvh_hand_shape_change,
                cfg.bvh_compensate_finger_endsites))
    {
        fprintf(stderr, "[pass6] BVHWriter::open failed — aborting export\n");
        return;
    }
    printf("[pass6] writing BVH (%.2f fps timeline) …\n", fps);

    // Per-track scratch: for each frame, is the track present?  Lets us emit
    // pad_ids without re-scanning the full session map.
    struct TrackState { int first; int last; const std::map<int,int>* fr_to_det; };
    std::map<int, TrackState> ts;
    for (const auto& t : tracks)
        ts[t.id] = { t.first_frame, t.last_frame, &t.frame_to_det };

    for (int f = 0; f < (int)frames.size(); ++f)
    {
        std::vector<fsb::MHRResult> results;
        std::vector<int>            ids;
        std::vector<int>            pad_ids;

        for (const auto& [id, st] : ts)
        {
            if (f < st.first || f > st.last) continue;   // track inactive
            auto it = st.fr_to_det->find(f);
            if (it != st.fr_to_det->end()) {
                results.push_back(frames[f].detections[it->second]);
                ids.push_back(id);
            } else {
                // Alive in [first,last] but no detection this frame — pad.
                pad_ids.push_back(id);
            }
        }
        w.write_frame_external(results, ids, pad_ids);
    }
    w.close();
}

// ════════════════════════════════════════════════════════════════════════════
//  main
// ════════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv)
{
    Config cfg;
    if (!parse_args(argc, argv, cfg)) return 1;

    // Load pipeline.  No GL window, no MHR mesh — we just need MHR params.
    fsb::Pipeline pipeline;
    {
        fsb::PipelineConfig pcfg;
        apply_common_to_pipeline_cfg(cfg, pcfg);
        pcfg.skip_body_model = false;  // we need keypoints_3d for jitter detection
        if (!pipeline.load(pcfg)) {
            fprintf(stderr, "Failed to load pipeline\n");
            return 1;
        }
    }

    // Also load the LBS struct ourselves so we can re-apply the hand-PCA
    // patch.  The live render binary does the same to fix mhr_model_params
    // arm slots when the pipeline runs with skip_body_model=true.  We DON'T
    // skip the body model here, so apply_hand_pose was already invoked
    // inside the pipeline (fast_sam_3dbody.cpp:945) and we don't need to
    // re-apply.  No-op, but kept here for symmetry / future use.

    cv::VideoCapture cap(cfg.from);
    if (!cap.isOpened()) {
        fprintf(stderr, "could not open video '%s'\n", cfg.from.c_str());
        return 1;
    }

    // PASS 1 — inference (decodes video, runs MHR, detects scene cuts) ---
    std::vector<FrameRecord> frames;
    std::vector<int>         scene_cuts;
    double fps = 30.0;
    if (!run_inference_pass(pipeline, cap, frames, scene_cuts, fps, cfg)) {
        fprintf(stderr, "no frames decoded; nothing to export\n");
        return 1;
    }
    cap.release();

    // PASS 2 — global tracking --------------------------------------------
    auto tracks = build_global_tracks(frames, cfg);

    // PASS 3 — gap interpolation (on by default; respects scene cuts) ----
    // Synthesise interpolated detections for any frame that's inside a
    // track's lifespan but missing in track.frame_to_det.  Without this,
    // the BVH writer would pad those frames with the previous pose
    // (visible as a frozen segment) — see the function's comment block.
    gap_interpolation_pass(frames, tracks, scene_cuts, cfg);

    // PASS 4 — jitter interpolation (opt-in; respects scene cuts) -------
    interpolate_jitter_pass(frames, tracks, scene_cuts, cfg);

    // PASS 5 — smoothing (per scene segment) ----------------------------
    smoothing_pass(frames, tracks, scene_cuts, (float)fps, cfg);

    // PASS 6 — BVH export -----------------------------------------------
    export_to_bvh(frames, tracks, fps, cfg);

    printf("done.\n");
    return 0;
}
