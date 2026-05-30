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

// NOTE: the pipeline passes (FrameRecord/Track/Config + the six passes) now
// live in src/offline_passes.{h,cpp} so the multi-view tool can reuse them
// (PLAN.md §6 / MULTIVIEW_PLAN.md).  This file is just CLI + orchestration.

#include "../src/offline_passes.h"
#include "../src/fast_sam_3dbody.h"
#include "../src/cli_common.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

using namespace offline;

static void print_usage(const char* prog)
{
    printf(
        "Usage: %s --onnx-dir DIR --gguf F.gguf --yolo F.onnx --from VIDEO.mp4 --bvh OUT.bvh [options]\n"
        "\n"
        "REQUIRED\n"
        "  --onnx-dir PATH            Directory with backbone / decoder / body_model ONNX files\n"
        "  --backbone NAME            Backbone filename in onnx-dir (default backbone.onnx;\n"
        "                             use backbone_int8.onnx after tools/quantize_backbone.py)\n"
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
        "  --no-scene-vit             Disable the ViT whole-frame embedding signal (on by default).\n"
        "  --scene-vit-threshold T      Cut when frame-to-frame backbone-embedding cosine drops below\n"
        "                             T (default 0.60).  The ViT signal counts double in the vote.\n"
        "  --scene-vit-veto-threshold T Suppress a cut when the embedding cosine stays >= T (default\n"
        "                             0.90) — the scene is unchanged, so the heuristics over-fired.\n"
        "                             Set to 1.0 to disable the veto.\n"
        "  --bvh-split-scenes         Write one set of BVH files per detected scene, people\n"
        "                             re-indexed per scene: <stem>_scene<S>_person<P>.bvh\n"
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
        A1("--scene-vit-threshold",     scene_vit_thresh,    std::stof)
        A1("--scene-vit-veto-threshold", scene_vit_veto_thresh, std::stof)
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
        if (!strcmp(argv[i], "--no-scene-vit"))           { c.scene_use_vit = false; continue; }
        if (!strcmp(argv[i], "--bvh-split-scenes"))       { c.bvh_split_scenes = true; continue; }
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
    export_to_bvh(frames, tracks, scene_cuts, fps, cfg);

    printf("done.\n");
    return 0;
}
