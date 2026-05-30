#pragma once
// ════════════════════════════════════════════════════════════════════════════
//  offline_passes.h
//
//  The offline BVH pipeline as a callable library.  Each "pass" used to be a
//  static function inside render/offline_sam_3dbody_render.cpp; they are
//  extracted here so other binaries (e.g. the multi-view fusion tool — see
//  MULTIVIEW_PLAN.md, which reuses Pass 1 inference + Pass 2 tracking per
//  stream) can drive the same stages without copy-pasting orchestration.
//
//  The binary render/offline_sam_3dbody_render.cpp is now a thin wrapper:
//  CLI parsing (its own print_usage / parse_args) + main() that calls these
//  passes in order.
//
//  Everything lives in namespace `offline` to keep the very generic `Config`
//  name out of the global scope (each binary still has its own local Config).
// ════════════════════════════════════════════════════════════════════════════

#include <array>
#include <chrono>
#include <map>
#include <string>
#include <vector>

#include <opencv2/videoio.hpp>      // cv::VideoCapture

#include "fast_sam_3dbody.h"        // fsb::MHRResult, fsb::Pipeline
#include "cli_common.h"             // CommonConfig

namespace offline
{

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

// Offline pipeline configuration.  Inherits the common subset from
// CommonConfig (cli_common.h):
//   --onnx-dir --gguf --yolo --from --cuda --trt --no-fp16 --thresh --nms
//   --bvh --bvh-template --no-bvh-*-shape-change --bvh-raw-fingers
//   --bw-cutoff --rot-clamp
// Offline-specific knobs (smoothing / tracking / scene / gap / jitter) below.
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

    // Process only a frame window: seek to start_frame, then stop after
    // max_frames decoded (0 = whole video).  Used by the multi-view front-end
    // to align per-stream inference to a common wall-clock window.
    int       start_frame          = 0;
    int       max_frames           = 0;

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
    // ViT signal: cosine similarity of the whole-frame backbone embedding
    // between consecutive frames.  A drop below scene_vit_thresh is a strong
    // semantic cut signal that counts double in the detector's vote.
    bool      scene_use_vit        = true;
    float     scene_vit_thresh     = 0.60f;
    // ViT veto: if the whole-frame embedding cosine stays this high the scene
    // is demonstrably unchanged, so suppress a cut even when the heuristics
    // (corner/hist/person) voted for one — they over-fire on fast-camera
    // game/action footage with a stable palette.  Set to 1.0 to disable veto.
    float     scene_vit_veto_thresh = 0.90f;

    // Split the output into one set of BVH files per detected scene, with
    // people re-indexed locally within each scene:
    //   <stem>_scene<S>_person<P>.bvh   (default: one <stem>_<id>.bvh per track
    //   spanning the whole clip).  Requires scene detection.
    bool      bvh_split_scenes      = false;

    // bvh_body_shape_change / bvh_hand_shape_change /
    // bvh_compensate_finger_endsites all come from CommonConfig now.
};

// ─── Passes ──────────────────────────────────────────────────────────────────
// Each operates on the shared FrameRecord/Track buffers; call them in order.

// PASS 1 — decode video + run MHR inference + detect scene cuts.
bool run_inference_pass(fsb::Pipeline& pipeline,
                        cv::VideoCapture& cap,
                        std::vector<FrameRecord>& out_frames,
                        std::vector<int>& out_scene_cuts,
                        double& out_fps,
                        const Config& cfg);

// PASS 2 — globally-optimal identity tracking across the whole clip.
std::vector<Track> build_global_tracks(std::vector<FrameRecord>& frames,
                                        const Config& cfg);

// PASS 3 — fill missing frames inside each track (respects scene cuts).
void gap_interpolation_pass(std::vector<FrameRecord>& frames,
                            std::vector<Track>& tracks,
                            const std::vector<int>& scene_cuts,
                            const Config& cfg);

// PASS 4 — replace high-velocity (jitter) frames by interpolation (opt-in).
void interpolate_jitter_pass(std::vector<FrameRecord>& frames,
                             const std::vector<Track>& tracks,
                             const std::vector<int>& scene_cuts,
                             const Config& cfg);

// PASS 5 — temporal smoothing per scene segment.
void smoothing_pass(std::vector<FrameRecord>& frames,
                    const std::vector<Track>& tracks,
                    const std::vector<int>& scene_cuts,
                    float fs, const Config& cfg);

// PASS 6 — write BVH file(s).
void export_to_bvh(const std::vector<FrameRecord>& frames,
                   const std::vector<Track>& tracks,
                   const std::vector<int>& scene_cuts,
                   double fps, const Config& cfg);

// Linear/SLERP interpolation of two MHR results (used by Pass 3/4; exposed
// because the multi-view refit reuses it — see MULTIVIEW_PLAN.md).
fsb::MHRResult interp_mhr(const fsb::MHRResult& a,
                          const fsb::MHRResult& b,
                          float t);

} // namespace offline
