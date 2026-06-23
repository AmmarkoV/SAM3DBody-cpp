#pragma once
// ============================================================================
// fast_sam_3dbody.h  –  C++ interface for SAM-3D-Body pipeline
//
// Pipeline stages
// ───────────────
//  1. YOLO Pose (ONNX / TRT engine)      → person bounding boxes
//  2. Backbone (backbone.onnx)            → image feature map  [B,1280,32,32]
//  3. Decoder  (decoder.onnx)             → pose token         [B,1024]
//  4. MHR head  (pipeline.gguf, ggml)    → raw pose params     [B,519]
//  5. Camera head (pipeline.gguf, ggml)  → camera params       [B,3]
//  6. Body model  (body_model.onnx)       → vertices + joints
//
// Inputs expected in BGR uint8 (OpenCV default).
// ============================================================================

#include <array>
#include <string>
#include <vector>
#include <cstdint>

namespace fsb {

// ─── Output per detected person ──────────────────────────────────────────────
struct MHRResult {
    // Bounding box in original image  [x1, y1, x2, y2]
    std::array<float, 4> bbox{};

    float focal_length = 0.f;          // Estimated / default focal length (pixels)

    // Camera translation  [tx, ty, tz]
    std::array<float, 3> pred_cam_t{};

    // ── Pose params ──────────────────────────────────────────────────────────
    // Global orientation – Euler ZYX  [rx, ry, rz]
    std::array<float, 3> global_rot{};

    // Body pose – MHR 133-dim Euler angles
    std::vector<float> body_pose;      // [133]

    // Shape betas  (SMPL-like identity blend shapes)
    std::vector<float> shape;          // [45]

    // Scale parameters
    std::vector<float> scale;          // [28]

    // Hand pose (left 54 + right 54 = 108)
    std::vector<float> hand_pose;      // [108]

    // Face expression
    std::vector<float> face_params;    // [72]

    // Raw model params fed to the LBS pipeline  [204]
    std::array<float, 204> mhr_model_params{};

    // ── Geometry (populated when Pipeline::Config::skip_body_model = false) ──
    std::vector<float> pred_vertices;  // [18439 × 3]  SMPL-like mesh
    std::vector<float> keypoints_3d;   // [70 × 3]     3-D joints
    std::vector<float> keypoints_2d;   // [70 × 2]     projected 2-D

    // Full MHR skeleton joint positions (see src/mhr_joint_table.h for names/order)
    // [127 × 3], same camera coordinate system as keypoints_3d (y,z negated, metres).
    std::vector<float> skeleton_3d;    // [127 × 3]

    // 2-D YOLO keypoints: 17 COCO joints × [x, y, confidence], image pixel coords
    std::vector<float> keypoints_yolo; // [17 × 3]   always populated if YOLO ran

    // ── Second-pass fields ────────────────────────────────────────────────────
    // Raw MHR FFN output before Euler conversion.
    //   [0:6]   global_rot_6d (6D continuous rotation)
    //   [6:266] body_cont[260] (23×6D + 58×sincos + 6trans)
    // Used by the Python second-pass to build prev_estimate for forward_decoder.
    std::array<float, 266> pred_pose_raw{};

    // Raw camera head FFN output [s, tx, ty] before the nonlinear
    // s/tx/ty → world-space pred_cam_t conversion.
    std::array<float, 3> pred_cam_raw{};
};

// ─── Pipeline configuration ───────────────────────────────────────────────────
struct PipelineConfig {
    // Paths
    std::string onnx_dir;           // Directory with backbone.onnx, decoder.onnx, body_model.onnx
    std::string backbone_name = "backbone.onnx"; // filename within onnx_dir; override for quantized variant
    std::string decoder_name  = "decoder.onnx";  // filename within onnx_dir; resolver swaps in decoder_fp16.onnx under --trt
    std::string gguf_path;          // Path to pipeline.gguf
    std::string yolo_path;          // YOLO model: .onnx or .engine (TRT)

    // Device
    int  cuda_device    = 0;        // CUDA device (-1 = CPU only)
    bool use_trt_ep     = false;    // Enable ONNX Runtime TensorRT EP (requires TRT install)
    bool use_fp16       = true;     // FP16 for ONNX EP

    // Inference options
    bool skip_body_model = false;   // Skip body model – no vertices/keypoints (faster)
    float person_thresh  = 0.50f;  // YOLO confidence threshold
    float person_nms_iou = 0.45f;  // YOLO NMS IoU threshold
    int  max_persons     = 0;      // 0 = unlimited; >0 = cap after NMS (top-N by conf)

    // Bounding-box detector provider (selects how the --yolo ONNX output is parsed).
    // Kept as an int for the C-ABI-friendly struct style. Extend with new kinds
    // (e.g. DET_YMAPNET) by adding an enum value, a parser, and a dispatch case.
    enum DetectorKind { DET_YOLO_POSE = 0, DET_LIBREYOLO = 1 /*, DET_YMAPNET = 2 */ };
    int  detector        = DET_YOLO_POSE; // 0 = YOLO11-pose (default), 1 = LibreYOLO (YOLOv9 bbox)

    // Camera intrinsics – set to 0 to use default (fx = image_width)
    float focal_x = 0.f;
    float focal_y = 0.f;
    float principal_x = 0.f;       // 0 = image_width  / 2
    float principal_y = 0.f;       // 0 = image_height / 2

    // Debug / diagnostic flags
    bool zero_face_params = true;   // Force face expression coefficients to 0 (default on; pass --dev-face to enable)
};

// ─── Pipeline class ───────────────────────────────────────────────────────────
class Pipeline {
public:
    Pipeline();
    ~Pipeline();

    // Non-copyable, moveable
    Pipeline(const Pipeline&)            = delete;
    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline(Pipeline&&)                 = default;
    Pipeline& operator=(Pipeline&&)      = default;

    // Load all models.  Returns false on failure.
    bool load(const PipelineConfig& cfg);

    // Release all resources.
    void free();

    // ── Core inference ────────────────────────────────────────────────────────
    // Process a single BGR image (width × height × 3, uint8).
    // Returns one MHRResult per detected person.
    std::vector<MHRResult> process_bgr(const uint8_t* bgr,
                                       int width, int height);

    // Convenience overload for OpenCV Mat (must be CV_8UC3 BGR).
    // Declared only if OpenCV is available; implemented in fast_sam_3dbody.cpp.
    struct cv_mat_tag {};
#if defined(FSB_HAS_OPENCV_MAT)
    std::vector<MHRResult> process_mat(const void* cv_mat_ptr);
#endif

    // ── Whole-frame ViT scene embedding ──────────────────────────────────────
    // Runs the backbone on the *entire* image (resized to the backbone's
    // 512×512 input), global-average-pools the [1280,32,32] feature map over
    // the spatial grid and L2-normalises the result.  Returns a 1280-d unit
    // vector whose cosine similarity to the previous frame's embedding is a
    // robust, semantic scene-cut signal (used by the offline detector).
    // Returns an empty vector if the backbone session isn't available.
    std::vector<float> scene_embedding(const uint8_t* bgr, int width, int height);

    // True after a successful load().
    bool is_loaded() const;

    // Print loaded model info.
    void print_info() const;

    // Print a one-line per-stage timing summary (per-frame averages of detection,
    // preprocess, backbone, decoder, MHR FFN and body model), plus the number of
    // frames and person crops processed across the lifetime of this Pipeline.
    void print_timing_summary() const;

private:
    struct Impl;
    Impl* impl_ = nullptr;
};

} // namespace fsb
