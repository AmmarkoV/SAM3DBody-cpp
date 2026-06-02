#pragma once
// ============================================================================
// fast_sam_3dbody_capi.h  –  Plain C API for ctypes / cffi access
// ============================================================================

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque pipeline handle
typedef void* FsbHandle;

// ── Config passed from Python ─────────────────────────────────────────────────
typedef struct {
    const char* onnx_dir;
    const char* gguf_path;
    const char* yolo_path;
    int   cuda_device;      // -1 = CPU
    int   skip_body_model;  // 0/1
    float person_thresh;
    float person_nms_iou;
    int   max_persons;      // 0 = unlimited
    float focal_x;
    float focal_y;
    float principal_x;
    float principal_y;
    int   zero_face_params;   // 0/1  — force face expression coefficients to zero
    int   detector;           // bbox provider: 0 = YOLO11-pose (default), 1 = LibreYOLO (YOLOv9 bbox)
} FsbConfig;

// ── Per-person result (fixed-size for easy ctypes mapping) ────────────────────
typedef struct {
    float bbox[4];          // x1, y1, x2, y2  (original image pixels)
    float focal_length;
    float pred_cam_t[3];    // tx, ty, tz
    float global_rot[3];    // Euler ZYX

    float body_pose[133];
    float shape[45];
    float scale[28];
    float hand_pose[108];
    float face_params[72];

    // 2-D YOLO keypoints: 17 COCO joints × [x, y, confidence]
    float yolo_kps[51];     // layout: [kp0_x, kp0_y, kp0_vis, kp1_x, ...]
    int   has_yolo_kps;     // 1 if YOLO ran and detected this person

    // 3-D keypoints (from body model; 0 when skip_body_model=1)
    float kps_3d[210];      // [70 × 3]
    float kps_2d[140];      // [70 × 2]
    int   has_kps;

    // ── Second-pass fields (added for --two-passes Python second decoder pass) ──
    //
    // pred_pose_raw: the raw MHR FFN output BEFORE Euler conversion.
    //   Layout (mirrors fast_sam_3dbody.cpp parse block ≈ line 755):
    //     [0:6]    global_rot_6d (6D continuous rotation, Zhou et al. 2019)
    //     [6:266]  body_cont[260] (23×6D + 58×sincos + 6trans)
    //   Together these 266 floats form the first part of the second-pass
    //   prev_estimate tensor: cat(pred_pose_raw, shape, scale, hand, face).
    //   Shape[45], scale[28], hand[108], face[72] are already in the fields above.
    //
    // pred_cam_raw: raw camera head FFN output [3] before the nonlinear
    //   s/tx/ty → pred_cam_t conversion.  Appended to prev_estimate when the
    //   loaded Python model has an init_camera attribute.
    //
    // IMPORTANT: these fields are appended at the END of FsbResult so that the
    // ctypes struct layout for older code is not disturbed.
    float pred_pose_raw[266];  // global_rot_6d[6] + body_cont[260]
    float pred_cam_raw[3];     // raw cam head output before s/tx/ty decode

    // Assembled model_params[204] used by native C LBS (hand + scale already decoded).
    // Layout: [0:3]=global_trans*10, [3:6]=global_rot ZYX, [6:136]=body_pose[:130],
    //         [136:204]=scale_out.  Mirrors Python mhr_forward(..., return_model_params=True).
    float mhr_model_params[204];
} FsbResult;

// ── Lifecycle ─────────────────────────────────────────────────────────────────
FsbHandle fsb_create(void);
void      fsb_destroy(FsbHandle h);

// Returns 1 on success, 0 on failure.
int fsb_load(FsbHandle h, const FsbConfig* cfg);

// ── Inference ─────────────────────────────────────────────────────────────────
// Process a BGR uint8 image.
// results    : pre-allocated array of FsbResult with at least max_results entries.
// max_results: capacity of results[].
// Returns number of persons written (≤ max_results).
int fsb_process_bgr(FsbHandle        h,
                    const uint8_t*   bgr,
                    int              width,
                    int              height,
                    FsbResult*       results,
                    int              max_results);

#ifdef __cplusplus
}
#endif
