#pragma once
// ============================================================================
// preprocess.hpp  –  per-person crop, normalisation, ray-condition,
//                    CLIFF condition, YOLO NMS helpers
// ============================================================================
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <cstring>
#include <vector>

namespace fsb {

// ─── image normalisation constants (from model_config.yaml) ──────────────────
static constexpr float IMAGE_MEAN[3]    = {0.485f, 0.456f, 0.406f};
static constexpr float IMAGE_STD[3]     = {0.229f, 0.224f, 0.225f};
static constexpr int   CROP_SIZE        = 512;
static constexpr int   FEAT_HW          = CROP_SIZE / 16;  // 32 – patch grid size
static constexpr int   PATCH_SIZE       = 16;
// BBoxScale padding factor – matches Python transforms BBoxScale(padding=1.25).
// The crop is expanded by this factor, and condition_info[2] = (bbox_size*1.25) / focal.
static constexpr float BBOX_SCALE_FACTOR = 1.25f;

// ─── Crop one person out of a BGR image and return normalised CHW float32 ─────
//
// bbox_x1/y1/x2/y2 : person bounding box in original image (float, unclamped)
// out_chw           : pre-allocated float[3 × CROP_SIZE × CROP_SIZE]
//
// The crop is a square centred on the bbox, padded with grey if needed.
// Normalised with IMAGE_MEAN / IMAGE_STD (RGB channel order).
//
// Also fills:
//   crop_cx, crop_cy       – bbox centre used for this crop
//   crop_size              – side length of the square crop (in original pixels)
inline void crop_and_normalise(
    const cv::Mat& bgr,              // full image [H,W,3] CV_8UC3
    float  bbox_x1, float bbox_y1,
    float  bbox_x2, float bbox_y2,
    float* out_chw,                  // [3, CROP_SIZE, CROP_SIZE]
    float& crop_cx, float& crop_cy,  // outputs: crop centre
    float& crop_size_out             // output: square side in source pixels
)
{
    const int img_w = bgr.cols;
    const int img_h = bgr.rows;

    // square crop centred on bbox
    float cx   = (bbox_x1 + bbox_x2) * 0.5f;
    float cy   = (bbox_y1 + bbox_y2) * 0.5f;
    float bw   = bbox_x2 - bbox_x1;
    float bh   = bbox_y2 - bbox_y1;
    // Expand by BBOX_SCALE_FACTOR (1.25) to match Python BBoxScale(padding=1.25).
    float side = std::max(bw, bh) * BBOX_SCALE_FACTOR;

    crop_cx      = cx;
    crop_cy      = cy;
    crop_size_out= side;

    int x1 = static_cast<int>(std::round(cx - side * 0.5f));
    int y1 = static_cast<int>(std::round(cy - side * 0.5f));
    int x2 = x1 + static_cast<int>(std::round(side));
    int y2 = y1 + static_cast<int>(std::round(side));

    // Clip and compute padding
    int pad_l = std::max(0, -x1);
    int pad_t = std::max(0, -y1);

    int sx1 = std::max(0, x1), sy1 = std::max(0, y1);
    int sx2 = std::min(img_w, x2), sy2 = std::min(img_h, y2);

    int roi_w = sx2 - sx1;
    int roi_h = sy2 - sy1;

    // Create padded square (114 grey = common YOLO convention)
    cv::Mat padded(y2 - y1, x2 - x1, CV_8UC3, cv::Scalar(114, 114, 114));
    if (roi_w > 0 && roi_h > 0) {
        bgr(cv::Rect(sx1, sy1, roi_w, roi_h))
            .copyTo(padded(cv::Rect(pad_l, pad_t, roi_w, roi_h)));
    }

    // Resize to CROP_SIZE × CROP_SIZE
    cv::Mat resized;
    cv::resize(padded, resized, {CROP_SIZE, CROP_SIZE}, 0, 0, cv::INTER_LINEAR);

    // BGR→RGB, uint8→float32 normalised, interleaved→CHW
    const int plane = CROP_SIZE * CROP_SIZE;
    for (int y = 0; y < CROP_SIZE; ++y) {
        const uchar* row = resized.ptr<uchar>(y);
        for (int x = 0; x < CROP_SIZE; ++x) {
            // OpenCV is BGR
            float b = row[3*x + 0] / 255.f;
            float g = row[3*x + 1] / 255.f;
            float r = row[3*x + 2] / 255.f;
            // normalise (RGB order matches PyTorch model)
            out_chw[0 * plane + y * CROP_SIZE + x] = (r - IMAGE_MEAN[0]) / IMAGE_STD[0];
            out_chw[1 * plane + y * CROP_SIZE + x] = (g - IMAGE_MEAN[1]) / IMAGE_STD[1];
            out_chw[2 * plane + y * CROP_SIZE + x] = (b - IMAGE_MEAN[2]) / IMAGE_STD[2];
        }
    }
}

// ─── Compute CLIFF condition info ─────────────────────────────────────────────
//
// CLIFF-style condition (USE_INTRIN_CENTER=true in config):
//   cond[0] = (bbox_cx - cam_cx) / focal_x
//   cond[1] = (bbox_cy - cam_cy) / focal_y
//   cond[2] = bbox_size           / focal_x
//
inline void compute_condition_info(
    float bbox_cx, float bbox_cy, float bbox_size,
    float focal_x, float focal_y,
    float cam_cx,  float cam_cy,
    float cond[3]   // output [3]
)
{
    cond[0] = (bbox_cx - cam_cx) / focal_x;
    cond[1] = (bbox_cy - cam_cy) / focal_y;
    cond[2] = bbox_size          / focal_x;
}

// ─── Compute ray_cond map  [2, FEAT_HW, FEAT_HW]  at patch resolution ────────
//
// The ONNX decoder expects ray directions at feature-map resolution (32×32),
// so we sample at each patch centre instead of per-pixel.
//
// Patch centre (px, py) in the 512×512 crop:
//   crop_x = px * PATCH_SIZE + PATCH_SIZE/2
//   crop_y = py * PATCH_SIZE + PATCH_SIZE/2
//
// Back-projected to original image then to normalised camera ray:
//   orig_x = (crop_x - CROP_SIZE/2) / scale + bbox_cx
//   ray_x  = (orig_x - cam_cx) / focal_x
//
// out_ray: float[2 × FEAT_HW × FEAT_HW]  layout [channel, y, x]
// channel 0 = ray_x,  channel 1 = ray_y
inline void compute_ray_cond(
    float bbox_cx, float bbox_cy, float crop_size_orig,
    float focal_x, float focal_y,
    float cam_cx,  float cam_cy,
    float* out_ray   // [2, FEAT_HW, FEAT_HW]
)
{
    const float scale   = static_cast<float>(CROP_SIZE) / crop_size_orig;
    const float half_cs = CROP_SIZE * 0.5f;
    const int   FHW     = FEAT_HW;
    const int   plane   = FHW * FHW;

    for (int py = 0; py < FHW; ++py) {
        for (int px = 0; px < FHW; ++px) {
            float crop_x = px * PATCH_SIZE + PATCH_SIZE * 0.5f;
            float crop_y = py * PATCH_SIZE + PATCH_SIZE * 0.5f;
            float orig_x = (crop_x - half_cs) / scale + bbox_cx;
            float orig_y = (crop_y - half_cs) / scale + bbox_cy;
            out_ray[0 * plane + py * FHW + px] = (orig_x - cam_cx) / focal_x;
            out_ray[1 * plane + py * FHW + px] = (orig_y - cam_cy) / focal_y;
        }
    }
}

// ─── YOLO output parsing & NMS ────────────────────────────────────────────────
//
// YOLO Pose output: [1, num_dets, 56]
//   columns 0-3  : cx, cy, w, h  (normalised 0..1)
//   column  4    : object confidence
//   columns 5-54 : class scores then keypoints (not used here)
//
struct PersonDet {
    float x1, y1, x2, y2;  // pixel coords
    float conf;
    // 17 COCO keypoints: [x, y, vis] each, in YOLO pixel space (0–640 before scale)
    float kps[51] = {};
    bool  has_kps = false;
};

static inline float iou(const PersonDet& a, const PersonDet& b) {
    float ix1 = std::max(a.x1, b.x1);
    float iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2);
    float iy2 = std::min(a.y2, b.y2);
    float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
    if (inter == 0.f) return 0.f;
    float ua = (a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - inter;
    return inter / (ua + 1e-6f);
}

// Parse YOLO Pose output tensor [num_dets, 56] (already transposed to row-major).
// Ultralytics ONNX export outputs cx,cy,w,h in YOLO input pixel coords (0-640).
// Caller scales to original image space via sx/sy after this call.
inline std::vector<PersonDet> parse_yolo_output(
    const float*  data,          // [num_dets × 56]
    int           num_dets,
    float         conf_thresh,
    float         nms_iou_thresh
)
{
    std::vector<PersonDet> raw;
    raw.reserve(64);

    for (int i = 0; i < num_dets; ++i) {
        const float* row = data + i * 56;
        float cx   = row[0], cy = row[1], w = row[2], h = row[3];
        float conf = row[4];
        if (conf < conf_thresh) continue;
        PersonDet d;
        d.x1   = cx - w * 0.5f;   // YOLO pixel space (0-640)
        d.y1   = cy - h * 0.5f;
        d.x2   = cx + w * 0.5f;
        d.y2   = cy + h * 0.5f;
        d.conf = conf;
        // Keypoints: columns 5..55 → 17 × (x, y, visibility)
        if (num_dets > 0 && 56 > 5) {
            std::memcpy(d.kps, row + 5, 51 * sizeof(float));
            d.has_kps = true;
        }
        raw.push_back(d);
    }

    // Sort descending by confidence
    std::sort(raw.begin(), raw.end(),
        [](const PersonDet& a, const PersonDet& b){ return a.conf > b.conf; });

    // Greedy NMS
    std::vector<bool> suppressed(raw.size(), false);
    std::vector<PersonDet> kept;
    for (size_t i = 0; i < raw.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(raw[i]);
        for (size_t j = i + 1; j < raw.size(); ++j) {
            if (!suppressed[j] && iou(raw[i], raw[j]) > nms_iou_thresh)
                suppressed[j] = true;
        }
    }
    return kept;
}

// Parse a YOLOv9 (LibreYOLO) detection output tensor [num_dets, num_feat],
// already transposed to row-major. Layout per row:
//   columns 0-3            : cx, cy, w, h  (YOLO input pixel coords, 0-640)
//   columns 4 .. num_feat-1: per-class scores (anchor-free, no objectness)
// COCO person is class 0, so confidence = column 4. A person-only model has
// num_feat == 5 and the same column-4 convention. Bbox-only: no keypoints.
// Caller scales to original image space via the same letterbox reversal used
// for the pose path.
inline std::vector<PersonDet> parse_yolov9_output(
    const float*  data,          // [num_dets × num_feat]
    int           num_dets,
    int           num_feat,      // 4 + num_classes (typically 84)
    float         conf_thresh,
    float         nms_iou_thresh
)
{
    std::vector<PersonDet> raw;
    raw.reserve(64);

    if (num_feat < 5) return raw;   // need at least 4 bbox + 1 class

    for (int i = 0; i < num_dets; ++i) {
        const float* row = data + (size_t)i * num_feat;
        float conf = row[4];                 // class 0 = person
        if (conf < conf_thresh) continue;
        float cx = row[0], cy = row[1], w = row[2], h = row[3];
        PersonDet d;
        d.x1   = cx - w * 0.5f;
        d.y1   = cy - h * 0.5f;
        d.x2   = cx + w * 0.5f;
        d.y2   = cy + h * 0.5f;
        d.conf = conf;
        d.has_kps = false;                   // detection-only: no keypoints
        raw.push_back(d);
    }

    // Sort descending by confidence
    std::sort(raw.begin(), raw.end(),
        [](const PersonDet& a, const PersonDet& b){ return a.conf > b.conf; });

    // Greedy NMS (shares iou() with the pose path)
    std::vector<bool> suppressed(raw.size(), false);
    std::vector<PersonDet> kept;
    for (size_t i = 0; i < raw.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(raw[i]);
        for (size_t j = i + 1; j < raw.size(); ++j) {
            if (!suppressed[j] && iou(raw[i], raw[j]) > nms_iou_thresh)
                suppressed[j] = true;
        }
    }
    return kept;
}

// ─── Convert continuous body params to 133-dim Euler (fast path) ──────────────
//
// Implements compact_cont_to_model_params_body_fast in C++.
// body_cont [260] → body_euler [133]
//
// Body pose parameterisation:
//   - 23 joints with 3 DOF → 23×6 = 138 continuous dims  (6D rotation)
//   - 58 joints with 1 DOF → 58×2 = 116 continuous dims  (sin,cos)
//   - 6  translation values→ 6    continuous dims
//   Total = 138 + 116 + 6 = 260  ✓
//
// Output 133 = 23×3 (euler 3-dof) + 58 (euler 1-dof) + 6 (trans) = 87 + 46 + ... hmm
//   Actually 23*3 = 69 + 58 + 6 = 133  ✓

// 6D rotation → 3D ZYX Euler angles (rx, ry, rz).
//
// Matches Python batchXYZfrom6D (mhr_utils.py) which:
//   1. Treats d6[0:3] as first COLUMN candidate, d6[3:6] as second COLUMN candidate.
//   2. Gram-Schmidt orthonormalises them → columns c0, c1.
//   3. c2 = c0 × c1 (right-hand frame).
//   4. Extracts ZYX Euler from the BOTTOM ROW and LEFT COLUMN of the matrix:
//        rx = atan2(R[2,1], R[2,2])
//        ry = asin(-R[2,0])
//        rz = atan2(R[1,0], R[0,0])
//
// Variable naming: eXY = element at row X of column Y of R.
//   col 0 → [e00, e01, e02] = [R[0,0], R[1,0], R[2,0]]
//   col 1 → [e10, e11, e12] = [R[0,1], R[1,1], R[2,1]]
//   col 2 → [e20, e21, e22] = [R[0,2], R[1,2], R[2,2]]
//
// ZYX extraction uses: R[2,0]=e02, R[2,1]=e12, R[2,2]=e22, R[1,0]=e01, R[0,0]=e00.
// (A previous bug used e20/e21/e10, which are the TRANSPOSED positions and extract
//  the angles for R^T = inverse rotation — all joints bent the wrong way.)
static inline void rot6d_to_euler(const float* d6, float* euler) {
    // ── Step 1: build column 0 (first 3 floats, normalised) ──────────────────
    float a0 = d6[0], a1 = d6[1], a2 = d6[2];
    float b0 = d6[3], b1 = d6[4], b2 = d6[5];

    float na  = std::sqrt(a0*a0 + a1*a1 + a2*a2) + 1e-8f;
    // col 0:  e00=R[0,0]  e01=R[1,0]  e02=R[2,0]
    float e00 = a0/na, e01 = a1/na, e02 = a2/na;

    // ── Step 2: Gram-Schmidt → column 1 ──────────────────────────────────────
    float dot = e00*b0 + e01*b1 + e02*b2;
    // col 1:  e10=R[0,1]  e11=R[1,1]  e12=R[2,1]
    float e10 = b0 - dot*e00;
    float e11 = b1 - dot*e01;
    float e12 = b2 - dot*e02;
    float nb  = std::sqrt(e10*e10 + e11*e11 + e12*e12) + 1e-8f;
    e10 /= nb; e11 /= nb; e12 /= nb;

    // ── Step 3: cross product → column 2 (not needed for angle extraction) ───
    // col 2:  e20=R[0,2]  e21=R[1,2]  e22=R[2,2]
    // e20 = e01*e12 - e02*e11   (unused in extraction)
    // e21 = e02*e10 - e00*e12   (unused in extraction)
    float e22 = e00*e11 - e01*e10;   // R[2,2] = cos(ry)*cos(rx)

    // ── Step 4: ZYX Euler extraction from bottom row and left column ─────────
    // For R = Rz(rz)*Ry(ry)*Rx(rx):
    //   R[2,0] = -sin(ry)                       → e02
    //   R[2,1] =  cos(ry)*sin(rx)               → e12
    //   R[2,2] =  cos(ry)*cos(rx)               → e22
    //   R[1,0] =  sin(rz)*cos(ry)               → e01
    //   R[0,0] =  cos(rz)*cos(ry)               → e00
    euler[0] = std::atan2(e12, e22);  // rx = atan2(R[2,1], R[2,2])
    euler[1] = std::asin(std::max(-1.f, std::min(1.f, -e02)));  // ry = asin(-R[2,0])
    euler[2] = std::atan2(e01, e00);  // rz = atan2(R[1,0], R[0,0])
}

// 3-DOF joint index layout in the 133-param vector
// (mirrors all_param_3dof_rot_idxs in mhr_utils.py)
static constexpr int BODY_3DOF_JOINT_IDXS[23][3] = {
    {0,2,4}, {6,8,10}, {12,13,14}, {15,16,17}, {18,19,20},
    {21,22,23}, {24,25,26}, {27,28,29}, {34,35,36}, {37,38,39},
    {44,45,46}, {53,54,55}, {64,65,66}, {85,69,73}, {86,70,79},
    {87,71,82}, {88,72,76}, {91,92,93}, {112,96,100}, {113,97,106},
    {114,98,109}, {115,99,103}, {130,131,132}
};
static constexpr int BODY_1DOF_IDXS[58] = {
    1,3,5,7,9,11,30,31,32,33,40,41,42,43,47,48,49,50,51,52,
    56,57,58,59,60,61,62,63,67,68,74,75,77,78,80,81,83,84,
    89,90,94,95,101,102,104,105,107,108,110,111,116,117,118,119,120,121,122,123
};
static constexpr int BODY_TRANS_IDXS[6] = {124,125,126,127,128,129};

inline void compact_cont_to_body_params(
    const float* body_cont,  // [260]
    float*       body_euler  // [133] out – caller must zero-initialise
)
{
    static constexpr int N3    = 23;
    static constexpr int N1    = 58;
    // 3-DOF region: first 23*6 = 138 floats
    for (int j = 0; j < N3; ++j) {
        float euler[3];
        rot6d_to_euler(body_cont + j * 6, euler);
        for (int k = 0; k < 3; ++k)
            body_euler[BODY_3DOF_JOINT_IDXS[j][k]] = euler[k];
    }
    // 1-DOF region: next 58*2 = 116 floats  (sin, cos)
    const float* p1 = body_cont + N3 * 6;
    for (int j = 0; j < N1; ++j) {
        float s = p1[j*2 + 0];
        float c = p1[j*2 + 1];
        body_euler[BODY_1DOF_IDXS[j]] = std::atan2(s, c);
    }
    // Translation region: last 6 floats
    const float* pt = body_cont + N3 * 6 + N1 * 2;
    for (int j = 0; j < 6; ++j)
        body_euler[BODY_TRANS_IDXS[j]] = pt[j];
}

// ─── Hand pose decode  (PCA + 6D/atan2 → 27 Euler params per hand) ───────────
//
// Mirrors Python:
//   replace_hands_in_pose(full_pose_params, hand_pose_params)
//     left, right            = split(hand_pose_params [108], [54, 54])
//     for h in (left, right):
//         decoded[54]    = hand_pose_mean + h @ hand_pose_comps
//         params[27]     = compact_cont_to_model_params_hand_fast(decoded)
//         full_pose_params[hand_joint_idxs_{l,r}] = params
//
// Joint DoF table _HAND_DOFS_IN_ORDER (mhr_utils.py):
//   {3,1,1, 3,1,1, 3,1,1, 3,1,1, 2,3,1,1}  (16 joints, sum=27 = N_HAND_OUT)
// 3-DoF joint  → 6 cont → 3 euler via rot6d_to_euler
// 1-DoF joint  → 2 cont → 1 atan2
// 2-DoF joint  → 4 cont → 2 atan2 (treated as two 1-DoF entries side-by-side)
static constexpr int HAND_DOFS_IN_ORDER[16] = {3,1,1, 3,1,1, 3,1,1, 3,1,1, 2,3,1,1};

// Decode one hand's 54-D PCA-decoded vector → 27-D Euler param vector.
// out[27] is fully written (no zero-init needed).
inline void compact_cont_to_hand_params(const float* cont54, float* out27)
{
    int cont_pos  = 0;   // running cursor in the 54-D cont vector
    int param_pos = 0;   // running cursor in the 27-D output
    for (int j = 0; j < 16; ++j) {
        int k = HAND_DOFS_IN_ORDER[j];
        if (k == 3) {
            float euler[3];
            rot6d_to_euler(cont54 + cont_pos, euler);
            out27[param_pos + 0] = euler[0];
            out27[param_pos + 1] = euler[1];
            out27[param_pos + 2] = euler[2];
            cont_pos  += 6;
            param_pos += 3;
        } else {
            // k == 1 or 2: each contributes k pairs (sin,cos) in cont and k atan2s in params.
            for (int i = 0; i < k; ++i) {
                float s = cont54[cont_pos + 0];
                float c = cont54[cont_pos + 1];
                out27[param_pos] = std::atan2(s, c);
                cont_pos  += 2;
                param_pos += 1;
            }
        }
    }
}

// Apply both hands' pose to an existing model_params [204] vector.
// hand_pose_params [108]   = the C engine's r.hand_pose (54 left + 54 right, raw 6D codes)
// hand_pose_mean   [54]    from body_model.lbs
// hand_pose_comps  [54×54] from body_model.lbs (row-major; matches Python .mm semantics)
// hand_joint_idxs_{l,r} [27] absolute indices in full_pose_params [136]
//   (i.e., index into model_params[0:136]; offsets 0..2 are global_trans, 3..5 global_rot)
inline void apply_hand_pose(
    float*       model_params204,
    const float* hand_pose_params,    // [108]
    const float* hand_pose_mean,      // [54]
    const float* hand_pose_comps,     // [54×54]  row-major (Python h.mm(comps) → out[i] = sum_k h[k]*comps[k,i])
    const int*   hand_joint_idxs_left,// [27]
    const int*   hand_joint_idxs_right) // [27]
{
    if (!model_params204 || !hand_pose_params || !hand_pose_mean ||
        !hand_pose_comps || !hand_joint_idxs_left || !hand_joint_idxs_right)
        return;

    auto decode_one = [&](const float* h54, const int* idx27)
    {
        // PCA decode: decoded[i] = mean[i] + Σ_k h54[k] * comps[k, i]
        float decoded[54];
        for (int i = 0; i < 54; ++i) decoded[i] = hand_pose_mean[i];
        for (int k = 0; k < 54; ++k) {
            float hk = h54[k];
            if (hk == 0.f) continue;
            const float* row = hand_pose_comps + (size_t)k * 54;
            for (int i = 0; i < 54; ++i) decoded[i] += hk * row[i];
        }
        // 6D / atan2 → 27 Euler params
        float params[27];
        compact_cont_to_hand_params(decoded, params);
        // Insert into model_params at absolute joint indices
        for (int i = 0; i < 27; ++i) {
            int idx = idx27[i];
            if (idx >= 0 && idx < 136) model_params204[idx] = params[i];
        }
    };

    decode_one(hand_pose_params + 0,  hand_joint_idxs_left);   // left  hand: cont[0:54]
    decode_one(hand_pose_params + 54, hand_joint_idxs_right);  // right hand: cont[54:108]
}

// ─── Assemble model_params [204] for the torch.jit body model ─────────────────
//
// body_model.onnx expects:
//   shape      [45]   identity blend shape betas
//   body_params[204]  = full_pose_params [136] + scales [68]
//   face       [72]
//
// Actually: the MHR model is called with (shape_params, model_params, expr_params).
// model_params=[204] = cat([full_pose_params, scales], dim=1)
// where full_pose_params=[136] = global_trans[3]+global_rot_euler[3]+body_pose[133]???
// The exact layout depends on the jit model.  We pass scale zeros for body_params[136:].
//
// From the code: model_params = torch.cat([full_pose_params, scales], dim=1)
//   full_pose_params [B,136] is assembled in _mhr_forward_core
//   scales           [B,68]  comes from scale_comps PCA decode
//
// For inference we zero-fill scales and set full_pose_params from predictions.
// Caller uses build_model_params_from_prediction() below.
//
// BUG (2026-04-27): The correct layout from the Python reference code
// (mhr_head.py _mhr_forward_core line 574-576) is:
//   full_pose_params = torch.cat([global_trans * 10, global_rot, body_pose_params], dim=1)
//   model_params     = torch.cat([full_pose_params, scales], dim=1)
// So model_params layout is:
//   [0:3]   = global_trans (scaled by 10, zeroed in single-view)
//   [3:6]   = global_rot_euler
//   [6:136] = body_pose_params (first 130 of 133 joints)
//   [136:204] = scales (zeroed)
// The current C++ implementation below puts global_rot at [0:2] and body_pose at [3:135],
// which is WRONG — it shifts everything by 3 positions into the wrong PT matrix columns.
// This causes garbage joint parameters and a deformed mesh.
//
// Additionally, hand joints should be zeroed (mhr_head.py line 433):
//   pred_pose_euler[:, mhr_param_hand_idxs] = 0
// And global_trans is zeroed (mhr_head.py line 427):
//   global_trans = torch.zeros_like(global_rot_euler)
struct ModelParams204 {
    float data[204] = {};
};

inline ModelParams204 build_model_params(
    const float* global_rot_euler,  // [3]  (ZYX Euler from rot6d_to_euler)
    const float* body_euler,        // [133]
    const float* scale_params,      // [28]  raw scale params (PCA codes)
    // scale_comps [28×68] and scale_mean [68] from the body model are not
    // available here – set scales to zero for a reasonable result
    bool         zero_scales = true
)
{
    ModelParams204 out{};
    // Layout from Python reference (mhr_head.py _mhr_forward_core line 574-584):
    //   full_pose_params = torch.cat([global_trans * 10, global_rot, body_pose_params], dim=1)
    //   model_params     = torch.cat([full_pose_params, scales], dim=1)
    //
    // Correct layout:
    //   [0:3]   = global_trans (scaled by 10, zeroed in single-view)
    //   [3:6]   = global_rot_euler
    //   [6:136] = body_pose_params (first 130 of 133 joints, hand joints zeroed)
    //   [136:204] = scales (zeroed)

    // [0:3] = global_trans (zeroed for single-view)
    out.data[0] = 0.0f;
    out.data[1] = 0.0f;
    out.data[2] = 0.0f;

    // [3:6] = global_rot in the ZYX order that Python stores via roma.rotmat_to_euler("ZYX"):
    //   roma.rotmat_to_euler("ZYX", R) → [rz, ry, rx]  (Z angle first, X angle last)
    //   rot6d_to_euler (C)             → [rx, ry, rz]  (X angle first — matches batchXYZfrom6D)
    // The PT matrix was trained with Python's [rz, ry, rx] layout, so we swap [0]↔[2].
    // Body-pose joints are NOT swapped: batchXYZfrom6D and rot6d_to_euler both use [rx,ry,rz].
    out.data[3] = global_rot_euler[2];  // rz  (Z angle)
    out.data[4] = global_rot_euler[1];  // ry  (Y angle)
    out.data[5] = global_rot_euler[0];  // rx  (X angle)

    // [6:136] = body_pose_params (first 130 of 133 joints)
    // Python code uses body_pose_params[..., :130] (line 568 of mhr_head.py)
    // This copies 130 floats from body_euler into [6:136]
    std::memcpy(out.data + 6, body_euler, 130 * sizeof(float));

    // Zero hand joint params (indices 62-115 in the 133-dim body_pose)
    // In model_params these become indices 68-121 (6 + 62 to 6 + 115)
    // Python code: pred_pose_euler[:, mhr_param_hand_idxs] = 0
    // mhr_param_hand_idxs = [62..115]
    for (int i = 68; i <= 121; ++i)
        out.data[i] = 0.0f;

    // [136:204] = scales (zeroed)
    // Already zeroed by default initialization
    (void)scale_params; (void)zero_scales;
    return out;
}

} // namespace fsb
