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
// Hand-crop padding factor — matches SAM3DBodyEstimator.transform_hand's
// GetBBoxCenterScale(padding=0.9) (sam_3d_body_estimator.py __init__),
// DIFFERENT from the body crop's 1.25. Both then go through the same
// TopdownAffine aspect-ratio fix (fixed_aspect_bbox_size). See PLAN.md,
// issue #15 "refined pose" plan — this was a real bug (hand crops were
// getting the body's 1.25 factor, inflating them ~1.4x too large).
static constexpr float HAND_BBOX_SCALE_FACTOR = 0.9f;
// "Human prior" aspect ratio (w/h) used by Python's TopdownAffine before it
// reshapes the crop to the model's square input — see fixed_aspect_bbox_size().
static constexpr float PRIOR_ASPECT_RATIO = 0.75f;

// ─── Reproduce Python's two-stage TopdownAffine aspect-ratio fix ─────────────
//
// Python (sam_3d_body/data/transforms/bbox_utils.py::fix_aspect_ratio) is
// called twice on the padded bbox scale (w0,h0) = (bw,bh) * BBOX_SCALE_FACTOR:
//   1. reshape to a 3:4 "human" prior aspect ratio (PRIOR_ASPECT_RATIO)
//   2. reshape that to the model's own square input aspect (1.0), which for a
//      square target always resolves to side = max(w1, h1).
//
// For boxes narrower than the 3:4 prior (typical tight photo crops) this
// collapses back to max(bw,bh)*BBOX_SCALE_FACTOR. For boxes wider than 3:4
// (e.g. a bbox padded out by flowing clothing) it pads noticeably more —
// omitting this step under-crops the network's input and distorts the
// regressed body proportions (see SAM3DBody-cpp issue #15).
inline float fixed_aspect_bbox_size(float bw, float bh, float scale_factor = BBOX_SCALE_FACTOR)
{
    float w0 = bw * scale_factor;
    float h0 = bh * scale_factor;

    float w1, h1;
    if (w0 > h0 * PRIOR_ASPECT_RATIO) { w1 = w0; h1 = w0 / PRIOR_ASPECT_RATIO; }
    else                              { w1 = h0 * PRIOR_ASPECT_RATIO; h1 = h0; }

    return std::max(w1, h1);
}

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
    float& crop_size_out,            // output: square side in source pixels
    // Python's GetBBoxCenterScale padding factor: 1.25 for the body crop
    // (self.transform), 0.9 for hand crops (self.transform_hand) — see
    // SAM3DBodyEstimator.__init__ in sam_3d_body_estimator.py. Both then go
    // through the SAME TopdownAffine aspect-ratio fix (fixed_aspect_bbox_size).
    float scale_factor = BBOX_SCALE_FACTOR
)
{
    const int img_w = bgr.cols;
    const int img_h = bgr.rows;

    // square crop centred on bbox
    float cx   = (bbox_x1 + bbox_x2) * 0.5f;
    float cy   = (bbox_y1 + bbox_y2) * 0.5f;
    float bw   = bbox_x2 - bbox_x1;
    float bh   = bbox_y2 - bbox_y1;
    float side = fixed_aspect_bbox_size(bw, bh, scale_factor);

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

    // Create padded square. Python's cv2.warpAffine (TopdownAffine) uses its
    // default borderValue, which is black (0,0,0) — match that, not the
    // grey(114) YOLO-letterbox convention (see SAM3DBody-cpp issue #15: a
    // grey pad here vs black in the reference pipeline showed up as a large
    // normalised-input divergence whenever the crop extends past the image).
    cv::Mat padded(y2 - y1, x2 - x1, CV_8UC3, cv::Scalar(0, 0, 0));
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

// ─── Small rotation-matrix helpers for the wrist-IK fusion (refined pose) ────
// (see PLAN.md, issue #15 "refined pose" plan). Ports the pieces of
// sam3d_body.py's run_inference "Doing IK" block that need real 3x3 rotation
// matrices, not the Euler-angle-only path the rest of this file uses.

// XYZW quaternion → row-major 3x3 rotation matrix.
inline void quat_to_mat3(const float q[4], float R[9])
{
    float x=q[0], y=q[1], z=q[2], w=q[3];
    float x2=x+x, y2=y+y, z2=z+z;
    float xx=x*x2, xy=x*y2, xz=x*z2;
    float yy=y*y2, yz=y*z2, zz=z*z2;
    float wx=w*x2, wy=w*y2, wz=w*z2;
    R[0]=1.f-(yy+zz); R[1]=xy-wz;      R[2]=xz+wy;
    R[3]=xy+wz;       R[4]=1.f-(xx+zz);R[5]=yz-wx;
    R[6]=xz-wy;       R[7]=yz+wx;      R[8]=1.f-(xx+yy);
}

// C = A @ B, all row-major 3x3.
inline void mat3_mul(const float A[9], const float B[9], float C[9])
{
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
        {
            float s = 0.f;
            for (int k = 0; k < 3; ++k) s += A[r*3+k]*B[k*3+c];
            C[r*3+c] = s;
        }
}

inline void mat3_transpose(const float A[9], float At[9])
{
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            At[c*3+r] = A[r*3+c];
}

// Minimal rotation matrix R such that R @ a = b, for unit vectors a, b
// (Rodrigues' rotation formula about axis = normalize(cross(a,b))). Used by
// the robust wrist-IK fusion (see POSEREFINE.md "design a robust fusion
// formula") to correct an FK-chain-derived rest-pose frame's forearm axis
// to match the keypoint-derived elbow->wrist direction, without disturbing
// the frame's twist/roll about that axis.
inline void mat3_rotate_a_to_b(const float a[3], const float b[3], float R[9])
{
    float cx = a[1]*b[2]-a[2]*b[1], cy = a[2]*b[0]-a[0]*b[2], cz = a[0]*b[1]-a[1]*b[0];
    float s = std::sqrt(cx*cx+cy*cy+cz*cz);
    float c = a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
    if (s < 1e-8f)
    {
        // a and b are (anti)parallel; identity if aligned, else any 180-deg
        // rotation about a perpendicular axis. Aligned case is by far the
        // common one here (correction is a small nudge), so just handle that.
        if (c > 0.f) { R[0]=1;R[1]=0;R[2]=0; R[3]=0;R[4]=1;R[5]=0; R[6]=0;R[7]=0;R[8]=1; return; }
        // Fallback for the (rare, degenerate) anti-parallel case: pick any
        // perpendicular axis and rotate 180 degrees about it.
        float perp[3] = { std::fabs(a[0])<0.9f ? 1.f:0.f, std::fabs(a[0])<0.9f ? 0.f:1.f, 0.f };
        float ax = a[1]*perp[2]-a[2]*perp[1], ay = a[2]*perp[0]-a[0]*perp[2], az = a[0]*perp[1]-a[1]*perp[0];
        float an = std::sqrt(ax*ax+ay*ay+az*az) + 1e-8f;
        ax/=an; ay/=an; az/=an;
        R[0]=2*ax*ax-1; R[1]=2*ax*ay;   R[2]=2*ax*az;
        R[3]=2*ax*ay;   R[4]=2*ay*ay-1; R[5]=2*ay*az;
        R[6]=2*ax*az;   R[7]=2*ay*az;   R[8]=2*az*az-1;
        return;
    }
    float kx=cx/s, ky=cy/s, kz=cz/s;              // rotation axis, unit
    float ang = std::atan2(s, c);
    float ca = std::cos(ang), sa = std::sin(ang), t = 1.f-ca;
    R[0]=t*kx*kx+ca;    R[1]=t*kx*ky-sa*kz; R[2]=t*kx*kz+sa*ky;
    R[3]=t*kx*ky+sa*kz; R[4]=t*ky*ky+ca;    R[5]=t*ky*kz-sa*kx;
    R[6]=t*kx*kz-sa*ky; R[7]=t*ky*kz+sa*kx; R[8]=t*kz*kz+ca;
}

// Geodesic angle between two rotation matrices (radians). Mirrors Python's
// rotation_angle_difference: angle = acos((trace(A@B^T) - 1) / 2).
inline float mat3_angle_diff(const float A[9], const float B[9])
{
    float Bt[9]; mat3_transpose(B, Bt);
    float R[9];  mat3_mul(A, Bt, R);
    float tr = R[0] + R[4] + R[8];
    float c  = std::max(-1.f, std::min(1.f, (tr - 1.f) * 0.5f));
    return std::acos(c);
}

// R = Rx(a) @ Rz(b) @ Ry(c)  →  (a,b,c), matching roma.rotmat_to_euler("XZY", …)
// (derived by hand from the same composition roma.euler_to_rotmat("XZY", …)
// uses — see PLAN.md for the derivation).
inline void rotmat_to_euler_xzy(const float R[9], float* a, float* b, float* c)
{
    // R[0]=R00 R[1]=R01 R[2]=R02 / R[3]=R10 R[4]=R11 R[5]=R12 / R[6]=R20 R[7]=R21 R[8]=R22
    *a = std::atan2(R[7], R[4]);                                   // atan2(R21, R11)
    *b = std::asin(std::max(-1.f, std::min(1.f, -R[1])));          // asin(-R01)
    *c = std::atan2(R[2], R[0]);                                   // atan2(R02, R00)
}

// Rx(a) @ Rz(b) @ Ry(c) → row-major 3x3. (Inverse of rotmat_to_euler_xzy.)
inline void euler_xzy_to_mat3(float a, float b, float c, float R[9])
{
    float ca=std::cos(a), sa=std::sin(a);
    float cb=std::cos(b), sb=std::sin(b);
    float cc=std::cos(c), sc=std::sin(c);
    R[0]=cb*cc;                R[1]=-sb;    R[2]=cb*sc;
    R[3]=ca*sb*cc+sa*sc;       R[4]=ca*cb;  R[5]=ca*sb*sc-sa*cc;
    R[6]=sa*sb*cc-ca*sc;       R[7]=sa*cb;  R[8]=sa*sb*sc+ca*cc;
}

// ─── Hand-crop wrist-centric → body-rooted transform (refined pose) ─────────
// (see PLAN.md, issue #15 "refined pose" plan). MHRHead.mhr_forward's
// enable_hand_model branch (mhr_head.py:181-196) — used only by
// head_pose_hand, never head_pose — transforms the hand-crop-predicted
// global_rot/global_trans from a wrist-centric frame into the SAME
// body-rooted frame model_params[3:6] normally represents, and zeros all
// non-hand body_pose_params/scales, before running the SAME full-body
// skinning model. Skipping this (an earlier version of this code did) means
// the resulting joint_global_rots — including the wrist joint used for the
// wrist-IK fusion — are in the wrong frame entirely, not just numerically
// off. Hand-verified end-to-end against real captured Python internals
// (mhr_forward hooked with the internal transform bypassed via an
// identity-substitution trick) to <1e-7 — see PLAN.md for the verification
// script and methodology.
//
// R = Rz(a2)@Ry(a1)@Rx(a0) for a 3-tuple (a0,a1,a2) in the SAME order as
// Python's `global_rot` tensor / model_params[3:6] (i.e. build_model_params's
// OUTPUT order, NOT rot6d_to_euler's raw (rx,ry,rz) order — swap [0]<->[2]
// first, same swap build_model_params does internally). This is roma's
// "xyz" EXTRINSIC convention (lowercase => extrinsic; hand-derived from
// roma's own intrinsic/extrinsic conversion rule and cross-checked against
// roma.euler_to_rotmat("xyz", ...) numerically — see PLAN.md).
inline void mp_rot_to_mat3(const float a[3], float R[9])
{
    float ca=std::cos(a[0]), sa=std::sin(a[0]);
    float cb=std::cos(a[1]), sb=std::sin(a[1]);
    float cc=std::cos(a[2]), sc=std::sin(a[2]);
    R[0]=cc*cb;           R[1]=cc*sb*sa-sc*ca;  R[2]=cc*sb*ca+sc*sa;
    R[3]=sc*cb;           R[4]=sc*sb*sa+cc*ca;  R[5]=sc*sb*ca-cc*sa;
    R[6]=-sb;             R[7]=cb*sa;           R[8]=cb*ca;
}

// Inverse of mp_rot_to_mat3: given R = Rz(a2)@Ry(a1)@Rx(a0), recover (a0,a1,a2).
inline void mat3_to_mp_rot(const float R[9], float a[3])
{
    a[0] = std::atan2(R[7], R[8]);                                 // atan2(R21, R22)
    a[1] = std::asin(std::max(-1.f, std::min(1.f, -R[6])));        // asin(-R20)
    a[2] = std::atan2(R[3], R[0]);                                 // atan2(R10, R00)
}

// mat3 @ vec3 (row-major R, column vector v)
inline void mat3_vec3(const float R[9], const float v[3], float out[3])
{
    out[0] = R[0]*v[0] + R[1]*v[1] + R[2]*v[2];
    out[1] = R[3]*v[0] + R[4]*v[1] + R[5]*v[2];
    out[2] = R[6]*v[0] + R[7]*v[1] + R[8]*v[2];
}

// Constants extracted from head_pose_hand on the sam-3d-body-dinov3
// checkpoint (nn.Parameter buffers — see PLAN.md's dump_hand_constants.py).
// Re-extract if the checkpoint ever changes.
static constexpr float HAND_RIGHT_WRIST_COORDS[3] = {-0.539864182472229f, 1.1133134365081787f, 0.1318483203649521f};
static constexpr float HAND_ROOT_COORDS[3]        = {0.0f, 0.9239869713783264f, 0.0f};
// Row-major 3x3 (local_to_world_wrist is stored the same way as a plain
// matrix in Python; no transpose needed — verified end-to-end, see above).
static constexpr float HAND_LOCAL_TO_WORLD_WRIST[9] = {
    0.6428927779197693f,  0.4922248423099518f, -0.5868593454360962f,
   -0.6405355930328369f,  0.7656170129776001f, -0.059537410736083984f,
    0.4200035333633423f,  0.4141804277896881f,  0.8074971437454224f
};
// model_params[204] indices to zero (145 of them — everything except the
// right-arm/hand chain + global trans/rot); the model already computes the
// hand-crop's own skeleton entirely in a "this is always a right hand"
// frame (see the KP_RIGHT_WRIST-always note elsewhere) regardless of which
// physical hand the crop came from — left crops are pre-flipped to look
// right before ever reaching the network.
static constexpr int HAND_NONHAND_PARAM_IDXS[145] = {
    6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,
    31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,
    55,56,57,58,59,60,61,62,63,64,65,66,67,95,96,97,98,99,100,101,102,103,
    104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,
    121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,
    138,139,140,141,142,143,145,146,147,148,149,150,151,152,153,179,180,
    181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,
    198,199,200,201,202,203
};

// Mirrors Python's fix_wrist_euler: try the "flipped" angle set (±pi on each
// axis, z additionally sign-flipped) and keep whichever set violates the
// joint limits less.
inline void fix_wrist_euler(float& x, float& z, float& y,
                            float lim_x0=-2.2f, float lim_x1=1.0f,
                            float lim_z0=-2.2f, float lim_z1=1.5f,
                            float lim_y0=-1.2f, float lim_y1=1.5f)
{
    constexpr float PI_F = 3.14159265358979323846f;
    auto wrap = [](float v){ return std::atan2(std::sin(v), std::cos(v)); };
    float x_alt =  wrap(x + PI_F);
    float z_alt =  wrap(-(z + PI_F));
    float y_alt =  wrap(y + PI_F);

    auto violation = [](float v, float lo, float hi){
        float below = std::max(0.f, lo - v);
        float above = std::max(0.f, v - hi);
        return below*below + above*above;
    };
    float v_orig = violation(x,lim_x0,lim_x1) + violation(z,lim_z0,lim_z1) + violation(y,lim_y0,lim_y1);
    float v_alt  = violation(x_alt,lim_x0,lim_x1) + violation(z_alt,lim_z0,lim_z1) + violation(y_alt,lim_y0,lim_y1);
    if (v_alt < v_orig) { x = x_alt; z = z_alt; y = y_alt; }
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
