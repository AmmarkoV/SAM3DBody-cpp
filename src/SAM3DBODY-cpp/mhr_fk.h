#pragma once
// mhr_fk.h — shared MHR (Momentum Human Rig) forward-kinematics core.
//
// Factored out of BVHWriter (2026-11) so every consumer of the 127-joint MHR
// skeleton — .bvh export, .arf export, and the live ROS/TF joint-local API
// (BVHWriter::compute_joint_locals(), CLIENTSERVER.md) — runs the exact same
// quaternion FK instead of maintaining parallel copies. This header carries
// no BVH-specific concepts (no template joints, no Euler-channel decisions,
// no rest-frame retarget) — those stay in bvh_writer.cpp.

#include <vector>

struct MHR_LBS_Data;
namespace fsb { struct MHRResult; }

namespace mhr_fk
{

// ── quaternion helpers (XYZW) ───────────────────────────────────────────────
void qmul(float* r, const float* a, const float* b);
void qconj(float* r, const float* a);
void qrot(float* o, const float* q, const float* v);
void euler_mhr_to_quat(float ex, float ey, float ez, float* q);

// One MHR joint's transform relative to its parent. For the root, "parent"
// is the camera frame: rotation = the body's global orientation, translation
// = MHRResult::pred_cam_t (metres).
struct JointLocal
{
    const char* name;    // MHR joint name (points into the static joint table)
    const char* parent;  // parent joint name, or "" for the root
    float       q[4];    // xyzw, parent-relative rotation
    float       t[3];    // metres, parent-relative translation
};

// Persistent FK scratch state for one MHR skeleton (127 joints), reused every
// frame/person. Does not own `lbs` — it must outlive the State.
class State
{
public:
    // Sizes scratch buffers and computes the rest-pose joint globals
    // (q_global_rest()) from lbs->joint_offsets/joint_prerotations.
    void init(const MHR_LBS_Data* lbs);
    bool is_init() const { return lbs_ != nullptr; }

    // Run FK for one MHRResult: fills q_local()/q_global()/t_global()/
    // s_global() for every joint (camera-frame, centimetre LBS units — same
    // convention as MHR_LBS_Data / mhr_lbs_compute()).
    void compute(const fsb::MHRResult& r);

    // Parent-relative quaternion+translation per joint, derived from the
    // CURRENT compute() state — call compute(r) first.
    void joint_locals(const fsb::MHRResult& r, std::vector<JointLocal>& out) const;

    // Rest-local bone vector from `ancestor` to `self` (MHR joint indices;
    // `ancestor` need not be `self`'s direct parent — a caller that collapses
    // chain joints not present in its own skeleton passes the nearest
    // ancestor it actually tracks). Uses the CURRENT compute() state.
    void rest_local_bone_vector(int self, int ancestor, float out[3]) const;

    int n_joints() const { return n_joints_; }
    // Raw decoded per-joint PT output for the CURRENT compute() state: 7
    // floats/joint (tx,ty,tz, ex,ey,ez euler, log2_scale). Exposed so a
    // consumer that needs the per-frame LOCAL translation/scale directly
    // (e.g. ARFWriter's AAU_JOINT 4x4 matrices) doesn't have to re-decode PT
    // itself — q_local() already gives the corresponding local rotation.
    const std::vector<float>& joint_params()  const { return joint_params_; }
    const std::vector<float>& q_local()       const { return q_local_; }
    const std::vector<float>& q_global()      const { return q_global_mhr_; }
    // Mutable access for the multi-view fusion path (write_frame_fused),
    // which averages several views' global rotations back into this buffer
    // before appending a frame. Every other consumer should treat q_global()
    // as read-only output of compute().
    std::vector<float>&       q_global_mut()        { return q_global_mhr_; }
    const std::vector<float>& t_global()      const { return t_global_mhr_; }
    const std::vector<float>& s_global()      const { return s_global_mhr_; }
    const std::vector<float>& q_global_rest() const { return q_global_mhr_rest_; }

private:
    const MHR_LBS_Data* lbs_      = nullptr;
    int                 n_joints_ = 0;

    std::vector<float> joint_params_;     // [n_joints*7] scratch: decoded PT output
    std::vector<float> q_local_;          // [n_joints*4] parent-relative local rotation
    std::vector<float> q_global_mhr_;     // [n_joints*4] global rotation (camera frame)
    std::vector<float> t_global_mhr_;     // [n_joints*3] global position (cm)
    std::vector<float> s_global_mhr_;     // [n_joints]   accumulated per-joint scale
    std::vector<float> q_global_mhr_rest_;// [n_joints*4] rest-pose global rotation
};

} // namespace mhr_fk
