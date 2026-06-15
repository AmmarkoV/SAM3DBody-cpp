#pragma once
// bvh_writer.h - BVH export for the MHR body-pose pipeline.
//
// New design (2026-05)
// ────────────────────
// We use the MotionCaptureLoader library (vendored in GraphicsEngine/) to
// parse the BVH template into a struct BVH_MotionCapture, store per-frame
// rotations and the root position into mc->motionValues, track each tracked
// bone's *measured* length from the MHR FK chain, and at close() rewrite
// the BVH joint OFFSETs to the median observed length before dumping via
// dumpBVHToBVH().  This drops the residual extremity error we used to get
// from the body.bvh / MHR rest-pose mismatch.

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include "export_macros.h"

#include "fast_sam_3dbody_capi.h"

struct MHR_LBS_Data;
struct BVH_MotionCapture;
namespace fsb { struct MHRResult; }

class FSB_API BVHWriter
{
public:
    // Public because the static NAME_MAP table in bvh_writer.cpp tags each entry
    // with its kind, and the table lives in an anonymous namespace.
    enum class SlotKind : char { Other, Body, Hand };

    bool open(const std::string& template_path,
              const std::string& out_path,
              float              frame_time = 1.0f / 30.0f,
              const std::string& lbs_path   = "",
              bool               rewrite_body_offsets       = true,
              bool               rewrite_hand_offsets       = true,
              bool               compensate_finger_endsites = true,
              bool               enforce_hand_limits        = false,
              bool               zero_hand_pose             = false,
              bool               sticky_hand_pose           = false,
              bool               rest_align                 = true,
              bool               dump_rest_dirs             = false);

    void write_frame(const std::vector<fsb::MHRResult>& results);

    // Same as write_frame, but the caller supplies the track IDs (e.g. from an
    // offline globally-optimal tracker) instead of relying on the internal
    // greedy IoU tracker.
    //
    //   results[i]    — i'th detection's MHR output
    //   track_ids[i]  — its track identity (must be ≥ 0)
    //   pad_ids       — IDs that already exist in this writer but were NOT
    //                   detected this frame and should continue with their
    //                   previous pose (caller controls track lifespan)
    //
    // The internal IoU tracker is bypassed entirely.  bbox-validity filtering
    // is also skipped — the caller has already decided which detections are
    // real.  session_frames_ increments by exactly one per call (matching
    // write_frame semantics).
    void write_frame_external(const std::vector<fsb::MHRResult>& results,
                              const std::vector<int>& track_ids,
                              const std::vector<int>& pad_ids = {});

    // ── Per-joint LOCAL transforms for live TF / ROS (CLIENTSERVER.md) ───────
    // One entry per MHR joint: its transform RELATIVE TO ITS PARENT joint —
    // rotation quaternion (xyzw) + translation (metres, in the parent's frame).
    // That is a TF transform set verbatim (parent_frame → child_frame).  The
    // root joint (parent == "") is placed in the camera frame: rotation = the
    // body's global orientation, translation = pred_cam_t.  Frame is the native
    // MHR camera-optical convention (x-right, y-down, z-forward); the consumer
    // remaps to its target (e.g. ROS REP-103) — see mocapnet_rosnode.  Reuses
    // the same FK as the BVH path (compute_per_frame_mhr_state); requires the
    // writer to be open() with a valid lbs_path.  Returns false otherwise.
    struct JointLocal {
        const char* name;    // MHR joint name (points into static joint table)
        const char* parent;  // parent joint name, or "" for the root
        float       q[4];    // xyzw, parent-relative rotation
        float       t[3];    // metres, parent-relative translation
    };
    bool compute_joint_locals(const fsb::MHRResult& r,
                              std::vector<JointLocal>& out);

    // ── Multi-view fused write (MULTIVIEW_PLAN.md: AverageFuser) ─────────────
    // One view of a person: its MHRResult (in its own camera frame), the
    // quaternion that rotates that camera frame into the shared world, and the
    // person's root already lifted to world (metres).
    struct FusedView {
        const fsb::MHRResult* mhr = nullptr;
        std::array<float,4>   q_world_cam{0,0,0,1};   // xyzw
        std::array<float,3>   root_world{};
    };
    struct FusedPerson {
        int                    track_id = -1;
        std::vector<FusedView> views;
    };
    // Write one fused timeline tick: for each person, average its views' world
    // joint rotations (the world prefix cancels for relative joints, so only the
    // root's world orientation is set) + mean world root; pad absent tracks.
    void write_frame_fused(const std::vector<FusedPerson>& persons,
                           const std::vector<int>& pad_ids = {});

    void close();

    bool is_open()           const { return mc_ != nullptr; }
    int  channels_per_frame()const { return total_channels_; }

    // Optional label inserted before the numeric id in per-person filenames:
    //   "<stem>_<prefix><id>.bvh".  Empty (default) keeps the documented
    //   "<stem>_<id>.bvh" convention.  The offline --bvh-split-scenes path
    //   sets this to e.g. "scene0_person" → "<stem>_scene0_person0.bvh".
    void set_id_label_prefix(const std::string& p) { id_prefix_ = p; }

    // Enable the foot-contact cleanup pass (root vertical leveling + 2-bone leg
    // IK to pin planted feet).  Runs per person at close() over the full motion
    // buffer, so it works for both the live and offline writers.  Off by
    // default — opt in via --foot-contact.
    void set_foot_contact(bool on) { foot_contact_ = on; }

    // Zero the root joint's position and rotation channels on every frame,
    // pinning the body at the origin facing forward (in-place motion).  Useful
    // for retargeting pipelines that want local pose without global travel.
    // Off by default — opt in via --bvh-static-root.
    void set_static_root(bool on) { static_root_ = on; }

    // Override the BVH "Frame Time" written at close().  Useful for live capture
    // when stale frames were dropped: the nominal camera period no longer matches
    // the real per-frame wall-clock spacing, so playback would run fast.  Passing
    // the measured effective period (wall_seconds / processed_frames) keeps the
    // exported motion at real-time speed.  Ignored if <= 0.
    void set_frame_time(float ft) { if (ft > 0.0f) frame_time_ = ft; }
    float frame_time() const { return frame_time_; }

    BVHWriter()  = default;
    ~BVHWriter() { if (is_open()) close(); }
    BVHWriter(const BVHWriter&)            = delete;
    BVHWriter& operator=(const BVHWriter&) = delete;

private:
    // ── Per-BVH-joint slot (resolved once for the template) ───────────────
    struct BvhSlot {
        int      bvh_jid;
        int      mhr_idx;          // -1 = unmapped
        int      ancestor_bvh_jid; // nearest mapped BVH ancestor (-1 = none)
        int      ancestor_mhr_idx;
        bool     is_root;
        SlotKind kind = SlotKind::Other;
    };

    // ── Per-tracked-person state ──────────────────────────────────────────
    struct PerPerson {
        int                                  id           = -1;
        int                                  frame_count  = 0;
        std::vector<float>                   motion;          // [frame_count * channels]
        // Running per-frame bone vectors per slot (3*n_samples each), used to
        // rewrite OFFSETs at close-time.  Indexed by BVH joint id.
        std::vector<std::vector<float>>      bone_samples;
    };

    // ── Lightweight 2D-bbox tracker (greedy IoU + missing-frame tolerance) ─
    struct Track {
        int    id;
        float  bbox[4];          // x1 y1 x2 y2
        int    last_seen_frame;  // session-frame index
    };

    // Persistent state
    BVH_MotionCapture* mc_              = nullptr;
    MHR_LBS_Data*      lbs_             = nullptr;
    std::string        out_path_;
    std::string        id_prefix_;       // inserted before <id> in filenames

    // Rest-frame retarget: per-slot shortest-arc quaternion (MHR rest bone dir
    // → body.bvh rest bone dir).  q_bone_align_[i] aligns the bone whose CHILD is
    // slot i (ancestor→i).  Used to re-aim each joint's rotation onto the
    // template's bone, fixing the flexion→twist / T-pose-vs-A-pose-arm leak.
    std::vector<float> q_bone_align_;    // [slots × 4]; identity where aligned
    // Per-slot "bake" quaternion = q_bone_align_ of this slot's SINGLE mapped
    // child (identity when the slot has zero or multiple mapped children).  A
    // single-child chain joint bakes its child's alignment into its own world
    // rotation so the child bone points where the MHR bone actually points; this
    // is what corrects the arm (clavicle→uparm→lowarm→wrist) and lower-leg aim
    // without disturbing branch joints (chest, hip) — see proto_bvh_arm_retarget.py.
    std::vector<float> q_slot_bake_;     // [slots × 4]; indexed by bvh_jid
    bool               rest_align_ = true;
    int                total_channels_  = 0;
    float              frame_time_      = 1.0f / 30.0f;
    int                session_frames_  = 0;
    bool               rewrite_body_offsets_       = true;
    bool               rewrite_hand_offsets_       = true;
    bool               compensate_finger_endsites_ = true;
    bool               enforce_hand_limits_        = false;
    bool               zero_hand_pose_             = false;
    bool               sticky_hand_pose_           = false;
    bool               foot_contact_               = false;
    bool               static_root_                = false;

    std::vector<BvhSlot> slots_;             // shared template
    int                  root_bvh_jid_ = -1;

    std::vector<Track>                       tracks_;
    int                                      next_track_id_ = 0;
    std::unordered_map<int, PerPerson>       people_;

    // Reusable per-frame scratch (MHR side).
    std::vector<float> joint_params_;
    std::vector<float> q_local_;
    std::vector<float> q_global_mhr_;
    std::vector<float> t_global_mhr_;
    std::vector<float> s_global_mhr_;   // per-joint accumulated scale (matches mhr_lbs_compute)
    std::vector<float> q_global_mhr_rest_;

    // Helpers
    bool  build_slots();
    void  compute_per_frame_mhr_state(const fsb::MHRResult& r);
    void  append_frame_for(PerPerson& p, const fsb::MHRResult& r);
    // Append a row using the CURRENT q_global_mhr_ (assumes it is already
    // populated); only r.pred_cam_t is read.  append_frame_for = compute + this.
    void  append_row_from_state(PerPerson& p, const fsb::MHRResult& r);
    void  pad_continuation_frame(PerPerson& p);
    void  rewrite_offsets_for(PerPerson& p);
    void  foot_contact_pass(PerPerson& p);
    bool  dump_one_person(const PerPerson& p);

    // Tracker
    std::vector<int>  assign_tracks(const std::vector<fsb::MHRResult>& results);
    static float bbox_iou(const float a[4], const float b[4]);
};
