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

struct MHR_LBS_Data;
struct BVH_MotionCapture;
namespace fsb { struct MHRResult; }

class BVHWriter
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
    // → body.bvh rest bone dir), applied as q_align·R·q_align⁻¹ to re-aim each
    // joint's rotation onto the template's bone.  Fixes flexion→twist leak when
    // the template rest pose differs from MHR's (e.g. T-pose vs A-pose arms).
    std::vector<float> q_bone_align_;    // [slots × 4]; identity where aligned
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
    std::vector<float> q_global_mhr_rest_;

    // Helpers
    bool  build_slots();
    void  compute_per_frame_mhr_state(const fsb::MHRResult& r);
    void  append_frame_for(PerPerson& p, const fsb::MHRResult& r);
    void  pad_continuation_frame(PerPerson& p);
    void  rewrite_offsets_for(PerPerson& p);
    void  foot_contact_pass(PerPerson& p);
    bool  dump_one_person(const PerPerson& p);

    // Tracker
    std::vector<int>  assign_tracks(const std::vector<fsb::MHRResult>& results);
    static float bbox_iou(const float a[4], const float b[4]);
};
