#pragma once
// ============================================================================
// focus.h  –  --focus: spend inference only where the image changed
//
// Two gates driven by the same low-level cue, the mean absolute intensity
// difference between frames:
//
//   FocusFrameGate  whole frame, before detection: when nothing moved since the
//                   detector last ran, its boxes stand and it is not run again.
//   FocusTracker    per detected person, after detection: people who have not
//                   moved keep their previous solution instead of being
//                   regressed.
// ============================================================================
#include "fast_sam_3dbody.h"
#include "preprocess.hpp"

#include <opencv2/core.hpp>
#include <cstdint>
#include <utility>
#include <vector>

namespace fsb {

// ─── --focus: dynamic distribution of inference over the detected people ──
//
// Implements the budget-allocation idea of
//
//   A. Qammaz, N. Kyriazis and A.A. Argyros, "Boosting the Performance of
//   Model-based 3D Tracking by Employing Low Level Motion Cues", In British
//   Machine Vision Conference (BMVC 2015), BMVA, pp. 144-1, Swansea, UK,
//   September 2015.
//
// There, an ensemble of collaborating trackers shares a pool of PSO budget
// and cheap low-level motion cues decide who gets B_max and who gets B_min
// (Eq. 7).  Here the "tracker" is one person crop through the backbone and
// decoder, and the budget is binary: regress this person again, or retain the
// solution we already hold for them.  The observation the paper rests on
// carries over directly — in most footage only some of the people present are
// moving at any instant, and re-solving someone who has not moved buys an
// answer we already have.
//
// WHICH CUES SURVIVE THE PORT.  The paper had RGBD input and four cues
// (Eq. 6): frame difference on colour m_c, the same on depth m_d, tracked
// velocity v, and structured depth noise e.  m_d and e need the depth stream
// and have no monocular counterpart, so they go.  v (Eq. 4, the change in the
// tracked 3D state) was implemented and then removed after measuring it: over
// 2682 person-frames from four clips, the people with the LARGEST frame-to-
// frame change in regressed camera translation had the most static images
// (median m_c of 0.07 for v > 50 mm, against 11.57 for v < 2 mm).  The signal
// is monocular depth ambiguity jittering t_z, not motion, so gating on it
// regresses the wrong people.  Eq. 6 therefore reduces to
//
//     m_i = (m_c,i > sensitivity) OR (t_i > 0) OR (r_i >= FOCUS_MAX_RETAIN).
//
// m_c is also the only cue that stays meaningful under a hard skip: it is
// computed from the raw frames, so it keeps reporting whether a person moved
// whether or not we regressed them.  A cue derived from the tracker's own
// output cannot do that — with B_min = 0 the state stops updating, the cue
// freezes, and the person is retained forever.  The paper never hits this
// because its B_min is 64 particles x 4 generations, not zero.
//
// t_i is the paper's timeout: once seen moving, a person stays on the
// expensive path for FOCUS_TIMEOUT more frames, so a pause mid-gesture does
// not drop them.  Per the paper a t_i that fires on its own only decrements,
// it does not re-arm.  r_i is the B_min analogue that bounds staleness: no
// one is retained more than FOCUS_MAX_RETAIN frames in a row, whatever the
// sensitivity is set to.
class FocusTracker
{
public:
    // Partition dets into the people to regress this frame and the people
    // whose previous solution we keep.  dets is rewritten to the former;
    // `retained` receives (original slot, solution) for the latter, and
    // `active_slot` maps each surviving det back to its original index so
    // process_mat can put the frame back together in detection order.
    // `motion`, when set, replaces box_motion() as the per-person cue
    // (Pipeline::set_focus_motion), measured against the frame each person was
    // last regressed on.
    void select(const cv::Mat& bgr, float sensitivity, bool debug,
                const FocusMotionFn& motion,
                std::vector<PersonDet>& dets,
                std::vector<std::pair<int, MHRResult>>& retained,
                std::vector<int>& active_slot);

    // Fold this frame's fresh solutions into the track table so the next frame
    // can retain them.  dets/results here are the regressed subset, still
    // index-aligned, because this runs before process_mat merges the retained
    // people back in.
    // `keep_key` (a motion cue is set) also stores bgr as those people's
    // keyframe.
    void commit(const std::vector<PersonDet>& dets,
                const std::vector<MHRResult>& results,
                const cv::Mat& bgr, bool keep_key);

private:
    static constexpr int   FOCUS_TIMEOUT    = 4;     // t_i, frames (paper's value)
    static constexpr int   FOCUS_MAX_RETAIN = 8;     // r_i cap: forced refresh
    static constexpr float FOCUS_MATCH_IOU  = 0.3f;  // det <-> track association

    struct FocusTrack
    {
        PersonDet det;              // box when we last regressed this person
        MHRResult result;           // the retained solution
        cv::Mat   key_bgr;          // frame `result` was regressed on (motion cue only)
        int       timeout = 0;      // t_i
        int       retained = 0;     // r_i, consecutive frames served from cache
        bool      matched = false;  // scratch, per frame
    };
    std::vector<FocusTrack> tracks_;
    cv::Mat                 prev_gray_;  // I_{t-1}, for Eq. 3
    uint64_t                n_regressed_ = 0, n_retained_ = 0;

    float box_motion(const cv::Mat& gray, const PersonDet& d,
                     float sensitivity) const;
};

// ─── --focus: skip person detection on static frames ──────────────────────
//
// FocusTracker decides who gets regressed, but it still needs boxes every
// frame, and on a frame where nobody is regressed the detector is ~90% of what
// is left.  This gate applies the same cue one level up: if nothing in the
// image moved since the last frame the detector actually ran on, that run's
// boxes are still the answer and the detector is not run again.
//
// The check runs on the letterboxed image at the detector's input resolution,
// the resize the detector needs anyway, so a missed skip costs one grey
// conversion and a frame difference on top of the detector.  What exactly is
// compared, and why, is in focus.cpp above FocusFrameGate::reuse().
class FocusFrameGate
{
public:
    // bgr is the full-resolution frame; det_w x det_h is the detector input
    // (640x640).  On a static frame writes the cached boxes (original-image
    // pixels, possibly none) to `dets` and returns true.  Otherwise returns
    // false and leaves the aspect-preserving resize the detector letterboxes
    // in letterboxed(), so the caller does not resize twice.
    bool reuse(const cv::Mat& bgr, int det_w, int det_h, float sensitivity,
               std::vector<PersonDet>& dets);

    // The detector ran on the frame reuse() just rejected and produced `dets`
    // (original-image pixels).  That frame becomes the new reference.
    void commit(const std::vector<PersonDet>& dets);

    const cv::Mat& letterboxed() const { return resized_; }
    // What the last reuse() measured, mean |dI| in 0-255: the busiest cached
    // box, and the busiest scene tile outside the boxes.  -1 = not measured
    // (no reference frame, skip cap reached, or decided before getting there).
    float          box_motion()   const { return box_motion_; }
    float          scene_motion() const { return scene_motion_; }

private:
    cv::Mat                resized_;   // this frame, scaled to the detector
    cv::Mat                gray_;      // ... in grey
    cv::Mat                diff_f_;    // |gray_ - key_gray_|, float scratch
    cv::Mat                key_gray_;  // frame the cached boxes came from
    std::vector<PersonDet> key_dets_;
    int                    skipped_ = 0;
    float                  box_motion_   = -1.f;
    float                  scene_motion_ = -1.f;
};

} // namespace fsb
