// ============================================================================
// focus.cpp  –  --focus whole-frame gate (see focus.h)
// ============================================================================
#include "focus.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace fsb {

// ─── FocusTracker ───────────────────────────────────────────────────────────

// Eq. 3 on the colour-intensity image: the mean per-pixel absolute difference
// inside b between this frame and the last, normalised by the box area N_b.
// This is exactly the quantity --focus takes as its sensitivity, in 0-255
// units.  Returns "definitely moving" when the box is degenerate or the
// previous frame is not comparable, so an unusable cue falls back to
// regressing rather than to serving a stale answer.
float FocusTracker::box_motion(const cv::Mat& gray, const PersonDet& d,
                               float sensitivity) const
{
    const float moving = sensitivity + 1.f;
    if (prev_gray_.empty() ||
        prev_gray_.size() != gray.size()) return moving;

    cv::Rect b(cv::Point((int)std::floor(d.x1), (int)std::floor(d.y1)),
               cv::Point((int)std::ceil (d.x2), (int)std::ceil (d.y2)));
    b &= cv::Rect(0, 0, gray.cols, gray.rows);
    if (b.width <= 0 || b.height <= 0) return moving;

    return (float)cv::norm(gray(b), prev_gray_(b), cv::NORM_L1) /
           (float)(b.width * b.height);           // N_b normalisation
}

void FocusTracker::select(const cv::Mat& bgr, float sensitivity, bool debug,
                          const FocusMotionFn& motion,
                          std::vector<PersonDet>& dets,
                          std::vector<std::pair<int, MHRResult>>& retained,
                          std::vector<int>& active_slot)
{
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

    for (auto& t : tracks_) t.matched = false;

    std::vector<PersonDet> active;
    const int n = (int)dets.size();
    for (int i = 0; i < n; ++i)
    {
        const PersonDet& d = dets[i];

        // Associate with the person we were tracking: best IoU against the
        // boxes we last regressed.  Greedy and per-frame, which is all this
        // needs — a mis-association costs one wasted regression, never a
        // wrong answer, because an unmatched detection is always regressed.
        int   best = -1;
        float best_iou = FOCUS_MATCH_IOU;
        for (int k = 0; k < (int)tracks_.size(); ++k)
        {
            if (tracks_[k].matched) continue;
            float v = iou(d, tracks_[k].det);
            if (v > best_iou) { best_iou = v; best = k; }
        }

        if (best < 0)                      // new person: no history, no cue
        {
            active_slot.push_back(i);
            active.push_back(d);
            continue;
        }

        FocusTrack& tr = tracks_[best];
        tr.matched = true;

        float m_c = -1.f;
        if (motion && !tr.key_bgr.empty() && tr.key_bgr.size() == bgr.size() &&
            bgr.isContinuous())
            m_c = motion(bgr.data, tr.key_bgr.data, bgr.cols, bgr.rows, tr.result);
        const bool from_motion = m_c >= 0.f;
        if (!from_motion) m_c = box_motion(gray, d, sensitivity);
        const bool  moved = m_c > sensitivity;
        const bool  stale = tr.retained >= FOCUS_MAX_RETAIN;

        const char* why = nullptr;
        if (moved)              { tr.timeout = FOCUS_TIMEOUT; why = "moved";   }
        else if (stale)         {                             why = "refresh"; }
        else if (tr.timeout > 0){ --tr.timeout;               why = "timeout"; }

        if (debug)
            printf("[FSB]   focus person %d: m_c=%.2f%s (sensitivity=%.2f) t_i=%d r_i=%d -> %s\n",
                   i, m_c, from_motion ? " [cue]" : "", sensitivity, tr.timeout, tr.retained,
                   why ? why : "RETAIN");
        if (debug && from_motion)   // the box cue it replaced, for comparison
            printf("[FSB]   focus person %d: box m_c=%.2f\n", i, box_motion(gray, d, sensitivity));

        if (!why)                          // m_i = 0: keep the answer we have
        {
            ++tr.retained;
            retained.emplace_back(i, tr.result);
            ++n_retained_;
            continue;
        }

        tr.retained = 0;
        active_slot.push_back(i);
        active.push_back(d);
    }

    // Tracks nobody matched this frame have left, or the detector lost them;
    // drop them so the association above cannot pair a new person with a
    // stale box.
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [](const FocusTrack& t){ return !t.matched; }),
                  tracks_.end());

    prev_gray_ = std::move(gray);
    n_regressed_ += active.size();
    dets = std::move(active);
}

void FocusTracker::commit(const std::vector<PersonDet>& dets,
                          const std::vector<MHRResult>& results,
                          const cv::Mat& bgr, bool keep_key)
{
    // One copy shared by everyone regressed on this frame (cv::Mat refcount).
    cv::Mat key = keep_key ? bgr.clone() : cv::Mat();
    for (size_t j = 0; j < results.size() && j < dets.size(); ++j)
    {
        const PersonDet& d = dets[j];

        int   best = -1;
        float best_iou = FOCUS_MATCH_IOU;
        for (int k = 0; k < (int)tracks_.size(); ++k)
        {
            float v = iou(d, tracks_[k].det);
            if (v > best_iou) { best_iou = v; best = k; }
        }
        if (best < 0)
        {
            tracks_.push_back(FocusTrack{});
            best = (int)tracks_.size() - 1;
        }
        tracks_[best].det     = d;
        tracks_[best].result  = results[j];
        tracks_[best].key_bgr = key;
    }
}

// ─── FocusFrameGate ─────────────────────────────────────────────────────────

// The gate asks two questions of the frame against the one the cached boxes
// came from, and skips the detector only when both answers are "no":
//
//   1. Did a person we already have move?  Measured exactly as FocusTracker
//      measures it, mean |dI| over each cached box, so a person the tracker
//      would retain never forces the detector on their own.  An earlier
//      version took the busiest small tile over the whole frame instead; on a
//      live webcam a seated person breathing and talking always had some 40 px
//      tile above 2.4 while their box mean sat at 1.1-1.3, and the gate never
//      skipped once.
//
//   2. Did someone appear?  Mean |dI| over coarse tiles with the cached boxes
//      masked out.  A newcomer the detector can find is person-sized, so the
//      tiles can be large enough to average away sensor noise and small
//      background motion (at 40 px the empty background of the same webcam
//      scene had a busiest tile of 2.5; at 120 px, 1.7).  Someone smaller than
//      a tile is still found within FOCUS_MAX_SKIP frames.
static constexpr int FOCUS_SCENE_TILE = 120;   // detector pixels
// Consecutive frames served from the cached boxes before the detector is run
// regardless, the frame-level twin of FOCUS_MAX_RETAIN.
static constexpr int FOCUS_MAX_SKIP   = 8;

bool FocusFrameGate::reuse(const cv::Mat& bgr, int det_w, int det_h,
                           float sensitivity, std::vector<PersonDet>& dets)
{
    // Same scale the detector's letterbox uses; the grey padding is constant
    // across frames so only the image content is compared.
    const float scale = std::min(float(det_w) / float(bgr.cols),
                                 float(det_h) / float(bgr.rows));
    const cv::Size sz((int)std::round(bgr.cols * scale),
                      (int)std::round(bgr.rows * scale));
    cv::resize(bgr, resized_, sz, 0, 0, cv::INTER_LINEAR);
    cv::cvtColor(resized_, gray_, cv::COLOR_BGR2GRAY);

    box_motion_ = scene_motion_ = -1.f;
    // Compare against the frame the boxes came from, not the previous one:
    // motion too slow to cross the sensitivity in any single frame still adds
    // up against a fixed reference.
    if (key_gray_.empty() || key_gray_.size() != gray_.size() ||
        skipped_ >= FOCUS_MAX_SKIP)
        return false;

    cv::Mat diff;
    cv::absdiff(gray_, key_gray_, diff);
    diff.convertTo(diff_f_, CV_32F);          // keep sub-integer means
    const cv::Rect frame(0, 0, diff_f_.cols, diff_f_.rows);

    // 1. people we have: per-box mean, then blank the box for question 2
    box_motion_ = 0.f;
    for (const PersonDet& d : key_dets_)
    {
        cv::Rect b(cv::Point((int)std::floor(d.x1 * scale), (int)std::floor(d.y1 * scale)),
                   cv::Point((int)std::ceil (d.x2 * scale), (int)std::ceil (d.y2 * scale)));
        b &= frame;
        if (b.width <= 0 || b.height <= 0) continue;
        box_motion_ = std::max(box_motion_, (float)cv::mean(diff_f_(b))[0]);
        diff_f_(b).setTo(0.f);
    }
    if (box_motion_ > sensitivity) return false;

    // 2. anyone new: busiest coarse tile of what is left.  A tile partly
    // covered by a box is diluted by the blanked pixels, which only matters
    // for someone right next to a person we track, and the skip cap covers it.
    cv::Mat tiles;
    cv::resize(diff_f_, tiles,
               cv::Size(std::max(1, diff_f_.cols / FOCUS_SCENE_TILE),
                        std::max(1, diff_f_.rows / FOCUS_SCENE_TILE)),
               0, 0, cv::INTER_AREA);
    double max_tile = 0.0;
    cv::minMaxLoc(tiles, nullptr, &max_tile);
    scene_motion_ = (float)max_tile;
    if (scene_motion_ > sensitivity) return false;

    ++skipped_;
    dets = key_dets_;
    return true;
}

void FocusFrameGate::commit(const std::vector<PersonDet>& dets)
{
    std::swap(key_gray_, gray_);
    key_dets_ = dets;
    skipped_  = 0;
}

} // namespace fsb
