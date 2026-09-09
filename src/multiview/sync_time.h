// ════════════════════════════════════════════════════════════════════════════
//  sync_time.h — per-stream temporal alignment from QR timecodes
//
//  Each camera films the flashing QR clock, so some frames carry an absolute
//  unix-ms timestamp (decoded by aruco_qr).  A camera's frame index and that
//  timestamp are linearly related:  t_unix_ms ≈ t0 + ms_per_frame · frame.
//  Fitting that line per stream gives a common clock: any master-timeline unix
//  time maps to a per-stream frame index and vice-versa (see MULTIVIEW_PLAN.md).
//
//  The fit must survive two things in real data:
//    * a staircase — the QR refreshes slower (~12–19 Hz) than the camera shoots
//      (30–60 fps), so one timestamp repeats over several frames;
//    * occasional QR mis-decodes producing a garbage timestamp.
//  We use a Theil-Sen estimator (median of pairwise slopes), which ignores the
//  zero-slope within-step pairs and is robust to a minority of outliers.
//
//  Pure C++ — no OpenCV.
// ════════════════════════════════════════════════════════════════════════════

#ifndef MULTIVIEW_SYNC_TIME_H_INCLUDED
#define MULTIVIEW_SYNC_TIME_H_INCLUDED

#include <utility>
#include <vector>

namespace mv
{

struct TimeFit
{
    bool   ok            = false;
    double t0_ms         = 0.0;   // unix ms at frame 0 (intercept)
    double ms_per_frame  = 0.0;   // slope
    double fps_eff       = 0.0;   // 1000 / ms_per_frame
    int    samples       = 0;     // QR timestamps used in the fit
    double residual_med_ms = 0.0; // |t - (t0 + slope·frame)| median
    double residual_max_ms = 0.0; // ... and max

    double frame_to_unix_ms(double frame) const { return t0_ms + ms_per_frame * frame; }
    double unix_ms_to_frame(double t_ms)  const
    { return ms_per_frame != 0.0 ? (t_ms - t0_ms) / ms_per_frame : 0.0; }
};

// Robust linear fit of (frame index, unix-ms) samples.  `samples` need not be
// sorted; duplicate timestamps (the staircase) and a minority of mis-decoded
// outliers are tolerated.  Returns ok=false if there is too little signal.
TimeFit fit_time(const std::vector<std::pair<int,double>>& samples);

} // namespace mv

#endif // MULTIVIEW_SYNC_TIME_H_INCLUDED
