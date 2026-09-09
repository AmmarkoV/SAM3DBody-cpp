// ════════════════════════════════════════════════════════════════════════════
//  sync_time.cpp — see sync_time.h.  Theil-Sen robust line fit.
// ════════════════════════════════════════════════════════════════════════════

#include "sync_time.h"

#include <algorithm>
#include <cmath>

namespace mv
{
namespace
{
double median(std::vector<double>& v)   // mutates (sorts) v
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n & 1) ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
}
} // anon

TimeFit fit_time(const std::vector<std::pair<int,double>>& s)
{
    TimeFit f;
    if (s.size() < 2) return f;

    // Frame span sets an adaptive minimum frame gap for pairwise slopes: far
    // pairs give a well-conditioned slope, and skipping equal-timestamp pairs
    // drops the staircase's zero slopes.
    int fmin = s[0].first, fmax = s[0].first;
    for (auto& p : s) { fmin = std::min(fmin, p.first); fmax = std::max(fmax, p.first); }
    int span = fmax - fmin;
    int min_gap = std::max(2, span / 50);

    std::vector<double> slopes;
    auto gather = [&](int gap){
        slopes.clear();
        for (size_t i = 0; i < s.size(); ++i)
            for (size_t j = i + 1; j < s.size(); ++j)
            {
                int df = s[j].first - s[i].first;
                if (df < 0) df = -df;
                if (df < gap) continue;
                if (s[i].second == s[j].second) continue;        // same QR step
                slopes.push_back((s[j].second - s[i].second) / (double)(s[j].first - s[i].first));
            }
    };
    gather(min_gap);
    if (slopes.empty()) gather(1);          // fallback: any distinct-frame pair
    if (slopes.empty()) return f;

    double b = median(slopes);

    std::vector<double> intercepts;
    intercepts.reserve(s.size());
    for (auto& p : s) intercepts.push_back(p.second - b * p.first);
    double a = median(intercepts);

    std::vector<double> resid;
    resid.reserve(s.size());
    double rmax = 0.0;
    for (auto& p : s) { double r = std::fabs(p.second - (a + b * p.first)); resid.push_back(r); rmax = std::max(rmax, r); }

    f.ok              = (b > 0.0);
    f.t0_ms           = a;
    f.ms_per_frame    = b;
    f.fps_eff         = (b > 1e-9) ? 1000.0 / b : 0.0;
    f.samples         = (int)s.size();
    f.residual_med_ms = median(resid);
    f.residual_max_ms = rmax;
    return f;
}

} // namespace mv
