// sync_time_test.cpp — fit_time on a synthetic QR-clock staircase + outliers.
//   30 fps camera, 12 Hz QR (so each timestamp repeats ~2-3 frames), plus a few
//   mis-decoded garbage timestamps.  The fit must recover ~30 fps and reject
//   the outliers.  Exits 0 on success, 1 on failure.

#include "sync_time.h"

#include <cmath>
#include <cstdio>
#include <vector>

int main()
{
    const double t0       = 1.7e12;        // unix ms at frame 0
    const double fps      = 30.0;
    const double ms_fr    = 1000.0 / fps;  // true ms per frame
    const double qr_hz    = 12.0;
    const double step_ms  = 1000.0 / qr_hz;

    std::vector<std::pair<int,double>> samples;
    unsigned int rng = 12345;
    for (int f = 0; f <= 600; ++f)
    {
        double real_t = t0 + f * ms_fr;
        // QR shows the timestamp quantised to its last refresh (the staircase).
        double shown  = t0 + std::floor((real_t - t0) / step_ms) * step_ms;

        rng = rng * 1103515245u + 12345u;
        if ((rng >> 16) % 20 == 0)         // ~5% gross mis-decodes
            shown = t0 + ((rng >> 8) % 100000) * 1.0 - 50000.0;

        samples.push_back({f, shown});
    }

    mv::TimeFit fit = mv::fit_time(samples);
    fprintf(stderr, "ok=%d  fps_eff=%.4f (want %.4f)  ms/frame=%.4f  t0=%.1f  resid med=%.1f max=%.1f ms  n=%d\n",
            fit.ok, fit.fps_eff, fps, fit.ms_per_frame, fit.t0_ms,
            fit.residual_med_ms, fit.residual_max_ms, fit.samples);

    int fail = 0;
    if (!fit.ok)                                 { fprintf(stderr,"  FAIL not ok\n"); fail=1; }
    if (std::fabs(fit.fps_eff - fps) > 0.5)      { fprintf(stderr,"  FAIL fps_eff off\n"); fail=1; }
    // Median residual should be within the staircase quantisation, not blown up
    // by the injected outliers.
    if (fit.residual_med_ms > step_ms)           { fprintf(stderr,"  FAIL residual too large\n"); fail=1; }
    // A frame near the middle should map back to ~its true time.
    double mid_err = std::fabs(fit.frame_to_unix_ms(300) - (t0 + 300*ms_fr));
    if (mid_err > step_ms)                       { fprintf(stderr,"  FAIL midpoint map off (%.1f ms)\n", mid_err); fail=1; }

    fprintf(stderr, fail ? "\nsync_time_test: FAILED\n" : "\nsync_time_test: OK\n");
    return fail;
}
