// ════════════════════════════════════════════════════════════════════════════
//  associate.cpp — see associate.h.
// ════════════════════════════════════════════════════════════════════════════

#include "associate.h"

#include "fast_sam_3dbody.h"   // fsb::MHRResult (pred_cam_t)

#include <algorithm>
#include <cmath>
#include <map>

namespace mv
{
namespace
{

// World root of a detection: T_world_cam · pred_cam_t (homogeneous point).
std::array<double,3> world_root(const Mat4& T, const std::array<float,3>& cam_t)
{
    double p[3] = { cam_t[0], cam_t[1], cam_t[2] };
    std::array<double,3> o;
    for (int i = 0; i < 3; ++i) o[i] = T[i*4+0]*p[0] + T[i*4+1]*p[1] + T[i*4+2]*p[2] + T[i*4+3];
    return o;
}
double dist3(const std::array<double,3>& a, const std::array<double,3>& b)
{ double dx=a[0]-b[0],dy=a[1]-b[1],dz=a[2]-b[2]; return std::sqrt(dx*dx+dy*dy+dz*dz); }

struct UF
{
    std::vector<int> p;
    explicit UF(int n) : p(n) { for (int i=0;i<n;++i) p[i]=i; }
    int find(int x){ while(p[x]!=x){ p[x]=p[p[x]]; x=p[x]; } return x; }
    void uni(int a,int b){ p[find(a)]=find(b); }
};

} // anon

Correspondences associate(const std::vector<StreamResult>& streams,
                          double master_fps, double match_thresh_m)
{
    Correspondences C;
    const int S = (int)streams.size();
    if (S == 0 || master_fps <= 0) return C;

    auto unix_of = [&](int s, int frel) -> double {
        return streams[s].t0_ms + (double)(streams[s].start_frame + frel) * 1000.0 / streams[s].fps_eff;
    };
    auto frel_of = [&](int s, double t) -> int {
        return (int)std::lround((t - streams[s].t0_ms) * streams[s].fps_eff / 1000.0) - streams[s].start_frame;
    };

    // ── Common overlap window over streams that produced timed frames ────────
    double t_lo = -1e300, t_hi = 1e300;
    int active = 0;
    for (int s = 0; s < S; ++s)
    {
        if (!streams[s].has_time || streams[s].fps_eff <= 0 || streams[s].frames.empty()) continue;
        double a = unix_of(s, 0), b = unix_of(s, (int)streams[s].frames.size() - 1);
        t_lo = std::max(t_lo, a); t_hi = std::min(t_hi, b); ++active;
    }
    if (active == 0 || t_lo > t_hi) return C;     // no common coverage

    const double dt = 1000.0 / master_fps;
    const double tol = dt;                         // a frame is "at" a tick within one master period
    C.t_start_ms = t_lo; C.dt_ms = dt;
    C.nticks = (int)std::floor((t_hi - t_lo) / dt) + 1;

    // ── Per (stream, tick): the present tracks with their world roots ────────
    struct Pres { int track_idx; int frel; int det; std::array<double,3> root; };
    std::vector<std::vector<std::vector<Pres>>> present(S);
    for (int s = 0; s < S; ++s)
    {
        present[s].assign(C.nticks, {});
        if (!streams[s].has_time || streams[s].frames.empty()) continue;
        for (int k = 0; k < C.nticks; ++k)
        {
            double t = t_lo + k * dt;
            int frel = frel_of(s, t);
            if (frel < 0 || frel >= (int)streams[s].frames.size()) continue;
            if (std::fabs(unix_of(s, frel) - t) > tol) continue;
            for (int ti = 0; ti < (int)streams[s].tracks.size(); ++ti)
            {
                const auto& tr = streams[s].tracks[ti];
                auto it = tr.frame_to_det.find(frel);
                if (it == tr.frame_to_det.end()) continue;
                int det = it->second;
                const auto& mhr = streams[s].frames[frel].detections[det];
                present[s][k].push_back({ ti, frel, det, world_root(streams[s].T_world_cam, mhr.pred_cam_t) });
            }
        }
    }

    // ── Track-level association: greedy per stream pair + union-find ─────────
    std::vector<int> base(S+1, 0);
    for (int s = 0; s < S; ++s) base[s+1] = base[s] + (int)streams[s].tracks.size();
    UF uf(base[S]);

    for (int i = 0; i < S; ++i)
        for (int j = i+1; j < S; ++j)
        {
            int Ti = (int)streams[i].tracks.size(), Tj = (int)streams[j].tracks.size();
            if (Ti == 0 || Tj == 0) continue;
            std::vector<std::vector<double>> sum(Ti, std::vector<double>(Tj, 0.0));
            std::vector<std::vector<int>>    cnt(Ti, std::vector<int>(Tj, 0));
            for (int k = 0; k < C.nticks; ++k)
                for (const auto& a : present[i][k])
                    for (const auto& b : present[j][k])
                    { sum[a.track_idx][b.track_idx] += dist3(a.root, b.root); cnt[a.track_idx][b.track_idx]++; }

            struct Cand { double d; int a, b; };
            std::vector<Cand> cands;
            for (int a = 0; a < Ti; ++a) for (int b = 0; b < Tj; ++b)
                if (cnt[a][b] > 0) cands.push_back({ sum[a][b]/cnt[a][b], a, b });
            std::sort(cands.begin(), cands.end(), [](const Cand&x,const Cand&y){ return x.d<y.d; });

            std::vector<char> ua(Ti,0), ub(Tj,0);
            for (const auto& c : cands)
            {
                if (ua[c.a] || ub[c.b]) continue;
                if (c.d > match_thresh_m) break;       // sorted; the rest are worse
                uf.uni(base[i]+c.a, base[j]+c.b);
                ua[c.a]=1; ub[c.b]=1;
            }
        }

    // ── Components -> global persons ─────────────────────────────────────────
    std::map<int,int> root_to_person;
    for (int s = 0; s < S; ++s)
        for (int ti = 0; ti < (int)streams[s].tracks.size(); ++ti)
        {
            int r = uf.find(base[s]+ti);
            if (!root_to_person.count(r)) root_to_person[r] = (int)root_to_person.size();
        }
    C.n_persons = (int)root_to_person.size();
    C.person_track.assign(C.n_persons, std::vector<int>(S, -1));
    for (int s = 0; s < S; ++s)
        for (int ti = 0; ti < (int)streams[s].tracks.size(); ++ti)
        {
            int p = root_to_person[uf.find(base[s]+ti)];
            C.person_track[p][s] = streams[s].tracks[ti].id;
        }

    // ── Per-tick person observations ─────────────────────────────────────────
    C.ticks.assign(C.nticks, std::vector<PersonTick>(C.n_persons));
    for (int s = 0; s < S; ++s)
        for (int k = 0; k < C.nticks; ++k)
            for (const auto& pr : present[s][k])
            {
                int p = root_to_person[uf.find(base[s]+pr.track_idx)];
                C.ticks[k][p].views.push_back({ s, pr.frel, pr.det });
            }

    return C;
}

} // namespace mv
