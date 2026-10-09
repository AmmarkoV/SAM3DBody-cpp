// skin_color.cpp  –  see skin_color.h
#include "skin_color.h"
#include "bbox_iou.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>

namespace fsb {

// The occlusion z-buffer only has to separate body parts from each other, so
// it is rasterised at a reduced resolution (longest side) to keep it cheap.
static constexpr int   ZBUF_MAX_SIDE = 512;
// A vertex counts as visible when it is at most this far (metres) behind the
// nearest surface at its pixel — it sits on its own triangles, so it is always
// "at" the z-buffer depth give or take a coarse pixel's worth of slope.
static constexpr float ZBUF_EPS      = 0.03f;
// Reject samples viewed more obliquely than this (cosine of the view angle):
// near the silhouette a small fit error lands the vertex on the background.
static constexpr float MIN_FACING    = 0.3f;
// discrepancy() needs at least this many vertices seen by both sides.
static constexpr size_t MIN_SHARED_VERTICES = 500;

float observation_change(const SkinObservation& a, const SkinObservation& b)
{
    if (a.w.size() != b.w.size() || a.rgb.size() != b.rgb.size()) return -1.f;
    double diff = 0.0, wsum = 0.0;
    size_t shared = 0;
    for (size_t i = 0; i < a.w.size(); ++i) {
        float wt = std::min(a.w[i], b.w[i]);
        if (wt <= 0.f) continue;
        double d = 0.0;
        for (int c = 0; c < 3; ++c) d += std::fabs(a.rgb[i*3+c] - b.rgb[i*3+c]);
        diff += wt * d / 3.0;
        wsum += wt;
        ++shared;
    }
    if (shared < MIN_SHARED_VERTICES || wsum <= 0.0) return -1.f;
    return (float)(diff / wsum);
}

// ── SkinObserver ─────────────────────────────────────────────────────────────

void SkinObserver::init(const unsigned int* indices, size_t n_indices, size_t n_vertices)
{
    n_vertices_ = n_vertices;
    tris_.assign(indices, indices + (n_indices / 3) * 3);
}

void SkinObserver::observe(SkinObservation& out,
                           const uint8_t* bgr, int w, int h,
                           const float* verts, const float* normals,
                           const float cam_t[3], float focal, float cx, float cy)
{
    SkinProjection pr;
    project(pr, w, h, verts, normals, cam_t, focal, cx, cy);
    sample(out, pr, bgr);
}

void SkinObserver::project(SkinProjection& out, int w, int h,
                           const float* verts, const float* normals,
                           const float cam_t[3], float focal, float cx, float cy)
{
    out.vertex.clear(); out.u.clear(); out.v.clear(); out.w.clear();
    out.img_w = w; out.img_h = h; out.n_vertices = n_vertices_;
    if (!n_vertices_ || w <= 0 || h <= 0) return;

    // ── Project every vertex (full-resolution pixels + camera depth) ─────────
    proj_.resize(n_vertices_ * 3);
    for (size_t i = 0; i < n_vertices_; ++i) {
        float x = verts[i*3+0] + cam_t[0];
        float y = verts[i*3+1] + cam_t[1];
        float z = verts[i*3+2] + cam_t[2];
        float* p = &proj_[i*3];
        p[2] = z;
        if (z < 1e-3f) { p[0] = p[1] = -1.f; continue; }   // behind the camera
        p[0] = focal * x / z + cx;
        p[1] = focal * y / z + cy;
    }

    // ── Occlusion z-buffer at reduced resolution ─────────────────────────────
    const float s  = std::min(1.f, (float)ZBUF_MAX_SIDE / (float)std::max(w, h));
    const int   zw = std::max(1, (int)(w * s)), zh = std::max(1, (int)(h * s));
    zbuf_.assign((size_t)zw * zh, std::numeric_limits<float>::infinity());
    for (size_t t = 0; t + 2 < tris_.size(); t += 3) {
        const float* A = &proj_[tris_[t]   * 3];
        const float* B = &proj_[tris_[t+1] * 3];
        const float* C = &proj_[tris_[t+2] * 3];
        if (A[2] < 1e-3f || B[2] < 1e-3f || C[2] < 1e-3f) continue;
        float ax = A[0]*s, ay = A[1]*s, bx = B[0]*s, by = B[1]*s, qx = C[0]*s, qy = C[1]*s;
        float area = (bx-ax)*(qy-ay) - (by-ay)*(qx-ax);
        if (std::fabs(area) < 1e-12f) continue;
        int x0 = std::max(0,      (int)std::floor(std::min({ax, bx, qx})));
        int x1 = std::min(zw - 1, (int)std::ceil (std::max({ax, bx, qx})));
        int y0 = std::max(0,      (int)std::floor(std::min({ay, by, qy})));
        int y1 = std::min(zh - 1, (int)std::ceil (std::max({ay, by, qy})));
        for (int py = y0; py <= y1; ++py)
            for (int px = x0; px <= x1; ++px) {
                float fx = px + 0.5f, fy = py + 0.5f;
                float w0 = ((bx-fx)*(qy-fy) - (by-fy)*(qx-fx)) / area;
                float w1 = ((qx-fx)*(ay-fy) - (qy-fy)*(ax-fx)) / area;
                float w2 = 1.f - w0 - w1;
                if (w0 < 0.f || w1 < 0.f || w2 < 0.f) continue;
                float z = w0*A[2] + w1*B[2] + w2*C[2];
                float& d = zbuf_[(size_t)py * zw + px];
                if (z < d) d = z;
            }
    }

    // ── Keep each visible, camera-facing vertex ──────────────────────────────
    for (size_t i = 0; i < n_vertices_; ++i) {
        const float* p = &proj_[i*3];
        float u = p[0], v = p[1], z = p[2];
        if (z < 1e-3f || u < 0.f || v < 0.f || u > w - 1.f || v > h - 1.f) continue;

        // cos of the angle between the normal and the direction to the eye.
        float ex = -(verts[i*3+0] + cam_t[0]);
        float ey = -(verts[i*3+1] + cam_t[1]);
        float ez = -z;
        float el = std::sqrt(ex*ex + ey*ey + ez*ez);
        const float* n = &normals[i*3];
        float facing = (n[0]*ex + n[1]*ey + n[2]*ez) / el;
        if (facing < MIN_FACING) continue;

        int zx = std::min(zw - 1, (int)(u * s)), zy = std::min(zh - 1, (int)(v * s));
        if (z > zbuf_[(size_t)zy * zw + zx] + ZBUF_EPS) continue;   // occluded

        out.vertex.push_back((unsigned int)i);
        out.u.push_back(u);
        out.v.push_back(v);
        out.w.push_back(facing * facing);
    }
}

void SkinObserver::sample(SkinObservation& out, const SkinProjection& proj, const uint8_t* bgr)
{
    out.rgb.assign(proj.n_vertices * 3, 0.f);
    out.w.assign(proj.n_vertices, 0.f);
    if (!bgr) return;
    const int w = proj.img_w, h = proj.img_h;
    for (size_t k = 0; k < proj.vertex.size(); ++k) {
        const size_t i = proj.vertex[k];
        const float  u = proj.u[k], v = proj.v[k];
        // Bilinear sample.
        int   x0 = (int)u, y0 = (int)v;
        int   x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
        float fu = u - x0, fv = v - y0;
        const uint8_t* p00 = bgr + ((size_t)y0 * w + x0) * 3;
        const uint8_t* p10 = bgr + ((size_t)y0 * w + x1) * 3;
        const uint8_t* p01 = bgr + ((size_t)y1 * w + x0) * 3;
        const uint8_t* p11 = bgr + ((size_t)y1 * w + x1) * 3;
        for (int c = 0; c < 3; ++c) {
            float val = (1-fu)*(1-fv)*p00[c] + fu*(1-fv)*p10[c]
                      + (1-fu)*fv*p01[c]     + fu*fv*p11[c];
            out.rgb[i*3 + (2 - c)] = val / 255.f;   // BGR -> RGB
        }
        out.w[i] = proj.w[k];
    }
}

// ── SkinColorAccumulator ─────────────────────────────────────────────────────

void SkinColorAccumulator::init(const unsigned int* indices, size_t n_indices,
                                size_t n_vertices)
{
    rgba_dirty_ = true;
    n_vertices_ = n_vertices;
    sum_rgb_.assign(n_vertices * 3, 0.0);
    sum_w_.assign(n_vertices, 0.0);
    colors_.assign(n_vertices * 3, 0.5f);

    std::vector<std::vector<unsigned int>> tmp(n_vertices);
    auto add_edge = [&](unsigned int a, unsigned int b) {
        auto& l = tmp[a];
        if (std::find(l.begin(), l.end(), b) == l.end()) l.push_back(b);
    };
    for (size_t t = 0; t + 2 < (n_indices / 3) * 3; t += 3) {
        unsigned int a = indices[t], b = indices[t+1], c = indices[t+2];
        add_edge(a, b); add_edge(b, a); add_edge(b, c);
        add_edge(c, b); add_edge(c, a); add_edge(a, c);
    }
    adj_offsets_.assign(n_vertices + 1, 0);
    for (size_t v = 0; v < n_vertices; ++v)
        adj_offsets_[v+1] = adj_offsets_[v] + (unsigned int)tmp[v].size();
    adj_.clear();
    adj_.reserve(adj_offsets_[n_vertices]);
    for (auto& l : tmp) adj_.insert(adj_.end(), l.begin(), l.end());
}

void SkinColorAccumulator::add(const SkinObservation& obs)
{
    if (obs.w.size() != n_vertices_) return;
    rgba_dirty_ = true;
    for (size_t i = 0; i < n_vertices_; ++i) {
        float wt = obs.w[i];
        if (wt <= 0.f) continue;
        for (int c = 0; c < 3; ++c) sum_rgb_[i*3+c] += wt * obs.rgb[i*3+c];
        sum_w_[i] += wt;
    }
}

void SkinColorAccumulator::set_default_color(const float rgb[3])
{
    rgba_dirty_ = true;
    has_default_ = true;
    for (int c = 0; c < 3; ++c) default_rgb_[c] = rgb[c];
}

void SkinColorAccumulator::set_override(const std::vector<unsigned int>& vertices, const float rgb[3])
{
    rgba_dirty_ = true;
    override_ = vertices;
    for (int c = 0; c < 3; ++c) override_rgb_[c] = rgb[c];
}

void SkinColorAccumulator::apply_override()
{
    for (unsigned int v : override_)
        if (v < n_vertices_)
            for (int c = 0; c < 3; ++c) colors_[v*3+c] = override_rgb_[c];
}

float SkinColorAccumulator::discrepancy(const SkinObservation& obs, const float* vertex_weight) const
{
    if (obs.w.size() != n_vertices_) return -1.f;
    double diff = 0.0, wsum = 0.0;
    size_t shared = 0;
    for (size_t i = 0; i < n_vertices_; ++i) {
        float wt = obs.w[i];
        if (wt <= 0.f || sum_w_[i] <= 0.0) continue;
        if (vertex_weight) wt *= vertex_weight[i];
        double d = 0.0;
        for (int c = 0; c < 3; ++c)
            d += std::fabs(sum_rgb_[i*3+c] / sum_w_[i] - obs.rgb[i*3+c]);
        diff += wt * d / 3.0;
        wsum += wt;
        ++shared;
    }
    if (shared < MIN_SHARED_VERTICES || wsum <= 0.0) return -1.f;
    return (float)(diff / wsum);
}

const std::vector<float>& SkinColorAccumulator::colors()
{
    // Observed vertices: their weighted mean.  Then grow outwards over the
    // mesh graph one ring at a time; each newly reached vertex takes the mean
    // of its already-coloured neighbours.
    std::vector<char>         done(n_vertices_, 0);
    std::vector<unsigned int> frontier, next;
    for (size_t i = 0; i < n_vertices_; ++i) {
        // --skin-color-mirror: a never-seen vertex borrows its seen mirror.
        size_t src = i;
        if (sum_w_[i] <= 0.0 && mirror_.size() == n_vertices_) src = mirror_[i];
        if (sum_w_[src] <= 0.0) {
            // --skin-color-default: unseen vertices keep the default, no fill.
            if (has_default_)
                for (int c = 0; c < 3; ++c) colors_[i*3+c] = default_rgb_[c];
            continue;
        }
        for (int c = 0; c < 3; ++c)
            colors_[i*3+c] = (float)(sum_rgb_[src*3+c] / sum_w_[src]);
        done[i] = 1;
        frontier.push_back((unsigned int)i);
    }
    // Nothing seen yet stays grey; with a default there is nothing to fill.
    if (frontier.empty() || has_default_) { apply_override(); return colors_; }

    while (!frontier.empty()) {
        next.clear();
        for (unsigned int v : frontier)
            for (unsigned int k = adj_offsets_[v]; k < adj_offsets_[v+1]; ++k)
                if (!done[adj_[k]]) { done[adj_[k]] = 2; next.push_back(adj_[k]); }
        // done==2 marks "in this ring": average only rings already finished.
        for (unsigned int v : next) {
            float acc[3] = {0.f, 0.f, 0.f};
            int   cnt = 0;
            for (unsigned int k = adj_offsets_[v]; k < adj_offsets_[v+1]; ++k)
                if (done[adj_[k]] == 1) {
                    for (int c = 0; c < 3; ++c) acc[c] += colors_[adj_[k]*3+c];
                    ++cnt;
                }
            for (int c = 0; c < 3; ++c) colors_[v*3+c] = acc[c] / cnt;
        }
        for (unsigned int v : next) done[v] = 1;
        frontier.swap(next);
    }
    apply_override();
    return colors_;
}

const std::vector<float>& SkinColorAccumulator::colors_rgba()
{
    if (!rgba_dirty_) return rgba_;
    rgba_dirty_ = false;
    colors();
    rgba_.resize(n_vertices_ * 4);
    for (size_t i = 0; i < n_vertices_; ++i) {
        for (int c = 0; c < 3; ++c) rgba_[i*4+c] = colors_[i*3+c];
        rgba_[i*4+3] = sum_w_[i] > 0.0 ? 1.f : 0.f;
    }
    return rgba_;
}

float SkinColorAccumulator::coverage() const
{
    if (!n_vertices_) return 0.f;
    size_t seen = 0;
    for (double w : sum_w_) seen += (w > 0.0);
    return (float)seen / (float)n_vertices_;
}

// ── PersonSlots ──────────────────────────────────────────────────────────────

PersonSlots::Params PersonSlots::Params::from_env()
{
    auto env = [](const char* name, float def) {
        const char* v = getenv(name);
        return v ? (float)atof(v) : def;
    };
    Params p;
    p.min_iou    = env("FSB_SLOT_MIN_IOU",    p.min_iou);
    p.app_scale  = env("FSB_SLOT_APP_SCALE",  p.app_scale);
    p.reid_sim   = env("FSB_SLOT_REID_SIM",   p.reid_sim);
    p.app_weight = env("FSB_SLOT_APP_WEIGHT", p.app_weight);
    p.confirm    = (int)env("FSB_SLOT_CONFIRM", (float)p.confirm);
    p.reid_confirm = env("FSB_SLOT_REID_CONFIRM", p.reid_confirm);
    p.margin     = env("FSB_SLOT_MARGIN", p.margin);
    p.motion     = env("FSB_SLOT_MOTION", p.motion ? 1.f : 0.f) != 0.f;
    return p;
}

std::vector<std::pair<int, std::array<float, 4>>> PersonSlots::recent_boxes(long max_age) const
{
    std::vector<std::pair<int, std::array<float, 4>>> out;
    for (size_t s = 0; s < slots_.size(); ++s)
        if (slots_[s].alive && frame_ + 1 - slots_[s].real_seen <= max_age)
            out.emplace_back((int)s, slots_[s].box);
    return out;
}

std::vector<int> PersonSlots::assign(const std::vector<std::array<float, 4>>& boxes,
                                     const std::vector<float>& discrepancy,
                                     const std::vector<char>& synthetic)
{
    // A missed slot's box is extrapolated for at most this many frames.
    static constexpr long  MAX_EXTRAPOLATE = 10;
    // A tentative slot may be merged into an older one only this young.
    static constexpr long  CONFIRM_WINDOW  = 30;

    ++frame_;
    const size_t S = slots_.size();
    const bool   have_app = discrepancy.size() == boxes.size() * S;
    auto app_of = [&](size_t b, size_t s) {
        float d = have_app ? discrepancy[b * S + s] : -1.f;
        return d < 0.f ? -1.f : std::max(0.f, 1.f - d / p_.app_scale);
    };

    // Where each slot's person should be now.
    std::vector<std::array<float, 4>> pred(S);
    for (size_t s = 0; s < S; ++s) {
        pred[s] = slots_[s].box;
        if (p_.motion) {
            const float k = (float)std::min(frame_ - slots_[s].seen, MAX_EXTRAPOLATE);
            for (int c = 0; c < 4; c += 2) { pred[s][c] += k * slots_[s].vx; pred[s][c+1] += k * slots_[s].vy; }
        }
    }

    struct Pair { float score; int box, slot; };
    std::vector<Pair> pairs;
    for (size_t b = 0; b < boxes.size(); ++b)
        for (size_t s = 0; s < S; ++s) {
            if (!slots_[s].alive) continue;
            const float iou = fsb::bbox_iou(boxes[b], pred[s]);
            const float app = app_of(b, s);
            if (app < 0.f) {
                if (iou >= p_.min_iou) pairs.push_back({iou, (int)b, (int)s});
                continue;
            }
            // Appearance alone re-identifies only when no confirmation is asked
            // for; otherwise it has to hold over frames (below).
            if (iou >= p_.min_iou || (p_.confirm <= 0 && app >= p_.reid_sim))
                pairs.push_back({(1.f - p_.app_weight) * iou + p_.app_weight * app, (int)b, (int)s});
        }
    std::sort(pairs.begin(), pairs.end(),
              [](const Pair& x, const Pair& y) { return x.score > y.score; });

    std::vector<int>  slot(boxes.size(), -1);
    std::vector<char> taken(S, 0);
    for (const Pair& q : pairs)
        if (slot[q.box] < 0 && !taken[q.slot]) { slot[q.box] = q.slot; taken[q.slot] = 1; }

    // Delayed re-identification: a young slot whose person keeps looking like
    // an older, currently absent slot is that person; merge and retire it.
    if (p_.confirm > 0 && have_app)
        for (size_t b = 0; b < boxes.size(); ++b) {
            const int t = slot[b];
            if (t < 0 || frame_ - slots_[t].created > CONFIRM_WINDOW) continue;
            // Candidates: older, alive, absent now, and not seen since t appeared.
            const float need = p_.reid_confirm >= 0.f ? p_.reid_confirm : p_.reid_sim;
            int best = -1; float best_app = -1.f, second = 0.f;
            for (size_t s = 0; s < S; ++s) {
                if ((int)s == t || taken[s] || !slots_[s].alive || slots_[s].created >= slots_[t].created ||
                    slots_[s].seen >= slots_[t].created)
                    continue;
                const float a = app_of(b, s);
                if (a > best_app) { second = std::max(second, best_app); best_app = a; best = (int)s; }
                else              second = std::max(second, a);
            }
            if (best >= 0 && (best_app < need || best_app - second < p_.margin)) best = -1;
            Slot& ts = slots_[t];
            if (best < 0)               { ts.cand = -1; ts.cand_frames = 0; continue; }
            if (ts.cand == best)        ++ts.cand_frames;
            else                        { ts.cand = best; ts.cand_frames = 1; }
            if (ts.cand_frames >= p_.confirm) {
                ts.alive = false;
                taken[best] = 1;
                slot[b] = best;
            }
        }

    for (size_t b = 0; b < boxes.size(); ++b) {
        if (slot[b] < 0) {
            Slot ns;
            ns.box = boxes[b];
            ns.seen = ns.created = ns.real_seen = frame_;
            slot[b] = (int)slots_.size();
            slots_.push_back(ns);
            continue;
        }
        Slot& s = slots_[slot[b]];
        if (p_.motion) {
            const long  age = std::max(1L, frame_ - s.seen);
            const float vx = (0.5f * (boxes[b][0] + boxes[b][2]) - 0.5f * (s.box[0] + s.box[2])) / age;
            const float vy = (0.5f * (boxes[b][1] + boxes[b][3]) - 0.5f * (s.box[1] + s.box[3])) / age;
            s.vx = age > MAX_EXTRAPOLATE ? 0.f : 0.5f * s.vx + 0.5f * vx;
            s.vy = age > MAX_EXTRAPOLATE ? 0.f : 0.5f * s.vy + 0.5f * vy;
        }
        s.box  = boxes[b];
        s.seen = frame_;
        if (b >= synthetic.size() || !synthetic[b]) s.real_seen = frame_;
    }
    return slot;
}

} // namespace fsb
