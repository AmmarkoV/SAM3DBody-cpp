#pragma once
// ============================================================================
// skin_color.h  –  accumulate a person's appearance onto the MHR mesh (--skin-color)
//
// body_mesh.tri carries no UV coordinates, so the appearance is stored per
// vertex (18439 of them, ~1 cm apart on an adult) instead of in a UV texture.
//
// Every frame, each posed vertex is projected into the source image with the
// same pinhole model the overlay uses (pixel = f * (v + cam_t).xy / (v + cam_t).z
// + c).  A vertex is observed only when it
//   * faces the camera (its normal points at the eye), and
//   * is not hidden behind another part of the body (a small CPU z-buffer of
//     the whole mesh is rasterised first).
// Observations are weighted by cos(view angle)^2, so near-frontal views
// dominate and the grazing silhouette samples, where the fit error lets the
// background leak in, count for little.  Each person's colours are the
// weighted running average over the whole session.
//
// Vertices never seen yet (the back, when the person never turns around) are
// filled from their nearest observed neighbours over the mesh graph, so the
// mesh is always fully coloured — unless set_default_color() gave them a
// colour of their own (--skin-color-default), e.g. black for the top of the
// head, which the camera rarely sees and the fill would paint with the
// forehead's skin tone.
//
// Who is who: detection order is not stable frame to frame, so PersonSlots
// matches detections to accumulators by box overlap and, when available, by
// how well each detection's pixels agree with each person's stored colours
// (SkinColorAccumulator::discrepancy on the CPU, SkinMatchGL on the GPU).
// ============================================================================

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace fsb {

// One detection's per-vertex view of the current frame.
struct SkinObservation {
    std::vector<float> rgb;   // [n_vertices x 3] RGB 0-1 under each vertex
    std::vector<float> w;     // [n_vertices] weight, 0 = not visible this frame
};

// Mean absolute colour change (0-1) between two observations of the same posed
// mesh, over the vertices visible in both — a frame difference restricted to
// the person's silhouette (--focus with --skin-color).  -1 when they share too
// few vertices to tell.
float observation_change(const SkinObservation& a, const SkinObservation& b);

// Projects a posed mesh into the image and samples the visible vertices.
class SkinObserver {
public:
    // Topology only; call once.  indices = 3 per triangle.
    void init(const unsigned int* indices, size_t n_indices, size_t n_vertices);

    //   bgr          : source image, w x h x 3 uint8, BGR
    //   verts        : posed vertices [n_vertices x 3], metres, LBS camera space
    //                  (the Y/Z-flipped space mhr_lbs_compute outputs)
    //   normals      : per-vertex normals in the same space (outward)
    //   cam_t        : MHRResult::pred_cam_t
    //   focal, cx, cy: pinhole intrinsics in pixels
    void observe(SkinObservation& out,
                 const uint8_t* bgr, int w, int h,
                 const float* verts, const float* normals,
                 const float cam_t[3], float focal, float cx, float cy);

private:
    size_t                    n_vertices_ = 0;
    std::vector<unsigned int> tris_;   // 3 per triangle
    std::vector<float>        zbuf_;   // scratch
    std::vector<float>        proj_;   // scratch: u, v, depth per vertex
};

// One person's colours, accumulated over the session.
class SkinColorAccumulator {
public:
    // Topology only; call once.  indices = 3 per triangle.
    void init(const unsigned int* indices, size_t n_indices, size_t n_vertices);

    void add(const SkinObservation& obs);

    // Colour of vertices never observed, instead of filling them from their
    // observed neighbours.  RGB 0-1; call after init().
    void set_default_color(const float rgb[3]);

    // Paint these vertices rgb (0-1) in colors(), over whatever was observed
    // (--skin-hair-cap).  Accumulation and discrepancy are unaffected.
    void set_override(const std::vector<unsigned int>& vertices, const float rgb[3]);

    // mirror[v] = v's left/right counterpart.  A never-seen vertex whose
    // mirror was seen takes the mirror's colour, before the default or the
    // neighbour fill (--skin-color-mirror).
    void set_mirror(const std::vector<unsigned int>& mirror) { mirror_ = mirror; }

    // Mean absolute colour difference (0-1) between an observation and the
    // stored colours, over the vertices both have seen.  -1 when they share
    // too little of the body to tell.
    float discrepancy(const SkinObservation& obs) const;

    // RGB in [0,1] per vertex [n_vertices x 3], unobserved vertices filled.
    // Valid until the next call.
    const std::vector<float>& colors();
    // Same, as RGBA with A = 1 where the vertex has actually been observed
    // and 0 where it was only filled in.
    const std::vector<float>& colors_rgba();

    // Fraction of vertices observed at least once.
    float coverage() const;

    bool initialized() const { return n_vertices_ > 0; }

private:
    size_t                    n_vertices_ = 0;
    std::vector<unsigned int> adj_offsets_; // CSR vertex adjacency
    std::vector<unsigned int> adj_;
    std::vector<double>       sum_rgb_;     // weighted colour sums
    std::vector<double>       sum_w_;
    std::vector<float>        colors_;
    std::vector<float>        rgba_;
    bool                      has_default_ = false;   // set_default_color
    float                     default_rgb_[3] = {0.f, 0.f, 0.f};
    std::vector<unsigned int> mirror_;                    // set_mirror
    std::vector<unsigned int> override_;                  // set_override
    float                     override_rgb_[3] = {0.f, 0.f, 0.f};
    void apply_override();
};

// Keeps each person on the same accumulator across frames.  Detection order is
// not stable (two people swap places in the result list frame to frame), which
// would average everyone into one grey blend.  Each detection is matched to
// the slot it resembles most (greedy): box overlap with the slot's last box,
// blended with appearance agreement when a discrepancy score is available.
// An unmatched detection opens a new slot.  Slots keep their last box while
// their person is missed; a strong appearance match alone can also bring a
// person back to their slot after they left the view.
class PersonSlots {
public:
    // boxes      : [x1,y1,x2,y2] per detection.
    // discrepancy: optional [n_boxes x size()] row-major, colour discrepancy
    //              0-1 of detection d vs slot s, < 0 = unknown.  Empty = boxes only.
    // Returns the slot of each box.
    std::vector<int> assign(const std::vector<std::array<float, 4>>& boxes,
                            const std::vector<float>& discrepancy = {});
    size_t size() const { return last_box_.size(); }

private:
    std::vector<std::array<float, 4>> last_box_;
};

} // namespace fsb
