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

// Where a posed mesh's visible vertices land in the image: pixel position and
// observation weight per visible vertex.  Depends on geometry only, so one
// projection can be sampled in many frames (--focus: the same retained pose in
// the current frame and its keyframe compares the same pixels).
struct SkinProjection {
    std::vector<unsigned int> vertex;   // visible vertex indices
    std::vector<float>        u, v, w;  // pixel position and weight, per entry
    int                       img_w = 0, img_h = 0;
    size_t                    n_vertices = 0;
};

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

    // observe() in two halves: the geometry (visibility, pixel positions,
    // weights) for a w x h image, then the colours under it in any frame of
    // that size.  observe() == project() + sample().
    void project(SkinProjection& out, int w, int h,
                 const float* verts, const float* normals,
                 const float cam_t[3], float focal, float cx, float cy);
    static void sample(SkinObservation& out, const SkinProjection& proj, const uint8_t* bgr);

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
    void set_mirror(const std::vector<unsigned int>& mirror) { mirror_ = mirror; rgba_dirty_ = true; }

    // Mean absolute colour difference (0-1) between an observation and the
    // stored colours, over the vertices both have seen.  -1 when they share
    // too little of the body to tell.  vertex_weight (n_vertices, optional)
    // scales each vertex's say, e.g. more for the head (--skin-face-weight).
    // gain_range > 0 first fits a per-channel illumination gain (clamped to
    // [1/(1+r), 1+r]) between the observation and the stored colours.
    // shared_out, if given, receives the number of vertices compared.
    // lum_weight >= 0 measures the difference in an orthonormal opponent space
    // (two chroma axes + luminance), with the luminance axis weighted by it
    // (1 ~ RGB, smaller = less sensitive to brightness); < 0 = plain RGB.
    float discrepancy(const SkinObservation& obs, const float* vertex_weight = nullptr,
                      float gain_range = 0.f, size_t* shared_out = nullptr,
                      float lum_weight = -1.f) const;

    // Discrepancy between two people's accumulated colours (offline merging of
    // tracklets): the same measure as discrepancy(), over vertices both have
    // observed, each weighted by the smaller of the two accumulated weights.
    float discrepancy_to(const SkinColorAccumulator& other, float gain_range = 0.f,
                         size_t* shared_out = nullptr) const;

    // RGB in [0,1] per vertex [n_vertices x 3], unobserved vertices filled.
    // Valid until the next call.
    const std::vector<float>& colors();
    // Same, as RGBA with A = 1 where the vertex has actually been observed
    // and 0 where it was only filled in.  Cached until the colours change
    // (add() or a setter), since every matching pass asks every slot for it.
    const std::vector<float>& colors_rgba();

    // Fraction of vertices observed at least once.
    float coverage() const;

    bool initialized() const { return n_vertices_ > 0; }

    // Raw weighted sums (--dump-slots: offline merging without re-running inference).
    const std::vector<double>& sum_rgb() const { return sum_rgb_; }   // [n_vertices x 3]
    const std::vector<double>& sum_w()   const { return sum_w_; }     // [n_vertices]

private:
    size_t                    n_vertices_ = 0;
    std::vector<unsigned int> adj_offsets_; // CSR vertex adjacency
    std::vector<unsigned int> adj_;
    std::vector<double>       sum_rgb_;     // weighted colour sums
    std::vector<double>       sum_w_;
    std::vector<float>        colors_;
    std::vector<float>        rgba_;
    bool                      rgba_dirty_ = true;
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
// person back to their slot after they left the view, either at once
// (confirm = 0) or once it has held for `confirm` consecutive frames: the
// person first gets a new, tentative slot, which is then merged into the old
// one and retired.  A confirmed re-entry may use a looser similarity
// (reid_confirm) if the best absent slot also beats the runner-up by
// `margin`; a slot seen at any time since the tentative one appeared is never
// a candidate (two people visible at once are two people).  With `motion`, a
// slot's box is extrapolated with its recent velocity while its person is missed.
// With pos3d, closeness in 3D (depth weighted down: monocular depth is noisy)
// joins box overlap and appearance, and a pair clearly apart in 3D is rejected
// however much the boxes overlap; with vmax, a pair is impossible if the
// person would have had to move faster than vmax since the slot was last seen.
class PersonSlots {
public:
    struct Params {
        float min_iou    = 0.2f;   // below: a different person, not the same one moved
        float app_scale  = 0.2f;   // discrepancy that maps to zero similarity
        float reid_sim   = 0.6f;   // similarity that re-identifies without box overlap
        float app_weight = 0.5f;   // appearance vs box overlap in a pair's score
        int   confirm    = 0;      // frames an appearance-only re-entry must hold, 0 = at once
        float reid_confirm = -1.f; // similarity a confirmed re-entry needs (< 0: reid_sim)
        float margin     = 0.f;    // ... and by how much it must beat the next absent slot
        bool  motion     = false;  // extrapolate missed slots' boxes with their velocity
        float pos3d      = 0.f;    // > 0: 3D proximity term, lateral tolerance (m); depth x3
        float vmax       = 0.f;    // > 0: veto matches needing a speed above this (m/s)
        float fps        = 30.f;   // frame rate, for vmax
        int   stale      = 0;      // > 0: a slot missed this many frames is not reclaimed on box
                                   // overlap / 3D proximity alone; it needs appearance (rc) or the
                                   // confirmed re-entry (a newcomer at a dormant spot: split, not swap)
        // Defaults, overridden by FSB_SLOT_{MIN_IOU,APP_SCALE,REID_SIM,APP_WEIGHT,
        // CONFIRM,REID_CONFIRM,MARGIN,MOTION,POS3D,VMAX,STALE} (parameter sweeps, tools/reid_eval).
        static Params from_env();
    };

    explicit PersonSlots(const Params& p = Params::from_env()) : p_(p) {}

    // boxes      : [x1,y1,x2,y2] per detection.
    // discrepancy: optional [n_boxes x size()] row-major, colour discrepancy
    //              0-1 of detection d vs slot s, < 0 = unknown.  Empty = boxes only.
    // synthetic  : optional, per box: true for boxes the caller made up (split
    //              from a merged detection); they move a slot but do not count
    //              as a real sighting (recent_boxes).
    // pos        : optional, per box: the person's 3D position in camera space
    //              (metres, MHRResult::pred_cam_t), for pos3d / vmax.
    // Returns the slot of each box.
    std::vector<int> assign(const std::vector<std::array<float, 4>>& boxes,
                            const std::vector<float>& discrepancy = {},
                            const std::vector<char>& synthetic = {},
                            const std::vector<std::array<float, 3>>& pos = {});
    void set_fps(float fps) { if (fps > 0.f) p_.fps = fps; }
    size_t size() const { return slots_.size(); }

    // Last box of every alive slot really detected within the last max_age
    // assign() calls (the caller's next frame), with its slot index.
    std::vector<std::pair<int, std::array<float, 4>>> recent_boxes(long max_age) const;
    const Params& params() const { return p_; }
    // The slot a confirmed re-entry merged each slot into (-1 = none, still its own person).
    // Rows written for a tentative slot before confirmation can be relabelled with
    // root_of() (output delayed by at most the confirmation window).
    int merged_into(size_t s) const { return s < slots_.size() ? slots_[s].merged_into : -1; }
    int root_of(int s) const { while (s >= 0 && merged_into(s) >= 0) s = merged_into(s); return s; }

private:
    struct Slot {
        std::array<float, 4> box;          // last box
        float vx = 0.f, vy = 0.f;          // box centre velocity, px per frame
        long  seen = 0, created = 0;       // frame numbers
        long  real_seen = 0;               // last frame matched to a real (not synthetic) box
        bool  has_pos = false;             // 3D position (pos3d / vmax)
        float pos[3] = {0, 0, 0};          // smoothed camera-space position, m
        float pvel[3] = {0, 0, 0};         // its velocity, m per frame (with motion)
        bool  alive = true;                // false once merged into an older slot
        int   merged_into = -1;            // ... that slot
        int   cand = -1, cand_frames = 0;  // re-entry candidate (older slot) and its streak
    };
    Params            p_;
    std::vector<Slot> slots_;
    long              frame_ = 0;
};

// Offline, revisable association: group person slots (tracklets) into people
// once the whole sequence is known.  frame_slot lists every (frame, slot)
// occurrence.  Slots ever present in the same frame are never merged
// (cannot-link).  The rest are merged by average linkage over their pairwise
// accumulated-colour discrepancy (discrepancy_to, with gain_range), most similar
// first, while the best eligible pair is <= max_disc.  Returns a new id per slot
// (0-based, numbered by first appearance).
std::vector<int> merge_tracklets(const std::vector<SkinColorAccumulator>& acc,
                                 const std::vector<std::pair<int, int>>& frame_slot,
                                 float max_disc, float gain_range = 0.f);

} // namespace fsb
