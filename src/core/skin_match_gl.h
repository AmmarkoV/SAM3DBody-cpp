#pragma once
// ============================================================================
// skin_match_gl.h  –  GPU appearance check for --skin-color person matching
//
// Tiled render-and-diff in the style of RGBDAcquisition's
// glx_testMultiViewportDiff.c: every (detection, stored person) pair gets one
// tile of an offscreen float framebuffer.  The tile shows that detection's
// posed mesh, cropped to its bounding box, painted with the stored person's
// per-vertex colours; a diff shader compares every fragment with the camera
// pixel it covers and writes
//     R = |stored - camera| (mean over RGB) x seen,  G = seen,  B = 1
// where "seen" is 1 on vertices the stored person has actually been observed
// at (filled-in guesses are ignored).  Instead of reading the whole
// framebuffer back and summing tiles on the CPU, glGenerateMipmap averages
// each power-of-two tile down to a couple of texels, so only a 16x16 level is
// read back.  Score of a tile = sum R / sum G.
//
// Needs a current OpenGL 3.3 context (GLEW initialised); lives in the GL
// renderer only, the core library stays GL-free.  The CPU equivalent is
// SkinColorAccumulator::discrepancy (skin_color.h).
// ============================================================================

#include <array>
#include <cstddef>
#include <vector>

namespace fsb {

class SkinMatchGL {
public:
    // GL objects are left to process exit, like the renderer's own: the
    // context is gone by the time a destructor would run.

    // Topology only; call once with a current GL context.  False on failure
    // (shader/FBO), in which case the caller should fall back to the CPU path.
    bool init(const unsigned int* indices, size_t n_indices, size_t n_vertices);

    // Colour discrepancy of every detection against every stored person.
    //   scene_tex  : GL texture holding the camera frame as RGB, row 0 = top
    //   img_w/h    : frame size the MVPs project into
    //   det_verts  : per detection, posed vertices [n_vertices x 3]
    //   det_mvp    : per detection, column-major projection * view
    //   det_boxes  : per detection, [x1,y1,x2,y2] pixels
    //   slot_rgba  : per stored person, SkinColorAccumulator::colors_rgba()
    // Returns [n_det x n_slots] row-major, 0-1, -1 = too little overlap to
    // tell (or more pairs than fit in the framebuffer).
    std::vector<float> score(unsigned int scene_tex, int img_w, int img_h,
                             const std::vector<const float*>&           det_verts,
                             const std::vector<std::array<float, 16>>&  det_mvp,
                             const std::vector<std::array<float, 4>>&   det_boxes,
                             const std::vector<const float*>&           slot_rgba);

private:
    size_t       n_vertices_ = 0;
    unsigned int n_indices_  = 0;
    unsigned int prog_ = 0, vao_ = 0, ebo_ = 0;
    unsigned int fbo_ = 0, color_tex_ = 0, depth_rb_ = 0;
    int          mvp_loc_ = -1, crop_loc_ = -1, scene_loc_ = -1;
    std::vector<unsigned int> pos_vbo_;   // one per detection, grown on demand
    std::vector<unsigned int> col_vbo_;   // one per stored person
};

} // namespace fsb
