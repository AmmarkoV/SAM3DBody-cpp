#pragma once
// ============================================================================
// skin_turntable_gl.h  –  rotating debug view of the --skin-color meshes
//
// Renders every person's current posed mesh with their accumulated per-vertex
// colours, side by side, each centred in its own column and spun about the
// vertical axis, into an offscreen framebuffer, and hands the image back.
// Lighting is fixed to the camera so the colours, not the shading, are what
// changes as the body turns — the back fills in as the person is seen from
// more sides.  Used by fast_sam_3dbody_render --skin-turntable.
//
// Needs a current OpenGL 3.3 context (GLEW initialised).
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <vector>

namespace fsb {

class SkinTextureGL;

class SkinTurntableGL {
public:
    // Topology + output size; call once with a current GL context.
    bool init(const unsigned int* indices, size_t n_indices, size_t n_vertices,
              int width, int height);

    // verts: per person, posed vertices [n_vertices x 3] in LBS camera space.
    // rgb  : per person, colours [n_vertices x 3], 0-1.
    // Returns width x height x 3 RGB, rows top to bottom.  Valid until the
    // next call.  No people = a cleared frame, so the video keeps its length.
    const std::vector<uint8_t>& render(const std::vector<const float*>& verts,
                                       const std::vector<const float*>& rgb,
                                       float angle_deg);

    // --skin-texture: draw person p with texture ids[p] (0 = vertex colours)
    // in the next render() calls.
    void set_textures(SkinTextureGL* tex, const std::vector<unsigned int>& ids) { tex_ = tex; tex_ids_ = ids; }
    // Transparent background: render() then returns RGBA (straight alpha, 0 off
    // the bodies; with textures, never-seen parts keep their texture alpha).
    void set_transparent(bool t) { transparent_ = t; }
    int  channels() const { return transparent_ ? 4 : 3; }
    // Background colour, 0-1 (default dark grey).  A key colour such as
    // magenta lets tools/reid_eval/make_viz.py cut the people out.
    void set_background(float r, float g, float b) { bg_[0] = r; bg_[1] = g; bg_[2] = b; }
    int width()  const { return w_; }
    int height() const { return h_; }

private:
    float                     bg_[3] = {0.12f, 0.12f, 0.14f};
    SkinTextureGL*            tex_ = nullptr;
    bool                      transparent_ = false;
    std::vector<unsigned int> tex_ids_;
    size_t                    n_vertices_ = 0;
    unsigned int              n_indices_  = 0;
    std::vector<unsigned int> indices_;
    int                       w_ = 0, h_ = 0;
    unsigned int prog_ = 0, vao_ = 0, ebo_ = 0, pos_vbo_ = 0, norm_vbo_ = 0, col_vbo_ = 0;
    unsigned int fbo_ = 0, color_rb_ = 0, depth_rb_ = 0;
    int          proj_loc_ = -1;
    std::vector<float>   pos_, norm_;   // scratch: centred + rotated mesh
    std::vector<uint8_t> pixels_, row_;
};

} // namespace fsb
