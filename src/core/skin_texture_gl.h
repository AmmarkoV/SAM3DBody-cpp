#pragma once
// ============================================================================
// skin_texture_gl.h  –  --skin-texture: per-person appearance in a UV texture
//
// The per-vertex model of skin_color.h stores one colour per vertex (~1 cm
// apart on the body).  This accumulates the same weighted mean per TEXEL of a
// UV atlas of the MHR mesh (onnx/body_mesh_uv.bin, made once by
// tools/gen_mhr_uv.py), all on the GPU:
//
//   1. the posed mesh is rendered from the camera into an eye-depth buffer;
//   2. the mesh is rasterised in UV space into the person's float texture:
//      every texel finds its surface point, projects it into the frame,
//      keeps it only if it faces the camera (cos >= 0.3) and is the nearest
//      surface there (depth test, 3 cm tolerance), samples the camera image
//      and ADDS (colour * cos^2, cos^2) by additive blending;
//   3. resolve() turns the sums into colours (unobserved texels fall back to
//      the per-vertex fill), dilates across chart seams and builds mipmaps.
//
// Same visibility rules and weights as SkinObserver, so the two models agree;
// matching (re-ID) keeps using the per-vertex model.  The atlas splits
// vertices along seams into "wedges" that point back at the original vertex,
// so skinning keeps running on the original 18439 vertices.
//
// Needs a current OpenGL 3.3 context (GLEW initialised); renderer only.
// ============================================================================

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace fsb {

class SkinTextureGL {
public:
    // The UV atlas (pure file read, no GL).  False if missing or malformed.
    bool load_atlas(const std::string& path, size_t n_vertices);
    // GL objects; tex_size = texture side in texels.  Call after load_atlas.
    bool init(int tex_size);
    bool ready() const { return prog_acc_ != 0; }

    // Fold one view of `slot`'s person into their texture.
    //   scene_tex : GL texture with the camera frame, RGB, row 0 = top
    //   verts/normals : posed mesh in LBS camera space (original vertices)
    //   mvp       : column-major projection * view of the overlay (mhr_camera_matrices)
    //   cam_t     : MHRResult::pred_cam_t (eye direction for the facing test)
    void accumulate(int slot, unsigned int scene_tex, int img_w, int img_h,
                    const float* verts, const float* normals,
                    const float mvp[16], const float cam_t[3]);

    // RGBA8 mipmapped texture of `slot`: observed texels = their mean, the rest
    // from fallback_rgb (per original vertex, 0-1, e.g. SkinColorAccumulator::
    // colors()).  Alpha is the evidence: unseen_alpha() for texels never
    // observed, rising to 1 once a texel has about one frontal view's weight.
    // 0 if the slot has no texture yet.
    unsigned int resolve(int slot, const float* fallback_rgb);

    // Texture-based appearance discrepancy (--skin-match gpu_tex), the
    // SkinMatchGL tiled render-and-compare with each detection's posed mesh
    // painted from the stored person's accumulated TEXTURE (texel means, only
    // texels observed at least once count) instead of per-vertex colours, in
    // larger tiles (128x256) so the texture detail survives.
    //   det_verts/det_mvp/det_boxes : per detection, as SkinMatchGL::score
    //   slots : which of this object's slots to compare against
    // Returns [n_det x slots.size()] row-major, 0-1, -1 = too little overlap.
    std::vector<float> score(unsigned int scene_tex, int img_w, int img_h,
                             const std::vector<const float*>&          det_verts,
                             const std::vector<std::array<float, 16>>& det_mvp,
                             const std::vector<std::array<float, 4>>&  det_boxes,
                             const std::vector<int>&                   slots);

    // Draw a posed mesh with a resolved texture into the current framebuffer.
    //   mvp       : column-major
    //   normal_sign / light : the normal is multiplied component-wise by
    //               normal_sign before lighting against `light` (both in the
    //               space the caller lights in); shading is the light 0.8..1.0
    //               ramp used for video colours (default.frag).
    //   use_tex_alpha: multiply `alpha` by the texture's evidence alpha (needs
    //               blending on; off = opaque texture).
    void draw(unsigned int tex, const float* verts, const float* normals, const float mvp[16],
              float alpha, const float normal_sign[3], const float light[3], bool use_tex_alpha = false);

    // Alpha of texels never observed (0 = transparent, default).
    void set_unseen_alpha(float a) { unseen_alpha_ = a; }
    float unseen_alpha() const { return unseen_alpha_; }

    // The resolved texture of `slot` as tex_size x tex_size RGBA, row 0 = top
    // (v = 0).  Empty if the slot has none.
    std::vector<uint8_t> read_rgba(int slot);

    int    tex_size() const { return tex_size_; }
    size_t n_wedges() const { return vref_.size(); }
    const std::vector<float>&        uv()      const { return uv_; }       // 2 per wedge, v = 0 top
    const std::vector<unsigned int>& vref()    const { return vref_; }     // original vertex per wedge
    const std::vector<unsigned int>& indices() const { return indices_; }  // 3 per triangle, over wedges

private:
    struct Slot {
        unsigned int acc_tex = 0, acc_fbo = 0;   // RGBA32F sums
        unsigned int out_tex = 0, out_fbo = 0;   // RGBA8 resolved
        bool         dirty = true;
    };
    Slot& slot(int s);
    void  upload_wedges(const float* verts, const float* normals, const float* rgb);

    size_t n_vertices_ = 0;
    int    tex_size_ = 0;
    float  unseen_alpha_ = 0.f;
    std::vector<float>        uv_;
    std::vector<unsigned int> vref_, indices_;
    std::vector<float>        wpos_, wnorm_, wcol_;   // per-wedge scratch
    std::vector<Slot>         slots_;

    unsigned int vao_ = 0, vbo_pos_ = 0, vbo_norm_ = 0, vbo_uv_ = 0, vbo_col_ = 0, ebo_ = 0;
    unsigned int prog_depth_ = 0, prog_acc_ = 0, prog_resolve_ = 0, prog_dilate_ = 0, prog_draw_ = 0;
    unsigned int depth_fbo_ = 0, depth_tex_ = 0, depth_rb_ = 0;
    int          depth_w_ = 0, depth_h_ = 0;
    unsigned int scratch_tex_ = 0, scratch_fbo_ = 0;   // dilation ping-pong
    unsigned int prog_score_ = 0, score_fbo_ = 0, score_tex_ = 0, score_rb_ = 0;   // score() tiles
    unsigned int quad_vao_ = 0;
};

} // namespace fsb
