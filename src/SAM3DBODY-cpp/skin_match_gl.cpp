// skin_match_gl.cpp  –  see skin_match_gl.h
#include "skin_match_gl.h"

#include <GL/glew.h>

#include <algorithm>
#include <cstdio>

namespace fsb {

// Framebuffer of TILES_X x TILES_Y tiles.  Everything is a power of two so
// that mip level REDUCE_LEVEL collapses each tile to exactly
// (TILE_W >> REDUCE_LEVEL) x (TILE_H >> REDUCE_LEVEL) = 1 x 2 texels.
// Tiles are tall because people are; a box is stretched to fill its tile,
// which is fine since both sides of the comparison are stretched alike.
static constexpr int FB_SIZE      = 1024;
static constexpr int TILE_W       = 64;
static constexpr int TILE_H       = 128;
static constexpr int REDUCE_LEVEL = 6;
static constexpr int TILES_X      = FB_SIZE / TILE_W;   // 16
static constexpr int TILES_Y      = FB_SIZE / TILE_H;   // 8
static constexpr int RED_SIZE     = FB_SIZE >> REDUCE_LEVEL;   // 16
static constexpr int RED_TILE_H   = TILE_H >> REDUCE_LEVEL;    // 2
// A tile needs this fraction of its pixels on already-observed skin to count.
static constexpr float MIN_SEEN_FRACTION = 0.02f;

static const char* kVert = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=2) in vec4 aColor;    // stored colour, a = observed
uniform mat4 uMVP;                    // full-frame projection * view
uniform vec4 uCrop;                   // box -> tile: x' = x*sx + ox*w, y' = y*sy + oy*w
out vec4 vFrameClip;
out vec4 vColor;
void main() {
    vec4 c = uMVP * vec4(aPos, 1.0);
    vFrameClip = c;
    vColor     = aColor;
    gl_Position = vec4(c.x * uCrop.x + uCrop.y * c.w,
                       c.y * uCrop.z + uCrop.w * c.w, c.z, c.w);
}
)";

static const char* kFrag = R"(
#version 330 core
in vec4 vFrameClip;
in vec4 vColor;
uniform sampler2D uScene;             // camera frame, RGB, row 0 = top
out vec4 fragColor;
void main() {
    vec2 ndc = vFrameClip.xy / vFrameClip.w;
    vec3 cam = texture(uScene, vec2(0.5 * (ndc.x + 1.0), 0.5 * (1.0 - ndc.y))).rgb;
    float seen = vColor.a;
    float d    = dot(abs(vColor.rgb - cam), vec3(1.0 / 3.0));
    fragColor  = vec4(d * seen, seen, 1.0, 0.0);
}
)";

static GLuint compile(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024] = {};
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        fprintf(stderr, "[skin-match] shader: %s\n", log);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

bool SkinMatchGL::init(const unsigned int* indices, size_t n_indices, size_t n_vertices)
{
    n_vertices_ = n_vertices;
    n_indices_  = (unsigned int)((n_indices / 3) * 3);

    GLuint vs = compile(GL_VERTEX_SHADER, kVert), fs = compile(GL_FRAGMENT_SHADER, kFrag);
    if (!vs || !fs) return false;
    prog_ = glCreateProgram();
    glAttachShader(prog_, vs);
    glAttachShader(prog_, fs);
    glLinkProgram(prog_);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = 0;
    glGetProgramiv(prog_, GL_LINK_STATUS, &ok);
    if (!ok) { fprintf(stderr, "[skin-match] program link failed\n"); return false; }
    mvp_loc_   = glGetUniformLocation(prog_, "uMVP");
    crop_loc_  = glGetUniformLocation(prog_, "uCrop");
    scene_loc_ = glGetUniformLocation(prog_, "uScene");

    glGenVertexArrays(1, &vao_);
    glBindVertexArray(vao_);
    glGenBuffers(1, &ebo_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, n_indices_ * sizeof(unsigned int), indices, GL_STATIC_DRAW);
    glBindVertexArray(0);

    // Float colour target (sums must not saturate at 8 bits) with a mip chain
    // for the tile reduction, plus depth so only the front surface is diffed.
    glGenTextures(1, &color_tex_);
    glBindTexture(GL_TEXTURE_2D, color_tex_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, FB_SIZE, FB_SIZE, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glGenerateMipmap(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenRenderbuffers(1, &depth_rb_);
    glBindRenderbuffer(GL_RENDERBUFFER, depth_rb_);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, FB_SIZE, FB_SIZE);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    GLint prev_fbo = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);
    glGenFramebuffers(1, &fbo_);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color_tex_, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rb_);
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)prev_fbo);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "[skin-match] framebuffer incomplete (0x%04X)\n", status);
        return false;
    }
    return true;
}

std::vector<float> SkinMatchGL::score(unsigned int scene_tex, int img_w, int img_h,
                                      const std::vector<const float*>&          det_verts,
                                      const std::vector<std::array<float, 16>>& det_mvp,
                                      const std::vector<std::array<float, 4>>&  det_boxes,
                                      const std::vector<const float*>&          slot_rgba)
{
    const size_t D = det_verts.size(), S = slot_rgba.size();
    std::vector<float> out(D * S, -1.f);
    if (!prog_ || !D || !S || img_w <= 0 || img_h <= 0) return out;
    const size_t n_pairs = std::min(D * S, (size_t)(TILES_X * TILES_Y));

    // ── Save the state we touch ──────────────────────────────────────────────
    GLint prev_fbo = 0, prev_prog = 0, prev_vao = 0, prev_vp[4], prev_active = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);
    glGetIntegerv(GL_CURRENT_PROGRAM, &prev_prog);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prev_vao);
    glGetIntegerv(GL_VIEWPORT, prev_vp);
    glGetIntegerv(GL_ACTIVE_TEXTURE, &prev_active);
    const GLboolean prev_blend = glIsEnabled(GL_BLEND), prev_depth = glIsEnabled(GL_DEPTH_TEST),
                    prev_cull  = glIsEnabled(GL_CULL_FACE), prev_scissor = glIsEnabled(GL_SCISSOR_TEST);

    // ── Upload this frame's meshes and colours ───────────────────────────────
    auto grow = [](std::vector<unsigned int>& v, size_t n) {
        while (v.size() < n) { GLuint b = 0; glGenBuffers(1, &b); v.push_back(b); }
    };
    grow(pos_vbo_, D);
    grow(col_vbo_, S);
    for (size_t d = 0; d < D; ++d) {
        glBindBuffer(GL_ARRAY_BUFFER, pos_vbo_[d]);
        glBufferData(GL_ARRAY_BUFFER, n_vertices_ * 3 * sizeof(float), det_verts[d], GL_STREAM_DRAW);
    }
    for (size_t s = 0; s < S; ++s) {
        glBindBuffer(GL_ARRAY_BUFFER, col_vbo_[s]);
        glBufferData(GL_ARRAY_BUFFER, n_vertices_ * 4 * sizeof(float), slot_rgba[s], GL_STREAM_DRAW);
    }

    // ── Render one tile per (detection, slot) pair ───────────────────────────
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glViewport(0, 0, FB_SIZE, FB_SIZE);
    glDisable(GL_SCISSOR_TEST);
    glClearColor(0.f, 0.f, 0.f, 0.f);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glEnable(GL_SCISSOR_TEST);

    glUseProgram(prog_);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, scene_tex);
    glUniform1i(scene_loc_, 2);
    glBindVertexArray(vao_);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(2);

    for (size_t k = 0; k < n_pairs; ++k) {
        const size_t d = k / S, s = k % S;
        const int tx = (int)(k % TILES_X) * TILE_W, ty = (int)(k / TILES_X) * TILE_H;
        glViewport(tx, ty, TILE_W, TILE_H);
        glScissor (tx, ty, TILE_W, TILE_H);

        // Box (pixels, y down) -> NDC of the full frame -> [-1,1] of the tile.
        const auto& b = det_boxes[d];
        float nx1 = 2.f * b[0] / img_w - 1.f, nx2 = 2.f * b[2] / img_w - 1.f;
        float ny1 = 1.f - 2.f * b[3] / img_h, ny2 = 1.f - 2.f * b[1] / img_h;   // bottom, top
        if (nx2 - nx1 < 1e-4f || ny2 - ny1 < 1e-4f) continue;
        glUniform4f(crop_loc_, 2.f / (nx2 - nx1), -(nx1 + nx2) / (nx2 - nx1),
                               2.f / (ny2 - ny1), -(ny1 + ny2) / (ny2 - ny1));
        glUniformMatrix4fv(mvp_loc_, 1, GL_FALSE, det_mvp[d].data());

        glBindBuffer(GL_ARRAY_BUFFER, pos_vbo_[d]);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
        glBindBuffer(GL_ARRAY_BUFFER, col_vbo_[s]);
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
        glDrawElements(GL_TRIANGLES, (GLsizei)n_indices_, GL_UNSIGNED_INT, nullptr);
    }

    // ── Reduce: mip level REDUCE_LEVEL holds each tile's averages ───────────
    static std::vector<float> red((size_t)RED_SIZE * RED_SIZE * 4);
    glBindTexture(GL_TEXTURE_2D, color_tex_);
    glGenerateMipmap(GL_TEXTURE_2D);
    glGetTexImage(GL_TEXTURE_2D, REDUCE_LEVEL, GL_RGBA, GL_FLOAT, red.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    for (size_t k = 0; k < n_pairs; ++k) {
        const int rx = (int)(k % TILES_X), ry = (int)(k / TILES_X) * RED_TILE_H;
        float diff = 0.f, seen = 0.f;
        for (int j = 0; j < RED_TILE_H; ++j) {
            const float* t = &red[((size_t)(ry + j) * RED_SIZE + rx) * 4];
            diff += t[0];
            seen += t[1];
        }
        if (seen / RED_TILE_H >= MIN_SEEN_FRACTION) out[k] = diff / seen;
    }

    // ── Restore ──────────────────────────────────────────────────────────────
    glBindVertexArray((GLuint)prev_vao);
    glUseProgram((GLuint)prev_prog);
    glActiveTexture((GLenum)prev_active);
    glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)prev_fbo);
    glViewport(prev_vp[0], prev_vp[1], prev_vp[2], prev_vp[3]);
    if (prev_blend)   glEnable(GL_BLEND);        else glDisable(GL_BLEND);
    if (prev_depth)   glEnable(GL_DEPTH_TEST);   else glDisable(GL_DEPTH_TEST);
    if (prev_cull)    glEnable(GL_CULL_FACE);    else glDisable(GL_CULL_FACE);
    if (prev_scissor) glEnable(GL_SCISSOR_TEST); else glDisable(GL_SCISSOR_TEST);
    return out;
}

} // namespace fsb
