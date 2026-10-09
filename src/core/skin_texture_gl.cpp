// skin_texture_gl.cpp  –  see skin_texture_gl.h
#include "skin_texture_gl.h"

#include <GL/glew.h>

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace fsb {

// Same tolerances as SkinObserver (skin_color.cpp).
static constexpr float DEPTH_EPS     = 0.03f;   // metres
static constexpr float MIN_FACING    = 0.3f;
static constexpr int   DEPTH_MAX_SIDE = 1024;   // eye-depth buffer resolution cap
static constexpr int   DILATE_PASSES = 4;       // even: the result ends in out_tex
// Accumulated weight at which a texel counts as fully seen: about one frontal
// view (weights are cos^2 of the view angle).
static constexpr float FULL_WEIGHT   = 1.0f;

// ── Shaders ──────────────────────────────────────────────────────────────────
static const char* kDepthVert = R"(
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 uMVP;
out float vW;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); vW = gl_Position.w; }
)";
static const char* kDepthFrag = R"(
#version 330 core
in float vW;
out vec4 o;
void main() { o = vec4(vW, 0.0, 0.0, 1.0); }   // eye distance (clip w)
)";

// Rasterise in UV space; every fragment is a texel's surface point.
static const char* kAccVert = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNorm;
layout(location=3) in vec2 aUV;
uniform mat4 uMVP;
out vec4 vClip;
out vec3 vPos;
out vec3 vNorm;
void main() {
    gl_Position = vec4(aUV.x * 2.0 - 1.0, (1.0 - aUV.y) * 2.0 - 1.0, 0.0, 1.0);
    vClip = uMVP * vec4(aPos, 1.0);
    vPos  = aPos;
    vNorm = aNorm;
}
)";
static const char* kAccFrag = R"(
#version 330 core
in vec4 vClip;
in vec3 vPos;
in vec3 vNorm;
uniform sampler2D uScene;   // camera frame, RGB, row 0 = top
uniform sampler2D uDepth;   // eye distance of the nearest surface, rendered with uMVP
uniform vec3  uCamT;
uniform float uMinFacing;
uniform float uDepthEps;
out vec4 o;
void main() {
    if (vClip.w <= 1e-4) discard;
    vec2 ndc = vClip.xy / vClip.w;
    if (abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0) discard;
    float facing = dot(normalize(vNorm), normalize(-(vPos + uCamT)));
    if (facing < uMinFacing) discard;
    if (vClip.w > texture(uDepth, ndc * 0.5 + 0.5).r + uDepthEps) discard;   // occluded
    vec3 c = texture(uScene, vec2(0.5 * (ndc.x + 1.0), 0.5 * (1.0 - ndc.y))).rgb;
    float w = facing * facing;
    o = vec4(c * w, w);
}
)";

// Sums -> colours; texels never observed take the per-vertex fill.  Alpha is
// the evidence: uUnseen for texels never observed, ramping to 1 as their
// accumulated weight reaches uFullWeight.  Never below 1/255, because alpha 0
// marks "outside every chart" for the dilation.
static const char* kResolveVert = R"(
#version 330 core
layout(location=2) in vec3 aCol;
layout(location=3) in vec2 aUV;
out vec3 vCol;
void main() {
    gl_Position = vec4(aUV.x * 2.0 - 1.0, (1.0 - aUV.y) * 2.0 - 1.0, 0.0, 1.0);
    vCol = aCol;
}
)";
static const char* kResolveFrag = R"(
#version 330 core
in vec3 vCol;
uniform sampler2D uAcc;
uniform float uUnseen;
uniform float uFullWeight;
out vec4 o;
void main() {
    vec4 a = texelFetch(uAcc, ivec2(gl_FragCoord.xy), 0);
    float alpha = max(mix(uUnseen, 1.0, clamp(a.a / uFullWeight, 0.0, 1.0)), 1.0 / 255.0);
    o = vec4(a.a > 1e-6 ? a.rgb / a.a : vCol, alpha);
}
)";

// Full-screen pass: empty texels (alpha 0, outside every chart) take the mean
// of their filled neighbours, so bilinear and mip sampling never pull in black.
static const char* kQuadVert = R"(
#version 330 core
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)";
static const char* kDilateFrag = R"(
#version 330 core
uniform sampler2D uSrc;
out vec4 o;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 c = texelFetch(uSrc, p, 0);
    if (c.a > 0.0) { o = c; return; }
    ivec2 sz = textureSize(uSrc, 0);
    vec4 sum = vec4(0.0); float n = 0.0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx) {
            ivec2 q = clamp(p + ivec2(dx, dy), ivec2(0), sz - 1);
            vec4 s = texelFetch(uSrc, q, 0);
            if (s.a > 0.0) { sum += s; n += 1.0; }
        }
    o = n > 0.0 ? sum / n : vec4(0.0);
}
)";

static const char* kDrawVert = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNorm;
layout(location=3) in vec2 aUV;
uniform mat4 uMVP;
out vec3 vNorm;
out vec2 vUV;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); vNorm = aNorm; vUV = aUV; }
)";
static const char* kDrawFrag = R"(
#version 330 core
in vec3 vNorm;
in vec2 vUV;
uniform sampler2D uTex;
uniform vec3  uNSign;
uniform vec3  uLight;
uniform float uAlpha;
uniform float uUseTexAlpha;
out vec4 o;
void main() {
    // default.frag's wrap lighting, in its light variant for video colours (0.8..1.0)
    vec3  N = normalize(vNorm * uNSign);
    float d = clamp((dot(N, normalize(uLight)) + 0.15) / 1.15, 0.0, 1.0);
    d = d * d * (3.0 - 2.0 * d);
    d = mix(1.0, d * 0.65 + 0.35, 0.3);
    vec4 t = texture(uTex, vec2(vUV.x, 1.0 - vUV.y));
    float dither = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
    o = vec4(t.rgb * d + (dither - 0.5) / 255.0, uAlpha * mix(1.0, t.a, uUseTexAlpha));
}
)";

// score(): tiled render-and-compare against the accumulated textures.  Each
// detection's posed mesh is cropped to its box and stretched into a tile; a
// texel counts once it has been observed.  Tiles are larger than SkinMatchGL's
// (128x256: ~7 mm per pixel on an adult) so texture detail is compared.
static constexpr int   SC_FB      = 1024;
static constexpr int   SC_TILE_W  = 128;
static constexpr int   SC_TILE_H  = 256;
static constexpr int   SC_REDUCE  = 7;                       // mip level: 1 x 2 texels per tile
static constexpr int   SC_TILES_X = SC_FB / SC_TILE_W;       // 8
static constexpr int   SC_TILES_Y = SC_FB / SC_TILE_H;       // 4
static constexpr int   SC_RED     = SC_FB >> SC_REDUCE;      // 8
static constexpr int   SC_RED_TH  = SC_TILE_H >> SC_REDUCE;  // 2
static constexpr float SC_MIN_SEEN = 0.02f;                  // as SkinMatchGL

static const char* kScoreVert = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=3) in vec2 aUV;
uniform mat4 uMVP;                    // full-frame projection * view
uniform vec4 uCrop;                   // box -> tile
out vec4 vFrameClip;
out vec2 vUV;
void main() {
    vec4 c = uMVP * vec4(aPos, 1.0);
    vFrameClip = c;
    vUV = aUV;
    gl_Position = vec4(c.x * uCrop.x + uCrop.y * c.w, c.y * uCrop.z + uCrop.w * c.w, c.z, c.w);
}
)";
static const char* kScoreFrag = R"(
#version 330 core
in vec4 vFrameClip;
in vec2 vUV;
uniform sampler2D uScene;             // camera frame, RGB, row 0 = top
uniform sampler2D uAcc;               // the stored person's sums (rgb * w, w), linear filtered
out vec4 o;
void main() {
    vec2 ndc = vFrameClip.xy / vFrameClip.w;
    vec3 cam = texture(uScene, vec2(0.5 * (ndc.x + 1.0), 0.5 * (1.0 - ndc.y))).rgb;
    vec4 a = texture(uAcc, vec2(vUV.x, 1.0 - vUV.y));   // filtered sums: their ratio is a weighted mean
    float seen = a.a > 1e-4 ? 1.0 : 0.0;
    vec3 mean = seen > 0.0 ? a.rgb / a.a : cam;
    o = vec4(dot(abs(mean - cam), vec3(1.0 / 3.0)) * seen, seen, 1.0, 0.0);
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
        fprintf(stderr, "[skin-texture] shader: %s\n", log);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

static GLuint program(const char* vs_src, const char* fs_src)
{
    GLuint vs = compile(GL_VERTEX_SHADER, vs_src), fs = compile(GL_FRAGMENT_SHADER, fs_src);
    if (!vs || !fs) return 0;
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    glDeleteShader(vs);
    glDeleteShader(fs);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { fprintf(stderr, "[skin-texture] program link failed\n"); glDeleteProgram(p); return 0; }
    return p;
}

static GLuint make_tex(GLenum internal, GLenum format, GLenum type, int w, int h, bool mip)
{
    GLuint t = 0;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, internal, w, h, 0, format, type, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, mip ? GL_LINEAR_MIPMAP_LINEAR : GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mip ? GL_LINEAR : GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    if (mip) glGenerateMipmap(GL_TEXTURE_2D);
    return t;
}

static GLuint make_fbo(GLuint color, GLuint depth_rb = 0)
{
    GLuint f = 0;
    glGenFramebuffers(1, &f);
    glBindFramebuffer(GL_FRAMEBUFFER, f);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
    if (depth_rb) glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rb);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "[skin-texture] framebuffer incomplete\n");
        glDeleteFramebuffers(1, &f);
        f = 0;
    }
    return f;
}

// Saves and restores the GL state these passes touch.
struct GLStateGuard {
    GLint fbo = 0, prog = 0, vao = 0, vp[4] = {0, 0, 0, 0}, active = 0, tex3 = 0, tex4 = 0;
    GLint blend_src = 0, blend_dst = 0;
    GLboolean blend, depth, cull, scissor;
    GLfloat clear[4];
    GLStateGuard() {
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &fbo);
        glGetIntegerv(GL_CURRENT_PROGRAM, &prog);
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &vao);
        glGetIntegerv(GL_VIEWPORT, vp);
        glGetIntegerv(GL_ACTIVE_TEXTURE, &active);
        glActiveTexture(GL_TEXTURE3); glGetIntegerv(GL_TEXTURE_BINDING_2D, &tex3);
        glActiveTexture(GL_TEXTURE4); glGetIntegerv(GL_TEXTURE_BINDING_2D, &tex4);
        glGetIntegerv(GL_BLEND_SRC_RGB, &blend_src);
        glGetIntegerv(GL_BLEND_DST_RGB, &blend_dst);
        glGetFloatv(GL_COLOR_CLEAR_VALUE, clear);
        blend = glIsEnabled(GL_BLEND); depth = glIsEnabled(GL_DEPTH_TEST);
        cull = glIsEnabled(GL_CULL_FACE); scissor = glIsEnabled(GL_SCISSOR_TEST);
    }
    ~GLStateGuard() {
        glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, (GLuint)tex3);
        glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D, (GLuint)tex4);
        glActiveTexture((GLenum)active);
        glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)fbo);
        glUseProgram((GLuint)prog);
        glBindVertexArray((GLuint)vao);
        glViewport(vp[0], vp[1], vp[2], vp[3]);
        glBlendFunc((GLenum)blend_src, (GLenum)blend_dst);
        glClearColor(clear[0], clear[1], clear[2], clear[3]);
        if (blend)   glEnable(GL_BLEND);        else glDisable(GL_BLEND);
        if (depth)   glEnable(GL_DEPTH_TEST);   else glDisable(GL_DEPTH_TEST);
        if (cull)    glEnable(GL_CULL_FACE);    else glDisable(GL_CULL_FACE);
        if (scissor) glEnable(GL_SCISSOR_TEST); else glDisable(GL_SCISSOR_TEST);
    }
};

// ── Atlas ────────────────────────────────────────────────────────────────────
bool SkinTextureGL::load_atlas(const std::string& path, size_t n_vertices)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[8];
    uint32_t hdr[4];
    bool ok = fread(magic, 1, 8, f) == 8 && !memcmp(magic, "MHRUV001", 8) &&
              fread(hdr, sizeof(uint32_t), 4, f) == 4;
    if (ok) {
        const size_t nw = hdr[0], nt = hdr[1];
        uv_.resize(nw * 2);
        vref_.resize(nw);
        indices_.resize(nt * 3);
        ok = fread(uv_.data(), sizeof(float), uv_.size(), f) == uv_.size() &&
             fread(vref_.data(), sizeof(uint32_t), vref_.size(), f) == vref_.size() &&
             fread(indices_.data(), sizeof(uint32_t), indices_.size(), f) == indices_.size();
        for (size_t k = 0; ok && k < nw; ++k) ok = vref_[k] < n_vertices;
        for (size_t k = 0; ok && k < indices_.size(); ++k) ok = indices_[k] < nw;
    }
    fclose(f);
    if (!ok) { uv_.clear(); vref_.clear(); indices_.clear(); return false; }
    n_vertices_ = n_vertices;
    return true;
}

bool SkinTextureGL::init(int tex_size)
{
    if (vref_.empty()) return false;
    tex_size_ = tex_size;
    prog_depth_   = program(kDepthVert, kDepthFrag);
    prog_acc_     = program(kAccVert, kAccFrag);
    prog_resolve_ = program(kResolveVert, kResolveFrag);
    prog_dilate_  = program(kQuadVert, kDilateFrag);
    prog_draw_    = program(kDrawVert, kDrawFrag);
    prog_score_   = program(kScoreVert, kScoreFrag);
    if (!prog_depth_ || !prog_acc_ || !prog_resolve_ || !prog_dilate_ || !prog_draw_ || !prog_score_) {
        prog_acc_ = 0;
        return false;
    }
    GLint prev_vao = 0;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prev_vao);
    glGenVertexArrays(1, &vao_);
    glBindVertexArray(vao_);
    const size_t nw = vref_.size();
    auto vbo = [&](GLuint& b, GLuint loc, int comps, const void* data, GLenum usage) {
        glGenBuffers(1, &b);
        glBindBuffer(GL_ARRAY_BUFFER, b);
        glBufferData(GL_ARRAY_BUFFER, nw * comps * sizeof(float), data, usage);
        glEnableVertexAttribArray(loc);
        glVertexAttribPointer(loc, comps, GL_FLOAT, GL_FALSE, 0, nullptr);
    };
    vbo(vbo_pos_,  0, 3, nullptr,    GL_DYNAMIC_DRAW);
    vbo(vbo_norm_, 1, 3, nullptr,    GL_DYNAMIC_DRAW);
    vbo(vbo_col_,  2, 3, nullptr,    GL_DYNAMIC_DRAW);
    vbo(vbo_uv_,   3, 2, uv_.data(), GL_STATIC_DRAW);
    glGenBuffers(1, &ebo_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_.size() * sizeof(unsigned int), indices_.data(), GL_STATIC_DRAW);
    glGenVertexArrays(1, &quad_vao_);
    glBindVertexArray((GLuint)prev_vao);

    GLint prev_fbo = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);
    scratch_tex_ = make_tex(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, tex_size_, tex_size_, false);
    scratch_fbo_ = make_fbo(scratch_tex_);
    glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)prev_fbo);
    glBindTexture(GL_TEXTURE_2D, 0);
    wpos_.resize(nw * 3);
    wnorm_.resize(nw * 3);
    wcol_.resize(nw * 3);
    return scratch_fbo_ != 0;
}

SkinTextureGL::Slot& SkinTextureGL::slot(int s)
{
    if ((int)slots_.size() <= s) slots_.resize(s + 1);
    Slot& sl = slots_[s];
    if (!sl.acc_fbo) {
        GLint prev_fbo = 0;
        glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);
        sl.acc_tex = make_tex(GL_RGBA32F, GL_RGBA, GL_FLOAT, tex_size_, tex_size_, false);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);   // score() samples it filtered
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        sl.acc_fbo = make_fbo(sl.acc_tex);
        sl.out_tex = make_tex(GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, tex_size_, tex_size_, true);
        sl.out_fbo = make_fbo(sl.out_tex);
        glBindFramebuffer(GL_FRAMEBUFFER, sl.acc_fbo);
        glClearColor(0.f, 0.f, 0.f, 0.f);
        glClear(GL_COLOR_BUFFER_BIT);
        glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)prev_fbo);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    return sl;
}

void SkinTextureGL::upload_wedges(const float* verts, const float* normals, const float* rgb)
{
    const size_t nw = vref_.size();
    for (size_t k = 0; k < nw; ++k) {
        const size_t v = vref_[k];
        for (int c = 0; c < 3; ++c) {
            if (verts)   wpos_[k*3 + c]  = verts[v*3 + c];
            if (normals) wnorm_[k*3 + c] = normals[v*3 + c];
            if (rgb)     wcol_[k*3 + c]  = rgb[v*3 + c];
        }
    }
    if (verts)   { glBindBuffer(GL_ARRAY_BUFFER, vbo_pos_);  glBufferSubData(GL_ARRAY_BUFFER, 0, nw * 3 * sizeof(float), wpos_.data()); }
    if (normals) { glBindBuffer(GL_ARRAY_BUFFER, vbo_norm_); glBufferSubData(GL_ARRAY_BUFFER, 0, nw * 3 * sizeof(float), wnorm_.data()); }
    if (rgb)     { glBindBuffer(GL_ARRAY_BUFFER, vbo_col_);  glBufferSubData(GL_ARRAY_BUFFER, 0, nw * 3 * sizeof(float), wcol_.data()); }
}

// ── Accumulation ─────────────────────────────────────────────────────────────
void SkinTextureGL::accumulate(int s, unsigned int scene_tex, int img_w, int img_h,
                               const float* verts, const float* normals,
                               const float mvp[16], const float cam_t[3])
{
    if (!ready() || s < 0 || img_w <= 0 || img_h <= 0) return;
    GLStateGuard guard;
    Slot& sl = slot(s);

    // (1) eye-depth buffer at (capped) frame resolution
    const float sc = std::min(1.f, (float)DEPTH_MAX_SIDE / (float)std::max(img_w, img_h));
    const int dw = std::max(1, (int)(img_w * sc)), dh = std::max(1, (int)(img_h * sc));
    if (dw != depth_w_ || dh != depth_h_) {
        if (depth_fbo_) { glDeleteFramebuffers(1, &depth_fbo_); glDeleteTextures(1, &depth_tex_); glDeleteRenderbuffers(1, &depth_rb_); }
        depth_tex_ = make_tex(GL_R32F, GL_RED, GL_FLOAT, dw, dh, false);
        glGenRenderbuffers(1, &depth_rb_);
        glBindRenderbuffer(GL_RENDERBUFFER, depth_rb_);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, dw, dh);
        depth_fbo_ = make_fbo(depth_tex_, depth_rb_);
        depth_w_ = dw; depth_h_ = dh;
    }
    upload_wedges(verts, normals, nullptr);
    glBindVertexArray(vao_);
    glDisable(GL_BLEND);
    glDisable(GL_CULL_FACE);
    glDisable(GL_SCISSOR_TEST);

    glBindFramebuffer(GL_FRAMEBUFFER, depth_fbo_);
    glViewport(0, 0, dw, dh);
    glClearColor(1e9f, 0.f, 0.f, 1.f);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glUseProgram(prog_depth_);
    glUniformMatrix4fv(glGetUniformLocation(prog_depth_, "uMVP"), 1, GL_FALSE, mvp);
    glDrawElements(GL_TRIANGLES, (GLsizei)indices_.size(), GL_UNSIGNED_INT, nullptr);

    // (2) UV-space pass, additive into the slot's sums
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glBindFramebuffer(GL_FRAMEBUFFER, sl.acc_fbo);
    glViewport(0, 0, tex_size_, tex_size_);
    glUseProgram(prog_acc_);
    glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, scene_tex);
    glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D, depth_tex_);
    glUniform1i(glGetUniformLocation(prog_acc_, "uScene"), 3);
    glUniform1i(glGetUniformLocation(prog_acc_, "uDepth"), 4);
    glUniformMatrix4fv(glGetUniformLocation(prog_acc_, "uMVP"), 1, GL_FALSE, mvp);
    glUniform3fv(glGetUniformLocation(prog_acc_, "uCamT"), 1, cam_t);
    glUniform1f(glGetUniformLocation(prog_acc_, "uMinFacing"), MIN_FACING);
    glUniform1f(glGetUniformLocation(prog_acc_, "uDepthEps"), DEPTH_EPS);
    glDrawElements(GL_TRIANGLES, (GLsizei)indices_.size(), GL_UNSIGNED_INT, nullptr);
    sl.dirty = true;
}

// ── Resolve ──────────────────────────────────────────────────────────────────
unsigned int SkinTextureGL::resolve(int s, const float* fallback_rgb)
{
    if (!ready() || s < 0 || s >= (int)slots_.size() || !slots_[s].acc_fbo) return 0;
    Slot& sl = slots_[s];
    if (!sl.dirty) return sl.out_tex;
    GLStateGuard guard;
    glDisable(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_SCISSOR_TEST);
    glViewport(0, 0, tex_size_, tex_size_);

    upload_wedges(nullptr, nullptr, fallback_rgb);
    glBindFramebuffer(GL_FRAMEBUFFER, sl.out_fbo);
    glClearColor(0.f, 0.f, 0.f, 0.f);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(prog_resolve_);
    glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, sl.acc_tex);
    glUniform1i(glGetUniformLocation(prog_resolve_, "uAcc"), 3);
    glUniform1f(glGetUniformLocation(prog_resolve_, "uUnseen"), unseen_alpha_);
    glUniform1f(glGetUniformLocation(prog_resolve_, "uFullWeight"), FULL_WEIGHT);
    glBindVertexArray(vao_);
    glDrawElements(GL_TRIANGLES, (GLsizei)indices_.size(), GL_UNSIGNED_INT, nullptr);

    // dilate across the chart borders: out -> scratch -> out ...
    glUseProgram(prog_dilate_);
    glUniform1i(glGetUniformLocation(prog_dilate_, "uSrc"), 3);
    glBindVertexArray(quad_vao_);
    for (int p = 0; p < DILATE_PASSES; ++p) {
        const bool to_scratch = (p % 2) == 0;
        glBindFramebuffer(GL_FRAMEBUFFER, to_scratch ? scratch_fbo_ : sl.out_fbo);
        glActiveTexture(GL_TEXTURE3);
        glBindTexture(GL_TEXTURE_2D, to_scratch ? sl.out_tex : scratch_tex_);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    glBindTexture(GL_TEXTURE_2D, sl.out_tex);
    glGenerateMipmap(GL_TEXTURE_2D);
    sl.dirty = false;
    return sl.out_tex;
}

// ── Texture-based matching ───────────────────────────────────────────────────
std::vector<float> SkinTextureGL::score(unsigned int scene_tex, int img_w, int img_h,
                                        const std::vector<const float*>&          det_verts,
                                        const std::vector<std::array<float, 16>>& det_mvp,
                                        const std::vector<std::array<float, 4>>&  det_boxes,
                                        const std::vector<int>&                   slots)
{
    const size_t D = det_verts.size(), S = slots.size();
    std::vector<float> out(D * S, -1.f);
    if (!ready() || !D || !S || img_w <= 0 || img_h <= 0) return out;
    GLStateGuard guard;
    if (!score_fbo_) {
        score_tex_ = make_tex(GL_RGBA32F, GL_RGBA, GL_FLOAT, SC_FB, SC_FB, false);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
        glGenerateMipmap(GL_TEXTURE_2D);
        glGenRenderbuffers(1, &score_rb_);
        glBindRenderbuffer(GL_RENDERBUFFER, score_rb_);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, SC_FB, SC_FB);
        score_fbo_ = make_fbo(score_tex_, score_rb_);
        if (!score_fbo_) return out;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, score_fbo_);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glUseProgram(prog_score_);
    glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_2D, scene_tex);
    glUniform1i(glGetUniformLocation(prog_score_, "uScene"), 3);
    glUniform1i(glGetUniformLocation(prog_score_, "uAcc"), 4);
    const GLint mvp_loc = glGetUniformLocation(prog_score_, "uMVP");
    const GLint crop_loc = glGetUniformLocation(prog_score_, "uCrop");
    glBindVertexArray(vao_);

    static std::vector<float> red((size_t)SC_RED * SC_RED * 4);
    const size_t cap = (size_t)SC_TILES_X * SC_TILES_Y, total = D * S;
    for (size_t base = 0; base < total; base += cap) {
        const size_t n = std::min(cap, total - base);
        glDisable(GL_SCISSOR_TEST);
        glViewport(0, 0, SC_FB, SC_FB);
        glClearColor(0.f, 0.f, 0.f, 0.f);
        glClearDepth(1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_SCISSOR_TEST);
        size_t uploaded = (size_t)-1;
        for (size_t k = 0; k < n; ++k) {
            const size_t d = (base + k) / S, s = (base + k) % S;
            const int sk = slots[s];
            if (sk < 0 || sk >= (int)slots_.size() || !slots_[sk].acc_tex) continue;
            const auto& b = det_boxes[d];
            float nx1 = 2.f * b[0] / img_w - 1.f, nx2 = 2.f * b[2] / img_w - 1.f;
            float ny1 = 1.f - 2.f * b[3] / img_h, ny2 = 1.f - 2.f * b[1] / img_h;
            if (nx2 - nx1 < 1e-4f || ny2 - ny1 < 1e-4f) continue;
            if (uploaded != d) { upload_wedges(det_verts[d], nullptr, nullptr); uploaded = d; }
            const int tx = (int)(k % SC_TILES_X) * SC_TILE_W, ty = (int)(k / SC_TILES_X) * SC_TILE_H;
            glViewport(tx, ty, SC_TILE_W, SC_TILE_H);
            glScissor (tx, ty, SC_TILE_W, SC_TILE_H);
            glUniform4f(crop_loc, 2.f / (nx2 - nx1), -(nx1 + nx2) / (nx2 - nx1),
                                  2.f / (ny2 - ny1), -(ny1 + ny2) / (ny2 - ny1));
            glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, det_mvp[d].data());
            glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D, slots_[sk].acc_tex);
            glDrawElements(GL_TRIANGLES, (GLsizei)indices_.size(), GL_UNSIGNED_INT, nullptr);
        }
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, score_tex_);
        glGenerateMipmap(GL_TEXTURE_2D);
        glGetTexImage(GL_TEXTURE_2D, SC_REDUCE, GL_RGBA, GL_FLOAT, red.data());
        for (size_t k = 0; k < n; ++k) {
            const int rx = (int)(k % SC_TILES_X), ry = (int)(k / SC_TILES_X) * SC_RED_TH;
            float diff = 0.f, seen = 0.f;
            for (int j = 0; j < SC_RED_TH; ++j) {
                const float* px = &red[((size_t)(ry + j) * SC_RED + rx) * 4];
                diff += px[0];
                seen += px[1];
            }
            if (seen / SC_RED_TH >= SC_MIN_SEEN) out[base + k] = diff / seen;
        }
    }
    return out;
}

// ── Drawing ──────────────────────────────────────────────────────────────────
void SkinTextureGL::draw(unsigned int tex, const float* verts, const float* normals, const float mvp[16],
                         float alpha, const float normal_sign[3], const float light[3], bool use_tex_alpha)
{
    if (!ready() || !tex) return;
    GLint prev_prog = 0, prev_vao = 0, prev_active = 0, prev_tex3 = 0;
    glGetIntegerv(GL_CURRENT_PROGRAM, &prev_prog);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prev_vao);
    glGetIntegerv(GL_ACTIVE_TEXTURE, &prev_active);
    glActiveTexture(GL_TEXTURE3);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prev_tex3);

    upload_wedges(verts, normals, nullptr);
    glUseProgram(prog_draw_);
    glBindTexture(GL_TEXTURE_2D, tex);
    glUniform1i(glGetUniformLocation(prog_draw_, "uTex"), 3);
    glUniformMatrix4fv(glGetUniformLocation(prog_draw_, "uMVP"), 1, GL_FALSE, mvp);
    glUniform3fv(glGetUniformLocation(prog_draw_, "uNSign"), 1, normal_sign);
    glUniform3fv(glGetUniformLocation(prog_draw_, "uLight"), 1, light);
    glUniform1f(glGetUniformLocation(prog_draw_, "uAlpha"), alpha);
    glUniform1f(glGetUniformLocation(prog_draw_, "uUseTexAlpha"), use_tex_alpha ? 1.f : 0.f);
    glBindVertexArray(vao_);
    glDrawElements(GL_TRIANGLES, (GLsizei)indices_.size(), GL_UNSIGNED_INT, nullptr);

    glBindTexture(GL_TEXTURE_2D, (GLuint)prev_tex3);
    glActiveTexture((GLenum)prev_active);
    glBindVertexArray((GLuint)prev_vao);
    glUseProgram((GLuint)prev_prog);
}

std::vector<uint8_t> SkinTextureGL::read_rgba(int s)
{
    std::vector<uint8_t> out;
    if (!ready() || s < 0 || s >= (int)slots_.size() || !slots_[s].out_tex) return out;
    const size_t row = (size_t)tex_size_ * 4;
    std::vector<uint8_t> gl((size_t)tex_size_ * row);
    GLint prev_active = 0, prev_tex3 = 0, prev_pack = 0;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &prev_active);
    glActiveTexture(GL_TEXTURE3);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prev_tex3);
    glGetIntegerv(GL_PACK_ALIGNMENT, &prev_pack);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glBindTexture(GL_TEXTURE_2D, slots_[s].out_tex);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, gl.data());
    glBindTexture(GL_TEXTURE_2D, (GLuint)prev_tex3);
    glPixelStorei(GL_PACK_ALIGNMENT, prev_pack);
    glActiveTexture((GLenum)prev_active);
    out.resize(gl.size());
    for (int y = 0; y < tex_size_; ++y)          // GL row 0 = bottom = v 1
        memcpy(&out[(size_t)y * row], &gl[(size_t)(tex_size_ - 1 - y) * row], row);
    return out;
}

} // namespace fsb
