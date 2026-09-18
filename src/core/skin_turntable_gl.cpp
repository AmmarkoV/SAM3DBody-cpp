// skin_turntable_gl.cpp  –  see skin_turntable_gl.h
#include "skin_turntable_gl.h"

#include <GL/glew.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace fsb {

// Virtual camera distance (metres) from each person's centre; the focal length
// is then picked so the body fills its column.
static constexpr float CAM_DIST = 3.0f;
static constexpr float Z_NEAR   = 0.1f, Z_FAR = 10.f;

// Positions arrive already centred, rotated and pushed CAM_DIST in front of
// the camera, in the pipeline's camera space (Y down, Z forward).  The
// projection mirrors mhr_camera_matrices: flip Y/Z into GL view space, then a
// pinhole with the principal point at the column centre.
static const char* kVert = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec3 aNorm;
layout(location=2) in vec3 aColor;
uniform vec4 uProj;                   // 2f/w, 2f/h, depth a, depth b
out vec3 vNorm;
out vec3 vColor;
void main() {
    gl_Position = vec4(aPos.x * uProj.x, -aPos.y * uProj.y, aPos.z * uProj.z + uProj.w, aPos.z);
    vNorm  = vec3(aNorm.x, -aNorm.y, -aNorm.z);   // into GL view space
    vColor = aColor;
}
)";

// Same wrap lighting as default.frag, with the light fixed to the camera.
static const char* kFrag = R"(
#version 330 core
in vec3 vNorm;
in vec3 vColor;
out vec4 fragColor;
void main() {
    vec3  N = normalize(vNorm);
    vec3  L = normalize(vec3(0.3, 0.5, 0.8));
    float d = clamp((dot(N, L) + 0.15) / 1.15, 0.0, 1.0);
    d = d * d * (3.0 - 2.0 * d);
    d = d * 0.65 + 0.35;
    fragColor = vec4(vColor * d, 1.0);
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
        fprintf(stderr, "[skin-turntable] shader: %s\n", log);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

bool SkinTurntableGL::init(const unsigned int* indices, size_t n_indices, size_t n_vertices,
                           int width, int height)
{
    n_vertices_ = n_vertices;
    n_indices_  = (unsigned int)((n_indices / 3) * 3);
    indices_.assign(indices, indices + n_indices_);
    w_ = width;
    h_ = height;

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
    if (!ok) { fprintf(stderr, "[skin-turntable] program link failed\n"); return false; }
    proj_loc_ = glGetUniformLocation(prog_, "uProj");

    glGenVertexArrays(1, &vao_);
    glBindVertexArray(vao_);
    glGenBuffers(1, &ebo_);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, n_indices_ * sizeof(unsigned int), indices, GL_STATIC_DRAW);
    GLuint* vbos[3] = {&pos_vbo_, &norm_vbo_, &col_vbo_};
    for (int a = 0; a < 3; ++a) {
        glGenBuffers(1, vbos[a]);
        glBindBuffer(GL_ARRAY_BUFFER, *vbos[a]);
        glBufferData(GL_ARRAY_BUFFER, n_vertices_ * 3 * sizeof(float), nullptr, GL_STREAM_DRAW);
        glEnableVertexAttribArray(a);
        glVertexAttribPointer(a, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    }
    glBindVertexArray(0);

    glGenRenderbuffers(1, &color_rb_);
    glBindRenderbuffer(GL_RENDERBUFFER, color_rb_);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, w_, h_);
    glGenRenderbuffers(1, &depth_rb_);
    glBindRenderbuffer(GL_RENDERBUFFER, depth_rb_);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, w_, h_);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);

    GLint prev_fbo = 0;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);
    glGenFramebuffers(1, &fbo_);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, color_rb_);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,  GL_RENDERBUFFER, depth_rb_);
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)prev_fbo);
    if (status != GL_FRAMEBUFFER_COMPLETE) {
        fprintf(stderr, "[skin-turntable] framebuffer incomplete (0x%04X)\n", status);
        return false;
    }
    pixels_.resize((size_t)w_ * h_ * 3);
    return true;
}

const std::vector<uint8_t>& SkinTurntableGL::render(const std::vector<const float*>& verts,
                                                    const std::vector<const float*>& rgb,
                                                    float angle_deg)
{
    if (!prog_) return pixels_;
    const size_t P = std::min(verts.size(), rgb.size());

    // ── Save the state we touch ──────────────────────────────────────────────
    GLint prev_fbo = 0, prev_prog = 0, prev_vao = 0, prev_vp[4], prev_pack = 4;
    GLfloat prev_clear[4];
    glGetIntegerv(GL_FRAMEBUFFER_BINDING, &prev_fbo);
    glGetIntegerv(GL_CURRENT_PROGRAM, &prev_prog);
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &prev_vao);
    glGetIntegerv(GL_VIEWPORT, prev_vp);
    glGetIntegerv(GL_PACK_ALIGNMENT, &prev_pack);
    glGetFloatv(GL_COLOR_CLEAR_VALUE, prev_clear);
    const GLboolean prev_blend = glIsEnabled(GL_BLEND), prev_depth = glIsEnabled(GL_DEPTH_TEST),
                    prev_cull  = glIsEnabled(GL_CULL_FACE), prev_scissor = glIsEnabled(GL_SCISSOR_TEST);

    glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glDisable(GL_SCISSOR_TEST);
    glViewport(0, 0, w_, h_);
    glClearColor(0.12f, 0.12f, 0.14f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glUseProgram(prog_);
    glBindVertexArray(vao_);

    const float cs = std::cos(angle_deg * 3.14159265f / 180.f);
    const float sn = std::sin(angle_deg * 3.14159265f / 180.f);
    pos_.resize(n_vertices_ * 3);
    norm_.resize(n_vertices_ * 3);

    // Centre, height and horizontal reach (the radius it sweeps turning) per
    // person; one focal length for everyone so their sizes stay comparable,
    // the largest one that still fits every body in its column.
    struct Extent { float c[3], reach, height; };
    std::vector<Extent> ext(P);
    float f = 1e9f;
    for (size_t p = 0; p < P; ++p) {
        const float* v = verts[p];
        float lo[3] = { 1e9f,  1e9f,  1e9f}, hi[3] = {-1e9f, -1e9f, -1e9f};
        for (size_t i = 0; i < n_vertices_; ++i)
            for (int c = 0; c < 3; ++c) {
                lo[c] = std::min(lo[c], v[i*3+c]);
                hi[c] = std::max(hi[c], v[i*3+c]);
            }
        const float cx = 0.5f * (lo[0] + hi[0]), cy = 0.5f * (lo[1] + hi[1]), cz = 0.5f * (lo[2] + hi[2]);
        float r2 = 0.f;
        for (size_t i = 0; i < n_vertices_; ++i) {
            float dx = v[i*3+0] - cx, dz = v[i*3+2] - cz;
            r2 = std::max(r2, dx*dx + dz*dz);
        }
        const float reach  = std::max(std::sqrt(r2), 1e-3f);
        const float height = std::max(hi[1] - lo[1], 1e-3f);
        ext[p] = {{cx, cy, cz}, reach, height};
        const int   cw     = (int)((p + 1) * w_ / P) - (int)(p * w_ / P);
        const float near_z = CAM_DIST - reach;
        f = std::min(f, std::min(0.85f * h_ * near_z / height, 0.9f * cw * near_z / (2.f * reach)));
    }

    for (size_t p = 0; p < P; ++p) {
        const float* v = verts[p];
        const float cx = ext[p].c[0], cy = ext[p].c[1], cz = ext[p].c[2];

        // Area-weighted normals of the posed mesh, then everything rotated
        // about the vertical axis through the centre.
        std::fill(norm_.begin(), norm_.end(), 0.f);
        for (unsigned int t = 0; t + 2 < n_indices_; t += 3) {
            const unsigned int a = indices_[t], b = indices_[t+1], c = indices_[t+2];
            const float* A = v + a*3; const float* B = v + b*3; const float* C = v + c*3;
            float e1[3] = {B[0]-A[0], B[1]-A[1], B[2]-A[2]}, e2[3] = {C[0]-A[0], C[1]-A[1], C[2]-A[2]};
            float n[3]  = {e1[1]*e2[2]-e1[2]*e2[1], e1[2]*e2[0]-e1[0]*e2[2], e1[0]*e2[1]-e1[1]*e2[0]};
            for (unsigned int k : {a, b, c})
                for (int q = 0; q < 3; ++q) norm_[k*3+q] += n[q];
        }
        for (size_t i = 0; i < n_vertices_; ++i) {
            float dx = v[i*3+0] - cx, dy = v[i*3+1] - cy, dz = v[i*3+2] - cz;
            pos_[i*3+0] =  cs * dx + sn * dz;
            pos_[i*3+1] =  dy;
            pos_[i*3+2] = -sn * dx + cs * dz + CAM_DIST;
            float nx = norm_[i*3+0], ny = norm_[i*3+1], nz = norm_[i*3+2];
            float len = std::sqrt(nx*nx + ny*ny + nz*nz);
            if (len < 1e-12f) len = 1.f;
            norm_[i*3+0] = ( cs * nx + sn * nz) / len;
            norm_[i*3+1] =   ny / len;
            norm_[i*3+2] = (-sn * nx + cs * nz) / len;
        }

        const int x0 = (int)(p * w_ / P), cw = (int)((p + 1) * w_ / P) - x0;
        glViewport(x0, 0, cw, h_);
        glUniform4f(proj_loc_, 2.f * f / cw, 2.f * f / h_,
                    (Z_FAR + Z_NEAR) / (Z_FAR - Z_NEAR), -2.f * Z_FAR * Z_NEAR / (Z_FAR - Z_NEAR));

        glBindBuffer(GL_ARRAY_BUFFER, pos_vbo_);
        glBufferSubData(GL_ARRAY_BUFFER, 0, n_vertices_ * 3 * sizeof(float), pos_.data());
        glBindBuffer(GL_ARRAY_BUFFER, norm_vbo_);
        glBufferSubData(GL_ARRAY_BUFFER, 0, n_vertices_ * 3 * sizeof(float), norm_.data());
        glBindBuffer(GL_ARRAY_BUFFER, col_vbo_);
        glBufferSubData(GL_ARRAY_BUFFER, 0, n_vertices_ * 3 * sizeof(float), rgb[p]);
        glDrawElements(GL_TRIANGLES, (GLsizei)n_indices_, GL_UNSIGNED_INT, nullptr);
    }

    // ── Read back, top row first ─────────────────────────────────────────────
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w_, h_, GL_RGB, GL_UNSIGNED_BYTE, pixels_.data());
    const size_t stride = (size_t)w_ * 3;
    row_.resize(stride);
    for (int y = 0; y < h_ / 2; ++y) {
        uint8_t* a = &pixels_[(size_t)y * stride];
        uint8_t* b = &pixels_[(size_t)(h_ - 1 - y) * stride];
        memcpy(row_.data(), a, stride);
        memcpy(a, b, stride);
        memcpy(b, row_.data(), stride);
    }

    // ── Restore ──────────────────────────────────────────────────────────────
    glPixelStorei(GL_PACK_ALIGNMENT, prev_pack);
    glClearColor(prev_clear[0], prev_clear[1], prev_clear[2], prev_clear[3]);
    glBindVertexArray((GLuint)prev_vao);
    glUseProgram((GLuint)prev_prog);
    glBindFramebuffer(GL_FRAMEBUFFER, (GLuint)prev_fbo);
    glViewport(prev_vp[0], prev_vp[1], prev_vp[2], prev_vp[3]);
    if (prev_blend)   glEnable(GL_BLEND);        else glDisable(GL_BLEND);
    if (prev_depth)   glEnable(GL_DEPTH_TEST);   else glDisable(GL_DEPTH_TEST);
    if (prev_cull)    glEnable(GL_CULL_FACE);    else glDisable(GL_CULL_FACE);
    if (prev_scissor) glEnable(GL_SCISSOR_TEST); else glDisable(GL_SCISSOR_TEST);
    return pixels_;
}

} // namespace fsb
