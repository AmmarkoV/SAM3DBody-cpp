// fast_sam_3dbody_render.cpp
// OpenGL overlay renderer: reads a camera/image source, runs the MHR body-pose
// pipeline, and draws the deformed 3D mesh on top of the input frame.
//
// Usage:
//   fast_sam_3dbody_render --onnx-dir DIR --gguf pipeline.gguf
//       --yolo yolo.onnx [--mesh body_mesh.tri] [--from 0|path]
//
// Controls: close the window to exit.

// GLEW must come before any other GL header.
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glx.h>

extern "C" {
#include "../GraphicsEngine/System/glx3.h"
#include "../GraphicsEngine/ModelLoader/model_loader_tri.h"
#include "../GraphicsEngine/ModelLoader/model_loader_transform_joints.h"
}

#include "../src/fast_sam_3dbody.h"
#include "../src/preprocess.hpp"   // for fsb::apply_hand_pose
#include "../src/outputFiltering.h" // for QuatLPF + euler_zyx_to_quat helpers
#include "../src/cli_common.h"      // shared --onnx-dir / --bvh / … parser
#include "mhr_pose_driver.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "../src/bvh_writer.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <time.h>

// ── Inline GLSL shaders ──────────────────────────────────────────────────────

// Background fullscreen quad — uses gl_VertexID, no VBO needed.
static const char* QUAD_VERT = R"glsl(
    #version 330 core
    const vec2 P[4] = vec2[](
        vec2(-1.0,-1.0), vec2(1.0,-1.0),
        vec2(-1.0, 1.0), vec2(1.0, 1.0)
    );
    const vec2 UV[4] = vec2[](
        vec2(0.0,1.0), vec2(1.0,1.0),
        vec2(0.0,0.0), vec2(1.0,0.0)
    );
    out vec2 vUV;
    void main() { gl_Position = vec4(P[gl_VertexID],0.0,1.0); vUV = UV[gl_VertexID]; }
)glsl";

static const char* QUAD_FRAG = R"glsl(
    #version 330 core
    in  vec2      vUV;
    uniform sampler2D uTex;
    out vec4 fragColor;
    void main() { fragColor = vec4(texture(uTex, vUV).rgb, 1.0); }
)glsl";

// The body-mesh shaders live in render/default.vert and render/default.frag so
// they can be edited without a rebuild; default.frag exposes a `uColor` uniform.

// ── GL helpers ───────────────────────────────────────────────────────────────

// Read a whole text file (a GLSL source) into a string.  Returns empty on
// failure and logs the path — the caller treats empty as fatal.
static std::string load_text_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[shader] cannot open %s\n", path); return {}; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::string s;
    if (n > 0) { s.resize((size_t)n); if (fread(&s[0], 1, (size_t)n, f) != (size_t)n) s.clear(); }
    fclose(f);
    return s;
}

static GLuint compile_shader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[512] = {}; glGetShaderInfoLog(s, sizeof(buf), nullptr, buf);
        fprintf(stderr, "[shader] %s\n", buf);
    }
    return s;
}

static GLuint link_program(const char* vs, const char* fs) 
{
    GLuint p = glCreateProgram();
    GLuint v = compile_shader(GL_VERTEX_SHADER,   vs);
    GLuint f = compile_shader(GL_FRAGMENT_SHADER, fs);
    glAttachShader(p, v); glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok = 0; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[512] = {}; glGetProgramInfoLog(p, sizeof(buf), nullptr, buf);
        fprintf(stderr, "[program] %s\n", buf);
    }
    glDeleteShader(v); glDeleteShader(f);
    return p;
}

// ── GPU mesh state ───────────────────────────────────────────────────────────

struct MeshGPU 
{
    GLuint vao, vbo_pos, vbo_norm, ebo;
    GLsizei n_indices;
};

static MeshGPU upload_mesh_once(const struct TRI_Model* m) 
{
    MeshGPU g{};
    g.n_indices = (GLsizei)m->header.numberOfIndices;

    glGenVertexArrays(1, &g.vao);
    glBindVertexArray(g.vao);

    // Positions — DYNAMIC (updated every frame via glBufferSubData)
    glGenBuffers(1, &g.vbo_pos);
    glBindBuffer(GL_ARRAY_BUFFER, g.vbo_pos);
    glBufferData(GL_ARRAY_BUFFER,
                 (GLsizeiptr)(m->header.numberOfVertices * sizeof(float)),
                 m->vertices, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    // Normals — STATIC (T-pose normals are good enough for an overlay)
    glGenBuffers(1, &g.vbo_norm);
    glBindBuffer(GL_ARRAY_BUFFER, g.vbo_norm);
    glBufferData(GL_ARRAY_BUFFER,
                 (GLsizeiptr)(m->header.numberOfNormals * sizeof(float)),
                 m->normal, GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

    // Indices — STATIC
    glGenBuffers(1, &g.ebo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 (GLsizeiptr)(m->header.numberOfIndices * sizeof(unsigned int)),
                 m->indices, GL_STATIC_DRAW);

    glBindVertexArray(0);
    return g;
}

// ── Background texture ───────────────────────────────────────────────────────

struct BgTex { GLuint id; int w, h; bool ready; };

static BgTex create_bg_tex()
{
    BgTex t{0, 0, 0, false};

    glGenTextures(1, &t.id);
    if (t.id == 0) {
        fprintf(stderr, "[GL] glGenTextures returned 0 — out of texture objects?\n");
        return t;
    }

    glBindTexture(GL_TEXTURE_2D, t.id);
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "[GL] glBindTexture(GL_TEXTURE_2D) failed\n");
        glDeleteTextures(1, &t.id);
        t.id = 0;
        return t;
    }

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "[GL] glTexParameteri failed\n");
        glBindTexture(GL_TEXTURE_2D, 0);
        glDeleteTextures(1, &t.id);
        t.id = 0;
        return t;
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    if (glGetError() != GL_NO_ERROR) {
        fprintf(stderr, "[GL] glBindTexture(0) failed\n");
        glDeleteTextures(1, &t.id);
        t.id = 0;
    }

    t.ready = true;
    return t;
}

// Upload a BGR frame. Converts to RGB so the sampler returns correct colours.
//
// Returns true on success, false if the upload could not be completed.  The
// caller should skip the GL pass for this frame on failure rather than crash
// — the renderer is otherwise long-lived and a single bad frame shouldn't
// take it down.
//
// Defensively guards against every path that's been observed (or is
// plausibly causing) the segfault that the previous inline comment flagged:
//
//   * Texture object never allocated (`t.id == 0`) — glTexImage2D into 0
//     is undefined behaviour on some drivers (intel-mesa segfaults, nvidia
//     silently writes nowhere).
//   * Input cv::Mat empty / null / wrong type.  `frame.clone()` upstream
//     can return an empty Mat under OpenCV OOM; the previous code's
//     `vis.empty()` check happened *after* `cvtColor` would have crashed.
//   * Non-3-channel input (e.g. greyscale fallback when the decoder
//     produces YUV420 and OpenCV converts to single-channel by mistake).
//   * Pathological dimensions (negative, zero, or > MAX_TEXTURE_SIZE
//     equivalent).  glTexImage2D with width/height beyond the GL
//     implementation's max gives GL_INVALID_VALUE — and on a few drivers
//     pre-write checks dereference an oversized row pointer first.
//   * Non-contiguous cv::Mat (`!isContinuous()`).  After cvtColor this
//     almost never happens but cvtColor on a sub-region of a larger Mat
//     can produce one; glTexImage2D reads the buffer as a flat
//     width*height*3 byte stream and would walk off the end of a strided
//     buffer.
//   * cv::cvtColor itself throwing.  Wrap in try/catch and return false
//     rather than letting the exception kill the process.
//   * Stale GL errors from earlier in the frame masking ours — drain
//     them before our own checks so we can correlate any error we see
//     here with one of our own calls.
static bool upload_bg_frame(BgTex& t, const cv::Mat& bgr)
{
    if (t.id == 0) {
        fprintf(stderr, "[GL] upload_bg_frame: texture not allocated (id=0)\n");
        return false;
    }
    if (bgr.empty() || bgr.data == nullptr) {
        fprintf(stderr, "[CV] upload_bg_frame: empty/null input Mat\n");
        return false;
    }
    if (bgr.type() != CV_8UC3) {
        fprintf(stderr, "[CV] upload_bg_frame: wrong type %d (need CV_8UC3=%d) — "
                        "%dx%d, channels=%d\n",
                bgr.type(), CV_8UC3, bgr.cols, bgr.rows, bgr.channels());
        return false;
    }
    if (bgr.cols <= 0 || bgr.rows <= 0 ||
        bgr.cols > 16384 || bgr.rows > 16384) {
        // 16384 is GL_MAX_TEXTURE_SIZE on every desktop GPU made since ~2010;
        // anything beyond is either a decoded-frame corruption or an HDR
        // 8K+ source we wouldn't want to render anyway.
        fprintf(stderr, "[CV] upload_bg_frame: bad dimensions %dx%d\n",
                bgr.cols, bgr.rows);
        return false;
    }

    cv::Mat rgb;
    try {
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    } catch (const cv::Exception& e) {
        fprintf(stderr, "[CV] upload_bg_frame: cvtColor threw: %s\n", e.what());
        return false;
    }
    if (rgb.empty() || rgb.data == nullptr || rgb.type() != CV_8UC3) {
        fprintf(stderr, "[CV] upload_bg_frame: cvtColor produced bad Mat "
                        "(empty=%d data=%p type=%d)\n",
                rgb.empty(), (void*)rgb.data, rgb.type());
        return false;
    }
    // glTexImage2D treats the pixel buffer as a flat (width*height*3) byte
    // stream when GL_UNPACK_ROW_LENGTH=0; non-contiguous Mats have row
    // padding that would make GL read past the end of valid memory.
    // Cheap and safe to force-pack.
    if (!rgb.isContinuous()) rgb = rgb.clone();

    // Drain any prior GL errors so our error checks below can be trusted.
    while (glGetError() != GL_NO_ERROR) { /* discard */ }

    GLint old_unpack = 4;
    glGetIntegerv(GL_UNPACK_ALIGNMENT, &old_unpack);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    bool ok = true;
    glBindTexture(GL_TEXTURE_2D, t.id);
    GLenum err = glGetError();
    if (err != GL_NO_ERROR) {
        fprintf(stderr, "[GL] glBindTexture(id=%u) failed: 0x%04X\n", t.id, err);
        ok = false;
    } else if (!t.ready || rgb.cols != t.w || rgb.rows != t.h) {
        // Dimensions changed (or first upload) — allocate storage.
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                     rgb.cols, rgb.rows, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
        err = glGetError();
        if (err != GL_NO_ERROR) {
            fprintf(stderr, "[GL] glTexImage2D %dx%d failed: 0x%04X\n",
                    rgb.cols, rgb.rows, err);
            // Mark texture invalid so we don't try to glTexSubImage2D into
            // it next frame (which would silently corrupt the display).
            t.ready = false;
            t.w = t.h = 0;
            ok = false;
        } else {
            t.w = rgb.cols; t.h = rgb.rows; t.ready = true;
        }
    } else {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                        rgb.cols, rgb.rows,
                        GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
        err = glGetError();
        if (err != GL_NO_ERROR) {
            fprintf(stderr, "[GL] glTexSubImage2D %dx%d failed: 0x%04X\n",
                    rgb.cols, rgb.rows, err);
            ok = false;
        }
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, old_unpack);
    return ok;
}

// ── 4x4 matrix multiply (column-major) ──────────────────────────────────────

static void mat4_mul(float dst[16], const float a[16], const float b[16]) 
{
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r) 
        {
            dst[c*4+r] = 0.f;
            for (int k = 0; k < 4; ++k)
                dst[c*4+r] += a[k*4+r] * b[c*4+k];
        }
}

static void mat4_print(const char * label,float m[16])
{
 fprintf(stderr,"%s\n",label);
 fprintf(stderr,"_________________________\n");
 fprintf(stderr,"%0.2f %0.2f %0.2f %0.2f\n",m[0],m[1],m[2],m[3]);
 fprintf(stderr,"%0.2f %0.2f %0.2f %0.2f\n",m[4],m[5],m[6],m[7]);
 fprintf(stderr,"%0.2f %0.2f %0.2f %0.2f\n",m[8],m[9],m[10],m[11]);
 fprintf(stderr,"%0.2f %0.2f %0.2f %0.2f\n",m[12],m[13],m[14],m[15]);
 fprintf(stderr,"_________________________\n");
}



int mat4_transpose(float * mat)
{
  if (mat!=0)
  {
  /*       -------  TRANSPOSE ------->
      0   1   2   3           0  4  8   12
      4   5   6   7           1  5  9   13
      8   9   10  11          2  6  10  14
      12  13  14  15          3  7  11  15   */

  float tmp;
  tmp = mat[1]; mat[1]=mat[4];  mat[4]=tmp;
  tmp = mat[2]; mat[2]=mat[8];  mat[8]=tmp;
  tmp = mat[3]; mat[3]=mat[12]; mat[12]=tmp;


  tmp = mat[6]; mat[6]=mat[9]; mat[9]=tmp;
  tmp = mat[13]; mat[13]=mat[7]; mat[7]=tmp;
  tmp = mat[14]; mat[14]=mat[11]; mat[11]=tmp;
  } else
  { //Believe it or not this is the fastest branch prediction :P
    return 0;
  }

 return 1;
}
// ── Callbacks required by glx3.c ─────────────────────────────────────────────

extern "C" 
{
    // Called by glx3_checkEvents() on key/mouse events.
    int handleUserInput(int key, int x, int y) { (void)key; (void)x; (void)y; return 1; }
    // Called by glx3_checkEvents() when the window is resized.
    int windowSizeUpdated(unsigned int w, unsigned int h) { (void)w; (void)h; return 1; }
}

// ── YOLO skeleton joint pairs (COCO 17-joint order) ─────────────────────────

static const int COCO_PAIRS[][2] = 
{
    {0,1},{0,2},{1,3},{2,4},                          // head
    {5,6},{5,7},{7,9},{6,8},{8,10},                   // arms
    {5,11},{6,12},{11,12},{11,13},{13,15},{12,14},{14,16} // torso+legs
};
static const int N_COCO_PAIRS = 17;

static void draw_yolo_skeleton(cv::Mat& img,
                                const std::vector<float>& kps,
                                float conf_thresh = 0.3f) 
{
    if ((int)kps.size() < 51) return;
    // Draw limb lines first, then joint dots on top
    for (int p = 0; p < N_COCO_PAIRS; ++p) {
        int a = COCO_PAIRS[p][0], b = COCO_PAIRS[p][1];
        if (kps[a*3+2] < conf_thresh || kps[b*3+2] < conf_thresh) continue;
        cv::line(img,
                 {(int)kps[a*3], (int)kps[a*3+1]},
                 {(int)kps[b*3], (int)kps[b*3+1]},
                 cv::Scalar(255, 128, 0), 2, cv::LINE_AA);
    }
    for (int k = 0; k < 17; ++k) {
        if (kps[k*3+2] < conf_thresh) continue;
        cv::circle(img, {(int)kps[k*3], (int)kps[k*3+1]},
                   5, cv::Scalar(0, 200, 255), -1, cv::LINE_AA);
    }
}

// ── Save GL framebuffer to file ──────────────────────────────────────────────

static void save_framebuffer(const std::string& path, int w, int h) {
    std::vector<uint8_t> px(w * h * 3);
    // Default GL_PACK_ALIGNMENT is 4 — for widths whose row byte-count (w*3)
    // is not divisible by 4 (e.g. 2250×3 = 6750 → 2 pad bytes/row) glReadPixels
    // writes over-aligned rows, shearing the saved image into a parallelogram.
    GLint old_pack = 4;
    glGetIntegerv(GL_PACK_ALIGNMENT, &old_pack);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, px.data());
    glPixelStorei(GL_PACK_ALIGNMENT, old_pack);
    // glReadPixels gives bottom-up rows; flip vertically
    cv::Mat img(h, w, CV_8UC3, px.data());
    cv::flip(img, img, 0);
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    cv::imwrite(path, img);
    printf("Saved: %s\n", path.c_str());
}

// 2nd-order Butterworth is now sourced from src/outputFiltering.h
// (struct ButterWorth + initButterWorth/filter, plus the wrap-aware
//  ButterWorthWrap shim that absorbs the unwrap-then-filter trick the
//  previous BwFilter class did inline).  Removing the local duplicate
//  eliminates the two-coefficient-conventions footgun PLAN.md called
//  out — there's now exactly one Butterworth implementation in the
//  project and any change to the smoothing math goes in one place.
//
// Helper: filter N channels in-place through a parallel bank of
// ButterWorthWrap filters.  `bank.size()` must equal `n`.
static inline void apply_bw_bank(std::vector<ButterWorthWrap>& bank,
                                 float* data, int n)
{
    if ((int)bank.size() != n) return;
    for (int i = 0; i < n; ++i)
        data[i] = filter_wrap(&bank[i], data[i]);
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, const char** argv) {
    std::string onnx_dir  = "./onnx";
    std::string gguf_path = "./onnx/pipeline.gguf";
    std::string yolo_path = "./onnx/yolo.onnx";
    std::string mesh_path = "./body_mesh.tri";
    std::string vert_path = "render/default.vert";
    std::string frag_path = "render/default.frag";
    float       mesh_color[3] = {0.65f, 0.75f, 0.9f};  // same for everyone for now
    float       shininess     = 0.0f;   // 0 = matte; --shiny turns on the chrome look
    float       transparency  = 0.7f;   // mesh opacity (--transparency)
    std::string lbs_path  = "";
    std::string src       = "0";
    std::string save_frames_prefix = "";
    int         save_frame_idx     = 0;
    std::string bvh_path           = "";
    std::string bvh_template       = "";
    // --headless creates a GLX Pbuffer (offscreen surface) instead of a
    // visible X11 window.  Used by scripts/video.sh in --save mode so the
    // long-running render can't be killed by an accidental window close,
    // the screen-saver, or any window-manager interaction with a stale
    // long-lived window.  The GL context is identical either way; only
    // the surface is offscreen, so glReadPixels (used by --save-frames)
    // still produces the same image.
    bool        headless           = false;
    bool        bvh_body_shape_change          = true;
    bool        bvh_hand_shape_change          = true;
    bool        bvh_compensate_finger_endsites = true;
    bool        bvh_enforce_hand_limits        = false;
    bool        bvh_zero_hand_pose             = false;
    bool        bvh_sticky_hand_pose           = false;
    bool        bvh_rest_align                 = true;
    bool        bvh_dump_rest_dirs             = false;
    bool        bvh_foot_contact               = false;
    bool        use_butterworth    = false;
    float       bw_cutoff         = 6.0f;   // Hz; higher = less lag, less smoothing
    bool        filter_root_rot   = false;  // enabled by --butterworth-root-rotation
    float       rot_clamp_deg     = 1.0f;   // rejection threshold in degrees/frame
    int  cuda_device  = 0;
    bool use_trt      = false;
    bool fp16         = true;
    bool zero_face    = true;
    int    render_w   = 0;   // GL window width  (0 = match input)
    int    render_h   = 0;   // GL window height (0 = match input)
    int    cap_w      = 0;   // capture width  (0 = driver default)
    int    cap_h      = 0;   // capture height (0 = driver default)
    double cap_fps    = 0.0; // capture fps    (0 = driver default)

    // Common flags go through the shared parser; binary-specific flags
    // (--mesh, --lbs, --save-frames, --render-size, --size, --fps,
    //  --butterworth*, --dev-face, --headless) stay handled inline.
    CommonConfig cc;
    for (int i = 1; i < argc; ++i) {
        if (parse_common_arg(argc, argv, i, cc)) continue;

#define A1(flag, field, conv) \
        if (!strcmp(argv[i], flag) && i+1<argc) { field = conv(argv[++i]); continue; }
        A1("--mesh",        mesh_path,          std::string)
        A1("--vert",        vert_path,          std::string)
        A1("--frag",        frag_path,          std::string)
        A1("--lbs",         lbs_path,           std::string)
        A1("--save-frames", save_frames_prefix, std::string)
#undef A1
        if (!strcmp(argv[i], "--mesh-color") && i+3 < argc)
            { mesh_color[0]=std::stof(argv[++i]); mesh_color[1]=std::stof(argv[++i]);
              mesh_color[2]=std::stof(argv[++i]); continue; }
        if (!strcmp(argv[i], "--transparency") && i+1 < argc)
            { transparency = 1.0f - std::stof(argv[++i]); continue; }  // value is transparency; uAlpha is opacity
        // --shiny enables the reflective look at a default strength; an optional
        // numeric argument (0..1) overrides it, e.g. --shiny 0.6
        if (!strcmp(argv[i], "--shiny")) {
            shininess = 0.85f;
            if (i+1 < argc && (isdigit((unsigned char)argv[i+1][0]) || argv[i+1][0]=='.'))
                shininess = std::stof(argv[++i]);
            continue;
        }
        if (!strcmp(argv[i], "--render-size") && i+2 < argc)
            { render_w = std::stoi(argv[++i]); render_h = std::stoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--size") && i+2 < argc)
            { cap_w = std::stoi(argv[++i]); cap_h = std::stoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--fps") && i+1 < argc)
            { cap_fps = std::stod(argv[++i]); continue; }
        if (!strcmp(argv[i], "--dev-face"))    { zero_face      = false; continue; }
        if (!strcmp(argv[i], "--butterworth"))              { use_butterworth  = true; continue; }
        if (!strcmp(argv[i], "--butterworth-root-rotation")){ filter_root_rot  = true; continue; }
        if (!strcmp(argv[i], "--headless"))                 { headless = true; continue; }
    }
    // Unpack the common parser's output into the local variables the
    // rest of main() expects.  Keeping the locals avoids a wholesale
    // refactor of the rendering loop's pipeline-config / BVH-writer /
    // filter-init code paths.
    onnx_dir                       = cc.onnx_dir;
    gguf_path                      = cc.gguf_path;
    yolo_path                      = cc.yolo_path;
    src                            = cc.from;
    cuda_device                    = cc.cuda_device;
    use_trt                        = cc.use_trt;
    fp16                           = cc.fp16;
    bvh_path                       = cc.bvh_path;
    bvh_template                   = cc.bvh_template;
    bvh_body_shape_change          = cc.bvh_body_shape_change;
    bvh_hand_shape_change          = cc.bvh_hand_shape_change;
    bvh_compensate_finger_endsites = cc.bvh_compensate_finger_endsites;
    bvh_enforce_hand_limits        = cc.bvh_enforce_hand_limits;
    bvh_zero_hand_pose             = cc.bvh_zero_hand_pose;
    bvh_sticky_hand_pose           = cc.bvh_sticky_hand_pose;
    bvh_rest_align                 = cc.bvh_rest_align;
    bvh_dump_rest_dirs             = cc.bvh_dump_rest_dirs;
    bvh_foot_contact               = cc.bvh_foot_contact;
    bw_cutoff                      = cc.bw_cutoff;
    rot_clamp_deg                  = cc.rot_clamp_deg;

    // ── Pipeline ─────────────────────────────────────────────────────────────
    fsb::Pipeline pipeline;
    {
        fsb::PipelineConfig cfg;
        cfg.onnx_dir        = onnx_dir;
        cfg.gguf_path       = gguf_path;
        cfg.yolo_path       = yolo_path;
        cfg.cuda_device     = cuda_device;
        cfg.use_trt_ep      = use_trt;
        cfg.use_fp16        = fp16;
        cfg.skip_body_model = true;    // LBS runs natively in C; skip body_model.onnx
        if (!pipeline.load(cfg)) {
            fprintf(stderr, "Failed to load pipeline\n"); return 1;
        }
    }

    // ── Video/image source ────────────────────────────────────────────────────
    bool is_image = false;
    cv::Mat static_img;
    cv::VideoCapture cap;
    {
        bool numeric = !src.empty() &&
                       (src[0]=='-' || isdigit((unsigned char)src[0]));
        if (numeric) {
            cap.open(std::stoi(src));
        } else {
            static_img = cv::imread(src);
            if (!static_img.empty()) {
                is_image = true;
            } else {
                cap.open(src);
                if (!cap.isOpened()) { fprintf(stderr,"Cannot open: %s\n", src.c_str()); return 1; }
            }
        }
        if (!is_image && cap.isOpened()) {
            if (cap_w > 0) cap.set(cv::CAP_PROP_FRAME_WIDTH,  cap_w);
            if (cap_h > 0) cap.set(cv::CAP_PROP_FRAME_HEIGHT, cap_h);
            if (cap_fps > 0.0) cap.set(cv::CAP_PROP_FPS,      cap_fps);
        }
    }

    // Determine initial window size from first frame
    cv::Mat probe;
    if (is_image) probe = static_img;
    else          cap >> probe;
    if (probe.empty()) { fprintf(stderr, "Empty frame\n"); return 1; }
    int frame_w = probe.cols;   // input frame dims — used for projection matrix
    int frame_h = probe.rows;
    int W = (render_w > 0) ? render_w : frame_w;
    int H = (render_h > 0) ? render_h : frame_h;

    // ── GLX surface ───────────────────────────────────────────────────────────
    // viewWindow=1 → normal visible X11 window
    // viewWindow=0 → offscreen GLX Pbuffer (no XMapWindow, no event source the
    //                user can interact with).  Pbuffers were the standard
    //                pre-EGL way to get offscreen GL on Linux/X11 and the
    //                fixed-pipeline glReadPixels we use to save frames works
    //                identically on them.
    if (!start_glx3_stuff(W, H, headless ? 0 : 1, argc, argv)) {
        fprintf(stderr, "Failed to start GLX %s\n",
                headless ? "Pbuffer" : "window"); return 1;
    }
    if (headless) printf("[headless] running offscreen — no GUI window\n");
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "GLEW init failed\n"); return 1;
    }

    // ── Shaders ───────────────────────────────────────────────────────────────
    GLuint prog_quad = link_program(QUAD_VERT, QUAD_FRAG);

    std::string mesh_vert_src = load_text_file(vert_path.c_str());
    std::string mesh_frag_src = load_text_file(frag_path.c_str());
    if (mesh_vert_src.empty() || mesh_frag_src.empty()) {
        fprintf(stderr, "Cannot load mesh shaders (%s, %s)\n",
                vert_path.c_str(), frag_path.c_str()); return 1;
    }
    GLuint prog_mesh = link_program(mesh_vert_src.c_str(), mesh_frag_src.c_str());
    GLint  mvp_loc   = glGetUniformLocation(prog_mesh, "uMVP");
    GLint  view_loc  = glGetUniformLocation(prog_mesh, "uView");
    GLint  color_loc = glGetUniformLocation(prog_mesh, "uColor");
    GLint  scene_loc = glGetUniformLocation(prog_mesh, "uScene");
    GLint  res_loc   = glGetUniformLocation(prog_mesh, "uResolution");
    GLint  shiny_loc = glGetUniformLocation(prog_mesh, "uShiny");
    GLint  alpha_loc = glGetUniformLocation(prog_mesh, "uAlpha");
    GLint  tex_loc   = glGetUniformLocation(prog_quad, "uTex");

    // ── Load body mesh from .tri ──────────────────────────────────────────────
    struct TRI_Model* tri_model = tri_allocateModel();
    if (!tri_loadModel(mesh_path.c_str(), tri_model)) {
        fprintf(stderr, "Cannot load mesh: %s\n", mesh_path.c_str()); return 1;
    }
    printf("Mesh loaded: %u vertices, %u indices\n",
           tri_model->header.numberOfVertices / 3,
           tri_model->header.numberOfIndices / 3);

    MeshGPU mesh_gpu = upload_mesh_once(tri_model);

    // ── Load LBS data ─────────────────────────────────────────────────────────
    if (lbs_path.empty()) lbs_path = onnx_dir + "/body_model.lbs";
    struct MHR_LBS_Data* lbs = mhr_lbs_load(lbs_path.c_str());
    if (!lbs) fprintf(stderr, "Warning: LBS data not loaded — mesh will not deform\n");
    if (lbs) {
        std::string corr_path = onnx_dir + "/correctives.bin";
        if (mhr_correctives_load(lbs, corr_path.c_str()))
            printf("[LBS] pose correctives loaded from %s\n", corr_path.c_str());
        else
            printf("[LBS] correctives.bin not found — rendering without pose correctives\n");
    }
    std::vector<float> lbs_out(MHR_VERTEX_FLOATS, 0.f);

    // ── Source FPS (used by both BVH writer and Butterworth init) ────────────
    float video_fps = 30.f;
    if (!is_image && cap.isOpened()) {
        double f = cap.get(cv::CAP_PROP_FPS);
        if (f > 1.0) video_fps = (float)f;
    }

    // ── BVH writer ────────────────────────────────────────────────────────────
    BVHWriter bvh_writer;
    if (!bvh_path.empty()) {
        if (bvh_template.empty()) bvh_template = "./body.bvh";
        if (!bvh_writer.open(bvh_template, bvh_path, 1.f / video_fps, lbs_path,
                             bvh_body_shape_change, bvh_hand_shape_change,
                             bvh_compensate_finger_endsites,
                             bvh_enforce_hand_limits,
                             bvh_zero_hand_pose,
                             bvh_sticky_hand_pose,
                             bvh_rest_align,
                             bvh_dump_rest_dirs))
            fprintf(stderr, "[BVH] Warning: could not open BVH writer\n");
        else {
            bvh_writer.set_foot_contact(bvh_foot_contact);
            printf("[BVH] Writing to %s (%.1f fps)\n", bvh_path.c_str(), video_fps);
        }
    }

    // ── Butterworth filter banks ──────────────────────────────────────────────
    //
    // One ButterWorthWrap per channel.  wrap_input=1 on mhr_model_params is
    // safe across the whole array — translation sub-ranges have per-frame
    // deltas well under π so the wrap is a no-op there, while the Euler
    // sub-ranges (joint rotations) get the wrap-correct integration that
    // prevents ±π-discontinuity flips.  pred_cam_t is metres, no wrap
    // possible, so wrap_input=0.
    //
    // Above-Nyquist cutoffs are caught here and disable the bank entirely
    // (same Nyquist guard the old BwFilter::init had, just hoisted up).
    std::vector<ButterWorthWrap> bw_mp;
    std::vector<ButterWorthWrap> bw_cam;
    bool bw_active = false;
    if (use_butterworth) {
        if (bw_cutoff <= 0.f || bw_cutoff >= video_fps * 0.5f) {
            printf("[BW] cutoff %.1f Hz is outside (0, Nyquist=%.1f Hz) — disabled\n",
                   bw_cutoff, video_fps * 0.5f);
        } else {
            bw_active = true;
            bw_mp.resize(204);
            bw_cam.resize(3);
            for (auto& w : bw_mp)  init_butterworth_wrap(&w, video_fps, bw_cutoff, 1);
            for (auto& w : bw_cam) init_butterworth_wrap(&w, video_fps, bw_cutoff, 0);
            float lag_ms = 1000.f / (3.14159f * bw_cutoff);
            printf("[BW] cutoff=%.1f Hz  sample=%.1f Hz  approx lag=%.0f ms (%.1f frames)\n",
                   bw_cutoff, video_fps, lag_ms, lag_ms * video_fps / 1000.f);
        }
    }
    // global_rot quaternion 1st-order SLERP-EMA — same QuatLPF primitive as
    // fast_sam_3dbody_run.  Filter directly on orientation (no Euler-wrap or
    // gimbal-lock artifacts); --rot-clamp is the geodesic SLERP-step clamp.
    QuatLPF root_rot_filter{};
    if (use_butterworth && filter_root_rot)
        init_quat_lpf(&root_rot_filter, video_fps, bw_cutoff);

    // Empty VAO for the quad (we use gl_VertexID in the vertex shader)
    GLuint quad_vao;
    glGenVertexArrays(1, &quad_vao);

    BgTex bg = create_bg_tex();
    if (bg.id == 0) {
        fprintf(stderr, "[GL] Failed to create background texture\n");
        return 1;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // ── Render loop ───────────────────────────────────────────────────────────
#define NS_NOW() ({ struct timespec _t; clock_gettime(CLOCK_MONOTONIC,&_t); (long long)_t.tv_sec*1000000000LL + _t.tv_nsec; })
    long long t_last_frame = NS_NOW();
    double    fps_ema      = 0.0;

    cv::Mat frame;
    while (glx3_checkEvents()) 
    {
        if (is_image) 
        {
            frame = static_img;
        } else 
        {
            cap >> frame;
            if (frame.empty()) break;
        }

        // Inference
        long long t_infer = NS_NOW();
        auto results = pipeline.process_bgr(frame.data, frame.cols, frame.rows);
        double latency_ms = (NS_NOW() - t_infer) / 1e6;

        // Patch arm/collar/head angles in mhr_model_params.
        // The pipeline runs with skip_body_model=true so its internal lbs_data is null,
        // meaning apply_hand_pose was a no-op inside fast_sam_3dbody.cpp — the arm joint
        // slots [68:121] in mhr_model_params are zeroed.  Re-apply here using the
        // renderer's own lbs, which has the hand PCA matrices loaded.  This must happen
        // before the Butterworth filter (so the filter smooths the true arm angles) and
        // before bvh_writer.write_frame (which reads from mhr_model_params).
        if (!results.empty() && lbs &&
            lbs->hand_pose_mean && lbs->hand_pose_comps &&
            lbs->hand_joint_idxs_left && lbs->hand_joint_idxs_right &&
            !results[0].hand_pose.empty()) {
            fsb::apply_hand_pose(results[0].mhr_model_params.data(),
                                  results[0].hand_pose.data(),
                                  lbs->hand_pose_mean, lbs->hand_pose_comps,
                                  lbs->hand_joint_idxs_left,
                                  lbs->hand_joint_idxs_right);
        }

        // Temporal smoothing — scalar Butterworth on linear channels,
        // QuatLPF SLERP-EMA on the root rotation.  The two gates are
        // independent because QuatLPF doesn't have a Nyquist constraint:
        // even if --bw-cutoff is above Nyquist (disabling bw_active), the
        // user can still ask for --butterworth-root-rotation and get a
        // working orientation filter.
        if (use_butterworth && !results.empty()) {
            if (bw_active) {
                apply_bw_bank(bw_mp,  results[0].mhr_model_params.data(), 204);
                apply_bw_bank(bw_cam, results[0].pred_cam_t.data(),       3);
            }

            // global_rot: quaternion-domain SLERP-EMA — only when
            // --butterworth-root-rotation is passed.  --rot-clamp is a
            // geodesic outlier clamp on the SLERP step in deg / frame; 0
            // disables it (pure EMA).
            if (filter_root_rot) {
                auto& gr = results[0].global_rot;
                float in_q[4], out_q[4];
                euler_zyx_to_quat(gr[0], gr[1], gr[2], in_q);
                float max_step_rad = (rot_clamp_deg > 0.0f)
                    ? rot_clamp_deg * (3.14159265359f / 180.0f) : 0.0f;
                filter_quat(&root_rot_filter, in_q, max_step_rad, out_q);
                quat_to_euler_zyx(out_q, &gr[0], &gr[1], &gr[2]);
            }
        }

        // BVH frame output
        if (bvh_writer.is_open())
            bvh_writer.write_frame(results);

        // Annotate frame: draw YOLO skeleton when LBS mesh is unavailable.
        cv::Mat vis = frame.clone();
        bool any_mesh = lbs && !results.empty();
        if (!any_mesh)
        {
            for (const auto& r : results)
                draw_yolo_skeleton(vis, r.keypoints_yolo);
        }

        // Upload background.  Failures are non-fatal: skip the background
        // quad for this frame and let the next frame retry.  Killing the
        // process here is the regression that produced truncated mp4s
        // (e.g. matrix_rendered.mp4 stopping at 14s with audio continuing
        // for the full 90s) — a single bad frame should not take down a
        // long render.
        bool bg_ok = upload_bg_frame(bg, vis);

        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glViewport(0, 0, W, H);

        // ── Background quad (only when we have a valid texture) ──────────────
        if (bg_ok && bg.ready) {
            glDisable(GL_DEPTH_TEST);
            glUseProgram(prog_quad);
            glUniform1i(tex_loc, 0);
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, bg.id);
            glBindVertexArray(quad_vao);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            glEnable(GL_DEPTH_TEST);
        }

        // ── Mesh overlay for each detected person ─────────────────────────────
        glUseProgram(prog_mesh);

        // Reflection uniforms (shared by every person this frame).  The scene
        // texture is the same background image the quad drew; bind it to unit 1
        // so the chrome shader can sample it.  uShiny=0 short-circuits the
        // reflection in the shader, so this is a no-op when --shiny is off.
        glUniform1f(shiny_loc, shininess);
        glUniform1f(alpha_loc, transparency);
        glUniform2f(res_loc, (float)W, (float)H);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, bg.id);
        glUniform1i(scene_loc, 1);

        for (const auto& r : results) {
            if (!lbs) continue;

            // Mesh tint — uniform for the whole body.  Constant for now; this
            // is where per-identity colouring will hook in later.
            glUniform3fv(color_loc, 1, mesh_color);

            // Decode scale PCA into model_params[136:204] if available.
            // Python: scales = scale_mean + scale_params @ scale_comps  ([28]→[68])
            // build_model_params zeros [136:204]; fill them here if lbs has scale data.
            std::array<float, 204> mp = r.mhr_model_params;
            // Apply hand pose (v3 lbs file required).  This overrides the zeroed
            // hand joint slots that build_model_params left in mp with the
            // PCA-decoded per-finger Euler angles, exactly as Python's
            // replace_hands_in_pose() does.
            if (lbs->hand_pose_mean && lbs->hand_pose_comps &&
                lbs->hand_joint_idxs_left && lbs->hand_joint_idxs_right &&
                !r.hand_pose.empty())
            {
                fsb::apply_hand_pose(mp.data(),
                                      r.hand_pose.data(),
                                      lbs->hand_pose_mean,
                                      lbs->hand_pose_comps,
                                      lbs->hand_joint_idxs_left,
                                      lbs->hand_joint_idxs_right);
            }
            if (lbs->scale_mean && lbs->scale_comps && !r.scale.empty()) {
                int ns = lbs->n_scale_out;  // 68
                int np = lbs->n_scale_pc;   // 28
                for (int i = 0; i < ns; ++i)
                    mp[136 + i] = lbs->scale_mean[i];
                for (int k = 0; k < np && k < (int)r.scale.size(); ++k)
                    for (int i = 0; i < ns; ++i)
                        mp[136 + i] += r.scale[k] * lbs->scale_comps[k * ns + i];
            }

            // Run native C LBS forward pass, stream result to GPU
            static const float zero_face_buf[72] = {};
            mhr_lbs_compute(lbs,
                            mp.data(),
                            r.shape.data(),
                            zero_face ? zero_face_buf : r.face_params.data(),
                            lbs_out.data(),
                            nullptr);
            mhr_update_mesh_vertices(tri_model, lbs_out.data());

            // First-frame verts dump for verify_transforms.py LBS comparison
            { static int verts_dumped = 0;
              if (!verts_dumped) {
                  verts_dumped = 1;
                  FILE* fp = fopen("/tmp/cpp_lbs_verts.bin", "wb");
                  if (fp) {
                      int hdr[2] = { (int)MHR_VERTEX_COUNT, 3 };
                      fwrite(hdr, sizeof(int), 2, fp);
                      fwrite(lbs_out.data(), sizeof(float), MHR_VERTEX_FLOATS, fp);
                      fclose(fp);
                      fprintf(stderr, "[LBS] wrote first-frame verts to /tmp/cpp_lbs_verts.bin\n");
                  }
              }
            }

#if 0 /* DEBUG: vertex bounds in model space — re-enable to diagnose mesh placement */
            // Debug: print vertex bounds in model space
            { float xmin=1e9f,xmax=-1e9f,ymin=1e9f,ymax=-1e9f,zmin=1e9f,zmax=-1e9f;
              for (int i=0; i<MHR_VERTEX_FLOATS; i+=3) {
                  if (lbs_out[i]<xmin)   xmin=lbs_out[i];
                  if (lbs_out[i]>xmax)   xmax=lbs_out[i];
                  if (lbs_out[i+1]<ymin) ymin=lbs_out[i+1];
                  if (lbs_out[i+1]>ymax) ymax=lbs_out[i+1];
                  if (lbs_out[i+2]<zmin) zmin=lbs_out[i+2];
                  if (lbs_out[i+2]>zmax) zmax=lbs_out[i+2];
              }
              printf("[mesh] model bounds: x[%.3f,%.3f] y[%.3f,%.3f] z[%.3f,%.3f]\n",
                     xmin,xmax, ymin,ymax, zmin,zmax);
            }
#endif

            glBindBuffer(GL_ARRAY_BUFFER, mesh_gpu.vbo_pos);
            glBufferSubData(GL_ARRAY_BUFFER, 0,
                            MHR_VERTEX_FLOATS * sizeof(float),
                            tri_model->vertices);

            // Build MVP = projection * view
            float proj[16], view[16], mvp[16];
            mhr_camera_matrices(proj, view,
                                r.focal_length, r.pred_cam_t.data(),
                                frame_w, frame_h);


            //view[0]=1.0; view[1]=0.0; view[2]=0.0; view[3]=0.0;
            //view[4]=0.0; view[5]=1.0; view[6]=0.0; view[7]=0.0;
            //view[8]=0.0; view[9]=0.0; view[10]=1.0; view[11]=100.0;
            //view[12]=0.0; view[13]=0.0; view[14]=0.0; view[15]=1.0;
            mat4_mul(mvp, proj, view);
            //mat4_transpose(mvp);
            //mat4_print("Projection",proj);
            //mat4_print("View",view);
            //mat4_print("MVP",mvp);

#if 0 /* DEBUG: view-space and clip-space bounds — re-enable to diagnose projection/clipping */
            // Debug: view-space and clip-space bounds
            { float vxmin=1e9f,vxmax=-1e9f,vymin=1e9f,vymax=-1e9f,vzmin=1e9f,vzmax=-1e9f;
              float cxmin=1e9f,cxmax=-1e9f,cymin=1e9f,cymax=-1e9f,czmin=1e9f,czmax=-1e9f,cwmin=1e9f,cwmax=-1e9f;
              for (int i=0; i<MHR_VERTEX_FLOATS; i+=3) {
                  // View space: apply full view matrix (diagonal -1 for Y,Z + translation)
                  float vx =  lbs_out[i]   + view[12];
                  float vy = -lbs_out[i+1] + view[13];
                  float vz = -lbs_out[i+2] + view[14];
                  if(vx<vxmin)vxmin=vx;
                  if(vx>vxmax)vxmax=vx;
                  if(vy<vymin)vymin=vy;
                  if(vy>vymax)vymax=vy;
                  if(vz<vzmin)vzmin=vz;
                  if(vz>vzmax)vzmax=vz;
                  // Clip space (MVP * vertex)
                  float wx = mvp[0]*lbs_out[i]   + mvp[4]*lbs_out[i+1] + mvp[8]*lbs_out[i+2] + mvp[12];
                  float wy = mvp[1]*lbs_out[i]   + mvp[5]*lbs_out[i+1] + mvp[9]*lbs_out[i+2] + mvp[13];
                  float wz = mvp[2]*lbs_out[i]   + mvp[6]*lbs_out[i+1] + mvp[10]*lbs_out[i+2]+ mvp[14];
                  float ww = mvp[3]*lbs_out[i]   + mvp[7]*lbs_out[i+1] + mvp[11]*lbs_out[i+2]+ mvp[15];
                  if(wx<cxmin)cxmin=wx;
                  if(wx>cxmax)cxmax=wx;
                  if(wy<cymin)cymin=wy;
                  if(wy>cymax)cymax=wy;
                  if(wz<czmin)czmin=wz;
                  if(wz>czmax)czmax=wz;
                  if(ww<cwmin)cwmin=ww;
                  if(ww>cwmax)cwmax=ww;
              }
              printf("[mesh] view bounds: x[%.3f,%.3f] y[%.3f,%.3f] z[%.3f,%.3f]\n", vxmin,vxmax, vymin,vymax, vzmin,vzmax);
              printf("[mesh] clip w=[%.3f,%.3f]  ndcX=[%.3f,%.3f]  ndcY=[%.3f,%.3f]  ndcZ=[%.3f,%.3f]\n",
                     cwmin,cwmax,
                     cxmin/cwmax, cxmax/cwmin, // worst-case NDC
                     cymin/cwmax, cymax/cwmin,
                     czmin/cwmax, czmax/cwmin);
            }
#endif

            glUniformMatrix4fv(mvp_loc, 1, GL_FALSE, mvp);
            glUniformMatrix4fv(view_loc, 1, GL_FALSE, view);

            glBindVertexArray(mesh_gpu.vao);
            glDrawElements(GL_TRIANGLES, mesh_gpu.n_indices,
                           GL_UNSIGNED_INT, nullptr);
            GLenum err = glGetError();
            if (err != GL_NO_ERROR)
                fprintf(stderr, "[GL] error 0x%04X after draw\n", err);
        }
        glBindVertexArray(0);

        // Save this frame before buffer swap when --save-frames is active
        if (!save_frames_prefix.empty()) {
            char path[4096];
            snprintf(path, sizeof(path), "%s%05d.jpg",
                     save_frames_prefix.c_str(), ++save_frame_idx);
            save_framebuffer(path, W, H);
        }

        glx3_endRedraw();

        // Status line: FPS (EMA), inference latency, subjects in view
        { long long t_now   = NS_NOW();
          double frame_ms   = (t_now - t_last_frame) / 1e6;
          t_last_frame      = t_now;
          fps_ema = (fps_ema == 0.0) ? (1000.0 / frame_ms)
                                     : (0.9 * fps_ema + 0.1 * (1000.0 / frame_ms));
          fprintf(stderr, "\r  FPS: %5.1f  Latency: %4.0f ms  Subjects: %d   ",
                  fps_ema, latency_ms, (int)results.size());
          fflush(stderr);
        }

        if (is_image) break;   // keep window open only for live sources
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    if (bvh_writer.is_open()) bvh_writer.close();
    mhr_lbs_free(lbs);
    tri_freeModel(tri_model);
    stop_glx3_stuff();
    return 0;
}
