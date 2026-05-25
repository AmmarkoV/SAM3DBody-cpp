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

// Body mesh — simple directional light from fixed direction.
static const char* MESH_VERT = R"glsl(
    #version 330 core
    layout(location=0) in vec3 aPos;
    layout(location=1) in vec3 aNorm;
    uniform mat4 uMVP;
    out vec3 vNorm;
    void main() {
        gl_Position = uMVP * vec4(aPos, 1.0);
        vNorm = aNorm;
    }
)glsl";

static const char* MESH_FRAG = R"glsl(
    #version 330 core
    in  vec3 vNorm;
    out vec4 fragColor;
    void main() {
        vec3 L = normalize(vec3(0.3, 0.8, 0.5));
        float d = clamp(dot(normalize(vNorm), L), 0.0, 1.0) * 0.7 + 0.3;
        fragColor = vec4(vec3(0.65, 0.75, 0.9) * d, 0.7);
    }
)glsl";

// ── GL helpers ───────────────────────────────────────────────────────────────

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
static void upload_bg_frame(BgTex& t, const cv::Mat& bgr)
{
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    if (rgb.empty() || rgb.data == nullptr) {
        fprintf(stderr, "[CV] upload_bg_frame: empty or null RGB image (%dx%d)\n", bgr.cols, bgr.rows);
        return;
    }

    // Enforce 1-byte unpack alignment — OpenCV data is tightly packed and
    // may not satisfy GL_UNPACK_ALIGNMENT=4, causing glTexImage2D to read
    // past the buffer end on rows where cols*3 % 4 != 0.
    GLint old_unpack;
    glGetIntegerv(GL_UNPACK_ALIGNMENT, &old_unpack);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glBindTexture(GL_TEXTURE_2D, t.id);
    if (!t.ready || bgr.cols != t.w || bgr.rows != t.h) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB,
                     bgr.cols, bgr.rows, 0,
                     GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
        t.w = bgr.cols; t.h = bgr.rows; t.ready = true;
    } else {
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0,
                        bgr.cols, bgr.rows,
                        GL_RGB, GL_UNSIGNED_BYTE, rgb.data);
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    glPixelStorei(GL_UNPACK_ALIGNMENT, old_unpack);
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

// ── 2nd-order Butterworth low-pass filter (Direct Form II transposed) ────────
// Default coefficients: fc=2 Hz at 30 fps
//   b = [0.02008, 0.04016, 0.02008], a = [1, -1.56102, 0.64135]
struct BwFilter {
    // Coefficients default to ~1.5 Hz at 30 fps; call init() to override.
    float b0 = 0.02008f, b1 = 0.04016f, b2 = 0.02008f;
    float a1 = -1.56102f, a2 = 0.64135f;
    bool  active = true;   // false when fc >= fs/2 (Nyquist): filter becomes a pass-through
    std::vector<float> s1, s2, prev_x, acc;
    bool ready = false;

    // Compute 2nd-order Butterworth coefficients for the given cutoff / sample rate.
    // Group delay at DC ≈ 1/(π·fc) seconds — halving fc doubles the lag.
    // fc must be < fs/2; above Nyquist the filter is disabled (pass-through).
    void init(float fc, float fs) {
        if (fc <= 0.f || fc >= fs * 0.5f) { active = false; return; }
        active = true;
        float ff  = fc / fs;
        float ita = 1.f / tanf(3.14159265f * ff);
        float q   = 1.41421356f;
        b0  =  1.f / (1.f + q * ita + ita * ita);
        b1  =  2.f * b0;
        b2  =  b0;
        // Stored negated to match the transposed-direct-form-II sign convention used in apply().
        a1  = -2.f * (ita * ita - 1.f) * b0;
        a2  =  (1.f - q * ita + ita * ita) * b0;
    }

    void apply(float* data, int n) {
        if (!active) return;
        if ((int)s1.size() != n) {
            s1.assign(n, 0.f); s2.assign(n, 0.f);
            prev_x.assign(n, 0.f); acc.assign(n, 0.f);
            ready = false;
        }
        if (!ready) {
            // Warm-start: output equals first input (no transient).
            // Initialise the unwrapped accumulator to the first raw value.
            for (int i = 0; i < n; ++i) {
                float x   = data[i];
                acc[i]    = x;
                prev_x[i] = x;
                s1[i] = (1.f - b0) * x;
                s2[i] = (b2 - a2)  * x;
            }
            ready = true;
            return;
        }
        for (int i = 0; i < n; ++i) {
            float x     = data[i];
            // Wrap the per-frame delta to [-π, π] before accumulating so that
            // Euler angle discontinuities (±π boundary crossings) don't cause the
            // filter to interpolate through the discontinuity and produce a flip.
            // For translation channels the delta is tiny (<< π) so wrapping is a no-op.
            float delta = x - prev_x[i];
            while (delta >  3.14159265f) delta -= 6.28318530f;
            while (delta < -3.14159265f) delta += 6.28318530f;
            acc[i]    += delta;
            prev_x[i]  = x;
            float y  = b0 * acc[i] + s1[i];
            s1[i]    = b1 * acc[i] - a1 * y + s2[i];
            s2[i]    = b2 * acc[i] - a2 * y;
            data[i]  = y;
        }
    }
};

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, const char** argv) {
    std::string onnx_dir  = "./onnx";
    std::string gguf_path = "./onnx/pipeline.gguf";
    std::string yolo_path = "./onnx/yolo.onnx";
    std::string mesh_path = "./body_mesh.tri";
    std::string lbs_path  = "";
    std::string src       = "0";
    std::string save_frames_prefix = "";
    int         save_frame_idx     = 0;
    std::string bvh_path           = "";
    std::string bvh_template       = "";
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

    for (int i = 1; i < argc; ++i) {
#define A1(flag, field, conv) \
        if (!strcmp(argv[i], flag) && i+1<argc) { field = conv(argv[++i]); continue; }
        A1("--onnx-dir", onnx_dir,  std::string)
        A1("--gguf",     gguf_path, std::string)
        A1("--yolo",     yolo_path, std::string)
        A1("--mesh",     mesh_path, std::string)
        A1("--lbs",      lbs_path,  std::string)
        A1("--from",        src,                std::string)
        A1("--save-frames", save_frames_prefix, std::string)
        A1("--cuda",        cuda_device,        std::stoi)
        A1("--bvh",          bvh_path,     std::string)
        A1("--bvh-template", bvh_template, std::string)
        A1("--bw-cutoff",    bw_cutoff,    std::stof)
#undef A1
        if (!strcmp(argv[i], "--render-size") && i+2 < argc)
            { render_w = std::stoi(argv[++i]); render_h = std::stoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--size") && i+2 < argc)
            { cap_w = std::stoi(argv[++i]); cap_h = std::stoi(argv[++i]); continue; }
        if (!strcmp(argv[i], "--fps") && i+1 < argc)
            { cap_fps = std::stod(argv[++i]); continue; }
        if (!strcmp(argv[i], "--trt"))         { use_trt        = true;  continue; }
        if (!strcmp(argv[i], "--no-fp16"))     { fp16           = false; continue; }
        if (!strcmp(argv[i], "--dev-face"))    { zero_face      = false; continue; }
        if (!strcmp(argv[i], "--butterworth"))              { use_butterworth  = true; continue; }
        if (!strcmp(argv[i], "--butterworth-root-rotation")){ filter_root_rot  = true; continue; }
        if (!strcmp(argv[i], "--rot-clamp") && i+1 < argc) { rot_clamp_deg = std::stof(argv[++i]); continue; }
    }

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

    // ── GLX window ────────────────────────────────────────────────────────────
    if (!start_glx3_stuff(W, H, 1, argc, argv)) {
        fprintf(stderr, "Failed to start GLX window\n"); return 1;
    }
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "GLEW init failed\n"); return 1;
    }

    // ── Shaders ───────────────────────────────────────────────────────────────
    GLuint prog_quad = link_program(QUAD_VERT, QUAD_FRAG);
    GLuint prog_mesh = link_program(MESH_VERT, MESH_FRAG);
    GLint  mvp_loc   = glGetUniformLocation(prog_mesh, "uMVP");
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
        if (!bvh_writer.open(bvh_template, bvh_path, 1.f / video_fps, lbs_path))
            fprintf(stderr, "[BVH] Warning: could not open BVH writer\n");
        else
            printf("[BVH] Writing to %s (%.1f fps)\n", bvh_path.c_str(), video_fps);
    }

    // ── Butterworth filter state (one filter per channel group) ──────────────
    BwFilter bw_mp;   // for mhr_model_params[204]
    BwFilter bw_cam;  // for pred_cam_t[3]
    if (use_butterworth) {
        bw_mp.init(bw_cutoff, video_fps);
        bw_cam.init(bw_cutoff, video_fps);
        float lag_ms = 1000.f / (3.14159f * bw_cutoff);
        printf("[BW] cutoff=%.1f Hz  sample=%.1f Hz  approx lag=%.0f ms (%.1f frames)\n",
               bw_cutoff, video_fps, lag_ms, lag_ms * video_fps / 1000.f);
    }
    // global_rot rejection state (same wrap-corrected rejection as fast_sam_3dbody_run)
    std::array<float, 3> prev_global_rot{};
    bool global_rot_init = false;

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

        // Temporal smoothing (Butterworth IIR)
        if (use_butterworth && !results.empty()) {
            bw_mp.apply(results[0].mhr_model_params.data(), 204);
            bw_cam.apply(results[0].pred_cam_t.data(), 3);

            // global_rot: wrap-corrected rejection — only when --butterworth-root-rotation
            // is passed. Off by default: first-frame lock-in to a bad prediction would
            // freeze the entire sequence at the wrong orientation.
            if (filter_root_rot && rot_clamp_deg > 0.0f) {
                auto& gr = results[0].global_rot;
                if (!global_rot_init) {
                    prev_global_rot = gr;
                    global_rot_init = true;
                } else {
                    const float max_rad = rot_clamp_deg * (3.14159265359f / 180.0f);
                    bool reject = false;
                    for (int k = 0; k < 3; ++k) {
                        float delta = gr[k] - prev_global_rot[k];
                        while (delta >  3.14159265359f) delta -= 2.0f * 3.14159265359f;
                        while (delta < -3.14159265359f) delta += 2.0f * 3.14159265359f;
                        if (fabsf(delta) > max_rad) { reject = true; break; }
                    }
                    if (reject) gr = prev_global_rot;
                    else        prev_global_rot = gr;
                }
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

        // Upload background (with optional skeleton annotation)
        if (!bg.ready)
        {
         fprintf(stderr, "[GL] error bg not ready\n");
         exit(1);
        }
        
        if ( (vis.empty()) )
        {
         fprintf(stderr, "[CV] error upload_bg_frame\n");
         exit(1);
        }
        upload_bg_frame(bg, vis); //<-- THIS SEGFAULTS!

        glClearColor(0.f, 0.f, 0.f, 1.f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glViewport(0, 0, W, H);

        // ── Background quad ───────────────────────────────────────────────────
        glDisable(GL_DEPTH_TEST);
        glUseProgram(prog_quad);
        glUniform1i(tex_loc, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, bg.id);
        glBindVertexArray(quad_vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glEnable(GL_DEPTH_TEST);

        // ── Mesh overlay for each detected person ─────────────────────────────
        glUseProgram(prog_mesh);
        for (const auto& r : results) {
            if (!lbs) continue;

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
