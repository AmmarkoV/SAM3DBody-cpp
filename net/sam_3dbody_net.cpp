// ============================================================================
// sam_3dbody_net.cpp — networked client/server for SAM-3D-Body (CLIENTSERVER.md)
//
// One binary, two modes:
//
//   --server   Runs the GPU pipeline behind an AmmarServer HTTP endpoint.
//              POST /infer  (a JPEG file)  → response body = that frame's poses
//              as line-oriented SAM3D text.  Also accumulates a server-side BVH.
//
//   --client   Grabs webcam frames, POSTs the newest one to the server, parses
//              the SAM3D text response.  Synchronous request/response keeps
//              exactly one frame in flight; the camera is drained to the newest
//              frame before each POST (drop-latest) so we stay real-time.
//
// The SAM3D text serializer (server) and parser (client) live together here so
// the wire format has a single source of truth.  Transport is AmmarServer /
// AmmClient (vendored under ./AmmarServer); the HTTP response *is* the return
// channel, so there is no custom framing.
// ============================================================================

#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <unistd.h>

#include <opencv2/opencv.hpp>

#include "fast_sam_3dbody.h"
#include "bvh_writer.h"

extern "C" {
#include "AmmServerlib.h"
#include "AmmClient.h"
}

// ── SAM3D text format version (bump when the contract changes) ───────────────
static const int SAM3D_FORMAT_VERSION = 1;

// ============================================================================
// Configuration (shared; only the relevant fields are used per mode)
// ============================================================================
struct NetConfig {
    bool server = false;
    bool client = false;

    // Network
    std::string host = "127.0.0.1";   // client: server address
    int         port = 8080;
    int         timeout_s = 10;        // client socket timeout

    // Client capture
    std::string from = "0";            // webcam index or /dev/videoX or file
    int  cap_w = 0, cap_h = 0;
    double cap_fps = 0.0;
    int  jpeg_quality = 80;
    bool raw_print = false;            // dump the raw response body

    // Server pipeline (mirrors main.cpp defaults)
    std::string onnx_dir = "./onnx";
    std::string gguf_path = "./onnx/pipeline.gguf";
    std::string yolo_path = "./onnx/yolo.onnx";
    int   cuda_device = 0;
    bool  use_trt = false;
    bool  skip_body = false;
    int   max_persons = 0;
    float thresh = 0.50f;

    // Server BVH (empty = don't write one)
    std::string bvh_path;
    std::string bvh_template = "./body_mhr.bvh";
    double      bvh_fps = 30.0;        // nominal frame time stamped into the BVH
    int         idle_finalize_s = 5;   // finalize the BVH after this idle gap

    // Per-joint TF transforms in the response (JLOCAL lines).  Needs the LBS
    // body model (onnx-dir/body_model.lbs); works even with --skip-body.
    bool        jlocal = false;
};

// ============================================================================
// SAM3D text serialization (server side)
// ============================================================================
static void append_floats(std::string& s, const char* tag,
                          const float* v, int n)
{
    char b[32];
    s += tag;
    for (int i = 0; i < n; ++i) {
        snprintf(b, sizeof(b), " %.6g", v[i]);
        s += b;
    }
    s += '\n';
}

// Build the response body for one processed frame.  betas_sent tracks which
// track ids have already had their (near-constant) shape/scale emitted, so we
// only send BETAS on a track's first frame.
//
// ── Wire-format extension contract (forward compatibility) ──────────────────
// The format is line-oriented: each line's first whitespace token is a record
// type the consumer dispatches on.  Unknown tokens MUST be skipped, so new
// record types are added without breaking existing clients; bump
// SAM3D_FORMAT_VERSION only for incompatible changes.  Currently emitted:
//   SAM3D P GROT CAMT BPOSE HPOSE BETAS KP3D KP2D END
//   JLOCAL <joint> <parent> qx qy qz qw tx ty tz   per-joint LOCAL transform —
//        parent-relative quaternion (xyzw) + translation (metres) = a TF
//        broadcast set; root parent is "camera".  Emitted per person when the
//        server runs with --jlocal (source: BVHWriter::compute_joint_locals).
// Reserved for future ROS / TF-broadcast clients (see CLIENTSERVER.md
// "Forward compatibility: ROS / TF outputs") — add as new per-person lines:
//   HIER   <joint> <parent> ...                     joint tree, once per track.
//   ROOTQ  qx qy qz qw tx ty tz                      camera->root as a quaternion.
//   CONF   <70 floats>                               per-joint confidence.
// Header may also gain: coord=<frame conv> units=m  (camera optical frame;
// client remaps to ROS REP-103) and a real capture stamp once t_ms round-trips.
// To avoid response bloat when several output types coexist, gate emission by a
// client-selected output set (a future per-request 'out=' selector / server
// --outputs flag); the default stays the full set above.
static std::string format_results(unsigned seq, unsigned t_ms,
                                  const std::vector<fsb::MHRResult>& res,
                                  const std::vector<int>& ids,
                                  std::unordered_map<int,bool>& betas_sent,
                                  const std::vector<std::vector<BVHWriter::JointLocal>>* jlocals)
{
    std::string s;
    s.reserve(4096);
    char b[128];

    snprintf(b, sizeof(b), "SAM3D %d frame=%u t_ms=%u persons=%zu\n",
             SAM3D_FORMAT_VERSION, seq, t_ms, res.size());
    s += b;

    for (size_t i = 0; i < res.size(); ++i) {
        const fsb::MHRResult& r = res[i];
        int id = ids[i];

        snprintf(b, sizeof(b),
                 "P %zu id=%d bbox=%.6g,%.6g,%.6g,%.6g focal=%.6g\n",
                 i, id, r.bbox[0], r.bbox[1], r.bbox[2], r.bbox[3],
                 r.focal_length);
        s += b;

        append_floats(s, "GROT", r.global_rot.data(), 3);
        append_floats(s, "CAMT", r.pred_cam_t.data(), 3);
        if (!r.body_pose.empty())
            append_floats(s, "BPOSE", r.body_pose.data(), (int)r.body_pose.size());
        if (!r.hand_pose.empty())
            append_floats(s, "HPOSE", r.hand_pose.data(), (int)r.hand_pose.size());

        // BETAS only on a track's first frame (shape/scale are per-identity).
        if (!betas_sent[id]) {
            betas_sent[id] = true;
            if (!r.shape.empty()) {
                s += "BETAS shape=";
                for (size_t k = 0; k < r.shape.size(); ++k) {
                    snprintf(b, sizeof(b), "%s%.6g", k ? "," : "", r.shape[k]);
                    s += b;
                }
                s += " scale=";
                for (size_t k = 0; k < r.scale.size(); ++k) {
                    snprintf(b, sizeof(b), "%s%.6g", k ? "," : "", r.scale[k]);
                    s += b;
                }
                s += '\n';
            }
        }

        // Optional keypoints — only populated when the body model ran
        // (i.e. NOT --skip-body).  KP3D is metres in camera space; KP2D is
        // projected image pixels.
        if (!r.keypoints_3d.empty()) {
            snprintf(b, sizeof(b), "KP3D %zu", r.keypoints_3d.size() / 3);
            s += b;
            for (size_t k = 0; k + 2 < r.keypoints_3d.size(); k += 3) {
                snprintf(b, sizeof(b), " %.6g,%.6g,%.6g",
                         r.keypoints_3d[k], r.keypoints_3d[k+1], r.keypoints_3d[k+2]);
                s += b;
            }
            s += '\n';
        }
        if (!r.keypoints_2d.empty()) {
            snprintf(b, sizeof(b), "KP2D %zu", r.keypoints_2d.size() / 2);
            s += b;
            for (size_t k = 0; k + 1 < r.keypoints_2d.size(); k += 2) {
                snprintf(b, sizeof(b), " %.6g,%.6g",
                         r.keypoints_2d[k], r.keypoints_2d[k+1]);
                s += b;
            }
            s += '\n';
        }

        // Per-joint local transforms (TF tree) — only when the server runs with
        // --jlocal.  One line per MHR joint: parent-relative quaternion (xyzw) +
        // translation (metres); the root's parent token is "camera".
        if (jlocals && i < jlocals->size()) {
            char jb[256];
            for (const auto& jl : (*jlocals)[i]) {
                const char* par = jl.parent[0] ? jl.parent : "camera";
                snprintf(jb, sizeof(jb),
                         "JLOCAL %s %s %.6g %.6g %.6g %.6g %.6g %.6g %.6g\n",
                         jl.name, par, jl.q[0], jl.q[1], jl.q[2], jl.q[3],
                         jl.t[0], jl.t[1], jl.t[2]);
                s += jb;
            }
        }
    }

    snprintf(b, sizeof(b), "END frame=%u\n", seq);
    s += b;
    return s;
}

// ============================================================================
// Server
// ============================================================================
namespace srv {

static NetConfig            g_cfg;
static fsb::Pipeline        g_pipeline;
static BVHWriter            g_bvh;
static bool                 g_bvh_enabled = false;   // write a server-side .bvh
static bool                 g_jlocal_enabled = false; // emit JLOCAL TF lines
static bool                 g_bvh_open    = false;
static std::mutex           g_mutex;          // guards pipeline + tracker + bvh
static std::unordered_map<int,bool> g_betas_sent;

static AmmServer_Instance*  g_server = nullptr;
static AmmServer_RH_Context g_inferCtx = {};
static AmmServer_RH_Context g_closeCtx = {};
static AmmServer_RH_Context g_rootCtx  = {};

static std::atomic<unsigned long> g_last_activity_ms{0};
static std::atomic<bool>          g_dirty{false};   // frames written since last finalize

// ── Lightweight greedy bbox-IoU tracker (same idea as the live BVHWriter) ────
struct Track { int id; float bbox[4]; int last_seen; };
static std::vector<Track> g_tracks;
static int g_next_id   = 0;
static int g_frame_idx = 0;
static const int RETIRE_FRAMES = 90;   // ~3 s @ 30 fps
static const float IOU_THRESH  = 0.10f;

static float iou(const float a[4], const float b[4])
{
    float x1 = std::max(a[0], b[0]), y1 = std::max(a[1], b[1]);
    float x2 = std::min(a[2], b[2]), y2 = std::min(a[3], b[3]);
    float iw = x2 - x1, ih = y2 - y1;
    if (iw <= 0 || ih <= 0) return 0.f;
    float inter = iw * ih;
    float ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter;
    return ua > 0 ? inter / ua : 0.f;
}

// Assign stable ids to this frame's detections; also report which existing
// (still-alive) tracks were NOT detected this frame so the BVH writer can pad
// them and keep all per-person frame counts in lock-step.
static std::vector<int> assign_tracks(const std::vector<fsb::MHRResult>& res,
                                      std::vector<int>& pad_ids)
{
    ++g_frame_idx;
    const int n = (int)res.size();
    std::vector<int> ids(n, -1);

    // Candidate (iou, det, trk) pairs above threshold, greedily matched.
    struct Cand { float iou; int d; int t; };
    std::vector<Cand> cand;
    for (int d = 0; d < n; ++d)
        for (int t = 0; t < (int)g_tracks.size(); ++t) {
            float v = iou(res[d].bbox.data(), g_tracks[t].bbox);
            if (v >= IOU_THRESH) cand.push_back({v, d, t});
        }
    std::sort(cand.begin(), cand.end(),
              [](const Cand& a, const Cand& b){ return a.iou > b.iou; });

    std::vector<char> used_d(n, 0), used_t(g_tracks.size(), 0);
    for (const Cand& c : cand) {
        if (used_d[c.d] || used_t[c.t]) continue;
        used_d[c.d] = used_t[c.t] = 1;
        ids[c.d] = g_tracks[c.t].id;
        std::memcpy(g_tracks[c.t].bbox, res[c.d].bbox.data(), sizeof(float)*4);
        g_tracks[c.t].last_seen = g_frame_idx;
    }
    // Unmatched detections spawn new tracks.
    for (int d = 0; d < n; ++d) {
        if (ids[d] >= 0) continue;
        Track tk; tk.id = g_next_id++; tk.last_seen = g_frame_idx;
        std::memcpy(tk.bbox, res[d].bbox.data(), sizeof(float)*4);
        g_tracks.push_back(tk);
        ids[d] = tk.id;
    }
    // Alive-but-missing tracks → pad; too-old tracks → retire.
    pad_ids.clear();
    std::vector<Track> kept;
    kept.reserve(g_tracks.size());
    for (const Track& t : g_tracks) {
        int age = g_frame_idx - t.last_seen;
        if (age == 0) { kept.push_back(t); continue; }      // detected this frame
        if (age <= RETIRE_FRAMES) { pad_ids.push_back(t.id); kept.push_back(t); }
        // else: drop
    }
    g_tracks.swap(kept);
    return ids;
}

static unsigned long now_ms() { return AmmClient_GetTickCountMilliseconds(); }

static void ensure_bvh_open()
{
    if ((!g_bvh_enabled && !g_jlocal_enabled) || g_bvh_open) return;
    std::string lbs = g_cfg.onnx_dir + "/body_model.lbs";
    // JLOCAL-only mode still needs an open writer (for its LBS + FK scratch); it
    // never calls write_frame, so a placeholder out path is fine — no files are
    // produced unless --bvh actually feeds frames.
    std::string out = g_cfg.bvh_path.empty() ? "/tmp/.sam3dnet_jlocal" : g_cfg.bvh_path;
    if (g_bvh.open(g_cfg.bvh_template, out,
                   1.0f / (float)g_cfg.bvh_fps, lbs)) {
        g_bvh_open = true;
    } else {
        fprintf(stderr, "[server] BVH/LBS open failed (template=%s lbs=%s) — "
                "disabling BVH + JLOCAL\n", g_cfg.bvh_template.c_str(), lbs.c_str());
        g_bvh_enabled = false;
        g_jlocal_enabled = false;
    }
}

// Finalize the current capture: write the BVH files and reset session state so
// a subsequent /infer starts a fresh one.
static void finalize_session(const char* why)
{
    if (g_bvh_open) {
        g_bvh.close();
        g_bvh_open = false;
        fprintf(stderr, "[server] BVH finalized (%s) → %s_<id>.bvh\n", why,
                g_cfg.bvh_path.c_str());
    }
    g_tracks.clear();
    g_betas_sent.clear();
    g_next_id = 0;
    g_frame_idx = 0;
    g_dirty = false;
}

// POST /infer — the hot path.
static void* infer_callback(AmmServer_DynamicRequest* rqst)
{
    unsigned int jpegSize = 0;
    const char* jpeg = _FILES(rqst, "uploadedfile", VALUE, &jpegSize);
    if (!jpeg || jpegSize == 0) {
        const char* msg = "SAM3D error no-image\n";
        size_t m = std::min((size_t)rqst->MAXcontentSize, strlen(msg));
        std::memcpy(rqst->content, msg, m);
        rqst->contentSize = m;
        return 0;
    }

    unsigned int seq  = _GETuint(rqst, "seq");
    unsigned int t_ms = _GETuint(rqst, "t_ms");

    std::string body;
    {
        std::lock_guard<std::mutex> lk(g_mutex);

        cv::Mat enc(1, (int)jpegSize, CV_8U, (void*)jpeg);
        cv::Mat bgr = cv::imdecode(enc, cv::IMREAD_COLOR);
        if (bgr.empty()) {
            const char* msg = "SAM3D error decode-failed\n";
            size_t m = std::min((size_t)rqst->MAXcontentSize, strlen(msg));
            std::memcpy(rqst->content, msg, m);
            rqst->contentSize = m;
            return 0;
        }

        std::vector<fsb::MHRResult> res =
            g_pipeline.process_bgr(bgr.data, bgr.cols, bgr.rows);

        std::vector<int> pad_ids;
        std::vector<int> ids = assign_tracks(res, pad_ids);

        if (g_bvh_enabled || g_jlocal_enabled) ensure_bvh_open();
        if (g_bvh_enabled && g_bvh_open) {
            g_bvh.write_frame_external(res, ids, pad_ids);
            g_dirty = true;
        }

        // Per-person TF transforms (JLOCAL), computed by the same writer FK.
        std::vector<std::vector<BVHWriter::JointLocal>> jlocals;
        if (g_jlocal_enabled && g_bvh_open) {
            jlocals.resize(res.size());
            for (size_t i = 0; i < res.size(); ++i)
                g_bvh.compute_joint_locals(res[i], jlocals[i]);
        }

        body = format_results(seq, t_ms, res, ids, g_betas_sent,
                              g_jlocal_enabled ? &jlocals : nullptr);
    }

    g_last_activity_ms = now_ms();

    if (body.size() > rqst->MAXcontentSize) {
        fprintf(stderr, "[server] response %zu > MAXcontentSize %lu — truncating\n",
                body.size(), rqst->MAXcontentSize);
        body.resize(rqst->MAXcontentSize);
    }
    std::memcpy(rqst->content, body.data(), body.size());
    rqst->contentSize = body.size();
    return 0;
}

// POST /close — flush the BVH for this capture.
static void* close_callback(AmmServer_DynamicRequest* rqst)
{
    {
        std::lock_guard<std::mutex> lk(g_mutex);
        finalize_session("client /close");
    }
    const char* msg = "SAM3D closed\n";
    size_t m = std::min((size_t)rqst->MAXcontentSize, strlen(msg));
    std::memcpy(rqst->content, msg, m);
    rqst->contentSize = m;
    return 0;
}

// GET / — minimal browser page: pick an image, POST it to /infer, show the
// SAM3D text response on the same page.  Uses the same multipart field name
// ("uploadedfile") as the CLI client, so it exercises the identical path.
// Embedded (not a disk file) so the server stays self-contained.  '\n' inside
// the JS is written as "\\n" so the served bytes are a literal backslash-n.
static const char INDEX_HTML[] =
"<!doctype html><html><head><meta charset=\"utf-8\"><title>SAM3D-net</title>"
"<style>body{font-family:sans-serif;margin:2em;max-width:900px}"
"pre{background:#111;color:#0f0;padding:1em;overflow:auto;white-space:pre-wrap}"
"img{max-width:320px;display:block;margin:1em 0;border:1px solid #ccc}</style>"
"</head><body><h2>SAM-3D-Body &mdash; upload an image</h2>"
"<input type=\"file\" id=\"f\" accept=\"image/*\"><img id=\"pv\">"
"<pre id=\"out\">Pick an image&hellip;</pre><script>"
"const f=document.getElementById('f'),out=document.getElementById('out'),pv=document.getElementById('pv');"
"f.addEventListener('change',async()=>{"
"const file=f.files[0];if(!file)return;"
"pv.src=URL.createObjectURL(file);out.textContent='Running\\u2026';"
"const fd=new FormData();fd.append('uploadedfile',file,file.name);"
"try{const t0=performance.now();"
"const r=await fetch('/infer',{method:'POST',body:fd});"
"const txt=await r.text();"
"out.textContent='('+Math.round(performance.now()-t0)+' ms)\\n'+txt;"
"}catch(e){out.textContent='Error: '+e;}});"
"</script></body></html>";

static void* root_callback(AmmServer_DynamicRequest* rqst)
{
    size_t n = sizeof(INDEX_HTML) - 1;   // exclude the trailing NUL
    if (n > rqst->MAXcontentSize) n = rqst->MAXcontentSize;
    memcpy(rqst->content, INDEX_HTML, n);
    rqst->contentSize = n;
    return 0;
}

static void on_terminate()
{
    std::lock_guard<std::mutex> lk(g_mutex);
    finalize_session("server shutdown");
}

static int run(const NetConfig& cfg)
{
    g_cfg = cfg;
    g_bvh_enabled = !cfg.bvh_path.empty();
    g_jlocal_enabled = cfg.jlocal;

    // ── Load the pipeline once, kept warm for the server's lifetime ─────────
    fsb::PipelineConfig pcfg;
    pcfg.onnx_dir        = cfg.onnx_dir;
    pcfg.gguf_path       = cfg.gguf_path;
    pcfg.yolo_path       = cfg.yolo_path;
    pcfg.cuda_device     = cfg.cuda_device;
    pcfg.use_trt_ep      = cfg.use_trt;
    pcfg.skip_body_model = cfg.skip_body;
    pcfg.max_persons     = cfg.max_persons;
    pcfg.person_thresh   = cfg.thresh;

    fprintf(stderr, "[server] loading pipeline…\n");
    if (!g_pipeline.load(pcfg)) {
        fprintf(stderr, "[server] pipeline load failed\n");
        return 1;
    }
    g_pipeline.print_info();

    // ── Start AmmarServer ───────────────────────────────────────────────────
    AmmServer_CheckIfHeaderBinaryAreTheSame(AMMAR_SERVER_HTTP_HEADER_SPEC);
    AmmServer_RegisterTerminationSignal((void*)&on_terminate);

    char bindIP[MAX_IP_STRING_SIZE];
    strncpy(bindIP, "0.0.0.0", MAX_IP_STRING_SIZE);

    // We serve no static files — point both roots at the cwd so Start never
    // fails on a missing public_html/.
    static char web_root[]  = "./";
    static char tmpl_root[] = "./";
    g_server = AmmServer_Start("sam3dnet", bindIP, cfg.port, 0, web_root, tmpl_root);
    if (!g_server) {
        fprintf(stderr, "[server] AmmServer_Start failed on port %d\n", cfg.port);
        g_pipeline.free();
        return 1;
    }

    // Per-resource max request size — must comfortably hold a JPEG frame plus
    // the multipart envelope (and the pose-text response).  8 MB is far above
    // any webcam-q80 frame.
    AmmServer_SetIntSettingValue(g_server, AMMSET_MAX_POST_TRANSACTION_SIZE,
                                 8 * 1024 * 1024);

    AmmServer_AddResourceHandler(g_server, &g_inferCtx, "/infer",
                                 8 * 1024 * 1024, 0, (void*)&infer_callback,
                                 DIFFERENT_PAGE_FOR_EACH_CLIENT | ENABLE_RECEIVING_FILES);
    AmmServer_DoNOTCacheResourceHandler(g_server, &g_inferCtx);

    AmmServer_AddResourceHandler(g_server, &g_closeCtx, "/close",
                                 4096, 0, (void*)&close_callback,
                                 DIFFERENT_PAGE_FOR_EACH_CLIENT);
    AmmServer_DoNOTCacheResourceHandler(g_server, &g_closeCtx);

    AmmServer_AddResourceHandler(g_server, &g_rootCtx, "/index.html",
                                 16 * 1024, 0, (void*)&root_callback,
                                 SAME_PAGE_FOR_ALL_CLIENTS);

    fprintf(stderr, "[server] ready on :%d  (POST /infer, /close)%s%s\n", cfg.port,
            g_bvh_enabled ? "  [BVH on]" : "",
            g_jlocal_enabled ? "  [JLOCAL on]" : "");

    // ── Main loop: idle-timeout BVH finalize ────────────────────────────────
    g_last_activity_ms = now_ms();
    while (AmmServer_Running(g_server)) {
        sleep(1);
        if (g_bvh_enabled && g_dirty) {
            unsigned long idle = now_ms() - g_last_activity_ms;
            if (idle > (unsigned long)cfg.idle_finalize_s * 1000UL) {
                std::lock_guard<std::mutex> lk(g_mutex);
                finalize_session("idle timeout");
            }
        }
    }

    on_terminate();
    AmmServer_Stop(g_server);
    g_pipeline.free();
    return 0;
}

} // namespace srv

// ============================================================================
// Client
// ============================================================================
namespace cli {

// Read a full HTTP response into buf: loop AmmClient_Recv until the header
// terminator (\r\n\r\n) is present AND we have Content-Length body bytes.
// AmmarServer sends the response header and body as separate segments, so a
// single recv can land short of the terminator.  Returns body pointer + length,
// or nullptr on failure.
static char* recv_http_response(AmmClient_Instance* inst,
                                std::vector<char>& buf,
                                unsigned int* outBodyLen)
{
    unsigned int total = 0;
    long contentLen = -1;
    while (total + 1 < buf.size()) {
        unsigned int chunk = (unsigned int)(buf.size() - 1 - total);
        if (!AmmClient_Recv(inst, buf.data() + total, &chunk)) return nullptr;
        if (chunk == 0) break;                 // peer closed
        total += chunk;
        buf[total] = 0;                        // null-terminate for strstr

        unsigned int afterHdr = total;
        char* body = AmmClient_seekEndOfHeader(buf.data(), &afterHdr);
        if (!body) continue;                   // header terminator not in yet
        if (contentLen < 0) {
            char* cl = strstr(buf.data(), "Content-Length:");
            if (cl) contentLen = atol(cl + 15);
        }
        if (contentLen < 0) { *outBodyLen = afterHdr; return body; }   // no CL
        if (afterHdr >= (unsigned int)contentLen) {
            *outBodyLen = (unsigned int)contentLen;
            return body;
        }
    }
    return nullptr;
}

// Parse + print a SAM3D response body.  Minimal tokenizer: dispatch on the
// first token of each line.  (A real consumer would store the float arrays;
// here we surface the structure so the wire format is verifiable.)
static void parse_and_print(const char* body, unsigned int len, bool raw)
{
    if (raw) { fwrite(body, 1, len, stdout); printf("\n"); return; }

    std::string s(body, len);
    size_t pos = 0;
    int persons = 0;
    int jlocal = 0;
    while (pos < s.size()) {
        size_t eol = s.find('\n', pos);
        if (eol == std::string::npos) eol = s.size();
        std::string line = s.substr(pos, eol - pos);
        pos = eol + 1;
        if (line.empty()) continue;

        if (line.compare(0, 6, "SAM3D ") == 0) {
            printf("[client] %s\n", line.c_str());
        } else if (line.compare(0, 2, "P ") == 0) {
            ++persons;
            printf("[client]   %s\n", line.c_str());   // index, id, bbox, focal
        } else if (line.compare(0, 5, "KP3D ") == 0) {
            // "KP3D <n> x,y,z x,y,z ..." — show the count and the first point.
            int n = 0; char first[64] = "";
            sscanf(line.c_str(), "KP3D %d %63s", &n, first);
            printf("[client]     %d 3D points (metres); kp0=%s\n", n, first);
        } else if (line.compare(0, 7, "JLOCAL ") == 0) {
            ++jlocal;
        }
        // GROT/CAMT/BPOSE/HPOSE/BETAS/KP2D skipped in this minimal printer
        // (use --raw to dump the full body, including all KP3D/JLOCAL values).
    }
    if (persons == 0) printf("[client]   (no persons)\n");
    if (jlocal)       printf("[client]   %d JLOCAL TF transforms\n", jlocal);
}

static int run(const NetConfig& cfg)
{
    // ── Open the capture source (webcam index, /dev/videoX, or a file) ───────
    cv::VideoCapture cap;
    bool src_is_int = !cfg.from.empty() &&
                      cfg.from.find_first_not_of("0123456789") == std::string::npos;
    if (src_is_int) cap.open(std::stoi(cfg.from));
    else            cap.open(cfg.from);
    if (!cap.isOpened()) {
        fprintf(stderr, "[client] cannot open source: %s\n", cfg.from.c_str());
        return 1;
    }
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);            // drop-latest: minimal queue
    if (cfg.cap_w  > 0)  cap.set(cv::CAP_PROP_FRAME_WIDTH,  cfg.cap_w);
    if (cfg.cap_h  > 0)  cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.cap_h);
    if (cfg.cap_fps> 0)  cap.set(cv::CAP_PROP_FPS,          cfg.cap_fps);

    // ── Connect ──────────────────────────────────────────────────────────────
    AmmClient_Instance* inst =
        AmmClient_Initialize(cfg.host.c_str(), cfg.port, cfg.timeout_s);
    if (!inst) {
        fprintf(stderr, "[client] cannot connect to %s:%d\n",
                cfg.host.c_str(), cfg.port);
        return 1;
    }
    fprintf(stderr, "[client] connected to %s:%d, streaming %s\n",
            cfg.host.c_str(), cfg.port, cfg.from.c_str());

    std::vector<uchar> jpeg;
    std::vector<int> jparams = {cv::IMWRITE_JPEG_QUALITY, cfg.jpeg_quality};
    const unsigned int RECV_CAP = 1024 * 1024;
    std::vector<char> recvbuf(RECV_CAP);

    unsigned int seq = 0;
    bool ok = true;
    while (ok) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            fprintf(stderr, "[client] capture ended\n");
            break;
        }

        if (!cv::imencode(".jpg", frame, jpeg, jparams)) {
            fprintf(stderr, "[client] jpeg encode failed\n");
            continue;
        }

        // seq + capture time travel as query params (SendFile carries the file).
        char uri[96];
        snprintf(uri, sizeof(uri), "/infer?seq=%u&t_ms=%lu",
                 seq, AmmClient_GetTickCountMilliseconds());

        unsigned long t0 = AmmClient_GetTickCountMicroseconds();
        if (!AmmClient_SendFile(inst, uri, "uploadedfile", "f.jpg", "image/jpeg",
                                (const char*)jpeg.data(), (unsigned int)jpeg.size(),
                                /*keepAlive=*/1)) {
            fprintf(stderr, "[client] send failed (seq %u)\n", seq);
            continue;
        }

        unsigned int bodyLen = 0;
        char* bodyStart = recv_http_response(inst, recvbuf, &bodyLen);
        unsigned long dt = AmmClient_GetTickCountMicroseconds() - t0;
        if (!bodyStart) { fprintf(stderr, "[client] no HTTP body (seq %u)\n", seq); continue; }

        printf("[client] seq %u  rtt %.1f ms\n", seq, dt / 1000.0);
        parse_and_print(bodyStart, bodyLen, cfg.raw_print);

        ++seq;
    }

    // Flush the server-side BVH for this session.
    AmmClient_SendFile(inst, "/close", "uploadedfile", "x", "text/plain",
                       "", 0, /*keepAlive=*/0);
    AmmClient_Close(inst);
    return 0;
}

} // namespace cli

// ============================================================================
// CLI
// ============================================================================
static void usage(const char* argv0)
{
    printf(
"sam_3dbody_net — networked SAM-3D-Body (CLIENTSERVER.md)\n\n"
"  %s --server [options]    run the GPU inference server\n"
"  %s --client [options]    stream a webcam to a server\n\n"
"Server options:\n"
"  --port N            bind port (default 8080)\n"
"  --onnx-dir DIR      models directory (default ./onnx)\n"
"  --gguf PATH         pipeline.gguf (default ./onnx/pipeline.gguf)\n"
"  --yolo PATH         yolo .onnx (default ./onnx/yolo.onnx)\n"
"  --cuda N            CUDA device (-1 = CPU; default 0)\n"
"  --trt               enable the TensorRT EP\n"
"  --skip-body         skip the body model (no vertices/keypoints)\n"
"  --max-persons N     cap persons per frame (0 = unlimited)\n"
"  --thresh T          YOLO person confidence (default 0.50)\n"
"  --bvh PATH          also write a server-side BVH (PATH_<id>.bvh)\n"
"  --bvh-template P    BVH template (default ./body_mhr.bvh)\n"
"  --bvh-fps Z         nominal BVH frame time (default 30)\n"
"  --idle-finalize S   finalize the BVH after S idle seconds (default 5)\n"
"  --jlocal            add per-joint TF transforms (JLOCAL lines) to responses\n"
"                      (parent-relative quat+offset, metres; needs body_model.lbs)\n\n"
"Client options:\n"
"  --host IP           server address (default 127.0.0.1)\n"
"  --port N            server port (default 8080)\n"
"  --from SRC          webcam index / /dev/videoX / file (default 0)\n"
"  --size W H          capture resolution\n"
"  --fps Z             capture framerate\n"
"  --jpeg-quality Q    JPEG quality 1-100 (default 80)\n"
"  --raw               print the raw SAM3D response body\n",
        argv0, argv0);
}

int main(int argc, char** argv)
{
    NetConfig c;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", what); exit(2); }
            return argv[++i];
        };
        if      (a == "--server")        c.server = true;
        else if (a == "--client")        c.client = true;
        else if (a == "--host")          c.host = next("--host");
        else if (a == "--port")          c.port = std::stoi(next("--port"));
        else if (a == "--from")          c.from = next("--from");
        else if (a == "--size")        { c.cap_w = std::stoi(next("--size")); c.cap_h = std::stoi(next("--size")); }
        else if (a == "--fps")           c.cap_fps = std::stod(next("--fps"));
        else if (a == "--jpeg-quality")  c.jpeg_quality = std::stoi(next("--jpeg-quality"));
        else if (a == "--raw")           c.raw_print = true;
        else if (a == "--onnx-dir")      c.onnx_dir = next("--onnx-dir");
        else if (a == "--gguf")          c.gguf_path = next("--gguf");
        else if (a == "--yolo")          c.yolo_path = next("--yolo");
        else if (a == "--cuda")          c.cuda_device = std::stoi(next("--cuda"));
        else if (a == "--trt")           c.use_trt = true;
        else if (a == "--skip-body")     c.skip_body = true;
        else if (a == "--max-persons")   c.max_persons = std::stoi(next("--max-persons"));
        else if (a == "--thresh")        c.thresh = std::stof(next("--thresh"));
        else if (a == "--bvh")           c.bvh_path = next("--bvh");
        else if (a == "--bvh-template")  c.bvh_template = next("--bvh-template");
        else if (a == "--bvh-fps")       c.bvh_fps = std::stod(next("--bvh-fps"));
        else if (a == "--idle-finalize") c.idle_finalize_s = std::stoi(next("--idle-finalize"));
        else if (a == "--jlocal")        c.jlocal = true;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown option: %s\n", a.c_str()); usage(argv[0]); return 2; }
    }

    if (c.server == c.client) {   // neither or both
        fprintf(stderr, "specify exactly one of --server / --client\n\n");
        usage(argv[0]);
        return 2;
    }
    return c.server ? srv::run(c) : cli::run(c);
}
