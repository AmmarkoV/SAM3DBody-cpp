#pragma once
// ════════════════════════════════════════════════════════════════════════════
//  cli_common.h
//
//  Shared CLI parsing for the three SAM3DBody-cpp binaries:
//      fast_sam_3dbody_run, fast_sam_3dbody_render, offline_sam_3dbody_render
//
//  Each binary used to have its own argv loop and parsed the same ~15 common
//  flags three times — with subtly different defaults (e.g. --rot-clamp was
//  1.0 in main, 1.0 in render, 30.0 in offline) and subtly different argv
//  conventions ("ARG1" macro vs hand-rolled strcmp).  This header collapses
//  the common subset into one parser so flag drift across binaries is no
//  longer a hand-maintenance task.
//
//  USAGE
//      CommonConfig cc;
//      cc.rot_clamp_deg = 30.0f;       // binary-specific default override
//      for (int i = 1; i < argc; ++i) {
//          if (parse_common_arg(argc, argv, i, cc)) continue;
//          // binary-specific flag dispatch lives here
//      }
//      ...
//      fsb::PipelineConfig pc;
//      apply_common_to_pipeline_cfg(cc, pc);
//
//  CONTRACT
//      parse_common_arg() returns true iff argv[i] matches a known common
//      flag.  Single-value flags (e.g. "--onnx-dir PATH") consume the next
//      argv element by incrementing i.  Boolean flags don't advance i.  The
//      outer for-loop's own ++i then moves past whichever element was last
//      consumed.
//
//      A flag NOT in the common set leaves i unchanged and returns false;
//      the binary's loop handles it locally.
//
//  WHAT IS / ISN'T COMMON
//      Common  ::=  the union of flags that have the same semantics in
//                   every binary that accepts them.  Binaries that don't
//                   read a particular field (e.g. `--thresh` ignored by
//                   the renderer) still let the parser populate it — the
//                   field just stays unread.  Accepting a flag we don't
//                   use is harmless and keeps the CLI uniform.
//
//      Not common ::= anything that's a mode-switch (e.g. --interpolate-
//                   jitter, --skip-body, --info) or has binary-specific
//                   semantics (e.g. --fps which means "webcam capture
//                   rate" in run but "source video FPS override" in
//                   offline).  Those stay in each binary's own argv loop.
//
//  CALLERS RESPONSIBLE FOR
//      * print_usage / --help — each binary still owns its own help text.
//        print_common_args_help() emits the common subset on demand so the
//        per-binary help can just call it as part of its own output.
// ════════════════════════════════════════════════════════════════════════════

#include <cctype>
#include <cstdio>
#include <cstdlib>    // ensure_trt_models(): std::system() to wget/unzip the TRT models
#include <cstring>
#include <filesystem> // ensure_models(): locate the repo's onnx/ relative to the exe
#include <fstream>    // resolve_backbone_defaults(): probe for backbone_fp16.onnx
#include <string>
#include <vector>     // ensure_models(): sentinel file list
#include <glob.h>     // resolve_detector_defaults(): find libreyolo*.onnx in onnx_dir
#include <unistd.h>   // ensure_trt_models(): readlink("/proc/self/exe") to locate setup_trt.sh

#include "fast_sam_3dbody.h"  // for fsb::PipelineConfig


struct CommonConfig
{
    // ── Pipeline (model paths + ONNX runtime knobs) ──────────────────────────
    std::string onnx_dir       = "./onnx";
    std::string gguf_path      = "./onnx/pipeline.gguf";
    std::string yolo_path      = "./onnx/yolo.onnx";
    // Backbone filename within onnx_dir.  cmake -DSAM3D_BACKBONE_QUANT=ON
    // bakes in "backbone_int8.onnx"; --backbone overrides at runtime.
#ifdef SAM3D_BACKBONE_QUANT
    std::string backbone_name  = "backbone_int8.onnx";
#else
    std::string backbone_name  = "backbone.onnx";
#endif
    std::string decoder_name   = "decoder.onnx";  // → decoder_fp16.onnx under --trt and on CPU
    int         cuda_device    = 0;       // -1 = CPU
    bool        use_trt        = false;
    bool        fp16           = true;    // can be disabled with --no-fp16

    // ── YOLO person detector tuning ──────────────────────────────────────────
    // The renderer doesn't use these (it inherits whatever the pipeline
    // chose internally) but accepting them keeps the CLI uniform — passing
    // `--thresh 0.6` to the renderer simply no-ops rather than erroring.
    float       person_thresh   = 0.50f;
    float       person_nms_iou  = 0.45f;
    int         max_persons     = 0;      // --max-persons N: 0 = unlimited; >0 = top-N by conf
    // --detector: bbox provider. "auto" (default) prefers a LibreYOLO model when
    // one is present in onnx_dir (or when --yolo points at a libreyolo/yolov9
    // export), else falls back to yolo-pose. resolve_detector_defaults() turns
    // this into a concrete name ("yolo-pose" | "libreyolo") before it is read.
    std::string detector        = "auto";
    // Did the user explicitly pass these on the command line?  Drives the
    // "auto" detector choice and the per-detector default threshold.
    bool        yolo_path_set   = false;  // --yolo given
    bool        thresh_set      = false;  // --thresh / --detector-threshold given
    bool        backbone_name_set = false; // --backbone given (pins the model; disables fp16 auto-prefer)

    // ── Input source ────────────────────────────────────────────────────────
    std::string from;             // file / webcam index / empty = required
    int         max_frames = -1; // --frames N: stop after N frames (-1 = unlimited)
    int         start_frame = 0; // --start N: skip to frame N before processing

    // ── BVH export ──────────────────────────────────────────────────────────
    std::string bvh_path;
    std::string bvh_template    = "./body_mhr.bvh";
    bool        bvh_body_shape_change          = true;   // --no-bvh-body-shape-change
    bool        bvh_hand_shape_change          = true;   // --no-bvh-hand-shape-change
    bool        bvh_compensate_finger_endsites = true;   // --bvh-raw-fingers
    bool        bvh_enforce_hand_limits        = true;   // default on; --no-enforce-hand-limits
    bool        bvh_zero_hand_pose             = false;  // --zero-hand-pose
    bool        bvh_sticky_hand_pose           = true;   // default on; --no-sticky-hand-pose
    bool        bvh_rest_align                 = true;    // --no-bvh-rest-align
    bool        bvh_dump_rest_dirs             = false;   // --dump-rest-dirs
    bool        bvh_foot_contact               = false;   // --foot-contact
    bool        bvh_static_root                = false;   // --bvh-static-root
    // Live streaming target: emit one BVH MOTION line per frame (single person)
    // to this path ("-" = stdout).  Consumed by scripts/webcam_gmr.sh →
    // tools/gmr_stream.py for the live webcam→robot pipeline.  Live binary only;
    // the offline binary ignores it.
    std::string bvh_stream_path;                          // --bvh-stream

    // Live shm transport (Linux/FSB_SHM only): publish each frame's BVH channels
    // into a SharedMemoryVideoBuffers generic buffer instead of (or besides) the
    // "@F" stdout line — no ASCII pipe, no per-frame temp .bvh on the consumer.
    // scripts/webcam_gmr.sh sets these automatically when the shm lib is present;
    // ignored by a binary built without FSB_SHM (falls back to --bvh-stream).
    std::string bvh_shm_descriptor;                       // --bvh-shm  (POSIX shm object name)
    std::string bvh_shm_stream = "bvh";                   // --bvh-shm-stream (feed name within it)

    // ── Filtering knobs ─────────────────────────────────────────────────────
    // Defaults match the live binaries; the offline binary overrides
    // rot_clamp_deg to 30.0 before invoking the parser (see comment in
    // its main()).
    float       bw_cutoff      = 6.0f;    // Hz
    float       rot_clamp_deg  = 1.0f;    // deg / frame
};


// ─── Argv walker ─────────────────────────────────────────────────────────────
// argv is taken as `const char* const*` so callers can pass either the C
// `const char **argv` (renderer) or the C++ `char **argv` (main / offline)
// without explicit casts.
inline bool parse_common_arg(int argc, const char* const* argv, int& i,
                             CommonConfig& c)
{
#define CLI_STR(flag, field)                                              \
    if (std::strcmp(argv[i], flag) == 0 && i + 1 < argc)                  \
    { c.field = argv[++i]; return true; }
#define CLI_INT(flag, field)                                              \
    if (std::strcmp(argv[i], flag) == 0 && i + 1 < argc)                  \
    { c.field = std::stoi(argv[++i]); return true; }
#define CLI_FLT(flag, field)                                              \
    if (std::strcmp(argv[i], flag) == 0 && i + 1 < argc)                  \
    { c.field = std::stof(argv[++i]); return true; }
#define CLI_BOOL(flag, field, val)                                        \
    if (std::strcmp(argv[i], flag) == 0)                                  \
    { c.field = (val); return true; }

    // Pipeline
    CLI_STR ("--onnx-dir",             onnx_dir)
    CLI_STR ("--gguf",                 gguf_path)
    // --yolo also records that the user pinned the model so the "auto" detector
    // selection won't second-guess their path.
    if (std::strcmp(argv[i], "--yolo") == 0 && i + 1 < argc)
    { c.yolo_path = argv[++i]; c.yolo_path_set = true; return true; }
    // --backbone records that the user pinned the model so resolve_backbone_defaults()
    // won't silently auto-upgrade them to backbone_fp16.onnx.
    if (std::strcmp(argv[i], "--backbone") == 0 && i + 1 < argc)
    { c.backbone_name = argv[++i]; c.backbone_name_set = true; return true; }
    CLI_STR ("--from",                 from)
    CLI_INT ("--frames",               max_frames)
    CLI_INT ("--start",                start_frame)
    CLI_INT ("--cuda",                 cuda_device)
    CLI_BOOL("--trt",                  use_trt, true)
    CLI_BOOL("--no-fp16",              fp16,    false)

    // Detector tuning.  --detector-threshold is the preferred, self-describing
    // spelling; --thresh is kept as a back-compat alias.  Both record that the
    // threshold was set so resolve_detector_defaults() won't override it with a
    // per-detector default.
    if ((std::strcmp(argv[i], "--detector-threshold") == 0 ||
         std::strcmp(argv[i], "--thresh") == 0) && i + 1 < argc)
    { c.person_thresh = std::stof(argv[++i]); c.thresh_set = true; return true; }
    CLI_FLT ("--nms",                  person_nms_iou)
    CLI_INT ("--max-persons",          max_persons)
    CLI_STR ("--detector",             detector)

    // BVH export
    CLI_STR ("--bvh",                  bvh_path)
    CLI_STR ("--bvh-template",         bvh_template)
    CLI_BOOL("--no-bvh-body-shape-change", bvh_body_shape_change,          false)
    CLI_BOOL("--no-bvh-hand-shape-change", bvh_hand_shape_change,          false)
    CLI_BOOL("--bvh-raw-fingers",          bvh_compensate_finger_endsites, false)
    // Hand limits + sticky hand pose are ON by default; keep the positive flags
    // (now explicit no-ops) for back-compat and add --no-… to disable.
    CLI_BOOL("--enforce-hand-limits",      bvh_enforce_hand_limits,        true)
    CLI_BOOL("--no-enforce-hand-limits",   bvh_enforce_hand_limits,        false)
    CLI_BOOL("--zero-hand-pose",           bvh_zero_hand_pose,             true)
    CLI_BOOL("--sticky-hand-pose",         bvh_sticky_hand_pose,           true)
    CLI_BOOL("--no-sticky-hand-pose",      bvh_sticky_hand_pose,           false)
    CLI_BOOL("--no-bvh-rest-align",        bvh_rest_align,                 false)
    CLI_BOOL("--dump-rest-dirs",           bvh_dump_rest_dirs,             true)
    CLI_BOOL("--foot-contact",             bvh_foot_contact,               true)
    CLI_BOOL("--bvh-static-root",          bvh_static_root,                true)
    CLI_STR ("--bvh-stream",               bvh_stream_path)
    CLI_STR ("--bvh-shm",                  bvh_shm_descriptor)
    CLI_STR ("--bvh-shm-stream",           bvh_shm_stream)

    // Filters
    CLI_FLT ("--bw-cutoff",            bw_cutoff)
    CLI_FLT ("--rot-clamp",            rot_clamp_deg)

#undef CLI_STR
#undef CLI_INT
#undef CLI_FLT
#undef CLI_BOOL
    return false;
}


// ─── Helpers ────────────────────────────────────────────────────────────────

// Map a --detector NAME string to a fsb::PipelineConfig::DetectorKind value.
// Unknown names warn and fall back to the default (yolo-pose). Extend the table
// (one row per kind) to add new providers, e.g. {"ymapnet", DET_YMAPNET}.
inline int detector_kind_from_string(const std::string& name)
{
    struct DetMap { const char* name; int kind; };
    static const DetMap kDetectors[] = {
        { "yolo-pose", fsb::PipelineConfig::DET_YOLO_POSE },
        { "libreyolo", fsb::PipelineConfig::DET_LIBREYOLO },
    };
    for (const auto& d : kDetectors)
        if (name == d.name) return d.kind;
    std::fprintf(stderr, "[cli] unknown --detector '%s'; using 'yolo-pose'\n",
                 name.c_str());
    return fsb::PipelineConfig::DET_YOLO_POSE;
}

// True if the model filename looks like a LibreYOLO / YOLOv9 detection export
// (vs an Ultralytics YOLO11-pose model), based on its basename.
inline bool path_looks_like_libreyolo(const std::string& p)
{
    std::string base = p;
    size_t slash = base.find_last_of("/\\");
    if (slash != std::string::npos) base = base.substr(slash + 1);
    for (auto& ch : base) ch = (char)std::tolower((unsigned char)ch);
    return base.find("libreyolo") != std::string::npos ||
           base.find("yolov9")    != std::string::npos ||
           base.find("yolo9")     != std::string::npos;
}

// ---------------------------------------------------------------------------
// Lazily fetch the model files when they are missing at startup.
//
// scripts/setup.sh fetches them at install time; this is the runtime recovery
// path for a checkout where that never ran (or where onnx/ was cleaned).  It
// shells out to tools/fetch_model.sh, which pulls the individual files from
// HuggingFace and verifies size + sha256 — the same delegation ensure_trt_models()
// below uses for setup_trt.sh, so curl/TLS stays out of this binary.
//
// Call FIRST, before resolve_detector_defaults() / resolve_backbone_defaults():
// both of those decide what to load by probing onnx_dir, so they need the files
// to be on disk before they run.
//
// refined_pose=true also fetches the 'refined' profile (the 28 iterative
// decoder_* ONNX graphs + pipeline_refined.gguf that --refined-pose loads).
//
// Two details that are easy to get wrong:
//
//   * The profile comes from the user's INTENT (--cuda), never from what is on
//     disk.  resolve_backbone_defaults() picks fp32-vs-bf16 by probing, but with
//     an empty onnx_dir that probe finds nothing and falls through to the bf16
//     default — which would fetch 5.1 GB of CUDA models for a --cuda -1 run that
//     then cannot load them (the CPU EP has no bf16 kernels).
//
//   * When the default './onnx' is missing we fetch into the onnx/ next to the
//     executable, not into the current directory.  Otherwise the classic "ran it
//     from build/" mistake (see the warning box in main()) stops being a warning
//     and turns into a multi-GB download into build/onnx.
//
// Best-effort throughout: on any failure we warn and return, leaving the existing
// "model missing" diagnostics to fire exactly as before.  SAM3D_AUTO_FETCH=0
// disables it outright; fetch_model.sh itself refuses to download when it has no
// terminal to prompt on, which keeps ROS nodes and CI jobs from silently pulling
// gigabytes.
inline void ensure_models(CommonConfig& c, bool refined_pose = false)
{
    const bool cpu = (c.cuda_device < 0);
    const char* profile;
    if (cpu)                      profile = "cpu";     // CPU EP: fp32 backbone + fp16 decoder
    else if (c.use_trt)           profile = "shared";  // the TRT pair is ensure_trt_models()' job
    else if (c.backbone_name_set) profile = "shared";  // user is driving the model choice
    else                          profile = "cuda";    // bf16 backbone + bf16 decoder

    // Deliberately NOT the 'trt' profile under --trt: setup_trt.sh also provisions
    // the TensorRT runtime venv, and fetching the TRT models here would make
    // ensure_trt_models() take its "already have them" early return and skip that.
    // Pulling the 5.1 GB 'cuda' fallback set would be worse still, since a TRT run
    // never loads it once the fp16 pair is on disk.

    // Mirrors the manifest in tools/fetch_model.sh.  Presence only — the script
    // does the size/sha verification, and re-running it is cheap and idempotent.
    std::vector<std::string> sentinels = {
        "pipeline.gguf", "body_model.lbs", "yolo.onnx",
        "correctives.bin", "keypoint_mapping.bin"
    };
    if (cpu) {
        // Fetched even when --backbone is pinned: pinning the backbone says
        // nothing about the decoder, and the bf16 decoder.onnx has no CPU kernels.
        sentinels.insert(sentinels.end(), {
            "backbone_fp32.onnx", "backbone_fp32.onnx.data",
            "decoder_fp16.onnx",  "decoder_fp16.onnx.data" });
    } else if (!c.use_trt && !c.backbone_name_set) {
        sentinels.insert(sentinels.end(), {
            "backbone.onnx", "backbone.onnx.data", "decoder.onnx" });
    }
    if (refined_pose) {
        // --refined-pose's iterative decoders (the 'refined' profile in
        // tools/fetch_model.sh) — 28 per-layer ONNX graphs plus the separate
        // hand-head GGUF. Without them the pipeline load fails outright, so
        // fetch them upfront like every other required model.
        sentinels.insert(sentinels.end(), {
            "pipeline_refined.gguf",
            "decoder_hand_pre.onnx", "decoder_hand_layer0.onnx",
            "decoder_hand_layer1.onnx", "decoder_hand_layer2.onnx",
            "decoder_hand_layer3.onnx", "decoder_hand_layer4.onnx",
            "decoder_hand_layer5.onnx", "decoder_hand_normfinal.onnx",
            "decoder_hand_update.onnx",
            "decoder_prompted_pre.onnx", "decoder_prompted_layer0.onnx",
            "decoder_prompted_layer1.onnx", "decoder_prompted_layer2.onnx",
            "decoder_prompted_layer3.onnx", "decoder_prompted_layer4.onnx",
            "decoder_prompted_layer5.onnx", "decoder_prompted_normfinal.onnx",
            "decoder_prompted_update.onnx",
            "decoder_pass1_pre.onnx", "decoder_pass1_layer0.onnx",
            "decoder_pass1_layer1.onnx", "decoder_pass1_layer2.onnx",
            "decoder_pass1_layer3.onnx", "decoder_pass1_layer4.onnx",
            "decoder_pass1_layer5.onnx", "decoder_pass1_normfinal.onnx",
            "decoder_pass1_update.onnx", "decoder_pass1_handbox.onnx" });
    }

    // Where the executable lives, so we can find both the repo's onnx/ and the
    // fetch script without depending on the working directory.
    std::string exe_dir;
    {
        char buf[4096];
        ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if (n > 0) {
            buf[n] = '\0';
            std::string exe(buf);
            size_t s = exe.find_last_of('/');
            if (s != std::string::npos) exe_dir = exe.substr(0, s);
        }
    }

    // Retarget the default './onnx' at the repo's onnx/ when the cwd has none.
    // Only ever fires in the case that fails outright today.
    if (c.onnx_dir == "./onnx" && !std::filesystem::exists("./onnx") && !exe_dir.empty()) {
        std::string repo_onnx = exe_dir + "/../onnx";
        std::error_code ec;
        std::filesystem::path canon = std::filesystem::weakly_canonical(repo_onnx, ec);
        if (!ec) {
            c.onnx_dir = canon.string();
            // gguf_path / yolo_path default to "./onnx/..." independently of
            // onnx_dir, so they have to move with it — otherwise the backbone
            // loads from the repo while YOLO still looks in the (absent) ./onnx.
            if (c.gguf_path == "./onnx/pipeline.gguf")
                c.gguf_path = c.onnx_dir + "/pipeline.gguf";
            if (!c.yolo_path_set && c.yolo_path == "./onnx/yolo.onnx")
                c.yolo_path = c.onnx_dir + "/yolo.onnx";
            std::fprintf(stderr,
                "[cli] './onnx' not in the working directory — using '%s' instead.\n",
                c.onnx_dir.c_str());
        }
    }

    // Locating models the user already has is not downloading, so this opt-out
    // is honoured only from here on.
    if (const char* e = std::getenv("SAM3D_AUTO_FETCH"))
        if (e[0] == '0' && e[1] == '\0') return;

    auto missing = [&] {
        for (const auto& f : sentinels)
            if (!std::ifstream(c.onnx_dir + "/" + f).good()) return true;
        return false;
    };
    if (!missing()) return;

    std::string script;
    if (!exe_dir.empty() && std::ifstream(exe_dir + "/../tools/fetch_model.sh").good())
        script = exe_dir + "/../tools/fetch_model.sh";
    else if (std::ifstream("tools/fetch_model.sh").good())
        script = "tools/fetch_model.sh";

    if (script.empty()) {
        std::fprintf(stderr,
            "[cli] models missing from '%s' and tools/fetch_model.sh not found; "
            "fetch them with:  bash scripts/setup.sh\n", c.onnx_dir.c_str());
        return;
    }

    const std::string profiles =
        std::string(profile) + (refined_pose ? " refined" : "");

    std::fprintf(stderr,
        "[cli] models missing from '%s' — running '%s %s' to fetch them…\n",
        c.onnx_dir.c_str(), script.c_str(), profiles.c_str());

    const std::string cmd = "bash '" + script + "' " + profiles +
                            " --onnx-dir '" + c.onnx_dir + "'";
    if (std::system(cmd.c_str()) != 0 || missing()) {
        std::fprintf(stderr,
            "[cli] model fetch did not complete — continuing; the load below will "
            "report what is still missing.\n");
    }
}

// Resolve the "auto" detector default and the per-detector confidence default.
// Call this once, after the argv loop and before apply_common_to_pipeline_cfg()
// (or before a binary reads c.detector / c.yolo_path directly).  It is the
// single place that implements "prefer LibreYOLO when available":
//
//   * detector == "auto" and the user did NOT pass --yolo  → scan onnx_dir for
//     a libreyolo*.onnx and, if found, adopt it (LibreYOLO preferred over the
//     Ultralytics yolo.onnx default).
//   * detector == "auto"                                   → pick libreyolo vs
//     yolo-pose from the (possibly user-pinned) model filename.
//   * threshold not set on the CLI                         → default 0.25 for
//     libreyolo (the tiny model scores people lower) else 0.50.
//
// An explicit --detector / --yolo / --detector-threshold always wins.
inline void resolve_detector_defaults(CommonConfig& c)
{
    if (c.detector == "auto") {
        // Prefer a LibreYOLO model on disk when the user hasn't pinned --yolo.
        if (!c.yolo_path_set) {
            std::string pattern = c.onnx_dir + "/libreyolo*.onnx";
            glob_t g{};
            if (glob(pattern.c_str(), 0, nullptr, &g) == 0 && g.gl_pathc > 0) {
                c.yolo_path = g.gl_pathv[0];
                std::fprintf(stderr,
                    "[cli] --detector auto: found LibreYOLO model '%s'; "
                    "preferring it over yolo-pose\n", c.yolo_path.c_str());
            }
            globfree(&g);
        }
        c.detector = path_looks_like_libreyolo(c.yolo_path) ? "libreyolo"
                                                            : "yolo-pose";
    }

    if (!c.thresh_set) {
        c.person_thresh =
            (detector_kind_from_string(c.detector) ==
             fsb::PipelineConfig::DET_LIBREYOLO) ? 0.25f : 0.50f;
    }
}

// Auto-fetch the TRT-ready models when --trt is requested but they're missing.
// resolve_backbone_defaults() swaps in backbone_fp16_trt.onnx / decoder_fp16.onnx
// under --trt only when they exist on disk; without them --trt silently falls back
// to the CUDA EP.  Rather than duplicate the download here, we delegate to
// tools/setup_trt.sh (one source of truth) with --skip-venv, so it just fetches
// the models — and it prompts before pulling the ~1.7 GB archive.
//
// Best-effort: on any failure (script not found, user declines the prompt, no
// network) we warn and return, and resolve_backbone_defaults() falls back to the
// CUDA EP exactly as before.  Skipped when the user pinned --backbone (they're
// driving the model choice) or on CPU (--cuda -1, where TRT/fp16 don't apply).
inline void ensure_trt_models(const CommonConfig& c)
{
    if (!c.use_trt)          return;
    if (c.cuda_device < 0)   return;   // CPU EP: TRT/fp16 N/A
    if (c.backbone_name_set) return;   // user pinned a model — don't second-guess

    auto exists = [&](const char* name) {
        return std::ifstream(c.onnx_dir + "/" + name).good();
    };
    if (exists("backbone_fp16_trt.onnx") && exists("decoder_fp16.onnx"))
        return;   // already have them

    // Locate setup_trt.sh relative to this executable (binaries live in build/,
    // so ../tools/), with the working dir as a fallback.
    std::string script;
    {
        char buf[4096];
        ssize_t n = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if (n > 0) {
            buf[n] = '\0';
            std::string exe(buf);
            size_t s = exe.find_last_of('/');
            if (s != std::string::npos) {
                std::string cand = exe.substr(0, s) + "/../tools/setup_trt.sh";
                if (std::ifstream(cand).good()) script = cand;
            }
        }
        if (script.empty() && std::ifstream("tools/setup_trt.sh").good())
            script = "tools/setup_trt.sh";
    }
    if (script.empty()) {
        std::fprintf(stderr,
            "[cli] TRT: backbone_fp16_trt.onnx / decoder_fp16.onnx missing and "
            "tools/setup_trt.sh not found; falling back to CUDA EP. "
            "Fetch them per DEPENDENCIES.md §6.\n");
        return;
    }

    std::fprintf(stderr,
        "[cli] TRT: backbone_fp16_trt.onnx / decoder_fp16.onnx missing in '%s'; "
        "running '%s' to fetch them…\n", c.onnx_dir.c_str(), script.c_str());

    const std::string cmd =
        "bash '" + script + "' --skip-venv --onnx-dir '" + c.onnx_dir + "'";
    if (std::system(cmd.c_str()) != 0) {
        std::fprintf(stderr,
            "[cli] TRT: setup_trt.sh did not fetch the models — falling back to CUDA EP.\n");
        return;
    }

    if (exists("backbone_fp16_trt.onnx") && exists("decoder_fp16.onnx"))
        std::fprintf(stderr, "[cli] TRT: prebuilt models ready in '%s'.\n",
                     c.onnx_dir.c_str());
    else
        std::fprintf(stderr,
            "[cli] TRT: models not fetched (declined or unavailable) — "
            "falling back to CUDA EP.\n");
}

// On CUDA, auto-prefer a float16 backbone when one has been exported next to the
// default backbone (tools/export_backbone_fp16.py → <onnx_dir>/backbone_fp16.onnx).
// The stock backbone is bfloat16; the fp16 remap is 3× smaller on disk/VRAM and a
// few % faster on ORT's CUDA EP, with identical output. (Not 2×: the export bakes
// in fp32 MatMul paths — see tools/export_backbone_fp16.py for the full story.)
//
// Only fires when ALL of the following hold, so it never surprises the user:
//   * the user did NOT pin a model with --backbone, and
//   * backbone_name is still the plain FP32 default ("backbone.onnx") — i.e. a
//     baked-in quant default (SAM3D_BACKBONE_QUANT → backbone_int8.onnx) wins, and
//   * we are running on a CUDA device (FP16 on the CPU EP is not a win), and
//   * <onnx_dir>/backbone_fp16.onnx actually exists.
//
// On CPU (--cuda -1) the swaps are a different story — see the block at the top of
// the body: there the bf16 stock models don't just run slower, they don't load at
// all, so the fp32 backbone / fp16 decoder are picked up as a correctness fix.
inline void resolve_backbone_defaults(CommonConfig& c)
{
    auto exists = [&](const std::string& name) {
        return std::ifstream(c.onnx_dir + "/" + name).good();
    };

    // ── CPU EP: neither stock model may be bfloat16 ────────────────────────────
    // backbone.onnx (355 bf16 tensors) and decoder.onnx (130) both ship as bf16
    // exports.  ORT's CUDA EP has bf16 kernels; the CPU EP has none, so it refuses
    // to even load the graph:
    //   "Could not find an implementation for MatMul(13) node with name
    //    '/init_to_token/MatMul'"
    // The fp32 backbone / fp16 decoder are the CPU-runnable remaps (the fp16
    // decoder matches fp32 to ~0.005% and is no slower on the CPU EP, so there is
    // no fp32 decoder), and scripts/setup.sh --cpu-backbone fetches both.  Prefer
    // them here so --cuda -1 works without the user having to name them, and say
    // exactly how to get them when they're absent instead of leaving the user with
    // ORT's cryptic MatMul error.
    if (c.cuda_device < 0) {
        if (c.decoder_name == "decoder.onnx") {
            if (exists("decoder_fp16.onnx")) {
                c.decoder_name = "decoder_fp16.onnx";
                std::fprintf(stderr,
                    "[cli] CPU: using 'decoder_fp16.onnx' (bf16 decoder.onnx has no CPU kernels).\n");
            } else {
                std::fprintf(stderr,
                    "[cli] CPU: '%s/decoder_fp16.onnx' is missing and the bf16 "
                    "decoder.onnx cannot load on the CPU EP.\n"
                    "      Fetch the CPU models with:  bash scripts/setup.sh --cpu-backbone\n",
                    c.onnx_dir.c_str());
            }
        }
        // Same story for the backbone, unless the user pinned one or a build-time
        // quant default (backbone_int8.onnx, which runs on CPU) is in play.
        if (!c.backbone_name_set && c.backbone_name == "backbone.onnx") {
            if (exists("backbone_fp32.onnx")) {
                c.backbone_name = "backbone_fp32.onnx";
                std::fprintf(stderr,
                    "[cli] CPU: using 'backbone_fp32.onnx' (bf16 backbone.onnx has no CPU kernels).\n");
            } else {
                std::fprintf(stderr,
                    "[cli] CPU: '%s/backbone_fp32.onnx' is missing and the bf16 "
                    "backbone.onnx cannot load on the CPU EP.\n"
                    "      Fetch the CPU models with:  bash scripts/setup.sh --cpu-backbone\n",
                    c.onnx_dir.c_str());
            }
        }
        return;   // the TRT / fp16-backbone swaps below are CUDA-only
    }

    // Under --trt, fetch the TRT-ready models on the fly if they're not on disk
    // yet, so the swaps below can engage instead of falling back to the CUDA EP.
    ensure_trt_models(c);

    // ── Decoder ────────────────────────────────────────────────────────────────
    // The stock decoder.onnx is bfloat16.  ORT's CUDA EP runs it fine, but the
    // TensorRT EP rejects bf16 subgraph-boundary tensors ("output tensor data
    // type: 16 not supported").  decoder_fp16.onnx is the bf16→fp16 remap that
    // TRT accepts (regenerate with:
    //   tools/export_backbone_fp16.py --input onnx/decoder.onnx
    //                                 --output onnx/decoder_fp16.onnx).
    // Pick it up automatically under --trt so the decoder runs on TRT instead of
    // crashing.
    if (c.use_trt && c.decoder_name == "decoder.onnx" && exists("decoder_fp16.onnx")) {
        c.decoder_name = "decoder_fp16.onnx";
        std::fprintf(stderr,
            "[cli] TRT: using 'decoder_fp16.onnx' (bf16 decoder.onnx is not TRT-compatible).\n");
    }

    // ── Backbone ───────────────────────────────────────────────────────────────
    if (c.backbone_name_set)            return;   // user pinned --backbone
    if (c.backbone_name != "backbone.onnx") return;   // non-default (e.g. int8) — respect it

    // Under --trt prefer the If-folded fp16 backbone: the stock fp16 backbone
    // still carries the rope_embed `If` subgraphs, which TRT cannot shape-infer
    // ("/rope_embed/If_1_output_0 has no shape specified"), forcing a silent
    // fall-back to the CUDA EP.  backbone_fp16_trt.onnx folds them out so the TRT
    // engine builds and the heavy GEMMs run on the fp16 tensor cores.
    if (c.use_trt && exists("backbone_fp16_trt.onnx")) {
        c.backbone_name = "backbone_fp16_trt.onnx";
        std::fprintf(stderr,
            "[cli] TRT: preferring 'backbone_fp16_trt.onnx' (If-folded, TRT-buildable).\n");
        return;
    }

    if (exists("backbone_fp16.onnx")) {
        c.backbone_name = "backbone_fp16.onnx";
        std::fprintf(stderr,
            "[cli] CUDA: found 'backbone_fp16.onnx'; preferring it over backbone.onnx "
            "(FP16: 3x smaller, ~6%% faster). Pin --backbone backbone.onnx to force the bf16 original.\n");
    }
}

// Populate the fields of a fsb::PipelineConfig that come straight from
// CommonConfig.  Binary-specific fields (`skip_body_model`, `principal_x`,
// `focal_x`, etc.) stay the caller's responsibility.
inline void apply_common_to_pipeline_cfg(const CommonConfig& c,
                                          fsb::PipelineConfig& pc)
{
    pc.onnx_dir       = c.onnx_dir;
    pc.backbone_name  = c.backbone_name;
    pc.decoder_name   = c.decoder_name;
    pc.gguf_path      = c.gguf_path;
    pc.yolo_path      = c.yolo_path;
    pc.cuda_device    = c.cuda_device;
    pc.use_trt_ep     = c.use_trt;
    pc.use_fp16       = c.fp16;
    pc.person_thresh  = c.person_thresh;
    pc.person_nms_iou = c.person_nms_iou;
    pc.max_persons    = c.max_persons;
    pc.detector       = detector_kind_from_string(c.detector);
}

// Centralised auto-derivation of the LBS path from --onnx-dir.  All three
// binaries do this the same way.
inline std::string default_lbs_path(const CommonConfig& c)
{
    return c.onnx_dir + "/body_model.lbs";
}

// Emit the common subset of --help.  Per-binary help texts call this so the
// shared rows stay consistent across binaries.
inline void print_common_args_help(FILE* fp)
{
    std::fprintf(fp,
        "Common (parsed by cli_common.h):\n"
        "  --onnx-dir PATH                Directory with backbone/decoder ONNX files\n"
        "  --backbone NAME                Backbone filename within onnx-dir (default backbone.onnx; on CUDA,\n"
        "                                 backbone_fp16.onnx is auto-preferred when present — see\n"
        "                                 tools/export_backbone_fp16.py; or backbone_int8.onnx via\n"
        "                                 tools/quantize_backbone.py)\n"
        "  --gguf     PATH                pipeline.gguf (MHR + camera heads)\n"
        "  --yolo     PATH                Detector model (.onnx); YOLO11-pose or a LibreYOLO/YOLOv9 export\n"
        "  --detector NAME                Bbox provider parsing --yolo output: auto (default; prefers a\n"
        "                                 libreyolo*.onnx in onnx-dir, else yolo-pose) | yolo-pose (56-ch\n"
        "                                 YOLO11-pose) | libreyolo (84-ch YOLOv9 detection, bbox-only)\n"
        "  --from     PATH                Input source (file path, or webcam index where supported)\n"
        "  --frames   N                   Stop after N frames (useful for quick tests; default unlimited)\n"
        "  --start    N                   Skip to frame N before processing (seek into the video; default 0)\n"
        "  --cuda     N                   CUDA device (-1 = CPU; default 0)\n"
        "  --trt                          Use ONNX Runtime TensorRT EP\n"
        "  --no-fp16                      Disable FP16\n"
        "  --detector-threshold F         Person confidence threshold (default 0.50; 0.25 for libreyolo,\n"
        "                                 whose tiny model scores people lower).  Alias: --thresh\n"
        "  --nms      F                   Detector NMS IoU (default 0.45)\n"
        "  --max-persons N                Cap processing to the top-N most-confident people (0 = unlimited)\n"
        "  --bvh      PATH                Write BVH motion-capture file(s); per-person filenames appended\n"
        "  --bvh-template PATH            BVH skeleton template (default ./body_mhr.bvh,\n"
        "                                 MHR-rest aligned; ./mocapnet.bvh for MakeHuman,\n"
        "                                 ./mixamo.bvh for a Mixamo 'mixamorig:' rig,\n"
        "                                 ./lafan.bvh for LAFAN1 names (feeds GMR robot retargeting))\n"
        "  --no-bvh-body-shape-change     Keep template body bone lengths\n"
        "  --no-bvh-hand-shape-change     Keep template hand/finger bone lengths\n"
        "  --bvh-raw-fingers              Do not rescale finger End-Site OFFSETs\n"
        "  --no-enforce-hand-limits       Disable the default clamp of finger joint angles to anatomical\n"
        "                                 limits (the clamp fixes wild splay when hands are not visible)\n"
        "  --zero-hand-pose               Always write neutral (straight) hand pose\n"
        "  --no-sticky-hand-pose          Disable the default 'inherit previous frame's hand pose when out\n"
        "                                 of limits' behaviour (neutral on first frame)\n"
        "  --no-bvh-rest-align            Disable rest-frame retarget (re-aiming joint rotations onto the\n"
        "                                 template's bones; on by default — fixes arms under-bending when\n"
        "                                 the template rest pose differs from MHR, e.g. T-pose vs A-pose)\n"
        "  --dump-rest-dirs               Print the per-bone template-vs-MHR rest-direction table at open\n"
        "  --foot-contact                 Clean up foot-skate: level the root to a fitted floor and run\n"
        "                                 2-bone leg IK to pin planted feet (offline; off by default)\n"
        "  --bvh-static-root              Zero the root position and rotation every frame, pinning the\n"
        "                                 body in place (in-place motion; off by default)\n"
        "  --bw-cutoff HZ                 Butterworth cutoff (default 6 Hz)\n"
        "  --rot-clamp DEG                Geodesic SLERP clamp on global_rot (default 1 deg/frame;\n"
        "                                 offline binary defaults to 30)\n");
}
