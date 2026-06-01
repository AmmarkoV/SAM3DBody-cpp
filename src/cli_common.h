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

#include <cstdio>
#include <cstring>
#include <string>

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
    int         cuda_device    = 0;       // -1 = CPU
    bool        use_trt        = false;
    bool        fp16           = true;    // can be disabled with --no-fp16

    // ── YOLO person detector tuning ──────────────────────────────────────────
    // The renderer doesn't use these (it inherits whatever the pipeline
    // chose internally) but accepting them keeps the CLI uniform — passing
    // `--thresh 0.6` to the renderer simply no-ops rather than erroring.
    float       person_thresh   = 0.50f;
    float       person_nms_iou  = 0.45f;

    // ── Input source ────────────────────────────────────────────────────────
    std::string from;             // file / webcam index / empty = required
    int         max_frames = -1; // --frames N: stop after N frames (-1 = unlimited)
    int         start_frame = 0; // --start N: skip to frame N before processing

    // ── BVH export ──────────────────────────────────────────────────────────
    std::string bvh_path;
    std::string bvh_template    = "./body.bvh";
    bool        bvh_body_shape_change          = true;   // --no-bvh-body-shape-change
    bool        bvh_hand_shape_change          = true;   // --no-bvh-hand-shape-change
    bool        bvh_compensate_finger_endsites = true;   // --bvh-raw-fingers
    bool        bvh_enforce_hand_limits        = false;  // --enforce-hand-limits
    bool        bvh_zero_hand_pose             = false;  // --zero-hand-pose
    bool        bvh_sticky_hand_pose           = false;  // --sticky-hand-pose
    bool        bvh_rest_align                 = true;    // --no-bvh-rest-align
    bool        bvh_dump_rest_dirs             = false;   // --dump-rest-dirs
    bool        bvh_foot_contact               = false;   // --foot-contact

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
    CLI_STR ("--yolo",                 yolo_path)
    CLI_STR ("--backbone",             backbone_name)
    CLI_STR ("--from",                 from)
    CLI_INT ("--frames",               max_frames)
    CLI_INT ("--start",                start_frame)
    CLI_INT ("--cuda",                 cuda_device)
    CLI_BOOL("--trt",                  use_trt, true)
    CLI_BOOL("--no-fp16",              fp16,    false)

    // YOLO tuning
    CLI_FLT ("--thresh",               person_thresh)
    CLI_FLT ("--nms",                  person_nms_iou)

    // BVH export
    CLI_STR ("--bvh",                  bvh_path)
    CLI_STR ("--bvh-template",         bvh_template)
    CLI_BOOL("--no-bvh-body-shape-change", bvh_body_shape_change,          false)
    CLI_BOOL("--no-bvh-hand-shape-change", bvh_hand_shape_change,          false)
    CLI_BOOL("--bvh-raw-fingers",          bvh_compensate_finger_endsites, false)
    CLI_BOOL("--enforce-hand-limits",      bvh_enforce_hand_limits,        true)
    CLI_BOOL("--zero-hand-pose",           bvh_zero_hand_pose,             true)
    CLI_BOOL("--sticky-hand-pose",         bvh_sticky_hand_pose,           true)
    CLI_BOOL("--no-bvh-rest-align",        bvh_rest_align,                 false)
    CLI_BOOL("--dump-rest-dirs",           bvh_dump_rest_dirs,             true)
    CLI_BOOL("--foot-contact",             bvh_foot_contact,               true)

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

// Populate the fields of a fsb::PipelineConfig that come straight from
// CommonConfig.  Binary-specific fields (`skip_body_model`, `principal_x`,
// `focal_x`, etc.) stay the caller's responsibility.
inline void apply_common_to_pipeline_cfg(const CommonConfig& c,
                                          fsb::PipelineConfig& pc)
{
    pc.onnx_dir       = c.onnx_dir;
    pc.backbone_name  = c.backbone_name;
    pc.gguf_path      = c.gguf_path;
    pc.yolo_path      = c.yolo_path;
    pc.cuda_device    = c.cuda_device;
    pc.use_trt_ep     = c.use_trt;
    pc.use_fp16       = c.fp16;
    pc.person_thresh  = c.person_thresh;
    pc.person_nms_iou = c.person_nms_iou;
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
        "  --backbone NAME                Backbone filename within onnx-dir (default backbone.onnx;\n"
        "                                 use backbone_int8.onnx after running tools/quantize_backbone.py)\n"
        "  --gguf     PATH                pipeline.gguf (MHR + camera heads)\n"
        "  --yolo     PATH                YOLO pose model (.onnx)\n"
        "  --from     PATH                Input source (file path, or webcam index where supported)\n"
        "  --frames   N                   Stop after N frames (useful for quick tests; default unlimited)\n"
        "  --start    N                   Skip to frame N before processing (seek into the video; default 0)\n"
        "  --cuda     N                   CUDA device (-1 = CPU; default 0)\n"
        "  --trt                          Use ONNX Runtime TensorRT EP\n"
        "  --no-fp16                      Disable FP16\n"
        "  --thresh   F                   YOLO person confidence (default 0.50)\n"
        "  --nms      F                   YOLO NMS IoU (default 0.45)\n"
        "  --bvh      PATH                Write BVH motion-capture file(s); per-person filenames appended\n"
        "  --bvh-template PATH            BVH skeleton template (default ./body.bvh)\n"
        "  --no-bvh-body-shape-change     Keep template body bone lengths\n"
        "  --no-bvh-hand-shape-change     Keep template hand/finger bone lengths\n"
        "  --bvh-raw-fingers              Do not rescale finger End-Site OFFSETs\n"
        "  --enforce-hand-limits          Clamp finger joint angles to anatomical limits\n"
        "                                 (fixes wild splay when hands are not visible)\n"
        "  --zero-hand-pose               Always write neutral (straight) hand pose\n"
        "  --sticky-hand-pose             Inherit previous frame's hand pose (neutral on first frame)\n"
        "  --no-bvh-rest-align            Disable rest-frame retarget (re-aiming joint rotations onto the\n"
        "                                 template's bones; on by default — fixes arms under-bending when\n"
        "                                 the template rest pose differs from MHR, e.g. T-pose vs A-pose)\n"
        "  --dump-rest-dirs               Print the per-bone template-vs-MHR rest-direction table at open\n"
        "  --foot-contact                 Clean up foot-skate: level the root to a fitted floor and run\n"
        "                                 2-bone leg IK to pin planted feet (offline; off by default)\n"
        "  --bw-cutoff HZ                 Butterworth cutoff (default 6 Hz)\n"
        "  --rot-clamp DEG                Geodesic SLERP clamp on global_rot (default 1 deg/frame;\n"
        "                                 offline binary defaults to 30)\n");
}
