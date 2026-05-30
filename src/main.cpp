#include "fast_sam_3dbody.h"
#include "bvh_writer.h"
#include "outputFiltering.h"
#include "cli_common.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <time.h>

// Monotonic nanosecond timestamp (POSIX; avoids chrono overhead in tight loops)
static uint64_t get_mono_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// CLI
//
// Config inherits the common subset from CommonConfig (cli_common.h):
//   --onnx-dir --gguf --yolo --from --cuda --trt --no-fp16 --thresh --nms
//   --bvh --bvh-template --no-bvh-*-shape-change --bvh-raw-fingers
//   --bw-cutoff --rot-clamp
// Runner-specific flags below.
// ---------------------------------------------------------------------------
struct Config : public CommonConfig
{
    Config() {
        from = "0";  // runner-only default: webcam 0 when --from is omitted
    }

    // Runner-specific output
    std::string csv_path;             // -o / --out

    // Runner-specific pipeline knobs
    bool        skip_body      = false;
    bool        zero_face      = true;
    float       focal_x        = 0.f;
    float       focal_y        = 0.f;
    float       cx             = 0.f;
    float       cy             = 0.f;
    bool        headless       = false;
    bool        info_only      = false;

    /* On a good bw_cutoff value..
    ┌────────────────────────┬──────────────────────┬────────────────────────────────────┐
    │      --bw-cutoff       │      Approx lag      │               Effect               │
    ├────────────────────────┼──────────────────────┼────────────────────────────────────┤
    │ 1.5 Hz                 │ ~212 ms / 6.4 frames │ Heavy smoothing, very laggy        │
    ├────────────────────────┼──────────────────────┼────────────────────────────────────┤
    │ 3 Hz                   │ ~106 ms / 3.2 frames │ Moderate smoothing                 │
    ├────────────────────────┼──────────────────────┼────────────────────────────────────┤
    │ 6 Hz                   │ ~53 ms / 1.6 frames  │ Light smoothing, barely noticeable │
    ├────────────────────────┼──────────────────────┼────────────────────────────────────┤
    │ 10 Hz                  │ ~32 ms / ~1 frame    │ Minimal smoothing, sub-frame lag   │
    ├────────────────────────┼──────────────────────┼────────────────────────────────────┤
    │ 12 Hz                  │ ~27 ms / 0.8 frames  │ Almost pass-through                │
    └────────────────────────┴──────────────────────┴────────────────────────────────────┘
    */
    bool        butterworth      = false;
    bool        filter_root_rot  = false;

    // Capture / display geometry — runner only.
    int         render_w    = 0;     // GL window width  (0 = match input)
    int         render_h    = 0;     // GL window height (0 = match input)
    int         cap_w       = 0;     // capture width  (0 = driver default)
    int         cap_h       = 0;     // capture height (0 = driver default)
    double      cap_fps     = 0.0;   // capture fps    (0 = driver default)
};

static void print_usage(const char* prog)
{
    printf("Usage: %s [options]\n\n", prog);
    printf("  --onnx-dir PATH   Directory with backbone/decoder/body_model ONNX files\n");
    printf("  --backbone NAME   Backbone filename in onnx-dir (default backbone.onnx;\n");
    printf("                    use backbone_int8.onnx after tools/quantize_backbone.py)\n");
    printf("  --gguf PATH       pipeline.gguf (MHR + camera heads)\n");
    printf("  --yolo PATH       YOLO pose model (.onnx or .engine)\n");
    printf("  --from SRC        Webcam index (0,1,..) or path to image/video\n");
    printf("  --size W H        Webcam capture resolution (default: driver default)\n");
    printf("  --fps Z           Webcam capture framerate  (default: driver default)\n");
    printf("  --cuda DEVICE     CUDA device index (default 0; -1 = CPU)\n");
    printf("  --trt             Enable ONNX Runtime TensorRT EP\n");
    printf("  --no-fp16         Disable FP16 for ONNX EP\n");
    printf("  --skip-body       Skip body model (no vertices / keypoints)\n");
    printf("  --dev-face        Enable face expression params (disabled by default)\n");
    printf("  --thresh T        YOLO person confidence threshold (default 0.50)\n");
    printf("  --nms T           YOLO NMS IoU threshold (default 0.45)\n");
    printf("  --fx F            Camera focal length x (0 = image width)\n");
    printf("  --fy F            Camera focal length y (0 = image width)\n");
    printf("  --cx F            Principal point x (0 = width/2)\n");
    printf("  --cy F            Principal point y (0 = height/2)\n");
    printf("  --render-size W H GL window width and height in pixels (default: match input)\n");
    printf("  -o / --out PATH   Write 3D keypoints to CSV (frame,skeleton_id,joint_x,y,z...)\n");
    printf("  --bvh PATH        Write BVH motion capture output to PATH (one file per tracked person)\n");
    printf("  --bvh-template P  BVH skeleton template (default ./body.bvh)\n");
    printf("  --no-bvh-body-shape-change  Keep body.bvh's authored body bone lengths (no median rewrite)\n");
    printf("  --no-bvh-hand-shape-change  Keep body.bvh's authored hand/finger bone lengths\n");
    printf("  --bvh-raw-fingers           Do NOT rescale finger End-Site OFFSETs to MHR fingertip lengths\n");
    printf("  --headless        Do not open display windows\n");
    printf("  --info            Print pipeline info and exit\n");
    printf("  --butterworth              Apply Butterworth low-pass filter to MHR output vectors\n");
    printf("  --bw-cutoff HZ             Butterworth cutoff frequency in Hz (default 120)\n");
    printf("  --butterworth-root-rotation  Also apply wrap-rejection to global_rot (off by default)\n");
    printf("  --rot-clamp DEG            Rejection threshold for root rotation in deg/frame (default 1)\n");
    printf("  --help / -h       This message\n");
}

static Config parse_args(int argc, char** argv)
{
    Config c;
    for (int i = 1; i < argc; ++i)
    {
        // Common flags first — see src/cli_common.h.  Anything that's NOT
        // a common flag falls through to the runner-specific handlers below.
        if (parse_common_arg(argc, argv, i, c)) continue;

#define ARG1(flag, field, conv) \
        if (!strcmp(argv[i], flag) && i+1 < argc) { c.field = conv(argv[++i]); continue; }
        // Runner-only: per-axis camera intrinsics + CSV path.
        ARG1("--fx",       focal_x,     std::stof)
        ARG1("--fy",       focal_y,     std::stof)
        ARG1("--cx",       cx,          std::stof)
        ARG1("--cy",       cy,          std::stof)
        ARG1("--out",      csv_path,    std::string)
        ARG1("-o",         csv_path,    std::string)
#undef ARG1
        if (!strcmp(argv[i], "--render-size") && i+2 < argc)
        {
            c.render_w = std::stoi(argv[++i]);
            c.render_h = std::stoi(argv[++i]);
            continue;
        }
        if (!strcmp(argv[i], "--size") && i+2 < argc)
        {
            c.cap_w = std::stoi(argv[++i]);
            c.cap_h = std::stoi(argv[++i]);
            continue;
        }
        if (!strcmp(argv[i], "--fps") && i+1 < argc)
        {
            c.cap_fps = std::stod(argv[++i]);
            continue;
        }
        if (!strcmp(argv[i], "--skip-body"))      { c.skip_body       = true;  continue; }
        if (!strcmp(argv[i], "--dev-face"))       { c.zero_face       = false; continue; }
        if (!strcmp(argv[i], "--headless"))       { c.headless        = true;  continue; }
        if (!strcmp(argv[i], "--info"))           { c.info_only       = true;  continue; }
        if (!strcmp(argv[i], "--butterworth"))    { c.butterworth     = true;  continue; }
        if (!strcmp(argv[i], "--butterworth-root-rotation")) { c.filter_root_rot = true; continue; }
        if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        fprintf(stderr, "Unknown option: %s\n", argv[i]);
        print_usage(argv[0]);
        std::exit(1);
    }
    return c;
}

// ---------------------------------------------------------------------------
// Print one MHRResult to stdout
// ---------------------------------------------------------------------------
static void print_result(int person_idx, const fsb::MHRResult& r)
{
    printf("  person[%d]  bbox=[%.1f,%.1f,%.1f,%.1f]  focal=%.1f  cam_t=[%.3f,%.3f,%.3f]\n",
           person_idx,
           r.bbox[0], r.bbox[1], r.bbox[2], r.bbox[3],
           r.focal_length,
           r.pred_cam_t[0], r.pred_cam_t[1], r.pred_cam_t[2]);
    printf("             global_rot=[%.4f,%.4f,%.4f]\n",
           r.global_rot[0], r.global_rot[1], r.global_rot[2]);

    // body pose summary (first 9 values)
    printf("             body_pose[0..8]=[");
    for (int j = 0; j < 9 && j < (int)r.body_pose.size(); ++j)
        printf("%.4f%s", r.body_pose[j], j+1<9 && j+1<(int)r.body_pose.size() ? "," : "");
    printf("...]\n");

    // shape summary
    printf("             shape[0..4]=[");
    for (int j = 0; j < 5 && j < (int)r.shape.size(); ++j)
        printf("%.4f%s", r.shape[j], j+1<5 && j+1<(int)r.shape.size() ? "," : "");
    printf("...]\n");

    if (!r.keypoints_3d.empty())
    {
        printf("             kp3d[0]=[%.3f,%.3f,%.3f]\n",
               r.keypoints_3d[0], r.keypoints_3d[1], r.keypoints_3d[2]);
    }
}

// ---------------------------------------------------------------------------
// MHR-70 joint names (matches fast_sam_3dbody_dump_csv.py / MHR70_NAMES)
// ---------------------------------------------------------------------------
static const char* KP_NAMES[70] =
{
    "nose",                    //  0
    "left_eye",                //  1
    "right_eye",               //  2
    "left_ear",                //  3
    "right_ear",               //  4
    "left_shoulder",           //  5
    "right_shoulder",          //  6
    "left_elbow",              //  7
    "right_elbow",             //  8
    "left_hip",                //  9
    "right_hip",               // 10
    "left_knee",               // 11
    "right_knee",              // 12
    "left_ankle",              // 13
    "right_ankle",             // 14
    "left_big_toe_tip",        // 15
    "left_small_toe_tip",      // 16
    "left_heel",               // 17
    "right_big_toe_tip",       // 18
    "right_small_toe_tip",     // 19
    "right_heel",              // 20
    "right_thumb_tip",         // 21
    "right_thumb_first_joint", // 22
    "right_thumb_second_joint",// 23
    "right_thumb_third_joint", // 24
    "right_index_tip",         // 25
    "right_index_first_joint", // 26
    "right_index_second_joint",// 27
    "right_index_third_joint", // 28
    "right_middle_tip",        // 29
    "right_middle_first_joint",// 30
    "right_middle_second_joint",//31
    "right_middle_third_joint",// 32
    "right_ring_tip",          // 33
    "right_ring_first_joint",  // 34
    "right_ring_second_joint", // 35
    "right_ring_third_joint",  // 36
    "right_pinky_tip",         // 37
    "right_pinky_first_joint", // 38
    "right_pinky_second_joint",// 39
    "right_pinky_third_joint", // 40
    "right_wrist",             // 41
    "left_thumb_tip",          // 42
    "left_thumb_first_joint",  // 43
    "left_thumb_second_joint", // 44
    "left_thumb_third_joint",  // 45
    "left_index_tip",          // 46
    "left_index_first_joint",  // 47
    "left_index_second_joint", // 48
    "left_index_third_joint",  // 49
    "left_middle_tip",         // 50
    "left_middle_first_joint", // 51
    "left_middle_second_joint",// 52
    "left_middle_third_joint", // 53
    "left_ring_tip",           // 54
    "left_ring_first_joint",   // 55
    "left_ring_second_joint",  // 56
    "left_ring_third_joint",   // 57
    "left_pinky_tip",          // 58
    "left_pinky_first_joint",  // 59
    "left_pinky_second_joint", // 60
    "left_pinky_third_joint",  // 61
    "left_wrist",              // 62
    "left_olecranon",          // 63
    "right_olecranon",         // 64
    "left_cubital_fossa",      // 65
    "right_cubital_fossa",     // 66
    "left_acromion",           // 67
    "right_acromion",          // 68
    "neck",                    // 69
};

// ---------------------------------------------------------------------------
// Skeleton edges for 2-D overlay  (MHR-70 joint indices)
// ---------------------------------------------------------------------------
// Colour key (by group): body=green, right-hand=blue, left-hand=red
static const int BODY_EDGES[][2] =
{
    // Head
    {0,1},{0,2},{1,3},{2,4},
    // Shoulders
    {5,6},
    // Left arm: shoulder→elbow→wrist
    {5,7},{7,62},
    // Right arm: shoulder→elbow→wrist
    {6,8},{8,41},
    // Torso
    {5,9},{6,10},{9,10},
    // Left leg
    {9,11},{11,13},{13,15},{13,17},
    // Right leg
    {10,12},{12,14},{14,18},{14,20},
    // Neck
    {5,69},{6,69},
};
static const int N_BODY_EDGES = (int)(sizeof(BODY_EDGES)/sizeof(BODY_EDGES[0]));

// Right hand: finger chains root→tip  (wrist = 41)
static const int RHAND_EDGES[][2] =
{
    {41,24},{24,23},{23,22},{22,21},   // thumb
    {41,28},{28,27},{27,26},{26,25},   // index
    {41,32},{32,31},{31,30},{30,29},   // middle
    {41,36},{36,35},{35,34},{34,33},   // ring
    {41,40},{40,39},{39,38},{38,37},   // pinky
};
static const int N_RHAND_EDGES = (int)(sizeof(RHAND_EDGES)/sizeof(RHAND_EDGES[0]));

// Left hand: finger chains root→tip  (wrist = 62)
static const int LHAND_EDGES[][2] =
{
    {62,45},{45,44},{44,43},{43,42},
    {62,49},{49,48},{48,47},{47,46},
    {62,53},{53,52},{52,51},{51,50},
    {62,57},{57,56},{56,55},{55,54},
    {62,61},{61,60},{60,59},{59,58},
};
static const int N_LHAND_EDGES = (int)(sizeof(LHAND_EDGES)/sizeof(LHAND_EDGES[0]));

// ---------------------------------------------------------------------------
// Draw 2-D skeleton overlay on an OpenCV BGR image
// ---------------------------------------------------------------------------
static void draw_skeleton_2d(cv::Mat& img, const std::vector<fsb::MHRResult>& results)
{
    for (const auto& r : results)
    {
        if (r.keypoints_2d.size() < 70*2) continue;
        const float* kp = r.keypoints_2d.data();

        auto pt = [&](int j) -> cv::Point
        {
            return { (int)kp[j*2], (int)kp[j*2+1] };
        };

        // Body edges — green
        for (int e = 0; e < N_BODY_EDGES; ++e)
            cv::line(img, pt(BODY_EDGES[e][0]), pt(BODY_EDGES[e][1]),
                     cv::Scalar(0,200,0), 2, cv::LINE_AA);

        // Right-hand edges — blue
        for (int e = 0; e < N_RHAND_EDGES; ++e)
            cv::line(img, pt(RHAND_EDGES[e][0]), pt(RHAND_EDGES[e][1]),
                     cv::Scalar(200,80,0), 1, cv::LINE_AA);

        // Left-hand edges — red
        for (int e = 0; e < N_LHAND_EDGES; ++e)
            cv::line(img, pt(LHAND_EDGES[e][0]), pt(LHAND_EDGES[e][1]),
                     cv::Scalar(0,80,200), 1, cv::LINE_AA);

        // Joint dots
        for (int j = 0; j < 70; ++j)
        {
            cv::Scalar col;
            int r_px;
            if      (j <= 20)
            {
                col = cv::Scalar(0,255,0);      // body
                r_px = 4;
            }
            else if (j <= 41)
            {
                col = cv::Scalar(255,100,0);    // right hand
                r_px = 3;
            }
            else if (j <= 62)
            {
                col = cv::Scalar(0,100,255);    // left hand
                r_px = 3;
            }
            else
            {
                col = cv::Scalar(0,220,220);    // extra
                r_px = 4;
            }
            cv::circle(img, pt(j), r_px, col, -1, cv::LINE_AA);
        }
    }
}

// ---------------------------------------------------------------------------
// Write CSV header
// ---------------------------------------------------------------------------
static void write_csv_header(std::ofstream& f)
{
    f << "frame,skeleton_id";
    for (int j = 0; j < 70; ++j)
        f << "," << KP_NAMES[j] << "_x," << KP_NAMES[j] << "_y," << KP_NAMES[j] << "_z";
    f << "\n";
}

// ---------------------------------------------------------------------------
// Write one row per detected person
// ---------------------------------------------------------------------------
static void write_csv_rows(std::ofstream& f, int frame_no,
                           const std::vector<fsb::MHRResult>& results)
{
    for (int i = 0; i < (int)results.size(); ++i)
    {
        const auto& r = results[i];
        if (r.keypoints_3d.size() < 70*3) continue;
        f << frame_no << "," << i;
        for (int j = 0; j < 70; ++j)
            f << "," << r.keypoints_3d[j*3]
              << "," << r.keypoints_3d[j*3+1]
              << "," << r.keypoints_3d[j*3+2];
        f << "\n";
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    Config c = parse_args(argc, argv);

    // -----------------------------------------------------------------------
    // Sanity-check working directory: warn if onnx/ is missing and the user
    // hasn't overridden --onnx-dir.  Common mistake: running from build/
    // instead of the repo root.
    // -----------------------------------------------------------------------
    {
        struct stat st{};
        if (c.onnx_dir == "./onnx" && stat("./onnx", &st) != 0)
        {
            fprintf(stderr,
                "\n"
                "  ╔══════════════════════════════════════════════════════════════╗\n"
                "  ║  WARNING: './onnx' not found in the current directory.       ║\n"
                "  ║                                                              ║\n"
                "  ║  You are probably running from the wrong directory.          ║\n"
                "  ║  Run from the repository root, or pass --onnx-dir:           ║\n"
                "  ║                                                              ║\n"
                "  ║    cd /path/to/SAM3DBody-cpp                                 ║\n"
                "  ║    ./build/fast_sam_3dbody_run --from YOURFILE.mp4           ║\n"
                "  ║                                                              ║\n"
                "  ╚══════════════════════════════════════════════════════════════╝\n"
                "\n");
        }
    }

    // -----------------------------------------------------------------------
    // Optional CSV output
    // -----------------------------------------------------------------------
    std::ofstream csv_out;
    if (!c.csv_path.empty())
    {
        csv_out.open(c.csv_path);
        if (!csv_out)
        {
            fprintf(stderr, "[main] Cannot open CSV for writing: %s\n", c.csv_path.c_str());
            return 1;
        }
        write_csv_header(csv_out);
        printf("[main] Writing 3D keypoints to: %s\n", c.csv_path.c_str());
    }

    // -----------------------------------------------------------------------
    // Optional BVH output
    // -----------------------------------------------------------------------
    BVHWriter bvh_writer;
    if (!c.bvh_path.empty())
    {
        // Auto-detect body_model.lbs from the onnx directory for finger animation
        std::string lbs_path = c.onnx_dir + "/body_model.lbs";

        if (!bvh_writer.open(c.bvh_template, c.bvh_path,
                             1.0f / 30.0f,    // default 30 fps; TODO: derive from cap_fps
                             lbs_path,
                             c.bvh_body_shape_change,
                             c.bvh_hand_shape_change,
                             c.bvh_compensate_finger_endsites,
                             c.bvh_enforce_hand_limits,
                             c.bvh_zero_hand_pose,
                             c.bvh_sticky_hand_pose,
                             c.bvh_rest_align,
                             c.bvh_dump_rest_dirs))
        {
            fprintf(stderr, "[main] BVH writer failed to open (continuing without BVH output).\n");
        }
        else
        {
            bvh_writer.set_foot_contact(c.bvh_foot_contact);
            printf("[main] Writing BVH to: %s  (template: %s)\n",
                   c.bvh_path.c_str(), c.bvh_template.c_str());
        }
    }

    fsb::PipelineConfig pcfg;
    apply_common_to_pipeline_cfg(c, pcfg);  // all shared pipeline fields
    pcfg.skip_body_model  = c.skip_body;
    pcfg.zero_face_params = c.zero_face;
    pcfg.focal_x          = c.focal_x;
    pcfg.focal_y          = c.focal_y;
    pcfg.principal_x      = c.cx;
    pcfg.principal_y      = c.cy;

    fsb::Pipeline pipeline;
    {
        auto t0 = Clock::now();
        if (!pipeline.load(pcfg))
        {
            fprintf(stderr, "[main] Pipeline load failed.\n");
            return 1;
        }
        printf("[main] Pipeline loaded in %.1f ms\n", ms_since(t0));
    }

    pipeline.print_info();
    if (c.info_only)
    {
        pipeline.free();
        return 0;
    }

    // -----------------------------------------------------------------------
    // Open input source
    // -----------------------------------------------------------------------
    cv::VideoCapture cap;
    bool is_image = false;

    bool src_is_int = !c.from.empty() &&
                      c.from.find_first_not_of("0123456789") == std::string::npos;

    // A webcam is an integer index OR a /dev/videoX path — not a recorded video file.
    // Only webcams need frame-sync skipping; video files have no real-time clock.
    bool is_webcam = src_is_int ||
                     (c.from.size() >= 10 &&
                      c.from.compare(0, 10, "/dev/video") == 0);

    if (src_is_int)
    {
        cap.open(std::stoi(c.from));
    }
    else
    {
        // Treat as image if it has a known image extension
        const char* img_exts[] = {".jpg",".jpeg",".png",".bmp",".tiff",".webp", nullptr};
        for (int k = 0; img_exts[k]; ++k)
        {
            auto ext = img_exts[k];
            auto elen = strlen(ext);
            if (c.from.size() >= elen &&
                    c.from.compare(c.from.size()-elen, elen, ext) == 0)
            {
                is_image = true;
                break;
            }
        }
        if (!is_image) cap.open(c.from);
    }

    if (!is_image && cap.isOpened())
    {
        if (c.cap_w > 0) cap.set(cv::CAP_PROP_FRAME_WIDTH,  c.cap_w);
        if (c.cap_h > 0) cap.set(cv::CAP_PROP_FRAME_HEIGHT, c.cap_h);
        if (c.cap_fps > 0.0) cap.set(cv::CAP_PROP_FPS,      c.cap_fps);
    }

    // Query the camera's native framerate for frame-sync logic.
    // For video files this is the encoded fps, which we don't want to skip against.
    double cam_fps = 0.0;
    if (is_webcam && cap.isOpened())
    {
        cam_fps = cap.get(cv::CAP_PROP_FPS);
        if (cam_fps <= 0.0) cam_fps = 30.0;   // sensible fallback if driver doesn't report
        printf("[main] Webcam FPS from driver: %.1f\n", cam_fps);
    }

    if (!is_image && !cap.isOpened())
    {
        fprintf(stderr, "[main] Cannot open input: %s\n", c.from.c_str());
        pipeline.free();
        return 1;
    }

    // -----------------------------------------------------------------------
    // Butterworth filter state (one bank of filters per person slot)
    // -----------------------------------------------------------------------
    struct PersonFilters
    {
        std::array<ButterWorth, 70*3> kp3d{};
        std::array<ButterWorth, 133>  body_pose{};
        std::array<ButterWorth, 108>  hand_pose{};
        std::array<ButterWorth, 3>    cam_t{};
        // global_rot uses a quaternion 1st-order SLERP-EMA instead of per-axis
        // Butterworth.  Per-axis filtering on Euler triples interpolates linearly
        // through the ±π wrap and through gimbal-lock, which manifests as visible
        // body flips.  QuatLPF runs on the orientation directly: hemisphere-
        // corrected SLERP, geodesic outlier clamp (--rot-clamp deg / frame).
        QuatLPF                       root_rot{};
    };
    std::vector<PersonFilters> bw_filters;
    const float bw_fps = (c.cap_fps > 0.0) ? (float)c.cap_fps : 30.0f;
    // Butterworth is only valid below the Nyquist frequency (fps/2).
    // Above that, tan(π·fc/fs)→0, all IIR coefficients collapse to zero, and
    // filter() outputs 0 after 3 frames — corrupting pred_cam_t, keypoints, etc.
    // When the cutoff is at or above Nyquist we skip the filter (pass-through).
    const bool use_bw = c.butterworth && (c.bw_cutoff < bw_fps * 0.5f);

    auto init_person_filters = [&](PersonFilters& pf)
    {
        if (!use_bw) return;
        for (auto& f : pf.kp3d)      initButterWorth(&f, bw_fps, c.bw_cutoff);
        for (auto& f : pf.body_pose) initButterWorth(&f, bw_fps, c.bw_cutoff);
        for (auto& f : pf.hand_pose) initButterWorth(&f, bw_fps, c.bw_cutoff);
        for (auto& f : pf.cam_t)     initButterWorth(&f, bw_fps, c.bw_cutoff);
        // Root-rotation quaternion filter uses the same cutoff frequency so the
        // user can keep tuning a single --bw-cutoff knob.  Initialisation is
        // lazy in filter_quat (first sample warms the state to the input).
        init_quat_lpf(&pf.root_rot, bw_fps, c.bw_cutoff);
    };

    if (c.butterworth)
    {
        if (use_bw)
            printf("[main] Butterworth filter: %.1f Hz cutoff at %.1f Hz sample rate"
                   "  |  root rot clamp: %.1f deg/frame\n",
                   c.bw_cutoff, bw_fps, c.rot_clamp_deg);
        else
            printf("[main] Butterworth SKIPPED (cutoff %.1f Hz >= Nyquist %.1f Hz)"
                   "  |  root rot clamp: %.1f deg/frame\n",
                   c.bw_cutoff, bw_fps * 0.5f, c.rot_clamp_deg);
    }

    // -----------------------------------------------------------------------
    // Inference loop
    // -----------------------------------------------------------------------
    cv::Mat  frame;
    int      frame_count  = 0;
    double   total_inf_ms = 0.0;
    auto     loop_start   = Clock::now();
    uint64_t t_capture_start_ns = 0;   // set on first webcam grab

    while (true)
    {
        if (is_image)
        {
            frame = cv::imread(c.from);
            if (frame.empty())
            {
                fprintf(stderr, "[main] Cannot read image.\n");
                break;
            }
        }
        else
        {
            // ---- webcam frame-sync: drain stale frames from the driver buffer ----
            // The camera keeps capturing at cam_fps regardless of how fast we process.
            // We use a monotonic clock to see how many camera frames have been produced
            // since we started, then skip (grab without decode) any we haven't consumed
            // so the next cap.read() returns a frame that is current right now.
            if (is_webcam && cam_fps > 0.0)
            {
                uint64_t now_ns = get_mono_ns();
                if (t_capture_start_ns == 0)
                {
                    // Pin the clock to the very first frame we're about to grab.
                    t_capture_start_ns = now_ns;
                }
                else
                {
                    uint64_t elapsed_ns = now_ns - t_capture_start_ns;
                    // How many frames the camera has produced since we began.
                    int64_t expected = (int64_t)((double)elapsed_ns * cam_fps / 1e9);
                    // Frames we still owe the camera — skip them (grab only, no decode).
                    int skip = (int)(expected - (int64_t)frame_count);
                    if (skip > 0)
                    {
                        // Cap to a reasonable window; we only need the *latest* frame,
                        // so skip at most what fits in ~1 second of camera output.
                        if (skip > (int)cam_fps) skip = (int)cam_fps;
                        for (int s = 0; s < skip; ++s)
                            if (!cap.grab()) break;
                    }
                }
            }
            if (!cap.read(frame) || frame.empty()) break;
        }

        auto t0 = Clock::now();
        std::vector<fsb::MHRResult> results =
            pipeline.process_bgr(frame.data, frame.cols, frame.rows);
        double inf_ms = ms_since(t0);

        // Apply Butterworth low-pass filter to each person's MHR output vectors
        if (c.butterworth)
        {
            // Grow filter bank as new person slots appear
            while (bw_filters.size() < results.size())
            {
                bw_filters.emplace_back();
                init_person_filters(bw_filters.back());
            }
            for (int i = 0; i < (int)results.size(); ++i)
            {
                auto& r  = results[i];
                auto& pf = bw_filters[i];

                if (use_bw)
                {
                    for (int k = 0; k < (int)r.keypoints_3d.size() && k < 70*3; ++k)
                        r.keypoints_3d[k] = filter(&pf.kp3d[k], r.keypoints_3d[k]);

                    for (int k = 0; k < (int)r.body_pose.size() && k < 133; ++k)
                        r.body_pose[k] = filter(&pf.body_pose[k], r.body_pose[k]);

                    for (int k = 0; k < (int)r.hand_pose.size() && k < 108; ++k)
                        r.hand_pose[k] = filter(&pf.hand_pose[k], r.hand_pose[k]);
                }

                // global_rot: quaternion-domain 1st-order SLERP-EMA.  The Euler
                // triple is stored ZYX-intrinsic ([rz, ry, rx]) so we convert
                // via the matching helper, filter on the orientation directly,
                // then write back in the same order.  --rot-clamp is now a
                // geodesic outlier clamp on the SLERP step (deg / frame); 0
                // disables it (pure EMA).
                if (c.filter_root_rot)
                {
                    float in_q[4], out_q[4];
                    euler_zyx_to_quat(r.global_rot[0], r.global_rot[1],
                                      r.global_rot[2], in_q);
                    float max_step_rad = (c.rot_clamp_deg > 0.0f)
                        ? c.rot_clamp_deg * (3.14159265359f / 180.0f) : 0.0f;
                    filter_quat(&pf.root_rot, in_q, max_step_rad, out_q);
                    quat_to_euler_zyx(out_q, &r.global_rot[0],
                                              &r.global_rot[1],
                                              &r.global_rot[2]);
                }

                if (use_bw)
                    for (int k = 0; k < 3; ++k)
                        r.pred_cam_t[k] = filter(&pf.cam_t[k], r.pred_cam_t[k]);
            }
        }
        total_inf_ms += inf_ms;
        ++frame_count;

        printf("frame %d  |  %.1f ms  |  %d person(s)\n",
               frame_count, inf_ms, (int)results.size());
        for (int i = 0; i < (int)results.size(); ++i)
            print_result(i, results[i]);

        // CSV
        if (csv_out.is_open())
            write_csv_rows(csv_out, frame_count, results);

        // BVH
        if (bvh_writer.is_open())
            bvh_writer.write_frame(results);

        // Visualization
        if (!c.headless)
        {
            cv::Mat vis = frame.clone();
            draw_skeleton_2d(vis, results);
            // HUD
            char hud[128];
            snprintf(hud, sizeof(hud), "frame %d | %.0f ms | %d person(s) | q=quit",
                     frame_count, inf_ms, (int)results.size());
            cv::putText(vis, hud, {10, 24},
                        cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0,255,255), 2, cv::LINE_AA);
            cv::imshow("Fast-SAM-3D-Body", vis);
            int key = cv::waitKey(is_image ? 0 : 1) & 0xFF;
            if (key == 'q' || key == 27) break;
        }

        if (is_image) break;

        // FPS every 30 frames
        if (frame_count % 30 == 0)
        {
            double wall_s    = ms_since(loop_start) / 1000.0;
            double inf_fps   = frame_count * 1000.0 / (total_inf_ms > 0 ? total_inf_ms : 1);
            double achieved  = frame_count / (wall_s > 0 ? wall_s : 1);
            if (is_webcam && cam_fps > 0.0)
                printf("[fps] cam=%.1f  achieved=%.1f  inf=%.1f\n",
                       cam_fps, achieved, inf_fps);
            else
                printf("[fps] inf=%.1f  wall=%.1f\n", inf_fps, achieved);
        }
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    if (frame_count > 0)
    {
        double wall_s = ms_since(loop_start) / 1000.0;
        printf("\n--- Summary (%d frames) ---\n", frame_count);
        printf("  Inf fps  : %.1f\n", frame_count * 1000.0 / (total_inf_ms > 0 ? total_inf_ms : 1));
        printf("  Wall fps : %.1f\n", frame_count / (wall_s > 0 ? wall_s : 1));
        printf("  Inf ms   : %.2f / frame\n", total_inf_ms / frame_count);
    }

    if (bvh_writer.is_open())
        bvh_writer.close();

    pipeline.free();
    return 0;
}
