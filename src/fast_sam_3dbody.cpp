// ============================================================================
// fast_sam_3dbody.cpp  –  SAM-3D-Body inference pipeline
//
// Stage map:
//  YOLO (ONNX/TRT)  → person bboxes
//  backbone.onnx    → [B,1280,32,32]  image features
//  decoder.onnx     → [B,1024]        pose token
//  pipeline.gguf    → [B,519]+[B,3]   MHR params + camera params  (ggml)
//  body_model.onnx  → [B,18439,3]     SMPL-like vertices  (optional)
// ============================================================================

#define FSB_HAS_OPENCV_MAT  1

#include "fast_sam_3dbody.h"
#include "preprocess.hpp"

// ── ggml headers ─────────────────────────────────────────────────────────────
#if __has_include(<ggml/ggml.h>)
#  include <ggml/ggml.h>
#  include <ggml/ggml-alloc.h>
#  include <ggml/ggml-backend.h>
#  include <ggml/ggml-cpu.h>
#  include <ggml/gguf.h>
#else
#  include <ggml.h>
#  include <ggml-alloc.h>
#  include <ggml-backend.h>
#  include <ggml-cpu.h>
#  include <gguf.h>
#endif
#if defined(GGML_USE_CUDA)
#  if __has_include(<ggml/ggml-cuda.h>)
#    include <ggml/ggml-cuda.h>
#  elif __has_include(<ggml-cuda.h>)
#    include <ggml-cuda.h>
#  endif
#endif

// ── ONNX Runtime ─────────────────────────────────────────────────────────────
#include <onnxruntime_cxx_api.h>

// ── OpenCV ───────────────────────────────────────────────────────────────────
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

// ── LBS ──────────────────────────────────────────────────────────────────────
#include "../GraphicsEngine/ModelLoader/model_loader_transform_joints.h"
#include "mhr_lbs_cuda.cuh"

// ── STL ──────────────────────────────────────────────────────────────────────
#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fsb
{

// ─────────────────────────────────────────────────────────────────────────────
// Timing helper
// ─────────────────────────────────────────────────────────────────────────────
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// ─────────────────────────────────────────────────────────────────────────────
// GGUF metadata
// ─────────────────────────────────────────────────────────────────────────────
struct GGUFMeta
{
    uint32_t decoder_dim  = 1024;
    uint32_t npose        = 519;
    uint32_t cam_out_dim  = 3;
    uint32_t num_vertices = 18439;
    uint32_t num_kps      = 70;
    float    default_focal= 800.f;
    float    person_thresh= 0.5f;
    float    nms_iou      = 0.45f;
};

static uint32_t gguf_u32(gguf_context* c, const char* k, uint32_t def=0)
{
    int id = gguf_find_key(c, k);
    return id>=0 ? gguf_get_val_u32(c, id) : def;
}

// ─────────────────────────────────────────────────────────────────────────────
// Small FFN  (MHR head / camera head)  – plain C++ CPU matmul
//
// Architecture: Linear(in, hid) + ReLU + Linear(hid, out)
// Weights loaded from GGUF (f16 weights converted to f32 on load).
//   {prefix}.fc0.{weight,bias}   –  shape [hid, in] / [hid]
//   {prefix}.fc1.{weight,bias}   –  shape [out, hid] / [out]
//
// Row-major storage: w0[i * in_dim + j] = weight from input j to hidden i.
// Inference: y = relu(x @ w0.T + b0) @ w1.T + b1
// ─────────────────────────────────────────────────────────────────────────────
struct CFFN
{
    std::vector<float> w0, b0, w1, b1;
    int in_dim=0, hid_dim=0, out_dim=0;
};

static bool cffn_load(CFFN& ffn,
                      gguf_context*  gctx,
                      ggml_context*  wctx,   // created by gguf_init_from_file
                      FILE*          fp,
                      size_t         data_base,
                      const std::string& prefix)
{
    // Read one weight tensor by name; convert f16→f32 if needed.
    // Shape comes from the ggml context created alongside the gguf context.
    auto read_f32 = [&](const char* suffix, std::vector<float>& out) -> bool
    {
        std::string name = prefix + suffix;

        // Get shape from the ggml context
        ggml_tensor* t = ggml_get_tensor(wctx, name.c_str());
        if (!t)
        {
            fprintf(stderr, "[FFN] tensor not found: %s\n", name.c_str());
            return false;
        }
        size_t n    = ggml_nelements(t);
        int64_t idx = gguf_find_tensor(gctx, name.c_str());
        size_t  off = gguf_get_tensor_offset(gctx, idx);
        int     type = (int)gguf_get_tensor_type(gctx, idx);

        std::fseek(fp, (long)(data_base + off), SEEK_SET);
        out.resize(n);
        if (type == GGML_TYPE_F32)
        {
            if (std::fread(out.data(), sizeof(float), n, fp) != n) return false;
        }
        else if (type == GGML_TYPE_F16)
        {
            std::vector<uint16_t> tmp(n);
            if (std::fread(tmp.data(), sizeof(uint16_t), n, fp) != n) return false;
            ggml_fp16_to_fp32_row(tmp.data(), out.data(), (int)n);
        }
        else
        {
            fprintf(stderr, "[FFN] unsupported weight type %d for %s\n", type, name.c_str());
            return false;
        }
        return true;
    };

    // Retrieve dimension info from ggml context tensors
    auto get_tensor = [&](const char* suffix) -> ggml_tensor*
    {
        return ggml_get_tensor(wctx, (prefix + suffix).c_str());
    };

    if (!read_f32(".fc0.weight", ffn.w0)) return false;
    if (!read_f32(".fc0.bias",   ffn.b0)) return false;
    if (!read_f32(".fc1.weight", ffn.w1)) return false;
    if (!read_f32(".fc1.bias",   ffn.b1)) return false;

    // ne[0]=Cin, ne[1]=Cout for weight matrices (GGML column-major vs numpy row-major)
    auto* w0t = get_tensor(".fc0.weight");
    auto* w1t = get_tensor(".fc1.weight");
    ffn.in_dim  = (int)w0t->ne[0];
    ffn.hid_dim = (int)w0t->ne[1];
    ffn.out_dim = (int)w1t->ne[1];
    return true;
}

// y = relu(x @ w.T + b)   x:[B,K]  w:[N,K]  b:[N]  → out:[B,N]
static void linear_relu(const float* x, const float* w, const float* b,
                        float* y, int B, int K, int N, bool relu)
{
    for (int bi = 0; bi < B; ++bi)
    {
        for (int n = 0; n < N; ++n)
        {
            float s = b[n];
            const float* xr = x + bi * K;
            const float* wr = w + n * K;
            for (int k = 0; k < K; ++k) s += xr[k] * wr[k];
            y[bi * N + n] = relu ? std::max(0.f, s) : s;
        }
    }
}

static std::vector<float> cffn_run(const CFFN& ffn, const float* x, int B)
{
    std::vector<float> h(B * ffn.hid_dim);
    linear_relu(x,       ffn.w0.data(), ffn.b0.data(),
                h.data(), B, ffn.in_dim,  ffn.hid_dim, true);

    std::vector<float> y(B * ffn.out_dim);
    linear_relu(h.data(), ffn.w1.data(), ffn.b1.data(),
                y.data(),  B, ffn.hid_dim, ffn.out_dim, false);
    return y;
}

// ─────────────────────────────────────────────────────────────────────────────
// ONNX Runtime session wrapper
// ─────────────────────────────────────────────────────────────────────────────
struct OrtSession
{
    Ort::Env*             env     = nullptr;
    Ort::Session*         session = nullptr;
    Ort::MemoryInfo       mem_info{ nullptr };
    std::vector<std::string>       input_names_s,  output_names_s;
    std::vector<const char*>       input_names,    output_names;

    bool load(Ort::Env& e, const std::string& path, bool cuda, int device,
              bool fp16_io = false, bool trt_ep = false)
    {
        // Try with the requested EP first; fall back to CPU if it fails to load
        // (e.g. libcudnn not installed, CUDA EP shared library missing).
        for (int attempt = 0; attempt < 2; ++attempt)
        {
            bool try_cuda = cuda && (attempt == 0);
            Ort::SessionOptions opts;
            opts.SetIntraOpNumThreads(1);
            opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            try
            {
                if (try_cuda && !trt_ep)
                {
                    OrtCUDAProviderOptions cp{};
                    cp.device_id = device;
                    opts.AppendExecutionProvider_CUDA(cp);
                }
#if defined(USE_TENSORRT_EP)
                if (try_cuda && trt_ep)
                {
                    OrtTensorRTProviderOptions tp{};
                    tp.device_id = device;
                    tp.trt_fp16_enable = fp16_io ? 1 : 0;
                    opts.AppendExecutionProvider_TensorRT(tp);
                }
#endif
                session = new Ort::Session(e, path.c_str(), opts);
                if (!try_cuda && cuda)
                    fprintf(stderr, "[ORT] WARNING: '%s' running on CPU (CUDA EP unavailable)\n",
                            path.c_str());
                break;  // success
            }
            catch (const Ort::Exception& ex)
            {
                if (try_cuda)
                {
                    fprintf(stderr, "[ORT] CUDA EP failed (%s)\n[ORT] Retrying '%s' on CPU…\n",
                            ex.what(), path.c_str());
                    continue;  // retry without CUDA
                }
                fprintf(stderr, "[ORT] load '%s' failed: %s\n", path.c_str(), ex.what());
                return false;
            }
        }
        if (!session) return false;
        env = &e;

        Ort::AllocatorWithDefaultOptions alloc;
        size_t n_in  = session->GetInputCount();
        size_t n_out = session->GetOutputCount();
        input_names_s.resize(n_in);
        output_names_s.resize(n_out);
        input_names.resize(n_in);
        output_names.resize(n_out);
        for (size_t i = 0; i < n_in;  ++i)
            input_names_s[i]  = session->GetInputNameAllocated(i,  alloc).get(),
                                input_names[i]    = input_names_s[i].c_str();
        for (size_t i = 0; i < n_out; ++i)
            output_names_s[i] = session->GetOutputNameAllocated(i, alloc).get(),
                                output_names[i]   = output_names_s[i].c_str();

        mem_info = Ort::MemoryInfo::CreateCpu(
                       OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        return true;
    }

    // Run with a single float32 input tensor (for backbone)
    std::vector<float> run1(const float* in_data,
                            const std::vector<int64_t>& in_shape,
                            size_t out_elems)
    {
        Ort::Value in_t = Ort::Value::CreateTensor<float>(
                              mem_info, const_cast<float*>(in_data), in_shape[0]*in_shape[1]*in_shape[2]*in_shape[3],
                              in_shape.data(), in_shape.size());
        auto out = session->Run(Ort::RunOptions{nullptr},
                                input_names.data(),  &in_t,    1,
                                output_names.data(), output_names.size());
        std::vector<float> result(out_elems);
        auto* src = out[0].GetTensorMutableData<float>();
        std::memcpy(result.data(), src, out_elems * sizeof(float));
        return result;
    }

    void free()
    {
        delete session;
        session = nullptr;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline::Impl
// ─────────────────────────────────────────────────────────────────────────────
struct Pipeline::Impl
{
    PipelineConfig  cfg;
    GGUFMeta        meta;
    bool            loaded = false;

    // ONNX Runtime
    Ort::Env        ort_env{ORT_LOGGING_LEVEL_WARNING, "fast_sam_3dbody"};
    OrtSession      sess_yolo, sess_backbone, sess_decoder, sess_body;

    // CPU FFNs for MHR + camera heads (weights loaded from GGUF)
    CFFN mhr_ffn, cam_ffn;

    // Keypoint mapping: sparse COO format for 70 MHR keypoints
    // Maps [vertices(18439) + joints(127)] → 70 keypoints
    struct KpEntry
    {
        int32_t row;
        int32_t col;
        float val;
    };
    std::vector<KpEntry> kp_mapping;

    // Native C LBS (body_model.lbs) — loaded when body_model.onnx is unavailable
    struct MHR_LBS_Data* lbs_data = nullptr;
    MHR_LBS_CUDACtx*    lbs_cuda = nullptr;   // GPU-accelerated path; null on CPU builds

    // ── per-stage timing accumulators ──────────────────────────────────────────
    // Wall time (ms) spent in each pipeline stage, summed across every
    // process_*() call.  print_timing_summary() reports the per-frame averages.
    struct StageTimers
    {
        double   detection  = 0.0;   // YOLO person detection
        double   preprocess = 0.0;   // crop / normalise / condition+ray info
        double   backbone   = 0.0;   // backbone.onnx
        double   decoder    = 0.0;   // decoder.onnx
        double   mhr_ffn    = 0.0;   // MHR + camera CPU FFN heads
        double   body_model = 0.0;   // body_model.onnx or native LBS
        uint64_t frames     = 0;     // images processed
        uint64_t persons    = 0;     // total person crops processed
    } timers;

    // ── load ──────────────────────────────────────────────────────────────────
    bool load(const PipelineConfig& c)
    {
        cfg = c;

        bool cuda = cfg.cuda_device >= 0;
        int  dev  = cfg.cuda_device;

        // ── ONNX sessions ─────────────────────────────────────────────────────
        auto opath = [&](const char* f)
        {
            return cfg.onnx_dir + "/" + f;
        };

        printf("[FSB] Loading backbone … ");
        fflush(stdout);
        if (!sess_backbone.load(ort_env, opath(cfg.backbone_name.c_str()), cuda, dev,
                                cfg.use_fp16, false))
            return false;
        printf("OK\n");

        printf("[FSB] Loading decoder  … ");
        fflush(stdout);
        if (!sess_decoder.load(ort_env, opath("decoder.onnx"), cuda, dev,
                               cfg.use_fp16, false))
            return false;
        printf("OK\n");

        if (!cfg.skip_body_model)
        {
            // Prefer body_model.onnx; fall back gracefully to body_model.pt
            // (body_model.pt requires LibTorch – planned via ggml, see TODO below)
            std::string bm_onnx = opath("body_model.onnx");
            std::ifstream bm_check(bm_onnx);
            if (bm_check.good())
            {
                bm_check.close();
                printf("[FSB] Loading body_model.onnx … ");
                fflush(stdout);
                if (!sess_body.load(ort_env, bm_onnx, cuda, dev, false, false))
                    return false;
                printf("OK\n");

                // Load keypoint mapping for 70 MHR keypoints
                std::string kp_path = opath("keypoint_mapping.bin");
                std::ifstream kp_f(kp_path, std::ios::binary);
                if (kp_f.is_open())
                {
                    uint32_t num_rows, num_cols, nnz;
                    kp_f.read(reinterpret_cast<char*>(&num_rows), 4);
                    kp_f.read(reinterpret_cast<char*>(&num_cols), 4);
                    kp_f.read(reinterpret_cast<char*>(&nnz), 4);
                    kp_mapping.reserve(nnz);
                    for (uint32_t i = 0; i < nnz; ++i)
                    {
                        KpEntry e;
                        kp_f.read(reinterpret_cast<char*>(&e.row), 4);
                        kp_f.read(reinterpret_cast<char*>(&e.col), 4);
                        kp_f.read(reinterpret_cast<char*>(&e.val), 4);
                        kp_mapping.push_back(e);
                    }
                    kp_f.close();
                    printf("[FSB] keypoint_mapping: %ux%u, %u non-zero entries\n",
                           num_rows, num_cols, nnz);
                }
                else
                {
                    printf("[FSB] keypoint_mapping.bin not found – 2D keypoint output disabled\n");
                }
            }
            else
            {
                printf("[FSB] body_model.onnx not found; trying body_model.lbs … ");
                fflush(stdout);
                std::string lbs_path = opath("body_model.lbs");
                lbs_data = mhr_lbs_load(lbs_path.c_str());
                if (lbs_data)
                {
                    printf("OK (%d joints, %d vertices)\n", lbs_data->n_joints, lbs_data->n_verts);
#ifdef FSB_CUDA
                    lbs_cuda = mhr_lbs_cuda_init(lbs_data);
                    if (lbs_cuda) printf("[FSB] LBS CUDA accelerated (GPU shape blend + scatter)\n");
#endif

                    // Load keypoint mapping even with LBS
                    std::string kp_path = opath("keypoint_mapping.bin");
                    std::ifstream kp_f(kp_path, std::ios::binary);
                    if (kp_f.is_open())
                    {
                        uint32_t num_rows, num_cols, nnz;
                        kp_f.read(reinterpret_cast<char*>(&num_rows), 4);
                        kp_f.read(reinterpret_cast<char*>(&num_cols), 4);
                        kp_f.read(reinterpret_cast<char*>(&nnz), 4);
                        kp_mapping.reserve(nnz);
                        for (uint32_t i = 0; i < nnz; ++i)
                        {
                            KpEntry e;
                            kp_f.read(reinterpret_cast<char*>(&e.row), 4);
                            kp_f.read(reinterpret_cast<char*>(&e.col), 4);
                            kp_f.read(reinterpret_cast<char*>(&e.val), 4);
                            kp_mapping.push_back(e);
                        }
                        kp_f.close();
                        printf("[FSB] keypoint_mapping: %ux%u, %u non-zero entries\n",
                               num_rows, num_cols, nnz);
                    }
                    else
                    {
                        printf("[FSB] keypoint_mapping.bin not found – 2D keypoint output disabled\n");
                    }
                }
                else
                {
                    printf("not found\n");
                    printf("[FSB] body_model.lbs not found; vertex/keypoint output disabled.\n");
                }
            }
        }

        // YOLO – optional (might not exist for image-only usage)
        if (!cfg.yolo_path.empty())
        {
            printf("[FSB] Loading YOLO … ");
            fflush(stdout);
            if (!sess_yolo.load(ort_env, cfg.yolo_path, cuda, dev, false, false))
            {
                fprintf(stderr, "[FSB] YOLO load failed – detection disabled\n");
            }
            else
            {
                printf("OK\n");
            }
        }

        // ── ggml / GGUF ───────────────────────────────────────────────────────
        printf("[FSB] Loading pipeline.gguf … ");
        fflush(stdout);
        if (!load_gguf(cfg.gguf_path)) return false;
        printf("OK\n");

        loaded = true;
        return true;
    }

    bool load_gguf(const std::string& path)
    {
        // Only use gguf for metadata + weight bytes; inference runs in plain C++.
        gguf_context* gctx = nullptr;
        ggml_context* tmp_ctx = nullptr;
        {
            struct gguf_init_params p
            {
                true, &tmp_ctx
            };
            gctx = gguf_init_from_file(path.c_str(), p);
        }
        if (!gctx)
        {
            fprintf(stderr, "[FSB] Cannot open GGUF: %s\n", path.c_str());
            return false;
        }

        meta.decoder_dim   = gguf_u32(gctx, "sam3dbody.decoder_dim", 1024);
        meta.npose         = gguf_u32(gctx, "sam3dbody.npose",        519);
        meta.default_focal = 800.f;
        meta.person_thresh = cfg.person_thresh;
        meta.nms_iou       = cfg.person_nms_iou;

        FILE* fp = std::fopen(path.c_str(), "rb");
        if (!fp)
        {
            gguf_free(gctx);
            if (tmp_ctx) ggml_free(tmp_ctx);
            return false;
        }
        size_t data_base = gguf_get_data_offset(gctx);

        bool ok = cffn_load(mhr_ffn, gctx, tmp_ctx, fp, data_base, "mhr_proj")
                  && cffn_load(cam_ffn, gctx, tmp_ctx, fp, data_base, "cam_proj");

        std::fclose(fp);
        gguf_free(gctx);
        if (tmp_ctx) ggml_free(tmp_ctx);
        if (!ok) return false;

        printf("[FSB] FFNs: MHR(%dx%d->%d) Cam(%dx%d->%d)\n",
               mhr_ffn.in_dim, mhr_ffn.hid_dim, mhr_ffn.out_dim,
               cam_ffn.in_dim, cam_ffn.hid_dim, cam_ffn.out_dim);
        return true;
    }

    // ── process_bgr ───────────────────────────────────────────────────────────
    std::vector<MHRResult> process_bgr(const uint8_t* bgr, int W, int H)
    {
        cv::Mat img(H, W, CV_8UC3, const_cast<uint8_t*>(bgr));
        return process_mat(img, W, H);
    }

    std::vector<MHRResult> process_mat(const cv::Mat& bgr, int W, int H)
    {
        auto t_total = Clock::now();

        // ── camera intrinsics ─────────────────────────────────────────────────
        // Default matches Python sam_3d_body/data/utils/prepare_batch.py:
        //   focal = sqrt(W^2 + H^2)        (image diagonal — when no FOV estimator)
        //   cx, cy = W/2, H/2
        // This is the value the Python decoder/FFN was trained against; using a
        // smaller default (e.g. W) produces a wrong condition_info → wrong
        // global_rot / pred_cam_t / pose params from the FFN.
        float default_focal = std::sqrt(float(W)*float(W) + float(H)*float(H));
        float fx = (cfg.focal_x    > 0.f) ? cfg.focal_x    : default_focal;
        float fy = (cfg.focal_y    > 0.f) ? cfg.focal_y    : default_focal;
        float cx = (cfg.principal_x> 0.f) ? cfg.principal_x: float(W) * 0.5f;
        float cy = (cfg.principal_y> 0.f) ? cfg.principal_y: float(H) * 0.5f;

        // ── person detection ──────────────────────────────────────────────────
        auto t0 = Clock::now();
        std::vector<PersonDet> dets;

        if (sess_yolo.session)
        {
            // YOLO11 input: 640×640.
            // We must match Ultralytics YOLO's default preprocessing (LetterBox):
            //   resize keeping aspect ratio, then pad to 640×640 with grey (114).
            // Naive resize to 640×640 stretches a 3:2 image and produces wrong
            // bboxes that diverge from the Python reference by tens of pixels.
            const int YW = 640, YH = 640;
            float scale = std::min(float(YW) / float(W), float(YH) / float(H));
            int new_w = (int)std::round(W * scale);
            int new_h = (int)std::round(H * scale);
            int pad_x = (YW - new_w) / 2;          // letterbox pad (left)
            int pad_y = (YH - new_h) / 2;          // letterbox pad (top)
            cv::Mat resized;
            cv::resize(bgr, resized, {new_w, new_h}, 0, 0, cv::INTER_LINEAR);
            cv::Mat yolo_in(YH, YW, CV_8UC3, cv::Scalar(114, 114, 114));
            resized.copyTo(yolo_in(cv::Rect(pad_x, pad_y, new_w, new_h)));
            // HWC uint8 → CHW float32 [0,1]
            std::vector<float> yolo_buf(3 * YH * YW);
            for (int y = 0; y < YH; ++y)
            {
                const uchar* row = yolo_in.ptr<uchar>(y);
                for (int x = 0; x < YW; ++x)
                {
                    yolo_buf[0*YH*YW + y*YW + x] = row[3*x+2] / 255.f; // R
                    yolo_buf[1*YH*YW + y*YW + x] = row[3*x+1] / 255.f; // G
                    yolo_buf[2*YH*YW + y*YW + x] = row[3*x+0] / 255.f; // B
                }
            }
            // Run YOLO – output shape: [1, num_dets, 56] (or [1, 56, num_dets] depending on export)
            Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<int64_t> in_shape{1, 3, YH, YW};
            Ort::Value in_t = Ort::Value::CreateTensor<float>(
                                  mi, yolo_buf.data(), yolo_buf.size(), in_shape.data(), 4);

            try
            {
                auto outs = sess_yolo.session->Run(
                                Ort::RunOptions{nullptr},
                                sess_yolo.input_names.data(),  &in_t,  1,
                                sess_yolo.output_names.data(), 1);

                auto info   = outs[0].GetTensorTypeAndShapeInfo();
                auto shape  = info.GetShape();
                // Output is [1, C, N] (channels-first → needs transpose) or
                // [1, N, C]. C is the per-detection feature count (56 for
                // YOLO11-pose, 84 for YOLOv9 detection); N is the anchor count
                // and is always the larger dim. Detect the layout by size so the
                // same code feeds either parser.
                int nd = 0, C = 0;
                const float* raw = outs[0].GetTensorData<float>();
                std::vector<float> row_major;

                if (shape.size() == 3)
                {
                    int d1 = (int)shape[1], d2 = (int)shape[2];
                    if (d1 <= d2)
                    {
                        // [1, C, N] → transpose to row-major [N, C]
                        C = d1; nd = d2;
                        row_major.resize((size_t)nd * C);
                        for (int j = 0; j < nd; ++j)
                            for (int k = 0; k < C; ++k)
                                row_major[(size_t)j*C + k] = raw[(size_t)k*nd + j];
                    }
                    else
                    {
                        // [1, N, C] → already row-major
                        C = d2; nd = d1;
                        row_major.assign(raw, raw + (size_t)nd * C);
                    }
                }
                // Parse per the selected provider. The letterbox reversal below
                // (YOLO coords → original image coords) is shared by both.
                switch (cfg.detector)
                {
                case PipelineConfig::DET_LIBREYOLO:
                    dets = parse_yolov9_output(row_major.data(), nd, C,
                                               cfg.person_thresh, cfg.person_nms_iou);
                    break;
                case PipelineConfig::DET_YOLO_POSE:
                default:
                    dets = parse_yolo_output(row_major.data(), nd,
                                             cfg.person_thresh, cfg.person_nms_iou);
                    break;
                }
                for (auto& d : dets)
                {
                    d.x1 = (d.x1 - pad_x) / scale;
                    d.x2 = (d.x2 - pad_x) / scale;
                    d.y1 = (d.y1 - pad_y) / scale;
                    d.y2 = (d.y2 - pad_y) / scale;
                    if (d.has_kps)
                    {
                        for (int k = 0; k < 17; ++k)
                        {
                            d.kps[k*3 + 0] = (d.kps[k*3 + 0] - pad_x) / scale;
                            d.kps[k*3 + 1] = (d.kps[k*3 + 1] - pad_y) / scale;
                        }
                    }
                }
            }
            catch (const Ort::Exception& e)
            {
                fprintf(stderr, "[FSB] YOLO inference error: %s\n", e.what());
            }
        }

        // Fallback: full image as single detection
        if (dets.empty())
        {
            dets.push_back({ 0.f, 0.f, float(W), float(H), 1.f });
        }
        // Apply max_persons cap (sorted by confidence from NMS)
        if (cfg.max_persons > 0 && (int)dets.size() > cfg.max_persons)
            dets.resize(cfg.max_persons);
        double dt_detect = ms(t0);
        timers.detection += dt_detect;
        printf("[FSB] detection: %.1f ms  persons: %zu\n", dt_detect, dets.size());

        // ── per-person crops ──────────────────────────────────────────────────
        const int B = (int)dets.size();
        timers.frames  += 1;
        timers.persons += (uint64_t)B;
        const int plane = CROP_SIZE * CROP_SIZE;

        // Pre-allocate batch buffers
        const int ray_plane = FEAT_HW * FEAT_HW;
        std::vector<float> batch_crops   (B * 3 * plane);
        std::vector<float> batch_cond    (B * 3);
        std::vector<float> batch_ray     (B * 2 * ray_plane);
        std::vector<float> crop_cx_v(B), crop_cy_v(B), crop_sz_v(B);

        t0 = Clock::now();
        for (int i = 0; i < B; ++i)
        {
            const auto& d = dets[i];
            float* img_ptr = batch_crops.data() + i * 3 * plane;
            float& ccx     = crop_cx_v[i];
            float& ccy     = crop_cy_v[i];
            float& csz     = crop_sz_v[i];

            crop_and_normalise(bgr, d.x1, d.y1, d.x2, d.y2,
                               img_ptr, ccx, ccy, csz);

            float* cond_ptr = batch_cond.data() + i * 3;
            compute_condition_info(ccx, ccy, csz, fx, fy, cx, cy, cond_ptr);

            float* ray_ptr = batch_ray.data() + i * 2 * ray_plane;
            compute_ray_cond(ccx, ccy, csz, fx, fy, cx, cy, ray_ptr);
        }
        double dt_pre = ms(t0);
        timers.preprocess += dt_pre;
        printf("[FSB] preprocess: %.1f ms\n", dt_pre);

        // ── backbone ─────────────────────────────────────────────────────────
        t0 = Clock::now();
        const int FEAT_HW = CROP_SIZE / 16;   // 32
        const int BACKBONE_DIM = 1280;
        const size_t feat_elems = (size_t)B * BACKBONE_DIM * FEAT_HW * FEAT_HW;

        std::vector<int64_t> img_shape{B, 3, CROP_SIZE, CROP_SIZE};
        Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value img_t = Ort::Value::CreateTensor<float>(
                               mi, batch_crops.data(), batch_crops.size(), img_shape.data(), 4);
        auto backbone_out = sess_backbone.session->Run(
                                Ort::RunOptions{nullptr},
                                sess_backbone.input_names.data(),  &img_t,  1,
                                sess_backbone.output_names.data(), 1);
        const float* feat_ptr = backbone_out[0].GetTensorData<float>();
        std::vector<float> features(feat_ptr, feat_ptr + feat_elems);
        double dt_bb = ms(t0);
        timers.backbone += dt_bb;
        printf("[FSB] backbone:   %.1f ms\n", dt_bb);

        // ── decoder ──────────────────────────────────────────────────────────
        t0 = Clock::now();
        const int DECODER_DIM = (int)meta.decoder_dim;
        const size_t token_elems = (size_t)B * DECODER_DIM;

        std::vector<int64_t> feat_shape{B, BACKBONE_DIM, FEAT_HW, FEAT_HW};
        std::vector<int64_t> cond_shape{B, 3};
        std::vector<int64_t> ray_shape {B, 2, FEAT_HW, FEAT_HW};

        Ort::Value feat_t = Ort::Value::CreateTensor<float>(
                                mi, features.data(), features.size(), feat_shape.data(), 4);
        Ort::Value cond_t = Ort::Value::CreateTensor<float>(
                                mi, batch_cond.data(), batch_cond.size(), cond_shape.data(), 2);
        Ort::Value ray_t  = Ort::Value::CreateTensor<float>(
                                mi, batch_ray.data(), batch_ray.size(), ray_shape.data(), 4);

        std::vector<Ort::Value> dec_inputs;
        dec_inputs.push_back(std::move(feat_t));
        dec_inputs.push_back(std::move(cond_t));
        dec_inputs.push_back(std::move(ray_t));

        std::vector<const char*>& dec_in_names  = sess_decoder.input_names;
        std::vector<const char*>& dec_out_names = sess_decoder.output_names;

        auto decoder_out = sess_decoder.session->Run(
                               Ort::RunOptions{nullptr},
                               dec_in_names.data(),  dec_inputs.data(),  dec_inputs.size(),
                               dec_out_names.data(), 1);
        const float* token_ptr = decoder_out[0].GetTensorData<float>();
        std::vector<float> pose_tokens(token_ptr, token_ptr + token_elems);
        double dt_dec = ms(t0);
        timers.decoder += dt_dec;
        printf("[FSB] decoder:    %.1f ms\n", dt_dec);

        // ── MHR head (CPU FFN) ────────────────────────────────────────────────
        t0 = Clock::now();
        std::vector<float> mhr_raw  = cffn_run(mhr_ffn, pose_tokens.data(), B);
        std::vector<float> cam_raw  = cffn_run(cam_ffn, pose_tokens.data(), B);
        double dt_ffn = ms(t0);
        timers.mhr_ffn += dt_ffn;
        printf("[FSB] MHR FFN:    %.1f ms\n", dt_ffn);

        // ── body model (optional) ─────────────────────────────────────────────
        std::vector<float> all_verts, all_skel;
        bool use_lbs_skel = false;  // true if skeleton from LBS (float32, [127,3])
        if (!cfg.skip_body_model && sess_body.session)
        {
            t0 = Clock::now();
            // Build per-person body model inputs
            const int NPOSE = (int)meta.npose;
            std::vector<float> batch_shape  (B * 45, 0.f);
            std::vector<float> batch_bparams(B * 204, 0.f);
            std::vector<float> batch_face   (B * 72,  0.f);

            for (int i = 0; i < B; ++i)
            {
                const float* raw_i = mhr_raw.data() + i * NPOSE;
                // Parse: global_rot_6d[6] + body_cont[260] + shape[45] + scale[28] + hand[108] + face[72]
                const float* global_rot_6d  = raw_i;
                const float* body_cont      = raw_i + 6;
                const float* shape          = raw_i + 266;
                const float* face           = raw_i + 447;

                // Convert global rot 6D → Euler
                float global_rot_euler[3];
                rot6d_to_euler(global_rot_6d, global_rot_euler);

                // Convert body continuous params → 133-dim Euler
                float body_euler[133] = {};
                compact_cont_to_body_params(body_cont, body_euler);

                // Build model_params [204]
                ModelParams204 mp = build_model_params(global_rot_euler, body_euler, nullptr, true);

                // Copy into batch buffers
                std::memcpy(batch_shape.data()   + i * 45,  shape, 45  * sizeof(float));
                std::memcpy(batch_bparams.data() + i * 204, mp.data, 204 * sizeof(float));
                if (!cfg.zero_face_params)
                    std::memcpy(batch_face.data() + i * 72, face, 72 * sizeof(float));
                // else: batch_face stays zero-initialised → neutral expression
            }

            std::vector<int64_t> shape_sh  {B, 45};
            std::vector<int64_t> bparam_sh {B, 204};
            std::vector<int64_t> face_sh   {B, 72};

            Ort::Value shape_t  = Ort::Value::CreateTensor<float>(mi, batch_shape.data(),   B*45,  shape_sh.data(),  2);
            Ort::Value bparam_t = Ort::Value::CreateTensor<float>(mi, batch_bparams.data(), B*204, bparam_sh.data(), 2);
            Ort::Value face_t   = Ort::Value::CreateTensor<float>(mi, batch_face.data(),    B*72,  face_sh.data(),   2);

            // apply_correctives = False (constant bool tensor)
            bool corr_val = false;
            std::vector<int64_t> scalar_sh{};
            Ort::Value corr_t = Ort::Value::CreateTensor<bool>(mi, &corr_val, 1,
                                scalar_sh.data(), 0);

            std::vector<Ort::Value> body_ins;
            body_ins.push_back(std::move(shape_t));
            body_ins.push_back(std::move(bparam_t));
            body_ins.push_back(std::move(face_t));
            body_ins.push_back(std::move(corr_t));

            auto body_out = sess_body.session->Run(
                                Ort::RunOptions{nullptr},
                                sess_body.input_names.data(),  body_ins.data(),  4,
                                sess_body.output_names.data(), 2);

            const float* vp = body_out[0].GetTensorData<float>();
            const float* sp = body_out[1].GetTensorData<float>();
            size_t vn = (size_t)B * 18439 * 3;
            size_t sn = (size_t)B * 127   * 8;
            all_verts.assign(vp, vp + vn);
            all_skel.assign(sp,  sp + sn);
            double dt_body = ms(t0);
            timers.body_model += dt_body;
            printf("[FSB] body_model: %.1f ms\n", dt_body);
        }
        else if (!cfg.skip_body_model && lbs_data)
        {
            // Native C LBS fallback: compute vertices + joint coordinates
            use_lbs_skel = true;
            t0 = Clock::now();
            const int NPOSE = (int)meta.npose;
            all_verts.resize(B * 18439 * 3);
            all_skel.resize(B * 127 * 3);  // LBS outputs joints as [127, 3]

            for (int i = 0; i < B; ++i)
            {
                const float* raw_i = mhr_raw.data() + i * NPOSE;
                const float* global_rot_6d = raw_i;
                const float* body_cont     = raw_i + 6;
                //const float* shape         = raw_i + 266;
                //const float* face          = raw_i + 447;

                float global_rot_euler[3];
                rot6d_to_euler(global_rot_6d, global_rot_euler);

                float body_euler[133] = {};
                compact_cont_to_body_params(body_cont, body_euler);

                ModelParams204 mp = build_model_params(global_rot_euler, body_euler, nullptr, true);

                // Apply hand pose PCA decode (mirrors render binary + Python replace_hands_in_pose)
                const float* hand_pose = raw_i + 339;  // layout: 6+260+45+28=339
                apply_hand_pose(mp.data, hand_pose,
                                lbs_data->hand_pose_mean, lbs_data->hand_pose_comps,
                                lbs_data->hand_joint_idxs_left, lbs_data->hand_joint_idxs_right);

                // Apply scale decode: scales = scale_mean + scale_params @ scale_comps
                const float* scale_params = raw_i + 311;  // layout: 6+260+45=311
                if (lbs_data->scale_mean && lbs_data->scale_comps)
                {
                    const int ns = lbs_data->n_scale_out;  // 68
                    const int np = lbs_data->n_scale_pc;   // 28
                    for (int j = 0; j < ns; ++j) mp.data[136+j] = lbs_data->scale_mean[j];
                    for (int k = 0; k < np; ++k)
                        for (int j = 0; j < ns; ++j)
                            mp.data[136+j] += scale_params[k] * lbs_data->scale_comps[k * ns + j];
                }

                float* verts_out  = all_verts.data() + (size_t)i * 18439 * 3;
                float* joints_out = all_skel.data() + (size_t)i * 127 * 3;

                static const float zero_face[72] = {};
#ifdef FSB_CUDA
                if (lbs_cuda) {
                    mhr_lbs_cuda_compute(lbs_cuda, lbs_data, mp.data,
                                         raw_i + 266,
                                         cfg.zero_face_params ? zero_face : raw_i + 447,
                                         verts_out, joints_out);
                } else
#endif
                {
                    mhr_lbs_compute(lbs_data,
                                    mp.data,
                                    raw_i + 266,  /* shape */
                                    cfg.zero_face_params ? zero_face : raw_i + 447,  /* face */
                                    verts_out,
                                    joints_out);
                }
                printf("[FSB] LBS person %d done\n", i);
            }
            double dt_lbs = ms(t0);
            timers.body_model += dt_lbs;
            printf("[FSB] LBS:      %.1f ms, verts=%zu skel=%zu\n", dt_lbs, all_verts.size(), all_skel.size());
        }

        // ── assemble MHRResult per person ────────────────────────────────────
        std::vector<MHRResult> results(B);
        const int NPOSE = (int)meta.npose;

        for (int i = 0; i < B; ++i)
        {
            MHRResult& r   = results[i];
            const auto& d  = dets[i];
            const float* p = mhr_raw.data() + i * NPOSE;

            r.bbox = { d.x1, d.y1, d.x2, d.y2 };

            // ── Second-pass raw fields ────────────────────────────────────────
            // Store the raw MHR FFN output (first 266 floats = global_rot_6d[6]
            // + body_cont[260]) before Euler conversion.  The Python --two-passes
            // path needs the 6D continuous representation to rebuild prev_estimate
            // for forward_decoder; the Euler angles already stored in global_rot /
            // body_pose cannot reconstruct it.
            std::memcpy(r.pred_pose_raw.data(), p, 266 * sizeof(float));

            // Camera: convert raw head output [s, tx, ty] → [tx+cx, ty+cy, tz]
            // Mirrors Python cam_raw_to_pred_cam_t in fast_sam_3dbody_frontend-3D.py
            const float* cam = cam_raw.data() + i * 3;
            // Also store cam_raw before conversion (needed for prev_estimate when
            // the Python model has init_camera — appended as extra 3 floats).
            std::memcpy(r.pred_cam_raw.data(), cam, 3 * sizeof(float));
            {
                float s_val   = -cam[0];           // sign flip (Python: s = -pred_cam[:,0])
                float tx      =  cam[1];
                float ty      = -cam[2];           // sign flip (Python: ty = -pred_cam[:,2])
                float bw      = d.x2 - d.x1;
                float bh      = d.y2 - d.y1;
                float bbox_cx = (d.x1 + d.x2) * 0.5f;
                float bbox_cy = (d.y1 + d.y2) * 0.5f;
                float bs      = std::max(bw, bh) * 1.25f * s_val + 1e-8f;
                float tz      = 2.0f * fx / bs;
                float cx_off  = 2.0f * (bbox_cx - cx) / bs;
                float cy_off  = 2.0f * (bbox_cy - cy) / bs;
                r.pred_cam_t  = { tx + cx_off, ty + cy_off, tz };
            }
            r.focal_length = fx;

            // Global rotation 6D → Euler
            const float* g6d = p;
            float ge[3];
            rot6d_to_euler(g6d, ge);
            // rot6d_to_euler returns [rx,ry,rz] but mhr_forward expects [rz,ry,rx]
            r.global_rot = { ge[2], ge[1], ge[0] };

            // Body pose
            const float* bc = p + 6;
            float be[133] = {};
            compact_cont_to_body_params(bc, be);
            r.body_pose.assign(be, be + 133);

            // Shape [45]
            r.shape.assign(p + 266, p + 266 + 45);

            // Scale [28]
            r.scale.assign(p + 311, p + 311 + 28);

            // Hand pose [108]
            r.hand_pose.assign(p + 339, p + 339 + 108);

            // Face [72]
            r.face_params.assign(p + 447, p + 447 + 72);

            // Model params [204] for native C LBS – includes hand pose + scale decode
            {
                float ge[3];
                rot6d_to_euler(p, ge);
                float be[133] = {};
                compact_cont_to_body_params(p + 6, be);
                ModelParams204 mp = build_model_params(ge, be, nullptr, true);
                // Hand pose PCA decode (mirrors Python replace_hands_in_pose)
                apply_hand_pose(mp.data, p + 339,
                                lbs_data ? lbs_data->hand_pose_mean   : nullptr,
                                lbs_data ? lbs_data->hand_pose_comps  : nullptr,
                                lbs_data ? lbs_data->hand_joint_idxs_left  : nullptr,
                                lbs_data ? lbs_data->hand_joint_idxs_right : nullptr);
                // Scale decode: scales = scale_mean + scale_params @ scale_comps
                if (lbs_data && lbs_data->scale_mean && lbs_data->scale_comps)
                {
                    const int ns = lbs_data->n_scale_out;
                    const int np = lbs_data->n_scale_pc;
                    for (int j = 0; j < ns; ++j) mp.data[136+j] = lbs_data->scale_mean[j];
                    for (int k = 0; k < np; ++k)
                        for (int j = 0; j < ns; ++j)
                            mp.data[136+j] += p[311+k] * lbs_data->scale_comps[k * ns + j];
                }
                std::memcpy(r.mhr_model_params.data(), mp.data, 204 * sizeof(float));
            }

            // YOLO 2D keypoints [17 × 3]
            if (d.has_kps)
                r.keypoints_yolo.assign(d.kps, d.kps + 51);

            // Vertices (optional)
            if (!all_verts.empty())
            {
                size_t off = (size_t)i * 18439 * 3;
                r.pred_vertices.assign(all_verts.begin() + off,
                                       all_verts.begin() + off + 18439*3);
                // mhr_lbs_compute already applies y,z flip + cm→m — no additional flip needed.

                // Compute 70 MHR keypoints from vertices + skeleton joints
                if (!kp_mapping.empty())
                {
                    // Extract joint coordinates from skeleton state
                    std::vector<float> joint_coords(127 * 3);
                    if (use_lbs_skel)
                    {
                        // LBS output: float32 [B, 127, 3], already in meters, already flipped
                        const float* skel_j = all_skel.data() + (size_t)i * 127 * 3;
                        std::copy(skel_j, skel_j + 127*3, joint_coords.begin());
                    }
                    else
                    {
                        // ONNX body model output: float32 [B, 127, 8]
                        // First 3 floats per joint are world position (x,y,z) in meters.
                        const float* skel_j = all_skel.data() + (size_t)i * 127 * 8;
                        for (int j = 0; j < 127; ++j)
                        {
                            joint_coords[j*3 + 0] =  skel_j[j*8 + 0];
                            joint_coords[j*3 + 1] = -skel_j[j*8 + 1];  // y flip
                            joint_coords[j*3 + 2] = -skel_j[j*8 + 2];  // z flip
                        }
                    }

                    // Apply keypoint_mapping: sparse matrix-vector multiply
                    // [vertices(18439*3) + joints(127*3)] → keypoints_3d[70*3]
                    const float* verts_ptr = r.pred_vertices.data();
                    const float* joints_ptr = joint_coords.data();
                    std::vector<float> kps_3d(70 * 3, 0.f);

                    for (const auto& entry : kp_mapping)
                    {
                        // Each keypoint has 3 consecutive rows (x,y,z)
                        float coord_val = entry.val;
                        for (int c = 0; c < 3; ++c)
                        {
                            int row = entry.row * 3 + c;
                            int col = entry.col;
                            float src_val = 0.f;
                            if (col < 18439)
                                src_val = verts_ptr[col * 3 + c];
                            else
                                src_val = joints_ptr[(col - 18439) * 3 + c];
                            kps_3d[row] += src_val * coord_val;
                        }
                    }

                    // kps_3d is already in the camera coordinate system (y,z negated)
                    // because both verts_ptr and joints_ptr are post-flip inputs.
                    // No additional flip is needed here.
                    r.keypoints_3d = std::move(kps_3d);

                    // Project to 2D: kps_cam = kps_3d + pred_cam_t, then perspective divide
                    std::vector<float> kps_2d(70 * 2);
                    for (int k = 0; k < 70; ++k)
                    {
                        float dz = r.keypoints_3d[k*3 + 2] + r.pred_cam_t[2];
                        float dx = r.keypoints_3d[k*3 + 0] + r.pred_cam_t[0];
                        float dy = r.keypoints_3d[k*3 + 1] + r.pred_cam_t[1];
                        if (dz < 1e-4f) dz = 1e-4f;
                        kps_2d[k*2 + 0] = dx / dz * fx + cx;
                        kps_2d[k*2 + 1] = dy / dz * fy + cy;
                    }
                    r.keypoints_2d = std::move(kps_2d);
                }
            }
        }

        printf("[FSB] total: %.1f ms  (%d persons)\n", ms(t_total), B);
        printf("[FSB] returning results vector\n");
        return results;
    }

    // ── Whole-frame ViT embedding (scene-cut signal) ────────────────────────────
    std::vector<float> scene_embedding(const cv::Mat& bgr)
    {
        if (!sess_backbone.session || bgr.empty()) return {};

        // Resize the WHOLE frame (stretch, no bbox crop) to the backbone input
        // and normalise exactly as crop_and_normalise does: BGR→RGB, /255,
        // (x-mean)/std, interleaved→CHW.  We want a global scene descriptor,
        // so the aspect-ratio distortion from a plain resize is harmless and
        // identical frame-to-frame.
        cv::Mat resized;
        cv::resize(bgr, resized, {CROP_SIZE, CROP_SIZE}, 0, 0, cv::INTER_LINEAR);

        const int plane = CROP_SIZE * CROP_SIZE;
        std::vector<float> chw((size_t)3 * plane);
        for (int y = 0; y < CROP_SIZE; ++y) {
            const uchar* row = resized.ptr<uchar>(y);
            for (int x = 0; x < CROP_SIZE; ++x) {
                float b = row[3*x + 0] / 255.f;
                float g = row[3*x + 1] / 255.f;
                float r = row[3*x + 2] / 255.f;
                chw[0 * plane + y * CROP_SIZE + x] = (r - IMAGE_MEAN[0]) / IMAGE_STD[0];
                chw[1 * plane + y * CROP_SIZE + x] = (g - IMAGE_MEAN[1]) / IMAGE_STD[1];
                chw[2 * plane + y * CROP_SIZE + x] = (b - IMAGE_MEAN[2]) / IMAGE_STD[2];
            }
        }

        const int BACKBONE_DIM = 1280;
        const int HW = FEAT_HW * FEAT_HW;   // 32×32 spatial grid
        std::vector<int64_t> img_shape{1, 3, CROP_SIZE, CROP_SIZE};
        Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value img_t = Ort::Value::CreateTensor<float>(
                               mi, chw.data(), chw.size(), img_shape.data(), 4);
        auto out = sess_backbone.session->Run(
                       Ort::RunOptions{nullptr},
                       sess_backbone.input_names.data(),  &img_t,  1,
                       sess_backbone.output_names.data(), 1);
        const float* feat = out[0].GetTensorData<float>();   // [1,1280,32,32]

        // Global-average-pool over the spatial grid → 1280-d, then L2-normalise.
        std::vector<float> emb(BACKBONE_DIM, 0.f);
        for (int c = 0; c < BACKBONE_DIM; ++c) {
            const float* ch = feat + (size_t)c * HW;
            float acc = 0.f;
            for (int k = 0; k < HW; ++k) acc += ch[k];
            emb[c] = acc / (float)HW;
        }
        double norm = 0.0;
        for (float v : emb) norm += (double)v * v;
        norm = std::sqrt(norm) + 1e-8;
        for (float& v : emb) v = (float)(v / norm);
        return emb;
    }

    void free_all()
    {
        // CFFN weights are plain vectors – cleaned up automatically
        mhr_ffn = CFFN{};
        cam_ffn = CFFN{};
        sess_backbone.free();
        sess_decoder.free();
        sess_body.free();
        sess_yolo.free();
        if (lbs_cuda) { mhr_lbs_cuda_free(lbs_cuda); lbs_cuda = nullptr; }
        if (lbs_data)
        {
            mhr_lbs_free(lbs_data);
            lbs_data = nullptr;
        }
        loaded = false;
    }

    // ── timing summary ──────────────────────────────────────────────────────────
    // One line with the per-frame average wall time of every pipeline stage,
    // plus the number of frames / person crops processed.  Printed to stderr so
    // it stays visible alongside the live FPS/Latency line even when a frontend
    // redirects stdout to a file (e.g. scripts/webcam.sh → /tmp/render_raw.txt).
    void print_timing_summary() const
    {
        if (timers.frames == 0)
        {
            fprintf(stderr, "[FSB] timing: no frames processed.\n");
            return;
        }
        const double n = (double)timers.frames;
        const double total = timers.detection + timers.preprocess + timers.backbone +
                             timers.decoder + timers.mhr_ffn + timers.body_model;
        // Leading newline: the live FPS/Latency line is redrawn with '\r' and no
        // trailing newline, so start the summary on a fresh line.
        fprintf(stderr,
               "\n[FSB] timing over %llu frame(s), %llu person crop(s)  |  "
               "detection=%.2f  preprocess=%.2f  backbone=%.2f  decoder=%.2f  "
               "mhr_ffn=%.2f  body_model=%.2f  |  total=%.2f ms/frame\n",
               (unsigned long long)timers.frames,
               (unsigned long long)timers.persons,
               timers.detection / n, timers.preprocess / n, timers.backbone / n,
               timers.decoder / n,   timers.mhr_ffn / n,    timers.body_model / n,
               total / n);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline  (public interface)
// ─────────────────────────────────────────────────────────────────────────────
Pipeline::Pipeline()  : impl_(new Impl) {}
Pipeline::~Pipeline()
{
    free();
    delete impl_;
}

bool Pipeline::load(const PipelineConfig& cfg)
{
    return impl_->load(cfg);
}
void Pipeline::free()
{
    if (impl_) impl_->free_all();
}
bool Pipeline::is_loaded() const
{
    return impl_ && impl_->loaded;
}
void Pipeline::print_info() const
{
    if (!impl_ || !impl_->loaded)
    {
        printf("[FSB] not loaded\n");
        return;
    }
    const auto& m = impl_->meta;
    printf("\n=== fast_sam_3dbody ===\n");
    printf("  decoder_dim : %u\n", m.decoder_dim);
    printf("  npose       : %u\n", m.npose);
    printf("  num_vertices: %u\n", m.num_vertices);
    printf("  num_kps     : %u\n", m.num_kps);
    printf("  default_f   : %.0f\n", m.default_focal);
    printf("=======================\n\n");
}
std::vector<MHRResult> Pipeline::process_bgr(const uint8_t* bgr, int w, int h)
{
    return impl_->process_bgr(bgr, w, h);
}
void Pipeline::print_timing_summary() const
{
    if (impl_) impl_->print_timing_summary();
}
std::vector<float> Pipeline::scene_embedding(const uint8_t* bgr, int w, int h)
{
    if (!impl_) return {};
    cv::Mat img(h, w, CV_8UC3, const_cast<uint8_t*>(bgr));
    return impl_->scene_embedding(img);
}

} // namespace fsb
