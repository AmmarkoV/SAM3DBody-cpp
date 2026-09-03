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
#include <filesystem>
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
        // Execution-provider preference ladder, most→least preferred.  We try
        // each in turn and fall back on failure, so a missing TensorRT runtime
        // degrades to the CUDA EP, and a missing CUDA EP degrades to CPU, rather
        // than aborting the load.
        //   --trt  →  [TensorRT, CUDA, CPU]
        //   --cuda →  [CUDA, CPU]
        //   CPU    →  [CPU]
        enum EP { EP_TRT, EP_CUDA, EP_CPU };
        std::vector<EP> ladder;
        if (cuda && trt_ep) ladder.push_back(EP_TRT);
        if (cuda)           ladder.push_back(EP_CUDA);
        ladder.push_back(EP_CPU);

        auto ep_name = [](EP ep) {
            return ep == EP_TRT ? "TensorRT" : ep == EP_CUDA ? "CUDA" : "CPU";
        };

        for (size_t a = 0; a < ladder.size(); ++a)
        {
            const EP ep = ladder[a];
            const bool last = (a + 1 == ladder.size());
            Ort::SessionOptions opts;
            opts.SetIntraOpNumThreads(1);
            opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            try
            {
                if (ep == EP_TRT)
                {
#if defined(USE_TENSORRT_EP)
                    OrtTensorRTProviderOptions tp{};
                    tp.device_id             = device;
                    tp.trt_fp16_enable       = fp16_io ? 1 : 0;
                    tp.trt_max_workspace_size = (size_t)2 << 30;   // 2 GB build scratch
                    // The legacy options struct is zero-initialised, but TRT
                    // rejects 0 for these two (they must be positive) — set ORT's
                    // documented defaults explicitly to silence the warnings.
                    tp.trt_max_partition_iterations = 1000;
                    tp.trt_min_subgraph_size        = 1;
                    // Persist built engines next to the model so we don't pay the
                    // (minutes-long) TensorRT engine build on every launch.  TRT
                    // keys cache files by model + input shape + TRT/GPU version, so
                    // the first run of each batch size builds, then reuses.
                    namespace fs = std::filesystem;
                    fs::path model_dir = fs::path(path).parent_path();
                    if (model_dir.empty()) model_dir = ".";
                    static std::string cache_dir;   // must outlive the Append call below
                    cache_dir = (model_dir / "trt_engine_cache").string();
                    std::error_code ec;
                    fs::create_directories(cache_dir, ec);
                    tp.trt_engine_cache_enable = 1;
                    tp.trt_engine_cache_path   = cache_dir.c_str();
                    opts.AppendExecutionProvider_TensorRT(tp);
                    fprintf(stderr,
                        "[ORT] '%s': TensorRT EP (fp16=%d, engine cache='%s').\n"
                        "      First run builds engines (can take minutes); later runs reuse them.\n",
                        path.c_str(), tp.trt_fp16_enable, cache_dir.c_str());
#else
                    continue;   // compiled without TRT — should not be in ladder, but be safe
#endif
                }
                else if (ep == EP_CUDA)
                {
                    OrtCUDAProviderOptions cp{};
                    cp.device_id = device;
                    opts.AppendExecutionProvider_CUDA(cp);
                }
                // EP_CPU: append nothing — the default CPU EP runs.

                session = new Ort::Session(e, path.c_str(), opts);
                if (ep == EP_CPU && cuda)
                    fprintf(stderr, "[ORT] WARNING: '%s' running on CPU (GPU EPs unavailable)\n",
                            path.c_str());
                break;  // success
            }
            catch (const Ort::Exception& ex)
            {
                if (!last)
                {
                    fprintf(stderr,
                        "[ORT] %s EP failed for '%s' (%s)\n[ORT] Falling back to %s…\n",
                        ep_name(ep), path.c_str(), ex.what(), ep_name(ladder[a + 1]));
                    continue;   // try the next EP down the ladder
                }
                fprintf(stderr, "[ORT] load '%s' failed on %s: %s\n",
                        path.c_str(), ep_name(ep), ex.what());
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

    // ── Refined pose (see PLAN.md) — only loaded when cfg.refined_pose ───────
    OrtSession sess_decoder_handbox;   // decoder_handbox_fp32.onnx: pose_token+hand_box+hand_cls
    OrtSession sess_decoder_hand;      // decoder_hand.onnx: hand-crop decoder pass
    OrtSession sess_decoder_prompted;  // decoder_prompted.onnx: keypoint-prompted body pass 2
    CFFN mhr_ffn_hand, cam_ffn_hand;   // gguf mhr_proj_hand / cam_proj_hand

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
                                cfg.use_fp16, cfg.use_trt_ep))
            return false;
        printf("OK\n");

        printf("[FSB] Loading decoder  … ");
        fflush(stdout);
        if (!sess_decoder.load(ort_env, opath(cfg.decoder_name.c_str()), cuda, dev,
                               cfg.use_fp16, cfg.use_trt_ep))
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
                if (!sess_body.load(ort_env, bm_onnx, cuda, dev, false, cfg.use_trt_ep))
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

        // ── Refined pose (see PLAN.md) — extra decoder passes, off by default ──
        if (cfg.refined_pose)
        {
            printf("[FSB] Loading decoder_handbox (fp32) … ");
            fflush(stdout);
            // fp32 (not bf16): ORT's CUDA EP has a bf16-specific race condition
            // on this graph's 2-token hand-box slice+MLP (non-deterministic
            // garbage on repeat runs of the identical file) — see PLAN.md.
            if (!sess_decoder_handbox.load(ort_env, opath(cfg.decoder_handbox_name.c_str()),
                                           cuda, dev, /*fp16_io=*/false, /*trt_ep=*/false))
                return false;
            printf("OK\n");

            printf("[FSB] Loading decoder_hand … ");
            fflush(stdout);
            if (!sess_decoder_hand.load(ort_env, opath(cfg.decoder_hand_name.c_str()),
                                        cuda, dev, cfg.use_fp16, cfg.use_trt_ep))
                return false;
            printf("OK\n");

            printf("[FSB] Loading decoder_prompted … ");
            fflush(stdout);
            if (!sess_decoder_prompted.load(ort_env, opath(cfg.decoder_prompted_name.c_str()),
                                            cuda, dev, cfg.use_fp16, cfg.use_trt_ep))
                return false;
            printf("OK\n");
        }

        // YOLO – optional (might not exist for image-only usage)
        if (!cfg.yolo_path.empty())
        {
            printf("[FSB] Loading YOLO … ");
            fflush(stdout);
            if (!sess_yolo.load(ort_env, cfg.yolo_path, cuda, dev, false, cfg.use_trt_ep))
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

        if (cfg.refined_pose)
        {
            std::string refined_path = cfg.gguf_refined_path;
            if (refined_path.empty())
            {
                // Derive "onnx/pipeline.gguf" -> "onnx/pipeline_refined.gguf".
                const std::string suffix = ".gguf";
                refined_path = cfg.gguf_path;
                if (refined_path.size() >= suffix.size() &&
                    refined_path.compare(refined_path.size()-suffix.size(), suffix.size(), suffix) == 0)
                    refined_path = refined_path.substr(0, refined_path.size()-suffix.size()) + "_refined.gguf";
                else
                    refined_path += "_refined.gguf";
            }
            printf("[FSB] Loading %s … ", refined_path.c_str());
            fflush(stdout);
            if (!load_gguf_hand(refined_path)) return false;
            printf("OK\n");
        }

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

    // Loads mhr_proj_hand/cam_proj_hand from a SEPARATE gguf file (see PLAN.md,
    // issue #15 "refined pose" plan — kept out of pipeline.gguf so its
    // HuggingFace manifest entry never has to change). Required (unlike the
    // hand tensors used to be, optionally, inside load_gguf) — only called at
    // all when cfg.refined_pose is set, in which case they must be present.
    bool load_gguf_hand(const std::string& path)
    {
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

        FILE* fp = std::fopen(path.c_str(), "rb");
        if (!fp)
        {
            gguf_free(gctx);
            if (tmp_ctx) ggml_free(tmp_ctx);
            return false;
        }
        size_t data_base = gguf_get_data_offset(gctx);

        bool ok = cffn_load(mhr_ffn_hand, gctx, tmp_ctx, fp, data_base, "mhr_proj_hand")
                  && cffn_load(cam_ffn_hand, gctx, tmp_ctx, fp, data_base, "cam_proj_hand");

        std::fclose(fp);
        gguf_free(gctx);
        if (tmp_ctx) ggml_free(tmp_ctx);
        if (!ok)
        {
            fprintf(stderr, "[FSB] --refined-pose requested but %s has no mhr_proj_hand/"
                            "cam_proj_hand tensors (re-export with --refined — see PLAN.md)\n",
                    path.c_str());
            return false;
        }

        printf("[FSB] Hand FFNs: MHR(%dx%d->%d) Cam(%dx%d->%d)\n",
               mhr_ffn_hand.in_dim, mhr_ffn_hand.hid_dim, mhr_ffn_hand.out_dim,
               cam_ffn_hand.in_dim, cam_ffn_hand.hid_dim, cam_ffn_hand.out_dim);
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

        if (!cfg.external_boxes.empty())
        {
            // External boxes replace detection outright; they are already in
            // original-image pixels so no letterbox reversal is needed.
            for (const auto& b : cfg.external_boxes)
            {
                PersonDet d;
                d.x1 = b[0]; d.y1 = b[1]; d.x2 = b[2]; d.y2 = b[3];
                d.conf = 1.f;
                dets.push_back(d);
            }
        }
        else if (sess_yolo.session)
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

        // Fallback: full image as single detection.
        // Only when no detector ran at all — i.e. no YOLO model loaded and no
        // external boxes — which is the image-only "assume a single centred
        // person" usage.  When a detector *did* run and returned nothing, the
        // frame genuinely contains no person; feeding the whole image would
        // make the regressor hallucinate a body in the middle of the frame.
        if (dets.empty() && !sess_yolo.session && cfg.external_boxes.empty())
        {
            dets.push_back({ 0.f, 0.f, float(W), float(H), 1.f });
        }
        // Apply max_persons cap (sorted by confidence from NMS)
        if (cfg.max_persons > 0 && (int)dets.size() > cfg.max_persons)
            dets.resize(cfg.max_persons);
        double dt_detect = ms(t0);
        timers.detection += dt_detect;
        printf("[FSB] detection: %.1f ms  persons: %zu\n", dt_detect, dets.size());

        // Nothing detected – no crops to regress.  Returning early also keeps
        // the ONNX sessions from being run with a zero-sized batch.
        if (dets.empty())
        {
            timers.frames += 1;
            return {};
        }

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

        // ── Refined pose: hand-box regression + hand-crop decoder passes ───────
        // (see PLAN.md, issue #15 "refined pose" plan). Off by default; adds a
        // second decoder pass (fp32, hand-box regression) plus up to 2×B more
        // decoder_hand.onnx forward passes (one per visible hand per person).
        // The validity gate / pass-2 keypoint-prompted decoder / wrist-IK
        // fusion that turn this into a corrected final body_pose run further
        // below, after the per-person results (incl. keypoints_2d) exist.
        std::vector<std::array<float,8>> hand_box_out(B);
        std::vector<std::array<float,4>> hand_cls_out(B);
        for (auto& a : hand_box_out) a.fill(0.f);
        for (auto& a : hand_cls_out) a.fill(0.f);
        // Hoisted to function scope: used again after the results-assembly loop
        // below (gate / pass-2 keypoint prompt / wrist-IK / splice).
        struct HandCropRef {
            int person; bool is_left;
            float orig_cx, orig_cy, orig_sz;   // hand crop geometry, original-image space
        };
        std::vector<HandCropRef> hand_refs;
        std::vector<float> hand_mhr_raw, hand_cam_raw;
        if (cfg.refined_pose && sess_decoder_handbox.session)
        {
            t0 = Clock::now();
            Ort::Value hb_feat_t = Ort::Value::CreateTensor<float>(
                                       mi, features.data(), features.size(), feat_shape.data(), 4);
            Ort::Value hb_cond_t = Ort::Value::CreateTensor<float>(
                                       mi, batch_cond.data(), batch_cond.size(), cond_shape.data(), 2);
            Ort::Value hb_ray_t  = Ort::Value::CreateTensor<float>(
                                       mi, batch_ray.data(), batch_ray.size(), ray_shape.data(), 4);
            std::vector<Ort::Value> hb_inputs;
            hb_inputs.push_back(std::move(hb_feat_t));
            hb_inputs.push_back(std::move(hb_cond_t));
            hb_inputs.push_back(std::move(hb_ray_t));

            auto hb_out = sess_decoder_handbox.session->Run(
                              Ort::RunOptions{nullptr},
                              sess_decoder_handbox.input_names.data(),  hb_inputs.data(), hb_inputs.size(),
                              sess_decoder_handbox.output_names.data(), sess_decoder_handbox.output_names.size());
            // outputs: [pose_token (unused — bf16 decoder's is used instead), hand_box, hand_cls]
            if (hb_out.size() >= 3)
            {
                const float* hb = hb_out[1].GetTensorData<float>();   // [B,2,4]
                const float* hc = hb_out[2].GetTensorData<float>();   // [B,2,2]
                for (int i = 0; i < B; ++i)
                {
                    std::memcpy(hand_box_out[i].data(), hb + (size_t)i * 8, 8 * sizeof(float));
                    std::memcpy(hand_cls_out[i].data(), hc + (size_t)i * 4, 4 * sizeof(float));
                }
            }
            printf("[FSB] handbox:    %.1f ms\n", ms(t0));

            // ── Build left/right hand crops from the regressed boxes ───────────
            // hand_box layout per person: [left_cx,cy,w,h, right_cx,cy,w,h],
            // normalised to the 512x512 body crop [0,1]. Convert to original-
            // image pixel center/size via the same affine used by compute_ray_cond
            // (crop = scale*(orig-bbox_c) + CROP_SIZE/2), then to a square xyxy box.
            // Left hand: Python flips the WHOLE source image before cropping so the
            // network always sees a "right-looking" hand; we instead crop the
            // unflipped box and flip the resulting tensor — equivalent (flip
            // commutes with a symmetric crop+pad) and cheaper. The geometry fed
            // to ray_cond/cond_info is mirrored around the full image width to
            // stay consistent with the flipped pixel content.
            std::vector<float> hbatch_crops, hbatch_cond, hbatch_ray;
            hbatch_crops.reserve((size_t)2 * B * 3 * plane);
            hbatch_cond.reserve((size_t)2 * B * 3);
            hbatch_ray.reserve((size_t)2 * B * 2 * ray_plane);

            for (int i = 0; i < B; ++i)
            {
                const float scale_i = float(CROP_SIZE) / crop_sz_v[i];
                for (int h = 0; h < 2; ++h)   // 0 = left, 1 = right
                {
                    const bool is_left = (h == 0);
                    const float* hb = hand_box_out[i].data() + h * 4;
                    float box_cx = hb[0] * CROP_SIZE, box_cy = hb[1] * CROP_SIZE;
                    float box_w  = hb[2] * CROP_SIZE, box_h  = hb[3] * CROP_SIZE;
                    float box_sz = std::max(box_w, box_h);

                    // crop-space → original-image-space (inverse of compute_ray_cond's
                    // forward affine; see fixed_aspect_bbox_size/compute_ray_cond above)
                    float orig_cx = (box_cx - CROP_SIZE * 0.5f) / scale_i + crop_cx_v[i];
                    float orig_cy = (box_cy - CROP_SIZE * 0.5f) / scale_i + crop_cy_v[i];
                    float orig_sz = box_sz / scale_i;

                    float hx1 = orig_cx - orig_sz * 0.5f, hx2 = orig_cx + orig_sz * 0.5f;
                    float hy1 = orig_cy - orig_sz * 0.5f, hy2 = orig_cy + orig_sz * 0.5f;

                    float* crop_ptr = nullptr;
                    hbatch_crops.resize(hbatch_crops.size() + 3 * plane);
                    crop_ptr = hbatch_crops.data() + hbatch_crops.size() - 3 * plane;
                    float hcx, hcy, hcsz;
                    crop_and_normalise(bgr, hx1, hy1, hx2, hy2, crop_ptr, hcx, hcy, hcsz,
                                       HAND_BBOX_SCALE_FACTOR);

                    float geom_cx = hcx, geom_cam_cx = cx;
                    if (is_left)
                    {
                        // Flip the crop tensor horizontally (per-row mirror, CHW layout)
                        for (int c = 0; c < 3; ++c)
                        {
                            float* plane_ptr = crop_ptr + c * plane;
                            for (int y = 0; y < CROP_SIZE; ++y)
                            {
                                float* row = plane_ptr + y * CROP_SIZE;
                                std::reverse(row, row + CROP_SIZE);
                            }
                        }
                        // Mirror the geometry around the FULL source image width so
                        // ray_cond/cond_info stay consistent with the flipped pixels.
                        geom_cx     = float(W) - hcx;
                        geom_cam_cx = float(W) - cx;
                    }

                    hbatch_cond.resize(hbatch_cond.size() + 3);
                    compute_condition_info(geom_cx, hcy, hcsz, fx, fy, geom_cam_cx, cy,
                                           hbatch_cond.data() + hbatch_cond.size() - 3);

                    hbatch_ray.resize(hbatch_ray.size() + 2 * ray_plane);
                    compute_ray_cond(geom_cx, hcy, hcsz, fx, fy, geom_cam_cx, cy,
                                     hbatch_ray.data() + hbatch_ray.size() - 2 * ray_plane);

                    hand_refs.push_back({i, is_left, hcx, hcy, hcsz});
                }
            }

            // ── decoder_hand.onnx: run all 2×B hand crops in one batch ─────────
            t0 = Clock::now();
            const int HB = (int)hand_refs.size();
            std::vector<int64_t> himg_shape{HB, 3, CROP_SIZE, CROP_SIZE};
            std::vector<int64_t> hcond_shape{HB, 3};
            std::vector<int64_t> hray_shape {HB, 2, FEAT_HW, FEAT_HW};

            Ort::Value hfeat_in_t = Ort::Value::CreateTensor<float>(
                                        mi, hbatch_crops.data(), hbatch_crops.size(), himg_shape.data(), 4);
            auto hand_backbone_out = sess_backbone.session->Run(
                                         Ort::RunOptions{nullptr},
                                         sess_backbone.input_names.data(),  &hfeat_in_t, 1,
                                         sess_backbone.output_names.data(), 1);
            const float* hfeat_ptr = hand_backbone_out[0].GetTensorData<float>();
            size_t hfeat_elems = (size_t)HB * BACKBONE_DIM * FEAT_HW * FEAT_HW;
            std::vector<float> hand_features(hfeat_ptr, hfeat_ptr + hfeat_elems);

            std::vector<int64_t> hfeat_shape{HB, BACKBONE_DIM, FEAT_HW, FEAT_HW};
            Ort::Value hd_feat_t = Ort::Value::CreateTensor<float>(
                                       mi, hand_features.data(), hand_features.size(), hfeat_shape.data(), 4);
            Ort::Value hd_cond_t = Ort::Value::CreateTensor<float>(
                                       mi, hbatch_cond.data(), hbatch_cond.size(), hcond_shape.data(), 2);
            Ort::Value hd_ray_t  = Ort::Value::CreateTensor<float>(
                                       mi, hbatch_ray.data(), hbatch_ray.size(), hray_shape.data(), 4);
            std::vector<Ort::Value> hd_inputs;
            hd_inputs.push_back(std::move(hd_feat_t));
            hd_inputs.push_back(std::move(hd_cond_t));
            hd_inputs.push_back(std::move(hd_ray_t));

            auto hd_out = sess_decoder_hand.session->Run(
                              Ort::RunOptions{nullptr},
                              sess_decoder_hand.input_names.data(),  hd_inputs.data(), hd_inputs.size(),
                              sess_decoder_hand.output_names.data(), 1);
            const float* hd_token_ptr = hd_out[0].GetTensorData<float>();
            std::vector<float> hand_pose_tokens(hd_token_ptr, hd_token_ptr + (size_t)HB * DECODER_DIM);

            hand_mhr_raw = cffn_run(mhr_ffn_hand, hand_pose_tokens.data(), HB);
            hand_cam_raw = cffn_run(cam_ffn_hand, hand_pose_tokens.data(), HB);
            printf("[FSB] hand decoder+ffn: %.1f ms  (%d hand crop(s))\n", ms(t0), HB);

            // hand_mhr_raw/hand_cam_raw/hand_refs are used below (after the
            // per-person results are assembled) for the validity gate, pass-2
            // keypoint prompt, and wrist-IK fusion. Report per-hand sanity now.
            for (int h = 0; h < HB; ++h)
            {
                const auto& ref = hand_refs[h];
                const float* raw = hand_mhr_raw.data() + (size_t)h * mhr_ffn_hand.out_dim;
                float max_abs = 0.f;
                for (int k = 0; k < mhr_ffn_hand.out_dim; ++k) max_abs = std::max(max_abs, std::fabs(raw[k]));
                printf("[FSB]   hand[%d] person=%d %s  max|raw|=%.3f  box=[%.3f %.3f %.3f %.3f]\n",
                       h, ref.person, ref.is_left ? "left " : "right",
                       max_abs,
                       hand_box_out[ref.person][(ref.is_left?0:4)+0],
                       hand_box_out[ref.person][(ref.is_left?0:4)+1],
                       hand_box_out[ref.person][(ref.is_left?0:4)+2],
                       hand_box_out[ref.person][(ref.is_left?0:4)+3]);
            }
        }

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
                                    joints_out,
                                    nullptr);
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
        // Pass-1's own wrist Euler [right(41,43,42), left(31,33,32)], captured
        // before any refined-pose splice — this is Python's `ori_local_wrist_rotmat`
        // (see run_inference), used as the reference pose for the rotation-agreement
        // gate below (PLAN.md step 7 TODO: this criterion is now implemented).
        std::vector<std::array<float,6>> pass1_wrist_euler(B);

        for (int i = 0; i < B; ++i)
        {
            MHRResult& r   = results[i];
            const auto& d  = dets[i];
            const float* p = mhr_raw.data() + i * NPOSE;

            r.bbox = { d.x1, d.y1, d.x2, d.y2 };

            if (cfg.refined_pose && !hand_box_out.empty())
            {
                r.has_hand_box = true;
                r.hand_box     = hand_box_out[i];
                r.hand_box_cls = hand_cls_out[i];
            }

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
                float bs      = fixed_aspect_bbox_size(bw, bh) * s_val + 1e-8f;
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
            pass1_wrist_euler[i] = { be[41], be[43], be[42], be[31], be[33], be[32] };

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

                    // Keep the full 127-joint skeleton (incl. root / c_spine0..3)
                    // for consumers that need joints absent from the 70 keypoints.
                    r.skeleton_3d = joint_coords;

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

        // ── Refined pose: validity gate + pass 2 + wrist-IK fusion + splice ────
        // (see PLAN.md, issue #15 "refined pose" plan). Mirrors sam3d_body.py's
        // run_inference Steps 3-5 ("replace hand pose estimation from the body
        // decoder" / "Doing IK"), with two documented simplifications:
        //   - the validity gate uses box-size + 2D-wrist-distance only (skips
        //     Python's rotation-agreement and full-70-keypoint-in-crop criteria)
        //   - kept everything else faithful, including the closed-form wrist-IK
        //     rotation solve (this is what the LBS out_joint_quats extension —
        //     see model_loader_transform_joints.c — was added for).
        if (cfg.refined_pose && sess_decoder_prompted.session && lbs_data && !kp_mapping.empty())
        {
            t0 = Clock::now();
            static constexpr int KP_RIGHT_WRIST = 41, KP_LEFT_WRIST = 62;
            static constexpr int KP_RIGHT_ELBOW = 8,  KP_LEFT_ELBOW = 7;
            static constexpr float HAND_BOX_SIZE_THRESH   = 64.f;    // px, original image
            // Python's own threshold is 0.25 (run_inference's hand_wrist_kps2d_thresh),
            // tuned for its native bf16 eager inference. This C++ port necessarily runs
            // the hand-box regression in fp32 (decoder_handbox_fp32.onnx — bf16+CUDA has
            // a confirmed ORT race condition on this op, see PLAN.md), which is NOT free:
            // hand-verified that Python's OWN fp32-eager computation (same crop, zero ONNX
            // involved) already diverges from its OWN bf16 production output by ~0.03-0.09
            // in normalised crop space — comparable to or larger than the gap measured
            // between this port and Python's bf16 output. So some of what this threshold
            // is rejecting is an unavoidable cost of the bf16-CUDA-race workaround, not
            // estimation error — relaxed to compensate. Calibrated against a small number
            // of real test images (not systematically tuned); revisit if it lets through
            // hands that are genuinely a bad match.
            static constexpr float HAND_WRIST_DIST_THRESH = 0.50f;   // normalised by hand crop size

            // ── per-hand-crop own FK: wrist 2D (full-image, unflipped) + wrist quat ──
            struct HandFK {
                bool ok = false;
                std::array<float,133> body_euler{};
                std::array<float,3>   global_rot_euler{};
                std::array<float,108> hand108{};
                std::array<float,28>  scale28{};
                std::array<float,45>  shape45{};
                std::array<float,2>   wrist2d{};      // full-image px, unflipped
                std::array<float,4>   wrist_quat{};   // XYZW, unflipped model space
                bool valid = false;
            };
            const int HB = (int)hand_refs.size();
            std::vector<HandFK> hfk(HB);

            std::vector<float> hv_scratch((size_t)lbs_data->n_verts * 3);
            std::vector<float> hj_scratch((size_t)lbs_data->n_joints * 3);
            std::vector<float> hq_scratch((size_t)lbs_data->n_joints * 4);
            static const float zero_face72[72] = {};

            for (int h = 0; h < HB; ++h)
            {
                const auto& ref = hand_refs[h];
                HandFK& F = hfk[h];
                const float* raw  = hand_mhr_raw.data() + (size_t)h * mhr_ffn_hand.out_dim;
                const float* camr = hand_cam_raw.data() + (size_t)h * 3;

                rot6d_to_euler(raw, F.global_rot_euler.data());
                compact_cont_to_body_params(raw + 6, F.body_euler.data());
                std::copy(raw + 339, raw + 339 + 108, F.hand108.begin());
                std::copy(raw + 311, raw + 311 + 28,  F.scale28.begin());
                std::copy(raw + 266, raw + 266 + 45,  F.shape45.begin());

                // head_camera_hand uses DEFAULT_SCALE_FACTOR_HAND=10 (model_config.yaml)
                // in Python's perspective_projection: bs = bbox_size*s*default_scale_factor
                // (camera_head.py:85). Hand-verified against real captured Python values
                // (sys.settrace on camera_project_hand, forced bbox) — this formula with
                // *10 reproduces Python's pred_cam_t to ~0.05 (small remaining gap traced
                // to the hand-box regression's own crop-normalised cx/cy differing by a
                // few percent from Python's, amplified ~3x by the body-crop zoom factor —
                // see PLAN.md). An earlier attempt at *10 without HAND_CAM_SCALE_FACTOR
                // named/isolated like this produced wildly wrong (off-screen) results;
                // re-verified step-by-step via the camdbg printf below this time.
                static constexpr float HAND_CAM_SCALE_FACTOR = 10.f;
                float geom_cx     = ref.is_left ? (float(W) - ref.orig_cx) : ref.orig_cx;
                float geom_cam_cx = ref.is_left ? (float(W) - cx)          : cx;
                float s_val = -camr[0], t_x = camr[1], t_y = -camr[2];
                float bs    = ref.orig_sz * s_val * HAND_CAM_SCALE_FACTOR + 1e-8f;
                float pred_cam_t[3] = {
                    t_x + 2.f*(geom_cx - geom_cam_cx)/bs,
                    t_y + 2.f*(ref.orig_cy - cy)/bs,
                    2.f*fx/bs
                };
                printf("[FSB]   camdbg h=%d %s: raw_cam=(%.6f,%.6f,%.6f) orig_cx=%.3f orig_cy=%.3f "
                       "orig_sz=%.3f fx=%.3f geom_cam_cx=%.3f cy=%.3f s_val=%.6f bs=%.4f "
                       "pred_cam_t=(%.4f,%.4f,%.4f)\n",
                       h, ref.is_left?"left":"right", camr[0], camr[1], camr[2],
                       ref.orig_cx, ref.orig_cy, ref.orig_sz, fx, geom_cam_cx, cy, s_val, bs,
                       pred_cam_t[0], pred_cam_t[1], pred_cam_t[2]);

                ModelParams204 mp = build_model_params(F.global_rot_euler.data(), F.body_euler.data(), nullptr, true);
                apply_hand_pose(mp.data, F.hand108.data(),
                                lbs_data->hand_pose_mean, lbs_data->hand_pose_comps,
                                lbs_data->hand_joint_idxs_left, lbs_data->hand_joint_idxs_right);
                if (lbs_data->scale_mean && lbs_data->scale_comps)
                {
                    int ns = lbs_data->n_scale_out, npc = lbs_data->n_scale_pc;
                    for (int j = 0; j < ns; ++j) mp.data[136+j] = lbs_data->scale_mean[j];
                    for (int k = 0; k < npc; ++k)
                        for (int j = 0; j < ns; ++j)
                            mp.data[136+j] += F.scale28[k] * lbs_data->scale_comps[k*ns+j];
                }

                // Wrist-centric → body-rooted transform (mhr_head.py's enable_hand_model
                // branch, only used by head_pose_hand) — see preprocess.hpp's
                // mp_rot_to_mat3 doc comment / PLAN.md for the derivation+verification.
                // mp.data[3:6] is ALREADY in the (rz,ry,rx) order this needs (build_model_params
                // put it there); global_trans_ori is always 0 (single-view inference).
                {
                    float R_ori[9]; mp_rot_to_mat3(mp.data + 3, R_ori);
                    float R_new[9]; mat3_mul(R_ori, HAND_LOCAL_TO_WORLD_WRIST, R_new);
                    float mp_rot_new[3]; mat3_to_mp_rot(R_new, mp_rot_new);

                    float diff[3] = {
                        HAND_RIGHT_WRIST_COORDS[0] - HAND_ROOT_COORDS[0],
                        HAND_RIGHT_WRIST_COORDS[1] - HAND_ROOT_COORDS[1],
                        HAND_RIGHT_WRIST_COORDS[2] - HAND_ROOT_COORDS[2]
                    };
                    float rotated[3]; mat3_vec3(R_new, diff, rotated);
                    mp.data[0] = -(rotated[0] + HAND_ROOT_COORDS[0]) * 10.f;
                    mp.data[1] = -(rotated[1] + HAND_ROOT_COORDS[1]) * 10.f;
                    mp.data[2] = -(rotated[2] + HAND_ROOT_COORDS[2]) * 10.f;
                    mp.data[3] = mp_rot_new[0];
                    mp.data[4] = mp_rot_new[1];
                    mp.data[5] = mp_rot_new[2];
                    for (int idx : HAND_NONHAND_PARAM_IDXS) mp.data[idx] = 0.f;
                }

                if (!mhr_lbs_compute(lbs_data, mp.data, F.shape45.data(), zero_face72,
                                     hv_scratch.data(), hj_scratch.data(), hq_scratch.data()))
                    continue;

                // Always joint 42 (r_wrist) — the hand-crop's own skeleton is always
                // computed in a "this is a right hand" frame regardless of ref.is_left
                // (same reason the keypoint lookup elsewhere always uses KP_RIGHT_WRIST).
                const int wrist_joint = 42;
                std::copy(hq_scratch.begin() + wrist_joint*4, hq_scratch.begin() + wrist_joint*4 + 4,
                          F.wrist_quat.begin());
                {
                    float Rdbg[9]; quat_to_mat3(F.wrist_quat.data(), Rdbg);
                    printf("[FSB]   wristquatdbg h=%d %s: R=[[%.4f %.4f %.4f] [%.4f %.4f %.4f] [%.4f %.4f %.4f]]\n",
                           h, ref.is_left?"left":"right",
                           Rdbg[0],Rdbg[1],Rdbg[2],Rdbg[3],Rdbg[4],Rdbg[5],Rdbg[6],Rdbg[7],Rdbg[8]);
                }

                // A hand crop's OWN predicted keypoints always report the wrist at
                // kps_right_wrist_idx (41), regardless of which hand it is — every
                // crop (right natural, left flipped-to-look-right) is normalised to
                // look right-handed before being fed to the network, and the network
                // has no notion of "this is actually the left hand" to report
                // differently. Confirmed against sam3d_body.py run_inference lines
                // ~1350-1354: both right_kps_full and left_kps_full index the SAME
                // kps_right_wrist_idx into their respective hand crop's own output;
                // only the BODY PASS's own (genuinely anatomical) keypoints use 41
                // vs 62 by side. Using KP_LEFT_WRIST here for the left crop (an
                // earlier version of this code did) was a real bug.
                //
                // The wrist keypoint's 3D position in the hand crop's own decode is
                // (empirically, essentially exact float32 zero — confirmed by hooking
                // camera_project_hand and printing pred_keypoints_3d[:,41] on a real
                // image) the ORIGIN: head_pose_hand's MHR forward is wrist-rooted for
                // this keypoint, unlike the body-rooted (pelvis) skeleton mhr_lbs_data
                // here represents. So skip the kp_mapping/LBS position lookup entirely
                // for this specific point and project the origin directly through
                // pred_cam_t — this is NOT an approximation, it matches Python's real
                // output to <1px (hand-verified against captured ground truth). Using
                // mhr_lbs_compute's body-rooted skeleton here (an earlier version of
                // this code did) was a real bug — see PLAN.md, this also means the
                // wrist_quat used below for the IK fusion may have the same rooting
                // mismatch and needs the same scrutiny (flagged, not yet fixed).
                float dz = pred_cam_t[2], dx = pred_cam_t[0], dy = pred_cam_t[1];
                if (dz < 1e-4f) dz = 1e-4f;
                float wx = dx/dz*fx + geom_cam_cx;
                float wy = dy/dz*fy + cy;
                if (ref.is_left) wx = float(W) - wx - 1.f;   // unflip back to normal image space
                F.wrist2d = {wx, wy};
                F.ok = true;

                int body_wrist_kp = ref.is_left ? KP_LEFT_WRIST : KP_RIGHT_WRIST;
                bool valid_box = ref.orig_sz > HAND_BOX_SIZE_THRESH;
                bool valid_dist = false;
                float dbg_dist = -1.f, dbg_bodyx = -1.f, dbg_bodyy = -1.f;
                if (results[ref.person].keypoints_2d.size() >= (size_t)(body_wrist_kp+1)*2)
                {
                    // Python normalises each hand's distance by the OTHER hand's own
                    // bbox_scale (run_inference lines ~1363-1368: right_kps_dist uses
                    // batch_lhand's scale, left_kps_dist uses batch_rhand's scale) —
                    // faithfully replicated here, not "fixed", even though it reads
                    // like it could be an upstream quirk. Falls back to this hand's
                    // own orig_sz if the sibling hand crop wasn't built (e.g. only
                    // one hand's box passed the earlier size check upstream — doesn't
                    // currently happen since both hands are always built, but keep
                    // the fallback for robustness).
                    float norm_sz = ref.orig_sz;
                    for (const auto& sib : hand_refs)
                        if (sib.person == ref.person && sib.is_left != ref.is_left) { norm_sz = sib.orig_sz; break; }

                    const float* body_wrist2d = &results[ref.person].keypoints_2d[body_wrist_kp*2];
                    float ddx = wx - body_wrist2d[0], ddy = wy - body_wrist2d[1];
                    float dist = std::sqrt(ddx*ddx + ddy*ddy) / std::max(1.f, norm_sz);
                    valid_dist = dist < HAND_WRIST_DIST_THRESH;
                    dbg_dist = dist; dbg_bodyx = body_wrist2d[0]; dbg_bodyy = body_wrist2d[1];
                }
                F.valid = valid_box && valid_dist;
                // Dev escape hatch for gate-threshold tuning/debugging without a
                // rebuild — bypasses only the distance check, box-size still applies.
                if (getenv("FSB_FORCE_HAND_VALID")) F.valid = valid_box;
                printf("[FSB]   gate-debug h=%d person=%d %s: orig_sz=%.1f valid_box=%d "
                       "hand_wrist2d=(%.1f,%.1f) body_wrist2d=(%.1f,%.1f) dist_norm=%.3f valid_dist=%d\n",
                       h, ref.person, ref.is_left?"left":"right", ref.orig_sz, valid_box,
                       wx, wy, dbg_bodyx, dbg_bodyy, dbg_dist, valid_dist);
            }
            printf("[FSB] hand FK + gate: %.1f ms  (%d/%d hand(s) valid)\n", ms(t0),
                   (int)std::count_if(hfk.begin(), hfk.end(), [](const HandFK& f){ return f.valid; }), HB);

            // ── per-person: keypoint prompt → decoder_prompted → decode → splice ──
            t0 = Clock::now();
            for (int i = 0; i < B; ++i)
            {
                MHRResult& r = results[i];
                // DIAGNOSTIC: snapshot pass-1's result so it can be restored below,
                // to isolate whether pass-2's unconditional replace (independent of
                // any hand-crop splice) is itself the source of visible distortion.
                MHRResult r_pass1_backup;
                bool skip_pass2 = getenv("FSB_SKIP_PASS2") != nullptr;
                if (skip_pass2) r_pass1_backup = r;
                int left_h = -1, right_h = -1;
                for (int h = 0; h < HB; ++h)
                    if (hand_refs[h].person == i) (hand_refs[h].is_left ? left_h : right_h) = h;

                // keypoint_prompt[4,3]: right_wrist, left_wrist, right_elbow, left_elbow.
                // Coords are crop-normalised to the BODY pass's own crop [-0.5,0.5]
                // then shifted to [0,1]; label==-2 marks an invalid/unused slot.
                float kp_prompt[4][3];
                auto set_prompt = [&](int slot, float full_x, float full_y, float label, bool valid)
                {
                    if (!valid) { kp_prompt[slot][0]=0.f; kp_prompt[slot][1]=0.f; kp_prompt[slot][2]=-2.f; return; }
                    float nx = (full_x - crop_cx_v[i]) / crop_sz_v[i];
                    float ny = (full_y - crop_cy_v[i]) / crop_sz_v[i];
                    bool in_range = nx>=-0.5f && nx<=0.5f && ny>=-0.5f && ny<=0.5f;
                    if (!in_range) { kp_prompt[slot][0]=0.f; kp_prompt[slot][1]=0.f; kp_prompt[slot][2]=-2.f; return; }
                    kp_prompt[slot][0] = nx + 0.5f;
                    kp_prompt[slot][1] = ny + 0.5f;
                    kp_prompt[slot][2] = label;
                };
                bool right_valid = right_h >= 0 && hfk[right_h].valid;
                bool left_valid  = left_h  >= 0 && hfk[left_h].valid;
                set_prompt(0, right_valid ? hfk[right_h].wrist2d[0] : 0.f,
                             right_valid ? hfk[right_h].wrist2d[1] : 0.f,
                             (float)KP_RIGHT_WRIST, right_valid);
                set_prompt(1, left_valid ? hfk[left_h].wrist2d[0] : 0.f,
                             left_valid ? hfk[left_h].wrist2d[1] : 0.f,
                             (float)KP_LEFT_WRIST, left_valid);
                bool have_kp2d = r.keypoints_2d.size() >= 70*2;
                set_prompt(2, have_kp2d ? r.keypoints_2d[KP_RIGHT_ELBOW*2+0] : 0.f,
                             have_kp2d ? r.keypoints_2d[KP_RIGHT_ELBOW*2+1] : 0.f,
                             (float)KP_RIGHT_ELBOW, right_valid && have_kp2d);
                set_prompt(3, have_kp2d ? r.keypoints_2d[KP_LEFT_ELBOW*2+0] : 0.f,
                             have_kp2d ? r.keypoints_2d[KP_LEFT_ELBOW*2+1] : 0.f,
                             (float)KP_LEFT_ELBOW, left_valid && have_kp2d);

                // prev_estimate[522] = cat(pred_pose_raw[266], shape[45], scale[28], hand[108], face[72], pred_cam_raw[3])
                float prev_est[522];
                float* pe = prev_est;
                std::memcpy(pe, r.pred_pose_raw.data(), 266*sizeof(float)); pe += 266;
                std::memcpy(pe, r.shape.data(),      45*sizeof(float));     pe += 45;
                std::memcpy(pe, r.scale.data(),      28*sizeof(float));     pe += 28;
                std::memcpy(pe, r.hand_pose.data(), 108*sizeof(float));     pe += 108;
                std::memcpy(pe, r.face_params.data(),72*sizeof(float));     pe += 72;
                std::memcpy(pe, r.pred_cam_raw.data(),3*sizeof(float));

                std::vector<int64_t> f_sh{1, BACKBONE_DIM, FEAT_HW, FEAT_HW};
                std::vector<int64_t> c_sh{1, 3};
                std::vector<int64_t> r_sh{1, 2, FEAT_HW, FEAT_HW};
                std::vector<int64_t> k_sh{1, 4, 3};
                std::vector<int64_t> p_sh{1, 1, 522};

                Ort::Value pf_t = Ort::Value::CreateTensor<float>(
                                      mi, features.data() + (size_t)i*BACKBONE_DIM*FEAT_HW*FEAT_HW,
                                      (size_t)BACKBONE_DIM*FEAT_HW*FEAT_HW, f_sh.data(), 4);
                Ort::Value pc_t = Ort::Value::CreateTensor<float>(
                                      mi, batch_cond.data() + (size_t)i*3, 3, c_sh.data(), 2);
                Ort::Value pr_t = Ort::Value::CreateTensor<float>(
                                      mi, batch_ray.data() + (size_t)i*2*ray_plane, 2*ray_plane, r_sh.data(), 4);
                Ort::Value pk_t = Ort::Value::CreateTensor<float>(mi, &kp_prompt[0][0], 12, k_sh.data(), 3);
                Ort::Value pp_t = Ort::Value::CreateTensor<float>(mi, prev_est, 522, p_sh.data(), 3);

                std::vector<Ort::Value> p2_inputs;
                p2_inputs.push_back(std::move(pf_t));
                p2_inputs.push_back(std::move(pc_t));
                p2_inputs.push_back(std::move(pr_t));
                p2_inputs.push_back(std::move(pk_t));
                p2_inputs.push_back(std::move(pp_t));

                auto p2_out = sess_decoder_prompted.session->Run(
                                  Ort::RunOptions{nullptr},
                                  sess_decoder_prompted.input_names.data(),  p2_inputs.data(), p2_inputs.size(),
                                  sess_decoder_prompted.output_names.data(), 1);
                const float* p2_token = p2_out[0].GetTensorData<float>();

                std::vector<float> p2_mhr = cffn_run(mhr_ffn, p2_token, 1);
                std::vector<float> p2_cam = cffn_run(cam_ffn, p2_token, 1);
                const float* p2 = p2_mhr.data();

                // Pass 2's decode REPLACES the pass-1 output wholesale (matches Python:
                // output.update({"mhr": pose_output}) is unconditional — only the
                // wrist/hand/scale/shape splice below is gated by per-hand validity).
                std::memcpy(r.pred_pose_raw.data(), p2, 266*sizeof(float));
                std::memcpy(r.pred_cam_raw.data(),  p2_cam.data(), 3*sizeof(float));
                float p2_global_rot_euler[3];
                rot6d_to_euler(p2, p2_global_rot_euler);
                printf("[FSB]   pass1v2dbg person=%d pass1_global_rot=(%.3f,%.3f,%.3f) "
                       "pass2_global_rot=(%.3f,%.3f,%.3f) pass1_cam_t=(%.3f,%.3f,%.3f) "
                       "pass2_cam=(s=%.3f,tx=%.3f,ty=%.3f)\n",
                       i, r.global_rot.size()>2?r.global_rot[0]:0.f, r.global_rot.size()>1?r.global_rot[1]:0.f,
                       r.global_rot.size()>0?r.global_rot[2]:0.f,
                       p2_global_rot_euler[2], p2_global_rot_euler[1], p2_global_rot_euler[0],
                       r.pred_cam_t[0], r.pred_cam_t[1], r.pred_cam_t[2],
                       p2_cam[0], p2_cam[1], p2_cam[2]);
                r.global_rot = { p2_global_rot_euler[2], p2_global_rot_euler[1], p2_global_rot_euler[0] };
                std::array<float,133> p2_body_euler{};
                compact_cont_to_body_params(p2 + 6, p2_body_euler.data());
                if (const char* dump_path = getenv("FSB_DUMP_P2_BODY_EULER"))
                {
                    FILE* fp = fopen(dump_path, "w");
                    if (fp)
                    {
                        for (float v : p2_body_euler) fprintf(fp, "%.8f\n", v);
                        fclose(fp);
                    }
                }
                r.shape.assign(p2 + 266, p2 + 266 + 45);
                r.scale.assign(p2 + 311, p2 + 311 + 28);
                r.hand_pose.assign(p2 + 339, p2 + 339 + 108);
                r.face_params.assign(p2 + 447, p2 + 447 + 72);
                {
                    std::array<float,3> pass1_cam_t = r.pred_cam_t;
                    float s_val = -p2_cam[0], t_x = p2_cam[1], t_y = -p2_cam[2];
                    float bw = r.bbox[2]-r.bbox[0], bh = r.bbox[3]-r.bbox[1];
                    float bbox_cx = (r.bbox[0]+r.bbox[2])*0.5f, bbox_cy = (r.bbox[1]+r.bbox[3])*0.5f;
                    float bs = fixed_aspect_bbox_size(bw,bh)*s_val + 1e-8f;
                    r.pred_cam_t = { t_x + 2.f*(bbox_cx-cx)/bs, t_y + 2.f*(bbox_cy-cy)/bs, 2.f*fx/bs };
                    printf("[FSB]   camv2dbg person=%d pass1_cam_t=(%.3f,%.3f,%.3f) pass2_cam_t=(%.3f,%.3f,%.3f)\n",
                           i, pass1_cam_t[0], pass1_cam_t[1], pass1_cam_t[2],
                           r.pred_cam_t[0], r.pred_cam_t[1], r.pred_cam_t[2]);
                }

                // ── wrist-IK fusion (only for hands that passed the gate) ──────────
                if (right_valid || left_valid)
                {
                    ModelParams204 mp2 = build_model_params(p2_global_rot_euler, p2_body_euler.data(), nullptr, true);
                    // hand/scale not needed for FK (arms only), zero_face/zero-shape are fine —
                    // only joint rotations are used, not vertices.
                    std::vector<float> zshape((size_t)lbs_data->n_shape_pc, 0.f);
                    std::vector<float> v2((size_t)lbs_data->n_verts*3), j2((size_t)lbs_data->n_joints*3),
                                       q2((size_t)lbs_data->n_joints*4);
                    if (mhr_lbs_compute(lbs_data, mp2.data, zshape.data(), zero_face72,
                                        v2.data(), j2.data(), q2.data()))
                    {
                        for (int lr = 0; lr < 2; ++lr)   // 0=right, 1=left
                        {
                            bool valid  = lr==0 ? right_valid : left_valid;
                            int  h      = lr==0 ? right_h     : left_h;
                            if (!valid) continue;

                            int lowarm_j     = lr==0 ? 40 : 76;
                            int wristtwist_j = lr==0 ? 41 : 77;
                            float lowarm_R[9]; quat_to_mat3(q2.data() + lowarm_j*4, lowarm_R);
                            float pre_R[9];    quat_to_mat3(lbs_data->joint_prerotations + wristtwist_j*4, pre_R);
                            float zero_rot_R[9]; mat3_mul(lowarm_R, pre_R, zero_rot_R);

                            float pred_global_R[9]; quat_to_mat3(hfk[h].wrist_quat.data(), pred_global_R);

                            // fused_local = zero_rot^T @ pred_global  (see PLAN.md derivation)
                            float zero_rot_T[9]; mat3_transpose(zero_rot_R, zero_rot_T);
                            float fused_R[9]; mat3_mul(zero_rot_T, pred_global_R, fused_R);

                            // Rotation-agreement gate (Python's `valid_angle`, run_inference
                            // "Doing IK" block): reject the splice if the hand crop's fused
                            // wrist rotation disagrees too much with pass-1's own wrist pose.
                            // Previously left out (PLAN.md step 7 TODO) — confirmed to matter:
                            // without it, a bad hand-crop scale/pose estimate can get spliced
                            // in even when the box+distance gate alone passed, corrupting the
                            // whole-body scale via the shared scale[8]/[9] PCA components.
                            {
                                const float* ori_e = pass1_wrist_euler[i].data() + (lr==0 ? 0 : 3);
                                float ori_R[9]; euler_xzy_to_mat3(ori_e[0], ori_e[1], ori_e[2], ori_R);
                                if (mat3_angle_diff(ori_R, fused_R) >= 1.4f) continue;
                            }

                            float wx, wz, wy;
                            rotmat_to_euler_xzy(fused_R, &wx, &wz, &wy);
                            fix_wrist_euler(wx, wz, wy);

                            // body_pose indices: right=[41,43,42], left=[31,33,32]
                            // DIAGNOSTIC: temporarily skipped via env var to isolate whether
                            // the wrist-rotation splice (not the scale splice, already disabled
                            // above) is the actual source of the visible mesh distortion.
                            if (!getenv("FSB_SKIP_WRIST_SPLICE"))
                            {
                                static const int idx_r[3] = {41,43,42}, idx_l[3] = {31,33,32};
                                const int* idx = lr==0 ? idx_r : idx_l;
                                p2_body_euler[idx[0]] = wx;
                                p2_body_euler[idx[1]] = wz;
                                p2_body_euler[idx[2]] = wy;
                            }

                            // hand[108] half swap (see run_inference lines ~1545-1569)
                            int src_off = lr==0 ? 54 : 0;   // right uses hand[54:], left uses hand[:54]
                            std::copy(hfk[h].hand108.begin()+src_off, hfk[h].hand108.begin()+src_off+54,
                                      r.hand_pose.begin() + (lr==0 ? 54 : 0));
                            // scale[8]/[9] swap: DISABLED (see PLAN.md "refinedpose" branch notes).
                            // hfk[h].scale28[8/9] comes from decoder_hand's single-pass (no
                            // iterative refinement) regression and is confirmed off by ~35-40%
                            // vs Python's real value even with fp32 weights (0.977-0.980 here
                            // vs Python's 0.719) -- this is what was producing the "whole body
                            // scale is wrong" regression once the validity gate started letting
                            // this splice fire. Root cause is architectural (decoder_hand.onnx
                            // is missing Python's per-layer do_interm_preds/keypoint_token_update
                            // loop, which needs the LBS body model between decoder layers --
                            // blocked from a single ONNX export by pymomentum). Re-enable once
                            // that iterative refinement is ported (native LBS + per-layer ONNX).
                            // if (r.scale.size() >= 10)
                            //     r.scale[lr==0 ? 8 : 9] = hfk[h].scale28[lr==0 ? 8 : 9];
                        }
                    }
                }
                r.body_pose.assign(p2_body_euler.begin(), p2_body_euler.end());

                // Rebuild derived fields (mhr_model_params, vertices/keypoints) from
                // the final spliced pose, mirroring the existing result-assembly code.
                ModelParams204 mp_final = build_model_params(p2_global_rot_euler, p2_body_euler.data(), nullptr, true);
                apply_hand_pose(mp_final.data, r.hand_pose.data(),
                                lbs_data->hand_pose_mean, lbs_data->hand_pose_comps,
                                lbs_data->hand_joint_idxs_left, lbs_data->hand_joint_idxs_right);
                if (lbs_data->scale_mean && lbs_data->scale_comps)
                {
                    int ns = lbs_data->n_scale_out, npc = lbs_data->n_scale_pc;
                    for (int j = 0; j < ns; ++j) mp_final.data[136+j] = lbs_data->scale_mean[j];
                    for (int k = 0; k < npc; ++k)
                        for (int j = 0; j < ns; ++j)
                            mp_final.data[136+j] += r.scale[k] * lbs_data->scale_comps[k*ns+j];
                }
                std::memcpy(r.mhr_model_params.data(), mp_final.data, 204*sizeof(float));

                if (!r.pred_vertices.empty())
                {
                    static const float zero_face_out[72] = {};
                    std::vector<float> fverts((size_t)lbs_data->n_verts*3), fjoints((size_t)lbs_data->n_joints*3);
                    if (mhr_lbs_compute(lbs_data, mp_final.data, r.shape.data(),
                                        cfg.zero_face_params ? zero_face_out : r.face_params.data(),
                                        fverts.data(), fjoints.data(), nullptr))
                    {
                        r.pred_vertices = fverts;
                        r.skeleton_3d   = fjoints;
                        {
                            float vmin[3]={1e9f,1e9f,1e9f}, vmax[3]={-1e9f,-1e9f,-1e9f};
                            for (size_t vi = 0; vi < fverts.size()/3; ++vi)
                                for (int c = 0; c < 3; ++c)
                                {
                                    float v = fverts[vi*3+c];
                                    vmin[c] = std::min(vmin[c], v);
                                    vmax[c] = std::max(vmax[c], v);
                                }
                            printf("[FSB]   vertdbg person=%d min=(%.3f,%.3f,%.3f) max=(%.3f,%.3f,%.3f) extent=(%.3f,%.3f,%.3f)\n",
                                   i, vmin[0],vmin[1],vmin[2], vmax[0],vmax[1],vmax[2],
                                   vmax[0]-vmin[0], vmax[1]-vmin[1], vmax[2]-vmin[2]);
                        }

                        std::vector<float> kps3d(70*3, 0.f);
                        for (const auto& e : kp_mapping)
                        {
                            for (int c = 0; c < 3; ++c)
                            {
                                float src = (e.col < 18439) ? fverts[e.col*3+c] : fjoints[(e.col-18439)*3+c];
                                kps3d[e.row*3+c] += src * e.val;
                            }
                        }
                        r.keypoints_3d = kps3d;
                        std::vector<float> kps2d(70*2);
                        for (int k = 0; k < 70; ++k)
                        {
                            float dz = kps3d[k*3+2] + r.pred_cam_t[2];
                            float dx = kps3d[k*3+0] + r.pred_cam_t[0];
                            float dy = kps3d[k*3+1] + r.pred_cam_t[1];
                            if (dz < 1e-4f) dz = 1e-4f;
                            kps2d[k*2+0] = dx/dz*fx + cx;
                            kps2d[k*2+1] = dy/dz*fy + cy;
                        }
                        r.keypoints_2d = kps2d;
                        printf("[FSB]   kp2ddbg person=%d l_sh=(%.1f,%.1f) r_sh=(%.1f,%.1f) "
                               "l_elb=(%.1f,%.1f) r_elb=(%.1f,%.1f) l_hip=(%.1f,%.1f) r_hip=(%.1f,%.1f) "
                               "r_wrist=(%.1f,%.1f) l_wrist=(%.1f,%.1f) scale8=%.4f scale9=%.4f\n",
                               i, kps2d[5*2],kps2d[5*2+1], kps2d[6*2],kps2d[6*2+1],
                               kps2d[7*2],kps2d[7*2+1], kps2d[8*2],kps2d[8*2+1],
                               kps2d[9*2],kps2d[9*2+1], kps2d[10*2],kps2d[10*2+1],
                               kps2d[41*2],kps2d[41*2+1], kps2d[62*2],kps2d[62*2+1],
                               r.scale.size()>8?r.scale[8]:-1.f, r.scale.size()>9?r.scale[9]:-1.f);
                    }
                }

                if (right_valid || left_valid)
                    printf("[FSB]   person=%d pass-2 applied  right_valid=%d left_valid=%d\n",
                           i, right_valid, left_valid);

                if (skip_pass2) r = r_pass1_backup;
            }
            printf("[FSB] pass-2 + IK + splice: %.1f ms  (%d person(s))\n", ms(t0), B);
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
