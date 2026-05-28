#pragma once
// ============================================================================
// mhr_lbs_cuda.cuh  –  GPU-accelerated MHR Linear Blend Skinning
//
// Accelerates the two dominant stages of mhr_lbs_compute():
//
//   Stage 1 – Shape blend (CPU: ~2.2 M MACs)
//     out[nv*3] = base_shape + Σ sc[i]*shape_vec[i] + Σ fc[i]*face_vec[i]
//     GPU: 18439 threads, each summing 45+72 weighted vectors → trivially parallel
//
//   Stage 6 – LBS scatter (CPU: ~51337 qrot + accumulate)
//     out_vert[vi] += Σ_j  w[j,vi] * (skin_t[j] + skin_s[j]*rotate(skin_q[j], unposed[vi]))
//     GPU: 18439 threads (one per vertex), CSR gather — no atomics, fully parallel
//
// Stages 2–5 (parameter transform, Euler→quat, FK, skin-TRS) involve only
// 127 joints and are kept on CPU.  The resulting skin_{t,q,s} arrays (4 KB)
// are uploaded to device before the scatter kernel.
//
// Correctives path: if correctives are loaded (corr_sp1_row != NULL) the
// sparse corrective net runs on CPU (complex SpMV with different sparsity),
// and only Stage 6 runs on GPU.  Stage 1 falls back to CPU in that case to
// avoid a mid-frame round-trip for the unposed mesh.
//
// CPU fallback: if FSB_CUDA is not defined, mhr_lbs_cuda.cuh is still
// includable — every function degrades to a stub returning 0/NULL so that
// fast_sam_3dbody.cpp can guard with a single `if (lbs_cuda)` check.
// ============================================================================

#ifdef FSB_CUDA

#include "../GraphicsEngine/ModelLoader/model_loader_transform_joints.h"

#ifdef __cplusplus
extern "C" {
#endif

struct MHR_LBS_CUDACtx;   // opaque; defined in mhr_lbs_cuda.cu

// Upload all static data from d to the GPU.
// Returns NULL on allocation failure or if CUDA is unavailable.
// Caller must call mhr_lbs_cuda_free() when done.
MHR_LBS_CUDACtx *mhr_lbs_cuda_init(const struct MHR_LBS_Data *d);

// GPU-accelerated forward pass.  Drop-in replacement for mhr_lbs_compute().
// Falls back to CPU for stages that cannot be GPU-accelerated (correctives).
// Returns 1 on success, 0 on failure.
int mhr_lbs_cuda_compute(MHR_LBS_CUDACtx        *ctx,
                          const struct MHR_LBS_Data *d,
                          const float *model_params,   /* [204]        */
                          const float *shape_coeffs,   /* [n_shape_pc] */
                          const float *face_coeffs,    /* [n_face_pc]  */
                          float       *out_verts,      /* [n_verts*3]  */
                          float       *out_joints);    /* [n_joints*3], may be NULL */

// Free all GPU memory allocated by mhr_lbs_cuda_init().
void mhr_lbs_cuda_free(MHR_LBS_CUDACtx *ctx);

#ifdef __cplusplus
} // extern "C"
#endif

#else // !FSB_CUDA  — stubs so callers compile without #ifdefs

struct MHR_LBS_CUDACtx;
struct MHR_LBS_Data;

static inline MHR_LBS_CUDACtx *mhr_lbs_cuda_init(const struct MHR_LBS_Data *) { return 0; }
static inline int mhr_lbs_cuda_compute(MHR_LBS_CUDACtx *, const struct MHR_LBS_Data *,
                                        const float *, const float *, const float *,
                                        float *, float *) { return 0; }
static inline void mhr_lbs_cuda_free(MHR_LBS_CUDACtx *) {}

#endif // FSB_CUDA
