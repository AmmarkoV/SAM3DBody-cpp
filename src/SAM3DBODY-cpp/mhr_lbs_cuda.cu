// ============================================================================
// mhr_lbs_cuda.cu  –  CUDA kernels for MHR Linear Blend Skinning
//
// See mhr_lbs_cuda.cuh for design notes.
// ============================================================================

#include "mhr_lbs_cuda.cuh"

#ifdef FSB_CUDA

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ─── Quaternion helpers (device copies; match model_loader_transform_joints.c) ─

#define MHR_LN2 0.693147180559945f

__device__ __forceinline__
void d_qmul(float *r, const float *a, const float *b)
{
    r[0] = b[0]*a[3] + b[3]*a[0] + b[2]*a[1] - b[1]*a[2];
    r[1] = b[1]*a[3] - b[2]*a[0] + b[3]*a[1] + b[0]*a[2];
    r[2] = b[2]*a[3] + b[1]*a[0] - b[0]*a[1] + b[3]*a[2];
    r[3] = b[3]*a[3] - b[0]*a[0] - b[1]*a[1] - b[2]*a[2];
}

__device__ __forceinline__
void d_qrot(float *out, const float *q, float vx, float vy, float vz)
{
    float qx=q[0], qy=q[1], qz=q[2], qw=q[3];
    float tx = 2.f*(qy*vz - qz*vy);
    float ty = 2.f*(qz*vx - qx*vz);
    float tz = 2.f*(qx*vy - qy*vx);
    out[0] = vx + qw*tx + (qy*tz - qz*ty);
    out[1] = vy + qw*ty + (qz*tx - qx*tz);
    out[2] = vz + qw*tz + (qx*ty - qy*tx);
}

// ─── Kernel 1: shape blend ────────────────────────────────────────────────────
//
// out[vi*3 : +3] = base_shape[vi*3 : +3]
//                + Σ_{i<n_sc} sc[i] * shape_vecs[i*nv*3 + vi*3 : +3]
//                + Σ_{i<n_fc} fc[i] * face_vecs [i*nv*3 + vi*3 : +3]
//
// Launch config: grid = ceil(nv/256), block = 256
// Memory access: for a fixed 'i', consecutive threads access consecutive
// float3 triads → 12-byte stride, fully coalesced.

__global__ void k_shape_blend(const float * __restrict__ base_shape,
                               const float * __restrict__ shape_vecs,
                               const float * __restrict__ face_vecs,
                               const float * __restrict__ sc,   // shape_coeffs [n_sc]
                               const float * __restrict__ fc,   // face_coeffs  [n_fc]
                               float       * __restrict__ out,  // unposed [nv*3]
                               int nv, int n_sc, int n_fc)
{
    int vi = blockIdx.x * blockDim.x + threadIdx.x;
    if (vi >= nv) return;

    float x = base_shape[vi*3+0];
    float y = base_shape[vi*3+1];
    float z = base_shape[vi*3+2];

    for (int i = 0; i < n_sc; ++i) {
        float c = sc[i];
        if (c == 0.f) continue;
        const float *sv = shape_vecs + (size_t)i * nv * 3 + vi * 3;
        x += c * sv[0];  y += c * sv[1];  z += c * sv[2];
    }
    for (int i = 0; i < n_fc; ++i) {
        float c = fc[i];
        if (c == 0.f) continue;
        const float *fv = face_vecs + (size_t)i * nv * 3 + vi * 3;
        x += c * fv[0];  y += c * fv[1];  z += c * fv[2];
    }

    out[vi*3+0] = x;  out[vi*3+1] = y;  out[vi*3+2] = z;
}

// ─── Kernel 2: LBS scatter (CSR gather) + coordinate flip ────────────────────
//
// For each vertex vi, gathers contributions from all influencing joints:
//   out[vi] += Σ_{k in CSR row vi} w[k] * (skin_t[ji] + skin_s[ji]*rotate(skin_q[ji], unposed[vi]))
//
// CSR layout (built at upload time from COO skin table, sorted by vertex):
//   vt_start[vi]  .. vt_start[vi+1]-1  → indices into vt_joint / vt_weight
//
// No atomic operations: each thread owns its output vertex exclusively.
// Launch config: grid = ceil(nv/256), block = 256

__global__ void k_lbs_scatter(const float * __restrict__ unposed,   // [nv*3]
                               const float * __restrict__ skin_t,    // [nj*3]  from CPU FK
                               const float * __restrict__ skin_q,    // [nj*4]  from CPU FK
                               const float * __restrict__ skin_s,    // [nj]    from CPU FK
                               const int   * __restrict__ vt_start,  // [nv+1]  CSR row ptrs
                               const int   * __restrict__ vt_joint,  // [ns]    joint idx
                               const float * __restrict__ vt_weight, // [ns]    blend weight
                               float       * __restrict__ out_verts, // [nv*3]
                               int nv)
{
    int vi = blockIdx.x * blockDim.x + threadIdx.x;
    if (vi >= nv) return;

    float ux = unposed[vi*3+0];
    float uy = unposed[vi*3+1];
    float uz = unposed[vi*3+2];

    float ox = 0.f, oy = 0.f, oz = 0.f;

    int start = vt_start[vi];
    int end   = vt_start[vi + 1];
    for (int k = start; k < end; ++k) {
        int   ji = vt_joint[k];
        float w  = vt_weight[k];
        float sx = skin_s[ji];
        float pv[3];
        d_qrot(pv, skin_q + ji * 4, ux, uy, uz);
        ox += w * (skin_t[ji*3+0] + sx * pv[0]);
        oy += w * (skin_t[ji*3+1] + sx * pv[1]);
        oz += w * (skin_t[ji*3+2] + sx * pv[2]);
    }

    // Step 7: Y,Z flip + cm→m (matches mhr_lbs_compute)
    out_verts[vi*3+0] =  ox * 0.01f;
    out_verts[vi*3+1] = -oy * 0.01f;
    out_verts[vi*3+2] = -oz * 0.01f;
}

// ─── CPU helpers (mirror of model_loader_transform_joints.c) ─────────────────
// Needed to compute the per-frame joint transforms on CPU (Steps 2–5).
// Kept local to this TU; same implementations as the .c file.

static void h_qmul(float *r, const float *a, const float *b)
{
    r[0] = b[0]*a[3] + b[3]*a[0] + b[2]*a[1] - b[1]*a[2];
    r[1] = b[1]*a[3] - b[2]*a[0] + b[3]*a[1] + b[0]*a[2];
    r[2] = b[2]*a[3] + b[1]*a[0] - b[0]*a[1] + b[3]*a[2];
    r[3] = b[3]*a[3] - b[0]*a[0] - b[1]*a[1] - b[2]*a[2];
}

static void h_qrot(float *out, const float *q, const float *v)
{
    float qx=q[0],qy=q[1],qz=q[2],qw=q[3];
    float vx=v[0],vy=v[1],vz=v[2];
    float tx=2.f*(qy*vz-qz*vy), ty=2.f*(qz*vx-qx*vz), tz=2.f*(qx*vy-qy*vx);
    out[0]=vx+qw*tx+(qy*tz-qz*ty);
    out[1]=vy+qw*ty+(qz*tx-qx*tz);
    out[2]=vz+qw*tz+(qx*ty-qy*tx);
}

static void h_euler_to_quat(float ex, float ey, float ez, float *q)
{
    float hx=ex*.5f, hy=ey*.5f, hz=ez*.5f;
    float qx[4]={sinf(hx),0,0,cosf(hx)};
    float qy[4]={0,sinf(hy),0,cosf(hy)};
    float qz[4]={0,0,sinf(hz),cosf(hz)};
    float tmp[4];
    h_qmul(tmp, qz, qy);
    h_qmul(q,   tmp, qx);
}

// ─── CSR skin table builder ───────────────────────────────────────────────────
// Converts COO skin table (joint_idx, vert_idx, weight)[ns] to vertex-indexed
// CSR so each vertex can gather its joint contributions without atomics.

struct CSRSkin {
    int   *start;   // [nv+1]
    int   *joint;   // [ns]
    float *weight;  // [ns]
};

static CSRSkin build_csr(const int *joint_idx, const int *vert_idx,
                          const float *weights, int ns, int nv)
{
    CSRSkin c{};
    c.start  = (int*)calloc(nv + 1, sizeof(int));
    c.joint  = (int*)malloc(ns * sizeof(int));
    c.weight = (float*)malloc(ns * sizeof(float));
    if (!c.start || !c.joint || !c.weight) return c;

    // Count entries per vertex
    for (int k = 0; k < ns; ++k) c.start[vert_idx[k] + 1]++;
    // Prefix-sum → row pointers
    for (int v = 0; v < nv; ++v) c.start[v+1] += c.start[v];
    // Fill (use temporary row cursor)
    int *cur = (int*)calloc(nv, sizeof(int));
    for (int k = 0; k < ns; ++k) {
        int vi = vert_idx[k];
        int pos = c.start[vi] + cur[vi]++;
        c.joint[pos]  = joint_idx[k];
        c.weight[pos] = weights[k];
    }
    free(cur);
    return c;
}

// ─── Context struct ───────────────────────────────────────────────────────────

struct MHR_LBS_CUDACtx {
    // Static GPU arrays (uploaded once at init)
    float *d_base_shape;     // [nv*3]
    float *d_shape_vecs;     // [n_sc * nv * 3]
    float *d_face_vecs;      // [n_fc * nv * 3]
    int   *d_vt_start;       // [nv+1]  CSR row pointers
    int   *d_vt_joint;       // [ns]
    float *d_vt_weight;      // [ns]

    // Per-frame device buffers (allocated once, reused every call)
    float *d_sc;             // [n_sc] shape coeffs
    float *d_fc;             // [n_fc] face coeffs
    float *d_unposed;        // [nv*3]
    float *d_skin_t;         // [nj*3]
    float *d_skin_q;         // [nj*4]
    float *d_skin_s;         // [nj]
    float *d_out_verts;      // [nv*3]

    // Dimensions (cached for kernel launches)
    int nv, nj, ns, n_sc, n_fc;

    // CPU scratch for Steps 2-5 (joint transforms)
    float *h_joint_params;   // [nj*7]
    float *h_input_vec;      // [npc]
    float *h_t_local;        // [nj*3]
    float *h_q_local;        // [nj*4]
    float *h_s_local;        // [nj]
    float *h_g_t;            // [nj*3]
    float *h_g_q;            // [nj*4]
    float *h_g_s;            // [nj]
    float *h_skin_t;         // [nj*3]
    float *h_skin_q;         // [nj*4]
    float *h_skin_s;         // [nj]
    // unposed scratch (only used when correctives present)
    float *h_unposed;        // [nv*3]  NULL when no correctives
};

// ─── Public API ───────────────────────────────────────────────────────────────

extern "C"
MHR_LBS_CUDACtx *mhr_lbs_cuda_init(const struct MHR_LBS_Data *d)
{
    if (!d) return nullptr;

    MHR_LBS_CUDACtx *ctx = new MHR_LBS_CUDACtx{};
    ctx->nv   = d->n_verts;
    ctx->nj   = d->n_joints;
    ctx->ns   = d->n_skin;
    ctx->n_sc = d->n_shape_pc;
    ctx->n_fc = d->n_face_pc;

    int nv = ctx->nv, nj = ctx->nj, ns = ctx->ns;
    int n_sc = ctx->n_sc, n_fc = ctx->n_fc;

#define CUDA_ALLOC_UPLOAD(dst, src, bytes)                          \
    do {                                                            \
        if (cudaMalloc(&(dst), (bytes)) != cudaSuccess) goto fail;  \
        if (cudaMemcpy((dst), (src), (bytes),                       \
                       cudaMemcpyHostToDevice) != cudaSuccess)      \
            goto fail;                                              \
    } while(0)

    // Static data
    CUDA_ALLOC_UPLOAD(ctx->d_base_shape,  d->base_shape,
                      (size_t)nv * 3 * sizeof(float));
    CUDA_ALLOC_UPLOAD(ctx->d_shape_vecs,  d->shape_vectors,
                      (size_t)n_sc * nv * 3 * sizeof(float));
    CUDA_ALLOC_UPLOAD(ctx->d_face_vecs,   d->face_vectors,
                      (size_t)n_fc * nv * 3 * sizeof(float));

    // Build CSR skin table on CPU, upload
    {
        CSRSkin csr = build_csr(d->skin_joint_idx, d->skin_vert_idx,
                                 d->skin_weights, ns, nv);
        if (!csr.start) goto fail;
        CUDA_ALLOC_UPLOAD(ctx->d_vt_start,  csr.start,  (size_t)(nv+1)*sizeof(int));
        CUDA_ALLOC_UPLOAD(ctx->d_vt_joint,  csr.joint,  (size_t)ns*sizeof(int));
        CUDA_ALLOC_UPLOAD(ctx->d_vt_weight, csr.weight, (size_t)ns*sizeof(float));
        free(csr.start); free(csr.joint); free(csr.weight);
    }

    // Per-frame device buffers
    if (cudaMalloc(&ctx->d_sc,        (size_t)n_sc*sizeof(float)) != cudaSuccess) goto fail;
    if (cudaMalloc(&ctx->d_fc,        (size_t)n_fc*sizeof(float)) != cudaSuccess) goto fail;
    if (cudaMalloc(&ctx->d_unposed,   (size_t)nv*3*sizeof(float)) != cudaSuccess) goto fail;
    if (cudaMalloc(&ctx->d_skin_t,    (size_t)nj*3*sizeof(float)) != cudaSuccess) goto fail;
    if (cudaMalloc(&ctx->d_skin_q,    (size_t)nj*4*sizeof(float)) != cudaSuccess) goto fail;
    if (cudaMalloc(&ctx->d_skin_s,    (size_t)nj*sizeof(float))   != cudaSuccess) goto fail;
    if (cudaMalloc(&ctx->d_out_verts, (size_t)nv*3*sizeof(float)) != cudaSuccess) goto fail;

#undef CUDA_ALLOC_UPLOAD

    // CPU scratch for joint transforms (Steps 2-5)
    {
        int npc = d->pt_cols;
        ctx->h_joint_params = (float*)malloc((size_t)nj * 7 * sizeof(float));
        ctx->h_input_vec    = (float*)malloc((size_t)npc * sizeof(float));
        ctx->h_t_local      = (float*)malloc((size_t)nj * 3 * sizeof(float));
        ctx->h_q_local      = (float*)malloc((size_t)nj * 4 * sizeof(float));
        ctx->h_s_local      = (float*)malloc((size_t)nj * sizeof(float));
        ctx->h_g_t          = (float*)malloc((size_t)nj * 3 * sizeof(float));
        ctx->h_g_q          = (float*)malloc((size_t)nj * 4 * sizeof(float));
        ctx->h_g_s          = (float*)malloc((size_t)nj * sizeof(float));
        ctx->h_skin_t       = (float*)malloc((size_t)nj * 3 * sizeof(float));
        ctx->h_skin_q       = (float*)malloc((size_t)nj * 4 * sizeof(float));
        ctx->h_skin_s       = (float*)malloc((size_t)nj * sizeof(float));
        if (!ctx->h_joint_params || !ctx->h_input_vec ||
            !ctx->h_t_local || !ctx->h_q_local || !ctx->h_s_local ||
            !ctx->h_g_t || !ctx->h_g_q || !ctx->h_g_s ||
            !ctx->h_skin_t || !ctx->h_skin_q || !ctx->h_skin_s)
            goto fail;
    }

    // Correctives: allocate CPU unposed scratch if needed (used as host staging)
    if (d->corr_sp1_row) {
        ctx->h_unposed = (float*)malloc((size_t)nv * 3 * sizeof(float));
        if (!ctx->h_unposed) goto fail;
        fprintf(stderr, "[LBS CUDA] correctives present — shape blend runs on CPU\n");
    }

    fprintf(stderr, "[LBS CUDA] init OK  nv=%d nj=%d ns=%d  static GPU: ~%.1f MB\n",
            nv, nj, ns,
            (float)((size_t)(nv*3 + (size_t)n_sc*nv*3 + (size_t)n_fc*nv*3
                    + (nv+1) + ns + ns) * sizeof(float)) / 1e6f);
    return ctx;

fail:
    fprintf(stderr, "[LBS CUDA] init failed — falling back to CPU\n");
    mhr_lbs_cuda_free(ctx);
    return nullptr;
}


extern "C"
int mhr_lbs_cuda_compute(MHR_LBS_CUDACtx        *ctx,
                           const struct MHR_LBS_Data *d,
                           const float *model_params,
                           const float *shape_coeffs,
                           const float *face_coeffs,
                           float       *out_verts,
                           float       *out_joints)
{
    if (!ctx || !d || !model_params || !shape_coeffs || !face_coeffs || !out_verts)
        return 0;

    int nv = ctx->nv, nj = ctx->nj, n_sc = ctx->n_sc, n_fc = ctx->n_fc;
    int npc = d->pt_cols;

    // ── Steps 2-5 on CPU: parameter transform → local TRS → FK → skin TRS ────
    // These involve only 127 joints — CPU cost is negligible (<0.1 ms).

    // Step 2: joint_params = PT @ model_params
    {
        int npr = d->pt_rows;
        memset(ctx->h_input_vec, 0, (size_t)npc * sizeof(float));
        memcpy(ctx->h_input_vec, model_params,
               (size_t)(npc < 204 ? npc : 204) * sizeof(float));
        for (int j = 0; j < npr; ++j) {
            float acc = 0.f;
            const float *row = d->PT + (size_t)j * npc;
            for (int k = 0; k < npc; ++k) acc += row[k] * ctx->h_input_vec[k];
            ctx->h_joint_params[j] = acc;
        }
    }

    // Step 3: local TRS per joint
    for (int j = 0; j < nj; ++j) {
        const float *jp  = ctx->h_joint_params + j * 7;
        const float *off = d->joint_offsets      + j * 3;
        const float *pre = d->joint_prerotations + j * 4;
        ctx->h_t_local[j*3+0] = off[0] + jp[0];
        ctx->h_t_local[j*3+1] = off[1] + jp[1];
        ctx->h_t_local[j*3+2] = off[2] + jp[2];
        float q_euler[4];
        h_euler_to_quat(jp[3], jp[4], jp[5], q_euler);
        h_qmul(ctx->h_q_local + j*4, pre, q_euler);
        ctx->h_s_local[j] = expf(jp[6] * MHR_LN2);
    }

    // Step 4: FK chain (parent < child guaranteed by sorted joint order)
    for (int j = 0; j < nj; ++j) {
        int p = d->joint_parents[j];
        if (p < 0) {
            memcpy(ctx->h_g_t + j*3, ctx->h_t_local + j*3, 3*sizeof(float));
            memcpy(ctx->h_g_q + j*4, ctx->h_q_local + j*4, 4*sizeof(float));
            ctx->h_g_s[j] = ctx->h_s_local[j];
        } else {
            ctx->h_g_s[j] = ctx->h_g_s[p] * ctx->h_s_local[j];
            h_qmul(ctx->h_g_q + j*4, ctx->h_g_q + p*4, ctx->h_q_local + j*4);
            float rt[3];
            h_qrot(rt, ctx->h_g_q + p*4, ctx->h_t_local + j*3);
            ctx->h_g_t[j*3+0] = ctx->h_g_t[p*3+0] + ctx->h_g_s[p] * rt[0];
            ctx->h_g_t[j*3+1] = ctx->h_g_t[p*3+1] + ctx->h_g_s[p] * rt[1];
            ctx->h_g_t[j*3+2] = ctx->h_g_t[p*3+2] + ctx->h_g_s[p] * rt[2];
        }
    }

    // Step 5: skin TRS = global ∘ inv_bind
    for (int j = 0; j < nj; ++j) {
        const float *ib = d->inv_bind_pose + j * 8;
        ctx->h_skin_s[j] = ctx->h_g_s[j] * ib[7];
        h_qmul(ctx->h_skin_q + j*4, ctx->h_g_q + j*4, ib + 3);
        float rt[3];
        h_qrot(rt, ctx->h_g_q + j*4, ib);
        ctx->h_skin_t[j*3+0] = ctx->h_g_t[j*3+0] + ctx->h_g_s[j] * rt[0];
        ctx->h_skin_t[j*3+1] = ctx->h_g_t[j*3+1] + ctx->h_g_s[j] * rt[1];
        ctx->h_skin_t[j*3+2] = ctx->h_g_t[j*3+2] + ctx->h_g_s[j] * rt[2];
    }

    // ── Upload per-frame data to GPU ──────────────────────────────────────────
    cudaMemcpy(ctx->d_skin_t, ctx->h_skin_t, (size_t)nj*3*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_skin_q, ctx->h_skin_q, (size_t)nj*4*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx->d_skin_s, ctx->h_skin_s, (size_t)nj*sizeof(float),   cudaMemcpyHostToDevice);

    // ── GPU Stage 1: shape blend (or CPU + upload when correctives present) ───
    if (ctx->h_unposed) {
        // Correctives path: CPU shape blend → apply correctives → upload unposed
        memcpy(ctx->h_unposed, d->base_shape, (size_t)nv * 3 * sizeof(float));
        for (int i = 0; i < n_sc; ++i) {
            float c = shape_coeffs[i];
            if (c == 0.f) continue;
            const float *sv = d->shape_vectors + (size_t)i * nv * 3;
            for (int k = 0; k < nv*3; ++k) ctx->h_unposed[k] += c * sv[k];
        }
        for (int i = 0; i < n_fc; ++i) {
            float c = face_coeffs[i];
            if (c == 0.f) continue;
            const float *fv = d->face_vectors + (size_t)i * nv * 3;
            for (int k = 0; k < nv*3; ++k) ctx->h_unposed[k] += c * fv[k];
        }
        // mhr_apply_correctives is static in the .c file; replicate inline
        if (d->corr_sp1_row) {
            int nfeat = d->corr_n_feat, nhid = d->corr_n_hidden, nout = d->corr_n_out;
            float *feat   = (float*)calloc((size_t)nfeat, sizeof(float));
            float *hidden = (float*)calloc((size_t)nhid,  sizeof(float));
            float *corr   = (float*)calloc((size_t)nout,  sizeof(float));
            if (feat && hidden && corr) {
                // Feature extraction: 6D pose from joints 2..N-1
                for (int j = 2; j < nj; ++j) {
                    const float *jp = ctx->h_joint_params + j * 7;
                    float ex=jp[3], ey=jp[4], ez=jp[5];
                    float cx=cosf(ex),sx=sinf(ex),cy=cosf(ey),sy=sinf(ey),cz=cosf(ez),sz=sinf(ez);
                    int fi = (j-2)*6;
                    feat[fi+0]=cz*cy-1.f; feat[fi+1]=sz*cy; feat[fi+2]=-sy;
                    feat[fi+3]=-sz*cx+cz*sy*sx; feat[fi+4]=cz*cx+sz*sy*sx-1.f; feat[fi+5]=cy*sx;
                }
                // Layer 1: sparse @ feat → hidden, ReLU
                for (int k = 0; k < d->corr_nnz1; ++k)
                    hidden[d->corr_sp1_row[k]] += d->corr_sp1_val[k] * feat[d->corr_sp1_col[k]];
                for (int i = 0; i < nhid; ++i) if (hidden[i] < 0.f) hidden[i] = 0.f;
                // Layer 2: sparse @ hidden → corr
                for (int k = 0; k < d->corr_nnz2; ++k)
                    corr[d->corr_sp2_row[k]] += d->corr_sp2_val[k] * hidden[d->corr_sp2_col[k]];
                for (int i = 0; i < nout; ++i) ctx->h_unposed[i] += corr[i];
            }
            free(feat); free(hidden); free(corr);
        }
        cudaMemcpy(ctx->d_unposed, ctx->h_unposed,
                   (size_t)nv*3*sizeof(float), cudaMemcpyHostToDevice);
    } else {
        // No correctives: GPU shape blend from pre-uploaded static data
        cudaMemcpy(ctx->d_sc, shape_coeffs, (size_t)n_sc*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(ctx->d_fc, face_coeffs,  (size_t)n_fc*sizeof(float), cudaMemcpyHostToDevice);

        int block = 256;
        int grid  = (nv + block - 1) / block;
        k_shape_blend<<<grid, block>>>(
            ctx->d_base_shape, ctx->d_shape_vecs, ctx->d_face_vecs,
            ctx->d_sc, ctx->d_fc, ctx->d_unposed,
            nv, n_sc, n_fc);
    }

    // ── GPU Stage 6: LBS scatter + coord flip ────────────────────────────────
    {
        int block = 256;
        int grid  = (nv + block - 1) / block;
        k_lbs_scatter<<<grid, block>>>(
            ctx->d_unposed, ctx->d_skin_t, ctx->d_skin_q, ctx->d_skin_s,
            ctx->d_vt_start, ctx->d_vt_joint, ctx->d_vt_weight,
            ctx->d_out_verts, nv);
    }

    // ── Download results ──────────────────────────────────────────────────────
    cudaMemcpy(out_verts, ctx->d_out_verts,
               (size_t)nv * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    // ── Step 8: joint world positions (from CPU FK g_t, already computed) ─────
    if (out_joints) {
        for (int j = 0; j < nj; ++j) {
            out_joints[j*3+0] =  ctx->h_g_t[j*3+0] * 0.01f;
            out_joints[j*3+1] = -ctx->h_g_t[j*3+1] * 0.01f;
            out_joints[j*3+2] = -ctx->h_g_t[j*3+2] * 0.01f;
        }
    }

    return 1;
}


extern "C"
void mhr_lbs_cuda_free(MHR_LBS_CUDACtx *ctx)
{
    if (!ctx) return;
#define CF(p) do { if (ctx->p) { cudaFree(ctx->p); ctx->p = nullptr; } } while(0)
#define HF(p) do { if (ctx->p) { free(ctx->p);     ctx->p = nullptr; } } while(0)
    CF(d_base_shape); CF(d_shape_vecs); CF(d_face_vecs);
    CF(d_vt_start);   CF(d_vt_joint);  CF(d_vt_weight);
    CF(d_sc);         CF(d_fc);
    CF(d_unposed);    CF(d_skin_t);    CF(d_skin_q);  CF(d_skin_s);
    CF(d_out_verts);
    HF(h_joint_params); HF(h_input_vec);
    HF(h_t_local); HF(h_q_local); HF(h_s_local);
    HF(h_g_t);     HF(h_g_q);     HF(h_g_s);
    HF(h_skin_t);  HF(h_skin_q);  HF(h_skin_s);
    HF(h_unposed);
#undef CF
#undef HF
    delete ctx;
}

#endif // FSB_CUDA
