// mhr_fk.cpp — see mhr_fk.h. Ported verbatim from BVHWriter's original
// compute_per_frame_mhr_state()/compute_joint_locals()/bone-sample logic
// (bvh_writer.cpp, pre-2026-11 refactor) — behavior-preserving extraction.

#include "mhr_fk.h"
#include "fast_sam_3dbody.h"
#include "mhr_joint_table.h"

extern "C" {
#include "ModelLoader/model_loader_transform_joints.h"
}

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace mhr_fk
{

// ─── quaternion helpers (XYZW) ──────────────────────────────────────────────
void qmul(float* r, const float* a, const float* b)
{
    r[0] = b[0]*a[3] + b[3]*a[0] + b[2]*a[1] - b[1]*a[2];
    r[1] = b[1]*a[3] - b[2]*a[0] + b[3]*a[1] + b[0]*a[2];
    r[2] = b[2]*a[3] + b[1]*a[0] - b[0]*a[1] + b[3]*a[2];
    r[3] = b[3]*a[3] - b[0]*a[0] - b[1]*a[1] - b[2]*a[2];
}
void qconj(float* r, const float* a)
{
    r[0]=-a[0];
    r[1]=-a[1];
    r[2]=-a[2];
    r[3]=a[3];
}
void qrot(float* o, const float* q, const float* v)
{
    float qx=q[0], qy=q[1], qz=q[2], qw=q[3], vx=v[0], vy=v[1], vz=v[2];
    float tx = 2.f*(qy*vz - qz*vy);
    float ty = 2.f*(qz*vx - qx*vz);
    float tz = 2.f*(qx*vy - qy*vx);
    o[0]=vx + qw*tx + (qy*tz - qz*ty);
    o[1]=vy + qw*ty + (qz*tx - qx*tz);
    o[2]=vz + qw*tz + (qx*ty - qy*tx);
}
void euler_mhr_to_quat(float ex, float ey, float ez, float* q)
{
    float hx=ex*0.5f, hy=ey*0.5f, hz=ez*0.5f;
    float qx[4]= {sinf(hx),0.f,0.f,cosf(hx)};
    float qy[4]= {0.f,sinf(hy),0.f,cosf(hy)};
    float qz[4]= {0.f,0.f,sinf(hz),cosf(hz)};
    float t[4];
    qmul(t, qz, qy);
    qmul(q, t, qx);
}

// ─── State ──────────────────────────────────────────────────────────────────
void State::init(const MHR_LBS_Data* lbs)
{
    lbs_ = lbs;
    const int nj = lbs->n_joints;
    n_joints_ = nj;

    joint_params_.assign((size_t)nj * 7, 0.f);
    q_local_.assign((size_t)nj * 4, 0.f);
    q_global_mhr_.assign((size_t)nj * 4, 0.f);
    t_global_mhr_.assign((size_t)nj * 3, 0.f);
    s_global_mhr_.assign((size_t)nj, 1.f);

    // MHR rest-pose globals (reused by every frame and every person).
    q_global_mhr_rest_.assign((size_t)nj * 4, 0.f);
    for (int j = 0; j < nj; ++j)
    {
        const float* p_q = lbs->joint_prerotations + j*4;
        int parent = lbs->joint_parents[j];
        if (parent < 0)
            memcpy(&q_global_mhr_rest_[j*4], p_q, 4*sizeof(float));
        else
            qmul(&q_global_mhr_rest_[j*4], &q_global_mhr_rest_[parent*4], p_q);
    }
}

void State::compute(const fsb::MHRResult& r)
{
    const int nj    = n_joints_;
    const int npc   = lbs_->pt_cols;
    const float* PT = lbs_->PT;
    const float* pre = lbs_->joint_prerotations;
    const int*  parents = lbs_->joint_parents;

    // Decode the scale PCA into model_params[136:204], matching the mesh path
    // (the render loop re-decodes scale before calling mhr_lbs_compute).  The
    // pipeline runs with skip_body_model=true, so its lbs_data is null and
    // r.mhr_model_params[136:204] is left ZERO — without this decode the FK
    // has scale=1 while the mesh has real per-joint scale, so bone lengths
    // measured from this FK would disagree with the deformed mesh.
    std::array<float,204> params = r.mhr_model_params;
    if (lbs_->scale_mean && lbs_->scale_comps && !r.scale.empty())
    {
        const int ns = lbs_->n_scale_out;   // 68
        const int np = lbs_->n_scale_pc;    // 28
        for (int j = 0; j < ns && 136 + j < 204; ++j)
            params[136 + j] = lbs_->scale_mean[j];
        for (int k = 0; k < np && k < (int)r.scale.size(); ++k)
            for (int j = 0; j < ns && 136 + j < 204; ++j)
                params[136 + j] += r.scale[k] * lbs_->scale_comps[k * ns + j];
    }

    const int take = std::min(npc, 204);
    for (int row = 0; row < nj * 7; ++row)
    {
        const float* prow = PT + (size_t)row * npc;
        float acc = 0.f;
        for (int k = 0; k < take; ++k) acc += prow[k] * params[k];
        joint_params_[row] = acc;
    }

    for (int j = 0; j < nj; ++j)
    {
        const float* jp = &joint_params_[j * 7];
        float q_euler[4];
        euler_mhr_to_quat(jp[3], jp[4], jp[5], q_euler);
        qmul(&q_local_[j*4], pre + j*4, q_euler);

        int p = parents[j];
        const float* off = lbs_->joint_offsets + j*3;
        // Per-joint local scale = exp2(log2_scale). mhr_lbs_compute scales
        // each child offset by the PARENT's accumulated global scale.
        float s_local = exp2f(jp[6]);
        if (p < 0)
        {
            memcpy(&q_global_mhr_[j*4], &q_local_[j*4], 4*sizeof(float));
            t_global_mhr_[j*3+0] = off[0] + jp[0];
            t_global_mhr_[j*3+1] = off[1] + jp[1];
            t_global_mhr_[j*3+2] = off[2] + jp[2];
            s_global_mhr_[j]     = s_local;
        }
        else
        {
            qmul(&q_global_mhr_[j*4], &q_global_mhr_[p*4], &q_local_[j*4]);
            float local_off[3] = { off[0] + jp[0], off[1] + jp[1], off[2] + jp[2] };
            float rt[3];
            qrot(rt, &q_global_mhr_[p*4], local_off);
            const float sp = s_global_mhr_[p];
            t_global_mhr_[j*3+0] = t_global_mhr_[p*3+0] + sp * rt[0];
            t_global_mhr_[j*3+1] = t_global_mhr_[p*3+1] + sp * rt[1];
            t_global_mhr_[j*3+2] = t_global_mhr_[p*3+2] + sp * rt[2];
            s_global_mhr_[j]     = sp * s_local;
        }
    }
}

void State::joint_locals(const fsb::MHRResult& r, std::vector<JointLocal>& out) const
{
    out.clear();
    out.reserve(n_joints_);
    for (int j = 0; j < n_joints_; ++j)
    {
        JointLocal jl;
        jl.name = (j < mhr_joint_table::N_JOINTS) ? mhr_joint_table::NAMES[j] : "?";
        const int p = lbs_->joint_parents[j];

        if (p < 0)
        {
            // Root: placed in the camera frame. Rotation = the body's global
            // orientation; translation = pred_cam_t (already metres).
            jl.parent = "";
            memcpy(jl.q, &q_global_mhr_[j*4], 4*sizeof(float));
            jl.t[0] = r.pred_cam_t[0];
            jl.t[1] = r.pred_cam_t[1];
            jl.t[2] = r.pred_cam_t[2];
        }
        else
        {
            jl.parent = (p < mhr_joint_table::N_JOINTS) ? mhr_joint_table::NAMES[p] : "?";
            // Parent-relative rotation is the raw MHR local rotation (xyzw).
            memcpy(jl.q, &q_local_[j*4], 4*sizeof(float));
            // Parent-relative translation = R_parent_global^-1 (t[j]-t[p]);
            // this absorbs the per-joint scale exactly. cm -> m.
            float d[3] = { t_global_mhr_[j*3+0] - t_global_mhr_[p*3+0],
                           t_global_mhr_[j*3+1] - t_global_mhr_[p*3+1],
                           t_global_mhr_[j*3+2] - t_global_mhr_[p*3+2] };
            float qpc[4]; qconj(qpc, &q_global_mhr_[p*4]);
            float rl[3];  qrot(rl, qpc, d);
            jl.t[0] = rl[0] * 0.01f;
            jl.t[1] = rl[1] * 0.01f;
            jl.t[2] = rl[2] * 0.01f;
        }
        out.push_back(jl);
    }
}

void State::rest_local_bone_vector(int self, int ancestor, float out[3]) const
{
    float dv_world[3] =
    {
        t_global_mhr_[self*3+0] - t_global_mhr_[ancestor*3+0],
        t_global_mhr_[self*3+1] - t_global_mhr_[ancestor*3+1],
        t_global_mhr_[self*3+2] - t_global_mhr_[ancestor*3+2],
    };
    float inv_cur[4];
    qconj(inv_cur, &q_global_mhr_[ancestor*4]);
    float dv_local[3];
    qrot(dv_local, inv_cur, dv_world);
    qrot(out, &q_global_mhr_rest_[ancestor*4], dv_local);
}

} // namespace mhr_fk
