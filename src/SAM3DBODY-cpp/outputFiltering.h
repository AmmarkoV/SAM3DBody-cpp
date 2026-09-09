#ifndef _BUTTERWORTH_FILTER_H_INCLUDED
#define _BUTTERWORTH_FILTER_H_INCLUDED

/** @file outputFiltering.hpp
*   @brief From Wikipedia :  The Butterworth filter is a type of signal processing filter designed to have a frequency response as flat as possible in the passband. 
*   It is also referred to as a maximally flat magnitude filter.
*   It was first described in 1930 by the British engineer and physicist Stephen Butterworth in his paper entitled "On the Theory of Filter Amplifiers"
*   https://en.wikipedia.org/wiki/Butterworth_filter
* 
*   The frequency response of the Butterworth filter is maximally flat (i.e. has no ripples) in the passband and rolls off towards zero in the stopband.
*   That's why it is used as a post-processing step if you don't disable it from the GUI. It should be noted that this is a relatively new addition to the codebase ( 30-10-2019 ) 
*   the original BMVC 2019 paper  ( https://www.youtube.com/watch?v=fH5e-KMBvM0 ) did not have any post processing done..!
*
*   However some sort of filtering had to be added after numerous comments regarding signal noise. And here it is, in a header-only vanilla C compatible version. 
*   Thanks to Stelios Piperakis (  https://github.com/mrsp ) for giving me the initial code implementation  that this filter is based on 
*   
*   @author Ammar Qammaz (AmmarkoV)
*/


#include <math.h>

/* ────────────────────────────────────────────────────────────────────────────
 *  QuatLPF — 1st-order SLERP-based EMA on unit quaternions (XYZW storage).
 *
 *  Why a separate filter, instead of running the scalar Butterworth above on
 *  the four channels of a quaternion (or three of an Euler triple):
 *
 *     • SO(3) is a curved 3-manifold, not a vector space.  The maximally-flat
 *       magnitude property that Butterworth gives is a guarantee about a
 *       scalar LTI system — it does not transfer to rotations.
 *     • Euler angles wrap at ±π.  A per-axis low-pass interpolates linearly
 *       through that discontinuity, which produces a visible body-flip for
 *       ~τ seconds after every wrap.
 *     • Unit quaternions q and −q represent the same orientation.  Without a
 *       hemisphere correction the four channels can flip sign frame-to-frame
 *       and any linear filter on them produces nonsense.
 *
 *  What this filter guarantees instead:
 *
 *     • Geodesic monotonicity — the output never moves further from the
 *       previous output than the input is.  (SLERP is a shortest-arc.)
 *     • No-flip continuity over any continuous SO(3) trajectory.
 *     • A single, frame-rate-independent distance metric (the rotation angle
 *       2·acos|dot(q1,q2)| in radians).
 *     • Exponential convergence to a constant input at time-constant
 *       τ = 1/(2π·fc)  — i.e. the same effective cutoff knob as the linear
 *       Butterworth, so --bw-cutoff continues to mean the same thing.
 *
 *  What it does NOT give you:
 *
 *     • Maximally-flat magnitude response: it's 1st-order (−6 dB/octave),
 *       not the 2nd-order Butterworth shape used on the linear channels.
 *       This is the *correct* tradeoff for rotations — see the README note.
 *
 *  Use it ONLY for the root-joint global rotation.  Per-joint body pose and
 *  hand pose Euler channels are still small enough per frame that the scalar
 *  Butterworth there works fine.
 *
 *  Storage convention: quaternions are XYZW everywhere (matches bvh_writer.cpp
 *  and model_loader_transform_joints.c).
 * ─────────────────────────────────────────────────────────────────────────── */
struct QuatLPF
{
    float q[4];          /* current filtered orientation, XYZW unit quat */
    float alpha;         /* EMA step in (0,1]; α=1 ⇒ pass-through */
    char  initialized;
};

static void init_quat_lpf(struct QuatLPF *qf, float fsampling, float fcutoff)
{
    float dt = 1.0f / fsampling;
    float w  = 2.0f * 3.14159265359f * fcutoff * dt;
    qf->alpha       = w / (1.0f + w);          /* bilinear-style 1st-order α */
    qf->initialized = 0;
    qf->q[0] = qf->q[1] = qf->q[2] = 0.0f;
    qf->q[3] = 1.0f;
}

/* Filter a single input quaternion in_q (XYZW) and write the smoothed
 * quaternion to out_q.
 *
 *   max_step_rad : geodesic outlier clamp.  If the input is more than this
 *                  many radians of rotation away from the previous filtered
 *                  orientation, the SLERP fraction is reduced so the output
 *                  moves by at most max_step_rad in this step.  Pass 0 to
 *                  disable the clamp (pure EMA).
 *
 * Implementation:
 *   1. Hemisphere correction — pick the sign of in_q so dot(qf->q, in_q) ≥ 0.
 *   2. Compute the geodesic angle θ = 2·acos(|dot|).
 *   3. Pick a SLERP fraction t = min(α, max_step_rad / θ).
 *   4. SLERP from qf->q toward in_q by t; renormalise; store + return.
 */
static void filter_quat(struct QuatLPF *qf,
                        const float in_q[4],
                        float max_step_rad,
                        float out_q[4])
{
    /* First sample: jam the filter state to the input so we don't start
     * SLERPing from the identity quaternion (which would give a long warm-up
     * during which the body slowly rotates from rest pose into its real
     * orientation). */
    if (!qf->initialized)
    {
        float n = sqrtf(in_q[0]*in_q[0] + in_q[1]*in_q[1] +
                        in_q[2]*in_q[2] + in_q[3]*in_q[3]);
        if (n < 1e-9f) n = 1.0f;
        for (int i = 0; i < 4; ++i) qf->q[i] = in_q[i] / n;
        qf->initialized = 1;
        for (int i = 0; i < 4; ++i) out_q[i] = qf->q[i];
        return;
    }

    /* Hemisphere correction so q and −q are not treated as a 180° flip. */
    float dot = qf->q[0]*in_q[0] + qf->q[1]*in_q[1] +
                qf->q[2]*in_q[2] + qf->q[3]*in_q[3];
    float corr[4];
    if (dot < 0.0f) {
        corr[0] = -in_q[0]; corr[1] = -in_q[1];
        corr[2] = -in_q[2]; corr[3] = -in_q[3];
        dot = -dot;
    } else {
        corr[0] = in_q[0]; corr[1] = in_q[1];
        corr[2] = in_q[2]; corr[3] = in_q[3];
    }

    /* Choose SLERP fraction.  Geodesic angle (full rotation): θ = 2·acos(dot). */
    float t = qf->alpha;
    if (max_step_rad > 0.0f)
    {
        float dot_clamped = dot > 1.0f ? 1.0f : (dot < -1.0f ? -1.0f : dot);
        float theta_geo   = 2.0f * acosf(dot_clamped);
        if (theta_geo > max_step_rad)
        {
            float t_clamp = max_step_rad / theta_geo;
            if (t_clamp < t) t = t_clamp;
        }
    }

    /* SLERP.  For nearly-aligned quaternions sin(Ω) → 0; fall back to LERP
     * + renormalise to avoid the division blowing up. */
    float omega       = acosf(dot > 1.0f ? 1.0f : (dot < -1.0f ? -1.0f : dot));
    float sin_omega   = sinf(omega);
    float a, b;
    if (sin_omega < 1e-6f) {
        a = 1.0f - t;
        b = t;
    } else {
        a = sinf((1.0f - t) * omega) / sin_omega;
        b = sinf( t          * omega) / sin_omega;
    }
    float nx = a * qf->q[0] + b * corr[0];
    float ny = a * qf->q[1] + b * corr[1];
    float nz = a * qf->q[2] + b * corr[2];
    float nw = a * qf->q[3] + b * corr[3];
    float n = sqrtf(nx*nx + ny*ny + nz*nz + nw*nw);
    if (n < 1e-9f) n = 1.0f;
    qf->q[0] = nx/n; qf->q[1] = ny/n; qf->q[2] = nz/n; qf->q[3] = nw/n;
    out_q[0] = qf->q[0]; out_q[1] = qf->q[1]; out_q[2] = qf->q[2]; out_q[3] = qf->q[3];
}

/* ─── Euler ↔ Quaternion helpers ─────────────────────────────────────────────
 *
 * MHR uses the ZYX intrinsic convention (R = Rz · Ry · Rx applied to a
 * column vector).  `global_rot` is stored as [rz, ry, rx] (see
 * fast_sam_3dbody.cpp where `r.global_rot = { ge[2], ge[1], ge[0] }`).
 *
 * These helpers keep the conversion in one place so the filter never sees
 * an Euler triple in the wrong order.
 */
static void euler_zyx_to_quat(float rz, float ry, float rx, float out_q[4])
{
    float hz = 0.5f * rz, hy = 0.5f * ry, hx = 0.5f * rx;
    float cz = cosf(hz), sz = sinf(hz);
    float cy = cosf(hy), sy = sinf(hy);
    float cx = cosf(hx), sx = sinf(hx);
    /* q = qz * qy * qx, Hamilton product, XYZW */
    out_q[0] = cz*cy*sx - sz*sy*cx;
    out_q[1] = cz*sy*cx + sz*cy*sx;
    out_q[2] = sz*cy*cx - cz*sy*sx;
    out_q[3] = cz*cy*cx + sz*sy*sx;
}

/* Inverse of the above: extract (rz, ry, rx) from a unit quaternion XYZW
 * representing R = Rz · Ry · Rx.
 *
 *   R[2][0] = -sin(ry)
 *   R[2][1] =  cos(ry)·sin(rx)
 *   R[2][2] =  cos(ry)·cos(rx)
 *   R[1][0] =  cos(ry)·sin(rz)
 *   R[0][0] =  cos(ry)·cos(rz)
 *
 * Near the gimbal-lock pole (|sin ry| → 1) cos(ry) → 0 so rx and rz cannot be
 * separated; we lock rz=0 and solve for rx alone — the same convention used
 * in bvh_writer.cpp.
 */
static void quat_to_euler_zyx(const float q[4], float *rz, float *ry, float *rx)
{
    float x = q[0], y = q[1], z = q[2], w = q[3];
    /* Rotation matrix entries we actually need: */
    float r20 = 2.0f*(x*z - w*y);     /* R[2][0] = -sin(ry)            */
    float r21 = 2.0f*(y*z + w*x);     /* R[2][1] = cos(ry)·sin(rx)     */
    float r22 = 1.0f - 2.0f*(x*x + y*y); /* R[2][2] = cos(ry)·cos(rx) */
    float r10 = 2.0f*(x*y + w*z);     /* R[1][0] = cos(ry)·sin(rz)     */
    float r00 = 1.0f - 2.0f*(y*y + z*z); /* R[0][0] = cos(ry)·cos(rz) */

    float s_ry = -r20;
    if (s_ry >  1.0f) s_ry =  1.0f;
    if (s_ry < -1.0f) s_ry = -1.0f;
    *ry = asinf(s_ry);

    if (fabsf(r20) < 0.99999f) {
        *rx = atan2f(r21, r22);
        *rz = atan2f(r10, r00);
    } else {
        /* Gimbal-lock: pick rz=0, solve rx */
        *rz = 0.0f;
        *rx = atan2f(-2.0f*(y*z - w*x), 1.0f - 2.0f*(x*x + z*z));
    }
}

/**
 * @brief The complete state of a Butterworth filter instance  
 */
struct ButterWorth
{
  //https://en.wikipedia.org/wiki/Butterworth_filter
  //https://github.com/mrsp/serow/blob/master/src/butterworthLPF.cpp
  float unfilteredValue;
  float filteredValue;
  //-----------
  char initialized;
  //-----------
  float  a;
  float  fx;
  float  fs;
  float  a1;
  float  a2;
  float  b0;
  float  b1;
  float  b2;
  float  ff;
  float  ita;
  float  q;
  int  i;
  float  y_p;
  float  y_pp;
  float  x_p;
  float  x_pp;
};

/**
 * @brief Initialize a "sensor" using  fsampling/fcutoff values
 * @param Butterworth filter instance
 * @param frequency of sampling
 * @param frequency of cutoff
 */
static void initButterWorth(struct ButterWorth * sensor,float fsampling,float fcutoff)
{
    sensor->fs = fsampling;
    sensor->fx = fcutoff;

    sensor->i   = 0;
    sensor->ff  = (float) sensor->fx/sensor->fs;
    sensor->ita = (float) 1.0/tan((float) 3.14159265359 * sensor->ff);
    sensor->q   = 1.41421356237;
    sensor->b0  = (float) 1.0 / (1.0 + sensor->q*sensor->ita + sensor->ita*sensor->ita);
    sensor->b1  = 2*sensor->b0;
    sensor->b2  = sensor->b0;
    sensor->a1  = 2.0 * (sensor->ita*sensor->ita - 1.0) * sensor->b0;
    sensor->a2  = -(1.0 - sensor->q*sensor->ita + sensor->ita*sensor->ita) * sensor->b0;
    sensor->a   =(float) (2.0*3.14159265359*sensor->ff)/(2.0*3.14159265359*sensor->ff+1.0); 
}


/**
 * @brief Filter a new incoming value and get the result
 * @param Butterworth filter instance
 * @param Unfiltered input value
 * @retval Filtered output value
 */
static float filter(struct ButterWorth * sensor,float unfilteredValue)
{
 sensor->unfilteredValue = unfilteredValue;   
 
 float y = sensor->unfilteredValue; 
 float out;
 if ((sensor->i>2)&&(1))
       {
        out = sensor->b0 * y + sensor->b1 * sensor->y_p + sensor->b2* sensor->y_pp + sensor->a1 * sensor->x_p + sensor->a2 * sensor->x_pp;
       }
    else
       {
        out = sensor->x_p + sensor->a * (y - sensor->x_p);
        sensor->i=sensor->i+1;
       }
     
    sensor->y_pp = sensor->y_p;
    sensor->y_p = y;
    sensor->x_pp = sensor->x_p;
    sensor->x_p = out;
    
    sensor->filteredValue = out;
    
    if (!sensor->initialized)
    {
        //Do a warmup..
        //Make sure we dont start from 0
        
        sensor->initialized=1;
        for (unsigned int i=0; i<5; i++)
        {
          filter(sensor,unfilteredValue);            
        }

         return filter(sensor,unfilteredValue);
    }
    
    return out; 
}

/* ────────────────────────────────────────────────────────────────────────────
 *  ButterWorthWrap — 2nd-order Butterworth with an optional ±π wrap-correct
 *                    pre-integrator for filtering Euler-angle channels.
 *
 *  The renderer used to carry a separate `BwFilter` class with this logic
 *  baked in.  It produced output identical to running the scalar
 *  ButterWorth above on a wrap-aware unwrapped accumulator, so we lift the
 *  unwrap into a thin shim around the existing filter() rather than
 *  duplicating coefficient math.
 *
 *  When wrap_input = 0:
 *      Equivalent to filter(&w->bw, x).
 *
 *  When wrap_input = 1:
 *      Track the previous input.  Each frame, compute (x − prev_in),
 *      wrap it into [-π, π], add to a running accumulator, filter the
 *      accumulator instead of the raw input.  Result is a smoothed
 *      *unwrapped* Euler value (may exceed ±π) which preserves SO(3)
 *      continuity through wraps.  Callers that need wrapped output can
 *      remod themselves; most pipelines (BVH export, mhr_lbs_compute) are
 *      perfectly fine with unwrapped Euler.
 *
 *  Use cases:
 *      * Euler-channel batches like `mhr_model_params[204]` where some
 *        sub-ranges are rotations and others are translations — the
 *        wrap is a no-op for translations (their per-frame deltas are
 *        << π), so wrap_input = 1 for the whole bank is safe.
 *      * pred_cam_t / keypoints_3d, where wrap_input = 0 keeps the
 *        filter strictly equivalent to the scalar ButterWorth path.
 * ─────────────────────────────────────────────────────────────────────────── */
struct ButterWorthWrap
{
    struct ButterWorth bw;
    float acc;             /* unwrapped accumulator, valid only when wrap_input=1 */
    float prev_in;         /* previous raw input, used to compute the wrapped delta */
    char  wrap_input;      /* 0 = bypass unwrap; behave as a plain ButterWorth */
    char  wrap_init;       /* 1 once acc/prev_in have been seeded to the first sample */
};

/* Not `static`: `inline` so TUs that include the header but never call
 * these helpers don't trigger -Wunused-function warnings. */
inline void init_butterworth_wrap(struct ButterWorthWrap* w,
                                  float fsampling, float fcutoff,
                                  int wrap_input)
{
    initButterWorth(&w->bw, fsampling, fcutoff);
    w->acc        = 0.f;
    w->prev_in    = 0.f;
    w->wrap_input = wrap_input ? 1 : 0;
    w->wrap_init  = 0;
}

inline float filter_wrap(struct ButterWorthWrap* w, float x)
{
    if (!w->wrap_input) return filter(&w->bw, x);

    /* First sample: seed the unwrapped accumulator to the input.  Subsequent
     * samples integrate wrapped deltas into it, which is mathematically
     * identical to the renderer's old BwFilter::apply but routed through the
     * shared scalar filter primitive. */
    if (!w->wrap_init) {
        w->acc       = x;
        w->prev_in   = x;
        w->wrap_init = 1;
        return filter(&w->bw, x);   /* acc == x on the very first call */
    }
    float delta = x - w->prev_in;
    while (delta >  3.14159265359f) delta -= 6.28318530718f;
    while (delta < -3.14159265359f) delta += 6.28318530718f;
    w->acc    += delta;
    w->prev_in = x;
    return filter(&w->bw, w->acc);
}

#endif
