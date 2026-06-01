// bvh_writer.cpp — multi-person BVH export driven by MHR pose output.
//
// Pipeline overview
// ─────────────────
//  open(template, out_path, ...):
//    bvh_loadBVH(template) → mc_   (we use *only* its hierarchy)
//    mhr_lbs_load(lbs)    → lbs_   (PT, prerotations, parents, offsets)
//    build the per-slot mapping table (BVH name → MHR index, nearest ancestor)
//    precompute MHR rest-pose joint quaternions
//
//  write_frame(results):
//    run the 2D-bbox tracker → vector<track_id> aligned with results
//    spawn a new PerPerson for previously-unseen track ids
//    for every active track, append one frame row to that person's motion
//      buffer (real MHR output when matched this frame, a duplicate of the
//      last frame when within the disappear tolerance)
//    increment session_frames_
//
//  close():
//    for each PerPerson with frames:
//      rewrite their BVH OFFSETs from the median per-bone length they saw
//      transplant their motion buffer into mc_ → dumpBVHToBVH("base_<id>.bvh")
//
// Single-person inputs still produce one BVH file (id = 0).

#include "bvh_writer.h"
#include "fast_sam_3dbody.h"
#include "mhr_joint_table.h"

extern "C" {
#include "ModelLoader/model_loader_transform_joints.h"
#include "bvh_loader.h"
#include "export/bvh_to_bvh.h"
#include "calculate/bvh_transform.h"
}

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{

constexpr float RAD2DEG   = 57.29577951308232f;
constexpr float POS_SCALE = 100.0f;             // metres → centimetres
constexpr float TRACK_IOU_THRESH   = 0.10f;
constexpr int   TRACK_MAX_MISSING  = 90;        // 3 s at 30 fps

// Reject obviously-broken YOLO bboxes that anchor at the image origin
// (we've seen low-confidence frames where the bbox is the whole image).
inline bool bbox_looks_valid(const std::array<float,4>& b)
{
    const float w = b[2] - b[0];
    const float h = b[3] - b[1];
    if (w < 8.f || h < 8.f) return false;
    if (b[0] == 0.f && b[1] == 0.f) return false;
    return true;
}

// ─── quaternion helpers (XYZW) ──────────────────────────────────────────────
inline void qmul(float* r, const float* a, const float* b)
{
    r[0] = b[0]*a[3] + b[3]*a[0] + b[2]*a[1] - b[1]*a[2];
    r[1] = b[1]*a[3] - b[2]*a[0] + b[3]*a[1] + b[0]*a[2];
    r[2] = b[2]*a[3] + b[1]*a[0] - b[0]*a[1] + b[3]*a[2];
    r[3] = b[3]*a[3] - b[0]*a[0] - b[1]*a[1] - b[2]*a[2];
}
inline void qconj(float* r, const float* a)
{
    r[0]=-a[0];
    r[1]=-a[1];
    r[2]=-a[2];
    r[3]=a[3];
}
inline void qrot(float* o, const float* q, const float* v)
{
    float qx=q[0], qy=q[1], qz=q[2], qw=q[3], vx=v[0], vy=v[1], vz=v[2];
    float tx = 2.f*(qy*vz - qz*vy);
    float ty = 2.f*(qz*vx - qx*vz);
    float tz = 2.f*(qx*vy - qy*vx);
    o[0]=vx + qw*tx + (qy*tz - qz*ty);
    o[1]=vy + qw*ty + (qz*tx - qx*tz);
    o[2]=vz + qw*tz + (qx*ty - qy*tx);
}
// Shortest-arc unit quaternion rotating direction `from` onto `to` (both need
// not be normalised).  Used by the rest-frame retarget to re-aim MHR rotation
// deltas onto the template's (differently-posed) bones.
inline void shortest_arc_quat(const float* from, const float* to, float* q)
{
    float fn = std::sqrt(from[0]*from[0]+from[1]*from[1]+from[2]*from[2]);
    float tn = std::sqrt(to[0]*to[0]+to[1]*to[1]+to[2]*to[2]);
    q[0]=q[1]=q[2]=0.f; q[3]=1.f;                       // identity fallback
    if (fn < 1e-9f || tn < 1e-9f) return;
    float u[3]={from[0]/fn,from[1]/fn,from[2]/fn};
    float v[3]={to[0]/tn,  to[1]/tn,  to[2]/tn};
    float d = u[0]*v[0]+u[1]*v[1]+u[2]*v[2];
    if (d >  0.999999f) return;                         // already aligned
    if (d < -0.999999f) {                               // antiparallel: 180°
        // pick any axis perpendicular to u
        float ax[3] = {1.f,0.f,0.f};
        if (std::fabs(u[0]) >= 0.9f) { ax[0]=0.f; ax[1]=1.f; }
        float c[3]={u[1]*ax[2]-u[2]*ax[1], u[2]*ax[0]-u[0]*ax[2], u[0]*ax[1]-u[1]*ax[0]};
        float cn=std::sqrt(c[0]*c[0]+c[1]*c[1]+c[2]*c[2])+1e-12f;
        q[0]=c[0]/cn; q[1]=c[1]/cn; q[2]=c[2]/cn; q[3]=0.f;
        return;
    }
    float c[3]={u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]};
    q[0]=c[0]; q[1]=c[1]; q[2]=c[2]; q[3]=1.f+d;
    float n=std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    q[0]/=n; q[1]/=n; q[2]/=n; q[3]/=n;
}
inline void euler_mhr_to_quat(float ex, float ey, float ez, float* q)
{
    float hx=ex*0.5f, hy=ey*0.5f, hz=ez*0.5f;
    float qx[4]= {sinf(hx),0.f,0.f,cosf(hx)};
    float qy[4]= {0.f,sinf(hy),0.f,cosf(hy)};
    float qz[4]= {0.f,0.f,sinf(hz),cosf(hz)};
    float t[4];
    qmul(t, qz, qy);
    qmul(q, t, qx);
}
inline void quat_to_mat3(const float* q, float m[9])
{
    float x=q[0], y=q[1], z=q[2], w=q[3];
    float xx=x*x, yy=y*y, zz=z*z, xy=x*y, xz=x*z, yz=y*z, wx=w*x, wy=w*y, wz=w*z;
    m[0]=1-2*(yy+zz);
    m[1]=2*(xy-wz);
    m[2]=2*(xz+wy);
    m[3]=2*(xy+wz);
    m[4]=1-2*(xx+zz);
    m[5]=2*(yz-wx);
    m[6]=2*(xz-wy);
    m[7]=2*(yz+wx);
    m[8]=1-2*(xx+yy);
}
inline void mat3_to_zxy(const float m[9], float& a, float& b, float& c)
{
    float s = std::max(-1.f, std::min(1.f, m[7]));
    b = asinf(s);
    if (fabsf(m[7]) < 0.99999f)
    {
        c = atan2f(-m[6], m[8]);
        a = atan2f(-m[1], m[4]);
    }
    else
    {
        c = 0.f;
        a = atan2f(m[3], m[0]);
    }
}
inline void mat3_to_zyx(const float m[9], float& a, float& b, float& c)
{
    float s = std::max(-1.f, std::min(1.f, -m[6]));
    b = asinf(s);
    if (fabsf(m[6]) < 0.99999f)
    {
        c = atan2f(m[7], m[8]);
        a = atan2f(m[3], m[0]);
    }
    else
    {
        c = 0.f;
        a = atan2f(-m[1], m[4]);
    }
}

// ─── small vec3 / mat3 helpers (foot-contact IK) ───────────────────────────
inline void v3sub(const float* a, const float* b, float* o){ o[0]=a[0]-b[0]; o[1]=a[1]-b[1]; o[2]=a[2]-b[2]; }
inline float v3dot(const float* a, const float* b){ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
inline float v3len(const float* a){ return std::sqrt(v3dot(a,a)); }
inline void  v3cross(const float* a, const float* b, float* o){
    o[0]=a[1]*b[2]-a[2]*b[1]; o[1]=a[2]*b[0]-a[0]*b[2]; o[2]=a[0]*b[1]-a[1]*b[0];
}
inline void v3norm(float* a){ float n=v3len(a); if(n>1e-12f){ a[0]/=n; a[1]/=n; a[2]/=n; } }

// Upper-left 3×3 rotation of a row-major 4×4 (Matrix4x4OfFloats.m), in the same
// row-major 9-float layout quat_to_mat3 / mat3_to_zxy use.
inline void mat3_from_m4(const float* m16, float* m9){
    m9[0]=m16[0]; m9[1]=m16[1]; m9[2]=m16[2];
    m9[3]=m16[4]; m9[4]=m16[5]; m9[5]=m16[6];
    m9[6]=m16[8]; m9[7]=m16[9]; m9[8]=m16[10];
}
// Rotation 3×3 → unit quaternion (XYZW), Shepperd's method.
inline void mat3_to_quat(const float* m, float* q){
    float tr = m[0]+m[4]+m[8];
    if (tr > 0.f){
        float s = std::sqrt(tr+1.f)*2.f;            // s = 4w
        q[3]=0.25f*s; q[0]=(m[7]-m[5])/s; q[1]=(m[2]-m[6])/s; q[2]=(m[3]-m[1])/s;
    } else if (m[0]>m[4] && m[0]>m[8]){
        float s = std::sqrt(1.f+m[0]-m[4]-m[8])*2.f; // s = 4x
        q[3]=(m[7]-m[5])/s; q[0]=0.25f*s; q[1]=(m[1]+m[3])/s; q[2]=(m[2]+m[6])/s;
    } else if (m[4]>m[8]){
        float s = std::sqrt(1.f+m[4]-m[0]-m[8])*2.f; // s = 4y
        q[3]=(m[2]-m[6])/s; q[0]=(m[1]+m[3])/s; q[1]=0.25f*s; q[2]=(m[5]+m[7])/s;
    } else {
        float s = std::sqrt(1.f+m[8]-m[0]-m[4])*2.f; // s = 4z
        q[3]=(m[3]-m[1])/s; q[0]=(m[2]+m[6])/s; q[1]=(m[5]+m[7])/s; q[2]=0.25f*s;
    }
    float n=std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    if(n>1e-12f){ q[0]/=n; q[1]/=n; q[2]/=n; q[3]/=n; }
}
inline float median_of(std::vector<float> v){
    if (v.empty()) return 0.f;
    std::sort(v.begin(), v.end());
    return v[v.size()/2];
}

// ─── Explicit BVH-name → MHR-name mapping ──────────────────────────────────
struct NameMap
{
    const char* bvh;
    const char* mhr;
    BVHWriter::SlotKind kind;
};
constexpr BVHWriter::SlotKind BODY = BVHWriter::SlotKind::Body;
constexpr BVHWriter::SlotKind HAND = BVHWriter::SlotKind::Hand;

constexpr NameMap NAME_MAP[] =
{
    { "hip","root",BODY }, { "abdomen","c_spine1",BODY }, { "chest","c_spine3",BODY },
    { "neck","c_neck",BODY }, { "head","c_head",BODY }, { "jaw","c_jaw",BODY },

    { "lCollar","l_clavicle",BODY }, { "lShldr","l_uparm",BODY },
    { "lForeArm","l_lowarm",BODY }, { "lHand","l_wrist",BODY },

    { "rCollar","r_clavicle",BODY }, { "rShldr","r_uparm",BODY },
    { "rForeArm","r_lowarm",BODY }, { "rHand","r_wrist",BODY },

    { "lThigh","l_upleg",BODY }, { "lShin","l_lowleg",BODY }, { "lFoot","l_foot",BODY },
    { "rThigh","r_upleg",BODY }, { "rShin","r_lowleg",BODY }, { "rFoot","r_foot",BODY },

    // Left hand fingers (BVH finger2..5 = index/middle/ring/pinky; finger1 = thumb).
    { "lthumb",      "l_thumb1",HAND }, { "finger1-2.l","l_thumb2",HAND }, { "finger1-3.l","l_thumb3",HAND },
    { "finger2-1.l","l_index1",HAND }, { "finger2-2.l","l_index2",HAND }, { "finger2-3.l","l_index3",HAND },
    { "finger3-1.l","l_middle1",HAND}, { "finger3-2.l","l_middle2",HAND}, { "finger3-3.l","l_middle3",HAND},
    { "finger4-1.l","l_ring1",HAND  }, { "finger4-2.l","l_ring2",HAND  }, { "finger4-3.l","l_ring3",HAND  },
    { "finger5-1.l","l_pinky1",HAND }, { "finger5-2.l","l_pinky2",HAND }, { "finger5-3.l","l_pinky3",HAND },

    // Right hand fingers
    { "rthumb",      "r_thumb1",HAND }, { "finger1-2.r","r_thumb2",HAND }, { "finger1-3.r","r_thumb3",HAND },
    { "finger2-1.r","r_index1",HAND }, { "finger2-2.r","r_index2",HAND }, { "finger2-3.r","r_index3",HAND },
    { "finger3-1.r","r_middle1",HAND}, { "finger3-2.r","r_middle2",HAND}, { "finger3-3.r","r_middle3",HAND},
    { "finger4-1.r","r_ring1",HAND  }, { "finger4-2.r","r_ring2",HAND  }, { "finger4-3.r","r_ring3",HAND  },
    { "finger5-1.r","r_pinky1",HAND }, { "finger5-2.r","r_pinky2",HAND }, { "finger5-3.r","r_pinky3",HAND },
};

// ─── Hand joint angle limits (--enforce-hand-limits) ───────────────────────
// Per-BVH-joint-name limits on the ZXY rotation channels written to the BVH.
// Channel order matches the template: Zrotation Xrotation Yrotation.
//
// Derived from MocapNET's createRandomHandDataset.sh natural-hand pose and
// the observed natural range-of-motion of human finger joints.  Applied when
// the BVHWriter::enforce_hand_limits_ flag is set.
//
// For finger proximal joints (finger[2-5]-1) the most common artefact from
// the MHR model on footage where hands are not clearly visible is a large Z
// offset (~35-40°) that makes fingers splay wide.  The ±20° Z limit fixes
// this while still allowing natural spread.
struct HandLimitEntry {
    const char* bvh_name;
    float z_min, z_max;   // Zrotation limits (degrees)
    float x_min, x_max;   // Xrotation limits (degrees)
    float y_min, y_max;   // Yrotation limits (degrees)
};

// clang-format off
static const HandLimitEntry HAND_LIMITS[] = {
    // ── proximal phalanges (knuckles) ──────────────────────────────────────
    // Z = spread/abduction  X = flex/extend  Y = axial twist
    { "finger2-1.r", -20, 20,  -5, 90,  -10, 10 },
    { "finger3-1.r", -20, 20,  -5, 90,  -10, 10 },
    { "finger4-1.r", -20, 20,  -5, 90,  -10, 10 },
    { "finger5-1.r", -22, 10,  -5, 90,  -10, 10 },  // pinky naturally adducts a bit
    { "finger2-1.l",  -20, 20,  -5, 90,  -10, 10 },
    { "finger3-1.l",  -20, 20,  -5, 90,  -10, 10 },
    { "finger4-1.l",  -20, 20,  -5, 90,  -10, 10 },
    { "finger5-1.l",  -10, 22,  -5, 90,  -10, 10 },

    // ── middle phalanges ──────────────────────────────────────────────────
    { "finger2-2.r", -10, 10,  -5, 90,  -5, 5 },
    { "finger3-2.r", -10, 10,  -5, 90,  -5, 5 },
    { "finger4-2.r", -10, 10,  -5, 90,  -5, 5 },
    { "finger5-2.r", -10, 10,  -5, 90,  -5, 5 },
    { "finger2-2.l", -10, 10,  -5, 90,  -5, 5 },
    { "finger3-2.l", -10, 10,  -5, 90,  -5, 5 },
    { "finger4-2.l", -10, 10,  -5, 90,  -5, 5 },
    { "finger5-2.l", -10, 10,  -5, 90,  -5, 5 },

    // ── distal phalanges ──────────────────────────────────────────────────
    { "finger2-3.r", -10, 10,  -5, 80,  -5, 5 },
    { "finger3-3.r", -10, 10,  -5, 80,  -5, 5 },
    { "finger4-3.r", -10, 10,  -5, 80,  -5, 5 },
    { "finger5-3.r", -10, 10,  -5, 80,  -5, 5 },
    { "finger2-3.l", -10, 10,  -5, 80,  -5, 5 },
    { "finger3-3.l", -10, 10,  -5, 80,  -5, 5 },
    { "finger4-3.l", -10, 10,  -5, 80,  -5, 5 },
    { "finger5-3.l", -10, 10,  -5, 80,  -5, 5 },

    // ── thumb (wider opposition range) ────────────────────────────────────
    { "rthumb",     -60, 60,  -60, 60,  -60, 60 },
    { "lthumb",     -60, 60,  -60, 60,  -60, 60 },
    { "finger1-2.r",-45, 45,  -45, 90,  -20, 20 },
    { "finger1-3.r",-30, 30,  -5,  80,  -10, 10 },
    { "finger1-2.l",-45, 45,  -45, 90,  -20, 20 },
    { "finger1-3.l",-30, 30,  -5,  80,  -10, 10 },
};
// clang-format on

static const HandLimitEntry* find_hand_limit(const char* name)
{
    for (const auto& e : HAND_LIMITS)
        if (strcmp(e.bvh_name, name) == 0) return &e;
    return nullptr;
}

static inline float clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

}  // namespace

// ─── Tracker ────────────────────────────────────────────────────────────────

float BVHWriter::bbox_iou(const float a[4], const float b[4])
{
    float ix1 = std::max(a[0], b[0]);
    float iy1 = std::max(a[1], b[1]);
    float ix2 = std::min(a[2], b[2]);
    float iy2 = std::min(a[3], b[3]);
    float iw = std::max(0.f, ix2 - ix1);
    float ih = std::max(0.f, iy2 - iy1);
    float inter = iw * ih;
    if (inter <= 0.f) return 0.f;
    float aa = std::max(0.f, a[2]-a[0]) * std::max(0.f, a[3]-a[1]);
    float bb = std::max(0.f, b[2]-b[0]) * std::max(0.f, b[3]-b[1]);
    float u  = aa + bb - inter;
    return u > 0.f ? inter / u : 0.f;
}

// Greedy IoU assignment.  Returns parallel vector of track_ids matched to each
// detection (creating new tracks as needed).  Tracks expired this frame are
// removed from tracks_.
std::vector<int> BVHWriter::assign_tracks(const std::vector<fsb::MHRResult>& results)
{
    const int F = session_frames_;
    std::vector<int> result_ids(results.size(), -1);

    // Score every (detection, track) pair, sort descending by IoU.
    struct Pair
    {
        int det;
        int track;
        float iou;
    };
    std::vector<Pair> pairs;
    pairs.reserve(results.size() * std::max((size_t)1, tracks_.size()));
    for (size_t d = 0; d < results.size(); ++d)
    {
        float db[4] = { results[d].bbox[0], results[d].bbox[1],
                        results[d].bbox[2], results[d].bbox[3]
                      };
        for (size_t t = 0; t < tracks_.size(); ++t)
        {
            float v = bbox_iou(db, tracks_[t].bbox);
            if (v >= TRACK_IOU_THRESH) pairs.push_back({(int)d, (int)t, v});
        }
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const Pair& a, const Pair& b)
    {
        return a.iou > b.iou;
    });

    std::vector<char> det_taken(results.size(), 0);
    std::vector<char> track_taken(tracks_.size(), 0);
    for (const auto& p : pairs)
    {
        if (det_taken[p.det] || track_taken[p.track]) continue;
        det_taken[p.det]     = 1;
        track_taken[p.track] = 1;
        result_ids[p.det]    = tracks_[p.track].id;
        const auto& bb = results[p.det].bbox;
        tracks_[p.track].bbox[0] = bb[0];
        tracks_[p.track].bbox[1] = bb[1];
        tracks_[p.track].bbox[2] = bb[2];
        tracks_[p.track].bbox[3] = bb[3];
        tracks_[p.track].last_seen_frame = F;
    }

    // Spawn new tracks for unmatched detections.
    for (size_t d = 0; d < results.size(); ++d)
    {
        if (det_taken[d]) continue;
        Track t;
        t.id = next_track_id_++;
        t.bbox[0] = results[d].bbox[0];
        t.bbox[1] = results[d].bbox[1];
        t.bbox[2] = results[d].bbox[2];
        t.bbox[3] = results[d].bbox[3];
        t.last_seen_frame = F;
        tracks_.push_back(t);
        result_ids[d] = t.id;
    }

    // Retire tracks that have been missing for too long.
    auto it = std::remove_if(tracks_.begin(), tracks_.end(),
                             [F](const Track& t)
    {
        return (F - t.last_seen_frame) > TRACK_MAX_MISSING;
    });
    tracks_.erase(it, tracks_.end());

    return result_ids;
}

// ─── MHR-side rest-pose FK (run once at open) ──────────────────────────────

bool BVHWriter::build_slots()
{
    std::unordered_map<std::string,int> bvh_name_to_id;
    for (unsigned int j = 0; j < mc_->jointHierarchySize; ++j)
    {
        const char* name = mc_->jointHierarchy[j].jointName;
        if (name && *name) bvh_name_to_id[std::string(name)] = (int)j;
        if (mc_->jointHierarchy[j].isRoot) root_bvh_jid_ = (int)j;
    }
    if (root_bvh_jid_ < 0)
    {
        fprintf(stderr, "[BVHWriter] template has no ROOT joint\n");
        return false;
    }

    std::unordered_map<std::string,int> mhr_name_to_idx;
    for (int j = 0; j < mhr_joint_table::N_JOINTS; ++j)
        mhr_name_to_idx[mhr_joint_table::NAMES[j]] = j;

    slots_.assign(mc_->jointHierarchySize, BvhSlot{});
    for (unsigned int j = 0; j < mc_->jointHierarchySize; ++j)
    {
        slots_[j].bvh_jid          = (int)j;
        slots_[j].mhr_idx          = -1;
        slots_[j].ancestor_bvh_jid = -1;
        slots_[j].ancestor_mhr_idx = -1;
        slots_[j].is_root          = mc_->jointHierarchy[j].isRoot;
    }
    int n_mapped = 0;
    for (const auto& nm : NAME_MAP)
    {
        auto bi = bvh_name_to_id.find(nm.bvh);
        if (bi == bvh_name_to_id.end()) continue;
        auto mi = mhr_name_to_idx.find(nm.mhr);
        if (mi == mhr_name_to_idx.end())
        {
            fprintf(stderr, "[BVHWriter] mapping target '%s' not in MHR table\n", nm.mhr);
            continue;
        }
        slots_[bi->second].mhr_idx = mi->second;
        slots_[bi->second].kind    = nm.kind;
        ++n_mapped;
    }

    for (unsigned int j = 0; j < mc_->jointHierarchySize; ++j)
    {
        if (slots_[j].mhr_idx < 0)        continue;
        if (mc_->jointHierarchy[j].isRoot) continue;
        int p = (int)mc_->jointHierarchy[j].parentJoint;
        while (p >= 0)
        {
            if (slots_[p].mhr_idx >= 0)
            {
                slots_[j].ancestor_bvh_jid = p;
                slots_[j].ancestor_mhr_idx = slots_[p].mhr_idx;
                break;
            }
            if (mc_->jointHierarchy[p].isRoot) break;
            p = (int)mc_->jointHierarchy[p].parentJoint;
        }
    }

    fprintf(stderr, "[BVHWriter] mapped %d BVH joints to MHR; root jID=%d\n",
            n_mapped, root_bvh_jid_);
    return n_mapped > 0;
}

bool BVHWriter::open(const std::string& template_path,
                     const std::string& out_path,
                     float              frame_time,
                     const std::string& lbs_path,
                     bool               rewrite_body_offsets,
                     bool               rewrite_hand_offsets,
                     bool               compensate_finger_endsites,
                     bool               enforce_hand_limits,
                     bool               zero_hand_pose,
                     bool               sticky_hand_pose,
                     bool               rest_align,
                     bool               dump_rest_dirs)
{
    out_path_                    = out_path;
    rewrite_body_offsets_        = rewrite_body_offsets;
    rewrite_hand_offsets_        = rewrite_hand_offsets;
    compensate_finger_endsites_  = compensate_finger_endsites;
    enforce_hand_limits_         = enforce_hand_limits;
    zero_hand_pose_              = zero_hand_pose;
    sticky_hand_pose_            = sticky_hand_pose;
    rest_align_                  = rest_align;

    mc_ = (BVH_MotionCapture*)calloc(1, sizeof(*mc_));
    if (!mc_) return false;
    if (!bvh_loadBVH(template_path.c_str(), mc_, 1.0f))
    {
        fprintf(stderr, "[BVHWriter] bvh_loadBVH('%s') failed\n", template_path.c_str());
        free(mc_);
        mc_ = nullptr;
        return false;
    }
    total_channels_ = (int)mc_->numberOfValuesPerFrame;
    frame_time_     = frame_time;

    if (mc_->motionValues)
    {
        free(mc_->motionValues);
        mc_->motionValues = nullptr;
    }
    mc_->motionValuesSize = 0;
    mc_->numberOfFrames   = 0;
    mc_->numberOfFramesEncountered = 0;

    if (lbs_path.empty())
    {
        fprintf(stderr, "[BVHWriter] lbs_path is empty — cannot match BVH joints\n");
        bvh_free(mc_);
        free(mc_);
        mc_ = nullptr;
        return false;
    }
    lbs_ = mhr_lbs_load(lbs_path.c_str());
    if (!lbs_)
    {
        fprintf(stderr, "[BVHWriter] mhr_lbs_load('%s') failed\n", lbs_path.c_str());
        bvh_free(mc_);
        free(mc_);
        mc_ = nullptr;
        return false;
    }

    if (!build_slots())
    {
        fprintf(stderr, "[BVHWriter] failed to map any BVH joint\n");
        mhr_lbs_free(lbs_);
        lbs_ = nullptr;
        bvh_free(mc_);
        free(mc_);
        mc_ = nullptr;
        return false;
    }

    // MHR rest-pose globals (re-used by every frame and every person).
    const int nj = lbs_->n_joints;
    q_global_mhr_rest_.assign((size_t)nj * 4, 0.f);
    {
        std::vector<float> g_t(nj * 3, 0.f);
        for (int j = 0; j < nj; ++j)
        {
            const float* p_q = lbs_->joint_prerotations + j*4;
            int parent = lbs_->joint_parents[j];
            if (parent < 0)
            {
                g_t[j*3+0] = lbs_->joint_offsets[j*3+0];
                g_t[j*3+1] = lbs_->joint_offsets[j*3+1];
                g_t[j*3+2] = lbs_->joint_offsets[j*3+2];
                memcpy(&q_global_mhr_rest_[j*4], p_q, 4*sizeof(float));
            }
            else
            {
                float rt[3];
                qrot(rt, &q_global_mhr_rest_[parent*4], lbs_->joint_offsets + j*3);
                g_t[j*3+0] = g_t[parent*3+0] + rt[0];
                g_t[j*3+1] = g_t[parent*3+1] + rt[1];
                g_t[j*3+2] = g_t[parent*3+2] + rt[2];
                qmul(&q_global_mhr_rest_[j*4], &q_global_mhr_rest_[parent*4], p_q);
            }
        }

        // ── Rest-frame retarget alignment (+ optional diagnostic) ─────────────
        // body.bvh's rest bone directions differ from MHR's (e.g. T-pose
        // template vs MHR's A-pose arms), so applying MHR rotation-deltas
        // verbatim aims each joint's flexion axis at the MHR bone, not the
        // template bone → bend leaks into twist.  For every mapped bone we
        // precompute the shortest-arc rotation MHR-rest-dir → body.bvh-rest-dir
        // and conjugate the per-frame local rotation by it (q_align·R·q_align⁻¹),
        // re-aiming the swing onto the template's bone.  Identity where the rest
        // directions already agree (torso, head, lower legs) → no-op there.
        // CLI flags drive this; the env vars remain as convenience overrides.
        const bool dump = dump_rest_dirs || (getenv("FSB_DUMP_REST_DIRS") != nullptr);
        if (getenv("FSB_NO_REST_ALIGN")) rest_align_ = false;
        q_bone_align_.assign(slots_.size() * 4, 0.f);
        {
            const int N = (int)mc_->jointHierarchySize;
            std::vector<std::array<float,3>> bvhrest(N);
            for (int j = 0; j < N; ++j) {
                const auto& jh = mc_->jointHierarchy[j];
                if (jh.isRoot || (int)jh.parentJoint < 0)
                    bvhrest[j] = { jh.offset[0], jh.offset[1], jh.offset[2] };
                else {
                    int par = (int)jh.parentJoint;
                    bvhrest[j] = { bvhrest[par][0] + jh.offset[0],
                                   bvhrest[par][1] + jh.offset[1],
                                   bvhrest[par][2] + jh.offset[2] };
                }
            }
            if (dump)
                fprintf(stderr,
                    "\n[rest-dir] bone                MHR-rest-dir            "
                    "BVH-rest-dir            angle\n");
            for (size_t i = 0; i < slots_.size(); ++i) {
                const BvhSlot& s = slots_[i];
                float* qa = &q_bone_align_[i*4];
                qa[0]=qa[1]=qa[2]=0.f; qa[3]=1.f;          // identity default
                if (s.is_root || s.mhr_idx < 0) continue;
                if (s.ancestor_mhr_idx < 0 || s.ancestor_bvh_jid < 0) continue;
                float md[3] = { g_t[s.mhr_idx*3+0]-g_t[s.ancestor_mhr_idx*3+0],
                                g_t[s.mhr_idx*3+1]-g_t[s.ancestor_mhr_idx*3+1],
                                g_t[s.mhr_idx*3+2]-g_t[s.ancestor_mhr_idx*3+2] };
                float bd[3] = { bvhrest[s.bvh_jid][0]-bvhrest[s.ancestor_bvh_jid][0],
                                bvhrest[s.bvh_jid][1]-bvhrest[s.ancestor_bvh_jid][1],
                                bvhrest[s.bvh_jid][2]-bvhrest[s.ancestor_bvh_jid][2] };
                shortest_arc_quat(md, bd, qa);             // MHR dir → BVH dir
                if (dump) {
                    float mn=std::sqrt(md[0]*md[0]+md[1]*md[1]+md[2]*md[2])+1e-9f;
                    float bn=std::sqrt(bd[0]*bd[0]+bd[1]*bd[1]+bd[2]*bd[2])+1e-9f;
                    float dt=(md[0]*bd[0]+md[1]*bd[1]+md[2]*bd[2])/(mn*bn);
                    dt=dt<-1.f?-1.f:(dt>1.f?1.f:dt);
                    fprintf(stderr, "[rest-dir] %-18s [%+.2f %+.2f %+.2f]  "
                            "[%+.2f %+.2f %+.2f]  %6.1f deg\n",
                            mc_->jointHierarchy[s.bvh_jid].jointName,
                            md[0]/mn,md[1]/mn,md[2]/mn, bd[0]/bn,bd[1]/bn,bd[2]/bn,
                            std::acos(dt)*57.29578f);
                }
            }
            if (dump) fprintf(stderr, "\n");
        }

        // ── Per-slot "bake" align (single-child-chain aim correction) ─────────
        // For each mapped slot, count its mapped child slots (those whose nearest
        // mapped ancestor is this slot).  If exactly one, bake that child's bone
        // alignment so this joint aims the child bone at the MHR direction; this
        // fixes the arm and lower-leg aim.  Branch joints (chest→neck+2 clavicles,
        // hip→2 thighs+spine) get identity and stay on the existing retarget, so
        // their (already-correct) downstream positions cannot regress.
        q_slot_bake_.assign(slots_.size() * 4, 0.f);
        std::vector<int> n_children(slots_.size(), 0);
        std::vector<int> only_child(slots_.size(), -1);
        for (size_t i = 0; i < slots_.size(); ++i)
        {
            const BvhSlot& s = slots_[i];
            if (s.mhr_idx < 0 || s.ancestor_bvh_jid < 0) continue;
            int a = s.ancestor_bvh_jid;              // slot index == bvh_jid
            ++n_children[a];
            only_child[a] = (int)i;
        }
        for (size_t i = 0; i < slots_.size(); ++i)
        {
            float* b = &q_slot_bake_[i*4];
            if (rest_align_ && n_children[i] == 1)
                memcpy(b, &q_bone_align_[only_child[i]*4], 4*sizeof(float));
            else { b[0]=b[1]=b[2]=0.f; b[3]=1.f; }   // identity
        }
    }

    joint_params_.assign((size_t)nj * 7, 0.f);
    q_local_.assign((size_t)nj * 4, 0.f);
    q_global_mhr_.assign((size_t)nj * 4, 0.f);
    t_global_mhr_.assign((size_t)nj * 3, 0.f);

    tracks_.clear();
    people_.clear();
    next_track_id_  = 0;
    session_frames_ = 0;

    fprintf(stderr, "[BVHWriter] template loaded: %u joints, %d channels/frame\n",
            mc_->jointHierarchySize, total_channels_);
    return true;
}

// ─── Per-frame MHR FK ──────────────────────────────────────────────────────

void BVHWriter::compute_per_frame_mhr_state(const fsb::MHRResult& r)
{
    const int nj   = lbs_->n_joints;
    const int npc  = lbs_->pt_cols;
    const float* PT = lbs_->PT;
    const float* pre = lbs_->joint_prerotations;
    const int*  parents = lbs_->joint_parents;

    const int take = std::min(npc, 204);
    for (int row = 0; row < nj * 7; ++row)
    {
        const float* prow = PT + (size_t)row * npc;
        float acc = 0.f;
        for (int k = 0; k < take; ++k) acc += prow[k] * r.mhr_model_params[k];
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
        if (p < 0)
        {
            memcpy(&q_global_mhr_[j*4], &q_local_[j*4], 4*sizeof(float));
            t_global_mhr_[j*3+0] = off[0] + jp[0];
            t_global_mhr_[j*3+1] = off[1] + jp[1];
            t_global_mhr_[j*3+2] = off[2] + jp[2];
        }
        else
        {
            qmul(&q_global_mhr_[j*4], &q_global_mhr_[p*4], &q_local_[j*4]);
            float local_off[3] = { off[0] + jp[0], off[1] + jp[1], off[2] + jp[2] };
            float rt[3];
            qrot(rt, &q_global_mhr_[p*4], local_off);
            t_global_mhr_[j*3+0] = t_global_mhr_[p*3+0] + rt[0];
            t_global_mhr_[j*3+1] = t_global_mhr_[p*3+1] + rt[1];
            t_global_mhr_[j*3+2] = t_global_mhr_[p*3+2] + rt[2];
        }
    }
}

// Write a single channel value into the in-row position `row[..]` resolved via
// the BVH library helper (which always returns an offset relative to fID=0 as
// long as numberOfFrames is set to ≥ 1 in mc_).
static inline void set_channel(BVH_MotionCapture* mc, float* row,
                               BVHJointID jID, unsigned int channelTypeID,
                               float value)
{
    unsigned int idx = bvh_resolveFrameAndJointAndChannelToMotionID(mc, jID, 0, channelTypeID);
    row[idx] = value;
}

void BVHWriter::append_frame_for(PerPerson& p, const fsb::MHRResult& r)
{
    compute_per_frame_mhr_state(r);
    append_row_from_state(p, r);
}

void BVHWriter::append_row_from_state(PerPerson& p, const fsb::MHRResult& r)
{
    // Append a new row of zeros, then fill it.
    p.motion.insert(p.motion.end(), total_channels_, 0.f);
    float* row = p.motion.data() + (size_t)p.frame_count * total_channels_;

    unsigned int saved_frames = mc_->numberOfFrames;
    float*       saved_buf    = mc_->motionValues;
    unsigned int saved_size   = mc_->motionValuesSize;
    mc_->numberOfFrames   = 1;
    mc_->motionValues     = row;
    mc_->motionValuesSize = total_channels_;

    auto delta_mhr = [&](int m, float* out)
    {
        float inv_rest[4];
        qconj(inv_rest, &q_global_mhr_rest_[m * 4]);
        qmul(out, &q_global_mhr_[m * 4], inv_rest);
    };

    // Previous frame row (for sticky hand mode).
    const float* prev_row = (p.frame_count > 0)
        ? p.motion.data() + (size_t)(p.frame_count - 1) * total_channels_
        : nullptr;

    for (size_t i = 0; i < slots_.size(); ++i)
    {
        const BvhSlot& s = slots_[i];
        if (s.mhr_idx < 0)                  continue;
        const auto& jh = mc_->jointHierarchy[s.bvh_jid];
        if (jh.isEndSite)                   continue;

        // ── Hand pose override modes ───────────────────────────────────────
        // zero_hand_pose: skip entirely — row is zero-initialised.
        if (s.kind == SlotKind::Hand && zero_hand_pose_)
            continue;

        float q_delta_self[4];
        delta_mhr(s.mhr_idx, q_delta_self);
        float q_local_bvh[4];
        if (s.is_root || s.ancestor_mhr_idx < 0)
        {
            memcpy(q_local_bvh, q_delta_self, 4*sizeof(float));
        }
        else
        {
            float q_delta_par[4];
            delta_mhr(s.ancestor_mhr_idx, q_delta_par);
            float inv_par[4];
            qconj(inv_par, q_delta_par);
            qmul(q_local_bvh, inv_par, q_delta_self);
        }

        // Rest-frame retarget (single-child-chain aim correction).
        //
        // We want each BVH bone to point where the MHR bone actually points
        // despite the rest-pose difference (body.bvh T-pose vs MHR A-pose arms).
        // A bone's direction is set by its PARENT's world rotation, so to aim the
        // bone we bake the bone's rest-alignment into the parent.  Telescoping the
        // target world rotations W[j]=Δ(m_j)·conj(A[j]) down the chain gives the
        // local rotation as a pre/post multiply on the plain delta-local:
        //     q_local = A[ancestor] · q_local · conj(A[self])
        // where A[x] = q_slot_bake_[x] is the alignment of x's single mapped child
        // (identity for branch/leaf joints).  This replaces the old self-conjugate
        // qa·R·qa⁻¹, which only rotated the delta's axis and could not bend the
        // template's horizontal arm down.  See proto_bvh_arm_retarget.py.
        if (rest_align_)
        {
            const float* A_self = &q_slot_bake_[i*4];
            const float  ident[4] = {0,0,0,1};
            const float* A_par = (s.ancestor_bvh_jid >= 0)
                               ? &q_slot_bake_[s.ancestor_bvh_jid*4] : ident;
            float inv_self[4], tmp[4], out[4];
            qconj(inv_self, A_self);
            qmul(tmp, q_local_bvh, inv_self);   // q_local · conj(A_self)
            qmul(out, A_par, tmp);              // A_par · (…)
            memcpy(q_local_bvh, out, 4*sizeof(float));
        }

        float m[9];
        quat_to_mat3(q_local_bvh, m);

        if (s.is_root && jh.hasPositionalChannels)
        {
            set_channel(mc_, row, s.bvh_jid, BVH_POSITION_X, r.pred_cam_t[0] * POS_SCALE);
            set_channel(mc_, row, s.bvh_jid, BVH_POSITION_Y, r.pred_cam_t[1] * POS_SCALE);
            set_channel(mc_, row, s.bvh_jid, BVH_POSITION_Z, r.pred_cam_t[2] * POS_SCALE);

            float a, b, c;
            mat3_to_zyx(m, a, b, c);
            set_channel(mc_, row, s.bvh_jid, BVH_ROTATION_Z, a * RAD2DEG);
            set_channel(mc_, row, s.bvh_jid, BVH_ROTATION_Y, b * RAD2DEG);
            set_channel(mc_, row, s.bvh_jid, BVH_ROTATION_X, c * RAD2DEG);
        }
        else
        {
            float a, b, c;
            mat3_to_zxy(m, a, b, c);
            float az = a * RAD2DEG, bx = b * RAD2DEG, cy = c * RAD2DEG;

            if (s.kind == SlotKind::Hand)
            {
                const HandLimitEntry* lim = nullptr;
                if (enforce_hand_limits_ || sticky_hand_pose_)
                {
                    const char* jname = mc_->jointHierarchy[s.bvh_jid].jointName;
                    lim = find_hand_limit(jname);
                }

                if (lim)
                {
                    auto within = [](float v, float lo, float hi)
                        { return v >= lo && v <= hi; };

                    if (sticky_hand_pose_)
                    {
                        // Limits-gated sticky: accept new pose only when within
                        // limits; otherwise inherit the previous frame's value.
                        // On the first frame (no prev_row) fall back to clamping.
                        auto resolve = [&](unsigned int ch) -> unsigned int {
                            return bvh_resolveFrameAndJointAndChannelToMotionID(
                                       mc_, (BVHJointID)s.bvh_jid, 0, ch);
                        };
                        unsigned int iz = resolve(BVH_ROTATION_Z);
                        unsigned int ix = resolve(BVH_ROTATION_X);
                        unsigned int iy = resolve(BVH_ROTATION_Y);
                        row[iz] = within(az, lim->z_min, lim->z_max) ? az
                                  : (prev_row ? prev_row[iz] : clampf(az, lim->z_min, lim->z_max));
                        row[ix] = within(bx, lim->x_min, lim->x_max) ? bx
                                  : (prev_row ? prev_row[ix] : clampf(bx, lim->x_min, lim->x_max));
                        row[iy] = within(cy, lim->y_min, lim->y_max) ? cy
                                  : (prev_row ? prev_row[iy] : clampf(cy, lim->y_min, lim->y_max));
                        continue;  // channels already written above
                    }
                    else if (enforce_hand_limits_)
                    {
                        az = clampf(az, lim->z_min, lim->z_max);
                        bx = clampf(bx, lim->x_min, lim->x_max);
                        cy = clampf(cy, lim->y_min, lim->y_max);
                    }
                }
                else if (sticky_hand_pose_)
                {
                    // No limit table entry for this joint: pure inherit (or neutral).
                    if (prev_row)
                    {
                        for (unsigned int ch : { (unsigned int)BVH_ROTATION_Z,
                                                 (unsigned int)BVH_ROTATION_X,
                                                 (unsigned int)BVH_ROTATION_Y })
                        {
                            unsigned int idx = bvh_resolveFrameAndJointAndChannelToMotionID(
                                                   mc_, (BVHJointID)s.bvh_jid, 0, ch);
                            row[idx] = prev_row[idx];
                        }
                        continue;
                    }
                    // No prev_row: leave neutral (zeros).
                    continue;
                }
            }

            set_channel(mc_, row, s.bvh_jid, BVH_ROTATION_Z, az);
            set_channel(mc_, row, s.bvh_jid, BVH_ROTATION_X, bx);
            set_channel(mc_, row, s.bvh_jid, BVH_ROTATION_Y, cy);
        }

        // Sample live bone vector for the OFFSET rewrite (length-only at close).
        if (!s.is_root && s.ancestor_mhr_idx >= 0)
        {
            const int mp = s.ancestor_mhr_idx;
            float dv_world[3] =
            {
                t_global_mhr_[s.mhr_idx*3+0] - t_global_mhr_[mp*3+0],
                t_global_mhr_[s.mhr_idx*3+1] - t_global_mhr_[mp*3+1],
                t_global_mhr_[s.mhr_idx*3+2] - t_global_mhr_[mp*3+2],
            };
            float inv_cur[4];
            qconj(inv_cur, &q_global_mhr_[mp*4]);
            float dv_local[3];
            qrot(dv_local, inv_cur, dv_world);
            float dv_rest[3];
            qrot(dv_rest, &q_global_mhr_rest_[mp*4], dv_local);
            auto& vec = p.bone_samples[i];
            vec.push_back(dv_rest[0]);
            vec.push_back(dv_rest[1]);
            vec.push_back(dv_rest[2]);
        }
    }

    mc_->numberOfFrames   = saved_frames;
    mc_->motionValues     = saved_buf;
    mc_->motionValuesSize = saved_size;
    ++p.frame_count;
}

void BVHWriter::pad_continuation_frame(PerPerson& p)
{
    // Duplicate the last row so the timeline stays continuous through a brief
    // missing-frame gap.  At start-of-track (no frames yet) we'd never call
    // this, so motion_ always has at least one frame to duplicate.
    if (p.frame_count == 0) return;
    const float* prev = p.motion.data() + (size_t)(p.frame_count - 1) * total_channels_;
    p.motion.insert(p.motion.end(), prev, prev + total_channels_);
    ++p.frame_count;
}

// ─── BVHWriter::write_frame ─────────────────────────────────────────────────

void BVHWriter::write_frame(const std::vector<fsb::MHRResult>& results)
{
    if (!mc_ || !lbs_) return;

    // Drop YOLO bboxes that look like noise before the tracker sees them.
    std::vector<fsb::MHRResult> filtered;
    filtered.reserve(results.size());
    for (const auto& r : results)
    {
        if (bbox_looks_valid(r.bbox)) filtered.push_back(r);
    }
    const std::vector<fsb::MHRResult>& dets = filtered;

    // Tracker: per detection → stable id.  Side-effects: spawns/retires tracks_.
    std::vector<int> ids = assign_tracks(dets);

    // For each detection, append a fresh frame to that person's buffer.
    // Note: PerPerson is created lazily here for new ids.
    std::vector<char> alive_this_frame;
    if (!people_.empty()) alive_this_frame.assign(next_track_id_, 0);

    for (size_t d = 0; d < dets.size(); ++d)
    {
        int id = ids[d];
        PerPerson& p = people_[id];
        if (p.id < 0)
        {
            p.id           = id;
            p.frame_count  = 0;
            p.bone_samples.assign(slots_.size(), std::vector<float> {});
        }
        append_frame_for(p, dets[d]);
        if ((int)alive_this_frame.size() <= id) alive_this_frame.resize(id + 1, 0);
        alive_this_frame[id] = 1;
    }

    // Active tracks that didn't get a detection this frame: duplicate the last
    // known pose so their BVH timeline remains continuous.  (We keep tracks
    // alive for TRACK_MAX_MISSING frames before retiring them entirely.)
    for (const auto& t : tracks_)
    {
        if (t.last_seen_frame == session_frames_) continue;   // matched this frame
        auto it = people_.find(t.id);
        if (it == people_.end()) continue;
        pad_continuation_frame(it->second);
    }

    ++session_frames_;
}

// ─── BVHWriter::write_frame_external ───────────────────────────────────────
//
// Multi-person frame write driven by caller-supplied track IDs.  Used by the
// offline binary, which runs a globally-optimal Hungarian tracker over the
// whole video and decides identities up front.
//
// Differences from write_frame:
//   - No internal tracker call (no assign_tracks, no tracks_ touching).
//   - No bbox-validity filter (caller has already vetted detections).
//   - Caller decides which previously-seen tracks deserve a continuation pad
//     this frame via the pad_ids list.  Tracks that the caller stops mentioning
//     simply stop accumulating motion samples — the writer doesn't auto-retire.

void BVHWriter::write_frame_external(const std::vector<fsb::MHRResult>& results,
                                     const std::vector<int>&            track_ids,
                                     const std::vector<int>&            pad_ids)
{
    if (!mc_ || !lbs_) return;
    if (results.size() != track_ids.size())
    {
        fprintf(stderr, "[BVHWriter] write_frame_external: results / track_ids "
                        "size mismatch (%zu vs %zu)\n",
                results.size(), track_ids.size());
        return;
    }

    // Append a real motion row per detection ──────────────────────────────────
    for (size_t d = 0; d < results.size(); ++d)
    {
        int id = track_ids[d];
        if (id < 0) continue;
        PerPerson& p = people_[id];
        if (p.id < 0)
        {
            p.id          = id;
            p.frame_count = 0;
            p.bone_samples.assign(slots_.size(), std::vector<float>{});
        }
        append_frame_for(p, results[d]);
        // Keep next_track_id_ ahead of any externally-assigned ID so a later
        // call to the internal tracker (in mixed-mode use) wouldn't reuse it.
        if (id >= next_track_id_) next_track_id_ = id + 1;
    }

    // Continuation pads for tracks the caller wants to keep alive ─────────────
    for (int id : pad_ids)
    {
        auto it = people_.find(id);
        if (it == people_.end()) continue;
        pad_continuation_frame(it->second);
    }

    ++session_frames_;
}

void BVHWriter::write_frame_fused(const std::vector<FusedPerson>& persons,
                                  const std::vector<int>& pad_ids)
{
    if (!mc_ || !lbs_) return;

    for (const auto& fp : persons)
    {
        if (fp.track_id < 0 || fp.views.empty()) continue;

        std::vector<float> qsum;            // [nj*4] sign-aligned world-rotation sum
        float root_sum[3] = {0.f,0.f,0.f};
        int   nv = 0;

        for (const auto& view : fp.views)
        {
            if (!view.mhr) continue;
            compute_per_frame_mhr_state(*view.mhr);     // fills q_global_mhr_ (camera frame)
            const size_t N = q_global_mhr_.size();
            if (qsum.empty()) qsum.assign(N, 0.f);
            else if (qsum.size() != N) continue;        // joint-count mismatch (shouldn't happen)

            for (size_t j = 0; j < N; j += 4)
            {
                float qw[4];
                qmul(qw, view.q_world_cam.data(), &q_global_mhr_[j]);   // world = q_world_cam · q_cam
                float* acc = &qsum[j];
                if (nv == 0) { for (int k=0;k<4;++k) acc[k]=qw[k]; }
                else {
                    float d = acc[0]*qw[0]+acc[1]*qw[1]+acc[2]*qw[2]+acc[3]*qw[3];
                    float s = (d < 0.f) ? -1.f : 1.f;
                    for (int k=0;k<4;++k) acc[k]+=s*qw[k];
                }
            }
            root_sum[0]+=view.root_world[0]; root_sum[1]+=view.root_world[1]; root_sum[2]+=view.root_world[2];
            ++nv;
        }
        if (nv == 0 || qsum.empty()) continue;

        // Normalise the averaged quaternions back into q_global_mhr_ (world frame).
        for (size_t j = 0; j < qsum.size(); j += 4)
        {
            float* a = &qsum[j];
            float n = std::sqrt(a[0]*a[0]+a[1]*a[1]+a[2]*a[2]+a[3]*a[3]);
            if (n < 1e-9f) n = 1.f;
            for (int k=0;k<4;++k) q_global_mhr_[j+k] = a[k]/n;
        }

        fsb::MHRResult fr;                  // only pred_cam_t is read by append_row_from_state
        fr.pred_cam_t = { root_sum[0]/nv, root_sum[1]/nv, root_sum[2]/nv };

        PerPerson& p = people_[fp.track_id];
        if (p.id < 0) { p.id = fp.track_id; p.frame_count = 0; p.bone_samples.assign(slots_.size(), std::vector<float>{}); }
        append_row_from_state(p, fr);
        if (fp.track_id >= next_track_id_) next_track_id_ = fp.track_id + 1;
    }

    for (int id : pad_ids) { auto it = people_.find(id); if (it != people_.end()) pad_continuation_frame(it->second); }
    ++session_frames_;
}

// ─── Close-time: rewrite OFFSETs, then dump one file per person ────────────

void BVHWriter::rewrite_offsets_for(PerPerson& p)
{
    const int N = (int)mc_->jointHierarchySize;
    std::vector<std::array<float,3>> rest_pos(N);
    for (int j = 0; j < N; ++j)
    {
        const auto& jh = mc_->jointHierarchy[j];
        if (jh.isRoot) rest_pos[j] = { jh.offset[0], jh.offset[1], jh.offset[2] };
        else
        {
            int par = (int)jh.parentJoint;
            rest_pos[j] = { rest_pos[par][0] + jh.offset[0],
                            rest_pos[par][1] + jh.offset[1],
                            rest_pos[par][2] + jh.offset[2]
                          };
        }
    }

    int n_rewritten = 0;
    int n_skipped   = 0;
    for (size_t i = 0; i < slots_.size(); ++i)
    {
        const BvhSlot& s = slots_[i];
        if (s.mhr_idx < 0 || s.is_root) continue;
        if (s.ancestor_bvh_jid < 0)     continue;
        // Honour per-category disable flags — keeps the template's authored
        // OFFSETs (T-pose proportions) for body or hand bones when requested.
        if (s.kind == SlotKind::Body && !rewrite_body_offsets_) { ++n_skipped; continue; }
        if (s.kind == SlotKind::Hand && !rewrite_hand_offsets_) { ++n_skipped; continue; }
        const auto& samples = p.bone_samples[i];
        size_t n = samples.size() / 3;
        if (n == 0) continue;

        std::vector<float> lens(n);
        for (size_t k = 0; k < n; ++k)
        {
            float x = samples[3*k+0], y = samples[3*k+1], z = samples[3*k+2];
            lens[k] = std::sqrt(x*x + y*y + z*z);
        }
        std::sort(lens.begin(), lens.end());
        float median_len = lens[lens.size() / 2];

        int anc = s.ancestor_bvh_jid;
        float dx = rest_pos[i][0] - rest_pos[anc][0];
        float dy = rest_pos[i][1] - rest_pos[anc][1];
        float dz = rest_pos[i][2] - rest_pos[anc][2];
        float cur = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (cur < 1e-3f) continue;

        float inv_cur = 1.f / cur;
        float dir[3] = { dx*inv_cur, dy*inv_cur, dz*inv_cur };
        float delta_len = median_len - cur;
        auto& jh = mc_->jointHierarchy[i];
        jh.offset[0] += delta_len * dir[0];
        jh.offset[1] += delta_len * dir[1];
        jh.offset[2] += delta_len * dir[2];
        rest_pos[i] = { rest_pos[anc][0] + median_len * dir[0],
                        rest_pos[anc][1] + median_len * dir[1],
                        rest_pos[anc][2] + median_len * dir[2]
                      };
        ++n_rewritten;
    }

    fprintf(stderr,
            "[BVHWriter] person %d: rewrote %d joint OFFSETs (skipped %d for "
            "--no-bvh-{body,hand}-shape-change)\n",
            p.id, n_rewritten, n_skipped);

    // ── Fingertip End-Site compensation ─────────────────────────────────
    // body.bvh's End-Site OFFSETs at finger tips tend to be 0.3–0.5 cm longer
    // than the MHR `*_null` extensions, which makes the rewritten hands look
    // slightly oversized even after the bone-length rewrite.  When this
    // compensation is enabled (default; off with --bvh-raw-fingers), rescale
    // each End-Site OFFSET on a Hand-tagged tip joint to the length of the
    // corresponding MHR `*_null` joint's rest offset.  Direction is preserved.
    if (compensate_finger_endsites_ && rewrite_hand_offsets_)
    {
        int n_endsite = 0;
        for (size_t i = 0; i < slots_.size(); ++i)
        {
            const BvhSlot& s = slots_[i];
            if (s.mhr_idx < 0 || s.kind != SlotKind::Hand) continue;

            // Find an End-Site child of this BVH joint.
            int es_jid = -1;
            for (unsigned int k = 0; k < mc_->jointHierarchySize; ++k) {
                if ((int)mc_->jointHierarchy[k].parentJoint == s.bvh_jid &&
                    mc_->jointHierarchy[k].isEndSite) { es_jid = (int)k; break; }
            }
            if (es_jid < 0) continue;

            // Find an MHR child whose name ends with "_null".
            int null_idx = -1;
            for (int k = 0; k < lbs_->n_joints; ++k) {
                if (lbs_->joint_parents[k] != s.mhr_idx) continue;
                const char* nm = mhr_joint_table::NAMES[k];
                size_t L = strlen(nm);
                if (L >= 5 && strcmp(nm + L - 5, "_null") == 0) { null_idx = k; break; }
            }
            if (null_idx < 0) continue;

            const float* mhr_null_off = lbs_->joint_offsets + null_idx*3;
            float target_len = std::sqrt(mhr_null_off[0]*mhr_null_off[0] +
                                         mhr_null_off[1]*mhr_null_off[1] +
                                         mhr_null_off[2]*mhr_null_off[2]);
            if (target_len < 1e-3f) continue;

            float* off = mc_->jointHierarchy[es_jid].offset;
            float cur  = std::sqrt(off[0]*off[0] + off[1]*off[1] + off[2]*off[2]);
            if (cur < 1e-3f) continue;
            float scale = target_len / cur;
            off[0] *= scale;
            off[1] *= scale;
            off[2] *= scale;
            ++n_endsite;
        }
        if (n_endsite > 0)
            fprintf(stderr,
                    "[BVHWriter] person %d: rescaled %d finger End-Site OFFSETs "
                    "(disable with --bvh-raw-fingers)\n", p.id, n_endsite);
    }
}

// Build "<base>_<prefix><id>.<ext>".  If out_path_ has no extension, append
// .bvh.  prefix is normally empty (→ "<base>_<id>.bvh"); --bvh-split-scenes
// sets it to e.g. "scene0_person".
static std::string per_person_path(const std::string& base,
                                   const std::string& prefix, int id)
{
    auto dot = base.find_last_of('.');
    auto slash = base.find_last_of("/\\");
    bool has_ext = (dot != std::string::npos) && (slash == std::string::npos || dot > slash);
    if (has_ext) return base.substr(0, dot) + "_" + prefix + std::to_string(id) + base.substr(dot);
    return base + "_" + prefix + std::to_string(id) + ".bvh";
}

// ════════════════════════════════════════════════════════════════════════════
//  Foot-contact cleanup (Phase 1: root vertical leveling + 2-bone leg IK)
//
//  Runs once per person at close(), AFTER rewrite_offsets_for() has set this
//  person's final bone lengths and mc_->motionValues points at their buffer —
//  so the MotionCaptureLoader FK (bvh_loadTransformForMotionBuffer) yields the
//  exact world joint positions the exported file will play.  Working in the
//  writer's own BVH space (cm) keeps everything self-consistent: no MHR /
//  keypoints_3d frame mismatch, no body_pose editing.
//
//  Convention: the camera frame is +X right / +Y down / +Z forward (see the
//  projection in fast_sam_3dbody.cpp), so the gravity axis is Y and the floor
//  is a (near-)constant Y.  We never assume the sign of "down": leveling pins
//  the planted ankle to a fitted constant Y, and contact detection only needs
//  horizontal (X,Z) speed + nearness to that constant — both sign-agnostic.
// ════════════════════════════════════════════════════════════════════════════
void BVHWriter::foot_contact_pass(PerPerson& p)
{
    const int F = p.frame_count;
    if (F < 3) return;
    const bool dbg = (std::getenv("FSB_FOOT_DEBUG") != nullptr);

    float SPEED_THR  = 2.0f;   // cm/frame below which a foot is "planted"
    float HEIGHT_BAND= 8.0f;   // cm around the fitted ground level
    const int MIN_RUN = 3;     // shortest accepted / bridged contact run (frames)
    const int BLEND   = 3;     // ramp frames at each contact edge
    if (const char* e = std::getenv("FSB_FOOT_SPEED")) SPEED_THR   = (float)atof(e);
    if (const char* e = std::getenv("FSB_FOOT_BAND"))  HEIGHT_BAND = (float)atof(e);

    auto find_jid = [&](const char* nm) -> int {
        for (unsigned int j = 0; j < mc_->jointHierarchySize; ++j)
            if (std::strcmp(mc_->jointHierarchy[j].jointName, nm) == 0) return (int)j;
        return -1;
    };
    struct Leg { int hip, knee, ankle; };
    Leg legs[2] = {
        { find_jid("lThigh"), find_jid("lShin"), find_jid("lFoot") },
        { find_jid("rThigh"), find_jid("rShin"), find_jid("rFoot") },
    };
    for (auto& L : legs)
        if (L.hip < 0 || L.knee < 0 || L.ankle < 0) {
            if (dbg) fprintf(stderr, "[foot] person %d: leg chain not in template; skip\n", p.id);
            return;
        }
    if (root_bvh_jid_ < 0 || !mc_->jointHierarchy[root_bvh_jid_].hasPositionalChannels)
        return;

    // ── FK every frame, caching the world transforms the IK needs ────────────
    struct JT  { float pos[3]; float q[4]; };
    struct FFK { JT hip[2], knee[2], ankle[2], hippar[2]; };
    std::vector<FFK> fk(F);

    BVH_Transform xf; memset(&xf, 0, sizeof(xf));
    if (!bvh_allocateTransform(mc_, &xf)) { if (dbg) fprintf(stderr,"[foot] alloc xf failed\n"); return; }

    auto grab = [&](int jid, JT& out){
        const BVH_TransformedJoint& tj = xf.joint[jid];
        // chainTransformation is the world matrix (pos3D = its m[3,7,11]); its
        // rotation part is the joint's world orientation.
        out.pos[0]=tj.pos3D[0]; out.pos[1]=tj.pos3D[1]; out.pos[2]=tj.pos3D[2];
        float m9[9]; mat3_from_m4(tj.chainTransformation.m, m9);
        mat3_to_quat(m9, out.q);
    };
    for (int f = 0; f < F; ++f){
        float* row = p.motion.data() + (size_t)f*total_channels_;
        bvh_loadTransformForMotionBuffer(mc_, row, &xf, /*populateTorso=*/0);
        for (int s = 0; s < 2; ++s){
            grab(legs[s].hip,   fk[f].hip[s]);
            grab(legs[s].knee,  fk[f].knee[s]);
            grab(legs[s].ankle, fk[f].ankle[s]);
            grab((int)mc_->jointHierarchy[legs[s].hip].parentJoint, fk[f].hippar[s]);
        }
    }
    bvh_freeTransform(&xf);

    // ── Per-foot contact detection (low horizontal speed + near ground Y) ────
    auto clean_runs = [&](std::vector<char>& c){
        // drop runs shorter than MIN_RUN, then bridge gaps shorter than MIN_RUN
        for (int f = 0; f < F; ){
            if (!c[f]) { ++f; continue; }
            int g = f; while (g < F && c[g]) ++g;
            if (g - f < MIN_RUN) for (int k=f;k<g;++k) c[k]=0;
            f = g;
        }
        for (int f = 0; f < F; ){
            if (c[f]) { ++f; continue; }
            int g = f; while (g < F && !c[g]) ++g;
            if (f > 0 && g < F && g - f < MIN_RUN) for (int k=f;k<g;++k) c[k]=1;
            f = g;
        }
    };

    std::vector<char> contact[2];
    float groundY[2] = {0.f, 0.f};
    bool  has_ground[2] = {false, false};
    std::vector<float> spd[2];
    for (int s = 0; s < 2; ++s){
        contact[s].assign(F, 0);
        spd[s].assign(F, 1e9f);
        for (int f = 1; f < F; ++f){
            float dx = fk[f].ankle[s].pos[0]-fk[f-1].ankle[s].pos[0];
            float dz = fk[f].ankle[s].pos[2]-fk[f-1].ankle[s].pos[2];
            spd[s][f] = std::sqrt(dx*dx+dz*dz);
        }
        if (F > 1) spd[s][0] = spd[s][1];
        std::vector<float> lowY;
        for (int f = 0; f < F; ++f) if (spd[s][f] < SPEED_THR) lowY.push_back(fk[f].ankle[s].pos[1]);
        if ((int)lowY.size() >= MIN_RUN){
            groundY[s] = median_of(lowY); has_ground[s] = true;
            for (int f = 0; f < F; ++f)
                contact[s][f] = (spd[s][f] < SPEED_THR) &&
                                (std::fabs(fk[f].ankle[s].pos[1]-groundY[s]) < HEIGHT_BAND);
            clean_runs(contact[s]);
        }
    }

    // Fitted floor = median planted-ankle Y across both feet's contact frames.
    std::vector<float> floorSamples;
    for (int s = 0; s < 2; ++s)
        for (int f = 0; f < F; ++f) if (contact[s][f]) floorSamples.push_back(fk[f].ankle[s].pos[1]);
    if (floorSamples.empty()){
        if (dbg) fprintf(stderr, "[foot] person %d: no contacts detected; pass is a no-op\n", p.id);
        return;
    }
    const float floorY = median_of(floorSamples);

    if (dbg){
        int nc[2]={0,0}; for(int s=0;s<2;++s) for(int f=0;f<F;++f) nc[s]+=contact[s][f];
        fprintf(stderr,
            "[foot] person %d: F=%d  L:ankleY[%.1f..%.1f] ground=%.1f contacts=%d  "
            "R:ground=%.1f contacts=%d  floorY=%.1f  (speed<%.1f band<%.1f)\n",
            p.id, F,
            [&]{float m= 1e9f; for(int f=0;f<F;++f) m=std::min(m,fk[f].ankle[0].pos[1]); return m;}(),
            [&]{float m=-1e9f; for(int f=0;f<F;++f) m=std::max(m,fk[f].ankle[0].pos[1]); return m;}(),
            has_ground[0]?groundY[0]:0.f, nc[0],
            has_ground[1]?groundY[1]:0.f, nc[1], floorY, SPEED_THR, HEIGHT_BAND);
    }

    // ── Analytic 2-bone IK: re-aim a leg so its ankle reaches `target`. ───────
    // Writes the hip + knee ZXY rotation channels of frame `row`.
    auto solve_leg_ik = [&](float* row, int f, int s, const float target[3]){
        const JT& H  = fk[f].hip[s];
        const JT& K  = fk[f].knee[s];
        const JT& A  = fk[f].ankle[s];
        const JT& HP = fk[f].hippar[s];

        float v1[3]; v3sub(K.pos, H.pos, v1); float L1 = v3len(v1);
        float v2[3]; v3sub(A.pos, K.pos, v2); float L2 = v3len(v2);
        if (L1 < 1e-4f || L2 < 1e-4f) return;

        float d[3]; v3sub(target, H.pos, d); float dist = v3len(d);
        const float eps = 1e-3f;
        float dmin = std::fabs(L1-L2)+eps, dmax = L1+L2-eps;
        if (dist < dmin) dist = dmin;
        if (dist > dmax) dist = dmax;
        if (dist < 1e-4f) return;
        float u[3] = { d[0]/v3len(d), d[1]/v3len(d), d[2]/v3len(d) };
        float Tc[3] = { H.pos[0]+u[0]*dist, H.pos[1]+u[1]*dist, H.pos[2]+u[2]*dist };

        float t_along = (dist*dist + L1*L1 - L2*L2) / (2.f*dist);
        float h2 = L1*L1 - t_along*t_along; float h = h2>0.f ? std::sqrt(h2) : 0.f;
        // bend direction: current knee offset projected off the hip→target axis
        float e[3]; { float pr = v3dot(v1,u); e[0]=v1[0]-pr*u[0]; e[1]=v1[1]-pr*u[1]; e[2]=v1[2]-pr*u[2]; }
        if (v3len(e) < 1e-4f){                       // straight leg: pick a stable perp
            float up[3]={0,1,0}; v3cross(u,up,e);
            if (v3len(e) < 1e-4f){ float ax[3]={1,0,0}; v3cross(u,ax,e); }
        }
        v3norm(e);
        float Knew[3] = { H.pos[0]+u[0]*t_along+e[0]*h,
                          H.pos[1]+u[1]*t_along+e[1]*h,
                          H.pos[2]+u[2]*t_along+e[2]*h };

        float nd1[3]; v3sub(Knew, H.pos, nd1);
        float Rdh[4]; shortest_arc_quat(v1, nd1, Rdh);
        float Rwh_new[4]; qmul(Rwh_new, Rdh, H.q);

        float v2mid[3]; qrot(v2mid, Rdh, v2);
        float nd2[3]; v3sub(Tc, Knew, nd2);
        float Rdk[4]; shortest_arc_quat(v2mid, nd2, Rdk);
        float tmp[4]; qmul(tmp, Rdh, K.q);
        float Rwk_new[4]; qmul(Rwk_new, Rdk, tmp);

        float cp[4], qlh[4]; qconj(cp, HP.q);      qmul(qlh, cp, Rwh_new);
        float ch[4], qlk[4]; qconj(ch, Rwh_new);   qmul(qlk, ch, Rwk_new);

        float m[9], az, bx, cy;
        quat_to_mat3(qlh, m); mat3_to_zxy(m, az, bx, cy);
        set_channel(mc_, row, (BVHJointID)legs[s].hip,  BVH_ROTATION_Z, az*RAD2DEG);
        set_channel(mc_, row, (BVHJointID)legs[s].hip,  BVH_ROTATION_X, bx*RAD2DEG);
        set_channel(mc_, row, (BVHJointID)legs[s].hip,  BVH_ROTATION_Y, cy*RAD2DEG);
        quat_to_mat3(qlk, m); mat3_to_zxy(m, az, bx, cy);
        set_channel(mc_, row, (BVHJointID)legs[s].knee, BVH_ROTATION_Z, az*RAD2DEG);
        set_channel(mc_, row, (BVHJointID)legs[s].knee, BVH_ROTATION_X, bx*RAD2DEG);
        set_channel(mc_, row, (BVHJointID)legs[s].knee, BVH_ROTATION_Y, cy*RAD2DEG);
    };

    // ── Horizontal lock target + ramp weight per foot/frame ──────────────────
    // For each contact run, the locked (X,Z) is the run's median; the weight
    // ramps 0→1→0 across BLEND frames at the run edges to avoid pops.
    std::vector<float> lockX[2], lockZ[2], lockW[2];
    for (int s = 0; s < 2; ++s){
        lockX[s].assign(F,0.f); lockZ[s].assign(F,0.f); lockW[s].assign(F,0.f);
        for (int f = 0; f < F; ){
            if (!contact[s][f]) { ++f; continue; }
            int g = f; while (g < F && contact[s][g]) ++g;     // run [f,g)
            std::vector<float> xs, zs;
            for (int k=f;k<g;++k){ xs.push_back(fk[k].ankle[s].pos[0]); zs.push_back(fk[k].ankle[s].pos[2]); }
            float mx = median_of(xs), mz = median_of(zs);
            for (int k=f;k<g;++k){
                int edge = std::min(k-f+1, g-k);               // 1..run/2
                float w = std::min(1.f, (float)edge/(float)BLEND);
                lockX[s][k]=mx; lockZ[s][k]=mz; lockW[s][k]=w;
            }
            f = g;
        }
    }

    // ── Apply: per frame, IK each contacting foot, then level the root in Y ───
    std::vector<float> dY(F, std::numeric_limits<float>::quiet_NaN());
    for (int f = 0; f < F; ++f){
        float* row = p.motion.data() + (size_t)f*total_channels_;
        float lev_sum = 0.f; int lev_n = 0;
        for (int s = 0; s < 2; ++s){
            if (!contact[s][f]) continue;
            const JT& A = fk[f].ankle[s];
            float w = lockW[s][f];
            float target[3] = {
                A.pos[0] + (lockX[s][f]-A.pos[0])*w,
                A.pos[1],                                   // vertical handled by leveling
                A.pos[2] + (lockZ[s][f]-A.pos[2])*w
            };
            solve_leg_ik(row, f, s, target);
            lev_sum += (floorY - A.pos[1]) * w;             // pull planted ankle to floor
            lev_n++;
        }
        if (lev_n) dY[f] = lev_sum / lev_n;
    }

    // Fill the non-contact frames by linear interpolation between known dY,
    // clamping flat past the first / last contact.
    {
        int prev = -1;
        for (int f = 0; f < F; ++f){
            if (std::isnan(dY[f])) continue;
            if (prev < 0) for (int k=0;k<f;++k) dY[k]=dY[f];       // head
            else { float a=dY[prev], b=dY[f]; for (int k=prev+1;k<f;++k){ float t=(float)(k-prev)/(f-prev); dY[k]=a+(b-a)*t; } }
            prev = f;
        }
        if (prev < 0) return;                                     // shouldn't happen
        for (int k=prev+1;k<F;++k) dY[k]=dY[prev];                // tail
    }

    unsigned int rootYidx = bvh_resolveFrameAndJointAndChannelToMotionID(
                                mc_, (BVHJointID)root_bvh_jid_, 0, BVH_POSITION_Y);
    for (int f = 0; f < F; ++f){
        float* row = p.motion.data() + (size_t)f*total_channels_;
        row[rootYidx] += dY[f];
    }

    if (dbg){
        float mn=1e9f,mx=-1e9f; for(int f=0;f<F;++f){ mn=std::min(mn,dY[f]); mx=std::max(mx,dY[f]); }
        // Re-FK the corrected buffer and measure planted-foot horizontal skate
        // (RMS deviation from each contact run's locked centre) before vs after.
        BVH_Transform x2; memset(&x2, 0, sizeof(x2));
        if (bvh_allocateTransform(mc_, &x2)){
            double se_before=0, se_after=0; long n=0;
            std::vector<std::array<float,2>> after(F);
            for (int s=0;s<2;++s){
                for (int f=0; f<F; ++f){
                    if (!contact[s][f]) continue;
                    float* row = p.motion.data() + (size_t)f*total_channels_;
                    bvh_loadTransformForMotionBuffer(mc_, row, &x2, 0);
                    after[f] = { x2.joint[legs[s].ankle].pos3D[0], x2.joint[legs[s].ankle].pos3D[2] };
                }
                for (int f=0; f<F; ++f){
                    if (!contact[s][f]) continue;
                    float bx=fk[f].ankle[s].pos[0]-lockX[s][f], bz=fk[f].ankle[s].pos[2]-lockZ[s][f];
                    float ax=after[f][0]-lockX[s][f],          az=after[f][1]-lockZ[s][f];
                    se_before += bx*bx+bz*bz; se_after += ax*ax+az*az; ++n;
                }
            }
            bvh_freeTransform(&x2);
            fprintf(stderr,
                "[foot] person %d: applied root ΔY=[%.1f, %.1f] cm + leg IK; planted-foot skate "
                "RMS %.2f → %.2f cm over %ld contact-frames\n",
                p.id, mn, mx,
                n? std::sqrt(se_before/n):0.0, n? std::sqrt(se_after/n):0.0, n);
        } else {
            fprintf(stderr, "[foot] person %d: applied root ΔY=[%.1f, %.1f] cm + leg IK on contacts\n",
                    p.id, mn, mx);
        }
    }
}

bool BVHWriter::dump_one_person(const PerPerson& p)
{
    // Snapshot the template's original OFFSETs so we can restore them after
    // this person's bone-length rewrites (each person has their own lengths).
    const int N = (int)mc_->jointHierarchySize;
    std::vector<std::array<float,3>> saved_offsets(N);
    for (int j = 0; j < N; ++j)
    {
        saved_offsets[j] = { mc_->jointHierarchy[j].offset[0],
                             mc_->jointHierarchy[j].offset[1],
                             mc_->jointHierarchy[j].offset[2]
                           };
    }

    PerPerson& mut = const_cast<PerPerson&>(p);
    rewrite_offsets_for(mut);

    mc_->motionValues       = mut.motion.data();
    mc_->motionValuesSize   = (unsigned int)mut.motion.size();
    mc_->numberOfFrames     = (unsigned int)mut.frame_count;
    mc_->numberOfFramesEncountered = mc_->numberOfFrames;
    mc_->frameTime          = frame_time_;

    // Foot-contact cleanup runs here: offsets are final and mc_->motionValues
    // points at this person's buffer, so the loader FK matches the export.
    if (foot_contact_) foot_contact_pass(mut);

    std::string path = per_person_path(out_path_, id_prefix_, p.id);
    int ok = dumpBVHToBVH(path.c_str(), mc_, /*hierarchy=*/1, /*motion=*/1);
    if (ok) printf("[BVHWriter] wrote %d frames → %s\n", p.frame_count, path.c_str());
    else    fprintf(stderr, "[BVHWriter] dumpBVHToBVH('%s') failed\n", path.c_str());

    // ── BVHTester-compatibility heads-up ────────────────────────────────────
    // BVHTester (./RGBDAcquisition/.../BVHTester) silently culls frames whose
    // root sits outside its default visible volume — the user discovers this
    // when the BVH "plays correctly in Blender but renders an empty scene in
    // BVHTester."  Since pred_cam_t directly drives the hip's BVH position
    // (× 100 to convert m → cm), the typical MHR output places the hip at
    // Z ≈ +200..+500 cm (subject-in-front-of-camera) which is exactly the
    // region BVHTester rejects as "behind camera".
    //
    // We always print the observed root XYZ range so users can sanity-check
    // their numbers, and only emit the load-bearing "use --setPositionRotation"
    // suggestion when the path leaves a conservative ±200 cm cube on any
    // axis — that threshold catches every realistic MHR run while staying
    // quiet on BVHs that BVHTester would already play directly (e.g. a
    // hand-edited or MocapNET-style clip with the hip already centred).
    if (ok && mc_->jointHierarchy[root_bvh_jid_].hasPositionalChannels &&
        p.frame_count > 0)
    {
        unsigned int xi = bvh_resolveFrameAndJointAndChannelToMotionID(
                              mc_, (BVHJointID)root_bvh_jid_, 0, BVH_POSITION_X);
        unsigned int yi = bvh_resolveFrameAndJointAndChannelToMotionID(
                              mc_, (BVHJointID)root_bvh_jid_, 0, BVH_POSITION_Y);
        unsigned int zi = bvh_resolveFrameAndJointAndChannelToMotionID(
                              mc_, (BVHJointID)root_bvh_jid_, 0, BVH_POSITION_Z);
        // The resolve helper returns ABSOLUTE indices assuming fID=0;
        // since each row is one frame's data, fID=0 makes the result
        // also the in-row offset.
        float mn[3] = {  std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::infinity() };
        float mx[3] = { -std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity() };
        const unsigned int ch_idx[3] = { xi, yi, zi };
        for (int f = 0; f < p.frame_count; ++f)
        {
            const float* row = mut.motion.data() + (size_t)f * total_channels_;
            for (int a = 0; a < 3; ++a) {
                float v = row[ch_idx[a]];
                if (v < mn[a]) mn[a] = v;
                if (v > mx[a]) mx[a] = v;
            }
        }
        printf("[BVHWriter] person %d: root path  "
               "X=[%.1f, %.1f] cm  Y=[%.1f, %.1f] cm  Z=[%.1f, %.1f] cm\n",
               p.id, mn[0], mx[0], mn[1], mx[1], mn[2], mx[2]);

        constexpr float BVHTESTER_HALF_VOL_CM = 200.0f;
        bool out = false;
        for (int a = 0; a < 3; ++a)
            if (mn[a] < -BVHTESTER_HALF_VOL_CM || mx[a] > BVHTESTER_HALF_VOL_CM) {
                out = true;
                break;
            }
        if (out) {
            // Pick a sane override position for the suggestion: keep the
            // observed X/Y centres, push Z to 1500 (= -150 cm in BVHTester's
            // negate-and-divide-by-10 convention, the same depth MocapNET
            // scripts use).
            int cx = (int)((mn[0] + mx[0]) * 0.5f * -10.0f);
            int cy = (int)((mn[1] + mx[1]) * 0.5f * -10.0f);
            fprintf(stderr,
                "[BVHWriter] person %d: root path falls outside BVHTester's default\n"
                "                ±%.0f-cm visible volume — the file plays correctly in\n"
                "                Blender but BVHTester will render an empty scene.\n"
                "                To preview in BVHTester:\n"
                "                  BVHTester --from %s \\\n"
                "                            --setPositionRotation %d %d 1500 0 0 0 \\\n"
                "                            --csv ./ %s 3d\n",
                p.id, BVHTESTER_HALF_VOL_CM, path.c_str(), cx, cy, "test");
        }
    }

    // Restore offsets for the next person.
    for (int j = 0; j < N; ++j)
    {
        mc_->jointHierarchy[j].offset[0] = saved_offsets[j][0];
        mc_->jointHierarchy[j].offset[1] = saved_offsets[j][1];
        mc_->jointHierarchy[j].offset[2] = saved_offsets[j][2];
    }
    mc_->motionValues     = nullptr;
    mc_->motionValuesSize = 0;
    return ok != 0;
}

void BVHWriter::close()
{
    if (!mc_) return;

    if (people_.empty())
    {
        printf("[BVHWriter] no detections seen; nothing to write\n");
    }
    else
    {
        // Sort by id so output filenames go in ascending order.
        std::vector<int> ids;
        ids.reserve(people_.size());
        for (auto& kv : people_) ids.push_back(kv.first);
        std::sort(ids.begin(), ids.end());
        for (int id : ids) dump_one_person(people_.at(id));
    }

    bvh_free(mc_);
    free(mc_);
    mc_ = nullptr;
    if (lbs_)
    {
        mhr_lbs_free(lbs_);
        lbs_ = nullptr;
    }
    tracks_.clear();
    people_.clear();
    slots_.clear();
}
