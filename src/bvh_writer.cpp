// bvh_writer.cpp – BVH export for MHR body-pose inference output.
//
// Euler convention (body joints)
// ──────────────────────────────
// rot6d_to_euler → ZYX: euler[0]=rx, [1]=ry, [2]=rz  (radians).
// compact_cont_to_body_params stores:
//   body_pose[ BODY_3DOF_JOINT_IDXS[j][0] ] = rx
//   body_pose[ BODY_3DOF_JOINT_IDXS[j][1] ] = ry
//   body_pose[ BODY_3DOF_JOINT_IDXS[j][2] ] = rz
// BVH body channels (non-root): Zrotation Xrotation Yrotation
//   → at bvh_offset+0 write rz, +1 write rx, +2 write ry.
//
// global_rot storage (fast_sam_3dbody.cpp line 917):
//   r.global_rot = { ge[2], ge[1], ge[0] }  where ge=[rx,ry,rz]
//   ⟹  global_rot[0]=rz, [1]=ry, [2]=rx
// BVH hip channels: Xpos Ypos Zpos  Zrot Yrot Xrot
//   → direct map: buf[3]=rz, buf[4]=ry, buf[5]=rx.
//
// Arm/collar/head joints (entries 12–20 in MHR_JOINT_SPECS)
// ──────────────────────────────────────────────────────────
// build_model_params() zeroes mhr_model_params[68:121] (body_pose indices 62-115)
// because the LBS mesh was trained expecting hand-PCA decoded angles there.
// apply_hand_pose() then fills those slots via the hand PCA decode:
//   - "right" hand PCA → collar, shoulder, head angles (body_pose 62-88)
//   - "left"  hand PCA → elbow, wrist angles (body_pose 89-115)
// So mhr_model_params[6+arm_idx] holds the correct arm angles after apply_hand_pose.
// r.body_pose[62:115] is near-zero (raw FFN compact-6D output) and must NOT be used.
//
// Hand joints
// ───────────
// apply_hand_pose places the 27 decoded Euler angles per hand directly into
// mhr_model_params[204] at positions given by hand_joint_idxs_left/right[27].
// We read those indices from body_model.lbs (version 3 section) and use them
// to extract angles from MHRResult::mhr_model_params.
//
// DOF order (per hand), from HAND_DOFS_IN_ORDER = [3,1,1, 3,1,1, 3,1,1, 3,1,1, 2,3,1,1]:
//   params[0..2]  = thumb MCP  (rx,ry,rz)
//   params[3]     = thumb PIP  (single angle)
//   params[4]     = thumb DIP
//   params[5..7]  = index MCP  (rx,ry,rz)
//   params[8]     = index PIP
//   params[9]     = index DIP
//   params[10..12]= middle MCP (rx,ry,rz)
//   params[13]    = middle PIP
//   params[14]    = middle DIP
//   params[15..17]= ring MCP   (rx,ry,rz)
//   params[18]    = ring PIP
//   params[19]    = ring DIP
//   params[20..21]= pinky CMC  (2 angles)
//   params[22..24]= pinky MCP  (rx,ry,rz)
//   params[25]    = pinky PIP
//   params[26]    = pinky DIP

#include "bvh_writer.h"
#include "fast_sam_3dbody.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

static constexpr float RAD2DEG   = 57.29577951308232f;
static constexpr float POS_SCALE = 100.0f;   // pred_cam_t metres → cm

// Convert SMPL-like ZYX Euler angles (R = Rz·Ry·Rx) to BVH body-joint ZXY Euler
// angles (BVH applies M = Rz·Rx·Ry for "Zrotation Xrotation Yrotation" channels).
// Builds the rotation matrix from ZYX then extracts ZXY; avoids gimbal artifacts.
static void zyx_to_zxy(float rx, float ry, float rz,
                        float& a /*Zrot*/, float& b /*Xrot*/, float& c /*Yrot*/)
{
    const float crx = cosf(rx), srx = sinf(rx);
    const float cry = cosf(ry), sry = sinf(ry);
    const float crz = cosf(rz), srz = sinf(rz);

    // ZYX matrix elements needed for ZXY decomposition:
    //   R = Rz(rz)*Ry(ry)*Rx(rx)
    const float R01 = crz*sry*srx - srz*crx;   // R[0][1]
    const float R11 = srz*sry*srx + crz*crx;   // R[1][1]
    const float R20 = -sry;                     // R[2][0]
    const float R21 =  cry*srx;                 // R[2][1]
    const float R22 =  cry*crx;                 // R[2][2]

    // ZXY decomposition: M = Rz(a)*Rx(b)*Ry(c)
    //   M[2][1] = sin(b)
    //   M[2][0] = -cos(b)*sin(c)  →  c = atan2(-M20, M22)
    //   M[0][1] = -sin(a)*cos(b)  →  a = atan2(-M01, M11)
    b = asinf(std::max(-1.f, std::min(1.f, R21)));
    c = atan2f(-R20, R22);
    a = atan2f(-R01, R11);
}

// ─────────────────────────────────────────────────────────────────────────────
// Major body joint mapping (22 active + 1 skipped)
// rx/ry/rz indices taken from BODY_3DOF_JOINT_IDXS in preprocess.hpp.
// Joint ordering is presumed SMPL-like; validate against mhr_utils.py if needed.
// ─────────────────────────────────────────────────────────────────────────────
struct MhrJointSpec {
    const char* bvh_name;
    int rx_idx, ry_idx, rz_idx;
};
// clang-format off
static constexpr MhrJointSpec MHR_JOINT_SPECS[23] = {
    /* 0  L_Hip      */ {"lButtock",  0,   2,   4  },
    /* 1  R_Hip      */ {"rButtock",  6,   8,   10 },
    /* 2  Spine1     */ {"abdomen",   12,  13,  14 },
    /* 3  L_Knee     */ {"lThigh",    15,  16,  17 },
    /* 4  R_Knee     */ {"rThigh",    18,  19,  20 },
    /* 5  Spine2     */ {"chest",     21,  22,  23 },
    /* 6  L_Ankle    */ {"lShin",     24,  25,  26 },
    /* 7  R_Ankle    */ {"rShin",     27,  28,  29 },
    /* 8  Spine3→neck1*/{"neck1",     34,  35,  36 },
    /* 9  L_Foot     */ {"lFoot",     37,  38,  39 },
    /* 10 R_Foot     */ {"rFoot",     44,  45,  46 },
    /* 11 Neck       */ {"neck",      53,  54,  55 },
    /* 12 L_Collar   */ {"lCollar",   64,  65,  66 },
    /* 13 R_Collar   */ {"rCollar",   85,  69,  73 },
    /* 14 Head       */ {"head",      86,  70,  79 },
    /* 15 L_Shoulder */ {"lShldr",    87,  71,  82 },
    /* 16 R_Shoulder */ {"rShldr",    88,  72,  76 },
    /* 17 L_Elbow    */ {"lForeArm",  91,  92,  93 },
    /* 18 R_Elbow    */ {"rForeArm",  112, 96,  100},
    /* 19 L_Wrist    */ {"lHand",     113, 97,  106},
    /* 20 R_Wrist    */ {"rHand",     114, 98,  109},
    /* 21 (skip)     */ {nullptr,     115, 99,  103},
    /* 22 Jaw        */ {"jaw",       130, 131, 132},
};
// clang-format on

// ─────────────────────────────────────────────────────────────────────────────
// BVH hierarchy parser
// ─────────────────────────────────────────────────────────────────────────────
static bool parse_bvh_hierarchy(
    const std::string&                   path,
    std::string&                         hierarchy_out,
    std::unordered_map<std::string,int>& name_to_offset,
    int&                                 total_channels)
{
    FILE* f = fopen(path.c_str(), "r");
    if (!f) {
        fprintf(stderr, "[BVHWriter] Cannot open template: %s\n", path.c_str());
        return false;
    }

    char        line[4096];
    std::string hierarchy;
    int         channel_cursor = 0;
    bool        found_motion   = false;
    char        pending_joint[256] = "";

    while (fgets(line, sizeof(line), f))
    {
        const char* p = line;
        while (*p == ' ' || *p == '\t') ++p;

        if (strncmp(p, "MOTION", 6) == 0 &&
            (p[6]=='\n' || p[6]=='\r' || p[6]=='\0'))
        {
            found_motion = true;
            break;
        }

        hierarchy += line;

        if ( (strncmp(p,"ROOT", 4)==0 && (p[4]==' '||p[4]=='\t')) ||
             (strncmp(p,"JOINT",5)==0 && (p[5]==' '||p[5]=='\t')) )
        {
            const char* q = p + ((p[0]=='R') ? 4 : 5);
            while (*q==' '||*q=='\t') ++q;
            size_t n = 0;
            while (q[n] && q[n]!=' ' && q[n]!='\t' && q[n]!='\r' && q[n]!='\n') ++n;
            if (n > 0 && n < sizeof(pending_joint)) {
                memcpy(pending_joint, q, n);
                pending_joint[n] = '\0';
            }
            continue;
        }

        if (strncmp(p, "End Site", 8) == 0) {
            pending_joint[0] = '\0';
            continue;
        }

        if (strncmp(p, "CHANNELS", 8) == 0 && (p[8]==' '||p[8]=='\t')) {
            int n_ch = 0;
            sscanf(p + 8, "%d", &n_ch);
            if (pending_joint[0] != '\0') {
                name_to_offset[std::string(pending_joint)] = channel_cursor;
                pending_joint[0] = '\0';
            }
            channel_cursor += n_ch;
            continue;
        }
    }

    fclose(f);

    if (!found_motion) {
        fprintf(stderr, "[BVHWriter] Template '%s' has no MOTION section.\n", path.c_str());
        return false;
    }

    hierarchy_out  = std::move(hierarchy);
    total_channels = channel_cursor;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// BVHWriter::open
// ─────────────────────────────────────────────────────────────────────────────
bool BVHWriter::open(const std::string& template_path,
                     const std::string& out_path,
                     float frame_time,
                     const std::string& /*lbs_path*/)
{
    std::string hierarchy;
    std::unordered_map<std::string,int> name_to_offset;
    int total_ch = 0;

    if (!parse_bvh_hierarchy(template_path, hierarchy, name_to_offset, total_ch))
        return false;

    if (total_ch <= 0) {
        fprintf(stderr, "[BVHWriter] Template parsed 0 channels.\n");
        return false;
    }
    if (total_ch != 498) {
        fprintf(stderr,
                "[BVHWriter] Warning: template has %d channels (expected 498).\n",
                total_ch);
    }

    // ── Resolve major body joints ────────────────────────────────────────────
    for (int j = 0; j < 23; ++j) {
        resolved_[j].bvh_offset = -1;
        resolved_[j].rx_idx     = MHR_JOINT_SPECS[j].rx_idx;
        resolved_[j].ry_idx     = MHR_JOINT_SPECS[j].ry_idx;
        resolved_[j].rz_idx     = MHR_JOINT_SPECS[j].rz_idx;

        const char* bvh_name = MHR_JOINT_SPECS[j].bvh_name;
        if (!bvh_name) continue;

        auto it = name_to_offset.find(bvh_name);
        if (it == name_to_offset.end())
            fprintf(stderr, "[BVHWriter] Warning: joint '%s' not found.\n", bvh_name);
        else
            resolved_[j].bvh_offset = it->second;
    }

    // ── Open output file ─────────────────────────────────────────────────────
    file_ = fopen(out_path.c_str(), "wb");
    if (!file_) {
        fprintf(stderr, "[BVHWriter] Cannot open output: %s\n", out_path.c_str());
        return false;
    }

    total_channels_ = total_ch;
    frame_time_     = frame_time;
    frame_count_    = 0;

    fwrite(hierarchy.data(), 1, hierarchy.size(), file_);
    fprintf(file_, "MOTION\n");
    frames_pos_ = ftell(file_);
    fprintf(file_, "Frames: %10d\n", 0);
    fprintf(file_, "Frame Time: %.6f\n", frame_time_);

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// BVHWriter::build_frame
// ─────────────────────────────────────────────────────────────────────────────
void BVHWriter::build_frame(const fsb::MHRResult& r, float* buf) const
{
    // ── Root joint (hip) – channels 0-5
    //    Xposition Yposition Zposition  Zrotation Yrotation Xrotation
    buf[0] = r.pred_cam_t[0] * POS_SCALE;
    buf[1] = r.pred_cam_t[1] * POS_SCALE;
    buf[2] = r.pred_cam_t[2] * POS_SCALE;
    // global_rot = [rz, ry, rx] → direct map to Zrot/Yrot/Xrot
    buf[3] = r.global_rot[0] * RAD2DEG;
    buf[4] = r.global_rot[1] * RAD2DEG;
    buf[5] = r.global_rot[2] * RAD2DEG;

    // ── Major body joints (BVH ZXY: Zrot Xrot Yrot; SMPL-like ZYX → converted via zyx_to_zxy)
    //
    // Source for each body_pose index:
    //   idx 0..61   → mhr_model_params[6+idx] == body_pose[idx]  (spine, hips, knees …)
    //   idx 62..115 → mhr_model_params[6+idx]  ← apply_hand_pose places the correct
    //                  arm/collar/head angles here via the hand-PCA decode.  r.body_pose
    //                  at these indices is near-zero (raw FFN compact-rep output).
    //   idx 116..129→ mhr_model_params[6+idx] == body_pose[idx]  (remaining joints)
    //   idx 130..132→ body_pose[idx]  (jaw; [6+130] would land in the scales region)
    if ((int)r.body_pose.size() >= 133 &&
        (int)r.mhr_model_params.size() >= 136) {
        const float* bp = r.body_pose.data();
        const float* mp = r.mhr_model_params.data();
        for (int j = 0; j < 23; ++j) {
            const ResolvedJoint& rj = resolved_[j];
            if (rj.bvh_offset < 0) continue;
            // For each channel pick the source with valid arm angles.
            auto get = [&](int idx) -> float {
                return (idx < 130) ? mp[6 + idx] : bp[idx];
            };
            // SMPL-like data gives ZYX Euler (rx,ry,rz); BVH "Zrot Xrot Yrot" is ZXY.
            // Re-decompose so the BVH player reconstructs the exact same rotation.
            float za, xb, yc;
            zyx_to_zxy(get(rj.rx_idx), get(rj.ry_idx), get(rj.rz_idx), za, xb, yc);
            buf[rj.bvh_offset + 0] = za * RAD2DEG;   // Zrotation
            buf[rj.bvh_offset + 1] = xb * RAD2DEG;   // Xrotation
            buf[rj.bvh_offset + 2] = yc * RAD2DEG;   // Yrotation
        }
    }

}
// Note: finger/hand BVH channels are left at zero.
// The MHR model's "hand PCA" encodes arm joints (collar, shoulder, elbow, wrist),
// not individual finger poses.  There is no per-finger prediction in this model.

// ─────────────────────────────────────────────────────────────────────────────
// BVHWriter::write_frame
// ─────────────────────────────────────────────────────────────────────────────
void BVHWriter::write_frame(const std::vector<fsb::MHRResult>& results)
{
    if (!file_) return;

    static constexpr int STACK_MAX = 1024;
    float        stack_buf[STACK_MAX];
    std::vector<float> heap_buf;
    float* buf;

    if (total_channels_ <= STACK_MAX) {
        buf = stack_buf;
    } else {
        heap_buf.assign(total_channels_, 0.f);
        buf = heap_buf.data();
    }
    memset(buf, 0, total_channels_ * sizeof(float));

    if (!results.empty())
        build_frame(results[0], buf);

    for (int i = 0; i < total_channels_; ++i) {
        if (i > 0) fputc(' ', file_);
        if (buf[i] == 0.f) fputc('0', file_);
        else fprintf(file_, "%.4f", buf[i]);
    }
    fputc('\n', file_);

    ++frame_count_;
}

// ─────────────────────────────────────────────────────────────────────────────
// BVHWriter::close
// ─────────────────────────────────────────────────────────────────────────────
void BVHWriter::close()
{
    if (!file_) return;
    fseek(file_, frames_pos_, SEEK_SET);
    fprintf(file_, "Frames: %10d\n", frame_count_);
    fclose(file_);
    file_ = nullptr;
    printf("[BVHWriter] Wrote %d frame(s).\n", frame_count_);
}
