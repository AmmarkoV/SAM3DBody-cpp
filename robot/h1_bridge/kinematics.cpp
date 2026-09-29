#include "kinematics.h"

#include <algorithm>
#include <cmath>

namespace h1b {
namespace {

struct Tf {
    double R[3][3];
    double p[3];
};

Tf identity() {
    Tf t{};
    for (int i = 0; i < 3; ++i) t.R[i][i] = 1.0;
    return t;
}

// Rodrigues rotation about a unit axis.
void axis_angle(const float ax[3], double th, double M[3][3]) {
    const double x = ax[0], y = ax[1], z = ax[2];
    const double c = std::cos(th), s = std::sin(th), C = 1.0 - c;
    M[0][0] = c + x * x * C;     M[0][1] = x * y * C - z * s; M[0][2] = x * z * C + y * s;
    M[1][0] = y * x * C + z * s; M[1][1] = c + y * y * C;     M[1][2] = y * z * C - x * s;
    M[2][0] = z * x * C - y * s; M[2][1] = z * y * C + x * s; M[2][2] = c + z * z * C;
}

// parent * origin(joint) * rot(axis, q)
Tf child(const Tf& parent, const JointSpec& j, double q) {
    Tf t;
    for (int i = 0; i < 3; ++i) {
        t.p[i] = parent.p[i];
        for (int k = 0; k < 3; ++k) t.p[i] += parent.R[i][k] * j.origin[k];
    }
    double RO[3][3];
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k) {
            RO[i][k] = 0.0;
            for (int m = 0; m < 3; ++m) RO[i][k] += parent.R[i][m] * j.R[m][k];
        }
    double Rq[3][3];
    axis_angle(j.axis, q, Rq);
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k) {
            t.R[i][k] = 0.0;
            for (int m = 0; m < 3; ++m) t.R[i][k] += RO[i][m] * Rq[m][k];
        }
    return t;
}

Vec3 apply(const Tf& t, double x, double y, double z) {
    return {t.R[0][0] * x + t.R[0][1] * y + t.R[0][2] * z + t.p[0],
            t.R[1][0] * x + t.R[1][1] * y + t.R[1][2] * z + t.p[1],
            t.R[2][0] * x + t.R[2][1] * y + t.R[2][2] * z + t.p[2]};
}

// Walk torso -> 7 arm joints for one side; returns the transforms of every link.
void arm_chain(const JointVec& q, int side, Tf& torso, Tf links[7]) {
    torso = child(identity(), kJoints[kWaist], q[kWaist]);
    const int b = side == 0 ? kL0 : kR0;
    Tf t = torso;
    for (int i = 0; i < 7; ++i) {
        t = child(t, kJoints[b + i], q[b + i]);
        links[i] = t;
    }
}

// Signed distance from a point to an axis-aligned box (negative inside).
double box_sdf(const Vec3& p, const float lo[3], const float hi[3], double margin) {
    double out2 = 0.0, in = -1e9;
    for (int i = 0; i < 3; ++i) {
        const double l = lo[i] - margin, h = hi[i] + margin;
        const double d = std::max(l - p[i], p[i] - h);   // >0 outside along this axis
        if (d > 0) out2 += d * d;
        in = std::max(in, d);
    }
    return out2 > 0 ? std::sqrt(out2) : in;
}

double dist(const Vec3& a, const Vec3& b) {
    return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) +
                     (a[2] - b[2]) * (a[2] - b[2]));
}

}  // namespace

ArmPoints forward_kinematics(const JointVec& q, double hand_len) {
    ArmPoints out{};
    for (int side = 0; side < 2; ++side) {
        Tf torso, L[7];
        arm_chain(q, side, torso, L);
        if (side == 0)
            for (int i = 0; i < 3; ++i)
                for (int k = 0; k < 3; ++k) out.torso_R[i][k] = torso.R[i][k];
        out.p[side][kElbow] = apply(L[3], 0, 0, 0);            // elbow_link origin
        out.p[side][kWrist] = apply(L[6], 0, 0, 0);            // wrist_yaw_link origin
        out.p[side][kHandMid] = apply(L[6], 0.5 * hand_len, 0, 0);
        out.p[side][kHandTip] = apply(L[6], hand_len, 0, 0);
    }
    return out;
}

void link_origins(const JointVec& q, Vec3 out[4]) {
    for (int side = 0; side < 2; ++side) {
        Tf torso, L[7];
        arm_chain(q, side, torso, L);
        out[2 * side + 0] = apply(L[3], 0, 0, 0);
        out[2 * side + 1] = apply(L[6], 0, 0, 0);
    }
}

CollisionReport self_collision(const JointVec& q, const CollisionGeometry& g) {
    static const char* kPointName[2][kNumArmPoints] = {
        {"L elbow", "L wrist", "L hand", "L fingertips"},
        {"R elbow", "R wrist", "R hand", "R fingertips"}};
    const double radius[kNumArmPoints] = {g.r_elbow, g.r_wrist, g.r_hand_mid, g.r_hand_tip};

    const ArmPoints a = forward_kinematics(q, g.hand_len);
    CollisionReport rep;
    double worst = 0.0;
    auto add = [&](double pen, const char* what) {
        if (pen <= 0) return;
        rep.penetration += pen;
        if (pen > worst) { worst = pen; rep.worst = what; }
    };

    for (int side = 0; side < 2; ++side)
        for (int k = 0; k < kNumArmPoints; ++k) {
            const Vec3& p = a.p[side][k];
            // torso frame = pelvis origin rotated by the waist: p_torso = R^T p
            Vec3 pt{};
            for (int i = 0; i < 3; ++i)
                pt[i] = a.torso_R[0][i] * p[0] + a.torso_R[1][i] * p[1] + a.torso_R[2][i] * p[2];
            for (const CollisionBox& b : kBoxes) {
                const double sd = box_sdf(b.frame == 1 ? pt : p, b.lo, b.hi, g.box_margin);
                add(radius[k] - sd, kPointName[side][k]);
            }
        }
    // hands (and wrists) must not meet each other
    for (int i = kWrist; i < kNumArmPoints; ++i)
        for (int j = kWrist; j < kNumArmPoints; ++j)
            add(g.min_hand_gap - dist(a.p[0][i], a.p[1][j]), "hands too close");
    return rep;
}

}  // namespace h1b
