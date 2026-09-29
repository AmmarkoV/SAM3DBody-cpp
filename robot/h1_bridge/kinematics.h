// kinematics.h — H1-2 upper-body forward kinematics + conservative self-collision.
//
// All geometry is generated from the URDF / MJCF (h1_2_model.h).  FK is checked
// against MuJoCo in test_bridge.cpp.  Everything is in the PELVIS frame, with the
// legs assumed at the standing zero pose (the bridge only runs upper-body teleop
// while Unitree's locomotion controller keeps the robot standing).
#pragma once
#include <array>

#include "h1_2_model.h"

namespace h1b {

using Vec3 = std::array<double, 3>;
using JointVec = std::array<double, kNumJoints>;

// Points of one arm that are tested for collision.
enum ArmPoint { kElbow = 0, kWrist = 1, kHandMid = 2, kHandTip = 3, kNumArmPoints = 4 };

struct ArmPoints {
    Vec3 p[2][kNumArmPoints];   // [left,right][ArmPoint], pelvis frame
    double torso_R[3][3];       // torso_link orientation (waist yaw)
};

struct CollisionGeometry {
    double hand_len = 0.17;       // wrist_yaw origin -> fingertips (m).  Dex5-1P: VERIFY on the hand.
    double r_elbow = 0.035;       // sphere radius around each tested point (m)
    double r_wrist = 0.035;
    double r_hand_mid = 0.045;
    double r_hand_tip = 0.03;
    double box_margin = 0.02;     // extra clearance added to every body box (m)
    double min_hand_gap = 0.10;   // hand-to-hand clearance (m): no clapping
};

ArmPoints forward_kinematics(const JointVec& q, double hand_len);

// Elbow / wrist_yaw link origins, exactly as MuJoCo reports them (for the FK test).
void link_origins(const JointVec& q, Vec3 out[4]);

struct CollisionReport {
    double penetration = 0.0;     // sum of all penetrations (m); 0 = collision-free
    const char* worst = nullptr;  // human-readable name of the deepest contact
};

CollisionReport self_collision(const JointVec& q, const CollisionGeometry& g);

}  // namespace h1b
