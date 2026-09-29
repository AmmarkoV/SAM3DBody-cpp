// safety.h — the H1-2 teleop safety supervisor.
//
// Pure logic, no I/O and no SDK: every input arrives through step() and every
// decision leaves through its return value, so the whole state machine is
// unit-tested tick by tick (test_bridge.cpp) without a robot.
//
// Nothing upstream is trusted (camera, model, IK, Python can all emit garbage or
// stall).  The supervisor alone decides what reaches the motors:
//
//   DISARMED ──(deadman held + arm button)──► RAMP_IN ──(weight=1)──► TRACKING
//      ▲                                         │                       │
//      │                          deadman released / shutdown frame      │
//      └──────────────── RAMP_OUT ◄──────────────┴───────────────────────┘
//   any fault (any state) ──► FAULT: ramp weight to 0, then stay latched until the
//   operator acknowledges (arm button with the deadman RELEASED).
#pragma once
#include <cstdint>

#include "kinematics.h"

namespace h1b {

enum class Mode : uint8_t { Disarmed, RampIn, Tracking, RampOut, Fault };
const char* mode_name(Mode m);

enum class FaultCode : uint8_t {
    None,
    StateStale,      // rt/lowstate missing, late or failed its CRC
    MotorError,      // a controlled motor reports an error word
    OverTemp,
    TrackingError,   // measured joints do not follow the command (obstruction / collision)
    TargetLost,      // no fresh retargeted pose for target_lost_s
    TargetInvalid,   // too many consecutive NaN / jumping targets
    LoopStall,       // the control loop itself ran late
    EStop,           // remote e-stop button, terminal key or SIGINT
};
const char* fault_name(FaultCode f);

// Remote buttons (lowstate.wireless_remote bytes 2-3; SDK xKeySwitchUnion bit order).
enum Button : uint16_t {
    kR1 = 1u << 0, kL1 = 1u << 1, kStart = 1u << 2, kSelect = 1u << 3,
    kR2 = 1u << 4, kL2 = 1u << 5, kF1 = 1u << 6, kF2 = 1u << 7,
    kA = 1u << 8, kB = 1u << 9, kX = 1u << 10, kY = 1u << 11,
    kUp = 1u << 12, kRight = 1u << 13, kDown = 1u << 14, kLeft = 1u << 15,
};

struct Config {
    // speed: harness default is 25% of vmax; >0.25 must be explicitly confirmed by main
    double speed_scale = 0.25;
    double vmax = 2.0;              // rad/s at speed_scale 1 (also capped by the URDF limit)
    double amax = 10.0;             // rad/s^2 at speed_scale 1
    double follow_tau = 0.08;       // s, time constant of the command follower
    double limit_margin = 0.10;     // fraction of each joint's range kept clear of its limits
    double waist_abs = 0.5;         // rad, waist yaw cap (URDF allows 2.35)
    double gain_scale = 1.0;        // <= 1: kp/kd never exceed Unitree's reference gains
    double ramp_in_s = 2.0;
    double ramp_out_s = 1.0;
    double target_hold_s = 0.15;    // older target -> hold the current goal
    double target_home_s = 1.0;     // older -> ease to the home pose
    double target_lost_s = 3.0;     // older -> fault
    double jump_max = 0.8;          // rad between consecutive accepted targets
    int jump_reject_fault = 5;      // consecutive rejected targets -> fault
    double state_stale_s = 0.05;
    double track_err_max = 0.35;    // rad
    double track_err_s = 0.2;
    int temp_fault_c = 85;
    double loop_stall_s = 0.05;
    CollisionGeometry collision;
    uint16_t deadman = kR1;         // held the whole time
    uint16_t arm = kX;              // pressed (with deadman held) to arm; alone to ack a fault
    uint16_t estop = kB;            // pressed at any time -> fault
};

struct StateIn {
    bool valid = false;             // received and passed CRC
    double age_s = 1e9;             // since that lowstate was received
    double q[kNumJoints] = {};
    bool motor_error = false;
    int max_temp_c = 0;
    uint16_t buttons = 0;
};

struct TargetIn {
    bool have = false;              // a frame is available (maybe not new)
    uint64_t counter = 0;
    double age_s = 1e9;             // since the writer published it
    uint32_t flags = 0;             // TELEOP_F_*
    double q[kNumJoints] = {};      // already remapped to the bridge's joint order
};

struct Out {
    Mode mode = Mode::Disarmed;
    FaultCode fault = FaultCode::None;
    float weight = 0.f;             // arm_sdk blend weight, [0,1]
    double q[kNumJoints] = {};      // commanded joint positions
    float kp[kNumJoints] = {};
    float kd[kNumJoints] = {};
    bool collision_blocked = false; // a step was refused this tick by the self-collision gate
    int rejected_targets = 0;       // consecutive rejected target frames
    double target_age_s = 1e9;
    const char* note = "";          // why we are where we are (for the operator)
};

// Safe home pose: Unitree's own init pose from h1_2_arm_sdk_dds_example (arms
// slightly abducted so they clear the torso).
extern const JointVec kHome;

class Supervisor {
public:
    explicit Supervisor(const Config& cfg);
    Out step(double dt, const StateIn& state, const TargetIn& target, bool external_estop);
    const Config& config() const { return cfg_; }
    // Joint limits actually enforced (URDF shrunk by limit_margin, waist capped).
    double lo(int j) const { return lo_[j]; }
    double hi(int j) const { return hi_[j]; }
    double vmax(int j) const { return vmax_[j]; }   // rad/s, speed_scale applied

private:
    void fault(FaultCode f, const char* note);
    void intake(const TargetIn& t);
    void follow(double dt);

    Config cfg_;
    double lo_[kNumJoints], hi_[kNumJoints], vmax_[kNumJoints];
    Mode mode_ = Mode::Disarmed;
    FaultCode fault_ = FaultCode::None;
    const char* note_ = "disarmed";
    double weight_ = 0.0;
    JointVec cmd_{}, vel_{}, goal_{}, accepted_{};
    bool have_accepted_ = false, have_cmd_ = false;
    uint64_t last_counter_ = 0;
    double accepted_age_ = 1e9;     // age of the accepted target, advanced every tick
    int rejects_ = 0;
    bool shutdown_ = false;
    double track_err_t_ = 0.0;
    uint16_t prev_buttons_ = 0;
    bool blocked_ = false;
};

}  // namespace h1b
