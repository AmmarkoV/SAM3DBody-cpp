// manual.h — slider teleop (h1_bridge --manual): the operator's target source.
//
// Pure logic, no UI and no SDK (unit-tested in test_bridge.cpp).  The UI only edits a
// PENDING pose; nothing moves until Apply, and Apply is accepted only if
// check_manual_pose() passes.  The accepted pose is then walked toward along a
// straight joint-space line, slower than the Supervisor's follower, so the path that
// was collision-checked is the path that is actually driven.  Everything still goes
// through the Supervisor (limits, velocity, per-step collision gate, deadman).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "safety.h"

namespace h1b {

// Fraction of each joint's Supervisor vmax the manual target moves at (< 1, so the
// follower keeps up and the driven path stays on the checked straight line).
constexpr double kManualSpeedFrac = 0.8;

struct PoseCheck {
    bool ok = false;
    std::string verdict;              // one line for the operator
    std::vector<std::string> notes;   // non-blocking remarks (near a limit, large move)
    int n_changed = 0;                // joints that move by more than 1 mrad
    int max_joint = -1;               // joint with the largest move
    double max_delta = 0.0;           // rad
    double eta_s = 0.0;               // time the manual walk takes to get there
};

// Is moving from `from` to `to` sane?  Blocks on: a non-finite value, a joint outside
// the Supervisor's ENFORCED range, a self-colliding target, or a straight-line path
// that collides (deeper than `from` already does).
PoseCheck check_manual_pose(const Supervisor& sup, const JointVec& from, const JointVec& to);

class ManualSource {
public:
    explicit ManualSource(const Supervisor& sup) : sup_(sup) {}

    // Start at the measured pose (clamped into the enforced range): arming then moves nothing.
    void init(const JointVec& q);
    bool ready() const { return ready_; }

    // Re-checks from the CURRENT published target; the goal changes only if it passes.
    PoseCheck apply(const JointVec& to);

    // One control tick: advance toward the goal and return a fresh target frame.
    TargetIn tick(double dt);

    const JointVec& goal() const { return goal_; }
    const JointVec& published() const { return pub_; }
    bool moving() const;

private:
    const Supervisor& sup_;
    JointVec goal_{}, pub_{};
    bool ready_ = false;
    uint64_t counter_ = 0;
};

}  // namespace h1b
