#include "manual.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "../teleop_shm.h"

namespace h1b {

namespace {

std::string fmt(const char* f, double a, double b = 0, double c = 0) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), f, a, b, c);
    return buf;
}

double manual_speed(const Supervisor& sup, int j) { return kManualSpeedFrac * sup.vmax(j); }

}  // namespace

PoseCheck check_manual_pose(const Supervisor& sup, const JointVec& from, const JointVec& to) {
    PoseCheck r;
    const CollisionGeometry& g = sup.config().collision;
    for (int j = 0; j < kNumJoints; ++j) {
        if (!std::isfinite(to[j]) || !std::isfinite(from[j])) {
            r.verdict = std::string("BLOCKED: non-finite value for ") + kJoints[j].name;
            return r;
        }
        if (to[j] < sup.lo(j) - 1e-9 || to[j] > sup.hi(j) + 1e-9) {
            r.verdict = std::string("BLOCKED: ") + kJoints[j].name +
                        fmt(" = %.3f outside the enforced range [%.2f, %.2f]", to[j], sup.lo(j), sup.hi(j));
            return r;
        }
        const double d = std::fabs(to[j] - from[j]);
        if (d > 1e-3) ++r.n_changed;
        if (d > r.max_delta) { r.max_delta = d; r.max_joint = j; }
        if (d > 1e-3) r.eta_s = std::max(r.eta_s, d / manual_speed(sup, j));
    }

    const CollisionReport end = self_collision(to, g);
    if (end.penetration > 0.0) {
        r.verdict = std::string("BLOCKED: target pose self-collides (") + (end.worst ? end.worst : "?") +
                    fmt(", %.0f mm)", end.penetration * 1000.0);
        return r;
    }
    // Straight-line path, sampled every <= 0.02 rad on the joint that moves most.
    const double pen0 = self_collision(from, g).penetration;
    const int n = std::max(1, int(std::ceil(r.max_delta / 0.02)));
    for (int i = 1; i < n; ++i) {
        const double s = double(i) / n;
        JointVec q;
        for (int j = 0; j < kNumJoints; ++j) q[j] = from[j] + s * (to[j] - from[j]);
        const CollisionReport c = self_collision(q, g);
        if (c.penetration > 0.0 && c.penetration > pen0 + 1e-6) {
            r.verdict = std::string("BLOCKED: path self-collides at ") + fmt("%.0f%%", 100.0 * s) + " (" +
                        (c.worst ? c.worst : "?") + "). Go via an intermediate pose.";
            return r;
        }
    }

    for (int j = 0; j < kNumJoints; ++j) {
        const double span = sup.hi(j) - sup.lo(j);
        if (std::fabs(to[j] - from[j]) > 1e-3 &&
            (to[j] - sup.lo(j) < 0.05 * span || sup.hi(j) - to[j] < 0.05 * span))
            r.notes.push_back(std::string("near its limit: ") + kJoints[j].name);
    }
    if (r.max_delta > 1.0)
        r.notes.push_back(std::string("large move: ") + kJoints[r.max_joint].name + fmt(" by %.2f rad", r.max_delta));

    r.ok = true;
    if (r.n_changed == 0)
        r.verdict = "OK: no change";
    else
        r.verdict = fmt("OK: %.0f joint(s), max move %.2f rad, about %.1f s", r.n_changed, r.max_delta, r.eta_s);
    return r;
}

PoseCheck check_takeover(const Supervisor& sup, bool driven, const JointVec& cmd, const JointVec& to) {
    PoseCheck r;
    for (int j = 0; j < kNumJoints; ++j) {
        if (!std::isfinite(to[j]) || !std::isfinite(cmd[j])) {
            r.verdict = std::string("BLOCKED: non-finite value for ") + kJoints[j].name;
            return r;
        }
        // The Supervisor clamps targets into the enforced range: compare what it will drive.
        const double d = std::fabs(std::clamp(to[j], sup.lo(j), sup.hi(j)) - cmd[j]);
        if (d > 1e-3) ++r.n_changed;
        if (d > r.max_delta) { r.max_delta = d; r.max_joint = j; }
    }
    if (!driven) {
        r.ok = true;
        r.verdict = "OK: arms not driven";
        return r;
    }
    if (r.max_delta > kTakeoverTol) {
        r.verdict = std::string("BLOCKED while armed: ") + kJoints[r.max_joint].name +
                    fmt(" is %.2f rad from the arms (pick-up needs <= %.2f). Match the pose, or release R1 first.",
                        r.max_delta, kTakeoverTol);
        return r;
    }
    r.ok = true;
    r.verdict = fmt("OK: pick-up within %.2f rad", r.max_delta);
    return r;
}

void ManualSource::init(const JointVec& q) {
    for (int j = 0; j < kNumJoints; ++j) pub_[j] = std::clamp(q[j], sup_.lo(j), sup_.hi(j));
    goal_ = pub_;
    ready_ = true;
}

PoseCheck ManualSource::apply(const JointVec& to) {
    PoseCheck c;
    if (!ready_) { c.verdict = "BLOCKED: no robot state yet"; return c; }
    c = check_manual_pose(sup_, pub_, to);
    if (c.ok) goal_ = to;
    return c;
}

bool ManualSource::moving() const {
    for (int j = 0; j < kNumJoints; ++j)
        if (std::fabs(goal_[j] - pub_[j]) > 1e-9) return true;
    return false;
}

TargetIn ManualSource::tick(double dt) {
    TargetIn t;
    if (!ready_) return t;
    // Largest step along the line that keeps every joint under its manual speed.
    double alpha = 1.0;
    for (int j = 0; j < kNumJoints; ++j) {
        const double rem = std::fabs(goal_[j] - pub_[j]);
        if (rem > 1e-12) alpha = std::min(alpha, manual_speed(sup_, j) * dt / rem);
    }
    for (int j = 0; j < kNumJoints; ++j) pub_[j] += alpha * (goal_[j] - pub_[j]);
    t.have = true;
    t.counter = ++counter_;
    t.age_s = 0.0;
    t.flags = TELEOP_F_TRACKING;
    for (int j = 0; j < kNumJoints; ++j) t.q[j] = pub_[j];
    return t;
}

}  // namespace h1b
