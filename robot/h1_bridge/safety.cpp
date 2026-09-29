#include "safety.h"

#include <algorithm>
#include <cmath>

#include "../teleop_shm.h"

namespace h1b {

const JointVec kHome = {0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0,
                        0.0};

const char* mode_name(Mode m) {
    switch (m) {
        case Mode::Disarmed: return "DISARMED";
        case Mode::RampIn: return "RAMP_IN";
        case Mode::Tracking: return "TRACKING";
        case Mode::RampOut: return "RAMP_OUT";
        case Mode::Fault: return "FAULT";
    }
    return "?";
}

const char* fault_name(FaultCode f) {
    switch (f) {
        case FaultCode::None: return "none";
        case FaultCode::StateStale: return "lowstate stale/invalid";
        case FaultCode::MotorError: return "motor error";
        case FaultCode::OverTemp: return "motor over-temperature";
        case FaultCode::TrackingError: return "tracking error (obstructed?)";
        case FaultCode::TargetLost: return "target stream lost";
        case FaultCode::TargetInvalid: return "target stream invalid";
        case FaultCode::LoopStall: return "control loop stalled";
        case FaultCode::EStop: return "E-STOP";
    }
    return "?";
}

Supervisor::Supervisor(const Config& cfg) : cfg_(cfg) {
    cfg_.speed_scale = std::clamp(cfg_.speed_scale, 0.0, 1.0);
    cfg_.gain_scale = std::clamp(cfg_.gain_scale, 0.0, 1.0);
    cfg_.limit_margin = std::clamp(cfg_.limit_margin, 0.02, 0.4);
    for (int j = 0; j < kNumJoints; ++j) {
        const JointSpec& s = kJoints[j];
        const double m = cfg_.limit_margin * (s.hi - s.lo);
        lo_[j] = s.lo + m;
        hi_[j] = s.hi - m;
        vmax_[j] = std::min<double>(cfg_.vmax, s.vel) * cfg_.speed_scale;
    }
    lo_[kWaist] = std::max(lo_[kWaist], -cfg_.waist_abs);
    hi_[kWaist] = std::min(hi_[kWaist], cfg_.waist_abs);
}

void Supervisor::fault(FaultCode f, const char* note) {
    if (mode_ == Mode::Fault) return;          // keep the FIRST cause
    mode_ = Mode::Fault;
    fault_ = f;
    note_ = note;
    vel_.fill(0.0);
}

void Supervisor::intake(const TargetIn& t) {
    if (!t.have || t.counter <= last_counter_) return;   // nothing new (or replayed)
    last_counter_ = t.counter;
    if (t.flags & TELEOP_F_SHUTDOWN) { shutdown_ = true; return; }
    if (!(t.flags & TELEOP_F_TRACKING)) return;           // writer's own idle easing: ignore

    JointVec c{};
    bool ok = true;
    for (int j = 0; j < kNumJoints; ++j) {
        if (!std::isfinite(t.q[j])) { ok = false; break; }
        c[j] = std::clamp(t.q[j], lo_[j], hi_[j]);
    }
    // Jump check only against a RECENT target: after a gap (person lost and re-found,
    // or re-arming) the new pose may legitimately be far away, and the follower's
    // velocity limit is what makes moving there safe.
    if (ok && have_accepted_ && accepted_age_ <= cfg_.target_hold_s)
        for (int j = 0; j < kNumJoints; ++j)
            if (std::fabs(c[j] - accepted_[j]) > cfg_.jump_max) { ok = false; break; }
    if (!ok) {
        if (++rejects_ >= cfg_.jump_reject_fault)
            fault(FaultCode::TargetInvalid, "too many NaN / jumping target frames");
        return;
    }
    rejects_ = 0;
    accepted_ = c;
    have_accepted_ = true;
    accepted_age_ = std::max(0.0, t.age_s);
}

// Velocity- and acceleration-limited first-order follower toward goal_, with a
// self-collision gate on every single step (so the PATH is checked, not just the goal).
void Supervisor::follow(double dt) {
    JointVec next = cmd_, v = vel_;
    for (int j = 0; j < kNumJoints; ++j) {
        const double e = goal_[j] - cmd_[j];
        const double vdes = std::clamp(e / cfg_.follow_tau, -vmax_[j], vmax_[j]);
        const double amax = cfg_.amax * cfg_.speed_scale;
        v[j] = vel_[j] + std::clamp(vdes - vel_[j], -amax * dt, amax * dt);
        v[j] = std::clamp(v[j], -vmax_[j], vmax_[j]);
        next[j] = cmd_[j] + v[j] * dt;
        if ((goal_[j] - next[j]) * e < 0) { next[j] = goal_[j]; v[j] = 0; }   // no overshoot
        next[j] = std::clamp(next[j], lo_[j], hi_[j]);
    }

    blocked_ = false;
    const double pen_now = self_collision(cmd_, cfg_.collision).penetration;
    auto acceptable = [&](const JointVec& q) {
        const double p = self_collision(q, cfg_.collision).penetration;
        return p <= 0.0 || p < pen_now - 1e-6;     // free, or strictly getting out
    };
    if (!acceptable(next)) {
        blocked_ = true;
        // Freeze the offending part(s): try reverting left arm / right arm / waist.
        static const int kParts[7] = {1, 2, 4, 3, 5, 6, 7};   // bitmask: 1=L, 2=R, 4=waist
        for (int mask : kParts) {
            JointVec t = next, tv = v;
            for (int j = 0; j < kNumJoints; ++j) {
                const int part = j < kR0 ? 1 : (j < kWaist ? 2 : 4);
                if (mask & part) { t[j] = cmd_[j]; tv[j] = 0; }
            }
            if (mask == 7 || acceptable(t)) { next = t; v = tv; break; }
        }
    }
    cmd_ = next;
    vel_ = v;
}

Out Supervisor::step(double dt, const StateIn& st, const TargetIn& tgt, bool external_estop) {
    Out o;
    const uint16_t pressed = st.buttons & ~prev_buttons_;   // rising edges
    prev_buttons_ = st.buttons;
    const bool state_ok = st.valid && st.age_s <= cfg_.state_stale_s;
    const bool deadman = state_ok && (st.buttons & cfg_.deadman) == cfg_.deadman;

    accepted_age_ += dt;
    intake(tgt);

    // ── universal checks: valid in every mode ────────────────────────────────
    if (external_estop || (state_ok && (st.buttons & cfg_.estop)))
        fault(FaultCode::EStop, "e-stop");
    if (mode_ != Mode::Disarmed && dt > cfg_.loop_stall_s)
        fault(FaultCode::LoopStall, "control tick late");
    if (mode_ != Mode::Disarmed && !state_ok)
        fault(FaultCode::StateStale, "lowstate stale or invalid");
    if (state_ok && st.motor_error) fault(FaultCode::MotorError, "a controlled motor reports an error");
    if (state_ok && st.max_temp_c >= cfg_.temp_fault_c) fault(FaultCode::OverTemp, "motor too hot");

    if (!have_cmd_ && state_ok) {
        for (int j = 0; j < kNumJoints; ++j) cmd_[j] = st.q[j];
        have_cmd_ = true;
    }

    switch (mode_) {
        case Mode::Disarmed: {
            weight_ = 0.0;
            vel_.fill(0.0);
            if (state_ok)                         // follow the measured pose: no jump on arming
                for (int j = 0; j < kNumJoints; ++j) cmd_[j] = st.q[j];
            goal_ = cmd_;
            note_ = "disarmed";
            if ((pressed & cfg_.arm) && deadman) {
                if (!state_ok) note_ = "cannot arm: no valid lowstate";
                else if (!have_accepted_ || accepted_age_ > cfg_.target_hold_s)
                    note_ = "cannot arm: no fresh target pose";
                else {
                    mode_ = Mode::RampIn;
                    shutdown_ = false;
                    track_err_t_ = 0.0;
                    note_ = "ramping in";
                }
            }
            break;
        }
        case Mode::RampIn:
        case Mode::Tracking: {
            if (!deadman) { mode_ = Mode::RampOut; note_ = "deadman released"; break; }
            if (shutdown_) { mode_ = Mode::RampOut; note_ = "writer shut down"; break; }
            if (accepted_age_ <= cfg_.target_hold_s) { goal_ = accepted_; note_ = "tracking"; }
            else if (accepted_age_ <= cfg_.target_home_s) { note_ = "target late: holding"; }
            else if (accepted_age_ <= cfg_.target_lost_s) {
                for (int j = 0; j < kNumJoints; ++j) goal_[j] = std::clamp(kHome[j], lo_[j], hi_[j]);
                note_ = "target lost: easing home";
            } else {
                fault(FaultCode::TargetLost, "no target pose for too long");
                break;
            }
            follow(dt);
            if (mode_ == Mode::RampIn) {
                weight_ = std::min(1.0, weight_ + dt / cfg_.ramp_in_s);
                if (weight_ >= 1.0) mode_ = Mode::Tracking;
            } else {
                double worst = 0.0;
                for (int j = 0; j < kNumJoints; ++j) worst = std::max(worst, std::fabs(cmd_[j] - st.q[j]));
                track_err_t_ = worst > cfg_.track_err_max ? track_err_t_ + dt : 0.0;
                if (track_err_t_ > cfg_.track_err_s)
                    fault(FaultCode::TrackingError, "joints not following the command");
            }
            break;
        }
        case Mode::RampOut:
        case Mode::Fault: {
            vel_.fill(0.0);                       // command frozen, weight hands control back
            weight_ = std::max(0.0, weight_ - dt / cfg_.ramp_out_s);
            if (mode_ == Mode::RampOut && weight_ <= 0.0) { mode_ = Mode::Disarmed; note_ = "disarmed"; }
            if (mode_ == Mode::Fault && weight_ <= 0.0 && (pressed & cfg_.arm) && !(st.buttons & cfg_.deadman)) {
                mode_ = Mode::Disarmed;           // acknowledged; arming is a separate press
                fault_ = FaultCode::None;
                rejects_ = 0;
                note_ = "fault acknowledged";
            }
            break;
        }
    }

    o.mode = mode_;
    o.fault = fault_;
    o.weight = static_cast<float>(weight_);
    for (int j = 0; j < kNumJoints; ++j) {
        o.q[j] = cmd_[j];
        o.kp[j] = static_cast<float>(kJoints[j].kp * cfg_.gain_scale);
        o.kd[j] = static_cast<float>(kJoints[j].kd * cfg_.gain_scale);
    }
    o.collision_blocked = blocked_;
    o.rejected_targets = rejects_;
    o.target_age_s = accepted_age_;
    o.note = note_;
    return o;
}

}  // namespace h1b
