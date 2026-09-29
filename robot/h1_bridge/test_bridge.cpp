// test_bridge.cpp — fault-injection + property tests for the H1-2 safety core.
// No robot, no SDK: the supervisor is driven tick by tick with synthetic inputs.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>

#include "../teleop_shm.h"
#include "h1_2_fk_golden.h"
#include "kinematics.h"
#include "safety.h"
#include "shm_reader.h"

using namespace h1b;

static int g_fail = 0, g_checks = 0;
#define CHECK(cond, ...)                                                        \
    do {                                                                        \
        ++g_checks;                                                             \
        if (!(cond)) {                                                          \
            ++g_fail;                                                           \
            std::printf("  FAIL %s:%d: %s  ", __FILE__, __LINE__, #cond);       \
            std::printf(__VA_ARGS__);                                           \
            std::printf("\n");                                                  \
        }                                                                       \
    } while (0)

// A robot whose joints follow the command perfectly while the weight is > 0.
struct Rig {
    Config cfg;
    Supervisor sup;
    StateIn st;
    TargetIn tg;
    uint64_t ctr = 0;
    Out o;
    bool robot_follows = true;
    explicit Rig(const Config& c = Config()) : cfg(c), sup(c) {
        st.valid = true;
        st.age_s = 0.002;
        for (int j = 0; j < kNumJoints; ++j) st.q[j] = kHome[j];
    }
    void send(const JointVec& q, uint32_t flags = TELEOP_F_TRACKING) {
        tg.have = true;
        tg.counter = ++ctr;
        tg.age_s = 0.01;
        tg.flags = flags;
        for (int j = 0; j < kNumJoints; ++j) tg.q[j] = q[j];
    }
    Out tick(double dt = 0.01, bool estop = false) {
        o = sup.step(dt, st, tg, estop);
        if (robot_follows && o.weight > 0)
            for (int j = 0; j < kNumJoints; ++j) st.q[j] = o.q[j];
        tg.age_s += dt;
        return o;
    }
    // keep streaming `q` at ~30 fps for `secs`
    void stream(const JointVec& q, double secs) {
        for (double t = 0; t < secs; t += 0.01) {
            if (std::fmod(t, 0.033) < 0.01) send(q);
            tick();
        }
    }
    void press(uint16_t b) {       // press + release b on top of whatever is held
        const uint16_t held = st.buttons;
        st.buttons = held | b; tick();
        st.buttons = held; tick();
    }
    // Stream a target that walks toward `q` at <= step rad/frame (like the Python
    // causal clamp), ~30 fps, for `secs`.  Returns true if the gate blocked any tick.
    bool approach(const JointVec& q, double secs, double step = 0.3) {
        bool blocked = false;
        for (double t = 0; t < secs; t += 0.01) {
            if (std::fmod(t, 0.033) < 0.01) {
                for (int j = 0; j < kNumJoints; ++j)
                    cur[j] += std::max(-step, std::min(step, q[j] - cur[j]));
                send(cur);
            }
            tick();
            blocked |= o.collision_blocked;
        }
        return blocked;
    }
    JointVec cur = kHome;          // last target sent by approach()
    void arm() {
        st.buttons = cfg.deadman;
        send(kHome);
        tick();
        press(cfg.arm);
    }
};

static JointVec pose(double lsp, double lsr, double lsy, double lel, double rsp, double rsr,
                     double rsy, double rel, double waist = 0) {
    JointVec q = kHome;
    q[kL0 + 0] = lsp; q[kL0 + 1] = lsr; q[kL0 + 2] = lsy; q[kL0 + 3] = lel;
    q[kR0 + 0] = rsp; q[kR0 + 1] = rsr; q[kR0 + 2] = rsy; q[kR0 + 3] = rel;
    q[kWaist] = waist;
    return q;
}

// ── tests ─────────────────────────────────────────────────────────────────────
static void test_fk_matches_mujoco() {
    double worst = 0;
    for (const auto& c : h1b_golden::kCases) {
        JointVec q;
        for (int j = 0; j < kNumJoints; ++j) q[j] = c.q[j];
        Vec3 p[4];
        link_origins(q, p);
        for (int k = 0; k < 4; ++k)
            for (int i = 0; i < 3; ++i) worst = std::max(worst, std::fabs(p[k][i] - c.p[k][i]));
    }
    CHECK(worst < 1e-5, "FK vs MuJoCo worst error %.3g m", worst);
}

static void test_starts_disarmed_and_follows_measured() {
    Rig r;
    r.st.q[kL0 + 3] = 0.7;
    Out o = r.tick();
    CHECK(o.mode == Mode::Disarmed && o.weight == 0.f, "mode %s w %.2f", mode_name(o.mode), o.weight);
    CHECK(std::fabs(o.q[kL0 + 3] - 0.7) < 1e-9, "disarmed command must mirror the measured pose");
}

static void test_arming_requires_deadman_and_fresh_target() {
    Rig r;
    r.send(kHome);
    r.press(r.cfg.arm);                                   // no deadman
    CHECK(r.o.mode == Mode::Disarmed, "armed without deadman");
    Rig r2;
    r2.st.buttons = r2.cfg.deadman;
    r2.tick();
    r2.press(r2.cfg.arm);                                 // no target ever
    CHECK(r2.o.mode == Mode::Disarmed, "armed without a target");
    Rig r3;
    r3.arm();
    CHECK(r3.o.mode == Mode::RampIn, "did not arm: %s", r3.o.note);
}

static void test_ramp_in_then_tracking_and_weight_rate() {
    Rig r;
    r.arm();
    float prev = r.o.weight;
    double t = 0;
    while (r.o.mode == Mode::RampIn && t < 5) {
        r.send(kHome);
        r.tick();
        CHECK(r.o.weight - prev <= 0.01 / r.cfg.ramp_in_s + 1e-6, "weight jumped %.4f", r.o.weight - prev);
        prev = r.o.weight;
        t += 0.01;
    }
    CHECK(r.o.mode == Mode::Tracking && r.o.weight == 1.f, "mode %s", mode_name(r.o.mode));
    CHECK(std::fabs(t - r.cfg.ramp_in_s) < 0.05, "ramp-in took %.2f s", t);
}

static void test_deadman_release_ramps_out() {
    Rig r;
    r.arm();
    r.stream(kHome, 2.5);
    r.st.buttons = 0;
    r.tick();
    CHECK(r.o.mode == Mode::RampOut, "mode %s", mode_name(r.o.mode));
    double t = 0.01;
    while (r.o.mode == Mode::RampOut && t < 3) { r.tick(); t += 0.01; }
    CHECK(r.o.mode == Mode::Disarmed && r.o.weight == 0.f, "mode %s", mode_name(r.o.mode));
    CHECK(std::fabs(t - r.cfg.ramp_out_s) < 0.05, "ramp-out took %.2f s", t);
}

static void test_state_stale_faults_and_latches() {
    Rig r;
    r.arm();
    r.stream(kHome, 2.5);
    r.st.age_s = 0.2;                                     // lowstate stopped arriving
    r.tick();
    CHECK(r.o.mode == Mode::Fault && r.o.fault == FaultCode::StateStale, "mode %s", mode_name(r.o.mode));
    r.st.age_s = 0.002;
    for (int i = 0; i < 200; ++i) r.tick();
    CHECK(r.o.weight == 0.f && r.o.mode == Mode::Fault, "fault must latch (mode %s)", mode_name(r.o.mode));
    r.st.buttons = r.cfg.deadman;
    r.press(r.cfg.arm);                                   // ack WITH deadman held: refused
    CHECK(r.o.mode == Mode::Fault, "ack must require the deadman released");
    r.st.buttons = 0;
    r.press(r.cfg.arm);
    CHECK(r.o.mode == Mode::Disarmed, "ack failed: %s", mode_name(r.o.mode));
}

static void test_nan_and_jump_targets_rejected_then_fault() {
    Rig r;
    r.arm();
    r.stream(kHome, 2.5);
    const double before = r.o.q[kL0 + 3];
    JointVec bad = kHome;
    bad[kL0 + 3] = std::nan("");
    r.send(bad); r.tick();
    CHECK(r.o.rejected_targets == 1 && std::fabs(r.o.q[kL0 + 3] - before) < 1e-6, "NaN leaked");
    JointVec jump = kHome;
    jump[kL0 + 3] += 1.5;
    for (int i = 0; i < r.cfg.jump_reject_fault - 1; ++i) { r.send(jump); r.tick(); }
    CHECK(r.o.mode == Mode::Fault && r.o.fault == FaultCode::TargetInvalid, "mode %s", mode_name(r.o.mode));
}

static void test_replayed_counter_ignored_and_target_timeouts() {
    Rig r;
    r.arm();
    JointVec q = kHome;
    q[kL0 + 3] = 0.8;
    r.stream(q, 2.5);
    CHECK(r.o.mode == Mode::Tracking, "mode %s", mode_name(r.o.mode));
    // stop sending: the same counter keeps being presented (reader returns the last frame)
    double t = 0;
    bool held = false, homed = false;
    while (r.o.mode == Mode::Tracking && t < 5) {
        r.tick(); t += 0.01;
        if (std::strstr(r.o.note, "holding")) held = true;
        if (std::strstr(r.o.note, "home")) homed = true;
    }
    CHECK(held && homed, "held=%d homed=%d", held, homed);
    CHECK(r.o.fault == FaultCode::TargetLost, "fault %s", fault_name(r.o.fault));
    CHECK(std::fabs(t - r.cfg.target_lost_s) < 0.1, "lost after %.2f s", t);
}

// Person lost for a moment, then re-found in a very different pose: the new pose
// must be accepted (the follower limits the motion), not rejected as a "jump".
static void test_reacquire_after_gap_accepts_far_pose() {
    Rig r;
    r.arm();
    r.stream(kHome, 2.5);
    for (int i = 0; i < 50; ++i) r.tick();               // 0.5 s without frames
    JointVec far = kHome;
    far[kL0 + 3] = 1.5;                                   // 1.5 rad from the last target
    r.stream(far, 4.0);
    CHECK(r.o.mode == Mode::Tracking && r.o.rejected_targets == 0, "mode %s rejects %d",
          mode_name(r.o.mode), r.o.rejected_targets);
    CHECK(std::fabs(r.o.q[kL0 + 3] - 1.5) < 1e-3, "did not reach the re-acquired pose (%.3f)", r.o.q[kL0 + 3]);
}

static void test_limits_and_velocity_are_enforced() {
    Rig r;
    r.arm();
    std::mt19937 rng(1);
    double worst_v = 0, worst_lim = 0;
    JointVec prev;
    for (int j = 0; j < kNumJoints; ++j) prev[j] = r.o.q[j];
    for (int k = 0; k < 150; ++k) {
        JointVec q;
        for (int j = 0; j < kNumJoints; ++j) {
            std::uniform_real_distribution<double> u(kJoints[j].lo - 1.0, kJoints[j].hi + 1.0);
            q[j] = u(rng);                                // deliberately beyond the URDF limits
        }
        r.sup = Supervisor(r.cfg);                        // fresh, so jumps are not rejected
        r.arm();
        r.cur = kHome;
        for (int i = 0; i < 60; ++i) {
            r.approach(q, 0.01);
            for (int j = 0; j < kNumJoints; ++j) {
                if (r.o.mode == Mode::Disarmed) break;
                const double vmax = std::min<double>(r.cfg.vmax, kJoints[j].vel) * r.cfg.speed_scale;
                worst_v = std::max(worst_v, std::fabs(r.o.q[j] - prev[j]) / 0.01 - vmax);
                worst_lim = std::max({worst_lim, r.sup.lo(j) - r.o.q[j], r.o.q[j] - r.sup.hi(j)});
            }
            for (int j = 0; j < kNumJoints; ++j) prev[j] = r.o.q[j];
        }
    }
    CHECK(worst_v <= 1e-6, "velocity limit exceeded by %.3g rad/s", worst_v);
    CHECK(worst_lim <= 1e-9, "joint limit exceeded by %.3g rad", worst_lim);
    CHECK(r.sup.hi(kWaist) <= r.cfg.waist_abs + 1e-12, "waist cap");
}

static void test_tracking_error_faults() {
    Rig r;
    r.arm();
    r.stream(kHome, 2.5);
    r.robot_follows = false;                              // arm is blocked by something
    JointVec q = kHome;
    q[kL0 + 3] = 0.6;
    r.stream(q, 3.0);
    CHECK(r.o.fault == FaultCode::TrackingError, "fault %s", fault_name(r.o.fault));
}

static void test_motor_error_temp_estop_stall() {
    { Rig r; r.arm(); r.stream(kHome, 2.5); r.st.motor_error = true; r.tick();
      CHECK(r.o.fault == FaultCode::MotorError, "%s", fault_name(r.o.fault)); }
    { Rig r; r.arm(); r.stream(kHome, 2.5); r.st.max_temp_c = 90; r.tick();
      CHECK(r.o.fault == FaultCode::OverTemp, "%s", fault_name(r.o.fault)); }
    { Rig r; r.arm(); r.stream(kHome, 2.5); r.st.buttons |= r.cfg.estop; r.tick();
      CHECK(r.o.fault == FaultCode::EStop, "%s", fault_name(r.o.fault)); }
    { Rig r; r.arm(); r.stream(kHome, 2.5); r.tick(0.01, true);
      CHECK(r.o.fault == FaultCode::EStop, "%s", fault_name(r.o.fault)); }
    { Rig r; r.arm(); r.stream(kHome, 2.5); r.tick(0.2);
      CHECK(r.o.fault == FaultCode::LoopStall, "%s", fault_name(r.o.fault)); }
    { Rig r; r.st.buttons = r.cfg.estop; r.tick();       // e-stop even while disarmed
      CHECK(r.o.mode == Mode::Fault, "%s", mode_name(r.o.mode)); }
}

static void test_shutdown_frame_ramps_out() {
    Rig r;
    r.arm();
    r.stream(kHome, 2.5);
    r.send(kHome, TELEOP_F_SHUTDOWN);
    r.tick();
    CHECK(r.o.mode == Mode::RampOut, "mode %s", mode_name(r.o.mode));
}

static void test_gain_ceiling() {
    Config c;
    c.gain_scale = 3.0;                                   // asks for MORE than Unitree's gains
    Rig r(c);
    r.tick();
    for (int j = 0; j < kNumJoints; ++j)
        CHECK(r.o.kp[j] <= kJoints[j].kp + 1e-6 && r.o.kd[j] <= kJoints[j].kd + 1e-6, "gain above ceiling");
}

static void test_home_pose_is_collision_free() {
    const CollisionReport rep = self_collision(kHome, CollisionGeometry{});
    CHECK(rep.penetration == 0.0, "home pose collides: %s (%.3f)", rep.worst ? rep.worst : "", rep.penetration);
}

// Random targets, many of which collide: the command must never ENTER collision.
static void test_collision_gate_never_enters_collision() {
    std::mt19937 rng(7);
    int blocked_ticks = 0, colliding_targets = 0;
    double worst = 0;
    Config c;
    c.speed_scale = 1.0;                                  // fast, to stress the gate
    for (int k = 0; k < 300; ++k) {
        Rig r(c);
        r.arm();
        JointVec q;
        do {                                              // odd trials: force a colliding target
            for (int j = 0; j < kNumJoints; ++j) {
                std::uniform_real_distribution<double> u(r.sup.lo(j), r.sup.hi(j));
                q[j] = u(rng);
            }
        } while ((k & 1) && self_collision(q, c.collision).penetration <= 0);
        if (self_collision(q, c.collision).penetration > 0) ++colliding_targets;
        for (int i = 0; i < 250; ++i) {
            r.approach(q, 0.01);
            JointVec cmd;
            for (int j = 0; j < kNumJoints; ++j) cmd[j] = r.o.q[j];
            worst = std::max(worst, self_collision(cmd, c.collision).penetration);
            blocked_ticks += r.o.collision_blocked;
        }
    }
    CHECK(worst == 0.0, "command entered collision (%.4f m)", worst);
    CHECK(colliding_targets >= 150 && blocked_ticks > 1000,
          "gate not exercised: %d colliding targets, %d blocked ticks", colliding_targets, blocked_ticks);
    std::printf("  (collision gate: %d/300 targets collided, %d blocked ticks)\n", colliding_targets, blocked_ticks);
}

static void test_clap_is_blocked() {
    // both arms forward, elbows bent, shoulders yawed inward: hands meet in front
    const JointVec clap = pose(-1.2, 0.05, -1.3, 1.2, -1.2, -0.05, 1.3, 1.2);
    Config c;
    Rig r(c);
    const CollisionReport target_rep = self_collision(clap, c.collision);
    r.arm();
    const bool blocked = r.approach(clap, 6.0);
    CHECK(blocked, "collision gate never engaged");
    JointVec cmd;
    for (int j = 0; j < kNumJoints; ++j) cmd[j] = r.o.q[j];
    CHECK(target_rep.penetration > 0, "test pose should collide (%s)", target_rep.worst ? target_rep.worst : "none");
    CHECK(self_collision(cmd, c.collision).penetration == 0.0, "clap reached");
}

static void test_crc_matches_zlib() {
    CHECK(teleop_crc32("123456789", 9) == 0xCBF43926u, "CRC-32 check value");
}

int main() {
    struct { const char* name; void (*fn)(); } tests[] = {
        {"fk_matches_mujoco", test_fk_matches_mujoco},
        {"starts_disarmed_and_follows_measured", test_starts_disarmed_and_follows_measured},
        {"arming_requires_deadman_and_fresh_target", test_arming_requires_deadman_and_fresh_target},
        {"ramp_in_then_tracking_and_weight_rate", test_ramp_in_then_tracking_and_weight_rate},
        {"deadman_release_ramps_out", test_deadman_release_ramps_out},
        {"state_stale_faults_and_latches", test_state_stale_faults_and_latches},
        {"nan_and_jump_targets_rejected_then_fault", test_nan_and_jump_targets_rejected_then_fault},
        {"replayed_counter_ignored_and_target_timeouts", test_replayed_counter_ignored_and_target_timeouts},
        {"reacquire_after_gap_accepts_far_pose", test_reacquire_after_gap_accepts_far_pose},
        {"limits_and_velocity_are_enforced", test_limits_and_velocity_are_enforced},
        {"tracking_error_faults", test_tracking_error_faults},
        {"motor_error_temp_estop_stall", test_motor_error_temp_estop_stall},
        {"shutdown_frame_ramps_out", test_shutdown_frame_ramps_out},
        {"gain_ceiling", test_gain_ceiling},
        {"home_pose_is_collision_free", test_home_pose_is_collision_free},
        {"collision_gate_never_enters_collision", test_collision_gate_never_enters_collision},
        {"clap_is_blocked", test_clap_is_blocked},
        {"crc_matches_zlib", test_crc_matches_zlib},
    };
    for (auto& t : tests) {
        const int before = g_fail;
        t.fn();
        std::printf("%s %s\n", g_fail == before ? "[ OK ]" : "[FAIL]", t.name);
    }
    std::printf("%d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
