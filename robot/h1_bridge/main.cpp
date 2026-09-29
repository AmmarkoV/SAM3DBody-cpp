// main.cpp — h1_bridge: tools/gmr_stream.py --sink teleop  ->  Unitree H1-2 arm_sdk.
//
//   h1_bridge <network_interface> [--shm NAME] [--robot NAME] [--domain N] [--rate HZ]
//             [--speed S [--allow-fast]] [--gain G]
//
// All the decisions live in the Supervisor (safety.h); this file is only I/O:
//   rt/lowstate (DDS)  ─► StateIn ─┐
//   /dev/shm/<NAME>    ─► TargetIn ┼─► Supervisor::step ─► Out ─► rt/arm_sdk (DDS)
//   SIGINT / any key   ─► e-stop  ─┘
// Unitree's locomotion controller keeps the robot standing; arm_sdk blends our arm +
// waist command in by the weight carried in motor_cmd[27].q (0 = theirs, 1 = ours).
//
// Remote: hold R1 (deadman) and press X to arm; release R1 to ramp out; B = e-stop;
// after a fault, release R1 and press X to acknowledge.  Ctrl-C / any key = e-stop,
// and the process exits once the weight has ramped back to 0 (a second Ctrl-C kills it
// immediately, leaving the controller to time out arm_sdk on its own).
#include <signal.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include "safety.h"
#include "shm_reader.h"

using namespace h1b;
using unitree_hg::msg::dds_::LowCmd_;
using unitree_hg::msg::dds_::LowState_;

namespace {

// Unitree's CRC (unitree_sdk2/example/h1/low_level/h1_2_ankle_track.cpp), which the
// SDK does not export: over all 32-bit words of the message except the trailing crc.
uint32_t crc32_core(const uint32_t* ptr, uint32_t len) {
    uint32_t crc = 0xFFFFFFFF;
    const uint32_t poly = 0x04c11db7;
    for (uint32_t i = 0; i < len; i++) {
        uint32_t xbit = 1u << 31;
        const uint32_t data = ptr[i];
        for (uint32_t bits = 0; bits < 32; bits++) {
            if (crc & 0x80000000) { crc <<= 1; crc ^= poly; }
            else crc <<= 1;
            if (data & xbit) crc ^= poly;
            xbit >>= 1;
        }
    }
    return crc;
}
template <class Msg>
uint32_t msg_crc(const Msg& m) {
    return crc32_core(reinterpret_cast<const uint32_t*>(&m), (sizeof(Msg) >> 2) - 1);
}

// Latest lowstate, written by the DDS thread, read by the control loop.
struct StateBuf {
    std::mutex m;
    StateIn s;                  // age_s unused here; computed from t_ns by the loop
    int64_t t_ns = 0;           // mono time of the last message that passed its CRC
    uint64_t crc_fail = 0;
    int err_motor = -1;         // first controlled motor reporting an error word
    uint32_t err_word = 0;
    int mode_machine = -1;
};
StateBuf g_state;

void on_lowstate(const void* p) {
    const LowState_& ls = *static_cast<const LowState_*>(p);
    const int64_t now = mono_now_ns();
    if (ls.crc() != msg_crc(ls)) {
        std::lock_guard<std::mutex> lk(g_state.m);
        ++g_state.crc_fail;                           // not taken: the state just ages
        return;
    }
    StateIn s;
    s.valid = true;
    int err_motor = -1;
    uint32_t err_word = 0;
    for (int j = 0; j < kNumJoints; ++j) {
        const auto& ms = ls.motor_state()[kJoints[j].motor];
        s.q[j] = ms.q();
        if (ms.motorstate() != 0 && err_motor < 0) { err_motor = kJoints[j].motor; err_word = ms.motorstate(); }
        s.max_temp_c = std::max<int>(s.max_temp_c, std::max(ms.temperature()[0], ms.temperature()[1]));
    }
    s.motor_error = err_motor >= 0;
    s.buttons = uint16_t(ls.wireless_remote()[2] | (ls.wireless_remote()[3] << 8));
    std::lock_guard<std::mutex> lk(g_state.m);
    g_state.s = s;
    g_state.t_ns = now;
    g_state.err_motor = err_motor;
    g_state.err_word = err_word;
    g_state.mode_machine = ls.mode_machine();
}

std::atomic<bool> g_sigint{false};
void on_sigint(int) {
    g_sigint = true;
    signal(SIGINT, SIG_DFL);                          // a second Ctrl-C kills us outright
}

// Terminal in non-canonical mode so a single key press is an e-stop.
struct RawTerminal {
    termios saved{};
    bool on = false;
    RawTerminal() {
        if (!isatty(STDIN_FILENO) || tcgetattr(STDIN_FILENO, &saved) != 0) return;
        termios t = saved;
        t.c_lflag &= ~(ICANON | ECHO);
        t.c_cc[VMIN] = 0;
        t.c_cc[VTIME] = 0;
        on = tcsetattr(STDIN_FILENO, TCSANOW, &t) == 0;
    }
    ~RawTerminal() { if (on) tcsetattr(STDIN_FILENO, TCSANOW, &saved); }
    bool key() const {
        char c;
        return on && ::read(STDIN_FILENO, &c, 1) == 1;
    }
};

[[noreturn]] void usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s <network_interface> [options]\n"
        "  --shm NAME      teleop shm written by gmr_stream.py --sink teleop (default h1_teleop)\n"
        "  --robot NAME    robot the shm must be for (default unitree_h1_2)\n"
        "  --domain N      DDS domain id (default 0 = real robot; unitree_mujoco uses 1)\n"
        "  --rate HZ       control / publish rate (default 50, as Unitree's arm_sdk example)\n"
        "  --speed S       fraction of the velocity/accel caps (default 0.25)\n"
        "  --allow-fast    required for --speed above 0.25\n"
        "  --gain G        fraction of Unitree's reference kp/kd, <= 1 (default 1)\n",
        argv0);
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argv[1][0] == '-') usage(argv[0]);
    const std::string iface = argv[1];
    std::string shm_name = "h1_teleop", robot = "unitree_h1_2";
    int domain = 0;
    double rate = 50.0;
    bool allow_fast = false;
    Config cfg;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> const char* { if (i + 1 >= argc) usage(argv[0]); return argv[++i]; };
        if (a == "--shm") shm_name = val();
        else if (a == "--robot") robot = val();
        else if (a == "--domain") domain = std::atoi(val());
        else if (a == "--rate") rate = std::atof(val());
        else if (a == "--speed") cfg.speed_scale = std::atof(val());
        else if (a == "--allow-fast") allow_fast = true;
        else if (a == "--gain") cfg.gain_scale = std::atof(val());
        else usage(argv[0]);
    }
    if (cfg.speed_scale > 0.25 && !allow_fast) {
        std::fprintf(stderr, "--speed %.2f > 0.25 needs --allow-fast\n", cfg.speed_scale);
        return 2;
    }
    if (!(rate >= 20.0 && rate <= 500.0)) {       // below 20 Hz every tick trips LoopStall
        std::fprintf(stderr, "--rate must be in [20, 500]\n");
        return 2;
    }
    Supervisor sup(cfg);

    unitree::robot::ChannelFactory::Instance()->Init(domain, iface);
    unitree::robot::ChannelPublisher<LowCmd_> pub("rt/arm_sdk");
    pub.InitChannel();
    unitree::robot::ChannelSubscriber<LowState_> sub("rt/lowstate");
    sub.InitChannel(on_lowstate, 1);

    signal(SIGINT, on_sigint);
    RawTerminal term;
    std::fprintf(stderr,
        "[h1_bridge] iface=%s domain=%d shm=%s robot=%s rate=%.0fHz speed=%.2f gain=%.2f\n"
        "[h1_bridge] remote: hold R1 + press X = arm | release R1 = ramp out | B = E-STOP\n"
        "[h1_bridge] Ctrl-C or any key = E-STOP (exits after the ramp-out)\n",
        iface.c_str(), domain, shm_name.c_str(), robot.c_str(), rate,
        sup.config().speed_scale, sup.config().gain_scale);

    TeleopShmReader reader;
    std::string shm_err, last_shm_err;
    int64_t next_connect_ns = 0;
    uint64_t counter_base = 0, last_counter = 0;  // keeps counters monotonic across writer restarts

    LowCmd_ cmd;
    const int64_t period_ns = int64_t(1e9 / rate);
    timespec deadline;
    clock_gettime(CLOCK_MONOTONIC, &deadline);
    int64_t prev_ns = mono_now_ns() - period_ns;
    int64_t next_status_ns = 0;
    Mode prev_mode = Mode::Disarmed;
    bool estop = false, printed_machine = false;
    uint64_t prev_crc_fail = 0;

    for (;;) {
        const int64_t now = mono_now_ns();
        const double dt = double(now - prev_ns) * 1e-9;
        prev_ns = now;

        // ── inputs ────────────────────────────────────────────────────────────
        StateIn st;
        int err_motor, mode_machine;
        uint32_t err_word;
        uint64_t crc_fail;
        {
            std::lock_guard<std::mutex> lk(g_state.m);
            st = g_state.s;
            st.age_s = g_state.t_ns ? double(now - g_state.t_ns) * 1e-9 : 1e9;
            err_motor = g_state.err_motor;
            err_word = g_state.err_word;
            mode_machine = g_state.mode_machine;
            crc_fail = g_state.crc_fail;
        }
        if (!printed_machine && mode_machine >= 0) {
            std::fprintf(stderr, "\n[h1_bridge] lowstate received (mode_machine=%d)\n", mode_machine);
            printed_machine = true;
        }

        // Read BEFORE the stale check: an exiting writer publishes F_SHUTDOWN and then
        // unlinks, and that last frame is only reachable through the old mapping.
        TargetIn tgt;
        if (reader.read(now, &tgt)) {
            tgt.counter += counter_base;
            last_counter = std::max(last_counter, tgt.counter);
        }
        if ((!reader.connected() || reader.stale()) && now >= next_connect_ns) {
            if (reader.connect(shm_name, robot, &shm_err)) {
                counter_base = last_counter;          // a restarted writer counts from 1 again
                std::fprintf(stderr, "\n[h1_bridge] connected to /dev/shm/%s\n", shm_name.c_str());
                last_shm_err.clear();
            } else {
                if (shm_err != last_shm_err) std::fprintf(stderr, "\n[h1_bridge] %s\n", shm_err.c_str());
                last_shm_err = shm_err;
                next_connect_ns = now + 500000000LL;
            }
        }

        if (g_sigint || term.key()) estop = true;

        // ── decide ────────────────────────────────────────────────────────────
        const Out o = sup.step(dt, st, tgt, estop);

        // ── output ────────────────────────────────────────────────────────────
        cmd.motor_cmd()[kArmSdkWeightIndex].q(o.weight);
        for (int j = 0; j < kNumJoints; ++j) {
            auto& mc = cmd.motor_cmd()[kJoints[j].motor];
            mc.q(float(o.q[j]));
            mc.dq(0.f);
            mc.kp(o.kp[j]);
            mc.kd(o.kd[j]);
            mc.tau(0.f);
        }
        cmd.crc(msg_crc(cmd));
        pub.Write(cmd);

        // ── operator feedback ─────────────────────────────────────────────────
        if (o.mode != prev_mode) {
            std::fprintf(stderr, "\n[h1_bridge] %s -> %s: %s", mode_name(prev_mode), mode_name(o.mode), o.note);
            if (o.mode == Mode::Fault) {
                std::fprintf(stderr, " [%s]", fault_name(o.fault));
                if (o.fault == FaultCode::MotorError && err_motor >= 0)
                    std::fprintf(stderr, " motor %d motorstate=0x%x", err_motor, err_word);
            }
            std::fprintf(stderr, "\n");
            prev_mode = o.mode;
        }
        if (crc_fail != prev_crc_fail) {
            std::fprintf(stderr, "\n[h1_bridge] lowstate CRC failures: %llu\n", (unsigned long long)crc_fail);
            prev_crc_fail = crc_fail;
        }
        if (now >= next_status_ns) {
            std::fprintf(stderr, "\r[%-8s] w=%.2f target_age=%5.2fs state_age=%5.3fs rej=%d%s  %-28s",
                         mode_name(o.mode), o.weight, std::min(o.target_age_s, 99.99),
                         std::min(st.age_s, 9.999), o.rejected_targets,
                         o.collision_blocked ? " COLLISION-BLOCK" : "", o.note);
            next_status_ns = now + 200000000LL;
        }

        if (estop && o.weight <= 0.f) break;          // control handed back: safe to leave

        // ── fixed-rate sleep (absolute deadlines; resync after an overrun) ──────
        deadline.tv_nsec += period_ns;
        while (deadline.tv_nsec >= 1000000000L) { deadline.tv_nsec -= 1000000000L; ++deadline.tv_sec; }
        const int64_t dl_ns = int64_t(deadline.tv_sec) * 1000000000LL + deadline.tv_nsec;
        if (dl_ns < mono_now_ns()) clock_gettime(CLOCK_MONOTONIC, &deadline);
        else clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &deadline, nullptr);
    }

    // A few more weight-0 frames so the controller sees the hand-back even if one is lost.
    cmd.motor_cmd()[kArmSdkWeightIndex].q(0.f);
    cmd.crc(msg_crc(cmd));
    for (int i = 0; i < 10; ++i) {
        pub.Write(cmd);
        usleep(useconds_t(period_ns / 1000));
    }
    std::fprintf(stderr, "\n[h1_bridge] weight 0, exiting\n");
    return 0;
}
