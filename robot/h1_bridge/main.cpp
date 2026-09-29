// main.cpp — h1_bridge: tools/gmr_stream.py --sink teleop  ->  Unitree H1-2 arm_sdk.
//
//   h1_bridge <network_interface> [--shm NAME] [--robot NAME] [--domain N] [--rate HZ]
//             [--speed S [--allow-fast]] [--gain G] [--ui] [--manual]
//
//   --ui      OpenCV monitor window (joint states, warnings, E-STOP button)
//   --manual  sliders instead of the camera stream (implies --ui); see manual.h
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
// immediately; what the robot then does with the last weight is UNVERIFIED).
//
// Operator traps (Develop mode, colliding remote combos, power-on joint zeros, the two
// different e-stops, ...) are printed as a checklist that must be acknowledged before
// anything is published, and re-warned live when they are detected (warn_*() below).
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <signal.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include "manual.h"
#include "safety.h"
#include "shm_reader.h"
#include "ui.h"

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
    float stick_max = 0.f;      // largest |lx|,|rx|,|ry|,|ly| on the remote (they move the base)
    int temp_c[kNumJoints] = {};
    uint32_t motorstate[kNumJoints] = {};
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
    int temp_c[kNumJoints];
    uint32_t mstate[kNumJoints];
    for (int j = 0; j < kNumJoints; ++j) {
        const auto& ms = ls.motor_state()[kJoints[j].motor];
        temp_c[j] = std::max(ms.temperature()[0], ms.temperature()[1]);
        mstate[j] = ms.motorstate();
        s.q[j] = ms.q();
        if (ms.motorstate() != 0 && err_motor < 0) { err_motor = kJoints[j].motor; err_word = ms.motorstate(); }
        s.max_temp_c = std::max<int>(s.max_temp_c, std::max(ms.temperature()[0], ms.temperature()[1]));
    }
    s.motor_error = err_motor >= 0;
    s.buttons = uint16_t(ls.wireless_remote()[2] | (ls.wireless_remote()[3] << 8));
    // Sticks: floats lx, rx, ry, L2(analog), ly at bytes 4..23 (SDK xRockerBtnDataStruct).
    float axes[5];
    std::memcpy(axes, ls.wireless_remote().data() + 4, sizeof(axes));
    float stick_max = 0.f;
    for (int i : {0, 1, 2, 4})
        if (std::isfinite(axes[i])) stick_max = std::max(stick_max, std::fabs(axes[i]));
    std::lock_guard<std::mutex> lk(g_state.m);
    g_state.stick_max = stick_max;
    std::copy(temp_c, temp_c + kNumJoints, g_state.temp_c);
    std::copy(mstate, mstate + kNumJoints, g_state.motorstate);
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

// ── operator warnings ─────────────────────────────────────────────────────────
// Sources: the H1-2 User Manual V1.0 (unitreeh1-2.pdf), Unitree's xr_teleoperate
// wiki ("Motion"), and unitree_sdk2's examples.  Anything we could not confirm for
// H1-2 firmware is labelled UNVERIFIED rather than stated as fact.
void warn(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
UiShared* g_ui = nullptr;           // set with --ui: warnings and events also go to the window

void warn(const char* fmt, ...) {
    char msg[1024];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    const bool tty = isatty(STDERR_FILENO);
    std::fprintf(stderr, "\n%s[h1_bridge] WARNING: %s%s\n", tty ? "\033[1;33m" : "", msg, tty ? "\033[0m" : "");
    ui_log(g_ui, std::string("WARNING: ") + msg);
}

// An event line for both the terminal and the UI log.
void event(const std::string& line) {
    std::fprintf(stderr, "\n[h1_bridge] %s\n", line.c_str());
    ui_log(g_ui, line);
}

void print_checklist(const Supervisor& sup, double rate) {
    const Config& c = sup.config();
    std::fprintf(stderr,
"\n"
"=========================== H1-2 ARM_SDK PRE-FLIGHT CHECKLIST ===========================\n"
" 1. ROBOT MODE.  The robot must be standing under Unitree's OWN motion controller\n"
"    (manual: L2+UP = ready, then R2+X = motion control).  arm_sdk only BLENDS our arm +\n"
"    waist command into that controller; it keeps balancing the legs.\n"
"    >>> DO NOT enter Develop / debug mode (L2+R2). <<<  It stops the built-in controller:\n"
"    nothing balances the robot and arm_sdk has nothing to blend into.  This bridge never\n"
"    publishes rt/lowcmd (the Develop-mode topic).\n"
" 2. GANTRY FIRST.  Run first sessions with the robot on the suspension frame.  Keep 2 m\n"
"    clear; arms may move at up to %.2f rad/s (--speed %.2f).\n"
" 3. JOINT ZEROS.  Arm and ankle zeros are latched from the POWER-ON pose (manual: \"Body\n"
"    placement\", arms at the limit position).  Powered on in any other pose, every angle is\n"
"    offset and this bridge's joint limits and self-collision model are WRONG.  If in doubt,\n"
"    power-cycle in the correct pose.  (Checked below once lowstate arrives: a joint outside\n"
"    its URDF range is reported.)\n"
" 4. TWO DIFFERENT E-STOPS.\n"
"      bridge  B / Ctrl-C / any key here  -> arms ramp back to Unitree control, robot STANDS.\n"
"      (--ui: also the E-STOP button, space/e in a window, or closing a window.)\n"
"      Unitree L2+B (manual e-stop)       -> DAMPING: whole robot goes limp and FALLS.\n"
"    Use B while standing free; L2+B only when hanging on the gantry.\n"
" 5. REMOTE COMBOS SHARE BUTTONS.  Bridge: hold R1 = deadman, press X = arm, B = e-stop.\n"
"    Unitree: L2+B damping, L2+R2 Develop mode, L2+A position mode, R2+X (re)start motion\n"
"    control, L2+UP ready, START stand<->walk.  On G1 firmware R1+X enters the waist motion\n"
"    program; what R1+X does on H1-2 firmware is UNVERIFIED.  Keep fingers OFF L2/R2 while\n"
"    arming (R2+X is one finger away from R1+X).  Detected combos are warned live.\n"
" 6. STICKS MOVE THE ROBOT.  While standing/walking the joysticks drive the base; do not\n"
"    touch them while the arms are armed (warned live).\n"
" 7. LINK LOSS IS UNVERIFIED.  What the controller does if rt/arm_sdk stops while the\n"
"    weight is > 0 (cable pulled, bridge killed by a 2nd Ctrl-C, crash) is NOT documented.\n"
"    Test it on the gantry before relying on it.  Use the wired 192.168.123.x link, not Wi-Fi.\n"
" 8. ONE PUBLISHER.  Nothing else may publish rt/arm_sdk or rt/lowcmd at the same time\n"
"    (xr_teleoperate, Unitree examples, a second h1_bridge).\n"
" 9. ALSO UNVERIFIED on this firmware:\n"
"      - motorstate != 0 is treated as a motor error (latches FAULT, even while disarmed);\n"
"      - mode_machine is printed but not checked;\n"
"      - the hand length in the collision model (%.2f m) is a guess; hands are NOT driven.\n"
" 10. --manual: sliders only edit a PENDING pose; nothing moves before APPLY, APPLY is\n"
"    refused unless the pose AND the straight path to it are inside the limits and\n"
"    collision-free, and the arms still move only while armed from the remote.  The\n"
"    collision model is simplified boxes/spheres: it does not know about the hands' fingers,\n"
"    tools, cables or anything near the robot.\n"
"    Run --ui on a LOCAL display: if the display connection dies (ssh -X drop, logout, xkill)\n"
"    the process is killed at once, with no ramp-out (see item 7).\n"
" 11. Rate %.0f Hz (Unitree's xr_teleoperate uses 250 Hz, their arm_sdk example 50 Hz).\n"
"==========================================================================================\n",
        c.vmax * c.speed_scale, c.speed_scale, c.collision.hand_len, rate);
}

// The DDS link to the robot is the 192.168.123.0/24 network (dev PC .162, optional .163).
void warn_network(const std::string& iface, int domain) {
    ifaddrs* ifs = nullptr;
    if (getifaddrs(&ifs) != 0) return;
    bool found = false, robot_net = false;
    char ip[INET_ADDRSTRLEN] = "?";
    for (ifaddrs* i = ifs; i; i = i->ifa_next) {
        if (!i->ifa_addr || i->ifa_addr->sa_family != AF_INET || iface != i->ifa_name) continue;
        const auto* sin = reinterpret_cast<const sockaddr_in*>(i->ifa_addr);
        inet_ntop(AF_INET, &sin->sin_addr, ip, sizeof(ip));
        found = true;
        robot_net |= (ntohl(sin->sin_addr.s_addr) >> 8) == 0xC0A87B;   // 192.168.123.x
    }
    freeifaddrs(ifs);
    if (!found)
        warn("interface '%s' has no IPv4 address: DDS will see nothing.", iface.c_str());
    else if (domain == 0 && !robot_net)
        warn("interface '%s' is %s, not on the robot network 192.168.123.0/24 "
             "(domain 0 = real robot).  Wrong interface?", iface.c_str(), ip);
}

// One-off sanity checks on the first valid lowstate.
void warn_first_state(const StateIn& s, int err_motor, uint32_t err_word) {
    for (int j = 0; j < kNumJoints; ++j) {
        const JointSpec& js = kJoints[j];
        if (s.q[j] < js.lo - 0.05 || s.q[j] > js.hi + 0.05)
            warn("%s reads %.3f rad, OUTSIDE its URDF range [%.2f, %.2f].  Likely a wrong "
                 "power-on zero (checklist item 3): limits and collision checks are invalid.  "
                 "Do not arm; power-cycle the robot in the manual's placement pose.",
                 js.name, s.q[j], js.lo, js.hi);
    }
    if (s.motor_error)
        warn("motor %d reports motorstate=0x%x at startup.  The bridge treats nonzero as an "
             "error and latches FAULT; if this is a normal status bit on this firmware "
             "(UNVERIFIED), it will never arm.", err_motor, err_word);
    if (s.max_temp_c >= 70)
        warn("hottest controlled motor is already at %d C (fault at %d C).", s.max_temp_c,
             Config{}.temp_fault_c);
}

// Unitree remote combos that share buttons with the bridge (manual + xr_teleoperate).
struct Combo { uint16_t keys; const char* what; };
constexpr Combo kUnitreeCombos[] = {
    {kL2 | kR2, "L2+R2 = Unitree DEVELOP mode: the built-in controller STOPS and nothing balances "
                "the robot.  Never use Develop mode with this bridge."},
    {kL2 | kB, "L2+B = Unitree DAMPING: the whole robot goes limp and falls unless it hangs on the "
               "gantry."},
    {kL2 | kA, "L2+A = Unitree position mode (Develop-mode diagnostic posture)."},
    {kR2 | kX, "R2+X = Unitree (re)starts its motion-control program."},
    {kL2 | kUp, "L2+UP = Unitree ready / locked standing."},
    {kStart, "START = Unitree toggles standing <-> walking."},
};

void warn_combos(uint16_t buttons, uint16_t prev) {
    for (const Combo& c : kUnitreeCombos)
        if ((buttons & c.keys) == c.keys && (prev & c.keys) != c.keys)
            warn("remote: %s", c.what);
}

[[noreturn]] void usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s <network_interface> [options]\n"
        "  --shm NAME      teleop shm written by gmr_stream.py --sink teleop (default h1_teleop)\n"
        "  --robot NAME    robot the shm must be for (default unitree_h1_2)\n"
        "  --domain N      DDS domain id (default 0 = real robot; unitree_mujoco uses 1)\n"
        "  --rate HZ       control / publish rate (default 50, as Unitree's arm_sdk example)\n"
        "  --speed S       fraction of the velocity/accel caps (default 0.25)\n"
        "  --allow-fast    required for --speed above 0.25\n"
        "  --gain G        fraction of Unitree's reference kp/kd, <= 1 (default 1)\n"
        "  --ui            OpenCV monitor window (joint states, warnings, E-STOP)\n"
        "  --manual        per-joint sliders instead of the camera stream (implies --ui)\n",
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
    bool allow_fast = false, use_ui = false, manual = false;
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
        else if (a == "--ui") use_ui = true;
        else if (a == "--manual") use_ui = manual = true;
        else usage(argv[0]);
    }
#ifndef H1B_HAVE_UI
    if (use_ui) {
        std::fprintf(stderr, "--ui / --manual need OpenCV: this h1_bridge was built without it\n");
        return 2;
    }
#endif
    if (cfg.speed_scale > 0.25 && !allow_fast) {
        std::fprintf(stderr, "--speed %.2f > 0.25 needs --allow-fast\n", cfg.speed_scale);
        return 2;
    }
    if (!(rate >= 20.0 && rate <= 500.0)) {       // below 20 Hz every tick trips LoopStall
        std::fprintf(stderr, "--rate must be in [20, 500]\n");
        return 2;
    }
    Supervisor sup(cfg);
    UiShared ui_shared;
    ui_shared.manual = manual;
    if (use_ui) g_ui = &ui_shared;

    // Nothing is published until the operator has read the checklist.
    print_checklist(sup, rate);
    if (isatty(STDIN_FILENO)) {
        std::fprintf(stderr, "Type 'yes' + ENTER to confirm you have checked all items: ");
        char line[64] = {};
        if (!std::fgets(line, sizeof(line), stdin) || std::strncmp(line, "yes", 3) != 0) {
            std::fprintf(stderr, "[h1_bridge] not confirmed, exiting (nothing was published)\n");
            return 1;
        }
    } else {
        warn("stdin is not a terminal: checklist NOT confirmed, and the any-key e-stop is "
             "unavailable (Ctrl-C and the remote's B still work).");
    }
    warn_network(iface, domain);

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
        "[h1_bridge] Ctrl-C or any key = E-STOP (exits after the ramp-out)\n"
        "[h1_bridge] target source: %s\n",
        iface.c_str(), domain, shm_name.c_str(), robot.c_str(), rate,
        sup.config().speed_scale, sup.config().gain_scale,
        manual ? "MANUAL sliders (the shm is ignored)" : "camera stream (shm)");

    // The control loop runs on its own thread when the UI (which needs the main thread) is up.
    auto control = [&]() {
        ManualSource manual_src(sup);

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
        const int64_t start_ns = mono_now_ns();
        int64_t next_nostate_warn_ns = start_ns + 3000000000LL, next_stick_warn_ns = 0;
        bool state_lost_warned = false;
        uint16_t prev_buttons = 0;

        for (;;) {
            const int64_t now = mono_now_ns();
            const double dt = double(now - prev_ns) * 1e-9;
            prev_ns = now;

            // ── inputs ────────────────────────────────────────────────────────────
            StateIn st;
            int err_motor, mode_machine;
            uint32_t err_word;
            uint64_t crc_fail;
            float stick_max;
            int temp_c[kNumJoints];
            uint32_t mstate[kNumJoints];
            {
                std::lock_guard<std::mutex> lk(g_state.m);
                st = g_state.s;
                st.age_s = g_state.t_ns ? double(now - g_state.t_ns) * 1e-9 : 1e9;
                err_motor = g_state.err_motor;
                err_word = g_state.err_word;
                mode_machine = g_state.mode_machine;
                crc_fail = g_state.crc_fail;
                stick_max = g_state.stick_max;
                std::copy(g_state.temp_c, g_state.temp_c + kNumJoints, temp_c);
                std::copy(g_state.motorstate, g_state.motorstate + kNumJoints, mstate);
            }
            if (!printed_machine && mode_machine >= 0) {
                event("lowstate received (mode_machine=" + std::to_string(mode_machine) +
                      "; its meaning for H1-2 is UNVERIFIED, not checked)");
                printed_machine = true;
                warn_first_state(st, err_motor, err_word);
            }
            if (!printed_machine && now >= next_nostate_warn_ns) {
                warn("no rt/lowstate after %.0f s: robot off, wrong interface, wrong --domain, or "
                     "another DDS network.  Nothing can be armed until it arrives.",
                     double(now - start_ns) * 1e-9);
                next_nostate_warn_ns = now + 10000000000LL;
            }
            if (printed_machine && st.age_s > 0.5 && !state_lost_warned) {
                warn("rt/lowstate stopped (%.1f s old): link lost or robot powered down.  If this "
                     "happens while armed, what the robot does with the last arm_sdk weight is "
                     "UNVERIFIED.", st.age_s);
                state_lost_warned = true;
            } else if (st.age_s <= 0.5) {
                state_lost_warned = false;
            }
            if (st.valid && st.age_s <= 0.5) {
                warn_combos(st.buttons, prev_buttons);
                prev_buttons = st.buttons;
            }

            // Read BEFORE the stale check: an exiting writer publishes F_SHUTDOWN and then
            // unlinks, and that last frame is only reachable through the old mapping.
            TargetIn tgt;
            if (manual) {
                if (!manual_src.ready() && st.valid && st.age_s < 0.5) {
                    JointVec q;
                    for (int j = 0; j < kNumJoints; ++j) q[j] = st.q[j];
                    manual_src.init(q);
                    event("manual: sliders start at the measured pose");
                }
                bool req = false;
                JointVec pose{};
                {
                    std::lock_guard<std::mutex> lk(ui_shared.m);
                    std::swap(req, ui_shared.apply_request);
                    pose = ui_shared.apply_pose;
                }
                if (req) {
                    const PoseCheck c = manual_src.apply(pose);   // authoritative re-check
                    {
                        std::lock_guard<std::mutex> lk(ui_shared.m);
                        ui_shared.apply_result = c.verdict;
                        ui_shared.apply_ok = c.ok;
                        ++ui_shared.apply_seq;
                    }
                    event(std::string("apply: ") + c.verdict);
                }
                tgt = manual_src.tick(dt);
            } else if (reader.read(now, &tgt)) {
                tgt.counter += counter_base;
                last_counter = std::max(last_counter, tgt.counter);
            }
            if (!manual && (!reader.connected() || reader.stale()) && now >= next_connect_ns) {
                if (reader.connect(shm_name, robot, &shm_err)) {
                    counter_base = last_counter;          // a restarted writer counts from 1 again
                    event("connected to /dev/shm/" + shm_name);
                    last_shm_err.clear();
                } else {
                    if (shm_err != last_shm_err) event(shm_err);
                    last_shm_err = shm_err;
                    next_connect_ns = now + 500000000LL;
                }
            }

            if (g_sigint || term.key() || ui_shared.estop) estop = true;

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
            if (o.weight > 0.f && stick_max > 0.2f && now >= next_stick_warn_ns) {
                warn("joystick deflected (%.2f) while the arms are driven: the sticks move the robot's "
                     "BASE under Unitree's controller.  Hands off the sticks.", stick_max);
                next_stick_warn_ns = now + 2000000000LL;
            }
            if (o.mode == Mode::RampIn && prev_mode == Mode::Disarmed)
                warn("ARMING: the arms are now blended to the target pose.  Keep R1 held, fingers "
                     "off L2/R2 and the sticks.  Stop = release R1 or press B (robot keeps standing).");
            if (o.mode != prev_mode) {
                std::string line = std::string(mode_name(prev_mode)) + " -> " + mode_name(o.mode) + ": " + o.note;
                if (o.mode == Mode::Fault) {
                    line += std::string(" [") + fault_name(o.fault) + "]";
                    if (o.fault == FaultCode::MotorError && err_motor >= 0) {
                        char b[64];
                        std::snprintf(b, sizeof(b), " motor %d motorstate=0x%x", err_motor, err_word);
                        line += b;
                    }
                }
                event(line);
                prev_mode = o.mode;
            }
            if (g_ui) {
                std::lock_guard<std::mutex> lk(ui_shared.m);
                ui_shared.have_state = st.valid;
                ui_shared.st = st;
                ui_shared.out = o;
                std::copy(temp_c, temp_c + kNumJoints, ui_shared.temp_c);
                std::copy(mstate, mstate + kNumJoints, ui_shared.motorstate);
                ui_shared.mode_machine = mode_machine;
                ui_shared.crc_fail = crc_fail;
                ui_shared.manual_ready = manual_src.ready();
                ui_shared.published = manual_src.published();
                ui_shared.goal = manual_src.goal();
            }
            if (crc_fail != prev_crc_fail) {
                event("lowstate CRC failures: " + std::to_string(crc_fail));
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
        event("weight 0, exiting");
        ui_shared.done = true;
    };

#ifdef H1B_HAVE_UI
    if (use_ui) {
        std::thread loop(control);
        run_ui(ui_shared, sup);          // returns once the control loop is done
        loop.join();
        return 0;
    }
#endif
    control();
    return 0;
}
