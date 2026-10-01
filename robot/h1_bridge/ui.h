// ui.h — operator UI for h1_bridge (--ui / --manual), OpenCV highgui.
//
// With --manual the slider window also has a FOLLOW CAMERA / BACK TO MANUAL toggle that
// switches the target source at runtime (vetted by check_takeover() on the control
// thread); while the camera drives, the sliders mirror its pose read-only.
//
// The UI runs on the MAIN thread (OpenCV's Qt backend requires it); the control loop
// runs on its own thread, so a stalled window can never stall control.  They meet
// only in UiShared.  The UI can request an e-stop and propose manual poses; it can
// NEVER arm (that stays the remote's deadman) and never writes a command itself:
// every manual pose is re-checked by the control thread (ManualSource::apply).
#pragma once
#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>

#include "manual.h"

namespace h1b {

struct UiShared {
    std::mutex m;
    // control -> UI (snapshot every tick)
    bool have_state = false;
    StateIn st;
    Out out;
    int temp_c[kNumJoints] = {};
    uint32_t motorstate[kNumJoints] = {};
    int mode_machine = -1;
    uint64_t crc_fail = 0;
    bool manual = false;                 // --manual: the slider window exists
    bool source_manual = false;          // current target source: sliders (true) or camera shm
    bool manual_ready = false;
    bool camera_fresh = false;           // a fresh TRACKING frame from the shm this tick...
    JointVec camera_q{};                 // ...clamped into the enforced range
    JointVec published{}, goal{};
    std::deque<std::string> log;         // warnings and events, oldest first
    std::string apply_result;            // verdict of the last Apply / source switch, from the control thread
    bool apply_ok = false;
    uint64_t apply_seq = 0;
    // UI -> control
    bool apply_request = false;
    JointVec apply_pose{};
    bool switch_request = false;
    bool switch_to_manual = false;
    std::atomic<bool> estop{false};      // latched: the control loop ramps out and exits
    std::atomic<bool> done{false};       // set by the control loop when it has exited
};

// Thread-safe; no-op on nullptr.  Keeps the last 200 lines.
inline void ui_log(UiShared* ui, const std::string& line) {
    if (!ui) return;
    std::lock_guard<std::mutex> lk(ui->m);
    ui->log.push_back(line);
    while (ui->log.size() > 200) ui->log.pop_front();
}

// Blocks until ui.done.  Closing a window or pressing space/'e' in it = e-stop.
void run_ui(UiShared& ui, const Supervisor& sup);

}  // namespace h1b
