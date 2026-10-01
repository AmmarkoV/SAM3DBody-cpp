// ui.cpp — see ui.h.  Two windows:
//   monitor: mode / fault / weight, per-joint measured vs commanded against the limits,
//            temperatures and motor status words, the warning log, a big E-STOP;
//   manual (--manual only): one trackbar per joint editing a PENDING pose, a front and a
//            side view of current / applied / pending, the safety verdict, and
//            APPLY / HOLD / HOME / RESET / E-STOP buttons, and FOLLOW CAMERA / BACK TO
//            MANUAL (while the camera drives, the sliders mirror its pose, read-only).
#include "ui.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <vector>

namespace h1b {

namespace {

const char* kMon = "h1_bridge: monitor";
const char* kMan = "h1_bridge: manual joints";

const char* kShort[kNumJoints] = {
    "L shoulder pitch", "L shoulder roll", "L shoulder yaw", "L elbow",
    "L wrist roll", "L wrist pitch", "L wrist yaw",
    "R shoulder pitch", "R shoulder roll", "R shoulder yaw", "R elbow",
    "R wrist roll", "R wrist pitch", "R wrist yaw",
    "waist yaw"};

const cv::Scalar kWhite(235, 235, 235), kGrey(120, 120, 120), kDark(45, 45, 45), kBg(28, 28, 28),
    kGreen(80, 200, 80), kRed(60, 60, 230), kYellow(60, 220, 240), kCyan(230, 210, 60), kOrange(0, 150, 255),
    kMagenta(220, 80, 220);

std::string fmt(const char* f, ...) {
    char buf[256];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return buf;
}

void text(cv::Mat& img, const std::string& s, cv::Point p, double scale = 0.45,
          const cv::Scalar& c = kWhite, int th = 1) {
    cv::putText(img, s, p, cv::FONT_HERSHEY_SIMPLEX, scale, c, th, cv::LINE_AA);
}

enum ButtonId { kNone = 0, kBtnEstop, kBtnApply, kBtnHold, kBtnHome, kBtnReset, kBtnSource };
struct Button { cv::Rect r; const char* label; cv::Scalar color; ButtonId id; bool enabled; };

struct Clicks { std::vector<Button> buttons; ButtonId clicked = kNone; };

void on_mouse(int ev, int x, int y, int, void* p) {
    if (ev != cv::EVENT_LBUTTONDOWN) return;
    auto* c = static_cast<Clicks*>(p);
    for (const Button& b : c->buttons)
        if (b.enabled && b.r.contains({x, y})) c->clicked = b.id;
}

void draw_button(cv::Mat& img, const Button& b, double scale = 0.6) {
    const cv::Scalar fill = b.enabled ? b.color : kDark;
    cv::rectangle(img, b.r, fill, cv::FILLED);
    cv::rectangle(img, b.r, kWhite, 1);
    int base = 0;
    const cv::Size ts = cv::getTextSize(b.label, cv::FONT_HERSHEY_SIMPLEX, scale, 2, &base);
    text(img, b.label, {b.r.x + (b.r.width - ts.width) / 2, b.r.y + (b.r.height + ts.height) / 2},
         scale, b.enabled ? kWhite : kGrey, 2);
}

cv::Scalar mode_color(Mode m) {
    switch (m) {
        case Mode::Tracking: return {40, 150, 40};
        case Mode::RampIn:
        case Mode::RampOut: return {0, 120, 220};
        case Mode::Fault: return {40, 40, 200};
        default: return {80, 80, 80};
    }
}

// Word-wrap to `width` characters.
std::vector<std::string> wrap(const std::string& s, size_t width) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t n = std::min(width, s.size() - i);
        if (i + n < s.size()) {
            const size_t sp = s.rfind(' ', i + n);
            if (sp != std::string::npos && sp > i) n = sp - i;
        }
        out.push_back(s.substr(i, n));
        i += n;
        while (i < s.size() && s[i] == ' ') ++i;
    }
    return out;
}

// ── monitor window ──────────────────────────────────────────────────────────
cv::Mat render_monitor(UiShared& ui, const Supervisor& sup, Clicks& clicks) {
    const int W = 980, H = 800;
    cv::Mat img(H, W, CV_8UC3, kBg);
    StateIn st;
    Out o;
    int temp[kNumJoints];
    uint32_t ms[kNumJoints];
    bool have;
    int mm;
    uint64_t crc;
    std::vector<std::string> log;
    bool manual;
    {
        std::lock_guard<std::mutex> lk(ui.m);
        st = ui.st; o = ui.out; have = ui.have_state; mm = ui.mode_machine; crc = ui.crc_fail;
        manual = ui.source_manual;
        std::copy(ui.temp_c, ui.temp_c + kNumJoints, temp);
        std::copy(ui.motorstate, ui.motorstate + kNumJoints, ms);
        log.assign(ui.log.begin(), ui.log.end());
    }
    const bool estop = ui.estop;

    // header
    cv::rectangle(img, {0, 0, W - 240, 100}, mode_color(o.mode), cv::FILLED);
    text(img, mode_name(o.mode), {14, 42}, 1.2, kWhite, 3);
    if (o.mode == Mode::Fault) text(img, fault_name(o.fault), {260, 40}, 0.7, kWhite, 2);
    text(img, o.note, {14, 78}, 0.6, kWhite, 1);
    // weight bar
    cv::rectangle(img, {14, 112, 300, 18}, kDark, cv::FILLED);
    cv::rectangle(img, {14, 112, int(300 * o.weight), 18}, kCyan, cv::FILLED);
    text(img, fmt("arm_sdk weight %.2f", o.weight), {324, 126});
    text(img, fmt("state age %s   target age %.2f s   CRC fails %llu   mode_machine %d   source %s",
                  have ? fmt("%.3f s", st.age_s).c_str() : "NONE", std::min(o.target_age_s, 99.99),
                  (unsigned long long)crc, mm, manual ? "MANUAL sliders" : "camera (shm)"),
         {14, 150}, 0.45, have && st.age_s < 0.05 ? kWhite : kYellow);
    if (o.collision_blocked) text(img, "COLLISION GATE BLOCKING", {520, 126}, 0.5, kOrange, 2);

    // E-STOP
    clicks.buttons.clear();
    Button es{{W - 230, 8, 220, 100}, estop ? "E-STOP SENT" : "E-STOP", {30, 30, 210}, kBtnEstop, !estop};
    clicks.buttons.push_back(es);
    draw_button(img, es, 0.9);
    text(img, "space / e / close window", {W - 222, 128}, 0.42, kGrey);

    // joint table
    const int y0 = 180, rh = 24, bx = 330, bw = 390;
    text(img, "joint", {14, y0 - 6}, 0.45, kGrey);
    text(img, "meas", {170, y0 - 6}, 0.45, kGrey);
    text(img, "cmd", {245, y0 - 6}, 0.45, kGrey);
    text(img, "URDF range, dark = enforced; white = meas, cyan = cmd", {bx, y0 - 6}, 0.4, kGrey);
    text(img, "temp", {740, y0 - 6}, 0.45, kGrey);
    text(img, "motorstate", {810, y0 - 6}, 0.45, kGrey);
    for (int j = 0; j < kNumJoints; ++j) {
        const int y = y0 + 8 + j * rh + (j >= kR0 ? 6 : 0) + (j >= kWaist ? 6 : 0);
        const JointSpec& js = kJoints[j];
        auto X = [&](double q) { return bx + int(bw * (q - js.lo) / (js.hi - js.lo)); };
        text(img, kShort[j], {14, y + 15});
        text(img, have ? fmt("%+.3f", st.q[j]) : "-", {165, y + 15});
        text(img, fmt("%+.3f", o.q[j]), {240, y + 15}, 0.45, kCyan);
        cv::rectangle(img, {bx, y + 3, bw, 16}, {70, 70, 70}, cv::FILLED);
        cv::rectangle(img, {X(sup.lo(j)), y + 3, X(sup.hi(j)) - X(sup.lo(j)), 16}, {50, 50, 50}, cv::FILLED);
        if (js.lo < 0 && js.hi > 0) cv::line(img, {X(0), y + 3}, {X(0), y + 18}, kGrey);
        if (have) {
            const bool out = st.q[j] < js.lo || st.q[j] > js.hi;
            const int x = std::clamp(X(st.q[j]), bx, bx + bw);
            cv::line(img, {x, y + 1}, {x, y + 20}, out ? kRed : kWhite, 2);
        }
        const int xc = X(o.q[j]);
        cv::fillConvexPoly(img, std::vector<cv::Point>{{xc, y + 11}, {xc - 5, y + 21}, {xc + 5, y + 21}}, kCyan);
        const cv::Scalar tc = temp[j] >= sup.config().temp_fault_c ? kRed : temp[j] >= 70 ? kOrange : kWhite;
        text(img, have ? fmt("%d C", temp[j]) : "-", {740, y + 15}, 0.45, tc);
        text(img, have ? fmt("0x%x", ms[j]) : "-", {810, y + 15}, 0.45, ms[j] ? kRed : kWhite);
    }

    // log (newest at the bottom)
    const int ly0 = 580;
    cv::rectangle(img, {8, ly0 - 20, W - 16, H - ly0 - 12}, kDark, 1);
    text(img, "warnings / events", {14, ly0 - 4}, 0.45, kGrey);
    std::vector<std::pair<std::string, cv::Scalar>> lines;
    for (const std::string& l : log) {
        const cv::Scalar c = l.rfind("WARNING", 0) == 0 ? kYellow : kWhite;
        for (const std::string& w : wrap(l, 128)) lines.push_back({w, c});
    }
    const int maxl = (H - ly0 - 28) / 17;
    const int first = std::max(0, int(lines.size()) - maxl);
    for (int i = first; i < int(lines.size()); ++i)
        text(img, lines[i].first, {14, ly0 + 16 + (i - first) * 17}, 0.4, lines[i].second);
    return img;
}

// ── manual window ───────────────────────────────────────────────────────────
struct Skel { Vec3 neck, head, sh[2], p[2][kNumArmPoints]; double R[3][3]; };

Skel skeleton(const JointVec& q, const CollisionGeometry& g) {
    const ArmPoints a = forward_kinematics(q, g.hand_len);
    Skel s;
    auto rot = [&](double x, double y, double z) {
        Vec3 o;
        for (int i = 0; i < 3; ++i) o[i] = a.torso_R[i][0] * x + a.torso_R[i][1] * y + a.torso_R[i][2] * z;
        return o;
    };
    s.neck = rot(0, 0, 0.45);
    s.head = rot(0, 0, 0.62);
    for (int side = 0; side < 2; ++side) {
        const float* o = kJoints[side ? kR0 : kL0].origin;
        s.sh[side] = rot(o[0], o[1], o[2]);
        for (int k = 0; k < kNumArmPoints; ++k) s.p[side][k] = a.p[side][k];
    }
    std::copy(&a.torso_R[0][0], &a.torso_R[0][0] + 9, &s.R[0][0]);
    return s;
}

constexpr double kS = 175.0;   // px per metre
// view 0: front, facing the robot (its left on the image right); view 1: side, from its right.
cv::Point proj(const Vec3& p, int view, const cv::Rect& panel) {
    const double u = view == 0 ? p[1] : p[0];
    return {panel.x + panel.width / 2 + int(kS * u), panel.y + 210 - int(kS * p[2])};
}

void draw_boxes(cv::Mat& img, const Skel& s, int view, const cv::Rect& panel) {
    for (const CollisionBox& b : kBoxes) {
        std::vector<cv::Point> pts;
        for (int c = 0; c < 8; ++c) {
            Vec3 v{c & 1 ? b.hi[0] : b.lo[0], c & 2 ? b.hi[1] : b.lo[1], c & 4 ? b.hi[2] : b.lo[2]};
            if (b.frame == 1) {
                Vec3 w;
                for (int i = 0; i < 3; ++i) w[i] = s.R[i][0] * v[0] + s.R[i][1] * v[1] + s.R[i][2] * v[2];
                v = w;
            }
            pts.push_back(proj(v, view, panel));
        }
        std::vector<cv::Point> hull;
        cv::convexHull(pts, hull);
        cv::polylines(img, hull, true, {75, 75, 75}, 1, cv::LINE_AA);
    }
}

void draw_skel(cv::Mat& img, const Skel& s, int view, const cv::Rect& panel, const cv::Scalar& c,
               int th, const CollisionGeometry* spheres) {
    auto P = [&](const Vec3& v) { return proj(v, view, panel); };
    cv::line(img, P({0, 0, 0}), P(s.neck), c, th, cv::LINE_AA);
    cv::circle(img, P(s.head), int(0.08 * kS), c, th, cv::LINE_AA);
    for (int side = 0; side < 2; ++side) {
        cv::line(img, P(s.neck), P(s.sh[side]), c, th, cv::LINE_AA);
        cv::line(img, P(s.sh[side]), P(s.p[side][kElbow]), c, th, cv::LINE_AA);
        cv::line(img, P(s.p[side][kElbow]), P(s.p[side][kWrist]), c, th, cv::LINE_AA);
        cv::line(img, P(s.p[side][kWrist]), P(s.p[side][kHandTip]), c, th, cv::LINE_AA);
        if (spheres) {
            const double r[kNumArmPoints] = {spheres->r_elbow, spheres->r_wrist, spheres->r_hand_mid,
                                             spheres->r_hand_tip};
            for (int k = 0; k < kNumArmPoints; ++k)
                cv::circle(img, P(s.p[side][k]), std::max(2, int(r[k] * kS)), c, 1, cv::LINE_AA);
        }
    }
}

struct ManualUi {
    bool created = false;
    int set_pos[kNumJoints] = {};     // slider position we last set programmatically...
    JointVec set_q{};                 // ...and the exact angle it stands for (no quantization)
    JointVec pending{}, checked_from{}, checked_to{};
    PoseCheck check;
    bool have_check = false;
    bool source_manual = true;        // the source the sliders were last set for
};

int to_pos(const Supervisor& sup, int j, double q) {
    return std::clamp(int(std::lround(1000.0 * (q - sup.lo(j)) / (sup.hi(j) - sup.lo(j)))), 0, 1000);
}
double from_pos(const Supervisor& sup, int j, int pos) {
    return sup.lo(j) + (sup.hi(j) - sup.lo(j)) * pos / 1000.0;
}
void set_sliders(const Supervisor& sup, ManualUi& mu, const JointVec& q) {
    for (int j = 0; j < kNumJoints; ++j) {
        mu.set_q[j] = std::clamp(q[j], sup.lo(j), sup.hi(j));
        mu.set_pos[j] = to_pos(sup, j, mu.set_q[j]);
        cv::setTrackbarPos(kShort[j], kMan, mu.set_pos[j]);
    }
}

cv::Mat render_manual(UiShared& ui, const Supervisor& sup, ManualUi& mu, Clicks& clicks) {
    const int W = 980, H = 540;
    cv::Mat img(H, W, CV_8UC3, kBg);
    bool ready, src_manual, cam_fresh;
    JointVec pub, goal, cmd, cam;
    std::string res;
    bool res_ok;
    Mode mode;
    {
        std::lock_guard<std::mutex> lk(ui.m);
        ready = ui.manual_ready; pub = ui.published; goal = ui.goal;
        for (int j = 0; j < kNumJoints; ++j) cmd[j] = ui.out.q[j];
        res = ui.apply_result; res_ok = ui.apply_ok; mode = ui.out.mode;
        src_manual = ui.source_manual; cam_fresh = ui.camera_fresh; cam = ui.camera_q;
    }
    clicks.buttons.clear();
    if (!ready && src_manual) {
        text(img, "waiting for rt/lowstate (sliders start at the measured pose)...", {20, 40}, 0.6, kYellow);
        return img;
    }
    if (!mu.created) {
        for (int j = 0; j < kNumJoints; ++j) cv::createTrackbar(kShort[j], kMan, nullptr, 1000);
        set_sliders(sup, mu, src_manual ? pub : cmd);
        mu.created = true;
        mu.source_manual = src_manual;
    }
    if (src_manual != mu.source_manual) {       // back to manual: the sliders start at the command
        if (src_manual) set_sliders(sup, mu, pub);
        mu.source_manual = src_manual;
        mu.have_check = false;
    }
    if (!src_manual) {
        // Camera drives: the sliders mirror its pose; dragging them does nothing.
        if (cam_fresh && cam != mu.set_q) set_sliders(sup, mu, cam);
        mu.pending = mu.set_q;
    } else {
        // A slider nobody has moved keeps its exact angle, so quantization never shows up as a move.
        for (int j = 0; j < kNumJoints; ++j) {
            const int pos = cv::getTrackbarPos(kShort[j], kMan);
            mu.pending[j] = pos == mu.set_pos[j] ? mu.set_q[j] : from_pos(sup, j, pos);
        }
        if (!mu.have_check || mu.checked_from != pub || mu.checked_to != mu.pending) {
            mu.check = check_manual_pose(sup, pub, mu.pending);
            mu.checked_from = pub;
            mu.checked_to = mu.pending;
            mu.have_check = true;
        }
    }
    const PoseCheck& c = mu.check;
    const CollisionGeometry& g = sup.config().collision;
    bool moving = false;
    if (src_manual)
        for (int j = 0; j < kNumJoints; ++j) moving |= std::fabs(goal[j] - pub[j]) > 1e-6;
    const bool driven = mode == Mode::RampIn || mode == Mode::Tracking;

    const Skel s_cmd = skeleton(cmd, g), s_goal = skeleton(goal, g), s_pend = skeleton(mu.pending, g),
               s_cam = skeleton(cam, g);
    const char* titles[2] = {"FRONT (facing the robot)", "SIDE (from its right)"};
    for (int view = 0; view < 2; ++view) {
        const cv::Rect panel{10 + view * 310, 10, 300, 400};
        cv::rectangle(img, panel, kDark, 1);
        text(img, titles[view], {panel.x + 8, panel.y + 18}, 0.45, kGrey);
        draw_boxes(img, s_cmd, view, panel);
        if (cam_fresh) draw_skel(img, s_cam, view, panel, kMagenta, 1, nullptr);
        draw_skel(img, s_cmd, view, panel, kWhite, 2, nullptr);
        if (src_manual) {
            if (moving) draw_skel(img, s_goal, view, panel, kYellow, 1, nullptr);
            draw_skel(img, s_pend, view, panel, c.ok ? kGreen : kRed, 1, &g);
        }
    }
    text(img, src_manual ? "white = command now   yellow = applied goal   green/red = sliders (pending)"
                         : "white = command now", {12, 432}, 0.42, kGrey);
    text(img, cam_fresh ? "magenta = camera (GMR) pose" : "camera: no fresh pose on the shm", {12, 454}, 0.42,
         cam_fresh ? kMagenta : kGrey);
    text(img, fmt("mode %s: the arms move only while ARMED from the remote (hold R1 + X)", mode_name(mode)),
         {12, 476}, 0.42, mode == Mode::Tracking ? kGreen : kYellow);
    // What FOLLOW CAMERA would do right now (the control thread re-checks on the click).
    if (src_manual) {
        std::string pick;
        bool pick_ok = false;
        if (!driven) { pick = "FOLLOW CAMERA: allowed (arms not driven)"; pick_ok = true; }
        else if (!cam_fresh) pick = "FOLLOW CAMERA: blocked, no fresh camera pose";
        else {
            const PoseCheck t = check_takeover(sup, true, cmd, cam);
            pick_ok = t.ok;
            pick = t.ok ? fmt("FOLLOW CAMERA: pick-up OK (%.2f rad)", t.max_delta)
                        : fmt("FOLLOW CAMERA: blocked, %s is %.2f rad away (needs <= %.2f)",
                              kShort[t.max_joint], t.max_delta, kTakeoverTol);
        }
        text(img, pick, {12, 498}, 0.42, pick_ok ? kGreen : kOrange);
    }

    // verdict + notes
    const int x0 = 640;
    int y = 56;
    if (src_manual) {
        text(img, "PENDING POSE CHECK", {x0, 30}, 0.5, kGrey);
        for (const std::string& l : wrap(c.verdict, 34)) { text(img, l, {x0, y}, 0.5, c.ok ? kGreen : kRed, 1); y += 20; }
        for (const std::string& n : c.notes)
            for (const std::string& l : wrap("! " + n, 38)) { text(img, l, {x0, y}, 0.45, kYellow); y += 18; }
    } else {
        text(img, "SOURCE: CAMERA (GMR shm)", {x0, 30}, 0.55, kMagenta, 2);
        for (const std::string& l : wrap("Sliders mirror the camera pose (read-only). BACK TO MANUAL "
                                         "freezes the arms at the current command.", 38)) {
            text(img, l, {x0, y}, 0.45, kWhite); y += 18;
        }
    }
    y += 8;
    if (!res.empty()) {
        text(img, "last request:", {x0, y}, 0.45, kGrey); y += 18;
        for (const std::string& l : wrap(res, 38)) { text(img, l, {x0, y}, 0.45, res_ok ? kGreen : kRed); y += 18; }
    }
    if (moving) text(img, "moving to the applied goal...", {x0, 250}, 0.5, kYellow);

    const bool estop = ui.estop;
    const bool m = src_manual && !estop;
    std::vector<Button> b = {
        {{x0, 265, 330, 55}, "APPLY", {40, 140, 40}, kBtnApply, m && c.ok && c.n_changed > 0},
        {{x0, 330, 105, 40}, "HOLD", {120, 90, 40}, kBtnHold, m},
        {{x0 + 112, 330, 105, 40}, "HOME", {90, 90, 90}, kBtnHome, m},
        {{x0 + 224, 330, 106, 40}, "RESET", {90, 90, 90}, kBtnReset, m},
        {{x0, 380, 330, 60}, estop ? "E-STOP SENT" : "E-STOP", {30, 30, 210}, kBtnEstop, !estop},
        {{x0, 450, 330, 45}, src_manual ? "FOLLOW CAMERA" : "BACK TO MANUAL",
         src_manual ? cv::Scalar(150, 50, 150) : cv::Scalar(40, 110, 40), kBtnSource, !estop},
    };
    for (const Button& bt : b) { draw_button(img, bt, bt.id == kBtnEstop || bt.id == kBtnApply ? 0.8 : 0.55); clicks.buttons.push_back(bt); }
    text(img, "HOLD = stop here.  HOME / RESET only set the sliders.", {x0, 518}, 0.38, kGrey);
    return img;
}

// Operator actions go to the terminal too, so a run can be reconstructed afterwards.
void ui_event(UiShared& ui, const std::string& line) {
    std::fprintf(stderr, "\n[h1_bridge] UI: %s\n", line.c_str());
    ui_log(&ui, "UI: " + line);
}

bool closed(const char* win) {
    try {
        return cv::getWindowProperty(win, cv::WND_PROP_VISIBLE) < 1;
    } catch (const cv::Exception&) {
        return true;
    }
}

}  // namespace

void run_ui(UiShared& ui, const Supervisor& sup) {
    const bool manual = ui.manual;
    Clicks mon_clicks, man_clicks;
    ManualUi mu;
    cv::namedWindow(kMon, cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_NORMAL);
    cv::setMouseCallback(kMon, on_mouse, &mon_clicks);
    cv::moveWindow(kMon, 0, 0);
    if (manual) {
        cv::namedWindow(kMan, cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_NORMAL);
        cv::setMouseCallback(kMan, on_mouse, &man_clicks);
        cv::moveWindow(kMan, 1000, 0);           // side by side, never stacked
    }
    bool shown = false, close_logged = false;
    while (!ui.done) {
        cv::imshow(kMon, render_monitor(ui, sup, mon_clicks));
        if (manual) cv::imshow(kMan, render_manual(ui, sup, mu, man_clicks));
        const int key = cv::waitKey(33);
        if (key == ' ' || key == 'e' || key == 'E') {
            ui.estop = true;
            ui_event(ui, "E-STOP (keyboard)");
        }
        for (Clicks* c : {&mon_clicks, &man_clicks}) {
            const ButtonId id = c->clicked;
            c->clicked = kNone;
            switch (id) {
                case kBtnEstop:
                    ui.estop = true;
                    ui_event(ui, "E-STOP (button)");
                    break;
                case kBtnApply: {
                    ui_event(ui, "APPLY clicked");
                    std::lock_guard<std::mutex> lk(ui.m);
                    ui.apply_request = true;
                    ui.apply_pose = mu.pending;
                    break;
                }
                case kBtnHold: {
                    ui_event(ui, "HOLD clicked");
                    std::lock_guard<std::mutex> lk(ui.m);
                    ui.apply_request = true;
                    ui.apply_pose = ui.published;
                    break;
                }
                case kBtnHome: {
                    JointVec h;
                    for (int j = 0; j < kNumJoints; ++j) h[j] = std::clamp(kHome[j], sup.lo(j), sup.hi(j));
                    set_sliders(sup, mu, h);
                    ui_event(ui, "HOME clicked (sliders only)");
                    break;
                }
                case kBtnReset: {
                    JointVec q;
                    {
                        std::lock_guard<std::mutex> lk(ui.m);
                        for (int j = 0; j < kNumJoints; ++j) q[j] = ui.out.q[j];
                    }
                    set_sliders(sup, mu, q);
                    ui_event(ui, "RESET clicked (sliders only)");
                    break;
                }
                case kBtnSource: {
                    bool to_manual;
                    {
                        std::lock_guard<std::mutex> lk(ui.m);
                        to_manual = !ui.source_manual;
                        ui.switch_request = true;
                        ui.switch_to_manual = to_manual;
                    }
                    ui_event(ui, to_manual ? "BACK TO MANUAL clicked" : "FOLLOW CAMERA clicked");
                    break;
                }
                case kNone: break;
            }
        }
        // Closing a window is an e-stop: nobody may be left without a view of the robot.
        if (shown && !ui.estop && (closed(kMon) || (manual && closed(kMan)))) {
            ui.estop = true;
            if (!close_logged) ui_event(ui, std::string("E-STOP (window closed: ") +
                                            (closed(kMon) ? kMon : kMan) + ")");
            close_logged = true;
        }
        shown = true;
    }
    cv::destroyAllWindows();
}

}  // namespace h1b
