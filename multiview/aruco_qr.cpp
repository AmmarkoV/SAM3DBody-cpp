// ════════════════════════════════════════════════════════════════════════════
//  aruco_qr.cpp — see aruco_qr.h.
//
//  Uses the OpenCV 4.6 "classic" aruco free-function API
//  (cv::aruco::detectMarkers / estimatePoseSingleMarkers) and cv::QRCodeDetector
//  (objdetect).  Marker pose is per-id (each marker is solved with its own
//  physical side length, mirroring tracker.py).
// ════════════════════════════════════════════════════════════════════════════

#include "aruco_qr.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/objdetect.hpp>

#include <algorithm>
#include <map>
#include <unordered_map>

namespace mv
{

int aruco_dict_id(const std::string& n)
{
    static const std::unordered_map<std::string,int> m = {
        {"DICT_4X4_50",  cv::aruco::DICT_4X4_50},  {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
        {"DICT_4X4_250", cv::aruco::DICT_4X4_250}, {"DICT_4X4_1000",cv::aruco::DICT_4X4_1000},
        {"DICT_5X5_50",  cv::aruco::DICT_5X5_50},  {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
        {"DICT_5X5_250", cv::aruco::DICT_5X5_250}, {"DICT_5X5_1000",cv::aruco::DICT_5X5_1000},
        {"DICT_6X6_50",  cv::aruco::DICT_6X6_50},  {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
        {"DICT_6X6_250", cv::aruco::DICT_6X6_250}, {"DICT_6X6_1000",cv::aruco::DICT_6X6_1000},
        {"DICT_7X7_50",  cv::aruco::DICT_7X7_50},  {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
        {"DICT_7X7_250", cv::aruco::DICT_7X7_250}, {"DICT_7X7_1000",cv::aruco::DICT_7X7_1000},
        {"DICT_ARUCO_ORIGINAL",   cv::aruco::DICT_ARUCO_ORIGINAL},
        {"DICT_APRILTAG_16h5",    cv::aruco::DICT_APRILTAG_16h5},
        {"DICT_APRILTAG_25h9",    cv::aruco::DICT_APRILTAG_25h9},
        {"DICT_APRILTAG_36h10",   cv::aruco::DICT_APRILTAG_36h10},
        {"DICT_APRILTAG_36h11",   cv::aruco::DICT_APRILTAG_36h11},
    };
    auto it = m.find(n);
    return it == m.end() ? -1 : it->second;
}

void parse_qr_kv(const std::string& txt, QrDet& q)
{
    q.raw = txt;
    size_t i = 0;
    while (i <= txt.size())
    {
        size_t amp = txt.find('&', i);
        std::string tok = txt.substr(i, amp == std::string::npos ? std::string::npos : amp - i);
        size_t eq = tok.find('=');
        if (eq != std::string::npos)
        {
            std::string k = tok.substr(0, eq), v = tok.substr(eq + 1);
            // Accept both payload conventions: the C tools (xQRSync/libQRSync)
            // emit t_unix_ms / t_perf_ms / frame, while sync.html emits the
            // short t / perf / f.  hz is common to both.
            if      (k == "t_unix_ms" || k == "t")    q.t_unix_ms = v;
            else if (k == "t_perf_ms" || k == "perf") q.t_perf_ms = v;
            else if (k == "frame"     || k == "f")    q.frame     = v;
            else if (k == "hz")                        q.hz        = v;
        }
        if (amp == std::string::npos) break;
        i = amp + 1;
    }
}

struct ArucoQrDetector::Impl
{
    cv::Ptr<cv::aruco::Dictionary>        dict;
    cv::Ptr<cv::aruco::DetectorParameters> params;
    cv::QRCodeDetector                    qr;

    double                 default_len;
    std::map<int,double>   len_by_id;

    bool                   have_calib = false;
    struct calibration     calib{};

    bool     ready = false;
    cv::Mat  K, dist;            // pose camera matrix + distortion (zero if rectifying)
    bool     rectify = false;
    cv::Mat  mapx, mapy;

    void build(int W, int H)
    {
        if (have_calib && calib.intrinsicParametersSet)
        {
            double fx = calib.intrinsic[CALIB_INTR_FX], fy = calib.intrinsic[CALIB_INTR_FY];
            double cx = calib.intrinsic[CALIB_INTR_CX], cy = calib.intrinsic[CALIB_INTR_CY];
            // Scale intrinsics if the calibration resolution differs from the video.
            if (calib.width > 0 && calib.height > 0 &&
                ((int)calib.width != W || (int)calib.height != H))
            {
                double sx = (double)W / calib.width, sy = (double)H / calib.height;
                fx *= sx; cx *= sx; fy *= sy; cy *= sy;
            }
            cv::Mat Kc = (cv::Mat_<double>(3,3) << fx,0,cx, 0,fy,cy, 0,0,1);
            cv::Mat Dc = (cv::Mat_<double>(5,1) << calib.k1, calib.k2, calib.p1, calib.p2, calib.k3);

            if (cv::countNonZero(Dc) > 0)
            {
                cv::Mat newK = cv::getOptimalNewCameraMatrix(Kc, Dc, cv::Size(W,H), 1.0);
                cv::initUndistortRectifyMap(Kc, Dc, cv::noArray(), newK, cv::Size(W,H),
                                            CV_32FC1, mapx, mapy);
                K = newK;
                dist = cv::Mat::zeros(5,1,CV_64F);
                rectify = true;
            }
            else { K = Kc; dist = Dc; rectify = false; }
        }
        else
        {
            double f = 0.9 * std::max(W, H);                 // approx pinhole (tracker.py)
            K = (cv::Mat_<double>(3,3) << f,0,W/2.0, 0,f,H/2.0, 0,0,1);
            dist = cv::Mat::zeros(5,1,CV_64F);
            rectify = false;
        }
        ready = true;
    }
};

ArucoQrDetector::ArucoQrDetector(const std::string& dict_name, double default_marker_length_m)
    : impl_(new Impl)
{
    int id = aruco_dict_id(dict_name);
    if (id < 0) id = cv::aruco::DICT_6X6_250;   // sensible default for an unknown name
    impl_->dict        = cv::aruco::getPredefinedDictionary(id);
    impl_->params      = cv::aruco::DetectorParameters::create();
    impl_->default_len = default_marker_length_m;
}

ArucoQrDetector::~ArucoQrDetector() { delete impl_; }

void ArucoQrDetector::set_calibration(const struct calibration& c)
{
    impl_->calib = c;
    impl_->have_calib = true;
    impl_->ready = false;          // rebuild camera model on next frame
}

void ArucoQrDetector::set_marker_length(int id, double len_m)
{
    impl_->len_by_id[id] = len_m;
}

bool ArucoQrDetector::calibrated() const { return impl_->have_calib; }

void ArucoQrDetector::process(const cv::Mat& bgr, FrameDetections& out)
{
    out.markers.clear();
    out.qrs.clear();
    if (bgr.empty()) return;

    if (!impl_->ready) impl_->build(bgr.cols, bgr.rows);

    cv::Mat frame = bgr;
    if (impl_->rectify)
    {
        cv::Mat r;
        cv::remap(bgr, r, impl_->mapx, impl_->mapy, cv::INTER_LINEAR);
        frame = r;
    }
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // ── ArUco markers + per-id pose ──────────────────────────────────────────
    std::vector<int>                       ids;
    std::vector<std::vector<cv::Point2f>>  corners;
    cv::aruco::detectMarkers(gray, impl_->dict, corners, ids, impl_->params);

    for (size_t i = 0; i < ids.size(); ++i)
    {
        double len = impl_->default_len;
        auto it = impl_->len_by_id.find(ids[i]);
        if (it != impl_->len_by_id.end()) len = it->second;

        std::vector<std::vector<cv::Point2f>> one{ corners[i] };
        std::vector<cv::Vec3d> rv, tv;
        cv::aruco::estimatePoseSingleMarkers(one, (float)len, impl_->K, impl_->dist, rv, tv);

        MarkerDet m;
        m.id = ids[i];
        for (int k = 0; k < 3; ++k) { m.rvec[k] = rv[0][k]; m.tvec[k] = tv[0][k]; }
        out.markers.push_back(m);
    }

    // ── QR codes (multi first, then single fallback; swallow OpenCV errors) ──
    // We only need the decoded payload, not the QR's geometry, so detect on a
    // downscaled gray — QRCodeDetector is the per-frame bottleneck on high-res
    // (e.g. 5K GoPro) footage, and the smaller image is also more stable.
    cv::Mat qr_gray = gray;
    const int QR_MAX_SIDE = 1600;
    int mx = std::max(gray.cols, gray.rows);
    if (mx > QR_MAX_SIDE)
    {
        double s = (double)QR_MAX_SIDE / mx;
        cv::resize(gray, qr_gray, cv::Size(), s, s, cv::INTER_AREA);
    }

    bool any = false;
    try
    {
        std::vector<std::string> infos;
        impl_->qr.detectAndDecodeMulti(qr_gray, infos);
        for (auto& s : infos)
            if (!s.empty()) { QrDet q; parse_qr_kv(s, q); out.qrs.push_back(q); any = true; }
    }
    catch (const cv::Exception&) { /* fall through to single */ }

    if (!any)
    {
        try
        {
            std::string s = impl_->qr.detectAndDecode(qr_gray);
            if (!s.empty()) { QrDet q; parse_qr_kv(s, q); out.qrs.push_back(q); }
        }
        catch (const cv::Exception&) { /* skip QR this frame */ }
    }
}

} // namespace mv
