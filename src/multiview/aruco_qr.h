// ════════════════════════════════════════════════════════════════════════════
//  aruco_qr.h — per-frame ArUco pose + QR timecode detection (OpenCV)
//
//  C++ port of the detection half of tracker.py: for one BGR frame it returns
//  the ArUco markers (id + Rodrigues rvec/tvec in the camera frame, metric when
//  a calibration is supplied) and the decoded QR payloads (raw + the parsed
//  t_unix_ms / t_perf_ms / frame / hz fields the QR clock emits).
//
//  This is the detection front-end for the multi-view sync step: the extrinsics
//  solver consumes the marker poses and the temporal solver consumes the QR
//  timestamps (see MULTIVIEW_PLAN.md).
//
//  OpenCV aruco types are kept behind a pimpl so this header only needs core.
// ════════════════════════════════════════════════════════════════════════════

#ifndef MULTIVIEW_ARUCO_QR_H_INCLUDED
#define MULTIVIEW_ARUCO_QR_H_INCLUDED

#include <array>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "calib.h"

namespace mv
{

struct MarkerDet
{
    int                   id = -1;
    std::array<double,3>  rvec{};   // Rodrigues rotation, camera frame
    std::array<double,3>  tvec{};   // translation (metres), camera frame
};

struct QrDet
{
    std::string raw;                // full decoded payload
    std::string t_unix_ms;          // parsed key=value fields ("" if absent)
    std::string t_perf_ms;
    std::string frame;
    std::string hz;
};

struct FrameDetections
{
    std::vector<MarkerDet> markers;
    std::vector<QrDet>     qrs;
};

// Map an OpenCV predefined-dictionary name ("DICT_6X6_250", …) to its enum id;
// returns -1 if unknown.
int  aruco_dict_id(const std::string& name);

// Parse a "k=v&k2=v2" QR payload into the typed QrDet fields (sets q.raw=txt).
void parse_qr_kv(const std::string& txt, QrDet& q);

class ArucoQrDetector
{
public:
    // dict_name e.g. "DICT_6X6_250"; default_marker_length_m is the physical
    // marker side used for pose when an id has no per-id override.
    ArucoQrDetector(const std::string& dict_name, double default_marker_length_m);
    ~ArucoQrDetector();

    ArucoQrDetector(const ArucoQrDetector&)            = delete;
    ArucoQrDetector& operator=(const ArucoQrDetector&) = delete;

    // Supply intrinsics/distortion (e.g. from a .calib) so marker pose is metric
    // and frames are undistorted before detection.  Without it an approximate
    // pinhole derived from the frame size is used and no undistortion is done.
    void set_calibration(const struct calibration& c);

    // Per-id physical marker side length (metres) override.
    void set_marker_length(int id, double len_m);

    // Detect markers + QR on one BGR frame.  The camera model / undistort maps
    // are built lazily from the first frame's size.
    void process(const cv::Mat& bgr, FrameDetections& out);

    bool calibrated() const;        // true once a calibration has been applied

private:
    struct Impl;
    Impl* impl_;
};

} // namespace mv

#endif // MULTIVIEW_ARUCO_QR_H_INCLUDED
